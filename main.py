from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer

from datasets.mm_dataset import MultimodalSentimentDataset
from models.mm_model import MultimodalSentimentModel
from trainer import train_one_experiment, evaluate, predict
from utils.io import read_guid_label_file
from utils.seed import set_seed
from utils.transforms import AnyResTransform


def build_transforms(image_size: int, train: bool, anyres: bool = False) -> transforms.Compose | AnyResTransform:
    if anyres:
        return AnyResTransform(image_size, train=train)

    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )


def save_test_predictions(
    src_test_file: str,
    dst_test_file: str,
    id2label: Dict[int, str],
    guid_pred: List[Tuple[str, int]],
) -> None:
    pred_map = {g: id2label[p] for g, p in guid_pred}

    out_lines: List[str] = []
    with open(src_test_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            raw = line.rstrip("\n")
            if not raw.strip():
                continue
            if "\t" in raw:
                parts = raw.split("\t")
                sep = "\t"
            elif "," in raw:
                parts = raw.split(",")
                sep = ","
            else:
                parts = raw.split()
                sep = " "
            guid = parts[0]
            pred = pred_map.get(guid)
            if pred is None:
                raise KeyError(f"Missing prediction for guid={guid}")
            if len(parts) == 1:
                out_lines.append(f"{guid}{sep}{pred}")
            else:
                parts[1] = pred
                out_lines.append(sep.join(parts))

    Path(dst_test_file).parent.mkdir(parents=True, exist_ok=True)
    with open(dst_test_file, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")


def run_experiment(config: Dict, modality: str, ckpt_name: str) -> str:
    set_seed(int(config["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() and not config["cpu"] else "cpu")
    print(f"Using device: {device}")

    labels: List[str] = config["labels"]
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    data_root = Path(config["data_root"])
    train_file = data_root / "train.txt"
    test_file = data_root / "test_without_label.txt"
    data_dir = Path(config["data_dir"]) if config["data_dir"] else data_root / "data"

    train_pairs = read_guid_label_file(str(train_file))
    train_pairs = [(g, y) for g, y in train_pairs if y is not None]

    guids = [g for g, _ in train_pairs]
    y_str = [y for _, y in train_pairs]
    train_list, val_list = train_test_split(
        train_pairs,
        test_size=float(config["val_ratio"]),
        random_state=int(config["seed"]),
        stratify=y_str,
    )

    tokenizer = AutoTokenizer.from_pretrained(config["text_model_name"])
    anyres = config.get("anyres", False)
    tf_train = build_transforms(int(config["image_size"]), train=True, anyres=anyres)
    tf_eval = build_transforms(int(config["image_size"]), train=False, anyres=anyres)

    ds_train = MultimodalSentimentDataset(
        train_list,
        data_dir=str(data_dir),
        label2id=label2id,
        tokenizer=tokenizer,
        max_length=int(config["max_length"]),
        image_transform=tf_train,
        is_test=False,
    )
    ds_val = MultimodalSentimentDataset(
        val_list,
        data_dir=str(data_dir),
        label2id=label2id,
        tokenizer=tokenizer,
        max_length=int(config["max_length"]),
        image_transform=tf_eval,
        is_test=False,
    )

    num_workers = int(config["num_workers"])
    persistent_workers = (num_workers > 0)

    train_loader = DataLoader(
        ds_train,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )

    model = MultimodalSentimentModel(
        text_model_name=config["text_model_name"],
        image_backbone=config.get("image_backbone", "resnet50"),
        proj_dim=int(config["proj_dim"]),
        num_labels=int(config["num_labels"]),
        dropout=float(config["dropout"]),
        modality=modality,
        freeze_backbones=bool(config.get("freeze_backbones", False)),
        anyres=anyres,
    )

    exp_dir = Path(config["output_dir"]) / ckpt_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    with open(exp_dir / "config_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    best_ckpt = train_one_experiment(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=str(exp_dir),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
        epochs=int(config["epochs"]),
        warmup_ratio=float(config["warmup_ratio"]),
        grad_clip_norm=float(config["grad_clip_norm"]),
        target_names=labels,
    )

    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    val_metrics, val_report = evaluate(model, val_loader, device, target_names=labels)
    with open(exp_dir / "val_metrics.json", "w", encoding="utf-8") as f:
        json.dump(val_metrics, f, ensure_ascii=False, indent=2)

    return best_ckpt


def main():
    # Load config directly
    config_path = "configs/default.json"
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Training / ablation
    best_mm = ""
    if config["mode"] in ["train", "all"]:
        best_mm = run_experiment(config, modality="multimodal", ckpt_name="multimodal")
        print(f"[OK] Best multimodal checkpoint: {best_mm}")

    if config["mode"] in ["ablation", "all"]:
        best_text = run_experiment(config, modality="text", ckpt_name="text_only")
        best_img = run_experiment(config, modality="image", ckpt_name="image_only")
        print(f"[OK] Best text-only checkpoint: {best_text}")
        print(f"[OK] Best image-only checkpoint: {best_img}")

        if not best_mm:
            best_mm = str(Path(config["output_dir"]) / "multimodal" / "best.pt")

    # Prediction
    if config["mode"] in ["predict", "all"]:
        device = torch.device("cuda" if torch.cuda.is_available() and not config["cpu"] else "cpu")

        labels: List[str] = config["labels"]
        label2id = {l: i for i, l in enumerate(labels)}
        id2label = {i: l for l, i in label2id.items()}

        data_root = Path(config["data_root"])
        test_file = data_root / "test_without_label.txt"
        data_dir = Path(config["data_dir"]) if config["data_dir"] else data_root / "data"

        test_pairs = read_guid_label_file(str(test_file))

        tokenizer = AutoTokenizer.from_pretrained(config["text_model_name"])
        anyres = config.get("anyres", False)
        tf_eval = build_transforms(int(config["image_size"]), train=False, anyres=anyres)

        ds_test = MultimodalSentimentDataset(
            test_pairs,
            data_dir=str(data_dir),
            label2id=label2id,
            tokenizer=tokenizer,
            max_length=int(config["max_length"]),
            image_transform=tf_eval,
            is_test=True,
        )
        test_loader = DataLoader(
            ds_test,
            batch_size=int(config["batch_size"]),
            shuffle=False,
            num_workers=int(config["num_workers"]),
            pin_memory=True,
        )

        # load model
        ckpt = config["ckpt_path"] or best_mm
        if not ckpt:
            raise ValueError("No checkpoint provided. Use ckpt_path in config or run training first.")
        model = MultimodalSentimentModel(
            text_model_name=config["text_model_name"],
            image_backbone=config.get("image_backbone", "resnet50"),
            proj_dim=int(config["proj_dim"]),
            num_labels=int(config["num_labels"]),
            dropout=float(config["dropout"]),
            modality="multimodal",
            freeze_backbones=False,
            anyres=anyres,
        )
        model.load_state_dict(torch.load(ckpt, map_location=device))

        guid_pred = predict(model, test_loader, device)

        output_test_file = config["output_test_file"] or str(Path(config["output_dir"]) / "test_with_label.txt")
        save_test_predictions(str(test_file), output_test_file, id2label, guid_pred)
        print(f"[OK] Test predictions written to: {output_test_file}")


if __name__ == "__main__":
    main()
