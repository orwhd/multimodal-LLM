from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from utils.metrics import compute_metrics, build_classification_report


def move_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        if k == "guid":
            out[k] = v
        else:
            out[k] = v.to(device)
    return out


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device, target_names: List[str]) -> Tuple[Dict[str, float], str]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []

    for batch in dataloader:
        batch = move_to_device(batch, device)
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
        )
        preds = torch.argmax(logits, dim=-1)
        y_pred.extend(preds.cpu().tolist())
        y_true.extend(batch["labels"].cpu().tolist())

    m = compute_metrics(y_true, y_pred)
    report = build_classification_report(y_true, y_pred, target_names=target_names)
    return m.to_dict(), report


def train_one_experiment(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    output_dir: str,
    lr: float,
    weight_decay: float,
    epochs: int,
    warmup_ratio: float,
    grad_clip_norm: float,
    target_names: List[str],
) -> str:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    criterion = nn.CrossEntropyLoss()

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    model.to(device)
    
    # AMP
    scaler = torch.cuda.amp.GradScaler()

    best_f1 = -1.0
    best_path = out_dir / "best.pt"
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        total_loss = 0.0
        for batch in pbar:
            batch = move_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch["pixel_values"],
                )
                loss = criterion(logits, batch["labels"])

            # Scale loss
            scaler.scale(loss).backward()

            if grad_clip_norm and grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / max(len(train_loader), 1)

        val_metrics, val_report = evaluate(model, val_loader, device, target_names=target_names)
        
        scheduler.step(val_metrics['macro_f1'])

        print(f"Epoch {epoch}/{epochs} | Train Loss: {avg_loss:.4f} | Val F1: {val_metrics['macro_f1']:.4f} | Val Acc: {val_metrics['acc']:.4f}")

        epoch_stats = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_metrics": val_metrics
        }
        history.append(epoch_stats)
        
        with open(out_dir / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        (out_dir / "logs").mkdir(exist_ok=True)
        with open(out_dir / "logs" / f"epoch_{epoch}.json", "w", encoding="utf-8") as f:
            json.dump(
                {"epoch": epoch, "train_loss": avg_loss, "val": val_metrics},
                f,
                ensure_ascii=False,
                indent=2,
            )

        # Update best
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), best_path)
            with open(out_dir / "best_report.txt", "w", encoding="utf-8") as f:
                f.write(val_report)

    return str(best_path)


@torch.no_grad()
def predict(model: nn.Module, dataloader: DataLoader, device: torch.device) -> List[Tuple[str, int]]:
    model.eval()
    model.to(device)
    results: List[Tuple[str, int]] = []
    for batch in tqdm(dataloader, desc="Predict", leave=False):
        guid = batch["guid"]
        batch = move_to_device(batch, device)
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
        )
        preds = torch.argmax(logits, dim=-1).cpu().tolist()
        results.extend(list(zip(guid, preds)))
    return results
