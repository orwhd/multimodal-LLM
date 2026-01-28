from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

from utils.io import find_image_file, find_text_file, load_text


@dataclass
class Sample:
    guid: str
    text_path: Path
    image_path: Path
    label: Optional[int]


class MultimodalSentimentDataset(Dataset):
    def __init__(
        self,
        pairs: List[Tuple[str, Optional[str]]],
        data_dir: str,
        label2id: Dict[str, int],
        tokenizer,
        max_length: int,
        image_transform: Optional[Callable] = None,
        is_test: bool = False,
    ) -> None:
        self.samples: List[Sample] = []
        self.data_dir = data_dir
        self.label2id = label2id
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_transform = image_transform
        self.is_test = is_test

        for guid, label_str in pairs:
            text_path = find_text_file(data_dir, guid)
            image_path = find_image_file(data_dir, guid)
            label_id = None
            if (label_str is not None) and (not is_test):
                if label_str not in label2id:
                    raise ValueError(f"Unknown label {label_str} for guid={guid}")
                label_id = label2id[label_str]
            self.samples.append(Sample(guid=guid, text_path=text_path, image_path=image_path, label=label_id))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        text = load_text(s.text_path)

        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        image = Image.open(s.image_path).convert("RGB")
        if self.image_transform is not None:
            image = self.image_transform(image)

        item = {
            "guid": s.guid,
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "pixel_values": image,
        }
        if not self.is_test:
            item["labels"] = torch.tensor(s.label, dtype=torch.long)
        return item
