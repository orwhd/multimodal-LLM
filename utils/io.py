from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple


IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def read_guid_label_file(path: str) -> List[Tuple[str, Optional[str]]]:
    pairs: List[Tuple[str, Optional[str]]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if "," in line:
                parts = line.split(",")
            else:
                parts = line.split()
            
            guid = parts[0].strip()
            label = parts[1].strip() if len(parts) > 1 else None
            
            if label and (label.lower() == "null" or label == ""):
                label = None
                
            pairs.append((guid, label))
    return pairs


def find_text_file(data_dir: str, guid: str) -> Path:
    return Path(data_dir) / f"{guid}.txt"


def find_image_file(data_dir: str, guid: str) -> Path:
    base = Path(data_dir)
    for ext in IMG_EXTS:
        p = base / f"{guid}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"Cannot find image file for guid={guid} under {data_dir}")


def load_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()
