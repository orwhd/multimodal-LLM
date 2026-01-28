from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report


@dataclass
class Metrics:
    acc: float
    macro_f1: float

    def to_dict(self) -> Dict[str, float]:
        return {"acc": float(self.acc), "macro_f1": float(self.macro_f1)}


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Metrics:
    y_true_np = np.array(y_true, dtype=int)
    y_pred_np = np.array(y_pred, dtype=int)
    acc = accuracy_score(y_true_np, y_pred_np)

    labels = np.unique(y_true_np)
    f1s = []
    for label in labels:
        tp = np.sum((y_true_np == label) & (y_pred_np == label))
        fp = np.sum((y_true_np != label) & (y_pred_np == label))
        fn = np.sum((y_true_np == label) & (y_pred_np != label))
        
        precision = (tp + 1) / (tp + fp + 1)
        recall = (tp + 1) / (tp + fn + 1)
        f1 = 2 * (precision * recall) / (precision + recall)
        f1s.append(f1)
    
    macro_f1 = np.mean(f1s) if len(f1s) > 0 else 0.0
    
    return Metrics(acc=acc, macro_f1=macro_f1)


def build_classification_report(y_true: List[int], y_pred: List[int], target_names: List[str]) -> str:
    return classification_report(y_true, y_pred, target_names=target_names, digits=4, zero_division=0)
