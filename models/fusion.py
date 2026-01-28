from __future__ import annotations

import torch
import torch.nn as nn


class LateFusionHead(nn.Module):
    """
    project text and image features to same dimension
    """

    def __init__(self, text_dim: int, image_dim: int, proj_dim: int, num_labels: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.text_proj = nn.Linear(text_dim, proj_dim)
        self.image_proj = nn.Linear(image_dim, proj_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, num_labels),
        )

    def forward(self, text_feat: torch.Tensor, image_feat: torch.Tensor) -> torch.Tensor:
        t = self.text_proj(text_feat)
        v = self.image_proj(image_feat)
        fused = torch.cat([t, v], dim=-1)
        return self.classifier(fused)


class TextOnlyHead(nn.Module):
    def __init__(self, text_dim: int, proj_dim: int, num_labels: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(text_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, num_labels),
        )

    def forward(self, text_feat: torch.Tensor) -> torch.Tensor:
        return self.net(text_feat)


class ImageOnlyHead(nn.Module):
    def __init__(self, image_dim: int, proj_dim: int, num_labels: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(image_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, num_labels),
        )

    def forward(self, image_feat: torch.Tensor) -> torch.Tensor:
        return self.net(image_feat)
