from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn

from models.text_encoder import TextEncoder
from models.image_encoder import ImageEncoder
from models.fusion import LateFusionHead, TextOnlyHead, ImageOnlyHead


Modality = Literal["multimodal", "text", "image"]


class MultimodalSentimentModel(nn.Module):
    def __init__(
        self,
        text_model_name: str,
        image_backbone: str,
        proj_dim: int,
        num_labels: int,
        dropout: float = 0.2,
        modality: Modality = "multimodal",
        freeze_backbones: bool = False,
        anyres: bool = False,
    ) -> None:
        super().__init__()
        self.modality: Modality = modality

        self.text_encoder = TextEncoder(text_model_name)
        self.image_encoder = ImageEncoder(backbone=image_backbone, pretrained=True, anyres=anyres)

        if freeze_backbones:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
            for p in self.image_encoder.parameters():
                p.requires_grad = False

        tdim = self.text_encoder.hidden_size
        vdim = self.image_encoder.feature_dim

        if modality == "multimodal":
            self.head = LateFusionHead(text_dim=tdim, image_dim=vdim, proj_dim=proj_dim, num_labels=num_labels, dropout=dropout)
        elif modality == "text":
            self.head = TextOnlyHead(text_dim=tdim, proj_dim=proj_dim, num_labels=num_labels, dropout=dropout)
        elif modality == "image":
            self.head = ImageOnlyHead(image_dim=vdim, proj_dim=proj_dim, num_labels=num_labels, dropout=dropout)
        else:
            raise ValueError(f"Unknown modality: {modality}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        if self.modality == "text":
            text_feat = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            logits = self.head(text_feat)
            return logits

        if self.modality == "image":
            img_feat = self.image_encoder(pixel_values=pixel_values)
            logits = self.head(img_feat)
            return logits

        text_feat = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        img_feat = self.image_encoder(pixel_values=pixel_values)
        logits = self.head(text_feat, img_feat)
        return logits
