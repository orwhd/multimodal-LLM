from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm


class ImageEncoder(nn.Module):
    """
    Image encoder: supports ResNet and ViT backbones.
    """

    def __init__(self, backbone: str = "resnet50", pretrained: bool = True, anyres: bool = False) -> None:
        super().__init__()
        self.anyres = anyres
        if backbone == "resnet50":
            m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            self.feature_dim = 2048
            self.backbone = nn.Sequential(*list(m.children())[:-1])
        elif backbone == "resnet18":
            m = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            self.feature_dim = 512
            self.backbone = nn.Sequential(*list(m.children())[:-1])
        elif backbone == "vit_b_16":
            weights = tvm.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = tvm.vit_b_16(weights=weights)
            self.feature_dim = 768
            self.model.heads = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        if self.anyres:
            self.feature_dim *= 5

        self.backbone_name = backbone

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.anyres:
            if pixel_values.dim() == 5:
                b, n, c, h, w = pixel_values.shape
                pixel_values = pixel_values.view(b * n, c, h, w)
            else:
                raise ValueError(f"Expected 5D input for AnyRes, got {pixel_values.shape}")

        if self.backbone_name.startswith("vit"):
            feat = self.model(pixel_values)
        else:
            feat = self.backbone(pixel_values)
            feat = feat.flatten(1)
            
        if self.anyres:
            feat = feat.reshape(b, -1)
            
        return feat
