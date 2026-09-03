"""EfficientNet classifier head (spec sections 7 and 9).

The wrapper exposes three things the pipeline needs:
  * logits over the 38 PlantVillage classes,
  * the pooled deep embedding used by the sparse encoder and t-SNE,
  * the last convolutional feature map, which Grad-CAM hooks.

`arch` is stored inside the checkpoint, so a B0 model trained on CPU and a B3
model trained on a GPU load through the same code path.
"""
from __future__ import annotations

from dataclasses import dataclass

import timm
import torch
import torch.nn as nn


@dataclass
class ClassifierOutput:
    logits: torch.Tensor          # (B, num_classes)
    embedding: torch.Tensor       # (B, feature_dim) pooled deep features
    feature_map: torch.Tensor     # (B, C, H, W) last conv activations


class LeafClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        arch: str = "efficientnet_b0",
        pretrained: bool = False,
        drop_rate: float = 0.3,
    ) -> None:
        super().__init__()
        self.arch = arch
        self.num_classes = num_classes
        # num_classes=0 gives a headless backbone that returns pooled features.
        self.backbone = timm.create_model(
            arch, pretrained=pretrained, num_classes=0, drop_rate=drop_rate
        )
        self.feature_dim = self.backbone.num_features
        self.dropout = nn.Dropout(drop_rate)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Last convolutional feature map, before pooling (Grad-CAM target)."""
        return self.backbone.forward_features(x)

    def forward(self, x: torch.Tensor) -> ClassifierOutput:
        feature_map = self.backbone.forward_features(x)
        embedding = self.backbone.forward_head(feature_map, pre_logits=True)
        logits = self.classifier(self.dropout(embedding))
        return ClassifierOutput(logits=logits, embedding=embedding, feature_map=feature_map)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> ClassifierOutput:
        self.eval()
        return self.forward(x)
