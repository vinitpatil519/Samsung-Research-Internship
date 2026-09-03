"""Stage 7: species + disease classification (spec section 9).

The classifier is trained on background-removed leaves, so inference runs on the
masked image produced by the U-Net++ stage - the same distribution it saw during
training.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..labels import label_for_name
from ..models.registry import DEVICE, registry


@dataclass
class Prediction:
    class_name: str
    index: int
    probability: float
    plant: str
    disease: str
    pathogen: str
    healthy: bool


@dataclass
class ClassifyResult:
    available: bool
    detail: str
    predictions: List[Prediction]
    probabilities: Optional[np.ndarray] = None   # full softmax vector
    embedding: Optional[np.ndarray] = None       # pooled deep features
    entropy: float = 0.0                         # predictive entropy in bits
    margin: float = 0.0                          # top1 - top2 probability


def run(cls_tensor: torch.Tensor, top_k: int = 3) -> ClassifyResult:
    model = registry.classifier()
    if model is None:
        return ClassifyResult(
            available=False,
            detail="classifier checkpoint missing - run backend/training/train_classifier.py",
            predictions=[],
        )

    class_names = registry.class_names()
    with torch.no_grad():
        out = model(cls_tensor.to(DEVICE))
        probs = F.softmax(out.logits, dim=1)[0].cpu().numpy().astype(np.float32)
        embedding = out.embedding[0].cpu().numpy().astype(np.float32)

    order = np.argsort(probs)[::-1]
    predictions: List[Prediction] = []
    for idx in order[: max(1, top_k)]:
        name = class_names[int(idx)]
        info = label_for_name(name)
        predictions.append(
            Prediction(
                class_name=name,
                index=int(idx),
                probability=float(probs[int(idx)] * 100.0),
                plant=info.plant,
                disease=info.disease,
                pathogen=info.pathogen,
                healthy=info.healthy,
            )
        )

    safe = np.clip(probs, 1e-12, 1.0)
    entropy = float(-(safe * np.log2(safe)).sum())
    margin = float((probs[order[0]] - probs[order[1]]) * 100.0) if len(order) > 1 else 100.0

    return ClassifyResult(
        available=True,
        detail=f"arch={model.arch}",
        predictions=predictions,
        probabilities=probs,
        embedding=embedding,
        entropy=entropy,
        margin=margin,
    )
