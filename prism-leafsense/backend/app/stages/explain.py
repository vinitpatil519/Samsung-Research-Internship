"""Stage 8: explainability - Grad-CAM and t-SNE placement (spec section 11)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ..models.registry import DEVICE, registry


@dataclass
class GradCamResult:
    available: bool
    detail: str
    cam: Optional[np.ndarray] = None      # (H, W) float32 in [0, 1]
    focus_ratio: float = 0.0              # fraction of the leaf above 0.5 activation


@dataclass
class TsneResult:
    available: bool
    detail: str
    sample: Optional[List[float]] = None          # [x, y] of the uploaded leaf
    points: Optional[np.ndarray] = None           # (N, 2) reference cloud
    labels: Optional[np.ndarray] = None           # (N,) class indices
    class_names: Optional[List[str]] = None
    neighbour_classes: Optional[List[str]] = None  # nearest training classes


def gradcam(cls_tensor: torch.Tensor, class_index: int, mask: Optional[np.ndarray] = None) -> GradCamResult:
    """Grad-CAM over the last EfficientNet feature map."""
    model = registry.classifier()
    if model is None:
        return GradCamResult(False, "classifier checkpoint missing")

    model.eval()
    x = cls_tensor.to(DEVICE).requires_grad_(False)

    feature_map = model.forward_features(x)
    feature_map.retain_grad()
    embedding = model.backbone.forward_head(feature_map, pre_logits=True)
    logits = model.classifier(embedding)

    model.zero_grad(set_to_none=True)
    logits[0, class_index].backward()

    grads = feature_map.grad
    if grads is None:
        return GradCamResult(False, "no gradients captured for Grad-CAM")

    weights = grads.mean(dim=(2, 3), keepdim=True)          # global-average-pooled gradients
    cam = F.relu((weights * feature_map).sum(dim=1, keepdim=True))
    cam = cam[0, 0].detach().cpu().numpy().astype(np.float32)

    span = float(cam.max() - cam.min())
    cam = (cam - cam.min()) / span if span > 1e-8 else np.zeros_like(cam)

    if mask is not None:
        cam = cv2.resize(cam, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_CUBIC)
        cam = cam * (mask > 0)
        leaf_pixels = int((mask > 0).sum())
        focus = float(((cam > 0.5) & (mask > 0)).sum() / leaf_pixels) if leaf_pixels else 0.0
    else:
        focus = float((cam > 0.5).mean())

    return GradCamResult(True, "grad-cam on last conv block", cam=cam, focus_ratio=focus)


def tsne_placement(latent: Optional[np.ndarray], k: int = 15, max_points: int = 2000) -> TsneResult:
    """Place the uploaded sample in the precomputed t-SNE map.

    t-SNE has no parametric transform, so the sample is positioned as the
    distance-weighted mean of its k nearest training neighbours in latent space.
    """
    reference = registry.tsne_reference()
    if reference is None:
        return TsneResult(False, "t-SNE reference not built - run backend/training/build_tsne.py")
    if latent is None:
        return TsneResult(False, "latent code unavailable")

    embeddings = reference["embeddings"]
    if latent.shape[0] != embeddings.shape[1]:
        return TsneResult(
            False,
            f"latent dimension mismatch: got {latent.shape[0]}, reference has {embeddings.shape[1]}",
        )

    distances = np.linalg.norm(embeddings - latent[None, :], axis=1)
    nearest = np.argsort(distances)[:k]
    weights = 1.0 / (distances[nearest] + 1e-6)
    weights = weights / weights.sum()
    sample = (reference["points"][nearest] * weights[:, None]).sum(axis=0)

    points = reference["points"]
    labels = reference["labels"]
    if len(points) > max_points:
        # Deterministic thinning keeps the payload small without reshaping clusters.
        step = len(points) // max_points + 1
        points = points[::step]
        labels = labels[::step]

    names = reference["class_names"]
    neighbour_classes = [names[int(labels_i)] for labels_i in reference["labels"][nearest[:5]]]

    return TsneResult(
        available=True,
        detail=f"{len(reference['points'])} reference points, k={k}",
        sample=[float(sample[0]), float(sample[1])],
        points=points.astype(np.float32),
        labels=labels.astype(np.int32),
        class_names=list(names),
        neighbour_classes=neighbour_classes,
    )
