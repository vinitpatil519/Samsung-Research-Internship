"""Stage 3: leaf segmentation with U-Net++ (spec section 6).

The trained checkpoint is the primary path. When no checkpoint exists yet the
stage falls back to a classical Excess-Green + Otsu segmentation and says so in
`method`, so the UI never presents a classical mask as a network output.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

from ..models.registry import DEVICE, registry


@dataclass
class SegmentResult:
    probability: np.ndarray   # (256, 256) float32 in [0, 1]
    mask: np.ndarray          # (256, 256) uint8 in {0, 1}
    method: str               # "unetpp" or "classical-exg-otsu"
    leaf_ratio: float         # fraction of pixels kept
    components: int           # connected components in the final mask


def _classical(image: np.ndarray) -> np.ndarray:
    """Excess-Green index + Otsu threshold, plus a saturation cue.

    Leaves are green or chlorotic-yellow; both stay well separated from grey
    backgrounds in ExG and in HSV saturation.
    """
    rgb = image.astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    exg = 2 * g - r - b

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1].astype(np.float32) / 255.0

    score = 0.65 * (exg - exg.min()) / (np.ptp(exg) + 1e-6) + 0.35 * sat
    score_u8 = (np.clip(score, 0, 1) * 255).astype(np.uint8)
    _, binary = cv2.threshold(score_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (binary > 0).astype(np.float32)


def _refine(prob: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Threshold, close small holes, drop specks, keep the dominant blobs."""
    mask = (prob > threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest = areas.max()
        keep = {i + 1 for i, area in enumerate(areas) if area >= max(64, 0.15 * largest)}
        mask = np.isin(labels, list(keep)).astype(np.uint8)

    # A mask covering almost nothing means segmentation failed; use everything
    # rather than handing the classifier an empty frame.
    if mask.mean() < 0.02:
        mask = np.ones_like(mask)
    return mask


def run(preprocessed_rgb: np.ndarray, seg_tensor: torch.Tensor) -> SegmentResult:
    model = registry.unet()
    if model is not None:
        with torch.no_grad():
            logits = model(seg_tensor.to(DEVICE))
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy().astype(np.float32)
        method = "unetpp"
    else:
        prob = _classical(preprocessed_rgb)
        method = "classical-exg-otsu"

    mask = _refine(prob)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return SegmentResult(
        probability=prob,
        mask=mask,
        method=method,
        leaf_ratio=float(mask.mean()),
        components=max(0, count - 1),
    )
