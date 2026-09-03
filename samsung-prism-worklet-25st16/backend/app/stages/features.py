"""Stage 5: feature extraction (spec section 7).

Deep features come from the EfficientNet embedding; handcrafted features are
GLCM texture, a Gabor filter bank, HSV colour histograms and shape descriptors.
The concatenation order defined here is the contract shared with the sparse
encoder training script - changing it invalidates the encoder checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import gabor_kernel
from skimage.filters.rank import entropy as local_entropy
from skimage.morphology import disk

GLCM_DISTANCES = (1, 2)
GLCM_ANGLES = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
GLCM_PROPS = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")
GLCM_LEVELS = 32

GABOR_FREQUENCIES = (0.1, 0.2, 0.3, 0.4)
GABOR_THETAS = tuple(np.pi * i / 6 for i in range(6))

HIST_BINS = 32

GLCM_DIM = len(GLCM_PROPS) * len(GLCM_DISTANCES) * len(GLCM_ANGLES)   # 48
GABOR_DIM = len(GABOR_FREQUENCIES) * len(GABOR_THETAS) * 2            # 48
COLOR_DIM = HIST_BINS * 3                                             # 96
SHAPE_DIM = 14
HANDCRAFTED_DIM = GLCM_DIM + GABOR_DIM + COLOR_DIM + SHAPE_DIM        # 206

_GABOR_KERNELS: List[np.ndarray] = [
    np.real(gabor_kernel(freq, theta=theta))
    for freq in GABOR_FREQUENCIES
    for theta in GABOR_THETAS
]


@dataclass
class FeatureResult:
    handcrafted: np.ndarray                      # (206,) float32
    parts: Dict[str, np.ndarray]                 # named sub-vectors
    gabor_energy: np.ndarray                     # (H, W) float32, normalised
    texture_entropy: np.ndarray                  # (H, W) float32, normalised
    colour_histograms: Dict[str, np.ndarray]     # H/S/V histograms
    shape_metrics: Dict[str, float] = field(default_factory=dict)


def _glcm_features(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Grey-level co-occurrence statistics over the leaf region."""
    quantised = (gray.astype(np.float32) / 256.0 * GLCM_LEVELS).astype(np.uint8)
    quantised = np.clip(quantised, 0, GLCM_LEVELS - 1)
    # Background pixels are pushed to level 0 and stay a constant, so they add
    # no spurious texture transitions inside the leaf.
    quantised = quantised * (mask > 0)

    glcm = graycomatrix(
        quantised,
        distances=list(GLCM_DISTANCES),
        angles=list(GLCM_ANGLES),
        levels=GLCM_LEVELS,
        symmetric=True,
        normed=True,
    )
    values = [np.asarray(graycoprops(glcm, prop), dtype=np.float32).ravel() for prop in GLCM_PROPS]
    return np.concatenate(values).astype(np.float32)


def _gabor_features(gray: np.ndarray, mask: np.ndarray):
    """Mean and std of each Gabor response, plus a combined energy map."""
    norm = gray.astype(np.float32) / 255.0
    stats: List[float] = []
    energy = np.zeros_like(norm)
    valid = mask > 0
    for kernel in _GABOR_KERNELS:
        response = cv2.filter2D(norm, cv2.CV_32F, kernel)
        magnitude = np.abs(response)
        energy = np.maximum(energy, magnitude)
        region = magnitude[valid] if valid.any() else magnitude
        stats.append(float(region.mean()))
        stats.append(float(region.std()))
    return np.asarray(stats, dtype=np.float32), energy


def _colour_features(rgb: np.ndarray, mask: np.ndarray):
    """HSV histograms computed over leaf pixels only."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    binary = (mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        binary = np.ones_like(binary)
    hists = {}
    for idx, name in enumerate(("hue", "saturation", "value")):
        ranges = [0, 180] if idx == 0 else [0, 256]
        hist = cv2.calcHist([hsv], [idx], binary, [HIST_BINS], ranges).ravel()
        total = hist.sum()
        hists[name] = (hist / total if total > 0 else hist).astype(np.float32)
    vector = np.concatenate([hists["hue"], hists["saturation"], hists["value"]])
    return vector.astype(np.float32), hists


def _shape_features(mask: np.ndarray):
    """Contour-derived shape descriptors plus the 7 Hu moments."""
    binary = (mask > 0).astype(np.uint8)
    metrics = {
        "area_ratio": float(binary.mean()),
        "perimeter": 0.0,
        "circularity": 0.0,
        "aspect_ratio": 0.0,
        "extent": 0.0,
        "solidity": 0.0,
        "eccentricity": 0.0,
    }
    hu = np.zeros(7, dtype=np.float32)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        x, y, w, h = cv2.boundingRect(contour)
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))

        metrics["perimeter"] = perimeter / max(binary.shape)
        metrics["circularity"] = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0
        metrics["aspect_ratio"] = w / h if h else 0.0
        metrics["extent"] = area / (w * h) if w * h else 0.0
        metrics["solidity"] = area / hull_area if hull_area else 0.0
        if len(contour) >= 5:
            (_, _), (major, minor), _ = cv2.fitEllipse(contour)
            major, minor = max(major, minor), min(major, minor)
            if major > 0:
                metrics["eccentricity"] = float(np.sqrt(max(0.0, 1 - (minor / major) ** 2)))

        moments = cv2.moments(binary)
        raw_hu = cv2.HuMoments(moments).ravel()
        # Log-scale compresses the 7 Hu moments into a comparable range.
        hu = np.sign(raw_hu) * np.log10(np.abs(raw_hu) + 1e-30)
        hu = hu.astype(np.float32)

    vector = np.concatenate([
        np.asarray(list(metrics.values()), dtype=np.float32),
        hu,
    ])
    return vector.astype(np.float32), metrics


def run(rgb: np.ndarray, mask: np.ndarray) -> FeatureResult:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    glcm_vec = _glcm_features(gray, mask)
    gabor_vec, gabor_energy = _gabor_features(gray, mask)
    colour_vec, hists = _colour_features(rgb, mask)
    shape_vec, shape_metrics = _shape_features(mask)

    handcrafted = np.concatenate([glcm_vec, gabor_vec, colour_vec, shape_vec]).astype(np.float32)
    handcrafted = np.nan_to_num(handcrafted, nan=0.0, posinf=0.0, neginf=0.0)

    ent = local_entropy(gray, disk(5)).astype(np.float32)
    ent = ent * (mask > 0)

    def norm(arr: np.ndarray) -> np.ndarray:
        span = float(arr.max() - arr.min())
        return (arr - arr.min()) / span if span > 1e-8 else np.zeros_like(arr)

    return FeatureResult(
        handcrafted=handcrafted,
        parts={
            "glcm": glcm_vec,
            "gabor": gabor_vec,
            "colour": colour_vec,
            "shape": shape_vec,
        },
        gabor_energy=norm(gabor_energy * (mask > 0)),
        texture_entropy=norm(ent),
        colour_histograms=hists,
        shape_metrics=shape_metrics,
    )
