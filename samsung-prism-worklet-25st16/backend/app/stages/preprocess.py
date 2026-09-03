"""Stage 1-2: preprocessing (spec section 5).

Resize to 256x256, Gaussian denoise, CLAHE contrast equalisation, then RGB
normalisation for the networks. Every intermediate is returned so the UI can
show what each operation did.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

from .. import config


@dataclass
class PreprocessResult:
    original: np.ndarray      # RGB uint8, EXIF-corrected, untouched geometry
    resized: np.ndarray       # 256x256 RGB uint8
    denoised: np.ndarray      # after Gaussian filtering
    equalised: np.ndarray     # after CLAHE on the L channel
    seg_tensor: torch.Tensor  # (1, 3, 256, 256) normalised for U-Net++
    cls_tensor: torch.Tensor  # (1, 3, 224, 224) normalised for EfficientNet
    noise_removed: float      # mean absolute change caused by denoising
    contrast_gain: float      # std-dev ratio after CLAHE


def to_tensor(image: np.ndarray, size: int) -> torch.Tensor:
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    arr = resized.astype(np.float32) / 255.0
    mean = np.array(config.IMAGENET_MEAN, dtype=np.float32)
    std = np.array(config.IMAGENET_STD, dtype=np.float32)
    arr = (arr - mean) / std
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)


def enhance(image: np.ndarray, size: int = config.SEG_SIZE):
    """Resize, denoise and CLAHE-equalise. Shared by inference and training.

    Training uses this same function, so the classifier never sees a different
    pixel distribution than the one the API feeds it.
    """
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)

    k = config.GAUSSIAN_KERNEL
    denoised = cv2.GaussianBlur(resized, (k, k), config.GAUSSIAN_SIGMA)

    # CLAHE on luminance only, so hue stays usable for colour features.
    lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(
        clipLimit=config.CLAHE_CLIP,
        tileGridSize=(config.CLAHE_GRID, config.CLAHE_GRID),
    )
    lab[..., 0] = clahe.apply(lab[..., 0])
    equalised = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return resized, denoised, equalised


def run(image: np.ndarray) -> PreprocessResult:
    resized, denoised, equalised = enhance(image, config.SEG_SIZE)
    noise_removed = float(np.mean(np.abs(resized.astype(np.float32) - denoised.astype(np.float32))))

    before = float(cv2.cvtColor(denoised, cv2.COLOR_RGB2GRAY).std())
    after = float(cv2.cvtColor(equalised, cv2.COLOR_RGB2GRAY).std())
    contrast_gain = after / before if before > 1e-6 else 1.0

    return PreprocessResult(
        original=image,
        resized=resized,
        denoised=denoised,
        equalised=equalised,
        seg_tensor=to_tensor(equalised, config.SEG_SIZE),
        cls_tensor=to_tensor(equalised, config.CLS_SIZE),
        noise_removed=noise_removed,
        contrast_gain=contrast_gain,
    )
