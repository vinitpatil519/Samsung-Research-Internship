"""Image encode/decode helpers shared by every pipeline stage.

All stage previews leave the backend as base64 PNG data URLs so the frontend can
drop them straight into an <img> tag with no extra round trip.
"""
from __future__ import annotations

import base64
import io
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

from . import config


def decode_upload(raw: bytes) -> np.ndarray:
    """Decode uploaded bytes to an RGB uint8 array, honouring EXIF rotation.

    Phone cameras store orientation in EXIF; without transpose a portrait photo
    reaches the model rotated by 90 degrees.
    """
    try:
        with Image.open(io.BytesIO(raw)) as img:
            img = ImageOps.exif_transpose(img)
            return np.array(img.convert("RGB"))
    except Exception as exc:  # noqa: BLE001 - surfaced as a 400 by the API layer
        raise ValueError(f"could not decode image: {exc}") from exc


def to_data_url(image: np.ndarray, max_side: Optional[int] = None) -> str:
    """Encode an RGB / grayscale uint8 array as a base64 PNG data URL."""
    arr = image
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if max_side is None:
        max_side = config.PREVIEW_SIZE
    h, w = arr.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        arr = cv2.resize(arr, (max(1, int(w * scale)), max(1, int(h * scale))),
                         interpolation=cv2.INTER_AREA)
    mode = "L" if arr.ndim == 2 else "RGB"
    buf = io.BytesIO()
    Image.fromarray(arr, mode=mode).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def normalise01(arr: np.ndarray) -> np.ndarray:
    """Scale an arbitrary float map into [0, 1]."""
    arr = arr.astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-8:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def colourise(map01: np.ndarray, colormap: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
    """Turn a [0, 1] single-channel map into an RGB heatmap."""
    gray = (np.clip(map01, 0, 1) * 255).astype(np.uint8)
    bgr = cv2.applyColorMap(gray, colormap)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def overlay_heatmap(
    image: np.ndarray, map01: np.ndarray, alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Blend a heatmap over an RGB image."""
    heat = colourise(map01, colormap)
    if heat.shape[:2] != image.shape[:2]:
        heat = cv2.resize(heat, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.clip((1 - alpha) * image.astype(np.float32) + alpha * heat.astype(np.float32),
                   0, 255).astype(np.uint8)


def draw_mask_outline(image: np.ndarray, mask: np.ndarray,
                      colour: Tuple[int, int, int] = (16, 185, 129)) -> np.ndarray:
    """Draw the mask boundary on a copy of the image."""
    out = image.copy()
    binary = (mask > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, colour, 2)
    return out


def apply_mask(image: np.ndarray, mask: np.ndarray,
               background: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """Keep leaf pixels, replace background with a flat colour."""
    m = (mask > 0.5).astype(np.float32)[..., None]
    bg = np.array(background, dtype=np.float32).reshape(1, 1, 3)
    return np.clip(image.astype(np.float32) * m + bg * (1 - m), 0, 255).astype(np.uint8)


def latent_strip(latent: np.ndarray, height: int = 64) -> np.ndarray:
    """Render a 1-D latent vector as a colour strip image."""
    vec = normalise01(latent.reshape(1, -1))
    side = int(np.ceil(np.sqrt(vec.size)))
    padded = np.zeros(side * side, dtype=np.float32)
    padded[: vec.size] = vec.ravel()
    grid = padded.reshape(side, side)
    grid = cv2.resize(grid, (side * 8, side * 8), interpolation=cv2.INTER_NEAREST)
    return colourise(grid, cv2.COLORMAP_VIRIDIS)


def histogram_image(hist: np.ndarray, colour: Tuple[int, int, int],
                    size: Tuple[int, int] = (180, 320)) -> np.ndarray:
    """Simple bar rendering of a 1-D histogram."""
    h, w = size
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    values = normalise01(hist)
    bins = len(values)
    bar_w = max(1, w // bins)
    for i, v in enumerate(values):
        x0 = i * bar_w
        y0 = int((1 - v) * (h - 4))
        cv2.rectangle(canvas, (x0, y0), (x0 + bar_w - 1, h - 1), colour, -1)
    return canvas
