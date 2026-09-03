"""Image encode/decode helpers shared by every pipeline stage.

All stage previews leave the backend as base64 PNG data URLs so the frontend can
drop them straight into an <img> tag with no extra round trip.
"""
from __future__ import annotations

import base64
import io
from typing import List, Optional, Tuple

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


# Palette reused by the classification bars and the t-SNE scatter (RGB).
PALETTE: List[Tuple[int, int, int]] = [
    (5, 150, 105), (37, 99, 235), (180, 83, 9), (124, 58, 237), (220, 38, 38),
    (8, 145, 178), (101, 163, 13), (219, 39, 119), (71, 85, 105), (202, 138, 4),
    (13, 148, 136), (79, 70, 229), (147, 51, 234), (22, 163, 74),
]

_ACCENT = (5, 150, 105)
_INK = (15, 23, 42)
_MUTED = (100, 116, 139)
_LINE = (226, 232, 240)


def _shorten(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "."


def probability_bars(
    entries: List[Tuple[str, float]],
    size: Tuple[int, int] = (460, 560),
    title: str = "Class probabilities (%)",
) -> np.ndarray:
    """Horizontal bar chart of the top class probabilities.

    `entries` is an ordered list of (label, percentage). The winning class is
    drawn in the accent colour, the rest in grey, so the decision margin is
    visible at a glance rather than only readable in the metrics table.
    """
    h, w = size
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(canvas, title, (14, 30), font, 0.52, _INK, 1, cv2.LINE_AA)
    cv2.line(canvas, (14, 42), (w - 14, 42), _LINE, 1)

    if not entries:
        cv2.putText(canvas, "no predictions", (14, 70), font, 0.45, _MUTED, 1, cv2.LINE_AA)
        return canvas

    top = 74
    row_h = min(78, (h - top - 16) // max(1, len(entries)))
    bar_left, bar_right = 14, w - 60

    for index, (label, percentage) in enumerate(entries):
        y = top + index * row_h
        colour = _ACCENT if index == 0 else (203, 213, 225)

        cv2.putText(canvas, _shorten(label, 46), (bar_left, y), font, 0.46,
                    _INK if index == 0 else _MUTED, 1, cv2.LINE_AA)

        track_top, track_bottom = y + 10, y + 10 + 18
        cv2.rectangle(canvas, (bar_left, track_top), (bar_right, track_bottom),
                      (241, 245, 249), -1)
        filled = int(bar_left + (bar_right - bar_left) * max(0.0, min(100.0, percentage)) / 100.0)
        if filled > bar_left:
            cv2.rectangle(canvas, (bar_left, track_top), (filled, track_bottom), colour, -1)

        cv2.putText(canvas, f"{percentage:5.2f}%", (bar_right + 6, track_bottom - 3),
                    font, 0.44, _INK if index == 0 else _MUTED, 1, cv2.LINE_AA)

    return canvas


def scatter_plot(
    points: np.ndarray,
    labels: np.ndarray,
    sample: Tuple[float, float],
    class_names: List[str],
    size: Tuple[int, int] = (460, 560),
    title: str = "t-SNE of the 256-d sparse latent code",
) -> np.ndarray:
    """Render the t-SNE reference cloud with the uploaded sample marked.

    Reference points are coloured by plant species (the genus prefix of the
    class name); the upload is drawn as a black cross-hair marker.
    """
    h, w = size
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(canvas, title, (14, 26), font, 0.45, _INK, 1, cv2.LINE_AA)

    pad_left, pad_right, pad_top, pad_bottom = 40, 16, 40, 46
    plot_w = w - pad_left - pad_right
    plot_h = h - pad_top - pad_bottom

    if points.size == 0:
        cv2.putText(canvas, "no reference points", (14, 70), font, 0.45, _MUTED, 1, cv2.LINE_AA)
        return canvas

    all_x = np.append(points[:, 0], sample[0])
    all_y = np.append(points[:, 1], sample[1])
    x_lo, x_hi = float(all_x.min()), float(all_x.max())
    y_lo, y_hi = float(all_y.min()), float(all_y.max())
    x_span = max(x_hi - x_lo, 1e-6)
    y_span = max(y_hi - y_lo, 1e-6)

    def to_px(x: float, y: float) -> Tuple[int, int]:
        px = pad_left + int((x - x_lo) / x_span * (plot_w - 1))
        # Screen y grows downward, so the axis is flipped.
        py = pad_top + int((1 - (y - y_lo) / y_span) * (plot_h - 1))
        return px, py

    # Plot frame and a light grid.
    cv2.rectangle(canvas, (pad_left, pad_top), (pad_left + plot_w, pad_top + plot_h), _LINE, 1)
    for i in range(1, 4):
        gx = pad_left + i * plot_w // 4
        gy = pad_top + i * plot_h // 4
        cv2.line(canvas, (gx, pad_top), (gx, pad_top + plot_h), (241, 245, 249), 1)
        cv2.line(canvas, (pad_left, gy), (pad_left + plot_w, gy), (241, 245, 249), 1)

    # Stable species -> colour mapping, so the legend matches across requests.
    species = sorted({class_names[int(i)].split("___")[0] for i in labels})
    colour_of = {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(species)}

    for (x, y), label in zip(points, labels):
        px, py = to_px(float(x), float(y))
        colour = colour_of[class_names[int(label)].split("___")[0]]
        cv2.circle(canvas, (px, py), 2, colour, -1, cv2.LINE_AA)

    # The upload: white halo, black cross-hair, so it reads over any cluster.
    sx, sy = to_px(float(sample[0]), float(sample[1]))
    cv2.circle(canvas, (sx, sy), 7, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(canvas, (sx, sy), 7, _INK, 2, cv2.LINE_AA)
    cv2.line(canvas, (sx - 11, sy), (sx + 11, sy), _INK, 1, cv2.LINE_AA)
    cv2.line(canvas, (sx, sy - 11), (sx, sy + 11), _INK, 1, cv2.LINE_AA)

    cv2.putText(canvas, "uploaded sample", (14, h - 26), font, 0.4, _INK, 1, cv2.LINE_AA)
    cv2.circle(canvas, (128, h - 30), 4, _INK, 2, cv2.LINE_AA)
    cv2.putText(canvas, f"{len(points)} training points, {len(species)} species",
                (14, h - 10), font, 0.38, _MUTED, 1, cv2.LINE_AA)
    return canvas
