"""Pipeline smoke test - runs in CI without any trained weights.

Verifies that the full stage graph executes on a synthetic image, that every
stage reports itself, and that stages needing missing checkpoints degrade
honestly instead of raising.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app import pipeline                          # noqa: E402
from app.labels import CLASS_NAMES, LABELS         # noqa: E402

EXPECTED_STAGES = [
    "input", "resize", "denoise", "clahe",
    "segmentation_probability", "segmentation_mask", "masked_leaf",
    "gabor", "texture", "colour",
    "sparse_encoder", "classifier", "gradcam", "tsne",
]


def synthetic_leaf(width: int = 420, height: int = 320) -> bytes:
    """Grey background with a green ellipse standing in for a leaf."""
    rng = np.random.default_rng(0)
    image = np.full((height, width, 3), 170, dtype=np.uint8)
    image += rng.integers(-12, 12, image.shape, dtype=np.int16).astype(np.uint8)

    yy, xx = np.mgrid[0:height, 0:width]
    ellipse = ((xx - width / 2) / (width / 3.2)) ** 2 + ((yy - height / 2) / (height / 2.6)) ** 2 <= 1
    image[ellipse] = (58, 140, 62)
    image[ellipse & (((xx + yy) % 37) < 4)] = (150, 120, 40)   # lesion-like speckle

    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=92)
    return buffer.getvalue()


def main() -> int:
    assert len(CLASS_NAMES) == 38, f"expected 38 classes, got {len(CLASS_NAMES)}"
    assert len(LABELS) == 38
    for name in CLASS_NAMES:
        info = LABELS[name]
        assert info.plant and info.disease, f"missing label text for {name}"
        assert info.description and info.treatment, f"missing card content for {name}"
        assert info.pathogen, f"missing pathogen for {name}"

    result = pipeline.analyse(synthetic_leaf())
    ids = [stage["id"] for stage in result["stages"]]
    assert ids == EXPECTED_STAGES, f"stage graph changed: {ids}"

    for stage in result["stages"]:
        if stage["id"] in {"sparse_encoder", "classifier", "tsne"} and not stage["available"]:
            continue
        assert stage["image"], f"stage {stage['id']} produced no image"

    segmented = pipeline.segment_only(synthetic_leaf())
    assert segmented["mask"].startswith("data:image/png;base64,")

    print(f"smoke ok: {len(ids)} stages, "
          f"classifier_available={result['available']}, {result['timing_ms']} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
