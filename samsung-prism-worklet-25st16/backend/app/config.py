"""Central configuration for the Samsung Research (Prism) Worklet:25ST16 backend."""
from __future__ import annotations

import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent

WEIGHTS_DIR = Path(os.getenv("WORKLET_WEIGHTS_DIR", PROJECT_ROOT / "weights"))
DATASET_DIR = Path(os.getenv("WORKLET_DATASET_DIR", PROJECT_ROOT / "datasets"))

# Weight files resolved through the model registry.
UNET_WEIGHTS = WEIGHTS_DIR / "unetpp_leaf.pt"
CLASSIFIER_WEIGHTS = WEIGHTS_DIR / "efficientnet_plantvillage.pt"
ENCODER_WEIGHTS = WEIGHTS_DIR / "sparse_encoder.pt"
TSNE_REFERENCE = WEIGHTS_DIR / "tsne_reference.npz"

# Image geometry.
SEG_SIZE = 256          # U-Net++ input resolution
CLS_SIZE = 224          # EfficientNet input resolution
PREVIEW_SIZE = 320      # size of the base64 previews returned per stage

# Preprocessing knobs (section 5 of the master spec).
GAUSSIAN_KERNEL = 5
GAUSSIAN_SIGMA = 1.0
CLAHE_CLIP = 2.0
CLAHE_GRID = 8

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Inference device: CPU by default, CUDA when available.
DEVICE = os.getenv("WORKLET_DEVICE", "auto")

MAX_UPLOAD_BYTES = int(os.getenv("WORKLET_MAX_UPLOAD_BYTES", 12 * 1024 * 1024))
ALLOWED_ORIGINS = os.getenv(
    "WORKLET_ALLOWED_ORIGINS",
    ",".join(
        f"http://{host}:{port}"
        for host in ("localhost", "127.0.0.1")
        for port in (3000, 3001, 3002, 3003)
    ),
).split(",")
