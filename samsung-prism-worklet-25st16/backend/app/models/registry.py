"""Weight-loading abstraction.

Every model is loaded once, lazily, and cached. If a checkpoint is missing the
registry reports it instead of silently returning an untrained network - the
pipeline then marks the affected stage as unavailable rather than inventing a
prediction.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .. import config
from ..labels import CLASS_NAMES, NUM_CLASSES
from .classifier import LeafClassifier
from .sparse_encoder import SparseStackedAutoencoder
from .unetpp import UNetPlusPlus

_LOCK = threading.Lock()


def resolve_device() -> torch.device:
    if config.DEVICE != "auto":
        return torch.device(config.DEVICE)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


DEVICE = resolve_device()


@dataclass
class ModelStatus:
    name: str
    loaded: bool
    path: str
    detail: str = ""
    metrics: Optional[dict] = None


class _Registry:
    def __init__(self) -> None:
        self._unet: Optional[UNetPlusPlus] = None
        self._classifier: Optional[LeafClassifier] = None
        self._encoder: Optional[SparseStackedAutoencoder] = None
        self._tsne: Optional[dict] = None
        self._status: dict = {}
        self._class_names = list(CLASS_NAMES)

    # ------------------------------------------------------------------ U-Net++
    def unet(self) -> Optional[UNetPlusPlus]:
        if self._unet is not None:
            return self._unet
        with _LOCK:
            if self._unet is not None:
                return self._unet
            path = Path(config.UNET_WEIGHTS)
            if not path.exists():
                self._status["unetpp"] = ModelStatus(
                    "unetpp", False, str(path), "checkpoint not found - using classical fallback"
                )
                return None
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            widths = tuple(ckpt.get("widths", (32, 64, 128, 256, 512)))
            model = UNetPlusPlus(widths=widths, deep_supervision=False)
            model.load_state_dict(ckpt["state_dict"], strict=True)
            model.eval().to(DEVICE)
            self._unet = model
            self._status["unetpp"] = ModelStatus(
                "unetpp", True, str(path), f"widths={widths}", ckpt.get("metrics")
            )
            return model

    # -------------------------------------------------------------- Classifier
    def classifier(self) -> Optional[LeafClassifier]:
        if self._classifier is not None:
            return self._classifier
        with _LOCK:
            if self._classifier is not None:
                return self._classifier
            path = Path(config.CLASSIFIER_WEIGHTS)
            if not path.exists():
                self._status["classifier"] = ModelStatus(
                    "classifier", False, str(path), "checkpoint not found - run training first"
                )
                return None
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            arch = ckpt.get("arch", "efficientnet_b0")
            names = ckpt.get("class_names", CLASS_NAMES)
            model = LeafClassifier(num_classes=len(names), arch=arch, pretrained=False)
            model.load_state_dict(ckpt["state_dict"], strict=True)
            # The resolution the checkpoint was trained at; inference must match
            # it, otherwise features land at the wrong scale.
            model.input_size = int(ckpt.get("size", config.CLS_SIZE))
            model.eval().to(DEVICE)
            self._classifier = model
            self._class_names = list(names)
            self._status["classifier"] = ModelStatus(
                "classifier", True, str(path),
                f"arch={arch} input={model.input_size}px", ckpt.get("metrics")
            )
            return model

    def classifier_input_size(self) -> int:
        model = self.classifier()
        return int(getattr(model, "input_size", config.CLS_SIZE)) if model else config.CLS_SIZE

    # ----------------------------------------------------------- Sparse encoder
    def encoder(self) -> Optional[SparseStackedAutoencoder]:
        if self._encoder is not None:
            return self._encoder
        with _LOCK:
            if self._encoder is not None:
                return self._encoder
            path = Path(config.ENCODER_WEIGHTS)
            if not path.exists():
                self._status["sparse_encoder"] = ModelStatus(
                    "sparse_encoder", False, str(path), "checkpoint not found"
                )
                return None
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            model = SparseStackedAutoencoder(
                input_dim=ckpt["input_dim"],
                hidden_dims=tuple(ckpt.get("hidden_dims", (2048, 1024, 512))),
                latent_dim=ckpt.get("latent_dim", 256),
            )
            model.load_state_dict(ckpt["state_dict"], strict=True)
            model.eval().to(DEVICE)
            # Standardisation statistics used during training travel with the
            # checkpoint so inference scales features identically.
            if "feature_mean" in ckpt and "feature_std" in ckpt:
                model.feature_stats = (
                    np.asarray(ckpt["feature_mean"], dtype=np.float32),
                    np.asarray(ckpt["feature_std"], dtype=np.float32),
                )
            self._encoder = model
            self._status["sparse_encoder"] = ModelStatus(
                "sparse_encoder", True, str(path),
                f"{ckpt['input_dim']}d -> {ckpt.get('latent_dim', 256)}d",
                ckpt.get("metrics"),
            )
            return model

    # ------------------------------------------------------------ t-SNE reference
    def tsne_reference(self) -> Optional[dict]:
        if self._tsne is not None:
            return self._tsne
        with _LOCK:
            if self._tsne is not None:
                return self._tsne
            path = Path(config.TSNE_REFERENCE)
            if not path.exists():
                self._status["tsne"] = ModelStatus("tsne", False, str(path), "reference not built")
                return None
            data = np.load(path, allow_pickle=True)
            self._tsne = {
                "points": data["points"],           # (N, 2) projected training samples
                "labels": data["labels"],           # (N,) class indices
                "embeddings": data["embeddings"],   # (N, D) latent codes for kNN projection
                "class_names": list(data["class_names"]),
            }
            self._status["tsne"] = ModelStatus(
                "tsne", True, str(path), f"{len(self._tsne['points'])} reference points"
            )
            return self._tsne

    # ----------------------------------------------------------------- Utilities
    def class_names(self) -> list:
        self.classifier()
        return self._class_names

    def status(self) -> dict:
        # Touch every loader so status reflects reality.
        self.unet()
        self.classifier()
        self.encoder()
        self.tsne_reference()
        for key in ("unetpp", "classifier", "sparse_encoder", "tsne"):
            self._status.setdefault(key, ModelStatus(key, False, "", "not initialised"))
        return {k: v.__dict__ for k, v in self._status.items()}

    def warmup(self) -> None:
        self.status()


registry = _Registry()
