"""Stage 6: Auto Sparse Stacked Encoder (spec section 8).

Compresses [deep embedding | handcrafted features] into a 256-d latent code.
Feature standardisation statistics travel inside the encoder checkpoint so
inference reproduces the training scaling exactly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from ..models.registry import DEVICE, registry


@dataclass
class EncoderResult:
    available: bool
    detail: str
    latent: Optional[np.ndarray] = None
    input_dim: int = 0
    latent_dim: int = 0
    sparsity: float = 0.0             # fraction of latent units below 0.05
    reconstruction_error: float = 0.0  # MSE on the standardised input


def build_input(deep: np.ndarray, handcrafted: np.ndarray) -> np.ndarray:
    """Concatenation contract shared with train_encoder.py."""
    return np.concatenate([deep.ravel(), handcrafted.ravel()]).astype(np.float32)


def run(deep: Optional[np.ndarray], handcrafted: np.ndarray) -> EncoderResult:
    model = registry.encoder()
    if model is None:
        return EncoderResult(False, "sparse encoder checkpoint missing")
    if deep is None:
        return EncoderResult(False, "deep embedding unavailable (classifier not loaded)")

    vector = build_input(deep, handcrafted)
    if vector.size != model.input_dim:
        return EncoderResult(
            False,
            f"feature dimension mismatch: got {vector.size}, checkpoint expects {model.input_dim}",
        )

    stats = getattr(model, "feature_stats", None)
    if stats is not None:
        mean, std = stats
        vector = (vector - mean) / std

    with torch.no_grad():
        x = torch.from_numpy(vector).unsqueeze(0).to(DEVICE)
        recon, latent = model(x)
        error = float(torch.nn.functional.mse_loss(recon, x).item())
        latent_np = latent[0].cpu().numpy().astype(np.float32)

    return EncoderResult(
        available=True,
        detail=f"{model.input_dim}d -> {model.latent_dim}d",
        latent=latent_np,
        input_dim=int(model.input_dim),
        latent_dim=int(model.latent_dim),
        sparsity=float((latent_np < 0.05).mean()),
        reconstruction_error=error,
    )
