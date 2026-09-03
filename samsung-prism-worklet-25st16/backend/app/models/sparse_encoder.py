"""Auto Sparse Stacked Encoder (spec section 8).

Compresses the concatenated deep + handcrafted feature vector down to a
256-dimensional latent code. Sparsity comes from an L1 penalty on the latent
activations plus a KL divergence term against a low target activation rate.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SparseStackedAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple = (2048, 1024, 512),
        latent_dim: int = 256,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        encoder_layers = []
        prev = input_dim
        for dim in hidden_dims:
            encoder_layers += [nn.Linear(prev, dim), nn.BatchNorm1d(dim), nn.ReLU(inplace=True)]
            prev = dim
        # Sigmoid latent keeps activations in [0, 1] so the KL sparsity term is valid.
        encoder_layers += [nn.Linear(prev, latent_dim), nn.Sigmoid()]
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        prev = latent_dim
        for dim in reversed(hidden_dims):
            decoder_layers += [nn.Linear(prev, dim), nn.BatchNorm1d(dim), nn.ReLU(inplace=True)]
            prev = dim
        decoder_layers += [nn.Linear(prev, input_dim)]
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon, latent


def kl_sparsity(latent: torch.Tensor, rho: float = 0.05, eps: float = 1e-8) -> torch.Tensor:
    """KL divergence between the target activation rate rho and the observed mean."""
    rho_hat = latent.mean(dim=0).clamp(eps, 1 - eps)
    rho_t = torch.full_like(rho_hat, rho)
    return (
        rho_t * torch.log(rho_t / rho_hat)
        + (1 - rho_t) * torch.log((1 - rho_t) / (1 - rho_hat))
    ).sum()


def sparse_ae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    latent: torch.Tensor,
    l1_weight: float = 1e-4,
    kl_weight: float = 1e-3,
    rho: float = 0.05,
):
    mse = nn.functional.mse_loss(recon, target)
    l1 = latent.abs().mean()
    kl = kl_sparsity(latent, rho)
    total = mse + l1_weight * l1 + kl_weight * kl
    return total, {"mse": mse.item(), "l1": l1.item(), "kl": kl.item()}
