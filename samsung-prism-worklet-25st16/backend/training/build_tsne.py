"""Build the t-SNE reference map (spec section 11).

Encodes the cached training features into 256-d latents, projects them to 2-D
with t-SNE, and stores both. At request time the API positions a new leaf by
distance-weighted interpolation over its nearest latent neighbours, because
t-SNE itself has no out-of-sample transform.

    python build_tsne.py --perplexity 30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.manifold import TSNE

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app import config                        # noqa: E402
from app.labels import CLASS_NAMES             # noqa: E402
from app.models.registry import registry       # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(Path(config.WEIGHTS_DIR) / "feature_cache.npz"))
    ap.add_argument("--out", default=str(config.TSNE_REFERENCE))
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--max-points", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cache = Path(args.cache)
    if not cache.exists():
        print(f"feature cache missing: {cache} - run train_encoder.py first")
        return 1

    encoder = registry.encoder()
    if encoder is None:
        print("sparse encoder checkpoint missing - run train_encoder.py first")
        return 1

    blob = np.load(cache)
    fused = np.concatenate([blob["deep"], blob["handcrafted"]], axis=1).astype(np.float32)
    fused = np.nan_to_num(fused, nan=0.0, posinf=0.0, neginf=0.0)
    labels = blob["labels"]

    if len(fused) > args.max_points:
        rng = np.random.default_rng(args.seed)
        keep = rng.choice(len(fused), size=args.max_points, replace=False)
        fused, labels = fused[keep], labels[keep]

    mean, std = encoder.feature_stats
    standardised = (fused - mean) / std

    with torch.no_grad():
        latents = encoder.encode(torch.from_numpy(standardised)).numpy().astype(np.float32)

    perplexity = min(args.perplexity, max(5.0, (len(latents) - 1) / 3))
    print(f"running t-SNE on {latents.shape} with perplexity {perplexity}", flush=True)
    points = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=args.seed,
    ).fit_transform(latents).astype(np.float32)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        points=points,
        labels=labels.astype(np.int32),
        embeddings=latents,
        class_names=np.asarray(CLASS_NAMES, dtype=object),
    )
    print(f"saved {out} with {len(points)} reference points")
    return 0


if __name__ == "__main__":
    sys.exit(main())
