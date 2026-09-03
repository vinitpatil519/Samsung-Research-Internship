"""Train the Auto Sparse Stacked Encoder (spec section 8).

Step 1 extracts the exact feature vector the inference pipeline builds -
EfficientNet embedding concatenated with GLCM, Gabor, colour and shape features
- and caches it. Step 2 trains the autoencoder on those cached vectors with L1 +
KL sparsity.

    python train_encoder.py --per-class 40 --epochs 60
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app import config                                       # noqa: E402
from app.labels import CLASS_NAMES                            # noqa: E402
from app.models.registry import DEVICE, registry              # noqa: E402
from app.models.sparse_encoder import (                       # noqa: E402
    SparseStackedAutoencoder, sparse_ae_loss,
)
from app.stages import features as feature_stage              # noqa: E402
from app.stages import preprocess, segment                    # noqa: E402
from training.dataset import build_samples, read_rgb, stratified_split  # noqa: E402


def extract_features(samples, cache_path: Path, limit_log: int = 100) -> dict:
    """Run the real inference pipeline over training images and cache vectors."""
    classifier = registry.classifier()
    if classifier is None:
        raise SystemExit("classifier checkpoint required - train the classifier first")

    deep_list, hand_list, labels = [], [], []
    started = time.time()
    for i, sample in enumerate(samples, 1):
        image = read_rgb(sample.color)
        pre = preprocess.run(image)
        seg = segment.run(pre.equalised, pre.seg_tensor)

        masked = pre.equalised * (seg.mask[..., None] > 0)
        feats = feature_stage.run(masked.astype(np.uint8), seg.mask)

        cls_tensor = preprocess.to_tensor(masked.astype(np.uint8), registry.classifier_input_size())
        with torch.no_grad():
            out = classifier(cls_tensor.to(DEVICE))
        deep_list.append(out.embedding[0].cpu().numpy().astype(np.float32))
        hand_list.append(feats.handcrafted)
        labels.append(sample.label)

        if i % limit_log == 0:
            rate = i / (time.time() - started)
            print(f"features {i}/{len(samples)} ({rate:.1f} img/s)", flush=True)

    data = {
        "deep": np.stack(deep_list),
        "handcrafted": np.stack(hand_list),
        "labels": np.asarray(labels, dtype=np.int64),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **data)
    print(f"cached features to {cache_path}: deep={data['deep'].shape} "
          f"handcrafted={data['handcrafted'].shape}", flush=True)
    return data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default=str(config.DATASET_DIR))
    ap.add_argument("--out", default=str(config.ENCODER_WEIGHTS))
    ap.add_argument("--cache", default=str(Path(config.WEIGHTS_DIR) / "feature_cache.npz"))
    ap.add_argument("--per-class", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--latent-dim", type=int, default=256)
    ap.add_argument("--hidden", default="2048,1024,512")
    ap.add_argument("--l1", type=float, default=1e-4)
    ap.add_argument("--kl", type=float, default=1e-3)
    ap.add_argument("--rho", type=float, default=0.05)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--rebuild-cache", action="store_true")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    cache_path = Path(args.cache)
    if cache_path.exists() and not args.rebuild_cache:
        blob = np.load(cache_path)
        data = {k: blob[k] for k in ("deep", "handcrafted", "labels")}
        print(f"loaded cached features {data['deep'].shape}", flush=True)
    else:
        samples = build_samples(Path(args.dataset_dir), CLASS_NAMES)
        if not samples:
            print("no samples found - run download_dataset.py first")
            return 1
        subset, _ = stratified_split(samples, val_fraction=0.001, per_class_cap=args.per_class)
        print(f"extracting features from {len(subset)} images", flush=True)
        data = extract_features(subset, cache_path)

    fused = np.concatenate([data["deep"], data["handcrafted"]], axis=1).astype(np.float32)
    fused = np.nan_to_num(fused, nan=0.0, posinf=0.0, neginf=0.0)

    mean = fused.mean(axis=0)
    std = fused.std(axis=0)
    std[std < 1e-6] = 1.0
    standardised = (fused - mean) / std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden = tuple(int(h) for h in args.hidden.split(","))
    model = SparseStackedAutoencoder(
        input_dim=standardised.shape[1], hidden_dims=hidden, latent_dim=args.latent_dim,
    ).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    tensor = torch.from_numpy(standardised)
    loader = DataLoader(TensorDataset(tensor), batch_size=args.batch_size,
                        shuffle=True, drop_last=len(tensor) > args.batch_size)

    print(f"training encoder: {standardised.shape[1]}d -> {args.latent_dim}d "
          f"on {len(tensor)} vectors", flush=True)
    last = {}
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {"loss": 0.0, "mse": 0.0, "l1": 0.0, "kl": 0.0}
        batches = 0
        for (batch,) in loader:
            batch = batch.to(device)
            optimiser.zero_grad(set_to_none=True)
            recon, latent = model(batch)
            loss, parts = sparse_ae_loss(recon, batch, latent, args.l1, args.kl, args.rho)
            loss.backward()
            optimiser.step()
            totals["loss"] += loss.item()
            for k, v in parts.items():
                totals[k] += v
            batches += 1
        batches = max(batches, 1)
        last = {k: v / batches for k, v in totals.items()}
        if epoch % 10 == 0 or epoch == args.epochs:
            print(f"[epoch {epoch}] " + " ".join(f"{k}={v:.5f}" for k, v in last.items()),
                  flush=True)

    model.eval()
    with torch.no_grad():
        latents = model.encode(tensor.to(device)).cpu().numpy()
    sparsity = float((latents < 0.05).mean())

    out_path = Path(args.out)
    torch.save({
        "state_dict": model.state_dict(),
        "input_dim": int(standardised.shape[1]),
        "hidden_dims": hidden,
        "latent_dim": int(args.latent_dim),
        "feature_mean": mean,
        "feature_std": std,
        "metrics": {**{k: round(float(v), 6) for k, v in last.items()},
                    "latent_sparsity": round(sparsity, 4)},
    }, out_path)
    print(f"saved {out_path} (latent sparsity {sparsity:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
