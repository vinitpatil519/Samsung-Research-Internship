"""Train U-Net++ for leaf/background segmentation (spec section 6).

Ground-truth masks come free from PlantVillage's `segmented` tree, and half the
training images are recomposited onto synthetic backgrounds so the network
learns to cut leaves out of cluttered real-world photos, not just grey studio
sheets.

    python train_unet.py --epochs 4 --per-class 120 --size 192
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app import config                                     # noqa: E402
from app.labels import CLASS_NAMES                          # noqa: E402
from app.models.unetpp import (                             # noqa: E402
    UNetPlusPlus, bce_dice_loss, dice_coefficient, iou_score,
)
from training.dataset import (                              # noqa: E402
    SegmentationDataset, build_samples, stratified_split,
)


def evaluate(model, loader, device) -> dict:
    model.eval()
    dice_total = iou_total = loss_total = 0.0
    batches = 0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            loss_total += bce_dice_loss(logits, masks).item()
            dice_total += dice_coefficient(logits, masks)
            iou_total += iou_score(logits, masks)
            batches += 1
    batches = max(batches, 1)
    return {
        "val_loss": loss_total / batches,
        "dice": dice_total / batches,
        "iou": iou_total / batches,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default=str(config.DATASET_DIR))
    ap.add_argument("--out", default=str(config.UNET_WEIGHTS))
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--per-class", type=int, default=120, help="cap images per class")
    ap.add_argument("--size", type=int, default=192)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    ap.add_argument("--widths", default="16,32,64,128,256")
    ap.add_argument("--time-budget-min", type=float, default=0.0,
                    help="stop after this many minutes (0 = no limit)")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    widths = tuple(int(w) for w in args.widths.split(","))

    samples = build_samples(Path(args.dataset_dir), CLASS_NAMES)
    if not samples:
        print("no paired color/segmented samples found - run download_dataset.py first")
        return 1
    train_samples, val_samples = stratified_split(samples, val_fraction=0.12,
                                                  per_class_cap=args.per_class)
    print(f"device={device} train={len(train_samples)} val={len(val_samples)} widths={widths}")

    train_loader = DataLoader(
        SegmentationDataset(train_samples, size=args.size, train=True),
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True,
    )
    val_loader = DataLoader(
        SegmentationDataset(val_samples, size=args.size, train=False, composite_prob=0.5, seed=7),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
    )

    model = UNetPlusPlus(widths=widths, deep_supervision=False).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=max(1, args.epochs * len(train_loader))
    )

    best_dice = -1.0
    started = time.time()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stop = False

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, (images, masks) in enumerate(train_loader, 1):
            images, masks = images.to(device), masks.to(device)
            optimiser.zero_grad(set_to_none=True)
            loss = bce_dice_loss(model(images), masks)
            loss.backward()
            optimiser.step()
            scheduler.step()
            running += loss.item()
            if step % 20 == 0:
                elapsed = (time.time() - started) / 60
                print(f"epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss {running / step:.4f} elapsed {elapsed:.1f}m", flush=True)
            if args.time_budget_min and (time.time() - started) / 60 > args.time_budget_min:
                print("time budget reached - stopping early")
                stop = True
                break

        metrics = evaluate(model, val_loader, device)
        metrics["train_loss"] = running / max(1, len(train_loader))
        metrics["epoch"] = epoch
        print(f"[epoch {epoch}] {json.dumps({k: round(v, 4) for k, v in metrics.items()})}",
              flush=True)

        if metrics["dice"] > best_dice:
            best_dice = metrics["dice"]
            torch.save({
                "state_dict": model.state_dict(),
                "widths": widths,
                "size": args.size,
                "metrics": {k: round(float(v), 5) for k, v in metrics.items()},
            }, out_path)
            print(f"saved {out_path} (dice {best_dice:.4f})", flush=True)
        if stop:
            break

    print(f"DONE best_dice={best_dice:.4f} minutes={(time.time() - started) / 60:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
