"""Fine-tune EfficientNet on background-removed PlantVillage leaves.

Spec sections 9 and 10: cross entropy with label smoothing, a choice of four
optimisers, cosine annealing, and early stopping. Training uses the `segmented`
tree so the model sees the same masked input the U-Net++ stage produces at
inference time.

    python train_classifier.py --epochs 4 --per-class 400 --size 192 --arch efficientnet_b0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app import config                                      # noqa: E402
from app.labels import CLASS_NAMES                           # noqa: E402
from app.models.classifier import LeafClassifier             # noqa: E402
from training.dataset import (                               # noqa: E402
    ClassificationDataset, build_samples, stratified_split,
)


def build_optimiser(name: str, params, lr: float, weight_decay: float):
    """Spec section 10 - all four optimisers are selectable."""
    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True,
                               weight_decay=weight_decay)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, alpha=0.9, momentum=0.9,
                                   eps=1e-3, weight_decay=weight_decay)
    if name == "lion":
        try:
            from lion_pytorch import Lion
        except ImportError as exc:
            raise SystemExit("pip install lion-pytorch to use --optimizer lion") from exc
        return Lion(params, lr=lr, weight_decay=weight_decay)
    raise SystemExit(f"unknown optimiser {name}")


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int) -> dict:
    model.eval()
    correct = total = 0
    top3 = 0
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for images, labels in loader:
        images = images.to(device)
        logits = model(images).logits.cpu()
        pred = logits.argmax(1)
        correct += (pred == labels).sum().item()
        total += labels.numel()
        top3 += (logits.topk(min(3, num_classes), dim=1).indices == labels[:, None]).any(1).sum().item()
        for t, p in zip(labels.tolist(), pred.tolist()):
            confusion[t, p] += 1

    # Macro precision / recall / F1 from the confusion matrix.
    tp = np.diag(confusion).astype(np.float64)
    fp = confusion.sum(0) - tp
    fn = confusion.sum(1) - tp
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) > 0)
    return {
        "accuracy": correct / max(1, total),
        "top3_accuracy": top3 / max(1, total),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "confusion": confusion,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default=str(config.DATASET_DIR))
    ap.add_argument("--out", default=str(config.CLASSIFIER_WEIGHTS))
    ap.add_argument("--arch", default="efficientnet_b0")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--per-class", type=int, default=400)
    ap.add_argument("--size", type=int, default=192)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--optimizer", default="adamw", choices=["adamw", "sgd", "rmsprop", "lion"])
    ap.add_argument("--scheduler", default="cosine", choices=["cosine", "plateau"])
    ap.add_argument("--patience", type=int, default=12, help="early-stopping patience")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--time-budget-min", type=float, default=0.0)
    ap.add_argument("--resume", action="store_true", help="continue from an existing checkpoint")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = build_samples(Path(args.dataset_dir), CLASS_NAMES)
    if not samples:
        print("no samples found - run download_dataset.py first")
        return 1
    train_samples, val_samples = stratified_split(samples, val_fraction=0.15,
                                                  per_class_cap=args.per_class)
    print(f"device={device} arch={args.arch} train={len(train_samples)} val={len(val_samples)}",
          flush=True)

    train_loader = DataLoader(
        ClassificationDataset(train_samples, size=args.size, train=True),
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True,
    )
    val_loader = DataLoader(
        ClassificationDataset(val_samples, size=args.size, train=False),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = LeafClassifier(num_classes=len(CLASS_NAMES), arch=args.arch,
                           pretrained=not args.resume)
    if args.resume and out_path.exists():
        ckpt = torch.load(out_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        print(f"resumed from {out_path}", flush=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimiser = build_optimiser(args.optimizer, model.parameters(), args.lr, args.weight_decay)
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=max(1, args.epochs * len(train_loader)))
        step_per_batch = True
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode="max", patience=1)
        step_per_batch = False

    best_acc = -1.0
    bad_epochs = 0
    started = time.time()
    stop = False

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for step, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.to(device), labels.to(device)
            optimiser.zero_grad(set_to_none=True)
            loss = criterion(model(images).logits, labels)
            loss.backward()
            optimiser.step()
            if step_per_batch:
                scheduler.step()
            running += loss.item()
            seen += 1
            if step % 25 == 0:
                print(f"epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss {running / seen:.4f} elapsed {(time.time() - started) / 60:.1f}m",
                      flush=True)
            if args.time_budget_min and (time.time() - started) / 60 > args.time_budget_min:
                print("time budget reached - stopping early", flush=True)
                stop = True
                break

        metrics = evaluate(model, val_loader, device, len(CLASS_NAMES))
        confusion = metrics.pop("confusion")
        metrics["train_loss"] = running / max(1, seen)
        metrics["epoch"] = epoch
        print(f"[epoch {epoch}] {json.dumps({k: round(float(v), 4) for k, v in metrics.items()})}",
              flush=True)

        if not step_per_batch:
            scheduler.step(metrics["accuracy"])

        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            bad_epochs = 0
            torch.save({
                "state_dict": model.state_dict(),
                "arch": args.arch,
                "class_names": CLASS_NAMES,
                "size": args.size,
                "metrics": {k: round(float(v), 5) for k, v in metrics.items()},
            }, out_path)
            np.save(out_path.with_suffix(".confusion.npy"), confusion)
            print(f"saved {out_path} (accuracy {best_acc:.4f})", flush=True)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("early stopping", flush=True)
                break
        if stop:
            break

    print(f"DONE best_accuracy={best_acc:.4f} minutes={(time.time() - started) / 60:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
