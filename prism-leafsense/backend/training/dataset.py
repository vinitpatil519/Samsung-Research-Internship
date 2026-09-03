"""Dataset plumbing shared by the training scripts.

PlantVillage ships three parallel trees: `color` (raw photo), `segmented`
(background already removed) and `grayscale`. We use color + segmented, which
gives a free ground-truth mask for every image: mask = non-black pixels of the
segmented file.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from app.config import SEG_SIZE
from app.stages.preprocess import enhance

IMAGENET_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)


@dataclass
class Sample:
    color: Path
    segmented: Path
    class_name: str
    label: int


def _index_tree(root: Path) -> dict:
    """class_name -> {stem: path} for one PlantVillage tree."""
    table: dict = {}
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        entries = {}
        for file in class_dir.iterdir():
            if file.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            # segmented files carry a `_final_masked` suffix; strip it to match.
            stem = file.stem.replace("_final_masked", "")
            entries[stem] = file
        table[class_dir.name] = entries
    return table


def build_samples(dataset_dir: Path, class_names: Sequence[str]) -> List[Sample]:
    """Pair every color image with its segmented counterpart."""
    root = Path(dataset_dir) / "plantvillage"
    colour_tree = _index_tree(root / "color")
    seg_tree = _index_tree(root / "segmented")

    index = {name: i for i, name in enumerate(class_names)}
    samples: List[Sample] = []
    for class_name, files in colour_tree.items():
        if class_name not in index:
            continue
        seg_files = seg_tree.get(class_name, {})
        for stem, colour_path in files.items():
            seg_path = seg_files.get(stem)
            if seg_path is None:
                continue
            samples.append(Sample(colour_path, seg_path, class_name, index[class_name]))
    samples.sort(key=lambda s: (s.class_name, s.color.name))
    return samples


def stratified_split(
    samples: Sequence[Sample],
    val_fraction: float = 0.15,
    per_class_cap: Optional[int] = None,
    seed: int = 1337,
) -> Tuple[List[Sample], List[Sample]]:
    """Class-balanced train/val split with an optional per-class cap.

    The cap keeps CPU training tractable and also flattens PlantVillage's 38x
    class imbalance (120 potato-healthy images vs 4527 orange HLB images).
    """
    rng = random.Random(seed)
    by_class: dict = {}
    for sample in samples:
        by_class.setdefault(sample.class_name, []).append(sample)

    train: List[Sample] = []
    val: List[Sample] = []
    for class_name in sorted(by_class):
        items = by_class[class_name][:]
        rng.shuffle(items)
        if per_class_cap:
            items = items[:per_class_cap]
        cut = max(1, int(len(items) * val_fraction))
        val.extend(items[:cut])
        train.extend(items[cut:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def read_rgb(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)      # handles non-ASCII paths on Windows
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"could not read {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def mask_from_segmented(segmented: np.ndarray) -> np.ndarray:
    """Non-black pixels of the segmented image are leaf."""
    gray = cv2.cvtColor(segmented, cv2.COLOR_RGB2GRAY)
    mask = (gray > 8).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def normalise(image: np.ndarray) -> np.ndarray:
    arr = image.astype(np.float32) / 255.0
    return ((arr - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)


def random_background(size: int, rng: random.Random) -> np.ndarray:
    """Synthetic clutter used to teach the segmenter real-world backgrounds.

    PlantVillage photographs every leaf on a uniform grey sheet. Compositing the
    leaf onto noisy, textured, coloured planes stops U-Net++ from learning
    'anything not grey is leaf'.
    """
    style = rng.randrange(4)
    if style == 0:                      # flat colour
        colour = np.array([rng.randrange(256) for _ in range(3)], dtype=np.float32)
        bg = np.ones((size, size, 3), dtype=np.float32) * colour
    elif style == 1:                    # vertical gradient
        top = np.array([rng.randrange(256) for _ in range(3)], dtype=np.float32)
        bottom = np.array([rng.randrange(256) for _ in range(3)], dtype=np.float32)
        ramp = np.linspace(0, 1, size, dtype=np.float32)[:, None, None]
        bg = top[None, None, :] * (1 - ramp) + bottom[None, None, :] * ramp
        bg = np.repeat(bg, size, axis=1)
    elif style == 2:                    # blurred noise (soil / gravel look)
        bg = np.random.default_rng(rng.randrange(1 << 30)).integers(
            0, 256, (size, size, 3)).astype(np.float32)
        bg = cv2.GaussianBlur(bg, (0, 0), rng.uniform(2, 9))
    else:                               # coarse green clutter (grass / canopy)
        small = np.random.default_rng(rng.randrange(1 << 30)).integers(
            0, 256, (max(2, size // 16), max(2, size // 16), 3)).astype(np.float32)
        small[..., 1] = np.clip(small[..., 1] * 1.3, 0, 255)
        bg = cv2.resize(small, (size, size), interpolation=cv2.INTER_CUBIC)
    return np.clip(bg, 0, 255).astype(np.uint8)


class SegmentationDataset(Dataset):
    """Color image + binary mask, with random-background compositing."""

    def __init__(self, samples: Sequence[Sample], size: int = 256,
                 train: bool = True, composite_prob: float = 0.5, seed: int = 0) -> None:
        self.samples = list(samples)
        self.size = size
        self.train = train
        self.composite_prob = composite_prob
        self.seed = seed

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        rng = random.Random((self.seed * 1_000_003 + idx) if not self.train else random.randrange(1 << 30))

        colour = cv2.resize(read_rgb(sample.color), (self.size, self.size), interpolation=cv2.INTER_AREA)
        segmented = cv2.resize(read_rgb(sample.segmented), (self.size, self.size), interpolation=cv2.INTER_AREA)
        mask = mask_from_segmented(segmented)

        if rng.random() < self.composite_prob:
            bg = random_background(self.size, rng)
            m = mask[..., None].astype(np.float32)
            colour = (colour.astype(np.float32) * m + bg.astype(np.float32) * (1 - m)).astype(np.uint8)

        if self.train:
            if rng.random() < 0.5:
                colour, mask = colour[:, ::-1].copy(), mask[:, ::-1].copy()
            if rng.random() < 0.5:
                colour, mask = colour[::-1].copy(), mask[::-1].copy()
            k = rng.randrange(4)
            if k:
                colour = np.rot90(colour, k).copy()
                mask = np.rot90(mask, k).copy()
            if rng.random() < 0.5:
                factor = rng.uniform(0.75, 1.3)
                colour = np.clip(colour.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        return (
            torch.from_numpy(normalise(colour)),
            torch.from_numpy(mask.astype(np.float32)).unsqueeze(0),
        )


class ClassificationDataset(Dataset):
    """Background-removed leaf + class index.

    The image is built through exactly the inference path: resize to 256,
    Gaussian denoise, CLAHE, then apply the leaf mask, then resize to the
    classifier input size. Only the mask source differs - ground truth here,
    U-Net++ prediction at inference - so the pixel statistics the classifier
    learns are the ones it will actually be served.
    """

    def __init__(self, samples: Sequence[Sample], size: int = 224, train: bool = True) -> None:
        self.samples = list(samples)
        self.size = size
        self.train = train

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        colour = read_rgb(sample.color)
        segmented = cv2.resize(read_rgb(sample.segmented), (SEG_SIZE, SEG_SIZE),
                               interpolation=cv2.INTER_AREA)
        mask = mask_from_segmented(segmented)

        _, _, equalised = enhance(colour, SEG_SIZE)
        masked = (equalised * (mask[..., None] > 0)).astype(np.uint8)
        image = cv2.resize(masked, (self.size, self.size), interpolation=cv2.INTER_AREA)

        if self.train:
            rng = random
            if rng.random() < 0.5:
                image = image[:, ::-1].copy()
            if rng.random() < 0.3:
                image = image[::-1].copy()
            k = rng.randrange(4)
            if k:
                image = np.rot90(image, k).copy()
            if rng.random() < 0.7:
                # Colour jitter in HSV; hue shift stays small so disease colour
                # cues survive.
                hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[..., 0] = (hsv[..., 0] + rng.uniform(-6, 6)) % 180
                hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.8, 1.25), 0, 255)
                hsv[..., 2] = np.clip(hsv[..., 2] * rng.uniform(0.75, 1.3), 0, 255)
                image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            if rng.random() < 0.3:
                image = cv2.GaussianBlur(image, (3, 3), rng.uniform(0.3, 1.2))

        return torch.from_numpy(normalise(image)), sample.label
