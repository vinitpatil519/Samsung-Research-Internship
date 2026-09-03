"""Inference pipeline orchestration.

Runs Preprocess -> U-Net++ -> Feature extraction -> Sparse encoder ->
EfficientNet -> Grad-CAM -> t-SNE and records a visual output for every stage,
so the UI can show exactly what each block produced.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from . import config, imaging
from .labels import label_for_name
from .models.registry import registry
from .stages import classify, encoder, explain, features, preprocess, segment


@dataclass
class Stage:
    id: str
    title: str
    model: str                       # which block produced it
    summary: str
    image: Optional[str] = None      # base64 PNG data URL
    metrics: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    available: bool = True


class _Clock:
    """Per-stage wall-clock timing."""

    def __init__(self) -> None:
        self.start = time.perf_counter()

    def lap(self) -> float:
        now = time.perf_counter()
        elapsed = (now - self.start) * 1000.0
        self.start = now
        return round(elapsed, 2)


def analyse(raw: bytes, top_k: int = 3) -> Dict[str, Any]:
    total_start = time.perf_counter()
    clock = _Clock()
    stages: List[Stage] = []

    image = imaging.decode_upload(raw)
    original_shape = image.shape

    # ---------------------------------------------------------- 1. Input
    stages.append(Stage(
        id="input",
        title="Input Image",
        model="upload",
        summary=f"Uploaded leaf photo, {original_shape[1]}x{original_shape[0]} pixels, "
                f"EXIF orientation corrected.",
        image=imaging.to_data_url(image),
        metrics={"width": int(original_shape[1]), "height": int(original_shape[0])},
        duration_ms=clock.lap(),
    ))

    # ------------------------------------------------- 2-4. Preprocessing
    pre = preprocess.run(image)
    pre_ms = clock.lap()
    stages.append(Stage(
        id="resize",
        title="Resize 256x256",
        model="preprocessing",
        summary="Area-interpolated resize to the fixed 256x256 network input.",
        image=imaging.to_data_url(pre.resized),
        metrics={"size": config.SEG_SIZE},
        duration_ms=pre_ms / 3,
    ))
    stages.append(Stage(
        id="denoise",
        title="Gaussian Denoise",
        model="preprocessing",
        summary=f"{config.GAUSSIAN_KERNEL}x{config.GAUSSIAN_KERNEL} Gaussian filter, "
                f"sigma {config.GAUSSIAN_SIGMA}; mean pixel change {pre.noise_removed:.2f}.",
        image=imaging.to_data_url(pre.denoised),
        metrics={"noise_removed": round(pre.noise_removed, 3)},
        duration_ms=pre_ms / 3,
    ))
    stages.append(Stage(
        id="clahe",
        title="CLAHE Contrast",
        model="preprocessing",
        summary=f"Adaptive histogram equalisation on the LAB L channel; "
                f"contrast gain {pre.contrast_gain:.2f}x.",
        image=imaging.to_data_url(pre.equalised),
        metrics={"contrast_gain": round(pre.contrast_gain, 3),
                 "clip_limit": config.CLAHE_CLIP},
        duration_ms=pre_ms / 3,
    ))

    # ------------------------------------------------- 5-7. Segmentation
    seg = segment.run(pre.equalised, pre.seg_tensor)
    seg_ms = clock.lap()
    is_unet = seg.method == "unetpp"
    stages.append(Stage(
        id="segmentation_probability",
        title="U-Net++ Probability Map",
        model="unetpp" if is_unet else "classical fallback",
        summary=("Per-pixel leaf probability from the nested U-Net decoder."
                 if is_unet else
                 "No U-Net++ checkpoint loaded; Excess-Green + Otsu score shown instead."),
        image=imaging.to_data_url(imaging.colourise(seg.probability)),
        metrics={"method": seg.method, "mean_probability": round(float(seg.probability.mean()), 4)},
        duration_ms=seg_ms / 3,
        available=is_unet,
    ))
    stages.append(Stage(
        id="segmentation_mask",
        title="Binary Leaf Mask",
        model="unetpp" if is_unet else "classical fallback",
        summary=f"Thresholded and morphologically cleaned mask; "
                f"leaf covers {seg.leaf_ratio * 100:.1f}% of the frame.",
        image=imaging.to_data_url((seg.mask * 255).astype(np.uint8)),
        metrics={"leaf_ratio": round(seg.leaf_ratio, 4), "components": seg.components},
        duration_ms=seg_ms / 3,
    ))

    masked = imaging.apply_mask(pre.equalised, seg.mask)
    outlined = imaging.draw_mask_outline(pre.equalised, seg.mask)
    stages.append(Stage(
        id="masked_leaf",
        title="Background Removed",
        model="segmentation output",
        summary="Leaf pixels kept, background zeroed - this image feeds the classifier.",
        image=imaging.to_data_url(masked),
        metrics={"outline_preview": imaging.to_data_url(outlined, max_side=256)},
        duration_ms=seg_ms / 3,
    ))

    # ----------------------------------------------- 8. Feature extraction
    feat = features.run(masked, seg.mask)
    feat_ms = clock.lap()
    stages.append(Stage(
        id="gabor",
        title="Gabor Filter Bank",
        model="handcrafted features",
        summary=f"{len(features.GABOR_FREQUENCIES)} frequencies x "
                f"{len(features.GABOR_THETAS)} orientations; peak response per pixel.",
        image=imaging.to_data_url(imaging.colourise(feat.gabor_energy, cv2.COLORMAP_MAGMA)),
        metrics={"dimensions": features.GABOR_DIM,
                 "mean_energy": round(float(feat.gabor_energy.mean()), 4)},
        duration_ms=feat_ms / 3,
    ))
    stages.append(Stage(
        id="texture",
        title="GLCM Texture Map",
        model="handcrafted features",
        summary=f"Local entropy over the leaf plus {features.GLCM_DIM} grey-level "
                f"co-occurrence statistics.",
        image=imaging.to_data_url(imaging.colourise(feat.texture_entropy, cv2.COLORMAP_CIVIDIS)),
        metrics={"glcm_dimensions": features.GLCM_DIM,
                 "mean_entropy": round(float(feat.texture_entropy.mean()), 4)},
        duration_ms=feat_ms / 3,
    ))
    hist_img = np.vstack([
        imaging.histogram_image(feat.colour_histograms["hue"], (16, 185, 129)),
        imaging.histogram_image(feat.colour_histograms["saturation"], (37, 99, 235)),
        imaging.histogram_image(feat.colour_histograms["value"], (100, 116, 139)),
    ])
    stages.append(Stage(
        id="colour",
        title="Colour Histogram + Shape",
        model="handcrafted features",
        summary=f"HSV histograms ({features.HIST_BINS} bins each) over leaf pixels and "
                f"{features.SHAPE_DIM} shape descriptors.",
        image=imaging.to_data_url(hist_img, max_side=420),
        metrics={
            "handcrafted_dimensions": features.HANDCRAFTED_DIM,
            "circularity": round(feat.shape_metrics.get("circularity", 0.0), 4),
            "solidity": round(feat.shape_metrics.get("solidity", 0.0), 4),
            "aspect_ratio": round(feat.shape_metrics.get("aspect_ratio", 0.0), 4),
        },
        duration_ms=feat_ms / 3,
    ))

    # ------------------------------------------------- 9. Classification
    cls_tensor = preprocess.to_tensor(masked, registry.classifier_input_size())
    cls = classify.run(cls_tensor, top_k=top_k)
    cls_ms = clock.lap()

    # -------------------------------------------- 10. Sparse encoder stage
    enc = encoder.run(cls.embedding, feat.handcrafted)
    enc_ms = clock.lap()
    stages.append(Stage(
        id="sparse_encoder",
        title="Sparse Stacked Encoder",
        model="sparse autoencoder",
        summary=(f"{enc.input_dim}-d fused features compressed to {enc.latent_dim}-d latent; "
                 f"{enc.sparsity * 100:.0f}% of units near zero."
                 if enc.available else enc.detail),
        image=imaging.to_data_url(imaging.latent_strip(enc.latent)) if enc.available else None,
        metrics={
            "input_dim": enc.input_dim,
            "latent_dim": enc.latent_dim,
            "sparsity": round(enc.sparsity, 4),
            "reconstruction_error": round(enc.reconstruction_error, 6),
        },
        duration_ms=enc_ms,
        available=enc.available,
    ))

    top = cls.predictions[0] if cls.predictions else None
    stages.append(Stage(
        id="classifier",
        title="EfficientNet Classification",
        model="efficientnet",
        summary=(f"{top.plant} - {top.disease} at {top.probability:.2f}% confidence."
                 if top else cls.detail),
        image=None,
        metrics={
            "entropy_bits": round(cls.entropy, 4),
            "margin": round(cls.margin, 2),
            "num_classes": len(registry.class_names()),
        },
        duration_ms=cls_ms,
        available=cls.available,
    ))

    # ------------------------------------------------- 11. Grad-CAM
    if cls.available and top is not None:
        cam = explain.gradcam(cls_tensor, top.index, mask=seg.mask)
    else:
        cam = explain.GradCamResult(False, "classifier unavailable")
    cam_ms = clock.lap()
    stages.append(Stage(
        id="gradcam",
        title="Grad-CAM Attention",
        model="efficientnet explainability",
        summary=(f"Regions driving the prediction; {cam.focus_ratio * 100:.1f}% of leaf area "
                 f"strongly activated." if cam.available else cam.detail),
        image=imaging.to_data_url(imaging.overlay_heatmap(masked, cam.cam)) if cam.available else None,
        metrics={"focus_ratio": round(cam.focus_ratio, 4)},
        duration_ms=cam_ms,
        available=cam.available,
    ))

    # ------------------------------------------------- 12. t-SNE placement
    tsne = explain.tsne_placement(enc.latent if enc.available else None)
    tsne_ms = clock.lap()
    stages.append(Stage(
        id="tsne",
        title="t-SNE Embedding",
        model="latent space",
        summary=("Uploaded leaf placed among training clusters by nearest latent neighbours."
                 if tsne.available else tsne.detail),
        image=None,
        metrics={"neighbours": tsne.neighbour_classes or []},
        duration_ms=tsne_ms,
        available=tsne.available,
    ))

    # ------------------------------------------------------- Final payload
    result: Dict[str, Any] = {
        "available": cls.available,
        "detail": cls.detail,
        "stages": [s.__dict__ for s in stages],
        "models": registry.status(),
        "timing_ms": round((time.perf_counter() - total_start) * 1000.0, 2),
    }

    if cls.available and top is not None:
        info = label_for_name(top.class_name)
        result.update({
            "plant": info.plant,
            "disease": info.disease,
            "pathogen": info.pathogen,
            "healthy": info.healthy,
            "confidence": round(top.probability, 2),
            "class_name": top.class_name,
            "description": info.description,
            "treatment": info.treatment,
            "top_predictions": [
                {
                    "class_name": p.class_name,
                    "plant": p.plant,
                    "disease": p.disease,
                    "pathogen": p.pathogen,
                    "confidence": round(p.probability, 2),
                    "healthy": p.healthy,
                }
                for p in cls.predictions
            ],
            "segmentation": {"method": seg.method, "leaf_ratio": round(seg.leaf_ratio, 4)},
            "uncertainty": {"entropy_bits": round(cls.entropy, 4), "margin": round(cls.margin, 2)},
        })

    if tsne.available:
        result["tsne"] = {
            "sample": tsne.sample,
            "points": tsne.points.tolist(),
            "labels": tsne.labels.tolist(),
            "class_names": tsne.class_names,
        }

    return result


def segment_only(raw: bytes) -> Dict[str, Any]:
    """POST /segment - segmentation without the classification head."""
    image = imaging.decode_upload(raw)
    pre = preprocess.run(image)
    seg = segment.run(pre.equalised, pre.seg_tensor)
    masked = imaging.apply_mask(pre.equalised, seg.mask)
    return {
        "method": seg.method,
        "leaf_ratio": round(seg.leaf_ratio, 4),
        "components": seg.components,
        "mask": imaging.to_data_url((seg.mask * 255).astype(np.uint8)),
        "probability": imaging.to_data_url(imaging.colourise(seg.probability)),
        "masked": imaging.to_data_url(masked),
    }
