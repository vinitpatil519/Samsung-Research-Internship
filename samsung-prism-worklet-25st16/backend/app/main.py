"""FastAPI application (spec section 13).

Endpoints:
    GET  /health   - liveness plus which checkpoints are loaded
    GET  /classes  - the 38 bilingual class cards
    POST /predict  - full pipeline with a visual output per stage
    POST /segment  - segmentation only
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from . import config, pipeline
from .labels import CLASS_NAMES, LABELS
from .models.registry import DEVICE, registry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("worklet25st16")

app = FastAPI(
    title="Samsung Research (Prism) Worklet:25ST16 API",
    version="1.0.0",
    description="U-Net++ segmentation, sparse encoding and EfficientNet leaf disease diagnosis.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in config.ALLOWED_ORIGINS if o.strip()] or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _warmup() -> None:
    status = registry.status()
    logger.info("device=%s", DEVICE)
    for name, info in status.items():
        logger.info("model %s loaded=%s %s", name, info["loaded"], info["detail"])


async def _read_upload(file: UploadFile) -> bytes:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty upload")
    if len(raw) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"image larger than {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
        )
    return raw


@app.get("/health")
def health() -> dict:
    status = registry.status()
    return {
        "status": "ok",
        "device": str(DEVICE),
        "models": status,
        "ready": status["classifier"]["loaded"],
    }


@app.get("/classes")
def classes() -> dict:
    return {
        "count": len(CLASS_NAMES),
        "classes": [LABELS[name].to_dict() for name in CLASS_NAMES],
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = 3) -> dict:
    raw = await _read_upload(file)
    try:
        result = pipeline.analyse(raw, top_k=top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("prediction failed")
        raise HTTPException(status_code=500, detail=f"inference failed: {exc}") from exc

    if not result.get("available"):
        # Pipeline ran, but no classifier weights: return the stages plus the reason.
        raise HTTPException(
            status_code=503,
            detail={"message": result.get("detail", "model unavailable"),
                    "models": result.get("models")},
        )
    return result


@app.post("/segment")
async def segment(file: UploadFile = File(...)) -> dict:
    raw = await _read_upload(file)
    try:
        return pipeline.segment_only(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
