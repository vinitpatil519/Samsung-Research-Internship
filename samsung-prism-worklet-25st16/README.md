# Samsung Research (Prism) Worklet:25ST16

End-to-end plant leaf disease diagnosis: **U-Net++ segmentation → deep + handcrafted feature
fusion → Auto Sparse Stacked Encoder → EfficientNet classification → Grad-CAM and t-SNE**, with a
Next.js UI that shows the output of *every* stage alongside the diagnosis and its uncertainty.

```
Upload/Camera → Resize → Gaussian → CLAHE → U-Net++ mask → Masked leaf
              → GLCM + Gabor + Colour + Shape → Sparse encoder (256-d)
              → EfficientNet (38 classes) → Grad-CAM → t-SNE placement
```

## Repository layout

```
samsung-prism-worklet-25st16/
├── backend/
│   ├── app/
│   │   ├── main.py            FastAPI app: /health /classes /predict /segment
│   │   ├── pipeline.py        stage orchestration, one visual output per stage
│   │   ├── labels.py          38 classes with pathogen, symptoms, management
│   │   ├── imaging.py         base64 previews, heatmaps, overlays
│   │   ├── config.py          paths, image sizes, preprocessing constants
│   │   ├── models/            unetpp.py, classifier.py, sparse_encoder.py, registry.py
│   │   └── stages/            preprocess, segment, features, encoder, classify, explain
│   ├── training/              download_dataset, dataset, train_unet, train_classifier,
│   │                          train_encoder, build_tsne
│   ├── tests/smoke.py         runs the whole pipeline on a synthetic leaf
│   └── requirements.txt
├── frontend/                  Next.js 15 + TypeScript + Tailwind v4 + Framer Motion + Recharts
├── weights/                   checkpoints (gitignored)
├── datasets/                  PlantVillage (gitignored)
├── docs/                      PROJECT_QA.md, methodology
└── docker-compose.yml
```

## Quick start

### 1. Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

`GET /health` reports which checkpoints are loaded. Without a classifier checkpoint the pipeline
still runs and returns preprocessing, segmentation and feature stages; `/predict` then answers
`503` with the reason instead of inventing a diagnosis.

### 2. Frontend

```bash
cd frontend
npm install
npm run dev          # http://localhost:3000
```

Set `NEXT_PUBLIC_API_URL` in `frontend/.env.local` if the backend is not on `localhost:8000`.

### 3. Docker

```bash
docker compose up --build
```

## Training the models

```bash
cd backend/training

# 1. PlantVillage (2.2 GB, resumable; extracts color/ and segmented/)
python download_dataset.py

# 2. Segmentation - masks come from the segmented tree, backgrounds are randomised
python train_unet.py --epochs 3 --per-class 50 --size 160 --threads 10

# 3. Classification - trained on background-removed leaves, as at inference
python train_classifier.py --epochs 4 --per-class 260 --size 160 --arch efficientnet_b0 --threads 10

# 4. Sparse encoder - extracts the real fused feature vector, then trains the autoencoder
python train_encoder.py --per-class 40 --epochs 80 --threads 10

# 5. t-SNE reference map used to place new uploads
python build_tsne.py
```

Every script writes to `weights/` and the backend picks checkpoints up on the next start. All
scripts accept `--time-budget-min` or `--per-class` to fit a CPU-only machine; add `--arch
efficientnet_b3` and raise `--per-class` on a GPU for the full spec configuration.

Optimisers from the spec are selectable: `--optimizer adamw|sgd|rmsprop|lion`, schedulers
`--scheduler cosine|plateau`, early stopping via `--patience`.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | liveness, device, per-model load status |
| GET | `/classes` | all 38 classes with pathogen, symptom and management metadata |
| POST | `/predict` | full pipeline; returns diagnosis + every stage image |
| POST | `/segment` | segmentation only (mask, probability map, masked leaf) |

`POST /predict` response (abridged):

```json
{
  "plant": "Tomato",
  "disease": "Early Blight",
  "pathogen": "Alternaria solani (fungus)",
  "healthy": false,
  "confidence": 98.42,
  "top_predictions": [{"class_name": "Tomato___Early_blight", "confidence": 98.42}],
  "description": "Brown spots with concentric target-like rings ...",
  "treatment": "Rotate crops, mulch, stake plants ...",
  "segmentation": {"method": "unetpp", "leaf_ratio": 0.41},
  "uncertainty": {"entropy_bits": 0.21, "margin": 96.8},
  "stages": [{"id": "clahe", "image": "data:image/png;base64,...", "metrics": {}}],
  "tsne": {"sample": [1.2, -4.5], "points": [[...]], "labels": [3]}
}
```

## Mobile capture

The analyzer uses `<input type="file" accept="image/*" capture="environment">`, which opens the rear
camera directly on both Android and iOS, and falls back to the gallery/file picker elsewhere. EXIF
orientation is corrected server-side so portrait phone photos are not rotated.

## Delivered checkpoint results

Trained on this machine, CPU only (no CUDA build), so B0 @ 160 px with a per-class cap rather than
the spec's B3 @ 256 px on the full 54k images.

| Model | Config | Result |
| --- | --- | --- |
| U-Net++ | 3 epochs, 1,628 train images, 160 px | **Dice 0.972**, IoU 0.946 |
| EfficientNet-B0 | 4 epochs, 8,086 train images, 160 px | **Accuracy 97.83%**, top-3 99.86%, macro F1 0.952 |
| Sparse encoder | 1,486-d → 256-d, 80 epochs | recon MSE 0.074, 65% latent units near zero |
| t-SNE reference | 1,443 latent codes | clean per-species clusters |

End-to-end through the live API on 40 held-out images: **top-1 90.0%, top-3 100%**, ~1.5–2.3 s per
image on CPU. The gap to the 97.8% validation figure comes from validation using ground-truth masks
while the API uses U-Net++ predicted masks; training the classifier on predicted masks (or with mask
jitter augmentation) closes it.

## Honesty rules baked into the code

- No mocked predictions. If a checkpoint is missing, the affected stage reports `available: false`
  and says which script to run.
- If the U-Net++ checkpoint is missing, segmentation falls back to Excess-Green + Otsu and labels
  the stage `classical fallback`, never `unetpp`.
- Confidence is reported with its margin and predictive entropy, so a low-separation prediction is
  visible rather than hidden behind one big number.

## Documentation

- `docs/PROJECT_QA.md` - 27 deep questions and answers covering the whole project.
- `docs/methodology.md` - the pipeline written up as a method section.
