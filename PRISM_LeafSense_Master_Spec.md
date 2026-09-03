# PRISM LeafSense --- Master Project Specification

> End-to-end multimodal plant disease diagnosis platform combining
> **U-Net++ segmentation + EfficientNet classification + Auto Sparse
> Stacked Encoder + t-SNE explainability** with a research-oriented UI.

## 1. Objective

Rebuild the Samsung PRISM internship project as a production-quality
research application.

**Core workflow**

1.  User uploads a leaf image.
2.  U-Net++ isolates the leaf from the background.
3.  Image preprocessing removes noise and normalizes size.
4.  Feature extraction combines deep and handcrafted features.
5.  Auto Sparse Stacked Encoder compresses features.
6.  EfficientNet predicts plant species + disease.
7.  Optimizers improve convergence during training.
8.  UI visualizes confidence, Grad-CAM, t-SNE embedding and research
    pipeline.

## 2. Expected Output

-   Plant species
-   Disease name
-   Healthy / Diseased status
-   Confidence score
-   Top-3 predictions
-   Segmentation mask
-   Grad-CAM heatmap
-   t-SNE visualization
-   Disease description
-   Treatment recommendation

## 3. Model Pipeline

Input → Preprocessing → U-Net++ → Feature Extraction → Sparse Encoder →
EfficientNet → Explainability → UI

## 4. Dataset

Use PlantVillage + optional real-world leaf datasets.

Expected classes:

-   Tomato
-   Potato
-   Pepper
-   Apple
-   Grape
-   Corn
-   Strawberry
-   Peach

Each contains Healthy + multiple diseases.

## 5. Preprocessing

-   Resize 256×256
-   RGB normalization
-   Gaussian filtering
-   CLAHE
-   Data augmentation
-   Background normalization

## 6. Segmentation --- U-Net++

Input: 256×256 RGB

Output: Binary mask

Loss: - Dice Loss - BCE Loss

Metric: - IoU - Dice Coefficient

Purpose: remove background and improve classification robustness.

## 7. Feature Extraction

### Deep Features

-   EfficientNet intermediate embeddings
-   1280-dimensional vector

### Handcrafted Features

-   GLCM texture
-   Gabor filters
-   Color histogram
-   Shape descriptors

Concatenate all vectors.

## 8. Auto Sparse Stacked Encoder

Purpose:

-   Dimensionality reduction
-   Noise removal
-   Latent representation learning

Architecture:

-   2048
-   1024
-   512
-   256 latent
-   Decoder symmetric

Regularization:

-   L1 sparsity
-   KL divergence

Output: 256-dimensional embedding.

## 9. Classification

Model: EfficientNet-B3

Outputs:

-   Plant Species
-   Disease Class

Loss:

-   Cross Entropy
-   Label Smoothing

Metrics:

-   Accuracy
-   Precision
-   Recall
-   F1
-   ROC-AUC

## 10. Optimizers

Implement all for experimentation:

  Optimizer        Purpose
  ---------------- --------------------
  AdamW            Default
  SGD + Momentum   Baseline
  RMSProp          EfficientNet paper
  Lion             Modern optimizer

Scheduler:

-   Cosine Annealing
-   ReduceLROnPlateau

Early stopping: patience = 12

## 11. Explainability

### Grad-CAM

Generate attention heatmap over infected regions.

### t-SNE

Project latent vectors to 2D.

User should see uploaded sample plotted among training clusters.

### Confidence

Show probability bars.

## 12. Frontend

Framework:

-   Next.js
-   Tailwind
-   Framer Motion
-   Recharts

Pages:

### Landing

-   Hero
-   Research overview
-   Samsung PRISM story
-   Architecture diagram

### Analyzer

-   Drag & Drop upload
-   Analyze button

### Results

Display:

-   Original image
-   Segmentation mask
-   Heatmap
-   Disease card
-   Confidence gauge
-   Top predictions
-   t-SNE plot
-   Recommendation

### Research

Explain every model with interactive cards.

## 13. Backend

FastAPI

Endpoints:

-   POST /predict
-   POST /segment
-   GET /health
-   GET /classes

Inference pipeline:

Upload → Segment → Features → Encoder → EfficientNet → GradCAM → JSON

## 14. JSON Response

``` json
{
  "plant":"Tomato",
  "disease":"Early Blight",
  "healthy":false,
  "confidence":98.42,
  "top_predictions":[
    ["Early Blight",98.4],
    ["Late Blight",1.1],
    ["Healthy",0.5]
  ],
  "gradcam":"...",
  "mask":"...",
  "embedding":[0.23,-0.51]
}
```

## 15. Folder Structure

``` text
prism-leafsense/
│
├── frontend/
├── backend/
├── models/
├── datasets/
├── notebooks/
├── docs/
│   ├── architecture.png
│   ├── research.md
│   └── methodology.md
├── weights/
└── README.md
```

## 16. Training Pipeline

1.  Prepare dataset
2.  Train U-Net++
3.  Generate masks
4.  Extract features
5.  Train sparse encoder
6.  Train EfficientNet
7.  Generate t-SNE embeddings
8.  Export ONNX/PyTorch weights

## 17. Research Metrics

Segmentation:

-   Dice
-   IoU

Classification:

-   Accuracy
-   Precision
-   Recall
-   F1
-   ROC
-   Confusion Matrix

Visualization:

-   t-SNE
-   Grad-CAM
-   Probability distribution

## 18. UI Theme

Style:

-   White research paper aesthetic
-   Emerald accents
-   Scientific typography
-   Animated pipeline
-   Interactive architecture diagram
-   Minimal glassmorphism

## 19. Stretch Goals

-   Multimodal text input
-   PDF research report export
-   Compare two leaves
-   Batch prediction
-   Mobile responsive PWA

## 20. Codex / Claude Build Instructions

The generated application must be production-ready.

Requirements:

-   TypeScript throughout
-   FastAPI backend
-   PyTorch models
-   Modular architecture
-   Clean API separation
-   ONNX inference support
-   Docker + docker-compose
-   GitHub Actions CI
-   Model weight loading abstraction
-   No mocked predictions (real inference pipeline)
-   Fully responsive UI
-   Interactive research visualizations

The goal is to resemble a published AI research demonstrator rather than
a basic image classifier.
