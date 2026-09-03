# Methodology

## 1. Problem statement

Given a single RGB photograph of a plant leaf, identify the plant species and the disease affecting
it (or confirm that it is healthy), report a calibrated confidence, and expose the intermediate
evidence that produced the decision.

## 2. Data

PlantVillage (Hughes & Salathé), 54,305 images across 38 classes spanning 14 species. The dataset
ships three parallel trees:

| Tree | Content | Used for |
| --- | --- | --- |
| `color` | raw photograph on a uniform background | segmentation input |
| `segmented` | the same image with the background already removed | mask ground truth, classifier input |
| `grayscale` | luminance-only copy | not used |

The `segmented` tree makes segmentation supervision free: the binary mask is simply the set of
non-black pixels of the segmented file, after a morphological open/close to remove speckle.

Class frequency is severely imbalanced (120 images for `Potato___healthy`, 4,527 for
`Orange___Haunglongbing`). Training uses a per-class cap, which both flattens the imbalance and
bounds epoch time.

## 3. Preprocessing

1. **EXIF transpose** - phone photos carry orientation in metadata; without correcting it a portrait
   capture reaches the network rotated by 90°.
2. **Resize to 256×256** with area interpolation (correct choice when downsampling).
3. **Gaussian blur**, 5×5, σ = 1.0 - removes sensor and JPEG noise that GLCM would otherwise read as
   texture.
4. **CLAHE** (clip 2.0, 8×8 tiles) on the **L channel of LAB** only. Equalising luminance without
   touching chroma preserves the colour cues (chlorosis, necrosis, rust pustules) that both the
   handcrafted colour features and the classifier depend on.
5. **ImageNet normalisation** to match the pretrained backbone statistics.

## 4. Segmentation - U-Net++

U-Net++ replaces U-Net's single skip connection per level with a dense nested grid: node `X(i,j)`
receives every shallower node at the same resolution plus the upsampled node one level below. The
encoder features that reach the decoder have therefore passed through several convolutions, which
narrows the semantic gap between raw encoder maps and the decoder's abstraction level.

- **Loss**: `BCE + Dice`. BCE gives per-pixel gradients; Dice optimises the overlap metric directly
  and is robust to the leaf/background area imbalance.
- **Metrics**: Dice coefficient and IoU on a held-out split.
- **Background randomisation**: PlantVillage photographs every leaf on a uniform grey sheet. Trained
  on that alone, a network learns "not grey ⇒ leaf" and fails on field photos. Half of the training
  batch is therefore recomposited over synthetic backgrounds - flat colours, gradients, blurred
  noise, coarse green clutter - using the ground-truth mask.
- **Post-processing**: threshold at 0.5, morphological close then open, keep connected components
  ≥ 15% of the largest. If the surviving mask covers < 2% of the frame the mask is reset to the full
  frame, so a failed segmentation degrades to "classify the whole image" rather than to an empty
  crop.

When no checkpoint is present the stage falls back to Excess-Green (`2G − R − B`) combined with HSV
saturation and Otsu thresholding, and reports `method: classical-exg-otsu` so the UI never presents
a classical mask as a network output.

## 5. Feature extraction

**Deep features**: the pooled EfficientNet embedding (1280-d for B0, 1536-d for B3).

**Handcrafted features** (206-d total), all computed over leaf pixels only:

| Family | Dimensions | Definition |
| --- | --- | --- |
| GLCM | 48 | 6 properties (contrast, dissimilarity, homogeneity, energy, correlation, ASM) × 2 distances × 4 angles, 32 grey levels |
| Gabor | 48 | 4 frequencies × 6 orientations, mean and standard deviation of each response magnitude |
| Colour | 96 | HSV histograms, 32 bins per channel, mask-weighted and L1-normalised |
| Shape | 14 | area ratio, perimeter, circularity, aspect ratio, extent, solidity, eccentricity, 7 log-scaled Hu moments |

GLCM and Gabor answer a question CNN embeddings answer only implicitly: *how* is the surface
patterned - powdery, speckled, striped, scorched. Colour histograms capture disease-specific hue
shifts; shape descriptors capture curl and distortion (e.g. leaf-curl virus).

## 6. Auto Sparse Stacked Encoder

The fused vector (deep ‖ handcrafted) is standardised with statistics stored in the checkpoint, then
compressed 2048 → 1024 → 512 → **256** with a symmetric decoder.

- The latent layer is sigmoid-bounded so activations lie in [0, 1] and a **KL divergence** term can
  drive the mean activation of each unit towards ρ = 0.05.
- An **L1 penalty** additionally suppresses individual activations.
- Objective: `MSE(reconstruction) + λ₁·L1(latent) + λ_KL·KL(ρ ‖ ρ̂)`.

Sparsity forces each latent unit to specialise, which produces a latent space where distance is
meaningful - exactly what the t-SNE placement and nearest-neighbour lookup need.

## 7. Classification - EfficientNet

EfficientNet's compound scaling grows depth, width and input resolution together under a single
coefficient, which is why even B0 is competitive at a fraction of the parameters of older
backbones. The head predicts all 38 species-disease combinations; species and health status are
derived from the winning class through the label table.

- **Loss**: cross entropy with label smoothing 0.1 - prevents the 99.9% overconfidence that
  PlantVillage's clean images otherwise induce.
- **Optimisers** (all implemented, selectable): AdamW (default), SGD + Nesterov momentum, RMSProp
  (the optimiser from the EfficientNet paper), Lion.
- **Schedulers**: cosine annealing or ReduceLROnPlateau. **Early stopping** with patience 12.
- **Metrics**: accuracy, top-3 accuracy, macro precision / recall / F1, and a saved confusion
  matrix.

The classifier is trained on background-removed leaves so that its training distribution matches
what the U-Net++ stage hands it at inference.

## 8. Explainability

**Grad-CAM**: the gradient of the winning logit with respect to the last convolutional feature map
is global-average-pooled into per-channel weights; the ReLU of the weighted channel sum gives the
class-discriminative heatmap. The map is masked to the leaf so background artefacts cannot appear as
evidence, and a `focus_ratio` reports how much of the leaf is strongly activated.

**t-SNE**: computed once, offline, over the latent codes of the training subset. t-SNE has no
out-of-sample transform, so a new upload is placed by inverse-distance-weighted interpolation of the
2-D coordinates of its k = 15 nearest latent neighbours. The five nearest training classes are
returned alongside, which doubles as a sanity check on the prediction.

**Confidence**: softmax probability, plus the top-1 − top-2 margin and the predictive entropy in
bits. A confident-but-ambiguous prediction (high probability, low margin) is therefore visible.

## 9. Serving

FastAPI exposes `/health`, `/classes`, `/predict` and `/segment`. Models load lazily through a
registry that caches instances and records load status; a missing checkpoint is reported, never
silently replaced by an untrained network. Each stage returns a base64 PNG preview plus its own
metrics and wall-clock duration, so the UI can render the full trace from a single response.
