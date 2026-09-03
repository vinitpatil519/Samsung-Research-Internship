# Samsung Research (Prism) Worklet:25ST16 — Project Q&A

27 questions covering the whole project: what it does, why each design choice was made, how each
model works, and where the limits are. Written so you can read it once and then explain the project
to anyone — a reviewer, a professor, or an interviewer.

---

### 1. Explain the project in one minute.

Worklet 25ST16 diagnoses plant leaf disease from a single photograph. A user uploads a leaf image or
takes one with a phone camera. The backend runs a five-block pipeline: preprocessing normalises the
image; a U-Net++ network segments the leaf away from the background; a feature block extracts deep
CNN features together with classical GLCM, Gabor, colour and shape descriptors; a sparse stacked
autoencoder compresses that fused vector to 256 dimensions; and an EfficientNet classifier predicts
one of 38 species-disease combinations. The UI returns the plant name, the disease, the confidence,
the implicated pathogen, the symptoms and a management recommendation, plus Grad-CAM attention, a
t-SNE view of where the sample sits in latent space, and, importantly, the actual image produced by
every single stage, so the diagnosis is auditable rather than a black box.

### 2. Why segment the leaf before classifying it, instead of classifying the raw photo?

Two reasons — one about accuracy, one about honesty.

A CNN trained on whole photographs can reach high accuracy by keying on the background: the tray,
the soil, the lighting rig, even JPEG artefacts specific to one capture session. That is a shortcut
that evaporates the moment the photo comes from a different setting. Removing the background forces
the classifier to use leaf pixels only.

Second, segmentation makes the system inspectable. The masked leaf is shown in the UI, so if the
mask is wrong you can see it immediately — and a wrong mask explains a wrong diagnosis. Without it,
a bad prediction is unexplainable.

### 3. What is U-Net++ and how is it different from a plain U-Net?

U-Net is an encoder-decoder with skip connections: each decoder level receives the raw encoder
feature map from the same resolution. The problem is that a shallow encoder map is semantically
"younger" than the decoder map it is concatenated with — the decoder has already been through the
bottleneck, the encoder feature has not. That mismatch is called the semantic gap.

U-Net++ fills the gap with a dense nested grid of intermediate convolution blocks. Node `X(i,j)`
takes every shallower node at its own resolution (`X(i,0) … X(i,j-1)`) plus the upsampled node from
one level below (`X(i+1,j-1)`). So encoder features are progressively refined through several
convolutions before they meet the decoder, and the network can also be deeply supervised — a head
on each of `X(0,1) … X(0,4)` — which lets you prune it to a shallower, faster model at inference
without retraining. Our implementation supports deep supervision; inference uses the deepest head.

### 4. Why the combined BCE + Dice loss and not just binary cross-entropy?

BCE is a per-pixel loss: it gives a clean gradient for every pixel independently but is blind to
overlap as a whole, and it is biased by class imbalance — if the leaf covers 30% of the frame,
predicting "background everywhere" already gets 70% of pixels right.

Dice loss optimises `2|P∩G| / (|P| + |G|)` directly, which is exactly the overlap metric we report,
and is naturally normalised by region size, so it does not collapse to the majority class. But Dice
alone has weak and unstable gradients when the prediction is nearly empty at initialisation.

Together, BCE supplies stable early gradients and Dice aligns training with the evaluation metric.
This pairing is the standard choice in medical and biological segmentation for exactly this reason.

### 5. Where did the segmentation ground truth come from? Did you annotate masks by hand?

No manual annotation was needed. PlantVillage ships three parallel trees for the same 54,305
images: `color` (the raw photo), `segmented` (the same image with the background already removed)
and `grayscale`. The binary mask for any image is therefore the set of non-black pixels of its
`segmented` counterpart, cleaned with a morphological open-then-close to remove speckle. Pairing is
by filename stem, since segmented files carry a `_final_masked` suffix.

That gives 54,305 free (image, mask) pairs — supervision for U-Net++ at zero annotation cost.

### 6. PlantVillage photographs every leaf on a uniform grey sheet. Doesn't that make your segmenter useless in the field?

It would, if trained naively — the network would learn "grey means background", which is a rule
about that studio, not about leaves. That is the dataset's best-known weakness.

The fix is background randomisation during training. For half of the training samples, the leaf is
cut out with its ground-truth mask and recomposited onto a synthetic background: a flat random
colour, a two-colour gradient, blurred random noise (soil and gravel look), or coarse green clutter
(grass and canopy look). The mask does not change. The network therefore sees the same leaf against
many different backgrounds and cannot solve the task with a colour rule — it has to learn leaf shape
and venation.

Honest limitation: synthetic backgrounds are still not real field photographs with overlapping
leaves, shadows and depth-of-field blur. The right next step is fine-tuning on a few hundred real
annotated field images.

### 7. Why Gaussian blur before CLAHE, and why apply CLAHE only to the L channel?

Order matters. CLAHE amplifies local contrast — if you run it first, it amplifies sensor noise and
JPEG blocking artefacts along with real detail, and the GLCM texture features downstream would then
measure that amplified noise. Denoising first (5×5 Gaussian, σ = 1.0) removes the high-frequency
junk, and CLAHE then stretches only real structure.

CLAHE is applied to the L (lightness) channel after converting RGB to LAB, then the image is
converted back. Equalising in RGB directly would shift the ratios between channels and therefore the
hue — and hue is diagnostic here: chlorosis is yellowing, rust is orange-brown, late blight is
grey-green. Destroying colour to improve contrast would break both the colour histogram features and
the classifier. Working in LAB changes brightness distribution while leaving chroma untouched.

### 8. You use a CNN. Why bother with handcrafted features at all?

Three reasons.

First, they measure something a pooled CNN embedding only encodes implicitly: the *surface statistics*
of the leaf. Powdery mildew is a fine white film, rust is discrete raised pustules, septoria is many
small circular spots, mosaic virus is a low-frequency mottle. GLCM and Gabor put explicit numbers on
"how is this surface patterned".

Second, they are interpretable and cheap. When a prediction is wrong you can look at the Gabor
energy map and the entropy map and see whether the texture evidence was even present.

Third — the project brief specifies a hybrid deep + handcrafted feature stage feeding a sparse
encoder, which mirrors the published architecture this work reproduces. The fused vector is 1280 (or
1536 for B3) deep dimensions plus 206 handcrafted dimensions.

### 9. Explain GLCM concretely — what is actually being computed?

A grey-level co-occurrence matrix counts how often a pixel with grey level *i* appears at a fixed
offset from a pixel with grey level *j*. We quantise the leaf to 32 grey levels (fewer levels means
a denser, more stable matrix), and build matrices for 2 distances (1 and 2 pixels) × 4 angles (0°,
45°, 90°, 135°), symmetric and normalised so each matrix is a joint probability distribution.

From each matrix we take 6 Haralick properties:

- **Contrast** — `Σ (i−j)² p(i,j)`: large when neighbouring pixels differ strongly (rough, spotty).
- **Dissimilarity** — like contrast, linear in `|i−j|`, so less dominated by extremes.
- **Homogeneity** — weights `p(i,j)` by `1/(1+(i−j)²)`: high for smooth surfaces.
- **Energy / ASM** — `Σ p(i,j)²`: high when a few transitions dominate, i.e. uniform texture.
- **Correlation** — linear dependency of grey levels along the offset, which picks up directional
  structure like the striped lesions of northern leaf blight.

6 properties × 2 distances × 4 angles = 48 numbers. Background pixels are forced to level 0 so they
form a constant region and do not create false transitions inside the statistic.

### 10. What do the Gabor filters add, and why 4 frequencies × 6 orientations?

A Gabor filter is a sinusoid multiplied by a Gaussian envelope — a band-pass filter tuned to one
spatial frequency and one orientation, and a standard model of early visual cortex receptive fields.
Convolving the leaf with a bank of them answers "how much energy does this leaf have at this scale,
in this direction".

Four frequencies (0.1–0.4 cycles/pixel) cover coarse mottling through fine speckle; six orientations
at 30° steps cover the vein and lesion directions without redundancy (Gabor responses are symmetric,
so beyond 180° you repeat yourself). We take the mean and standard deviation of the response
magnitude over leaf pixels for each of the 24 filters, giving 48 features, and the per-pixel maximum
across the bank is returned as the visual "Gabor energy" map shown in the UI.

### 11. What is the Auto Sparse Stacked Encoder actually for? Isn't the CNN embedding enough for classification?

It is not in the classification path — the classifier head sits directly on the EfficientNet
embedding. The sparse encoder serves the *representation* side of the system: it takes the fused
deep + handcrafted vector (roughly 1,486 dimensions for B0) and compresses it to a 256-dimensional
latent code that is used for the t-SNE map and for nearest-neighbour retrieval.

Three things it buys:

- **Dimensionality reduction** — 1,486 → 256, which makes the neighbour search and the t-SNE
  tractable and less subject to the curse of dimensionality.
- **Denoising** — an autoencoder trained to reconstruct must discard the components of the input
  that carry no reconstructable structure.
- **Disentangling** — sparsity forces each unit to respond to a specific pattern rather than every
  unit encoding a little of everything, which is what makes latent distances meaningful.

"Stacked" means the encoder is a stack of layers (2048 → 1024 → 512 → 256) with a symmetric decoder;
"auto" is the autoencoder objective, reconstructing its own input rather than a label.

### 12. You use both an L1 penalty and a KL divergence for sparsity. What is the difference?

They constrain different things.

The **L1 penalty**, `λ·mean(|h|)`, is per-activation: it pushes every individual latent value toward
zero, for every sample. It shrinks magnitudes.

The **KL term** is per-unit and averaged over the batch. For each latent unit *j* we compute its mean
activation `ρ̂ⱼ` over the batch and penalise `KL(ρ ‖ ρ̂ⱼ) = ρ log(ρ/ρ̂ⱼ) + (1−ρ) log((1−ρ)/(1−ρ̂ⱼ))`
with target ρ = 0.05. This says: each unit should be active for about 5% of inputs. It does not care
which 5%.

So L1 says "be small"; KL says "be selective". KL is the term that produces specialised, feature-
detector-like units. It requires activations in [0, 1] to be a valid Bernoulli KL, which is why the
latent layer uses a sigmoid rather than a ReLU.

### 13. Why EfficientNet rather than ResNet, VGG or a plain CNN?

EfficientNet's contribution is compound scaling. Earlier practice scaled one dimension at a time —
deeper (ResNet), or wider, or higher input resolution. The EfficientNet paper showed those three are
coupled: a deeper network needs more resolution to have detail worth its depth, and more width to
carry the features. It scales all three together by a single coefficient φ, with the ratios found by
grid search, on top of a mobile-inverted-bottleneck (MBConv) base found by neural architecture
search.

The practical payoff is accuracy per parameter. EfficientNet-B0 gets ImageNet accuracy comparable to
a ResNet-50 with roughly a fifth of the parameters, which matters a lot when the target is a
demonstrator that has to run on CPU and, eventually, on a phone. The spec calls for B3; the code
takes `--arch` so B0 (fast, CPU-trainable) and B3 (the spec configuration, GPU) load through the
same path — the architecture name is stored inside the checkpoint.

### 14. Why label smoothing? What goes wrong without it?

Standard cross entropy asks the model to put probability 1.0 on the true class, which drives logits
apart without bound. On a clean, visually easy dataset like PlantVillage that produces a model
reporting 99.99% confidence on almost everything — including on inputs it should be unsure about.
Confidence stops carrying information.

Label smoothing replaces the hard target with `1−ε` on the true class and `ε/(K−1)` spread over the
rest (ε = 0.1 here). The optimal logit gap becomes finite, the model stops over-sharpening, and the
resulting probabilities are better calibrated and generalise slightly better. Since the UI shows a
confidence number to a user who may act on it, calibration is a product requirement, not a detail.

### 15. Four optimisers are implemented. Which one and why?

- **AdamW** (default) — Adam with decoupled weight decay. Adam's per-parameter adaptive step sizes
  make fine-tuning robust to learning-rate choice; the "W" fixes the fact that L2 inside Adam's
  update is not equivalent to true weight decay. Best default for short fine-tuning runs.
- **SGD + Nesterov momentum** — the classical baseline. Often generalises marginally better with a
  long, well-tuned schedule, but needs more epochs and more tuning than a CPU budget allows.
- **RMSProp** — what the original EfficientNet paper used (with specific ε and decay), included for
  faithful reproduction.
- **Lion** — a recent optimiser found by symbolic search; it keeps only momentum and uses the sign of
  the update, so it is memory-light and works well at large batch sizes.

Selectable with `--optimizer`, paired with cosine annealing or ReduceLROnPlateau and early stopping
with patience 12.

### 16. PlantVillage is heavily imbalanced — 120 images for one class, 4,527 for another. How do you handle that?

With a per-class cap in the stratified split (`--per-class`). Every class contributes at most N
training images, so the effective distribution is far flatter than the raw dataset and the loss is
not dominated by orange HLB and tomato yellow leaf curl. The cap also bounds epoch time, which is
what makes CPU training feasible at all.

Classes below the cap keep all their images, so the small classes are not thrown away. Because the
metric set includes **macro** precision, recall and F1 — averaged over classes, not over samples — a
model that quietly ignores `Potato___healthy` cannot hide behind overall accuracy.

Alternatives that would also work, and would be the next step with more compute: class-weighted
cross entropy, a weighted sampler, or focal loss.

### 17. How is the train/validation split done, and is there a leakage risk?

The split is stratified per class with a fixed seed: each class is shuffled, capped, and split
85/15. Both the split and the class ordering are deterministic and stored in the checkpoint, so
evaluation numbers are reproducible.

The leakage risk to be aware of — and to name in an interview, because it is the standard critique
of PlantVillage results — is that the dataset contains **multiple photographs of the same physical
leaf**. A purely random split can put one photo of a leaf in train and another in validation, which
inflates the score. PlantVillage publishes a `leaf-map.json` grouping shots of the same leaf, and a
leaf-disjoint split (or the dataset's official train/test lists, also shipped in the repository) is
the rigorous choice. This implementation uses a random stratified split, so the reported validation
accuracy should be read as an optimistic figure; grouping by leaf id is the first thing to change
before publishing a number.

### 18. Explain Grad-CAM step by step.

Grad-CAM answers "which spatial regions of the last convolutional feature map raised the score of
class *c*".

1. Run the forward pass and take the last conv feature map `A ∈ ℝ^{K×H×W}` (K channels).
2. Take the logit `y_c` for the predicted class and backpropagate to `A`, giving `∂y_c/∂A`.
3. Global-average-pool those gradients over space to get one weight per channel:
   `α_k = (1/HW) Σ_{i,j} ∂y_c/∂A_k(i,j)`. A channel whose activation raises `y_c` gets a positive
   weight.
4. Take the weighted channel sum and apply ReLU: `L = ReLU(Σ_k α_k A_k)`. The ReLU keeps only
   evidence *for* the class, discarding evidence against it.
5. Normalise to [0, 1], upsample to image size, overlay.

We additionally multiply by the leaf mask and report a `focus_ratio` — the fraction of leaf area
above 0.5 activation — so a diffuse, unfocused explanation is measurable and not just visually
judged.

### 19. Why mask the Grad-CAM to the leaf? Isn't that hiding information?

It is the opposite: the classifier only ever receives the masked leaf, so any activation outside the
leaf region would be an artefact of upsampling a 7×7 feature map to 224×224, not evidence the model
actually used. Showing it would invite over-interpretation. What matters — and is preserved — is
whether the attention concentrates on lesions or drifts across healthy tissue, which the focus ratio
quantifies.

### 20. How does the t-SNE view work, and how can a new upload appear on a map computed offline?

t-SNE converts pairwise distances in high-dimensional space into conditional probabilities of
"being a neighbour", then places points in 2-D so the low-dimensional neighbour distribution matches
the high-dimensional one, minimising the KL divergence between them. It preserves local
neighbourhoods well; global distances and cluster sizes in a t-SNE plot are *not* meaningful, which
is a caveat worth stating.

Crucially, t-SNE is not a learned mapping — it optimises the coordinates themselves, so there is no
transform to apply to a new point. The map is therefore computed once offline over the latent codes
of the training subset and stored with those codes. At request time the new sample's 256-d latent is
compared to the stored latents, the k = 15 nearest are found, and the sample is placed at the
inverse-distance-weighted mean of their 2-D positions. The classes of the five nearest neighbours
are also returned, which is a useful independent sanity check: if the classifier says "tomato early
blight" but every latent neighbour is a grape disease, something is wrong.

### 21. How trustworthy is the confidence number?

A softmax probability is a score, not a calibrated probability, and modern networks are
systematically overconfident. Three things are done about it.

First, label smoothing during training limits logit sharpening. Second, the API reports two
additional quantities alongside the top probability: the **margin** (top-1 minus top-2) and the
**predictive entropy** in bits over all 38 classes. A prediction can be 92% confident with a 2%
margin — meaning two classes are nearly tied — and that situation is visible instead of hidden.
Third, out-of-distribution inputs are flagged indirectly: the segmentation leaf ratio, the Grad-CAM
focus ratio and the t-SNE neighbour classes all disagree characteristically when the input is not
one of the 38 known conditions.

The rigorous next step is **temperature scaling**: fit a single scalar T on the validation set by
minimising negative log-likelihood of `softmax(logits/T)`, then divide logits by T at inference. It
costs almost nothing and measurably improves calibration error.

### 22. Walk me through the system architecture.

**Backend — FastAPI.** `POST /predict` receives the image, and `app/pipeline.py` runs the stages in
order, recording for each one a base64 PNG preview, its own metrics, and its wall-clock duration.
Models are loaded lazily through `models/registry.py`, which caches each instance and records
whether its checkpoint exists. `GET /health` exposes that status, `GET /classes` returns the 38
38 class cards with pathogen, symptom and management text, and `POST /segment` runs
segmentation alone.

**Frontend — Next.js 15 + TypeScript + Tailwind.** The analyzer page holds the upload widget, the
result card (diagnosis, animated confidence gauge, top-3 bars, symptoms, management), the
horizontally scrolling stage rail with one card per pipeline stage, and the Recharts t-SNE scatter
with the upload marked as a star.

**Why base64 previews rather than image URLs:** the stage images are ephemeral, per-request artefacts.
Serving them as URLs would require either server-side session storage with a cleanup policy or a
blob store, plus 14 additional HTTP round trips per analysis. Inlining them keeps the backend
stateless — which is what makes it horizontally scalable — at the cost of about 30–40% base64
overhead on payload size, mitigated by capping previews at 320 px.

### 23. How does the camera work on mobile, and what breaks if you get it wrong?

The camera button is `<input type="file" accept="image/*" capture="environment">`. The `capture`
attribute tells both Android Chrome and iOS Safari to open the rear camera directly instead of the
gallery; on desktop the attribute is ignored and it degrades to a normal file picker. This is
deliberately preferred over `getUserMedia`, which needs an HTTPS origin, an explicit permission
prompt, manual video-frame capture and per-browser quirk handling — for a single still photo, the
file input is more robust and works offline-installed too.

The failure mode that bites everyone: **EXIF orientation**. Phones store the sensor as landscape and
record the rotation in metadata. Decoding without honouring it feeds the network a sideways leaf and
quietly degrades accuracy. The backend calls `ImageOps.exif_transpose` before anything else.

### 24. Why does the response carry a pathogen name, symptom description and management action rather than only the class label?

Because "Tomato___Early_blight" is a dataset directory name, not a finding. A reviewer, and anyone
acting on the output, needs to know *what organism* is implicated (Alternaria solani), *what
observable evidence* supports that class (concentric target-like rings on older lower leaves) and
*what follows from it* (rotation, mulch, staking, a chlorothalonil or mancozeb schedule). The
description is also what makes the Grad-CAM checkable: if the stated symptom is "spots on the lower
older leaves" and the attention map is lit on the leaf tip, the prediction deserves suspicion even
at high confidence.

It is a static curated table in `app/labels.py`, not a generated text. That matters for a research
demonstrator: the vocabulary is small and finite (38 classes, 21 distinct conditions), the wording is
identical on every run, there is no network dependency or per-request cost, and there is no
generative model in the loop that could invent a treatment. Every string is auditable in one file
and versioned with the code.

### 25. What happens when a model checkpoint is missing, and why is that designed the way it is?

Nothing is faked. The registry reports the checkpoint as not loaded, and the pipeline marks the
affected stages `available: false` with the name of the training script to run. `POST /predict`
returns 503 with that reason instead of a made-up diagnosis.

Segmentation is the one stage with a fallback — Excess-Green (`2G − R − B`) plus HSV saturation and
Otsu thresholding — because everything downstream needs *some* mask. Even then, the stage reports
`method: "classical-exg-otsu"`, and the UI displays that label, so a classical mask is never
presented as a U-Net++ output.

This matters more than it looks. A demo that silently degrades to random predictions is worse than
one that stops, because the failure is invisible until someone acts on a wrong answer.

### 26. What results did you get, and what limits them?

Training was CPU-only on this machine (no CUDA build available), so the delivered checkpoints use
EfficientNet-B0 at 160 px with a per-class cap of 260, rather than the spec's B3 at 256 px on the
full 54k images.

**Segmentation (U-Net++, 3 epochs, 1,628 train / 222 val images, ~32 min):**

| Metric | Value |
| --- | --- |
| Dice coefficient | 0.972 |
| IoU | 0.946 |
| Validation loss (BCE + Dice) | 0.131 |

**Classification (EfficientNet-B0, 4 epochs, 8,086 train / 1,426 val images, ~112 min):**

| Metric | Value |
| --- | --- |
| Accuracy | 97.83% |
| Top-3 accuracy | 99.86% |
| Macro precision | 0.952 |
| Macro recall | 0.953 |
| Macro F1 | 0.952 |

**Sparse encoder:** 1,486-d → 256-d, reconstruction MSE 0.074, 65% of latent units near zero.
**t-SNE reference:** 1,443 points, clean per-species clusters.
**Latency:** roughly 1.5–2.3 s per image end to end on CPU, all 14 stage previews included.

The number that matters most is the one measured through the **live API**, not on the validation
loader: on 40 held-out images sent to `POST /predict` as real uploads, top-1 was **90.0%** and top-3
**100%**. That is ~8 points below the 97.8% validation figure, and the gap is informative — the
validation loader uses the dataset's ground-truth masks, while the API uses masks predicted by
U-Net++. A slightly different mask boundary shifts the crop and the scale of the leaf the classifier
sees. The fix is to train the classifier on U-Net-predicted masks (or with random mask
erosion/dilation as augmentation) so it becomes invariant to that boundary jitter; that costs one
more training run, not an architecture change.

`GET /health` reports the metrics stored inside each checkpoint, so the numbers the UI shows are
always the ones that checkpoint actually achieved, never a number typed into a slide.

Three honest limits on any number from this dataset:

1. **Split leakage** — a random split can separate two photos of the same physical leaf across train
   and validation (see Q17), so accuracy is optimistic.
2. **Domain gap** — PlantVillage is studio-lit, single-leaf, uniform-background. Real field
   photographs have shadows, overlapping foliage, multiple diseases on one leaf, and stages of
   severity the dataset does not span.
3. **Closed set** — the model must answer with one of 38 classes. Photograph a wheat leaf, or a
   deficiency rather than a disease, and it will still return its most similar known class.

Scaling up is a matter of compute, not code: `--arch efficientnet_b3 --size 256 --per-class 1200` on
a GPU runs the same scripts.

### 27. If you had two more months, what would you do next?

In priority order:

1. **A leaf-disjoint split and temperature scaling**, so the reported accuracy is defensible and the
   confidence is calibrated. Cheap, and it changes what the numbers mean.
2. **Real field data.** A few thousand annotated in-field photographs — including nutrient
   deficiencies and pest damage, not just the 38 pathogens — with fine-tuning on top of the
   PlantVillage-pretrained weights. This is the single biggest accuracy gain available.
3. **Out-of-distribution rejection**, so a non-leaf photo or an unknown disease returns "not
   recognised" rather than a confident wrong class. Energy-based scoring on the logits, or a
   distance threshold in the sparse latent space, both fit the existing architecture.
4. **ONNX export and on-device inference**, so the phone runs the model offline. That is the form
   this has to take to be useful in a field with no signal — and it is why the classifier is B0-sized
   and the weight loading already goes through an abstraction layer.
5. **Severity estimation**, not just classification: the mask and the Grad-CAM already give the
   infected fraction of leaf area, which is what actually determines whether to spray.
