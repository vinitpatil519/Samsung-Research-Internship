"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

type Card = {
  id: string;
  title: string;
  claim: string;
  detail: string;
  facts: [string, string][];
};

const CARDS: Card[] = [
  {
    id: "preprocessing",
    title: "Preprocessing",
    claim: "Make every photo look like the same camera took it.",
    detail:
      "Images are resized to 256x256, denoised with a 5x5 Gaussian (sigma 1.0), then contrast-equalised with CLAHE on the L channel of LAB. Equalising luminance only keeps hue intact, which matters because chlorosis and necrosis are colour cues the later stages depend on.",
    facts: [
      ["Input size", "256 x 256 RGB"],
      ["Denoise", "Gaussian 5x5, sigma 1.0"],
      ["Contrast", "CLAHE clip 2.0, 8x8 tiles"],
    ],
  },
  {
    id: "unetpp",
    title: "U-Net++ segmentation",
    claim: "Cut the leaf out before anything else looks at it.",
    detail:
      "A nested U-Net with dense skip pathways: every decoder node receives all shallower nodes at its own resolution plus the upsampled node below it, which shortens the semantic gap that plain U-Net skips leave. Trained with BCE + Dice on masks derived from the PlantVillage segmented tree, with half the batch composited onto synthetic backgrounds so the network does not learn 'not grey means leaf'.",
    facts: [
      ["Loss", "BCE + Dice"],
      ["Metrics", "Dice coefficient, IoU"],
      ["Output", "Binary mask, morphologically cleaned"],
    ],
  },
  {
    id: "features",
    title: "Feature extraction",
    claim: "Deep features see shape; handcrafted features measure texture.",
    detail:
      "The EfficientNet embedding is concatenated with 48 GLCM statistics (6 properties x 2 distances x 4 angles), 48 Gabor descriptors (4 frequencies x 6 orientations, mean and standard deviation), 96 HSV histogram bins and 14 shape descriptors including the 7 Hu moments - 206 handcrafted dimensions in total.",
    facts: [
      ["GLCM", "contrast, dissimilarity, homogeneity, energy, correlation, ASM"],
      ["Gabor", "4 frequencies x 6 orientations"],
      ["Shape", "circularity, solidity, extent, eccentricity, Hu moments"],
    ],
  },
  {
    id: "encoder",
    title: "Auto Sparse Stacked Encoder",
    claim: "Compress the fused vector without keeping the noise.",
    detail:
      "A stacked autoencoder narrows 2048 -> 1024 -> 512 -> 256 with a symmetric decoder. The latent layer is sigmoid-bounded so a KL divergence term can push the mean activation towards a 5% target, and an L1 penalty keeps individual codes sparse. Standardisation statistics are stored in the checkpoint so inference scales features exactly as training did.",
    facts: [
      ["Latent", "256 dimensions"],
      ["Regularisation", "L1 + KL divergence (rho = 0.05)"],
      ["Reported", "sparsity and reconstruction error per request"],
    ],
  },
  {
    id: "efficientnet",
    title: "EfficientNet classification",
    claim: "One backbone, two answers: species and disease.",
    detail:
      "EfficientNet's compound scaling balances depth, width and resolution, which is why a B0 fine-tune already separates 38 PlantVillage classes. Training uses cross entropy with 0.1 label smoothing, cosine-annealed learning rate, and early stopping with patience 12. AdamW, SGD with momentum, RMSProp and Lion are all selectable from the CLI.",
    facts: [
      ["Classes", "38 (14 species)"],
      ["Loss", "Cross entropy + label smoothing 0.1"],
      ["Metrics", "accuracy, top-3, macro precision / recall / F1, confusion matrix"],
    ],
  },
  {
    id: "explain",
    title: "Explainability",
    claim: "Show where the decision came from, not just the number.",
    detail:
      "Grad-CAM weights the last convolutional feature map by the gradient of the winning logit, so bright regions are the pixels that raised that class score; the map is masked to the leaf so background artefacts cannot masquerade as evidence. t-SNE projects the 256-d latent space once, offline; a new upload is placed by distance-weighted interpolation over its nearest training neighbours, because t-SNE has no out-of-sample transform.",
    facts: [
      ["Grad-CAM", "last conv block, gradient-weighted"],
      ["t-SNE", "precomputed map, kNN placement"],
      ["Confidence", "softmax probability, margin and entropy"],
    ],
  },
];

export default function ResearchPage() {
  const [open, setOpen] = useState<string | null>("unetpp");

  return (
    <div className="space-y-8">
      <header>
        <p className="mono-label">Method</p>
        <h1 className="mt-1 text-3xl font-semibold tracking-tight">How LeafSense works</h1>
        <p className="mt-2 max-w-2xl text-sm text-muted">
          Six blocks, each with a job the next one depends on. Open a card for the details.
        </p>
      </header>

      <div className="grid gap-3">
        {CARDS.map((card) => {
          const expanded = open === card.id;
          return (
            <article
              key={card.id}
              className={`rounded-2xl border bg-white p-5 transition ${
                expanded ? "border-accent" : "border-line"
              }`}
            >
              <button
                type="button"
                onClick={() => setOpen(expanded ? null : card.id)}
                className="flex w-full items-start justify-between gap-4 text-left"
              >
                <div>
                  <h2 className="text-lg font-semibold">{card.title}</h2>
                  <p className="mt-0.5 text-sm text-muted">{card.claim}</p>
                </div>
                <span className="mt-1 text-lg text-muted">{expanded ? "−" : "+"}</span>
              </button>

              <AnimatePresence initial={false}>
                {expanded && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.22 }}
                    className="overflow-hidden"
                  >
                    <p className="mt-4 text-sm leading-relaxed">{card.detail}</p>
                    <dl className="mt-4 grid gap-2 sm:grid-cols-3">
                      {card.facts.map(([key, value]) => (
                        <div key={key} className="rounded-lg border border-line bg-paper p-3">
                          <dt className="mono-label">{key}</dt>
                          <dd className="mt-1 text-xs">{value}</dd>
                        </div>
                      ))}
                    </dl>
                  </motion.div>
                )}
              </AnimatePresence>
            </article>
          );
        })}
      </div>
    </div>
  );
}
