import Link from "next/link";

const PIPELINE = [
  { id: "01", name: "Input", detail: "Upload or camera capture, EXIF-corrected" },
  { id: "02", name: "Preprocessing", detail: "Resize 256, Gaussian denoise, CLAHE" },
  { id: "03", name: "U-Net++", detail: "Nested decoder, BCE + Dice, binary leaf mask" },
  { id: "04", name: "Feature extraction", detail: "Deep embedding + GLCM, Gabor, colour, shape" },
  { id: "05", name: "Sparse encoder", detail: "Fused vector compressed to a 256-d latent" },
  { id: "06", name: "EfficientNet", detail: "38 PlantVillage classes, label smoothing" },
  { id: "07", name: "Explainability", detail: "Grad-CAM heatmap and t-SNE placement" },
];

const HIGHLIGHTS = [
  { value: "38", label: "disease classes" },
  { value: "14", label: "plant species" },
  { value: "206", label: "handcrafted feature dimensions" },
  { value: "14", label: "visual stage outputs" },
];

export default function HomePage() {
  return (
    <div className="space-y-14">
      <section className="paper-grid -mx-4 rounded-3xl px-4 py-12 sm:-mx-6 sm:px-10">
        <p className="mono-label">Samsung PRISM · research demonstrator</p>
        <h1 className="mt-3 max-w-3xl text-4xl font-semibold tracking-tight sm:text-5xl">
          Leaf disease diagnosis you can watch happen,{" "}
          <span className="text-accent">stage by stage</span>.
        </h1>
        <p className="mt-4 max-w-2xl text-base text-muted">
          LeafSense segments the leaf with U-Net++, fuses deep and handcrafted features, compresses
          them through a sparse stacked encoder, and classifies species and disease with
          EfficientNet. Every intermediate image is returned, not hidden.
        </p>
        <div className="mt-7 flex flex-wrap gap-3">
          <Link
            href="/analyze"
            className="rounded-lg bg-accent px-5 py-3 text-sm font-semibold text-white transition hover:bg-emerald-700"
          >
            Analyze a leaf
          </Link>
          <Link
            href="/research"
            className="rounded-lg border border-line bg-white px-5 py-3 text-sm font-medium transition hover:border-accent hover:text-accent"
          >
            Read the method
          </Link>
        </div>

        <dl className="mt-10 grid grid-cols-2 gap-4 sm:grid-cols-4">
          {HIGHLIGHTS.map((item) => (
            <div key={item.label} className="rounded-xl border border-line bg-white/70 p-4">
              <dt className="text-2xl font-semibold text-accent">{item.value}</dt>
              <dd className="mt-1 text-xs text-muted">{item.label}</dd>
            </div>
          ))}
        </dl>
      </section>

      <section>
        <div className="mono-label">Architecture</div>
        <h2 className="mt-1 text-2xl font-semibold tracking-tight">The pipeline</h2>
        <ol className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {PIPELINE.map((step) => (
            <li key={step.id} className="rounded-xl border border-line bg-white p-4">
              <span className="mono-label">{step.id}</span>
              <h3 className="mt-1 text-sm font-semibold">{step.name}</h3>
              <p className="mt-1 text-xs leading-relaxed text-muted">{step.detail}</p>
            </li>
          ))}
        </ol>
      </section>

      <section className="grid gap-5 sm:grid-cols-2">
        <article className="rounded-2xl border border-line bg-white p-6">
          <h2 className="text-lg font-semibold">Why segmentation first</h2>
          <p className="mt-2 text-sm leading-relaxed text-muted">
            A classifier trained on whole photos can key on the background - the tray, the soil, the
            lighting. U-Net++ removes the background, so EfficientNet sees only leaf pixels and the
            same distribution it was trained on. The masked image the classifier receives is shown in
            the results, so you can check that the cut is clean.
          </p>
        </article>
        <article className="rounded-2xl border border-line bg-white p-6">
          <h2 className="text-lg font-semibold">Reported with its uncertainty</h2>
          <p className="mt-2 text-sm leading-relaxed text-muted">
            Every prediction ships with the top-1 probability, the top-1 minus top-2 margin and the
            predictive entropy over all 38 classes, so a confident-but-ambiguous decision is visible
            rather than hidden behind a single number. Missing checkpoints are reported as missing;
            no stage ever substitutes a fabricated result.
          </p>
        </article>
      </section>
    </div>
  );
}
