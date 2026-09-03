"use client";

import { useCallback, useEffect, useRef, useState } from "react";

type Props = {
  onAnalyze: (file: File) => void;
  busy: boolean;
};

export default function UploadPanel({ onAnalyze, busy }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fileInput = useRef<HTMLInputElement>(null);
  const cameraInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!file) {
      setPreview(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const accept = useCallback((candidate: File | undefined | null) => {
    if (!candidate) return;
    if (!candidate.type.startsWith("image/")) {
      setError("That file is not an image.");
      return;
    }
    if (candidate.size > 12 * 1024 * 1024) {
      setError("Image is larger than 12 MB.");
      return;
    }
    setError(null);
    setFile(candidate);
  }, []);

  return (
    <section className="rounded-2xl border border-line bg-white p-5 shadow-sm sm:p-6">
      <div className="mono-label mb-3">Step 1 · Leaf image</div>

      <div
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          accept(event.dataTransfer.files?.[0]);
        }}
        onClick={() => fileInput.current?.click()}
        className={`flex min-h-52 cursor-pointer flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed p-5 text-center transition ${
          dragging ? "border-accent bg-accent-soft" : "border-line bg-paper hover:border-accent"
        }`}
      >
        {preview ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={preview}
            alt="Selected leaf"
            className="max-h-56 w-auto rounded-lg object-contain"
          />
        ) : (
          <>
            <span className="grid h-11 w-11 place-items-center rounded-full bg-accent-soft text-xl text-accent">
              ⬆
            </span>
            <p className="text-sm font-medium">Tap to upload or drop a leaf photo</p>
            <p className="text-xs text-muted">JPG · PNG · HEIC · max 12 MB</p>
          </>
        )}
      </div>

      <div className="mt-4 grid grid-cols-1 gap-2 sm:grid-cols-3">
        <button
          type="button"
          onClick={() => fileInput.current?.click()}
          className="rounded-lg border border-line bg-white px-4 py-2.5 text-sm font-medium transition hover:border-accent hover:text-accent"
        >
          Choose file
        </button>
        <button
          type="button"
          onClick={() => cameraInput.current?.click()}
          className="rounded-lg border border-line bg-white px-4 py-2.5 text-sm font-medium transition hover:border-accent hover:text-accent"
        >
          Open camera
        </button>
        <button
          type="button"
          disabled={!file || busy}
          onClick={() => file && onAnalyze(file)}
          className="rounded-lg bg-accent px-4 py-2.5 text-sm font-semibold text-white transition disabled:cursor-not-allowed disabled:opacity-40"
        >
          {busy ? "Analysing…" : "Analyze leaf"}
        </button>
      </div>

      {error && <p className="mt-3 text-sm text-red-600">{error}</p>}

      <input
        ref={fileInput}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(event) => accept(event.target.files?.[0])}
      />
      {/* capture="environment" opens the rear camera directly on Android and iOS. */}
      <input
        ref={cameraInput}
        type="file"
        accept="image/*"
        capture="environment"
        className="hidden"
        onChange={(event) => accept(event.target.files?.[0])}
      />
    </section>
  );
}
