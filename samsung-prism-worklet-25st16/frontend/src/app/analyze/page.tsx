"use client";

import { useCallback, useEffect, useState } from "react";
import ResultCard from "@/components/ResultCard";
import StageRail from "@/components/StageRail";
import TsneChart from "@/components/TsneChart";
import UploadPanel from "@/components/UploadPanel";
import { health, predict, type HealthResponse, type PredictResponse } from "@/lib/api";

function ModelStrip({ status }: { status: HealthResponse | null }) {
  if (!status) {
    return (
      <p className="text-xs text-red-600">
        Backend unreachable - start it with: uvicorn app.main:app --port 8000
      </p>
    );
  }
  return (
    <div className="flex flex-wrap items-center gap-2 text-[11px]">
      <span className="mono-label">Models</span>
      {Object.entries(status.models).map(([name, info]) => (
        <span
          key={name}
          title={info.detail}
          className={`rounded-full px-2 py-0.5 ${
            info.loaded ? "bg-accent-soft text-accent" : "bg-slate-100 text-muted"
          }`}
        >
          {name} {info.loaded ? "✓" : "–"}
        </span>
      ))}
      <span className="rounded-full bg-slate-100 px-2 py-0.5 text-muted">{status.device}</span>
    </div>
  );
}

export default function AnalyzePage() {
  const [status, setStatus] = useState<HealthResponse | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    health().then(setStatus).catch(() => setStatus(null));
  }, []);

  const run = useCallback(async (file: File) => {
    setBusy(true);
    setError(null);
    try {
      setResult(await predict(file));
    } catch (exception) {
      setResult(null);
      setError(exception instanceof Error ? exception.message : "prediction failed");
    } finally {
      setBusy(false);
    }
  }, []);

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight">Analyze a leaf</h1>
        <p className="mt-1 text-sm text-muted">
          Upload a photo or capture one with the camera. Every pipeline stage returns its own
          output.
        </p>
      </header>

      <ModelStrip status={status} />

      <UploadPanel onAnalyze={run} busy={busy} />

      {error && (
        <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
          {error}
        </div>
      )}

      {busy && (
        <div className="rounded-xl border border-line bg-white p-4 text-sm text-muted">
          Running U-Net++, feature extraction and EfficientNet…
        </div>
      )}

      {result && (
        <>
          <ResultCard result={result} />
          <StageRail stages={result.stages} />
          {result.tsne && <TsneChart tsne={result.tsne} />}
        </>
      )}
    </div>
  );
}
