"use client";

import { motion } from "framer-motion";
import type { PredictResponse } from "@/lib/api";

function ConfidenceGauge({ value, healthy }: { value: number; healthy: boolean }) {
  const radius = 52;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference * (1 - Math.min(100, Math.max(0, value)) / 100);
  const colour = healthy ? "#059669" : value >= 75 ? "#b45309" : "#dc2626";

  return (
    <svg viewBox="0 0 130 130" className="h-32 w-32 shrink-0" role="img" aria-label={`${value}%`}>
      <circle cx="65" cy="65" r={radius} fill="none" stroke="#e2e8f0" strokeWidth="10" />
      <motion.circle
        cx="65"
        cy="65"
        r={radius}
        fill="none"
        stroke={colour}
        strokeWidth="10"
        strokeLinecap="round"
        transform="rotate(-90 65 65)"
        strokeDasharray={circumference}
        initial={{ strokeDashoffset: circumference }}
        animate={{ strokeDashoffset: offset }}
        transition={{ duration: 0.9, ease: "easeOut" }}
      />
      <text x="65" y="62" textAnchor="middle" className="fill-ink text-[22px] font-semibold">
        {value.toFixed(1)}%
      </text>
      <text x="65" y="82" textAnchor="middle" className="fill-slate-500 text-[10px]">
        confidence
      </text>
    </svg>
  );
}

export default function ResultCard({ result }: { result: PredictResponse }) {
  if (!result.plant || !result.disease) return null;

  const healthy = Boolean(result.healthy);
  const confidence = result.confidence ?? 0;

  return (
    <section className="rounded-2xl border border-line bg-white p-5 shadow-sm sm:p-6">
      <div className="mono-label mb-3">Diagnosis</div>

      <div className="flex flex-col gap-5 sm:flex-row sm:items-center">
        <ConfidenceGauge value={confidence} healthy={healthy} />

        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <h2 className="text-2xl font-semibold tracking-tight">{result.plant}</h2>
            <span
              className={`rounded-full px-2.5 py-1 text-xs font-medium ${
                healthy ? "bg-accent-soft text-accent" : "bg-amber-50 text-amber-700"
              }`}
            >
              {healthy ? "Healthy" : "Diseased"}
            </span>
          </div>

          <p className="mt-1 text-lg font-medium text-ink">{result.disease}</p>
          {result.pathogen && result.pathogen !== "-" && (
            <p className="text-sm italic text-muted">{result.pathogen}</p>
          )}

          <div className="mt-3 flex flex-wrap gap-x-5 gap-y-1 text-xs text-muted">
            <span>Segmentation: {result.segmentation?.method}</span>
            <span>Leaf area: {((result.segmentation?.leaf_ratio ?? 0) * 100).toFixed(1)}%</span>
            <span>Margin: {result.uncertainty?.margin.toFixed(1)}%</span>
            <span>Entropy: {result.uncertainty?.entropy_bits.toFixed(2)} bits</span>
            <span>Latency: {result.timing_ms.toFixed(0)} ms</span>
          </div>
        </div>
      </div>

      <div className="mt-5 grid gap-4 sm:grid-cols-2">
        <div className="rounded-xl border border-line bg-paper p-4">
          <div className="mono-label mb-1">Symptoms</div>
          <p className="text-sm leading-relaxed">{result.description}</p>
        </div>
        <div className="rounded-xl border border-accent/30 bg-accent-soft p-4">
          <div className="mono-label mb-1 text-accent">Management</div>
          <p className="text-sm leading-relaxed">{result.treatment}</p>
        </div>
      </div>

      <div className="mt-5">
        <div className="mono-label mb-2">Top predictions</div>
        <ul className="space-y-2">
          {(result.top_predictions ?? []).map((prediction, index) => (
            <li key={prediction.class_name}>
              <div className="flex items-baseline justify-between gap-3 text-sm">
                <span className="truncate">
                  {prediction.plant} · {prediction.disease}
                </span>
                <span className="font-mono text-xs">{prediction.confidence.toFixed(2)}%</span>
              </div>
              <div className="mt-1 h-2 overflow-hidden rounded-full bg-slate-100">
                <motion.div
                  className={index === 0 ? "h-full bg-accent" : "h-full bg-slate-300"}
                  initial={{ width: 0 }}
                  animate={{ width: `${prediction.confidence}%` }}
                  transition={{ duration: 0.6, delay: index * 0.08 }}
                />
              </div>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}
