"use client";

import { motion } from "framer-motion";
import { useState } from "react";
import type { Stage } from "@/lib/api";

function MetricList({ metrics }: { metrics: Record<string, unknown> }) {
  const rows = Object.entries(metrics).filter(
    ([key, value]) =>
      key !== "outline_preview" &&
      value !== null &&
      value !== undefined &&
      !(Array.isArray(value) && value.length === 0),
  );
  if (rows.length === 0) return null;
  return (
    <dl className="mt-3 grid grid-cols-2 gap-x-3 gap-y-1 text-[11px]">
      {rows.map(([key, value]) => (
        <div key={key} className="contents">
          <dt className="truncate text-muted">{key.replace(/_/g, " ")}</dt>
          <dd className="truncate text-right font-mono">
            {Array.isArray(value) ? value.length : String(value)}
          </dd>
        </div>
      ))}
    </dl>
  );
}

export default function StageRail({ stages }: { stages: Stage[] }) {
  const [zoom, setZoom] = useState<Stage | null>(null);

  return (
    <section className="rounded-2xl border border-line bg-white p-5 shadow-sm sm:p-6">
      <div className="mb-4 flex items-baseline justify-between gap-3">
        <div>
          <div className="mono-label">Pipeline trace</div>
          <h2 className="text-lg font-semibold">Output of every stage</h2>
        </div>
        <span className="text-xs text-muted">{stages.length} stages</span>
      </div>

      <div className="stage-rail -mx-1 flex snap-x gap-4 overflow-x-auto px-1 pb-3">
        {stages.map((stage, index) => (
          <motion.article
            key={stage.id}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.04, duration: 0.25 }}
            className={`w-64 shrink-0 snap-start rounded-xl border p-3 ${
              stage.available ? "border-line bg-paper" : "border-dashed border-line bg-slate-50"
            }`}
          >
            <div className="flex items-center justify-between">
              <span className="mono-label">
                {String(index + 1).padStart(2, "0")} · {stage.model}
              </span>
              <span className="font-mono text-[10px] text-muted">
                {stage.duration_ms.toFixed(0)} ms
              </span>
            </div>

            <h3 className="mt-1 text-sm font-semibold">{stage.title}</h3>

            <div className="mt-2 aspect-square w-full overflow-hidden rounded-lg border border-line bg-white">
              {stage.image ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={stage.image}
                  alt={stage.title}
                  onClick={() => setZoom(stage)}
                  className="h-full w-full cursor-zoom-in object-contain"
                />
              ) : (
                <div className="grid h-full place-items-center px-3 text-center text-[11px] text-muted">
                  {stage.available ? "numeric output only" : "stage unavailable"}
                </div>
              )}
            </div>

            <p className="mt-2 text-[11px] leading-snug text-muted">{stage.summary}</p>
            <MetricList metrics={stage.metrics} />
          </motion.article>
        ))}
      </div>

      {zoom && (
        <div
          className="fixed inset-0 z-50 grid place-items-center bg-ink/70 p-4"
          onClick={() => setZoom(null)}
        >
          <div className="max-h-full w-full max-w-2xl overflow-auto rounded-2xl bg-white p-4">
            <div className="mb-2 flex items-center justify-between">
              <h3 className="text-sm font-semibold">{zoom.title}</h3>
              <button type="button" className="text-sm text-muted" onClick={() => setZoom(null)}>
                ✕
              </button>
            </div>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={zoom.image ?? ""} alt={zoom.title} className="w-full rounded-lg" />
            <p className="mt-3 text-xs text-muted">{zoom.summary}</p>
          </div>
        </div>
      )}
    </section>
  );
}
