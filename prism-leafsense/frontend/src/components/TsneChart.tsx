"use client";

import { useMemo } from "react";
import {
  CartesianGrid,
  Legend,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ResponsiveContainer,
} from "recharts";
import type { TsnePayload } from "@/lib/api";

/** Colour by plant genus so 38 classes stay readable. */
const PALETTE = [
  "#059669", "#2563eb", "#b45309", "#7c3aed", "#dc2626",
  "#0891b2", "#65a30d", "#db2777", "#475569", "#ca8a04",
  "#0d9488", "#4f46e5", "#9333ea", "#16a34a",
];

export default function TsneChart({ tsne }: { tsne: TsnePayload }) {

  const groups = useMemo(() => {
    const byPlant = new Map<string, { x: number; y: number; label: string }[]>();
    tsne.points.forEach((point, index) => {
      const className = tsne.class_names[tsne.labels[index]] ?? "unknown";
      const plant = className.split("___")[0].replace(/_/g, " ");
      if (!byPlant.has(plant)) byPlant.set(plant, []);
      byPlant.get(plant)!.push({ x: point[0], y: point[1], label: className.replace(/_/g, " ") });
    });
    return Array.from(byPlant.entries()).sort((a, b) => a[0].localeCompare(b[0]));
  }, [tsne]);

  return (
    <section className="rounded-2xl border border-line bg-white p-5 shadow-sm sm:p-6">
      <div className="mono-label mb-1">Latent space</div>
      <h2 className="mb-3 text-lg font-semibold">t-SNE of the 256-d sparse code</h2>

      <div className="h-80 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 8, right: 8, bottom: 8, left: 0 }}>
            <CartesianGrid stroke="#eef2f7" />
            <XAxis type="number" dataKey="x" tick={{ fontSize: 10 }} stroke="#94a3b8" />
            <YAxis type="number" dataKey="y" tick={{ fontSize: 10 }} stroke="#94a3b8" />
            <Tooltip
              cursor={{ strokeDasharray: "3 3" }}
              contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e2e8f0" }}
              formatter={(_value, _name, entry) => [
                (entry?.payload as { label?: string })?.label ?? "",
                "",
              ]}
            />
            <Legend wrapperStyle={{ fontSize: 10 }} />
            {groups.map(([plant, data], index) => (
              <Scatter
                key={plant}
                name={plant}
                data={data}
                fill={PALETTE[index % PALETTE.length]}
                fillOpacity={0.45}
                shape="circle"
                legendType="circle"
              />
            ))}
            <Scatter
              name="Your leaf"
              data={[{ x: tsne.sample[0], y: tsne.sample[1], label: "uploaded sample" }]}
              fill="#0f172a"
              shape="star"
              legendType="star"
            />
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      <p className="mt-2 text-xs text-muted">
        The dark star is your upload, positioned by its nearest neighbours in latent space.
      </p>
    </section>
  );
}
