/**
 * Typed client for the FastAPI backend.
 *
 * Every stage image arrives as a base64 PNG data URL, so results render with no
 * second round trip.
 */

export const API_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ?? "http://localhost:8000";

export type Stage = {
  id: string;
  title: string;
  model: string;
  summary: string;
  image: string | null;
  metrics: Record<string, unknown>;
  duration_ms: number;
  available: boolean;
};

export type TopPrediction = {
  class_name: string;
  plant: string;
  disease: string;
  pathogen: string;
  confidence: number;
  healthy: boolean;
};

export type ModelStatus = {
  name: string;
  loaded: boolean;
  path: string;
  detail: string;
  metrics: Record<string, number> | null;
};

export type TsnePayload = {
  sample: [number, number];
  points: number[][];
  labels: number[];
  class_names: string[];
};

export type PredictResponse = {
  available: boolean;
  detail: string;
  stages: Stage[];
  models: Record<string, ModelStatus>;
  timing_ms: number;
  plant?: string;
  disease?: string;
  pathogen?: string;
  healthy?: boolean;
  confidence?: number;
  class_name?: string;
  description?: string;
  treatment?: string;
  top_predictions?: TopPrediction[];
  segmentation?: { method: string; leaf_ratio: number };
  uncertainty?: { entropy_bits: number; margin: number };
  tsne?: TsnePayload;
};

export type HealthResponse = {
  status: string;
  device: string;
  ready: boolean;
  models: Record<string, ModelStatus>;
};

export class ApiError extends Error {
  constructor(message: string, readonly status: number, readonly payload?: unknown) {
    super(message);
    this.name = "ApiError";
  }
}

export async function predict(file: File, topK = 3): Promise<PredictResponse> {
  const body = new FormData();
  body.append("file", file);

  const response = await fetch(`${API_URL}/predict?top_k=${topK}`, { method: "POST", body });
  if (!response.ok) {
    let detail: unknown = await response.text();
    try {
      detail = JSON.parse(detail as string).detail ?? detail;
    } catch {
      /* plain-text error body */
    }
    const message =
      typeof detail === "string"
        ? detail
        : (detail as { message?: string })?.message ?? "prediction failed";
    throw new ApiError(message, response.status, detail);
  }
  return response.json();
}

export async function health(): Promise<HealthResponse> {
  const response = await fetch(`${API_URL}/health`, { cache: "no-store" });
  if (!response.ok) throw new ApiError("backend unreachable", response.status);
  return response.json();
}

export async function classes() {
  const response = await fetch(`${API_URL}/classes`, { cache: "no-store" });
  if (!response.ok) throw new ApiError("could not load classes", response.status);
  return response.json();
}
