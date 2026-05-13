export type Signal = {
  label: string;
  confidence: number;
  source: string;
  passed: boolean;
  weight: number;
  weighted_score: number;
};

export type DetectionResponse = {
  request_id: string;
  title: string;
  summary: string;
  scene: string;
  risk_level: "low" | "watch" | "high" | "critical";
  fusion_confidence: number;
  weapon_count: number;
  weapon_detections: Array<{ label: string; class_id?: number; confidence: number; box: number[] }>;
  has_weapon_yolo: boolean;
  gun_score_police: number;
  knife_score_police: number;
  police_detected: boolean;
  police_score: number;
  violence_score_police: number;
  violence_score_lstm: number;
  accident_class: number | null;
  accident_confidence: number;
  signals: Signal[];
  component_errors: Record<string, string>;
  component_latency_ms: Record<string, number>;
  image_base64: string;
  saved_path?: string | null;
};

export type LogItem = {
  timestamp: string;
  source: string;
  summary: string;
  risk_level: string;
  confidence: number;
  errors: Record<string, string>;
  latency_ms: Record<string, number>;
};

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

async function parseJson<T>(response: Response): Promise<T> {
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const message = payload?.detail ?? `Request failed with status ${response.status}`;
    throw new Error(typeof message === "string" ? message : JSON.stringify(message));
  }
  return payload as T;
}

export async function uploadMedia(file: File, mode: "auto" | "image" | "video", threshold: number, saveEvidence: boolean) {
  const form = new FormData();
  form.append("file", file);
  form.append("mode", mode);
  form.append("threshold", String(threshold));
  form.append("save_evidence", String(saveEvidence));
  const response = await fetch(`${API_BASE}/upload`, { method: "POST", body: form });
  return parseJson<DetectionResponse>(response);
}

export async function liveDetect(blob: Blob, threshold: number, detectionEnabled: boolean, frameIndex: number, saveEvidence = false) {
  const form = new FormData();
  form.append("frame", blob, `frame-${frameIndex}.jpg`);
  form.append("threshold", String(threshold));
  form.append("detection_enabled", String(detectionEnabled));
  form.append("frame_index", String(frameIndex));
  form.append("save_evidence", String(saveEvidence));
  const response = await fetch(`${API_BASE}/live-detect`, { method: "POST", body: form });
  return parseJson<DetectionResponse>(response);
}

export async function fetchLogs() {
  const response = await fetch(`${API_BASE}/logs?limit=80`);
  return parseJson<{ items: LogItem[] }>(response);
}

export async function fetchHealth() {
  const response = await fetch(`${API_BASE}/health`);
  return parseJson<{ status: string; models_loaded: boolean; models?: Record<string, string>; error?: string }>(response);
}

export function imageSrc(result?: DetectionResponse | null) {
  return result?.image_base64 ? `data:image/jpeg;base64,${result.image_base64}` : "";
}
