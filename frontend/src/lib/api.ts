// Backend API configuration and helpers

export const API_BASE_URL = "http://127.0.0.1:5000";

export interface AudioModule {
  status: string;
  score?: number | null;
  model_prediction?: string;
  model_confidence?: number;
  duration?: number;
  error?: string;
  message?: string;
  details?: string[];
  features?: {
    pitch_mean: number;
    pitch_std: number;
    energy: number;
    mfcc_mean: number;
    pause_ratio: number;
    speech_rate: number;
  };
}

export interface FacialModule {
  status: string;
  score?: number | null;
  class?: string | null;
  error?: string;
  message?: string;
  frames_processed?: number;
  face_detected_frames?: number;
  features?: {
    blink_rate: number;
    eye_contact_percentage: number;
    head_movement_frequency: number;
    emotion_stability: number;
    emotion_confidence: number;
  } | null;
}

export interface FusionModule {
  status: string;
  analysis_mode?: string;
  message?: string;
  final_confidence_class?: string;
  final_confidence_score?: number | null;
  features_used?: {
    voice_confidence_score: number | null;
    facial_confidence_score: number | null;
  };
  error?: string;
}

export interface AnalysisResult {
  source_id: string;
  timestamp: string;
  analysis_mode: string;
  audio_module: AudioModule;
  facial_module: FacialModule;
  fusion_module: FusionModule;
  warnings: string[];
}

export interface HistoryEntry {
  id: string;
  timestamp: string;
  analysis_mode: string;
  confidence_class: string;
  confidence_score: number | null;
  audio_score: number | null;
  facial_score: number | null;
  audio_status: string;
  facial_status: string;
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/health`);
    return res.ok;
  } catch {
    return false;
  }
}

/**
 * Poll the backend health endpoint until it responds or we run out of retries.
 * Used at startup to wait for the sidecar backend to finish launching.
 *
 * @param maxRetries  Number of polling attempts (default 30 → ~30 s)
 * @param intervalMs  Milliseconds between attempts
 * @returns true if the backend became reachable, false if timed out
 */
export async function waitForBackend(
  maxRetries = 30,
  intervalMs = 1000,
): Promise<boolean> {
  for (let i = 0; i < maxRetries; i++) {
    const ok = await checkHealth();
    if (ok) return true;
    await new Promise((r) => setTimeout(r, intervalMs));
  }
  return false;
}

export async function analyzeRecording(blob: Blob, filename = "recording.webm", sourceId = "recording"): Promise<AnalysisResult> {
  const formData = new FormData();
  formData.append("video", blob, filename);
  formData.append("source_id", sourceId);

  const res = await fetch(`${API_BASE_URL}/api/analyze`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: "Unknown error" }));
    throw new Error(err.error || `Server error: ${res.status}`);
  }

  return res.json();
}

export async function fetchLatestResult(): Promise<AnalysisResult | null> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/latest-result`);
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

export async function fetchHistory(): Promise<HistoryEntry[]> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/history`);
    if (!res.ok) return [];
    const data = await res.json();
    return data.history || [];
  } catch {
    return [];
  }
}
