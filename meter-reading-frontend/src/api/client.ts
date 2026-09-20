/**
 * src/api/client.ts
 *
 * Thin wrapper over the backend's 4 endpoints. Nothing in here renders
 * anything -- this file's only job is "talk to the API and return typed
 * data", so components never construct a fetch() call or a URL string
 * themselves. Mirrors the backend's own layering (routes call db/crud.py
 * rather than writing SQL inline) -- same idea, one level up.
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;

/** Thrown when the API responds with a non-2xx status. Carries the
 * backend's own `detail` message (FastAPI's HTTPException shape)
 * so callers can show something more useful than a generic error. */
export class ApiError extends Error {
  status: number;

  constructor(status: number, detail: string) {
    super(detail);
    this.status = status;
  }
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, options);

  if (!res.ok) {
    // FastAPI's HTTPException responses look like { "detail": "..." } --
    // fall back to the status text if the body isn't in that shape for
    // some reason (e.g. a network-level error page).
    const body = await res.json().catch(() => null);
    throw new ApiError(res.status, body?.detail ?? res.statusText);
  }

  return res.json() as Promise<T>;
}

// ---- POST /gauges ---------------------------------------------------

export interface EnqueueResponse {
  job_id: string;
  status: string;
}

/** Uploads a photo and enqueues Stage 1 processing. Returns immediately
 * with a job_id -- the actual pipeline work happens asynchronously. */
export function enqueueGauge(file: File): Promise<EnqueueResponse> {
  const formData = new FormData();
  formData.append("file", file);

  return request<EnqueueResponse>("/gauges", {
    method: "POST",
    body: formData,
    // No Content-Type header set deliberately -- the browser sets the
    // correct multipart/form-data boundary itself when the body is a
    // FormData object. Setting it manually here would omit the boundary
    // and the backend's UploadFile parsing would fail.
  });
}

// ---- GET /jobs/{job_id} ----------------------------------------------

export interface JobStatusResponse {
  job_id: string;
  status: string;
  detail: string | null;
  gauge_id: string | null;
}

/** Polls a job's current status. Callers are responsible for deciding
 * when/how often to call this again (see useJobPolling, next file). */
export function getJobStatus(jobId: string): Promise<JobStatusResponse> {
  return request<JobStatusResponse>(`/jobs/${jobId}`);
}

// ---- GET /gauges/{gauge_id} --------------------------------------------

export interface CandidateOut {
  text: string;
  parsed_value: number;
  rank: number;
  is_forced_zero: boolean;
}

export interface GaugeKeypoints {
  center: [number, number];
  tip: [number, number];
  min: [number, number];
  max: [number, number];
}

export interface GaugeResultResponse {
  gauge_id: string;
  image_url: string | null;
  bbox: [number, number, number, number] | null;
  keypoints: GaugeKeypoints | null;
  pos_a_candidates: CandidateOut[];
  pos_b_candidates: CandidateOut[];
}

/** Fetches a gauge's keypoints + Stage 3 candidates, once its job is "done". */
export function getGaugeResult(gaugeId: string): Promise<GaugeResultResponse> {
  return request<GaugeResultResponse>(`/gauges/${gaugeId}`);
}

// ---- POST /gauges/{gauge_id}/explain ------------------------------------

export interface ExplainResponse {
  position_a: string | null;
  position_b: string | null;
}

/** Asks the LLM agent to comment on which candidate looks most plausible
 * for each position -- advisory only, never a decision. */
export function explainCandidates(gaugeId: string): Promise<ExplainResponse> {
  return request<ExplainResponse>(`/gauges/${gaugeId}/explain`, { method: "POST" });
}

// ---- GET /gauges (history list) ----------------------------------------

export interface HistoryItem {
  gauge_id: string;
  image_url: string | null;
  created_at: string;
  chosen_min_value: number;
  chosen_max_value: number;
  computed_value: number;
  actual_value: number | null;
  judgment: "correct" | "incorrect" | "unclear" | null;
  swapped: boolean;
}

/** Lists finished judgments, most recent first. */
export function getHistory(limit = 50, offset = 0): Promise<HistoryItem[]> {
  return request<HistoryItem[]>(`/gauges?limit=${limit}&offset=${offset}`);
}

// ---- POST /gauges/{gauge_id}/preview -----------------------------------

export interface PreviewRequest {
  chosen_min_value: number;
  chosen_max_value: number;
  swapped?: boolean;
}

export interface PreviewResponse {
  computed_value: number;
}

/** Computes what the reading WOULD be for a given min/max choice,
 * WITHOUT saving anything -- lets the person see the number before
 * deciding whether to judge it correct/incorrect. */
export function previewSelection(gaugeId: string, body: PreviewRequest): Promise<PreviewResponse> {
  return request<PreviewResponse>(`/gauges/${gaugeId}/preview`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

// ---- POST /gauges/{gauge_id}/selection ----------------------------------

export interface SelectionRequest {
  chosen_min_value: number;
  chosen_max_value: number;
  swapped?: boolean;
  judgment?: "correct" | "incorrect" | "unclear";
  actual_value?: number;
  manual_correction?: string;
  note?: string;
}

export interface SelectionResponse {
  gauge_id: string;
  computed_value: number;
  swapped: boolean;
}

/** Submits the human's final min/max choice; the backend computes and
 * returns the actual gauge reading via pointer_math.angle_utils. */
export function submitSelection(
  gaugeId: string,
  body: SelectionRequest,
): Promise<SelectionResponse> {
  return request<SelectionResponse>(`/gauges/${gaugeId}/selection`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}