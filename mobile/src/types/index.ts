export interface LabelScore {
  label: string;
  score: number | null;
  supported: boolean;
  severity: string;
}

export interface ModalityInfo {
  decision: "accepted" | "rejected" | "uncertain" | "technical_failure" | null;
  reason: string | null;
  model: string | null;
  latency_ms: number | null;
}

export interface Analysis {
  analysis_id: string;
  status: "queued" | "processing" | "completed" | "failed";
  created_at: string;
  modality: ModalityInfo;
  labels: LabelScore[];
  inference_latency_ms: number | null;
  total_latency_ms: number | null;
  report_available: boolean;
  grad_cam_available: boolean;
  disclaimer: string;
}

export interface AnalysisCreateData {
  patient_id?: string;
  exam_date?: string;
  modality?: string;
  view?: string;
  institution?: string;
  referring_physician?: string;
  generate_grad_cam?: boolean;
}

export interface ReportExportRequest {
  format: "pdf" | "html";
  impression?: string;
  reviewer_name?: string;
  review_status?: string;
  notes?: string;
}

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  model_loaded: boolean;
  device: string;
}

export const DISCLAIMER =
  "AI-generated research draft requiring review by a qualified radiologist. Not a final clinical diagnosis. Prediction scores are not calibrated probabilities or clinical confidence.";

export const SEVERITY_LABELS: Record<string, string> = {
  high: "High",
  moderate: "Moderate",
  low: "Low",
  minimal: "Minimal",
  "not assessed": "Not assessed",
};

export const SEVERITY_COLORS: Record<string, string> = {
  high: "#dc2626",
  moderate: "#ea580c",
  low: "#ca8a04",
  minimal: "#16a34a",
  "not assessed": "#6b7280",
};

export const MODALITY_COLORS: Record<string, string> = {
  accepted: "#166534",
  rejected: "#991b1b",
  uncertain: "#92400e",
  technical_failure: "#6b7280",
};
