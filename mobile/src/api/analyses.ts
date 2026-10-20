import { getApiClient } from "./client";
import type {
  Analysis,
  AnalysisCreateData,
  HealthResponse,
  ReportExportRequest,
} from "../types";

const api = getApiClient();

export const healthCheck = async (): Promise<HealthResponse> => {
  const res = await api.get<HealthResponse>("/health");
  return res.data;
};

export const readinessCheck = async (): Promise<{ ready: boolean; checks: Record<string, string> }> => {
  const res = await api.get<{ ready: boolean; checks: Record<string, string> }>("/ready");
  return res.data;
};

export const createAnalysis = async (
  fileUri: string,
  data: AnalysisCreateData,
  mimeType: string = "image/png"
): Promise<Analysis> => {
  const formData = new FormData();
  formData.append("file", {
    uri: fileUri,
    type: mimeType,
    name: "upload.png",
  } as any);
  formData.append("patient_id", data.patient_id ?? "");
  formData.append("exam_date", data.exam_date ?? "");
  formData.append("modality", data.modality ?? "Chest X-ray");
  formData.append("view", data.view ?? "");
  formData.append("institution", data.institution ?? "");
  formData.append("referring_physician", data.referring_physician ?? "");
  formData.append("generate_grad_cam", String(data.generate_grad_cam ?? false));

  const res = await api.post<Analysis>("/analyses", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data;
};

export const getAnalysis = async (analysisId: string): Promise<Analysis> => {
  const res = await api.get<Analysis>(`/analyses/${analysisId}`);
  return res.data;
};

export const downloadReport = async (
  analysisId: string,
  format: "pdf" | "html" = "pdf",
  requestData?: ReportExportRequest
): Promise<{ uri: string; format: string }> => {
  // Trigger report generation (returns metadata)
  const res = await api.post(`/report/${analysisId}`, requestData ?? { format });
  // Return download URL
  return {
    uri: `${api.defaults.baseURL.replace("/api/v1", "")}/api/v1/report/${analysisId}/download?format=${format}`,
    format,
  };
};
