// Chest Xpert Mobile — API client configuration
// The backend URL must be set via environment variable or app config.
// Never hardcode credentials or internal IPs in the source.

import axios, { AxiosInstance, AxiosError } from "axios";

const API_BASE_URL = process.env.EXPO_PUBLIC_API_BASE_URL || "http://localhost:8000";
const API_KEY = process.env.EXPO_PUBLIC_API_KEY || "";

export const API_BASE = API_BASE_URL;

let api: AxiosInstance;

export function getApiClient(): AxiosInstance {
  if (!api) {
    api = axios.create({
      baseURL: `${API_BASE_URL}/api/v1`,
      timeout: 30000,
      headers: {
        "Content-Type": "application/json",
        ...(API_KEY ? { "X-API-Key": API_KEY } : {}),
      },
    });
    api.interceptors.response.use(
      (res) => res,
      (err: AxiosError) => {
        console.error("API error:", err.response?.status, err.message);
        return Promise.reject(err);
      }
    );
  }
  return api;
}

export default getApiClient();
