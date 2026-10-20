import { create } from "zustand";
import type { Analysis, AnalysisCreateData } from "../types";

interface AnalysisState {
  currentAnalysis: Analysis | null;
  isLoading: boolean;
  error: string | null;
  setAnalysis: (analysis: Analysis | null) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  reset: () => void;
  createAnalysis: (fileUri: string, data: AnalysisCreateData) => Promise<void>;
  fetchAnalysis: (analysisId: string) => Promise<void>;
}

import { createAnalysis as apiCreateAnalysis, getAnalysis as apiGetAnalysis } from "../api/analyses";

export const useAnalysisStore = create<AnalysisState>((set) => ({
  currentAnalysis: null,
  isLoading: false,
  error: null,
  setAnalysis: (analysis) => set({ currentAnalysis: analysis }),
  setLoading: (isLoading) => set({ isLoading }),
  setError: (error) => set({ error }),
  reset: () => set({ currentAnalysis: null, isLoading: false, error: null }),
  createAnalysis: async (fileUri, data) => {
    set({ isLoading: true, error: null });
    try {
      const analysis = await apiCreateAnalysis(fileUri, data);
      set({ currentAnalysis: analysis, isLoading: false });
    } catch (err: any) {
      set({ error: err?.response?.data?.detail || err.message || "Analysis failed", isLoading: false });
    }
  },
  fetchAnalysis: async (analysisId) => {
    set({ isLoading: true, error: null });
    try {
      const analysis = await apiGetAnalysis(analysisId);
      set({ currentAnalysis: analysis, isLoading: false });
    } catch (err: any) {
      set({ error: err?.response?.data?.detail || err.message || "Failed to load analysis", isLoading: false });
    }
  },
}));
