import React, { useCallback, useEffect, useState } from "react";
import {
  View,
  Text,
  ScrollView,
  ActivityIndicator,
  Alert,
  Linking,
} from "react-native";
import { useNavigation, useRoute } from "@react-navigation/native";
import * as Sharing from "expo-sharing";

import { Colors, spacing, typography } from "../theme";
import type { RootStackParamList } from "../navigation/AppNavigator";
import { useAnalysisStore } from "../hooks/useAnalysisStore";
import { fetchAnalysis } from "../api/analyses";
import { Card } from "../components/Card";
import { CompositeButton } from "../components/CompositeButton";
import { LabelRow } from "../components/LabelRow";
import { StatusBar } from "expo-status-bar";
import { DISCLAIMER } from "../types";

export function ReportScreen() {
  const navigation = useNavigation<React.NavigationProp<RootStackParamList>>();
  const route = useRoute<RootStackParamList>();
  const analysisId = (route.params as { analysisId?: string })?.analysisId;

  const { currentAnalysis, isLoading, error, setAnalysis, setLoading, setError } =
    useAnalysisStore();

  const [reportUri, setReportUri] = useState<string | null>(null);

  const loadAnalysis = useCallback(async () => {
    if (!analysisId) return;
    setLoading(true);
    setError(null);
    try {
      const analysis = await fetchAnalysis(analysisId);
      setAnalysis(analysis);
    } catch (e: any) {
      setError(e?.response?.data?.detail || e.message || "Failed to load report");
    } finally {
      setLoading(false);
    }
  }, [analysisId, setAnalysis, setLoading, setError]);

  useEffect(() => {
    loadAnalysis();
  }, [loadAnalysis]);

  const onShareReport = useCallback(async () => {
    if (!reportUri) return;
    try {
      await Sharing.shareAsync(reportUri, { dialogTitle: "Share Chest Xpert Report" });
    } catch (e) {
      Alert.alert("Share failed", "Unable to share the report at this time.");
    }
  }, [reportUri]);

  const onOpenInBrowser = useCallback(() => {
    if (!reportUri) return;
    Linking.openURL(reportUri).catch(() => {});
  }, [reportUri]);

  const onBack = useCallback(() => navigation.goBack(), [navigation]);

  if (isLoading && !currentAnalysis) {
    return (
      <View style={{ flex: 1, alignItems: "center", justifyContent: "center", padding: spacing.md }}>
        <ActivityIndicator size="large" color={Colors.primary} />
        <Text style={{ ...typography.body, marginTop: spacing.md }}>Loading report...</Text>
      </View>
    );
  }

  if (error && !currentAnalysis) {
    return (
      <View style={{ flex: 1, alignItems: "center", justifyContent: "center", padding: spacing.md }}>
        <Text style={{ ...typography.body, color: Colors.danger, textAlign: "center" }}>
          {error}
        </Text>
        <CompositeButton title="Retry" onPress={loadAnalysis} type="primary" marginTop={spacing.md} />
        <CompositeButton title="Back" onPress={onBack} type="outline" marginTop={spacing.sm} />
      </View>
    );
  }

  const analysis = currentAnalysis;
  if (!analysis) return null;

  const supportedLabels = analysis.labels.filter((l) => l.supported);

  return (
    <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={{ padding: spacing.md }}>
      <StatusBar style="auto" />

      <View style={{ alignItems: "center", marginBottom: spacing.lg }}>
        <Text style={{ ...typography.h2, textAlign: "center" }}>Analysis Report</Text>
        <Text style={{ ...typography.bodyMuted, textAlign: "center", marginTop: spacing.sm }}>
          Analysis ID: {analysis.analysis_id}
        </Text>
      </View>

      <Card style={{ marginBottom: spacing.md }}>
        <Text style={{ ...typography.h3, marginBottom: spacing.sm }}>Modality Validation</Text>
        <View style={{ flexDirection: "row", alignItems: "center", gap: spacing.sm }}>
          <View
            style={{
              width: 10,
              height: 10,
              borderRadius: 5,
              backgroundColor:
                analysis.modality.decision === "accepted"
                  ? Colors.success
                  : analysis.modality.decision === "rejected"
                  ? Colors.danger
                  : Colors.warning,
            }}
          />
          <Text style={{ ...typography.body }}>
            {analysis.modality.decision === "accepted"
              ? "Accepted — chest X-ray"
              : analysis.modality.decision === "rejected"
              ? "Rejected — not a chest X-ray"
              : analysis.modality.decision === "uncertain"
              ? "Uncertain"
              : "Technical failure"}
          </Text>
        </View>
        {analysis.modality.reason && (
          <Text style={{ ...typography.caption, marginTop: spacing.xs }}>
            {analysis.modality.reason}
          </Text>
        )}
      </Card>

      <Card style={{ marginBottom: spacing.md }}>
        <Text style={{ ...typography.h3, marginBottom: spacing.sm }}>Qwen Report Generation</Text>
        <View style={{ flexDirection: "row", alignItems: "center", gap: spacing.sm }}>
          <View
            style={{
              width: 10,
              height: 10,
              borderRadius: 5,
              backgroundColor:
                analysis.verification.available ? Colors.success : Colors.warning,
            }}
          />
          <Text style={{ ...typography.body }}>
            {analysis.verification.available
              ? "Qwen report generated"
              : "Qwen unavailable — template fallback used"}
          </Text>
        </View>
        {analysis.verification.model && (
          <Text style={{ ...typography.caption, marginTop: spacing.xs }}>
            Model: {analysis.verification.model}
          </Text>
        )}
        {analysis.verification.error && (
          <Text style={{ ...typography.caption, marginTop: spacing.xs }}>
            {analysis.verification.error}
          </Text>
        )}
      </Card>

      <Card style={{ marginBottom: spacing.md }}>
        <Text style={{ ...typography.h3, marginBottom: spacing.sm }}>Findings (18 labels)</Text>
        {supportedLabels.map((label) => (
          <LabelRow key={label.label} label={label} />
        ))}
        {analysis.labels.filter((l) => !l.supported).length > 0 && (
          <Text style={{ ...typography.caption, marginTop: spacing.sm }}>
            Some labels are not supported by the selected model.
          </Text>
        )}
      </Card>

      <Card style={{ marginBottom: spacing.md }}>
        <Text style={{ ...typography.h3, marginBottom: spacing.sm }}>Actions</Text>
        <CompositeButton
          title="View PDF Report"
          onPress={() => {
            setReportUri(
              `${"http://localhost:8000"}/api/v1/report/${analysis.analysis_id}/download?format=pdf`
            );
          }}
          icon="📄"
          type="primary"
        />
        <CompositeButton
          title="Share Report"
          onPress={onShareReport}
          icon="↗"
          type="secondary"
          marginTop={spacing.sm}
        />
        <CompositeButton
          title="Open in Browser"
          onPress={onOpenInBrowser}
          icon="🌐"
          type="outline"
          marginTop={spacing.sm}
        />
      </Card>

      <CompositeButton title="Back to Analysis" onPress={onBack} icon="◀" type="text" />

      <Text style={{ ...typography.caption, textAlign: "center", marginTop: spacing.lg }}>
        {DISCLAIMER}
      </Text>
    </ScrollView>
  );
}
