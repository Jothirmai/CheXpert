import React from "react";
import {
  View,
  Text,
  ScrollView,
  TouchableOpacity,
} from "react-native";
import { useNavigation } from "@react-navigation/native";

import { Colors, spacing, typography } from "../theme";
import { CompositeButton } from "../components/CompositeButton";
import { Card } from "../components/Card";
import { StatusBar } from "expo-status-bar";

export function MethodologyScreen() {
  const navigation = useNavigation();

  return (
    <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={{ padding: spacing.md }}>
      <StatusBar style="auto" />

      <View style={{ alignItems: "center", marginBottom: spacing.lg }}>
        <Text style={{ ...typography.h2, textAlign: "center" }}>Models & Methodology</Text>
      </View>

      <Card style={{ marginBottom: spacing.md }}>
        <Text style={{ ...typography.h3 }}>Classifier</Text>
        <Text style={{ ...typography.body }}>
          torchxrayvision DenseNet-121 (densenet121-res224-all) — pre-trained
          weights by other researchers (Irvin et al., CheXpert). Not trained by
          this project.
        </Text>
        <Text style={{ ...typography.bodyMuted }}>
          Input: 224×224 grayscale · Output: 18 independent pathology scores
          (sigmoid, not calibrated probabilities).
        </Text>
      </Card>

      <Card style={{ marginBottom: spacing.md }}>
        <Text style={{ ...typography.h3 }}>Supported Findings (18)</Text>
        <Text style={{ ...typography.body }}>
          Atelectasis, Consolidation, Infiltration, Pneumothorax, Edema,
          Emphysema, Fibrosis, Effusion, Pneumonia, Pleural Thickening,
          Cardiomegaly, Nodule, Mass, Hernia, Lung Lesion, Fracture, Lung
          Opacity, Enlarged Cardiomediastinum.
        </Text>
      </Card>

      <Card style={{ marginBottom: spacing.md }}>
        <Text style={{ ...typography.h3 }}>Modality Validation</Text>
        <Text style={{ ...typography.body }}>
          Images are validated as chest X-rays before inference using CLIP
          zero-shot classification with a heuristic fallback. Not a validated
          clinical test.
        </Text>
      </Card>

      <Card style={{ marginBottom: spacing.md }}>
        <Text style={{ ...typography.h3 }}>Explainability</Text>
        <Text style={{ ...typography.body }}>
          Grad-CAM heatmaps (target layer: features.norm5) are available for
          supported findings. Explanations, not proof of disease.
        </Text>
      </Card>

      <CompositeButton title="Back" onPress={() => navigation.goBack()} type="outline" />

      <Text style={{ ...typography.caption, textAlign: "center", marginTop: spacing.lg }}>
        AI-generated research draft requiring review by a qualified radiologist. Not a final clinical diagnosis.
      </Text>
    </ScrollView>
  );
}
