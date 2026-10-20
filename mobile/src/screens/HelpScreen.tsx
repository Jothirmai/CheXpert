import React from "react";
import {
  View,
  Text,
  ScrollView,
  TouchableOpacity,
  Linking,
} from "react-native";
import { useNavigation } from "@react-navigation/native";

import { Colors, spacing, typography } from "../theme";
import { CompositeButton } from "../components/CompositeButton";
import { Card } from "../components/Card";
import { StatusBar } from "expo-status-bar";

export function HelpScreen() {
  const navigation = useNavigation();

  const onEmail = () => {
    Linking.openURL("mailto:?subject=Chest Xpert Support").catch(() => {});
  };

  return (
    <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={{ padding: spacing.md }}>
      <StatusBar style="auto" />

      <View style={{ alignItems: "center", marginBottom: spacing.lg }}>
        <Text style={{ ...typography.h2, textAlign: "center" }}>Help & Troubleshooting</Text>
      </View>

      <Card style={{ marginBottom: spacing.md }}>
        <Text style={{ ...typography.h3 }}>Common Issues</Text>
        <Text style={{ ...typography.body, marginTop: spacing.sm }}>
          <Text style={{ fontWeight: "600" }}>Image rejected as not a chest X-ray:</Text>{" "}
          Make sure you are uploading a frontal or lateral chest radiograph.
          Photographs, documents, and cartoons will be rejected.
        </Text>
        <Text style={{ ...typography.body, marginTop: spacing.sm }}>
          <Text style={{ fontWeight: "600" }}>Analysis takes too long:</Text>{" "}
          The backend may be processing other jobs. Wait and refresh the report.
        </Text>
        <Text style={{ ...typography.body, marginTop: spacing.sm }}>
          <Text style={{ fontWeight: "600" }}>Report not available:</Text>{" "}
          Ensure the analysis has completed successfully before requesting the report.
        </Text>
      </Card>

      <Card style={{ marginBottom: spacing.md }}>
        <Text style={{ ...typography.h3 }}>Supported File Types</Text>
        <Text style={{ ...typography.body }}>
          PNG, JPG, JPEG, DICOM (.dcm/.dicom). Maximum 50 MB.
        </Text>
      </Card>

      <Card style={{ marginBottom: spacing.md }}>
        <Text style={{ ...typography.h3 }}>Privacy</Text>
        <Text style={{ ...typography.body }}>
          Images are sent to your configured backend for inference. Reports are
          not stored permanently unless configured. No patient data is logged.
        </Text>
      </Card>

      <CompositeButton title="Contact Support" onPress={onEmail} icon="✉" type="outline" />
      <CompositeButton title="Back" onPress={() => navigation.goBack()} type="outline" marginTop={spacing.sm} />

      <Text style={{ ...typography.caption, textAlign: "center", marginTop: spacing.lg }}>
        AI-generated research draft requiring review by a qualified radiologist. Not a final clinical diagnosis.
      </Text>
    </ScrollView>
  );
}
