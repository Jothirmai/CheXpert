import React, { useCallback } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  ScrollView,
  Image,
  ActivityIndicator,
  Platform,
  Linking,
} from "react-native";
import { useNavigation } from "@react-navigation/native";

import { Colors, spacing, typography } from "../theme";
import type { RootStackParamList } from "../navigation/AppNavigator";
import { CompositeButton } from "../components/CompositeButton";
import { Card } from "../components/Card";
import { HealthCheckStatus } from "../components/HealthCheckStatus";

export function HomeScreen() {
  const navigation = useNavigation<React.NavigationProp<RootStackParamList>>();

  const onNewAnalysis = useCallback(() => {
    navigation.navigate("Analysis");
  }, [navigation]);

  const onCheckStatus = useCallback(async () => {
    // Placeholder — the app checks backend health on demand.
    Linking.openURL("http://localhost:8000/api/v1/health").catch(() => {});
  }, []);

  return (
    <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={{ padding: spacing.md }}>
      <StatusBarComponent />

      <View style={{ alignItems: "center", marginBottom: spacing.lg }}>
        <View style={{ width: 64, height: 64, borderRadius: 16, backgroundColor: Colors.surface, alignItems: "center", justifyContent: "center", marginBottom: spacing.md }}>
          <Text style={{ fontSize: 32 }}>🩺</Text>
        </View>
        <Text style={{ ...typography.h1, textAlign: "center" }}>Chest Xpert</Text>
        <Text style={{ ...typography.bodyMuted, textAlign: "center", marginTop: spacing.sm }}>
          Chest X-ray multi-label findings inference
        </Text>
      </View>

      <Card style={{ marginBottom: spacing.md }}>
        <Text style={{ ...typography.h3, marginBottom: spacing.sm }}>Quick Actions</Text>
        <CompositeButton
          title="New X-ray Analysis"
          onPress={onNewAnalysis}
          icon="📷"
          type="primary"
        />
        <CompositeButton
          title="Check Backend Status"
          onPress={onCheckStatus}
          icon="🔌"
          type="secondary"
          marginTop={spacing.sm}
        />
      </Card>

      <Card style={{ marginBottom: spacing.md }}>
        <Text style={{ ...typography.h3, marginBottom: spacing.sm }}>About</Text>
        <Text style={{ ...typography.body }}>
          Chest Xpert predicts 18 chest X-ray findings using a pre-trained
          DenseNet-121 model. No training or fine-tuning is performed.
        </Text>
        <CompositeButton
          title="Models & Methodology"
          onPress={() => navigation.navigate("Methodology")}
          icon="📋"
          type="outline"
          marginTop={spacing.sm}
        />
      </Card>

      <Card style={{ marginBottom: spacing.md }}>
        <Text style={{ ...typography.h3, marginBottom: spacing.sm }}>Help</Text>
        <Text style={{ ...typography.body }}>
          Need assistance with capture, upload, or report interpretation?
        </Text>
        <CompositeButton
          title="Help & Troubleshooting"
          onPress={() => navigation.navigate("Help")}
          icon="❓"
          type="outline"
          marginTop={spacing.sm}
        />
      </Card>

      <Text style={{ ...typography.caption, textAlign: "center", marginTop: spacing.lg }}>
        AI-generated research draft requiring review by a qualified radiologist. Not a final clinical diagnosis.
      </Text>
      <StatusBar />
    </ScrollView>
  );
}

function StatusBarComponent() {
  return <StatusBar style="auto" />;
}
