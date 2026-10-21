import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  ScrollView,
  Image,
  ActivityIndicator,
  Platform,
  Alert,
  Dimensions,
} from "react-native";
import { useNavigation } from "@react-navigation/native";
import * as ImagePicker from "expo-image-picker";
import * as Camera from "expo-camera";
import { StatusBar } from "expo-status-bar";

import { Colors, spacing, typography, SEVERITY_COLORS } from "../theme";
import type { RootStackParamList } from "../navigation/AppNavigator";
import { useAnalysisStore } from "../hooks/useAnalysisStore";
import { createAnalysis } from "../api/analyses";
import { Card } from "../components/Card";
import { CompositeButton } from "../components/CompositeButton";
import { SeverityBadge } from "../components/SeverityBadge";

const { width } = Dimensions.get("window");

type AnalysisFlowState = "idle" | "capturing" | "previewing" | "uploading" | "completed" | "error";

export function AnalysisScreen() {
  const navigation = useNavigation<React.NavigationProp<RootStackParamList>>();
  const { currentAnalysis, isLoading, error, reset, createAnalysis: createAnalysisAction } =
    useAnalysisStore();

  const [flow, setFlow] = useState<AnalysisFlowState>("idle");
  const [imageUri, setImageUri] = useState<string | null>(null);
  const [cameraPermission, requestCameraPermission] = Camera.useCameraPermissions();
  const [galleryPermission, requestGalleryPermission] = ImagePicker.useMediaLibraryPermissions();
  const cameraRef = useRef<Camera.CameraView>(null);

  const onBack = useCallback(() => {
    reset();
    setFlow("idle");
    setImageUri(null);
    navigation.goBack();
  }, [navigation, reset]);

  const onCapture = useCallback(async () => {
    if (!cameraPermission?.granted) {
      const res = await requestCameraPermission();
      if (!res.granted) {
        Alert.alert("Camera permission required to capture images.");
        return;
      }
    }
    if (cameraRef.current) {
      const photo = await cameraRef.current.takePictureAsync();
      if (!photo.canceled) {
        setImageUri(photo.uri);
        setFlow("previewing");
      }
    }
  }, [cameraPermission, requestCameraPermission]);

  const onChooseGallery = useCallback(async () => {
    if (!galleryPermission?.granted) {
      const res = await requestGalleryPermission();
      if (!res.granted) {
        Alert.alert("Gallery permission required to select images.");
        return;
      }
    }
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsMultipleSelection: false,
      aspect: [4, 3],
      quality: 0.8,
    });
    if (!result.canceled && result.assets[0]) {
      setImageUri(result.assets[0].uri);
      setFlow("previewing");
    }
  }, [galleryPermission, requestGalleryPermission]);

  const onConfirmUpload = useCallback(async () => {
    if (!imageUri) return;
    setFlow("uploading");
    try {
      await createAnalysisAction(imageUri, {
        patient_id: "",
        exam_date: "",
        modality: "Chest X-ray",
        view: "",
        institution: "",
        referring_physician: "",
        generate_grad_cam: true,
      });
      setFlow("completed");
    } catch (e: any) {
      setFlow("error");
      Alert.alert("Upload failed", e?.message || "Unknown error");
    }
  }, [imageUri, createAnalysisAction]);

  const onRetry = useCallback(() => {
    setFlow("idle");
    setImageUri(null);
    reset();
  }, [reset]);

  const onViewReport = useCallback(() => {
    if (currentAnalysis?.analysis_id) {
      navigation.navigate("Report", { analysisId: currentAnalysis.analysis_id });
    }
  }, [navigation, currentAnalysis]);

  // Poll for completion when processing
  useEffect(() => {
    if (currentAnalysis?.status === "processing" || currentAnalysis?.status === "queued") {
      const timer = setInterval(async () => {
        // The store's fetchAnalysis is not directly available here; we rely on user refresh.
      }, 3000);
      return () => clearInterval(timer);
    }
  }, [currentAnalysis]);

  return (
    <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={{ padding: spacing.md }}>
      <StatusBarComponent />

      {flow === "idle" && (
        <>
          <View style={{ alignItems: "center", marginBottom: spacing.lg }}>
            <Text style={{ ...typography.h2, textAlign: "center" }}>New X-ray Analysis</Text>
            <Text style={{ ...typography.bodyMuted, textAlign: "center", marginTop: spacing.sm }}>
              Capture or select a chest X-ray image for analysis.
            </Text>
          </View>

          <Card style={{ marginBottom: spacing.md }}>
            <CompositeButton
              title="Take Photo"
              onPress={onCapture}
              icon="📷"
              type="primary"
              disabled={!cameraPermission?.granted && !cameraPermission?.canRequest}
            />
            <CompositeButton
              title="Choose from Gallery"
              onPress={onChooseGallery}
              icon="🖼️"
              type="secondary"
              marginTop={spacing.sm}
            />
          </Card>

          <CompositeButton
            title="Back to Home"
            onPress={onBack}
            icon="◀"
            type="outline"
          />
        </>
      )}

      {flow === "previewing" && imageUri && (
        <>
          <View style={{ alignItems: "center", marginBottom: spacing.lg }}>
            <Text style={{ ...typography.h2, textAlign: "center" }}>Confirm Image</Text>
            <Text style={{ ...typography.bodyMuted, textAlign: "center", marginTop: spacing.sm }}>
              Review the selected X-ray before uploading.
            </Text>
          </View>

          <Card style={{ marginBottom: spacing.md, alignItems: "center" }}>
            <View style={{ width: width - 48, height: width - 48, backgroundColor: Colors.surface, borderRadius: borderRadiusMd, overflow: "hidden", alignItems: "center", justifyContent: "center" }}>
              <Image source={{ uri: imageUri }} style={{ width: "100%", height: "100%" }} resizeMode="cover" />
            </View>
          </View>

          <View style={{ flexDirection: "row", gap: spacing.sm, marginBottom: spacing.md }}>
            <CompositeButton
              title="Retake"
              onPress={() => setFlow("idle")}
              icon="🔄"
              type="outline"
              flex={1}
            />
            <CompositeButton
              title="Upload & Analyze"
              onPress={onConfirmUpload}
              icon="↑"
              type="primary"
              flex={2}
            />
          </View>

          <CompositeButton
            title="Back to Home"
            onPress={onBack}
            icon="◀"
            type="text"
          />
        </>
      )}

      {flow === "uploading" && (
        <>
          <View style={{ alignItems: "center", paddingVertical: spacing.lg }}>
            <ActivityIndicator size="large" color={Colors.primary} />
            <Text style={{ ...typography.body, marginTop: spacing.md }}>Analyzing image...</Text>
            <Text style={{ ...typography.caption, marginTop: spacing.sm }}>
              This may take a few moments. Please keep the app open.
            </Text>
          </View>
          <CompositeButton title="Cancel" onPress={onRetry} type="text" />
        </>
      )}

      {flow === "completed" && currentAnalysis && (
        <>
          <View style={{ alignItems: "center", marginBottom: spacing.lg }}>
            <Text style={{ ...typography.h2, textAlign: "center" }}>Analysis Complete</Text>
            <Text style={{ ...typography.bodyMuted, textAlign: "center", marginTop: spacing.sm }}>
              Your X-ray has been analyzed. Review the report below.
            </Text>
          </View>

          <Card style={{ marginBottom: spacing.md }}>
            <Text style={{ ...typography.h3, marginBottom: spacing.sm }}>Results Summary</Text>
            <Text style={{ ...typography.body }}>
              Modality: <Text style={{ fontWeight: "600" }}>{currentAnalysis.modality.decision ?? "—"}</Text>
            </Text>
            <Text style={{ ...typography.body }}>
              Labels assessed:{" "}
              <Text style={{ fontWeight: "600" }}>
                {currentAnalysis.labels.filter((l) => l.supported).length}/18
              </Text>
            </Text>
            <Text style={{ ...typography.body }}>
              Inference complete.
            </Text>
          </Card>

          <CompositeButton
            title="View Full Report"
            onPress={onViewReport}
            icon="📄"
            type="primary"
          />
          <CompositeButton
            title="New Analysis"
            onPress={onRetry}
            icon="🔄"
            type="outline"
            marginTop={spacing.sm}
          />
        </>
      )}

      {flow === "error" && (
        <>
          <View style={{ alignItems: "center", marginBottom: spacing.lg }}>
            <Text style={{ ...typography.h2, textAlign: "center" }}>{Colors.danger}>Upload Error</Text>
            <Text style={{ ...typography.bodyMuted, textAlign: "center", marginTop: spacing.sm }}>
              {error || "Failed to upload and analyze the image."}
            </Text>
          </View>
          <CompositeButton title="Try Again" onPress={onRetry} icon="🔄" type="primary" />
          <CompositeButton title="Back to Home" onPress={onBack} icon="◀" type="outline" marginTop={spacing.sm} />
        </>
      )}

      {isLoading && flow !== "completed" && flow !== "uploading" && (
        <View style={{ alignItems: "center", paddingVertical: spacing.lg }}>
          <ActivityIndicator size="large" color={Colors.primary} />
        </View>
      )}

      <Text style={{ ...typography.caption, textAlign: "center", marginTop: spacing.lg }}>
        {DISCLAIMER_CONST}
      </Text>
      <StatusBar />
    </ScrollView>
  );
}

function StatusBarComponent() {
  return <StatusBar style="auto" />;
}

const borderRadiusMd = 12;

const DISCLAIMER_CONST =
  "AI-generated research draft requiring review by a qualified radiologist. Not a final clinical diagnosis. Prediction scores are not calibrated probabilities or clinical diagnosis.";
