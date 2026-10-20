import React from "react";
import { View, Text, StyleSheet } from "react-native";

import { Colors, spacing, typography, SEVERITY_COLORS } from "../theme";
import type { LabelScore } from "../types";

interface LabelRowProps {
  label: LabelScore;
}

export function LabelRow({ label }: LabelRowProps) {
  const color = SEVERITY_COLORS[label.severity] || Colors.textMuted;
  return (
    <View style={styles.row}>
      <View style={styles.labelContainer}>
        <Text style={styles.labelName}>{label.label}</Text>
        <Text style={styles.labelScore}>
          {label.score != null ? label.score.toFixed(3) : "n/a"}
        </Text>
      </View>
      <View style={styles.badgeContainer}>
        <View style={[styles.badge, { backgroundColor: `${color}20`, borderColor: color }]}>
          <Text style={[styles.badgeText, { color }]}>{label.severity}</Text>
        </View>
        <Text style={styles.supported}>
          {label.supported ? "Supported" : "Unsupported"}
        </Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  row: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingVertical: 6,
    borderBottomWidth: 1,
    borderBottomColor: Colors.border,
  },
  labelContainer: {
    flex: 1,
  },
  labelName: {
    ...typography.body,
    fontWeight: "600" as const,
  },
  labelScore: {
    ...typography.bodyMuted,
  },
  badgeContainer: {
    alignItems: "flex-end",
  },
  badge: {
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
    borderWidth: 1,
    marginBottom: 2,
  },
  badgeText: {
    fontSize: 9,
    fontWeight: "700" as const,
    textTransform: "uppercase" as const,
  },
  supported: {
    fontSize: 10,
    color: Colors.textMuted,
  },
});
