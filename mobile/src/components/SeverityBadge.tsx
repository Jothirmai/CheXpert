import React from "react";
import { View, Text, StyleSheet } from "react-native";

import { Colors, spacing, typography, SEVERITY_COLORS } from "../theme";

interface SeverityBadgeProps {
  severity: string;
}

export function SeverityBadge({ severity }: SeverityBadgeProps) {
  const color = SEVERITY_COLORS[severity] || Colors.textMuted;
  return (
    <View style={[styles.badge, { backgroundColor: `${color}20`, borderColor: color }]}>
      <Text style={[styles.badgeText, { color }]}>{severity.charAt(0).toUpperCase() + severity.slice(1)}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  badge: {
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
    borderWidth: 1,
    alignSelf: "flex-start",
  },
  badgeText: {
    fontSize: 9,
    fontWeight: "700" as const,
    textTransform: "uppercase" as const,
  },
});
