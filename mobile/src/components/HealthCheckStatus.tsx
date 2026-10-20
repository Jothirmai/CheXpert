import React from "react";
import { View, Text, ActivityIndicator, StyleSheet } from "react-native";

import { Colors, spacing, typography } from "../theme";

interface HealthCheckStatusProps {
  ok: boolean;
  message: string;
}

export function HealthCheckStatus({ ok, message }: HealthCheckStatusProps) {
  return (
    <View style={[styles.container, ok && styles.ok, !ok && styles.error]}>
      {ok ? (
        <View style={styles.dot} />
      ) : (
        <ActivityIndicator size="small" color={Colors.danger} />
      )}
      <Text style={[styles.text, ok ? styles.textOk : styles.textError]}>{message}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    paddingVertical: 4,
  },
  dot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: Colors.success,
  },
  ok: { backgroundColor: "rgba(22, 101, 52, 0.1)" },
  error: { backgroundColor: "rgba(153, 27, 27, 0.1)" },
  text: { ...typography.caption },
  textOk: { color: Colors.success },
  textError: { color: Colors.danger },
});
