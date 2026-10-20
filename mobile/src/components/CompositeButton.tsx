import React from "react";
import { View, Text, TouchableOpacity, StyleSheet, ActivityIndicator } from "react-native";

import { Colors, spacing, typography } from "../theme";

type ButtonType = "primary" | "secondary" | "outline" | "text";

interface CompositeButtonProps {
  title: string;
  onPress: () => void;
  type?: ButtonType;
  icon?: string;
  disabled?: boolean;
  flex?: number;
  marginTop?: number;
}

const typeStyles: Record<ButtonType, any> = {
  primary: { backgroundColor: Colors.primary },
  secondary: { backgroundColor: Colors.secondary },
  outline: { backgroundColor: "transparent", borderWidth: 1, borderColor: Colors.primary },
  text: { backgroundColor: "transparent" },
};

const typeTextColors: Record<ButtonType, string> = {
  primary: Colors.white,
  secondary: Colors.white,
  outline: Colors.primary,
  text: Colors.secondary,
};

export function CompositeButton({
  title,
  onPress,
  type = "primary",
  icon,
  disabled = false,
  flex,
  marginTop,
}: CompositeButtonProps) {
  return (
    <TouchableOpacity
      style={[
        styles.button,
        typeStyles[type],
        disabled && { opacity: 0.5, borderColor: Colors.textMuted },
        flex ? { flex } : {},
        marginTop ? { marginTop } : {},
      ]}
      onPress={onPress}
      disabled={disabled}
    >
      {disabled ? (
        <ActivityIndicator size="small" color={typeTextColors[type]} />
      ) : (
        <>
          {icon && <Text style={{ marginRight: 4 }}>{icon}</Text>}
          <Text style={{ ...styles.title, color: typeTextColors[type] }}>{title}</Text>
        </>
      )}
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  button: {
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderRadius: 8,
    alignItems: "center",
    flexDirection: "row",
    justifyContent: "center",
    minHeight: 48,
  },
  title: {
    ...typography.body,
    fontWeight: "600" as const,
  },
});
