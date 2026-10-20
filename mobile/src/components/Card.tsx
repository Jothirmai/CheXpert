import React from "react";
import { View, StyleSheet } from "react-native";

import { Colors, spacing, borderRadius } from "../theme";

interface CardProps {
  children: React.ReactNode;
  style?: any;
}

export function Card({ children, style }: CardProps) {
  return (
    <View
      style={[
        styles.card,
        { padding: spacing.md, borderRadius: borderRadius.md },
        style,
      ]}
    >
      {children}
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: Colors.surface,
    borderWidth: 1,
    borderColor: Colors.border,
  },
});
