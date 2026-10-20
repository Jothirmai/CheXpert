export const Colors = {
  primary: "#0f172a",
  primaryDark: "#020617",
  secondary: "#3b82f6",
  success: "#166534",
  warning: "#92400e",
  danger: "#991b1b",
  background: "#ffffff",
  surface: "#f8fafc",
  border: "#e2e8f0",
  text: "#1e293b",
  textMuted: "#64748b",
  white: "#ffffff",
  overlay: "rgba(15, 23, 42, 0.85)",
};

export const spacing = {
  xs: 4,
  sm: 8,
  md: 16,
  lg: 24,
  xl: 32,
};

export const borderRadius = {
  sm: 6,
  md: 10,
  lg: 16,
};

export const typography = {
  h1: { fontSize: 24, fontWeight: "700" as const, color: Colors.text },
  h2: { fontSize: 20, fontWeight: "600" as const, color: Colors.text },
  h3: { fontSize: 16, fontWeight: "600" as const, color: Colors.text },
  body: { fontSize: 14, color: Colors.text },
  bodyMuted: { fontSize: 13, color: Colors.textMuted },
  caption: { fontSize: 11, color: Colors.textMuted },
};
