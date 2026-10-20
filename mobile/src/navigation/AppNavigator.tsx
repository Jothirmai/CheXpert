import { createNativeStackNavigator } from "@react-navigation/stack";
import React from "react";
import { View, Text } from "react-native";

import { HomeScreen } from "../screens/HomeScreen";
import { AnalysisScreen } from "../screens/AnalysisScreen";
import { ReportScreen } from "../screens/ReportScreen";
import { MethodologyScreen } from "../screens/MethodologyScreen";
import { HelpScreen } from "../screens/HelpScreen";

const Stack = createNativeStackNavigator();

export type RootStackParamList = {
  Home: undefined;
  Analysis: { analysisId?: string };
  Report: { analysisId: string };
  Methodology: undefined;
  Help: undefined;
};

export function AppNavigator() {
  return (
    <Stack.Navigator screenOptions={{ headerShown: false }}>
      <Stack.Screen name="Home" component={HomeScreen} />
      <Stack.Screen name="Analysis" component={AnalysisScreen} />
      <Stack.Screen name="Report" component={ReportScreen} />
      <Stack.Screen name="Methodology" component={MethodologyScreen} />
      <Stack.Screen name="Help" component={HelpScreen} />
    </Stack.Navigator>
  );
}
