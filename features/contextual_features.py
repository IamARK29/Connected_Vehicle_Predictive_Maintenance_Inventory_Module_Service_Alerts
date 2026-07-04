"""
Contextual Feature Engine — derives road type, weather, elevation,
and load context from raw telemetry and trip data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class ContextualFeatureEngine:

    def compute(self, trip_row: dict, session_df: pd.DataFrame, dealer_city: str = "") -> dict:
        return {
            "road_type":            self.road_type(trip_row, session_df),
            "stop_go_ratio":        self.stop_go_ratio(session_df),
            "elevation_stress":     self.elevation_stress(session_df, trip_row),
            "rain_intensity":       self.rain_intensity(session_df),
            "thermal_zone":         self.thermal_zone(session_df),
            "load_condition_proxy": self.load_condition_proxy(session_df),
            "night_fraction":       self.night_driving_fraction(session_df),
        }

    def road_type(self, trip_row: dict, df: pd.DataFrame) -> str:
        avg_speed = trip_row.get("averageSpeed", 0)
        if avg_speed > 70:
            return "highway"
        if self.stop_go_ratio(df) > 0.35:
            return "urban"
        if avg_speed < 30:
            return "urban"
        return "suburban"

    def stop_go_ratio(self, df: pd.DataFrame) -> float:
        """Fraction of driving time with vehSpeed raw < 50 (= 5 kph)."""
        if "vehSysPwrMod" not in df.columns:
            return 0.0
        driving = df[df["vehSysPwrMod"] == 2]
        if len(driving) == 0:
            return 0.0
        if "vehSpeed" not in df.columns:
            return 0.0
        return round(float((driving["vehSpeed"] < 50).sum() / len(driving)), 3)

    def elevation_stress(self, df: pd.DataFrame, trip_row: dict) -> float:
        """Sum of |delta Altitude| / trip_km. Altitude: raw*0.1 = m."""
        if "Altitude" not in df.columns:
            return 0.0
        alt_m = df["Altitude"] * 0.1
        trip_km = max(trip_row.get("odometer", 1), 1)
        return round(float(alt_m.diff().abs().sum() / trip_km), 2)

    def rain_intensity(self, df: pd.DataFrame) -> int:
        """0=dry 1=light 2=moderate 3=heavy from vehRainDetected (CH-14)."""
        if "vehRainDetected" not in df.columns:
            return 0
        return int(df["vehRainDetected"].max())

    def thermal_zone(self, df: pd.DataFrame) -> str:
        """From vehOutsideTemp (already raw=physical for this signal)."""
        if "vehOutsideTemp" not in df.columns:
            return "moderate"
        t = float(df["vehOutsideTemp"].mean())
        if t < 15:
            return "cold"
        if t < 30:
            return "moderate"
        if t < 40:
            return "hot"
        return "extreme"

    def load_condition_proxy(self, df: pd.DataFrame) -> float:
        """Mean accelerator % at constant speed = proxy for vehicle load."""
        if "vehSpeed" not in df.columns or "vehAccelPos" not in df.columns:
            return 0.0
        spd = df["vehSpeed"] * 0.1
        mean_spd = spd.mean()
        if mean_spd < 1:
            return 0.0
        cruise = df[(spd >= mean_spd * 0.95) & (spd <= mean_spd * 1.05)]
        if len(cruise) < 10:
            return 0.0
        return round(float((cruise["vehAccelPos"] * 0.4).mean()), 2)

    def night_driving_fraction(self, df: pd.DataFrame) -> float:
        if "vehSysPwrMod" not in df.columns:
            return 0.0
        driving = df[df["vehSysPwrMod"] == 2]
        if len(driving) == 0:
            return 0.0
        if "vehNightDetected" not in driving.columns:
            return 0.0
        return round(float((driving["vehNightDetected"] == 1).sum() / len(driving)), 3)
