"""
Brake wear feature engineering pipeline.

Produces features that drive the brake-pad replacement prediction model.
All window computations are relative to the latest timestamp in the input df.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from features.base_pipeline import FeaturePipeline
from features.derived_utils import (
    detect_harsh_brake,
    compute_brake_stress_index,
    detect_regen_event,
)


class BrakeFeaturePipeline(FeaturePipeline):

    def compute(
        self,
        vin: str,
        df: pd.DataFrame,
        label_df: pd.DataFrame | None = None,
        *,
        last_service_odo: float | None = None,
        **_ctx: Any,
    ) -> pd.DataFrame:
        df = self._normalize(df)
        if df.empty or "timestamp" not in df.columns:
            return pd.DataFrame()

        t_max = df["timestamp"].max()
        odo   = df["odometer"] if "odometer" in df.columns else pd.Series(dtype=float)
        speed = df["speed"]     if "speed"    in df.columns else pd.Series(0.0, index=df.index)
        bpos  = df["brake_pos"] if "brake_pos" in df.columns else pd.Series(0.0, index=df.index)
        apos  = df["accel_pos"] if "accel_pos" in df.columns else pd.Series(0.0, index=df.index)

        # ── Apply derived utility functions ──────────────────────────────
        df = detect_harsh_brake(df)
        df = compute_brake_stress_index(df)
        df = detect_regen_event(df)

        # ── 7d and 30d windows ────────────────────────────────────────────
        d7  = self._last_days(df, 7)
        d30 = self._last_days(df, 30)

        bpos30  = d30["brake_pos"].fillna(0) if "brake_pos" in d30.columns else pd.Series(dtype=float)
        speed30 = d30["speed"].fillna(0)     if "speed"     in d30.columns else pd.Series(dtype=float)
        apos30  = d30["accel_pos"].fillna(0) if "accel_pos" in d30.columns else pd.Series(dtype=float)
        bpos7   = d7["brake_pos"].fillna(0)  if "brake_pos" in d7.columns  else pd.Series(dtype=float)
        odo30   = d30["odometer"]            if "odometer"  in d30.columns else pd.Series(dtype=float)

        # ── Cumulative brake stress index (via derived BSI column) ────────
        brake_stress_cumulative = float(df["bsi"].sum()) if "bsi" in df.columns else float(
            (bpos.fillna(0) * speed.fillna(0) / 1000).sum()
        )

        # ── Harsh brake rates from derived column ─────────────────────────
        harsh_7d_mask  = d7["is_harsh_brake"].to_numpy()  if "is_harsh_brake" in d7.columns  else np.array([False] * len(d7))
        harsh_30d_mask = d30["is_harsh_brake"].to_numpy() if "is_harsh_brake" in d30.columns else np.array([False] * len(d30))
        odo7 = d7["odometer"] if "odometer" in d7.columns else pd.Series(dtype=float)
        harsh_brake_rate_7d  = self._rate_per_100km(harsh_7d_mask,  odo7)
        harsh_brake_rate_30d = self._rate_per_100km(harsh_30d_mask, odo30)

        # ── Contextual adjustments ────────────────────────────────────────
        rain_intensity = 0
        road_type = "mixed"
        elevation_stress = 0.0
        try:
            from features.contextual_features import ContextualFeatureEngine
            cfe = ContextualFeatureEngine()
            rain_intensity = cfe.rain_intensity(df)
            elevation_stress = cfe.elevation_stress(df, {"odometer": float(odo.iloc[-1] - odo.iloc[0]) if len(odo) > 1 else 1})
            road_type = cfe.road_type({"averageSpeed": float(speed.mean())}, df)
        except Exception:
            pass
        rain_multiplier = 1.0 + 0.3 * (rain_intensity / 3)
        harsh_brake_rate_7d  = round(harsh_brake_rate_7d * rain_multiplier, 4)
        harsh_brake_rate_30d = round(harsh_brake_rate_30d * rain_multiplier, 4)

        # ── High-speed stops ───────────────────────────────────────────────
        high_speed_stop_mask = ((speed30 > 80) & (bpos30 > 40)).to_numpy() if len(speed30) > 0 else np.array([])
        high_speed_stop_count_30d = int(high_speed_stop_mask.sum()) if len(high_speed_stop_mask) > 0 else 0

        # ── Average brake intensity (7d) ───────────────────────────────────
        braking_7d = bpos7[bpos7 > 20]
        avg_brake_intensity_7d = float(braking_7d.mean()) if len(braking_7d) > 0 else 0.0

        # ── Regen fraction from derived column ─────────────────────────────
        regen_mask = df["is_regen_event"] if "is_regen_event" in df.columns else pd.Series(False, index=df.index)
        regen_fraction = float(regen_mask.sum() / max(len(df), 1))

        # ── Additional B2 features ─────────────────────────────────────────
        effective_brake_km       = float(odo30.max() - odo30.min()) if len(odo30) > 1 else 0.0
        abs_activation_rate_30d  = 0.0   # ABS data unavailable
        esc_activation_rate_30d  = 0.0   # ESC data unavailable
        brake_pedal_travel_proxy = float(bpos30[bpos30 > 10].mean()) if (bpos30 > 10).any() else 0.0
        brake_thermal_stress     = float((bpos30.fillna(0) * speed30.fillna(0) ** 2).sum() / 1_000_000)
        wot_event_count_30d      = int((apos30 > 225).sum())   # raw > 225 = ~90% throttle
        accel_smoothness_score   = float(np.clip(100 - apos30.diff().abs().mean() * 2, 0, 100)) if len(apos30) > 1 else 50.0

        # ── Service-based features ─────────────────────────────────────────
        km_since_last_brake_service = float(odo.iloc[-1] - last_service_odo) if (
            last_service_odo is not None and len(odo) > 0
        ) else np.nan
        days_since_last_brake_service = 0.0   # injected by refresh job from service history

        # ── 95th-pctile decel / brake heat ────────────────────────────────
        ax30 = d30["accel_x"].fillna(0) if "accel_x" in d30.columns else pd.Series(dtype=float)
        if len(ax30) > 0 and len(bpos30) > 0:
            braking_ax = ax30[bpos30 > 20].abs()
            decel_g_95th_30d = float(np.nanpercentile(braking_ax, 95)) if len(braking_ax) > 0 else 0.0
        else:
            decel_g_95th_30d = 0.0
        brake_heat_proxy = float((bpos.fillna(0) * speed.fillna(0) ** 2).sum() / 1_000_000)

        # ── Current state ──────────────────────────────────────────────────
        brake_front_mm             = float(df["brake_front_mm"].iloc[-1]) if "brake_front_mm" in df.columns else np.nan
        brake_rear_mm              = float(df["brake_rear_mm"].iloc[-1])  if "brake_rear_mm"  in df.columns else np.nan
        brake_fluid_pct            = float(df["brake_fluid_pct"].iloc[-1]) if "brake_fluid_pct" in df.columns else np.nan
        brake_fluid_warning_active = int(brake_fluid_pct < 80) if not np.isnan(brake_fluid_pct) else 0

        # ── Labels ────────────────────────────────────────────────────────
        days_to_failure, within_30 = self._label_days_to_failure(
            vin, label_df, "brake_degradation", t_max
        )

        return self._row(vin, t_max, {
            "brake_stress_cumulative":        brake_stress_cumulative,
            "harsh_brake_rate_7d":            harsh_brake_rate_7d,
            "harsh_brake_rate_30d":           harsh_brake_rate_30d,
            "high_speed_stop_count_30d":      high_speed_stop_count_30d,
            "avg_brake_intensity_7d":         avg_brake_intensity_7d,
            "regen_fraction":                 regen_fraction,
            "effective_brake_km":             effective_brake_km,
            "abs_activation_rate_30d":        abs_activation_rate_30d,
            "esc_activation_rate_30d":        esc_activation_rate_30d,
            "brake_pedal_travel_proxy":       brake_pedal_travel_proxy,
            "km_since_last_brake_service":    km_since_last_brake_service,
            "days_since_last_brake_service":  days_since_last_brake_service,
            "brake_thermal_stress":           brake_thermal_stress,
            "wot_event_count_30d":            wot_event_count_30d,
            "accel_smoothness_score":         accel_smoothness_score,
            "deceleration_g_95th_30d":        decel_g_95th_30d,
            "brake_heat_proxy":               brake_heat_proxy,
            "brake_fluid_warning_active":     brake_fluid_warning_active,
            "brake_front_mm":                 brake_front_mm,
            "brake_rear_mm":                  brake_rear_mm,
            "brake_fluid_pct":                brake_fluid_pct,
            # Contextual
            "road_type":                          road_type,
            "rain_intensity":                     rain_intensity,
            "elevation_stress":                   elevation_stress,
            # Targets
            "days_to_brake_replacement":          days_to_failure,
            "brake_replacement_within_30_days":   within_30,
        })
