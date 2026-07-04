"""
Engine / oil degradation feature engineering pipeline.

Computes all features used by the oil-change prediction model.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from features.base_pipeline import FeaturePipeline
from features.derived_utils import compute_oil_degradation_index


class EngineFeaturePipeline(FeaturePipeline):

    def compute(
        self,
        vin: str,
        df: pd.DataFrame,
        label_df: pd.DataFrame | None = None,
        *,
        last_oil_change_odo: float | None = None,
        last_oil_change_date: str | None = None,
        **_ctx: Any,
    ) -> pd.DataFrame:
        df = self._normalize(df)
        if df.empty or "timestamp" not in df.columns:
            return pd.DataFrame()

        t_max = df["timestamp"].max()

        speed       = df["speed"].fillna(0)      if "speed"       in df.columns else pd.Series(0.0, index=df.index)
        rpm         = df["rpm"].fillna(0)         if "rpm"         in df.columns else pd.Series(0.0, index=df.index)
        sys_pwr     = df["sys_pwr_mod"].fillna(0) if "sys_pwr_mod" in df.columns else pd.Series(0, index=df.index)
        coolant     = df["coolant_temp"].fillna(np.nan) if "coolant_temp" in df.columns else pd.Series(dtype=float)
        oil_life    = df["oil_life_pct"].fillna(np.nan) if "oil_life_pct" in df.columns else pd.Series(dtype=float)
        fuel_con    = df["fuel_consumed"].fillna(0) if "fuel_consumed" in df.columns else pd.Series(0.0, index=df.index)
        odo         = df["odometer"]             if "odometer"    in df.columns else pd.Series(dtype=float)

        d7   = self._last_days(df, 7)
        d30  = self._last_days(df, 30)
        d90  = self._last_days(df, 90)

        # ── km / days since last oil change ──────────────────────────────
        km_since_oil_change = float(odo.iloc[-1] - last_oil_change_odo) if (
            last_oil_change_odo is not None and len(odo) > 0
        ) else 0.0

        days_since_oil_change = 0.0
        if last_oil_change_date:
            try:
                lcd = pd.to_datetime(last_oil_change_date, utc=True)
                days_since_oil_change = float((t_max - lcd).total_seconds() / 86400)
            except Exception:
                pass

        # ── Cold starts (30d) ─────────────────────────────────────────────
        # A cold start = session start (sys_pwr_mod transitions to 2/3) when
        # coolant was < 40°C at that moment.
        cold_start_count_30d = 0
        if "coolant_temp" in d30.columns and "sys_pwr_mod" in d30.columns:
            pwr30 = d30["sys_pwr_mod"].fillna(0)
            cool30 = d30["coolant_temp"].fillna(100)
            session_starts = (pwr30.diff() > 1).fillna(False)
            cold_start_count_30d = int(((cool30 < 40) & session_starts).sum())

        # ── Contextual adjustments ────────────────────────────────────────
        elevation_stress = 0.0
        load_condition_proxy = 0.0
        thermal_zone = "moderate"
        try:
            from features.contextual_features import ContextualFeatureEngine
            cfe = ContextualFeatureEngine()
            thermal_zone = cfe.thermal_zone(df)
            elevation_stress = cfe.elevation_stress(df, {"odometer": float(odo.iloc[-1] - odo.iloc[0]) if len(odo) > 1 else 1})
            load_condition_proxy = cfe.load_condition_proxy(df)
        except Exception:
            pass
        cold_start_weight = 1.5 if thermal_zone == "cold" else 1.0
        cold_start_count_30d = int(cold_start_count_30d * cold_start_weight)

        # ── High-RPM duration (30d, minutes) ──────────────────────────────
        high_rpm_30d = rpm[rpm.index.isin(d30.index)] if len(d30) > 0 else pd.Series(dtype=float)
        high_rpm_duration_minutes_30d = float((high_rpm_30d > 4000).sum() / 60)

        # ── Coolant overtemp events (30d) ─────────────────────────────────
        cool30 = d30["coolant_temp"].fillna(0) if "coolant_temp" in d30.columns else pd.Series(dtype=float)
        coolant_overtemp_count_30d = int((cool30 > 100).sum())

        # ── Average coolant temp during driving sessions (7d) ─────────────
        if "coolant_temp" in d7.columns and "sys_pwr_mod" in d7.columns:
            driving_mask = d7["sys_pwr_mod"].isin([2, 3])
            avg_coolant_temp_7d = float(d7.loc[driving_mask, "coolant_temp"].mean()) if driving_mask.any() else 0.0
        elif "coolant_temp" in d7.columns:
            avg_coolant_temp_7d = float(d7["coolant_temp"].mean()) if len(d7) > 0 else 0.0
        else:
            avg_coolant_temp_7d = 0.0

        # ── Fuel consumption deviation: 7d vs 90d baseline ────────────────
        def _mean_fuel_rate(seg: pd.DataFrame) -> float:
            if "fuel_consumed" not in seg.columns or "odometer" not in seg.columns:
                return np.nan
            km  = float(seg["odometer"].max() - seg["odometer"].min())
            fv  = float(seg["fuel_consumed"].max() - seg["fuel_consumed"].min())
            return fv / km * 100 if km > 1 else np.nan

        fuel_7d  = _mean_fuel_rate(d7)
        fuel_90d = _mean_fuel_rate(d90)
        fuel_consumption_deviation = float((fuel_7d - fuel_90d) / fuel_90d * 100) if (
            fuel_90d and not np.isnan(fuel_90d) and fuel_90d > 0 and not np.isnan(fuel_7d)
        ) else 0.0

        # ── Idle hours (30d): speed=0, rpm>0, sys_pwr_mod = running ───────
        if "speed" in d30.columns and "rpm" in d30.columns:
            idle_mask = (d30["speed"].fillna(0) < 2) & (d30["rpm"].fillna(0) > 600)
            idle_hours_30d = float(idle_mask.sum() / 3600)
        else:
            idle_hours_30d = 0.0

        # ── Short trip fraction (30d): trips < 8 km ───────────────────────
        short_trip_fraction_30d = 0.0
        if "sys_pwr_mod" in d30.columns and "odometer" in d30.columns:
            pwr30 = d30["sys_pwr_mod"].fillna(0)
            odo30 = d30["odometer"].fillna(method=None)
            session_end_mask = (pwr30.diff(-1) < 0).fillna(False)
            session_start_mask = (pwr30.diff() > 1).fillna(False)
            starts = odo30[session_start_mask].values
            ends   = odo30[session_end_mask].values
            n_pairs = min(len(starts), len(ends))
            if n_pairs > 0:
                trip_kms = ends[:n_pairs] - starts[:n_pairs]
                short_trip_fraction_30d = float((trip_kms < 8).sum() / n_pairs)

        # ── Oil degradation index — via derived_utils ─────────────────────
        _empty_svc = pd.DataFrame(columns=["DescriptionOne", "Mileage"])
        oil_degradation_index = compute_oil_degradation_index(df, _empty_svc) / 100.0

        # ── Current state ─────────────────────────────────────────────────
        oil_life_pct_now    = float(oil_life.iloc[-1]) if len(oil_life) > 0 else np.nan
        oil_pressure_warn   = 0   # field not in synthetic data
        mil_warning         = 0

        # ── Labels ────────────────────────────────────────────────────────
        days_to_failure, within_14 = self._label_days_to_failure(
            vin, label_df, "oil_degradation", t_max
        )
        oil_change_due_within_14 = int(
            (not np.isnan(oil_life_pct_now) and oil_life_pct_now < 15) or within_14
        )

        # ── Simple aggregate features expected by tests ────────────────────
        rpm7       = d7["rpm"].fillna(0) if "rpm" in d7.columns else pd.Series(dtype=float)
        coolant7   = d7["coolant_temp"].fillna(0) if "coolant_temp" in d7.columns else pd.Series(dtype=float)
        rpm_mean_7d          = float(rpm7.mean())    if len(rpm7) > 0    else 0.0
        coolant_temp_max_7d  = float(coolant7.max()) if len(coolant7) > 0 else 0.0

        # ── Additional B2 features ─────────────────────────────────────────
        high_rpm_stress_index = float(min(1.0, high_rpm_duration_minutes_30d / 120))
        rpm30 = d30["rpm"].fillna(0)   if "rpm"   in d30.columns else pd.Series(dtype=float)
        spd30 = d30["speed"].fillna(0) if "speed" in d30.columns else pd.Series(dtype=float)
        rpm_to_speed_ratio_anomaly = float(
            ((rpm30 / (spd30 + 1)).replace([np.inf, -np.inf], np.nan) - 30).abs().mean()
        ) if len(rpm30) > 0 else 0.0
        fuel_consumption_deviation_pct = fuel_consumption_deviation
        apos30 = d30["accel_pos"].fillna(0) if "accel_pos" in d30.columns else pd.Series(dtype=float)
        engine_load_proxy = float(apos30.mean() / 255 * 100) if len(apos30) > 0 else 0.0
        towing_load_indicator = int(
            len(rpm30) > 0 and (spd30 > 40).any() and (rpm30 > 3500).any() and
            float(rpm30[(spd30 > 40) & (rpm30 > 3500)].mean()) > 3500
        )
        mil_recurrence_flag = int(mil_warning)
        gear30 = d30["gear_pos"].fillna(0) if "gear_pos" in d30.columns else pd.Series(dtype=float)
        overrev  = int((rpm30 > 5000).sum())
        lugging  = int(((rpm30 < 1000) & (spd30 > 30) & (gear30 > 2)).sum())
        gear_efficiency_score = float(np.clip(100 - overrev / max(len(rpm30), 1) * 1000
                                              - lugging / max(len(rpm30), 1) * 800, 0, 100))

        return self._row(vin, t_max, {
            "oil_degradation_index":           oil_degradation_index,
            "km_since_oil_change":             km_since_oil_change,
            "days_since_oil_change":           days_since_oil_change,
            "cold_start_count_30d":            cold_start_count_30d,
            "short_trip_fraction_30d":         short_trip_fraction_30d,
            "coolant_overtemp_count_30d":      coolant_overtemp_count_30d,
            "avg_coolant_temp_7d":             avg_coolant_temp_7d,
            "high_rpm_stress_index":           high_rpm_stress_index,
            "rpm_to_speed_ratio_anomaly":      rpm_to_speed_ratio_anomaly,
            "fuel_consumption_deviation_pct":  fuel_consumption_deviation_pct,
            "idle_hours_30d":                  idle_hours_30d,
            "engine_load_proxy":               engine_load_proxy,
            "towing_load_indicator":           towing_load_indicator,
            "mil_recurrence_flag":             mil_recurrence_flag,
            "gear_efficiency_score":           gear_efficiency_score,
            "high_rpm_duration_minutes_30d":   high_rpm_duration_minutes_30d,
            "rpm_mean_7d":                     rpm_mean_7d,
            "coolant_temp_max_7d":             coolant_temp_max_7d,
            "oil_pressure_warning_active":     oil_pressure_warn,
            "mil_warning_active":              mil_warning,
            "oil_life_pct":                    oil_life_pct_now,
            # Contextual
            "elevation_stress":                elevation_stress,
            "load_condition_proxy":            load_condition_proxy,
            # Targets
            "days_to_engine_oil_change":       days_to_failure,
            "engine_oil_within_30_days":       oil_change_due_within_14,
        })
