"""
Tyre pressure and wear feature engineering pipeline.

Computes per-corner pressure trends, axle imbalance, temperature-corrected
values, lateral-G stress, and service-history-based wear estimates.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from features.base_pipeline import FeaturePipeline
from features.derived_utils import compute_tyre_stress_per_trip, correct_tyre_pressure_for_temp

_POSITIONS = ["fl", "fr", "rl", "rr"]
_TYRE_COLS = {p: f"tyre_{p}" for p in _POSITIONS}  # internal col names

# Reference temperature for pressure correction (25°C, 0.10 kPa/°C per spec)
_TYRE_TEMP_REF = 25.0
_TYRE_TEMP_K   = 0.10


class TyreFeaturePipeline(FeaturePipeline):

    def compute(
        self,
        vin: str,
        df: pd.DataFrame,
        label_df: pd.DataFrame | None = None,
        *,
        last_tyre_service_odo: float | None = None,
        **_ctx: Any,
    ) -> pd.DataFrame:
        df = self._normalize(df)
        if df.empty or "timestamp" not in df.columns:
            return pd.DataFrame()

        t_max = df["timestamp"].max()
        d7    = self._last_days(df, 7)
        d14   = self._last_days(df, 14)
        d30   = self._last_days(df, 30)

        out_tmp = df["outside_temp"].fillna(25.0) if "outside_temp" in df.columns else pd.Series(25.0, index=df.index)
        odo     = df["odometer"]                  if "odometer"    in df.columns else pd.Series(dtype=float)
        speed30 = d30["speed"].fillna(0)          if "speed"       in d30.columns else pd.Series(dtype=float)
        bpos30  = d30["brake_pos"].fillna(0)      if "brake_pos"   in d30.columns else pd.Series(dtype=float)
        ax30    = d30["accel_x"].fillna(0)        if "accel_x"       in d30.columns else pd.Series(dtype=float)
        ay30    = d30["accel_y"].fillna(0)        if "accel_y"       in d30.columns else pd.Series(dtype=float)
        steer30 = d30["steering_angle"].fillna(0) if "steering_angle" in d30.columns else pd.Series(dtype=float)

        features: dict[str, Any] = {}

        for pos in _POSITIONS:
            col = f"tyre_{pos}"
            if col not in df.columns:
                for suffix in ["_7d_avg", "_trend_14d", "_temp_corrected"]:
                    features[f"pressure_{pos}{suffix}"] = np.nan
                continue

            t_all = df[col].fillna(np.nan)
            t7    = d7[col].fillna(np.nan)    if col in d7.columns  else pd.Series(dtype=float)
            t14   = d14[col].fillna(np.nan)   if col in d14.columns else pd.Series(dtype=float)
            tmp14 = d14["outside_temp"].fillna(25.0) if "outside_temp" in d14.columns else pd.Series(25.0, index=d14.index)

            features[f"pressure_{pos}_7d_avg"]    = float(t7.mean())       if len(t7) > 0  else np.nan
            features[f"pressure_{pos}_trend_14d"] = self._slope(t14)

            # Temperature-corrected pressure via derived_utils (Charles's Law)
            if len(t14) > 0 and len(tmp14) > 0:
                corrected_vals = [
                    correct_tyre_pressure_for_temp(float(p_val), float(t_val))
                    for p_val, t_val in zip(t14.dropna(), tmp14.loc[t14.dropna().index].fillna(25.0))
                ]
                features[f"pressure_{pos}_temp_corrected"] = float(np.mean(corrected_vals)) if corrected_vals else np.nan
            else:
                features[f"pressure_{pos}_temp_corrected"] = np.nan

        # ── Minimum tyre pressure (7d, any corner) — expected by tests ───
        all_7d_pressures = [
            features.get(f"pressure_{p}_7d_avg", np.nan) for p in _POSITIONS
        ]
        tyre_pressure_min_7d = float(np.nanmin(all_7d_pressures)) if any(
            not np.isnan(v) for v in all_7d_pressures
        ) else 0.0

        # ── Axle imbalance ────────────────────────────────────────────────
        fl7 = d7["tyre_fl"].fillna(np.nan) if "tyre_fl" in d7.columns else pd.Series(dtype=float)
        fr7 = d7["tyre_fr"].fillna(np.nan) if "tyre_fr" in d7.columns else pd.Series(dtype=float)
        rl7 = d7["tyre_rl"].fillna(np.nan) if "tyre_rl" in d7.columns else pd.Series(dtype=float)
        rr7 = d7["tyre_rr"].fillna(np.nan) if "tyre_rr" in d7.columns else pd.Series(dtype=float)

        pressure_axle_imbalance_front = float((fl7 - fr7).abs().mean()) if len(fl7) > 0 and len(fr7) > 0 else np.nan
        pressure_axle_imbalance_rear  = float((rl7 - rr7).abs().mean()) if len(rl7) > 0 and len(rr7) > 0 else np.nan

        # ── Harsh brake events per 100km (30d) ────────────────────────────
        harsh_brake_mask = (bpos30 > 70) & (ax30 < -75)
        odo30            = d30["odometer"] if "odometer" in d30.columns else pd.Series(dtype=float)
        harsh_brake_per_100km_30d = self._rate_per_100km(harsh_brake_mask.to_numpy(), odo30)

        # ── Lateral G 95th percentile (30d) ────────────────────────────────
        # Use real tboxAccelY if available; fall back to steering angle rate proxy
        if ay30.abs().max() > 0.01:
            lateral_g_95th_30d = float(np.nanpercentile(ay30.abs(), 95)) if len(ay30) > 0 else np.nan
            sudden_turn_mask   = (ay30.abs() > 0.3).to_numpy()  # > 0.3g = sudden turn
        else:
            steer_rate         = steer30.diff().abs()
            lateral_g_95th_30d = float(np.nanpercentile(steer_rate, 95)) if len(steer_rate) > 0 else np.nan
            sudden_turn_mask   = (steer_rate > 30).to_numpy()

        # ── Sudden turns per 100km (30d) ───────────────────────────────────
        sudden_turn_per_100km_30d = self._rate_per_100km(sudden_turn_mask, odo30)

        # ── ESC activations per 100km (30d) ───────────────────────────────
        esc_activation_rate_30d = 0.0  # ESC status not in synthetic data

        # ── km since last tyre service ─────────────────────────────────────
        km_since_last_tyre_service = float(odo.iloc[-1] - last_tyre_service_odo) if (
            last_tyre_service_odo is not None and len(odo) > 0
        ) else (float(odo.iloc[-1]) % 40_000.0 if len(odo) > 0 else 0.0)
        # When service history unknown, modulo 40k simulates position in current tyre lifecycle

        # ── Average speed 30d ─────────────────────────────────────────────
        driving_mask = speed30 > 5
        avg_speed_30d = float(speed30[driving_mask].mean()) if driving_mask.any() else 0.0

        # ── Current tyre pressures ────────────────────────────────────────
        for pos in _POSITIONS:
            col = f"tyre_{pos}"
            features[f"{pos}_current"] = float(df[col].iloc[-1]) if col in df.columns else np.nan

        # ── TPMS status from real wheelTyreMonitorStatus signal ──────────
        # 0=OK, 1=Deflation Warning, 4=System Fault, 5=Sensor N/A
        if "tpms_status" in df.columns:
            tpms_mode = int(df["tpms_status"].fillna(0).mode().iloc[0])
            tpms_deflation_count = int((df["tpms_status"].fillna(0) == 1).sum())
        else:
            tpms_mode            = 0
            tpms_deflation_count = 0
        features["tpms_status"]            = tpms_mode
        features["tpms_deflation_count"]   = tpms_deflation_count

        # ── Blast risk score: low pressure + high speed ────────────────────
        # Risk = speed × max(0, target_kpa - current_kpa) / 1000
        target_kpa  = 230.0
        min_p_7d    = float(np.nanmin([features.get(f"pressure_{p}_7d_avg", np.nan) for p in _POSITIONS]) or target_kpa)
        blast_risk  = float(speed30.fillna(0).mean() * max(0, target_kpa - min_p_7d) / 1000)
        features["tyre_blast_risk_score"] = round(blast_risk, 4)

        # ── Labels — physics-based self-supervision + service event ─────────
        # Physics: warn when any corner drops below 195 kPa (normal ~226-232 kPa).
        # Extrapolate using worst-corner 14d trend to predict days to threshold.
        _PRESSURE_WARN_KPA = 195.0
        worst_drop_rate = min(
            float(features.get(f"pressure_{p}_trend_14d", 0.0) or 0.0)
            for p in _POSITIONS
        )
        if worst_drop_rate < -0.01:
            days_to_physics = float(np.clip(
                (min_p_7d - _PRESSURE_WARN_KPA) / abs(worst_drop_rate), 0, 365
            ))
        else:
            days_to_physics = 365.0
        # Also flag by wear km: typical tyre life = 40,000 km; warn at 35,000
        _TYRE_WEAR_WARN_KM = 35_000.0
        within_physics = int(
            min_p_7d < _PRESSURE_WARN_KPA or
            tpms_deflation_count > 2 or
            features.get("tyre_blast_risk_score", 0.0) > 0.5 or
            km_since_last_tyre_service >= _TYRE_WEAR_WARN_KM
        )

        days_to_failure, within_svc = self._label_days_to_failure(
            vin, label_df, "tyre_puncture", t_max
        )
        if np.isnan(days_to_failure):
            days_to_failure = days_to_physics
        within_30 = int(within_svc or within_physics)

        km_to_replacement = float(
            (last_tyre_service_odo + 40_000 - odo.iloc[-1]) if (
                last_tyre_service_odo is not None and len(odo) > 0
            ) else 0.0
        )

        # ── Tyre stress via derived_utils ──────────────────────────────────
        fl_avg = features.get("pressure_fl_7d_avg", np.nan)
        mean_pressure_kpa = np.nanmean([
            features.get("pressure_fl_7d_avg", np.nan),
            features.get("pressure_fr_7d_avg", np.nan),
            features.get("pressure_rl_7d_avg", np.nan),
            features.get("pressure_rr_7d_avg", np.nan),
        ])
        trip_summary = {
            "harshBreakingNum": int(harsh_brake_mask.sum()),
            "suddenTurnNum":    int(sudden_turn_mask.sum()),
            "maxSpeed":         float(speed30.max()) if len(speed30) > 0 else 0.0,
            "odometer":         float(odo30.max() - odo30.min()) if len(odo30) > 1 else 0.0,
        }
        tyre_stress_cumulative = compute_tyre_stress_per_trip(
            trip_summary,
            mean_pressure_kpa if not np.isnan(mean_pressure_kpa) else 230.0,
        )

        # ── Pressure drop rates (per-corner: 7d avg vs 14d trend) ─────────
        for pos in _POSITIONS:
            features[f"pressure_{pos}_7d_avg"]  = features.get(f"pressure_{pos}_7d_avg", np.nan)
            features[f"pressure_drop_rate_{pos}"] = features.get(f"pressure_{pos}_trend_14d", np.nan)

        features.update({
            "axle_imbalance_front":          pressure_axle_imbalance_front,
            "axle_imbalance_rear":           pressure_axle_imbalance_rear,
            "pressure_axle_imbalance_front": pressure_axle_imbalance_front,
            "pressure_axle_imbalance_rear":  pressure_axle_imbalance_rear,
            "tpms_status":                   features.get("tpms_status", 0),
            "tyre_stress_cumulative":        tyre_stress_cumulative,
            "km_since_last_tyre_service":    km_since_last_tyre_service,
            "tyre_pressure_min_7d":          tyre_pressure_min_7d,
            "effective_tyre_km":             float(odo.iloc[-1]) if len(odo) > 0 else np.nan,
            "uneven_wear_indicator":         int(
                pressure_axle_imbalance_front > 10 or pressure_axle_imbalance_rear > 10
            ),
            "harsh_brake_per_100km_30d":     harsh_brake_per_100km_30d,
            "lateral_g_95th_30d":            lateral_g_95th_30d,
            "sudden_turn_per_100km_30d":     sudden_turn_per_100km_30d,
            "esc_activation_rate_30d":       esc_activation_rate_30d,
            "avg_speed_30d":                 avg_speed_30d,
            # Targets
            "days_to_tyre_replacement":            days_to_failure,
            "tyre_within_30_days":                 within_30,
            "km_to_replacement":                   km_to_replacement,
        })

        return self._row(vin, t_max, features)
