"""
Driver behaviour scoring feature engineering pipeline.

Computes 7 weighted score components and a composite drive score.
Supports peer-percentile benchmarking in batch mode.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from features.base_pipeline import FeaturePipeline
from features.derived_utils import compute_composite_drive_score

# ── Score component weights (must sum to 1.0) ──────────────────────────────
_WEIGHTS = {
    "smooth_accel":    0.20,
    "smooth_brake":    0.20,
    "gear_efficiency": 0.10,
    "speed":           0.15,
    "fuel":            0.15,
    "cornering":       0.10,
    "idle":            0.10,
}

# Vehicle-class fuel benchmarks (L/100km)
_FUEL_BENCHMARKS = {
    "GLOSTER": 14.0, "HECTOR": 10.0, "ZSEV": 0.0,
    "ASTOR": 8.0, "DEFAULT": 9.5,
}


class DriverBehaviourFeaturePipeline(FeaturePipeline):

    def compute(
        self,
        vin: str,
        df: pd.DataFrame,
        label_df: pd.DataFrame | None = None,
        *,
        model_code: str = "DEFAULT",
        fuel_type: str = "ICE",
        **_ctx: Any,
    ) -> pd.DataFrame:
        df = self._normalize(df)
        if df.empty or "timestamp" not in df.columns:
            return pd.DataFrame()

        t_max = df["timestamp"].max()
        d7    = self._last_days(df, 7)
        d30   = self._last_days(df, 30)

        speed30   = d30["speed"].fillna(0)        if "speed"          in d30.columns else pd.Series(dtype=float)
        bpos30    = d30["brake_pos"].fillna(0)    if "brake_pos"      in d30.columns else pd.Series(dtype=float)
        apos30    = d30["accel_pos"].fillna(0)    if "accel_pos"      in d30.columns else pd.Series(dtype=float)
        rpm30     = d30["rpm"].fillna(0)          if "rpm"            in d30.columns else pd.Series(dtype=float)
        gear30    = d30["gear_pos"].fillna(0)     if "gear_pos"       in d30.columns else pd.Series(dtype=float)
        steer30   = d30["steering_angle"].fillna(0) if "steering_angle" in d30.columns else pd.Series(dtype=float)
        pwr30     = d30["sys_pwr_mod"].fillna(0)  if "sys_pwr_mod"   in d30.columns else pd.Series(0, index=d30.index)
        fcon30    = d30["fuel_consumed"].fillna(0) if "fuel_consumed" in d30.columns else pd.Series(0, index=d30.index)
        odo30     = d30["odometer"]               if "odometer"       in d30.columns else pd.Series(dtype=float)

        n30 = max(len(speed30), 1)
        km30 = float(odo30.max() - odo30.min()) if len(odo30) > 1 else 1.0

        # ── 1. Smooth acceleration score (20%) ─────────────────────────────
        # Harsh accel: accel_pos change > 30 in 1s
        accel_rate = apos30.diff().abs()
        harsh_accel_events = int((accel_rate > 30).sum())
        harsh_accel_rate   = harsh_accel_events / km30 * 100 if km30 > 0 else 0.0
        smooth_accel_score = float(np.clip(100 - harsh_accel_rate * 15, 0, 100))

        # ── 2. Smooth brake score (20%) ────────────────────────────────────
        harsh_brake_mask   = (bpos30 > 70).to_numpy()
        harsh_brake_events = int(harsh_brake_mask.sum())
        harsh_brake_rate   = harsh_brake_events / km30 * 100 if km30 > 0 else 0.0

        # High-speed stops (speed > 80 when hard-braking)
        high_spd_stop_mask = ((speed30 > 80) & (bpos30 > 40)).to_numpy()
        high_speed_stop_rate = int(high_spd_stop_mask.sum()) / km30 * 100 if km30 > 0 else 0.0

        smooth_brake_score = float(np.clip(100 - harsh_brake_rate * 12 - high_speed_stop_rate * 8, 0, 100))

        # ── 3. Gear efficiency score (10%) ─────────────────────────────────
        # Overrev: rpm > 5000 while speed > 20; lugging: rpm < 1000 while speed > 30 gear > 2
        overrev_mask  = ((rpm30 > 5000) & (speed30 > 20)).to_numpy()
        lugging_mask  = ((rpm30 < 1000) & (speed30 > 30) & (gear30 > 2)).to_numpy()
        overrev_rate  = overrev_mask.sum() / n30 * 100
        lugging_rate  = lugging_mask.sum() / n30 * 100
        gear_efficiency_score = float(np.clip(100 - overrev_rate * 10 - lugging_rate * 8, 0, 100))

        # ── 4. Speed score (15%) ──────────────────────────────────────────
        over80_mask  = (speed30 > 80).to_numpy()
        over120_mask = (speed30 > 120).to_numpy()
        over80_rate  = over80_mask.sum() / n30 * 100
        over120_rate = over120_mask.sum() / n30 * 100
        speed_score  = float(np.clip(100 - over80_rate * 5 - over120_rate * 20, 0, 100))

        # ── 5. Fuel efficiency score (15%) ────────────────────────────────
        fuel_score = 50.0   # neutral default for EVs
        if fuel_type in ("ICE", "PHEV") and km30 > 1:
            fuel_used_l = float(fcon30.iloc[-1] - fcon30.iloc[0]) if len(fcon30) > 1 else 0.0
            actual_l100 = fuel_used_l / km30 * 100
            benchmark   = _FUEL_BENCHMARKS.get(model_code, _FUEL_BENCHMARKS["DEFAULT"])
            fuel_score  = float(np.clip(100 * benchmark / actual_l100, 0, 100)) if actual_l100 > 0 else 50.0

        # ── 6. Cornering score (10%) ──────────────────────────────────────
        steer_rate    = steer30.diff().abs()
        sudden_turns  = int((steer_rate > 30).sum())
        sudden_turn_rate = sudden_turns / km30 * 100 if km30 > 0 else 0.0
        cornering_score  = float(np.clip(100 - sudden_turn_rate * 12, 0, 100))

        # ── 7. Idle score (10%) ───────────────────────────────────────────
        idle_secs     = int(((speed30 < 2) & (pwr30 > 0)).sum())
        idle_fraction = idle_secs / n30
        idle_score    = float(np.clip(100 - idle_fraction * 50, 0, 100))

        # ── Composite drive score via derived_utils ───────────────────────
        component_scores = {
            "smooth_accel":    smooth_accel_score,
            "smooth_brake":    smooth_brake_score,
            "gear_efficiency": gear_efficiency_score,
            "speed":           speed_score,
            "fuel":            fuel_score,
            "cornering":       cornering_score,
            "idle":            idle_score,
        }
        composite_drive_score = compute_composite_drive_score(
            smooth_accel     = smooth_accel_score,
            smooth_brake     = smooth_brake_score,
            gear_eff         = gear_efficiency_score,
            speed_compliance = speed_score,
            fuel_eff         = fuel_score,
            cornering        = cornering_score,
            idle             = idle_score,
        )

        # ── Weekly score trend (4 weeks) ──────────────────────────────────
        weekly_score_trend = _weekly_trend(df, fuel_type, model_code)

        # ── Worst behaviour component ─────────────────────────────────────
        worst_behaviour = min(component_scores, key=component_scores.get)

        # ── Per-window averages for supplementary features ─────────────────
        avg_speed_30d = float(speed30[speed30 > 5].mean()) if (speed30 > 5).any() else 0.0
        overSpeed80_count_30d  = int(over80_mask.sum())
        overSpeed120_count_30d = int(over120_mask.sum())

        # ── Fuel efficiency vs VIN 90d baseline ───────────────────────────
        d90 = self._last_days(df, 90)
        km90 = float(d90["odometer"].max() - d90["odometer"].min()) if "odometer" in d90.columns and len(d90) > 1 else 1.0
        fcon90 = d90["fuel_consumed"].fillna(0) if "fuel_consumed" in d90.columns else pd.Series(dtype=float)
        fuel_used_90 = float(fcon90.iloc[-1] - fcon90.iloc[0]) if len(fcon90) > 1 else 0.0
        baseline_l100 = fuel_used_90 / km90 * 100 if km90 > 1 and fuel_used_90 > 0 else np.nan
        fuel_7d_val  = _mean_fuel_rate(d7)
        fuel_efficiency_vs_baseline = float(
            (baseline_l100 - fuel_7d_val) / baseline_l100 * 100
        ) if baseline_l100 and not np.isnan(fuel_7d_val) else 0.0

        return self._row(vin, t_max, {
            # Component scores
            "smooth_accel_score":           smooth_accel_score,
            "smooth_brake_score":           smooth_brake_score,
            "gear_efficiency_score":        gear_efficiency_score,
            "speed_score":                  speed_score,
            "fuel_score":                   fuel_score,
            "cornering_score":              cornering_score,
            "idle_score":                   idle_score,
            # Composite and meta
            "composite_drive_score":        composite_drive_score,
            "weekly_score_trend":           weekly_score_trend,
            "worst_behaviour":              worst_behaviour,
            "peer_percentile":              50.0,   # filled by compute_batch; default 50th pctile
            "fuel_efficiency_vs_baseline":  fuel_efficiency_vs_baseline,
            # Detailed event counts / rates
            "harsh_accel_rate_30d":         harsh_accel_rate,
            "harsh_brake_rate_30d":         harsh_brake_rate,
            "overSpeed80_count_30d":        overSpeed80_count_30d,
            "overSpeed120_count_30d":       overSpeed120_count_30d,
            # Standardised snake_case aliases used by driver_score_model
            "overspeed_80_fraction_30d":    round(float(over80_mask.mean()), 4),
            "overspeed_120_count_30d":      overSpeed120_count_30d,
            "overrev_rate_30d":             round(float(overrev_rate), 4),
            "lugging_rate_30d":             round(float(lugging_rate), 4),
            "night_driving_fraction_30d":   0.0,   # placeholder — requires timestamp hour
            "fuel_efficiency_score":        round(float(fuel_score), 2),
            "sudden_turn_rate_30d":         sudden_turn_rate,
            "idle_fraction_30d":            idle_fraction,
            "avg_speed_30d":                avg_speed_30d,
        })

    def compute_batch(
        self,
        fleet_df: pd.DataFrame,
        telemetry_dir=None,
        label_df=None,
        **ctx,
    ) -> pd.DataFrame:
        """Compute features for all VINs, then fill peer_percentile column."""
        result = super().compute_batch(fleet_df, telemetry_dir, label_df, **ctx)
        if not result.empty and "composite_drive_score" in result.columns:
            scores = result["composite_drive_score"]
            result["peer_percentile"] = scores.rank(pct=True) * 100
        return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _weekly_trend(df: pd.DataFrame, fuel_type: str, model_code: str) -> float:
    """Slope of weekly composite drive scores over last 4 weeks."""
    if "timestamp" not in df.columns or len(df) < 1000:
        return 0.0
    t_max = df["timestamp"].max()
    weekly_scores: list[float] = []
    for week in range(4, 0, -1):
        end   = t_max - pd.Timedelta(days=(week - 1) * 7)
        start = end   - pd.Timedelta(days=7)
        seg   = df[(df["timestamp"] >= start) & (df["timestamp"] < end)]
        if len(seg) < 100:
            weekly_scores.append(np.nan)
            continue
        # Quick composite from available columns
        speed = seg["speed"].fillna(0) if "speed" in seg.columns else pd.Series(0, index=seg.index)
        bpos  = seg["brake_pos"].fillna(0) if "brake_pos" in seg.columns else pd.Series(0, index=seg.index)
        n     = max(len(speed), 1)
        s_spd = float(np.clip(100 - (speed > 80).sum() / n * 500, 0, 100))
        s_brk = float(np.clip(100 - (bpos > 70).sum() / n * 1200, 0, 100))
        weekly_scores.append((s_spd + s_brk) / 2)

    valid = [s for s in weekly_scores if not np.isnan(s)]
    if len(valid) < 2:
        return 0.0
    return float(np.polyfit(np.arange(len(valid)), valid, 1)[0])


def _mean_fuel_rate(seg: pd.DataFrame) -> float:
    """L/100km for a segment."""
    if "fuel_consumed" not in seg.columns or "odometer" not in seg.columns:
        return np.nan
    km  = float(seg["odometer"].max() - seg["odometer"].min())
    fv  = float(seg["fuel_consumed"].max() - seg["fuel_consumed"].min())
    return fv / km * 100 if km > 1 else np.nan
