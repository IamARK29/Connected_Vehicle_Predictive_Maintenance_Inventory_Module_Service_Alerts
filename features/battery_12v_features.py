"""
12V auxiliary battery health feature engineering pipeline.

Detects resting voltage decline, parasitic drain, cold-voltage dips,
and overnight drop — all leading indicators of imminent no-start events.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from features.base_pipeline import FeaturePipeline
from features.derived_utils import get_resting_voltage, compute_battery_12v_health_score


class Battery12VFeaturePipeline(FeaturePipeline):

    def compute(
        self,
        vin: str,
        df: pd.DataFrame,
        label_df: pd.DataFrame | None = None,
        *,
        manufacture_year: int = 2022,
        **_ctx: Any,
    ) -> pd.DataFrame:
        df = self._normalize(df)
        if df.empty or "timestamp" not in df.columns:
            return pd.DataFrame()

        t_max   = df["timestamp"].max()
        d7      = self._last_days(df, 7)
        d14     = self._last_days(df, 14)

        batt    = df["batt_12v"].fillna(np.nan) if "batt_12v"     in df.columns else pd.Series(dtype=float)
        pwr     = df["sys_pwr_mod"].fillna(-1)  if "sys_pwr_mod"  in df.columns else pd.Series(-1, index=df.index)
        out_tmp = df["outside_temp"].fillna(np.nan) if "outside_temp" in df.columns else pd.Series(dtype=float)
        odo     = df["odometer"]                if "odometer"     in df.columns else pd.Series(dtype=float)

        batt7   = d7["batt_12v"].fillna(np.nan) if "batt_12v"    in d7.columns  else pd.Series(dtype=float)
        pwr7    = d7["sys_pwr_mod"].fillna(-1)  if "sys_pwr_mod" in d7.columns  else pd.Series(-1, index=d7.index)
        batt14  = d14["batt_12v"].fillna(np.nan) if "batt_12v"   in d14.columns else pd.Series(dtype=float)
        pwr14   = d14["sys_pwr_mod"].fillna(-1) if "sys_pwr_mod" in d14.columns else pd.Series(-1, index=d14.index)
        tmp7    = d7["outside_temp"].fillna(np.nan) if "outside_temp" in d7.columns else pd.Series(dtype=float)

        # ── Resting voltage: via derived_utils (door-settled filter) ─────
        resting_series     = get_resting_voltage(df)
        resting7_series    = resting_series.loc[d7.index] if not d7.empty else pd.Series(dtype=float)
        resting14_series   = resting_series.loc[d14.index] if not d14.empty else pd.Series(dtype=float)

        # Fall back to simple off-mask if derived resting series is empty
        off_mask7  = pwr7  == 0
        off_mask14 = pwr14 == 0
        resting7   = resting7_series.dropna() if not resting7_series.dropna().empty else batt7[off_mask7]
        resting14  = resting14_series.dropna() if not resting14_series.dropna().empty else batt14[off_mask14]

        resting_voltage_7d_avg    = float(resting7.mean())  if len(resting7)  > 0 else np.nan
        resting_voltage_trend_14d = self._slope(resting14)

        # ── Overnight voltage drop ─────────────────────────────────────────
        # For each OFF→ON transition: voltage at ON_start minus previous ON_end
        overnight_voltage_drop_avg = _compute_overnight_drop(batt, pwr)

        # ── Cold cranking voltage dip ──────────────────────────────────────
        # At each startup, capture the minimum voltage in first 5 seconds
        cold_voltage_delta = _compute_cold_voltage_delta(batt, pwr)

        # ── Voltage under load: minimum during RPM spike at session start ──
        # Approximated by: min batt_12v in first 30s of driving when speed > 0
        voltage_under_load_proxy = _compute_under_load_voltage(df)

        # ── Parasitic drain rate (V/hour during extended OFF periods) ──────
        parasitic_drain_rate_7d = _compute_parasitic_drain(batt7, pwr7)

        # ── Battery age and odometer proxy ────────────────────────────────
        current_year   = t_max.year if hasattr(t_max, "year") else datetime.now().year
        battery_age_days = float((current_year - manufacture_year) * 365 + 180)
        total_km_proxy   = float(odo.iloc[-1]) if len(odo) > 0 else np.nan

        # ── Lights-on engine-off events (7d) ─────────────────────────────
        # Synthetic data doesn't have light status; use as 0 placeholder
        lights_on_engine_off_count_7d = 0

        # ── Average outside temp (7d) ─────────────────────────────────────
        avg_outside_temp_7d = float(tmp7.dropna().mean()) if len(tmp7.dropna()) > 0 else 0.0

        # ── Health score via derived_utils ────────────────────────────────
        battery_12v_health_score = compute_battery_12v_health_score(
            resting_v     = resting_voltage_7d_avg if not np.isnan(resting_voltage_7d_avg) else 12.0,
            trend_per_day = resting_voltage_trend_14d,
            cranking_v    = float(voltage_under_load_proxy) if not np.isnan(voltage_under_load_proxy) else 10.5,
            age_years     = battery_age_days / 365.0,
        )
        health_score = _health_score(
            resting_voltage_7d_avg, resting_voltage_trend_14d,
            overnight_voltage_drop_avg, cold_voltage_delta,
            parasitic_drain_rate_7d, battery_age_days,
        )

        # ── Current batt voltage ──────────────────────────────────────────
        voltage_current = float(batt.iloc[-1]) if len(batt) > 0 else np.nan

        # ── Labels ────────────────────────────────────────────────────────
        days_to_failure, within_7 = self._label_days_to_failure(
            vin, label_df, "12v_battery_failure", t_max
        )

        cold_weather_risk = max(1.0, 1.0 + (10.0 - float(tmp7.fillna(20).mean())) / 10.0) if len(tmp7) > 0 else 1.0
        overnight_drop_avg_7d = overnight_voltage_drop_avg

        # ── Simple aggregate features expected by tests ─────────────────
        batt7_valid = batt7.dropna()
        batt_12v_mean_7d = float(batt7_valid.mean()) if len(batt7_valid) > 0 else 0.0
        batt_12v_min_7d  = float(batt7_valid.min())  if len(batt7_valid) > 0 else 0.0

        return self._row(vin, t_max, {
            "batt_12v_mean_7d":                 batt_12v_mean_7d,
            "batt_12v_min_7d":                  batt_12v_min_7d,
            "resting_voltage_7d_avg":           resting_voltage_7d_avg,
            "resting_voltage_trend_14d":        resting_voltage_trend_14d,
            "overnight_drop_avg_7d":            overnight_drop_avg_7d,
            "cranking_voltage_dip_avg":         cold_voltage_delta,
            "voltage_recovery_rate":            voltage_under_load_proxy,
            "parasitic_drain_rate":             parasitic_drain_rate_7d,
            "battery_12v_health_score":         battery_12v_health_score,
            "battery_age_years":                battery_age_days / 365.0,
            "light_on_engine_off_events_7d":    lights_on_engine_off_count_7d,
            "cold_weather_risk_multiplier":     cold_weather_risk,
            # Legacy / extra
            "overnight_voltage_drop_avg":       overnight_voltage_drop_avg,
            "cold_voltage_delta":               cold_voltage_delta,
            "voltage_under_load_proxy":         voltage_under_load_proxy,
            "parasitic_drain_rate_7d":          parasitic_drain_rate_7d,
            "battery_age_days":                 battery_age_days,
            "total_km_proxy":                   total_km_proxy,
            "lights_on_engine_off_count_7d":    lights_on_engine_off_count_7d,
            "avg_outside_temp_7d":              avg_outside_temp_7d,
            "voltage_current":                  voltage_current,
            "health_score":                     health_score,
            # Targets
            "days_to_battery_12v_failure":      days_to_failure,
            "battery_12v_within_30_days":       within_7,
        })


# ── Internal helpers ─────────────────────────────────────────────────────────

def _compute_overnight_drop(batt: pd.Series, pwr: pd.Series) -> float:
    """Average voltage drop between end of one session and start of next."""
    drops: list[float] = []
    off_start_v: float | None = None

    for i in range(1, len(pwr)):
        prev, cur = int(pwr.iloc[i - 1]), int(pwr.iloc[i])
        bv = float(batt.iloc[i]) if not np.isnan(batt.iloc[i]) else None
        prev_bv = float(batt.iloc[i - 1]) if not np.isnan(batt.iloc[i - 1]) else None
        # Engine going OFF
        if prev > 0 and cur == 0:
            off_start_v = prev_bv
        # Engine starting again
        if prev == 0 and cur > 0 and off_start_v is not None and bv is not None:
            drops.append(off_start_v - bv)
            off_start_v = None

    return float(np.mean(drops)) if drops else 0.0


def _compute_cold_voltage_delta(batt: pd.Series, pwr: pd.Series) -> float:
    """Voltage dip during cranking = pre-start voltage minus minimum in first 5s."""
    dips: list[float] = []
    for i in range(1, len(pwr) - 5):
        if int(pwr.iloc[i - 1]) == 0 and int(pwr.iloc[i]) > 0:
            pre_v = float(batt.iloc[i - 1]) if not np.isnan(batt.iloc[i - 1]) else None
            window = batt.iloc[i:i + 5].dropna().values
            if pre_v and len(window) > 0:
                dips.append(pre_v - float(window.min()))
    return float(np.mean(dips)) if dips else 0.0


def _compute_under_load_voltage(df: pd.DataFrame) -> float:
    """Min 12V during acceleration at session start (first 30s with speed > 0)."""
    if "batt_12v" not in df.columns or "sys_pwr_mod" not in df.columns:
        return 0.0
    pwr  = df["sys_pwr_mod"].fillna(-1).values.astype(int)
    batt = df["batt_12v"].fillna(np.nan).values
    speed = df["speed"].fillna(0).values if "speed" in df.columns else np.zeros(len(df))

    minima: list[float] = []
    in_session = False
    session_start = 0
    for i in range(1, len(pwr)):
        if pwr[i - 1] == 0 and pwr[i] > 0:
            in_session = True
            session_start = i
        if in_session and (i - session_start) <= 30 and speed[i] > 1:
            pass  # collecting
        elif in_session and (i - session_start) == 31:
            window = batt[session_start:i]
            valid  = window[np.isfinite(window)]
            if len(valid) > 0:
                minima.append(float(valid.min()))
            in_session = False

    return float(np.mean(minima)) if minima else 0.0


def _compute_parasitic_drain(batt7: pd.Series, pwr7: pd.Series) -> float:
    """V/hour drop rate during extended OFF periods (> 1 hour)."""
    if len(batt7) == 0 or len(pwr7) == 0:
        return np.nan

    rates: list[float] = []
    off_streak = 0
    off_start_v: float | None = None

    for i in range(len(pwr7)):
        p  = int(pwr7.iloc[i])
        bv = float(batt7.iloc[i]) if not np.isnan(batt7.iloc[i]) else None
        if p == 0:
            off_streak += 1
            if off_streak == 1:
                off_start_v = bv
            elif off_streak > 3600 and off_start_v is not None and bv is not None:
                hours = off_streak / 3600
                rates.append((off_start_v - bv) / hours)
        else:
            off_streak = 0
            off_start_v = None

    return float(np.mean(rates)) if rates else 0.0


def _health_score(
    resting_v: float, v_trend: float, overnight_drop: float,
    cold_delta: float, drain_rate: float, age_days: float,
) -> float:
    """Composite 0-100 health score (100 = fully healthy)."""
    score = 100.0

    if not np.isnan(resting_v):
        # Ideal resting voltage ≥ 12.6V; every 0.1V below costs 5 points
        score -= max(0.0, (12.6 - resting_v) / 0.1 * 5)

    if not np.isnan(v_trend):
        # Declining trend: each 0.001V/step drop costs 3 points
        score -= max(0.0, -v_trend * 1000 * 3)

    if not np.isnan(overnight_drop):
        # Normal overnight drop < 0.2V; above that costs points
        score -= max(0.0, (overnight_drop - 0.2) * 30)

    if not np.isnan(cold_delta):
        # Normal cranking dip < 1.0V; larger dip = worse battery
        score -= max(0.0, (cold_delta - 1.0) * 15)

    if not np.isnan(drain_rate):
        # Parasitic drain > 0.02 V/h is high
        score -= max(0.0, (drain_rate - 0.02) * 200)

    # Age penalty: 2% per year after 3 years
    score -= max(0.0, (age_days / 365 - 3) * 2)

    return float(np.clip(score, 0, 100))
