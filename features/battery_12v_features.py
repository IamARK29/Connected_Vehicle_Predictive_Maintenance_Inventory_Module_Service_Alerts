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

# ── TBox parasitic draw constants ─────────────────────────────────────────────
# The TBox unit draws current continuously while parked.  This draw is EXPECTED
# and constant regardless of battery health — mixing it into the overnight-drop
# metric causes the model to over-penalise healthy batteries on long parks.

TBOX_CURRENT_DRAW_MA: dict[str, int] = {
    "deep_sleep": 15,    # minimal keep-alive, no data being sent
    "heartbeat":  28,    # periodic position pings, cellular radio active
    "active":     42,    # OTA in progress or live telemetry streaming
}

# Lead-acid 45 Ah: ~0.022 V per Ah drained from the 12.6 V resting state.
# Varies with battery age and temperature; 0.022 is a reasonable mid-point.
LEAD_ACID_VOLT_DROP_PER_AH: float = 0.022


def estimate_tbox_voltage_draw(
    park_duration_hours: float,
    avg_cell_signal_dbm: float,
    battery_age_years: float = 1.0,  # noqa: ARG001 (reserved for future calibration)
) -> float:
    """Estimate the voltage contribution of TBox parasitic draw during a park.

    Args:
        park_duration_hours:  hours the vehicle was parked
        avg_cell_signal_dbm:  mean cellSignalStrength during park (dBm).
                              > -85  → heartbeat mode (more draw)
                              > -100 → light deep-sleep
                              ≤ -100 → deep sleep (least draw)
        battery_age_years:    reserved for future age-based calibration

    Returns:
        Estimated voltage drop attributable to TBox in Volts.
    """
    if avg_cell_signal_dbm > -85:
        draw_ma = TBOX_CURRENT_DRAW_MA["heartbeat"]
    elif avg_cell_signal_dbm > -100:
        draw_ma = TBOX_CURRENT_DRAW_MA["deep_sleep"] + 5
    else:
        draw_ma = TBOX_CURRENT_DRAW_MA["deep_sleep"]

    charge_consumed_ah = (draw_ma / 1000.0) * park_duration_hours
    return round(charge_consumed_ah * LEAD_ACID_VOLT_DROP_PER_AH, 4)


def compute_overnight_drop_battery_only(
    raw_overnight_drop_v: float,
    park_duration_hours: float,
    avg_cell_signal_dbm: float,
    battery_age_years: float = 1.0,
) -> dict[str, float]:
    """Separate overnight voltage drop into TBox draw and true battery self-discharge.

    Args:
        raw_overnight_drop_v:  total measured drop (V_at_engine_off − V_at_next_start)
        park_duration_hours:   hours between the two measurements
        avg_cell_signal_dbm:   mean cell signal during park (-128 if no signal)
        battery_age_years:     age of the 12V battery in years

    Returns dict with three keys for use in health scoring and alerts.
    """
    tbox_drop = estimate_tbox_voltage_draw(park_duration_hours, avg_cell_signal_dbm, battery_age_years)
    battery_self_discharge_v = max(0.0, raw_overnight_drop_v - tbox_drop)
    return {
        "overnight_drop_total_v":          round(raw_overnight_drop_v, 4),
        "overnight_drop_tbox_component_v": round(tbox_drop, 4),
        "overnight_drop_battery_only_v":   round(battery_self_discharge_v, 4),
    }


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
        # Allow vrow to override the manufacture_year default
        _vrow = _ctx.get("vrow", {})
        _mfr = getattr(_vrow, "manufacture_year", None) or (
            _vrow.get("manufacture_year", None) if hasattr(_vrow, "get") else None
        )
        if _mfr is not None:
            manufacture_year = int(_mfr)
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

        # ── Battery age and odometer proxy ────────────────────────────────
        current_year     = t_max.year if hasattr(t_max, "year") else datetime.now().year
        battery_age_days = float((current_year - manufacture_year) * 365 + 180)
        battery_age_years = battery_age_days / 365.0
        total_km_proxy   = float(odo.iloc[-1]) if len(odo) > 0 else np.nan

        # ── Overnight voltage drop with TBox parasitic separation ─────────
        overnight_decomposed = _compute_overnight_drop_separated(batt, pwr, df, battery_age_years)
        overnight_voltage_drop_avg = overnight_decomposed["overnight_drop_total_v"]

        # ── Cold cranking voltage dip ──────────────────────────────────────
        # At each startup, capture the minimum voltage in first 5 seconds
        cold_voltage_delta = _compute_cold_voltage_delta(batt, pwr)

        # ── Voltage under load: minimum during RPM spike at session start ──
        # Approximated by: min batt_12v in first 30s of driving when speed > 0
        voltage_under_load_proxy = _compute_under_load_voltage(df)

        # ── Parasitic drain rate (V/hour during extended OFF periods) ──────
        parasitic_drain_rate_7d = _compute_parasitic_drain(batt7, pwr7)

        # ── Lights-on engine-off events (7d) — from real TBox light signals ─
        # vehDipLight or vehMainLight = 1 when engine off (SysPwrMod == 0): parasitic drain
        if "dip_light" in d7.columns or "main_light" in d7.columns:
            lights7 = (
                d7.get("dip_light",  pd.Series(0, index=d7.index)).fillna(0) |
                d7.get("main_light", pd.Series(0, index=d7.index)).fillna(0)
            )
            engine_off7 = d7["sys_pwr_mod"].fillna(-1) == 0 if "sys_pwr_mod" in d7.columns else pd.Series(False, index=d7.index)
            lights_on_engine_off_count_7d = int((lights7.astype(bool) & engine_off7).sum())
        else:
            lights_on_engine_off_count_7d = 0

        # ── AC load drain (7d) — vehAC on when SysPwrMod == 1 (ACC mode) ──
        if "ac_on" in d7.columns and "sys_pwr_mod" in d7.columns:
            acc_drain7 = (d7["ac_on"].fillna(0) > 0) & (d7["sys_pwr_mod"].fillna(0) == 1)
            ac_acc_drain_minutes_7d = float(acc_drain7.sum() / 60)
        else:
            ac_acc_drain_minutes_7d = 0.0

        # ── Average outside temp (7d) ─────────────────────────────────────
        avg_outside_temp_7d = float(tmp7.dropna().mean()) if len(tmp7.dropna()) > 0 else 0.0

        # ── Health score via derived_utils ────────────────────────────────
        battery_12v_health_score = compute_battery_12v_health_score(
            resting_v     = resting_voltage_7d_avg if not np.isnan(resting_voltage_7d_avg) else 12.0,
            trend_per_day = resting_voltage_trend_14d,
            cranking_v    = float(voltage_under_load_proxy) if not np.isnan(voltage_under_load_proxy) else 10.5,
            age_years     = battery_age_days / 365.0,
        )
        # Use battery-only drop in health score so TBox parasitic draw does not
        # incorrectly degrade the score of a healthy battery on a long park.
        battery_only_drop = overnight_decomposed["overnight_drop_battery_only_v"]
        health_score = _health_score(
            resting_voltage_7d_avg, resting_voltage_trend_14d,
            battery_only_drop, cold_voltage_delta,
            parasitic_drain_rate_7d, battery_age_days,
        )

        # ── Current batt voltage ──────────────────────────────────────────
        voltage_current = float(batt.iloc[-1]) if len(batt) > 0 else np.nan

        # ── Labels — voltage-trend self-supervision ───────────────────────
        # Physics: failure threshold ~11.8V. Extrapolate resting_voltage trend.
        _FAIL_THRESHOLD_V = 11.8
        _WARN_THRESHOLD_V = 12.2
        v_now = resting_voltage_7d_avg if not np.isnan(resting_voltage_7d_avg) else 12.6
        v_trend = resting_voltage_trend_14d if not np.isnan(resting_voltage_trend_14d) else 0.0
        if v_trend < -0.001:
            days_to_physics = float(np.clip((v_now - _FAIL_THRESHOLD_V) / abs(v_trend), 0, 730))
        else:
            days_to_physics = 730.0
        # Also flag by age: 12V batteries typically need replacement after 3 years
        _BATTERY_WARN_AGE_YEARS = 3.0
        within_physics = int(
            v_now < _WARN_THRESHOLD_V or
            overnight_voltage_drop_avg > 0.3 or
            (battery_age_days / 365.0) >= _BATTERY_WARN_AGE_YEARS
        )

        days_to_failure, within_svc = self._label_days_to_failure(
            vin, label_df, "12v_battery_failure", t_max
        )
        if np.isnan(days_to_failure):
            days_to_failure = days_to_physics
        within_7 = int(within_svc or within_physics)

        cold_weather_risk = max(1.0, 1.0 + (10.0 - float(tmp7.fillna(20).mean())) / 10.0) if len(tmp7) > 0 else 1.0
        overnight_drop_avg_7d = overnight_voltage_drop_avg
        overnight_drop_tbox_v    = overnight_decomposed["overnight_drop_tbox_component_v"]
        overnight_drop_battery_v = battery_only_drop

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
            "ac_acc_drain_minutes_7d":          ac_acc_drain_minutes_7d,
            "cold_weather_risk_multiplier":     cold_weather_risk,
            # Legacy / extra
            "overnight_voltage_drop_avg":       overnight_voltage_drop_avg,
            "overnight_drop_tbox_component_v":  overnight_drop_tbox_v,
            "overnight_drop_battery_only_v":    overnight_drop_battery_v,
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
    """Average voltage drop between engine-off and next engine-on (vectorised)."""
    if len(pwr) < 2:
        return 0.0
    p = pwr.fillna(-1).values.astype(float)
    b = batt.values
    # transitions: off→on (+1 diff going positive), on→off (+1 diff going negative
    p_int = (p > 0).astype(int)
    diff  = np.diff(p_int)
    off_idx = np.where(diff == -1)[0]     # indices where engine turns off
    on_idx  = np.where(diff ==  1)[0]     # indices where engine turns on
    drops: list[float] = []
    for oi in off_idx:
        # find next on-transition after this off
        next_on = on_idx[on_idx > oi]
        if len(next_on) == 0:
            break
        ni = next_on[0]
        bv_off = b[oi] if np.isfinite(b[oi]) else None
        bv_on  = b[ni + 1] if ni + 1 < len(b) and np.isfinite(b[ni + 1]) else None
        if bv_off is not None and bv_on is not None:
            drops.append(float(bv_off - bv_on))
    return float(np.mean(drops)) if drops else 0.0


def _compute_overnight_drop_separated(
    batt: pd.Series,
    pwr: pd.Series,
    df: pd.DataFrame,
    battery_age_years: float = 1.0,
) -> dict[str, float]:
    """Decompose each park-period drop into TBox draw and true self-discharge.

    Uses timestamps from df (after _normalize → reset_index) so positional
    indices from np.where align with df.iloc[i].
    """
    _zero = {"overnight_drop_total_v": 0.0, "overnight_drop_tbox_component_v": 0.0,
              "overnight_drop_battery_only_v": 0.0}
    if len(pwr) < 2:
        return _zero

    p_int = (pwr.fillna(-1).values > 0).astype(int)
    b     = batt.values
    diff  = np.diff(p_int)
    off_idx = np.where(diff == -1)[0]
    on_idx  = np.where(diff ==  1)[0]

    # Timestamps for park duration; fall back if not available
    ts_arr = df["timestamp"].values if "timestamp" in df.columns else None

    # Cell signal for TBox mode inference; default deep-sleep when absent
    sig_col  = next((c for c in ("cellSignalStrength", "cell_signal") if c in df.columns), None)
    sig_vals = df[sig_col].values if sig_col else None

    totals, tbox_drops, batt_drops = [], [], []

    for oi in off_idx:
        next_on = on_idx[on_idx > oi]
        if len(next_on) == 0:
            break
        ni = next_on[0]

        bv_off = b[oi]     if np.isfinite(b[oi])     else None
        bv_on  = b[ni + 1] if ni + 1 < len(b) and np.isfinite(b[ni + 1]) else None
        if bv_off is None or bv_on is None:
            continue
        raw_drop = float(bv_off - bv_on)

        # Park duration from timestamps
        park_h = 8.0  # default: 8-hour overnight assumption
        if ts_arr is not None:
            try:
                t_off = pd.Timestamp(ts_arr[oi])
                t_on  = pd.Timestamp(ts_arr[min(ni + 1, len(ts_arr) - 1)])
                park_h = max(0.0, (t_on - t_off).total_seconds() / 3600.0)
            except Exception:
                pass

        # Mean cell signal during park window
        avg_sig = -110.0  # default: deep sleep when signal not available
        if sig_vals is not None and ni > oi:
            park_sigs = sig_vals[oi:ni + 1]
            valid_sigs = park_sigs[np.isfinite(park_sigs.astype(float))]
            if len(valid_sigs) > 0:
                avg_sig = float(valid_sigs.mean())

        tbox_drop = estimate_tbox_voltage_draw(park_h, avg_sig, battery_age_years)
        batt_only = max(0.0, raw_drop - tbox_drop)

        totals.append(raw_drop)
        tbox_drops.append(tbox_drop)
        batt_drops.append(batt_only)

    if not totals:
        return _zero
    return {
        "overnight_drop_total_v":          round(float(np.mean(totals)),     4),
        "overnight_drop_tbox_component_v": round(float(np.mean(tbox_drops)), 4),
        "overnight_drop_battery_only_v":   round(float(np.mean(batt_drops)), 4),
    }


def _compute_cold_voltage_delta(batt: pd.Series, pwr: pd.Series) -> float:
    """Voltage dip during cranking — vectorised."""
    if len(pwr) < 2:
        return 0.0
    p_int = (pwr.fillna(-1).values > 0).astype(int)
    b     = batt.values
    on_idx = np.where(np.diff(p_int) == 1)[0]   # positions just before engine start
    dips: list[float] = []
    for i in on_idx:
        pre_v = b[i] if i < len(b) and np.isfinite(b[i]) else None
        window = b[i + 1: i + 6]
        valid  = window[np.isfinite(window)]
        if pre_v is not None and len(valid) > 0:
            dips.append(float(pre_v - valid.min()))
    return float(np.mean(dips)) if dips else 0.0


def _compute_under_load_voltage(df: pd.DataFrame) -> float:
    """Min 12V in first 30 rows of each drive session (vectorised)."""
    if "batt_12v" not in df.columns or "sys_pwr_mod" not in df.columns:
        return 0.0
    p_int  = (df["sys_pwr_mod"].fillna(-1).values > 0).astype(int)
    b      = df["batt_12v"].values
    on_idx = np.where(np.diff(p_int) == 1)[0]
    minima: list[float] = []
    for i in on_idx:
        window = b[i + 1: i + 31]
        valid  = window[np.isfinite(window)]
        if len(valid) > 0:
            minima.append(float(valid.min()))
    return float(np.mean(minima)) if minima else 0.0


def _compute_parasitic_drain(batt7: pd.Series, pwr7: pd.Series) -> float:
    """V/hour drop rate during extended OFF runs (> 1 hour, vectorised)."""
    if len(batt7) == 0 or len(pwr7) == 0:
        return np.nan
    p_int = (pwr7.fillna(-1).values > 0).astype(int)
    b     = batt7.values
    # Find contiguous OFF runs using group IDs
    group = np.cumsum(np.concatenate([[0], np.diff(p_int) != 0]))
    rates: list[float] = []
    for g in np.unique(group):
        mask = group == g
        if int(p_int[mask][0]) != 0:
            continue
        run_b = b[mask]
        valid = run_b[np.isfinite(run_b)]
        if len(valid) < 2 or len(valid) < 3600:  # < 1 hour at 1-sample/s
            continue
        hours = len(valid) / 3600.0
        rates.append(float(valid[0] - valid[-1]) / hours)
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
