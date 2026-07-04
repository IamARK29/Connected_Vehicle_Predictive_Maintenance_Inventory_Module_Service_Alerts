"""
HV Battery State-of-Health (SoH) feature engineering pipeline.

Works with EV and PHEV vehicles. Computes per-cycle energy metrics,
cell voltage spread trends, thermal fault counts, and range degradation.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from features.base_pipeline import FeaturePipeline
from features.derived_utils import compute_soh_from_charge_session


class HVBatteryFeaturePipeline(FeaturePipeline):

    def compute(
        self,
        vin: str,
        df: pd.DataFrame,
        label_df: pd.DataFrame | None = None,
        *,
        battery_capacity_kwh: float | None = None,
        fuel_type: str = "EV",
        **_ctx: Any,
    ) -> pd.DataFrame:
        df = self._normalize(df)
        if df.empty or "timestamp" not in df.columns:
            return pd.DataFrame()

        # Only meaningful for EV / PHEV
        if fuel_type not in ("EV", "PHEV"):
            return self._row(vin, df["timestamp"].max(), {"fuel_type": fuel_type, "hv_applicable": 0})

        t_max = df["timestamp"].max()
        d7    = self._last_days(df, 7)
        d30   = self._last_days(df, 30)
        d90   = self._last_days(df, 90)

        soc        = df["soc"].fillna(np.nan)              if "soc"                   in df.columns else pd.Series(dtype=float)
        # soh does NOT exist as a TBox signal — derived via Coulomb counting in pipeline
        pack_vol   = df["bms_pack_vol"].fillna(np.nan)     if "bms_pack_vol"          in df.columns else pd.Series(dtype=float)
        pack_crnt  = df["bms_pack_crnt"].fillna(np.nan)    if "bms_pack_crnt"         in df.columns else pd.Series(dtype=float)
        cell_max_v = df["cell_max_vol"].fillna(np.nan)     if "cell_max_vol"          in df.columns else pd.Series(dtype=float)
        cell_min_v = df["cell_min_vol"].fillna(np.nan)     if "cell_min_vol"          in df.columns else pd.Series(dtype=float)
        cell_max_t = df["cell_max_temp"].fillna(np.nan)    if "cell_max_temp"         in df.columns else pd.Series(dtype=float)
        cell_min_t = df["cell_min_temp"].fillna(np.nan)    if "cell_min_temp"         in df.columns else pd.Series(dtype=float)
        # Real TBox thermal runaway and fault signals
        bms_cmu_flt   = df["bms_cmu_fault"].fillna(0)          if "bms_cmu_fault"         in df.columns else pd.Series(0, index=df.index)
        bms_cv_flt    = df["bms_cell_volt_fault"].fillna(0)     if "bms_cell_volt_fault"   in df.columns else pd.Series(0, index=df.index)
        bms_pt_flt    = df["bms_pack_temp_fault"].fillna(0)     if "bms_pack_temp_fault"   in df.columns else pd.Series(0, index=df.index)
        dcdc_temp     = df["dcdc_temp"].fillna(np.nan)          if "dcdc_temp"             in df.columns else pd.Series(dtype=float)
        is_chg        = df["is_charging"].fillna(0)             if "is_charging"           in df.columns else pd.Series(0, index=df.index)
        dc_or_ac      = df["dc_or_ac"]                          if "dc_or_ac"              in df.columns else pd.Series("", index=df.index)
        used_since_chg = df["used_battery_since_charge"].fillna(np.nan) if "used_battery_since_charge" in df.columns else pd.Series(dtype=float)

        # ── SoH estimation — multi-factor physics model ───────────────────
        # Primary: Coulomb counting from charge sessions; falls back to
        # physics formula that uses age + odometer + cell spread + temperature.
        soh_from_session = None
        if battery_capacity_kwh and len(df) >= 60:
            soh_from_session = compute_soh_from_charge_session(df, battery_capacity_kwh)
        soh_coulomb = _estimate_soh_from_cycles(soc, pack_vol, pack_crnt, battery_capacity_kwh)

        # Multi-factor physics fallback (much richer than age-only)
        vrow = _ctx.get("vrow", {})
        mfr_year = int(getattr(vrow, "manufacture_year", None) or
                       (vrow.get("manufacture_year", 2022) if hasattr(vrow, "get") else 2022))
        age_years = float(t_max.year - mfr_year)
        odo_km    = float(df["odometer"].iloc[-1]) if "odometer" in df.columns and len(df) > 0 else 0.0
        # Cell voltage spread from raw data (available before spread_all computed below)
        _cmx = df["cell_max_vol"].fillna(np.nan) if "cell_max_vol" in df.columns else pd.Series(dtype=float)
        _cmn = df["cell_min_vol"].fillna(np.nan) if "cell_min_vol" in df.columns else pd.Series(dtype=float)
        _spread_now = float((_cmx - _cmn).dropna().mean()) if len((_cmx - _cmn).dropna()) > 0 else 0.02
        # High cell temperature events accelerate degradation
        _cmax_t = df["cell_max_temp"].fillna(0) if "cell_max_temp" in df.columns else pd.Series(0.0, index=df.index)
        _high_t_frac = float((_cmax_t > 45).mean())

        cal_deg     = max(0.0, age_years - 1.0) * 2.5           # calendar: 2.5%/yr after yr-1
        cyc_deg     = min(12.0, odo_km / 10_000.0 * 0.8)        # cycle: 0.8%/10K km, cap 12%
        spread_deg  = max(0.0, (_spread_now - 0.02) / 0.40 * 15.0)  # spread: 0→15% over 0.02→0.42V
        thermal_deg = min(5.0, _high_t_frac * 50.0)             # thermal: up to 5%
        soh_physics = float(np.clip(100.0 - cal_deg - cyc_deg - spread_deg - thermal_deg, 60.0, 100.0))

        # Pick best available estimate in priority order
        for candidate in (soh_from_session, soh_coulomb, soh_physics):
            if candidate is not None and not np.isnan(float(candidate)) and 60.0 <= float(candidate) <= 110.0:
                soh_estimated: float = float(candidate)
                break
        else:
            soh_estimated = soh_physics

        # ── SoH trend: slope of rolling 30d SoH estimates across the 90d window ─
        soh_trend_slope_90d = _compute_soh_trend_slope(
            df, t_max, battery_capacity_kwh
        )

        # ── Cell voltage spread ───────────────────────────────────────────
        spread_all = (cell_max_v - cell_min_v).dropna()
        spread_30  = (d30["cell_max_vol"].fillna(np.nan) - d30["cell_min_vol"].fillna(np.nan)).dropna() if (
            "cell_max_vol" in d30.columns and "cell_min_vol" in d30.columns
        ) else pd.Series(dtype=float)

        cell_voltage_spread           = float(spread_all.mean())     if len(spread_all) > 0 else np.nan
        cell_voltage_spread_p95_30d   = float(np.nanpercentile(spread_30, 95)) if len(spread_30) > 0 else np.nan
        cell_voltage_spread_trend_30d = self._slope(spread_30)

        # ── Cell temperature delta ─────────────────────────────────────────
        temp_delta = (cell_max_t - cell_min_t).dropna()
        avg_cell_temp_delta = float(temp_delta.mean()) if len(temp_delta) > 0 else np.nan

        # ── Charge session analysis ────────────────────────────────────────
        charge_stats = _analyse_charge_sessions(df, d30, battery_capacity_kwh)

        # ── Thermal fault count (30d) — from real TBox BMS fault signals ────
        cmu_flt30  = d30["bms_cmu_fault"].fillna(0)        if "bms_cmu_fault"       in d30.columns else pd.Series(0, index=d30.index)
        cv_flt30   = d30["bms_cell_volt_fault"].fillna(0)  if "bms_cell_volt_fault" in d30.columns else pd.Series(0, index=d30.index)
        pt_flt30   = d30["bms_pack_temp_fault"].fillna(0)  if "bms_pack_temp_fault" in d30.columns else pd.Series(0, index=d30.index)
        cell_max30 = d30["cell_max_temp"].fillna(0)        if "cell_max_temp"        in d30.columns else pd.Series(0, index=d30.index)

        # BMS-reported fault events (any level > 0)
        bms_cmu_fault_count_30d  = int((cmu_flt30  > 0).sum())
        bms_cv_fault_count_30d   = int((cv_flt30   > 0).sum())
        bms_pt_fault_count_30d   = int((pt_flt30   > 0).sum())
        # Cell overtemp events as fallback thermal fault indicator
        cell_overtemp_count_30d  = int((cell_max30 > 45).sum())
        thermal_fault_count_30d  = max(bms_cmu_fault_count_30d + bms_cv_fault_count_30d + bms_pt_fault_count_30d,
                                       cell_overtemp_count_30d)
        # Maximum fault severity in last 30d (0=None, 1=L1, 2=L2, 3=L3)
        max_fault_severity_30d   = max(int(cmu_flt30.max()), int(pt_flt30.max()), int(cv_flt30.max()))

        # ── DCDC converter thermal stress ─────────────────────────────────
        dcdc30              = d30["dcdc_temp"].fillna(np.nan) if "dcdc_temp" in d30.columns else pd.Series(dtype=float)
        dcdc_temp_max_30d   = float(dcdc30.max()) if not dcdc30.dropna().empty else 0.0  # 0 = absent/nominal

        # ── DC fast charge ratio (30d) ─────────────────────────────────────
        dc_or_ac30          = d30["dc_or_ac"] if "dc_or_ac" in d30.columns else pd.Series("", index=d30.index)
        is_chg30            = d30["is_charging"].fillna(0) if "is_charging" in d30.columns else pd.Series(0, index=d30.index)
        chg_events          = is_chg30 > 0
        dc_charge_sessions  = int(((dc_or_ac30 == "dc") & chg_events).sum())
        total_chg_seconds   = int(chg_events.sum())
        dc_charge_ratio_30d = float(dc_charge_sessions / max(total_chg_seconds, 1))

        # ── Isolation resistance min (30d) ─────────────────────────────────
        isolation_resistance_min_30d = 0.0  # not in TBox spec, placeholder

        # ── Range per kWh trend (30d) ─────────────────────────────────────
        range_per_kwh_30d_trend = _range_per_kwh_trend(d30, battery_capacity_kwh)

        # ── Simple aggregate features ─────────────────────────────────────
        soc7        = d7["soc"].fillna(np.nan) if "soc" in d7.columns else pd.Series(dtype=float)
        soc_mean_7d = float(soc7.mean())       if not soc7.dropna().empty else (float(soc.mean()) if not soc.dropna().empty else 0.0)
        # SOH is derived; use Coulomb-counting estimate as the canonical value
        soh_mean    = soh_estimated if soh_estimated is not None and not np.isnan(soh_estimated) else 0.0

        # ── Current state ─────────────────────────────────────────────────
        soc_current = float(soc.iloc[-1])      if len(soc) > 0       else np.nan
        soh_current = soh_estimated             # best available estimate

        # ── Labels — SOH trend-based self-supervision ─────────────────────
        # Physics: extrapolate SOH trend to 80% threshold.
        # soh_trend_slope_90d is in %/day (negative = degrading).
        _SOH_WARN_THRESHOLD = 85.0   # warn before hitting 80%
        # Age-based degradation: compute from vrow regardless of SOH path taken
        _vrow = _ctx.get("vrow", {})
        _mfr_year = int(getattr(_vrow, "manufacture_year", None) or
                        (_vrow.get("manufacture_year", 2022) if hasattr(_vrow, "get") else 2022))
        _age_years = float(t_max.year - _mfr_year)
        if not np.isnan(soh_estimated):
            if soh_trend_slope_90d < -0.001:
                # Days until SOH drops to 80% at current slope
                days_to_physics = float(np.clip(
                    (soh_estimated - 80.0) / abs(soh_trend_slope_90d), 0, 3650
                ))
            else:
                days_to_physics = 3650.0  # stable, very far future
            # Positive label: trend extrapolation reaches <80% within 90d,
            # OR SoH already below the warn threshold and actively degrading.
            # Age alone is NOT sufficient — avoids labelling the whole fleet positive.
            within_physics = int(
                (soh_estimated < _SOH_WARN_THRESHOLD and soh_trend_slope_90d < -0.005) or
                (not np.isnan(days_to_physics) and days_to_physics < 90)
            )
        else:
            days_to_physics = np.nan
            within_physics  = 0

        days_to_failure, within_svc = self._label_days_to_failure(
            vin, label_df, "hv_battery_degradation", t_max
        )
        if np.isnan(days_to_failure):
            days_to_failure = days_to_physics
        soh_below_80_within_90 = int(
            within_svc or within_physics or
            (not np.isnan(soh_current) and soh_current < 80)
        )

        # ── OTA features ──────────────────────────────────────────────────
        ota_feats = {"days_since_last_bms_ota": None, "bms_ota_count_90d": 0, "post_bms_ota_efficiency_delta": 0.0}
        try:
            from ingestion.ota_tracker import OTATracker
            ota_feats = OTATracker().get_ota_features(vin)
        except Exception:
            pass

        # ── Contextual features ───────────────────────────────────────────
        thermal_zone = "moderate"
        elevation_stress = 0.0
        try:
            from features.contextual_features import ContextualFeatureEngine
            cfe = ContextualFeatureEngine()
            thermal_zone = cfe.thermal_zone(df)
            elevation_stress = cfe.elevation_stress(df, {"odometer": float(df["odometer"].iloc[-1] - df["odometer"].iloc[0]) if "odometer" in df.columns and len(df) > 1 else 1})
        except Exception:
            pass

        return self._row(vin, t_max, {
            "hv_applicable":                    1,
            "fuel_type":                        fuel_type,
            "soc_mean_7d":                      soc_mean_7d,
            "soh_mean":                         soh_mean,
            "soh_estimated":                    soh_estimated,
            "soh_trend_slope_90d":              soh_trend_slope_90d,
            "cell_voltage_spread":              cell_voltage_spread,
            "cell_voltage_spread_trend_30d":    cell_voltage_spread_trend_30d,
            "cell_voltage_spread_p95_30d":      cell_voltage_spread_p95_30d,
            "avg_cell_temp_delta":              avg_cell_temp_delta,
            "dc_charge_count_30d":              charge_stats["dc_count"],
            "charge_duration_deviation":        charge_stats["duration_deviation"],
            "avg_charge_c_rate_30d":            charge_stats["avg_c_rate"],
            "thermal_fault_count_30d":          thermal_fault_count_30d,
            # Real TBox BMS fault signals
            "bms_cmu_fault_count_30d":          bms_cmu_fault_count_30d,
            "bms_cv_fault_count_30d":           bms_cv_fault_count_30d,
            "bms_pt_fault_count_30d":           bms_pt_fault_count_30d,
            "max_fault_severity_30d":           max_fault_severity_30d,
            "dcdc_temp_max_30d":                dcdc_temp_max_30d,
            "dc_charge_ratio_30d":              dc_charge_ratio_30d,
            "isolation_resistance_min_30d":     isolation_resistance_min_30d,
            "range_per_kwh_30d_trend":          range_per_kwh_30d_trend,
            "soc_current":                      soc_current,
            "soh_current":                      soh_current,
            # Contextual
            "thermal_cold":                     int(thermal_zone == "cold"),
            "thermal_moderate":                 int(thermal_zone == "moderate"),
            "thermal_hot":                      int(thermal_zone == "hot"),
            "thermal_extreme":                  int(thermal_zone == "extreme"),
            "elevation_stress":                 elevation_stress,
            "days_since_last_bms_ota":          ota_feats.get("days_since_last_bms_ota"),
            "bms_ota_count_90d":                ota_feats.get("bms_ota_count_90d", 0),
            "post_bms_ota_efficiency_delta":    ota_feats.get("post_bms_ota_efficiency_delta", 0.0),
            # Targets
            "days_to_hv_failure":               days_to_failure,
            "soh_below_80_within_90_days":      soh_below_80_within_90,
        })


# ── SoH trend helpers ─────────────────────────────────────────────────────

def _compute_soh_trend_slope(
    df: pd.DataFrame,
    t_max: pd.Timestamp,
    battery_capacity_kwh: float | None,
) -> float:
    """
    Estimate SoH degradation rate in %/day using cell voltage spread slope
    across the 90-day window — more reliable than Coulomb counting when
    charging sessions are sparse.  Negative = degrading.
    """
    if "timestamp" not in df.columns or "cell_max_vol" not in df.columns or "cell_min_vol" not in df.columns:
        return -2.5 / 365.0

    soh_points: list[tuple[float, float]] = []

    for days_ago in (90, 60, 30, 0):
        t_end   = t_max - pd.Timedelta(days=days_ago)
        t_start = t_end  - pd.Timedelta(days=30)
        seg = df[(df["timestamp"] >= t_start) & (df["timestamp"] <= t_end)]
        if len(seg) < 20:
            continue

        # Cell voltage spread as SoH proxy: higher spread → lower SoH
        cmx = seg["cell_max_vol"].fillna(np.nan)
        cmn = seg["cell_min_vol"].fillna(np.nan)
        spread = float((cmx - cmn).dropna().mean()) if len((cmx - cmn).dropna()) > 0 else np.nan
        if np.isnan(spread):
            continue

        # Also try Coulomb counting if current data available
        soc_s  = seg["soc"].fillna(np.nan)  if "soc"          in seg.columns else pd.Series(dtype=float)
        vol_s  = seg["bms_pack_vol"].fillna(np.nan) if "bms_pack_vol" in seg.columns else pd.Series(dtype=float)
        crnt_s = seg["bms_pack_crnt"].fillna(np.nan) if "bms_pack_crnt" in seg.columns else pd.Series(dtype=float)
        soh_cc = _estimate_soh_from_cycles(soc_s, vol_s, crnt_s, battery_capacity_kwh)

        if soh_cc is not None and not np.isnan(float(soh_cc)) and 60 <= float(soh_cc) <= 110:
            soh_val = float(soh_cc)
        else:
            # Physics estimate from spread: 0.02V = ~100%, 0.42V = ~85%
            soh_val = float(np.clip(100.0 - max(0.0, (spread - 0.02) / 0.40 * 15.0), 60.0, 100.0))

        soh_points.append((-float(days_ago), soh_val))

    if len(soh_points) >= 2:
        x = np.array([p[0] for p in soh_points])
        y = np.array([p[1] for p in soh_points])
        if np.std(x) > 0 and np.std(y) > 1e-6:
            return float(np.polyfit(x, y, 1)[0])

    return -2.5 / 365.0


# ── Charge cycle helpers ───────────────────────────────────────────────────

def _estimate_soh_from_cycles(
    soc: pd.Series,
    pack_vol: pd.Series,
    pack_crnt: pd.Series,
    nominal_kwh: float | None,
) -> float:
    """
    Estimate SoH from charge events: energy delivered per full charge / nominal.

    Uses Ah-counting: energy_kwh = Σ(I × V × Δt) over sessions where SOC
    goes from ~20% to ~90%.
    """
    if nominal_kwh is None or len(soc) < 10:
        return float(soc.mean()) if len(soc) > 0 else np.nan

    # Identify charging phases: SOC increasing, current negative (charging)
    soc_vals = soc.fillna(np.nan).values
    if len(pack_crnt) == 0 or len(pack_vol) == 0:
        # Fall back to direct SoH field mean
        return float(soc_vals[np.isfinite(soc_vals)].mean()) if np.any(np.isfinite(soc_vals)) else np.nan

    crnt = pack_crnt.fillna(0).values
    vol  = pack_vol.fillna(0).values

    # Energy per second (W) — positive = discharging, negative = charging
    power_w = crnt * vol  # current sign: negative during charging
    dt_s = 1.0            # 1-Hz data

    charge_mask = (crnt < -1) & (soc_vals > 20) & (soc_vals < 95)
    if not np.any(charge_mask):
        return float(np.nanmean(soc_vals)) if np.any(np.isfinite(soc_vals)) else np.nan

    energy_delivered_kwh = float(np.abs(power_w[charge_mask]).sum() * dt_s / 3600 / 1000)
    delta_soc_sum        = float(np.abs(np.diff(soc_vals[charge_mask])).sum())

    if delta_soc_sum < 5:
        return np.nan

    # Scale to a 0→100% equivalent charge
    full_charge_kwh = energy_delivered_kwh / (delta_soc_sum / 100)
    soh = float(np.clip(full_charge_kwh / nominal_kwh * 100, 0, 110))
    return soh


def _analyse_charge_sessions(
    df: pd.DataFrame, d30: pd.DataFrame, nominal_kwh: float | None
) -> dict:
    """Compute DC charge count, C-rate, duration deviation from 30-day window."""
    out = {"dc_count": 0, "avg_c_rate": 0.0, "duration_deviation": 0.0}

    if "bms_pack_crnt" not in d30.columns or "soc" not in d30.columns:
        return out

    crnt30 = d30["bms_pack_crnt"].fillna(0)
    soc30  = d30["soc"].fillna(np.nan)

    # Charge sessions: runs where current < -5A
    charge_flag = (crnt30 < -5).astype(int)
    session_ids = (charge_flag.diff().fillna(0) > 0).cumsum()
    charge_sessions = charge_flag * session_ids

    c_rates: list[float] = []
    durations: list[float] = []

    for sid in charge_sessions.unique():
        if sid == 0:
            continue
        seg_mask = charge_sessions == sid
        seg_crnt = crnt30[seg_mask].abs()
        seg_soc  = soc30[seg_mask]
        if len(seg_crnt) < 10:
            continue
        if nominal_kwh:
            # C-rate = current / capacity_Ah; assume pack at ~350V
            cap_ah   = nominal_kwh * 1000 / 350
            avg_cr   = float(seg_crnt.mean() / cap_ah) if cap_ah > 0 else np.nan
            c_rates.append(avg_cr)
            # > 0.5C = DC fast charge
            if avg_cr > 0.5:
                out["dc_count"] += 1
        durations.append(len(seg_crnt) / 60)  # minutes

    out["avg_c_rate"] = float(np.mean(c_rates)) if c_rates else 0.0

    # Duration deviation: actual vs expected (4h for 70% delta at 0.25C)
    if durations:
        expected_min = 70 / 25 * 60 if nominal_kwh and nominal_kwh > 30 else 70 / 50 * 60
        out["duration_deviation"] = float(np.mean(durations) - expected_min)

    return out


# Alias for backwards-compatibility with tests
BatteryHVFeaturePipeline = HVBatteryFeaturePipeline


def _range_per_kwh_trend(d30: pd.DataFrame, nominal_kwh: float | None) -> float:
    """Slope of (km driven / kWh used) per discharge session over 30d."""
    if "soc" not in d30.columns or "odometer" not in d30.columns or nominal_kwh is None:
        return 0.0

    soc30 = d30["soc"].fillna(np.nan)
    odo30 = d30["odometer"].fillna(np.nan)

    discharge_flag = (soc30.diff() < -0.5).astype(int)
    sids = (discharge_flag.diff().fillna(0) > 0).cumsum() * discharge_flag

    ratios: list[float] = []
    for sid in sids.unique():
        if sid == 0:
            continue
        mask = sids == sid
        dsoc = float(soc30[mask].iloc[0] - soc30[mask].iloc[-1])
        dkm  = float(odo30[mask].iloc[-1] - odo30[mask].iloc[0])
        if dsoc > 5 and dkm > 0:
            kwh_used = dsoc / 100 * nominal_kwh
            ratios.append(dkm / kwh_used)

    return float(np.polyfit(np.arange(len(ratios)), ratios, 1)[0]) if len(ratios) >= 2 else 0.0
