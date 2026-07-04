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

        soc        = df["soc"].fillna(np.nan)         if "soc"           in df.columns else pd.Series(dtype=float)
        soh        = df["soh"].fillna(np.nan)         if "soh"           in df.columns else pd.Series(dtype=float)
        pack_vol   = df["bms_pack_vol"].fillna(np.nan) if "bms_pack_vol" in df.columns else pd.Series(dtype=float)
        pack_crnt  = df["bms_pack_crnt"].fillna(np.nan) if "bms_pack_crnt" in df.columns else pd.Series(dtype=float)
        cell_max_v = df["cell_max_vol"].fillna(np.nan) if "cell_max_vol"  in df.columns else pd.Series(dtype=float)
        cell_min_v = df["cell_min_vol"].fillna(np.nan) if "cell_min_vol"  in df.columns else pd.Series(dtype=float)
        cell_max_t = df["cell_max_temp"].fillna(np.nan) if "cell_max_temp" in df.columns else pd.Series(dtype=float)
        cell_min_t = df["cell_min_temp"].fillna(np.nan) if "cell_min_temp" in df.columns else pd.Series(dtype=float)

        # ── SoH estimation: try derived_utils Coulomb-counting first ─────
        soh_from_session = None
        if battery_capacity_kwh and len(df) >= 60:
            soh_from_session = compute_soh_from_charge_session(df, battery_capacity_kwh)
        soh_estimated = soh_from_session if soh_from_session is not None else \
            _estimate_soh_from_cycles(soc, pack_vol, pack_crnt, battery_capacity_kwh)

        # ── SoH trend over 90 days ────────────────────────────────────────
        soh_90 = d90["soh"].fillna(np.nan) if "soh" in d90.columns else pd.Series(dtype=float)
        soh_trend_slope_90d = self._slope(soh_90.resample("D", on=d90["timestamp"] if "timestamp" in d90.columns else None).mean() if False else soh_90)

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

        # ── Thermal fault count (30d) ──────────────────────────────────────
        thermal_fault_count_30d = 0
        if "cell_max_temp" in d30.columns:
            thermal_fault_count_30d = int((d30["cell_max_temp"].fillna(0) > 45).sum())

        # ── Isolation resistance min (30d) ─────────────────────────────────
        # Not in synthetic data; placeholder
        isolation_resistance_min_30d = 0.0

        # ── Range per kWh trend (30d) ─────────────────────────────────────
        range_per_kwh_30d_trend = _range_per_kwh_trend(d30, battery_capacity_kwh)

        # ── Simple aggregate features expected by tests ─────────────────
        soc7        = d7["soc"].fillna(np.nan) if "soc" in d7.columns else pd.Series(dtype=float)
        soc_mean_7d = float(soc7.mean())       if not soc7.dropna().empty else (float(soc.mean()) if not soc.dropna().empty else 0.0)
        soh_mean    = float(soh.dropna().mean()) if not soh.dropna().empty else (soh_estimated if soh_estimated is not None and not np.isnan(soh_estimated) else 0.0)

        # ── Current state ─────────────────────────────────────────────────
        soc_current = float(soc.iloc[-1]) if len(soc) > 0 else np.nan
        soh_current = float(soh.iloc[-1]) if len(soh) > 0 else np.nan

        # ── Labels ────────────────────────────────────────────────────────
        days_to_failure, within_90 = self._label_days_to_failure(
            vin, label_df, "hv_battery_degradation", t_max
        )
        soh_below_80_within_90 = int(
            within_90 or (not np.isnan(soh_current) and soh_current < 80)
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

    out["avg_c_rate"] = float(np.mean(c_rates)) if c_rates else np.nan

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
