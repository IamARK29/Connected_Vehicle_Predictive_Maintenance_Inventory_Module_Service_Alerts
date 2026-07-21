"""
EV Health Router — GET /api/ev/{vin}/health

Runs EVChargingFeatureEngine, EVMotorFeatureEngine, EVDCDCFeatureEngine,
and RangeAnxietyPredictor in a single call and returns structured component
health scores, per-feature values, threshold alerts, and range estimate.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import get_current_user

router = APIRouter(prefix="/ev", tags=["EV Health"])
log = logging.getLogger(__name__)
DATA_DIR = Path(os.getenv("DATA_DIR", "data/synthetic"))

EV_FUEL_TYPES = {"EV", "PHEV", "BEV"}
_SEVERITY_PENALTY = {"CRITICAL": 35, "HIGH": 20, "MEDIUM": 10}

# ── Feature metadata: key → (display label, unit, decimal places) ─────────────

_CHARGING_META: dict[str, tuple[str, str, int]] = {
    "charge_acceptance_ratio":       ("Charge Acceptance Ratio",    "",       3),
    "charge_acceptance_trend_30d":   ("Acceptance Trend (30d)",     "/day",   5),
    "charge_duration_deviation_pct": ("Duration Deviation",         "%",      1),
    "end_voltage_deficit_v":         ("End-Voltage Deficit",        "V",      1),
    "dc_fraction_30d":               ("DC Fast-Charge Fraction",    "",       3),
    "avg_soc_at_charge_start":       ("Avg SoC at Charge Start",   "%",      1),
    "avg_soc_after_overnight_ac":    ("Overnight AC End SoC",       "%",      1),
    "total_charge_sessions_30d":     ("Total Sessions (30d)",       "",       0),
    "dc_charge_sessions_30d":        ("DC Sessions (30d)",          "",       0),
}

_MOTOR_META: dict[str, tuple[str, str, int]] = {
    "torque_efficiency_mean_30d":  ("Torque Delivery Efficiency", "",      3),
    "inv_temp_max_30d":            ("Inverter Temp Peak",          "°C",    1),
    "inv_temp_mean_30d":           ("Inverter Temp Mean",          "°C",    1),
    "inv_temp_per_kw":             ("Inverter Temp / Output kW",   "°C/kW", 3),
    "stator_temp_max_30d":         ("Stator Temp Peak",            "°C",    1),
    "stator_temp_mean_30d":        ("Stator Temp Mean",            "°C",    1),
    "torque_ripple_proxy_nm":      ("Torque Ripple Proxy",         "Nm",    2),
    "motor_rpm_deviation_mean":    ("RPM Deviation (mean)",        "rpm",   0),
}

_DCDC_META: dict[str, tuple[str, str, int]] = {
    "dcdc_output_v_mean_30d":    ("Output Voltage Mean",         "V",  2),
    "dcdc_output_v_min_30d":     ("Output Voltage Minimum",      "V",  2),
    "dcdc_baseline_temp_c":      ("Under-Bonnet Baseline Temp",  "°C", 1),
    "dcdc_temp_max_30d":         ("Converter Temp Peak",         "°C", 1),
    "dcdc_temp_mean_30d":        ("Converter Temp Mean",         "°C", 1),
    "dcdc_temp_rise_max_c":      ("Temp Rise (Peak)",            "°C", 1),
    "dcdc_temp_rise_mean_c":     ("Temp Rise (Mean)",            "°C", 1),
    "high_load_voltage_droop_v": ("High-Load Voltage Droop",     "V",  3),
    "dcdc_startup_recovery_v":   ("Startup Voltage Recovery",    "V",  3),
    "dcdc_thermal_cycles_total": ("Thermal Cycles (30d)",        "",   0),
}


# ── Data helpers ──────────────────────────────────────────────────────────────

def _load_fleet_row(vin: str) -> dict | None:
    for name in ("fleet.csv", "fleet_master.csv"):
        p = DATA_DIR / name
        if p.exists():
            try:
                df = pd.read_csv(p)
                match = df[df["vin"] == vin]
                if not match.empty:
                    return match.iloc[0].to_dict()
            except Exception:
                pass
    return None


def _load_telemetry(vin: str) -> pd.DataFrame:
    for pattern in [f"telemetry_{vin}.csv", f"{vin}_telemetry.csv"]:
        p = DATA_DIR / pattern
        if p.exists():
            try:
                return pd.read_csv(p, nrows=5000)
            except Exception:
                pass
    p = DATA_DIR / "telemetry.csv"
    if p.exists():
        try:
            df = pd.read_csv(p, nrows=100_000)
            if "vin" in df.columns:
                return df[df["vin"] == vin].copy().reset_index(drop=True)
        except Exception:
            pass
    return pd.DataFrame()


def _synthesise_charge_sessions(telem: pd.DataFrame) -> pd.DataFrame:
    """
    Derive charge sessions from telemetry when no dedicated sessions CSV exists.
    Detects contiguous windows where SoC is rising; classifies DC if session
    duration < 60 min (proxy for fast charging).
    """
    if "vehBMSPackSOC" not in telem.columns:
        return pd.DataFrame()

    df = telem.copy()
    # vehBMSPackSOC is already decoded percent (0-100) in the pipeline
    df["soc"] = df["vehBMSPackSOC"]
    df["d_soc"] = df["soc"].diff().fillna(0)

    now = pd.Timestamp.now(tz="UTC")
    sessions: list[dict] = []
    in_charge = False
    seg_start = 0

    for idx in range(len(df)):
        rising = df["d_soc"].iloc[idx] > 0.5  # 0.5 pp threshold in physical % units
        if not in_charge and rising:
            in_charge = True
            seg_start = idx
        elif in_charge and not rising:
            seg = df.iloc[seg_start:idx]
            if len(seg) >= 3:
                duration_min = float(len(seg) * 5)
                soc_s = float(seg["soc"].iloc[0])
                soc_e = float(seg["soc"].iloc[-1])
                # HV pack end voltage (vehBMSPackVol, already decoded volts)
                end_v = float(seg["vehBMSPackVol"].iloc[-1]) if "vehBMSPackVol" in seg.columns else 400.0
                sessions.append({
                    "session_id":             len(sessions) + 1,
                    "start_ts":               now - pd.Timedelta(days=(len(sessions) + 1) * 2),
                    "end_ts":                 now - pd.Timedelta(days=(len(sessions) + 1) * 2, hours=-(duration_min / 60)),
                    "charge_type":            "DC" if duration_min < 60 else "AC",
                    "soc_start_pct":          soc_s,
                    "soc_end_pct":            soc_e,
                    "duration_min":           duration_min,
                    "energy_kwh":             max(0.0, (soc_e - soc_s) * 0.01 * 50.3),
                    "end_voltage_v":          end_v,
                    "expected_end_voltage_v": 408.0,
                })
            in_charge = False
            if len(sessions) >= 20:
                break

    return pd.DataFrame(sessions) if sessions else pd.DataFrame()


# ── Scoring helpers ───────────────────────────────────────────────────────────

def _compute_alerts(features: dict, thresholds: dict) -> list[dict]:
    """Extract threshold violations from a feature dict. Returns alert list only."""
    alerts: list[dict] = []
    for key, (severity, threshold, comparison) in thresholds.items():
        val = features.get(key)
        if val is None:
            continue
        try:
            triggered = (
                (comparison == "gt" and float(val) > threshold) or
                (comparison == "lt" and float(val) < threshold)
            )
        except (TypeError, ValueError):
            continue
        if triggered:
            alerts.append({
                "feature":    key,
                "severity":   severity,
                "value":      val,
                "threshold":  threshold,
                "comparison": comparison,
            })
    return alerts


def _apply_thresholds(
    features: dict, thresholds: dict
) -> tuple[int | None, list[dict]]:
    """Legacy helper: pure threshold-based score (still used for overall EV health)."""
    has_data = any(v is not None for v in features.values())
    if not has_data:
        return None, []
    alerts = _compute_alerts(features, thresholds)
    score = 100 - sum(_SEVERITY_PENALTY.get(a["severity"], 10) for a in alerts)
    return max(0, min(100, score)), alerts


def _score_dcdc(raw: dict, fleet_row: dict) -> tuple[int | None, list[dict]]:
    """
    Continuous DCDC health score.

    Combines odometer-based aging with telemetry headroom signals so that
    a healthy vehicle still shows gradual degradation over its lifetime
    (not just a binary 100 / alert-penalised score).

    Degradation model:
        -1 pt per 10,000 km  (DCDC unit life ~200,000 km)
        -0–12 pt based on output voltage proximity to alert threshold
        -0–10 pt based on mean temperature rise above 15°C margin
        -0–8  pt based on high-load voltage droop
        Alert penalties (half-weight, already partially captured above)
    """
    has_data = any(v is not None for v in raw.values())
    if not has_data:
        return None, []

    from features.ev_dcdc_features import ALERT_THRESHOLDS as DCDC_THR
    alerts = _compute_alerts(raw, DCDC_THR)

    odo   = float(fleet_row.get("initial_odometer", 0) or 0)
    score = max(40.0, 100.0 - odo / 10_000.0)

    v_mean = raw.get("dcdc_output_v_mean_30d")
    if v_mean is not None and float(v_mean) < 14.0:
        # Linear -12 pts as voltage drops from 14.0V to 13.5V (alert boundary)
        score -= max(0.0, (14.0 - float(v_mean)) / (14.0 - 13.5) * 12.0)

    temp_rise = raw.get("dcdc_temp_rise_mean_c")
    if temp_rise is not None and float(temp_rise) > 15.0:
        score -= min(10.0, (float(temp_rise) - 15.0) * 0.4)

    droop = raw.get("high_load_voltage_droop_v")
    if droop is not None and float(droop) > 0.15:
        score -= min(8.0, (float(droop) - 0.15) / 0.35 * 8.0)

    # Alert threshold crossings: half-weight (continuous bands already penalise above)
    for a in alerts:
        score -= _SEVERITY_PENALTY.get(a["severity"], 10) * 0.5

    return max(0, min(100, int(round(score)))), alerts


def _score_motor(raw: dict, fleet_row: dict) -> tuple[int | None, list[dict]]:
    """
    Continuous motor & inverter health score.

        -1 pt per 15,000 km  (motor life ~300,000 km)
        -0–20 pt from torque efficiency drop below 0.97 (if available)
        -0–12 pt from inverter peak temp above operating margin
        Alert penalties (half-weight)
    """
    has_data = any(v is not None for v in raw.values())
    if not has_data:
        return None, []

    from features.ev_motor_features import ALERT_THRESHOLDS as MOT_THR
    alerts = _compute_alerts(raw, MOT_THR)

    odo   = float(fleet_row.get("initial_odometer", 0) or 0)
    score = max(50.0, 100.0 - odo / 15_000.0)

    eff = raw.get("torque_efficiency_mean_30d")
    if eff is not None:
        # Perfect ≈ 0.97; alert ≤ 0.88. Linear 0–20 pt penalty.
        score -= max(0.0, (0.97 - float(eff)) / (0.97 - 0.88) * 20.0)

    # inv_temp_max_30d: pipeline stores decoded value (generator actual °C - 40).
    # Normal operating headroom: computed values up to ~60 (=100°C actual) are ok.
    inv_max = raw.get("inv_temp_max_30d")
    if inv_max is not None and float(inv_max) > 55.0:
        score -= min(12.0, (float(inv_max) - 55.0) / 45.0 * 12.0)

    for a in alerts:
        score -= _SEVERITY_PENALTY.get(a["severity"], 10) * 0.5

    return max(0, min(100, int(round(score)))), alerts


def _score_charging(raw: dict, fleet_row: dict) -> tuple[int | None, list[dict]]:
    """
    Continuous charging health score.

        -1 pt per 12,000 km  (acceptance degrades with HV battery aging)
        -0–15 pt from charge acceptance ratio below 0.95
        -0–8  pt from high DC fast-charge fraction (>30% DC stresses cells)
        Alert penalties (half-weight)
    """
    has_data = any(v is not None for v in raw.values())
    if not has_data:
        return None, []

    from features.ev_charging_features import ALERT_THRESHOLDS as CHG_THR
    alerts = _compute_alerts(raw, CHG_THR)

    odo   = float(fleet_row.get("initial_odometer", 0) or 0)
    score = max(50.0, 100.0 - odo / 12_000.0)

    ratio = raw.get("charge_acceptance_ratio")
    if ratio is not None:
        r = float(ratio)
        if r < 0.95:
            score -= min(15.0, (0.95 - r) / (0.95 - 0.80) * 15.0)

    dc_frac = raw.get("dc_fraction_30d")
    if dc_frac is not None and float(dc_frac) > 0.30:
        score -= min(8.0, (float(dc_frac) - 0.30) / 0.70 * 8.0)

    for a in alerts:
        score -= _SEVERITY_PENALTY.get(a["severity"], 10) * 0.5

    return max(0, min(100, int(round(score)))), alerts


def _status_label(score: int | None) -> str:
    if score is None: return "No Data"
    if score >= 85:   return "Good"
    if score >= 70:   return "Fair"
    if score >= 50:   return "Poor"
    return "Critical"


def _format_features(raw: dict, meta: dict) -> dict:
    out = {}
    for key, (label, unit, decimals) in meta.items():
        val = raw.get(key)
        out[key] = {
            "label": label,
            "value": round(float(val), decimals) if val is not None else None,
            "unit":  unit,
        }
    return out


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.get(
    "/{vin}/health",
    summary="EV powertrain health — charging, motor, DC-DC, and range estimate",
    response_description="Component health scores, feature values, threshold alerts, range",
)
async def get_ev_health(
    vin: str,
    current_user: Annotated[dict, Depends(get_current_user)],
    soc_pct: float | None = Query(None, ge=0, le=100, description="Current SoC override (%)"),
    outside_temp_c: float = Query(25.0, ge=-20, le=55, description="Ambient temperature (°C)"),
    ac_is_on: bool = Query(False, description="Cabin AC active"),
):
    fleet_row = _load_fleet_row(vin)
    if not fleet_row:
        raise HTTPException(status_code=404, detail=f"Vehicle {vin} not found")

    fuel_type = str(fleet_row.get("fuel_type", "")).upper()
    if fuel_type not in EV_FUEL_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"EV health monitoring requires an EV or PHEV. This VIN is {fuel_type}.",
        )

    # ── Load telemetry ────────────────────────────────────────────────────────
    telem = _load_telemetry(vin)
    if "vin" not in telem.columns and not telem.empty:
        telem["vin"] = vin
    if not telem.empty and "timestamp" not in telem.columns:
        telem["timestamp"] = pd.date_range(
            end=pd.Timestamp.now(tz="UTC"), periods=len(telem), freq="5min"
        )
    if not telem.empty:
        telem["timestamp"] = pd.to_datetime(telem["timestamp"], utc=True, errors="coerce")

    # ── Charging ──────────────────────────────────────────────────────────────
    from features.ev_charging_features import EVChargingFeatureEngine
    sessions_df = _synthesise_charge_sessions(telem)
    chg_raw = EVChargingFeatureEngine().compute(sessions_df, telem, vin, fleet_row)
    chg_score, chg_alerts = _score_charging(chg_raw, fleet_row)

    # ── Motor & Inverter ──────────────────────────────────────────────────────
    from features.ev_motor_features import EVMotorFeatureEngine
    mot_raw = EVMotorFeatureEngine().compute(telem, vin)
    mot_score, mot_alerts = _score_motor(mot_raw, fleet_row)

    # ── DC-DC Converter ───────────────────────────────────────────────────────
    from features.ev_dcdc_features import EVDCDCFeatureEngine
    dcd_raw = EVDCDCFeatureEngine().compute(telem, vin)
    dcd_score, dcd_alerts = _score_dcdc(dcd_raw, fleet_row)

    # ── Range Estimate ────────────────────────────────────────────────────────
    from models.range_anxiety_model import RangeAnxietyPredictor

    live_soc = soc_pct
    live_temp = outside_temp_c
    live_ac   = ac_is_on

    try:
        from twin.vehicle_twin import TwinManager
        twin = TwinManager().get(vin)
        if twin:
            if live_soc is None and twin.soc_pct:
                live_soc = twin.soc_pct
            live_temp = twin.outside_temp_c
            live_ac   = twin.ac_is_on
    except Exception:
        pass

    live_soc = float(live_soc or 60.0)

    fs_range: dict[str, Any] = {"composite_drive_score": 70.0, "km_per_day_30d_avg": 40.0}
    range_result = RangeAnxietyPredictor().predict(
        vin=vin,
        current_soc_pct=live_soc,
        current_outside_temp_c=float(live_temp),
        ac_is_on=bool(live_ac),
        feature_store_features=fs_range,
        fleet_row=fleet_row,
    )

    # ── Overall EV health (weighted: motor 40%, charging 35%, dcdc 25%) ───────
    component_scores = [
        (chg_score, 0.35),
        (mot_score, 0.40),
        (dcd_score, 0.25),
    ]
    valid = [(s, w) for s, w in component_scores if s is not None]
    if valid:
        total_w = sum(w for _, w in valid)
        overall = int(round(sum(s * w for s, w in valid) / total_w))
    else:
        overall = 0

    return {
        "vin":                     vin,
        "model_name":              str(fleet_row.get("model_name", "")),
        "fuel_type":               fuel_type,
        "rated_range_km":          float(fleet_row.get("rated_range_km", 0) or 0),
        "battery_capacity_kwh":    float(fleet_row.get("battery_capacity_kwh", 0) or 0),
        "overall_ev_health_score": overall,
        "overall_status":          _status_label(overall),
        "computed_at":             datetime.now(timezone.utc).isoformat(),
        "range": {
            **range_result,
            "current_soc_pct": live_soc,
            "outside_temp_c":  float(live_temp),
            "ac_is_on":        bool(live_ac),
        },
        "components": {
            "charging": {
                "health_score": chg_score,
                "status":       _status_label(chg_score),
                "features":     _format_features(chg_raw, _CHARGING_META),
                "alerts":       chg_alerts,
            },
            "motor": {
                "health_score": mot_score,
                "status":       _status_label(mot_score),
                "features":     _format_features(mot_raw, _MOTOR_META),
                "alerts":       mot_alerts,
            },
            "dcdc": {
                "health_score": dcd_score,
                "status":       _status_label(dcd_score),
                "features":     _format_features(dcd_raw, _DCDC_META),
                "alerts":       dcd_alerts,
            },
        },
    }
