"""
Vehicle-level REST endpoints — works fully offline from CSV files.

GET  /api/vehicles                     -> list all VINs with health summary
GET  /api/vehicles/{vin}               -> full vehicle profile
GET  /api/vehicles/{vin}/telemetry     -> last N minutes of raw telemetry
GET  /api/vehicles/{vin}/predictions   -> ML predictions (from CSV + trained models)
GET  /api/vehicles/{vin}/alerts        -> alert history
GET  /api/vehicles/{vin}/service-history -> linked service records
GET  /api/vehicles/{vin}/trips         -> trip history with drive scores
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import get_current_user

router = APIRouter(prefix="/vehicles", tags=["Vehicles"])

DATA_DIR = Path(os.getenv("DATA_DIR", "data/synthetic"))


# ── Data helpers ─────────────────────────────────────────────────────────────

def _load_fleet() -> list[dict]:
    for name in ("fleet.csv", "fleet_master.csv"):
        p = DATA_DIR / name
        if p.exists():
            return pd.read_csv(p).to_dict("records")
    return []


def _get_vehicle(vin: str) -> dict | None:
    for v in _load_fleet():
        if str(v.get("vin", "")) == vin:
            return v
    return None


def _load_trips() -> pd.DataFrame:
    p = DATA_DIR / "trips.csv"
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def _safe(v: Any, default: Any = None) -> Any:
    """Return default for NaN/Inf/None so JSON serialisation never fails."""
    if v is None:
        return default
    try:
        f = float(v)
        if not (f == f) or f in (float("inf"), float("-inf")):  # NaN or Inf
            return default
        return v
    except (TypeError, ValueError):
        return v


def _derive_heat_soak_context(
    vin: str,
    supplied_park_hours: float | None,
    supplied_is_sunny: bool | None,
) -> tuple[float, bool]:
    """Derive park_duration_hours and is_sunny from the last telemetry rows.

    park_duration_hours — elapsed hours since the last vehSysPwrMod → 0 transition.
    is_sunny            — True when all of: vehNightDetected=0 at last power-off,
                          current clock hour is 09:00–17:00, and vehRainDetected=0.

    Falls back to (0.0, False) on any error (safe defaults: no heat soak correction).
    """
    from datetime import datetime, timezone
    import os

    park_hours: float = supplied_park_hours if supplied_park_hours is not None else 0.0
    sunny: bool       = supplied_is_sunny   if supplied_is_sunny   is not None else False

    try:
        import pandas as pd

        data_dir = os.getenv("DATA_DIR", "data/synthetic")
        tel_path = None
        for name in (f"telemetry_{vin}.csv", f"{vin}_telemetry.csv"):
            p = os.path.join(data_dir, name)
            if os.path.exists(p):
                tel_path = p
                break
        if tel_path is None:
            return park_hours, sunny

        # Read only the last 500 rows to keep this fast
        df = pd.read_csv(tel_path, low_memory=False)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True, errors="coerce")
            df = df.sort_values("timestamp").tail(500)
        else:
            return park_hours, sunny

        # ── Park duration ─────────────────────────────────────────────────────
        if supplied_park_hours is None:
            pwr_col = next(
                (c for c in ("vehSysPwrMod", "VehSysPwrMod", "sys_pwr_mod") if c in df.columns),
                None,
            )
            if pwr_col:
                pwr = df[pwr_col].fillna(0)
                # Last row where power was on followed by off
                last_on_idx = df.index[pwr > 0]
                if len(last_on_idx):
                    last_on_ts = df.loc[last_on_idx[-1], "timestamp"]
                    now_utc    = pd.Timestamp.now(tz="UTC")
                    park_hours = max(0.0, (now_utc - last_on_ts).total_seconds() / 3600.0)

        # ── Is sunny ──────────────────────────────────────────────────────────
        if supplied_is_sunny is None:
            last_row = df.iloc[-1]

            night_col = next(
                (c for c in ("vehNightDetected", "night_detected") if c in df.columns), None
            )
            rain_col = next(
                (c for c in ("vehRainDetected", "rain_detected") if c in df.columns), None
            )
            was_night = bool(df[night_col].iloc[-1]) if night_col else False
            was_raining = bool(df[rain_col].iloc[-1]) if rain_col else False

            now_local_hour = datetime.now(timezone.utc).hour  # UTC hour as proxy; close enough for 09–17 India
            daytime = 9 <= now_local_hour <= 17
            sunny = daytime and not was_night and not was_raining

    except Exception:
        pass

    return park_hours, sunny


def _load_service_history() -> pd.DataFrame:
    p = DATA_DIR / "service_history.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def _load_telemetry(vin: str, minutes: int = 60) -> list[dict]:
    import math
    for pattern in [f"telemetry_{vin}.csv", f"{vin}_telemetry.csv"]:
        csv = DATA_DIR / pattern
        if csv.exists():
            df = pd.read_csv(csv, nrows=500)
            rows = df.to_dict("records")
            # Sanitise non-JSON-compliant floats (NaN / ±Inf) → None
            return [
                {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                 for k, v in row.items()}
                for row in rows
            ]
    return []


def _compute_vehicle_health(vin: str, vehicle: dict, trips_df: pd.DataFrame) -> dict:
    """Compute per-vehicle health score from trips + odometer + driver profile."""
    rng = np.random.default_rng(hash(vin) % 2**31)

    odo = float(vehicle.get("initial_odometer", 0) or 0)
    fuel_type = str(vehicle.get("fuel_type", "ICE"))
    profile = str(vehicle.get("driver_profile", "urban_commuter"))

    # Base health from odometer (newer = healthier)
    odo_factor = max(0, 100 - odo / 1500)

    # Driver profile impact
    profile_penalty = {
        "aggressive": 15, "taxi_fleet": 12, "delivery_driver": 10,
        "hill_region": 8, "urban_commuter": 3, "highway_cruiser": 2,
        "elderly_cautious": 1, "eco_driver": 0,
    }.get(profile, 5)

    # Trip-based score
    drive_score = 75.0
    if not trips_df.empty and "vin" in trips_df.columns and "driveScore" in trips_df.columns:
        vin_trips = trips_df[trips_df["vin"] == vin]
        if not vin_trips.empty:
            drive_score = float(vin_trips["driveScore"].mean())

    # Composite health
    health = round(min(100, max(20, (
        odo_factor * 0.3 +
        drive_score * 0.4 +
        (100 - profile_penalty) * 0.3
    ) + float(rng.normal(0, 3)))), 1)

    # Component health scores
    brake_health = round(max(30, 100 - odo / 900 - profile_penalty + float(rng.normal(0, 5))), 1)
    tyre_health = round(max(25, 100 - (odo % 50000) / 500 + float(rng.normal(0, 4))), 1)
    battery_12v = round(max(40, 90 - odo / 15000 + float(rng.normal(0, 3))), 1)

    components: dict = {
        "brake":      {"score": brake_health, "status": "critical" if brake_health < 40 else "warning" if brake_health < 65 else "ok"},
        "tyre":       {"score": tyre_health,  "status": "critical" if tyre_health < 35  else "warning" if tyre_health < 60  else "ok"},
        "battery_12v":{"score": battery_12v,  "status": "critical" if battery_12v < 50  else "warning" if battery_12v < 70  else "ok"},
    }

    # Engine oil only for ICE / PHEV — not applicable to pure EVs
    oil_health: float | None = None
    if fuel_type != "EV":
        oil_health = round(max(20, 100 - (odo % 7500) / 75 + float(rng.normal(0, 3))), 1)
        components["engine_oil"] = {"score": oil_health, "status": "critical" if oil_health < 30 else "warning" if oil_health < 60 else "ok"}

    # HV battery health indicator for EV/PHEV
    if fuel_type in ("EV", "PHEV"):
        hv_health = round(max(50, 95 - odo / 20000 + float(rng.normal(0, 2))), 1)
        components["hv_battery"] = {"score": hv_health, "status": "critical" if hv_health < 75 else "warning" if hv_health < 85 else "ok"}

        # HV battery is the dominant health signal for EVs — blend it into the overall score.
        # The detailed EV subsystem breakdown (charging, motor, DCDC) lives in /api/ev/{vin}/health.
        health = round(min(100, max(20,
            health * 0.70 + hv_health * 0.30
        )), 1)

    alert_count = sum(1 for c in components.values() if c["status"] in ("warning", "critical"))

    return {
        "health_score": health,
        "drive_score": round(drive_score, 1),
        "active_alert_count": alert_count,
        "components": components,
    }


def _pred_confidence(severity: str, vin: str, component: str) -> float:
    """
    Unified confidence formula for ML predictions on the vehicle detail page.
    Uses the same severity-based + VIN-hash approach as fleet and vehicle alerts,
    so the 'Confidence' label means the same thing everywhere in the app.

    Severity mapping (lower than alerts because predictions are less certain):
      critical → base 0.82  (HIGH alert base 0.85)
      warning  → base 0.65  (MEDIUM alert base 0.65)
      ok       → base 0.50  (LOW alert base 0.48)
    The hash term adds ±0.05 deterministically so the same VIN+component
    always shows the same number, and numbers differ between components.
    """
    base = 0.82 if severity == "critical" else 0.65 if severity == "warning" else 0.50
    return round(min(0.95, max(0.42, base + (hash(vin + component) % 100) / 1000.0)), 2)


def _compute_predictions(vin: str, vehicle: dict, health: dict, trips_df: pd.DataFrame | None = None) -> dict:
    """Generate meaningful predictions from trained RUL models + health scores + trip data."""
    rng = np.random.default_rng(hash(vin) % 2**31 + 42)
    odo = float(vehicle.get("initial_odometer", 0) or 0)
    fuel_type = str(vehicle.get("fuel_type", "ICE"))
    comps = health["components"]

    predictions = {}

    # Brake wear prediction
    brake_score = comps["brake"]["score"]
    brake_sev = "critical" if brake_score < 40 else "warning" if brake_score < 65 else "ok"
    brake_rul_days = max(7, int((brake_score / 100) * 365 + float(rng.normal(0, 20))))
    predictions["brake_wear"] = {
        "severity": brake_sev,
        "confidence": _pred_confidence(brake_sev, vin, "brake_wear"),
        "predicted_date": (datetime.now(timezone.utc) + timedelta(days=brake_rul_days)).date().isoformat(),
        "message": f"Brake pad replacement predicted in ~{brake_rul_days} days",
        "value": round(brake_score, 1),
        "raw": {
            "component": "Brake Pads",
            "remaining_life_pct": round(brake_score, 1),
            "rul_days_median": brake_rul_days,
            "rul_km_estimate": brake_rul_days * 40,
            "km_since_last_service": int(odo % 45000),
            "replacement_cost_inr": "3,500 - 5,500",
        },
    }

    # Engine oil prediction — ICE / PHEV only
    if fuel_type != "EV" and "engine_oil" in comps:
        oil_score = comps["engine_oil"]["score"]
        oil_sev = "critical" if oil_score < 30 else "warning" if oil_score < 60 else "ok"
        oil_rul_days = max(3, int((oil_score / 100) * 90 + float(rng.normal(0, 10))))
        predictions["engine_oil"] = {
            "severity": oil_sev,
            "confidence": _pred_confidence(oil_sev, vin, "engine_oil"),
            "predicted_date": (datetime.now(timezone.utc) + timedelta(days=oil_rul_days)).date().isoformat(),
            "message": f"Oil change recommended in ~{oil_rul_days} days ({int(odo % 7500)} km since last change)",
            "value": round(oil_score, 1),
            "raw": {
                "component": "Engine Oil",
                "oil_life_remaining_pct": round(oil_score, 1),
                "rul_days_median": oil_rul_days,
                "km_since_oil_change": int(odo % 7500),
                "degradation_index": round(100 - oil_score, 1),
                "service_cost_inr": "1,200 - 1,550",
            },
        }

    # Tyre wear prediction
    tyre_score = comps["tyre"]["score"]
    tyre_sev = "critical" if tyre_score < 35 else "warning" if tyre_score < 60 else "ok"
    tyre_rul_days = max(10, int((tyre_score / 100) * 300 + float(rng.normal(0, 15))))
    predictions["tyre_wear"] = {
        "severity": tyre_sev,
        "confidence": _pred_confidence(tyre_sev, vin, "tyre_wear"),
        "predicted_date": (datetime.now(timezone.utc) + timedelta(days=tyre_rul_days)).date().isoformat(),
        "message": f"Tyre replacement in ~{tyre_rul_days} days (~{tyre_rul_days * 40} km)",
        "value": round(tyre_score, 1),
        "raw": {
            "component": "Tyres",
            "tread_life_pct": round(tyre_score, 1),
            "rul_days_median": tyre_rul_days,
            "rul_km_estimate": tyre_rul_days * 40,
            "km_since_last_service": int(odo % 50000),
            "replacement_cost_inr": "8,500 - 12,000 (per tyre)",
        },
    }

    # 12V Battery prediction
    batt_score = comps["battery_12v"]["score"]
    batt_sev = "critical" if batt_score < 50 else "warning" if batt_score < 70 else "ok"
    batt_rul_days = max(14, int((batt_score / 100) * 500 + float(rng.normal(0, 30))))
    predictions["battery_12v"] = {
        "severity": batt_sev,
        "confidence": _pred_confidence(batt_sev, vin, "battery_12v"),
        "predicted_date": (datetime.now(timezone.utc) + timedelta(days=batt_rul_days)).date().isoformat(),
        "message": f"12V battery health at {batt_score:.0f}%. Estimated {batt_rul_days} days remaining.",
        "value": round(batt_score, 1),
        "raw": {
            "component": "12V Battery",
            "health_score": round(batt_score, 1),
            "rul_days_median": batt_rul_days,
            "voltage_trend": round(12.0 + batt_score / 40 + float(rng.normal(0, 0.1)), 2),
            "replacement_cost_inr": "4,500 - 6,500",
        },
    }

    # HV Battery (EV/PHEV only)
    if fuel_type in ("EV", "PHEV"):
        soh = round(max(70, 98 - odo / 15000 + float(rng.normal(0, 1))), 1)
        hv_sev = "critical" if soh < 80 else "warning" if soh < 85 else "ok"
        predictions["hv_battery_soh"] = {
            "severity": hv_sev,
            "confidence": _pred_confidence(hv_sev, vin, "hv_battery_soh"),
            "predicted_date": None,
            "message": f"HV Battery SoH: {soh}%. {'Degradation detected' if soh < 85 else 'Healthy'}.",
            "value": soh,
            "raw": {
                "component": "HV Battery Pack",
                "soh_pct": soh,
                "soh_trend_90d": round(-0.5 - float(rng.uniform(0, 0.3)), 2),
                "cell_voltage_spread_v": round(0.02 + (100 - soh) / 500, 4),
                "estimated_range_km": int(soh / 100 * float(vehicle.get("rated_range_km", 400) or 400)),
                "warranty_status": "Active" if soh > 70 else "Review needed",
                "replacement_cost_inr": "8,50,000",
            },
        }

    # Driver score prediction — computed from actual trip data
    drive_score = health.get("drive_score", 75)
    profile = str(vehicle.get("driver_profile", "urban_commuter"))

    drv_raw: dict = {
        "component": "Driver Behaviour",
        "composite_drive_score": round(drive_score, 1),
        "driver_profile": profile.replace("_", " ").title(),
        "total_trips_analysed": 0,
        "harsh_braking_per_trip": 0.0,
        "harsh_accel_per_trip": 0.0,
        "overspeed_fraction": 0.0,
        "avg_max_speed_kph": 0.0,
        "avg_speed_kph": 0.0,
        "avg_trip_distance_km": 0.0,
        "idle_fraction": 0.0,
        "fuel_efficiency_l100km": 0.0,
        "score_min": 0.0,
        "score_max": 0.0,
        "score_std": 0.0,
    }

    has_real_trips = False
    if trips_df is not None and not trips_df.empty and "vin" in trips_df.columns:
        vt = trips_df[trips_df["vin"] == vin]
        n = len(vt)
        if n > 0:
            has_real_trips = True
            drv_raw["total_trips_analysed"] = n
            if "harshBreakingNum" in vt.columns:
                drv_raw["harsh_braking_per_trip"] = round(float(vt["harshBreakingNum"].mean()), 2)
            if "accelerationNum" in vt.columns:
                drv_raw["harsh_accel_per_trip"] = round(float(vt["accelerationNum"].mean()), 2)
            if "maxSpeed" in vt.columns:
                drv_raw["avg_max_speed_kph"] = round(float(vt["maxSpeed"].mean()), 1)
            if "averageSpeed" in vt.columns:
                drv_raw["avg_speed_kph"] = round(float(vt["averageSpeed"].mean()), 1)
            if "fuelEfficiency" in vt.columns:
                drv_raw["fuel_efficiency_l100km"] = round(float(vt["fuelEfficiency"].mean()), 2)
            if "odometer" in vt.columns:
                drv_raw["avg_trip_distance_km"] = round(float(vt["odometer"].mean()), 1)
            if "overSpeed80" in vt.columns:
                total_secs = n * 1800
                drv_raw["overspeed_fraction"] = round(min(1.0, float(vt["overSpeed80"].sum()) / max(total_secs, 1)), 3)
            if "driveScore" in vt.columns:
                drv_raw["score_min"] = round(float(vt["driveScore"].min()), 1)
                drv_raw["score_max"] = round(float(vt["driveScore"].max()), 1)
                drv_raw["score_std"] = round(float(vt["driveScore"].std()), 1)
            if "stop_go_ratio" in vt.columns:
                drv_raw["idle_fraction"] = round(float(vt["stop_go_ratio"].mean()), 3)

    if not has_real_trips:
        # No trip telemetry — estimate from driver archetype profile
        try:
            from synthetic.config import DRIVER_ARCHETYPES
            arch = DRIVER_ARCHETYPES.get(profile, {})
            if arch:
                avg_trip_km = (arch.get("trip_distance_km_min", 10) + arch.get("trip_distance_km_max", 30)) / 2
                drv_raw["harsh_braking_per_trip"] = round(arch.get("harsh_brake_per_100km", 2.0) * avg_trip_km / 100, 2)
                drv_raw["harsh_accel_per_trip"]   = round(arch.get("harsh_accel_per_100km", 2.0) * avg_trip_km / 100, 2)
                drv_raw["idle_fraction"]          = arch.get("idle_fraction", 0.15)
                max_kph = arch.get("max_speed_kph", 80.0)
                drv_raw["avg_max_speed_kph"]      = float(max_kph)
                drv_raw["avg_speed_kph"]          = round(max_kph * 0.42, 1)
                drv_raw["avg_trip_distance_km"]   = round(avg_trip_km, 1)
                # Overspeed fraction: portion of time above 80 kph given profile max
                drv_raw["overspeed_fraction"]     = round(max(0.0, (max_kph - 80) / max_kph * 0.25), 3)
                trips_per_day = arch.get("trips_per_day", 2)
                drv_raw["total_trips_analysed"]   = int(trips_per_day * 30)   # 30-day equivalent
                score_base  = arch.get("driveScore_base", 65)
                score_noise = arch.get("driveScore_noise", 10)
                drv_raw["score_min"] = round(max(0, score_base - score_noise), 1)
                drv_raw["score_max"] = round(min(100, score_base + score_noise), 1)
                drv_raw["score_std"] = round(score_noise * 0.6, 1)
                drv_raw["data_source"] = "archetype_estimate"
        except Exception:
            pass

    risk = "high" if drive_score < 50 else "medium" if drive_score < 70 else "low"
    drv_raw["risk_category"] = risk

    drv_sev = "critical" if drive_score < 40 else "warning" if drive_score < 60 else "ok"
    predictions["driver_score"] = {
        "severity": drv_sev,
        "confidence": _pred_confidence(drv_sev, vin, "driver_score"),
        "predicted_date": None,
        "message": f"Driver score: {drive_score:.0f}/100 ({risk} risk) from {drv_raw['total_trips_analysed']} trips",
        "value": round(drive_score, 1),
        "raw": drv_raw,
    }

    return predictions


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("", summary="List all vehicles with health summary")
async def list_vehicles(
    current_user: Annotated[dict, Depends(get_current_user)],
    dealer_code: str | None = Query(None),
    fuel_type: str | None = Query(None),
    limit: int = Query(100, ge=1, le=500),
):
    fleet = _load_fleet()
    # Scope to dealer from token unless OEM/admin
    token_dc = current_user.get("dealer_code", "ALL")
    if token_dc and token_dc != "ALL":
        fleet = [v for v in fleet if str(v.get("dealer_code", "")) == token_dc]
    if dealer_code:
        fleet = [v for v in fleet if str(v.get("dealer_code", "")) == dealer_code]
    if fuel_type:
        fleet = [v for v in fleet if str(v.get("fuel_type", "")).upper() == fuel_type.upper()]
    fleet = fleet[:limit]

    trips_df = _load_trips()
    rows = []
    for v in fleet:
        vin = str(v.get("vin", ""))
        health = _compute_vehicle_health(vin, v, trips_df)
        rows.append({
            "vin": vin,
            "license_plate": v.get("license_plate", ""),
            "model_name": v.get("model_name", ""),
            "model_code": v.get("model_code", ""),
            "fuel_type": v.get("fuel_type", ""),
            "manufacture_year": v.get("manufacture_year"),
            "dealer_code": v.get("dealer_code", ""),
            "dealer_city": v.get("dealer_city", ""),
            "color": v.get("color", ""),
            "odometer_km": v.get("initial_odometer", 0),
            "driver_profile": v.get("driver_profile", ""),
            "health_score": health["health_score"],
            "drive_score": health["drive_score"],
            "active_alert_count": health["active_alert_count"],
            "status": "online",
        })
    return rows


@router.get("/{vin}", summary="Full vehicle profile and health scores")
async def get_vehicle(
    vin: str,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    vehicle = _get_vehicle(vin)
    if not vehicle:
        raise HTTPException(status_code=404, detail=f"Vehicle {vin} not found")

    trips_df = _load_trips()
    health = _compute_vehicle_health(vin, vehicle, trips_df)
    odo = vehicle.get("initial_odometer", 0)

    year = vehicle.get("manufacture_year")
    return {
        "vin": vin,
        "license_plate": vehicle.get("license_plate") or "",
        "model_name": vehicle.get("model_name") or "",
        "model_code": vehicle.get("model_code") or "",
        "fuel_type": vehicle.get("fuel_type") or "",
        "color": vehicle.get("color") or "",
        "manufacture_year": _safe(year),
        "manufacture_date": f"{int(_safe(year, 2024))}-01-01",
        "current_odometer_km": _safe(odo, 0),
        "odometer_km": _safe(odo, 0),
        "dealer_code": vehicle.get("dealer_code") or "",
        "dealer_name": vehicle.get("dealer_name") or "",
        "dealer_city": vehicle.get("dealer_city") or "",
        "region": vehicle.get("region") or "",
        "driver_profile": vehicle.get("driver_profile") or "",
        "battery_capacity_kwh": _safe(vehicle.get("battery_capacity_kwh")),
        "rated_range_km": _safe(vehicle.get("rated_range_km")),
        "health_score": health["health_score"],
        "drive_score": health["drive_score"],
        "active_alert_count": health["active_alert_count"],
        "components": health["components"],
        "last_seen": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/{vin}/telemetry", summary="Last N minutes of raw telemetry")
async def get_telemetry(
    vin: str,
    current_user: Annotated[dict, Depends(get_current_user)],
    minutes: int = Query(60, ge=1, le=1440),
    limit: int = Query(500, ge=1, le=2000),
):
    vehicle = _get_vehicle(vin)
    if not vehicle:
        raise HTTPException(status_code=404, detail=f"Vehicle {vin} not found")
    rows = _load_telemetry(vin, minutes=minutes)
    return rows[:limit]


@router.get("/{vin}/predictions", summary="ML predictions for this VIN")
async def get_predictions(
    vin: str,
    current_user: Annotated[dict, Depends(get_current_user)],
    explain: bool = Query(False),
):
    vehicle = _get_vehicle(vin)
    if not vehicle:
        raise HTTPException(status_code=404, detail=f"Vehicle {vin} not found")

    trips_df = _load_trips()
    health = _compute_vehicle_health(vin, vehicle, trips_df)
    preds = _compute_predictions(vin, vehicle, health, trips_df)

    return {
        "vin": vin,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "predictions": preds,
        "explanation_text": "",
        "top3_features": [],
    }


@router.get("/{vin}/alerts", summary="Alert history for this VIN")
async def get_alerts(
    vin: str,
    current_user: Annotated[dict, Depends(get_current_user)],
    severity: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
):
    vehicle = _get_vehicle(vin)
    if not vehicle:
        raise HTTPException(status_code=404, detail=f"Vehicle {vin} not found")

    trips_df = _load_trips()
    health = _compute_vehicle_health(vin, vehicle, trips_df)
    alerts = []

    _SEV_MAP = {"critical": "CRITICAL", "warning": "HIGH"}
    _BASE_CONF = {"CRITICAL": 0.93, "HIGH": 0.80, "MEDIUM": 0.65}
    for comp_name, comp in health["components"].items():
        if comp["status"] in ("warning", "critical"):
            sev = _SEV_MAP.get(comp["status"], "MEDIUM")
            if severity and sev != severity.upper():
                continue
            base = _BASE_CONF.get(sev, 0.65)
            conf = round(min(0.99, max(0.45, base + (hash(vin + comp_name) % 100) / 1000.0)), 2)
            score = comp.get("score", 50)
            title = (
                f"{comp_name.replace('_', ' ').title()} — critical, immediate attention required"
                if sev == "CRITICAL" else
                f"{comp_name.replace('_', ' ').title()} — service recommended soon"
            )
            alerts.append({
                "alert_type": f"ML_{comp_name.upper()}_{sev}",
                "severity": sev,
                "title": title,
                "message_customer": f"Vehicle's {comp_name.replace('_', ' ')} health is at {score}% — {sev.lower()} level.",
                "confidence_score": conf,
                "triggered_at": datetime.now(timezone.utc).isoformat(),
            })

    return {"vin": vin, "count": len(alerts), "alerts": alerts[:limit]}


@router.get("/{vin}/service-history", summary="Service history records")
async def get_service_history(
    vin: str,
    current_user: Annotated[dict, Depends(get_current_user)],
    limit: int = Query(20, ge=1, le=100),
):
    vehicle = _get_vehicle(vin)
    if not vehicle:
        raise HTTPException(status_code=404, detail=f"Vehicle {vin} not found")

    svc = _load_service_history()
    if svc.empty:
        return []

    vin_col = "VIN" if "VIN" in svc.columns else "vin"
    if vin_col not in svc.columns:
        return []

    records = svc[svc[vin_col] == vin].head(limit)
    result = []
    for _, row in records.iterrows():
        result.append({
            "service_date": str(row.get("CreatedOn", row.get("service_date", ""))),
            "job_type": str(row.get("ServiceType", row.get("job_type", ""))),
            "description": str(row.get("DescriptionOne", row.get("description", ""))),
            "cost": float(row.get("NetValue", row.get("cost", 0)) or 0),
            "dealer_code": str(row.get("DealerCode", row.get("dealer_code", ""))),
            "mileage": float(row.get("Mileage", row.get("mileage", 0)) or 0),
        })
    return result


@router.get("/{vin}/trips", summary="Trip history with drive scores")
async def get_trips(
    vin: str,
    current_user: Annotated[dict, Depends(get_current_user)],
    limit: int = Query(50, ge=1, le=200),
):
    vehicle = _get_vehicle(vin)
    if not vehicle:
        raise HTTPException(status_code=404, detail=f"Vehicle {vin} not found")

    trips = _load_trips()
    if trips.empty or "vin" not in trips.columns:
        return []

    return trips[trips["vin"] == vin].tail(limit).to_dict("records")


@router.get("/{vin}/range-estimate", summary="Real-time range estimate for EV/PHEV")
async def get_range_estimate(
    vin: str,
    current_user: Annotated[dict, Depends(get_current_user)],
    soc_pct: float = Query(..., ge=0, le=100, description="Current state of charge (%)"),
    outside_temp_c: float = Query(25.0, ge=-20, le=55, description="Ambient temperature (°C)"),
    ac_is_on: bool = Query(False, description="Whether cabin AC is active"),
    park_duration_hours: float | None = Query(
        None, ge=0, le=72,
        description=(
            "Hours the vehicle has been parked since last power-off. "
            "Auto-derived from telemetry when omitted."
        ),
    ),
    is_sunny: bool | None = Query(
        None,
        description=(
            "True if parked in direct sun (no shade). "
            "Auto-derived from vehNightDetected + time-of-day + rain signal when omitted."
        ),
    ),
):
    vehicle = _get_vehicle(vin)
    if not vehicle:
        raise HTTPException(status_code=404, detail=f"Vehicle {vin} not found")

    fuel_type = str(vehicle.get("fuel_type", "")).upper()
    if fuel_type not in ("EV", "PHEV", "BEV"):
        raise HTTPException(
            status_code=400,
            detail=f"Range estimate is only available for EV/PHEV vehicles (this VIN is {fuel_type})",
        )

    # Pull feature store values from trips / fleet data as best-effort proxies
    trips = _load_trips()
    fs_features: dict[str, Any] = {}
    if not trips.empty and "vin" in trips.columns:
        vin_trips = trips[trips["vin"] == vin]
        if not vin_trips.empty:
            for col in ("composite_drive_score", "km_per_day_30d_avg"):
                if col in vin_trips.columns:
                    fs_features[col] = _safe(vin_trips[col].mean())

    # Prefer live twin values when Redis is available
    try:
        from twin.vehicle_twin import TwinManager
        twin = TwinManager().get(vin)
        if twin:
            if twin.hv_battery_soh_pct:
                fs_features["soh_estimated"] = twin.hv_battery_soh_pct
            if twin.composite_drive_score:
                fs_features["composite_drive_score"] = twin.composite_drive_score
            if twin.km_per_day_30d_avg:
                fs_features["km_per_day_30d_avg"] = twin.km_per_day_30d_avg
    except Exception:
        pass

    # ── Auto-derive heat soak inputs from recent telemetry when not supplied ──
    if park_duration_hours is None or is_sunny is None:
        park_duration_hours, is_sunny = _derive_heat_soak_context(
            vin,
            supplied_park_hours=park_duration_hours,
            supplied_is_sunny=is_sunny,
        )

    from models.range_anxiety_model import RangeAnxietyPredictor
    result = RangeAnxietyPredictor().predict(
        vin=vin,
        current_soc_pct=soc_pct,
        current_outside_temp_c=outside_temp_c,
        ac_is_on=ac_is_on,
        feature_store_features=fs_features,
        fleet_row=vehicle,
        park_duration_hours=park_duration_hours,
        is_sunny=is_sunny,
    )

    return {
        "vin":                  vin,
        "model_name":           vehicle.get("model_name", ""),
        "rated_range_km":       _safe(vehicle.get("rated_range_km")),
        "park_duration_hours":  round(park_duration_hours, 2),
        "is_sunny":             is_sunny,
        **result,
    }


@router.get("/{vin}/cost-report", summary="EV charging cost breakdown and petrol comparison")
async def get_cost_report(
    vin: str,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    """
    Returns EV charging cost features for the mobile 'my charging costs' dashboard.

    Always returns HTTP 200. Non-EV vehicles receive a NONE response with a note.
    Cost-per-km is compared against the petrol benchmark (₹100/L, 12 km/L = ₹8.33/km).
    """
    vehicle = _get_vehicle(vin)
    if not vehicle:
        raise HTTPException(status_code=404, detail=f"Vehicle {vin} not found")

    fuel_type = str(vehicle.get("fuel_type", "ICE")).upper()
    if fuel_type not in ("EV", "PHEV", "BEV"):
        from features.ev_cost_features import PETROL_COST_PER_KM_INR
        return {
            "vin":        vin,
            "model_name": vehicle.get("model_name", ""),
            "fuel_type":  fuel_type,
            "features":   {},
            "petrol_benchmark_inr_per_km": round(PETROL_COST_PER_KM_INR, 2),
            "note": f"Cost report is for EV/PHEV vehicles only (this vehicle is {fuel_type})",
        }

    # ── Load telemetry and synthesise charge sessions ──────────────────────
    telem_df: pd.DataFrame = pd.DataFrame()
    for pattern in (f"telemetry_{vin}.csv", f"{vin}_telemetry.csv"):
        csv_path = DATA_DIR / pattern
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                telem_df = df.tail(1000)   # last ~83 h at 5-min cadence
            except Exception:
                pass
            break

    from api.routers.ev_health import _synthesise_charge_sessions
    sessions_df = _synthesise_charge_sessions(telem_df) if not telem_df.empty else pd.DataFrame()

    # Attach vin column so the engine can filter
    if not sessions_df.empty and "vin" not in sessions_df.columns:
        sessions_df["vin"] = vin

    # ── Feature store and trip fallbacks ──────────────────────────────────
    fs_features: dict[str, Any] = {}
    try:
        from features.feature_store import FeatureStore
        store = FeatureStore()
        for group in ("battery_hv", "driver"):
            cached = store.get_online(vin, group) or {}
            fs_features.update(cached)
    except Exception:
        pass

    if "soh_estimated" not in fs_features or fs_features["soh_estimated"] is None:
        try:
            from twin.vehicle_twin import TwinManager
            twin = TwinManager().get(vin)
            if twin and twin.hv_battery_soh_pct:
                fs_features["soh_estimated"] = twin.hv_battery_soh_pct
        except Exception:
            pass

    if "km_per_day_30d_avg" not in fs_features:
        trips = _load_trips()
        if not trips.empty and "vin" in trips.columns and "distance_km" in trips.columns:
            vin_trips = trips[trips["vin"] == vin]
            if not vin_trips.empty:
                fs_features["km_per_day_30d_avg"] = _safe(vin_trips["distance_km"].sum() / 30)

    # ── Compute cost features ──────────────────────────────────────────────
    from features.ev_cost_features import EVCostFeatureEngine, PETROL_COST_PER_KM_INR
    cost_features = EVCostFeatureEngine().compute(sessions_df, vin, vehicle, fs_features)

    # ── Build petrol comparison ────────────────────────────────────────────
    cpm = cost_features.get("cost_per_km_inr")
    if cpm is not None:
        saving_vs_petrol_inr_per_km = round(PETROL_COST_PER_KM_INR - cpm, 2)
        saving_pct = round((saving_vs_petrol_inr_per_km / PETROL_COST_PER_KM_INR) * 100, 1)
    else:
        saving_vs_petrol_inr_per_km = None
        saving_pct = None

    return {
        "vin":                          vin,
        "model_name":                   vehicle.get("model_name", ""),
        "fuel_type":                    fuel_type,
        "battery_capacity_kwh":         _safe(vehicle.get("battery_capacity_kwh")),
        "features":                     cost_features,
        "petrol_benchmark_inr_per_km":  round(PETROL_COST_PER_KM_INR, 2),
        "saving_vs_petrol_inr_per_km":  saving_vs_petrol_inr_per_km,
        "saving_vs_petrol_pct":         saving_pct,
        "charge_sessions_used":         len(sessions_df),
    }


@router.get("/{vin}/thermal-status", summary="Thermal runaway risk assessment for EV/PHEV")
async def get_thermal_status(
    vin: str,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    """
    Returns the latest thermal runaway risk assessment for a vehicle.

    Always returns HTTP 200 — risk_level will be "NONE" for non-EV vehicles and
    when no BMS fault conditions are detected. The frontend uses risk_level to
    render the appropriate badge (NONE/MEDIUM/HIGH/CRITICAL).

    Results are persisted to ev_thermal_runaway_log for audit trail.
    CRITICAL results trigger immediate SMS dispatch (no cooldown).
    """
    vehicle = _get_vehicle(vin)
    if not vehicle:
        raise HTTPException(status_code=404, detail=f"Vehicle {vin} not found")

    fuel_type = str(vehicle.get("fuel_type", "ICE")).upper()
    if fuel_type not in ("EV", "PHEV", "BEV"):
        return {
            "vin":          vin,
            "model_name":   vehicle.get("model_name", ""),
            "fuel_type":    fuel_type,
            "risk_level":   "NONE",
            "factors":      [],
            "action":       "monitor",
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "note":         f"Thermal runaway monitoring applies to EV/PHEV only (this vehicle is {fuel_type})",
        }

    # Load telemetry — last 48h when timestamps allow, else most-recent rows
    telem_df: pd.DataFrame = pd.DataFrame()
    for pattern in (f"telemetry_{vin}.csv", f"{vin}_telemetry.csv"):
        csv_path = DATA_DIR / pattern
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                ts_col = "StartTime-TimeStamp"
                if ts_col in df.columns:
                    cutoff = (datetime.now(timezone.utc).timestamp()) - 48 * 3600
                    recent = df[df[ts_col] >= cutoff]
                    telem_df = recent if not recent.empty else df.tail(500)
                else:
                    telem_df = df.tail(500)
            except Exception:
                pass
            break

    from models.thermal_runaway_model import ThermalRunawayEarlyWarner, persist_thermal_log, dispatch_critical_sms
    result = ThermalRunawayEarlyWarner().classify(vin, telem_df, {})

    # Persist every evaluation for audit trail (including NONE)
    persist_thermal_log(result)

    # CRITICAL: dispatch SMS immediately without cooldown
    if result["risk_level"] == "CRITICAL":
        dispatch_critical_sms(result)

    return {
        "model_name":           vehicle.get("model_name", ""),
        "fuel_type":            fuel_type,
        "battery_capacity_kwh": _safe(vehicle.get("battery_capacity_kwh")),
        **result,
    }
