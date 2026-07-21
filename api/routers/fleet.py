"""
Fleet-level aggregated endpoints — works fully offline from CSV files.

GET /api/fleet/health-summary        -> aggregate health KPIs
GET /api/fleet/alerts/active         -> alerts (from CSV rule evaluation)
GET /api/fleet/maintenance-calendar  -> upcoming predicted service events
GET /api/fleet/driver-scores         -> ranked driver scores from trips.csv
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, Query

from api.dependencies import get_current_user
from api.schemas import (
    FleetHealthSummary, MaintenanceEvent, DriverScoreEntry,
)

router = APIRouter(prefix="/fleet", tags=["Fleet"])

DATA_DIR = Path(os.getenv("DATA_DIR", "data/synthetic"))


# ── CSV data helpers (no DB required) ─────────────────────────────────────────

def _load_fleet() -> list[dict]:
    for name in ("fleet.csv", "fleet_master.csv"):
        p = DATA_DIR / name
        if p.exists():
            return pd.read_csv(p).to_dict("records")
    return []


def _load_trips() -> pd.DataFrame:
    p = DATA_DIR / "trips.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def _load_service_history() -> pd.DataFrame:
    p = DATA_DIR / "service_history.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


_PROFILE_PENALTY: dict[str, float] = {
    "aggressive": 15, "taxi_fleet": 12, "delivery_driver": 10,
    "hill_region": 8, "urban_commuter": 3, "highway_cruiser": 2,
    "elderly_cautious": 1, "eco_driver": 0,
}


def _fleet_health_scores(fleet: list[dict], trips_df: pd.DataFrame | None = None) -> dict[str, float]:
    """
    Replicate _compute_vehicle_health() from vehicles.py using the identical formula
    so alert severity is consistent with what the Predictions tab shows.
    """
    score_map: dict[str, float] = {}
    for v in fleet:
        vin     = str(v.get("vin", ""))
        odo     = float(v.get("initial_odometer", 0) or 0)
        profile = str(v.get("driver_profile", "urban_commuter") or "urban_commuter")
        penalty = _PROFILE_PENALTY.get(profile, 5)
        odo_factor = max(0.0, 100.0 - odo / 1500.0)

        drive_score = 75.0
        if trips_df is not None and not trips_df.empty:
            if "vin" in trips_df.columns and "driveScore" in trips_df.columns:
                vin_trips = trips_df[trips_df["vin"] == vin]
                if not vin_trips.empty:
                    drive_score = float(vin_trips["driveScore"].mean())

        rng = np.random.default_rng(hash(vin) % 2**31)
        health = float(np.clip(round(
            odo_factor * 0.3 + drive_score * 0.4 + (100.0 - penalty) * 0.3
            + float(rng.normal(0, 3)), 1
        ), 20, 100))
        score_map[vin] = health
    return score_map


_COMP_ALERTS: dict[str, dict] = {
    # component → {status → alert definition}
    "brake": {
        "critical": {"alert_type": "ML_BRAKE_SYSTEM_FAILURE", "title": "Brake system critical — safety hazard",
                     "action": "Do not drive — arrange tow to nearest workshop", "cost_min": 8000, "cost_max": 30000},
        "warning":  {"alert_type": "ML_BRAKE_REPLACEMENT",    "title": "Brake pad replacement needed",
                     "action": "Schedule brake service within 2 weeks",           "cost_min": 3500, "cost_max": 8000},
    },
    "tyre": {
        "critical": {"alert_type": "ML_TYRE_CRITICAL",  "title": "Tyre wear critical — safety risk",
                     "action": "Do not drive — replace tyres immediately",         "cost_min": 15000, "cost_max": 30000},
        "warning":  {"alert_type": "ML_TYRE_ADVISORY",  "title": "Tyre wear increasing",
                     "action": "Inspect tyres and check wheel alignment",          "cost_min": 8500,  "cost_max": 15000},
    },
    "battery_12v": {
        "critical": {"alert_type": "ML_12V_BATTERY_FAILURE", "title": "12V battery failure imminent",
                     "action": "Battery replacement required immediately",          "cost_min": 5500, "cost_max": 8500},
        "warning":  {"alert_type": "ML_12V_ADVISORY",        "title": "12V battery weakening",
                     "action": "Test battery at next service",                     "cost_min": 4500, "cost_max": 6500},
    },
    "engine_oil": {
        "critical": {"alert_type": "ML_ENGINE_OIL_CRITICAL", "title": "Engine oil critically degraded — engine risk",
                     "action": "Stop driving — oil change required immediately",   "cost_min": 1500, "cost_max": 25000},
        "warning":  {"alert_type": "ML_OIL_CHANGE_DUE",     "title": "Engine oil change overdue",
                     "action": "Schedule oil change this week",                    "cost_min": 1200, "cost_max": 2500},
    },
    "hv_battery": {
        "critical": {"alert_type": "ML_HV_BATTERY_CRITICAL", "title": "HV traction battery critical degradation",
                     "action": "EV drivetrain inspection required immediately",    "cost_min": 40000, "cost_max": 200000},
        "warning":  {"alert_type": "ML_HV_SOH_DECLINE",     "title": "HV battery SoH declining",
                     "action": "Schedule battery health assessment",               "cost_min": 10000, "cost_max": 50000},
    },
}
_STATUS_TO_SEV = {"critical": "CRITICAL", "warning": "HIGH"}
_SEV_CONF      = {"CRITICAL": 0.93, "HIGH": 0.80, "MEDIUM": 0.65, "LOW": 0.50}
_SEV_ORDER     = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}


def _vehicle_components(vin: str, v: dict) -> dict[str, tuple[float, str]]:
    """
    Compute per-component health scores using the IDENTICAL formula as
    _compute_vehicle_health() in vehicles.py, so fleet alerts always match
    what the Predictions tab shows for individual vehicles.
    """
    odo     = float(v.get("initial_odometer", 0) or 0)
    fuel    = str(v.get("fuel_type", "ICE") or "ICE")
    penalty = _PROFILE_PENALTY.get(str(v.get("driver_profile", "") or ""), 5)

    rng = np.random.default_rng(hash(vin) % 2**31)
    _ = float(rng.normal(0, 3))                          # advance past composite health draw

    brake = round(max(30.0, 100 - odo / 900 - penalty  + float(rng.normal(0, 5))), 1)
    tyre  = round(max(25.0, 100 - (odo % 50000) / 500  + float(rng.normal(0, 4))), 1)
    bat12 = round(max(40.0, 90  - odo / 15000          + float(rng.normal(0, 3))), 1)

    comps: dict[str, tuple[float, str]] = {
        "brake":       (brake, "critical" if brake < 40 else "warning" if brake < 65 else "ok"),
        "tyre":        (tyre,  "critical" if tyre  < 35 else "warning" if tyre  < 60 else "ok"),
        "battery_12v": (bat12, "critical" if bat12 < 50 else "warning" if bat12 < 70 else "ok"),
    }

    if fuel != "EV":
        oil = round(max(20.0, 100 - (odo % 7500) / 75 + float(rng.normal(0, 3))), 1)
        comps["engine_oil"] = (oil, "critical" if oil < 30 else "warning" if oil < 60 else "ok")

    if fuel in ("EV", "PHEV"):
        hv = round(max(50.0, 95 - odo / 20000 + float(rng.normal(0, 2))), 1)
        comps["hv_battery"] = (hv, "critical" if hv < 75 else "warning" if hv < 85 else "ok")

    return comps


def _generate_alerts_from_fleet(fleet: list[dict], trips_df: pd.DataFrame | None = None) -> list[dict]:
    """
    Generate component-level alerts that exactly mirror what the Vehicle Detail
    Predictions tab shows — no random skip, no stale seed, CRITICAL always fires
    for vehicles with critical-health components.
    """
    alerts: list[dict] = []

    for v in fleet:
        vin        = str(v.get("vin", ""))
        model_name = str(v.get("model_name", "") or "")
        profile    = str(v.get("driver_profile", "urban_commuter") or "urban_commuter")

        vh = abs(hash(vin + "alerts_v3"))   # stable, version-tagged hash

        comps = _vehicle_components(vin, v)

        for comp_name, (score, status) in comps.items():
            if status == "ok":
                continue
            sev = _STATUS_TO_SEV[status]            # "critical" → "CRITICAL", "warning" → "HIGH"
            defn = _COMP_ALERTS.get(comp_name, {}).get(status)
            if not defn:
                continue

            # Hours ago: CRITICAL = 1–24h, HIGH = 12–72h
            hrs_min, hrs_max = (1, 24) if sev == "CRITICAL" else (12, 72)
            # Spread across the window deterministically using comp hash
            comp_seed = abs(hash(vin + comp_name))
            hrs = hrs_min + (comp_seed % max(hrs_max - hrs_min, 1))

            conf = round(min(0.99, _SEV_CONF[sev] + (comp_seed % 80) / 1000.0), 2)
            comp_label = comp_name.replace("_", " ").title()

            alerts.append({
                "vin":                vin,
                "alert_type":         defn["alert_type"],
                "severity":           sev,
                "title":              defn["title"],
                "message_customer":   f"{comp_label} health at {score}% on {model_name} ({vin[-8:]})",
                "recommended_action": defn["action"],
                "estimated_cost_min": defn["cost_min"],
                "estimated_cost_max": defn["cost_max"],
                "confidence_score":   conf,
                "triggered_at": (datetime.now(timezone.utc) - timedelta(hours=hrs)).isoformat(),
            })

        # LOW advisory for aggressive/taxi drivers — once per vehicle
        if profile in ("aggressive", "taxi_fleet") and (vh % 2) == 0:
            alerts.append({
                "vin":                vin,
                "alert_type":         "ML_DRIVER_ADVISORY",
                "severity":           "LOW",
                "title":              "Driving behaviour advisory",
                "message_customer":   f"Aggressive driving detected on {model_name} ({vin[-8:]})",
                "recommended_action": "Review driving habits to improve vehicle longevity",
                "estimated_cost_min": 0,
                "estimated_cost_max": 0,
                "confidence_score":   round(0.52 + (vh % 60) / 1000.0, 2),
                "triggered_at": (datetime.now(timezone.utc) - timedelta(hours=48 + vh % 120)).isoformat(),
            })

    alerts.sort(key=lambda a: (_SEV_ORDER.get(a["severity"], 4), a["triggered_at"]))
    return alerts


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/health-summary", response_model=FleetHealthSummary)
async def fleet_health_summary(current_user: Annotated[dict, Depends(get_current_user)]):
    fleet = _load_fleet()
    trips = _load_trips()
    dc = current_user.get("dealer_code", "ALL")
    if dc and dc != "ALL":
        fleet = [v for v in fleet if str(v.get("dealer_code", "")) == dc]
    alerts = _generate_alerts_from_fleet(fleet, trips)
    n = len(fleet)

    critical = sum(1 for a in alerts if a.get("severity") == "CRITICAL")
    high = sum(1 for a in alerts if a.get("severity") == "HIGH")
    medium = sum(1 for a in alerts if a.get("severity") == "MEDIUM")
    due = len({a["vin"] for a in alerts if a.get("severity") in ("CRITICAL", "HIGH")})

    avg_score = 0.0
    if not trips.empty and "driveScore" in trips.columns:
        avg_score = round(float(trips["driveScore"].mean()), 1)
    else:
        avg_score = round(max(0.0, 85.0 - (critical * 8 + high * 3 + medium * 1)), 1)

    return FleetHealthSummary(
        total_vehicles=n,
        online_now=max(1, n * 3 // 4),
        active_alerts_critical=critical,
        active_alerts_high=high,
        active_alerts_medium=medium,
        vehicles_due_service=due,
        fleet_avg_health_score=avg_score,
        generated_at=datetime.now(timezone.utc),
    )


@router.get("/alerts/active")
async def active_alerts(
    current_user: Annotated[dict, Depends(get_current_user)],
    severity: str | None = Query(None),
    hours: int = Query(168, ge=1, le=720),
    limit: int = Query(200, ge=1, le=1000),
):
    fleet = _load_fleet()
    trips = _load_trips()
    dc = current_user.get("dealer_code", "ALL")
    if dc and dc != "ALL":
        fleet = [v for v in fleet if str(v.get("dealer_code", "")) == dc]

    alerts = _generate_alerts_from_fleet(fleet, trips)

    # Apply the hours window — only return alerts triggered within [now - hours, now]
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    alerts = [
        a for a in alerts
        if datetime.fromisoformat(a["triggered_at"]) >= cutoff
    ]

    if severity:
        alerts = [a for a in alerts if a.get("severity", "").upper() == severity.upper()]

    return {
        "count": len(alerts),
        "hours": hours,
        "alerts": alerts[:limit],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/maintenance-calendar", response_model=list[MaintenanceEvent])
async def maintenance_calendar(
    current_user: Annotated[dict, Depends(get_current_user)],
    days: int = Query(90, ge=7, le=365),
    severity: str | None = Query(None),
    limit: int = Query(100, ge=1, le=500),
):
    fleet = _load_fleet()
    svc = _load_service_history()
    events: list[MaintenanceEvent] = []
    rng = np.random.default_rng(99)

    services = [
        ("brake_wear", "MEDIUM", 45000, 15000),
        ("engine_oil", "MEDIUM", 7500, 1500),
        ("tyre_wear", "MEDIUM", 50000, 15000),
        ("battery_12v", "HIGH", 60000, 20000),
    ]

    for v in fleet:
        vin = str(v.get("vin", ""))
        odo = float(v.get("initial_odometer", 0) or 0)
        model_name = str(v.get("model_name", ""))
        plate = str(v.get("license_plate", ""))

        for svc_type, sev, interval_km, variance in services:
            if severity and sev.upper() != severity.upper():
                continue
            km_remaining = interval_km - (odo % interval_km) + float(rng.normal(0, variance / 3))
            days_until = max(1, int(km_remaining / max(float(rng.uniform(30, 80)), 1)))
            if 0 < days_until <= days:
                events.append(MaintenanceEvent(
                    vin=vin,
                    license_plate=plate,
                    model_name=model_name,
                    alert_type=svc_type,
                    severity=sev,
                    predicted_date=(datetime.now(timezone.utc).date().__add__(
                        __import__("datetime").timedelta(days=days_until)
                    )).isoformat(),
                    days_until=days_until,
                    confidence=round(float(rng.uniform(0.6, 0.9)), 2),
                ))

    events.sort(key=lambda e: (e.days_until or 999))
    return events[:limit]


_PROFILE_BASE: dict[str, float] = {
    "eco_driver":       92.0,
    "elderly_cautious": 86.0,
    "highway_cruiser":  78.0,
    "urban_commuter":   72.0,
    "hill_region":      64.0,
    "delivery_driver":  56.0,
    "taxi_fleet":       50.0,
    "aggressive":       42.0,
}


@router.get("/driver-scores", response_model=list[DriverScoreEntry])
async def driver_scores(
    current_user: Annotated[dict, Depends(get_current_user)],
    limit: int = Query(50, ge=1, le=500),
):
    fleet = _load_fleet()
    trips = _load_trips()
    scores: list[DriverScoreEntry] = []

    for v in fleet:
        vin = str(v.get("vin", ""))
        plate = str(v.get("license_plate", "") or "")
        profile = str(v.get("driver_profile", "urban_commuter") or "urban_commuter")

        # Profile-based base score gives meaningful spread across archetypes.
        # VIN hash adds ±5 deterministic jitter so same-profile vehicles still differ.
        base = _PROFILE_BASE.get(profile, 72.0)
        vin_jitter = ((hash(vin) % 1000) / 1000.0 - 0.5) * 10.0  # ±5
        profile_score = base + vin_jitter

        trip_score: float | None = None
        if not trips.empty and "vin" in trips.columns and "driveScore" in trips.columns:
            vin_trips = trips[trips["vin"] == vin]
            if not vin_trips.empty:
                trip_score = float(vin_trips["driveScore"].mean())

        # Blend: 60% profile-based (to maintain archetype spread) + 40% actual trips
        if trip_score is not None:
            score = round(0.6 * profile_score + 0.4 * trip_score, 1)
        else:
            score = round(profile_score, 1)

        score = max(0.0, min(100.0, score))
        risk = "low" if score >= 70 else "medium" if score >= 50 else "high"
        scores.append(DriverScoreEntry(
            vin=vin,
            license_plate=plate,
            driver_name=profile.replace("_", " ").title(),
            score=score,
            risk_category=risk,
        ))

    scores.sort(key=lambda s: s.score, reverse=True)
    for i, s in enumerate(scores):
        s.rank = i + 1
        s.percentile = round(((len(scores) - i) / max(len(scores), 1)) * 100, 1)

    return scores[:limit]
