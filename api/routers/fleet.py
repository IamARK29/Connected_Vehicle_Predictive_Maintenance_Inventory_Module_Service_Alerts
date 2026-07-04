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


def _generate_alerts_from_fleet(fleet: list[dict]) -> list[dict]:
    """Generate realistic alerts from fleet data based on odometer and driver profile."""
    alerts = []
    rng = np.random.default_rng(42)

    alert_defs = [
        {"alert_type": "ML_BRAKE_REPLACEMENT", "severity": "HIGH",     "title": "Brake pad replacement needed",
         "action": "Schedule brake service within 2 weeks", "cost_min": 3500, "cost_max": 8000,
         "odo_thresh": 35000, "profiles": ["aggressive", "taxi_fleet", "delivery_driver"]},
        {"alert_type": "ML_OIL_CHANGE_DUE",    "severity": "HIGH",     "title": "Oil change overdue",
         "action": "Schedule oil change this week", "cost_min": 1200, "cost_max": 2500,
         "odo_thresh": 5000, "profiles": ["aggressive", "taxi_fleet", "hill_region"]},
        {"alert_type": "ML_BRAKE_WARNING",      "severity": "MEDIUM",   "title": "Brake wear accelerating",
         "action": "Inspect brakes at next service", "cost_min": 3000, "cost_max": 6000,
         "odo_thresh": 25000, "profiles": ["urban_commuter", "hill_region"]},
        {"alert_type": "ML_OIL_ADVISORY",       "severity": "MEDIUM",   "title": "Oil change due soon",
         "action": "Schedule oil change within 30 days", "cost_min": 1200, "cost_max": 1800,
         "odo_thresh": 4000, "profiles": ["urban_commuter", "highway_cruiser"]},
        {"alert_type": "ML_12V_ADVISORY",       "severity": "MEDIUM",   "title": "12V battery weakening",
         "action": "Test battery at next service", "cost_min": 4500, "cost_max": 6500,
         "odo_thresh": 50000, "profiles": ["elderly_cautious", "urban_commuter"]},
        {"alert_type": "ML_TYRE_ADVISORY",      "severity": "MEDIUM",   "title": "Tyre wear increasing",
         "action": "Inspect tyres, check alignment", "cost_min": 8500, "cost_max": 15000,
         "odo_thresh": 30000, "profiles": ["aggressive", "hill_region", "taxi_fleet"]},
        {"alert_type": "ML_HV_SOH_DECLINE",     "severity": "MEDIUM",   "title": "HV battery SoH declining",
         "action": "Schedule battery assessment", "cost_min": 10000, "cost_max": 50000,
         "odo_thresh": 60000, "profiles": ["taxi_fleet", "delivery_driver"], "fuel_types": ["EV", "PHEV"]},
        {"alert_type": "ML_DRIVER_ADVISORY",     "severity": "LOW",      "title": "Driving behaviour advisory",
         "action": "Review driving habits", "cost_min": 0, "cost_max": 0,
         "odo_thresh": 0, "profiles": ["aggressive"]},
    ]

    for v in fleet:
        vin = str(v.get("vin", ""))
        odo = float(v.get("initial_odometer", 0) or 0)
        profile = str(v.get("driver_profile", "urban_commuter"))
        fuel = str(v.get("fuel_type", "ICE"))
        model_name = str(v.get("model_name", ""))

        for ad in alert_defs:
            if odo % max(ad["odo_thresh"], 1) > ad["odo_thresh"] * 0.7 or profile in ad.get("profiles", []):
                if "fuel_types" in ad and fuel not in ad["fuel_types"]:
                    continue
                if float(rng.random()) > 0.4:
                    continue
                hours_ago = int(rng.integers(1, 168))
                alerts.append({
                    "vin": vin,
                    "alert_type": ad["alert_type"],
                    "severity": ad["severity"],
                    "title": ad["title"],
                    "message_customer": f"{ad['title']} - {model_name} ({vin[-8:]})",
                    "recommended_action": ad["action"],
                    "estimated_cost_min": ad["cost_min"],
                    "estimated_cost_max": ad["cost_max"],
                    "confidence_score": round(min(0.95, max(0.42,
                        (0.85 if ad["severity"] == "HIGH" else 0.65 if ad["severity"] == "MEDIUM" else 0.48)
                        + (hash(vin + ad["alert_type"]) % 100) / 1000.0
                    )), 2),
                    "triggered_at": (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat(),
                })

    alerts.sort(key=lambda a: {"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(a["severity"], 3))
    return alerts


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/health-summary", response_model=FleetHealthSummary)
async def fleet_health_summary(current_user: Annotated[dict, Depends(get_current_user)]):
    fleet = _load_fleet()
    dc = current_user.get("dealer_code", "ALL")
    if dc and dc != "ALL":
        fleet = [v for v in fleet if str(v.get("dealer_code", "")) == dc]
    alerts = _generate_alerts_from_fleet(fleet)
    n = len(fleet)

    critical = sum(1 for a in alerts if a.get("severity") == "CRITICAL")
    high = sum(1 for a in alerts if a.get("severity") == "HIGH")
    medium = sum(1 for a in alerts if a.get("severity") == "MEDIUM")
    due = len({a["vin"] for a in alerts if a.get("severity") in ("CRITICAL", "HIGH")})

    trips = _load_trips()
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
    alerts = _generate_alerts_from_fleet(fleet)
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
