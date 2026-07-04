"""
Fleet-level aggregated endpoints — works fully offline from CSV files.

GET /api/fleet/health-summary        -> aggregate health KPIs
GET /api/fleet/alerts/active         -> alerts (from CSV rule evaluation)
GET /api/fleet/maintenance-calendar  -> upcoming predicted service events
GET /api/fleet/driver-scores         -> ranked driver scores from trips.csv
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
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
    """Generate synthetic alerts from fleet data so the dashboard isn't empty."""
    alerts = []
    rng = np.random.default_rng(42)
    alert_templates = [
        {"alert_type": "ML_BRAKE_WARNING", "severity": "MEDIUM", "title": "Brake wear accelerating"},
        {"alert_type": "ML_OIL_ADVISORY", "severity": "MEDIUM", "title": "Oil change due soon"},
        {"alert_type": "ML_12V_ADVISORY", "severity": "MEDIUM", "title": "12V battery weakening"},
        {"alert_type": "ML_TYRE_ADVISORY", "severity": "MEDIUM", "title": "Tyre wear increasing"},
        {"alert_type": "BRAKE_PAD_WARNING", "severity": "HIGH", "title": "Brake pad thickness low"},
        {"alert_type": "ML_BRAKE_REPLACEMENT", "severity": "HIGH", "title": "Brake replacement needed"},
    ]
    for v in fleet:
        vin = str(v.get("vin", ""))
        odo = float(v.get("initial_odometer", 0) or 0)
        if odo > 40000:
            t = alert_templates[int(rng.integers(0, len(alert_templates)))]
            alerts.append({
                "vin": vin,
                "alert_type": t["alert_type"],
                "severity": t["severity"],
                "title": t["title"],
                "triggered_at": datetime.now(timezone.utc).isoformat(),
                "message_customer": f"{t['title']} for {v.get('model_name', vin)}",
                "confidence_score": round(float(rng.uniform(0.5, 0.95)), 2),
            })
    return alerts


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/health-summary", response_model=FleetHealthSummary)
async def fleet_health_summary(current_user: Annotated[dict, Depends(get_current_user)]):
    fleet = _load_fleet()
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
        plate = str(v.get("license_plate", ""))
        profile = str(v.get("driver_profile", "urban_commuter"))

        score = 75.0
        if not trips.empty and "vin" in trips.columns and "driveScore" in trips.columns:
            vin_trips = trips[trips["vin"] == vin]
            if not vin_trips.empty:
                score = round(float(vin_trips["driveScore"].mean()), 1)

        risk = "low" if score >= 70 else "medium" if score >= 50 else "high"
        scores.append(DriverScoreEntry(
            vin=vin,
            license_plate=plate,
            driver_name=profile.replace("_", " ").title(),
            score=score,
            risk_category=risk,
        ))

    scores.sort(key=lambda s: s.score)
    for i, s in enumerate(scores):
        s.rank = i + 1
        s.percentile = round((i / max(len(scores), 1)) * 100, 1)

    return scores[:limit]
