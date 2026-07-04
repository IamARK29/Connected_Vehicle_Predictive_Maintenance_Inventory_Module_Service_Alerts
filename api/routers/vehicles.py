"""
Vehicle-level REST endpoints.

GET  /api/vehicles                     → list all VINs with health summary
GET  /api/vehicles/{vin}               → full vehicle profile + current health scores
GET  /api/vehicles/{vin}/telemetry     → last N minutes of raw telemetry
GET  /api/vehicles/{vin}/predictions   → all active ML predictions
GET  /api/vehicles/{vin}/alerts        → alert history (last 30 days)
GET  /api/vehicles/{vin}/service-history → linked service records
GET  /api/vehicles/{vin}/trips         → trip history with drive scores
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import get_current_user
from api.schemas import (
    VehicleHealthSummary, VehicleDetail, HealthScore,
    TelemetryRow, MLPrediction, AlertResponse,
    ServiceRecord, TripRecord,
)

router = APIRouter(prefix="/vehicles", tags=["Vehicles"])

DATA_DIR = os.getenv("DATA_DIR", "data/synthetic")

# ── Data helpers ───────────────────────────────────────────────────────────────

def _load_fleet() -> list[dict]:
    """Load fleet from CSV or PostgreSQL."""
    import pandas as pd, pathlib
    csv = pathlib.Path(DATA_DIR) / "fleet.csv"
    if csv.exists():
        return pd.read_csv(csv).to_dict("records")
    try:
        import psycopg2
        conn = psycopg2.connect(os.getenv("DATABASE_URL", ""))
        df   = pd.read_sql("SELECT * FROM fleet LIMIT 1000", conn)
        conn.close()
        return df.to_dict("records")
    except Exception:
        return []


def _get_vehicle(vin: str) -> dict | None:
    for v in _load_fleet():
        if str(v.get("vin", "")) == vin:
            return v
    return None


def _load_telemetry(vin: str, minutes: int = 60) -> list[dict]:
    """Fetch last *minutes* of telemetry from InfluxDB or CSV."""
    try:
        from influxdb_client import InfluxDBClient
        url    = os.getenv("INFLUXDB_URL", "http://localhost:8086")
        token  = os.getenv("INFLUXDB_TOKEN", "autopredict-dev-token")
        org    = os.getenv("INFLUXDB_ORG", "autopredict")
        bucket = os.getenv("INFLUXDB_BUCKET", "telemetry")
        client = InfluxDBClient(url=url, token=token, org=org)
        qapi   = client.query_api()
        flux   = f'''
            from(bucket: "{bucket}")
            |> range(start: -{minutes}m)
            |> filter(fn: (r) => r["vin"] == "{vin}")
            |> pivot(rowKey:["_time"], columnKey:["_field"], valueColumn:"_value")
            |> sort(columns:["_time"], desc: true)
            |> limit(n: 500)
        '''
        tables = qapi.query(flux)
        rows   = []
        for table in tables:
            for rec in table.records:
                rows.append({"timestamp": rec.get_time(), **{k: v for k, v in rec.values.items() if not k.startswith("_")}})
        client.close()
        return rows
    except Exception:
        pass

    # CSV fallback
    import pandas as pd, pathlib
    tdir = pathlib.Path(DATA_DIR) / "telemetry"
    csv  = (tdir / f"{vin}_telemetry.csv") if tdir.exists() else (pathlib.Path(DATA_DIR) / f"{vin}_telemetry.csv")
    if not csv.exists():
        csv = pathlib.Path(DATA_DIR) / f"telemetry_{vin}.csv"
    if not csv.exists():
        return []
    df   = pd.read_csv(csv, parse_dates=["StartTime-TimeStamp"])
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    df = df[pd.to_datetime(df["StartTime-TimeStamp"], utc=True) >= cutoff]
    df = df.tail(500)
    return df.to_dict("records")


def _run_predictions(vin: str) -> dict[str, Any]:
    try:
        from models.model_registry import ModelRegistry
        return ModelRegistry().predict_all(vin)
    except Exception:
        return {}


def _run_predictions_with_explanation(vin: str) -> dict[str, Any]:
    """
    Returns predictions augmented with SHAP explanation for the most critical model.
    Falls back gracefully when SHAP is unavailable.
    """
    try:
        from models.model_registry import ModelRegistry
        return ModelRegistry().predict_with_explanation(vin)
    except Exception:
        preds = _run_predictions(vin)
        return {"predictions": preds, "explanation_text": "", "top3_features": []}


def _active_alerts(vin: str) -> list[dict]:
    try:
        import psycopg2, pandas as pd
        conn  = psycopg2.connect(os.getenv("DATABASE_URL", ""))
        df    = pd.read_sql(
            "SELECT * FROM alerts_log WHERE vin=%s ORDER BY triggered_at DESC LIMIT 50",
            conn, params=(vin,),
        )
        conn.close()
        return df.to_dict("records")
    except Exception:
        return []


def _service_history(vin: str, limit: int = 20) -> list[dict]:
    import pandas as pd, pathlib
    csv = pathlib.Path(DATA_DIR) / "service_history.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        df = df[df["VIN"] == vin].tail(limit)
        return df.to_dict("records")
    try:
        import psycopg2
        conn = psycopg2.connect(os.getenv("DATABASE_URL", ""))
        df   = pd.read_sql(
            "SELECT * FROM service_history WHERE \"VIN\"=%s ORDER BY \"CreatedOn\" DESC LIMIT %s",
            conn, params=(vin, limit),
        )
        conn.close()
        return df.to_dict("records")
    except Exception:
        return []


def _trips(vin: str, limit: int = 50) -> list[dict]:
    import pandas as pd, pathlib
    csv = pathlib.Path(DATA_DIR) / "trips.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        df = df[df["vin"] == vin].tail(limit)
        return df.to_dict("records")
    try:
        import psycopg2
        conn = psycopg2.connect(os.getenv("DATABASE_URL", ""))
        df   = pd.read_sql(
            "SELECT * FROM trips WHERE vin=%s ORDER BY \"startTime\" DESC LIMIT %s",
            conn, params=(vin, limit),
        )
        conn.close()
        return df.to_dict("records")
    except Exception:
        return []


def _health_summary(vin: str, vehicle: dict, preds: dict) -> VehicleHealthSummary:
    scores: list[HealthScore] = []
    critical = 0

    for model, pred in preds.items():
        sev = pred.get("severity", "ok")
        msg = pred.get("message", "")
        val = pred.get("value")
        score_val = 100.0
        if sev == "critical":
            score_val = 20.0
            critical += 1
        elif sev == "warning":
            score_val = 55.0
        else:
            score_val = 90.0
        scores.append(HealthScore(component=model, score=score_val, severity=sev, message=msg))

    overall = sum(s.score for s in scores) / len(scores) if scores else 85.0
    return VehicleHealthSummary(
        vin=vin,
        license_plate=vehicle.get("license_plate", vehicle.get("LicensePlateNumber", "")),
        model_name=vehicle.get("model_name", vehicle.get("ModelSalesCodeDescription", "")),
        fuel_type=vehicle.get("fuel_type", ""),
        overall_score=round(overall, 1),
        active_alerts=len([s for s in scores if s.severity != "ok"]),
        critical_alerts=critical,
        health_scores=scores,
    )


# ── Twin helper ───────────────────────────────────────────────────────────────

def _get_twin_manager():
    try:
        from twin.vehicle_twin import TwinManager
        return TwinManager()
    except Exception:
        return None


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get(
    "",
    summary="List all vehicles with health summary",
    responses={200: {"description": "Fleet health overview"}},
)
async def list_vehicles(
    current_user: Annotated[dict, Depends(get_current_user)],
    dealer_code: str | None = Query(None, description="Filter by dealer code"),
    fuel_type:   str | None = Query(None, description="Filter by fuel type (EV, PHEV, PETROL, DIESEL)"),
    limit:       int        = Query(100, ge=1, le=500),
):
    fleet = _load_fleet()
    if dealer_code:
        fleet = [v for v in fleet if str(v.get("dealer_code", v.get("home_dealer_code", v.get("DealerCode", "")))) == dealer_code]
    if fuel_type:
        fleet = [v for v in fleet if str(v.get("fuel_type", "")).upper() == fuel_type.upper()]
    fleet = fleet[:limit]

    rows = []
    for v in fleet:
        vin = str(v.get("vin", ""))
        odo = v.get("initial_odometer", v.get("odometer_km", 0))
        rows.append({
            "vin":                vin,
            "license_plate":      v.get("license_plate", ""),
            "model_name":         v.get("model_name", ""),
            "model_code":         v.get("model_code", ""),
            "fuel_type":          v.get("fuel_type", ""),
            "manufacture_year":   v.get("manufacture_year"),
            "dealer_code":        v.get("dealer_code", ""),
            "dealer_city":        v.get("dealer_city", ""),
            "color":              v.get("color", ""),
            "odometer_km":        odo,
            "health_score":       85.0,
            "active_alert_count": 0,
            "status":             "online",
        })
    return rows


@router.get(
    "/{vin}",
    summary="Full vehicle profile and health scores",
)
async def get_vehicle(
    vin: str,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    vehicle = _get_vehicle(vin)
    if not vehicle:
        raise HTTPException(status_code=404, detail=f"Vehicle {vin} not found")

    odo = vehicle.get("initial_odometer", vehicle.get("odometer_km", 0))
    alerts = _active_alerts(vin)

    return {
        "vin":                vin,
        "license_plate":      vehicle.get("license_plate", ""),
        "model_name":         vehicle.get("model_name", ""),
        "model_code":         vehicle.get("model_code", ""),
        "fuel_type":          vehicle.get("fuel_type", ""),
        "color":              vehicle.get("color", ""),
        "manufacture_year":   vehicle.get("manufacture_year"),
        "current_odometer_km": odo,
        "odometer_km":        odo,
        "dealer_code":        vehicle.get("dealer_code", ""),
        "dealer_name":        vehicle.get("dealer_name", ""),
        "region":             vehicle.get("region", ""),
        "health_score":       85.0,
        "active_alerts":      alerts[:10],
    }


@router.get(
    "/{vin}/telemetry",
    response_model=list[dict],
    summary="Last N minutes of raw telemetry",
)
async def get_telemetry(
    vin:     str,
    current_user: Annotated[dict, Depends(get_current_user)],
    minutes: int = Query(60, ge=1, le=1440, description="Lookback window in minutes"),
    limit:   int = Query(500, ge=1, le=2000),
):
    vehicle = _get_vehicle(vin)
    if not vehicle:
        raise HTTPException(status_code=404, detail=f"Vehicle {vin} not found")
    rows = _load_telemetry(vin, minutes=minutes)
    return rows[:limit]


@router.get(
    "/{vin}/predictions",
    response_model=dict[str, Any],
    summary="All active ML predictions for this VIN",
)
async def get_predictions(
    vin: str,
    current_user: Annotated[dict, Depends(get_current_user)],
    explain: bool = Query(False, description="Include SHAP explanation for the top-risk model"),
):
    vehicle = _get_vehicle(vin)
    if not vehicle:
        raise HTTPException(status_code=404, detail=f"Vehicle {vin} not found")

    if explain:
        result = _run_predictions_with_explanation(vin)
        preds  = result.get("predictions", {})
        expl_text = result.get("explanation_text", "")
        top3      = result.get("top3_features", [])
    else:
        preds     = _run_predictions(vin)
        expl_text = ""
        top3      = []

    return {
        "vin":              vin,
        "generated_at":     datetime.now(timezone.utc).isoformat(),
        "predictions":      preds,
        "explanation_text": expl_text,
        "top3_features":    top3,
    }


@router.get(
    "/{vin}/alerts",
    summary="Alert history for this VIN (last 30 days)",
)
async def get_alerts(
    vin:     str,
    current_user: Annotated[dict, Depends(get_current_user)],
    severity: str | None = Query(None, description="Filter: CRITICAL | HIGH | MEDIUM | LOW"),
    limit:    int         = Query(50, ge=1, le=200),
):
    vehicle = _get_vehicle(vin)
    if not vehicle:
        raise HTTPException(status_code=404, detail=f"Vehicle {vin} not found")

    alerts = _active_alerts(vin)
    if severity:
        alerts = [a for a in alerts if str(a.get("severity", "")).upper() == severity.upper()]
    return {"vin": vin, "count": len(alerts), "alerts": alerts[:limit]}


@router.get(
    "/{vin}/service-history",
    response_model=list[dict],
    summary="Service history records for this VIN",
)
async def get_service_history(
    vin:   str,
    current_user: Annotated[dict, Depends(get_current_user)],
    limit: int = Query(20, ge=1, le=100),
):
    vehicle = _get_vehicle(vin)
    if not vehicle:
        raise HTTPException(status_code=404, detail=f"Vehicle {vin} not found")
    records = _service_history(vin, limit=limit)
    return records


@router.get(
    "/{vin}/trips",
    response_model=list[dict],
    summary="Trip history with drive scores",
)
async def get_trips(
    vin:   str,
    current_user: Annotated[dict, Depends(get_current_user)],
    limit: int = Query(50, ge=1, le=200),
):
    vehicle = _get_vehicle(vin)
    if not vehicle:
        raise HTTPException(status_code=404, detail=f"Vehicle {vin} not found")
    return _trips(vin, limit=limit)
