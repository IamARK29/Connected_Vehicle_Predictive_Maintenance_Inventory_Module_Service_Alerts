"""Telemetry ingestion and retrieval endpoints."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_current_user
from api.schemas import TelemetryIngest
from ingestion.validators import validate_telemetry

router = APIRouter(prefix="/telemetry", tags=["Telemetry"])

_telemetry_cache: dict[str, list[dict]] = {}  # vin -> [records]


@router.post("/ingest", status_code=202)
async def ingest_telemetry(
    payload: TelemetryIngest,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    data = payload.model_dump()
    data["timestamp"] = data["timestamp"].isoformat()

    errors = validate_telemetry(data)
    if errors:
        raise HTTPException(status_code=422, detail={"validation_errors": errors})

    if payload.vin not in _telemetry_cache:
        _telemetry_cache[payload.vin] = []
    _telemetry_cache[payload.vin].append(data)

    if len(_telemetry_cache[payload.vin]) > 10000:
        _telemetry_cache[payload.vin] = _telemetry_cache[payload.vin][-5000:]

    try:
        from ingestion.db_writer import write_telemetry_sync
        write_telemetry_sync(payload.vin, data)
    except Exception:
        pass

    return {"accepted": True, "vin": payload.vin}


@router.post("/ingest/batch", status_code=202)
async def ingest_batch(
    records: list[TelemetryIngest],
    current_user: Annotated[dict, Depends(get_current_user)],
):
    accepted = 0
    for rec in records:
        data = rec.model_dump()
        data["timestamp"] = data["timestamp"].isoformat()
        if not validate_telemetry(data):
            if rec.vin not in _telemetry_cache:
                _telemetry_cache[rec.vin] = []
            _telemetry_cache[rec.vin].append(data)
            accepted += 1
    return {"accepted": accepted, "total": len(records)}


@router.get("/{vin}/latest")
async def get_latest_telemetry(vin: str, current_user: Annotated[dict, Depends(get_current_user)]):
    records = _telemetry_cache.get(vin, [])
    if not records:
        raise HTTPException(status_code=404, detail=f"No telemetry for vehicle {vin}")
    return records[-1]


@router.get("/{vin}/history")
async def get_telemetry_history(
    vin: str,
    hours: int = 24,
    current_user: Annotated[dict, Depends(get_current_user)] = None,
):
    records = _telemetry_cache.get(vin, [])
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    filtered = [r for r in records if r.get("timestamp", "") >= cutoff]
    return {"vin": vin, "hours": hours, "count": len(filtered), "records": filtered[-500:]}
