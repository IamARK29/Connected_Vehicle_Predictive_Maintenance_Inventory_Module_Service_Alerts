"""Alert management endpoints — evaluate, list, acknowledge."""
from __future__ import annotations

import json
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_current_user
from api.schemas import AlertResponse
from alerts.rule_engine import evaluate
from alerts.ml_alert_engine import run_ml_alerts

router = APIRouter(prefix="/alerts", tags=["Alerts"])

_alert_store: dict[str, list[dict]] = {}  # vin -> [alerts]
_acknowledged: set[str] = set()


def _alert_key(alert) -> str:
    return f"{alert.vin}:{alert.rule_id}:{alert.triggered_at}"


@router.post("/{vin}/evaluate", response_model=list[AlertResponse])
async def evaluate_vehicle_alerts(
    vin: str,
    telemetry: dict,
    use_ml: bool = True,
    current_user: Annotated[dict, Depends(get_current_user)] = None,
):
    rule_alerts = evaluate(telemetry, vin)
    ml_alerts = run_ml_alerts(telemetry, vin) if use_ml else []
    all_alerts = rule_alerts + ml_alerts

    if vin not in _alert_store:
        _alert_store[vin] = []

    alert_dicts = []
    for a in all_alerts:
        d = {
            "rule_id": a.rule_id, "vin": a.vin, "component": a.component,
            "severity": a.severity, "title": a.title, "message": a.message,
            "triggered_at": a.triggered_at, "metadata": a.metadata,
        }
        _alert_store[vin].append(d)
        alert_dicts.append(d)

    if len(_alert_store[vin]) > 1000:
        _alert_store[vin] = _alert_store[vin][-500:]

    return alert_dicts


@router.get("/{vin}", response_model=list[AlertResponse])
async def get_vehicle_alerts(
    vin: str,
    severity: str | None = None,
    current_user: Annotated[dict, Depends(get_current_user)] = None,
):
    alerts = _alert_store.get(vin, [])
    if severity:
        alerts = [a for a in alerts if a.get("severity") == severity]
    return alerts


@router.get("/", response_model=list[AlertResponse])
async def get_all_alerts(
    severity: str | None = None,
    current_user: Annotated[dict, Depends(get_current_user)] = None,
):
    all_alerts = []
    for vin_alerts in _alert_store.values():
        all_alerts.extend(vin_alerts)
    if severity:
        all_alerts = [a for a in all_alerts if a.get("severity") == severity]
    all_alerts.sort(key=lambda a: a.get("triggered_at", ""), reverse=True)
    return all_alerts[:200]


@router.post("/{vin}/{rule_id}/acknowledge")
async def acknowledge_alert(vin: str, rule_id: str, current_user: Annotated[dict, Depends(get_current_user)]):
    _acknowledged.add(f"{vin}:{rule_id}")
    return {"acknowledged": True, "vin": vin, "rule_id": rule_id}
