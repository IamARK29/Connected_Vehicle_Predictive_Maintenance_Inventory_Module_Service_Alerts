"""Maintenance scheduling endpoints."""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_current_user
from api.schemas import MaintenanceScheduleItem
from agent.cost_estimator import estimate_repair_cost

router = APIRouter(prefix="/maintenance", tags=["Maintenance"])

_schedule_store: dict[str, list[dict]] = {}

SERVICE_INTERVALS = {
    "engine_oil": 10000,
    "air_filter": 20000,
    "brake_fluid": 40000,
    "tyre_rotation": 10000,
    "brake_pads": 50000,
    "spark_plugs": 40000,
}


@router.get("/{vin}/schedule", response_model=list[MaintenanceScheduleItem])
async def get_maintenance_schedule(
    vin: str,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    return _schedule_store.get(vin, [])


@router.post("/{vin}/schedule", response_model=MaintenanceScheduleItem, status_code=201)
async def create_maintenance_item(
    vin: str,
    item: MaintenanceScheduleItem,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    record = item.model_dump()
    record["id"] = str(uuid.uuid4())
    record["created_at"] = datetime.now(timezone.utc).isoformat()
    record["vin"] = vin

    cost_info = estimate_repair_cost(item.component, item.service_type)
    if "total_cost_inr" in cost_info:
        record["estimated_cost_inr"] = cost_info["total_cost_inr"]

    if vin not in _schedule_store:
        _schedule_store[vin] = []
    _schedule_store[vin].append(record)
    return record


@router.get("/{vin}/upcoming")
async def get_upcoming_maintenance(
    vin: str,
    current_odometer: float = 0,
    current_user: Annotated[dict, Depends(get_current_user)] = None,
):
    upcoming = []
    for component, interval_km in SERVICE_INTERVALS.items():
        next_due_km = (current_odometer // interval_km + 1) * interval_km
        km_remaining = next_due_km - current_odometer
        service_type = component.replace("_", " ")
        cost = estimate_repair_cost(component.replace("_", " "), service_type)
        upcoming.append({
            "component": component,
            "service_type": service_type,
            "current_odometer_km": current_odometer,
            "due_at_km": next_due_km,
            "km_remaining": km_remaining,
            "priority": "urgent" if km_remaining < 500 else "upcoming",
            "estimated_cost_inr": cost.get("total_cost_inr"),
        })

    upcoming.sort(key=lambda x: x["km_remaining"])
    return upcoming


@router.patch("/{vin}/schedule/{item_id}/complete")
async def mark_completed(vin: str, item_id: str, current_user: Annotated[dict, Depends(get_current_user)]):
    items = _schedule_store.get(vin, [])
    for item in items:
        if item.get("id") == item_id:
            item["status"] = "completed"
            item["completed_at"] = datetime.now(timezone.utc).isoformat()
            return item
    raise HTTPException(status_code=404, detail=f"Schedule item {item_id} not found")


@router.get("/fleet/overdue")
async def get_fleet_overdue(current_user: Annotated[dict, Depends(get_current_user)]):
    overdue = []
    now = datetime.now(timezone.utc)
    for vin, items in _schedule_store.items():
        for item in items:
            if item.get("status") == "completed":
                continue
            due = item.get("due_date")
            if due and datetime.fromisoformat(due) < now:
                overdue.append({**item, "vin": vin, "days_overdue": (now - datetime.fromisoformat(due)).days})
    return overdue
