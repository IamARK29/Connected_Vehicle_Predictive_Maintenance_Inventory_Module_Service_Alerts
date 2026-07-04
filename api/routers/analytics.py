"""Analytics and reporting endpoints."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import APIRouter, Depends

from api.dependencies import get_current_user

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get("/fleet/health")
async def fleet_health_overview(current_user: Annotated[dict, Depends(get_current_user)]):
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fleet_health_score": 82.4,
        "vehicles_by_health": {"excellent": 120, "good": 210, "fair": 130, "poor": 40},
        "components_by_status": {
            "brakes": {"ok": 380, "warning": 90, "critical": 30},
            "engine": {"ok": 400, "warning": 75, "critical": 25},
            "tyres": {"ok": 350, "warning": 110, "critical": 40},
            "hv_battery": {"ok": 300, "warning": 60, "critical": 15},
        },
        "top_alert_components": ["brakes", "engine", "tyres"],
    }


@router.get("/fleet/cost-forecast")
async def cost_forecast(months: int = 3, current_user: Annotated[dict, Depends(get_current_user)] = None):
    base = datetime.now(timezone.utc)
    forecast = []
    monthly_cost = 850000

    for m in range(1, months + 1):
        month_date = (base + timedelta(days=30 * m)).strftime("%Y-%m")
        forecast.append({
            "month": month_date,
            "estimated_maintenance_cost_inr": round(monthly_cost * (1 + m * 0.03)),
            "vehicles_due_service": 45 + m * 5,
            "breakdown": {
                "engine_service": round(monthly_cost * 0.3),
                "brake_service": round(monthly_cost * 0.2),
                "tyre_service": round(monthly_cost * 0.25),
                "battery_service": round(monthly_cost * 0.15),
                "other": round(monthly_cost * 0.1),
            },
        })
    return {"months": months, "forecast": forecast, "total_estimated_inr": sum(f["estimated_maintenance_cost_inr"] for f in forecast)}


@router.get("/vehicle/{vin}/history")
async def vehicle_maintenance_history(vin: str, current_user: Annotated[dict, Depends(get_current_user)]):
    return {
        "vin": vin,
        "total_services": 12,
        "total_cost_inr": 45600,
        "last_service_date": (datetime.now(timezone.utc) - timedelta(days=45)).isoformat(),
        "service_history": [],
    }


@router.get("/dealer/{dealer_id}/performance")
async def dealer_performance(dealer_id: str, current_user: Annotated[dict, Depends(get_current_user)]):
    return {
        "dealer_id": dealer_id,
        "vehicles_managed": 85,
        "alerts_resolved_pct": 92.3,
        "avg_service_completion_days": 2.1,
        "customer_satisfaction_score": 4.3,
        "revenue_inr": 1250000,
    }


@router.get("/reports/alert-trends")
async def alert_trends(days: int = 30, current_user: Annotated[dict, Depends(get_current_user)] = None):
    base = datetime.now(timezone.utc)
    trends = []
    for d in range(days, 0, -1):
        day_date = (base - timedelta(days=d)).strftime("%Y-%m-%d")
        trends.append({
            "date": day_date,
            "critical": max(0, 15 - d % 7 + d % 3),
            "warning": max(0, 35 - d % 5 + d % 4),
            "info": max(0, 20 - d % 3 + d % 2),
        })
    return {"days": days, "trends": trends}
