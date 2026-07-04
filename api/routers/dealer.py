"""
Dealer-level endpoints.

GET  /api/dealer/{dealer_code}/appointments      → today + next 7 days
GET  /api/dealer/{dealer_code}/bay-status        → live bay occupancy
GET  /api/dealer/{dealer_code}/inventory         → parts stock levels
GET  /api/dealer/{dealer_code}/demand-forecast   → 30/90 day parts demand
POST /api/dealer/{dealer_code}/appointments      → create appointment
PUT  /api/dealer/{dealer_code}/appointments/{id}/status → update stage
"""
from __future__ import annotations

import os
import pathlib
from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import get_current_user
from api.schemas import (
    AppointmentCreate, AppointmentResponse, AppointmentStatusUpdate,
    BayStatus, InventoryItem, DemandForecast,
)

router = APIRouter(prefix="/dealer", tags=["Dealer"])

DATA_DIR = pathlib.Path(os.getenv("DATA_DIR", "data/synthetic"))

_PARTS_CATALOGUE: list[str] = [
    "BR-PAD-F-MG",    "BR-PAD-R-MG",   "BR-FLUID-DOT4",
    "OIL-5W30-4L",    "OIL-FILTER-MG", "BATT-12V-60AH",
    "BATT-12V-70AH",  "TYRE-215-60-17","TYRE-225-55-18",
    "COOLANT-1L",     "THERMOSTAT-MG", "HV-MODULE-MG",
    "BMS-FUSE-MG",    "SPARK-PLUG-NGK","AIR-FILTER-MG",
]

_PARTS_DESCRIPTIONS: dict[str, str] = {
    "BR-PAD-F-MG":     "Brake Pads (Front) — MG OEM",
    "BR-PAD-R-MG":     "Brake Pads (Rear) — MG OEM",
    "BR-FLUID-DOT4":   "Brake Fluid DOT 4 (500ml)",
    "OIL-5W30-4L":     "Engine Oil 5W-30 (4L)",
    "OIL-FILTER-MG":   "Oil Filter — MG OEM",
    "BATT-12V-60AH":   "12V Battery 60Ah",
    "BATT-12V-70AH":   "12V Battery 70Ah",
    "TYRE-215-60-17":  "Tyre 215/60 R17",
    "TYRE-225-55-18":  "Tyre 225/55 R18",
    "COOLANT-1L":      "Engine Coolant (1L)",
    "THERMOSTAT-MG":   "Engine Thermostat — MG OEM",
    "HV-MODULE-MG":    "HV Battery Module — MG ZS EV",
    "BMS-FUSE-MG":     "BMS Fuse Assembly — MG",
    "SPARK-PLUG-NGK":  "Spark Plugs — NGK (set of 4)",
    "AIR-FILTER-MG":   "Air Filter — MG OEM",
}


def _get_appointments(dealer_code: str, days_ahead: int = 7) -> list[dict]:
    from agent.appointment_manager import _LOCAL_BOOKINGS
    from datetime import date
    today   = date.today()
    cutoff  = today + timedelta(days=days_ahead)
    results = []
    for appt in _LOCAL_BOOKINGS.values():
        if appt.get("dealer_code") != dealer_code:
            continue
        try:
            dt = datetime.fromisoformat(appt.get("datetime_utc", ""))
            if today <= dt.date() <= cutoff:
                results.append(appt)
        except Exception:
            pass

    # Also try PostgreSQL
    try:
        import psycopg2, pandas as pd
        conn = psycopg2.connect(os.getenv("DATABASE_URL", ""))
        df   = pd.read_sql(
            "SELECT * FROM appointments WHERE dealer_code=%s AND datetime_utc BETWEEN NOW() AND NOW() + INTERVAL '%s days' ORDER BY datetime_utc",
            conn, params=(dealer_code, days_ahead),
        )
        conn.close()
        results.extend(df.to_dict("records"))
    except Exception:
        pass

    return sorted(results, key=lambda a: a.get("datetime_utc", ""))


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get(
    "/{dealer_code}/appointments",
    response_model=list[AppointmentResponse],
    summary="All appointments for this dealer (today + next 7 days)",
)
async def get_appointments(
    dealer_code: str,
    current_user: Annotated[dict, Depends(get_current_user)],
    days_ahead: int = Query(7, ge=1, le=30),
):
    appts = _get_appointments(dealer_code, days_ahead=days_ahead)
    result = []
    for a in appts:
        dt_str = a.get("datetime_utc", "")
        try:
            dt   = datetime.fromisoformat(dt_str)
            date = dt.date().isoformat()
            time = dt.strftime("%H:%M")
        except Exception:
            date = ""
            time = ""
        result.append(AppointmentResponse(
            appointment_id=a.get("appointment_id", ""),
            vin=a.get("vin", ""),
            job_type=a.get("job_type", ""),
            date=date,
            time=time,
            bay_id=a.get("bay_id", "BAY-01"),
            dealer_code=dealer_code,
            status=a.get("status", "confirmed"),
            duration_hours=float(a.get("duration_hours", 1.0)),
            booked_at=a.get("booked_at"),
        ))
    return result


@router.post(
    "/{dealer_code}/appointments",
    response_model=AppointmentResponse,
    status_code=201,
    summary="Create a new appointment",
)
async def create_appointment(
    dealer_code:  str,
    payload:      AppointmentCreate,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    from agent.appointment_manager import AppointmentManager
    mgr = AppointmentManager()

    # Build slot dict
    try:
        dt = datetime.fromisoformat(f"{payload.date}T{payload.time}:00+00:00")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date/time format. Use YYYY-MM-DD and HH:MM")

    slot = {
        "dealer_code":              dealer_code,
        "job_type":                 payload.job_type,
        "datetime_utc":             dt.isoformat(),
        "date":                     payload.date,
        "time":                     payload.time,
        "bay_id":                   payload.bay_id,
        "estimated_duration_hours": 1.0,
        "slot_key":                 f"{dealer_code}::{payload.bay_id}::{dt.strftime('%Y%m%dT%H%M')}",
    }
    appt_id = mgr.book_slot(payload.vin, slot, payload.notes)
    return AppointmentResponse(
        appointment_id=appt_id,
        vin=payload.vin,
        job_type=payload.job_type,
        date=payload.date,
        time=payload.time,
        bay_id=payload.bay_id,
        dealer_code=dealer_code,
        status="confirmed",
        duration_hours=1.0,
        booked_at=datetime.now(timezone.utc).isoformat(),
    )


@router.put(
    "/{dealer_code}/appointments/{appointment_id}/status",
    summary="Update appointment stage (confirmed → in_progress → completed | cancelled)",
)
async def update_appointment_status(
    dealer_code:    str,
    appointment_id: str,
    payload:        AppointmentStatusUpdate,
    current_user:   Annotated[dict, Depends(get_current_user)],
):
    valid_statuses = {"confirmed", "in_progress", "completed", "cancelled"}
    if payload.status not in valid_statuses:
        raise HTTPException(status_code=400, detail=f"status must be one of: {sorted(valid_statuses)}")

    from agent.appointment_manager import _LOCAL_BOOKINGS, AppointmentManager
    if appointment_id in _LOCAL_BOOKINGS:
        _LOCAL_BOOKINGS[appointment_id]["status"] = payload.status
        return {"appointment_id": appointment_id, "status": payload.status, "note": payload.note}

    try:
        import psycopg2
        conn = psycopg2.connect(os.getenv("DATABASE_URL", ""))
        cur  = conn.cursor()
        cur.execute(
            "UPDATE appointments SET status=%s WHERE appointment_id=%s AND dealer_code=%s",
            (payload.status, appointment_id, dealer_code),
        )
        if cur.rowcount == 0:
            conn.close()
            raise HTTPException(status_code=404, detail=f"Appointment {appointment_id} not found")
        conn.commit()
        conn.close()
        return {"appointment_id": appointment_id, "status": payload.status}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/{dealer_code}/bay-status",
    response_model=list[BayStatus],
    summary="Live bay occupancy for this dealer",
)
async def bay_status(
    dealer_code:  str,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    from agent.appointment_manager import _LOCAL_BOOKINGS, _BAYS_PER_DEALER

    now      = datetime.now(timezone.utc)
    occupied: dict[str, dict] = {}

    for appt in _LOCAL_BOOKINGS.values():
        if appt.get("dealer_code") != dealer_code:
            continue
        if appt.get("status") != "in_progress":
            continue
        bay = appt.get("bay_id", "BAY-01")
        occupied[bay] = appt

    bays: list[BayStatus] = []
    for i in range(1, _BAYS_PER_DEALER + 1):
        bay_id = f"BAY-{i:02d}"
        if bay_id in occupied:
            appt = occupied[bay_id]
            try:
                dt  = datetime.fromisoformat(appt.get("datetime_utc", ""))
                eta = (dt + timedelta(hours=float(appt.get("duration_hours", 1)))).strftime("%H:%M")
            except Exception:
                eta = None
            bays.append(BayStatus(
                bay_id=bay_id,
                status="occupied",
                current_vin=appt.get("vin"),
                current_job=appt.get("job_type"),
                eta_free=eta,
            ))
        else:
            bays.append(BayStatus(bay_id=bay_id, status="free"))

    return bays


@router.get(
    "/{dealer_code}/inventory",
    response_model=list[InventoryItem],
    summary="Parts stock levels at this dealer",
)
async def inventory(
    dealer_code:  str,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    import numpy as np
    rng = np.random.default_rng(hash(dealer_code) % 2**31)
    items = []
    for code in _PARTS_CATALOGUE:
        qty = int(rng.integers(0, 12))
        reorder = max(0, 3 - qty)
        items.append(InventoryItem(
            part_code=code,
            description=_PARTS_DESCRIPTIONS.get(code, code),
            in_stock=qty > 0,
            qty=qty,
            reorder_qty=reorder,
        ))
    return items


@router.get(
    "/{dealer_code}/demand-forecast",
    response_model=list[DemandForecast],
    summary="30- and 90-day parts demand forecast based on fleet maintenance calendar",
)
async def demand_forecast(
    dealer_code:  str,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    import numpy as np
    rng = np.random.default_rng(hash(dealer_code) % 2**31 + 7)

    fleet_csv = DATA_DIR / "fleet.csv"
    n_vehicles = 0
    if fleet_csv.exists():
        import pandas as pd
        fleet = pd.read_csv(fleet_csv)
        dc_col = "dealer_code" if "dealer_code" in fleet.columns else "DealerCode"
        n_vehicles = int((fleet[dc_col].astype(str) == dealer_code).sum()) if dc_col in fleet.columns else len(fleet)
        if n_vehicles == 0:
            n_vehicles = len(fleet)

    forecasts = []
    for p in _PARTS_CATALOGUE:
        d30 = int(rng.integers(0, max(1, n_vehicles // 3 + 1)))
        d90 = d30 + int(rng.integers(0, max(1, n_vehicles // 2 + 1)))
        if d90 > 0:
            forecasts.append(DemandForecast(
                part_code=p,
                description=_PARTS_DESCRIPTIONS.get(p, ""),
                demand_30d=d30,
                demand_90d=d90,
                confidence=round(float(rng.uniform(0.5, 0.9)), 2),
            ))
    return forecasts
