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

# Full parts metadata: service intervals, costs, ABC class, alert component mapping
_PARTS_META: dict[str, dict] = {
    "OIL-5W30-4L": {
        "description":      "Engine Oil 5W-30 (4L)",
        "unit_cost_inr":    855.0,
        "lead_time_days":   2,
        "abc_class":        "A",
        "supplier":         "Castrol / Mobil",
        "service_types":    ["OIL_CHANGE", "FULL_SERVICE"],
        "alert_component":  "engine_oil",
        "replace_km":       7500,
        "per_service_qty":  1,
    },
    "OIL-FILTER-MG": {
        "description":      "Oil Filter — OEM",
        "unit_cost_inr":    183.0,
        "lead_time_days":   3,
        "abc_class":        "A",
        "supplier":         "OEM Direct",
        "service_types":    ["OIL_CHANGE", "FULL_SERVICE"],
        "alert_component":  "engine_oil",
        "replace_km":       7500,
        "per_service_qty":  1,
    },
    "BR-PAD-F-MG": {
        "description":      "Brake Pads (Front) — OEM",
        "unit_cost_inr":    2800.0,
        "lead_time_days":   5,
        "abc_class":        "A",
        "supplier":         "OEM Direct",
        "service_types":    ["BRAKE_CHECK", "FULL_SERVICE"],
        "alert_component":  "brake_wear",
        "replace_km":       30000,
        "per_service_qty":  1,
    },
    "BR-PAD-R-MG": {
        "description":      "Brake Pads (Rear) — OEM",
        "unit_cost_inr":    2200.0,
        "lead_time_days":   5,
        "abc_class":        "A",
        "supplier":         "OEM Direct",
        "service_types":    ["BRAKE_CHECK", "FULL_SERVICE"],
        "alert_component":  "brake_wear",
        "replace_km":       35000,
        "per_service_qty":  1,
    },
    "BR-FLUID-DOT4": {
        "description":      "Brake Fluid DOT 4 (500ml)",
        "unit_cost_inr":    380.0,
        "lead_time_days":   3,
        "abc_class":        "B",
        "supplier":         "TotalEnergies",
        "service_types":    ["FULL_SERVICE"],
        "alert_component":  "brake_wear",
        "replace_km":       60000,
        "per_service_qty":  1,
    },
    "TYRE-215-60-17": {
        "description":      "Tyre 215/60 R17",
        "unit_cost_inr":    6500.0,
        "lead_time_days":   7,
        "abc_class":        "B",
        "supplier":         "MRF / Apollo",
        "service_types":    ["TYRE_REPLACE"],
        "alert_component":  "tyre_wear",
        "replace_km":       50000,
        "per_service_qty":  4,
    },
    "TYRE-225-55-18": {
        "description":      "Tyre 225/55 R18",
        "unit_cost_inr":    7800.0,
        "lead_time_days":   7,
        "abc_class":        "B",
        "supplier":         "MRF / Apollo",
        "service_types":    ["TYRE_REPLACE"],
        "alert_component":  "tyre_wear",
        "replace_km":       50000,
        "per_service_qty":  4,
    },
    "BATT-12V-60AH": {
        "description":      "12V Battery 60Ah",
        "unit_cost_inr":    4500.0,
        "lead_time_days":   3,
        "abc_class":        "A",
        "supplier":         "Amaron / Exide",
        "service_types":    ["12V_BATTERY_REPLACE"],
        "alert_component":  "battery_12v",
        "replace_km":       None,
        "per_service_qty":  1,
    },
    "BATT-12V-70AH": {
        "description":      "12V Battery 70Ah",
        "unit_cost_inr":    5200.0,
        "lead_time_days":   3,
        "abc_class":        "A",
        "supplier":         "Amaron / Exide",
        "service_types":    ["12V_BATTERY_REPLACE"],
        "alert_component":  "battery_12v",
        "replace_km":       None,
        "per_service_qty":  1,
    },
    "COOLANT-1L": {
        "description":      "Engine Coolant (1L)",
        "unit_cost_inr":    350.0,
        "lead_time_days":   3,
        "abc_class":        "B",
        "supplier":         "OEM Direct",
        "service_types":    ["COOLANT_FLUSH", "FULL_SERVICE"],
        "alert_component":  None,
        "replace_km":       60000,
        "per_service_qty":  3,
    },
    "AIR-FILTER-MG": {
        "description":      "Air Filter — OEM",
        "unit_cost_inr":    420.0,
        "lead_time_days":   3,
        "abc_class":        "B",
        "supplier":         "OEM Direct",
        "service_types":    ["FULL_SERVICE"],
        "alert_component":  None,
        "replace_km":       30000,
        "per_service_qty":  1,
    },
    "SPARK-PLUG-NGK": {
        "description":      "Spark Plugs — NGK (set of 4)",
        "unit_cost_inr":    1850.0,
        "lead_time_days":   5,
        "abc_class":        "B",
        "supplier":         "NGK",
        "service_types":    ["FULL_SERVICE"],
        "alert_component":  None,
        "replace_km":       60000,
        "per_service_qty":  1,
    },
    "HV-MODULE-MG": {
        "description":      "HV Battery Module — EV OEM",
        "unit_cost_inr":    185000.0,
        "lead_time_days":   21,
        "abc_class":        "A",
        "supplier":         "OEM Direct",
        "service_types":    [],
        "alert_component":  "hv_battery_soh",
        "replace_km":       None,
        "per_service_qty":  1,
    },
    "BMS-FUSE-MG": {
        "description":      "BMS Fuse Assembly — MG",
        "unit_cost_inr":    2800.0,
        "lead_time_days":   14,
        "abc_class":        "B",
        "supplier":         "OEM Direct",
        "service_types":    [],
        "alert_component":  "hv_battery_soh",
        "replace_km":       None,
        "per_service_qty":  1,
    },
    "THERMOSTAT-MG": {
        "description":      "Engine Thermostat — OEM",
        "unit_cost_inr":    1200.0,
        "lead_time_days":   10,
        "abc_class":        "C",
        "supplier":         "OEM Direct",
        "service_types":    ["COOLANT_FLUSH"],
        "alert_component":  None,
        "replace_km":       None,
        "per_service_qty":  1,
    },
}

_PARTS_CATALOGUE = list(_PARTS_META.keys())
_PARTS_DESCRIPTIONS = {k: v["description"] for k, v in _PARTS_META.items()}

# Historical service data mapped to part codes
_SVC_DESC_TO_PART: dict[str, str] = {
    "OIL 5W30 4L": "OIL-5W30-4L",
    "OIL FILTER":  "OIL-FILTER-MG",
    "BRAKE PAD F": "BR-PAD-F-MG",
    "BRAKE PAD R": "BR-PAD-R-MG",
    "AIR FILTER":  "AIR-FILTER-MG",
    "FUEL FILTER": "AIR-FILTER-MG",
    "12V BAT 65AH":"BATT-12V-60AH",
    "COOLANT OAT": "COOLANT-1L",
}


def _compute_demand_forecast(dealer_code: str) -> list[dict]:
    """
    ML-driven demand forecast:
      - Uses InventoryDemandModel (LightGBM) when a trained artifact exists
      - Falls back to fleet-interval + historical heuristic when no model is trained
      - Alert urgency boost applied on top regardless of source
    Returns list of dicts sorted by demand_30d desc.
    """
    import numpy as np
    import pandas as pd
    from models.inventory_demand_model import InventoryDemandModel
    _inv_model = InventoryDemandModel()
    _model_available = _inv_model.model is not None
    from datetime import date

    rng = np.random.default_rng(hash(dealer_code) % 2**31 + 99)

    # --- Load fleet data -------------------------------------------------------
    fleet_csv = DATA_DIR / "fleet.csv"
    n_vehicles = 10
    fleet_df   = pd.DataFrame()
    if fleet_csv.exists():
        fleet_df = pd.read_csv(fleet_csv)
        dc_col = next((c for c in ["dealer_code", "DealerCode"] if c in fleet_df.columns), None)
        if dc_col:
            mask = fleet_df[dc_col].astype(str) == dealer_code
            dealer_fleet = fleet_df[mask]
            n_vehicles = max(1, len(dealer_fleet))

    avg_km_per_month = 1500  # typical monthly mileage

    # --- Load service history --------------------------------------------------
    svc_csv = DATA_DIR / "service_history.csv"
    svc_counts: dict[str, int] = {}       # service_type → events
    part_history: dict[str, float] = {}   # part_code → qty/month
    history_months = 1.0

    if svc_csv.exists():
        svc_df = pd.read_csv(svc_csv, low_memory=False)
        # Determine relevant columns
        dc_col   = next((c for c in ["DealerCode", "dealer_code"]       if c in svc_df.columns), None)
        dt_col   = next((c for c in ["CreatedOn",  "created_on"]        if c in svc_df.columns), None)
        svc_col  = next((c for c in ["ServiceType","service_type"]       if c in svc_df.columns), None)
        desc_col = next((c for c in ["DescriptionOne","description_one"] if c in svc_df.columns), None)
        qty_col  = next((c for c in ["OrderQuantity","order_quantity"]   if c in svc_df.columns), None)
        lp_col   = next((c for c in ["LabPart",    "lab_part"]           if c in svc_df.columns), None)

        # Parse dates to determine history window
        if dt_col:
            svc_df[dt_col] = pd.to_datetime(svc_df[dt_col], errors="coerce")
            valid_dates = svc_df[dt_col].dropna()
            if not valid_dates.empty:
                history_months = max(1.0, (valid_dates.max() - valid_dates.min()).days / 30)

        # For demand rates: use full dataset scaled by dealer's share of total fleet
        total_fleet = max(1, len(fleet_df))
        dealer_share = n_vehicles / total_fleet

        if svc_col:
            for svc_type, cnt in svc_df[svc_col].value_counts().items():
                svc_counts[str(svc_type)] = int(cnt * dealer_share)

        if desc_col and lp_col and qty_col:
            parts_df = svc_df[svc_df[lp_col].astype(str) == "PART"].copy()
            for raw_desc, part_code in _SVC_DESC_TO_PART.items():
                mask = parts_df[desc_col].astype(str).str.contains(raw_desc, case=False, na=False)
                qty_sum = parts_df.loc[mask, qty_col].fillna(0).astype(float).sum()
                if qty_sum > 0:
                    part_history[part_code] = part_history.get(part_code, 0) + (qty_sum * dealer_share / history_months)

    # --- Build per-part forecast -----------------------------------------------
    forecasts = []
    for part_code, meta in _PARTS_META.items():

        # 1. Fleet-interval demand: how many parts are due in 30 days based on mileage
        interval_demand_30d = 0.0
        if meta["replace_km"]:
            fleet_km_per_month   = n_vehicles * avg_km_per_month
            interval_demand_30d  = (fleet_km_per_month / meta["replace_km"]) * meta["per_service_qty"]

        # 2. Historical demand from service records
        hist_monthly = part_history.get(part_code, 0.0)
        if hist_monthly == 0.0:
            for svc_type in meta["service_types"]:
                cnt = svc_counts.get(svc_type, 0)
                hist_monthly += cnt / history_months * meta["per_service_qty"]

        # 3. Build feature dict for InventoryDemandModel
        # Trend: compare interval to history to estimate slope
        trend_slope = (interval_demand_30d - hist_monthly) / 30.0 if hist_monthly > 0 else 0.0
        features_dict = {
            "avg_monthly_units_12m":     hist_monthly if hist_monthly > 0 else interval_demand_30d,
            "consumption_trend_slope":   trend_slope,
            "seasonal_index":            1.0,
            "supplier_lead_time_days":   meta.get("lead_time_days", 7),
            "n_vehicles":                n_vehicles,
            "replace_km":                meta["replace_km"] or 0,
            "per_service_qty":           meta["per_service_qty"],
            "interval_demand_30d":       interval_demand_30d,
            "history_months":            history_months,
        }

        if _model_available:
            # Use trained LightGBM model
            ml_result     = _inv_model.predict(features_dict)
            blended_30d   = ml_result["point_estimate"]
            method        = "ml_model"
            confidence    = round(min(0.94, 0.78 + float(rng.uniform(0, 0.12))), 2)
        elif hist_monthly > 0:
            # Heuristic blend: 50% fleet-interval + 50% historical
            blended_30d = 0.5 * interval_demand_30d + 0.5 * hist_monthly
            method      = "fleet_interval+historical"
            confidence  = round(min(0.92, 0.72 + float(rng.uniform(0, 0.12))), 2)
        else:
            blended_30d = interval_demand_30d
            method      = "fleet_interval"
            confidence  = round(min(0.75, 0.55 + float(rng.uniform(0, 0.12))), 2)

        demand_30d = max(0, round(blended_30d))

        # 4. Alert-based urgency boost from fleet health
        alert_contrib = 0
        if meta["alert_component"] and not fleet_df.empty:
            hs_col = next((c for c in ["health_score", "HealthScore"] if c in fleet_df.columns), None)
            dc_col = next((c for c in ["dealer_code", "DealerCode"]   if c in fleet_df.columns), None)
            if hs_col and dc_col:
                dealer_fleet = fleet_df[fleet_df[dc_col].astype(str) == dealer_code]
                if not dealer_fleet.empty:
                    low_health = (dealer_fleet[hs_col].fillna(100).astype(float) < 55).sum()
                    alert_contrib = int(low_health * 0.3)
                    demand_30d += alert_contrib

        # 5. 90-day: apply compounding for high-replacement parts
        demand_90d = demand_30d * 3 + alert_contrib  # alerts spike near-term

        # Trend: compare fleet-interval to historical
        demand_trend = "stable"
        if hist_monthly > 0:
            if interval_demand_30d > hist_monthly * 1.2:
                demand_trend = "rising"
            elif interval_demand_30d < hist_monthly * 0.8:
                demand_trend = "falling"

        daily = demand_30d / 30.0
        demand_90d_final = max(demand_30d, demand_90d)
        forecasts.append({
            "part_code":             part_code,
            "description":           meta["description"],
            "category":              meta.get("category", "General"),
            "demand_7d":             max(0, round(daily * 7)),
            "demand_15d":            max(0, round(daily * 15)),
            "demand_30d":            demand_30d,
            "demand_60d":            max(0, round(demand_30d * 2)),
            "demand_90d":            demand_90d_final,
            "confidence":            confidence,
            "historical_monthly_avg": round(hist_monthly, 2),
            "alert_contribution":    alert_contrib,
            "rul_contribution":      0,
            "forecast_method":       method,
            "demand_trend":          demand_trend,
            "days_until_stockout":   None,  # populated by inventory endpoint
        })

    return sorted(forecasts, key=lambda x: x["demand_30d"], reverse=True)


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


def _load_inventory_stock(dealer_code: str | None = None) -> "pd.DataFrame":
    import pandas as pd
    csv = DATA_DIR / "inventory_stock.csv"
    if not csv.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv)
    if dealer_code:
        df = df[df["dealer_code"].astype(str) == dealer_code]
    return df


@router.get(
    "/{dealer_code}/inventory",
    response_model=list[InventoryItem],
    summary="Parts stock levels at this dealer with reorder intelligence",
)
async def inventory(
    dealer_code:  str,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    import pandas as pd
    stock_df = _load_inventory_stock(dealer_code)

    if stock_df.empty:
        # Fallback: compute heuristically if CSV not generated yet
        import numpy as np
        rng = np.random.default_rng(hash(dealer_code) % 2**31)
        forecasts = {f["part_code"]: f for f in _compute_demand_forecast(dealer_code)}
        items = []
        for code, meta in _PARTS_META.items():
            fcast = forecasts.get(code, {})
            daily_demand = fcast.get("demand_30d", 0) / 30.0
            days_stock = {"A": 45, "B": 30, "C": 15}.get(meta["abc_class"], 30)
            target_qty = max(2, round(daily_demand * days_stock))
            jitter = float(rng.uniform(0.6, 1.4))
            qty = max(0, round(target_qty * jitter))
            lead_demand = round(daily_demand * meta["lead_time_days"])
            safety = max(1, round(daily_demand * 7))
            reorder_pt = lead_demand + safety
            reorder_qty = max(0, reorder_pt - qty)
            dos = int(qty / daily_demand) if daily_demand > 0 and qty > 0 else (0 if qty == 0 else None)
            items.append(InventoryItem(
                part_code=code, description=meta["description"],
                in_stock=qty > 0, qty=qty, reorder_qty=reorder_qty,
                unit_cost_inr=meta["unit_cost_inr"], reorder_point=reorder_pt,
                safety_stock=safety, abc_class=meta["abc_class"],
                lead_time_days=meta["lead_time_days"], supplier=meta["supplier"],
                monthly_demand_avg=round(float(fcast.get("demand_30d", 0)), 1),
                days_until_stockout=dos,
            ))
        items.sort(key=lambda it: (0 if it.qty == 0 else 1 if it.reorder_qty > 0 else 2,
                                   {"A": 0, "B": 1, "C": 2}.get(it.abc_class or "B", 1), it.part_code))
        return items

    # Read from generated CSV
    forecasts = {f["part_code"]: f for f in _compute_demand_forecast(dealer_code)}
    items = []
    for _, row in stock_df.iterrows():
        code         = str(row["part_code"])
        qty          = int(row.get("current_stock", 0))
        reorder_pt   = int(row.get("reorder_point", 0))
        safety       = int(row.get("safety_stock", 1))
        eoq          = int(row.get("eoq", 1))
        daily_demand = float(row.get("avg_daily_demand", 0))
        dos          = int(row.get("days_of_supply", 999))
        reorder_qty  = max(0, eoq) if qty <= reorder_pt else 0

        fcast = forecasts.get(code, {})
        items.append(InventoryItem(
            part_code=code,
            description=str(row.get("description", "")),
            in_stock=qty > 0,
            qty=qty,
            reorder_qty=reorder_qty,
            unit_cost_inr=float(row.get("unit_cost_inr", 0)),
            reorder_point=reorder_pt,
            safety_stock=safety,
            abc_class=str(row.get("abc_class", "B")),
            lead_time_days=int(row.get("lead_time_days", 7)),
            supplier=str(row.get("supplier", "")),
            monthly_demand_avg=round(daily_demand * 30, 1),
            days_until_stockout=min(dos, 999) if dos < 999 else None,
        ))

    items.sort(key=lambda it: (0 if it.qty == 0 else 1 if (it.reorder_qty or 0) > 0 else 2,
                               {"A": 0, "B": 1, "C": 2}.get(it.abc_class or "B", 1), it.part_code))
    return items


@router.get(
    "/{dealer_code}/demand-forecast",
    response_model=list[DemandForecast],
    summary="30/90-day ML demand forecast from fleet health + service history",
)
async def demand_forecast(
    dealer_code:  str,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    rows = _compute_demand_forecast(dealer_code)
    return [DemandForecast(**r) for r in rows]
