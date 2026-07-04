"""
AutoPredict Appointment Manager.

AppointmentManager:
  get_available_slots(dealer_code, job_type, days_ahead=7)
    → List[{date, time, bay_id, estimated_duration_hours}]
  book_slot(vin, slot, job_card_template)
    → appointment_id
  cancel_slot(appointment_id)
  get_parts_availability(parts_list, dealer_code)
    → {part_code: {in_stock: bool, qty: int, eta_days: int}}

Appointments are stored in PostgreSQL (with in-process dict fallback).
"""
from __future__ import annotations

import logging
import os
import random
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

log = logging.getLogger(__name__)

PG_DSN = os.getenv("DATABASE_URL", "")

# Alert-type → estimated service duration hours
_JOB_DURATIONS: dict[str, float] = {
    "THERMAL_RUNAWAY":          12.0,
    "LOW_BRAKE_FLUID":           0.5,
    "OIL_PRESSURE_CRITICAL":     4.0,
    "ENGINE_OVERTEMP":           3.0,
    "BRAKE_PAD_CRITICAL":        1.5,
    "BRAKE_PAD_WARNING":         1.5,
    "12V_BATTERY_CRITICAL":      0.5,
    "12V_BATTERY_RISK":          0.5,
    "TYRE_DEFLATION":            0.5,
    "BMS_PACK_TEMP_WARNING":     3.0,
    "HV_BATTERY_SOH_CRITICAL":   8.0,
    "CELL_VOLTAGE_IMBALANCE":    3.0,
    "ENGINE_OIL_LOW":            0.5,
    "ENGINE_OIL_ADVISORY":       0.5,
    "ML_BRAKE_REPLACEMENT":      2.0,
    "ML_OIL_CHANGE_DUE":         0.5,
    "ML_12V_FAILURE_RISK":       0.5,
    "ML_TYRE_REPLACEMENT":       1.5,
    "ML_PUNCTURE_DETECTED":      0.5,
    "ML_HV_SOH_DECLINE":         4.0,
    "ML_CELL_ANOMALY":           4.0,
    "ML_FUEL_ANOMALY":           1.5,
    "DEFAULT":                   1.0,
}

# Number of service bays per dealer (simulated)
_BAYS_PER_DEALER = 6
_WORKING_HOURS   = list(range(9, 18))  # 09:00–17:00

# In-process booking store (fallback when PostgreSQL unavailable)
_LOCAL_BOOKINGS: dict[str, dict] = {}

# Simulated spare-parts inventory seed (deterministic via dealer_code hash)
_PARTS_CATALOGUE: list[str] = [
    "BR-PAD-F-MG",   # Brake pads front
    "BR-PAD-R-MG",   # Brake pads rear
    "BR-FLUID-DOT4",  # Brake fluid DOT4
    "OIL-5W30-4L",   # Engine oil 5W30
    "OIL-FILTER-MG", # Oil filter
    "BATT-12V-60AH", # 12V battery 60Ah
    "BATT-12V-70AH", # 12V battery 70Ah
    "TYRE-215-60-17", # Tyre 215/60 R17
    "TYRE-225-55-18", # Tyre 225/55 R18
    "COOLANT-1L",    # Engine coolant
    "THERMOSTAT-MG", # Thermostat
    "HV-MODULE-MG",  # HV battery module
    "BMS-FUSE-MG",   # BMS fuse
    "SPARK-PLUG-NGK", # Spark plugs
]


class AppointmentManager:
    """
    Manages service appointment slots and parts pre-order for AutoPredict.

    Methods use PostgreSQL when DATABASE_URL is set; fall back to an
    in-process dict store for local/offline development.
    """

    # ── Slot finding ──────────────────────────────────────────────────────────

    def get_available_slots(
        self,
        dealer_code: str,
        job_type: str,
        days_ahead: int = 7,
    ) -> list[dict]:
        """
        Return up to (bays × days_ahead) available slots at *dealer_code*
        for *job_type*, spread across the next *days_ahead* working days.

        Each slot: {date, time, bay_id, estimated_duration_hours}
        """
        duration_h = _JOB_DURATIONS.get(job_type, _JOB_DURATIONS["DEFAULT"])
        booked_keys = self._booked_slot_keys(dealer_code)

        now  = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        base = now.replace(hour=0, minute=0)

        slots: list[dict] = []
        for day_offset in range(1, days_ahead + 1):
            day = base + timedelta(days=day_offset)
            if day.weekday() == 6:  # skip Sunday (MG dealerships closed)
                continue
            for bay in range(1, _BAYS_PER_DEALER + 1):
                for hour in _WORKING_HOURS:
                    # Skip hours that don't allow the job to finish by 18:00
                    if hour + duration_h > 18:
                        continue
                    slot_dt  = day.replace(hour=hour)
                    slot_key = _slot_key(dealer_code, bay, slot_dt)
                    if slot_key not in booked_keys:
                        slots.append({
                            "date":                       slot_dt.date().isoformat(),
                            "time":                       slot_dt.strftime("%H:%M"),
                            "datetime_utc":               slot_dt.isoformat(),
                            "bay_id":                     f"BAY-{bay:02d}",
                            "dealer_code":                dealer_code,
                            "job_type":                   job_type,
                            "estimated_duration_hours":   duration_h,
                            "slot_key":                   slot_key,
                        })
                        if len(slots) >= 12:
                            return slots
        return slots

    # ── Booking ───────────────────────────────────────────────────────────────

    def book_slot(
        self,
        vin: str,
        slot: dict,
        job_card_template: str = "",
    ) -> str:
        """
        Confirm a booking for *vin* at the given *slot* dict.

        Returns: appointment_id (UUID string)
        """
        appt_id = str(uuid.uuid4())
        record  = {
            "appointment_id":     appt_id,
            "vin":                vin,
            "dealer_code":        slot.get("dealer_code", ""),
            "bay_id":             slot.get("bay_id", ""),
            "job_type":           slot.get("job_type", ""),
            "datetime_utc":       slot.get("datetime_utc", ""),
            "duration_hours":     slot.get("estimated_duration_hours", 1.0),
            "status":             "confirmed",
            "job_card":           job_card_template,
            "booked_at":          datetime.now(timezone.utc).isoformat(),
            "slot_key":           slot.get("slot_key", ""),
        }
        _LOCAL_BOOKINGS[appt_id] = record
        self._persist_booking(record)
        log.info("Appointment booked: %s VIN=%s %s %s", appt_id, vin, slot.get("date"), slot.get("time"))
        return appt_id

    def cancel_slot(self, appointment_id: str) -> bool:
        """Cancel appointment by ID. Returns True if found and cancelled."""
        if appointment_id in _LOCAL_BOOKINGS:
            _LOCAL_BOOKINGS[appointment_id]["status"] = "cancelled"
            self._cancel_in_db(appointment_id)
            log.info("Appointment cancelled: %s", appointment_id)
            return True
        # try DB
        return self._cancel_in_db(appointment_id)

    def get_appointment(self, appointment_id: str) -> dict | None:
        if appointment_id in _LOCAL_BOOKINGS:
            return _LOCAL_BOOKINGS[appointment_id]
        return self._load_from_db(appointment_id)

    # ── Parts availability ────────────────────────────────────────────────────

    def get_parts_availability(
        self,
        parts_list: list[str],
        dealer_code: str,
    ) -> dict[str, dict]:
        """
        Return availability for each part code at the given dealer.

        {part_code: {in_stock: bool, qty: int, eta_days: int}}

        Uses a deterministic pseudo-random seed so results are consistent
        for the same dealer × part combination (good for demos).
        """
        result: dict[str, dict] = {}
        for part_code in parts_list:
            seed = hash(f"{dealer_code}:{part_code}") % (2**31)
            rng  = random.Random(seed)
            qty  = rng.randint(0, 5)
            result[part_code] = {
                "part_code":  part_code,
                "in_stock":   qty > 0,
                "qty":        qty,
                "eta_days":   0 if qty > 0 else rng.randint(2, 7),
                "dealer_code": dealer_code,
            }
        return result

    def suggest_parts(self, alert_type: str) -> list[str]:
        """Return a list of likely-needed part codes for an alert type."""
        _ALERT_PARTS: dict[str, list[str]] = {
            "BRAKE_PAD_CRITICAL":    ["BR-PAD-F-MG", "BR-PAD-R-MG"],
            "BRAKE_PAD_WARNING":     ["BR-PAD-F-MG", "BR-PAD-R-MG"],
            "ML_BRAKE_REPLACEMENT":  ["BR-PAD-F-MG", "BR-PAD-R-MG"],
            "LOW_BRAKE_FLUID":       ["BR-FLUID-DOT4"],
            "ENGINE_OIL_LOW":        ["OIL-5W30-4L", "OIL-FILTER-MG"],
            "ENGINE_OIL_ADVISORY":   ["OIL-5W30-4L", "OIL-FILTER-MG"],
            "ML_OIL_CHANGE_DUE":     ["OIL-5W30-4L", "OIL-FILTER-MG"],
            "ML_OIL_ADVISORY":       ["OIL-5W30-4L", "OIL-FILTER-MG"],
            "12V_BATTERY_CRITICAL":  ["BATT-12V-60AH"],
            "12V_BATTERY_RISK":      ["BATT-12V-60AH"],
            "ML_12V_FAILURE_RISK":   ["BATT-12V-60AH"],
            "TYRE_DEFLATION":        ["TYRE-215-60-17"],
            "ML_TYRE_REPLACEMENT":   ["TYRE-215-60-17"],
            "ML_PUNCTURE_DETECTED":  ["TYRE-215-60-17"],
            "ENGINE_OVERTEMP":       ["COOLANT-1L", "THERMOSTAT-MG"],
            "HV_BATTERY_SOH_CRITICAL": ["HV-MODULE-MG"],
            "ML_HV_SOH_DECLINE":     ["HV-MODULE-MG"],
            "CELL_VOLTAGE_IMBALANCE":  ["BMS-FUSE-MG"],
        }
        key = alert_type.replace("ML_", "")
        return _ALERT_PARTS.get(alert_type, _ALERT_PARTS.get(key, []))

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _booked_slot_keys(self, dealer_code: str) -> set[str]:
        """Collect all booked slot_key values for this dealer (local + DB)."""
        keys = {
            b["slot_key"]
            for b in _LOCAL_BOOKINGS.values()
            if b.get("dealer_code") == dealer_code and b.get("status") == "confirmed"
        }
        try:
            import psycopg2
            if not PG_DSN:
                return keys
            conn = psycopg2.connect(PG_DSN)
            cur  = conn.cursor()
            cur.execute("SELECT slot_key FROM appointments WHERE dealer_code=%s AND status='confirmed'", (dealer_code,))
            keys.update(r[0] for r in cur.fetchall())
            conn.close()
        except Exception:
            pass
        return keys

    def _persist_booking(self, record: dict) -> None:
        try:
            import psycopg2
            if not PG_DSN:
                return
            conn = psycopg2.connect(PG_DSN)
            cur  = conn.cursor()
            cur.execute(
                """
                INSERT INTO appointments
                  (appointment_id, vin, dealer_code, bay_id, job_type,
                   datetime_utc, duration_hours, status, job_card, booked_at, slot_key)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (appointment_id) DO NOTHING
                """,
                (record["appointment_id"], record["vin"], record["dealer_code"],
                 record["bay_id"], record["job_type"], record["datetime_utc"],
                 record["duration_hours"], record["status"], record["job_card"],
                 record["booked_at"], record["slot_key"]),
            )
            conn.commit()
            conn.close()
        except Exception as exc:
            log.debug("Appointment persist failed: %s", exc)

    def _cancel_in_db(self, appointment_id: str) -> bool:
        try:
            import psycopg2
            if not PG_DSN:
                return False
            conn = psycopg2.connect(PG_DSN)
            cur  = conn.cursor()
            cur.execute(
                "UPDATE appointments SET status='cancelled' WHERE appointment_id=%s",
                (appointment_id,),
            )
            affected = cur.rowcount
            conn.commit()
            conn.close()
            return affected > 0
        except Exception as exc:
            log.debug("Appointment cancel in DB failed: %s", exc)
            return False

    def _load_from_db(self, appointment_id: str) -> dict | None:
        try:
            import psycopg2
            if not PG_DSN:
                return None
            conn = psycopg2.connect(PG_DSN)
            cur  = conn.cursor()
            cur.execute("SELECT * FROM appointments WHERE appointment_id=%s", (appointment_id,))
            row = cur.fetchone()
            if row:
                cols = [d[0] for d in cur.description]
                return dict(zip(cols, row))
            conn.close()
        except Exception as exc:
            log.debug("Appointment load from DB failed: %s", exc)
        return None


def _slot_key(dealer_code: str, bay: int, dt: datetime) -> str:
    return f"{dealer_code}::BAY{bay:02d}::{dt.strftime('%Y%m%dT%H%M')}"


# ── Backward-compat function wrappers (used by old agent stub) ─────────────────

def find_available_slots(dealer_id: str, service_type: str, days_ahead: int = 7) -> list[dict]:
    return AppointmentManager().get_available_slots(dealer_id, service_type, days_ahead)


def book_appointment(dealer_id: str, vin: str, service_type: str, slot_datetime: str, customer_name: str) -> dict:
    mgr  = AppointmentManager()
    slot = {
        "dealer_code": dealer_id,
        "job_type": service_type,
        "datetime_utc": slot_datetime,
        "bay_id": "BAY-01",
        "estimated_duration_hours": 1.0,
        "slot_key": _slot_key(dealer_id, 1, datetime.fromisoformat(slot_datetime)),
    }
    appt_id = mgr.book_slot(vin, slot, "")
    return {"success": True, "appointment_id": appt_id, "vin": vin,
            "slot_datetime": slot_datetime, "customer_name": customer_name}
