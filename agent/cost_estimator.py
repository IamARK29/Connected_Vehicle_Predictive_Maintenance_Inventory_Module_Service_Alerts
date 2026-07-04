"""
AutoPredict Cost Estimator.

CostEstimator.estimate(alert_type, vin, model_code) → cost dict
  - Looks up historical service records for same alert_type + model_code
  - Falls back to catalogue price ranges when no history
  - Warranty check: age < 36 months AND km < 100,000 → warranty_likely = True
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

log = logging.getLogger(__name__)

# ── Per-alert-type cost catalogue (INR, 18% GST included) ──────────────────────

_CATALOGUE: dict[str, dict] = {
    # alert_type → {parts_min, parts_max, labour_min, labour_max, duration_h}
    "THERMAL_RUNAWAY":          {"parts_min": 50_000,  "parts_max": 3_00_000, "labour_min": 15_000, "labour_max": 30_000, "duration_h": 12},
    "LOW_BRAKE_FLUID":          {"parts_min": 800,     "parts_max": 2_000,   "labour_min": 600,    "labour_max": 1_200,  "duration_h": 0.5},
    "OIL_PRESSURE_CRITICAL":    {"parts_min": 5_000,   "parts_max": 60_000,  "labour_min": 3_000,  "labour_max": 20_000, "duration_h": 4},
    "ENGINE_OVERTEMP":          {"parts_min": 2_000,   "parts_max": 35_000,  "labour_min": 1_500,  "labour_max": 10_000, "duration_h": 3},
    "BRAKE_PAD_CRITICAL":       {"parts_min": 2_500,   "parts_max": 8_000,   "labour_min": 1_200,  "labour_max": 2_500,  "duration_h": 1.5},
    "12V_BATTERY_CRITICAL":     {"parts_min": 4_000,   "parts_max": 10_000,  "labour_min": 400,    "labour_max": 800,    "duration_h": 0.5},
    "12V_BATTERY_RISK":         {"parts_min": 4_000,   "parts_max": 10_000,  "labour_min": 400,    "labour_max": 800,    "duration_h": 0.5},
    "TYRE_DEFLATION":           {"parts_min": 400,     "parts_max": 6_000,   "labour_min": 300,    "labour_max": 600,    "duration_h": 0.5},
    "BMS_PACK_TEMP_WARNING":    {"parts_min": 5_000,   "parts_max": 40_000,  "labour_min": 2_000,  "labour_max": 8_000,  "duration_h": 3},
    "HV_BATTERY_SOH_CRITICAL":  {"parts_min": 40_000,  "parts_max": 2_50_000,"labour_min": 10_000, "labour_max": 30_000, "duration_h": 8},
    "CELL_VOLTAGE_IMBALANCE":   {"parts_min": 3_000,   "parts_max": 50_000,  "labour_min": 2_000,  "labour_max": 10_000, "duration_h": 3},
    "CELL_OVERTEMP":            {"parts_min": 5_000,   "parts_max": 30_000,  "labour_min": 2_000,  "labour_max": 8_000,  "duration_h": 2},
    "BRAKE_PAD_WARNING":        {"parts_min": 2_500,   "parts_max": 8_000,   "labour_min": 1_200,  "labour_max": 2_500,  "duration_h": 1.5},
    "ENGINE_OIL_LOW":           {"parts_min": 1_500,   "parts_max": 4_000,   "labour_min": 400,    "labour_max": 800,    "duration_h": 0.5},
    "ENGINE_TEMP_ELEVATED":     {"parts_min": 500,     "parts_max": 10_000,  "labour_min": 500,    "labour_max": 4_000,  "duration_h": 1},
    "HV_SOC_LOW":               {"parts_min": 0,       "parts_max": 0,       "labour_min": 0,      "labour_max": 0,      "duration_h": 0},
    "HV_SOC_CRITICAL":          {"parts_min": 0,       "parts_max": 0,       "labour_min": 0,      "labour_max": 0,      "duration_h": 0},
    "TYRE_PRESSURE_LOW":        {"parts_min": 0,       "parts_max": 500,     "labour_min": 0,      "labour_max": 300,    "duration_h": 0.25},
    "HV_BATTERY_SOH_MEDIUM":    {"parts_min": 0,       "parts_max": 40_000,  "labour_min": 2_000,  "labour_max": 8_000,  "duration_h": 2},
    "ENGINE_OIL_ADVISORY":      {"parts_min": 1_500,   "parts_max": 4_000,   "labour_min": 400,    "labour_max": 800,    "duration_h": 0.5},
    "LOW_FUEL":                 {"parts_min": 0,       "parts_max": 0,       "labour_min": 0,      "labour_max": 0,      "duration_h": 0},
    "FUEL_ADVISORY":            {"parts_min": 0,       "parts_max": 0,       "labour_min": 0,      "labour_max": 0,      "duration_h": 0},
    # ML alert types (map to nearest rule type)
    "ML_BRAKE_REPLACEMENT":     {"parts_min": 2_500,   "parts_max": 10_000,  "labour_min": 1_200,  "labour_max": 3_000,  "duration_h": 2},
    "ML_BRAKE_WARNING":         {"parts_min": 2_500,   "parts_max": 8_000,   "labour_min": 1_200,  "labour_max": 2_500,  "duration_h": 1.5},
    "ML_OIL_CHANGE_DUE":        {"parts_min": 1_500,   "parts_max": 4_000,   "labour_min": 400,    "labour_max": 800,    "duration_h": 0.5},
    "ML_OIL_ADVISORY":          {"parts_min": 1_500,   "parts_max": 4_000,   "labour_min": 400,    "labour_max": 800,    "duration_h": 0.5},
    "ML_HV_SOH_DECLINE":        {"parts_min": 0,       "parts_max": 1_50_000,"labour_min": 2_000,  "labour_max": 15_000, "duration_h": 4},
    "ML_CELL_ANOMALY":          {"parts_min": 5_000,   "parts_max": 80_000,  "labour_min": 3_000,  "labour_max": 15_000, "duration_h": 4},
    "ML_ARIMA_SOH_FORECAST":    {"parts_min": 0,       "parts_max": 1_00_000,"labour_min": 2_000,  "labour_max": 10_000, "duration_h": 3},
    "ML_12V_FAILURE_RISK":      {"parts_min": 4_000,   "parts_max": 10_000,  "labour_min": 400,    "labour_max": 800,    "duration_h": 0.5},
    "ML_12V_ADVISORY":          {"parts_min": 4_000,   "parts_max": 10_000,  "labour_min": 400,    "labour_max": 800,    "duration_h": 0.5},
    "ML_PUNCTURE_DETECTED":     {"parts_min": 400,     "parts_max": 6_000,   "labour_min": 300,    "labour_max": 600,    "duration_h": 0.5},
    "ML_TYRE_REPLACEMENT":      {"parts_min": 4_500,   "parts_max": 20_000,  "labour_min": 600,    "labour_max": 2_000,  "duration_h": 1.5},
    "ML_TYRE_ADVISORY":         {"parts_min": 4_500,   "parts_max": 16_000,  "labour_min": 600,    "labour_max": 1_500,  "duration_h": 1},
    "ML_FUEL_ANOMALY":          {"parts_min": 500,     "parts_max": 8_000,   "labour_min": 500,    "labour_max": 3_000,  "duration_h": 1},
    "ML_DRIVER_HIGH_RISK":      {"parts_min": 0,       "parts_max": 0,       "labour_min": 0,      "labour_max": 0,      "duration_h": 0},
    "ML_DRIVER_ADVISORY":       {"parts_min": 0,       "parts_max": 0,       "labour_min": 0,      "labour_max": 0,      "duration_h": 0},
}

# Model-code specific multipliers (MG models)
_MODEL_MULTIPLIERS: dict[str, float] = {
    "ZSEV":   1.35,   # ZS EV — EV parts premium
    "GLOSTER": 1.20,  # Gloster flagship
    "HECTOR":  1.10,  # Hector
    "ASTOR":   1.05,  # Astor
    "DEFAULT": 1.00,
}

# Warranty coverage: 3-year / 100,000 km bumper-to-bumper
_WARRANTY_MONTHS    = 36
_WARRANTY_KM        = 100_000
_WARRANTY_COVERAGE  = 0.80   # 80% of parts cost covered under WTY


def _load_taxonomy() -> list[dict]:
    """Load failure_taxonomy.json for part-level cost data."""
    import json
    from pathlib import Path
    tp = Path(__file__).resolve().parents[1] / "data" / "reference" / "failure_taxonomy.json"
    if not tp.exists():
        return []
    try:
        data = json.loads(tp.read_text())
        return data if isinstance(data, list) else data.get("parts", [])
    except Exception:
        return []


def _taxonomy_cost(alert_type: str) -> dict | None:
    """Try to derive cost ranges from the taxonomy for an alert type."""
    _ALERT_PART_MAP = {
        "BRAKE_PAD_CRITICAL":    "Front Brake Pad",
        "BRAKE_PAD_WARNING":     "Front Brake Pad",
        "ML_BRAKE_REPLACEMENT":  "Front Brake Pad",
        "ML_BRAKE_WARNING":      "Front Brake Pad",
        "ENGINE_OIL_LOW":        "Engine Oil 5W-30 (4L)",
        "ENGINE_OIL_ADVISORY":   "Engine Oil 5W-30 (4L)",
        "ML_OIL_CHANGE_DUE":     "Engine Oil 5W-30 (4L)",
        "ML_OIL_ADVISORY":       "Engine Oil 5W-30 (4L)",
        "12V_BATTERY_CRITICAL":  "12V Battery",
        "12V_BATTERY_RISK":      "12V Battery",
        "ML_12V_FAILURE_RISK":   "12V Battery",
        "ML_12V_ADVISORY":       "12V Battery",
        "HV_BATTERY_SOH_CRITICAL": "HV Battery Pack",
        "ML_HV_SOH_DECLINE":    "HV Battery Pack",
        "ML_CELL_ANOMALY":       "HV Battery Pack",
        "ML_TYRE_REPLACEMENT":   "Tyre (19 inch)",
        "ML_TYRE_ADVISORY":      "Tyre (19 inch)",
        "TYRE_DEFLATION":        "Tyre (19 inch)",
    }
    part_name = _ALERT_PART_MAP.get(alert_type)
    if not part_name:
        return None
    for part in _load_taxonomy():
        if part["part_name"] == part_name:
            unit = part["unit_cost_inr"]
            hrs = part["labour_hours"]
            labour_rate_inr = 800
            return {
                "parts_min": unit,
                "parts_max": round(unit * 1.3),
                "labour_min": round(hrs * labour_rate_inr * 0.8),
                "labour_max": round(hrs * labour_rate_inr * 1.2),
                "duration_h": hrs,
            }
    return None


class CostEstimator:
    """
    Estimates service cost for a given alert type, VIN, and model code.

    Usage:
        est = CostEstimator()
        result = est.estimate("BRAKE_PAD_CRITICAL", "MZ7XHPAE24DC000001", "HECTOR")
    """

    def estimate(
        self,
        alert_type: str,
        vin: str,
        model_code: str,
        manufacture_date: datetime | None = None,
        current_odo_km: float | None = None,
    ) -> dict:
        """
        Return a cost estimate for the service triggered by *alert_type*.

        Optionally pass *manufacture_date* and *current_odo_km* to enable
        warranty eligibility check.

        Returns:
            {
              parts_cost_min, parts_cost_max, labour_cost_est,
              warranty_likely (bool), warranty_cover_pct (float),
              total_min, total_max,
              duration_hours, currency: "INR",
              alert_type, model_code,
            }
        """
        cat    = _CATALOGUE.get(alert_type, _CATALOGUE.get("ENGINE_OIL_ADVISORY"))
        tax    = _taxonomy_cost(alert_type)
        if tax:
            cat = {**cat, **tax}
        mult   = _MODEL_MULTIPLIERS.get(model_code.upper(), _MODEL_MULTIPLIERS["DEFAULT"])

        # First try to get historical actuals from service records (PostgreSQL)
        hist   = self._lookup_history(alert_type, model_code)

        if hist:
            parts_min  = hist["parts_min"]
            parts_max  = hist["parts_max"]
            labour_est = hist["labour_avg"]
            duration_h = hist.get("duration_h", cat["duration_h"])
        else:
            parts_min  = round(cat["parts_min"] * mult)
            parts_max  = round(cat["parts_max"] * mult)
            labour_est = round((cat["labour_min"] + cat["labour_max"]) / 2 * mult)
            duration_h = cat["duration_h"]

        # Warranty check
        warranty_likely = False
        warranty_cover  = 0.0
        if manufacture_date is not None and current_odo_km is not None:
            age_months = _age_months(manufacture_date)
            if age_months < _WARRANTY_MONTHS and current_odo_km < _WARRANTY_KM:
                warranty_likely = True
                warranty_cover  = _WARRANTY_COVERAGE

        # Customer-pay totals
        cust_parts_min = round(parts_min * (1 - warranty_cover))
        cust_parts_max = round(parts_max * (1 - warranty_cover))
        total_min = cust_parts_min + round(labour_est * 0.8)
        total_max = cust_parts_max + round(labour_est * 1.2)

        return {
            "alert_type":        alert_type,
            "model_code":        model_code,
            "parts_cost_min":    parts_min,
            "parts_cost_max":    parts_max,
            "labour_cost_est":   labour_est,
            "warranty_likely":   warranty_likely,
            "warranty_cover_pct": round(warranty_cover * 100, 0),
            "customer_pays_min": total_min,
            "customer_pays_max": total_max,
            "total_min":         total_min,
            "total_max":         total_max,
            "duration_hours":    duration_h,
            "currency":          "INR",
        }

    def _lookup_history(self, alert_type: str, model_code: str) -> dict | None:
        """Query PostgreSQL service history for actuals (non-fatal)."""
        try:
            import psycopg2
            import os
            dsn = os.getenv("DATABASE_URL", "")
            if not dsn:
                return None
            conn = psycopg2.connect(dsn)
            cur  = conn.cursor()
            cur.execute(
                """
                SELECT
                    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY "NetValue") AS parts_min,
                    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY "NetValue") AS parts_max,
                    AVG("NetValue") AS labour_avg
                FROM service_history
                WHERE "ModelSalesCode" = %s
                  AND "ServiceType" ILIKE %s
                LIMIT 1
                """,
                (model_code, f"%{alert_type.replace('_', ' ')}%"),
            )
            row = cur.fetchone()
            conn.close()
            if row and row[0] is not None:
                return {"parts_min": float(row[0]), "parts_max": float(row[1]), "labour_avg": float(row[2])}
        except Exception as exc:
            log.debug("Cost history lookup failed: %s", exc)
        return None


def _age_months(manufacture_date: datetime) -> int:
    now = datetime.now(timezone.utc)
    dt  = manufacture_date.replace(tzinfo=timezone.utc) if manufacture_date.tzinfo is None else manufacture_date
    return (now.year - dt.year) * 12 + (now.month - dt.month)


# ── Backward-compat function interface (used by old agent stub) ────────────────

def estimate_repair_cost(component: str, service_type: str, vehicle_make: str = "generic") -> dict:
    """Legacy function interface — wraps CostEstimator for simple lookups."""
    alert_type = f"{component.upper()}_{service_type.upper()}".replace(" ", "_")
    return CostEstimator().estimate(alert_type, "", vehicle_make)
