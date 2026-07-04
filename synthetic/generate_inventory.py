"""
Generate synthetic inventory stock data for AutoPredict.

Produces:
  data/synthetic/inventory_stock.csv       — current stock levels per dealer × part
  data/synthetic/inventory_transactions.csv — 12 months of transaction history
"""
from __future__ import annotations

import math
import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data/synthetic")

# ── Parts catalogue (matches dealer.py _PARTS_META) ──────────────────────────
PARTS = {
    "OIL-5W30-4L":     {"description": "Engine Oil 5W-30 (4L)",          "unit_cost": 855,    "lead_days": 2,  "abc": "A", "fuel_types": {"ICE","PHEV"}, "replace_km": 7500,  "qty_per_svc": 1},
    "OIL-FILTER-MG":   {"description": "Oil Filter — MG OEM",            "unit_cost": 183,    "lead_days": 3,  "abc": "A", "fuel_types": {"ICE","PHEV"}, "replace_km": 7500,  "qty_per_svc": 1},
    "BR-PAD-F-MG":     {"description": "Brake Pads Front — MG OEM",      "unit_cost": 2800,   "lead_days": 5,  "abc": "A", "fuel_types": None,           "replace_km": 30000, "qty_per_svc": 1},
    "BR-PAD-R-MG":     {"description": "Brake Pads Rear — MG OEM",       "unit_cost": 2200,   "lead_days": 5,  "abc": "A", "fuel_types": None,           "replace_km": 35000, "qty_per_svc": 1},
    "BR-FLUID-DOT4":   {"description": "Brake Fluid DOT 4 (500ml)",       "unit_cost": 380,    "lead_days": 3,  "abc": "B", "fuel_types": None,           "replace_km": 60000, "qty_per_svc": 1},
    "TYRE-215-60-17":  {"description": "Tyre 215/60 R17",                 "unit_cost": 6500,   "lead_days": 7,  "abc": "B", "fuel_types": None,           "replace_km": 50000, "qty_per_svc": 4},
    "TYRE-225-55-18":  {"description": "Tyre 225/55 R18",                 "unit_cost": 7800,   "lead_days": 7,  "abc": "B", "fuel_types": None,           "replace_km": 50000, "qty_per_svc": 4},
    "BATT-12V-60AH":   {"description": "12V Battery 60Ah",                "unit_cost": 4500,   "lead_days": 3,  "abc": "A", "fuel_types": None,           "replace_km": None,  "qty_per_svc": 1},
    "BATT-12V-70AH":   {"description": "12V Battery 70Ah",                "unit_cost": 5200,   "lead_days": 3,  "abc": "A", "fuel_types": None,           "replace_km": None,  "qty_per_svc": 1},
    "COOLANT-1L":      {"description": "Engine Coolant (1L)",              "unit_cost": 350,    "lead_days": 3,  "abc": "B", "fuel_types": {"ICE","PHEV"}, "replace_km": 60000, "qty_per_svc": 3},
    "AIR-FILTER-MG":   {"description": "Air Filter — MG OEM",             "unit_cost": 420,    "lead_days": 3,  "abc": "B", "fuel_types": None,           "replace_km": 30000, "qty_per_svc": 1},
    "SPARK-PLUG-NGK":  {"description": "Spark Plugs NGK (set of 4)",      "unit_cost": 1850,   "lead_days": 5,  "abc": "B", "fuel_types": {"ICE"},        "replace_km": 60000, "qty_per_svc": 1},
    "HV-MODULE-MG":    {"description": "HV Battery Module — MG ZS EV",   "unit_cost": 185000, "lead_days": 21, "abc": "A", "fuel_types": {"EV","PHEV"},  "replace_km": None,  "qty_per_svc": 1},
    "BMS-FUSE-MG":     {"description": "BMS Fuse Assembly — MG",          "unit_cost": 2800,   "lead_days": 14, "abc": "B", "fuel_types": {"EV","PHEV"},  "replace_km": None,  "qty_per_svc": 1},
    "WIPER-BLADE-MG":  {"description": "Wiper Blade Set — MG OEM",        "unit_cost": 650,    "lead_days": 2,  "abc": "C", "fuel_types": None,           "replace_km": 20000, "qty_per_svc": 1},
    "CABIN-FILTER-MG": {"description": "Cabin Air Filter — MG OEM",       "unit_cost": 780,    "lead_days": 3,  "abc": "C", "fuel_types": None,           "replace_km": 20000, "qty_per_svc": 1},
    "THERMOSTAT-MG":   {"description": "Engine Thermostat — MG OEM",      "unit_cost": 1200,   "lead_days": 10, "abc": "C", "fuel_types": {"ICE","PHEV"}, "replace_km": None,  "qty_per_svc": 1},
    "OBD-HARNESS-MG":  {"description": "OBD Diagnostic Harness",          "unit_cost": 3500,   "lead_days": 7,  "abc": "C", "fuel_types": None,           "replace_km": None,  "qty_per_svc": 1},
}

AVG_KM_PER_MONTH = 1500
HISTORY_MONTHS   = 12
SERVICE_LEVEL    = 0.95      # 95% → Z = 1.645
Z                = 1.645
ORDERING_COST    = 500       # ₹ per order
HOLDING_RATE     = 0.20      # 20% of unit cost per year


def _eoq(annual_demand: float, unit_cost: float) -> int:
    if annual_demand <= 0 or unit_cost <= 0:
        return 1
    h = unit_cost * HOLDING_RATE
    return max(1, round(math.sqrt(2 * annual_demand * ORDERING_COST / h)))


def _safety_stock(sigma_demand_daily: float, lead_days: int) -> float:
    return Z * sigma_demand_daily * math.sqrt(lead_days)


def main() -> None:
    rng = random.Random(42)
    np_rng = np.random.default_rng(42)

    fleet = pd.read_csv(DATA_DIR / "fleet_master.csv")
    dealers = fleet.groupby(["dealer_code", "dealer_name", "dealer_city"])["vin"].count().reset_index()
    dealers.rename(columns={"vin": "n_vehicles"}, inplace=True)

    svc_history: pd.DataFrame | None = None
    svc_path = DATA_DIR / "service_history.csv"
    if svc_path.exists():
        svc_history = pd.read_csv(svc_path, low_memory=False)

    stock_rows: list[dict] = []
    txn_rows:   list[dict] = []
    today = date.today()

    for _, dealer_row in dealers.iterrows():
        dc        = dealer_row["dealer_code"]
        n_veh     = int(dealer_row["n_vehicles"])
        dealer_rng = random.Random(hash(dc) % 2**32)

        for part_code, meta in PARTS.items():
            # ── Compute monthly demand from fleet size + replace interval ────
            if meta["replace_km"]:
                fleet_km_per_month  = n_veh * AVG_KM_PER_MONTH
                monthly_base        = (fleet_km_per_month / meta["replace_km"]) * meta["qty_per_svc"]
            else:
                # Age-based / event-based parts: rough empirical rates
                event_rates = {
                    "BATT-12V-60AH": 0.04, "BATT-12V-70AH": 0.03,
                    "HV-MODULE-MG":  0.005, "BMS-FUSE-MG": 0.02,
                    "THERMOSTAT-MG": 0.01,  "OBD-HARNESS-MG": 0.02,
                }
                monthly_base = n_veh * event_rates.get(part_code, 0.02)

            # Add noise across months
            sigma_monthly   = max(0.3, monthly_base * 0.25)
            sigma_daily     = sigma_monthly / 30 ** 0.5
            annual_demand   = monthly_base * 12
            avg_daily       = monthly_base / 30.0

            # ── Inventory parameters ─────────────────────────────────────────
            lead_days       = meta["lead_days"]
            eoq             = _eoq(annual_demand, meta["unit_cost"])
            ss              = max(1, round(_safety_stock(sigma_daily, lead_days)))
            rop             = round(avg_daily * lead_days + ss)          # Reorder Point
            max_stock       = rop + eoq

            # ── Simulate 12-month transaction history ────────────────────────
            stock_level = int(rng.uniform(rop * 0.5, rop + eoq))        # starting stock
            txn_date    = today - timedelta(days=HISTORY_MONTHS * 30)
            last_restock_date = txn_date

            while txn_date <= today:
                # Monthly demand draw
                daily_demand = max(0.0, float(np_rng.normal(avg_daily, sigma_daily)))
                if daily_demand > 0 and rng.random() < daily_demand:
                    qty_issued = max(1, round(daily_demand * rng.uniform(0.5, 2.0)))
                    if stock_level >= qty_issued:
                        stock_level -= qty_issued
                        txn_rows.append({
                            "date":             txn_date.isoformat(),
                            "dealer_code":      dc,
                            "part_code":        part_code,
                            "transaction_type": "ISSUE",
                            "quantity":         qty_issued,
                            "reference":        f"JO-{dc}-{txn_date.strftime('%Y%m%d')}-{rng.randint(1000,9999)}",
                            "stock_after":      stock_level,
                        })

                # Trigger restock when stock drops below ROP
                if stock_level <= rop:
                    restock_qty = eoq
                    delivery_date = txn_date + timedelta(days=lead_days)
                    if delivery_date <= today:
                        stock_level += restock_qty
                        last_restock_date = delivery_date
                        txn_rows.append({
                            "date":             delivery_date.isoformat(),
                            "dealer_code":      dc,
                            "part_code":        part_code,
                            "transaction_type": "RECEIPT",
                            "quantity":         restock_qty,
                            "reference":        f"PO-{dc}-{delivery_date.strftime('%Y%m%d')}-{rng.randint(100,999)}",
                            "stock_after":      stock_level,
                        })

                txn_date += timedelta(days=1)

            # ── Compute last 90-day stats for display ─────────────────────────
            recent_issues = [
                t for t in txn_rows
                if t["dealer_code"] == dc
                and t["part_code"] == part_code
                and t["transaction_type"] == "ISSUE"
                and t["date"] >= (today - timedelta(days=90)).isoformat()
            ]
            qty_90d = sum(t["quantity"] for t in recent_issues)
            avg_daily_90d = qty_90d / 90.0

            # Days of supply at current consumption rate
            if avg_daily_90d > 0:
                days_of_supply = round(stock_level / avg_daily_90d)
            else:
                days_of_supply = 999

            # Stockout risk: probability of stockout before next reorder arrives
            if avg_daily_90d > 0 and sigma_daily > 0:
                deficit = stock_level - (avg_daily_90d * lead_days + ss)
                stockout_prob = max(0.0, min(1.0, float(np_rng.normal(
                    0.5 - deficit / (sigma_daily * lead_days + 0.001), 0.15
                ))))
            else:
                stockout_prob = 0.0

            stock_status = (
                "STOCKOUT"  if stock_level == 0 else
                "CRITICAL"  if stock_level < ss else
                "LOW"       if stock_level <= rop else
                "OK"
            )

            last_sold_txns = [t for t in txn_rows if t["dealer_code"] == dc and t["part_code"] == part_code and t["transaction_type"] == "ISSUE"]
            last_sold = max((t["date"] for t in last_sold_txns), default="")
            is_slow_mover = len(recent_issues) < 2 and avg_daily_90d < 0.05

            stock_rows.append({
                "dealer_code":        dc,
                "dealer_city":        dealer_row["dealer_city"],
                "part_code":          part_code,
                "description":        meta["description"],
                "abc_class":          meta["abc"],
                "supplier":           _supplier(part_code),
                "unit_cost_inr":      meta["unit_cost"],
                "current_stock":      max(0, stock_level),
                "unit":               "qty",
                "eoq":                eoq,
                "safety_stock":       ss,
                "reorder_point":      rop,
                "max_stock":          max_stock,
                "lead_time_days":     lead_days,
                "avg_daily_demand":   round(avg_daily, 4),
                "demand_std_daily":   round(sigma_daily, 4),
                "avg_qty_90d":        round(qty_90d, 1),
                "days_of_supply":     min(999, days_of_supply),
                "stockout_prob":      round(stockout_prob, 3),
                "stock_status":       stock_status,
                "last_restocked":     last_restock_date.isoformat(),
                "last_sold":          last_sold,
                "is_slow_mover":      is_slow_mover,
                "inventory_value_inr": round(max(0, stock_level) * meta["unit_cost"], 2),
                "service_types":      _svc_types(part_code),
            })

    stock_df = pd.DataFrame(stock_rows)
    stock_df.to_csv(DATA_DIR / "inventory_stock.csv", index=False)
    print(f"Generated inventory_stock.csv: {len(stock_df)} rows ({len(dealers)} dealers × {len(PARTS)} parts)")

    txn_df = pd.DataFrame(txn_rows).sort_values("date")
    txn_df.to_csv(DATA_DIR / "inventory_transactions.csv", index=False)
    print(f"Generated inventory_transactions.csv: {len(txn_df)} rows")

    # Quick summary
    alerts = stock_df[stock_df["stock_status"].isin(["CRITICAL", "STOCKOUT"])]
    print(f"  Reorder alerts: {len(alerts)} items need attention")
    print(f"  Total inventory value: ₹{stock_df['inventory_value_inr'].sum():,.0f}")


def _supplier(part_code: str) -> str:
    m = {
        "OIL-5W30-4L": "Castrol India", "OIL-FILTER-MG": "MG OEM Direct",
        "BR-PAD-F-MG": "MG OEM Direct", "BR-PAD-R-MG": "MG OEM Direct",
        "BR-FLUID-DOT4": "TotalEnergies", "TYRE-215-60-17": "MRF",
        "TYRE-225-55-18": "Apollo Tyres", "BATT-12V-60AH": "Amaron",
        "BATT-12V-70AH": "Exide", "COOLANT-1L": "MG OEM Direct",
        "AIR-FILTER-MG": "MG OEM Direct", "SPARK-PLUG-NGK": "NGK India",
        "HV-MODULE-MG": "MG OEM Direct", "BMS-FUSE-MG": "MG OEM Direct",
        "WIPER-BLADE-MG": "MG OEM Direct", "CABIN-FILTER-MG": "MG OEM Direct",
        "THERMOSTAT-MG": "MG OEM Direct", "OBD-HARNESS-MG": "MG OEM Direct",
    }
    return m.get(part_code, "MG OEM Direct")


def _svc_types(part_code: str) -> str:
    m = {
        "OIL-5W30-4L": "OIL_CHANGE,FULL_SERVICE",
        "OIL-FILTER-MG": "OIL_CHANGE,FULL_SERVICE",
        "BR-PAD-F-MG": "BRAKE_CHECK,FULL_SERVICE",
        "BR-PAD-R-MG": "BRAKE_CHECK,FULL_SERVICE",
        "BR-FLUID-DOT4": "BRAKE_CHECK,FULL_SERVICE",
        "TYRE-215-60-17": "TYRE_ROTATION,FULL_SERVICE",
        "TYRE-225-55-18": "TYRE_ROTATION,FULL_SERVICE",
        "BATT-12V-60AH": "12V_BATTERY_REPLACE",
        "BATT-12V-70AH": "12V_BATTERY_REPLACE",
        "COOLANT-1L": "COOLANT_FLUSH,FULL_SERVICE",
        "AIR-FILTER-MG": "FULL_SERVICE",
        "SPARK-PLUG-NGK": "FULL_SERVICE",
        "HV-MODULE-MG": "HV_BATTERY_INSPECT",
        "BMS-FUSE-MG": "HV_BATTERY_INSPECT",
        "WIPER-BLADE-MG": "FULL_SERVICE",
        "CABIN-FILTER-MG": "FULL_SERVICE",
    }
    return m.get(part_code, "")


if __name__ == "__main__":
    main()
