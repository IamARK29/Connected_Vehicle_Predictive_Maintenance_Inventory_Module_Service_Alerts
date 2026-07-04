"""
Synthetic parts inventory generator.

Produces a realistic parts inventory CSV calibrated to the fleet size.
Output: data/synthetic/parts_inventory.csv
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


_PARTS_META = [
    {"part_code": "OIL-5W30-4L",    "description": "Engine Oil 5W-30 (4L)",            "unit_cost_inr": 855,   "abc_class": "A", "lead_time_days": 2,  "supplier": "Castrol / Mobil",   "replace_km": 7500,  "per_service_qty": 1},
    {"part_code": "OIL-FILTER-MG",   "description": "Oil Filter — MG OEM",              "unit_cost_inr": 183,   "abc_class": "A", "lead_time_days": 3,  "supplier": "MG OEM Direct",     "replace_km": 7500,  "per_service_qty": 1},
    {"part_code": "BR-PAD-F-MG",     "description": "Brake Pads (Front) — MG OEM",      "unit_cost_inr": 2800,  "abc_class": "A", "lead_time_days": 5,  "supplier": "MG OEM Direct",     "replace_km": 30000, "per_service_qty": 1},
    {"part_code": "BR-PAD-R-MG",     "description": "Brake Pads (Rear) — MG OEM",       "unit_cost_inr": 2200,  "abc_class": "A", "lead_time_days": 5,  "supplier": "MG OEM Direct",     "replace_km": 35000, "per_service_qty": 1},
    {"part_code": "BR-FLUID-DOT4",   "description": "Brake Fluid DOT 4 (500ml)",         "unit_cost_inr": 380,   "abc_class": "B", "lead_time_days": 3,  "supplier": "Castrol / Bosch",   "replace_km": 40000, "per_service_qty": 1},
    {"part_code": "TYRE-MG-195-55",  "description": "Tyre 195/55 R16 (MG Astor/Hector)", "unit_cost_inr": 4200, "abc_class": "A", "lead_time_days": 7,  "supplier": "MRF / Apollo",      "replace_km": 50000, "per_service_qty": 4},
    {"part_code": "AIR-FILTER-MG",   "description": "Air Filter — MG OEM",              "unit_cost_inr": 590,   "abc_class": "B", "lead_time_days": 4,  "supplier": "MG OEM Direct",     "replace_km": 15000, "per_service_qty": 1},
    {"part_code": "BATT-12V-MF60",   "description": "12V Battery 60Ah MF (Hector/Astor)", "unit_cost_inr": 4800, "abc_class": "A", "lead_time_days": 2, "supplier": "Amaron / Exide",    "replace_km": None,  "per_service_qty": 1},
    {"part_code": "COOLANT-OAT-1L",  "description": "Engine Coolant OAT (1L)",           "unit_cost_inr": 320,   "abc_class": "C", "lead_time_days": 2,  "supplier": "MG OEM Direct",     "replace_km": 60000, "per_service_qty": 3},
    {"part_code": "WIPER-B-MG",      "description": "Wiper Blade Bosch Rear — MG",       "unit_cost_inr": 450,   "abc_class": "C", "lead_time_days": 3,  "supplier": "Bosch India",       "replace_km": None,  "per_service_qty": 2},
    {"part_code": "HV-MODULE-MG",    "description": "HV Battery Module (ZS EV)",         "unit_cost_inr": 95000, "abc_class": "A", "lead_time_days": 21, "supplier": "MG OEM Direct",     "replace_km": None,  "per_service_qty": 1},
    {"part_code": "BMS-FUSE-MG",     "description": "BMS Safety Fuse 200A (EV)",         "unit_cost_inr": 1200,  "abc_class": "B", "lead_time_days": 10, "supplier": "MG OEM Direct",     "replace_km": None,  "per_service_qty": 1},
    {"part_code": "THERMOSTAT-MG",   "description": "Engine Thermostat — MG Hector",     "unit_cost_inr": 980,   "abc_class": "C", "lead_time_days": 7,  "supplier": "MG OEM Direct",     "replace_km": 80000, "per_service_qty": 1},
]


def generate_parts_inventory(
    fleet_df: pd.DataFrame,
    data_dir: Path,
    start_date: str = "2024-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_vehicles     = len(fleet_df)
    avg_km_per_mo  = 1200.0
    start_dt       = datetime.strptime(start_date[:10], "%Y-%m-%d")

    rows = []
    for meta in _PARTS_META:
        replace_km = meta["replace_km"]

        # Estimate monthly demand from fleet size and replacement interval
        if replace_km:
            monthly_demand = (n_vehicles * avg_km_per_mo / replace_km) * meta["per_service_qty"]
        else:
            # Non-km-driven: age-based replacements (roughly 5-15% of fleet per year)
            monthly_demand = n_vehicles * rng.uniform(0.04, 0.12) / 12

        monthly_demand = max(0.0, monthly_demand)
        demand_30d = round(monthly_demand)

        # Safety stock = 7-day buffer; reorder point = lead-time demand + safety
        daily_demand   = monthly_demand / 30.0
        safety_stock   = max(1, round(daily_demand * 7))
        reorder_point  = max(safety_stock + 1, round(daily_demand * meta["lead_time_days"] + safety_stock))

        # Current stock: seeded by dealer+part, calibrated around demand
        if meta["abc_class"] == "A":
            target_days = rng.integers(30, 60)
        elif meta["abc_class"] == "B":
            target_days = rng.integers(20, 45)
        else:
            target_days = rng.integers(10, 30)

        qty = max(0, round(daily_demand * int(target_days) + rng.integers(-2, 3)))

        # Determine status
        in_stock = qty > 0
        reorder_qty = max(0, reorder_point - qty) if qty <= reorder_point else 0

        # Days until stockout
        if daily_demand > 0 and qty > 0:
            days_until_stockout = int(qty / daily_demand)
        elif qty == 0:
            days_until_stockout = 0
        else:
            days_until_stockout = 999

        # Last restock date: sometime in past 0-30 days
        last_restock_days_ago = int(rng.integers(0, 30))
        last_restock_date = (start_dt - timedelta(days=last_restock_days_ago)).strftime("%Y-%m-%d")

        rows.append({
            "part_code":            meta["part_code"],
            "description":          meta["description"],
            "abc_class":            meta["abc_class"],
            "supplier":             meta["supplier"],
            "unit_cost_inr":        meta["unit_cost_inr"],
            "lead_time_days":       meta["lead_time_days"],
            "replace_km":           replace_km if replace_km else "",
            "qty_on_hand":          qty,
            "reorder_point":        reorder_point,
            "safety_stock":         safety_stock,
            "reorder_qty":          reorder_qty,
            "in_stock":             in_stock,
            "monthly_demand_est":   round(monthly_demand, 2),
            "demand_30d_est":       demand_30d,
            "demand_90d_est":       max(demand_30d, demand_30d * 3),
            "days_until_stockout":  days_until_stockout if days_until_stockout < 999 else None,
            "last_restock_date":    last_restock_date,
            "total_stock_value_inr": round(qty * meta["unit_cost_inr"], 2),
        })

    df = pd.DataFrame(rows)
    out_path = data_dir / "parts_inventory.csv"
    df.to_csv(out_path, index=False)
    print(f"parts_inventory.csv: {len(df)} SKUs -> {out_path}")
    return df


if __name__ == "__main__":
    import sys
    from pathlib import Path as P

    data_dir = P(sys.argv[1]) if len(sys.argv) > 1 else P("data/synthetic")
    fleet_path = data_dir / "fleet.csv"
    if not fleet_path.exists():
        sys.exit(f"Fleet not found: {fleet_path}")
    fleet_df = pd.read_csv(fleet_path)
    generate_parts_inventory(fleet_df, data_dir)
