"""
Synthetic Service History Generator.

Produces service_history.csv matching the SERVICE_COL_MAP schema expected
by ingestion/file_ingestor.py.

Logic:
 - Schedule-based services are triggered by odometer milestones
 - Failure-specific unscheduled repairs come from failures_manifest.csv
 - Each service event produces one or more line items (parts + labour)
 - IssueType distribution: 60% WTY (warranty) / 40% PAID
"""
from __future__ import annotations

import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

from synthetic.config import SyntheticConfig

fake = Faker("en_IN")

# ── Scheduled service milestones (km intervals) ────────────────────────────

_SCHEDULES: list[dict] = [
    {"service_type": "OIL_CHANGE",      "km_interval":  7_500, "km_first":  7_500},
    {"service_type": "TYRE_ROTATION",   "km_interval": 10_000, "km_first": 10_000},
    {"service_type": "FULL_SERVICE",    "km_interval": 20_000, "km_first": 20_000},
    {"service_type": "BRAKE_CHECK",     "km_interval": 45_000, "km_first": 45_000},
    {"service_type": "AC_SERVICE",      "km_interval": 30_000, "km_first": 30_000},
    {"service_type": "COOLANT_FLUSH",   "km_interval": 60_000, "km_first": 60_000},
]

# ── Part price catalogue (INR) ─────────────────────────────────────────────

_PARTS_CATALOGUE: dict[str, list[tuple[str, str, float, float]]] = {
    # service_type → [(OrderItem, MaterialGroup, UnitPrice_min, UnitPrice_max)]
    "OIL_CHANGE": [
        ("OIL-5W30-4L",  "CONSUMABLE", 800,  1_000),
        ("OIL-FILTER",   "FILTER",     180,    280),
        ("LABOUR-OIL",   "LABOUR",     300,    500),
    ],
    "TYRE_ROTATION": [
        ("LABOUR-TYRE",  "LABOUR",     200,    400),
    ],
    "FULL_SERVICE": [
        ("OIL-5W30-4L",  "CONSUMABLE", 800,  1_000),
        ("OIL-FILTER",   "FILTER",     180,    280),
        ("AIR-FILTER",   "FILTER",     400,    700),
        ("FUEL-FILTER",  "FILTER",     350,    600),
        ("LABOUR-FULL",  "LABOUR",   1_200,  2_000),
    ],
    "BRAKE_CHECK": [
        ("BRAKE-PAD-F",  "SPARE",    2_000,  3_500),
        ("BRAKE-PAD-R",  "SPARE",    1_500,  2_500),
        ("LABOUR-BRAKE", "LABOUR",     800,  1_400),
    ],
    "AC_SERVICE": [
        ("AC-GAS-R134",  "CONSUMABLE", 600,    900),
        ("CABIN-FILTER", "FILTER",     500,    800),
        ("LABOUR-AC",    "LABOUR",     500,    900),
    ],
    "COOLANT_FLUSH": [
        ("COOLANT-OAT",  "CONSUMABLE", 700,  1_100),
        ("LABOUR-COOL",  "LABOUR",     400,    700),
    ],
    # Failure-driven repairs
    "BRAKE_REPAIR": [
        ("BRAKE-PAD-F",  "SPARE",    2_200,  4_000),
        ("BRAKE-DISC-F", "SPARE",    3_500,  6_000),
        ("BRAKE-FLUID",  "CONSUMABLE", 350,    500),
        ("LABOUR-BRAKE", "LABOUR",   1_000,  1_800),
    ],
    "OIL_SYSTEM_REPAIR": [
        ("OIL-5W30-4L",  "CONSUMABLE", 800,  1_200),
        ("OIL-FILTER",   "FILTER",     180,    280),
        ("ENGINE-GASKET","SPARE",    1_200,  3_000),
        ("LABOUR-ENG",   "LABOUR",   1_500,  3_000),
    ],
    "HV_BATTERY_SERVICE": [
        ("BMS-MODULE",   "SPARE",   15_000, 35_000),
        ("HV-CELL-SET",  "SPARE",   80_000,200_000),
        ("LABOUR-HV",    "LABOUR",   5_000, 12_000),
    ],
    "12V_BATTERY_REPLACE": [
        ("12V-BAT-65AH", "SPARE",    4_500,  6_500),
        ("LABOUR-BAT",   "LABOUR",     300,    500),
    ],
    "TYRE_REPLACE": [
        ("TYRE-205-55R16","SPARE",   4_000,  5_500),
        ("LABOUR-TYRE",  "LABOUR",     300,    600),
    ],
    "COOLING_REPAIR": [
        ("THERMOSTAT",   "SPARE",      800,  1_800),
        ("RADIATOR-CAP", "SPARE",      200,    400),
        ("COOLANT-OAT",  "CONSUMABLE", 700,  1_100),
        ("LABOUR-COOL",  "LABOUR",     800,  1_600),
    ],
}

_FAILURE_SERVICE_MAP = {
    "brake_degradation":      "BRAKE_REPAIR",
    "oil_degradation":        "OIL_SYSTEM_REPAIR",
    "hv_battery_degradation": "HV_BATTERY_SERVICE",
    "12v_battery_failure":    "12V_BATTERY_REPLACE",
    "tyre_puncture":          "TYRE_REPLACE",
    "overheating":            "COOLING_REPAIR",
}

_TAX_RATE = 0.18  # GST 18%


def generate_service_history(
    fleet_df: pd.DataFrame | None = None,
    cfg: SyntheticConfig | None = None,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    cfg      = cfg or SyntheticConfig()
    data_dir = data_dir or Path("data/synthetic")
    rng      = random.Random(cfg.seed)
    np_rng   = np.random.default_rng(cfg.seed)
    Faker.seed(cfg.seed)

    if fleet_df is None:
        fleet_df = pd.read_csv(data_dir / "fleet_master.csv")

    # Load failure manifest to add unscheduled failure repairs
    manifest_df = pd.DataFrame()
    manifest_path = data_dir / "failures_manifest.csv"
    if manifest_path.exists():
        manifest_df = pd.read_csv(manifest_path)

    start_dt = datetime.strptime(cfg.start_date, "%Y-%m-%d")
    end_dt   = start_dt + timedelta(days=cfg.num_days)

    all_rows: list[dict] = []

    for _, vrow in fleet_df.iterrows():
        vin = str(vrow["vin"])

        # Build scheduled service events from odometer milestones
        odo_start = float(vrow["initial_odometer"])
        avg_daily = cfg.avg_daily_km[str(vrow["driver_profile"])]

        scheduled_events = _build_scheduled_events(
            odo_start=odo_start,
            avg_daily_km=avg_daily,
            start_dt=start_dt,
            end_dt=end_dt,
        )

        # Add failure-specific unscheduled events
        fail_events = _build_failure_events(vin, manifest_df)

        all_events = sorted(scheduled_events + fail_events, key=lambda e: e["date"])

        for ev in all_events:
            rows = _build_line_items(ev, vrow, rng, np_rng, cfg)
            all_rows.extend(rows)
            print(f"  {vin}: {ev['service_type']} @ {ev['date'].date()} -> {len(rows)} line items")

    result = pd.DataFrame(all_rows)
    out_csv = data_dir / "service_history.csv"
    result.to_csv(out_csv, index=False)
    print(f"service_history.csv: {len(result)} line items -> {out_csv}")
    return result


def _build_scheduled_events(
    odo_start: float,
    avg_daily_km: float,
    start_dt: datetime,
    end_dt: datetime,
) -> list[dict]:
    events: list[dict] = []

    for sched in _SCHEDULES:
        km_interval = sched["km_interval"]
        km_first    = sched["km_first"]

        # First service at km_first from start of simulation
        km_target = odo_start + (km_first - (odo_start % km_first)) % km_first
        if km_target == odo_start:
            km_target += km_interval

        while True:
            days_offset = (km_target - odo_start) / max(avg_daily_km, 1)
            svc_date    = start_dt + timedelta(days=days_offset)
            if svc_date >= end_dt:
                break
            events.append({
                "service_type": sched["service_type"],
                "date":         svc_date,
                "odometer":     km_target,
                "unscheduled":  False,
            })
            km_target += km_interval

    return events


def _build_failure_events(vin: str, manifest_df: pd.DataFrame) -> list[dict]:
    events: list[dict] = []
    if manifest_df.empty:
        return events
    vfails = manifest_df[manifest_df["vin"] == vin]
    for _, frow in vfails.iterrows():
        ftype    = str(frow["failure_type"])
        svc_type = _FAILURE_SERVICE_MAP.get(ftype, "FULL_SERVICE")
        # Service happens shortly after the failure date
        fail_dt  = datetime.strptime(str(frow["failure_date"]), "%Y-%m-%d")
        svc_dt   = fail_dt + timedelta(days=random.randint(1, 5))
        events.append({
            "service_type":   svc_type,
            "date":           svc_dt,
            "odometer":       None,  # will estimate in line-item builder
            "unscheduled":    True,
            "failure_type":   ftype,
        })
    return events


def _build_line_items(
    event: dict,
    vrow: pd.Series,
    rng: random.Random,
    np_rng: np.random.Generator,
    cfg: SyntheticConfig,
) -> list[dict]:
    svc_type  = event["service_type"]
    svc_date  = event["date"]
    odo       = event.get("odometer") or (float(vrow["initial_odometer"]) + rng.uniform(5_000, 80_000))
    unsch     = event.get("unscheduled", False)

    dealer    = _pick_dealer(vrow, cfg, rng)
    issue_type = "WTY" if rng.random() < 0.60 else "PAID"
    order_num = uuid.uuid4().hex[:8].upper()

    parts_list = _PARTS_CATALOGUE.get(svc_type, _PARTS_CATALOGUE["FULL_SERVICE"])
    rows: list[dict] = []

    for i, (order_item, mat_group, price_min, price_max) in enumerate(parts_list):
        unit_price = round(rng.uniform(price_min, price_max), 2)
        qty        = 1 if mat_group != "LABOUR" else 1.0
        net_value  = round(unit_price * qty, 2)
        tax        = round(net_value * _TAX_RATE, 2)
        total_val  = round(net_value + tax, 2)

        # Warranty: dealer absorbs or manufacturer absorbs
        if issue_type == "WTY":
            warr_contrib = round(total_val * rng.uniform(0.7, 1.0), 2)
            insur_contrib = 0.0
        else:
            warr_contrib  = 0.0
            insur_contrib = round(total_val * rng.uniform(0.0, 0.2), 2)
        disc_contrib = round(total_val * rng.uniform(0.0, 0.05), 2)

        rows.append({
            "DealerCode":               dealer["code"],
            "Region":                   dealer["region"],
            "CompanyCode":              "MG001",
            "CreatedOn":                svc_date.strftime("%Y-%m-%d"),
            "CreatedOnTime":            svc_date.strftime("%H:%M:%S"),
            "Zone":                     dealer["region"],
            "DealerName":               dealer["name"],
            "DealerCity":               dealer["city"],
            "LicensePlateNumber":       _vin_to_plate(str(vrow["vin"]), rng),
            "VIN":                      str(vrow["vin"]),
            "Status":                   "CLOSED",
            "ServiceType":              svc_type,
            "ModelSalesCode":           str(vrow["model_code"]),
            "ModelSalesCodeDescription": str(vrow["model_name"]),
            "Color":                    str(vrow.get("color", "Pearl White")),
            "Mileage":                  round(odo, 0),
            "OrderItem":                f"{order_num}-{i+1:02d}",
            "LabPart":                  "PART" if mat_group != "LABOUR" else "LAB",
            "MaterialGroup":            mat_group,
            "DescriptionOne":           order_item.replace("-", " "),
            "OrderQuantity":            qty,
            "UnitPrice":                unit_price,
            "NetValue":                 net_value,
            "Tax":                      tax,
            "TotalValue":               total_val,
            "GrossValue":               total_val,
            "KeyField":                 f"WO-{order_num}",
            "IssueType":                issue_type,
            "WarrantyContribution":     warr_contrib,
            "InsuranceContribution":    insur_contrib,
            "DiscountContribution":     disc_contrib,
        })

    return rows


def _pick_dealer(vrow: pd.Series, cfg: SyntheticConfig, rng: random.Random) -> dict:
    """Pick dealer — 70% home dealer, 30% any dealer."""
    home_code = str(vrow.get("dealer_code", "DL001"))
    for d in cfg.dealers:
        if d.code == home_code:
            home_dealer = {"code": d.code, "name": d.name, "city": d.city, "region": d.region}
            break
    else:
        home_dealer = {"code": "DL001", "name": "MG Motors New Delhi", "city": "New Delhi", "region": "North"}

    if rng.random() < 0.70:
        return home_dealer

    d = rng.choice(cfg.dealers)
    return {"code": d.code, "name": d.name, "city": d.city, "region": d.region}


_STATES = ["DL", "MH", "KA", "TN", "TS", "GJ", "WB", "PB"]

def _vin_to_plate(vin: str, rng: random.Random) -> str:
    state  = rng.choice(_STATES)
    dist   = rng.randint(1, 99)
    series = rng.choice(["AA", "AB", "AC", "BA", "BB"])
    num    = rng.randint(1000, 9999)
    return f"{state}{dist:02d}{series}{num}"


if __name__ == "__main__":
    generate_service_history()
