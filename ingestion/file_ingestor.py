"""
File Ingestor — Mode B ingestion.

Handles CSV and JSON uploads for telemetry, trips, and service history.
Writes telemetry to InfluxDB; trips and service_history to PostgreSQL.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text

from ingestion.db_writer import write_influx

log = logging.getLogger(__name__)

PG_URL = os.getenv("POSTGRES_URL", "postgresql://autopredict:autopredict@localhost:5432/autopredict")

# ── Column mappings from real sample-data headers ──────────────────────────

TELEMETRY_COL_MAP = {
    "VIN": "vin",
    "StartTime-TimeStamp": "timestamp",
    "StartTime-Date": "date",
    "VehSpeed": "veh_speed",
    "VehSysPwrMod": "veh_sys_pwr_mod",
    "VehRPM": "veh_rpm",
    "VehGearPos": "veh_gear_pos",
    "VehSteeringAngle": "veh_steering_angle",
    "VehBrakePos": "veh_brake_pos",
    "VehAccelPos": "veh_accel_pos",
    "VehBatt": "veh_batt",
    "VehOdo": "veh_odo",
    "FuelTankLevel": "fuel_tank_level",
    "BMSPackVol": "bms_pack_vol",
    "BMSPackCrnt": "bms_pack_crnt",
    "BMSPackSOC": "bms_pack_soc",
    "BMSPackSOH": "bms_pack_soh",
    "BMSCellMaxVol": "bms_cell_max_vol",
    "BMSCellMinVol": "bms_cell_min_vol",
    "BMSCellMaxTemp": "bms_cell_max_temp",
    "BMSCellMinTemp": "bms_cell_min_temp",
    "TyrePressureFL": "tyre_pressure_fl",
    "TyrePressureFR": "tyre_pressure_fr",
    "TyrePressureRL": "tyre_pressure_rl",
    "TyrePressureRR": "tyre_pressure_rr",
    "TyreTempFL": "tyre_temp_fl",
    "TyreTempFR": "tyre_temp_fr",
    "TyreTempRL": "tyre_temp_rl",
    "TyreTempRR": "tyre_temp_rr",
    "GNSSLat": "gnss_lat",
    "GNSSLong": "gnss_long",
    "GNSSAlt": "gnss_alt",
    "GNSSHead": "gnss_head",
    "GNSSSats": "gnss_sats",
}

TRIP_COL_MAP = {
    "tripId": "trip_id",
    "vin": "vin",
    "startTime": "start_time",
    "endTime": "end_time",
    "startOdometer": "start_odometer",
    "endOdometer": "end_odometer",
    "odometer": "odometer",
    "averageSpeed": "average_speed",
    "maxSpeed": "max_speed",
    "vehFuelConsumed": "fuel_consumed",
    "fuelEfficiency": "fuel_efficiency",
    "driveScore": "drive_score",
    "powerMode": "power_mode",
    "overSpeedNum": "over_speed_num",
    "overSpeed80": "over_speed_80",
    "overSpeed120": "over_speed_120",
    "harshBreakingNum": "harsh_braking_num",
    "suddenTurnNum": "sudden_turn_num",
    "accelerationNum": "acceleration_num",
    "startPoint_lat": "start_lat",
    "startPoint_long": "start_long",
    "endPoint_lat": "end_lat",
    "endPoint_long": "end_long",
}

SERVICE_COL_MAP = {
    "DealerCode": "dealer_code",
    "Region": "region",
    "CompanyCode": "company_code",
    "CreatedOn": "created_on",
    "CreatedOnTime": "created_on_time",
    "Zone": "zone",
    "DealerName": "dealer_name",
    "DealerCity": "dealer_city",
    "LicensePlateNumber": "license_plate",
    "VIN": "vin",
    "Status": "status",
    "ServiceType": "service_type",
    "ModelSalesCode": "model_sales_code",
    "ModelSalesCodeDescription": "model_description",
    "Color": "color",
    "Mileage": "mileage",
    "OrderItem": "order_item",
    "LabPart": "lab_part",
    "MaterialGroup": "material_group",
    "DescriptionOne": "description",
    "OrderQuantity": "order_quantity",
    "UnitPrice": "unit_price",
    "NetValue": "net_value",
    "Tax": "tax",
    "TotalValue": "total_value",
    "GrossValue": "gross_value",
    "KeyField": "key_field",
    "IssueType": "issue_type",
    "WarrantyContribution": "warranty_contribution",
    "InsuranceContribution": "insurance_contribution",
    "DiscountContribution": "discount_contribution",
}

_DDL_TRIPS = """
CREATE TABLE IF NOT EXISTS trips (
    id BIGSERIAL PRIMARY KEY,
    trip_id VARCHAR(50),
    vin VARCHAR(17),
    start_time TIMESTAMPTZ,
    end_time TIMESTAMPTZ,
    start_odometer FLOAT,
    end_odometer FLOAT,
    odometer FLOAT,
    average_speed FLOAT,
    max_speed FLOAT,
    fuel_consumed FLOAT,
    fuel_efficiency FLOAT,
    drive_score FLOAT,
    power_mode INTEGER,
    over_speed_num INTEGER,
    over_speed_80 INTEGER,
    over_speed_120 INTEGER,
    harsh_braking_num INTEGER,
    sudden_turn_num INTEGER,
    acceleration_num INTEGER,
    start_lat FLOAT,
    start_long FLOAT,
    end_lat FLOAT,
    end_long FLOAT,
    raw_data JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_trips_vin ON trips (vin);
CREATE INDEX IF NOT EXISTS idx_trips_start ON trips (start_time);
"""

_DDL_SERVICE = """
CREATE TABLE IF NOT EXISTS service_history (
    id BIGSERIAL PRIMARY KEY,
    dealer_code VARCHAR(20),
    region VARCHAR(50),
    company_code VARCHAR(20),
    created_on DATE,
    zone VARCHAR(50),
    dealer_name VARCHAR(100),
    dealer_city VARCHAR(50),
    license_plate VARCHAR(20),
    vin VARCHAR(17),
    status VARCHAR(20),
    service_type VARCHAR(50),
    model_sales_code VARCHAR(20),
    model_description VARCHAR(200),
    color VARCHAR(30),
    mileage FLOAT,
    order_item VARCHAR(20),
    lab_part VARCHAR(20),
    material_group VARCHAR(20),
    description VARCHAR(200),
    order_quantity FLOAT,
    unit_price FLOAT,
    net_value FLOAT,
    tax FLOAT,
    total_value FLOAT,
    gross_value FLOAT,
    key_field VARCHAR(50),
    issue_type VARCHAR(50),
    warranty_contribution FLOAT,
    insurance_contribution FLOAT,
    discount_contribution FLOAT,
    raw_data JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_service_vin ON service_history (vin);
CREATE INDEX IF NOT EXISTS idx_service_dealer ON service_history (dealer_code);
"""

_pg_engine = None


def _get_pg_engine():
    global _pg_engine
    if _pg_engine is None:
        _pg_engine = create_engine(PG_URL, pool_pre_ping=True)
        with _pg_engine.connect() as conn:
            conn.execute(text(_DDL_TRIPS))
            conn.execute(text(_DDL_SERVICE))
            conn.commit()
    return _pg_engine


class FileIngestor:
    """Ingest CSV or JSON files into InfluxDB and PostgreSQL."""

    def ingest_telemetry_csv(self, filepath: str | Path, progress_cb=None) -> dict:
        """Read a telemetry CSV, write each row to InfluxDB as vehicle_telemetry."""
        df = pd.read_csv(filepath, low_memory=False)
        df = df.rename(columns={k: v for k, v in TELEMETRY_COL_MAP.items() if k in df.columns})

        if "vin" not in df.columns:
            return {"error": "No VIN column found", "uploaded": 0, "failed": 0}

        uploaded = failed = 0
        total = len(df)

        for i, row in enumerate(df.itertuples(index=False)):
            row_dict = row._asdict()
            vin = str(row_dict.get("vin", "UNKNOWN"))
            ts  = row_dict.get("timestamp") or datetime.now(timezone.utc).isoformat()

            numeric_fields = {
                k: float(v) for k, v in row_dict.items()
                if k not in ("vin", "timestamp", "date")
                and v is not None
                and str(v).strip() not in ("", "nan", "None")
                and _is_numeric(v)
            }
            try:
                write_influx("vehicle_telemetry", {"vin": vin}, numeric_fields, ts)
                uploaded += 1
            except Exception as exc:
                log.debug("Row %d failed: %s", i, exc)
                failed += 1

            if progress_cb and i % 500 == 0:
                progress_cb(i, total)

        return {"uploaded": uploaded, "failed": failed, "total": total, "errors": []}

    def ingest_trip_csv(self, filepath: str | Path, progress_cb=None) -> dict:
        """Read a trips CSV and insert rows into the trips PostgreSQL table."""
        df = pd.read_csv(filepath, low_memory=False)
        df = df.rename(columns={k: v for k, v in TRIP_COL_MAP.items() if k in df.columns})

        engine = _get_pg_engine()
        uploaded = failed = 0
        errors: list[str] = []
        total = len(df)

        KNOWN_COLS = set(TRIP_COL_MAP.values())
        db_cols = [c for c in df.columns if c in KNOWN_COLS]

        for i, row in enumerate(df[db_cols].itertuples(index=False)):
            row_dict = {db_cols[j]: _safe_val(v) for j, v in enumerate(row)}
            row_dict["raw_data"] = json.dumps({})
            try:
                cols = ", ".join(row_dict.keys())
                placeholders = ", ".join(f":{k}" for k in row_dict.keys())
                with engine.connect() as conn:
                    conn.execute(text(f"INSERT INTO trips ({cols}) VALUES ({placeholders})"), row_dict)
                    conn.commit()
                uploaded += 1
            except Exception as exc:
                failed += 1
                if len(errors) < 20:
                    errors.append(f"Row {i}: {exc}")

            if progress_cb and i % 200 == 0:
                progress_cb(i, total)

        return {"uploaded": uploaded, "failed": failed, "total": total, "errors": errors}

    def ingest_service_history_csv(self, filepath: str | Path, progress_cb=None) -> dict:
        """Read a service history CSV and insert into service_history table."""
        df = pd.read_csv(filepath, low_memory=False)
        df = df.rename(columns={k: v for k, v in SERVICE_COL_MAP.items() if k in df.columns})

        engine = _get_pg_engine()
        uploaded = failed = 0
        errors: list[str] = []
        total = len(df)

        KNOWN_COLS = set(SERVICE_COL_MAP.values())
        db_cols = [c for c in df.columns if c in KNOWN_COLS]

        for i, row in enumerate(df[db_cols].itertuples(index=False)):
            row_dict = {db_cols[j]: _safe_val(v) for j, v in enumerate(row)}
            row_dict["raw_data"] = json.dumps({})
            try:
                cols = ", ".join(row_dict.keys())
                placeholders = ", ".join(f":{k}" for k in row_dict.keys())
                with engine.connect() as conn:
                    conn.execute(text(f"INSERT INTO service_history ({cols}) VALUES ({placeholders})"), row_dict)
                    conn.commit()
                uploaded += 1
            except Exception as exc:
                failed += 1
                if len(errors) < 20:
                    errors.append(f"Row {i}: {exc}")

            if progress_cb and i % 200 == 0:
                progress_cb(i, total)

        return {"uploaded": uploaded, "failed": failed, "total": total, "errors": errors}

    def ingest_json(self, filepath: str | Path, data_type: str) -> dict:
        """Ingest a JSON file. data_type: 'telemetry' | 'trips' | 'service'."""
        with open(filepath) as f:
            data = json.load(f)

        records = data if isinstance(data, list) else [data]
        tmp = Path(str(filepath) + ".tmp.csv")
        pd.DataFrame(records).to_csv(tmp, index=False)

        try:
            if data_type == "telemetry":
                return self.ingest_telemetry_csv(tmp)
            elif data_type == "trips":
                return self.ingest_trip_csv(tmp)
            elif data_type == "service":
                return self.ingest_service_history_csv(tmp)
            else:
                return {"error": f"Unknown data_type: {data_type}"}
        finally:
            tmp.unlink(missing_ok=True)


def _is_numeric(val) -> bool:
    try:
        float(val)
        return True
    except (TypeError, ValueError):
        return False


def _safe_val(v):
    if v is None:
        return None
    s = str(v).strip()
    if s in ("", "nan", "NaN", "None", "NaT"):
        return None
    return v
