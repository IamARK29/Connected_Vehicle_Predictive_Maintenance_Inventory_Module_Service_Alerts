"""
InfluxDB + PostgreSQL writers for TBox channel data.
Measurements follow the 19-name convention from the Big Data spec.

CLI:
  python -m ingestion.db_writer setup-buckets
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

log = logging.getLogger(__name__)

INFLUX_URL    = os.getenv("INFLUXDB_URL", "http://localhost:8086")
INFLUX_TOKEN  = os.getenv("INFLUXDB_TOKEN", "autopredict-dev-token")
INFLUX_ORG    = os.getenv("INFLUXDB_ORG", "autopredict")
INFLUX_BUCKET = os.getenv("INFLUXDB_BUCKET", "telemetry")
PG_URL        = os.getenv("POSTGRES_URL", "postgresql://autopredict:autopredict@localhost:5432/autopredict")

# Fields stored as strings (tags / metadata), not InfluxDB numeric fields
_STRING_FIELDS = {"imei", "iccid", "networkOperator", "milDtcCodes"}
_BOOL_FIELDS = {
    "driverDoorAjar", "passDoorAjar", "rlDoorAjar", "rrDoorAjar", "bootDoorAjar",
    "bonnetAjar", "driverDoorLock", "passDoorLock", "rlDoorLock", "rrDoorLock",
    "centralLock", "cruiseActive", "cruiseAccelerate", "cruiseDecelerate",
    "cruiseCancel", "cruiseResume", "chargePlugConnected", "chargeActive",
    "hornActive", "milActive", "absActive", "tcsActive", "espActive",
    "headLightsOn", "highBeamOn", "fogFrontOn", "fogRearOn", "leftTurnSignal",
    "rightTurnSignal", "hazardOn", "drlOn", "parkingLightsOn", "rainSensorActive",
    "nightModeActive", "wiperRearActive", "autoLightsActive", "engineRunning",
    "hvacACActive", "hvacRecircActive", "hvacRearDefrost", "hvacFrontDefrost",
    "seatbeltDriver", "seatbeltPass", "seatbeltRL", "seatbeltRR", "seatbeltRCentre",
    "seatbeltRMid", "parkingBrakeActive", "airbagDeployedAny", "airbagDriverDeployed",
    "airbagPassDeployed", "airbagSideRLDeployed", "airbagSideRRDeployed",
    "thermalRunawayActive", "vehBMSBalancing",
}

_influx_client: InfluxDBClient | None = None
_pg_engine = None


def _get_influx_write_api():
    global _influx_client
    if _influx_client is None:
        _influx_client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
    return _influx_client.write_api(write_options=SYNCHRONOUS)


def _get_pg_engine():
    global _pg_engine
    if _pg_engine is None:
        from sqlalchemy import create_engine
        _pg_engine = create_engine(PG_URL, pool_pre_ping=True)
        _ensure_state_table()
    return _pg_engine


def _ensure_state_table() -> None:
    from sqlalchemy import text
    ddl = """
    CREATE TABLE IF NOT EXISTS vehicle_state (
        vin VARCHAR(17) PRIMARY KEY,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        latest_data JSONB NOT NULL DEFAULT '{}'
    );
    """
    with _pg_engine.connect() as conn:
        conn.execute(text(ddl))
        conn.commit()


def write_influx(
    measurement: str,
    tags: dict[str, str],
    fields: dict[str, Any],
    timestamp: str | datetime | None = None,
) -> None:
    """Write a single point to InfluxDB."""
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    point = Point(measurement).time(timestamp, WritePrecision.SECONDS)
    for k, v in tags.items():
        point = point.tag(k, str(v))

    for k, v in fields.items():
        if k in _STRING_FIELDS:
            continue  # skip string metadata from numeric fields
        if k in _BOOL_FIELDS:
            point = point.field(k, bool(v))
        elif isinstance(v, bool):
            point = point.field(k, v)
        else:
            try:
                point = point.field(k, float(v))
            except (TypeError, ValueError):
                point = point.field(k, str(v))

    try:
        write_api = _get_influx_write_api()
        write_api.write(bucket=INFLUX_BUCKET, record=point)
    except Exception as exc:
        log.error("InfluxDB write failed [%s]: %s", measurement, exc)


def write_influx_batch(points: list[Point]) -> None:
    """Write a batch of pre-built Points to InfluxDB."""
    if not points:
        return
    try:
        write_api = _get_influx_write_api()
        write_api.write(bucket=INFLUX_BUCKET, record=points)
    except Exception as exc:
        log.error("InfluxDB batch write failed (%d points): %s", len(points), exc)


def write_postgres_state(vin: str, latest_values: dict[str, Any]) -> None:
    """Upsert the latest channel values for a VIN into vehicle_state."""
    from sqlalchemy import text
    engine = _get_pg_engine()
    upsert = text("""
        INSERT INTO vehicle_state (vin, updated_at, latest_data)
        VALUES (:vin, NOW(), :data::jsonb)
        ON CONFLICT (vin) DO UPDATE
            SET updated_at = NOW(),
                latest_data = vehicle_state.latest_data || :data::jsonb
    """)
    try:
        with engine.connect() as conn:
            conn.execute(upsert, {"vin": vin, "data": json.dumps(latest_values)})
            conn.commit()
    except Exception as exc:
        log.error("PostgreSQL state write failed for VIN %s: %s", vin, exc)


# ── InfluxDB bucket management ─────────────────────────────────────────────

_BUCKET_SPECS = [
    ("tbox_hf",          30),    # 30-day retention — high-frequency drive data
    ("tbox_standard",    90),    # 90-day retention — standard 1 Hz telemetry
    ("tbox_lf",          180),   # 180-day retention — low-frequency / config
    ("tbox_aggregated",  0),     # infinite retention — 5-min aggregated views
    ("tbox_dtc",         730),   # 2-year retention — diagnostic trouble codes
]


def setup_influx_buckets() -> dict[str, str]:
    """
    Create all required InfluxDB buckets (idempotent — safe to run on every startup).
    Returns {bucket_name: "created" | "exists" | "error"}.
    """
    from influxdb_client import BucketRetentionRules
    from influxdb_client.client.exceptions import ApiException

    results: dict[str, str] = {}
    try:
        client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
        buckets_api = client.buckets_api()
        org_id = client.organizations_api().find_organizations(org=INFLUX_ORG)[0].id

        for name, retention_days in _BUCKET_SPECS:
            try:
                existing = buckets_api.find_bucket_by_name(name)
                if existing:
                    results[name] = "exists"
                    continue
            except Exception:
                pass
            try:
                retention = (
                    [BucketRetentionRules(type="expire", every_seconds=retention_days * 86400)]
                    if retention_days > 0 else []
                )
                buckets_api.create_bucket(
                    bucket_name=name,
                    org_id=org_id,
                    retention_rules=retention,
                )
                results[name] = "created"
                log.info("InfluxDB bucket created: %s (retention=%sd)", name, retention_days or "∞")
            except ApiException as exc:
                if "already exists" in str(exc).lower() or exc.status == 422:
                    results[name] = "exists"
                else:
                    log.error("Failed to create bucket %s: %s", name, exc)
                    results[name] = "error"
            except Exception as exc:
                log.error("Failed to create bucket %s: %s", name, exc)
                results[name] = "error"

        client.close()
    except Exception as exc:
        log.error("setup_influx_buckets failed: %s", exc)
    return results


# ── Legacy compatibility for the telemetry router ──────────────────────────

def write_telemetry_sync(data: dict[str, Any]) -> None:
    """Write a generic telemetry dict to InfluxDB (legacy / file-upload path)."""
    vin = data.get("vin", "UNKNOWN")
    ts = data.get("timestamp") or datetime.now(timezone.utc).isoformat()
    numeric = {
        k: v for k, v in data.items()
        if k not in ("vin", "timestamp") and isinstance(v, (int, float))
    }
    write_influx("vehicle_telemetry", {"vin": vin}, numeric, ts)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) > 1 and sys.argv[1] == "setup-buckets":
        results = setup_influx_buckets()
        for bucket, status in results.items():
            print(f"  {status:10s}  {bucket}")
    else:
        print("Usage: python -m ingestion.db_writer setup-buckets")
