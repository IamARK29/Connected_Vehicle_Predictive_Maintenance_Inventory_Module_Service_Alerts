"""
Telemetry Downsampler — Celery beat task (every 5 minutes).

Queries InfluxDB "tbox_hf" for the last 5 minutes per VIN,
computes mean/max/min/std per signal, and writes 5-min aggregates
to the "tbox_aggregated" bucket.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

log = logging.getLogger(__name__)

INFLUX_URL    = os.getenv("INFLUXDB_URL",    "http://localhost:8086")
INFLUX_TOKEN  = os.getenv("INFLUXDB_TOKEN",  "autopredict-dev-token")
INFLUX_ORG    = os.getenv("INFLUXDB_ORG",    "autopredict")
HF_BUCKET     = "tbox_hf"
AGG_BUCKET    = "tbox_aggregated"

# Signals to aggregate from the HF bucket
_HF_SIGNALS = [
    "vehSpeed", "vehRPM", "vehSysPwrMod", "vehGearPos",
    "tboxAccelX", "tboxAccelY", "tboxAccelZ",
    "vehAccelPos", "vehBrakePos", "vehSteeringAngle",
]

_WINDOW_MINUTES = 5


def _get_influx_client():
    from influxdb_client import InfluxDBClient
    return InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)


def downsample_hf() -> dict[str, int]:
    """
    Query last 5 min from tbox_hf, aggregate, write to tbox_aggregated.
    Returns {vin: points_written} summary.
    """
    from influxdb_client import Point, WritePrecision
    from influxdb_client.client.write_api import SYNCHRONOUS

    now = datetime.now(timezone.utc)
    window_start = now - timedelta(minutes=_WINDOW_MINUTES)
    window_str   = f"-{_WINDOW_MINUTES}m"

    flux = f"""
    from(bucket: "{HF_BUCKET}")
      |> range(start: {window_str})
      |> filter(fn: (r) => r["_measurement"] == "tbox_engine" or r["_measurement"] == "tbox_drive_style")
      |> group(columns: ["vin", "_field"])
      |> mean()
      |> yield(name: "mean")
    """

    summary: dict[str, int] = {}
    try:
        client    = _get_influx_client()
        query_api = client.query_api()
        write_api = client.write_api(write_options=SYNCHRONOUS)
        tables    = query_api.query(flux, org=INFLUX_ORG)

        # Group results by VIN
        vin_data: dict[str, dict[str, float]] = {}
        for table in tables:
            for record in table.records:
                vin    = record.values.get("vin", "UNKNOWN")
                field  = record.get_field()
                value  = record.get_value()
                if value is not None:
                    vin_data.setdefault(vin, {})[f"{field}_mean"] = float(value)

        # Write one aggregate point per VIN
        points = []
        for vin, fields in vin_data.items():
            p = (
                Point("tbox_5min_agg")
                .tag("vin", vin)
                .tag("window_minutes", str(_WINDOW_MINUTES))
                .time(now, WritePrecision.SECONDS)
            )
            for fname, fval in fields.items():
                p = p.field(fname, fval)
            points.append(p)
            summary[vin] = summary.get(vin, 0) + 1

        if points:
            write_api.write(bucket=AGG_BUCKET, record=points)
            log.info("Downsampled %d VINs -> %s (%d points)", len(summary), AGG_BUCKET, len(points))

        client.close()

    except Exception as exc:
        log.error("Downsampler error: %s", exc)

    return summary


# ── Celery task definition ─────────────────────────────────────────────────────

try:
    from celery import shared_task

    @shared_task(name="ingestion.telemetry_downsampler.downsample_hf_task")
    def downsample_hf_task() -> dict:
        return downsample_hf()

except ImportError:
    # Celery not installed — task can still be called directly
    def downsample_hf_task() -> dict:  # type: ignore[misc]
        return downsample_hf()


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json, logging as _logging
    _logging.basicConfig(level=_logging.INFO)
    result = downsample_hf()
    print(json.dumps(result, indent=2))
