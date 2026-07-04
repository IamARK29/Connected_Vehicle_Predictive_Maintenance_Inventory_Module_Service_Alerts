"""
Parquet Archiver — Celery beat task that runs at 03:00 daily.

For each InfluxDB bucket with finite retention, queries data in the
last 24h before the expiry window and writes to:
    data/archive/{bucket}/{year}/{month}/{vin}/data.parquet
Uses pyarrow with snappy compression.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

ARCHIVE_DIR = Path(os.getenv("ARCHIVE_DIR", "data/archive"))

INFLUXDB_URL = os.getenv("INFLUXDB_URL", "http://localhost:8086")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN", "autopredict-dev-token")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG", "autopredict")

BUCKET_RETENTION: dict[str, dict[str, Any]] = {
    "tbox_hf": {
        "retention_days": 1,
        "description": "high-frequency 10 Hz drive data",
    },
    "tbox_standard": {
        "retention_days": 7,
        "description": "standard 1 Hz vehicle state",
    },
    "tbox_lf": {
        "retention_days": 7,
        "description": "low-frequency / config",
    },
    "tbox_dtc": {
        "retention_days": 730,
        "description": "diagnostic trouble codes",
    },
}


class ParquetArchiver:
    """Archives InfluxDB data about to expire into Parquet files."""

    def archive_all(self) -> dict[str, int]:
        results: dict[str, int] = {}

        for bucket, config in BUCKET_RETENTION.items():
            try:
                count = self._archive_bucket(bucket, config["retention_days"])
                results[bucket] = count
                if count > 0:
                    log.info("Archived %d rows from %s", count, bucket)
            except Exception as exc:
                log.error("Archive failed for %s: %s", bucket, exc)
                results[bucket] = -1

        return results

    def _archive_bucket(self, bucket: str, retention_days: int) -> int:
        import pandas as pd

        try:
            from influxdb_client import InfluxDBClient
        except ImportError:
            log.warning("influxdb_client not installed — skipping archive for %s", bucket)
            return 0

        client = InfluxDBClient(
            url=INFLUXDB_URL,
            token=INFLUXDB_TOKEN,
            org=INFLUXDB_ORG,
        )
        query_api = client.query_api()

        start_offset = f"-{retention_days}d"
        stop_offset = f"-{retention_days - 1}d" if retention_days > 1 else "-0d"

        flux = f'''
            from(bucket: "{bucket}")
            |> range(start: {start_offset}, stop: {stop_offset})
            |> pivot(rowKey:["_time"], columnKey:["_field"], valueColumn:"_value")
            |> sort(columns:["_time"])
        '''

        try:
            tables = query_api.query(flux)
        except Exception as exc:
            log.debug("InfluxDB query failed for %s: %s", bucket, exc)
            client.close()
            return 0

        rows: list[dict] = []
        for table in tables:
            for record in table.records:
                row = {
                    "timestamp": record.get_time(),
                    "vin": record.values.get("vin", ""),
                }
                for k, v in record.values.items():
                    if not k.startswith("_") and k not in ("result", "table"):
                        row[k] = v
                rows.append(row)

        client.close()

        if not rows:
            return 0

        df = pd.DataFrame(rows)

        if "vin" not in df.columns:
            df["vin"] = "unknown"

        now = datetime.now(timezone.utc)
        year = str(now.year)
        month = f"{now.month:02d}"

        total_written = 0
        for vin, vin_df in df.groupby("vin"):
            out_dir = ARCHIVE_DIR / bucket / year / month / str(vin)
            out_dir.mkdir(parents=True, exist_ok=True)

            out_path = out_dir / "data.parquet"

            if out_path.exists():
                existing = pd.read_parquet(out_path)
                vin_df = pd.concat([existing, vin_df], ignore_index=True)
                vin_df = vin_df.drop_duplicates(subset=["timestamp"], keep="last")

            vin_df.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
            total_written += len(vin_df)

        return total_written


# ── Celery task ──────────────────────────────────────────────────────────────

try:
    from celery import Celery
    from celery.schedules import crontab
    import os as _os

    _celery = Celery(
        "parquet_archiver",
        broker=_os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    )

    @_celery.task(name="parquet_archiver.archive_daily")
    def archive_daily_task() -> dict:
        return ParquetArchiver().archive_all()

    _celery.conf.beat_schedule = {
        "archive-daily": {
            "task": "parquet_archiver.archive_daily",
            "schedule": crontab(hour=3, minute=0),
        },
    }

except ImportError:
    pass
