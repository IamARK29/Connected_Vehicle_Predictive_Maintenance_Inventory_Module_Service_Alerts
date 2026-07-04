"""Celery tasks for async file ingestion (Mode B)."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import redis

from celery_app import celery
from ingestion.file_ingestor import FileIngestor

log = logging.getLogger(__name__)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

_redis: redis.Redis | None = None


def _get_redis() -> redis.Redis:
    global _redis
    if _redis is None:
        _redis = redis.from_url(REDIS_URL, decode_responses=True)
    return _redis


def _set_progress(job_id: str, pct: int, message: str, result: dict | None = None) -> None:
    payload = {"pct": pct, "message": message, "result": result or {}}
    try:
        _get_redis().setex(f"upload:job:{job_id}", 3600, json.dumps(payload))
    except Exception:
        pass


def _make_progress_cb(job_id: str, phase: str):
    def cb(done: int, total: int) -> None:
        pct = int(done / total * 90) if total else 0
        _set_progress(job_id, pct, f"{phase}: {done}/{total} rows")
    return cb


@celery.task(bind=True, name="ingest_file")
def ingest_file_task(self, job_id: str, filepath: str, data_type: str) -> dict:
    """Process an uploaded file and write to InfluxDB/PostgreSQL."""
    _set_progress(job_id, 5, f"Starting {data_type} ingestion")

    ingestor = FileIngestor()
    path = Path(filepath)

    try:
        if data_type == "telemetry":
            result = ingestor.ingest_telemetry_csv(path, _make_progress_cb(job_id, "Telemetry"))
        elif data_type == "trips":
            result = ingestor.ingest_trip_csv(path, _make_progress_cb(job_id, "Trips"))
        elif data_type == "service":
            result = ingestor.ingest_service_history_csv(path, _make_progress_cb(job_id, "Service"))
        elif data_type == "json_telemetry":
            result = ingestor.ingest_json(path, "telemetry")
        elif data_type == "json_trips":
            result = ingestor.ingest_json(path, "trips")
        elif data_type == "json_service":
            result = ingestor.ingest_json(path, "service")
        else:
            result = {"error": f"Unknown data_type: {data_type}"}

        _set_progress(job_id, 100, "Complete", result)
        return result

    except Exception as exc:
        error = {"error": str(exc), "uploaded": 0, "failed": 0}
        _set_progress(job_id, 100, f"Failed: {exc}", error)
        raise
    finally:
        path.unlink(missing_ok=True)
