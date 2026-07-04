"""
Synthetic data generation + model training endpoints.

POST /api/synthetic/generate  -> generate fleet CSVs in background
POST /api/synthetic/train     -> train ML models from generated CSVs
GET  /api/synthetic/status/{job_id} -> poll progress
"""
from __future__ import annotations

import json
import os
import pathlib
import traceback
import uuid
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends
from pydantic import BaseModel, Field

from api.dependencies import get_current_user

router = APIRouter(prefix="/synthetic", tags=["Synthetic Data"])

DATA_DIR  = os.getenv("DATA_DIR", "data/synthetic")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

_PROGRESS: dict[str, dict] = {}


class SyntheticGenRequest(BaseModel):
    num_vehicles:    int   = Field(default=10,   ge=1,   le=100)
    num_days:        int   = Field(default=90,   ge=7,   le=365)
    failure_rate:    float = Field(default=0.05, ge=0.0, le=0.30)
    sessions_per_day: int  = Field(default=4,    ge=2,   le=24)


def _set_progress(job_id: str, pct: int, message: str, result: dict | None = None) -> None:
    payload = {"pct": pct, "message": message, "result": result or {}}
    _PROGRESS[job_id] = payload
    try:
        import redis as _redis
        r = _redis.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=2)
        r.setex(f"upload:job:{job_id}", 3600, json.dumps(payload))
    except Exception:
        pass


def get_progress(job_id: str) -> dict:
    if job_id in _PROGRESS:
        return _PROGRESS[job_id]
    try:
        import redis as _redis
        r = _redis.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=2)
        raw = r.get(f"upload:job:{job_id}")
        if raw:
            return json.loads(raw)
    except Exception:
        pass
    return {"pct": -1, "message": "Job not found", "result": {}}


# ---------------------------------------------------------------------------
# Background: generate synthetic data
# ---------------------------------------------------------------------------

def _run_generation(job_id: str, num_vehicles: int, num_days: int, failure_rate: float, sessions_per_day: int = 4) -> None:
    import sys
    project_root = str(pathlib.Path(__file__).resolve().parents[2])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    out_dir = pathlib.Path(DATA_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        _set_progress(job_id, 5, "Generating fleet profiles...")
        from synthetic.config import SyntheticConfig
        from synthetic.generate_fleet import generate_fleet

        cfg = SyntheticConfig(
            num_vehicles=num_vehicles,
            num_days=num_days,
            failure_injection_rate=failure_rate,
            sessions_per_day=sessions_per_day,
        )
        fleet_df = generate_fleet(cfg)
        fleet_df.to_csv(out_dir / "fleet.csv", index=False)
        _set_progress(job_id, 15, f"Fleet: {num_vehicles} VINs. Generating telemetry...")

        from synthetic.generate_telemetry import TelemetryGenerator
        gen = TelemetryGenerator(cfg)
        gen.generate_all(fleet_df, out_dir)
        _set_progress(job_id, 55, "Telemetry done. Generating trips...")

        from synthetic.generate_trips import generate_trips
        generate_trips(fleet_df, cfg, out_dir)
        _set_progress(job_id, 70, "Trips done. Generating service history...")

        try:
            from synthetic.generate_service_history import generate_service_history
            generate_service_history(fleet_df, cfg, out_dir)
        except Exception as exc:
            print(f"Service history skipped: {exc}")

        _set_progress(job_id, 78, "Generating DTC events...")
        try:
            import pandas as pd
            from synthetic.generate_dtcs import DTCGenerator
            manifest_path = out_dir / "failures_manifest.csv"
            manifest_df = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame(
                columns=["VIN", "failure_type", "failure_date"]
            )
            dtc_df = DTCGenerator().generate(fleet_df, manifest_df, num_days)
            # Write explicitly to out_dir so it's always in the right place
            dtc_out = out_dir / "dtc_events.csv"
            dtc_df.to_csv(dtc_out, index=False)
            _set_progress(job_id, 84, f"DTC events: {len(dtc_df)} rows. Generating OTA events...")
        except Exception as exc:
            print(f"DTC generation skipped: {exc}")
            _set_progress(job_id, 84, "DTC skipped. Generating OTA events...")

        try:
            from synthetic.generate_ota import generate_ota
            start_date_str = cfg.start_date if isinstance(cfg.start_date, str) else str(cfg.start_date)[:10]
            generate_ota(fleet_df, out_dir, start_date_str, num_days)
            _set_progress(job_id, 90, "OTA events done. Generating parts inventory...")
        except Exception as exc:
            print(f"OTA generation skipped: {exc}")
            _set_progress(job_id, 90, "OTA skipped. Generating parts inventory...")

        try:
            from synthetic.generate_parts_inventory import generate_parts_inventory
            start_date_str = cfg.start_date if isinstance(cfg.start_date, str) else str(cfg.start_date)[:10]
            generate_parts_inventory(fleet_df, out_dir, start_date_str)
            _set_progress(job_id, 96, "Parts inventory done.")
        except Exception as exc:
            print(f"Parts inventory generation skipped: {exc}")

        _set_progress(job_id, 100,
                     f"Data generated: {num_vehicles} VINs, {num_days} days. "
                     f"Telemetry, trips, service history, DTCs, OTA events, parts inventory ready.",
                     {"vehicles": num_vehicles, "days": num_days, "failure_rate": failure_rate,
                      "status": "success", "next_step": "train"})

    except Exception as exc:
        traceback.print_exc()
        _set_progress(job_id, 100,
                     f"Generation failed: {exc}",
                     {"status": "error", "error": str(exc)})


# ---------------------------------------------------------------------------
# Background: train models from CSV
# ---------------------------------------------------------------------------

def _run_training(job_id: str) -> None:
    import sys
    project_root = str(pathlib.Path(__file__).resolve().parents[2])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    out_dir = pathlib.Path(DATA_DIR)

    try:
        _set_progress(job_id, 5, "Loading fleet and telemetry CSVs...")
        from models.train_all import train_all

        def _cb(pct: int, msg: str):
            _set_progress(job_id, pct, msg)

        summary = train_all(
            data_dir=out_dir,
            experiment="autopredict-web",
            snapshot_interval_days=14,
            progress_callback=_cb,
        )
        trained = int((summary["status"] == "ok").sum()) if not summary.empty else 0
        total = len(summary)
        _set_progress(job_id, 100,
                     f"Training complete: {trained}/{total} models trained successfully.",
                     {"status": "success", "trained": trained, "total": total,
                      "summary": summary.to_dict("records") if not summary.empty else []})

    except Exception as exc:
        traceback.print_exc()
        _set_progress(job_id, 100,
                     f"Training failed: {exc}",
                     {"status": "error", "error": str(exc)})


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/generate", status_code=202, summary="Generate synthetic fleet data")
async def generate_synthetic_data(
    payload:          SyntheticGenRequest,
    background_tasks: BackgroundTasks,
    current_user:     Annotated[dict, Depends(get_current_user)],
):
    job_id = uuid.uuid4().hex
    _set_progress(job_id, 0, "Queued")
    background_tasks.add_task(
        _run_generation, job_id,
        payload.num_vehicles, payload.num_days, payload.failure_rate, payload.sessions_per_day,
    )
    return {
        "job_id":   job_id,
        "status":   "queued",
        "poll_url": f"/api/synthetic/status/{job_id}",
        "params":   payload.model_dump(),
    }


@router.post("/train", status_code=202, summary="Train ML models from generated data")
async def train_models(
    background_tasks: BackgroundTasks,
    current_user:     Annotated[dict, Depends(get_current_user)],
):
    job_id = uuid.uuid4().hex
    _set_progress(job_id, 0, "Training queued")
    background_tasks.add_task(_run_training, job_id)
    return {
        "job_id":   job_id,
        "status":   "queued",
        "poll_url": f"/api/synthetic/status/{job_id}",
    }


@router.get("/status/{job_id}", summary="Poll job progress")
async def get_generation_status(job_id: str):
    return get_progress(job_id)
