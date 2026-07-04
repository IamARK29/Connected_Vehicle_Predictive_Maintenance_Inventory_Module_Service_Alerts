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
    num_vehicles: int   = Field(default=10,   ge=1,   le=100)
    num_days:     int   = Field(default=90,   ge=7,   le=365)
    failure_rate: float = Field(default=0.05, ge=0.0, le=0.30)


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

def _run_generation(job_id: str, num_vehicles: int, num_days: int, failure_rate: float) -> None:
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

        _set_progress(job_id, 80, "Generating DTCs and OTA events...")
        try:
            from synthetic.generate_dtcs import DTCGenerator
            import pandas as pd
            manifest_path = out_dir / "failures_manifest.csv"
            if manifest_path.exists():
                manifest_df = pd.read_csv(manifest_path)
                DTCGenerator().generate(fleet_df, manifest_df)
        except Exception as exc:
            print(f"DTC generation skipped: {exc}")

        try:
            from synthetic.generate_ota import generate_ota
            generate_ota(fleet_df, out_dir, cfg.start_date, cfg.num_days)
        except Exception:
            pass

        _set_progress(job_id, 100,
                     f"Data generated: {num_vehicles} VINs, {num_days} days. Ready to train models.",
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
        payload.num_vehicles, payload.num_days, payload.failure_rate,
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
