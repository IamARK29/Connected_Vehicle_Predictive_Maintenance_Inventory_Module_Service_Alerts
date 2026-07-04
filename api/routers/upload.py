"""
File upload endpoints — Mode B ingestion.

Accepts CSV/JSON uploads, queues a Celery task, and streams
progress to the client via WebSocket /ws/upload/{job_id}.
"""
from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Annotated

import redis
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect

from fastapi.responses import StreamingResponse
from api.dependencies import get_current_user

router = APIRouter(prefix="/upload", tags=["File Upload"])

REDIS_URL    = os.getenv("REDIS_URL", "redis://localhost:6379/0")
UPLOAD_DIR   = Path(os.getenv("UPLOAD_DIR", "data/uploads"))
MAX_FILE_MB  = int(os.getenv("MAX_UPLOAD_MB", "500"))
MAX_BYTES    = MAX_FILE_MB * 1024 * 1024

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

_redis: redis.Redis | None = None


def _get_redis() -> redis.Redis | None:
    global _redis
    if _redis is None:
        try:
            r = redis.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=2)
            r.ping()
            _redis = r
        except Exception:
            pass
    return _redis


def _get_job_progress(job_id: str) -> dict:
    r = _get_redis()
    if r:
        raw = r.get(f"upload:job:{job_id}")
        if raw:
            return json.loads(raw)
    return {"pct": 0, "message": "Queued", "result": {}}


async def _save_upload(file: UploadFile) -> Path:
    content = await file.read()
    if len(content) > MAX_BYTES:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_FILE_MB} MB limit")
    suffix = Path(file.filename or "upload").suffix or ".csv"
    dest = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    dest.write_bytes(content)
    return dest


def _queue_task(filepath: Path, data_type: str) -> str:
    job_id = uuid.uuid4().hex
    try:
        from ingestion.tasks import ingest_file_task
        ingest_file_task.apply_async(args=[job_id, str(filepath), data_type], task_id=job_id)
    except Exception as exc:
        # Celery not available — run synchronously and store progress in memory
        from ingestion.file_ingestor import FileIngestor
        ingestor = FileIngestor()
        result = {"error": f"Celery unavailable ({exc}); ran synchronously"}
        try:
            if data_type == "telemetry":
                result = ingestor.ingest_telemetry_csv(filepath)
            elif data_type == "trips":
                result = ingestor.ingest_trip_csv(filepath)
            elif data_type == "service":
                result = ingestor.ingest_service_history_csv(filepath)
        except Exception as e2:
            result = {"error": str(e2), "uploaded": 0, "failed": 0}
        finally:
            filepath.unlink(missing_ok=True)

        r = _get_redis()
        if r:
            r.setex(f"upload:job:{job_id}", 3600, json.dumps({"pct": 100, "message": "Complete", "result": result}))
    return job_id


# ── CSV templates ──────────────────────────────────────────────────────────

_TEMPLATES: dict[str, str] = {
    "telemetry": "StartTime-TimeStamp,VIN,vehSpeed,vehSysPwrMod,vehGearPos,tboxAccelX,tboxAccelY,tboxAccelZ,vehBatt,vehOdo,vehCoolantTemp,vehOutsideTemp,vehAC,vehBrkFludLvlLow,vehABSF,vehOilPressureWarning,vehMILWarning,frontLeftTyrePressure,frontRrightTyrePressure,rearLeftTyrePressure,rearRightTyrePressure,wheelTyreMonitorStatus,gnssLat,gnssLong,vehRPM,FuelTankLevel,vehBMSPackSOC,vehBMSPackVol,vehBMSPackCrnt,vehBMSCellMaxTem,vehBMSCMUFlt,vehBMSPackTemFlt\n",
    "trips": "tripId,vin,startTime,endTime,startOdometer,endOdometer,odometer,averageSpeed,maxSpeed,vehFuelConsumed,fuelEfficiency,driveScore,harshBreakingNum,suddenTurnNum,accelerationNum,startPoint_lat,startPoint_long,endPoint_lat,endPoint_long\n",
    "service": "VIN,CreatedOn,ServiceType,DealerCode,DealerName,ModelSalesCode,DescriptionOne,OrderQuantity,NetValue,Mileage\n",
}


@router.get("/templates/{template_name}")
async def download_template(template_name: str):
    """Download a CSV template for telemetry, trips, or service uploads."""
    key = template_name.replace(".csv", "")
    if key not in _TEMPLATES:
        raise HTTPException(status_code=404, detail=f"Template '{template_name}' not found. Available: {list(_TEMPLATES.keys())}")
    import io
    content = _TEMPLATES[key]
    return StreamingResponse(
        io.BytesIO(content.encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={key}_template.csv"},
    )


# ── Endpoints ───────────────────────────────────────────────────────────────

@router.post("/telemetry")
async def upload_telemetry(
    file: Annotated[UploadFile, File(description="Telemetry CSV")],
    current_user: Annotated[dict, Depends(get_current_user)],
):
    """Upload a telemetry CSV for async ingestion into InfluxDB."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files accepted")
    path = await _save_upload(file)
    job_id = _queue_task(path, "telemetry")
    return {"job_id": job_id, "status": "queued", "filename": file.filename, "ws": f"/ws/upload/{job_id}"}


@router.post("/trips")
async def upload_trips(
    file: Annotated[UploadFile, File(description="Trips CSV")],
    current_user: Annotated[dict, Depends(get_current_user)],
):
    """Upload a trips CSV for async ingestion into PostgreSQL trips table."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files accepted")
    path = await _save_upload(file)
    job_id = _queue_task(path, "trips")
    return {"job_id": job_id, "status": "queued", "filename": file.filename, "ws": f"/ws/upload/{job_id}"}


@router.post("/service-history")
async def upload_service_history(
    file: Annotated[UploadFile, File(description="Service history CSV")],
    current_user: Annotated[dict, Depends(get_current_user)],
):
    """Upload a service history CSV for async ingestion into PostgreSQL."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files accepted")
    path = await _save_upload(file)
    job_id = _queue_task(path, "service")
    return {"job_id": job_id, "status": "queued", "filename": file.filename, "ws": f"/ws/upload/{job_id}"}


@router.post("/json")
async def upload_json(
    file: Annotated[UploadFile, File(description="JSON data file")],
    data_type: str = "telemetry",
    current_user: Annotated[dict, Depends(get_current_user)] = None,
):
    """Upload a JSON file. data_type: telemetry | trips | service."""
    if data_type not in ("telemetry", "trips", "service"):
        raise HTTPException(status_code=400, detail="data_type must be telemetry | trips | service")
    path = await _save_upload(file)
    job_id = _queue_task(path, f"json_{data_type}")
    return {"job_id": job_id, "status": "queued", "filename": file.filename, "ws": f"/ws/upload/{job_id}"}


@router.get("/status/{job_id}")
async def get_upload_status(job_id: str, current_user: Annotated[dict, Depends(get_current_user)]):
    """Poll upload job progress (alternative to WebSocket)."""
    return _get_job_progress(job_id)


# ── WebSocket progress stream ───────────────────────────────────────────────

@router.websocket("/ws/{job_id}")
async def upload_progress_ws(websocket: WebSocket, job_id: str):
    """Stream upload progress updates until job reaches 100%."""
    import asyncio
    await websocket.accept()
    try:
        while True:
            progress = _get_job_progress(job_id)
            await websocket.send_json(progress)
            if progress.get("pct", 0) >= 100:
                break
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await websocket.send_json({"error": str(exc)})
        except Exception:
            pass
