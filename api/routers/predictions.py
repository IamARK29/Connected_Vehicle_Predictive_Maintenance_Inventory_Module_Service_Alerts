"""Predictive maintenance inference endpoints."""
from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_current_user
from api.schemas import PredictionRequest, PredictionResponse

router = APIRouter(prefix="/predictions", tags=["Predictions"])

_MODEL_MAP = {
    "brakes": "models.brake_wear_model",
    "engine": "models.engine_oil_model",
    "hv_battery": "models.hv_battery_soh_model",
    "battery_12v": "models.battery_12v_model",
    "tyres": "models.tyre_wear_model",
    "fuel": "models.fuel_anomaly_model",
    "driver": "models.driver_score_model",
}


def _run_model(module_path: str, telemetry: dict) -> dict:
    import importlib
    module = importlib.import_module(module_path)
    return module.predict(telemetry)


@router.post("/{component}", response_model=PredictionResponse)
async def predict_component(
    component: str,
    payload: PredictionRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    module_path = _MODEL_MAP.get(component)
    if not module_path:
        raise HTTPException(status_code=404, detail=f"No model for component: {component}. Valid: {list(_MODEL_MAP.keys())}")

    try:
        result = _run_model(module_path, payload.telemetry)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail=f"Model not trained yet for {component}. Run training first.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(exc)}")

    return PredictionResponse(vin=payload.vin, component=component, predictions=result)


@router.post("/all", response_model=list[PredictionResponse])
async def predict_all(
    payload: PredictionRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    results = []
    for component, module_path in _MODEL_MAP.items():
        try:
            result = _run_model(module_path, payload.telemetry)
            results.append(PredictionResponse(vin=payload.vin, component=component, predictions=result))
        except Exception:
            pass
    return results
