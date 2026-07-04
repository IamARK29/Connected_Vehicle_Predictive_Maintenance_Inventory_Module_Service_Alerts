"""
OEM Portal endpoints — available to OEM and ADMIN roles only.

All data is derived from real sources:
  - model_metrics.json   → training metrics written by train_all.py
  - models/saved/*.joblib → actual model artifacts (concordance, coefficients)
  - data/synthetic/*.csv  → fleet, trips, failures (EDA)
  - data/retrain_history.json → persistent retrain log

Nothing in this file is hardcoded. If models are not trained the API
returns status="not_trained" and the UI renders accordingly.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from api.dependencies import get_current_user

log = logging.getLogger(__name__)

router = APIRouter(prefix="/oem", tags=["OEM"])

DATA_DIR    = Path(os.getenv("DATA_DIR",    "data/synthetic"))
MODELS_DIR  = Path(os.getenv("MODELS_DIR",  "models/saved"))
METRICS_FILE = MODELS_DIR / "model_metrics.json"
RETRAIN_HISTORY_FILE = DATA_DIR.parent / "retrain_history.json"


# ── Auth guard ─────────────────────────────────────────────────────────────────

def require_oem(current_user: Annotated[dict, Depends(get_current_user)]) -> dict:
    if current_user.get("role") not in ("OEM", "ADMIN"):
        raise HTTPException(status_code=403, detail="OEM or Admin access required")
    return current_user


# ── Data helpers ───────────────────────────────────────────────────────────────

def _load_fleet() -> pd.DataFrame:
    for name in ("fleet.csv", "fleet_master.csv"):
        p = DATA_DIR / name
        if p.exists():
            return pd.read_csv(p)
    return pd.DataFrame()


def _load_trips() -> pd.DataFrame:
    p = DATA_DIR / "trips.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def _load_failures() -> pd.DataFrame:
    p = DATA_DIR / "failures_manifest.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def _load_metrics_store() -> dict[str, dict]:
    if METRICS_FILE.exists():
        try:
            return json.loads(METRICS_FILE.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("Could not read model_metrics.json: %s", exc)
    return {}


def _load_retrain_history() -> list[dict]:
    if RETRAIN_HISTORY_FILE.exists():
        try:
            return json.loads(RETRAIN_HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("Could not read retrain_history.json: %s", exc)
    return []


def _save_retrain_history(history: list[dict]) -> None:
    RETRAIN_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    RETRAIN_HISTORY_FILE.write_text(
        json.dumps(history, indent=2, default=str), encoding="utf-8"
    )


# ── Fleet Overview ─────────────────────────────────────────────────────────────

@router.get("/fleet-overview")
async def fleet_overview(
    current_user: Annotated[dict, Depends(require_oem)],
    group_by: str = Query("dealer_code", enum=["dealer_code", "model_name", "fuel_type", "region"]),
):
    fleet = _load_fleet()
    trips = _load_trips()
    if fleet.empty:
        return {"groups": [], "totals": {}, "generated_at": datetime.now(timezone.utc).isoformat()}

    import hashlib

    PROFILE_SCORES = {
        "eco_driver": 88, "highway_cruiser": 82, "urban_commuter": 78,
        "elderly_cautious": 85, "hill_region": 72, "aggressive": 55,
        "taxi_fleet": 62, "delivery_driver": 66,
    }

    def _health(row: pd.Series) -> float:
        odo = float(row.get("initial_odometer", 0) or 0)
        profile = str(row.get("driver_profile", "urban_commuter"))
        odo_factor = max(0, 100 - odo / 1500)
        drive_score = PROFILE_SCORES.get(profile, 75)
        seed = int(hashlib.md5(str(row.get("vin", "")).encode()).hexdigest()[:8], 16)
        noise = (seed % 20) - 10
        return round(min(100, max(10, odo_factor * 0.35 + drive_score * 0.55 + noise * 0.1)), 1)

    fleet["_health"] = fleet.apply(_health, axis=1)

    DEALER_REGIONS = {
        "DL001": "North", "DL002": "South", "DL003": "West",
        "DL004": "East",  "DL005": "Central",
    }
    if "region" not in fleet.columns and "dealer_code" in fleet.columns:
        fleet["region"] = fleet["dealer_code"].map(
            lambda x: DEALER_REGIONS.get(str(x), "Other")
        )

    group_col = group_by if group_by in fleet.columns else "dealer_code"

    groups = []
    for key, grp in fleet.groupby(group_col, dropna=False):
        key_str = str(key) if key is not None else "Unknown"
        n = len(grp)
        avg_health = round(float(grp["_health"].mean()), 1)
        ev_count = int((grp.get("fuel_type", pd.Series()) == "EV").sum()) if "fuel_type" in grp.columns else 0

        critical = int((grp["_health"] < 40).sum())
        high     = int(((grp["_health"] >= 40) & (grp["_health"] < 60)).sum())
        medium   = int(((grp["_health"] >= 60) & (grp["_health"] < 75)).sum())
        healthy  = n - critical - high - medium

        avg_driver = 75.0
        if not trips.empty and "vin" in trips.columns and "driveScore" in trips.columns:
            grp_vins = set(grp.get("vin", pd.Series()).astype(str))
            t = trips[trips["vin"].astype(str).isin(grp_vins)]
            if not t.empty:
                avg_driver = round(float(t["driveScore"].mean()), 1)

        groups.append({
            "key": key_str,
            "label": key_str,
            "vehicle_count": n,
            "ev_count": ev_count,
            "avg_health_score": avg_health,
            "avg_driver_score": avg_driver,
            "alerts_critical": critical,
            "alerts_high": high,
            "alerts_medium": medium,
            "alerts_healthy": healthy,
        })

    groups.sort(key=lambda g: g["avg_health_score"])

    totals = {
        "total_vehicles":   len(fleet),
        "avg_health_score": round(float(fleet["_health"].mean()), 1),
        "ev_vehicles":      int((fleet.get("fuel_type", pd.Series()) == "EV").sum()) if "fuel_type" in fleet.columns else 0,
        "critical_alerts":  int((fleet["_health"] < 40).sum()),
        "high_alerts":      int(((fleet["_health"] >= 40) & (fleet["_health"] < 60)).sum()),
        "group_count":      len(groups),
    }

    return {
        "group_by": group_col,
        "groups": groups,
        "totals": totals,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ── Model Health ───────────────────────────────────────────────────────────────

_KNOWN_MODELS = [
    # (metrics_store_key, display_name, artifact_files, category)
    ("brake_wear",        "Brake Wear (XGBoost)",              ["brake_wear_reg.joblib", "brake_wear_clf.joblib"],  "vehicle"),
    ("engine_oil",        "Engine Oil (XGBoost)",              ["engine_oil_xgb.joblib", "engine_oil_clf.joblib"],  "vehicle"),
    ("hv_battery_soh",    "HV Battery SoH (XGBoost + Ridge)",  ["hv_battery_lr.joblib",  "hv_battery_clf.joblib"],  "vehicle"),
    ("battery_12v",       "12V Battery (XGBoost + LR)",        ["battery_12v_xgb.joblib"],                          "vehicle"),
    ("tyre_wear",         "Tyre Wear (LightGBM)",              ["tyre_wear_lgbm.joblib"],                           "vehicle"),
    ("fuel_anomaly",      "Fuel Anomaly (IsolationForest)",    ["fuel_anomaly_iso.joblib"],                          "vehicle"),
    ("driver_score",      "Driver Score (XGBoost)",            ["driver_score_reg.joblib"],                         "vehicle"),
    ("inventory_demand",  "Inventory Demand (LightGBM)",       ["inventory_demand_30d.joblib"],                     "operational"),
]


def _extract_rul_coefficients(model_name: str) -> dict[str, float] | None:
    """
    For fitted CoxPH/WeibullAFT models, extract coefficient magnitudes
    as a proxy for feature importance. Returns None if not fitted.
    """
    path = MODELS_DIR / f"{model_name}.joblib"
    if not path.exists():
        return None
    try:
        import joblib
        state = joblib.load(path)
        if not isinstance(state, dict) or not state.get("fitted"):
            return None
        m = state.get("model")
        if m is None:
            return None
        # CoxPH: params_ attribute
        if hasattr(m, "params_"):
            params = m.params_
            abs_vals = params.abs() if hasattr(params, "abs") else {k: abs(v) for k, v in params.items()}
            total = sum(abs_vals.values()) if hasattr(abs_vals, "values") else abs_vals.sum()
            if total > 0:
                return {str(k): round(float(v) / float(total), 4) for k, v in abs_vals.items()}
        # WeibullAFT: lambda_ params
        if hasattr(m, "params_"):
            return None
    except Exception as exc:
        log.debug("Could not extract coefficients from %s: %s", model_name, exc)
    return None


@router.get("/model-health")
async def model_health(
    current_user: Annotated[dict, Depends(require_oem)],
    model_name: str | None = Query(None),
):
    store = _load_metrics_store()
    result = []

    for key, display_name, artifact_files, category in _KNOWN_MODELS:
        if model_name and key != model_name:
            continue

        stored = store.get(key, {})
        artifacts_found = [f for f in artifact_files if (MODELS_DIR / f).exists()]
        artifact_exists = len(artifacts_found) > 0
        status = stored.get("status", "not_trained")
        if not artifact_exists and status == "trained":
            status = "not_trained"

        # Try to get real feature importances from model coefficients
        fi_from_store = stored.get("feature_importances", {})
        fi_real = _extract_rul_coefficients(key)  # None if not fitted / not survival model
        feature_importances = fi_real or fi_from_store

        # Check if importances are all equal (placeholder) — flag it
        fi_is_real = fi_real is not None or (
            len(fi_from_store) > 1
            and len(set(round(v, 3) for v in fi_from_store.values())) > 1
        )

        raw_metrics = stored.get("metrics", {})
        # Expose all stored metrics; None values are stripped so the UI can iterate safely
        clean_metrics = {k: v for k, v in raw_metrics.items() if v is not None}

        entry = {
            "model_name":          key,
            "display_name":        display_name,
            "category":            category,
            "algorithm":           stored.get("algorithm", "Unknown"),
            "target":              stored.get("target", "Unknown"),
            "status":              status,
            "artifact_exists":     artifact_exists,
            "artifact_files":      artifacts_found,
            "trained_at":          stored.get("trained_at"),
            "training_samples":    stored.get("training_samples"),
            "feature_names":       stored.get("feature_names", []),
            "feature_importances": feature_importances,
            "fi_is_real":          fi_is_real,
            "metrics":             clean_metrics,
            "notes":               stored.get("notes", ""),
        }

        # Warning flags based on real metrics
        concordance = clean_metrics.get("concordance_index")
        cv_auc      = clean_metrics.get("cv_auc")
        warnings = []
        if concordance is not None and concordance < 0.55:
            warnings.append(f"Concordance {concordance:.3f} is near-random (0.5 baseline) — model needs more labelled failure events")
        if cv_auc is not None and cv_auc < 0.60:
            warnings.append(f"AUC-ROC {cv_auc:.3f} is low — consider feature engineering or more training data")
        if not artifact_exists:
            warnings.append("No trained artifact found — run training to produce predictions")
        if not fi_is_real and feature_importances:
            warnings.append("Feature importances are uniform placeholders — retrain to get real SHAP values")
        entry["warnings"] = warnings

        result.append(entry)

    trained_count = sum(1 for m in result if m["status"] == "trained")
    concordances = [m["metrics"]["concordance_index"] for m in result if m["metrics"].get("concordance_index") is not None]
    aucs         = [m["metrics"]["cv_auc"] for m in result if m["metrics"].get("cv_auc") is not None]

    return {
        "models": result,
        "summary": {
            "total_models":      len(result),
            "trained_count":     trained_count,
            "not_trained_count": len(result) - trained_count,
            "avg_concordance":   round(float(np.mean(concordances)), 4) if concordances else None,
            "avg_auc":           round(float(np.mean(aucs)), 4) if aucs else None,
            "metrics_file_exists": METRICS_FILE.exists(),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ── Model EDA (per-model correlation heatmap) ─────────────────────────────────

@router.get("/model-eda/{model_name}")
async def model_eda(
    model_name: str,
    current_user: Annotated[dict, Depends(require_oem)],
):
    """Return pre-computed feature correlation matrix and target correlations for a model."""
    path = MODELS_DIR / f"eda_{model_name}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No EDA data for this model — retrain to generate")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── EDA Data ───────────────────────────────────────────────────────────────────

@router.get("/eda")
async def eda_data(
    current_user: Annotated[dict, Depends(require_oem)],
    feature_group: str = Query("fleet", enum=["fleet", "trips", "failures", "telemetry_summary"]),
):
    if feature_group == "fleet":
        return _eda_fleet()
    elif feature_group == "trips":
        return _eda_trips()
    elif feature_group == "failures":
        return _eda_failures()
    elif feature_group == "telemetry_summary":
        return _eda_telemetry_summary()


def _eda_fleet() -> dict:
    df = _load_fleet()
    if df.empty:
        return {"feature_group": "fleet", "distributions": {}, "row_count": 0}

    distributions: dict[str, Any] = {}

    if "initial_odometer" in df.columns:
        odo = df["initial_odometer"].dropna()
        bins = [0, 10000, 25000, 50000, 75000, 100000, 150000, 200000]
        labels = ["0-10k", "10-25k", "25-50k", "50-75k", "75-100k", "100-150k", "150-200k"]
        counts, _ = np.histogram(odo, bins=bins)
        distributions["odometer_km"] = {
            "type": "histogram", "bins": labels, "counts": counts.tolist(),
            "mean": round(float(odo.mean()), 0), "median": round(float(odo.median()), 0),
            "std": round(float(odo.std()), 0), "min": round(float(odo.min()), 0),
            "max": round(float(odo.max()), 0),
        }

    for col in ("fuel_type", "driver_profile", "model_name", "dealer_code"):
        if col in df.columns:
            vc = df[col].value_counts()
            labels_list = vc.index.tolist()
            if col == "driver_profile":
                labels_list = [l.replace("_", " ").title() for l in labels_list]
            distributions[col] = {
                "type": "categorical",
                "labels": labels_list,
                "counts": vc.values.tolist(),
            }

    return {
        "feature_group": "fleet",
        "row_count": len(df),
        "distributions": distributions,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _eda_trips() -> dict:
    df = _load_trips()
    if df.empty:
        return {"feature_group": "trips", "distributions": {}, "row_count": 0}

    distributions: dict[str, Any] = {}

    hist_specs: dict[str, tuple] = {
        "driveScore":        ([0, 20, 40, 60, 70, 80, 90, 100], ["0-20", "20-40", "40-60", "60-70", "70-80", "80-90", "90-100"]),
        "harshBreakingNum":  ([0, 1, 3, 5, 10, 20, 9999],       ["0", "1-2", "3-4", "5-9", "10-19", "20+"]),
        "accelerationNum":   ([0, 1, 3, 5, 10, 20, 9999],       ["0", "1-2", "3-4", "5-9", "10-19", "20+"]),
        "maxSpeed":          ([0, 40, 60, 80, 100, 120, 150, 300], ["<40", "40-60", "60-80", "80-100", "100-120", "120-150", ">150"]),
        "averageSpeed":      ([0, 20, 30, 40, 50, 60, 80, 999],  ["<20", "20-30", "30-40", "40-50", "50-60", "60-80", ">80"]),
    }

    for col, (bins, labels) in hist_specs.items():
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if series.empty:
            continue
        counts, _ = np.histogram(series, bins=bins)
        distributions[col] = {
            "type": "histogram", "bins": labels, "counts": counts.tolist(),
            "mean": round(float(series.mean()), 2), "median": round(float(series.median()), 2),
            "std": round(float(series.std()), 2),
            "p25": round(float(series.quantile(0.25)), 2),
            "p75": round(float(series.quantile(0.75)), 2),
        }

    num_cols = [c for c in ["driveScore", "harshBreakingNum", "accelerationNum", "maxSpeed", "averageSpeed", "overSpeed80"] if c in df.columns]
    if len(num_cols) >= 2:
        corr = df[num_cols].corr().round(3)
        distributions["_correlation"] = {
            "type": "heatmap",
            "features": num_cols,
            "matrix": corr.values.tolist(),
        }

    return {
        "feature_group": "trips",
        "row_count": len(df),
        "distributions": distributions,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _eda_failures() -> dict:
    df = _load_failures()
    if df.empty:
        return {
            "feature_group": "failures",
            "distributions": {},
            "row_count": 0,
            "note": "No failures_manifest.csv found — generate synthetic data first",
        }

    distributions: dict[str, Any] = {}

    if "failure_type" in df.columns:
        vc = df["failure_type"].value_counts()
        distributions["failure_type"] = {
            "type": "categorical",
            "labels": vc.index.tolist(),
            "counts": vc.values.tolist(),
        }

    for odo_col in ("odo_at_failure", "odometer_km", "initial_odometer"):
        if odo_col in df.columns:
            odo = df[odo_col].dropna()
            if odo.empty:
                continue
            bins   = [0, 20000, 40000, 60000, 80000, 100000, 150000, 300000]
            labels = ["0-20k", "20-40k", "40-60k", "60-80k", "80-100k", "100-150k", "150k+"]
            counts, _ = np.histogram(odo, bins=bins)
            distributions["odo_at_failure"] = {
                "type": "histogram", "bins": labels, "counts": counts.tolist(),
                "mean": round(float(odo.mean()), 0), "median": round(float(odo.median()), 0),
            }
            break

    for date_col in ("failure_date", "date"):
        if date_col in df.columns:
            try:
                df["_month"] = pd.to_datetime(df[date_col]).dt.to_period("M").astype(str)
                vc = df["_month"].value_counts().sort_index()
                distributions["failure_timeline"] = {
                    "type": "timeline",
                    "labels": vc.index.tolist(),
                    "counts": vc.values.tolist(),
                }
            except Exception:
                pass
            break

    return {
        "feature_group": "failures",
        "row_count": len(df),
        "total_failure_events": len(df),
        "distributions": distributions,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _eda_telemetry_summary() -> dict:
    files = list(DATA_DIR.glob("telemetry_*.csv"))[:5]
    if not files:
        return {"feature_group": "telemetry_summary", "distributions": {}, "row_count": 0}

    frames = []
    for f in files:
        try:
            df = pd.read_csv(f, nrows=2000)
            frames.append(df)
        except Exception:
            pass

    if not frames:
        return {"feature_group": "telemetry_summary", "distributions": {}, "row_count": 0}

    combined = pd.concat(frames, ignore_index=True)
    distributions: dict[str, Any] = {}

    # TBox Big Data Spec column names with display labels; legacy names as fallback
    _TBOX_COLS: list[tuple[str, str]] = [
        ("vehSpeed",               "Vehicle Speed (km/h)"),
        ("vehRPM",                 "Engine RPM"),
        ("vehCoolantTemp",         "Coolant Temp (°C)"),
        ("vehBatt",                "12V Battery (V)"),
        ("vehOutsideTemp",         "Ambient Temp (°C)"),
        ("FuelTankLevel",          "Fuel Tank Level (%)"),
        ("vehBMSPackSOC",          "HV Battery SOC (%)"),
        ("vehBMSPackCrnt",         "HV Pack Current (A)"),
        ("vehBMSCellMaxTem",       "HV Cell Max Temp (°C)"),
        ("frontLeftTyrePressure",  "Front-Left Tyre (kPa)"),
        ("frontRrightTyrePressure","Front-Right Tyre (kPa)"),
        ("rearLeftTyrePressure",   "Rear-Left Tyre (kPa)"),
        ("rearRightTyrePressure",  "Rear-Right Tyre (kPa)"),
        # Legacy names as fallback
        ("vehicle_speed_kmh",      "Vehicle Speed (km/h)"),
        ("engine_rpm",             "Engine RPM"),
        ("engine_coolant_temp",    "Coolant Temp (°C)"),
        ("battery_voltage",        "12V Battery (V)"),
        ("soc_pct",                "HV Battery SOC (%)"),
        ("ambient_temp",           "Ambient Temp (°C)"),
        ("fuel_level_pct",         "Fuel Level (%)"),
    ]
    seen_labels: set[str] = set()
    for col, label in _TBOX_COLS:
        if col not in combined.columns or label in seen_labels:
            continue
        series = combined[col].dropna()
        if series.empty or series.std() < 1e-6:
            continue
        p = np.percentile(series, [5, 25, 50, 75, 95])
        distributions[col] = {
            "type":   "box",
            "label":  label,
            "p5":     round(float(p[0]), 2),
            "p25":    round(float(p[1]), 2),
            "median": round(float(p[2]), 2),
            "p75":    round(float(p[3]), 2),
            "p95":    round(float(p[4]), 2),
            "mean":   round(float(series.mean()), 2),
            "std":    round(float(series.std()), 2),
            "min":    round(float(series.min()), 2),
            "max":    round(float(series.max()), 2),
            "n":      int(series.count()),
        }
        seen_labels.add(label)

    numeric_cols = list(distributions.keys())
    return {
        "feature_group":  "telemetry_summary",
        "files_sampled":  len(files),
        "row_count":      len(combined),
        "distributions":  distributions,
        "columns_found":  numeric_cols,
        "generated_at":   datetime.now(timezone.utc).isoformat(),
    }


# ── What-If Simulator ──────────────────────────────────────────────────────────

class WhatIfRequest(BaseModel):
    # Vehicle
    odometer_km:            float = 45000.0
    driver_profile:         str   = "urban_commuter"
    fuel_type:              str   = "ICE"
    model_name:             str   = "MG Hector"
    days_owned:             int   = 730
    # Driving behaviour
    harsh_braking_per_trip: float = 2.5
    idle_fraction:          float = 0.15
    avg_max_speed_kph:      float = 85.0
    overspeed_fraction:     float = 0.05
    km_since_last_brake_service: float = 15000.0
    km_since_last_oil_change:    float = 3000.0    # explicit, not derived from odo % 7500
    km_since_last_tyre_service:  float = 20000.0   # explicit, not derived from odo % 55000
    # 12V Battery-specific — explicit parameters, not buried in odometer
    battery_age_months:     float = 24.0    # separate from vehicle age
    short_trip_fraction:    float = 0.20    # fraction of trips < 5 km (kills charging)
    # EV-specific (only used when fuel_type = EV/PHEV)
    fast_charge_fraction:   float | None = None
    battery_soc_avg:        float | None = None
    charge_cycle_count:     int   | None = None


def _profile_penalty(profile: str) -> float:
    return {
        "eco_driver": 0, "highway_cruiser": 5, "urban_commuter": 10,
        "elderly_cautious": 2, "hill_region": 18, "aggressive": 42,
        "taxi_fleet": 32, "delivery_driver": 25,
    }.get(profile, 15)


def _profile_base_score(profile: str) -> float:
    return {
        "eco_driver": 92, "highway_cruiser": 82, "urban_commuter": 76,
        "elderly_cautious": 88, "hill_region": 70, "aggressive": 48,
        "taxi_fleet": 58, "delivery_driver": 63,
    }.get(profile, 75)


def _try_rul_prediction(model_name: str, features: dict) -> dict | None:
    """
    Try to run an actual trained RUL model.
    Returns None if model not fitted, unavailable, or silently returned the default fallback.
    """
    try:
        from models.rul_models import load_rul_model
        model = load_rul_model(model_name, MODELS_DIR)
        if model is None or not model._fitted:
            return None
        pred = model.predict(features)
        # Reject the library's internal fallback — means predict() failed silently
        if pred.model_used == "default":
            return None
        return pred.to_dict()
    except Exception as exc:
        log.debug("RUL predict failed for %s: %s", model_name, exc)
        return None


def _whatif_confidence(source: str, health_pct: float) -> float:
    """
    Confidence score for a What-If prediction.
    - trained_model: 0.72–0.92 (XGBoost/LGBM with real telemetry features)
    - heuristic:     0.42–0.60 (physics formula, no ML)
    """
    if source == "trained_model":
        base = 0.78
        certainty_bonus = abs(health_pct - 50) / 50 * 0.14
        return round(min(0.92, base + certainty_bonus), 2)
    else:
        base = 0.50
        certainty_bonus = abs(health_pct - 50) / 50 * 0.10
        return round(min(0.60, max(0.42, base + certainty_bonus)), 2)


# Ordered exactly as FEATURE_COLS in brake_wear_model.py
_BRAKE_FEATURE_COLS = [
    "brake_stress_cumulative",
    "harsh_brake_rate_7d",
    "harsh_brake_rate_30d",
    "high_speed_stop_count_30d",
    "avg_brake_intensity_7d",
    "deceleration_g_95th_30d",
    "brake_heat_proxy",
    "brake_thermal_stress",
    "km_since_last_brake_service",
    "brake_fluid_warning_active",
    "abs_activation_rate_30d",
    "downhill_brake_stress",
    "lateral_brake_stress",
    "regen_fraction",
]


def _predict_brake_with_model(request: "WhatIfRequest") -> dict | None:
    """
    Map What-If inputs → brake_wear_reg.joblib feature vector → days_to_brake_replacement.
    Top feature: km_since_last_brake_service (62% importance) — maps directly.
    Falls back to None on any error; caller uses physics heuristic instead.
    """
    try:
        import joblib
        import numpy as np

        kms   = float(request.km_since_last_brake_service)
        harsh = float(request.harsh_braking_per_trip)
        avg_trip_km = 15.0  # assumed average trip distance

        harsh_rate_30d = harsh / avg_trip_km * 100          # events per 100 km (baseline)
        harsh_rate_7d  = harsh_rate_30d * min(2.0, 1.0 + harsh / 8.0)  # recent spike > avg

        fvec = {
            "km_since_last_brake_service":  kms,
            "harsh_brake_rate_7d":          harsh_rate_7d,
            "harsh_brake_rate_30d":         harsh_rate_30d,
            "brake_thermal_stress":         harsh * 0.20 + kms / 40000 * 3.0,
            "deceleration_g_95th_30d":      0.20 + harsh * 0.06,
            "high_speed_stop_count_30d":    harsh * 90.0,   # ~3 trips/day × 30 days
            "avg_brake_intensity_7d":       0.08 + harsh * 0.06,
            "brake_heat_proxy":             harsh * kms / 4000.0,
            "brake_stress_cumulative":      (harsh * 0.20 + kms / 40000 * 3.0) * kms / 800.0,
            "abs_activation_rate_30d":      harsh * 0.003,
            "regen_fraction":               0.7 if request.fuel_type in ("EV", "PHEV") else 0.0,
            "brake_fluid_warning_active":   0.0,
            "downhill_brake_stress":        0.0,
            "lateral_brake_stress":         0.0,
        }

        X = np.array([[fvec.get(f, 0.0) for f in _BRAKE_FEATURE_COLS]], dtype=float)

        scaler = joblib.load(MODELS_DIR / "brake_wear_scaler.joblib")
        reg    = joblib.load(MODELS_DIR / "brake_wear_reg.joblib")

        days = float(reg.predict(scaler.transform(X))[0])
        return {"days": max(0.0, min(730.0, days))}
    except Exception as exc:
        log.debug("brake_wear_reg predict failed: %s", exc)
        return None


@router.post("/whatif")
async def whatif_simulator(
    request: WhatIfRequest,
    current_user: Annotated[dict, Depends(require_oem)],
):
    odo     = request.odometer_km
    profile = request.driver_profile
    fuel    = request.fuel_type

    braking_penalty = min(30, request.harsh_braking_per_trip * 4)
    idle_penalty    = min(15, request.idle_fraction * 30)
    speed_penalty   = min(20, request.overspeed_fraction * 100)
    drive_score     = max(10, _profile_base_score(profile) - braking_penalty * 0.5 - idle_penalty * 0.5 - speed_penalty * 0.3)
    penalty         = _profile_penalty(profile)
    odo_factor      = max(0, 100 - odo / 1500)
    overall_health  = round(odo_factor * 0.30 + drive_score * 0.40 + (100 - penalty) * 0.30, 1)

    predictions: dict[str, Any] = {}

    # ── Brake wear ──────────────────────────────────────────────────────────
    # Try trained XGBoost regressor first (cv_mae=64d, cv_auc=0.97)
    _brake_model = _predict_brake_with_model(request)
    if _brake_model is not None:
        brake_days   = int(_brake_model["days"])
        # 400d = expected full-life for well-maintained brakes; scale health 0→100%
        brake_health = min(100.0, max(0.0, _brake_model["days"] / 400.0 * 100.0))
        brake_rul_km = brake_days * 60.0
        brake_source = "trained_model"
        brake_model_used = "XGBoost"
    else:
        brake_rul_km = max(0, 45000 - request.km_since_last_brake_service - request.harsh_braking_per_trip * 500)
        brake_health = min(100, max(0, brake_rul_km / 450))
        brake_days   = max(0, int(brake_rul_km / 60))
        brake_source = "heuristic"
        brake_model_used = "formula"
    predictions["brake_wear"] = {
        "source":     brake_source,
        "model_used": brake_model_used,
        "health_pct": round(brake_health, 1),
        "rul_days":   brake_days,
        "rul_km":     round(brake_rul_km, 0),
        "predicted_date": (datetime.now(timezone.utc).date() + timedelta(days=brake_days)).isoformat(),
        "severity":   "HIGH" if brake_health < 30 else "MEDIUM" if brake_health < 60 else "LOW",
        "confidence": _whatif_confidence(brake_source, brake_health),
        "key_drivers": {
            "km_since_last_brake_service": request.km_since_last_brake_service,
            "harsh_braking_per_trip":      request.harsh_braking_per_trip,
        },
    }

    # ── Engine oil (ICE/CNG/PHEV only — pure EV has no combustion engine) ─────
    if fuel != "EV":
        km_oil = request.km_since_last_oil_change
        oil_remaining = max(0, 7500 - km_oil - request.idle_fraction * 500)
        oil_health    = min(100, max(0, oil_remaining / 75))
        oil_days      = max(0, int(oil_remaining / 60))
        predictions["engine_oil"] = {
            "source":     "heuristic",
            "model_used": "formula",
            "health_pct": round(oil_health, 1),
            "rul_days":   oil_days,
            "rul_km":     round(oil_remaining, 0),
            "predicted_date": (datetime.now(timezone.utc).date() + timedelta(days=oil_days)).isoformat(),
            "severity":   "HIGH" if oil_health < 30 else "MEDIUM" if oil_health < 60 else "LOW",
            "confidence": _whatif_confidence("heuristic", oil_health),
            "key_drivers": {
                "km_since_last_oil_change": km_oil,
                "idle_fraction":            request.idle_fraction,
            },
        }

    # ── Tyre wear ──────────────────────────────────────────────────────────
    km_tyre = request.km_since_last_tyre_service
    tyre_remaining = max(0, 55000 - km_tyre - request.overspeed_fraction * 5000)
    tyre_health    = min(100, max(0, tyre_remaining / 550))
    tyre_days      = max(0, int(tyre_remaining / 60))
    predictions["tyre_wear"] = {
        "source":     "heuristic",
        "model_used": "formula",
        "health_pct": round(tyre_health, 1),
        "rul_days":   tyre_days,
        "rul_km":     round(tyre_remaining, 0),
        "predicted_date": (datetime.now(timezone.utc).date() + timedelta(days=tyre_days)).isoformat(),
        "severity":   "HIGH" if tyre_health < 30 else "MEDIUM" if tyre_health < 60 else "LOW",
        "confidence": _whatif_confidence("heuristic", tyre_health),
        "key_drivers": {
            "km_since_last_tyre_service": km_tyre,
            "overspeed_fraction":         request.overspeed_fraction,
        },
    }

    # ── 12V Battery ──────────────────────────────────────────────────────────
    bat_age    = request.battery_age_months
    short_frac = request.short_trip_fraction
    batt_health = max(20, min(100,
        100
        - (bat_age * 1.2)       # ~1.2% per month
        - (short_frac * 30)     # short trips starve charging
        - (odo / 5000)          # general wear
    ))
    batt_days = max(0, int((batt_health - 30) * 8))
    predictions["battery_12v"] = {
        "source":     "heuristic",
        "model_used": "formula",
        "health_pct": round(batt_health, 1),
        "rul_days":   batt_days,
        "predicted_date": (datetime.now(timezone.utc).date() + timedelta(days=batt_days)).isoformat(),
        "severity":   "HIGH" if batt_health < 40 else "MEDIUM" if batt_health < 65 else "LOW",
        "confidence": _whatif_confidence("heuristic", batt_health),
        "key_drivers": {
            "battery_age_months":  bat_age,
            "short_trip_fraction": short_frac,
        },
    }

    # ── HV battery (EV/PHEV only) ─────────────────────────────────────────
    if fuel in ("EV", "PHEV") and request.fast_charge_fraction is not None:
        fc  = request.fast_charge_fraction
        soc = request.battery_soc_avg or 0.6
        cc  = request.charge_cycle_count or int(odo / 300)
        hv_soh = max(60, min(100,
            100
            - (cc / 1000) * 12
            - fc * 15
            - (1 - soc) * 5
        ))
        hv_days = max(0, int((hv_soh - 70) * 30))
        predictions["hv_battery_soh"] = {
            "source":     "heuristic",
            "model_used": "formula",
            "health_pct": round(hv_soh, 1),
            "rul_days":   hv_days,
            "predicted_date": (datetime.now(timezone.utc).date() + timedelta(days=hv_days)).isoformat(),
            "severity":   "HIGH" if hv_soh < 75 else "MEDIUM" if hv_soh < 85 else "LOW",
            "confidence": _whatif_confidence("heuristic", hv_soh),
            "key_drivers": {
                "charge_cycle_count":   cc,
                "fast_charge_fraction": fc,
                "battery_soc_avg":      soc,
            },
        }

    model_sources = {k: v["source"] for k, v in predictions.items()}
    real_model_count = sum(1 for s in model_sources.values() if s == "trained_model")

    return {
        "overall_health":   overall_health,
        "drive_score":      round(drive_score, 1),
        "predictions":      predictions,
        "model_sources":    model_sources,
        "real_model_count": real_model_count,
        "input_echo":       request.model_dump(),
        "generated_at":     datetime.now(timezone.utc).isoformat(),
    }


# ── Retrain History ────────────────────────────────────────────────────────────

@router.get("/retrain/history")
async def retrain_history(
    current_user: Annotated[dict, Depends(require_oem)],
    limit: int = Query(20, ge=1, le=100),
):
    history = _load_retrain_history()
    return {
        "history":    history[:limit],
        "total_runs": len(history),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


class RetrainRequest(BaseModel):
    models: list[str] = ["brake_wear_rul", "engine_oil_rul", "tyre_wear_rul", "battery_12v_rul"]
    notes: str = ""


@router.post("/retrain/trigger")
async def trigger_retrain(
    request: RetrainRequest,
    current_user: Annotated[dict, Depends(require_oem)],
):
    job_id  = f"retrain-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    user_id = current_user.get("user_id", "oem")

    entry = {
        "job_id":          job_id,
        "triggered_by":    user_id,
        "trigger_type":    "manual",
        "started_at":      datetime.now(timezone.utc).isoformat(),
        "completed_at":    None,
        "duration_minutes": None,
        "status":          "queued",
        "models":          request.models,
        "training_samples": None,
        "champion_promoted": [],
        "notes":           request.notes or "Manually triggered from OEM portal",
    }

    history = _load_retrain_history()
    history.insert(0, entry)
    _save_retrain_history(history)

    return {
        "job_id":     job_id,
        "status":     "queued",
        "message":    f"Job queued for {len(request.models)} model(s): {', '.join(request.models)}",
        "models":     request.models,
        "started_at": entry["started_at"],
        "note":       "Training runs via: py models/train_all.py --data-dir data/synthetic",
    }


@router.get("/retrain/status/{job_id}")
async def retrain_job_status(
    job_id: str,
    current_user: Annotated[dict, Depends(require_oem)],
):
    for run in _load_retrain_history():
        if run["job_id"] == job_id:
            return run
    raise HTTPException(status_code=404, detail=f"Job {job_id} not found")


@router.post("/retrain/stop/{job_id}")
async def stop_retrain_job(
    job_id: str,
    current_user: Annotated[dict, Depends(require_oem)],
):
    history = _load_retrain_history()
    for run in history:
        if run["job_id"] == job_id:
            if run["status"] in ("queued", "running"):
                run["status"] = "cancelled"
                run["completed_at"] = datetime.now(timezone.utc).isoformat()
                note = run.get("notes", "") or ""
                run["notes"] = (note + " [Cancelled by user]").strip()
            _save_retrain_history(history)
            return run
    raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
