"""
Fuel Anomaly Detection (unsupervised).

  iso: Isolation Forest on fuel/engine efficiency features
       contamination=0.05 — no labels required

Artefacts: models/saved/fuel_anomaly_{iso,scaler}.joblib
"""
from __future__ import annotations

import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from models.model_registry import MODEL_DIR, _mlflow_log

log = logging.getLogger(__name__)

FEATURE_COLS = [
    "fuel_consumption_deviation_pct",   # deviation from 90-day baseline
    "rpm_to_speed_ratio_anomaly",       # abnormal RPM/speed ratio (clutch slip, drag)
    "high_rpm_stress_index",            # sustained high-RPM events
    "coolant_overtemp_count_30d",       # thermal stress events
    "oil_pressure_warning_active",      # binary TBox warning signal
    "mil_warning_active",               # binary MIL warning signal
    "idle_hours_30d",                   # excessive idling
    "cold_start_count_30d",             # cold-start stress frequency
]

_iso:    Any = None
_scaler: StandardScaler | None = None


def _load():
    global _iso, _scaler
    if _iso is None:
        _iso    = joblib.load(MODEL_DIR / "fuel_anomaly_iso.joblib")
        _scaler = joblib.load(MODEL_DIR / "fuel_anomaly_scaler.joblib")
    return _iso, _scaler


# ── train ─────────────────────────────────────────────────────────────────────

def train(features_df: pd.DataFrame, experiment: str = "autopredict-v1") -> dict:
    global _iso, _scaler

    df    = features_df.copy()
    avail = [c for c in FEATURE_COLS if c in df.columns]

    if len(avail) < 2:
        return {"skipped": True, "reason": "insufficient feature columns"}

    df_clean = df[avail].fillna(0)
    if len(df_clean) < 10:
        return {"skipped": True, "reason": "insufficient data"}

    scaler   = StandardScaler()
    X        = scaler.fit_transform(df_clean)

    iso = IsolationForest(
        n_estimators=300,
        contamination=0.05,
        max_samples="auto",
        random_state=42,
    )
    iso.fit(X)

    scores       = iso.score_samples(X)
    anomaly_mask = iso.predict(X) == -1
    anomaly_rate = float(anomaly_mask.mean())

    metrics = {
        "anomaly_rate":      round(anomaly_rate, 4),
        "avg_anomaly_score": round(float(scores.mean()), 4),
        "n_samples":         len(df_clean),
        "n_features":        len(avail),
    }
    _mlflow_log(experiment, "fuel_anomaly", metrics, {})

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(iso,    MODEL_DIR / "fuel_anomaly_iso.joblib")
    joblib.dump(scaler, MODEL_DIR / "fuel_anomaly_scaler.joblib")
    _iso, _scaler = iso, scaler

    return metrics


# ── evaluate ──────────────────────────────────────────────────────────────────

def evaluate(features_df: pd.DataFrame) -> dict:
    iso, scaler = _load()
    avail   = [c for c in FEATURE_COLS if c in features_df.columns]
    X       = scaler.transform(features_df[avail].fillna(0))
    scores  = iso.score_samples(X)
    anomaly_rate = float((iso.predict(X) == -1).mean())
    return {
        "anomaly_rate":      round(anomaly_rate, 4),
        "avg_anomaly_score": round(float(scores.mean()), 4),
        "min_anomaly_score": round(float(scores.min()), 4),
    }


# ── predict_single ────────────────────────────────────────────────────────────

def predict_single(vin: str) -> dict:
    from features.engine_features import EngineFeaturePipeline
    feats = EngineFeaturePipeline().compute_from_influx(vin, lookback_days=30)
    if feats is None or feats.empty:
        return {"severity": "unknown", "error": f"No engine data for VIN {vin}"}
    return predict_batch(feats).iloc[0].to_dict()


# ── predict_batch ─────────────────────────────────────────────────────────────

def predict_batch(features_df: pd.DataFrame) -> pd.DataFrame:
    iso, scaler = _load()
    avail = [c for c in FEATURE_COLS if c in features_df.columns]
    X     = scaler.transform(features_df[avail].fillna(0))

    scores    = iso.score_samples(X)
    is_anomaly = (iso.predict(X) == -1).astype(int)

    # Normalise score to [0,1] probability proxy (lower score = more anomalous)
    score_min, score_max = scores.min(), scores.max()
    score_range = max(score_max - score_min, 1e-9)
    anomaly_prob = np.clip((score_min - scores) / score_range, 0, 1)

    return pd.DataFrame({
        "vin":                features_df["vin"].values if "vin" in features_df.columns else [""] * len(X),
        "fuel_anomaly":       is_anomaly,
        "anomaly_score":      np.round(scores, 4),
        "anomaly_probability": np.round(anomaly_prob, 4),
        "urgency":            np.where(is_anomaly == 1,
                                np.where(anomaly_prob > 0.7, "critical", "warning"),
                                "ok"),
    })
