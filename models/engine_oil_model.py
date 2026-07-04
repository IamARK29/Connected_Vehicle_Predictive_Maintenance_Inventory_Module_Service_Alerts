


"""
Engine Oil Degradation Models.

  odi:   Physics ODI formula (deterministic — no ML needed)
  xgb:   XGBoost correction layer — learns residual between ODI and service date
  clf:   XGBoost classifier → oil_change_due_within_14_days

Artefacts: models/saved/engine_oil_{xgb,clf,scaler}.joblib
"""
from __future__ import annotations

import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.preprocessing import StandardScaler

from models.model_registry import MODEL_DIR, _optuna_xgb, _shap_importance, _mlflow_log, _cv_metrics

log = logging.getLogger(__name__)

FEATURE_COLS = [
    "km_since_oil_change",
    "days_since_oil_change",
    "cold_start_count_30d",
    "high_rpm_duration_minutes_30d",
    "coolant_overtemp_count_30d",
    "avg_coolant_temp_7d",
    "fuel_consumption_deviation_pct",
    "idle_hours_30d",
    "short_trip_fraction_30d",
    "high_rpm_stress_index",
    # Real binary signals from TBox spec (replaces fake oil_life_pct)
    "oil_pressure_warning_active",  # from vehOilPressureWarning
    "mil_warning_active",           # from vehMILWarning
    "gear_efficiency_score",
]
TARGET_REG = "oil_degradation_index"
TARGET_CLF = "engine_oil_within_30_days"

_xgb: Any = None
_clf: Any = None
_scaler: StandardScaler | None = None


def _load():
    global _xgb, _clf, _scaler
    if _xgb is None:
        _xgb    = joblib.load(MODEL_DIR / "engine_oil_xgb.joblib")
        _clf    = joblib.load(MODEL_DIR / "engine_oil_clf.joblib")
        _scaler = joblib.load(MODEL_DIR / "engine_oil_scaler.joblib")
    return _xgb, _clf, _scaler


# ── Physics ODI (deterministic) ────────────────────────────────────────────

def oil_degradation_index(
    km_since_change: float,
    cold_starts_30d: float,
    coolant_overtemp_30d: float,
    high_rpm_min_30d: float,
    fuel_deviation_pct: float,
) -> float:
    """
    Physics-based Oil Degradation Index (0 = fresh, 1 = change required).

    Weights: km 35%, cold-starts 20%, thermal 20%, high-RPM 15%, fuel-enrich 10%.
    """
    km_norm    = min(1.0, km_since_change / 7500)
    cold_norm  = min(1.0, cold_starts_30d / 20)
    therm_norm = min(1.0, coolant_overtemp_30d / 10)
    rpm_norm   = min(1.0, high_rpm_min_30d / 120)
    fuel_norm  = min(1.0, max(0.0, fuel_deviation_pct) / 30)
    return float(
        0.35 * km_norm +
        0.20 * cold_norm +
        0.20 * therm_norm +
        0.15 * rpm_norm +
        0.10 * fuel_norm
    )


# ── train ─────────────────────────────────────────────────────────────────────

def train(features_df: pd.DataFrame, experiment: str = "autopredict-v1") -> dict:
    import xgboost as xgb

    global _xgb, _clf, _scaler
    df = (features_df.sort_values("computed_at")
          if "computed_at" in features_df.columns else features_df).copy()

    avail = [c for c in FEATURE_COLS if c in df.columns]
    df_reg = df.dropna(subset=avail + [TARGET_REG])
    df_clf = df.dropna(subset=avail + [TARGET_CLF])

    if len(df_reg) < 10:
        return {"skipped": True, "reason": "insufficient data"}

    scaler = StandardScaler()
    X_reg  = scaler.fit_transform(df_reg[avail].fillna(0))
    y_reg  = df_reg[TARGET_REG].values.astype(float)
    X_clf  = scaler.transform(df_clf[avail].fillna(0))
    y_clf  = df_clf[TARGET_CLF].values.astype(int)

    # XGBoost correction layer (residual between physics ODI and actual)
    best_reg = _optuna_xgb(X_reg, y_reg, task="reg", n_trials=30)
    best_clf = _optuna_xgb(X_clf, y_clf, task="clf", n_trials=30)

    cv_reg = _cv_metrics(xgb.XGBRegressor, X_reg, y_reg, best_reg, task="reg")
    cv_clf = _cv_metrics(xgb.XGBClassifier, X_clf, y_clf,
                         {**best_clf, "use_label_encoder": False, "eval_metric": "logloss"},
                         task="clf")

    xgb_model = xgb.XGBRegressor(**best_reg)
    xgb_model.fit(X_reg, y_reg)

    clf_model = xgb.XGBClassifier(**{**best_clf, "use_label_encoder": False, "eval_metric": "logloss"})
    clf_model.fit(X_clf, y_clf)

    shap_top = _shap_importance(xgb_model, X_reg, avail)
    metrics  = {**cv_reg, **cv_clf, "n_reg": len(df_reg), "n_clf": len(df_clf)}
    _mlflow_log(experiment, "engine_oil", metrics, best_reg, shap_top)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(xgb_model, MODEL_DIR / "engine_oil_xgb.joblib")
    joblib.dump(clf_model, MODEL_DIR / "engine_oil_clf.joblib")
    joblib.dump(scaler,    MODEL_DIR / "engine_oil_scaler.joblib")
    _xgb, _clf, _scaler = xgb_model, clf_model, scaler

    return metrics


# ── evaluate ──────────────────────────────────────────────────────────────────

def evaluate(features_df: pd.DataFrame) -> dict:
    xgb_m, clf_m, scaler = _load()
    avail = [c for c in FEATURE_COLS if c in features_df.columns]
    metrics: dict[str, Any] = {}

    df_reg = features_df.dropna(subset=avail + [TARGET_REG])
    if len(df_reg) > 0:
        X = scaler.transform(df_reg[avail].fillna(0))
        p = xgb_m.predict(X)
        metrics["mae"]  = round(float(mean_absolute_error(df_reg[TARGET_REG], p)), 4)
        metrics["rmse"] = round(float(np.sqrt(mean_squared_error(df_reg[TARGET_REG], p))), 4)

    df_clf = features_df.dropna(subset=avail + [TARGET_CLF])
    if len(df_clf) > 0 and df_clf[TARGET_CLF].sum() > 0:
        X = scaler.transform(df_clf[avail].fillna(0))
        metrics["auc"] = round(float(roc_auc_score(df_clf[TARGET_CLF], clf_m.predict_proba(X)[:, 1])), 4)

    return metrics


# ── predict_single ────────────────────────────────────────────────────────────

def predict_single(vin: str) -> dict:
    from features.engine_features import EngineFeaturePipeline
    feats = EngineFeaturePipeline().compute_from_influx(vin, lookback_days=90)
    if feats is None or feats.empty:
        return {"severity": "unknown", "error": f"No telemetry for VIN {vin}"}
    return predict_batch(feats).iloc[0].to_dict()


# ── predict_batch ─────────────────────────────────────────────────────────────

def predict_batch(features_df: pd.DataFrame) -> pd.DataFrame:
    xgb_m, clf_m, scaler = _load()
    avail = [c for c in FEATURE_COLS if c in features_df.columns]
    X     = scaler.transform(features_df[avail].fillna(0))

    odi_pred  = np.clip(xgb_m.predict(X), 0, 1)
    prob_14   = clf_m.predict_proba(X)[:, 1]
    # Estimate km until change: at linear rate, remaining = (1-odi) × 7500
    km_remaining = np.clip((1 - odi_pred) * 7500, 0, 7500)
    days_remaining = km_remaining / 65  # assume 65 km/day average

    return pd.DataFrame({
        "vin":                     features_df["vin"].values if "vin" in features_df.columns else [""] * len(X),
        "oil_degradation_index":   np.round(odi_pred, 3),
        "oil_change_prob_14d":     np.round(prob_14, 4),
        "km_to_oil_change":        np.round(km_remaining, 0),
        "days_until_oil_change":   np.round(days_remaining, 1),
        "urgency":                 np.where(odi_pred > 0.85, "critical",
                                   np.where(odi_pred > 0.65, "warning", "ok")),
    })
