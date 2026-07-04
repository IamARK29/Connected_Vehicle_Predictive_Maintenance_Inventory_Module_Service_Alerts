"""
Driver Behaviour Scoring Models.

  reg:  XGBoost regressor → composite_drive_score (0–100)
  clf:  XGBoost classifier → high_risk_driver (score < 50)

Artefacts: models/saved/driver_score_{reg,clf}.joblib
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
    "harsh_accel_rate_30d",
    "harsh_brake_rate_30d",
    "overspeed_80_fraction_30d",
    "overspeed_120_count_30d",
    "idle_fraction_30d",
    "overrev_rate_30d",
    "lugging_rate_30d",
    "avg_speed_30d",
    "sudden_turn_rate_30d",
    "night_driving_fraction_30d",
    "fuel_efficiency_score",
    "peer_percentile",
]
TARGET_REG = "composite_drive_score"
TARGET_CLF = "high_risk_driver"

_reg:    Any = None
_clf:    Any = None
_scaler: StandardScaler | None = None


def _load():
    global _reg, _clf, _scaler
    if _reg is None:
        _reg    = joblib.load(MODEL_DIR / "driver_score_reg.joblib")
        _clf    = joblib.load(MODEL_DIR / "driver_risk_clf.joblib")
        _scaler = joblib.load(MODEL_DIR / "driver_score_scaler.joblib")
    return _reg, _clf, _scaler


# ── train ─────────────────────────────────────────────────────────────────────

def train(features_df: pd.DataFrame, experiment: str = "autopredict-v1") -> dict:
    import xgboost as xgb

    global _reg, _clf, _scaler

    df = (features_df.sort_values("computed_at")
          if "computed_at" in features_df.columns else features_df).copy()

    avail  = [c for c in FEATURE_COLS if c in df.columns]

    # Derive TARGET_CLF if not present
    if TARGET_CLF not in df.columns and TARGET_REG in df.columns:
        df[TARGET_CLF] = (df[TARGET_REG] < 50).astype(int)

    df_reg = df.dropna(subset=avail + [TARGET_REG])
    df_clf = df.dropna(subset=avail + [TARGET_CLF])

    if len(df_reg) < 10:
        return {"skipped": True, "reason": "insufficient data"}

    scaler = StandardScaler()
    X_reg  = scaler.fit_transform(df_reg[avail].fillna(0))
    y_reg  = df_reg[TARGET_REG].values.astype(float)
    X_clf  = scaler.transform(df_clf[avail].fillna(0))
    y_clf  = df_clf[TARGET_CLF].values.astype(int)

    print("    [driver_score] Optuna HPO (reg) ...", end=" ", flush=True)
    best_reg = _optuna_xgb(X_reg, y_reg, task="reg", n_trials=30)
    print("done")
    print("    [driver_score] Optuna HPO (clf) ...", end=" ", flush=True)
    best_clf = _optuna_xgb(X_clf, y_clf, task="clf", n_trials=30)
    print("done")

    clf_params = {**best_clf, "use_label_encoder": False, "eval_metric": "logloss"}

    reg = xgb.XGBRegressor(**best_reg)
    reg.fit(X_reg, y_reg)

    clf = xgb.XGBClassifier(**clf_params)
    clf.fit(X_clf, y_clf)

    cv_reg = _cv_metrics(xgb.XGBRegressor, X_reg, y_reg, best_reg, task="reg")
    cv_clf = _cv_metrics(xgb.XGBClassifier, X_clf, y_clf, clf_params, task="clf")

    shap_top = _shap_importance(reg, X_reg, avail)
    metrics  = {**cv_reg, **cv_clf, "n_reg": len(df_reg), "n_clf": len(df_clf)}
    _mlflow_log(experiment, "driver_score", metrics, best_reg, shap_top)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(reg,    MODEL_DIR / "driver_score_reg.joblib")
    joblib.dump(clf,    MODEL_DIR / "driver_risk_clf.joblib")
    joblib.dump(scaler, MODEL_DIR / "driver_score_scaler.joblib")
    _reg, _clf, _scaler = reg, clf, scaler

    return metrics


# ── evaluate ──────────────────────────────────────────────────────────────────

def evaluate(features_df: pd.DataFrame) -> dict:
    reg, clf, scaler = _load()
    avail   = [c for c in FEATURE_COLS if c in features_df.columns]
    metrics: dict[str, Any] = {}

    df_reg = features_df.dropna(subset=avail + [TARGET_REG])
    if len(df_reg) > 0:
        X = scaler.transform(df_reg[avail].fillna(0))
        p = reg.predict(X)
        metrics["mae"]  = round(float(mean_absolute_error(df_reg[TARGET_REG], p)), 3)
        metrics["rmse"] = round(float(np.sqrt(mean_squared_error(df_reg[TARGET_REG], p))), 3)

    df_clf = features_df.dropna(subset=avail + [TARGET_CLF])
    if len(df_clf) > 0 and df_clf[TARGET_CLF].sum() > 0:
        X = scaler.transform(df_clf[avail].fillna(0))
        metrics["auc"] = round(float(roc_auc_score(df_clf[TARGET_CLF], clf.predict_proba(X)[:, 1])), 4)

    return metrics


# ── predict_single ────────────────────────────────────────────────────────────

def predict_single(vin: str) -> dict:
    from features.driver_behaviour_features import DriverBehaviourFeaturePipeline
    feats = DriverBehaviourFeaturePipeline().compute_from_influx(vin, lookback_days=30)
    if feats is None or feats.empty:
        return {"error": f"No driver behaviour data for VIN {vin}"}
    return predict_batch(feats).iloc[0].to_dict()


# ── predict_batch ─────────────────────────────────────────────────────────────

def predict_batch(features_df: pd.DataFrame) -> pd.DataFrame:
    reg, clf, scaler = _load()
    avail = [c for c in FEATURE_COLS if c in features_df.columns]
    X     = scaler.transform(features_df[avail].fillna(0))

    score_pred  = np.clip(reg.predict(X), 0, 100)
    risk_prob   = clf.predict_proba(X)[:, 1]

    return pd.DataFrame({
        "vin":               features_df["vin"].values if "vin" in features_df.columns else [""] * len(X),
        "composite_drive_score": np.round(score_pred, 1),
        "high_risk_probability": np.round(risk_prob, 4),
        "driver_score":      np.round(score_pred, 1),
        "risk_category":     np.where(score_pred < 40, "high",
                             np.where(score_pred < 65, "medium", "low")),
        "urgency":           np.where(risk_prob > 0.7, "critical",
                             np.where(risk_prob > 0.4, "warning", "ok")),
    })
