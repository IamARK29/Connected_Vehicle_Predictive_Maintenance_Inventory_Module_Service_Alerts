"""
12V Battery Failure Prediction.

  lr:  Logistic Regression (interpretable)
  xgb: XGBoost classifier (accuracy)
  Ensemble probability = mean(lr_prob, xgb_prob)

Artefacts: models/saved/battery_12v_{lr,xgb,scaler}.joblib
"""
from __future__ import annotations

import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from models.model_registry import MODEL_DIR, _optuna_xgb, _shap_importance, _mlflow_log, _cv_metrics

log = logging.getLogger(__name__)

FEATURE_COLS = [
    "resting_voltage_7d_avg",
    "resting_voltage_trend_14d",
    "overnight_voltage_drop_avg",
    "cold_voltage_delta",
    "voltage_under_load_proxy",
    "parasitic_drain_rate_7d",
    "battery_age_days",
    "total_km_proxy",
    "lights_on_engine_off_count_7d",
    "avg_outside_temp_7d",
    "voltage_current",
]
TARGET_CLF = "no_start_within_7_days"

_lr:     Any = None
_xgb:    Any = None
_scaler: StandardScaler | None = None


def _load():
    global _lr, _xgb, _scaler
    if _lr is None:
        _lr     = joblib.load(MODEL_DIR / "battery_12v_lr.joblib")
        _xgb    = joblib.load(MODEL_DIR / "battery_12v_xgb.joblib")
        _scaler = joblib.load(MODEL_DIR / "battery_12v_scaler.joblib")
    return _lr, _xgb, _scaler


# ── train ─────────────────────────────────────────────────────────────────────

def train(features_df: pd.DataFrame, experiment: str = "autopredict-v1") -> dict:
    import xgboost as xgb

    global _lr, _xgb, _scaler

    df = (features_df.sort_values("computed_at")
          if "computed_at" in features_df.columns else features_df).copy()

    avail  = [c for c in FEATURE_COLS if c in df.columns]
    df_clf = df.dropna(subset=avail + [TARGET_CLF])

    if len(df_clf) < 10:
        return {"skipped": True, "reason": "insufficient data"}

    scaler = StandardScaler()
    X      = scaler.fit_transform(df_clf[avail].fillna(0))
    y      = df_clf[TARGET_CLF].values.astype(int)

    # ── Logistic Regression (interpretable) ───────────────────────────────
    lr = LogisticRegression(C=0.5, max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(X, y)

    # ── XGBoost ───────────────────────────────────────────────────────────
    best = _optuna_xgb(X, y, task="clf", n_trials=30)
    clf_params = {**best, "use_label_encoder": False, "eval_metric": "logloss"}
    xgb_clf = xgb.XGBClassifier(**clf_params)
    xgb_clf.fit(X, y)

    cv_lr  = _cv_metrics(
        LogisticRegression,
        X, y,
        {"C": 0.5, "max_iter": 1000, "class_weight": "balanced", "random_state": 42},
        task="clf",
    )
    cv_xgb = _cv_metrics(xgb.XGBClassifier, X, y, clf_params, task="clf")

    ens_prob = (lr.predict_proba(X)[:, 1] + xgb_clf.predict_proba(X)[:, 1]) / 2
    ens_auc  = float(roc_auc_score(y, ens_prob)) if y.sum() > 0 else np.nan

    shap_top = _shap_importance(xgb_clf, X, avail)
    metrics  = {
        "lr_cv_auc":          cv_lr.get("cv_auc", np.nan),
        "xgb_cv_auc":         cv_xgb.get("cv_auc", np.nan),
        "ensemble_train_auc": round(ens_auc, 4) if not np.isnan(ens_auc) else np.nan,
        "n_clf":              len(df_clf),
    }
    _mlflow_log(experiment, "battery_12v", metrics, best, shap_top)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(lr,      MODEL_DIR / "battery_12v_lr.joblib")
    joblib.dump(xgb_clf, MODEL_DIR / "battery_12v_xgb.joblib")
    joblib.dump(scaler,  MODEL_DIR / "battery_12v_scaler.joblib")
    _lr, _xgb, _scaler = lr, xgb_clf, scaler

    return metrics


# ── evaluate ──────────────────────────────────────────────────────────────────

def evaluate(features_df: pd.DataFrame) -> dict:
    lr, xgb_clf, scaler = _load()
    avail  = [c for c in FEATURE_COLS if c in features_df.columns]
    df_clf = features_df.dropna(subset=avail + [TARGET_CLF])
    if len(df_clf) == 0 or df_clf[TARGET_CLF].sum() == 0:
        return {}

    X = scaler.transform(df_clf[avail].fillna(0))
    ens_prob = (lr.predict_proba(X)[:, 1] + xgb_clf.predict_proba(X)[:, 1]) / 2
    return {
        "ensemble_auc": round(float(roc_auc_score(df_clf[TARGET_CLF], ens_prob)), 4),
        "lr_auc":       round(float(roc_auc_score(df_clf[TARGET_CLF], lr.predict_proba(X)[:, 1])), 4),
        "xgb_auc":      round(float(roc_auc_score(df_clf[TARGET_CLF], xgb_clf.predict_proba(X)[:, 1])), 4),
    }


# ── predict_single ────────────────────────────────────────────────────────────

def predict_single(vin: str) -> dict:
    from features.battery_12v_features import Battery12VFeaturePipeline
    feats = Battery12VFeaturePipeline().compute_from_influx(vin, lookback_days=30)
    if feats is None or feats.empty:
        return {"error": f"No 12V battery data for VIN {vin}"}
    return predict_batch(feats).iloc[0].to_dict()


# ── predict_batch ─────────────────────────────────────────────────────────────

def predict_batch(features_df: pd.DataFrame) -> pd.DataFrame:
    lr, xgb_clf, scaler = _load()
    avail = [c for c in FEATURE_COLS if c in features_df.columns]
    X     = scaler.transform(features_df[avail].fillna(0))

    prob_lr  = lr.predict_proba(X)[:, 1]
    prob_xgb = xgb_clf.predict_proba(X)[:, 1]
    prob_ens = (prob_lr + prob_xgb) / 2.0

    days_est = np.clip(7 * (1 - prob_ens) / np.maximum(prob_ens, 1e-6), 1, 365)

    return pd.DataFrame({
        "vin":                 features_df["vin"].values if "vin" in features_df.columns else [""] * len(X),
        "no_start_probability": np.round(prob_ens, 4),
        "lr_probability":       np.round(prob_lr, 4),
        "xgb_probability":      np.round(prob_xgb, 4),
        "days_to_12v_failure":  np.round(days_est, 1),
        "urgency":              np.where(prob_ens > 0.6, "critical",
                                np.where(prob_ens > 0.3, "warning", "ok")),
    })
