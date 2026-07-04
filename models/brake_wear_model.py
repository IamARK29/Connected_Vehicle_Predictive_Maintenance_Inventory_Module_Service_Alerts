"""
Brake Wear Prediction Models.

  reg:     XGBoost regressor     → days_to_brake_replacement
  clf:     XGBoost classifier    → brake_replacement_within_30_days
  cox:     CoxPH survival model  → hazard function (optional, lifelines)

Training: Optuna 50-trial HPO, TimeSeriesSplit(5), SHAP top-5, MLflow.
Artefacts: models/saved/brake_wear_{reg,clf,scaler}.joblib
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    roc_auc_score, precision_recall_curve,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from models.model_registry import MODEL_DIR, _optuna_xgb, _shap_importance, _mlflow_log, _cv_metrics

log = logging.getLogger(__name__)

FEATURE_COLS = [
    "brake_stress_cumulative",
    "harsh_brake_rate_7d",
    "harsh_brake_rate_30d",
    "high_speed_stop_count_30d",
    "avg_brake_intensity_7d",
    "deceleration_g_95th_30d",
    "brake_heat_proxy",
    "brake_thermal_stress",
    "km_since_last_brake_service",
    # Real binary signals from TBox spec (replaces fake pad mm / fluid pct)
    "brake_fluid_warning_active",   # from vehBrkFludLvlLow
    "abs_activation_rate_30d",      # from vehABSF
    "downhill_brake_stress",        # from tboxAccelZ × brake_pos
    "lateral_brake_stress",         # from tboxAccelY × brake_pos
    "regen_fraction",               # EV only, 0 for ICE
]
TARGET_REG = "days_to_brake_replacement"
TARGET_CLF = "brake_replacement_within_30_days"

# Lazy-loaded artefacts
_reg    : Any = None
_clf    : Any = None
_scaler : StandardScaler | None = None


def _load():
    global _reg, _clf, _scaler
    if _reg is None:
        _reg    = joblib.load(MODEL_DIR / "brake_wear_reg.joblib")
        _clf    = joblib.load(MODEL_DIR / "brake_wear_clf.joblib")
        _scaler = joblib.load(MODEL_DIR / "brake_wear_scaler.joblib")
    return _reg, _clf, _scaler


# ── train ─────────────────────────────────────────────────────────────────────

def train(features_df: pd.DataFrame, experiment: str = "autopredict-v1") -> dict:
    """
    Train on pre-computed feature rows from BrakeFeaturePipeline.compute_batch().
    Runs Optuna HPO (50 trials) with TimeSeriesSplit(5) for both reg and clf.
    """
    import xgboost as xgb

    global _reg, _clf, _scaler

    df = (features_df.sort_values("computed_at")
          if "computed_at" in features_df.columns else features_df).copy()

    df_reg = df.dropna(subset=[c for c in FEATURE_COLS if c in df.columns] + [TARGET_REG])
    df_clf = df.dropna(subset=[c for c in FEATURE_COLS if c in df.columns] + [TARGET_CLF])

    avail_cols = [c for c in FEATURE_COLS if c in df.columns]
    if len(df_reg) < 10:
        log.warning("brake_wear train: only %d regression rows — skipping", len(df_reg))
        return {"skipped": True, "reason": "insufficient data"}

    scaler = StandardScaler()
    X_reg  = scaler.fit_transform(df_reg[avail_cols].fillna(0))
    y_reg  = df_reg[TARGET_REG].values.astype(float)
    X_clf  = scaler.transform(df_clf[avail_cols].fillna(0))
    y_clf  = df_clf[TARGET_CLF].values.astype(int)

    # ── Optuna HPO ────────────────────────────────────────────────────────
    print("    [brake_wear] Optuna HPO (50 trials, reg) ...", end=" ", flush=True)
    best_reg = _optuna_xgb(X_reg, y_reg, task="reg", n_trials=50)
    print("done")
    print("    [brake_wear] Optuna HPO (30 trials, clf) ...", end=" ", flush=True)
    best_clf = _optuna_xgb(X_clf, y_clf, task="clf", n_trials=30)
    print("done")

    # ── CV metrics ────────────────────────────────────────────────────────
    cv_reg = _cv_metrics(xgb.XGBRegressor, X_reg, y_reg, best_reg, task="reg")
    cv_clf = _cv_metrics(xgb.XGBClassifier, X_clf, y_clf,
                         {**best_clf, "use_label_encoder": False, "eval_metric": "logloss"},
                         task="clf")

    # ── Final models on full data ─────────────────────────────────────────
    reg = xgb.XGBRegressor(**best_reg)
    reg.fit(X_reg, y_reg)

    clf_params = {**best_clf, "use_label_encoder": False, "eval_metric": "logloss"}
    clf = xgb.XGBClassifier(**clf_params)
    clf.fit(X_clf, y_clf)

    # Recall @ 80% precision (on clf)
    clf_proba = clf.predict_proba(X_clf)[:, 1]
    r_at_p80  = 0.0
    if y_clf.sum() > 0:
        prec, rec, _ = precision_recall_curve(y_clf, clf_proba)
        if (prec >= 0.8).any():
            r_at_p80 = float(rec[prec >= 0.8][0])

    # ── SHAP ─────────────────────────────────────────────────────────────
    shap_top = _shap_importance(reg, X_reg, avail_cols)

    # ── Optional CoxPH ────────────────────────────────────────────────────
    cox_ci = np.nan
    try:
        from lifelines import CoxPHFitter
        cox_df = df_reg[[*avail_cols, TARGET_REG]].copy()
        cox_df["event"] = 1
        cox_df = cox_df.rename(columns={TARGET_REG: "duration"}).dropna()
        if len(cox_df) >= 20:
            cox = CoxPHFitter(penalizer=0.1)
            cox.fit(cox_df, duration_col="duration", event_col="event")
            cox_ci = round(float(cox.concordance_index_), 4)
            joblib.dump(cox, MODEL_DIR / "brake_wear_cox.joblib")
    except Exception as exc:
        log.debug("CoxPH skipped: %s", exc)

    metrics = {
        **cv_reg, **cv_clf,
        "cox_concordance_index": cox_ci,
        "recall_at_p80":         round(r_at_p80, 4),
        "n_reg":                 len(df_reg),
        "n_clf":                 len(df_clf),
    }

    _mlflow_log(experiment, "brake_wear", metrics, best_reg, shap_top,
                {"reg": MODEL_DIR / "brake_wear_reg.joblib"})

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(reg,    MODEL_DIR / "brake_wear_reg.joblib")
    joblib.dump(clf,    MODEL_DIR / "brake_wear_clf.joblib")
    joblib.dump(scaler, MODEL_DIR / "brake_wear_scaler.joblib")
    _reg, _clf, _scaler = reg, clf, scaler

    return metrics


# ── evaluate ──────────────────────────────────────────────────────────────────

def evaluate(features_df: pd.DataFrame) -> dict:
    reg, clf, scaler = _load()
    avail = [c for c in FEATURE_COLS if c in features_df.columns]
    metrics: dict[str, Any] = {}

    df_reg = features_df.dropna(subset=[TARGET_REG] + avail)
    if len(df_reg) > 0:
        X = scaler.transform(df_reg[avail].fillna(0))
        p = reg.predict(X)
        metrics["mae"]  = round(float(mean_absolute_error(df_reg[TARGET_REG], p)), 3)
        metrics["rmse"] = round(float(np.sqrt(mean_squared_error(df_reg[TARGET_REG], p))), 3)

    df_clf = features_df.dropna(subset=[TARGET_CLF] + avail)
    if len(df_clf) > 0 and df_clf[TARGET_CLF].sum() > 0:
        X = scaler.transform(df_clf[avail].fillna(0))
        metrics["auc"] = round(float(roc_auc_score(df_clf[TARGET_CLF], clf.predict_proba(X)[:, 1])), 4)

    return metrics


# ── predict_single ────────────────────────────────────────────────────────────

def predict_single(vin: str) -> dict:
    """Fetch features from InfluxDB for *vin* and return brake wear prediction."""
    from features.brake_features import BrakeFeaturePipeline
    feats = BrakeFeaturePipeline().compute_from_influx(vin, lookback_days=30)
    if feats is None or feats.empty:
        return {"severity": "unknown", "error": f"No telemetry data for VIN {vin}"}
    return predict_batch(feats).iloc[0].to_dict()


# ── predict_batch ─────────────────────────────────────────────────────────────

def predict_batch(features_df: pd.DataFrame) -> pd.DataFrame:
    """Run inference on a pre-computed feature DataFrame."""
    reg, clf, scaler = _load()
    avail = [c for c in FEATURE_COLS if c in features_df.columns]
    X = scaler.transform(features_df[avail].fillna(0))

    days_pred  = reg.predict(X)
    prob_clf   = clf.predict_proba(X)[:, 1]

    out = pd.DataFrame({
        "vin":                          features_df["vin"].values if "vin" in features_df.columns else [""] * len(X),
        "days_to_replacement_predicted": np.round(days_pred, 1),
        "replacement_prob_30d":          np.round(prob_clf, 4),
        "replacement_predicted_by":     (pd.Timestamp.utcnow() + pd.to_timedelta(np.round(days_pred), unit="D")).strftime("%Y-%m-%d"),
        "urgency":                       np.where(prob_clf > 0.7, "critical",
                                         np.where(prob_clf > 0.4, "warning", "ok")),
    })
    return out
