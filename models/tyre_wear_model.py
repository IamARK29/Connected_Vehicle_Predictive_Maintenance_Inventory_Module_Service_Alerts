"""
Tyre Wear & Replacement Prediction.

  lgbm:  LightGBM regressor → km_to_tyre_replacement (per-corner worst case)
  clf:   LightGBM classifier → tyre_replacement_within_30_days
  rules: Puncture rule engine — pressure_drop_rate_kpa_per_day > 5 → immediate alert

Artefacts: models/saved/tyre_wear_{lgbm,clf,scaler}.joblib
"""
from __future__ import annotations

import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from models.model_registry import MODEL_DIR, _shap_importance, _mlflow_log, _cv_metrics

log = logging.getLogger(__name__)

POSITIONS = ["fl", "fr", "rl", "rr"]

FEATURE_COLS = [
    # Per-corner pressure — names match tyre_features.py pipeline output
    *[f"pressure_{p}_7d_avg"         for p in POSITIONS],
    *[f"pressure_{p}_trend_14d"      for p in POSITIONS],
    *[f"pressure_{p}_temp_corrected" for p in POSITIONS],
    # Axle imbalance
    "axle_imbalance_front",
    "axle_imbalance_rear",
    # Pressure drop rate per corner
    *[f"pressure_drop_rate_{p}" for p in POSITIONS],
    # Real TPMS signal from wheelTyreMonitorStatus
    "tpms_status",
    "tpms_deflation_count",
    "tyre_blast_risk_score",
    # Dynamics — real tboxAccelY when available, steer proxy fallback
    "lateral_g_95th_30d",
    "harsh_brake_per_100km_30d",
    "sudden_turn_per_100km_30d",
    # Usage
    "km_since_last_tyre_service",
    "avg_speed_30d",
    "tyre_stress_cumulative",
]
TARGET_REG = "km_to_replacement"
TARGET_CLF = "tyre_within_30_days"

# Immediate puncture alert threshold (kPa/day)
PUNCTURE_THRESHOLD_KPA_PER_DAY = 5.0

_lgbm:   Any = None
_clf:    Any = None
_scaler: StandardScaler | None = None


def _load():
    global _lgbm, _clf, _scaler
    if _lgbm is None:
        _lgbm   = joblib.load(MODEL_DIR / "tyre_wear_lgbm.joblib")
        _clf    = joblib.load(MODEL_DIR / "tyre_wear_clf.joblib")
        _scaler = joblib.load(MODEL_DIR / "tyre_wear_scaler.joblib")
    return _lgbm, _clf, _scaler


def _optuna_lgbm(X, y, task: str = "reg", n_trials: int = 40) -> dict:
    """Optuna HPO for LightGBM. Returns best params dict."""
    try:
        import optuna
        import lightgbm as lgb
        from sklearn.metrics import mean_absolute_error, roc_auc_score

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        tscv = TimeSeriesSplit(n_splits=5)

        def objective(trial):
            params = {
                "n_estimators":     trial.suggest_int("n_est", 100, 500),
                "num_leaves":       trial.suggest_int("num_leaves", 20, 150),
                "learning_rate":    trial.suggest_float("lr", 0.01, 0.3, log=True),
                "subsample":        trial.suggest_float("sub", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("col", 0.5, 1.0),
                "reg_alpha":        trial.suggest_float("alpha", 1e-8, 1.0, log=True),
                "random_state": 42,
                "verbose": -1,
            }
            scores = []
            for tr, te in tscv.split(X):
                if len(te) < 2:
                    continue
                if task == "clf":
                    m = lgb.LGBMClassifier(**params)
                    m.fit(X[tr], y[tr])
                    if y[te].sum() == 0:
                        continue
                    preds = m.predict_proba(X[te])[:, 1]
                    s = -roc_auc_score(y[te], preds)
                else:
                    m = lgb.LGBMRegressor(**params)
                    m.fit(X[tr], y[tr])
                    preds = m.predict(X[te])
                    s = mean_absolute_error(y[te], preds)
                if np.isfinite(s):
                    scores.append(s)
            valid = [s for s in scores if np.isfinite(s)]
            return float(np.mean(valid)) if valid else 1e9

        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        p = study.best_params
        return {
            "n_estimators": p["n_est"], "num_leaves": p["num_leaves"],
            "learning_rate": p["lr"],   "subsample": p["sub"],
            "colsample_bytree": p["col"], "reg_alpha": p["alpha"],
            "random_state": 42, "verbose": -1,
        }
    except Exception as exc:
        log.warning("Optuna LightGBM HPO failed: %s", exc)
        return {"n_estimators": 200, "num_leaves": 63, "learning_rate": 0.05,
                "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42, "verbose": -1}


# ── train ─────────────────────────────────────────────────────────────────────

def train(features_df: pd.DataFrame, experiment: str = "autopredict-v1") -> dict:
    import lightgbm as lgb

    global _lgbm, _clf, _scaler

    df = (features_df.sort_values("computed_at")
          if "computed_at" in features_df.columns else features_df).copy()

    avail  = [c for c in FEATURE_COLS if c in df.columns]
    df_reg = df.dropna(subset=avail + [TARGET_REG])
    df_clf = df.dropna(subset=avail + [TARGET_CLF])

    if len(df_reg) < 10:
        return {"skipped": True, "reason": "insufficient data"}

    scaler = StandardScaler()
    X_reg  = np.nan_to_num(scaler.fit_transform(df_reg[avail].fillna(0)), nan=0.0, posinf=0.0, neginf=0.0)
    y_reg  = df_reg[TARGET_REG].values.astype(float)
    X_clf  = np.nan_to_num(scaler.transform(df_clf[avail].fillna(0)), nan=0.0, posinf=0.0, neginf=0.0)
    y_clf  = df_clf[TARGET_CLF].values.astype(int)

    print("    [tyre_wear] Optuna HPO (reg) ...", end=" ", flush=True)
    best_reg = _optuna_lgbm(X_reg, y_reg, task="reg", n_trials=40)
    print("done")
    print("    [tyre_wear] Optuna HPO (clf) ...", end=" ", flush=True)
    best_clf = _optuna_lgbm(X_clf, y_clf, task="clf", n_trials=30)
    print("done")

    lgbm_reg = lgb.LGBMRegressor(**best_reg)
    lgbm_reg.fit(X_reg, y_reg)

    lgbm_clf = lgb.LGBMClassifier(**best_clf)
    lgbm_clf.fit(X_clf, y_clf)

    # CV metrics via sklearn-compatible wrappers
    cv_reg = _cv_metrics(lgb.LGBMRegressor, X_reg, y_reg, best_reg, task="reg")
    cv_clf = _cv_metrics(lgb.LGBMClassifier, X_clf, y_clf, best_clf, task="clf")

    # SHAP via lightgbm built-in (fast)
    try:
        fi   = lgbm_reg.feature_importances_
        idx  = np.argsort(fi)[::-1][:5]
        shap_top = {avail[i]: round(float(fi[i]), 4) for i in idx}
    except Exception:
        shap_top = {}

    metrics = {
        **cv_reg, **cv_clf,
        "n_reg": len(df_reg),
        "n_clf": len(df_clf),
    }
    _mlflow_log(experiment, "tyre_wear", metrics, best_reg, shap_top)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(lgbm_reg, MODEL_DIR / "tyre_wear_lgbm.joblib")
    joblib.dump(lgbm_clf, MODEL_DIR / "tyre_wear_clf.joblib")
    joblib.dump(scaler,   MODEL_DIR / "tyre_wear_scaler.joblib")
    _lgbm, _clf, _scaler = lgbm_reg, lgbm_clf, scaler

    return metrics


# ── evaluate ──────────────────────────────────────────────────────────────────

def evaluate(features_df: pd.DataFrame) -> dict:
    lgbm, clf, scaler = _load()
    avail   = [c for c in FEATURE_COLS if c in features_df.columns]
    metrics: dict[str, Any] = {}

    df_reg = features_df.dropna(subset=avail + [TARGET_REG])
    if len(df_reg) > 0:
        X = np.nan_to_num(scaler.transform(df_reg[avail].fillna(0)), nan=0.0, posinf=0.0, neginf=0.0)
        p = lgbm.predict(X)
        metrics["mae"]  = round(float(mean_absolute_error(df_reg[TARGET_REG], p)), 1)
        metrics["rmse"] = round(float(np.sqrt(mean_squared_error(df_reg[TARGET_REG], p))), 1)

    df_clf = features_df.dropna(subset=avail + [TARGET_CLF])
    if len(df_clf) > 0 and df_clf[TARGET_CLF].sum() > 0:
        X = np.nan_to_num(scaler.transform(df_clf[avail].fillna(0)), nan=0.0, posinf=0.0, neginf=0.0)
        metrics["auc"] = round(float(roc_auc_score(df_clf[TARGET_CLF], clf.predict_proba(X)[:, 1])), 4)

    return metrics


# ── Puncture rule engine ───────────────────────────────────────────────────────

def _puncture_alert(features_df: pd.DataFrame) -> np.ndarray:
    """Returns boolean array: True where a puncture is detected."""
    if "max_pressure_drop_rate_kpa_per_day" in features_df.columns:
        return features_df["max_pressure_drop_rate_kpa_per_day"].fillna(0).values > PUNCTURE_THRESHOLD_KPA_PER_DAY
    return np.zeros(len(features_df), dtype=bool)


# ── predict_single ────────────────────────────────────────────────────────────

def predict_single(vin: str) -> dict:
    from features.tyre_features import TyreFeaturePipeline
    feats = TyreFeaturePipeline().compute_from_influx(vin, lookback_days=30)
    if feats is None or feats.empty:
        return {"severity": "unknown", "error": f"No tyre data for VIN {vin}"}
    return predict_batch(feats).iloc[0].to_dict()


# ── predict_batch ─────────────────────────────────────────────────────────────

def predict_batch(features_df: pd.DataFrame) -> pd.DataFrame:
    lgbm, clf, scaler = _load()
    avail = [c for c in FEATURE_COLS if c in features_df.columns]
    X     = np.nan_to_num(scaler.transform(features_df[avail].fillna(0)), nan=0.0, posinf=0.0, neginf=0.0)

    km_pred    = np.clip(lgbm.predict(X), 0, 80_000)
    prob_30d   = clf.predict_proba(X)[:, 1]
    puncture   = _puncture_alert(features_df).astype(int)

    urgency = np.where(
        puncture == 1,    "critical",
        np.where(prob_30d > 0.7, "critical",
        np.where(prob_30d > 0.4, "warning", "ok"))
    )

    return pd.DataFrame({
        "vin":                          features_df["vin"].values if "vin" in features_df.columns else [""] * len(X),
        "km_to_tyre_replacement":       np.round(km_pred, 0),
        "tyre_replacement_prob_30d":    np.round(prob_30d, 4),
        "puncture_alert":               puncture,
        "urgency":                      urgency,
    })
