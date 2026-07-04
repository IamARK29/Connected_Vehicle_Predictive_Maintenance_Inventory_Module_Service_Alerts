"""
HV Battery State-of-Health Prediction Models (EV / PHEV only).

  lr:   Linear regression → soh_estimated (from charge-cycle energy)
  arima: ARIMA(2,1,1) per-VIN SoH time-series forecaster (inference only)
  iso:  Isolation Forest cell anomaly detector
  clf:  XGBoost classifier → soh_below_80_within_90_days

Artefacts: models/saved/hv_battery_{lr,clf,iso,scaler}.joblib
"""
from __future__ import annotations

import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.preprocessing import StandardScaler

from models.model_registry import MODEL_DIR, _optuna_xgb, _shap_importance, _mlflow_log, _cv_metrics

log = logging.getLogger(__name__)

FEATURE_COLS = [
    "soh_trend_slope_90d",          # Coulomb-counting derived slope (not current SOH — avoids target leakage)
    "cell_voltage_spread",
    "cell_voltage_spread_trend_30d",
    "cell_voltage_spread_p95_30d",
    "avg_cell_temp_delta",
    "dc_charge_count_30d",
    "charge_duration_deviation",
    "avg_charge_c_rate_30d",
    "thermal_fault_count_30d",
    "range_per_kwh_30d_trend",
    "soc_current",
    # Real TBox BMS fault signals
    "bms_cmu_fault_count_30d",      # from vehBMSCMUFlt
    "bms_cv_fault_count_30d",       # from vehBMSCellVoltFlt
    "bms_pt_fault_count_30d",       # from vehBMSPackTemFlt
    "max_fault_severity_30d",       # max fault level (0-3)
    "dcdc_temp_max_30d",            # from vehHVDCDCTem
    "dc_charge_ratio_30d",          # DC fast charge ratio
]
TARGET_REG = "soh_estimated"
TARGET_CLF = "soh_below_80_within_90_days"

# Isolation Forest features (cell anomaly)
ISO_COLS = ["cell_voltage_spread", "avg_cell_temp_delta", "charge_duration_deviation"]

_lr:     Any = None
_clf:    Any = None
_iso:    Any = None
_scaler: StandardScaler | None = None


def _load():
    global _lr, _clf, _iso, _scaler
    if _lr is None:
        _lr     = joblib.load(MODEL_DIR / "hv_battery_lr.joblib")
        _clf    = joblib.load(MODEL_DIR / "hv_battery_clf.joblib")
        _iso    = joblib.load(MODEL_DIR / "hv_battery_iso.joblib")
        _scaler = joblib.load(MODEL_DIR / "hv_battery_scaler.joblib")
    return _lr, _clf, _iso, _scaler


# ── train ─────────────────────────────────────────────────────────────────────

def train(features_df: pd.DataFrame, experiment: str = "autopredict-v1") -> dict:
    import xgboost as xgb
    from sklearn.ensemble import IsolationForest

    global _lr, _clf, _iso, _scaler

    # Filter EV/PHEV only
    df = features_df.copy()
    if "hv_applicable" in df.columns:
        df = df[df["hv_applicable"] == 1]
    if "fuel_type" in df.columns:
        df = df[df["fuel_type"].isin(["EV", "PHEV"])]

    if "computed_at" in df.columns:
        df = df.sort_values("computed_at")

    avail     = [c for c in FEATURE_COLS if c in df.columns]
    avail_iso = [c for c in ISO_COLS     if c in df.columns]

    df_reg = df.dropna(subset=avail + [TARGET_REG])
    df_clf = df.dropna(subset=avail + [TARGET_CLF])

    if len(df_reg) < 5:
        log.warning("hv_battery_soh: only %d EV rows — skipping", len(df_reg))
        return {"skipped": True, "reason": "insufficient EV data"}

    scaler  = StandardScaler()
    X_reg   = scaler.fit_transform(df_reg[avail].fillna(0))
    y_reg   = df_reg[TARGET_REG].values.astype(float)
    X_clf   = scaler.transform(df_clf[avail].fillna(0)) if len(df_clf) >= 5 else X_reg
    y_clf   = df_clf[TARGET_CLF].values.astype(int)    if len(df_clf) >= 5 else np.zeros(len(y_reg), dtype=int)

    # ── Linear regression for SoH estimation (interpretable) ──────────────
    lr = Ridge(alpha=1.0)
    lr.fit(X_reg, y_reg)
    lr_preds = lr.predict(X_reg)
    lr_mae   = float(mean_absolute_error(y_reg, lr_preds))
    lr_rmse  = float(np.sqrt(mean_squared_error(y_reg, lr_preds)))

    # ── XGBoost binary classifier ──────────────────────────────────────────
    if y_clf.sum() == 0:
        log.warning("hv_battery_soh: y_clf all-zeros — skipping classifier")
        return {"skipped": True, "reason": "no positive classifier labels"}
    best_clf = _optuna_xgb(X_clf, y_clf, task="clf", n_trials=30)
    clf_params = {**best_clf, "use_label_encoder": False, "eval_metric": "logloss",
                  "base_score": float(np.clip(y_clf.mean(), 0.01, 0.99))}
    clf = xgb.XGBClassifier(**clf_params)
    clf.fit(X_clf, y_clf)
    cv_clf = _cv_metrics(xgb.XGBClassifier, X_clf, y_clf, clf_params, task="clf")

    # ── Isolation Forest for cell anomaly ────────────────────────────────
    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    X_iso = df_reg[avail_iso].fillna(0).values if avail_iso else X_reg[:, :2]
    iso.fit(X_iso)

    # ── SHAP ──────────────────────────────────────────────────────────────
    shap_top = _shap_importance(clf, X_clf, avail)

    metrics = {
        "lr_mae":  round(lr_mae, 3),
        "lr_rmse": round(lr_rmse, 3),
        **cv_clf,
        "n_ev":    len(df_reg),
    }
    _mlflow_log(experiment, "hv_battery_soh", metrics, best_clf, shap_top)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(lr,     MODEL_DIR / "hv_battery_lr.joblib")
    joblib.dump(clf,    MODEL_DIR / "hv_battery_clf.joblib")
    joblib.dump(iso,    MODEL_DIR / "hv_battery_iso.joblib")
    joblib.dump(scaler, MODEL_DIR / "hv_battery_scaler.joblib")
    _lr, _clf, _iso, _scaler = lr, clf, iso, scaler

    return metrics


# ── evaluate ──────────────────────────────────────────────────────────────────

def evaluate(features_df: pd.DataFrame) -> dict:
    lr, clf, iso, scaler = _load()
    avail = [c for c in FEATURE_COLS if c in features_df.columns]
    metrics: dict[str, Any] = {}

    df_reg = features_df.dropna(subset=avail + [TARGET_REG])
    if len(df_reg) > 0:
        X  = scaler.transform(df_reg[avail].fillna(0))
        p  = lr.predict(X)
        metrics["lr_mae"]  = round(float(mean_absolute_error(df_reg[TARGET_REG], p)), 3)
        metrics["lr_rmse"] = round(float(np.sqrt(mean_squared_error(df_reg[TARGET_REG], p))), 3)

    df_clf = features_df.dropna(subset=avail + [TARGET_CLF])
    if len(df_clf) > 0 and df_clf[TARGET_CLF].sum() > 0:
        X = scaler.transform(df_clf[avail].fillna(0))
        metrics["clf_auc"] = round(float(roc_auc_score(df_clf[TARGET_CLF], clf.predict_proba(X)[:, 1])), 4)

    return metrics


# ── predict_single ────────────────────────────────────────────────────────────

def predict_single(vin: str) -> dict:
    from features.battery_hv_features import HVBatteryFeaturePipeline
    feats = HVBatteryFeaturePipeline().compute_from_influx(vin, lookback_days=90)
    if feats is None or feats.empty:
        return {"severity": "unknown", "error": f"No EV/HV data for VIN {vin}"}

    # Base prediction
    result = predict_batch(feats).iloc[0].to_dict()

    # ARIMA forecast for SOH trend
    arima_forecast = _arima_soh_forecast(vin)
    if arima_forecast:
        result.update(arima_forecast)

    return result


def _arima_soh_forecast(vin: str) -> dict:
    """Fit ARIMA(2,1,1) on per-VIN SoH history and forecast 90 days."""
    try:
        from statsmodels.tsa.arima.model import ARIMA
        from features.battery_hv_features import HVBatteryFeaturePipeline
        from features.base_pipeline import _COL_ALIASES

        pipe = HVBatteryFeaturePipeline()
        df   = pipe._query_influx(vin, lookback_days=90)
        if df.empty or "soh" not in df.rename(columns=_COL_ALIASES).columns:
            return {}
        df = pipe._normalize(df)
        if "soh" not in df.columns:
            return {}

        soh_daily = (df.set_index("timestamp")["soh"]
                       .resample("D").mean().dropna())
        if len(soh_daily) < 14:
            return {}

        model = ARIMA(soh_daily, order=(2, 1, 1))
        fit   = model.fit()
        fc    = fit.forecast(steps=90)
        min_fc = float(fc.min())
        return {
            "arima_soh_90d_forecast_min": round(min_fc, 1),
            "arima_soh_below_80_forecast": int(min_fc < 80),
        }
    except Exception as exc:
        log.debug("ARIMA forecast failed for %s: %s", vin, exc)
        return {}


# ── predict_batch ─────────────────────────────────────────────────────────────

def predict_batch(features_df: pd.DataFrame) -> pd.DataFrame:
    lr, clf, iso, scaler = _load()
    avail     = [c for c in FEATURE_COLS if c in features_df.columns]
    avail_iso = [c for c in ISO_COLS     if c in features_df.columns]
    X         = scaler.transform(features_df[avail].fillna(0))

    soh_pred   = np.clip(lr.predict(X), 50, 105)
    prob_below80 = clf.predict_proba(X)[:, 1]
    X_iso      = features_df[avail_iso].fillna(0).values if avail_iso else X[:, :2]
    cell_anomaly = (iso.predict(X_iso) == -1).astype(int)

    return pd.DataFrame({
        "vin":                     features_df["vin"].values if "vin" in features_df.columns else [""] * len(X),
        "predicted_soh_pct":       np.round(soh_pred, 1),
        "prob_soh_below_80_90d":   np.round(prob_below80, 4),
        "cell_anomaly_detected":   cell_anomaly,
        "battery_health":          np.where(soh_pred > 85, "good",
                                   np.where(soh_pred > 70, "degraded", "poor")),
        "urgency":                 np.where(prob_below80 > 0.7, "critical",
                                   np.where(prob_below80 > 0.4, "warning", "ok")),
    })
