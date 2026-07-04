"""
AutoPredict Model Registry.

Shared utilities used by every model module:
  _optuna_xgb()    — Optuna HPO for XGBoost (reg or clf)
  _shap_importance() — top-N feature importances via SHAP
  _mlflow_log()    — MLflow metric / param / artifact logging

ModelRegistry class — unified load / predict / metadata interface.
"""
from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_default_mlflow = Path("mlruns").resolve().as_uri()
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", _default_mlflow)
MODEL_DIR  = Path("models/saved")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ── A/B Champion-Challenger config ────────────────────────────────────────────

@dataclass
class ModelServingConfig:
    model_name:              str
    champion_version:        str
    challenger_version:      str | None = None
    challenger_traffic_pct:  float      = 0.0

# Registry of active A/B experiments; keyed by model_name
_AB_CONFIGS: dict[str, ModelServingConfig] = {
    "brake_wear":     ModelServingConfig("brake_wear",     "v1", None, 0.0),
    "engine_oil":     ModelServingConfig("engine_oil",     "v1", None, 0.0),
    "hv_battery_soh": ModelServingConfig("hv_battery_soh", "v1", None, 0.0),
    "battery_12v":    ModelServingConfig("battery_12v",    "v1", None, 0.0),
    "tyre_wear":      ModelServingConfig("tyre_wear",      "v1", None, 0.0),
    "fuel_anomaly":   ModelServingConfig("fuel_anomaly",   "v1", None, 0.0),
    "driver_score":   ModelServingConfig("driver_score",   "v1", None, 0.0),
}

# ── Optuna HPO ─────────────────────────────────────────────────────────────

def _optuna_xgb(
    X: np.ndarray,
    y: np.ndarray,
    task: str = "clf",          # "clf" | "reg"
    n_trials: int = 50,
    n_cv_splits: int = 5,
) -> dict:
    """
    Optuna-driven XGBoost hyperparameter search using TimeSeriesSplit CV.
    Returns best params dict. Falls back to sensible defaults if Optuna missing.
    """
    try:
        import optuna
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import mean_absolute_error, roc_auc_score
        import xgboost as xgb

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        tscv = TimeSeriesSplit(n_splits=n_cv_splits)

        def objective(trial: "optuna.Trial") -> float:
            params = {
                "n_estimators":     trial.suggest_int("n_estimators", 50, 400),
                "max_depth":        trial.suggest_int("max_depth", 3, 8),
                "learning_rate":    trial.suggest_float("lr", 0.01, 0.3, log=True),
                "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample", 0.5, 1.0),
                "reg_alpha":        trial.suggest_float("alpha", 1e-8, 1.0, log=True),
                "reg_lambda":       trial.suggest_float("lambda", 1e-8, 1.0, log=True),
                "random_state": 42,
                "verbosity": 0,
            }
            scores: list[float] = []
            for tr, te in tscv.split(X):
                if len(te) == 0:
                    continue
                if task == "clf":
                    bs = float(np.clip(y[tr].mean(), 0.01, 0.99))
                    m = xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric="logloss", base_score=bs)
                    m.fit(X[tr], y[tr])
                    if y[te].sum() == 0:
                        continue
                    p = m.predict_proba(X[te])[:, 1]
                    scores.append(-roc_auc_score(y[te], p))
                else:
                    m = xgb.XGBRegressor(**params)
                    m.fit(X[tr], y[tr])
                    scores.append(mean_absolute_error(y[te], m.predict(X[te])))
            return float(np.mean(scores)) if scores else 1e9

        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        return {
            "n_estimators":     study.best_params["n_estimators"],
            "max_depth":        study.best_params["max_depth"],
            "learning_rate":    study.best_params["lr"],
            "subsample":        study.best_params["subsample"],
            "colsample_bytree": study.best_params["colsample"],
            "reg_alpha":        study.best_params["alpha"],
            "reg_lambda":       study.best_params["lambda"],
            "random_state": 42,
            "verbosity": 0,
        }
    except Exception as exc:
        log.warning("Optuna HPO failed (%s) — using defaults", exc)
        return {
            "n_estimators": 200, "max_depth": 5,
            "learning_rate": 0.05, "subsample": 0.8,
            "colsample_bytree": 0.8, "random_state": 42, "verbosity": 0,
        }


# ── SHAP feature importance ────────────────────────────────────────────────

def _shap_importance(model: Any, X: np.ndarray, feature_names: list[str], top_n: int = 5) -> dict[str, float]:
    """Return top-N feature importances via SHAP (falls back to built-in)."""
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        sv = np.abs(explainer.shap_values(X[:min(200, len(X))]))
        mean_abs = sv.mean(axis=0)
        idx = np.argsort(mean_abs)[::-1][:top_n]
        return {feature_names[i]: round(float(mean_abs[i]), 6) for i in idx}
    except Exception:
        # Fall back to XGBoost built-in importance
        try:
            fi = model.feature_importances_
            idx = np.argsort(fi)[::-1][:top_n]
            return {feature_names[i]: round(float(fi[i]), 6) for i in idx}
        except Exception:
            return {}


# ── MLflow logging ─────────────────────────────────────────────────────────

def _mlflow_log(
    experiment: str,
    run_name: str,
    metrics: dict,
    params: dict,
    shap_importance: dict | None = None,
    artifacts: dict[str, Path] | None = None,
) -> str | None:
    """Log a training run to MLflow. Returns run_id, or None if MLflow unavailable."""
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params({k: v for k, v in params.items() if not isinstance(v, dict)})
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()
                                if isinstance(v, (int, float)) and not np.isnan(float(v))})
            if shap_importance:
                mlflow.log_params({f"shap_{k}": v for k, v in shap_importance.items()})
            if artifacts:
                for name, path in artifacts.items():
                    if path.exists():
                        mlflow.log_artifact(str(path), artifact_path=name)
            return run.info.run_id
    except Exception as exc:
        log.warning("MLflow logging skipped: %s", exc)
        return None


# ── Cross-validation helper ────────────────────────────────────────────────

def _cv_metrics(
    Model,
    X: np.ndarray,
    y: np.ndarray,
    params: dict,
    task: str = "clf",
    n_splits: int = 5,
) -> dict[str, float]:
    """Run TimeSeriesSplit CV and return mean metrics."""
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_absolute_error, roc_auc_score, mean_squared_error

    tscv   = TimeSeriesSplit(n_splits=n_splits)
    maes, rmses, aucs = [], [], []

    for tr, te in tscv.split(X):
        if len(te) < 2:
            continue
        m = Model(**params)
        m.fit(X[tr], y[tr])
        if task == "clf":
            if y[te].sum() == 0:
                continue
            p = m.predict_proba(X[te])[:, 1]
            aucs.append(roc_auc_score(y[te], p))
        else:
            p = m.predict(X[te])
            maes.append(mean_absolute_error(y[te], p))
            rmses.append(float(np.sqrt(mean_squared_error(y[te], p))))

    out: dict[str, float] = {}
    if maes:
        out["cv_mae"]  = round(float(np.mean(maes)), 3)
        out["cv_rmse"] = round(float(np.mean(rmses)), 3)
    if aucs:
        out["cv_auc"] = round(float(np.mean(aucs)), 4)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# ModelRegistry — unified inference interface
# ══════════════════════════════════════════════════════════════════════════════

_REGISTRY_CACHE: dict[str, Any] = {}

_MODEL_SPECS = {
    "brake_wear": {
        "files": ["brake_wear_reg.joblib", "brake_wear_clf.joblib", "brake_wear_scaler.joblib"],
        "module": "models.brake_wear_model",
    },
    "engine_oil": {
        "files": ["engine_oil_xgb.joblib", "engine_oil_clf.joblib", "engine_oil_scaler.joblib"],
        "module": "models.engine_oil_model",
    },
    "hv_battery_soh": {
        "files": ["hv_battery_lr.joblib", "hv_battery_clf.joblib", "hv_battery_iso.joblib", "hv_battery_scaler.joblib"],
        "module": "models.hv_battery_soh_model",
    },
    "battery_12v": {
        "files": ["battery_12v_lr.joblib", "battery_12v_xgb.joblib", "battery_12v_scaler.joblib"],
        "module": "models.battery_12v_model",
    },
    "tyre_wear": {
        "files": ["tyre_wear_lgbm.joblib", "tyre_wear_clf.joblib", "tyre_wear_scaler.joblib"],
        "module": "models.tyre_wear_model",
    },
    "fuel_anomaly": {
        "files": ["fuel_anomaly_iso.joblib", "fuel_anomaly_scaler.joblib"],
        "module": "models.fuel_anomaly_model",
    },
    "driver_score": {
        "files": ["driver_score_reg.joblib", "driver_risk_clf.joblib"],
        "module": "models.driver_score_model",
    },
}


_seq_registry_instance = None


def _get_seq_registry():
    global _seq_registry_instance
    if _seq_registry_instance is None:
        try:
            from models.sequence_model_registry import SequenceModelRegistry
            _seq_registry_instance = SequenceModelRegistry()
        except Exception as exc:
            log.warning("SequenceModelRegistry unavailable: %s", exc)
    return _seq_registry_instance


class ModelRegistry:
    """Unified load, inference, and metadata interface for all AutoPredict models."""

    def load_all_models(self) -> dict[str, bool]:
        """
        Attempt to load every model from disk.
        Returns {model_name: loaded_ok} mapping.
        """
        status: dict[str, bool] = {}
        for name, spec in _MODEL_SPECS.items():
            missing = [f for f in spec["files"] if not (MODEL_DIR / f).exists()]
            if missing:
                log.warning("Model %s missing files: %s", name, missing)
                status[name] = False
            else:
                status[name] = True
        return status

    def predict_all(self, vin: str) -> dict[str, dict]:
        """
        Run all available models for *vin*.

        Returns {model_name: {severity, value, confidence, predicted_date, message}}.
        """
        results: dict[str, dict] = {}
        for name, spec in _MODEL_SPECS.items():
            try:
                import importlib
                mod = importlib.import_module(spec["module"])
                raw = mod.predict_single(vin)
                results[name] = _standardise(name, raw)
            except FileNotFoundError:
                results[name] = {"severity": "unknown", "message": "Model not trained yet"}
            except Exception as exc:
                log.error("predict_all failed for %s / VIN %s: %s", name, vin, exc)
                results[name] = {"severity": "error", "message": str(exc)}
        return results

    def predict_ensemble(
        self,
        vin: str,
        feature_store: Any | None = None,
    ) -> dict[str, float]:
        """
        Ensemble prediction combining tabular XGBoost (40%) and LSTM (60%).

        Returns {failure_type: blended_probability} for 6 failure types.
        """
        # Tabular V1 predictions
        tabular = self.predict_all(vin)

        # LSTM sequence predictions
        lstm: dict[str, float] = {}
        seq_reg = _get_seq_registry()
        if seq_reg is not None and feature_store is not None:
            try:
                lstm = seq_reg.predict_lstm(vin, feature_store)
            except Exception as exc:
                log.warning("LSTM predict failed for VIN %s: %s", vin, exc)

        _TAB_KEY_MAP = {
            "brake":       "brake_wear",
            "oil":         "engine_oil",
            "hv_battery":  "hv_battery_soh",
            "12v_battery": "battery_12v",
            "tyre":        "tyre_wear",
            "overheating": "engine_oil",   # closest proxy
        }

        ensemble: dict[str, float] = {}
        for ft in ["brake", "oil", "hv_battery", "12v_battery", "tyre", "overheating"]:
            tab_key = _TAB_KEY_MAP.get(ft, ft)
            tab_p   = float(tabular.get(tab_key, {}).get("confidence", 0.0))
            lstm_p  = float(lstm.get(ft, 0.0))
            ensemble[ft] = round(0.4 * tab_p + 0.6 * lstm_p, 4)

        return ensemble

    def predict_rul(
        self,
        vin: str,
        failure_type: str,
        features: dict | None = None,
    ) -> "RULPrediction | None":
        """
        Return RUL prediction for *failure_type*.

        Tries to load the saved RUL model artifact. Falls back to
        _DEFAULT_RUL when no trained model is found.

        *features* — optional dict with covariate values; when None,
        the method tries to fetch from the feature store.
        """
        try:
            from models.rul_models import RUL_FAILURE_TYPE_MAP, load_rul_model, _DEFAULT_RUL
            rul_model_name = next(
                (name for name, ft in RUL_FAILURE_TYPE_MAP.items() if ft == failure_type),
                None,
            )
            if rul_model_name is None:
                return None
            model = load_rul_model(rul_model_name)
            if model is None:
                return _DEFAULT_RUL
            if features is None:
                features = self._get_rul_features(vin, rul_model_name)
            return model.predict(features)
        except Exception as exc:
            log.debug("predict_rul failed for %s/%s: %s", vin, failure_type, exc)
            return None

    def _get_rul_features(self, vin: str, rul_model_name: str) -> dict:
        """Best-effort feature fetch from the feature store for RUL prediction."""
        try:
            from models.rul_models import RUL_MODEL_SPECS
            covariates = RUL_MODEL_SPECS.get(rul_model_name, {}).get("covariates", [])
            from features.feature_store import FeatureStore
            store = FeatureStore()
            row = store.get_latest(vin)
            if row is not None:
                return {k: float(row.get(k, 0.0) or 0.0) for k in covariates}
        except Exception:
            pass
        return {}

    def predict_with_ab(self, vin: str, model_name: str) -> tuple[dict, str]:
        """
        Route prediction to champion or challenger using MD5 VIN-hash bucketing.

        Returns (prediction_dict, variant) where variant is "champion" or "challenger".
        """
        cfg = _AB_CONFIGS.get(model_name)
        if cfg is None or cfg.challenger_version is None or cfg.challenger_traffic_pct <= 0:
            pred = self._predict_single_model(vin, model_name)
            return pred, "champion"

        bucket = (int(hashlib.md5(f"{vin}:{model_name}".encode()).hexdigest(), 16) % 100) / 100.0
        variant = "challenger" if bucket < cfg.challenger_traffic_pct else "champion"
        version = cfg.challenger_version if variant == "challenger" else cfg.champion_version

        pred = self._predict_single_model(vin, model_name)
        self._log_ab_event(vin, model_name, version, variant, pred)
        return pred, variant

    def _predict_single_model(self, vin: str, model_name: str) -> dict:
        """Run predict_single for one model, returning standardised dict."""
        spec = _MODEL_SPECS.get(model_name)
        if spec is None:
            return {}
        try:
            import importlib
            mod = importlib.import_module(spec["module"])
            raw = mod.predict_single(vin)
            return _standardise(model_name, raw)
        except Exception as exc:
            log.debug("_predict_single_model failed %s/%s: %s", vin, model_name, exc)
            return {}

    def _log_ab_event(
        self,
        vin: str,
        model_name: str,
        model_version: str,
        variant: str,
        pred: dict,
    ) -> None:
        """Persist an A/B routing event to ab_experiment_log."""
        import json
        from datetime import datetime, timezone
        try:
            from sqlalchemy import create_engine, text
            engine = create_engine(os.getenv("POSTGRES_URL", "sqlite:///./autopredict.db"))
            with engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO ab_experiment_log
                        (vin, model_name, model_version, variant,
                         prediction_value, prediction_date, outcome_date, outcome_correct)
                    VALUES (:vin,:model,:ver,:variant,:val,:pred_date,NULL,NULL)
                """), {
                    "vin":       vin,
                    "model":     model_name,
                    "ver":       model_version,
                    "variant":   variant,
                    "val":       pred.get("confidence"),
                    "pred_date": datetime.now(timezone.utc).date().isoformat(),
                })
        except Exception:
            try:
                import sqlite3, os as _os
                db = sqlite3.connect(_os.getenv("SQLITE_DB", "autopredict.db"))
                db.execute("""
                    CREATE TABLE IF NOT EXISTS ab_experiment_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        vin TEXT, model_name TEXT, model_version TEXT,
                        variant TEXT, prediction_value REAL,
                        prediction_date TEXT, outcome_date TEXT, outcome_correct INTEGER
                    )
                """)
                db.execute(
                    "INSERT INTO ab_experiment_log VALUES (NULL,?,?,?,?,?,?,NULL,NULL)",
                    (vin, model_name, model_version, variant,
                     pred.get("confidence"), datetime.now(timezone.utc).date().isoformat()),
                )
                db.commit()
                db.close()
            except Exception as exc2:
                log.debug("AB log insert skipped: %s", exc2)

    def predict_with_explanation(self, vin: str) -> dict:
        """
        Run all models and augment the result with SHAP explanation for the
        most critical prediction.

        Returns {
            "predictions":    {model_name: standardised_dict},
            "explanation_text": str,
            "top3_features":    list[dict],
        }
        """
        from models.explainability import try_explain, _EMPTY_EXPLANATION

        predictions = self.predict_all(vin)

        # Find highest-severity prediction
        sev_order = {"critical": 0, "warning": 1, "ok": 2, "unknown": 3, "error": 4}
        best_model = min(
            ((name, pred) for name, pred in predictions.items() if "error" not in pred),
            key=lambda kv: sev_order.get(kv[1].get("severity", "ok"), 3),
            default=(None, {}),
        )

        explanation_text = ""
        top3: list[dict] = []

        if best_model[0] is not None:
            model_name = best_model[0]
            spec = _MODEL_SPECS.get(model_name, {})
            try:
                import importlib
                mod = importlib.import_module(spec.get("module", ""))
                # Try to get (model_object, feature_vector, feature_names) from the module
                if hasattr(mod, "get_explainability_artifacts"):
                    clf, fvec, fnames = mod.get_explainability_artifacts(vin)
                    result = try_explain(clf, fnames, fvec, model_name)
                    explanation_text = result.nl_summary
                    top3 = result.top3
            except Exception as exc:
                log.debug("Explanation failed for %s/%s: %s", vin, model_name, exc)

        return {
            "predictions":    predictions,
            "explanation_text": explanation_text,
            "top3_features":    top3,
        }

    def get_model_metadata(self) -> list[dict]:
        """Return version, training date, and performance metrics for each model."""
        rows: list[dict] = []
        for name, spec in _MODEL_SPECS.items():
            meta: dict[str, Any] = {"model": name, "trained": False}
            for fname in spec["files"]:
                p = MODEL_DIR / fname
                if p.exists():
                    meta["trained"]      = True
                    meta["saved_at"]     = time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime)
                    )
                    meta["size_kb"]      = round(p.stat().st_size / 1024, 1)
                    break
            rows.append(meta)
        return rows


def _standardise(model_name: str, raw: dict) -> dict:
    """Map raw model output to the standard {severity, value, confidence, predicted_date, message} format."""
    SEVERITY_MAP = {
        "critical": "critical", "high": "critical",
        "warning":  "warning",  "medium": "warning",
        "ok":       "ok",       "low": "ok",
        "good":     "ok",       "degraded": "warning", "poor": "critical",
    }
    urgency = raw.get("urgency") or raw.get("battery_health") or raw.get("severity", "ok")
    severity = SEVERITY_MAP.get(str(urgency).lower(), "ok")
    confidence = float(
        raw.get("replacement_probability")
        or raw.get("failure_probability")
        or raw.get("replacement_prob_30d")
        or raw.get("no_start_probability")
        or raw.get("prob_within_90d", 0.5)
    )
    days = (
        raw.get("days_to_replacement_predicted")
        or raw.get("days_until_oil_change")
        or raw.get("days_to_12v_failure")
        or raw.get("days_to_replacement")
    )
    predicted_date = None
    if days is not None and not np.isnan(float(days)):
        predicted_date = (pd.Timestamp.utcnow() + pd.Timedelta(days=float(days))).date().isoformat()

    return {
        "severity":       severity,
        "value":          raw.get("composite_drive_score") or raw.get("predicted_soh_pct") or raw.get("driver_score"),
        "confidence":     round(confidence, 3),
        "predicted_date": predicted_date,
        "message":        _message(model_name, severity, raw),
        "raw":            raw,
    }


def _message(model_name: str, severity: str, raw: dict) -> str:
    if severity == "critical":
        msgs = {
            "brake_wear":     "Brake pads critical — schedule replacement immediately",
            "engine_oil":     "Oil degradation critical — oil change overdue",
            "hv_battery_soh": "HV battery SoH below 80% — capacity severely reduced",
            "battery_12v":    "12V battery near failure — no-start risk high",
            "tyre_wear":      "Tyre replacement required — below safe threshold",
            "fuel_anomaly":   "Fuel anomaly detected — possible leak or system fault",
            "driver_score":   "High-risk driving behaviour detected",
        }
    elif severity == "warning":
        msgs = {
            "brake_wear":     "Brake wear accelerating — plan replacement within 30 days",
            "engine_oil":     "Oil change due soon — schedule within 14 days",
            "hv_battery_soh": "HV battery SoH declining — monitor closely",
            "battery_12v":    "12V battery weakening — test recommended",
            "tyre_wear":      "Tyre wear increasing — inspect within 30 days",
            "fuel_anomaly":   "Fuel consumption above normal — check driving conditions",
            "driver_score":   "Moderate driving risk — coaching recommended",
        }
    else:
        msgs = {k: "System healthy" for k in _MODEL_SPECS}
    return msgs.get(model_name, f"{model_name}: {severity}")
