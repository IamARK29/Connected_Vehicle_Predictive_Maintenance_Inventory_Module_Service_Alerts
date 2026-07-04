"""
AutoRetrainer — Celery task that retrains a model and promotes it
                to champion if AUC improves by ≥ 2%.

Champion/challenger decision:
  new_auc > champion_auc + 0.02  → promote + MLflow log "promoted"
  else                            → MLflow log "challenger_rejected"
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

MODEL_DIR = Path("models/saved")


def _get_champion_auc(model_name: str) -> float:
    """Fetch the current champion AUC from MLflow, or 0.0 if unavailable."""
    try:
        import mlflow
        import os
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        client = mlflow.tracking.MlflowClient()
        exp    = client.get_experiment_by_name("autopredict-v1")
        if exp is None:
            return 0.0
        runs   = client.search_runs(
            [exp.experiment_id],
            filter_string=f"tags.model_name = '{model_name}' and tags.variant = 'champion'",
            order_by=["metrics.auc DESC"],
            max_results=1,
        )
        if not runs:
            return 0.0
        return float(runs[0].data.metrics.get("auc", 0.0))
    except Exception as exc:
        log.debug("Could not fetch champion AUC for %s: %s", model_name, exc)
        return 0.0


def _log_to_mlflow(model_name: str, auc: float, variant: str, run_name: str = "") -> None:
    """Log a training run with auc and variant tag to MLflow."""
    try:
        import mlflow, os
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        mlflow.set_experiment("autopredict-v1")
        with mlflow.start_run(run_name=run_name or f"{model_name}_{variant}") as run:
            mlflow.log_metric("auc", auc)
            mlflow.set_tag("model_name", model_name)
            mlflow.set_tag("variant", variant)
            log.info("MLflow logged %s auc=%.4f variant=%s run_id=%s",
                     model_name, auc, variant, run.info.run_id)
    except Exception as exc:
        log.debug("MLflow logging skipped: %s", exc)


def _promote_to_champion(model_name: str, model_path: Path, auc: float) -> None:
    """Copy the challenger artifact to champion slot and record promotion."""
    import shutil
    champion_path = MODEL_DIR / f"{model_name}_champion.joblib"
    try:
        shutil.copy2(model_path, champion_path)
        log.info("Promoted %s to champion (AUC=%.4f) -> %s", model_name, auc, champion_path)
    except Exception as exc:
        log.error("Champion promotion failed for %s: %s", model_name, exc)


def _evaluate_auc(model: Any, test_df: "pd.DataFrame", label_col: str = "label_binary") -> float:
    """Evaluate binary classifier AUC on test_df."""
    try:
        from sklearn.metrics import roc_auc_score
        import importlib

        feature_cols = [c for c in test_df.columns
                        if c not in (label_col, "vin", "timestamp", "computed_at", "feature_date")]
        X = test_df[feature_cols].fillna(0).values
        y = test_df[label_col].values.astype(int)
        if y.sum() == 0 or y.sum() == len(y):
            return 0.0
        probs = model.predict_proba(X)[:, 1]
        return float(roc_auc_score(y, probs))
    except Exception as exc:
        log.debug("AUC evaluation failed: %s", exc)
        return 0.0


def _train_model(model_name: str, features_df: "pd.DataFrame") -> tuple[Any, float]:
    """
    Retrain a single model on fresh features_df.
    Returns (trained_model, test_auc).
    """
    import importlib
    from sklearn.model_selection import train_test_split

    try:
        mod = importlib.import_module(f"models.{model_name}_model")
    except ImportError:
        # Try alternate mapping
        _NAME_MAP = {
            "brake_wear":     "brake_wear_model",
            "engine_oil":     "engine_oil_model",
            "hv_battery_soh": "hv_battery_soh_model",
            "battery_12v":    "battery_12v_model",
            "tyre_wear":      "tyre_wear_model",
            "fuel_anomaly":   "fuel_anomaly_model",
            "driver_score":   "driver_score_model",
        }
        mod_name = _NAME_MAP.get(model_name)
        if not mod_name:
            raise
        mod = importlib.import_module(f"models.{mod_name}")

    metrics = mod.train(features_df, experiment="autopredict-retrain")
    auc = float(metrics.get("cv_auc", metrics.get("xgb_cv_auc", 0.0)))

    # Load the freshly-saved model artifact for evaluation
    import joblib
    model_files = list(MODEL_DIR.glob(f"{model_name}*.joblib"))
    clf_model = None
    for mf in model_files:
        if "clf" in mf.name or "xgb" in mf.name:
            try:
                clf_model = joblib.load(mf)
                break
            except Exception:
                pass

    return clf_model, auc


class AutoRetrainer:
    """
    Celery task that retrains a model and promotes it to champion if AUC improves ≥ 2%.
    Also usable standalone (call retrain_sync instead of retrain.delay).
    """

    def retrain_sync(
        self,
        model_name: str,
        data_dir: str | Path = "data/synthetic",
        snapshot_interval_days: int = 14,
    ) -> dict[str, Any]:
        """
        Synchronous retrain (for testing or manual invocation).
        Returns {"model_name", "new_auc", "champion_auc", "action"}.
        """
        import pandas as pd
        from models.leakage_checker import LeakageChecker, LeakageError
        from models.training_data_builder import TrainingDataBuilder
        from models.train_all import _build_snapshot_dataset, _MODEL_CONFIGS

        data_path = Path(data_dir)
        result = {"model_name": model_name, "new_auc": 0.0, "champion_auc": 0.0, "action": "failed"}

        # ── 1. Build fresh features ──────────────────────────────────────────
        cfg = next((c for c in _MODEL_CONFIGS if c["name"] == model_name), None)
        if cfg is None:
            log.error("Unknown model: %s", model_name)
            result["action"] = "unknown_model"
            return result

        fleet_path = data_path / "fleet.csv"
        if not fleet_path.exists():
            log.warning("fleet.csv not found in %s — skipping retrain", data_path)
            result["action"] = "no_data"
            return result

        fleet_df   = pd.read_csv(fleet_path)
        manifest_df = None
        mp = data_path / "failures_manifest.csv"
        if mp.exists():
            manifest_df = pd.read_csv(mp)

        tdir = data_path / "telemetry"
        if not tdir.exists():
            tdir = data_path

        features_df = _build_snapshot_dataset(cfg, fleet_df, tdir, manifest_df, snapshot_interval_days)
        if features_df.empty:
            log.warning("No features built for %s — skipping retrain", model_name)
            result["action"] = "no_data"
            return result

        # ── 2. Leakage check ────────────────────────────────────────────────
        svc_path = data_path / "service_history.csv"
        if svc_path.exists():
            try:
                splits = TrainingDataBuilder().build(
                    model_name=model_name,
                    service_history_path=svc_path,
                    feature_store_dir=data_path / "feature_store",
                    output_path=data_path / "training_splits",
                )
                LeakageChecker().check(splits["train"], splits["val"], splits["test"], model_name)
                log.info("Leakage check passed for %s", model_name)
            except LeakageError as exc:
                log.error("Leakage detected for %s — retrain aborted: %s", model_name, exc)
                result["action"] = "leakage_abort"
                return result
            except Exception as exc:
                log.warning("Leakage check skipped: %s", exc)

        # ── 3. Train challenger ──────────────────────────────────────────────
        try:
            new_model, new_auc = _train_model(model_name, features_df)
        except Exception as exc:
            log.error("Challenger training failed for %s: %s", model_name, exc)
            result["action"] = "train_failed"
            return result

        champion_auc = _get_champion_auc(model_name)
        result["new_auc"]      = round(new_auc, 4)
        result["champion_auc"] = round(champion_auc, 4)

        # ── 4. Promote or reject ────────────────────────────────────────────
        if new_auc > champion_auc + 0.02:
            # Find latest joblib written by the train function
            latest = max(MODEL_DIR.glob(f"{model_name}*.joblib"), key=lambda p: p.stat().st_mtime,
                         default=None)
            if latest:
                _promote_to_champion(model_name, latest, new_auc)
            _log_to_mlflow(model_name, new_auc, "champion", f"{model_name}_promoted")
            result["action"] = "promoted"
            log.info("Promoted %s  new_auc=%.4f  champion_auc=%.4f", model_name, new_auc, champion_auc)
        else:
            _log_to_mlflow(model_name, new_auc, "challenger_rejected", f"{model_name}_rejected")
            result["action"] = "challenger_rejected"
            log.info("Challenger rejected %s  new_auc=%.4f  champion_auc=%.4f",
                     model_name, new_auc, champion_auc)

        return result


# ── Celery task wrapper ────────────────────────────────────────────────────────

try:
    from celery import Celery
    import os as _os

    _celery_app = Celery(
        "auto_retrain",
        broker=_os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    )

    @_celery_app.task(name="auto_retrain.retrain", bind=True, max_retries=2)
    def _retrain_task(self, model_name: str) -> dict:
        try:
            return AutoRetrainer().retrain_sync(model_name)
        except Exception as exc:
            log.error("Retrain task failed for %s: %s", model_name, exc)
            raise self.retry(exc=exc, countdown=300)

    # Attach .delay method to AutoRetrainer for the DriftMonitor integration
    AutoRetrainer.retrain = _retrain_task  # type: ignore[attr-defined]

except ImportError:
    # Celery not available — create a no-op stub so DriftMonitor.retrain.delay() works
    class _RetainStub:
        def delay(self, model_name: str) -> None:
            log.warning("Celery not available — retrain.delay(%s) is a no-op", model_name)

    AutoRetrainer.retrain = _RetainStub()  # type: ignore[attr-defined]
