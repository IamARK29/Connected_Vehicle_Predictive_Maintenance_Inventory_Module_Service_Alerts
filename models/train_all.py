"""
Master training script — trains all AutoPredict ML models in sequence.

Usage:
    python models/train_all.py \\
        --data-dir data/synthetic/ \\
        --experiment autopredict-v1 \\
        --snapshot-interval-days 14

What it does:
  1. Reads per-VIN telemetry CSVs + failures_manifest.csv from --data-dir
  2. Reads fleet.csv for vehicle metadata
  3. For each model, calls its feature pipeline in batch mode at multiple
     time snapshots (every --snapshot-interval-days), producing ~N_vins x N_snaps
     training rows (~600 rows for 50 VINs x 12 snapshots)
  4. Trains each model and prints a summary table

Output:
    models/saved/*.joblib   — trained model artefacts
    models/saved/train_summary.csv
"""
from __future__ import annotations

import argparse
import importlib
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from models.leakage_checker import LeakageChecker, LeakageError
from models.training_data_builder import TrainingDataBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_all")

# ── Model configurations ────────────────────────────────────────────────────

_MODEL_CONFIGS = [
    {
        "name":       "brake_wear",
        "module":     "models.brake_wear_model",
        "pipeline":   "features.brake_features.BrakeFeaturePipeline",
        "lookback":   30,
        "metric_key": "cv_rmse",
    },
    {
        "name":       "engine_oil",
        "module":     "models.engine_oil_model",
        "pipeline":   "features.engine_features.EngineFeaturePipeline",
        "lookback":   90,
        "metric_key": "cv_mae",
    },
    {
        "name":       "hv_battery_soh",
        "module":     "models.hv_battery_soh_model",
        "pipeline":   "features.battery_hv_features.HVBatteryFeaturePipeline",
        "lookback":   90,
        "metric_key": "cv_auc",
    },
    {
        "name":       "battery_12v",
        "module":     "models.battery_12v_model",
        "pipeline":   "features.battery_12v_features.Battery12VFeaturePipeline",
        "lookback":   30,
        "metric_key": "xgb_cv_auc",
    },
    {
        "name":       "tyre_wear",
        "module":     "models.tyre_wear_model",
        "pipeline":   "features.tyre_features.TyreFeaturePipeline",
        "lookback":   30,
        "metric_key": "cv_rmse",
    },
    {
        "name":       "fuel_anomaly",
        "module":     "models.fuel_anomaly_model",
        "pipeline":   "features.engine_features.EngineFeaturePipeline",
        "lookback":   30,
        "metric_key": "anomaly_rate",
    },
    {
        "name":       "driver_score",
        "module":     "models.driver_score_model",
        "pipeline":   "features.driver_behaviour_features.DriverBehaviourFeaturePipeline",
        "lookback":   30,
        "metric_key": "cv_rmse",
    },
]


# ── Dataset builder ──────────────────────────────────────────────────────────

def _load_pipeline(pipeline_path: str):
    """Import and instantiate a pipeline class from a dotted path."""
    module_path, cls_name = pipeline_path.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)()


def _build_snapshot_dataset(
    cfg: dict,
    fleet_df: pd.DataFrame,
    telemetry_dir: Path,
    manifest_df: pd.DataFrame | None,
    snapshot_interval_days: int,
) -> pd.DataFrame:
    """
    Call the feature pipeline at multiple snapshot points for each VIN,
    producing a dataset large enough for TimeSeriesSplit training.

    For a synthetic dataset with 50 VINs and 180 days:
      180 / 14 = ~12 snapshots → 50 × 12 = ~600 rows
    """
    pipe = _load_pipeline(cfg["pipeline"])

    # Determine snapshot dates from telemetry files
    all_rows: list[pd.DataFrame] = []

    for _, vrow in fleet_df.iterrows():
        vin = vrow["vin"]
        csv_path = telemetry_dir / f"{vin}_telemetry.csv"
        if not csv_path.exists():
            csv_path = telemetry_dir / f"telemetry_{vin}.csv"
        if not csv_path.exists():
            continue

        try:
            tel = pd.read_csv(csv_path, low_memory=False)
        except Exception as exc:
            log.warning("Failed reading %s: %s", csv_path, exc)
            continue

        # Rename timestamp column to expected internal name
        if "StartTime-TimeStamp" in tel.columns:
            tel = tel.rename(columns={"StartTime-TimeStamp": "timestamp"})

        # Convert unix epoch integers to datetime
        if "timestamp" in tel.columns:
            ts = tel["timestamp"]
            if pd.api.types.is_numeric_dtype(ts):
                tel["timestamp"] = pd.to_datetime(ts, unit="s", utc=True)
            else:
                tel["timestamp"] = pd.to_datetime(ts, utc=True)

        tel = tel.sort_values("timestamp")
        if tel.empty:
            continue

        # Downsample large files: keep every Nth row to cap at ~50K rows
        if len(tel) > 50_000:
            step = len(tel) // 50_000
            tel = tel.iloc[::step].copy()
            log.info("  Downsampled %s: %d rows (step=%d)", vin, len(tel), step)

        ts_start = tel["timestamp"].min()
        ts_end   = tel["timestamp"].max()
        lookback  = cfg["lookback"]

        # Walk through snapshot points
        snap_date = ts_start + pd.Timedelta(days=lookback)
        while snap_date <= ts_end:
            window = tel[tel["timestamp"] <= snap_date].copy()
            if len(window) < 10:
                snap_date += pd.Timedelta(days=snapshot_interval_days)
                continue

            try:
                row = pipe.compute(vin, window, label_df=manifest_df,
                                   vrow=vrow, t_ref=snap_date)
                if row is not None and not row.empty:
                    all_rows.append(row)
            except Exception as exc:
                log.debug("Pipeline error VIN %s @ %s: %s", vin, snap_date.date(), exc)

            snap_date += pd.Timedelta(days=snapshot_interval_days)

    if not all_rows:
        return pd.DataFrame()

    return pd.concat(all_rows, ignore_index=True)


# ── Main training loop ───────────────────────────────────────────────────────

def _run_leakage_check(
    name: str,
    data_dir: Path,
    feature_store_dir: Path,
    train_cutoff: str,
    val_cutoff:   str,
) -> None:
    """
    Attempt to build training data and run leakage checks.
    If service history is missing, logs a warning and skips (non-fatal).
    Exits with code 1 on LeakageError.
    """
    svc_path = data_dir / "service_history.csv"
    if not svc_path.exists():
        log.debug("Service history not found — skipping leakage check for %s", name)
        return

    build_dir = data_dir / "training_splits"
    try:
        builder = TrainingDataBuilder()
        splits  = builder.build(
            model_name           = name,
            service_history_path = svc_path,
            feature_store_dir    = feature_store_dir,
            output_path          = build_dir,
            train_cutoff         = train_cutoff,
            val_cutoff           = val_cutoff,
        )
        LeakageChecker().check(
            splits["train"], splits["val"], splits["test"], name
        )
        log.info("Leakage checks passed for %s", name)
    except LeakageError as exc:
        print(f"\n[LEAKAGE] {exc}")
        sys.exit(1)
    except Exception as exc:
        log.warning("Leakage check skipped for %s: %s", name, exc)


def train_all(
    data_dir: Path,
    experiment: str = "autopredict-v1",
    snapshot_interval_days: int = 14,
    models_to_run: list[str] | None = None,
    feature_store_dir: Path | None = None,
    train_cutoff: str = "2024-06-01",
    val_cutoff:   str = "2024-09-01",
    progress_callback: "callable | None" = None,
) -> pd.DataFrame:
    """
    Train all models and return a summary DataFrame.

    Returns a DataFrame with columns:
      model | metric | metric_value | training_rows | training_time_s | status
    """
    telemetry_dir = data_dir / "telemetry"
    if not telemetry_dir.exists():
        # try flat layout (CSVs directly in data_dir)
        telemetry_dir = data_dir

    fleet_path = data_dir / "fleet.csv"
    if not fleet_path.exists():
        log.error("fleet.csv not found in %s", data_dir)
        sys.exit(1)

    fleet_df   = pd.read_csv(fleet_path)
    log.info("Fleet: %d vehicles", len(fleet_df))

    manifest_path = data_dir / "failures_manifest.csv"
    manifest_df   = pd.read_csv(manifest_path) if manifest_path.exists() else None
    if manifest_df is not None:
        log.info("Failures manifest: %d rows", len(manifest_df))

    summary_rows: list[dict] = []

    total_models = len([c for c in _MODEL_CONFIGS if not models_to_run or c["name"] in models_to_run])

    for model_idx, cfg in enumerate(_MODEL_CONFIGS):
        name = cfg["name"]
        if models_to_run and name not in models_to_run:
            continue

        pct = int(10 + (model_idx / max(total_models, 1)) * 80)
        if progress_callback:
            progress_callback(pct, f"Training {name} ({model_idx+1}/{total_models})...")

        print(f"\n{'-'*60}")
        print(f"  Training: {name} ({model_idx+1}/{total_models})")
        print(f"{'-'*60}")

        t0 = time.perf_counter()

        # ── Leakage check (uses feature store offline data if available) ──
        fs_dir = feature_store_dir or (data_dir / "feature_store")
        _run_leakage_check(name, data_dir, fs_dir, train_cutoff, val_cutoff)

        # ── Build snapshot dataset ─────────────────────────────────────
        log.info("Building snapshot dataset for %s ...", name)
        features_df = _build_snapshot_dataset(
            cfg, fleet_df, telemetry_dir, manifest_df, snapshot_interval_days
        )
        n_rows = len(features_df)
        log.info("  Dataset: %d rows, %d columns", n_rows, features_df.shape[1] if n_rows else 0)

        if n_rows == 0:
            log.warning("  No training data — skipping %s", name)
            summary_rows.append({
                "model": name, "metric": cfg["metric_key"],
                "metric_value": None, "training_rows": 0,
                "training_time_s": 0, "status": "skipped (no data)",
            })
            continue

        # ── Train ─────────────────────────────────────────────────────
        try:
            mod     = importlib.import_module(cfg["module"])
            metrics = mod.train(features_df, experiment=experiment)
            elapsed = round(time.perf_counter() - t0, 1)

            metric_val = metrics.get(cfg["metric_key"])
            if metric_val is None:
                # pick any numeric metric
                for k, v in metrics.items():
                    if isinstance(v, float) and not (isinstance(v, float) and np.isnan(v)):
                        metric_val = round(v, 4)
                        break

            status = "skipped" if metrics.get("skipped") else "ok"
            summary_rows.append({
                "model":            name,
                "metric":           cfg["metric_key"],
                "metric_value":     round(float(metric_val), 4) if metric_val is not None else None,
                "training_rows":    n_rows,
                "training_time_s":  elapsed,
                "status":           status,
            })
            print(f"  Done in {elapsed}s  |  {cfg['metric_key']}={metric_val}")

        except Exception as exc:
            elapsed = round(time.perf_counter() - t0, 1)
            log.error("  Training failed for %s: %s", name, exc, exc_info=True)
            summary_rows.append({
                "model": name, "metric": cfg["metric_key"],
                "metric_value": None, "training_rows": n_rows,
                "training_time_s": elapsed, "status": f"error: {exc}",
            })

    # ── Train RUL survival models ───────────────────────────────────────────
    if not models_to_run or any(m in (models_to_run or []) for m in
                                 ["brake_wear", "engine_oil", "battery_12v", "tyre_wear"]):
        print(f"\n{'-'*60}")
        print(f"  Training: RUL survival models")
        print(f"{'-'*60}")
        t0_rul = time.perf_counter()
        try:
            from models.rul_models import train_rul_models
            combined_rows = []
            for cfg in _MODEL_CONFIGS:
                if cfg["name"] in ("brake_wear", "engine_oil", "battery_12v", "tyre_wear"):
                    chunk = _build_snapshot_dataset(
                        cfg, fleet_df, telemetry_dir, manifest_df, snapshot_interval_days
                    )
                    if not chunk.empty:
                        combined_rows.append(chunk)
            if combined_rows:
                rul_df = pd.concat(combined_rows, ignore_index=True)
                if "days_to_failure" not in rul_df.columns:
                    rul_df["days_to_failure"] = np.random.uniform(30, 365, len(rul_df))
                if "label_binary" not in rul_df.columns:
                    rul_df["label_binary"] = (rul_df["days_to_failure"] < 90).astype(int)
                rul_results = train_rul_models(rul_df)
                elapsed_rul = round(time.perf_counter() - t0_rul, 1)
                for name, ok in rul_results.items():
                    summary_rows.append({
                        "model": name, "metric": "survival",
                        "metric_value": None, "training_rows": len(rul_df),
                        "training_time_s": elapsed_rul, "status": "ok" if ok else "failed",
                    })
                print(f"  RUL models done in {elapsed_rul}s: {rul_results}")
            else:
                log.warning("  No data for RUL models — skipping")
        except Exception as exc:
            log.error("  RUL training failed: %s", exc, exc_info=True)

    summary = pd.DataFrame(summary_rows)
    return summary


# ── CLI ──────────────────────────────────────────────────────────────────────

def _print_table(df: pd.DataFrame) -> None:
    if df.empty:
        print("No models trained.")
        return
    cols = ["model", "metric", "metric_value", "training_rows", "training_time_s", "status"]
    cols = [c for c in cols if c in df.columns]
    print("\n" + "=" * 72)
    print("  AutoPredict Training Summary")
    print("=" * 72)
    print(df[cols].to_string(index=False))
    print("=" * 72)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train all AutoPredict ML models")
    ap.add_argument("--data-dir",         default="data/synthetic", help="Synthetic data directory")
    ap.add_argument("--experiment",       default="autopredict-v1", help="MLflow experiment name")
    ap.add_argument("--snapshot-interval-days", type=int, default=14, help="Days between training snapshots")
    ap.add_argument("--models",           nargs="*", help="Subset of model names to train")
    ap.add_argument("--output",           default="models/saved/train_summary.csv")
    ap.add_argument("--feature-store-dir", default=None, help="Feature store offline root")
    ap.add_argument("--train-cutoff",     default="2024-06-01")
    ap.add_argument("--val-cutoff",       default="2024-09-01")
    args = ap.parse_args()

    summary = train_all(
        data_dir               = Path(args.data_dir),
        experiment             = args.experiment,
        snapshot_interval_days = args.snapshot_interval_days,
        models_to_run          = args.models,
        feature_store_dir      = Path(args.feature_store_dir) if args.feature_store_dir else None,
        train_cutoff           = args.train_cutoff,
        val_cutoff             = args.val_cutoff,
    )

    _print_table(summary)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    log.info("Summary saved to %s", out)


if __name__ == "__main__":
    main()
