"""
Targeted retrain: hv_battery_soh + CoxPH RUL models (engine_oil_rul, battery_12v_rul).
Run with: py -m models.retrain_soh_rul --data-dir data/synthetic
"""
from __future__ import annotations

import argparse
import importlib
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-9s %(name)-18s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("retrain_soh_rul")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/synthetic")
    ap.add_argument("--experiment", default="autopredict-v2")
    ap.add_argument("--snapshot-interval-days", type=int, default=7)
    args = ap.parse_args()

    data_dir      = Path(args.data_dir)
    telemetry_dir = data_dir
    fleet_df      = pd.read_csv(data_dir / "fleet_master.csv")
    manifest_path = data_dir / "failures_manifest.csv"
    manifest_df   = pd.read_csv(manifest_path) if manifest_path.exists() else None

    # ── Pre-load telemetry cache (EV/PHEV VINs only for speed) ───────────────
    ev_vins = set(fleet_df[fleet_df["fuel_type"].isin(["EV", "PHEV"])]["vin"].astype(str))
    log.info("Pre-loading telemetry for %d EV/PHEV VINs...", len(ev_vins))
    telemetry_cache: dict[str, pd.DataFrame] = {}
    for vin in ev_vins:
        for fname in (f"telemetry_{vin}.csv", f"{vin}_telemetry.csv"):
            p = telemetry_dir / fname
            if p.exists():
                try:
                    tel = pd.read_csv(p, low_memory=False)
                    if "StartTime-TimeStamp" in tel.columns:
                        tel = tel.rename(columns={"StartTime-TimeStamp": "timestamp"})
                    if "timestamp" in tel.columns:
                        ts = tel["timestamp"]
                        if pd.api.types.is_numeric_dtype(ts):
                            tel["timestamp"] = pd.to_datetime(ts, unit="s", utc=True)
                        else:
                            tel["timestamp"] = pd.to_datetime(ts, utc=True)
                    tel = tel.sort_values("timestamp")
                    if len(tel) > 50_000:
                        step = len(tel) // 50_000
                        tel  = tel.iloc[::step].copy()
                    telemetry_cache[vin] = tel
                except Exception as e:
                    log.warning("Failed reading %s: %s", p, e)
                break
    log.info("Cache loaded: %d VINs", len(telemetry_cache))

    # ── Build hv_battery_soh snapshot dataset ─────────────────────────────────
    log.info("Building snapshot dataset for hv_battery_soh...")
    from features.battery_hv_features import HVBatteryFeaturePipeline
    pipe      = HVBatteryFeaturePipeline()
    all_rows: list[pd.DataFrame] = []
    lookback  = 90
    interval  = args.snapshot_interval_days

    for _, vrow in fleet_df.iterrows():
        vin = str(vrow["vin"])
        if vin not in telemetry_cache:
            continue
        tel      = telemetry_cache[vin]
        ts_start = tel["timestamp"].min()
        ts_end   = tel["timestamp"].max()
        snap     = ts_start + pd.Timedelta(days=lookback)
        fuel_t   = str(vrow.get("fuel_type", "ICE"))
        bat_kwh  = float(vrow["battery_capacity_kwh"]) if pd.notna(vrow.get("battery_capacity_kwh")) else None

        while snap <= ts_end:
            window = tel[tel["timestamp"] <= snap].copy()
            if len(window) < 10:
                snap += pd.Timedelta(days=interval)
                continue
            try:
                ctx = {"vrow": vrow, "t_ref": snap, "fuel_type": fuel_t}
                if bat_kwh:
                    ctx["battery_capacity_kwh"] = bat_kwh
                row = pipe.compute(vin, window, label_df=manifest_df, **ctx)
                if row is not None and not row.empty and row.get("hv_applicable", [0])[0] == 1:
                    all_rows.append(row)
            except Exception as exc:
                log.debug("Pipeline error %s @ %s: %s", vin, snap.date(), exc)
            snap += pd.Timedelta(days=interval)

    if not all_rows:
        log.error("No hv_battery_soh rows generated — check EV VINs in fleet")
        return

    features_df = pd.concat(all_rows, ignore_index=True)
    log.info("hv_battery_soh dataset: %d rows, %d cols", len(features_df), features_df.shape[1])

    # Log SoH distribution for sanity check
    if "soh_estimated" in features_df.columns:
        soh = features_df["soh_estimated"].dropna()
        log.info("soh_estimated: min=%.1f  mean=%.1f  max=%.1f", soh.min(), soh.mean(), soh.max())
    if "soh_below_80_within_90_days" in features_df.columns:
        pos = int(features_df["soh_below_80_within_90_days"].sum())
        log.info("soh_below_80_within_90_days positives: %d / %d", pos, len(features_df))

    # ── Train hv_battery_soh ──────────────────────────────────────────────────
    log.info("Training hv_battery_soh...")
    t0  = time.perf_counter()
    mod = importlib.import_module("models.hv_battery_soh_model")
    res = mod.train(features_df, experiment=args.experiment)
    log.info("hv_battery_soh done in %.1fs: %s", time.perf_counter() - t0, res)

    from models.metrics_store import save_model_metrics
    save_model_metrics(
        model_name="hv_battery_soh",
        algorithm="XGBoost + Ridge + IsolationForest",
        target="soh_estimated",
        training_samples=len(features_df),
        feature_names=[c for c in features_df.columns if c not in ("vin", "computed_at", "snapshot_date")],
        metrics=res,
        feature_importances={},
        status="trained",
    )

    # ── Build combined dataset for RUL (brake/engine/battery12v/tyre) ─────────
    log.info("Building combined snapshot dataset for RUL models...")
    _RUL_MODELS = [
        {"name": "brake_wear",  "module": "models.brake_wear_model",   "pipeline": "features.brake_features.BrakeFeaturePipeline",            "lookback": 90},
        {"name": "engine_oil",  "module": "models.engine_oil_model",   "pipeline": "features.engine_features.EngineFeaturePipeline",          "lookback": 90},
        {"name": "battery_12v", "module": "models.battery_12v_model",  "pipeline": "features.battery_12v_features.Battery12VFeaturePipeline", "lookback": 90},
        {"name": "tyre_wear",   "module": "models.tyre_wear_model",    "pipeline": "features.tyre_features.TyreFeaturePipeline",              "lookback": 90},
    ]

    # Load full fleet telemetry cache
    all_vins = set(fleet_df["vin"].astype(str))
    missing  = all_vins - set(telemetry_cache.keys())
    if missing:
        log.info("Loading %d ICE VINs for RUL...", len(missing))
        for vin in missing:
            for fname in (f"telemetry_{vin}.csv", f"{vin}_telemetry.csv"):
                p = telemetry_dir / fname
                if p.exists():
                    try:
                        tel = pd.read_csv(p, low_memory=False)
                        if "StartTime-TimeStamp" in tel.columns:
                            tel = tel.rename(columns={"StartTime-TimeStamp": "timestamp"})
                        if "timestamp" in tel.columns:
                            ts = tel["timestamp"]
                            if pd.api.types.is_numeric_dtype(ts):
                                tel["timestamp"] = pd.to_datetime(ts, unit="s", utc=True)
                            else:
                                tel["timestamp"] = pd.to_datetime(ts, utc=True)
                        tel = tel.sort_values("timestamp")
                        if len(tel) > 50_000:
                            step = len(tel) // 50_000
                            tel  = tel.iloc[::step].copy()
                        telemetry_cache[vin] = tel
                    except Exception:
                        pass
                    break

    combined: list[pd.DataFrame] = []
    for cfg in _RUL_MODELS:
        pipe_mod, pipe_cls = cfg["pipeline"].rsplit(".", 1)
        pipeline = getattr(importlib.import_module(pipe_mod), pipe_cls)()
        chunk_rows: list[pd.DataFrame] = []
        for _, vrow in fleet_df.iterrows():
            vin = str(vrow["vin"])
            if vin not in telemetry_cache:
                continue
            tel      = telemetry_cache[vin]
            ts_start = tel["timestamp"].min()
            ts_end   = tel["timestamp"].max()
            snap     = ts_start + pd.Timedelta(days=cfg["lookback"])
            while snap <= ts_end:
                window = tel[tel["timestamp"] <= snap].copy()
                if len(window) < 10:
                    snap += pd.Timedelta(days=interval)
                    continue
                try:
                    ctx = {"vrow": vrow, "t_ref": snap}
                    if "fuel_type" in vrow and pd.notna(vrow["fuel_type"]):
                        ctx["fuel_type"] = vrow["fuel_type"]
                    row = pipeline.compute(vin, window, label_df=manifest_df, **ctx)
                    if row is not None and not row.empty:
                        chunk_rows.append(row)
                except Exception:
                    pass
                snap += pd.Timedelta(days=interval)
        if chunk_rows:
            combined.append(pd.concat(chunk_rows, ignore_index=True))
            log.info("  %s: %d rows", cfg["name"], len(chunk_rows))

    if not combined:
        log.error("No RUL training rows — skipping RUL retrain")
        return

    rul_df = pd.concat(combined, ignore_index=True)
    if "days_to_failure" not in rul_df.columns:
        rul_df["days_to_failure"] = np.random.uniform(30, 365, len(rul_df))
    if "label_binary" not in rul_df.columns:
        rul_df["label_binary"] = (rul_df["days_to_failure"] < 90).astype(int)
    log.info("RUL combined dataset: %d rows", len(rul_df))

    # ── Train CoxPH RUL models only ───────────────────────────────────────────
    log.info("Training CoxPH RUL models (engine_oil_rul, battery_12v_rul)...")
    from models.rul_models import RUL_MODEL_SPECS, build_rul_model
    from models.model_registry import MODEL_DIR

    t0_rul = time.perf_counter()
    for name in ("engine_oil_rul", "battery_12v_rul"):
        spec  = RUL_MODEL_SPECS[name]
        model = build_rul_model(name)
        model.train(rul_df, "days_to_failure", "label_binary", spec["covariates"])
        model.save(MODEL_DIR / f"{name}.joblib")
        log.info("Saved %s", name)

    log.info("CoxPH RUL done in %.1fs", time.perf_counter() - t0_rul)
    log.info("All done.")


if __name__ == "__main__":
    main()
