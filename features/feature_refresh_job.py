"""
Feature refresh job — computes all feature groups for one or all VINs and
writes results to both the online (Redis) and offline (Parquet) feature store.

CLI:
    python -m features.feature_refresh_job --mode all   --lookback 90
    python -m features.feature_refresh_job --mode single --vin <VIN>
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_DATA_DIR   = Path(os.getenv("DATA_DIR",     "data/synthetic"))
_PG_URL     = os.getenv("POSTGRES_URL",      "")

# ── Lazy pipeline imports to avoid heavy deps at import time ─────────────────

def _load_pipelines():
    from features.brake_features            import BrakeFeaturePipeline
    from features.engine_features           import EngineFeaturePipeline
    from features.battery_hv_features       import HVBatteryFeaturePipeline
    from features.battery_12v_features      import Battery12VFeaturePipeline
    from features.tyre_features             import TyreFeaturePipeline
    from features.driver_behaviour_features import DriverBehaviourFeaturePipeline
    return {
        "brake":       BrakeFeaturePipeline(),
        "engine":      EngineFeaturePipeline(),
        "battery_hv":  HVBatteryFeaturePipeline(),
        "battery_12v": Battery12VFeaturePipeline(),
        "tyre":        TyreFeaturePipeline(),
        "driver":      DriverBehaviourFeaturePipeline(),
    }


class FeatureRefreshJob:
    """Celery-compatible job that refreshes feature groups for one or all VINs."""

    def refresh_all_vins(self, lookback_days: int = 90) -> dict:
        fleet_path = next(
            (_DATA_DIR / n for n in ("fleet.csv", "fleet_master.csv") if (_DATA_DIR / n).exists()),
            None,
        )
        if fleet_path is None:
            log.error("fleet.csv / fleet_master.csv not found in %s", _DATA_DIR)
            return {"total": 0, "failed": 0, "elapsed_s": 0}

        fleet_df = pd.read_csv(fleet_path)
        vins     = fleet_df["vin"].tolist()
        log.info("Refreshing features for %d VINs", len(vins))

        t0     = time.perf_counter()
        failed = 0
        for vin in vins:
            try:
                self.refresh_single_vin(vin, lookback_days=lookback_days)
            except Exception as exc:
                log.error("VIN %s refresh failed: %s", vin, exc)
                failed += 1

        elapsed = round(time.perf_counter() - t0, 1)
        log.info(
            "Refresh complete: total=%d  failed=%d  elapsed=%.1fs",
            len(vins), failed, elapsed,
        )
        return {"total": len(vins), "failed": failed, "elapsed_s": elapsed}

    def refresh_single_vin(self, vin: str, lookback_days: int = 90) -> None:
        from features.feature_store import FeatureStore

        store    = FeatureStore()
        pipelines = _load_pipelines()
        today    = date.today()

        # 1. Load telemetry ─────────────────────────────────────────────────
        df = self._load_telemetry(vin, lookback_days)
        if df.empty:
            log.warning("No telemetry data for VIN %s — skipping feature refresh", vin)
            return

        # 2. Load auxiliary data ────────────────────────────────────────────
        service_history = self._load_service_history(vin)
        dtc_events      = self._load_dtc_events(vin)
        label_df        = self._load_failures_manifest()
        fleet_row       = self._get_fleet_row(vin)

        ctx: dict[str, Any] = {
            "fuel_type":            fleet_row.get("fuel_type", "ICE"),
            "battery_capacity_kwh": fleet_row.get("battery_capacity_kwh"),
            "manufacture_year":     int(fleet_row.get("manufacture_year", 2022)),
        }

        # Find latest oil / brake / tyre service odo from service history
        if not service_history.empty and "DescriptionOne" in service_history.columns:
            oil_svc = service_history[
                service_history["DescriptionOne"].str.contains("ENGINE OIL|OIL CHANGE", case=False, na=False)
            ]
            ctx["last_oil_change_odo"] = float(oil_svc["Mileage"].max()) if len(oil_svc) > 0 else None

        # 3. Run all 6 pipelines ────────────────────────────────────────────
        group_map = {
            "brake":       "brake",
            "engine":      "engine",
            "battery_hv":  "battery_hv",
            "battery_12v": "battery_12v",
            "tyre":        "tyre",
            "driver":      "driver",
        }

        for pipe_key, group_name in group_map.items():
            pipe = pipelines[pipe_key]
            try:
                result_df = pipe.compute(vin, df.copy(), label_df=label_df, **ctx)
                if result_df is None or result_df.empty:
                    log.warning("Pipeline %s returned empty for VIN %s", pipe_key, vin)
                    continue
                features = result_df.iloc[0].to_dict()
                features = {k: (None if (isinstance(v, float) and np.isnan(v)) else v)
                            for k, v in features.items()}

                # 4. Write to feature store ─────────────────────────────────
                store.set_online(vin, group_name, features)
                store.write_offline(vin, group_name, features, today)
                log.debug("Wrote features for VIN %s group %s", vin, group_name)
            except Exception as exc:
                log.error("Pipeline %s failed for VIN %s: %s", pipe_key, vin, exc)

        # Build vehicle_state group
        self._write_vehicle_state(vin, df, fleet_row, store, today)

        # 5. Update PostgreSQL daily health ────────────────────────────────
        self._update_pg_health(vin)

    # ── Data loaders ────────────────────────────────────────────────────────

    def _load_telemetry(self, vin: str, lookback_days: int) -> pd.DataFrame:
        # Try InfluxDB first
        try:
            from features.base_pipeline import FeaturePipeline
            from influxdb_client import InfluxDBClient
            url   = os.getenv("INFLUXDB_URL",   "http://localhost:8086")
            token = os.getenv("INFLUXDB_TOKEN", "autopredict-dev-token")
            org   = os.getenv("INFLUXDB_ORG",   "autopredict")
            client = InfluxDBClient(url=url, token=token, org=org)
            flux = f"""
from(bucket: "tbox_standard")
  |> range(start: -{lookback_days}d)
  |> filter(fn: (r) => r["vin"] == "{vin}")
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> sort(columns: ["_time"])
"""
            tables = client.query_api().query_data_frame(flux)
            client.close()
            df = pd.concat(tables, ignore_index=True) if isinstance(tables, list) else tables
            if not df.empty:
                log.info("Loaded %d rows from InfluxDB for VIN %s", len(df), vin)
                return df
        except Exception as exc:
            log.debug("InfluxDB unavailable for VIN %s: %s", vin, exc)

        # CSV fallback — try several naming conventions
        for name in [f"{vin}_telemetry.csv", f"telemetry_{vin}.csv", "telemetry_combined.csv"]:
            p = _DATA_DIR / name
            if p.exists():
                df = pd.read_csv(p, low_memory=False)
                if "vin" in df.columns:
                    df = df[df["vin"] == vin]
                if not df.empty:
                    log.info("Loaded %d rows from CSV %s for VIN %s", len(df), p.name, vin)
                    return df

        log.warning("No telemetry found for VIN %s", vin)
        return pd.DataFrame()

    def _load_service_history(self, vin: str) -> pd.DataFrame:
        for name in ["service_history.csv", "service_records.csv"]:
            p = _DATA_DIR / name
            if p.exists():
                df = pd.read_csv(p)
                if "vin" in df.columns:
                    return df[df["vin"] == vin]
                return df
        return pd.DataFrame()

    def _load_dtc_events(self, vin: str) -> pd.DataFrame:
        for name in ["dtc_events.csv", "dtc_combined.csv"]:
            p = _DATA_DIR / name
            if p.exists():
                df = pd.read_csv(p)
                if "vin" in df.columns:
                    return df[df["vin"] == vin]
        return pd.DataFrame()

    def _load_failures_manifest(self) -> pd.DataFrame | None:
        p = _DATA_DIR / "failures_manifest.csv"
        return pd.read_csv(p) if p.exists() else None

    def _get_fleet_row(self, vin: str) -> dict:
        p = next(
            (_DATA_DIR / n for n in ("fleet.csv", "fleet_master.csv") if (_DATA_DIR / n).exists()),
            None,
        )
        if p is None:
            return {}
        df = pd.read_csv(p)
        rows = df[df["vin"] == vin]
        return rows.iloc[0].to_dict() if not rows.empty else {}

    def _write_vehicle_state(
        self, vin: str, df: pd.DataFrame, fleet_row: dict,
        store, today: date,
    ) -> None:
        state: dict[str, Any] = {
            "last_seen_timestamp": str(df["timestamp"].max()) if "timestamp" in df.columns else None,
            "is_ev":               int(str(fleet_row.get("fuel_type", "ICE")).upper() == "EV"),
            "battery_capacity_kwh": fleet_row.get("battery_capacity_kwh"),
        }
        # Snapshot latest signal values
        for col, key, scale in [
            ("odometer",   "current_odometer_km", 1.0),
            ("fuel_level", "fuel_level_pct",       1.0),
            ("soc",        "soc_pct",              0.1),
            ("sys_pwr_mod","power_mode",            1.0),
        ]:
            if col in df.columns:
                state[key] = float(df[col].iloc[-1]) * scale
        store.set_online(vin, "vehicle_state", state)
        store.write_offline(vin, "vehicle_state", state, today)

    def _update_pg_health(self, vin: str) -> None:
        if not _PG_URL:
            return
        try:
            import sqlalchemy as sa
            engine = sa.create_engine(_PG_URL)
            with engine.connect() as conn:
                conn.execute(
                    sa.text(
                        "INSERT INTO vehicle_daily_health (vin, health_date, updated_at) "
                        "VALUES (:vin, :d, :ts) "
                        "ON CONFLICT (vin, health_date) DO UPDATE SET updated_at=:ts"
                    ),
                    {"vin": vin, "d": date.today(), "ts": datetime.now(timezone.utc)},
                )
                conn.commit()
        except Exception as exc:
            log.debug("PG health update skipped for VIN %s: %s", vin, exc)


# ── Celery task wrapper ───────────────────────────────────────────────────────

try:
    from celery import shared_task

    @shared_task(name="features.feature_refresh_job.refresh_single_vin_task")
    def refresh_single_vin_task(vin: str, lookback_days: int = 90) -> None:
        FeatureRefreshJob().refresh_single_vin(vin, lookback_days=lookback_days)

    @shared_task(name="features.feature_refresh_job.refresh_all_vins_task")
    def refresh_all_vins_task(lookback_days: int = 90) -> dict:
        return FeatureRefreshJob().refresh_all_vins(lookback_days=lookback_days)

except ImportError:
    def refresh_single_vin_task(vin: str, lookback_days: int = 90) -> None:  # type: ignore[misc]
        FeatureRefreshJob().refresh_single_vin(vin, lookback_days=lookback_days)

    def refresh_all_vins_task(lookback_days: int = 90) -> dict:  # type: ignore[misc]
        return FeatureRefreshJob().refresh_all_vins(lookback_days=lookback_days)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description="AutoPredict feature refresh job")
    ap.add_argument("--mode",     choices=["all", "single"], required=True)
    ap.add_argument("--vin",      default=None, help="VIN (required for --mode single)")
    ap.add_argument("--lookback", type=int, default=90, help="Lookback days")
    args = ap.parse_args()

    job = FeatureRefreshJob()
    if args.mode == "single":
        if not args.vin:
            ap.error("--vin is required for --mode single")
        job.refresh_single_vin(args.vin, lookback_days=args.lookback)
        print(f"Refresh complete for VIN {args.vin}")
    else:
        result = job.refresh_all_vins(lookback_days=args.lookback)
        print(f"Refresh complete: {result}")


if __name__ == "__main__":
    main()
