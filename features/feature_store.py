"""
Feature Store — online (Redis) and offline (Parquet) dual-layer store.

Online  Redis:   key="features:{vin}:{group}"  value=JSON  TTL=25 hours
Offline Parquet: data/feature_store/{group}/date={as_of_date}/features.parquet
"""
from __future__ import annotations

import json
import logging
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_REDIS_URL      = os.getenv("REDIS_URL",      "redis://localhost:6379/0")
_OFFLINE_ROOT   = Path(os.getenv("FEATURE_STORE_DIR", "data/feature_store"))

# ── Feature group schemas ─────────────────────────────────────────────────────

FEATURE_GROUPS: dict[str, list[str]] = {
    "brake": [
        "brake_stress_cumulative", "harsh_brake_rate_7d", "harsh_brake_rate_30d",
        "high_speed_stop_count_30d", "avg_brake_intensity_7d",
        "regen_fraction", "effective_brake_km",
        "abs_activation_rate_30d", "esc_activation_rate_30d",
        "brake_pedal_travel_proxy", "km_since_last_brake_service",
        "days_since_last_brake_service", "brake_thermal_stress",
        "wot_event_count_30d", "accel_smoothness_score",
    ],
    "engine": [
        "oil_degradation_index", "km_since_oil_change", "days_since_oil_change",
        "cold_start_count_30d", "short_trip_fraction_30d",
        "coolant_overtemp_count_30d", "avg_coolant_temp_7d",
        "high_rpm_stress_index", "rpm_to_speed_ratio_anomaly",
        "fuel_consumption_deviation_pct", "idle_hours_30d",
        "engine_load_proxy", "towing_load_indicator",
        "mil_recurrence_flag", "gear_efficiency_score",
    ],
    "battery_hv": [
        "soh_estimated", "soh_trend_slope_90d",
        "cell_voltage_spread", "cell_spread_trend_30d", "cell_spread_p95_30d",
        "cell_temp_delta", "cell_temp_delta_trend_30d",
        "dc_charge_fraction_30d", "charge_duration_vs_expected",
        "isolation_resistance_min_30d", "range_per_kwh_30d_trend",
        "thermal_fault_count_30d", "busbar_resistance_proxy",
        "charge_c_rate_avg_30d",
    ],
    "battery_12v": [
        "resting_voltage_7d_avg", "resting_voltage_trend_14d",
        "overnight_drop_avg_7d", "cranking_voltage_dip_avg",
        "voltage_recovery_rate", "parasitic_drain_rate",
        "battery_12v_health_score", "battery_age_years",
        "light_on_engine_off_events_7d", "cold_weather_risk_multiplier",
    ],
    "tyre": [
        "pressure_fl_7d_avg", "pressure_fr_7d_avg",
        "pressure_rl_7d_avg", "pressure_rr_7d_avg",
        "pressure_drop_rate_fl", "pressure_drop_rate_fr",
        "pressure_drop_rate_rl", "pressure_drop_rate_rr",
        "axle_imbalance_front", "axle_imbalance_rear",
        "tpms_status", "tyre_stress_cumulative",
        "km_since_last_tyre_service", "effective_tyre_km", "uneven_wear_indicator",
    ],
    "driver": [
        "composite_drive_score", "smooth_accel_score", "smooth_brake_score",
        "gear_efficiency_score", "speed_compliance_score",
        "fuel_efficiency_score", "cornering_score", "idle_score",
        "weekly_score_trend", "peer_percentile",
        "fuel_consumption_deviation_pct", "idle_fuel_waste_L_30d",
        "hvac_load_factor", "upshift_rpm_avg", "engine_braking_ratio",
    ],
    "vehicle_state": [
        "current_odometer_km", "fuel_level_pct", "soc_pct",
        "is_charging", "power_mode", "last_seen_timestamp",
        "is_ev", "battery_capacity_kwh",
    ],
}

# Model → feature groups consumed by that model's prediction vector
_MODEL_FEATURE_GROUPS: dict[str, list[str]] = {
    "brake_wear":     ["brake", "vehicle_state"],
    "engine_oil":     ["engine", "vehicle_state"],
    "hv_battery_soh": ["battery_hv", "vehicle_state"],
    "battery_12v":    ["battery_12v", "vehicle_state"],
    "tyre_wear":      ["tyre", "vehicle_state"],
    "driver_score":   ["driver", "vehicle_state"],
    "fuel_anomaly":   ["engine", "driver", "vehicle_state"],
}


class FeatureStore:

    def __init__(
        self,
        redis_url: str = _REDIS_URL,
        offline_root: Path | str = _OFFLINE_ROOT,
    ) -> None:
        self._redis_url   = redis_url
        self._offline_dir = Path(offline_root)
        self._redis: Any  = None

    # ── Redis connection ────────────────────────────────────────────────────

    def _get_redis(self):
        if self._redis is None:
            import redis
            self._redis = redis.Redis.from_url(self._redis_url, decode_responses=True)
        return self._redis

    # ── Online layer ────────────────────────────────────────────────────────

    def get_online(self, vin: str, group: str) -> dict | None:
        key = f"features:{vin}:{group}"
        try:
            val = self._get_redis().get(key)
            return json.loads(val) if val else None
        except Exception as exc:
            log.warning("Redis GET %s failed: %s", key, exc)
            return None

    def set_online(
        self, vin: str, group: str, features_dict: dict, ttl_hours: int = 25
    ) -> None:
        key = f"features:{vin}:{group}"
        try:
            self._get_redis().set(
                key,
                json.dumps(features_dict, default=str),
                ex=ttl_hours * 3600,
            )
        except Exception as exc:
            log.warning("Redis SET %s failed: %s", key, exc)

    # ── Offline layer ───────────────────────────────────────────────────────

    def get_offline(self, vin: str, group: str, as_of_date: date) -> dict | None:
        group_dir = self._offline_dir / group
        if not group_dir.exists():
            return None

        target    = as_of_date.isoformat()
        date_dirs = sorted(group_dir.glob("date=*"))
        available = [d.name.removeprefix("date=") for d in date_dirs]
        eligible  = [d for d in available if d <= target]
        if not eligible:
            return None

        best_date    = max(eligible)
        parquet_path = group_dir / f"date={best_date}" / "features.parquet"
        if not parquet_path.exists():
            return None

        try:
            df       = pd.read_parquet(parquet_path)
            vin_rows = df[df["vin"] == vin]
            if vin_rows.empty:
                return None
            return vin_rows.iloc[-1].to_dict()
        except Exception as exc:
            log.error("Parquet read failed %s: %s", parquet_path, exc)
            return None

    def write_offline(
        self, vin: str, group: str, features_dict: dict, record_date: date
    ) -> None:
        out_dir = self._offline_dir / group / f"date={record_date.isoformat()}"
        out_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = out_dir / "features.parquet"

        row    = {"vin": vin, "record_date": record_date.isoformat(), **features_dict}
        new_df = pd.DataFrame([row])

        try:
            if parquet_path.exists():
                existing = pd.read_parquet(parquet_path)
                existing = existing[existing["vin"] != vin]
                combined = pd.concat([existing, new_df], ignore_index=True)
            else:
                combined = new_df
            combined.to_parquet(parquet_path, index=False)
        except Exception as exc:
            log.error("Parquet write failed %s: %s", parquet_path, exc)

    # ── Model feature vector ────────────────────────────────────────────────

    def get_feature_vector(self, vin: str, model_name: str) -> np.ndarray:
        """Return a float32 vector with all features for *model_name*.
        Missing features are 0.0 with a warning log.
        """
        groups = _MODEL_FEATURE_GROUPS.get(model_name, [])
        vector: list[float] = []

        for group in groups:
            features_dict = self.get_online(vin, group) or {}
            for col in FEATURE_GROUPS.get(group, []):
                val = features_dict.get(col, None)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    log.warning(
                        "Feature %s.%s missing for VIN %s → 0.0", group, col, vin
                    )
                    val = 0.0
                try:
                    vector.append(float(val))
                except (TypeError, ValueError):
                    log.warning("Non-numeric feature %s.%s=%r -> 0.0", group, col, val)
                    vector.append(0.0)

        return np.array(vector, dtype=np.float32)
