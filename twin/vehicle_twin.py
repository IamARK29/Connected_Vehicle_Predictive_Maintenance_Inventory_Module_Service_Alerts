"""
Vehicle Digital Twin — single Redis-backed representation of each vehicle.

Redis key: "twin:{vin}"  stored as a single JSON string.

VehicleTwin holds identity, live state, degradation, predictions,
alerts, DTCs, driver profile, and service summary.

TwinManager provides typed CRUD and domain-specific update methods.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

log = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
TWIN_TTL_SECONDS = 60 * 60 * 24 * 30  # 30 days


@dataclass
class VehicleTwin:
    # ── IDENTITY (set from fleet_master.csv, never changes) ──────────────────
    vin: str = ""
    model_code: str = ""
    model_name: str = ""
    fuel_type: str = ""
    manufacture_year: int = 0
    dealer_code: str = ""
    dealer_city: str = ""
    color: str = ""
    battery_capacity_kwh: float = 0.0
    rated_range_km: float = 0.0
    warranty_expires_date: str = ""

    # ── LIVE STATE (updated on every standard-tier telemetry event) ───────────
    last_seen_timestamp: str = ""
    power_mode: int = 0
    speed_kph: float = 0.0
    odometer_km: float = 0.0
    fuel_level_pct: float = 0.0
    soc_pct: float = 0.0
    battery_12v_v: float = 0.0
    coolant_temp_c: float = 0.0
    outside_temp_c: float = 25.0
    ac_is_on: bool = False
    is_charging: bool = False
    gps_lat: float = 0.0
    gps_lon: float = 0.0
    tyre_pressure_kpa: dict = field(default_factory=lambda: {"fl": 0.0, "fr": 0.0, "rl": 0.0, "rr": 0.0})

    # ── DEGRADATION STATE (updated by daily feature refresh) ─────────────────
    brake_life_pct_remaining: float = 100.0
    oil_degradation_index: float = 0.0
    hv_battery_soh_pct: float = 0.0
    battery_12v_health_score: float = 100.0
    tyre_health: dict = field(default_factory=lambda: {"fl": 100.0, "fr": 100.0, "rl": 100.0, "rr": 100.0})

    # ── PREDICTIONS (updated after each model run) ───────────────────────────
    failure_probs: dict = field(default_factory=dict)
    failure_stages: dict = field(default_factory=dict)
    rul_days: dict = field(default_factory=dict)
    next_predicted_service: dict = field(default_factory=dict)
    predicted_range_km: float = 0.0
    range_anxiety_flag: bool = False

    # ── ALERTS & DTC ─────────────────────────────────────────────────────────
    active_alerts: list = field(default_factory=list)
    active_dtcs: list = field(default_factory=list)

    # ── DRIVER PROFILE ───────────────────────────────────────────────────────
    driver_archetype: str = ""
    composite_drive_score: float = 0.0
    drive_score_7d_avg: float = 0.0
    km_per_day_30d_avg: float = 0.0
    primary_road_type: str = "mixed"

    # ── SERVICE SUMMARY ──────────────────────────────────────────────────────
    last_service_date: str = ""
    last_service_odometer_km: float = 0.0
    total_service_visits: int = 0
    warranty_active: bool = True

    # ── COMPUTED SCORE ───────────────────────────────────────────────────────
    health_summary_score: float = 100.0

    def to_dict(self) -> dict:
        return asdict(self)


class TwinManager:
    """Redis-backed CRUD and domain update methods for VehicleTwin."""

    def __init__(self) -> None:
        self._redis = None

    def _get_redis(self):
        if self._redis is None:
            try:
                import redis
                self._redis = redis.Redis.from_url(REDIS_URL, decode_responses=True)
            except Exception:
                self._redis = None
        return self._redis

    def _key(self, vin: str) -> str:
        return f"twin:{vin}"

    def get(self, vin: str) -> VehicleTwin | None:
        r = self._get_redis()
        if r is None:
            return self._get_fallback(vin)
        try:
            raw = r.get(self._key(vin))
            if raw is None:
                return self._get_fallback(vin)
            data = json.loads(raw)
            return _dict_to_twin(data)
        except Exception:
            return self._get_fallback(vin)

    def _get_fallback(self, vin: str) -> VehicleTwin | None:
        if vin in _IN_MEMORY_STORE:
            return _dict_to_twin(_IN_MEMORY_STORE[vin])
        return None

    def _save(self, vin: str, twin: VehicleTwin) -> None:
        data = twin.to_dict()
        r = self._get_redis()
        if r is not None:
            try:
                r.set(self._key(vin), json.dumps(data), ex=TWIN_TTL_SECONDS)
                return
            except Exception:
                pass
        _IN_MEMORY_STORE[vin] = data

    def get_all_vins(self) -> list[str]:
        r = self._get_redis()
        if r is not None:
            try:
                keys = r.keys("twin:*")
                return [k.replace("twin:", "", 1) for k in keys]
            except Exception:
                pass
        return list(_IN_MEMORY_STORE.keys())

    # ── Domain update methods ────────────────────────────────────────────────

    def update_from_telemetry(
        self,
        vin: str,
        channel_id: int,
        decoded_payload: dict[str, Any],
    ) -> None:
        twin = self.get(vin) or self._init_from_fleet_master(vin)

        if channel_id in (1, 2):
            if "gnssLat" in decoded_payload:
                twin.gps_lat = float(decoded_payload["gnssLat"])
            if "gnssLong" in decoded_payload:
                twin.gps_lon = float(decoded_payload["gnssLong"])

        elif channel_id == 3:
            if "vehSysPwrMod" in decoded_payload:
                twin.power_mode = int(decoded_payload["vehSysPwrMod"])
            if "vehSpeed" in decoded_payload:
                twin.speed_kph = float(decoded_payload["vehSpeed"])

        elif channel_id == 15:
            if "vehOdo" in decoded_payload:
                twin.odometer_km = float(decoded_payload["vehOdo"])
            if "vehFuelLev" in decoded_payload:
                twin.fuel_level_pct = float(decoded_payload["vehFuelLev"])
            if "vehBatt" in decoded_payload:
                raw_batt = decoded_payload["vehBatt"]
                twin.battery_12v_v = round(float(raw_batt) * 0.1, 1)
            if "vehCoolantTemp" in decoded_payload:
                twin.coolant_temp_c = float(decoded_payload["vehCoolantTemp"])

        elif channel_id in (19, 20, 21):
            if "vehBMSPackSOC" in decoded_payload:
                twin.soc_pct = float(decoded_payload["vehBMSPackSOC"])
            if "chargeRemainTime" in decoded_payload:
                twin.is_charging = float(decoded_payload["chargeRemainTime"]) > 0
            # Update range estimate whenever SoC changes (EV/PHEV only)
            if twin.fuel_type in ("EV", "PHEV", "BEV") and twin.soc_pct > 0:
                try:
                    from models.range_anxiety_model import RangeAnxietyPredictor
                    fs_features = {
                        "soh_estimated":           twin.hv_battery_soh_pct or 100.0,
                        "composite_drive_score":   twin.composite_drive_score or 70.0,
                        "km_per_day_30d_avg":      twin.km_per_day_30d_avg or 40.0,
                    }
                    fleet_row = {
                        "rated_range_km":      twin.rated_range_km,
                        "battery_capacity_kwh": twin.battery_capacity_kwh,
                    }
                    result = RangeAnxietyPredictor().predict(
                        vin=vin,
                        current_soc_pct=twin.soc_pct,
                        current_outside_temp_c=twin.outside_temp_c,
                        ac_is_on=twin.ac_is_on,
                        feature_store_features=fs_features,
                        fleet_row=fleet_row,
                    )
                    twin.predicted_range_km = float(result["predicted_range_km"])
                    twin.range_anxiety_flag = bool(result["range_anxiety_flag"])
                except Exception as exc:
                    log.debug("Range predictor failed for %s: %s", vin, exc)

        elif channel_id == 23:
            outside_temp = decoded_payload.get("vehOutsideTemp", 25.0)
            twin.outside_temp_c = float(outside_temp)
            if "vehAcStatus" in decoded_payload:
                twin.ac_is_on = bool(decoded_payload["vehAcStatus"])
            pressures = {}
            _TYRE_MAP = {
                "frontLeftTyrePressure": "fl",
                "frontRightTyrePressure": "fr",
                "rearLeftTyrePressure": "rl",
                "rearRightTyrePressure": "rr",
            }
            for signal, pos in _TYRE_MAP.items():
                if signal in decoded_payload and decoded_payload[signal] is not None:
                    raw_p = float(decoded_payload[signal])
                    pressures[pos] = _correct_tyre_pressure_for_temp(raw_p, float(outside_temp))
            if pressures:
                twin.tyre_pressure_kpa.update(pressures)

        twin.last_seen_timestamp = datetime.now(timezone.utc).isoformat()
        self._save(vin, twin)

    def update_predictions(
        self,
        vin: str,
        model_results: dict,
        stages: dict,
        rul: dict,
    ) -> None:
        twin = self.get(vin)
        if not twin:
            return
        twin.failure_probs = model_results
        twin.failure_stages = {k: int(v) for k, v in stages.items()}
        twin.rul_days = rul
        twin.next_predicted_service = self._compute_next_service(model_results, stages)
        twin.health_summary_score = self._compute_health_score(twin)
        self._save(vin, twin)

    def update_degradation(self, vin: str, features: dict) -> None:
        twin = self.get(vin)
        if not twin:
            return
        twin.brake_life_pct_remaining = max(0.0, 100.0 - features.get("brake_stress_cumulative", 0.0) / 50.0)
        twin.oil_degradation_index = features.get("oil_degradation_index", 0.0)
        twin.hv_battery_soh_pct = features.get("soh_estimated", 0.0)
        twin.battery_12v_health_score = features.get("battery_12v_health_score", 100.0)
        twin.composite_drive_score = features.get("composite_drive_score", 0.0)
        self._save(vin, twin)

    def update_alerts(self, vin: str, alerts: list[dict]) -> None:
        twin = self.get(vin)
        if not twin:
            return
        twin.active_alerts = [
            {
                "type": a.get("alert_type", a.get("type", "")),
                "severity": a.get("severity", ""),
                "triggered_at": a.get("triggered_at", ""),
                "explanation_text": a.get("explanation_text", ""),
                "top3_features": a.get("top3_features", []),
            }
            for a in alerts
        ]
        self._save(vin, twin)

    def update_dtcs(self, vin: str, dtcs: list[dict]) -> None:
        twin = self.get(vin)
        if not twin:
            return
        twin.active_dtcs = [
            {
                "dtc_code": d.get("dtc_code", ""),
                "serious_level": d.get("serious_level", ""),
                "confirmed": d.get("confirmed", False),
                "first_seen_date": d.get("first_seen_date", ""),
            }
            for d in dtcs
        ]
        self._save(vin, twin)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _init_from_fleet_master(self, vin: str) -> VehicleTwin:
        twin = VehicleTwin(vin=vin)
        try:
            import pandas as pd
            from pathlib import Path
            data_dir = os.getenv("DATA_DIR", "data/synthetic")
            for fname in ("fleet.csv", "fleet_master.csv"):
                csv = Path(data_dir) / fname
                if csv.exists():
                    df = pd.read_csv(csv)
                    match = df[df["vin"] == vin]
                    if not match.empty:
                        row = match.iloc[0]
                        twin.model_code = str(row.get("model_code", ""))
                        twin.model_name = str(row.get("model_name", ""))
                        twin.fuel_type = str(row.get("fuel_type", ""))
                        twin.manufacture_year = int(row.get("manufacture_year", 0))
                        twin.dealer_code = str(row.get("dealer_code", ""))
                        twin.dealer_city = str(row.get("dealer_city", ""))
                        twin.color = str(row.get("color", ""))
                        twin.battery_capacity_kwh = float(row.get("battery_capacity_kwh", 0.0))
                        twin.rated_range_km = float(row.get("rated_range_km", 0.0))
                        twin.odometer_km = float(row.get("initial_odometer", 0.0))
                        yr = twin.manufacture_year
                        twin.warranty_expires_date = f"{yr + 3}-12-31" if yr else ""
                        twin.warranty_active = twin.warranty_expires_date >= datetime.now(timezone.utc).strftime("%Y-%m-%d")
                        break
        except Exception as exc:
            log.debug("Fleet master lookup failed for %s: %s", vin, exc)
        return twin

    def _compute_next_service(self, model_results: dict, stages: dict) -> dict:
        from datetime import timedelta
        worst_type = ""
        worst_prob = 0.0
        for ft, prob in model_results.items():
            p = float(prob) if isinstance(prob, (int, float)) else 0.0
            if p > worst_prob:
                worst_prob = p
                worst_type = ft
        if not worst_type:
            return {}
        days_est = max(7, int((1.0 - worst_prob) * 180))
        est_date = (datetime.now(timezone.utc) + timedelta(days=days_est)).date().isoformat()
        return {"type": worst_type, "est_date": est_date, "est_km": 0}

    def _compute_health_score(self, twin: VehicleTwin) -> float:
        scores = [
            twin.brake_life_pct_remaining,
            100.0 - twin.oil_degradation_index,
            twin.battery_12v_health_score,
        ]
        if twin.fuel_type in ("EV", "PHEV", "BEV"):
            scores.append(twin.hv_battery_soh_pct)
        valid = [s for s in scores if s > 0]
        return round(min(valid), 1) if valid else 0.0


# ── In-memory fallback when Redis is unavailable ─────────────────────────────

_IN_MEMORY_STORE: dict[str, dict] = {}


def _dict_to_twin(data: dict) -> VehicleTwin:
    known = {f.name for f in VehicleTwin.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in known}
    return VehicleTwin(**filtered)


def _correct_tyre_pressure_for_temp(
    pressure_kpa: float,
    outside_temp_c: float,
    reference_temp_c: float = 25.0,
) -> float:
    """P_corrected = P_measured * (273.15 + ref) / (273.15 + measured_temp)"""
    if outside_temp_c < -50 or outside_temp_c > 60:
        return pressure_kpa
    return round(
        pressure_kpa * (273.15 + reference_temp_c) / (273.15 + outside_temp_c),
        2,
    )
