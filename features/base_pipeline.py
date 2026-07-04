"""
Feature pipeline base class.

All feature pipelines share:
 - Column normalisation (PascalCase CSV  ↔  snake_case InfluxDB)
 - compute(vin, df)            → one summary row (inference / training snapshot)
 - compute_from_influx(vin)    → queries InfluxDB, then calls compute()
 - compute_batch(fleet_df)     → iterates all VINs from CSV dir or InfluxDB
 - helper: _slope, _last_days, _safe, _rate_per_100km
"""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

INFLUX_URL    = os.getenv("INFLUXDB_URL",    "http://localhost:8086")
INFLUX_TOKEN  = os.getenv("INFLUXDB_TOKEN",  "autopredict-dev-token")
INFLUX_ORG    = os.getenv("INFLUXDB_ORG",    "autopredict")
INFLUX_BUCKET = os.getenv("INFLUXDB_BUCKET", "telemetry")

# ── Unified column alias map ────────────────────────────────────────────────
# Maps both PascalCase (synthetic CSV) and snake_case (InfluxDB output) to
# the internal names used by all pipelines.

_COL_ALIASES: dict[str, str] = {
    # ── Identity / timestamp ─────────────────────────────────────────────────
    "VIN":                    "vin",
    "StartTime-TimeStamp":    "timestamp",
    "StartTime-Date":         "date",
    "_time":                  "timestamp",

    # ── Motion — real TBox camelCase names (generator output) ───────────────
    "vehSpeed":               "speed",          "VehSpeed":               "speed",          "veh_speed":           "speed",
    "vehSysPwrMod":           "sys_pwr_mod",    "VehSysPwrMod":           "sys_pwr_mod",    "veh_sys_pwr_mod":     "sys_pwr_mod",
    "vehRPM":                 "rpm",            "VehRPM":                 "rpm",            "veh_rpm":             "rpm",
    "vehGearPos":             "gear_pos",       "VehGearPos":             "gear_pos",       "veh_gear_pos":        "gear_pos",
    "vehSteeringAngle":       "steering_angle", "VehSteeringAngle":       "steering_angle", "veh_steering_angle":  "steering_angle",
    "vehAccelPos":            "accel_pos",      "VehAccelPos":            "accel_pos",      "veh_accel_pos":       "accel_pos",
    "vehBrakePos":            "brake_pos",      "VehBrakePos":            "brake_pos",      "veh_brake_pos":       "brake_pos",
    # Real TBox accelerometer names (spec: tboxAccelX/Y/Z)
    "tboxAccelX":             "accel_x",        "VehAccelX":              "accel_x",        "veh_accel_x":         "accel_x",
    "tboxAccelY":             "accel_y",        "tbox_accel_y":           "accel_y",
    "tboxAccelZ":             "accel_z",        "tbox_accel_z":           "accel_z",

    # ── Powertrain ─────────────────────────────────────────────────────────
    "vehBatt":                "batt_12v",       "VehBatt":                "batt_12v",       "veh_batt":            "batt_12v",
    "vehOdo":                 "odometer",       "VehOdo":                 "odometer",       "veh_odo":             "odometer",
    "FuelTankLevel":          "fuel_level",     "fuel_tank_level":        "fuel_level",
    "vehFuelConsumed":        "fuel_consumed",  "VehFuelConsumed":        "fuel_consumed",  "veh_fuel_consumed":   "fuel_consumed",
    "vehCoolantTemp":         "coolant_temp",   "VehCoolantTemp":         "coolant_temp",   "veh_coolant_temp":    "coolant_temp",
    "vehOutsideTemp":         "outside_temp",   "VehOutsideTemp":         "outside_temp",   "veh_outside_temp":    "outside_temp",
    "vehInsideTemp":          "inside_temp",    "veh_inside_temp":        "inside_temp",

    # ── HVAC ──────────────────────────────────────────────────────────────
    "vehAC":                  "ac_on",          "VehAC":                  "ac_on",
    "vehACFanSpeed":          "ac_fan_speed",   "VehACFanSpeed":          "ac_fan_speed",

    # ── Lighting ──────────────────────────────────────────────────────────
    "vehDipLight":            "dip_light",
    "vehMainLight":           "main_light",
    "vehSideLight":           "side_light",
    "vehRainDetected":        "rain_detected",
    "vehNightDetected":       "night_detected",
    "vehHorn":                "horn",
    "vehSeatBeltDrv":         "seatbelt_drv",

    # ── Real binary warning signals (ICE + common) ────────────────────────
    "vehOilPressureWarning":  "oil_pressure_warning",
    "vehMILWarning":          "mil_warning",
    "vehBrkFludLvlLow":       "brake_fluid_low",
    "vehABSF":                "abs_failure",

    # ── Real tyre column names per TBox Big Data Spec ─────────────────────
    "frontLeftTyrePressure":   "tyre_fl",   "TyrePressureFL":  "tyre_fl",   "tyre_pressure_fl":  "tyre_fl",
    "frontRrightTyrePressure": "tyre_fr",   "TyrePressureFR":  "tyre_fr",   "tyre_pressure_fr":  "tyre_fr",
    "rearLeftTyrePressure":    "tyre_rl",   "TyrePressureRL":  "tyre_rl",   "tyre_pressure_rl":  "tyre_rl",
    "rearRightTyrePressure":   "tyre_rr",   "TyrePressureRR":  "tyre_rr",   "tyre_pressure_rr":  "tyre_rr",
    "wheelTyreMonitorStatus":  "tpms_status",

    # ── HV battery — real BMS column names per TBox spec ─────────────────
    "vehBMSPackVol":          "bms_pack_vol",       "BMSPackVol":         "bms_pack_vol",
    "vehBMSPackCrnt":         "bms_pack_crnt",      "BMSPackCrnt":        "bms_pack_crnt",
    "vehBMSPackSOC":          "soc",                "BMSPackSOC":         "soc",           "bms_pack_soc": "soc",
    "vehBMSPackSOCV":         "soc_valid",          "SOCValid":           "soc_valid",
    "vehBMSCellMaxVol":       "cell_max_vol",       "BMSCellMaxVol":      "cell_max_vol",
    "vehBMSCellMinVol":       "cell_min_vol",       "BMSCellMinVol":      "cell_min_vol",
    "vehBMSCellMaxTem":       "cell_max_temp",      "BMSCellMaxTemp":     "cell_max_temp",
    "vehBMSCellMinTem":       "cell_min_temp",      "BMSCellMinTemp":     "cell_min_temp",
    "vehHVDCDCTem":           "dcdc_temp",
    "vehBMSCMUFlt":           "bms_cmu_fault",
    "vehBMSCellVoltFlt":      "bms_cell_volt_fault",
    "vehBMSPackTemFlt":       "bms_pack_temp_fault",
    "vehBMSHVILClsd":         "bms_hvil_closed",
    "vehBMSBscSta":           "bms_status",

    # ── EV charging signals ───────────────────────────────────────────────
    "chargingGunIsConnected":     "charging_gun_connected",
    "vehIsCharging":              "is_charging",
    "dcOrAC":                     "dc_or_ac",
    "usedBatterySinceLastCharge": "used_battery_since_charge",
    "mileageSinceLastCharge":     "mileage_since_charge",
    "vehElecRange":               "elec_range",
    "vehEPTRdy":                  "ept_ready",
    "vehTMInvtrTem":              "motor_inv_temp",
    "vehTMSttrTem":               "motor_str_temp",

    # ── GNSS — real lowercase spec names + old PascalCase ────────────────
    "gnssLat":                "lat",       "GNSSLat":    "lat",    "gnss_lat":   "lat",
    "gnssLong":               "long",      "GNSSLong":   "long",   "gnss_long":  "long",
    "gnssAlt":                "alt",       "GNSSAlt":    "alt",    "gnss_alt":   "alt",
    "gnssHead":               "gnss_head", "GNSSHead":   "gnss_head",
    "gnssSats":               "gnss_sats", "GNSSSats":   "gnss_sats",

    # ── Labels (synthetic only) ───────────────────────────────────────────
    "_failure_type":          "failure_type",
    "_failure_intensity":     "failure_intensity",
}


class FeaturePipeline(ABC):
    """Abstract base for all feature engineering pipelines."""

    MEASUREMENT = "vehicle_telemetry"

    # ── Abstract interface ─────────────────────────────────────────────────

    @abstractmethod
    def compute(
        self,
        vin: str,
        df: pd.DataFrame,
        label_df: pd.DataFrame | None = None,
        **ctx: Any,
    ) -> pd.DataFrame:
        """
        Compute features from a time-sorted telemetry DataFrame.

        Parameters
        ----------
        vin :      Vehicle identifier.
        df :       Telemetry rows — accepts both CSV PascalCase and
                   InfluxDB snake_case column names.
        label_df : Optional failures_manifest DataFrame for target labels.
        **ctx :    Extra context (e.g. last_service_odo, manufacture_year).

        Returns
        -------
        Single-row DataFrame with all computed features.
        """

    # ── Online mode ────────────────────────────────────────────────────────

    def compute_from_influx(
        self, vin: str, lookback_days: int = 30, **ctx: Any
    ) -> pd.DataFrame:
        """Query InfluxDB for *lookback_days* of history, then compute features."""
        df = self._query_influx(vin, lookback_days)
        if df.empty:
            log.warning("No InfluxDB data for VIN %s (last %dd)", vin, lookback_days)
            return pd.DataFrame()
        return self.compute(vin, df, **ctx)

    # ── Batch training mode ────────────────────────────────────────────────

    def compute_batch(
        self,
        fleet_df: pd.DataFrame,
        telemetry_dir: Path | str | None = None,
        label_df: pd.DataFrame | None = None,
        **ctx: Any,
    ) -> pd.DataFrame:
        """
        Compute features for every VIN in *fleet_df*.

        If *telemetry_dir* is given, loads per-VIN CSVs from that directory.
        Otherwise, queries InfluxDB for each VIN.
        """
        tdir = Path(telemetry_dir) if telemetry_dir else None
        rows: list[pd.DataFrame] = []

        for _, vrow in fleet_df.iterrows():
            vin = str(vrow["vin"])
            vin_ctx: dict[str, Any] = {
                "manufacture_year": int(vrow.get("manufacture_year", 2022)),
                "fuel_type":        str(vrow.get("fuel_type", "ICE")),
                "battery_capacity_kwh": vrow.get("battery_capacity_kwh"),
                **ctx,
            }
            try:
                if tdir:
                    csv_path = tdir / f"telemetry_{vin}.csv"
                    if not csv_path.exists():
                        log.warning("CSV not found: %s", csv_path)
                        continue
                    df = pd.read_csv(csv_path, low_memory=False)
                    row = self.compute(vin, df, label_df, **vin_ctx)
                else:
                    row = self.compute_from_influx(vin, **vin_ctx)
                if row is not None and not row.empty:
                    rows.append(row)
            except Exception as exc:
                log.error("Feature computation failed for VIN %s: %s", vin, exc)

        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    # ── Column normalisation ───────────────────────────────────────────────

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        rename = {k: v for k, v in _COL_ALIASES.items() if k in df.columns}
        df = df.rename(columns=rename)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True, errors="coerce")
            df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def _last_days(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        if "timestamp" not in df.columns or df.empty:
            return df
        t_max  = df["timestamp"].max()
        cutoff = t_max - pd.Timedelta(days=n)
        return df[df["timestamp"] >= cutoff]

    # ── Feature helpers ────────────────────────────────────────────────────

    @staticmethod
    def _slope(series: pd.Series) -> float:
        """OLS slope of *series* (index = equally-spaced time steps)."""
        arr   = series.dropna().values
        if len(arr) < 2:
            return 0.0
        x = np.arange(len(arr), dtype=float)
        return float(np.polyfit(x, arr, 1)[0])

    @staticmethod
    def _safe(val: Any, default: float = np.nan) -> float:
        try:
            v = float(val)
            return v if np.isfinite(v) else default
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _rate_per_100km(mask: "np.ndarray[bool]", odo: pd.Series) -> float:
        """Count of True events in *mask* per 100 km driven."""
        km = float(odo.max() - odo.min()) if len(odo) > 0 else 0.0
        if km < 0.1:
            return 0.0
        return float(mask.sum()) / km * 100

    @staticmethod
    def _label_days_to_failure(
        vin: str, label_df: pd.DataFrame, failure_type: str, t_ref: "pd.Timestamp"
    ) -> tuple[float, int]:
        """
        Return (days_to_failure, within_30_days) from failures_manifest.

        Returns (np.nan, 0) if VIN has no matching failure in label_df.
        """
        if label_df is None or label_df.empty:
            return np.nan, 0
        mask = (label_df["vin"] == vin) & (label_df["failure_type"] == failure_type)
        vf   = label_df[mask]
        if vf.empty:
            return np.nan, 0
        fail_dt = pd.to_datetime(vf["failure_date"].iloc[0], utc=True)
        days    = (fail_dt - t_ref).total_seconds() / 86400
        return float(days), int(0 < days <= 30)

    # ── InfluxDB query ─────────────────────────────────────────────────────

    _influx_reachable: bool | None = None

    def _query_influx(self, vin: str, lookback_days: int) -> pd.DataFrame:
        try:
            if FeaturePipeline._influx_reachable is None:
                import socket
                host = INFLUX_URL.split("://")[-1].split(":")[0]
                port = int(INFLUX_URL.split(":")[-1].rstrip("/") or "8086")
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(0.3)
                FeaturePipeline._influx_reachable = s.connect_ex((host, port)) == 0
                s.close()
            if not FeaturePipeline._influx_reachable:
                return pd.DataFrame()

            from influxdb_client import InfluxDBClient
            client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
            query_api = client.query_api()
            flux = f"""
from(bucket: "{INFLUX_BUCKET}")
  |> range(start: -{lookback_days}d)
  |> filter(fn: (r) => r["_measurement"] == "{self.MEASUREMENT}")
  |> filter(fn: (r) => r["vin"] == "{vin}")
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> sort(columns: ["_time"])
"""
            tables = query_api.query_data_frame(flux)
            client.close()
            if isinstance(tables, list):
                if not tables:
                    return pd.DataFrame()
                df = pd.concat(tables, ignore_index=True)
            else:
                df = tables
            df = df.rename(columns={"_time": "timestamp"})
            return df
        except Exception as exc:
            log.error("InfluxDB query failed for VIN %s: %s", vin, exc)
            return pd.DataFrame()

    # ── Common row constructor ─────────────────────────────────────────────

    @staticmethod
    def _row(vin: str, ts: "pd.Timestamp | None", features: dict[str, Any]) -> pd.DataFrame:
        row = {"vin": vin, "computed_at": ts or pd.Timestamp.utcnow(), **features}
        return pd.DataFrame([row])
