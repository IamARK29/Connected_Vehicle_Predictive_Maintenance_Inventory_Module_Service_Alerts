"""
EV Motor Feature Engine — CH-21 signal group

Computes 8 features covering torque delivery efficiency, inverter/stator
thermal stress, torque ripple, and motor-speed fidelity for EV/PHEV drivetrains.

Signal scaling (TBox Big Data Spec, CH-21):
    vehEPTTrInptShaftToq  * 0.5  - 848   → Nm  (shaft input torque)
    vehTMActuToq          * 0.5  - 512   → Nm  (motor actual torque)
    vehTMInvtrTem         - 40           → °C  (inverter temperature)
    vehTMSttrTem          - 40           → °C  (stator temperature)
    vehTMSpd              - 32768        → rpm (motor speed)
    vehTMInvtrCrnt        - 1024         → A   (inverter DC current, signed)

Validity flags: value == 1 means the signal is invalid/unavailable.
Call _valid(df, col) to get a boolean mask safe for missing columns.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Drivetrain constants (MG ZS EV approximate)
_FINAL_DRIVE_RATIO  = 8.05
_TYRE_RADIUS_M      = 0.338   # 235/50 R19

ALERT_THRESHOLDS: dict[str, tuple[str, float, str]] = {
    # feature_key: (severity, threshold, comparison)
    "inv_temp_max_30d":            ("CRITICAL", 100.0, "gt"),
    "stator_temp_max_30d":         ("HIGH",     130.0, "gt"),
    "torque_efficiency_mean_30d":  ("MEDIUM",     0.88, "lt"),
    "torque_ripple_proxy_nm":      ("MEDIUM",    15.0, "gt"),
}


def _valid(df: pd.DataFrame, flag_col: str) -> "pd.Series[bool]":
    """Return a boolean mask that is True wherever the validity flag is NOT 1.

    If the flag column is absent, all rows are treated as valid (True).
    """
    col = df.get(flag_col)
    if col is None:
        return pd.Series(True, index=df.index)
    return col != 1


class EVMotorFeatureEngine:
    """
    Compute EV motor health features from 30-day TBox telemetry.

    Usage
    -----
    engine = EVMotorFeatureEngine()
    features = engine.compute(df, vin)   # df = full fleet telemetry DataFrame
    """

    def compute(self, df: pd.DataFrame, vin: str) -> dict[str, float | None]:
        """All signals from CH-21. Validity flag checks applied before use."""
        vin_df = df[df["vin"] == vin].copy() if "vin" in df.columns else df.copy()

        if vin_df.empty:
            return self._null_features()

        # ── TORQUE DELIVERY EFFICIENCY ────────────────────────────────────────
        # vehEPTTrInptShaftToq: physical = raw * 0.5 - 848
        # vehTMActuToq:         physical = raw * 0.5 - 512
        # Large persistent gap = mechanical loss (bearing wear, coupling issue)
        valid_toq = (
            _valid(vin_df, "vehEPTTrInptShaftToqV") &
            _valid(vin_df, "vehTMActuToqV") &
            vin_df["vehTMActuToq"].notna()
        ) if "vehTMActuToq" in vin_df.columns else pd.Series(False, index=vin_df.index)

        torque_efficiency_mean_30d: float | None = None
        if valid_toq.any() and "vehEPTTrInptShaftToq" in vin_df.columns:
            shaft_toq  = vin_df["vehEPTTrInptShaftToq"] * 0.5 - 848
            actual_toq = vin_df["vehTMActuToq"]          * 0.5 - 512
            # Only evaluate when motor is producing meaningful torque
            meaningful = actual_toq.abs() > 10
            torque_gap = (actual_toq - shaft_toq).where(valid_toq & meaningful)
            denom = actual_toq.abs().where(valid_toq & meaningful).clip(lower=1).mean()
            if denom and not np.isnan(denom):
                raw_eff = 1.0 - (torque_gap.abs().mean() / denom)
                torque_efficiency_mean_30d = round(float(np.clip(raw_eff, 0.0, 1.0)), 3)
        else:
            actual_toq = pd.Series(dtype=float)

        # ── INVERTER TEMPERATURE ──────────────────────────────────────────────
        # vehTMInvtrTem: physical = raw * 1.0 - 40  (°C)
        inv_temp_max_30d:  float | None = None
        inv_temp_mean_30d: float | None = None
        inv_temp_per_kw:   float | None = None

        if "vehTMInvtrTem" in vin_df.columns:
            inv_temp = vin_df["vehTMInvtrTem"] - 40.0
            inv_temp_max_30d  = round(float(inv_temp.max()),  1)
            inv_temp_mean_30d = round(float(inv_temp.mean()), 1)

            # ── INVERTER TEMPERATURE vs POWER RATIO ──────────────────────────
            # Excessive temp for a given power level = cooling degradation / aging IGBT
            if "vehTMInvtrCrnt" in vin_df.columns and "vehTMInvtrVol" in vin_df.columns:
                motor_power_kw = (vin_df["vehTMInvtrCrnt"] - 1024) * vin_df["vehTMInvtrVol"] / 1000.0
                high_power = motor_power_kw > 10
                if high_power.sum() > 100:
                    ratio = inv_temp.where(high_power) / motor_power_kw.where(high_power)
                    inv_temp_per_kw = round(float(ratio.mean()), 3)
                else:
                    inv_temp_per_kw = 0.0
        else:
            inv_temp = pd.Series(dtype=float)

        # ── STATOR TEMPERATURE ────────────────────────────────────────────────
        # vehTMSttrTem: physical = raw * 1.0 - 40  (°C)
        stator_temp_max_30d:  float | None = None
        stator_temp_mean_30d: float | None = None

        if "vehTMSttrTem" in vin_df.columns:
            stator_temp = vin_df["vehTMSttrTem"] - 40.0
            stator_temp_max_30d  = round(float(stator_temp.max()),  1)
            stator_temp_mean_30d = round(float(stator_temp.mean()), 1)

        # ── TORQUE RIPPLE DETECTION ───────────────────────────────────────────
        # Healthy motor produces smooth torque at constant commanded input.
        # High std during steady-speed motoring = winding or bearing fault.
        torque_ripple_proxy_nm: float | None = None

        if valid_toq.any() and not actual_toq.empty and "vehSpeed" in vin_df.columns:
            steady_speed = vin_df["vehSpeed"].diff().abs() < 5
            commanded    = (
                vin_df["vehAccelPos"] > 50
                if "vehAccelPos" in vin_df.columns
                else pd.Series(True, index=vin_df.index)
            )
            ripple_series = actual_toq.where(valid_toq & steady_speed & commanded)
            if ripple_series.count() >= 30:
                torque_ripple_proxy_nm = round(float(ripple_series.std()), 2)

        # ── MOTOR SPEED vs EXPECTED ───────────────────────────────────────────
        # vehTMSpd: physical = raw - 32768  (rpm)
        # Compare against kinematically expected RPM from wheel speed.
        motor_rpm_deviation_mean: float | None = None

        if "vehTMSpd" in vin_df.columns and "vehSpeed" in vin_df.columns:
            motor_rpm = vin_df["vehTMSpd"] - 32768.0
            # vehSpeed unit = 0.1 km/h → convert to m/s then to wheel rpm
            wheel_speed_ms      = vin_df["vehSpeed"] * 0.1 / 3.6
            expected_wheel_rpm  = wheel_speed_ms / (2 * np.pi * _TYRE_RADIUS_M)
            expected_motor_rpm  = expected_wheel_rpm * _FINAL_DRIVE_RATIO
            valid_rpm = _valid(vin_df, "vehTMSpdV") & (vin_df["vehSpeed"] > 100)
            if valid_rpm.sum() >= 30:
                deviation = (motor_rpm.where(valid_rpm) - expected_motor_rpm.where(valid_rpm)).abs()
                motor_rpm_deviation_mean = round(float(deviation.mean()), 0)

        return {
            "torque_efficiency_mean_30d": torque_efficiency_mean_30d,
            "inv_temp_max_30d":           inv_temp_max_30d,
            "inv_temp_mean_30d":          inv_temp_mean_30d,
            "inv_temp_per_kw":            inv_temp_per_kw,
            "stator_temp_max_30d":        stator_temp_max_30d,
            "stator_temp_mean_30d":       stator_temp_mean_30d,
            "torque_ripple_proxy_nm":     torque_ripple_proxy_nm,
            "motor_rpm_deviation_mean":   motor_rpm_deviation_mean,
        }

    @staticmethod
    def _null_features() -> dict[str, None]:
        return {
            "torque_efficiency_mean_30d": None,
            "inv_temp_max_30d":           None,
            "inv_temp_mean_30d":          None,
            "inv_temp_per_kw":            None,
            "stator_temp_max_30d":        None,
            "stator_temp_mean_30d":       None,
            "torque_ripple_proxy_nm":     None,
            "motor_rpm_deviation_mean":   None,
        }
