"""
EV DC-DC Converter Feature Engine

Monitors the bidirectional DC-DC converter that steps HV pack voltage (300–450V)
down to 12–14.4V to charge the 12V auxiliary battery and power LV systems.

Signal mapping:
    vehBatt                      → V   (12V rail / DC-DC output voltage; pipeline stores decoded volts)
    vehHVDCDCTem  - 40           → °C  (DC-DC converter temperature; raw offset-encoded)
    vehBMSPackCrnt               → A   (HV pack current; pipeline stores decoded amps)
    vehSysPwrMod  == 2           → RUN mode (DC-DC active)
    vehSysPwrMod  == 0           → OFF (DC-DC inactive, 12V on own battery)

Failure presentation: DC-DC faults often surface as 12V battery symptoms
(chronic under-charge, slow start), which is why these features are wired
into the battery_12v FeatureStore group.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# 12V rail healthy range when DC-DC is active
_DCDC_NOMINAL_LOW  = 13.8  # V — minimum healthy output
_DCDC_NOMINAL_HIGH = 14.4  # V — maximum healthy output
_RUN_MODE          = 2
_OFF_MODE          = 0
_HIGH_LOAD_CRNT_A  = 50.0  # HV pack current threshold for "high load" classification

ALERT_THRESHOLDS: dict[str, tuple[str, float, str]] = {
    # feature_key: (severity, threshold, comparison)
    # Temperature rise above session-start baseline — not absolute value.
    # Absolute temperature is not meaningful because under-bonnet temp after
    # two hours of sun parking in summer India can already be 65–80°C.
    "dcdc_temp_rise_max_c":      ("CRITICAL", 55.0,  "gt"),   # 55°C rise = severe
    "dcdc_temp_rise_mean_c":     ("MEDIUM",   25.0,  "gt"),   # 25°C mean rise = elevated
    "dcdc_output_v_mean_30d":    ("HIGH",     13.5,  "lt"),
    "dcdc_output_v_min_30d":     ("CRITICAL", 12.8,  "lt"),
    "high_load_voltage_droop_v": ("MEDIUM",    0.5,  "gt"),
}


class EVDCDCFeatureEngine:
    """
    Compute 7 DC-DC converter health features from 30-day TBox telemetry.

    Usage
    -----
    engine = EVDCDCFeatureEngine()
    features = engine.compute(df, vin)   # df = full fleet telemetry DataFrame
    """

    def compute(self, df: pd.DataFrame, vin: str) -> dict[str, float | None]:
        vin_df = df[df["vin"] == vin].copy() if "vin" in df.columns else df.copy()

        if vin_df.empty or "vehSysPwrMod" not in vin_df.columns:
            return self._null_features()

        running = vin_df["vehSysPwrMod"] == _RUN_MODE

        # ── DC-DC OUTPUT VOLTAGE WHILE RUNNING ────────────────────────────────
        # When in RUN mode the DC-DC should hold the 12V rail at 13.8–14.4V.
        # Below 13.5V = underperforming; below 12.8V while running = fault.
        dcdc_output_v_mean_30d: float | None = None
        dcdc_output_v_min_30d:  float | None = None

        if "vehBatt" in vin_df.columns and running.any():
            output_v = vin_df["vehBatt"].where(running)
            _mean = output_v.mean()
            _min  = output_v.min()
            if not np.isnan(_mean):
                dcdc_output_v_mean_30d = round(float(_mean), 2)
            if not np.isnan(_min):
                dcdc_output_v_min_30d = round(float(_min), 2)
        else:
            output_v = pd.Series(dtype=float, index=vin_df.index)

        # ── DC-DC CONVERTER TEMPERATURE ───────────────────────────────────────
        # vehHVDCDCTem: physical = raw - 40  (°C)
        # Under-bonnet baseline: average DCDC temp in the first 2 min after each
        # key-on (before driving heat builds up). If no sessions in data, fall back
        # to ambient + 15°C (passive heat soak when parked).
        dcdc_temp_max_30d:   float | None = None
        dcdc_temp_mean_30d:  float | None = None
        dcdc_baseline_temp_c: float | None = None
        dcdc_temp_rise_max_c: float | None = None
        dcdc_temp_rise_mean_c: float | None = None

        if "vehHVDCDCTem" in vin_df.columns:
            dcdc_temp = vin_df["vehHVDCDCTem"] - 40.0

            # Session-start baseline: first 120 rows (2 min at 1 Hz) after 0→2 transition
            pwr_vals = vin_df["vehSysPwrMod"].fillna(0).values if "vehSysPwrMod" in vin_df.columns else None
            baselines: list[float] = []
            if pwr_vals is not None:
                session_start_positions = np.where(np.diff(pwr_vals.astype(float), prepend=0.0) == 2)[0]
                vin_df_reset = vin_df.reset_index(drop=True)
                for pos in session_start_positions:
                    window_end = min(int(pos) + 120, len(vin_df_reset))
                    startup_win = vin_df_reset.iloc[int(pos):window_end]
                    if len(startup_win) > 5:
                        baselines.append(float((startup_win["vehHVDCDCTem"] - 40.0).mean()))

            if baselines:
                dcdc_baseline_temp_c = round(float(np.mean(baselines)), 1)
            elif "vehOutsideTemp" in vin_df.columns:
                dcdc_baseline_temp_c = round(float(vin_df["vehOutsideTemp"].mean()) + 15.0, 1)
            else:
                dcdc_baseline_temp_c = round(25.0 + 15.0, 1)   # conservative ambient fallback

            _tmax  = dcdc_temp.max()
            _tmean = dcdc_temp.mean()
            if not np.isnan(_tmax):
                dcdc_temp_max_30d      = round(float(_tmax),  1)
                dcdc_temp_rise_max_c   = round(float(_tmax)  - dcdc_baseline_temp_c, 1)
            if not np.isnan(_tmean):
                dcdc_temp_mean_30d     = round(float(_tmean), 1)
                dcdc_temp_rise_mean_c  = round(float(_tmean) - dcdc_baseline_temp_c, 1)

        # ── OUTPUT VOLTAGE DROOP UNDER HIGH HV LOAD ───────────────────────────
        # When the HV pack is under high current demand the DC-DC must work
        # harder; an aging converter droops below its idle-load output.
        high_load_voltage_droop_v: float | None = None

        if (
            dcdc_output_v_mean_30d is not None
            and "vehBMSPackCrnt" in vin_df.columns
        ):
            hv_current_a = vin_df["vehBMSPackCrnt"]
            high_load    = hv_current_a.abs() > _HIGH_LOAD_CRNT_A
            output_at_high_load = output_v.where(running & high_load).mean()
            if not np.isnan(output_at_high_load):
                droop = dcdc_output_v_mean_30d - float(output_at_high_load)
                high_load_voltage_droop_v = round(droop, 3)

        # ── STARTUP VOLTAGE RECOVERY ──────────────────────────────────────────
        # At every OFF→RUN transition the DC-DC should engage immediately,
        # causing a sharp positive jump in the 12V rail.
        # Near-zero or negative delta = converter slow to engage (control fault).
        # Note: we diff raw vehBatt (not masked) so the pre-start parked voltage
        # is available as the reference, then gate on the just_started mask.
        dcdc_startup_recovery_v: float | None = None

        if "vehBatt" in vin_df.columns and running.any():
            raw_v        = vin_df["vehBatt"]
            off_to_run   = (vin_df["vehSysPwrMod"].shift(1) == _OFF_MODE)
            just_started = running & off_to_run
            if just_started.any():
                delta = (raw_v - raw_v.shift(1)).where(just_started)
                _rec = delta.mean()
                if not np.isnan(_rec):
                    dcdc_startup_recovery_v = round(float(_rec), 3)

        # ── THERMAL CYCLE COUNT ────────────────────────────────────────────────
        # Each OFF→RUN transition = one thermal stress cycle (solder joint fatigue).
        dcdc_thermal_cycles_total = int((vin_df["vehSysPwrMod"].diff() == _RUN_MODE).sum())

        return {
            "dcdc_output_v_mean_30d":    dcdc_output_v_mean_30d,
            "dcdc_output_v_min_30d":     dcdc_output_v_min_30d,
            "dcdc_baseline_temp_c":      dcdc_baseline_temp_c,
            "dcdc_temp_max_30d":         dcdc_temp_max_30d,
            "dcdc_temp_mean_30d":        dcdc_temp_mean_30d,
            "dcdc_temp_rise_max_c":      dcdc_temp_rise_max_c,
            "dcdc_temp_rise_mean_c":     dcdc_temp_rise_mean_c,
            "high_load_voltage_droop_v": high_load_voltage_droop_v,
            "dcdc_startup_recovery_v":   dcdc_startup_recovery_v,
            "dcdc_thermal_cycles_total": dcdc_thermal_cycles_total,
        }

    @staticmethod
    def _null_features() -> dict[str, None | int]:
        return {
            "dcdc_output_v_mean_30d":    None,
            "dcdc_output_v_min_30d":     None,
            "dcdc_baseline_temp_c":      None,
            "dcdc_temp_max_30d":         None,
            "dcdc_temp_mean_30d":        None,
            "dcdc_temp_rise_max_c":      None,
            "dcdc_temp_rise_mean_c":     None,
            "high_load_voltage_droop_v": None,
            "dcdc_startup_recovery_v":   None,
            "dcdc_thermal_cycles_total": 0,
        }
