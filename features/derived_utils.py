"""
Derived telemetry parameter utilities for AutoPredict feature pipelines.

All functions accept DataFrames with either Big Data Spec raw column names
(vehBrakePos, vehSpeed, …) or the internal normalized names used by the
pipeline base class (brake_pos, speed, …).  A _col() helper resolves the
first available alias so callers need not worry about naming conventions.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Column alias resolution ───────────────────────────────────────────────────
# Maps each canonical (raw Big Data Spec) name to its fallback alternatives.
_COL_MAP: dict[str, list[str]] = {
    "vehBrakePos":           ["vehBrakePos", "VehBrakePos", "brake_pos"],
    "vehSpeed":              ["vehSpeed", "VehSpeed", "speed"],
    "tboxAccelX":            ["tboxAccelX", "VehAccelX", "accel_x"],
    "tboxAccelY":            ["tboxAccelY", "accel_y"],
    "tboxAccelZ":            ["tboxAccelZ", "accel_z"],
    "vehSysPwrMod":          ["vehSysPwrMod", "VehSysPwrMod", "sys_pwr_mod"],
    "vehAccelPos":           ["vehAccelPos", "VehAccelPos", "accel_pos"],
    "vehRPM":                ["vehRPM", "VehRPM", "rpm"],
    "vehGearPos":            ["vehGearPos", "VehGearPos", "gear_pos"],
    "vehOdo":                ["vehOdo", "VehOdo", "odometer"],
    "vehCoolantTemp":        ["vehCoolantTemp", "VehCoolantTemp", "coolant_temp"],
    "vehOutsideTemp":        ["vehOutsideTemp", "VehOutsideTemp", "outside_temp"],
    "vehBMSPackSOC":         ["vehBMSPackSOC", "BMSPackSOC", "soc"],
    "vehBMSPackSOCV":        ["vehBMSPackSOCV", "SOCValid", "soc_valid"],
    "vehPackVol":            ["vehPackVol", "BMSPackVol", "bms_pack_vol"],
    "vehPackCrnt":           ["vehPackCrnt", "BMSPackCrnt", "bms_pack_crnt"],
    "vehEPTTrInptShaftToq":  ["vehEPTTrInptShaftToq"],
    "vehEPTTrInptShaftToqV": ["vehEPTTrInptShaftToqV"],
    "vehDoorFrontDrv":       ["vehDoorFrontDrv", "door_front_drv"],
    "vehDoorFrontPas":       ["vehDoorFrontPas", "door_front_pas"],
    "vehDoorRearLeft":       ["vehDoorRearLeft", "door_rear_left"],
    "vehDoorRearRight":      ["vehDoorRearRight", "door_rear_right"],
    "vehBonnet":             ["vehBonnet", "bonnet"],
    "vehBoot":               ["vehBoot", "boot"],
}


def _col(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    """Return the first matching column for *name*, NaN filled with *default*."""
    for alias in _COL_MAP.get(name, [name]):
        if alias in df.columns:
            return df[alias].fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def _col_opt(df: pd.DataFrame, name: str) -> pd.Series:
    """Like _col() but fills NaN with NaN (optional/nullable signal)."""
    for alias in _COL_MAP.get(name, [name]):
        if alias in df.columns:
            return df[alias]
    return pd.Series(np.nan, index=df.index, dtype=float)


def _speed_diff(df: pd.DataFrame, speed_col: str | None) -> pd.Series:
    """Per-VIN diff of the speed column; returns zeros if column absent."""
    if speed_col is None:
        return pd.Series(0.0, index=df.index)
    if "vin" in df.columns:
        return df.groupby("vin")[speed_col].diff().fillna(0.0)
    return df[speed_col].fillna(0.0).diff().fillna(0.0)


# ── SESSION DETECTION ─────────────────────────────────────────────────────────

def assign_session_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Consecutive rows where vehSysPwrMod > 0 form one session."""
    sort_cols = [c for c in ["vin", "timestamp"] if c in df.columns]
    df = df.sort_values(sort_cols).copy() if sort_cols else df.copy()
    pwr = _col(df, "vehSysPwrMod")
    is_on = pwr > 0
    new_session = is_on & (~is_on.shift(1, fill_value=False))
    df["session_id"] = new_session.cumsum()
    df.loc[~is_on, "session_id"] = pd.NA
    return df


# ── HARSH BRAKE DETECTION ─────────────────────────────────────────────────────

def detect_harsh_brake(df: pd.DataFrame) -> pd.DataFrame:
    """Flag rows where ALL 3 conditions are true simultaneously:
    1. vehBrakePos raw > 175   (=70% physical, 175×0.4=70.0)
    2. tboxAccelX raw < -75    (=-0.3g physical, -75×0.004=-0.3)
    3. ΔvehSpeed per second < -80 raw counts (=-8 kph/s physical)
    """
    sort_cols = [c for c in ["vin", "timestamp"] if c in df.columns]
    df = df.sort_values(sort_cols).copy() if sort_cols else df.copy()

    speed_col = next((c for c in ["vehSpeed", "VehSpeed", "speed"] if c in df.columns), None)
    brake = _col(df, "vehBrakePos")
    accel_x = _col(df, "tboxAccelX")
    speed_delta = _speed_diff(df, speed_col)

    df["is_harsh_brake"] = (
        (brake > 175) & (accel_x < -75) & (speed_delta < -80)
    ).fillna(False)
    return df


# ── BRAKE STRESS INDEX ────────────────────────────────────────────────────────

def compute_brake_stress_index(df: pd.DataFrame) -> pd.DataFrame:
    """BSI per second = brake_pct × (speed_kph / 100)²
    Only when braking active (vehBrakePos raw > 25 = 10% physical).
    Represents kinetic energy dissipation (KE ∝ v²).
    """
    df = df.copy()
    brake = _col(df, "vehBrakePos")
    speed = _col(df, "vehSpeed")
    brake_pct = brake * 0.4        # raw → %
    speed_kph = speed * 0.1        # raw → kph
    bsi = brake_pct * (speed_kph / 100.0) ** 2
    df["bsi"] = bsi.where(brake > 25, 0.0)
    return df


# ── REGEN DETECTION (EV) ──────────────────────────────────────────────────────

def detect_regen_event(df: pd.DataFrame) -> pd.DataFrame:
    """Regen = negative motor torque while decelerating, minimal friction braking.
    vehEPTTrInptShaftToq: raw 1696 = 0 Nm.  raw < 1696 → negative (regen).
    """
    sort_cols = [c for c in ["vin", "timestamp"] if c in df.columns]
    df = df.sort_values(sort_cols).copy() if sort_cols else df.copy()

    speed_col = next((c for c in ["vehSpeed", "VehSpeed", "speed"] if c in df.columns), None)
    torque = _col(df, "vehEPTTrInptShaftToq", default=1696)
    torque_valid = _col_opt(df, "vehEPTTrInptShaftToqV")
    brake = _col(df, "vehBrakePos")
    speed_delta = _speed_diff(df, speed_col)

    valid_torque = (torque_valid.fillna(0) != 1)

    df["is_regen_event"] = (
        valid_torque &
        (torque < 1696) &
        (speed_delta < 0) &
        (brake < 75)
    ).fillna(False)
    return df


# ── ENGINE OIL DEGRADATION INDEX ─────────────────────────────────────────────

def compute_oil_degradation_index(
    df_vin: pd.DataFrame,
    service_history_vin: pd.DataFrame,
    oil_change_interval_km: float = 7500,
    road_impacts_per_100km: float = 0.0,
) -> float:
    """ODI = 0.30×F_km + 0.18×F_cold + 0.18×F_thermal + 0.14×F_rpm + 0.10×F_fuel + 0.10×F_road
    Returns 0.0–100.0 (100 = change immediately).

    road_impacts_per_100km: from compute_road_roughness_index()["impacts_per_100km"].
    Road impacts accelerate oil contamination via bearing surface abrasion.
    """
    odo = _col_opt(df_vin, "vehOdo")
    current_km = float(odo.dropna().max()) if not odo.dropna().empty else 0.0

    svc = pd.DataFrame()
    if len(service_history_vin) > 0 and "DescriptionOne" in service_history_vin.columns:
        svc = service_history_vin[
            service_history_vin["DescriptionOne"].str.contains(
                "ENGINE OIL|OIL FILTER|OIL CHANGE", case=False, na=False
            )
        ]
    last_oil_km = float(svc["Mileage"].max()) if (
        len(svc) > 0 and "Mileage" in svc.columns and not svc["Mileage"].isna().all()
    ) else (current_km - 3750)
    F_km = min(1.0, (current_km - last_oil_km) / max(oil_change_interval_km, 1))

    pwr = _col(df_vin, "vehSysPwrMod")
    coolant = _col_opt(df_vin, "vehCoolantTemp")
    outside = _col_opt(df_vin, "vehOutsideTemp")
    rpm = _col_opt(df_vin, "vehRPM")

    pwr_diff = pwr.diff().fillna(0)
    cold_starts = int(((pwr_diff > 0) & (pwr > 0) & (coolant.fillna(100) < 40)).sum())
    cold_weight = 1.5 if float(outside.fillna(20).mean()) < 10 else 1.0
    F_cold = min(1.0, (cold_starts * cold_weight) / 30)

    driving_count = max(int((pwr == 2).sum()), 1)
    overtemp = int((coolant.fillna(0) > 100).sum())
    F_thermal = min(1.0, overtemp / driving_count)

    high_rpm = int((rpm.fillna(0) > 4000).sum())
    F_rpm = min(1.0, high_rpm / driving_count)

    F_road = min(1.0, road_impacts_per_100km / 100.0)

    odi = (0.30 * F_km + 0.18 * F_cold + 0.18 * F_thermal +
           0.14 * F_rpm + 0.10 * 0.0 + 0.10 * F_road)
    return round(min(100.0, odi * 100), 2)


# ── HV BATTERY SOH ────────────────────────────────────────────────────────────

def compute_soh_from_charge_session(
    session_df: pd.DataFrame,
    nominal_capacity_kwh: float,
) -> float | None:
    """Coulomb-counting SoH from a qualifying charge session (20%→95% SoC).
    Returns SoH % or None if session does not qualify.
    """
    soc_raw = _col_opt(session_df, "vehBMSPackSOC")
    voltage = _col_opt(session_df, "vehPackVol") * 0.25    # raw → V
    current = _col_opt(session_df, "vehPackCrnt") * 0.05   # raw → A
    soc_valid = _col_opt(session_df, "vehBMSPackSOCV")

    valid = (soc_valid.fillna(0) != 1)
    soc = (soc_raw * 0.1).where(valid)                     # raw → %

    soc_clean = soc.dropna()
    if soc_clean.empty or soc_clean.min() > 22 or soc_clean.max() < 93:
        return None

    window = (soc >= 20) & (soc <= 95) & valid
    if window.sum() < 60:
        return None

    if "timestamp" in session_df.columns:
        dt = session_df["timestamp"].diff().dt.total_seconds().fillna(1.0)
    else:
        dt = pd.Series(1.0, index=session_df.index)

    energy_wh  = (voltage * current.abs() * dt)[window].sum()
    energy_kwh = energy_wh / 1000.0
    Q_rated    = nominal_capacity_kwh * 0.75    # expected energy 20%→95%
    if Q_rated <= 0:
        return None
    return round(min(100.0, energy_kwh / Q_rated * 100), 2)


# ── 12V BATTERY RESTING VOLTAGE ───────────────────────────────────────────────

def get_resting_voltage(df: pd.DataFrame) -> pd.Series:
    """Valid only when: SysPwrMod=0 + all doors closed + 30 min settled."""
    pwr = _col(df, "vehSysPwrMod")

    # Detect whether raw (÷10 = V) or physical (already V)
    raw_col = next((c for c in ["vehBatt", "VehBatt"] if c in df.columns), None)
    norm_col = "batt_12v" if "batt_12v" in df.columns else None
    if raw_col:
        batt = df[raw_col].fillna(np.nan) * 0.1    # raw → V
    elif norm_col:
        batt = df[norm_col].fillna(np.nan)
    else:
        return pd.Series(np.nan, index=df.index, dtype=float)

    parked = pwr == 0
    doors_closed = (
        (_col(df, "vehDoorFrontDrv")  == 0) &
        (_col(df, "vehDoorFrontPas")  == 0) &
        (_col(df, "vehDoorRearLeft")  == 0) &
        (_col(df, "vehDoorRearRight") == 0) &
        (_col(df, "vehBonnet")        == 0) &
        (_col(df, "vehBoot")          == 0)
    )
    last_on = pwr.ne(0).cumsum()
    minutes_parked = df.groupby(last_on).cumcount()
    settled = minutes_parked >= 30
    return batt.where(parked & doors_closed & settled)


# ── 12V BATTERY HEALTH SCORE ──────────────────────────────────────────────────

def compute_battery_12v_health_score(
    resting_v: float,
    trend_per_day: float,
    cranking_v: float,
    age_years: float,
) -> float:
    """Score 0–100. Weights: 0.35 voltage + 0.30 trend + 0.25 crank + 0.10 age."""
    f_v     = min(1.0, max(0.0, (resting_v  - 11.8) / 0.8))
    f_trend = min(1.0, max(0.0, 1.0 + trend_per_day / 0.010))
    f_crank = min(1.0, max(0.0, (cranking_v - 9.5)  / 1.5))
    f_age   = min(1.0, max(0.0, 1.0 - age_years / 5.0))
    return round(100.0 * (0.35 * f_v + 0.30 * f_trend + 0.25 * f_crank + 0.10 * f_age), 1)


# ── TYRE STRESS INDEX ─────────────────────────────────────────────────────────

def compute_tyre_stress_per_trip(
    trip_row: dict,
    mean_pressure_kpa: float,
    target_pressure_kpa: float = 230.0,
) -> float:
    """TSI = harsh_brakes×2 + sudden_turns×1.5 + speed_stress + under_inflation_penalty"""
    under_pct     = max(0.0, (target_pressure_kpa - mean_pressure_kpa) / max(target_pressure_kpa, 1))
    under_penalty = under_pct * 3.0
    max_speed     = float(trip_row.get("maxSpeed",  0) or 0)
    distance      = float(trip_row.get("odometer",  0) or 0)
    speed_stress  = (max_speed / 120.0) ** 2 * distance * 0.1
    return round(
        float(trip_row.get("harshBreakingNum", 0) or 0) * 2.0 +
        float(trip_row.get("suddenTurnNum",    0) or 0) * 1.5 +
        speed_stress + under_penalty,
        3,
    )


# ── DRIVER SCORE ──────────────────────────────────────────────────────────────

def compute_composite_drive_score(
    smooth_accel: float,
    smooth_brake: float,
    gear_eff: float,
    speed_compliance: float,
    fuel_eff: float,
    cornering: float,
    idle: float,
) -> float:
    """Weights: accel=0.20, brake=0.20, gear=0.10, speed=0.15, fuel=0.15, corner=0.10, idle=0.10"""
    score = (
        0.20 * smooth_accel     +
        0.20 * smooth_brake     +
        0.10 * gear_eff         +
        0.15 * speed_compliance +
        0.15 * fuel_eff         +
        0.10 * cornering        +
        0.10 * idle
    )
    return round(min(100.0, max(0.0, score)), 1)


# ── BRAKE GLAZING FACTOR (monsoon / wet-park correction) ─────────────────────
# Brake pads develop a glazed surface when they get wet and then dry without
# being used — common during India monsoon. The glazed surface has lower
# friction at session start; the driver presses harder, generating more wear
# than the base BSI (pedal travel proxy alone) would predict.

# Coefficient used in harsh-brake-rate rain adjustment.
# Named here so it can be revised from one place if field data updates it.
_RAIN_HARSH_BRAKE_COEFF = 0.3   # rate multiplied by (1 + coeff × rain_intensity/3)


def compute_session_start_glazing_factor(
    park_duration_hours: float,
    rain_intensity_during_park: int,
    ambient_humidity_proxy: float = 0.0,
) -> float:
    """Glazing multiplier for the BSI of the first 3 braking events in a session.

    Called with rain data read from actual vehRainDetected in the telemetry
    park window — no values hardcoded here.

    Args:
        park_duration_hours:        hours since last drive session ended
        rain_intensity_during_park: max vehRainDetected in park window (0–3)
        ambient_humidity_proxy:     abs(vehInsideTemp − vehOutsideTemp) during park;
                                    reserved for future calibration (not yet used)

    Returns:
        1.0 (no glazing) to 1.8 (severe glazing).
    """
    if rain_intensity_during_park == 0:
        return 1.0

    if park_duration_hours < 0.5:
        glazing_factor = 1.0     # still wet — glaze has not formed yet
    elif park_duration_hours < 2:
        glazing_factor = 1.3     # partial glaze forming
    elif park_duration_hours < 8:
        glazing_factor = 1.6     # moderate: overnight or mid-day park in rain
    else:
        glazing_factor = 1.8     # long park with rain then partial drying

    # Light rain glazes less severely than heavy rain
    if rain_intensity_during_park == 1:
        glazing_factor = 1.0 + (glazing_factor - 1.0) * 0.5

    return round(glazing_factor, 2)


def compute_brake_stress_index_with_glazing(
    df: pd.DataFrame,
    glazing_factor: float,
) -> pd.DataFrame:
    """BSI calculation with a session-start glazing multiplier.

    Applies a decaying multiplier to the first 3 braking events after a wet park.
    After 3 events the glaze has worn through and base BSI resumes.

    Decay: event 1 = full multiplier, event 2 = 70 %, event 3 = 40 %.

    Args:
        df:             session DataFrame (should cover a single driving session)
        glazing_factor: from compute_session_start_glazing_factor()
    """
    df = df.copy()
    brake = _col(df, "vehBrakePos")
    speed = _col(df, "vehSpeed")
    brake_pct = brake * 0.4         # raw → %
    speed_kph = speed * 0.1         # raw → kph
    base_bsi  = brake_pct * (speed_kph / 100.0) ** 2
    df["bsi"] = base_bsi.where(brake > 25, 0.0)

    if glazing_factor <= 1.0:
        return df

    # Braking events: raw brake pos > 50 = 20% physical pedal travel
    brake_events = brake > 50
    event_number = brake_events.cumsum()

    glazing_decay = {
        1: glazing_factor,
        2: 1.0 + (glazing_factor - 1.0) * 0.70,
        3: 1.0 + (glazing_factor - 1.0) * 0.40,
    }
    for event_n, multiplier in glazing_decay.items():
        mask = brake_events & (event_number == event_n)
        df.loc[mask, "bsi"] = df.loc[mask, "bsi"] * multiplier

    return df


# ── TYRE CARCASS TEMPERATURE MODEL ───────────────────────────────────────────
# On a 44°C day in Nagpur, a tyre at 100 kph on concrete runs 18–22°C above
# ambient. City stop-and-go adds 6–10°C from hysteresis flexing.
# Using ambient alone underestimates true tyre temp by 15–20°C and causes
# false low-pressure alerts on correctly-inflated hot tyres.

def estimate_tyre_carcass_temp(
    ambient_temp_c: float,
    speed_kph: float,
    recent_driving_minutes: float,
) -> float:
    """Estimate tyre carcass temperature above ambient from driving conditions.

    Args:
        ambient_temp_c:           vehOutsideTemp decoded (physical °C)
        speed_kph:                current or recent average speed (physical kph)
        recent_driving_minutes:   minutes elapsed since session start
    """
    if speed_kph < 40:
        peak_rise = 6.0
    elif speed_kph < 80:
        peak_rise = 12.0
    else:
        peak_rise = 20.0

    # Heat builds toward peak over ~12 minutes of sustained driving
    time_factor = min(1.0, recent_driving_minutes / 12.0)
    return round(ambient_temp_c + peak_rise * time_factor, 1)


def correct_tyre_pressure_for_temp(
    pressure_kpa: float,
    ambient_temp_c: float,
    speed_kph: float = 0.0,
    recent_driving_minutes: float = 0.0,
    reference_temp_c: float = 25.0,
) -> float:
    """Charles's Law pressure correction using estimated carcass temperature.

    At rest (speed=0, minutes<3): uses ambient directly.
    During/after driving: uses estimated carcass temperature so hot-tyre
    readings are not falsely flagged as low pressure.

    P_ref = P_measured × (273.15 + ref_temp) / (273.15 + effective_tyre_temp)
    """
    if recent_driving_minutes < 3 or speed_kph < 10:
        effective_temp = ambient_temp_c
    else:
        effective_temp = estimate_tyre_carcass_temp(
            ambient_temp_c, speed_kph, recent_driving_minutes
        )
    return round(
        pressure_kpa * (273.15 + reference_temp_c) / (273.15 + effective_temp),
        2,
    )


def correct_tyre_pressure_monsoon_adjusted(
    pressure_kpa: float,
    ambient_temp_c: float,
    speed_kph: float,
    recent_driving_minutes: float,
    rain_intensity: float,
    reference_temp_c: float = 25.0,
) -> float:
    """Monsoon-season variant: wet road spray cools tyre surface by ~4°C.

    vehRainDetected >= 2 at speed > 40 kph = moderate-to-heavy rain on highway.
    Reduces peak_rise to prevent over-correcting pressure during monsoon drives.
    """
    rain_cooling = 4.0 if (rain_intensity >= 2 and speed_kph > 40) else 0.0
    return correct_tyre_pressure_for_temp(
        pressure_kpa,
        ambient_temp_c - rain_cooling,
        speed_kph,
        recent_driving_minutes,
        reference_temp_c,
    )


# ── ROAD ROUGHNESS STRESS INDEX ───────────────────────────────────────────────
# tboxAccelZ raw × 0.004 = g.  At rest on a flat road the Z-axis reads ≈ +1.0g
# (gravity).  A pothole or speed bump registers as a deviation from 1.0g.
# Cities like Mumbai and Kolkata can produce 30–100+ impacts per 100 km.

def detect_road_impact_events(
    df: pd.DataFrame,
    impact_threshold_g: float = 1.4,
) -> pd.DataFrame:
    """Detect pothole and speed-bump impacts from the vertical accelerometer.

    Args:
        df:                 Telemetry DataFrame (raw or normalised column names).
        impact_threshold_g: Deviation from 1.0g that constitutes a significant
                            impact. 1.4g = moderate, 2.0g = severe, 3.5g = extreme.

    Adds columns:
        is_road_impact   (bool)
        impact_severity  (int 0–3)
    """
    sort_cols = [c for c in ["vin", "timestamp"] if c in df.columns]
    df = df.sort_values(sort_cols).copy() if sort_cols else df.copy()

    # default 250 raw ≈ 1.0g at rest so deviation ≈ 0 on flat road when column missing
    accel_z_raw = _col(df, "tboxAccelZ", default=250.0)
    accel_z_g   = accel_z_raw * 0.004
    deviation   = (accel_z_g - 1.0).abs()

    speed_kph = _col(df, "vehSpeed") * 0.1

    impact_mask = (deviation > impact_threshold_g) & (speed_kph > 5)
    df["is_road_impact"] = impact_mask.fillna(False)

    df["impact_severity"] = 0
    df.loc[df["is_road_impact"] & (deviation > 1.4) & (deviation <= 2.0), "impact_severity"] = 1
    df.loc[df["is_road_impact"] & (deviation > 2.0) & (deviation <= 3.0), "impact_severity"] = 2
    df.loc[df["is_road_impact"] & (deviation > 3.0),                      "impact_severity"] = 3

    return df


def compute_road_roughness_index(
    df: pd.DataFrame,
    distance_km: float | None = None,
) -> dict[str, float | int]:
    """Summarise road roughness experienced over the provided DataFrame window.

    Args:
        df:          Telemetry DataFrame for the analysis window.
        distance_km: Pre-computed distance in km.  When provided, skips odo
                     column lookup (use this when passing a normalised df where
                     the odo unit is already known).

    Returns dict:
        road_roughness_index    0–100 (0=smooth, 100=extremely rough)
        road_stress_multiplier  1.00–1.50 applied to tyre/suspension wear models
        impacts_per_100km       raw impact rate
        severe_impacts_30d      count of severity ≥ 2 events
    """
    _null: dict[str, float | int] = {
        "road_roughness_index":   0.0,
        "road_stress_multiplier": 1.0,
        "impacts_per_100km":      0.0,
        "severe_impacts_30d":     0,
    }

    df = detect_road_impact_events(df)

    if distance_km is None:
        odo_col = next((c for c in ["odometer", "vehOdo", "VehOdo"] if c in df.columns), None)
        if odo_col is None:
            return _null
        if "vin" in df.columns:
            distance_km = float(
                df.groupby("vin")[odo_col].apply(lambda x: x.max() - x.min()).sum()
            )
        else:
            distance_km = float(df[odo_col].max() - df[odo_col].min())

    if distance_km < 10:
        return _null

    impacts_per_100km  = float(df["is_road_impact"].sum()) / distance_km * 100
    severe_impacts     = int((df["impact_severity"] >= 2).sum())

    # Benchmark: smooth highway < 5/100km; typical Indian urban 30–80/100km; very rough > 100/100km
    road_roughness_index   = min(100.0, impacts_per_100km * 0.80)
    road_stress_multiplier = 1.0 + min(0.50, impacts_per_100km / 200.0)

    return {
        "road_roughness_index":   round(road_roughness_index, 1),
        "road_stress_multiplier": round(road_stress_multiplier, 3),
        "impacts_per_100km":      round(impacts_per_100km, 1),
        "severe_impacts_30d":     severe_impacts,
    }

