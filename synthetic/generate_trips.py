"""
Synthetic Trip Aggregator.

Reads per-VIN telemetry CSVs from data/synthetic/ and aggregates them
into trip-level records matching the TRIP_COL_MAP schema expected by
ingestion/file_ingestor.py.

Each driving session (detected by VehSysPwrMod transitions) becomes one
trip record with all computed drive-score components.
"""
from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from synthetic.config import SyntheticConfig, DRIVER_ARCHETYPES


# Drive-score component weights (must sum to 1.0)
_SCORE_WEIGHTS = {
    "speed_compliance":  0.25,
    "smooth_accel":      0.20,
    "smooth_brake":      0.20,
    "harsh_braking":     0.15,
    "idle_fraction":     0.10,
    "overspeed_80":      0.05,
    "overspeed_120":     0.05,
}


def generate_trips(
    fleet_df: pd.DataFrame | None = None,
    cfg: SyntheticConfig | None = None,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    cfg      = cfg or SyntheticConfig()
    data_dir = data_dir or Path("data/synthetic")

    if fleet_df is None:
        fleet_df = pd.read_csv(data_dir / "fleet_master.csv")

    all_trips: list[dict] = []

    for _, vrow in fleet_df.iterrows():
        vin       = str(vrow["vin"])
        fuel_type = str(vrow["fuel_type"])
        tel_csv   = data_dir / f"telemetry_{vin}.csv"

        if not tel_csv.exists():
            print(f"  [trips] WARNING: {tel_csv.name} not found, skipping")
            continue

        df = pd.read_csv(tel_csv, low_memory=False)
        trips = _extract_trips(df, vin, vrow, fuel_type)
        all_trips.extend(trips)
        print(f"  {vin}: {len(trips)} trips")

    result = pd.DataFrame(all_trips)
    out_csv = data_dir / "trips.csv"
    result.to_csv(out_csv, index=False)
    print(f"trips.csv: {len(result)} records -> {out_csv}")
    return result


def _extract_trips(df: pd.DataFrame, vin: str, vrow: pd.Series, fuel_type: str) -> list[dict]:
    """Segment a per-VIN telemetry DataFrame into trip records."""
    trips: list[dict] = []

    pwr_col = next((c for c in ("vehSysPwrMod", "VehSysPwrMod") if c in df.columns), None)
    if pwr_col is None or len(df) < 10:
        return trips

    # Detect session boundaries: transitions from non-2 to 2 (start) and 2 to non-2 (end)
    pwr = df[pwr_col].fillna(0).astype(int).to_numpy()
    running = (pwr == 2) | (pwr == 3)

    # Find contiguous running blocks
    starts = np.where(np.diff(running.astype(int), prepend=0) == 1)[0]
    ends   = np.where(np.diff(running.astype(int), prepend=0) == -1)[0]

    # Ensure paired
    if len(ends) == 0 or (len(starts) > 0 and starts[0] > ends[0] if len(ends) > 0 else False):
        ends = np.append(ends, len(running))
    if len(starts) > len(ends):
        ends = np.append(ends, len(running))
    if len(ends) > len(starts):
        starts = np.insert(starts, 0, 0)

    n_pairs = min(len(starts), len(ends))
    for i in range(n_pairs):
        seg = df.iloc[starts[i]:ends[i]]
        if len(seg) < 5:  # skip tiny segments (< 5 rows at any sample interval)
            continue
        trip = _compute_trip(seg, vin, vrow, fuel_type)
        if trip:
            trips.append(trip)

    return trips


def _compute_trip(seg: pd.DataFrame, vin: str, vrow: pd.Series, fuel_type: str) -> dict | None:
    """Compute all trip fields from a single session segment."""
    # Column name resolution — telemetry uses camelCase
    def _col(*candidates):
        for c in candidates:
            if c in seg.columns:
                return c
        return None

    n = len(seg)
    spd_col   = _col("vehSpeed",        "VehSpeed")
    odo_col   = _col("vehOdo",          "VehOdo")
    ts_col    = _col("StartTime-TimeStamp")
    lat_col   = _col("gnssLat",         "GNSSLat")
    lon_col   = _col("gnssLong",        "GNSSLong")
    fuel_col  = _col("vehFuelConsumed", "VehFuelConsumed")
    brake_col = _col("vehBrakePos",     "VehBrakePos")
    steer_col = _col("vehSteeringAngle","VehSteeringAngle")
    accel_col = _col("vehAccelPos",     "VehAccelPos")

    speed = seg[spd_col].fillna(0).to_numpy(dtype=float) if spd_col else np.zeros(n)
    odo   = seg[odo_col].ffill().to_numpy(dtype=float)   if odo_col else np.zeros(n)

    start_odo = float(odo[0])
    end_odo   = float(odo[-1])
    trip_km   = max(0.0, end_odo - start_odo)

    if trip_km < 0.1:
        return None  # parked-only segment

    avg_speed = float(np.mean(speed[speed > 1])) if np.any(speed > 1) else 0.0
    max_speed = float(np.max(speed))

    # Timestamps
    if ts_col:
        start_time = pd.to_datetime(seg[ts_col].iloc[0],  unit="s", utc=True).isoformat()
        end_time   = pd.to_datetime(seg[ts_col].iloc[-1], unit="s", utc=True).isoformat()
    else:
        start_time = end_time = ""

    # GPS
    start_lat  = float(seg[lat_col].iloc[0])  if lat_col else 0.0
    start_long = float(seg[lon_col].iloc[0])  if lon_col else 0.0
    end_lat    = float(seg[lat_col].iloc[-1]) if lat_col else 0.0
    end_long   = float(seg[lon_col].iloc[-1]) if lon_col else 0.0

    # Fuel consumed (ICE/PHEV only)
    fuel_consumed = 0.0
    if fuel_col and fuel_type in ("ICE", "PHEV"):
        fc = seg[fuel_col].fillna(0).to_numpy(dtype=float)
        fuel_consumed = max(0.0, float(fc[-1] - fc[0]))
    fuel_eff = round(fuel_consumed / trip_km * 100, 2) if trip_km > 0 and fuel_consumed > 0 else 0.0

    # Over-speed events
    over_80    = int(np.sum(speed > 80))
    over_120   = int(np.sum(speed > 120))
    over_total = int(np.sum(speed > 80))

    # Harsh braking: brake_pos > 70 while speed > 20
    harsh_braking = 0
    bp = seg[brake_col].fillna(0).to_numpy(dtype=float) if brake_col else np.zeros(n)
    if brake_col:
        harsh_braking = int(np.sum((bp > 70) & (speed > 20)))

    # Sudden turns: steering rate > threshold (scaled for sample interval)
    sudden_turns = 0
    if steer_col:
        steer = seg[steer_col].fillna(0).to_numpy(dtype=float)
        steer_rate = np.abs(np.gradient(steer))
        sudden_turns = int(np.sum(steer_rate > 5))  # 5°/step (was 30°/s at 1Hz)

    # Rapid acceleration events
    accel_events = 0
    ap = seg[accel_col].fillna(0).to_numpy(dtype=float) if accel_col else np.zeros(n)
    if accel_col:
        accel_events = int(np.sum(np.abs(np.gradient(ap)) > 15))

    # Idle fraction
    idle_pct = float(np.sum(speed < 2) / n)

    # Drive score (7 components)
    drive_score = _compute_drive_score(
        speed=speed,
        brake_pos=bp,
        accel_pos=ap,
        harsh_braking=harsh_braking,
        idle_fraction=idle_pct,
        over_80=over_80,
        over_120=over_120,
        n=n,
    )

    # ── Archetype-aware drive score ──────────────────────────────────────
    driver_profile = str(vrow.get("driver_profile", "urban_commuter"))
    archetype = DRIVER_ARCHETYPES.get(driver_profile, DRIVER_ARCHETYPES.get("urban_commuter", {}))
    arch_score = float(np.clip(
        archetype.get("driveScore_base", 65) + np.random.normal(0, archetype.get("driveScore_noise", 10)),
        0, 100,
    ))
    final_score = round((drive_score + arch_score) / 2, 1)

    # ── Contextual columns ───────────────────────────────────────────────
    if driver_profile == "delivery_driver":
        road_type = "urban"
    elif driver_profile == "highway_cruiser":
        road_type = "highway"
    else:
        road_type = "mixed"

    stop_go_ratio = round(archetype.get("idle_fraction", 0.15), 3)
    elev = archetype.get("elevation_change_m_per_km", 0)

    dealer_city = str(vrow.get("dealer_city", ""))
    region = str(vrow.get("region", ""))
    month = 1
    if start_time:
        try:
            month = int(start_time[5:7])
        except Exception:
            pass

    rain_int = 0
    if region in ("West", "South") and 6 <= month <= 9:
        rain_int = int(np.random.choice([0, 1, 2, 3], p=[0.3, 0.3, 0.25, 0.15]))

    if "Delhi" in dealer_city and month in (5, 6):
        thermal = "extreme"
    elif "Mumbai" in dealer_city:
        thermal = "hot"
    elif "Bangalore" in dealer_city:
        thermal = "moderate"
    elif "Chennai" in dealer_city:
        thermal = "hot"
    else:
        thermal = "moderate"

    load_proxy = 0.35 if driver_profile == "hill_region" else 0.0

    return {
        "tripId":           str(uuid.uuid4()),
        "vin":              vin,
        "startTime":        start_time,
        "endTime":          end_time,
        "startOdometer":    round(start_odo, 2),
        "endOdometer":      round(end_odo, 2),
        "odometer":         round(trip_km, 2),
        "averageSpeed":     round(avg_speed, 1),
        "maxSpeed":         round(max_speed, 1),
        "vehFuelConsumed":  round(fuel_consumed, 3),
        "fuelEfficiency":   fuel_eff,
        "driveScore":       round(final_score, 1),
        "powerMode":        2,
        "overSpeedNum":     over_total,
        "overSpeed80":      over_80,
        "overSpeed120":     over_120,
        "harshBreakingNum": harsh_braking,
        "suddenTurnNum":    sudden_turns,
        "accelerationNum":  accel_events,
        "startPoint_lat":   round(start_lat, 6),
        "startPoint_long":  round(start_long, 6),
        "endPoint_lat":     round(end_lat, 6),
        "endPoint_long":    round(end_long, 6),
        "road_type":                road_type,
        "stop_go_ratio":            stop_go_ratio,
        "elevation_change_m_per_km": elev,
        "rain_intensity":           rain_int,
        "thermal_zone":             thermal,
        "load_condition_proxy":     load_proxy,
    }


def _compute_drive_score(
    speed: np.ndarray,
    brake_pos: np.ndarray,
    accel_pos: np.ndarray,
    harsh_braking: int,
    idle_fraction: float,
    over_80: int,
    over_120: int,
    n: int,
) -> float:
    """Seven-component drive score, 0-100 (higher = safer/more efficient)."""
    # 1. Speed compliance (0=all over limit, 100=no over-limit seconds)
    speed_compliance = max(0.0, 100 - (over_80 / max(n, 1)) * 1000)

    # 2. Smooth acceleration (low rate of change in accel pedal)
    accel_changes = np.abs(np.gradient(accel_pos))
    smooth_accel  = max(0.0, 100 - float(np.mean(accel_changes)) * 5)

    # 3. Smooth braking (low rate of change in brake pedal)
    brake_changes = np.abs(np.gradient(brake_pos))
    smooth_brake  = max(0.0, 100 - float(np.mean(brake_changes)) * 5)

    # 4. Harsh braking penalty
    harsh_score = max(0.0, 100 - harsh_braking * 8)

    # 5. Idle fraction penalty (high idle = poor efficiency)
    idle_score = max(0.0, 100 - idle_fraction * 200)

    # 6. Over 80 kph penalty
    over80_score = max(0.0, 100 - (over_80 / max(n, 1)) * 500)

    # 7. Over 120 kph penalty
    over120_score = max(0.0, 100 - (over_120 / max(n, 1)) * 2000)

    components = {
        "speed_compliance": speed_compliance,
        "smooth_accel":     smooth_accel,
        "smooth_brake":     smooth_brake,
        "harsh_braking":    harsh_score,
        "idle_fraction":    idle_score,
        "overspeed_80":     over80_score,
        "overspeed_120":    over120_score,
    }

    score = sum(_SCORE_WEIGHTS[k] * v for k, v in components.items())
    return float(np.clip(score, 0, 100))


if __name__ == "__main__":
    generate_trips()
