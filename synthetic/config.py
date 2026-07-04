"""Synthetic data generation configuration."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DealerRecord:
    code: str
    name: str
    city: str
    region: str
    lat: float
    long: float


@dataclass
class ModelRecord:
    code: str
    name: str
    fuel_type: str        # ICE | EV | PHEV
    weight_pct: float     # distribution weight (must sum to 1.0)
    battery_kwh: float | None = None
    rated_range_km: float | None = None
    fuel_tank_l: float | None = None
    base_fuel_l100km: float | None = None  # ICE/PHEV baseline


DRIVER_ARCHETYPES: dict[str, dict] = {
    "urban_commuter": {
        "weight": 0.30, "max_speed_kph": 65, "accel_rate_kph_per_s": 4.0,
        "brake_rate_kph_per_s": 5.0, "upshift_rpm": 2400, "downshift_rpm": 1400,
        "harsh_brake_per_100km": 3.0, "harsh_accel_per_100km": 2.5,
        "idle_fraction": 0.35, "cruise_prob": 0.05,
        "trip_distance_km_min": 5, "trip_distance_km_max": 18,
        "trips_per_day": 3, "driveScore_base": 65, "driveScore_noise": 10,
    },
    "highway_cruiser": {
        "weight": 0.15, "max_speed_kph": 120, "accel_rate_kph_per_s": 3.0,
        "brake_rate_kph_per_s": 3.0, "upshift_rpm": 2200, "downshift_rpm": 1600,
        "harsh_brake_per_100km": 0.5, "harsh_accel_per_100km": 0.8,
        "idle_fraction": 0.05, "cruise_prob": 0.70,
        "trip_distance_km_min": 50, "trip_distance_km_max": 200,
        "trips_per_day": 1, "driveScore_base": 88, "driveScore_noise": 7,
    },
    "aggressive": {
        "weight": 0.15, "max_speed_kph": 140, "accel_rate_kph_per_s": 9.0,
        "brake_rate_kph_per_s": 9.0, "upshift_rpm": 3200, "downshift_rpm": 2000,
        "harsh_brake_per_100km": 7.0, "harsh_accel_per_100km": 8.0,
        "idle_fraction": 0.10, "cruise_prob": 0.05,
        "trip_distance_km_min": 10, "trip_distance_km_max": 60,
        "trips_per_day": 3, "driveScore_base": 42, "driveScore_noise": 12,
    },
    "eco_driver": {
        "weight": 0.10, "max_speed_kph": 80, "accel_rate_kph_per_s": 1.5,
        "brake_rate_kph_per_s": 2.0, "upshift_rpm": 1900, "downshift_rpm": 1200,
        "harsh_brake_per_100km": 0.3, "harsh_accel_per_100km": 0.5,
        "idle_fraction": 0.08, "cruise_prob": 0.40,
        "trip_distance_km_min": 10, "trip_distance_km_max": 50,
        "trips_per_day": 2, "driveScore_base": 93, "driveScore_noise": 5,
    },
    "taxi_fleet": {
        "weight": 0.10, "max_speed_kph": 80, "accel_rate_kph_per_s": 5.0,
        "brake_rate_kph_per_s": 6.0, "upshift_rpm": 2600, "downshift_rpm": 1500,
        "harsh_brake_per_100km": 4.0, "harsh_accel_per_100km": 3.5,
        "idle_fraction": 0.40, "cruise_prob": 0.10,
        "trip_distance_km_min": 5, "trip_distance_km_max": 25,
        "trips_per_day": 8, "driveScore_base": 58, "driveScore_noise": 10,
    },
    "delivery_driver": {
        "weight": 0.08, "max_speed_kph": 50, "accel_rate_kph_per_s": 5.5,
        "brake_rate_kph_per_s": 6.5, "upshift_rpm": 2800, "downshift_rpm": 1600,
        "harsh_brake_per_100km": 5.0, "harsh_accel_per_100km": 4.0,
        "idle_fraction": 0.50, "cruise_prob": 0.00,
        "trip_distance_km_min": 0.5, "trip_distance_km_max": 3,
        "trips_per_day": 15, "driveScore_base": 48, "driveScore_noise": 12,
    },
    "hill_region": {
        "weight": 0.07, "max_speed_kph": 60, "accel_rate_kph_per_s": 3.5,
        "brake_rate_kph_per_s": 4.0, "upshift_rpm": 2500, "downshift_rpm": 1300,
        "harsh_brake_per_100km": 2.0, "harsh_accel_per_100km": 3.0,
        "idle_fraction": 0.15, "cruise_prob": 0.00,
        "trip_distance_km_min": 5, "trip_distance_km_max": 30,
        "trips_per_day": 2, "driveScore_base": 67, "driveScore_noise": 10,
        "elevation_change_m_per_km": 15,
    },
    "elderly_cautious": {
        "weight": 0.05, "max_speed_kph": 55, "accel_rate_kph_per_s": 1.0,
        "brake_rate_kph_per_s": 2.5, "upshift_rpm": 2100, "downshift_rpm": 1100,
        "harsh_brake_per_100km": 0.5, "harsh_accel_per_100km": 0.8,
        "idle_fraction": 0.30, "cruise_prob": 0.00,
        "trip_distance_km_min": 3, "trip_distance_km_max": 20,
        "trips_per_day": 2, "driveScore_base": 75, "driveScore_noise": 8,
    },
}

assert abs(sum(v["weight"] for v in DRIVER_ARCHETYPES.values()) - 1.0) < 0.001


@dataclass
class SyntheticConfig:
    num_vehicles: int = 50
    num_days: int = 180
    start_date: str = "2024-01-01"

    # 60% ICE, 30% EV, 10% PHEV — enforced via model weight_pct
    vehicle_types: list = field(default_factory=lambda: ["ICE", "EV", "PHEV"])

    # Legacy driver profiles (kept for backward-compat with pre-F code)
    driver_profiles: list = field(default_factory=lambda: list(DRIVER_ARCHETYPES.keys()))
    driver_profile_weights: list = field(default_factory=lambda: [v["weight"] for v in DRIVER_ARCHETYPES.values()])

    failure_injection_rate: float = 0.15  # 15% of VINs get labelled failure events

    seed: int = 42

    dealers: list = field(default_factory=lambda: [
        DealerRecord("DL001", "MG Motors New Delhi",   "New Delhi",  "North",  28.61, 77.20),
        DealerRecord("DL002", "MG Motors Mumbai",      "Mumbai",     "West",   19.08, 72.88),
        DealerRecord("DL003", "MG Motors Bangalore",   "Bangalore",  "South",  12.97, 77.59),
        DealerRecord("DL004", "MG Motors Chennai",     "Chennai",    "South",  13.08, 80.27),
        DealerRecord("DL005", "MG Motors Hyderabad",   "Hyderabad",  "South",  17.38, 78.49),
        DealerRecord("DL006", "MG Motors Pune",        "Pune",       "West",   18.52, 73.86),
        DealerRecord("DL007", "MG Motors Kolkata",     "Kolkata",    "East",   22.57, 88.36),
        DealerRecord("DL008", "MG Motors Ahmedabad",   "Ahmedabad",  "West",   23.03, 72.58),
    ])

    models: list = field(default_factory=lambda: [
        ModelRecord("GLOSTER", "MG Gloster",  "ICE",  0.25, fuel_tank_l=75.0, base_fuel_l100km=12.0),
        ModelRecord("HECTOR",  "MG Hector",   "ICE",  0.35, fuel_tank_l=50.0, base_fuel_l100km=9.0),
        ModelRecord("ZSEV",    "MG ZS EV",    "EV",   0.30, battery_kwh=50.3, rated_range_km=461),
        ModelRecord("ASTOR",   "MG Astor",    "PHEV", 0.10, battery_kwh=13.0, rated_range_km=80,
                    fuel_tank_l=45.0, base_fuel_l100km=7.0),
    ])

    # Failure types that can be injected
    failure_types: list = field(default_factory=lambda: [
        "brake_degradation",
        "oil_degradation",
        "hv_battery_degradation",
        "12v_battery_failure",
        "tyre_puncture",
        "overheating",
    ])

    # Max km/day per driver profile (used for service schedule estimation)
    avg_daily_km: dict = field(default_factory=lambda: {
        "urban_commuter":   40,
        "highway_cruiser":  120,
        "aggressive":       95,
        "eco_driver":       50,
        "taxi_fleet":       100,
        "delivery_driver":  30,
        "hill_region":      40,
        "elderly_cautious": 25,
        # Legacy aliases
        "normal":           65,
        "eco":              50,
    })
