"""
Generate fleet_master.csv — 50 VINs with model, dealer, driver-profile attributes.

MG VIN format: MZ7X<model_code_2char><year_char><dealer_code_2char><6-digit-seq>
Example: MZ7XGLS4DL123456
"""
from __future__ import annotations

import random
import string
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

from synthetic.config import SyntheticConfig, DRIVER_ARCHETYPES

fake = Faker("en_IN")

# VIN year codes per WMI standard
_YEAR_CODES = {
    2020: "L", 2021: "M", 2022: "N", 2023: "P", 2024: "R", 2025: "S",
}
_COLORS = ["Pearl White", "Starry Black", "Glaze Red", "Aurora Silver", "Candy White",
           "Nordic Blue", "Evanescent Purple", "Harbour Grey"]


def _make_vin(model_code: str, year: int, dealer_code: str, seq: int) -> str:
    """Generate a realistic MG VIN."""
    mc = model_code[:2].upper()
    yr = _YEAR_CODES.get(year, "R")
    dc = dealer_code[2:4].upper()  # e.g. "DL001" → "00"
    return f"MZ7X{mc}{yr}{dc}{seq:06d}"


def generate_fleet(cfg: SyntheticConfig | None = None) -> pd.DataFrame:
    cfg = cfg or SyntheticConfig()
    rng = np.random.default_rng(cfg.seed)
    random.seed(cfg.seed)
    Faker.seed(cfg.seed)

    # Build weighted model list
    model_weights = [m.weight_pct for m in cfg.models]
    rows = []

    for idx in range(cfg.num_vehicles):
        # Pick model by weight
        model = rng.choice(cfg.models, p=model_weights)  # type: ignore[arg-type]
        dealer = rng.choice(cfg.dealers)  # type: ignore[arg-type]
        archetype_names = list(DRIVER_ARCHETYPES.keys())
        archetype_weights = [DRIVER_ARCHETYPES[n]["weight"] for n in archetype_names]
        driver_profile = str(rng.choice(archetype_names, p=archetype_weights))

        year = int(rng.integers(2021, 2025))
        seq = 100000 + idx
        vin = _make_vin(model.code, year, dealer.code, seq)

        # Initial odometer: age-based (months since manufacture × avg monthly km)
        months_old = (2024 - year) * 12 + 6
        avg_monthly_km = cfg.avg_daily_km[driver_profile] * 30
        initial_odo = float(int(months_old * avg_monthly_km * rng.uniform(0.7, 1.3)))

        row = {
            "vin":                vin,
            "model_code":         model.code,
            "model_name":         model.name,
            "fuel_type":          model.fuel_type,
            "manufacture_year":   year,
            "dealer_code":        dealer.code,
            "dealer_name":        dealer.name,
            "dealer_city":        dealer.city,
            "region":             dealer.region,
            "home_lat":           dealer.lat + float(rng.uniform(-0.5, 0.5)),
            "home_long":          dealer.long + float(rng.uniform(-0.5, 0.5)),
            "driver_profile":     driver_profile,
            "color":              rng.choice(_COLORS),
            "initial_odometer":   initial_odo,
            # EV / PHEV fields
            "battery_capacity_kwh":  model.battery_kwh,
            "rated_range_km":        model.rated_range_km,
            "fuel_tank_l":           model.fuel_tank_l,
            "base_fuel_l100km":      model.base_fuel_l100km,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    out = Path("data/synthetic")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "fleet_master.csv", index=False)
    print(f"fleet_master.csv: {len(df)} vehicles")
    return df


if __name__ == "__main__":
    generate_fleet()
