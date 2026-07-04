"""
Synthetic OTA event generator.

Generates firmware update events for fleet VINs:
  - TBOX: Jan-15 1.2.3→1.3.0, Jul-15 1.3.0→1.4.0 (all VINs)
  - VCU:  Oct-01 one update/year (all VINs)
  - BMS:  Apr-01 2.1.0→2.2.0 (EV VINs only)
  - Success rate: 98% (2% random failure)
  - Post-BMS OTA: reduce avgElecConsumption by 3% for 30 days

Output: data/synthetic/ota_events.csv
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


class OTAGenerator:

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def generate(
        self,
        fleet_df: pd.DataFrame,
        start_date: str = "2024-01-01",
        num_days: int = 180,
    ) -> pd.DataFrame:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = start_dt + timedelta(days=num_days)
        rows: list[dict] = []

        for _, vrow in fleet_df.iterrows():
            vin = str(vrow["vin"])
            fuel_type = str(vrow.get("fuel_type", "ICE"))
            year = int(vrow.get("manufacture_year", 2024))

            # TBOX: Jan-15 and Jul-15
            for month, from_v, to_v in [(1, "1.2.3", "1.3.0"), (7, "1.3.0", "1.4.0")]:
                ota_dt = datetime(year if year >= 2024 else 2024, month, 15, 2, 0)
                if start_dt <= ota_dt <= end_dt:
                    rows.append(self._event(vin, "TBOX", from_v, to_v, ota_dt, f"TBOX-{year}-{month:02d}"))

            # VCU: Oct-01
            vcu_dt = datetime(year if year >= 2024 else 2024, 10, 1, 3, 0)
            if start_dt <= vcu_dt <= end_dt:
                rows.append(self._event(vin, "VCU", "3.0.0", "3.1.0", vcu_dt, f"VCU-{year}"))

            # BMS: Apr-01 (EV/PHEV only)
            if fuel_type in ("EV", "PHEV"):
                bms_dt = datetime(year if year >= 2024 else 2024, 4, 1, 4, 0)
                if start_dt <= bms_dt <= end_dt:
                    rows.append(self._event(vin, "BMS", "2.1.0", "2.2.0", bms_dt, f"BMS-{year}"))

        df = pd.DataFrame(rows)
        return df

    def _event(self, vin: str, component: str, from_v: str, to_v: str,
               start_dt: datetime, campaign_id: str) -> dict:
        duration_min = int(self.rng.integers(15, 45))
        end_dt = start_dt + timedelta(minutes=duration_min)
        success = bool(self.rng.random() < 0.98)
        return {
            "vin": vin,
            "ota_campaign_id": campaign_id,
            "component": component,
            "from_version": from_v,
            "to_version": to_v,
            "ota_start_time": start_dt.isoformat(),
            "ota_complete_time": end_dt.isoformat(),
            "ota_success": success,
        }


def generate_ota(
    fleet_df: pd.DataFrame | None = None,
    data_dir: Path | None = None,
    start_date: str = "2024-01-01",
    num_days: int = 180,
) -> pd.DataFrame:
    data_dir = data_dir or Path("data/synthetic")
    if fleet_df is None:
        for fname in ("fleet.csv", "fleet_master.csv"):
            p = data_dir / fname
            if p.exists():
                fleet_df = pd.read_csv(p)
                break
    if fleet_df is None or fleet_df.empty:
        return pd.DataFrame()

    gen = OTAGenerator()
    df = gen.generate(fleet_df, start_date, num_days)
    out_csv = data_dir / "ota_events.csv"
    df.to_csv(out_csv, index=False)
    print(f"ota_events.csv: {len(df)} events -> {out_csv}")
    return df


if __name__ == "__main__":
    generate_ota()
