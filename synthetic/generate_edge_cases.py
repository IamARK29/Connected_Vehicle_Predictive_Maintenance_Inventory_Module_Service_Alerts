"""
Edge Case Generator for AutoPredict.

Produces corrupted/adversarial telemetry datasets to stress-test the ingestion
pipeline (validators, SignalDecoder, PhysicalConsistencyChecker).

8 modes:
  duplicate_packets        — 5% duplicate rows (same vin+timestamp)
  out_of_order_timestamps  — 3% timestamps shifted ±30 seconds
  missing_intervals        — 3 random 15-min data gaps deleted
  sensor_stuck             — vehCoolantTemp constant for 300 consecutive rows
  random_spikes            — vehSpeed=9999 + vehRPM=30000 outlier injections
  invalid_enum             — vehGearPos=14 + vehSysPwrMod=7 (out-of-spec enum)
  validity_flag_mismatch   — vehBMSPackSOCV=1 with normal SOC value (20 rows)
  brake_accel_conflict     — vehBrakePos=219 AND vehAccelPos=213 (5 rows)
"""
from __future__ import annotations

import logging
import os
import random
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DATA_DIR = Path(os.getenv("DATA_DIR", "data/synthetic"))
EC_DIR   = DATA_DIR / "edge_cases"

MODES = [
    "duplicate_packets",
    "out_of_order_timestamps",
    "missing_intervals",
    "sensor_stuck",
    "random_spikes",
    "invalid_enum",
    "validity_flag_mismatch",
    "brake_accel_conflict",
]


class EdgeCaseGenerator:

    def generate(self, base_df: pd.DataFrame, mode: str) -> pd.DataFrame:
        """
        Apply the specified corruption mode to *base_df* and return the result.

        Args:
            base_df:  Clean telemetry DataFrame (must have timestamp column).
            mode:     One of the 8 corruption modes.

        Returns:
            Corrupted DataFrame. Does NOT modify base_df in place.
        """
        if mode not in MODES:
            raise ValueError(f"Unknown mode {mode!r}. Must be one of: {MODES}")

        df   = base_df.copy()
        rng  = random.Random(42)
        np_rng = np.random.default_rng(42)

        handler = getattr(self, f"_mode_{mode}")
        return handler(df, rng, np_rng)

    # ── Mode implementations ──────────────────────────────────────────────────

    def _mode_duplicate_packets(self, df, rng, np_rng):
        n_dup = max(1, int(len(df) * 0.05))
        dupes = df.sample(n=n_dup, random_state=42)
        return pd.concat([df, dupes], ignore_index=True)

    def _mode_out_of_order_timestamps(self, df, rng, np_rng):
        if "timestamp" not in df.columns:
            return df
        df = df.copy()
        ts = df["timestamp"]
        if pd.api.types.is_numeric_dtype(ts):
            df["timestamp"] = pd.to_datetime(ts, unit="s", utc=True)
        else:
            df["timestamp"] = pd.to_datetime(ts, utc=True)
        n_shift = max(1, int(len(df) * 0.03))
        idx     = np_rng.choice(df.index, size=n_shift, replace=False)
        shifts  = [timedelta(seconds=int(s)) for s in np_rng.integers(-30, 30, size=n_shift)]
        for i, delta in zip(idx, shifts):
            df.at[i, "timestamp"] = df.at[i, "timestamp"] + delta
        return df

    def _mode_missing_intervals(self, df, rng, np_rng):
        if "timestamp" not in df.columns:
            return df
        df = df.copy()
        ts = df["timestamp"]
        if pd.api.types.is_numeric_dtype(ts):
            df["timestamp"] = pd.to_datetime(ts, unit="s", utc=True)
        else:
            df["timestamp"] = pd.to_datetime(ts, utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
        if len(df) < 100:
            return df

        ts_min = df["timestamp"].min()
        ts_max = df["timestamp"].max()
        span   = (ts_max - ts_min).total_seconds()

        mask = pd.Series(True, index=df.index)
        for _ in range(3):
            gap_start_s = rng.uniform(0, max(1, span - 900))
            gap_start   = ts_min + timedelta(seconds=gap_start_s)
            gap_end     = gap_start + timedelta(minutes=15)
            gap_mask    = (df["timestamp"] >= gap_start) & (df["timestamp"] < gap_end)
            mask &= ~gap_mask

        # For sparse data (1 row per hour+), timestamp gaps may hit zero rows.
        # Guarantee at least 5% removal by random drop as fallback.
        if mask.all():
            drop_n = max(1, len(df) // 20)
            drop_idx = np_rng.choice(df.index, size=drop_n, replace=False)
            mask.iloc[drop_idx] = False

        return df[mask].reset_index(drop=True)

    def _mode_sensor_stuck(self, df, rng, np_rng):
        df = df.copy()
        col = "vehCoolantTemp" if "vehCoolantTemp" in df.columns else \
              ("coolant_temp" if "coolant_temp" in df.columns else None)
        if col is None or len(df) < 300:
            return df
        start = rng.randint(0, len(df) - 300)
        stuck_val = float(df[col].iloc[start])
        df.iloc[start:start + 300, df.columns.get_loc(col)] = stuck_val
        return df

    def _mode_random_spikes(self, df, rng, np_rng):
        df = df.copy()
        spike_rows = np_rng.choice(df.index, size=min(5, len(df)), replace=False)
        if "vehSpeed" in df.columns:
            df.loc[spike_rows, "vehSpeed"] = 9999
        elif "speed" in df.columns:
            df.loc[spike_rows, "speed"] = 9999

        spike_rows2 = np_rng.choice(df.index, size=min(5, len(df)), replace=False)
        if "vehRPM" in df.columns:
            df.loc[spike_rows2, "vehRPM"] = 30000
        elif "rpm" in df.columns:
            df.loc[spike_rows2, "rpm"] = 30000
        return df

    def _mode_invalid_enum(self, df, rng, np_rng):
        df = df.copy()
        rows1 = np_rng.choice(df.index, size=min(5, len(df)), replace=False)
        rows2 = np_rng.choice(df.index, size=min(5, len(df)), replace=False)
        gear_col = "vehGearPos" if "vehGearPos" in df.columns else \
                   ("gear_pos" if "gear_pos" in df.columns else None)
        pwr_col  = "vehSysPwrMod" if "vehSysPwrMod" in df.columns else \
                   ("sys_pwr_mod" if "sys_pwr_mod" in df.columns else None)
        if gear_col:
            df.loc[rows1, gear_col] = 14
        if pwr_col:
            df.loc[rows2, pwr_col] = 7
        return df

    def _mode_validity_flag_mismatch(self, df, rng, np_rng):
        df = df.copy()
        flag_col = "vehBMSPackSOCV"
        soc_col  = "vehBMSPackSOC" if "vehBMSPackSOC" in df.columns else \
                   ("soc" if "soc" in df.columns else None)
        if soc_col is None:
            return df
        if flag_col not in df.columns:
            df[flag_col] = 0
        rows = np_rng.choice(df.index, size=min(20, len(df)), replace=False)
        df.loc[rows, flag_col] = 1   # mark invalid even though SOC value is present
        return df

    def _mode_brake_accel_conflict(self, df, rng, np_rng):
        df = df.copy()
        rows = np_rng.choice(df.index, size=min(5, len(df)), replace=False)
        brake_col = "vehBrakePos" if "vehBrakePos" in df.columns else \
                    ("brake_pos" if "brake_pos" in df.columns else None)
        accel_col = "vehAccelPos" if "vehAccelPos" in df.columns else \
                    ("accel_pos" if "accel_pos" in df.columns else None)
        if brake_col:
            df.loc[rows, brake_col] = 219   # 219 × 0.4 = 87.6%
        if accel_col:
            df.loc[rows, accel_col] = 213   # 213 × 0.4 = 85.2%
        return df


def generate_all(base_df: pd.DataFrame, output_dir: Path | None = None) -> dict[str, Path]:
    """Generate all 8 edge-case variants and save CSVs."""
    out_dir = output_dir or EC_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    gen   = EdgeCaseGenerator()
    paths = {}
    for mode in MODES:
        try:
            corrupted = gen.generate(base_df, mode)
            p = out_dir / f"ec_{mode}.csv"
            corrupted.to_csv(p, index=False)
            paths[mode] = p
            log.info("Edge case %s -> %d rows -> %s", mode, len(corrupted), p)
        except Exception as exc:
            log.error("Edge case %s failed: %s", mode, exc)
    return paths


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    src = DATA_DIR / "telemetry_combined.csv"
    if not src.exists():
        sys.exit(f"Source file not found: {src}  (run generate_telemetry.py first)")

    base_df = pd.read_csv(src, nrows=5000)
    paths   = generate_all(base_df)
    for mode, p in paths.items():
        print(f"  {mode:<30s}  {p}")
