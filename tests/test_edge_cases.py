"""
Edge case tests for the ingestion pipeline.

Generates each of the 8 corruption modes and verifies:
  1. The corruption is actually present in the output
  2. The validators/decoder correctly flag or reject the corrupted data
  3. The output CSV is written to data/synthetic/edge_cases/ec_{mode}.csv

If data/synthetic/telemetry_combined.csv does not exist, a minimal synthetic
DataFrame is created in-memory so tests can run without prior generate_telemetry.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("POSTGRES_URL",   "sqlite:///./test_autopredict.db")
os.environ.setdefault("INFLUXDB_URL",   "http://localhost:8086")
os.environ.setdefault("INFLUXDB_TOKEN", "autopredict-dev-token")
os.environ.setdefault("REDIS_URL",      "redis://localhost:6379/0")
os.environ.setdefault("SECRET_KEY",     "test-secret-key")

from synthetic.generate_edge_cases import EdgeCaseGenerator, MODES, EC_DIR
from ingestion.signal_registry import SignalDecoder
from ingestion.validators import TelemetryValidator, PhysicalConsistencyChecker


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_base_df(n: int = 500) -> pd.DataFrame:
    """Build a minimal but realistic telemetry DataFrame for edge case injection."""
    rng = np.random.default_rng(0)
    now = datetime.now(timezone.utc)
    ts  = [now + timedelta(seconds=i * 10) for i in range(n)]
    return pd.DataFrame({
        "timestamp":       ts,
        "vin":             "MH01TEST0001",
        # Raw (pre-physical) values matching the Big Data Spec
        "vehSpeed":        rng.integers(0, 800,  size=n).astype(float),    # raw ×0.1 = kph
        "vehRPM":          rng.integers(700, 3000, size=n).astype(float),
        "vehSysPwrMod":    np.full(n, 2),                                  # enum 2=Run
        "vehGearPos":      rng.integers(1, 6, size=n).astype(float),
        "vehBrakePos":     rng.integers(0, 50,  size=n).astype(float),     # raw; ×0.4 = %
        "vehAccelPos":     rng.integers(0, 100, size=n).astype(float),
        "vehCoolantTemp":  rng.integers(75, 100, size=n).astype(float),
        "vehBMSPackSOC":   rng.integers(200, 900, size=n).astype(float),   # raw ×0.1 = %
        "vehBMSPackSOCV":  np.zeros(n),                                    # validity flag
    })


@pytest.fixture(scope="module")
def base_df() -> pd.DataFrame:
    src = ROOT / "data" / "synthetic" / "telemetry_combined.csv"
    if src.exists():
        df = pd.read_csv(src, nrows=1000)
        if df.empty:
            return _make_base_df()
        # Ensure timestamp column exists
        if "timestamp" not in df.columns and "StartTime-TimeStamp" in df.columns:
            df = df.rename(columns={"StartTime-TimeStamp": "timestamp"})
        return df
    return _make_base_df()


@pytest.fixture(scope="module")
def gen() -> EdgeCaseGenerator:
    return EdgeCaseGenerator()


# ── Helper ────────────────────────────────────────────────────────────────────

def _save(df: pd.DataFrame, mode: str) -> Path:
    EC_DIR.mkdir(parents=True, exist_ok=True)
    p = EC_DIR / f"ec_{mode}.csv"
    df.to_csv(p, index=False)
    return p


# ── Mode tests ────────────────────────────────────────────────────────────────

def test_duplicate_packets_corruption_present(gen, base_df):
    corrupted = gen.generate(base_df, "duplicate_packets")
    assert len(corrupted) > len(base_df), "Duplicates should increase row count"
    assert len(corrupted) <= len(base_df) * 1.10, "Should be approx 5% more rows"
    p = _save(corrupted, "duplicate_packets")
    assert p.exists()


def test_duplicate_packets_detected_by_dedup(gen, base_df):
    corrupted = gen.generate(base_df, "duplicate_packets")
    if "timestamp" in corrupted.columns and "vin" in corrupted.columns:
        dupes = corrupted.duplicated(subset=["vin", "timestamp"], keep=False)
        assert dupes.sum() > 0, "No duplicate vin+timestamp pairs found in output"


def test_out_of_order_timestamps_corruption_present(gen, base_df):
    if "timestamp" not in base_df.columns:
        pytest.skip("No timestamp column in base_df")
    corrupted = gen.generate(base_df, "out_of_order_timestamps")
    orig_ts = pd.to_datetime(base_df["timestamp"]).reset_index(drop=True)
    new_ts  = pd.to_datetime(corrupted["timestamp"]).reset_index(drop=True)
    assert not orig_ts.equals(new_ts), "Timestamps should have changed"
    p = _save(corrupted, "out_of_order_timestamps")
    assert p.exists()


def test_missing_intervals_reduces_rows(gen, base_df):
    if "timestamp" not in base_df.columns:
        pytest.skip("No timestamp column in base_df")
    corrupted = gen.generate(base_df, "missing_intervals")
    assert len(corrupted) < len(base_df), "Missing intervals should reduce row count"
    p = _save(corrupted, "missing_intervals")
    assert p.exists()


def test_sensor_stuck_has_constant_run(gen, base_df):
    corrupted = gen.generate(base_df, "sensor_stuck")
    col = "vehCoolantTemp" if "vehCoolantTemp" in corrupted.columns else \
          ("coolant_temp" if "coolant_temp" in corrupted.columns else None)
    if col:
        values = corrupted[col].values
        # Find a run of ≥ 100 identical values
        found = False
        for i in range(len(values) - 100):
            if len(set(values[i:i + 100])) == 1:
                found = True
                break
        assert found, f"Expected 300-row stuck run in {col}"
    p = _save(corrupted, "sensor_stuck")
    assert p.exists()


def test_random_spikes_in_speed(gen, base_df):
    corrupted = gen.generate(base_df, "random_spikes")
    speed_col = "vehSpeed" if "vehSpeed" in corrupted.columns else \
                ("speed" if "speed" in corrupted.columns else None)
    if speed_col:
        assert (corrupted[speed_col] == 9999).any(), "Expected vehSpeed=9999 spike rows"
    p = _save(corrupted, "random_spikes")
    assert p.exists()


def test_random_spikes_rejected_by_decoder(gen, base_df):
    corrupted = gen.generate(base_df, "random_spikes")
    if "vehSpeed" in corrupted.columns:
        spike_rows = corrupted[corrupted["vehSpeed"] == 9999]
        for _, row in spike_rows.head(3).iterrows():
            result = SignalDecoder.decode("vehSpeed", float(row["vehSpeed"]))
            assert result is None, f"Spike vehSpeed=9999 should decode to None, got {result}"


def test_invalid_enum_values_present(gen, base_df):
    corrupted = gen.generate(base_df, "invalid_enum")
    gear_col = "vehGearPos" if "vehGearPos" in corrupted.columns else \
               ("gear_pos" if "gear_pos" in corrupted.columns else None)
    pwr_col  = "vehSysPwrMod" if "vehSysPwrMod" in corrupted.columns else \
               ("sys_pwr_mod" if "sys_pwr_mod" in corrupted.columns else None)
    if gear_col:
        assert (corrupted[gear_col] == 14).any(), "Expected vehGearPos=14 injection"
    if pwr_col:
        assert (corrupted[pwr_col] == 7).any(), "Expected vehSysPwrMod=7 injection"
    p = _save(corrupted, "invalid_enum")
    assert p.exists()


def test_invalid_enum_rejected_by_validator(gen, base_df):
    corrupted = gen.generate(base_df, "invalid_enum")
    validator = TelemetryValidator()
    # CH-3 defines vehSysPwrMod with enum_values=[0,1,2,3]; injected value 7 is invalid
    if "vehSysPwrMod" in corrupted.columns:
        bad_rows = corrupted[corrupted["vehSysPwrMod"] == 7].head(3)
        for _, row in bad_rows.iterrows():
            payload = {"vehSysPwrMod": int(row["vehSysPwrMod"])}
            ok, _, errors, _ = validator.validate(3, payload)
            assert not ok or any("vehSysPwrMod" in e for e in errors), \
                f"vehSysPwrMod=7 should fail ch3 validation, got ok={ok}, errors={errors}"


def test_validity_flag_mismatch_present(gen, base_df):
    corrupted = gen.generate(base_df, "validity_flag_mismatch")
    assert "vehBMSPackSOCV" in corrupted.columns, "Flag column should exist"
    assert (corrupted["vehBMSPackSOCV"] == 1).sum() >= 1, "Expected ≥1 row with flag=1"
    p = _save(corrupted, "validity_flag_mismatch")
    assert p.exists()


def test_validity_flag_gated_by_decoder(gen, base_df):
    corrupted = gen.generate(base_df, "validity_flag_mismatch")
    soc_col   = "vehBMSPackSOC" if "vehBMSPackSOC" in corrupted.columns else None
    if soc_col is None:
        pytest.skip("vehBMSPackSOC column not in base_df")

    gated_rows = corrupted[corrupted["vehBMSPackSOCV"] == 1].head(5)
    for _, row in gated_rows.iterrows():
        result = SignalDecoder.decode_row(dict(row))
        assert result.get("vehBMSPackSOC") is None, \
            "Gated SOC should decode to None when vehBMSPackSOCV==1"


def test_brake_accel_conflict_present(gen, base_df):
    corrupted = gen.generate(base_df, "brake_accel_conflict")
    brake_col = "vehBrakePos" if "vehBrakePos" in corrupted.columns else \
                ("brake_pos" if "brake_pos" in corrupted.columns else None)
    accel_col = "vehAccelPos" if "vehAccelPos" in corrupted.columns else \
                ("accel_pos" if "accel_pos" in corrupted.columns else None)
    if brake_col and accel_col:
        conflict = (corrupted[brake_col] == 219) & (corrupted[accel_col] == 213)
        assert conflict.any(), "Expected brake=219 + accel=213 conflict rows"
    p = _save(corrupted, "brake_accel_conflict")
    assert p.exists()


def test_brake_accel_conflict_detected_by_checker(gen, base_df):
    corrupted    = gen.generate(base_df, "brake_accel_conflict")
    checker      = PhysicalConsistencyChecker()
    brake_col    = "vehBrakePos" if "vehBrakePos" in corrupted.columns else \
                   ("brake_pos" if "brake_pos" in corrupted.columns else None)
    accel_col    = "vehAccelPos" if "vehAccelPos" in corrupted.columns else \
                   ("accel_pos" if "accel_pos" in corrupted.columns else None)

    if brake_col is None or accel_col is None:
        pytest.skip("brake/accel columns not present")

    conflict_rows = corrupted[(corrupted[brake_col] == 219) & (corrupted[accel_col] == 213)]
    found_violation = False
    for _, row in conflict_rows.head(5).iterrows():
        d = dict(row)
        violations = checker.check(d)
        if "BRAKE_ACCEL_CONFLICT" in violations:
            found_violation = True
            break
    assert found_violation, "PhysicalConsistencyChecker should flag BRAKE_ACCEL_CONFLICT"


# ── All modes generate and save ────────────────────────────────────────────────

@pytest.mark.parametrize("mode", MODES)
def test_all_modes_save_csv(gen, base_df, mode):
    corrupted = gen.generate(base_df, mode)
    assert not corrupted.empty, f"Mode {mode!r} produced empty DataFrame"
    p = _save(corrupted, mode)
    assert p.exists()
    assert p.stat().st_size > 0


# ── PhysicalConsistencyChecker unit tests ──────────────────────────────────────

def test_checker_brake_accel_conflict():
    checker = PhysicalConsistencyChecker()
    # brake=219→87.6%, accel=213→85.2%  both > 30 → conflict
    row = {"vehBrakePos": 219, "vehAccelPos": 213, "vehRPM": 2000, "vehSysPwrMod": 2}
    assert "BRAKE_ACCEL_CONFLICT" in checker.check(row)


def test_checker_no_conflict_normal():
    checker = PhysicalConsistencyChecker()
    row = {"vehBrakePos": 20, "vehAccelPos": 100, "vehRPM": 2000, "vehSysPwrMod": 2}
    assert "BRAKE_ACCEL_CONFLICT" not in checker.check(row)


def test_checker_speed_rpm_mismatch():
    checker = PhysicalConsistencyChecker()
    # speed=0, rpm=2500 while mode=Run → mismatch
    row = {"vehSpeed": 0, "vehRPM": 2500, "vehSysPwrMod": 2,
           "vehBrakePos": 0, "vehAccelPos": 0}
    assert "SPEED_RPM_MISMATCH" in checker.check(row)


def test_checker_high_speed_low_gear():
    checker = PhysicalConsistencyChecker()
    # speed=1300 raw (130 kph), gear=1 → implausible
    row = {"vehSpeed": 1300, "vehRPM": 5000, "vehSysPwrMod": 2,
           "vehGearPos": 1, "vehBrakePos": 0, "vehAccelPos": 200}
    assert "HIGH_SPEED_LOW_GEAR" in checker.check(row)


def test_checker_clean_row_no_violations():
    checker = PhysicalConsistencyChecker()
    row = {"vehSpeed": 600, "vehRPM": 2500, "vehSysPwrMod": 2,
           "vehGearPos": 4, "vehBrakePos": 0, "vehAccelPos": 120}
    assert checker.check(row) == []
