"""Tests for feature engineering pipelines.

Each pipeline is tested against a 100-row synthetic DataFrame.
Assertions:
  - compute() returns a non-empty DataFrame
  - Output has at least one row
  - No NaN values in numeric columns (except known optional fields)
  - Column names match the expected schema
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.brake_features import BrakeFeaturePipeline
from features.battery_12v_features import Battery12VFeaturePipeline
from features.battery_hv_features import BatteryHVFeaturePipeline
from features.engine_features import EngineFeaturePipeline
from features.tyre_features import TyreFeaturePipeline
from features.driver_behaviour_features import DriverBehaviourFeaturePipeline

VIN = "MH01MZ7X0001"

# Columns allowed to be NaN (optional or missing from synthetic df)
_ALLOWED_NAN = {
    "days_to_brake_replacement",
    "brake_replacement_within_30_days",
    "km_since_last_brake_service",
    "days_to_engine_oil_change",
    "engine_oil_within_30_days",
    "days_to_battery_12v_failure",
    "battery_12v_within_30_days",
    "days_to_hv_failure",
    "hv_within_30_days",
    "days_to_tyre_replacement",
    "tyre_within_30_days",
}


def _check_output(result: pd.DataFrame, expected_cols: list[str]) -> None:
    assert not result.empty, "Pipeline returned empty DataFrame"
    assert len(result) >= 1

    # All expected feature columns present
    missing = [c for c in expected_cols if c not in result.columns]
    assert not missing, f"Missing columns: {missing}"

    # No unexpected NaN in non-label numeric columns
    for col in result.select_dtypes(include=[np.number]).columns:
        if col in _ALLOWED_NAN:
            continue
        n_nan = result[col].isna().sum()
        assert n_nan == 0, f"Unexpected NaN in column '{col}' ({n_nan} rows)"


# ── Brake ──────────────────────────────────────────────────────────────────────

BRAKE_EXPECTED_COLS = [
    "vin",
    "brake_stress_cumulative",
    "harsh_brake_rate_7d",
    "harsh_brake_rate_30d",
    "high_speed_stop_count_30d",
    "avg_brake_intensity_7d",
    "deceleration_g_95th_30d",
    "brake_heat_proxy",
]


def test_brake_pipeline_basic(telemetry_df):
    pipe = BrakeFeaturePipeline()
    result = pipe.compute(VIN, telemetry_df)
    _check_output(result, BRAKE_EXPECTED_COLS)


def test_brake_pipeline_vin_in_output(telemetry_df):
    pipe = BrakeFeaturePipeline()
    result = pipe.compute(VIN, telemetry_df)
    assert result["vin"].iloc[0] == VIN


def test_brake_pipeline_empty_df():
    pipe = BrakeFeaturePipeline()
    result = pipe.compute(VIN, pd.DataFrame())
    assert result.empty


def test_brake_pipeline_no_timestamp():
    pipe = BrakeFeaturePipeline()
    df = pd.DataFrame({"speed": [10, 20], "brake_pos": [5, 10]})
    result = pipe.compute(VIN, df)
    assert result.empty


def test_brake_pipeline_stress_nonnegative(telemetry_df):
    pipe = BrakeFeaturePipeline()
    result = pipe.compute(VIN, telemetry_df)
    assert result["brake_stress_cumulative"].iloc[0] >= 0


# ── Battery 12V ───────────────────────────────────────────────────────────────

BATT12V_EXPECTED_COLS = [
    "vin",
    "batt_12v_mean_7d",
    "batt_12v_min_7d",
]


def test_battery_12v_pipeline_basic(telemetry_df):
    pipe = Battery12VFeaturePipeline()
    result = pipe.compute(VIN, telemetry_df)
    _check_output(result, BATT12V_EXPECTED_COLS)


def test_battery_12v_voltage_in_range(telemetry_df):
    pipe = Battery12VFeaturePipeline()
    result = pipe.compute(VIN, telemetry_df)
    mean_v = result["batt_12v_mean_7d"].iloc[0]
    assert 0 < mean_v < 20, f"12V mean out of range: {mean_v}"


def test_battery_12v_empty_df():
    pipe = Battery12VFeaturePipeline()
    result = pipe.compute(VIN, pd.DataFrame())
    assert result.empty


# ── HV Battery ────────────────────────────────────────────────────────────────

HV_EXPECTED_COLS = [
    "vin",
    "soc_mean_7d",
    "soh_mean",
]


def test_hv_battery_pipeline_basic(telemetry_df):
    pipe = BatteryHVFeaturePipeline()
    result = pipe.compute(VIN, telemetry_df)
    _check_output(result, HV_EXPECTED_COLS)


def test_hv_battery_soc_in_range(telemetry_df):
    pipe = BatteryHVFeaturePipeline()
    result = pipe.compute(VIN, telemetry_df)
    soc = result["soc_mean_7d"].iloc[0]
    assert 0 <= soc <= 100, f"SOC mean out of range: {soc}"


def test_hv_battery_empty_df():
    pipe = BatteryHVFeaturePipeline()
    result = pipe.compute(VIN, pd.DataFrame())
    assert result.empty


# ── Engine ────────────────────────────────────────────────────────────────────

ENGINE_EXPECTED_COLS = [
    "vin",
    "rpm_mean_7d",
    "coolant_temp_max_7d",
]


def test_engine_pipeline_basic(telemetry_df):
    pipe = EngineFeaturePipeline()
    result = pipe.compute(VIN, telemetry_df)
    _check_output(result, ENGINE_EXPECTED_COLS)


def test_engine_rpm_nonnegative(telemetry_df):
    pipe = EngineFeaturePipeline()
    result = pipe.compute(VIN, telemetry_df)
    assert result["rpm_mean_7d"].iloc[0] >= 0


def test_engine_empty_df():
    pipe = EngineFeaturePipeline()
    result = pipe.compute(VIN, pd.DataFrame())
    assert result.empty


# ── Tyre ──────────────────────────────────────────────────────────────────────

TYRE_EXPECTED_COLS = [
    "vin",
    "tyre_pressure_min_7d",
]


def test_tyre_pipeline_basic(telemetry_df):
    pipe = TyreFeaturePipeline()
    result = pipe.compute(VIN, telemetry_df)
    _check_output(result, TYRE_EXPECTED_COLS)


def test_tyre_empty_df():
    pipe = TyreFeaturePipeline()
    result = pipe.compute(VIN, pd.DataFrame())
    assert result.empty


# ── Driver Behaviour ──────────────────────────────────────────────────────────

DRIVER_EXPECTED_COLS = [
    "vin",
    "harsh_brake_rate_30d",
    "harsh_accel_rate_30d",
]


def test_driver_behaviour_pipeline_basic(telemetry_df):
    pipe = DriverBehaviourFeaturePipeline()
    result = pipe.compute(VIN, telemetry_df)
    _check_output(result, DRIVER_EXPECTED_COLS)


def test_driver_score_in_range(telemetry_df):
    pipe = DriverBehaviourFeaturePipeline()
    result = pipe.compute(VIN, telemetry_df)
    if "driver_score" in result.columns:
        score = result["driver_score"].iloc[0]
        assert 0 <= score <= 100, f"Driver score out of range: {score}"


def test_driver_empty_df():
    pipe = DriverBehaviourFeaturePipeline()
    result = pipe.compute(VIN, pd.DataFrame())
    assert result.empty


# ── Multi-VIN batch ───────────────────────────────────────────────────────────

def test_brake_pipeline_multiple_vins(multi_vin_df, sample_vins):
    pipe = BrakeFeaturePipeline()
    for vin in sample_vins:
        df_vin = multi_vin_df[multi_vin_df["vin"] == vin].copy()
        result = pipe.compute(vin, df_vin)
        assert not result.empty, f"Empty result for VIN {vin}"
        assert result["vin"].iloc[0] == vin
