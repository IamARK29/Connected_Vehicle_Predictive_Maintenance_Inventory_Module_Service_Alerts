"""Tests for EV feature engines and physics models.

Covers:
  - EVChargingFeatureEngine   (features/ev_charging_features.py)
  - EVMotorFeatureEngine      (features/ev_motor_features.py)
  - EVDCDCFeatureEngine       (features/ev_dcdc_features.py)
  - RangeAnxietyPredictor     (models/range_anxiety_model.py)
  - ThermalRunawayEarlyWarner (models/thermal_runaway_model.py)
  - EVCostFeatureEngine       (features/ev_cost_features.py)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

VIN = "MZ7XZSN00100000"


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _motor_df(n: int = 200) -> pd.DataFrame:
    """Minimal CH-21 telemetry DataFrame for EVMotorFeatureEngine."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "vin":               [VIN] * n,
        "vehTMInvtrTem":     rng.uniform(70, 90, n),   # raw; physical = raw - 40
        "vehTMSttrTem":      rng.uniform(100, 120, n),
        "vehTMActuToq":      rng.uniform(1024, 1224, n),  # raw; phys = raw*0.5-512
        "vehEPTTrInptShaftToq": rng.uniform(1800, 1990, n),  # raw; phys = raw*0.5-848
        "vehTMSpd":          rng.uniform(33000, 34000, n),  # raw; phys = raw-32768
    })


def _dcdc_df(n: int = 200) -> pd.DataFrame:
    """Minimal telemetry DataFrame for EVDCDCFeatureEngine."""
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "vin":         [VIN] * n,
        "vehSysPwrMod": [2] * n,           # RUN mode = 2
        "vehBatt":     rng.uniform(13.8, 14.4, n),  # decoded volts (pipeline stores physical values)
        "vehHVDCDCTem": rng.uniform(60, 80, n),  # raw -40 -> 20-40°C
    })


def _charging_sessions(n: int = 10) -> pd.DataFrame:
    """Minimal charge-session rows for EVChargingFeatureEngine."""
    rng = np.random.default_rng(1)
    now = pd.Timestamp.now(tz="UTC")
    return pd.DataFrame({
        "vin":         [VIN] * n,
        "start_ts":    [now - pd.Timedelta(days=i) for i in range(n)],
        "end_ts":      [now - pd.Timedelta(days=i) + pd.Timedelta(hours=4) for i in range(n)],
        "charge_type": ["AC"] * n,
        "soc_start_pct":  rng.uniform(20, 35, n),
        "soc_end_pct":    rng.uniform(85, 95, n),
        "duration_min":   rng.uniform(200, 280, n),
        "energy_kwh":     rng.uniform(30, 40, n),
        "end_voltage_v":  rng.uniform(395, 405, n),
        "expected_end_voltage_v": [400.0] * n,
    })


def _thermal_df_safe(n: int = 50) -> pd.DataFrame:
    """BMS telemetry with all faults at 0 (safe vehicle)."""
    return pd.DataFrame({
        "vin":                 [VIN] * n,
        "vehBMSPreThrmlFltInd": [0] * n,
        "vehBMSCMUFlt":        [0] * n,
        "vehBMSCellVoltFlt":   [0] * n,
        "vehBMSPackTemFlt":    [0] * n,
        "vehBMSPackVoltFlt":   [0] * n,
        "vehBMSCellMaxTem":    [100] * n,  # 100 * 0.5 = 50°C
        "vehBMSCellMinTem":    [96] * n,   # 96  * 0.5 = 48°C  -> delta 2°C
        "vehBMSHVILClsd":      [1] * n,    # interlock closed = safe
        "vehBMSBscSta":        [1] * n,
    })


def _thermal_df_critical(n: int = 50) -> pd.DataFrame:
    """BMS telemetry that triggers Rule 1 (pre-thermal fault indicator)."""
    df = _thermal_df_safe(n)
    df["vehBMSPreThrmlFltInd"] = 1
    return df


# ── EVChargingFeatureEngine ────────────────────────────────────────────────────

class TestEVChargingFeatureEngine:
    def test_import(self):
        from features.ev_charging_features import EVChargingFeatureEngine
        assert EVChargingFeatureEngine

    def test_empty_sessions_returns_nulls(self):
        from features.ev_charging_features import EVChargingFeatureEngine
        result = EVChargingFeatureEngine().compute(
            pd.DataFrame(), pd.DataFrame(), VIN, {}
        )
        assert isinstance(result, dict)
        # Count fields (total_charge_sessions_30d, dc_charge_sessions_30d) return 0;
        # all non-count fields must be None.
        count_keys = {"total_charge_sessions_30d", "dc_charge_sessions_30d"}
        assert all(v is None for k, v in result.items() if k not in count_keys)

    def test_with_sessions_returns_dict(self):
        from features.ev_charging_features import EVChargingFeatureEngine
        result = EVChargingFeatureEngine().compute(
            _charging_sessions(), pd.DataFrame(), VIN,
            {"battery_capacity_kwh": 50.3},
        )
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_alert_thresholds_exported(self):
        from features.ev_charging_features import ALERT_THRESHOLDS
        assert "charge_acceptance_ratio" in ALERT_THRESHOLDS


# ── EVMotorFeatureEngine ───────────────────────────────────────────────────────

class TestEVMotorFeatureEngine:
    def test_import(self):
        from features.ev_motor_features import EVMotorFeatureEngine
        assert EVMotorFeatureEngine

    def test_empty_df_returns_nulls(self):
        from features.ev_motor_features import EVMotorFeatureEngine
        result = EVMotorFeatureEngine().compute(pd.DataFrame(), VIN)
        assert isinstance(result, dict)
        assert all(v is None for v in result.values())

    def test_with_telemetry_computes_temps(self):
        from features.ev_motor_features import EVMotorFeatureEngine
        result = EVMotorFeatureEngine().compute(_motor_df(), VIN)
        assert isinstance(result, dict)
        assert result.get("inv_temp_max_30d") is not None
        assert result.get("stator_temp_max_30d") is not None

    def test_inverter_temp_physical_units(self):
        """Raw vehTMInvtrTem 70-90 → physical 30-50°C after -40 offset."""
        from features.ev_motor_features import EVMotorFeatureEngine
        result = EVMotorFeatureEngine().compute(_motor_df(), VIN)
        inv_max = result.get("inv_temp_max_30d")
        assert inv_max is not None
        assert 20.0 <= inv_max <= 60.0, f"Expected 20-60°C, got {inv_max}"

    def test_alert_thresholds_exported(self):
        from features.ev_motor_features import ALERT_THRESHOLDS
        assert "inv_temp_max_30d" in ALERT_THRESHOLDS
        assert "torque_efficiency_mean_30d" in ALERT_THRESHOLDS

    def test_wrong_vin_returns_nulls(self):
        from features.ev_motor_features import EVMotorFeatureEngine
        result = EVMotorFeatureEngine().compute(_motor_df(), "UNKNOWN_VIN")
        assert all(v is None for v in result.values())


# ── EVDCDCFeatureEngine ────────────────────────────────────────────────────────

class TestEVDCDCFeatureEngine:
    def test_import(self):
        from features.ev_dcdc_features import EVDCDCFeatureEngine
        assert EVDCDCFeatureEngine

    def test_empty_df_returns_nulls(self):
        from features.ev_dcdc_features import EVDCDCFeatureEngine
        result = EVDCDCFeatureEngine().compute(pd.DataFrame(), VIN)
        assert isinstance(result, dict)
        # dcdc_thermal_cycles_total returns 0; all other fields return None.
        count_keys = {"dcdc_thermal_cycles_total"}
        assert all(v is None for k, v in result.items() if k not in count_keys)

    def test_healthy_output_voltage_range(self):
        """vehBatt decoded 13.8-14.4 V; mean should be in that range."""
        from features.ev_dcdc_features import EVDCDCFeatureEngine
        result = EVDCDCFeatureEngine().compute(_dcdc_df(), VIN)
        mean_v = result.get("dcdc_output_v_mean_30d")
        assert mean_v is not None
        assert 13.5 <= mean_v <= 14.6, f"Expected 13.5-14.6V, got {mean_v}"

    def test_temp_physical_units(self):
        """Raw 60-80 - 40 = 20-40°C."""
        from features.ev_dcdc_features import EVDCDCFeatureEngine
        result = EVDCDCFeatureEngine().compute(_dcdc_df(), VIN)
        temp_max = result.get("dcdc_temp_max_30d")
        assert temp_max is not None
        assert 15.0 <= temp_max <= 45.0, f"Expected 15-45°C, got {temp_max}"

    def test_alert_thresholds_exported(self):
        from features.ev_dcdc_features import ALERT_THRESHOLDS
        assert "dcdc_output_v_mean_30d" in ALERT_THRESHOLDS
        assert "dcdc_output_v_min_30d" in ALERT_THRESHOLDS

    def test_off_mode_excluded(self):
        """Rows with vehSysPwrMod != 2 should not contribute to running stats."""
        from features.ev_dcdc_features import EVDCDCFeatureEngine
        df = _dcdc_df(100)
        df.loc[50:, "vehSysPwrMod"] = 0  # half rows = OFF
        result = EVDCDCFeatureEngine().compute(df, VIN)
        # Should still compute from the running half
        assert result.get("dcdc_output_v_mean_30d") is not None


# ── RangeAnxietyPredictor ──────────────────────────────────────────────────────

class TestRangeAnxietyPredictor:
    def test_import(self):
        from models.range_anxiety_model import RangeAnxietyPredictor
        assert RangeAnxietyPredictor

    def test_predict_returns_dict(self):
        from models.range_anxiety_model import RangeAnxietyPredictor
        result = RangeAnxietyPredictor().predict(
            vin=VIN,
            current_soc_pct=80.0,
            current_outside_temp_c=25.0,
            ac_is_on=False,
            feature_store_features={
                "soh_estimated": 95,
                "range_per_kwh_30d_trend": 20.0,
                "composite_drive_score": 75,
                "km_per_day_30d_avg": 60,
                "kwh_per_100km_std_30d": 2.0,
            },
            fleet_row={"rated_range_km": 338, "battery_capacity_kwh": 50.3},
        )
        assert "predicted_range_km" in result
        assert result["predicted_range_km"] > 0

    def test_hot_weather_ac_reduces_range(self):
        """35% SoC + 42°C + AC on must give < 200 km (as specified in the range anxiety spec)."""
        from models.range_anxiety_model import RangeAnxietyPredictor
        result = RangeAnxietyPredictor().predict(
            vin=VIN,
            current_soc_pct=35.0,
            current_outside_temp_c=42.0,
            ac_is_on=True,
            feature_store_features={
                "soh_estimated": 88,
                "range_per_kwh_30d_trend": 17.0,
                "composite_drive_score": 60,
                "km_per_day_30d_avg": 80,
                "kwh_per_100km_std_30d": 2.5,
            },
            fleet_row={"rated_range_km": 338, "battery_capacity_kwh": 50.3},
        )
        km = result["predicted_range_km"]
        assert km < 200, f"Expected < 200 km at 35% SoC / 42°C / AC on, got {km}"

    def test_high_soc_gives_more_range_than_low(self):
        from models.range_anxiety_model import RangeAnxietyPredictor
        predictor = RangeAnxietyPredictor()
        common = dict(
            vin=VIN,
            current_outside_temp_c=25.0,
            ac_is_on=False,
            feature_store_features={"soh_estimated": 95},
            fleet_row={"rated_range_km": 338, "battery_capacity_kwh": 50.3},
        )
        high = predictor.predict(current_soc_pct=90.0, **common)["predicted_range_km"]
        low  = predictor.predict(current_soc_pct=20.0, **common)["predicted_range_km"]
        assert high > low

    def test_anxiety_flag_present(self):
        from models.range_anxiety_model import RangeAnxietyPredictor
        result = RangeAnxietyPredictor().predict(
            vin=VIN,
            current_soc_pct=10.0,
            current_outside_temp_c=38.0,
            ac_is_on=True,
            feature_store_features={"soh_estimated": 80},
            fleet_row={"rated_range_km": 338, "battery_capacity_kwh": 50.3},
        )
        assert "range_anxiety_flag" in result
        assert result["range_anxiety_flag"] is True


# ── ThermalRunawayEarlyWarner ─────────────────────────────────────────────────

class TestThermalRunawayEarlyWarner:
    def test_import(self):
        from models.thermal_runaway_model import ThermalRunawayEarlyWarner
        assert ThermalRunawayEarlyWarner

    def test_safe_vehicle_returns_none(self):
        from models.thermal_runaway_model import ThermalRunawayEarlyWarner
        result = ThermalRunawayEarlyWarner().classify(VIN, _thermal_df_safe(), {})
        assert result["risk_level"] == "NONE"
        assert result["factors"] == []
        assert result["action"] == "monitor"

    def test_critical_pre_thermal_fault(self):
        """Rule 1: vehBMSPreThrmlFltInd == 1 → CRITICAL regardless of other signals."""
        from models.thermal_runaway_model import ThermalRunawayEarlyWarner
        result = ThermalRunawayEarlyWarner().classify(VIN, _thermal_df_critical(), {})
        assert result["risk_level"] == "CRITICAL"
        assert result["action"] == "STOP_VEHICLE_IMMEDIATELY_CONTACT_DEALER"
        signals = [f["signal"] for f in result["factors"]]
        assert "vehBMSPreThrmlFltInd" in signals

    def test_level3_fault_is_critical(self):
        """Rule 2: any FAULT_LEVELS signal >= 3 → CRITICAL."""
        from models.thermal_runaway_model import ThermalRunawayEarlyWarner
        df = _thermal_df_safe()
        df["vehBMSPackTemFlt"] = 3
        result = ThermalRunawayEarlyWarner().classify(VIN, df, {})
        assert result["risk_level"] == "CRITICAL"

    def test_two_level2_faults_is_high(self):
        """Rule 3: two or more signals at level 2 simultaneously → HIGH."""
        from models.thermal_runaway_model import ThermalRunawayEarlyWarner
        df = _thermal_df_safe()
        df["vehBMSCMUFlt"]      = 2
        df["vehBMSCellVoltFlt"] = 2
        result = ThermalRunawayEarlyWarner().classify(VIN, df, {})
        assert result["risk_level"] in ("HIGH", "CRITICAL")

    def test_empty_df_returns_none(self):
        from models.thermal_runaway_model import ThermalRunawayEarlyWarner
        result = ThermalRunawayEarlyWarner().classify(VIN, pd.DataFrame(), {})
        assert result["risk_level"] == "NONE"

    def test_missing_columns_handled_gracefully(self):
        """Missing CH-22 columns must not crash — _col() returns default."""
        from models.thermal_runaway_model import ThermalRunawayEarlyWarner
        df = pd.DataFrame({"vin": [VIN] * 10})  # no BMS columns at all
        result = ThermalRunawayEarlyWarner().classify(VIN, df, {})
        assert result["risk_level"] == "NONE"

    def test_result_schema(self):
        from models.thermal_runaway_model import ThermalRunawayEarlyWarner
        result = ThermalRunawayEarlyWarner().classify(VIN, _thermal_df_safe(), {})
        for key in ("vin", "risk_level", "factors", "action", "evaluated_at"):
            assert key in result, f"Missing key: {key}"

    def test_fault_levels_dict(self):
        from models.thermal_runaway_model import ThermalRunawayEarlyWarner
        fl = ThermalRunawayEarlyWarner.FAULT_LEVELS
        assert "vehBMSCMUFlt"      in fl
        assert "vehBMSCellVoltFlt" in fl
        assert "vehBMSPackTemFlt"  in fl
        assert "vehBMSPackVoltFlt" in fl
        assert all(3 in levels for levels in fl.values())


# ── EVCostFeatureEngine ────────────────────────────────────────────────────────

class TestEVCostFeatureEngine:
    def test_import(self):
        from features.ev_cost_features import EVCostFeatureEngine, PETROL_COST_PER_KM_INR
        assert EVCostFeatureEngine
        assert PETROL_COST_PER_KM_INR > 0

    def test_petrol_benchmark(self):
        """₹100/L at 12 km/L = ₹8.33/km."""
        from features.ev_cost_features import PETROL_COST_PER_KM_INR
        assert abs(PETROL_COST_PER_KM_INR - 8.333) < 0.01

    def test_empty_sessions_returns_all_none(self):
        from features.ev_cost_features import EVCostFeatureEngine
        result = EVCostFeatureEngine().compute(pd.DataFrame(), VIN, {}, {})
        assert isinstance(result, dict)
        assert all(v is None for v in result.values())

    def test_ev_health_column_names_normalised(self):
        """ev_health synthesises charge_type/duration_min/energy_kwh — must be accepted."""
        from features.ev_cost_features import EVCostFeatureEngine
        sessions = pd.DataFrame([
            {"vin": VIN, "charge_type": "AC", "duration_min": 360, "energy_kwh": 18.0},
            {"vin": VIN, "charge_type": "DC", "duration_min": 45,  "energy_kwh": 22.0},
            {"vin": VIN, "charge_type": "AC", "duration_min": 420, "energy_kwh": 20.0},
        ])
        result = EVCostFeatureEngine().compute(
            sessions, VIN, {},
            {"soh_estimated": 92, "km_per_day_30d_avg": 50},
        )
        assert result["cost_per_km_inr"] is not None
        assert result["kwh_per_km"] is not None

    def test_dc_charge_premium_positive(self):
        """DC sessions should cost more than AC → premium > 0."""
        from features.ev_cost_features import EVCostFeatureEngine
        sessions = pd.DataFrame([
            {"vin": VIN, "charge_type": "AC", "duration_min": 360, "energy_kwh": 18.0},
            {"vin": VIN, "charge_type": "DC", "duration_min": 45,  "energy_kwh": 22.0},
            {"vin": VIN, "charge_type": "AC", "duration_min": 420, "energy_kwh": 20.0},
        ])
        result = EVCostFeatureEngine().compute(
            sessions, VIN, {},
            {"soh_estimated": 92, "km_per_day_30d_avg": 50},
        )
        assert result["dc_charge_premium_inr_30d"] > 0

    def test_projected_cost_at_80pct_soh_higher(self):
        """Lower SoH means more kWh needed per km → projected cost > current."""
        from features.ev_cost_features import EVCostFeatureEngine
        sessions = pd.DataFrame([
            {"vin": VIN, "charge_type": "AC", "duration_min": 360, "energy_kwh": 18.0},
            {"vin": VIN, "charge_type": "DC", "duration_min": 45,  "energy_kwh": 22.0},
            {"vin": VIN, "charge_type": "AC", "duration_min": 420, "energy_kwh": 20.0},
        ])
        result = EVCostFeatureEngine().compute(
            sessions, VIN, {},
            {"soh_estimated": 92, "km_per_day_30d_avg": 50},
        )
        assert result["projected_cost_per_km_at_80pct_soh"] > result["cost_per_km_inr"]

    def test_home_vs_dc_mirrors_dc_premium(self):
        from features.ev_cost_features import EVCostFeatureEngine
        sessions = pd.DataFrame([
            {"vin": VIN, "charge_type": "AC", "duration_min": 360, "energy_kwh": 18.0},
            {"vin": VIN, "charge_type": "DC", "duration_min": 45,  "energy_kwh": 22.0},
        ])
        result = EVCostFeatureEngine().compute(sessions, VIN, {}, {"soh_estimated": 92})
        assert result["home_vs_dc_cost_difference"] == result["dc_charge_premium_inr_30d"]

    def test_output_has_all_8_features(self):
        from features.ev_cost_features import EVCostFeatureEngine
        result = EVCostFeatureEngine().compute(pd.DataFrame(), VIN, {}, {})
        expected = {
            "cost_per_km_inr",
            "kwh_per_km",
            "total_charging_cost_inr_30d",
            "energy_wasted_kwh_30d",
            "energy_waste_cost_inr_30d",
            "dc_charge_premium_inr_30d",
            "projected_cost_per_km_at_80pct_soh",
            "home_vs_dc_cost_difference",
        }
        assert expected.issubset(result.keys())
