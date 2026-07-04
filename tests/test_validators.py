"""Tests for the 23-channel TBox telemetry validator."""
from __future__ import annotations

import pytest
from ingestion.validators import CHANNEL_RULES, TelemetryValidator


@pytest.fixture(scope="module")
def v() -> TelemetryValidator:
    return TelemetryValidator()


# ── Helper ─────────────────────────────────────────────────────────────────────

def _ok(validator, channel, payload):
    valid, cleaned, errors, warnings = validator.validate(channel, payload)
    assert valid, f"Ch{channel} should be valid but got errors: {errors}"
    return cleaned, warnings


def _fail(validator, channel, payload):
    valid, cleaned, errors, warnings = validator.validate(channel, payload)
    assert not valid, f"Ch{channel} should be invalid but passed. cleaned={cleaned}"
    return errors


# ── Channel 1 — GNSS Position ──────────────────────────────────────────────────

def test_ch1_valid(v):
    cleaned, _ = _ok(v, 1, {"gnssTime": 1_700_000_000, "gnssLong": 72.8, "gnssLat": 19.0})
    assert "gnssTime" in cleaned
    assert "gnssLong" in cleaned


def test_ch1_missing_required_field(v):
    errors = _fail(v, 1, {"gnssTime": 1_700_000_000})
    assert any("gnssLong" in e or "gnssLat" in e for e in errors)


def test_ch1_gnsslong_boundary(v):
    # exactly at max
    _ok(v, 1, {"gnssTime": 0, "gnssLong": 90, "gnssLat": 180})
    # one step over
    _fail(v, 1, {"gnssTime": 0, "gnssLong": 90.1, "gnssLat": 0})


def test_ch1_gnsstime_negative(v):
    errors = _fail(v, 1, {"gnssTime": -1, "gnssLong": 0, "gnssLat": 0})
    assert any("gnssTime" in e for e in errors)


# ── Channel 2 — GNSS Quality ──────────────────────────────────────────────────

def test_ch2_valid(v):
    _ok(v, 2, {"altitude": 100, "gnssHead": 180, "gnssSats": 8, "hdop": 1.2, "gpsStatus": 3})


def test_ch2_invalid_enum_gps_status(v):
    errors = _fail(v, 2, {"gpsStatus": 9})
    assert any("gpsStatus" in e for e in errors)


def test_ch2_altitude_out_of_range(v):
    _fail(v, 2, {"altitude": 9000})


def test_ch2_boundary_altitude(v):
    _ok(v, 2, {"altitude": 8900})
    _fail(v, 2, {"altitude": 8901})


# ── Channel 3 — Vehicle Dynamics ──────────────────────────────────────────────

def test_ch3_speed_scaling(v):
    cleaned, _ = _ok(v, 3, {"vehSpeed": 1000, "vehRPM": 2000, "vehSysPwrMod": 2, "vehGearPos": 3})
    assert abs(cleaned["vehSpeed"] - 100.0) < 1e-6   # 1000 × 0.1 = 100 kph


def test_ch3_speed_over_max(v):
    _fail(v, 3, {"vehSpeed": 4001})


def test_ch3_power_mode_enum(v):
    _fail(v, 3, {"vehSysPwrMod": 5})


def test_ch3_rpm_boundary(v):
    _ok(v, 3, {"vehRPM": 12750})
    _fail(v, 3, {"vehRPM": 12751})


# ── Channel 4 — Accelerometers + Pedals ───────────────────────────────────────

def test_ch4_valid(v):
    _ok(v, 4, {
        "tboxAccelX": 0, "tboxAccelY": 0, "tboxAccelZ": 9,
        "vehAccelPos": 50, "vehBrakePos": 0, "vehSteeringAngle": -10,
    })


def test_ch4_accel_out_of_range(v):
    _fail(v, 4, {"tboxAccelX": 501})
    _fail(v, 4, {"tboxAccelX": -501})


def test_ch4_brake_pos_boundary(v):
    _ok(v, 4, {"vehBrakePos": 255})
    _fail(v, 4, {"vehBrakePos": 256})


def test_ch4_steering_negative_boundary(v):
    _ok(v, 4, {"vehSteeringAngle": -32768})
    _fail(v, 4, {"vehSteeringAngle": -32769})


# ── Channel 5 — Doors ─────────────────────────────────────────────────────────

def test_ch5_valid_booleans(v):
    payload = {
        "driverDoorAjar": 0, "passDoorAjar": 1, "rlDoorAjar": 0, "rrDoorAjar": 0,
        "bootDoorAjar": 0, "bonnetAjar": 0, "driverDoorLock": 1, "passDoorLock": 1,
        "rlDoorLock": 1, "rrDoorLock": 1, "centralLock": 1,
    }
    cleaned, _ = _ok(v, 5, payload)
    assert cleaned["passDoorAjar"] is True
    assert cleaned["driverDoorAjar"] is False


def test_ch5_invalid_boolean(v):
    _fail(v, 5, {"driverDoorAjar": 2})


# ── Channel 6 — Windows ───────────────────────────────────────────────────────

def test_ch6_valid(v):
    _ok(v, 6, {"driverWindowPos": 0, "passWindowPos": 128, "sunroofPos": 255})


def test_ch6_out_of_range(v):
    _fail(v, 6, {"driverWindowPos": 256})


# ── Channel 7 — Vehicle Status ────────────────────────────────────────────────

def test_ch7_valid(v):
    _ok(v, 7, {"ignitionStatus": 2, "seatbeltDriver": 1, "parkingBrakeActive": 0})


def test_ch7_bad_ignition_enum(v):
    _fail(v, 7, {"ignitionStatus": 5})


# ── Channel 8 — Cruise Control ────────────────────────────────────────────────

def test_ch8_valid(v):
    _ok(v, 8, {"cruiseActive": 1, "cruiseSetSpeed": 100})


def test_ch8_set_speed_boundary(v):
    _ok(v, 8, {"cruiseSetSpeed": 255})
    _fail(v, 8, {"cruiseSetSpeed": 256})


# ── Channel 9 — Temperatures ──────────────────────────────────────────────────

def test_ch9_valid(v):
    _ok(v, 9, {"ambientAirTemp": 35, "coolantTemp": 90, "engineOilTemp": 100})


def test_ch9_out_of_range_low(v):
    _fail(v, 9, {"coolantTemp": -129})


def test_ch9_boundary(v):
    _ok(v, 9, {"coolantTemp": -128})
    _ok(v, 9, {"coolantTemp": 127})
    _fail(v, 9, {"coolantTemp": 128})


# ── Channel 10 — HVAC ─────────────────────────────────────────────────────────

def test_ch10_valid(v):
    _ok(v, 10, {"hvacMode": 2, "hvacFanSpeed": 5, "hvacSetTemp": 22, "hvacACActive": 1})


def test_ch10_bad_hvac_mode(v):
    _fail(v, 10, {"hvacMode": 6})


def test_ch10_fan_speed_boundary(v):
    _ok(v, 10, {"hvacFanSpeed": 7})
    _fail(v, 10, {"hvacFanSpeed": 8})


# ── Channel 11 — Lights ───────────────────────────────────────────────────────

def test_ch11_valid(v):
    _ok(v, 11, {"headLightsOn": 1, "hazardOn": 0, "drlOn": 1})


# ── Channel 12 — Rain / Night ─────────────────────────────────────────────────

def test_ch12_valid(v):
    _ok(v, 12, {"rainSensorActive": 0, "wiperFrontSpeed": 3, "nightModeActive": 1})


def test_ch12_wiper_enum(v):
    _fail(v, 12, {"wiperFrontSpeed": 5})


# ── Channel 13 — Vehicle General ──────────────────────────────────────────────

def test_ch13_valid(v):
    _ok(v, 13, {"odometer": 50000, "fuelLevel": 65, "battVoltage12V": 12.6, "engineRunning": 1})


def test_ch13_fuel_out_of_range(v):
    _fail(v, 13, {"fuelLevel": 101})


def test_ch13_batt_12v_boundary(v):
    _ok(v, 13, {"battVoltage12V": 20})
    _fail(v, 13, {"battVoltage12V": 20.1})


# ── Channel 14 — Horn ─────────────────────────────────────────────────────────

def test_ch14_valid(v):
    _ok(v, 14, {"hornActive": 0})


# ── Channel 15 — MIL + Safety ─────────────────────────────────────────────────

def test_ch15_valid(v):
    _ok(v, 15, {"milActive": 1, "milDtcCount": 3, "milDtcCodes": "P0420,P0430", "absActive": 0})


def test_ch15_dtc_count_boundary(v):
    _ok(v, 15, {"milDtcCount": 255})
    _fail(v, 15, {"milDtcCount": 256})


# ── Channel 16 — Seat Belts ───────────────────────────────────────────────────

def test_ch16_valid(v):
    _ok(v, 16, {"seatbeltDriver": 1, "seatbeltPass": 0, "seatbeltRL": 1, "seatbeltRR": 0})


# ── Channel 17 — Airbag ───────────────────────────────────────────────────────

def test_ch17_valid(v):
    _ok(v, 17, {"airbagDeployedAny": 0, "airbagDriverDeployed": 0})


def test_ch17_deployed_true(v):
    cleaned, _ = _ok(v, 17, {"airbagDeployedAny": 1, "airbagDriverDeployed": 1})
    assert cleaned["airbagDeployedAny"] is True


# ── Channel 18 — Network ──────────────────────────────────────────────────────

def test_ch18_valid(v):
    _ok(v, 18, {
        "imei": "123456789012345", "iccid": "8991101200003204510",
        "networkOperator": "Airtel", "networkType": 4, "signalStrength": -70,
    })


def test_ch18_bad_network_type(v):
    _fail(v, 18, {"networkType": 6})


def test_ch18_signal_boundary(v):
    _ok(v, 18, {"signalStrength": -128})
    _ok(v, 18, {"signalStrength": 127})
    _fail(v, 18, {"signalStrength": 128})


# ── Channel 19 — HV Battery Pack ──────────────────────────────────────────────

def test_ch19_valid(v):
    cleaned, _ = _ok(v, 19, {
        "vehPackVol": 1400, "vehPackCrnt": 1000, "vehPackSOC": 75,
        "vehPackSOH": 92, "vehPackMaxTemp": 35, "vehPackMinTemp": 28,
    })
    assert abs(cleaned["vehPackVol"] - 350.0) < 1e-4   # 1400 × 0.25 = 350V


def test_ch19_soc_boundary(v):
    _ok(v, 19, {"vehPackSOC": 100})
    _fail(v, 19, {"vehPackSOC": 101})


# ── Channel 20 — Charging ─────────────────────────────────────────────────────

def test_ch20_valid(v):
    _ok(v, 20, {
        "chargePlugConnected": 1, "chargeActive": 1, "chargeMode": 2,
        "chargeVoltage": 1840, "chargeCurrent": 1300, "chargePower": 22000,
        "chargeTargetSOC": 80, "timeToFullCharge": 90,
    })


def test_ch20_bad_charge_mode(v):
    _fail(v, 20, {"chargeMode": 4})


# ── Channel 21 — EV RVM ───────────────────────────────────────────────────────

def test_ch21_validity_flags(v):
    payload = {
        "vehBMSPackSOCV": 1,   # marks vehBMSPackSOC as invalid
        "vehBMSPackSOC": 75,
        "vehBMSPackVol": 1400,
        "vehBMSPackCrnt": 0,
        "vehBMSPackSOH": 95,
        "vehTMSpeed": 5000,
        "vehTMTorque": 200,
        "vehTMTemp": 55,
    }
    valid, cleaned, errors, warnings = TelemetryValidator().validate(21, payload)
    assert valid
    assert any("vehBMSPackSOC" in w for w in warnings)
    assert "vehBMSPackSOC_invalid" in cleaned


def test_ch21_tm_torque_negative(v):
    _ok(v, 21, {"vehTMTorque": -32768})


def test_ch21_bms_cell_vol_boundary(v):
    _ok(v, 21, {"vehBMSCellMaxVol": 5000})
    _fail(v, 21, {"vehBMSCellMaxVol": 5001})


# ── Channel 22 — Thermal Runaway ──────────────────────────────────────────────

def test_ch22_valid(v):
    _ok(v, 22, {"thermalRunawayLevel": 0, "thermalRunawayZone": 0, "thermalRunawayActive": 0})


def test_ch22_active_level3(v):
    cleaned, _ = _ok(v, 22, {"thermalRunawayLevel": 3, "thermalRunawayActive": 1})
    assert cleaned["thermalRunawayLevel"] == 3


def test_ch22_bad_level_enum(v):
    _fail(v, 22, {"thermalRunawayLevel": 4})


def test_ch22_zone_boundary(v):
    _ok(v, 22, {"thermalRunawayZone": 255})
    _fail(v, 22, {"thermalRunawayZone": 256})


# ── Channel 23 — Tyres ────────────────────────────────────────────────────────

def test_ch23_valid(v):
    payload = {
        "tyrePressureFL": 44, "tyrePressureFR": 43, "tyrePressureRL": 45, "tyrePressureRR": 44,
        "tyreTempFL": 35, "tyreTempFR": 36, "tyreTempRL": 34, "tyreTempRR": 35,
        "wheelTyreMonitorStatus": 0, "wheelTyrePressureStatus": 0,
    }
    _ok(v, 23, payload)


def test_ch23_pressure_below_min(v):
    _fail(v, 23, {"tyrePressureFL": 0})


def test_ch23_pressure_boundary(v):
    _ok(v, 23, {"tyrePressureFL": 1})
    _ok(v, 23, {"tyrePressureFL": 128})
    _fail(v, 23, {"tyrePressureFL": 129})


def test_ch23_monitor_status_enum(v):
    _ok(v, 23, {"wheelTyreMonitorStatus": 6})
    _fail(v, 23, {"wheelTyreMonitorStatus": 7})


# ── Unknown channel ────────────────────────────────────────────────────────────

def test_unknown_channel(v):
    valid, cleaned, errors, _ = v.validate(99, {"foo": 1})
    assert not valid
    assert any("Unknown" in e for e in errors)


# ── All 23 channels present in CHANNEL_RULES ──────────────────────────────────

def test_all_23_channels_defined():
    assert set(CHANNEL_RULES.keys()) == set(range(1, 24))
