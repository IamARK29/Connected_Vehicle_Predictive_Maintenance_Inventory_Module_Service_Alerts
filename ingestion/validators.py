"""TBox telemetry channel validator — 23-channel Big Data spec."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FieldRule:
    min_val: float | None = None
    max_val: float | None = None
    scale: float | None = None  # raw × scale = engineering value
    required: bool = False
    enum_values: list | None = None
    data_type: str = "number"  # number | boolean | string | enum
    validity_flag_for: list[str] = field(default_factory=list)  # if =1, listed fields are invalid


CHANNEL_RULES: dict[int, dict[str, FieldRule]] = {
    1: {  # GNSS Position + Time
        "gnssTime":  FieldRule(min_val=0, required=True, data_type="number"),
        "gnssLong":  FieldRule(min_val=-90, max_val=90, required=True),   # per spec (swapped from convention)
        "gnssLat":   FieldRule(min_val=-180, max_val=180, required=True),
    },
    2: {  # GNSS Quality
        "altitude":   FieldRule(min_val=-100, max_val=8900),
        "gnssHead":   FieldRule(min_val=0, max_val=359),
        "gnssSats":   FieldRule(min_val=0, max_val=15),
        "hdop":       FieldRule(min_val=0, max_val=25.5),
        "gpsStatus":  FieldRule(enum_values=[0, 1, 2, 3], data_type="enum"),
    },
    3: {  # Vehicle Dynamics
        "vehSpeed":      FieldRule(min_val=0, max_val=4000, scale=0.1),   # ÷10 = kph
        "vehRPM":        FieldRule(min_val=0, max_val=12750),
        "vehSysPwrMod":  FieldRule(enum_values=[0, 1, 2, 3], data_type="enum"),
        "vehGearPos":    FieldRule(min_val=0, max_val=15),
    },
    4: {  # TBox Accelerometers + Pedals
        "tboxAccelX":        FieldRule(min_val=-500, max_val=500),
        "tboxAccelY":        FieldRule(min_val=-500, max_val=500),
        "tboxAccelZ":        FieldRule(min_val=-500, max_val=500),
        "vehAccelPos":       FieldRule(min_val=0, max_val=255),
        "vehBrakePos":       FieldRule(min_val=0, max_val=255),
        "vehSteeringAngle":  FieldRule(min_val=-32768, max_val=32767),
        "vehFuelConsumed":   FieldRule(min_val=0),
    },
    5: {  # Doors
        "driverDoorAjar":  FieldRule(data_type="boolean"),
        "passDoorAjar":    FieldRule(data_type="boolean"),
        "rlDoorAjar":      FieldRule(data_type="boolean"),
        "rrDoorAjar":      FieldRule(data_type="boolean"),
        "bootDoorAjar":    FieldRule(data_type="boolean"),
        "bonnetAjar":      FieldRule(data_type="boolean"),
        "driverDoorLock":  FieldRule(data_type="boolean"),
        "passDoorLock":    FieldRule(data_type="boolean"),
        "rlDoorLock":      FieldRule(data_type="boolean"),
        "rrDoorLock":      FieldRule(data_type="boolean"),
        "centralLock":     FieldRule(data_type="boolean"),
    },
    6: {  # Windows
        "driverWindowPos":  FieldRule(min_val=0, max_val=255),
        "passWindowPos":    FieldRule(min_val=0, max_val=255),
        "rlWindowPos":      FieldRule(min_val=0, max_val=255),
        "rrWindowPos":      FieldRule(min_val=0, max_val=255),
        "sunroofPos":       FieldRule(min_val=0, max_val=255),
    },
    7: {  # Vehicle Status (booleans + enums)
        "ignitionStatus":    FieldRule(enum_values=[0, 1, 2, 3], data_type="enum"),
        "seatbeltDriver":    FieldRule(data_type="boolean"),
        "seatbeltPass":      FieldRule(data_type="boolean"),
        "seatbeltRL":        FieldRule(data_type="boolean"),
        "seatbeltRR":        FieldRule(data_type="boolean"),
        "seatbeltRCentre":   FieldRule(data_type="boolean"),
        "parkingBrakeActive": FieldRule(data_type="boolean"),
    },
    8: {  # Cruise Control
        "cruiseActive":      FieldRule(data_type="boolean"),
        "cruiseAccelerate":  FieldRule(data_type="boolean"),
        "cruiseDecelerate":  FieldRule(data_type="boolean"),
        "cruiseCancel":      FieldRule(data_type="boolean"),
        "cruiseResume":      FieldRule(data_type="boolean"),
        "cruiseSetSpeed":    FieldRule(min_val=0, max_val=255),
    },
    9: {  # Temperatures (-128 to 127 °C)
        "ambientAirTemp":      FieldRule(min_val=-128, max_val=127),
        "coolantTemp":         FieldRule(min_val=-128, max_val=127),
        "intakeAirTemp":       FieldRule(min_val=-128, max_val=127),
        "transmissionTemp":    FieldRule(min_val=-128, max_val=127),
        "engineOilTemp":       FieldRule(min_val=-128, max_val=127),
    },
    10: {  # HVAC
        "hvacMode":             FieldRule(enum_values=[0, 1, 2, 3, 4], data_type="enum"),
        "hvacFanSpeed":         FieldRule(min_val=0, max_val=7),
        "hvacSetTemp":          FieldRule(min_val=-128, max_val=127),
        "hvacACActive":         FieldRule(data_type="boolean"),
        "hvacRecircActive":     FieldRule(data_type="boolean"),
        "hvacRearDefrost":      FieldRule(data_type="boolean"),
        "hvacFrontDefrost":     FieldRule(data_type="boolean"),
    },
    11: {  # Lights
        "headLightsOn":    FieldRule(data_type="boolean"),
        "highBeamOn":      FieldRule(data_type="boolean"),
        "fogFrontOn":      FieldRule(data_type="boolean"),
        "fogRearOn":       FieldRule(data_type="boolean"),
        "leftTurnSignal":  FieldRule(data_type="boolean"),
        "rightTurnSignal": FieldRule(data_type="boolean"),
        "hazardOn":        FieldRule(data_type="boolean"),
        "drlOn":           FieldRule(data_type="boolean"),
        "parkingLightsOn": FieldRule(data_type="boolean"),
    },
    12: {  # Rain/Night
        "rainSensorActive":  FieldRule(data_type="boolean"),
        "nightModeActive":   FieldRule(data_type="boolean"),
        "wiperFrontSpeed":   FieldRule(enum_values=[0, 1, 2, 3, 4], data_type="enum"),
        "wiperRearActive":   FieldRule(data_type="boolean"),
        "autoLightsActive":  FieldRule(data_type="boolean"),
    },
    13: {  # Vehicle General
        "odometer":         FieldRule(min_val=0),
        "fuelLevel":        FieldRule(min_val=0, max_val=100),
        "battVoltage12V":   FieldRule(min_val=0, max_val=20),
        "engineRunning":    FieldRule(data_type="boolean"),
        "vehRange":         FieldRule(min_val=0),
    },
    14: {  # Horn
        "hornActive": FieldRule(data_type="boolean"),
    },
    15: {  # MIL + Safety Systems
        "milActive":     FieldRule(data_type="boolean"),
        "milDtcCount":   FieldRule(min_val=0, max_val=255),
        "milDtcCodes":   FieldRule(data_type="string"),
        "absActive":     FieldRule(data_type="boolean"),
        "tcsActive":     FieldRule(data_type="boolean"),
        "espActive":     FieldRule(data_type="boolean"),
    },
    16: {  # Seat Belt (detailed)
        "seatbeltDriver":  FieldRule(data_type="boolean"),
        "seatbeltPass":    FieldRule(data_type="boolean"),
        "seatbeltRL":      FieldRule(data_type="boolean"),
        "seatbeltRR":      FieldRule(data_type="boolean"),
        "seatbeltRMid":    FieldRule(data_type="boolean"),
    },
    17: {  # Airbag
        "airbagDeployedAny":      FieldRule(data_type="boolean"),
        "airbagDriverDeployed":   FieldRule(data_type="boolean"),
        "airbagPassDeployed":     FieldRule(data_type="boolean"),
        "airbagSideRLDeployed":   FieldRule(data_type="boolean"),
        "airbagSideRRDeployed":   FieldRule(data_type="boolean"),
    },
    18: {  # Network
        "imei":            FieldRule(data_type="string"),
        "iccid":           FieldRule(data_type="string"),
        "networkOperator": FieldRule(data_type="string"),
        "networkType":     FieldRule(enum_values=[0, 1, 2, 3, 4, 5], data_type="enum"),
        "signalStrength":  FieldRule(min_val=-128, max_val=127),
    },
    19: {  # Battery Pack Info (HV)
        "vehPackVol":        FieldRule(min_val=0, max_val=4000, scale=0.25),    # raw→V
        "vehPackCrnt":       FieldRule(min_val=-1000, max_val=60000, scale=0.05),  # raw→A
        "vehPackSOC":        FieldRule(min_val=0, max_val=100),
        "vehPackSOH":        FieldRule(min_val=0, max_val=100),
        "vehPackMaxTemp":    FieldRule(min_val=-128, max_val=127),
        "vehPackMinTemp":    FieldRule(min_val=-128, max_val=127),
        "vehPackMaxCellVol": FieldRule(min_val=0, max_val=5000),   # mV
        "vehPackMinCellVol": FieldRule(min_val=0, max_val=5000),
        "vehConsumption":    FieldRule(min_val=0),
        "vehRegeneration":   FieldRule(min_val=0),
    },
    20: {  # Charging
        "chargePlugConnected": FieldRule(data_type="boolean"),
        "chargeActive":        FieldRule(data_type="boolean"),
        "chargeMode":          FieldRule(enum_values=[0, 1, 2, 3], data_type="enum"),
        "chargeVoltage":       FieldRule(min_val=0, max_val=4000, scale=0.25),
        "chargeCurrent":       FieldRule(min_val=-1000, max_val=60000, scale=0.05),
        "chargePower":         FieldRule(min_val=0),
        "chargeTargetSOC":     FieldRule(min_val=0, max_val=100),
        "timeToFullCharge":    FieldRule(min_val=0),
    },
    21: {  # EV RVM — BMS Suite + Traction Motor + Fault Codes
        # Validity flags: value=1 means the associated data field is INVALID
        "vehBMSPackSOCV":    FieldRule(enum_values=[0, 1], data_type="enum", validity_flag_for=["vehBMSPackSOC"]),
        "vehBMSPackVolV":    FieldRule(enum_values=[0, 1], data_type="enum", validity_flag_for=["vehBMSPackVol"]),
        "vehBMSPackCrntV":   FieldRule(enum_values=[0, 1], data_type="enum", validity_flag_for=["vehBMSPackCrnt"]),
        # BMS data
        "vehBMSPackVol":     FieldRule(min_val=0, max_val=4000, scale=0.25),
        "vehBMSPackCrnt":    FieldRule(min_val=-1000, max_val=60000, scale=0.05),
        "vehBMSPackSOC":     FieldRule(min_val=0, max_val=100),
        "vehBMSPackSOH":     FieldRule(min_val=0, max_val=100),
        "vehBMSCellMaxVol":  FieldRule(min_val=0, max_val=5000),
        "vehBMSCellMinVol":  FieldRule(min_val=0, max_val=5000),
        "vehBMSCellMaxTemp": FieldRule(min_val=-128, max_val=127),
        "vehBMSCellMinTemp": FieldRule(min_val=-128, max_val=127),
        "vehBMSFaultCode":   FieldRule(min_val=0),
        "vehBMSBalancing":   FieldRule(data_type="boolean"),
        # Traction Motor
        "vehTMSpeed":        FieldRule(min_val=0, max_val=20000),
        "vehTMTorque":       FieldRule(min_val=-32768, max_val=32767),
        "vehTMTemp":         FieldRule(min_val=-128, max_val=127),
        "vehTMVoltage":      FieldRule(min_val=0, max_val=4000, scale=0.25),
        "vehTMCurrent":      FieldRule(min_val=-1000, max_val=60000, scale=0.05),
        "vehTMFaultCode":    FieldRule(min_val=0),
        "vehTMEfficiency":   FieldRule(min_val=0, max_val=100),
    },
    22: {  # Thermal Runaway
        "thermalRunawayLevel":   FieldRule(enum_values=[0, 1, 2, 3], data_type="enum"),
        "thermalRunawayZone":    FieldRule(min_val=0, max_val=255),
        "thermalRunawayActive":  FieldRule(data_type="boolean"),
    },
    23: {  # Tyre (1–128 per spec; wheelTyreMonitorStatus 0–6)
        "tyrePressureFL":          FieldRule(min_val=1, max_val=128),
        "tyrePressureFR":          FieldRule(min_val=1, max_val=128),
        "tyrePressureRL":          FieldRule(min_val=1, max_val=128),
        "tyrePressureRR":          FieldRule(min_val=1, max_val=128),
        "tyreTempFL":              FieldRule(min_val=-128, max_val=127),
        "tyreTempFR":              FieldRule(min_val=-128, max_val=127),
        "tyreTempRL":              FieldRule(min_val=-128, max_val=127),
        "tyreTempRR":              FieldRule(min_val=-128, max_val=127),
        "wheelTyreMonitorStatus":  FieldRule(enum_values=[0, 1, 2, 3, 4, 5, 6], data_type="enum"),
        "wheelTyrePressureStatus": FieldRule(enum_values=[0, 1, 2, 3], data_type="enum"),
    },
}

CHANNEL_MEASUREMENT: dict[int, str] = {
    1:  "tbox_position",
    2:  "tbox_position",
    3:  "tbox_engine",
    4:  "tbox_drive_style",
    5:  "tbox_doors",
    6:  "tbox_windows",
    7:  "tbox_vehicle_status",
    8:  "tbox_cruise",
    9:  "tbox_temperature",
    10: "tbox_hvac",
    11: "tbox_lights",
    12: "tbox_rain_night",
    13: "tbox_vehicle_general",
    14: "tbox_horn",
    15: "tbox_mil",
    16: "tbox_seat_belt",
    17: "tbox_airbag",
    18: "tbox_network",
    19: "tbox_battery_info",
    20: "tbox_charging",
    21: "tbox_ev_rvm",
    22: "tbox_thermal_runaway",
    23: "tbox_tyre",
}


class TelemetryValidator:
    """Validates a TBox channel payload and returns engineering-scaled values."""

    def validate(
        self, channel_id: int, payload: dict[str, Any]
    ) -> tuple[bool, dict[str, Any], list[str], list[str]]:
        """
        Returns (is_valid, cleaned_data, errors, warnings).
        cleaned_data contains values already scaled to engineering units.
        """
        rules = CHANNEL_RULES.get(channel_id)
        if not rules:
            return False, {}, [f"Unknown channel_id: {channel_id}"], []

        errors: list[str] = []
        warnings: list[str] = []
        cleaned: dict[str, Any] = {}

        # Collect fields that validity flags mark as invalid
        invalid_fields: set[str] = set()
        for fname, rule in rules.items():
            if rule.validity_flag_for and fname in payload:
                if payload[fname] == 1:
                    invalid_fields.update(rule.validity_flag_for)
                    warnings.extend(f"{f} marked invalid by validity flag {fname}" for f in rule.validity_flag_for)

        for fname, rule in rules.items():
            raw = payload.get(fname)

            if raw is None:
                if rule.required:
                    errors.append(f"Required field missing: {fname}")
                continue

            if rule.data_type == "boolean":
                if raw not in (0, 1, True, False):
                    errors.append(f"{fname}: expected boolean 0/1, got {raw!r}")
                    continue
                cleaned[fname] = bool(raw)

            elif rule.data_type == "enum":
                if raw not in (rule.enum_values or []):
                    errors.append(f"{fname}: {raw!r} not in {rule.enum_values}")
                    continue
                cleaned[fname] = int(raw)

            elif rule.data_type == "string":
                cleaned[fname] = str(raw)

            else:  # number
                try:
                    fval = float(raw)
                except (TypeError, ValueError):
                    errors.append(f"{fname}: expected number, got {raw!r}")
                    continue

                if rule.min_val is not None and fval < rule.min_val:
                    errors.append(f"{fname}: {fval} below min {rule.min_val}")
                    continue
                if rule.max_val is not None and fval > rule.max_val:
                    errors.append(f"{fname}: {fval} exceeds max {rule.max_val}")
                    continue

                eng_val = fval * rule.scale if rule.scale is not None else fval
                if fname in invalid_fields:
                    cleaned[f"{fname}_invalid"] = True
                else:
                    cleaned[fname] = eng_val

        return len(errors) == 0, cleaned, errors, warnings


class PhysicalConsistencyChecker:
    """
    Cross-signal physical plausibility checks.

    Operates on DECODED (physical) values — call after TelemetryValidator and
    SignalDecoder have already been applied.  Returns list of violation codes;
    empty list means no violations detected.
    """

    def check(self, row: dict) -> list[str]:
        violations: list[str] = []

        # Simultaneous heavy brake + heavy throttle
        brake_pct = (row.get("vehBrakePos") or 0) * 0.4
        accel_pct = (row.get("vehAccelPos") or 0) * 0.4
        if brake_pct > 30 and accel_pct > 30:
            violations.append("BRAKE_ACCEL_CONFLICT")

        # Stationary vehicle with elevated RPM while engine is running
        speed_raw = row.get("vehSpeed") or 0
        rpm       = row.get("vehRPM") or 0
        pwr_mod   = row.get("vehSysPwrMod", 0)
        if speed_raw == 0 and rpm > 800 and pwr_mod not in (0, 3):
            violations.append("SPEED_RPM_MISMATCH")

        # Moving fast in an improbably low gear
        gear = row.get("vehGearPos") or 0
        if speed_raw > 1200 and gear in (0, 1, 2):
            violations.append("HIGH_SPEED_LOW_GEAR")

        return violations
