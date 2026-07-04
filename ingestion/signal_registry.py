"""
AutoPredict Signal Registry — Single Source of Truth for Big Data Spec encodings.

Never hardcode scale/offset/validity flags elsewhere — import from here.

physical = raw * scale + offset
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SignalDefinition:
    name:                 str
    channel_id:           int
    scale:                float
    offset:               float        = 0.0
    unit:                 str          = ""
    physical_min:         float        = float("-inf")
    physical_max:         float        = float("inf")
    invalid_raw_value:    float | None = None
    signed:               bool         = False
    validity_flag_field:  str | None   = None


def _s(name, ch, scale, unit, mn, mx, *,
        offset=0.0, signed=False, invalid=None, flag=None) -> SignalDefinition:
    return SignalDefinition(
        name=name, channel_id=ch, scale=scale, offset=offset, unit=unit,
        physical_min=mn, physical_max=mx, invalid_raw_value=invalid,
        signed=signed, validity_flag_field=flag,
    )


SIGNAL_REGISTRY: dict[str, SignalDefinition] = {

    # ── CH-01 Position 1 ───────────────────────────────────────────────────
    "gnssTime":  _s("gnssTime",  1, 1.0,       "unix_s", 0,          9_999_999_999),
    "gnssLong":  _s("gnssLong",  1, 0.000001,  "deg",    -180,       180,   signed=True),
    "gnssLat":   _s("gnssLat",   1, 0.000001,  "deg",    -90,        90,    signed=True),

    # ── CH-02 Position 2 ───────────────────────────────────────────────────
    "Altitude":  _s("Altitude",  2, 0.1,        "m",     -100,       8900),
    "gnssHead":  _s("gnssHead",  2, 1.0,        "deg",   0,          359),
    "gnssSats":  _s("gnssSats",  2, 1.0,        "count", 0,          15),
    "hdop":      _s("hdop",      2, 0.1,        "",      0,          25.5),
    "gpsStatus": _s("gpsStatus", 2, 1.0,        "enum",  0,          3),

    # ── CH-03 Engine Effort (HIGH FREQUENCY — 10 Hz) ───────────────────────
    "vehSpeed":     _s("vehSpeed",     3, 0.1,  "kph",  0,   400),
    "vehRPM":       _s("vehRPM",       3, 1.0,  "rpm",  0,   12750),
    "vehSysPwrMod": _s("vehSysPwrMod", 3, 1.0,  "enum", 0,   3),
    "vehGearPos":   _s("vehGearPos",   3, 1.0,  "enum", 0,   13),

    # ── CH-04 Drive Style (HIGH FREQUENCY — 10 Hz) ─────────────────────────
    "tboxAccelX":       _s("tboxAccelX",       4, 0.004, "g",   -2.0,  2.0,   signed=True),
    "tboxAccelY":       _s("tboxAccelY",       4, 0.004, "g",   -2.0,  2.0,   signed=True),
    "tboxAccelZ":       _s("tboxAccelZ",       4, 0.004, "g",   -2.0,  2.0,   signed=True),
    "vehAccelPos":      _s("vehAccelPos",      4, 0.4,   "%",   0,     100),
    "vehBrakePos":      _s("vehBrakePos",      4, 0.4,   "%",   0,     100),   # raw 175 = 70.0%
    "vehSteeringAngle": _s("vehSteeringAngle", 4, 0.1,   "deg", -2048, 2047,  signed=True),
    "vehFuelConsumed":  _s("vehFuelConsumed",  4, 1.0,   "ml",  0,     100_000),

    # ── CH-09 Temperature ──────────────────────────────────────────────────
    "vehOutsideTemp": _s("vehOutsideTemp", 9, 1.0, "C", -128, 127, signed=True),
    "vehInsideTemp":  _s("vehInsideTemp",  9, 1.0, "C", -128, 127, signed=True),

    # ── CH-10 HVAC ─────────────────────────────────────────────────────────
    "vehAC":               _s("vehAC",               10, 1.0, "bool", 0,  1),
    "vehACAuto":           _s("vehACAuto",           10, 1.0, "bool", 0,  1),
    "vehACFanSpeed":       _s("vehACFanSpeed",       10, 1.0, "enum", 0,  15),
    "vehACDrvTargetTemp":  _s("vehACDrvTargetTemp",  10, 1.0, "C",    16, 32),
    "vehACPassTargetTemp": _s("vehACPassTargetTemp", 10, 1.0, "C",    16, 32),

    # ── CH-14 Rain / Night ─────────────────────────────────────────────────
    "vehRainDetected":  _s("vehRainDetected",  14, 1.0, "enum", 0, 3),
    "vehNightDetected": _s("vehNightDetected", 14, 1.0, "bool", 0, 1),

    # ── CH-15 General Vehicle ──────────────────────────────────────────────
    "vehFuelLev":     _s("vehFuelLev",     15, 0.4,  "%",  0,    100),
    "vehBatt":        _s("vehBatt",        15, 0.1,  "V",  0,    40),      # raw 145 = 14.5V
    "vehCoolantTemp": _s("vehCoolantTemp", 15, 1.0,  "C",  -128, 255, signed=True),
    "vehOdo":         _s("vehOdo",         15, 1.0,  "km", 0,    1_000_000),
    "fuelRange":      _s("fuelRange",      15, 0.1,  "km", 0,    100_000),
    "fuelTankLevel":  _s("fuelTankLevel",  15, 1.0,  "L",  0,    200),

    # ── CH-18 Network ──────────────────────────────────────────────────────
    "cellSignalStrength": _s("cellSignalStrength", 18, 1.0, "dBm",  -128, 127, signed=True),
    "cellRAT":            _s("cellRAT",            18, 1.0, "enum", 0,    7),

    # ── CH-19 Battery Info ─────────────────────────────────────────────────
    "vehPackVol":           _s("vehPackVol",           19, 0.25,  "V",          0,   1023.75),
    "vehPackCrnt":          _s("vehPackCrnt",          19, 0.05,  "A",          -50, 3000),
    "avgElecConsumption":   _s("avgElecConsumption",   19, 0.1,   "kwh/100km",  -2,  100),
    "dayElecConsumption":   _s("dayElecConsumption",   19, 0.1,   "kwh",        0,   100_000),

    # ── CH-20 Charging ─────────────────────────────────────────────────────
    "vehElecRange":          _s("vehElecRange",          20, 0.1, "km",      0, 1_000_000),
    "chargeRemainTime":      _s("chargeRemainTime",      20, 1.0, "min",     0, 1000),
    "chargingStartTime":     _s("chargingStartTime",     20, 1.0, "unix_s",  0, 9_999_999_999),
    "chargingEndTime":       _s("chargingEndTime",       20, 1.0, "unix_s",  0, 9_999_999_999),
    "mileageSinceLastCharge":_s("mileageSinceLastCharge",20, 0.1, "km",     0, 1_000_000),

    # ── CH-21 EV RVM — BMS Pack ────────────────────────────────────────────
    "vehBMSPackVol":  _s("vehBMSPackVol",  21, 0.25,  "V",  0,      1023.75,
                          flag="vehBMSPackVolV"),
    "vehBMSPackCrnt": _s("vehBMSPackCrnt", 21, 0.05,  "A",  -1000,  2276.75,
                          offset=-1000, signed=True, flag="vehBMSPackCrntV"),
    # physical = raw*0.05 - 1000; raw=20004 → 0.2A
    "vehBMSPackSOC":  _s("vehBMSPackSOC",  21, 0.1,   "%",  0,      102.3,
                          flag="vehBMSPackSOCV"),

    # ── CH-21 EV RVM — BMS Cell ────────────────────────────────────────────
    "vehBMSCellMaxVol": _s("vehBMSCellMaxVol", 21, 0.001, "V",  0,   8.191,
                            flag="vehBMSCellMaxVolV"),
    "vehBMSCellMinVol": _s("vehBMSCellMinVol", 21, 0.001, "V",  0,   8.191,
                            flag="vehBMSCellMinVolV"),
    "vehBMSCellMaxTem": _s("vehBMSCellMaxTem", 21, 0.5,   "C",  -40, 87.5,
                            offset=-40, flag="vehBMSCellMaxTemV"),
    # physical = raw*0.5 - 40; raw=87 → 3.5°C
    "vehBMSCellMinTem": _s("vehBMSCellMinTem", 21, 0.5,   "C",  -40, 87.5,
                            offset=-40, flag="vehBMSCellMinTemV"),
    "vehBMSPtIsltnRstc":_s("vehBMSPtIsltnRstc",21, 0.5,  "kOhm",0,  8191.5,
                            flag="vehBMSPtIsltnRstcV"),

    # ── CH-21 EV RVM — Motor / Inverter ────────────────────────────────────
    "vehEPTTrInptShaftToq": _s("vehEPTTrInptShaftToq", 21, 0.5,  "Nm",  -848, 1199.5,
                                offset=-848, signed=True, flag="vehEPTTrInptShaftToqV"),
    # raw 1696 = 0 Nm (1696*0.5-848=0); raw<1696 = negative (regen)
    "vehTMInvtrCrnt": _s("vehTMInvtrCrnt", 21, 1.0,  "A",   -1024, 1023,
                          offset=-1024, signed=True, flag="vehTMInvtrCrntV"),
    "vehTMInvtrTem":  _s("vehTMInvtrTem",  21, 1.0,  "C",   -40,   215,
                          offset=-40, flag="vehTMInvtrCrntV"),
    "vehTMSpd":       _s("vehTMSpd",       21, 1.0,  "rpm", -32768, 32767,
                          offset=-32768, signed=True, flag="vehTMSpdV"),
    "vehTMActuToq":   _s("vehTMActuToq",   21, 0.5,  "Nm",  -512,  510.5,
                          offset=-512, signed=True, flag="vehTMActuToqV"),
    "vehTMSttrTem":   _s("vehTMSttrTem",   21, 1.0,  "C",   -40,   215,   offset=-40),
    "vehHVDCDCTem":   _s("vehHVDCDCTem",   21, 1.0,  "C",   -40,   215,   offset=-40),
    "vehChargerHVolt":    _s("vehChargerHVolt",     21, 0.02,  "V", 0,      1310.7),
    "vehChargerHVCurrent":_s("vehChargerHVCurrent", 21, 0.1,   "A", -204.8, 204.7,
                              offset=-204.8, signed=True),

    # ── CH-22 Thermal Runaway ──────────────────────────────────────────────
    "vehBMSCMUFlt":          _s("vehBMSCMUFlt",          22, 1.0, "level", 0, 3),
    "vehBMSCellVoltFlt":     _s("vehBMSCellVoltFlt",     22, 1.0, "level", 0, 3),
    "vehBMSPackTemFlt":      _s("vehBMSPackTemFlt",       22, 1.0, "level", 0, 3),
    "vehBMSPackVoltFlt":     _s("vehBMSPackVoltFlt",      22, 1.0, "level", 0, 3),
    "vehBMSPreThrmlFltInd":  _s("vehBMSPreThrmlFltInd",  22, 1.0, "bool",  0, 1),
    "vehBMSFltIndReq":       _s("vehBMSFltIndReq",        22, 1.0, "enum",  0, 2),
    "vehVCUSecyThrmlFltInd": _s("vehVCUSecyThrmlFltInd", 22, 1.0, "enum",  0, 2),

    # ── CH-23 Tyre Pressure ────────────────────────────────────────────────
    "frontRightTyrePressure": _s("frontRightTyrePressure", 23, 4.0, "kPa", 4, 508, invalid=128),
    "frontLeftTyrePressure":  _s("frontLeftTyrePressure",  23, 4.0, "kPa", 4, 508, invalid=128),
    "rearRightTyrePressure":  _s("rearRightTyrePressure",  23, 4.0, "kPa", 4, 508, invalid=128),
    "rearLeftTyrePressure":   _s("rearLeftTyrePressure",   23, 4.0, "kPa", 4, 508, invalid=128),
    "wheelTyreMonitorStatus": _s("wheelTyreMonitorStatus", 23, 1.0, "enum", 0, 6),
    # CRITICAL: status==3 or status==5 → skip pressure reading for that wheel
}


class SignalDecoder:
    """Converts raw TBox integer values to physical engineering units and back."""

    @staticmethod
    def decode(signal_name: str, raw_value: Any) -> float | None:
        """
        Returns physical value, or None if:
          - raw_value is None
          - raw matches invalid_raw_value sentinel
          - decoded physical is outside [physical_min, physical_max]
        """
        sig = SIGNAL_REGISTRY.get(signal_name)
        if sig is None or raw_value is None:
            return None
        if sig.invalid_raw_value is not None and raw_value == sig.invalid_raw_value:
            return None
        try:
            physical = float(raw_value) * sig.scale + sig.offset
        except (TypeError, ValueError):
            return None
        if physical < sig.physical_min or physical > sig.physical_max:
            return None
        return round(physical, 6)

    @staticmethod
    def encode(signal_name: str, physical_value: float) -> int:
        """Convert a physical value back to raw integer."""
        sig = SIGNAL_REGISTRY[signal_name]
        return round((physical_value - sig.offset) / sig.scale)

    @staticmethod
    def is_gated(signal_name: str, row: dict) -> bool:
        """Returns True if the validity flag for this signal is set to 1 (= INVALID)."""
        sig = SIGNAL_REGISTRY.get(signal_name)
        if sig is None or sig.validity_flag_field is None:
            return False
        return row.get(sig.validity_flag_field) == 1

    @staticmethod
    def decode_row(row: dict) -> dict:
        """
        Decode every signal present in *row* that exists in SIGNAL_REGISTRY.
        Gated signals → None.  Unknown keys are passed through unchanged.
        """
        out: dict[str, Any] = {}
        for key, raw in row.items():
            if key not in SIGNAL_REGISTRY:
                out[key] = raw
                continue
            if SignalDecoder.is_gated(key, row):
                out[key] = None
            else:
                out[key] = SignalDecoder.decode(key, raw)
        return out
