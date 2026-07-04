"""
AutoPredict Rule-Based Alert Engine.

Class RuleEngine.evaluate(vin, state) → List[Alert]

Severity matrix (dispatch SLA):
  CRITICAL  — immediate dispatch, < 30 seconds
  HIGH      — < 1 hour
  MEDIUM    — next business slot
  LOW       — in-app / batch notification

State dict keys accept both PascalCase (raw telemetry) and snake_case aliases.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

log = logging.getLogger(__name__)

# ── Alert dataclass ────────────────────────────────────────────────────────────

@dataclass
class Alert:
    vin:                  str
    alert_type:           str
    severity:             str          # CRITICAL | HIGH | MEDIUM | LOW
    title:                str
    message_customer:     str
    message_dealer:       str
    recommended_action:   str
    estimated_cost_min:   float        # INR
    estimated_cost_max:   float        # INR
    confidence_score:     float        # 0.0–1.0
    model_version:        str = "rule/1.0"
    triggered_at:         datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    data_snapshot_json:   str = "{}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["triggered_at"] = self.triggered_at.isoformat()
        return d


# ── Field accessor (handles PascalCase / snake_case / numeric aliases) ─────────

def _get(state: dict, *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in state:
            return state[k]
    return default


def _any_low_pressure(state: dict, threshold: float) -> str | None:
    """Return position label if any tyre is below threshold (bar), else None."""
    positions = {
        "fl": ("TyrePressureFL", "tyre_pressure_fl", "tyre_fl"),
        "fr": ("TyrePressureFR", "tyre_pressure_fr", "tyre_fr"),
        "rl": ("TyrePressureRL", "tyre_pressure_rl", "tyre_rl"),
        "rr": ("TyrePressureRR", "tyre_pressure_rr", "tyre_rr"),
    }
    for pos, keys in positions.items():
        val = _get(state, *keys)
        if val is not None and float(val) < threshold:
            return pos
    return None


# ── Rule definitions ───────────────────────────────────────────────────────────

@dataclass
class _Rule:
    alert_type:         str
    severity:           str
    title:              str
    message_customer:   str
    message_dealer:     str
    recommended_action: str
    cost_min:           float
    cost_max:           float
    cooldown_hours:     float = 24.0


def _evaluate_rules(vin: str, state: dict) -> list[Alert]:
    alerts: list[Alert] = []

    def _add(rule: _Rule, confidence: float = 1.0) -> None:
        snap = {k: v for k, v in state.items() if isinstance(v, (int, float, bool, str))}
        alerts.append(Alert(
            vin=vin,
            alert_type=rule.alert_type,
            severity=rule.severity,
            title=rule.title,
            message_customer=rule.message_customer,
            message_dealer=rule.message_dealer,
            recommended_action=rule.recommended_action,
            estimated_cost_min=rule.cost_min,
            estimated_cost_max=rule.cost_max,
            confidence_score=round(confidence, 3),
            data_snapshot_json=json.dumps(snap),
        ))

    speed       = float(_get(state, "VehSpeed", "veh_speed", "speed", default=0))
    pwr_mode    = _get(state, "VehSysPwrMod", "veh_sys_pwr_mod", "power_mode", default="OFF")
    batt_12v    = float(_get(state, "VehBatt", "veh_batt", "battery_12v_v", default=12.6))
    bms_temp_flt= int(_get(state, "vehBMSPackTemFlt", "bms_pack_temp_flt", default=0))
    cmu_flt     = int(_get(state, "vehBMSCMUFlt", "bms_cmu_flt", default=0))
    brake_fluid_low = bool(_get(state, "vehBrkFludLvlLow", "brake_fluid_low", default=False))
    oil_pressure_warn = bool(_get(state, "vehOilPressureWarning", "oil_pressure_warning", default=False))
    tyre_monitor = int(_get(state, "wheelTyreMonitorStatus", "tyre_monitor_status", default=0))
    coolant_temp = float(_get(state, "VehCoolantTemp", "coolant_temp", default=90))
    oil_life_pct = float(_get(state, "EnginOilLifePct", "engine_oil_life_pct", "oil_life_pct", default=100))
    brake_front  = float(_get(state, "BrakePadFrontMM", "brake_pad_front_mm", "brake_front_mm", default=10))
    brake_rear   = float(_get(state, "BrakePadRearMM",  "brake_pad_rear_mm",  "brake_rear_mm",  default=10))
    brake_fluid  = float(_get(state, "BrakeFluidPct", "brake_fluid_pct", default=100))
    soc          = float(_get(state, "BMSPackSOC", "bms_pack_soc", "soc", default=50))
    soh          = float(_get(state, "BMSPackSOH", "bms_pack_soh", "soh", default=100))
    cell_max_temp= float(_get(state, "BMSCellMaxTemp", "bms_cell_max_temp", default=30))
    cell_vol_max = float(_get(state, "BMSCellMaxVol", "bms_cell_max_vol", default=3.7))
    cell_vol_min = float(_get(state, "BMSCellMinVol", "bms_cell_min_vol", default=3.7))
    fuel_pct     = float(_get(state, "FuelTankLevel", "fuel_level_pct", "fuel_tank_level", default=50))
    off_minutes  = float(_get(state, "off_duration_minutes", default=0))

    # ── CRITICAL rules ────────────────────────────────────────────────────────

    if bms_temp_flt == 3 or cmu_flt == 3:
        _add(_Rule(
            alert_type="THERMAL_RUNAWAY",
            severity="CRITICAL",
            title="HV Battery Thermal Runaway Risk",
            message_customer="⚠️ Critical battery fault detected. Stop the vehicle immediately in a safe, open area and call emergency services.",
            message_dealer="BMS fault code 3 active (thermal runaway). Requires immediate BMS inspection. Do NOT charge vehicle.",
            recommended_action="Pull over safely. Do not charge. Call roadside assistance immediately.",
            cost_min=50_000,
            cost_max=3_00_000,
            cooldown_hours=0.0,
        ))

    if brake_fluid_low or brake_fluid < 10:
        _add(_Rule(
            alert_type="LOW_BRAKE_FLUID",
            severity="CRITICAL",
            title="Critical: Low Brake Fluid",
            message_customer="Brake fluid level is critically low. Do not drive — braking effectiveness is severely reduced.",
            message_dealer="BrakeFluidPct below threshold. Inspect for leak and replace fluid. Check brake pad wear.",
            recommended_action="Do not drive. Tow to nearest service centre for immediate inspection.",
            cost_min=2_000,
            cost_max=8_000,
        ))

    if oil_pressure_warn and speed > 50:
        _add(_Rule(
            alert_type="OIL_PRESSURE_CRITICAL",
            severity="CRITICAL",
            title="Critical: Low Oil Pressure at Speed",
            message_customer="Engine oil pressure warning at high speed. Reduce speed and pull over safely. Engine damage risk.",
            message_dealer="Oil pressure warning at {:.0f} km/h. Inspect oil pump, level, and engine internals.".format(speed),
            recommended_action="Reduce speed, pull over safely, switch off engine. Do not restart. Call roadside assistance.",
            cost_min=10_000,
            cost_max=80_000,
        ))

    if coolant_temp > 115:
        _add(_Rule(
            alert_type="ENGINE_OVERTEMP",
            severity="CRITICAL",
            title="Critical: Engine Overheating",
            message_customer="Engine temperature is critically high. Stop the vehicle immediately to prevent engine damage.",
            message_dealer=f"Coolant temp {coolant_temp:.0f}°C. Inspect thermostat, coolant level, radiator, and water pump.",
            recommended_action="Pull over and switch off engine. Allow to cool for 30 minutes. Check coolant level.",
            cost_min=5_000,
            cost_max=40_000,
        ))

    if min(brake_front, brake_rear) < 2.0:
        pos = "front" if brake_front < brake_rear else "rear"
        _add(_Rule(
            alert_type="BRAKE_PAD_CRITICAL",
            severity="CRITICAL",
            title=f"Critical: {pos.title()} Brake Pads Worn Out",
            message_customer=f"Your {pos} brake pads are critically thin. Schedule immediate replacement.",
            message_dealer=f"{pos.title()} brake pad <2mm. Immediate pad replacement required. Inspect rotors.",
            recommended_action="Avoid hard braking. Schedule service immediately.",
            cost_min=3_000,
            cost_max=15_000,
        ))

    if batt_12v < 11.5:
        _add(_Rule(
            alert_type="12V_BATTERY_CRITICAL",
            severity="CRITICAL",
            title="Critical: 12V Battery Voltage",
            message_customer=f"12V battery voltage is very low ({batt_12v:.1f}V). Vehicle may not start.",
            message_dealer=f"12V battery at {batt_12v:.1f}V. Test battery health and charging system.",
            recommended_action="Test battery immediately. Carry jump-start cable.",
            cost_min=4_000,
            cost_max=12_000,
        ))

    # ── HIGH rules ────────────────────────────────────────────────────────────

    if batt_12v < 12.1 and str(pwr_mode).upper() in ("OFF", "0") and off_minutes > 30:
        _add(_Rule(
            alert_type="12V_BATTERY_RISK",
            severity="HIGH",
            title="12V Battery Risk: Voltage Drop While Parked",
            message_customer=f"Your 12V battery voltage dropped to {batt_12v:.1f}V while the vehicle was off. Check for parasitic drain.",
            message_dealer=f"12V at {batt_12v:.1f}V after {off_minutes:.0f}min off. Suspect parasitic drain or failing battery.",
            recommended_action="Have battery and charging system tested at next service.",
            cost_min=4_000,
            cost_max=12_000,
        ), confidence=0.85)

    if tyre_monitor == 1:
        low_pos = _any_low_pressure(state, 1.8) or "unknown"
        _add(_Rule(
            alert_type="TYRE_DEFLATION",
            severity="HIGH",
            title="Tyre Deflation Detected",
            message_customer=f"TPMS alert: tyre pressure loss detected ({low_pos.upper()}). Slow down and check tyres.",
            message_dealer=f"TPMS status=1. Low pressure at {low_pos.upper()}. Inspect for puncture or valve leak.",
            recommended_action="Reduce speed to <60 km/h. Check tyre pressure. Look for puncture.",
            cost_min=500,
            cost_max=6_000,
        ))

    if bms_temp_flt == 2:
        _add(_Rule(
            alert_type="BMS_PACK_TEMP_WARNING",
            severity="HIGH",
            title="HV Battery Temperature Warning",
            message_customer=f"Battery temperature elevated ({cell_max_temp:.0f}°C). Avoid fast charging until inspected.",
            message_dealer=f"BMSPackTemFlt=2. Cell max temp {cell_max_temp:.0f}°C. Inspect thermal management system.",
            recommended_action="Avoid DC fast charging. Schedule service within 24 hours.",
            cost_min=8_000,
            cost_max=60_000,
        ), confidence=0.9)

    if soh < 70:
        _add(_Rule(
            alert_type="HV_BATTERY_SOH_CRITICAL",
            severity="HIGH",
            title=f"HV Battery State-of-Health Low ({soh:.0f}%)",
            message_customer=f"Your EV battery health is at {soh:.0f}%. Range and performance may be significantly reduced.",
            message_dealer=f"BMSPackSOH={soh:.0f}%. Below 70% threshold. Schedule battery health report with customer.",
            recommended_action="Schedule battery assessment. Review warranty coverage.",
            cost_min=20_000,
            cost_max=2_50_000,
        ), confidence=0.8)

    if cell_max_temp > 45:
        _add(_Rule(
            alert_type="CELL_OVERTEMP",
            severity="HIGH",
            title="HV Cell Temperature Elevated",
            message_customer=f"Battery cell temperature is high ({cell_max_temp:.0f}°C). Avoid fast charging.",
            message_dealer=f"Cell max temp {cell_max_temp:.0f}°C. Check cooling loop and coolant pump.",
            recommended_action="Park in shade. Avoid fast charging. Monitor at next drive.",
            cost_min=5_000,
            cost_max=30_000,
        ), confidence=0.85)

    if (cell_vol_max - cell_vol_min) > 0.15:
        spread = cell_vol_max - cell_vol_min
        _add(_Rule(
            alert_type="CELL_VOLTAGE_IMBALANCE",
            severity="HIGH",
            title=f"HV Cell Voltage Imbalance ({spread:.3f}V spread)",
            message_customer="Battery cell voltage imbalance detected. Schedule a battery health check.",
            message_dealer=f"Cell spread {spread:.3f}V (max {cell_vol_max:.3f}V, min {cell_vol_min:.3f}V). Check BMS and balancing circuits.",
            recommended_action="Schedule battery inspection. Avoid deep discharge.",
            cost_min=8_000,
            cost_max=80_000,
        ), confidence=0.8)

    # ── MEDIUM rules ──────────────────────────────────────────────────────────

    if 2.0 <= min(brake_front, brake_rear) < 4.0:
        pos = "front" if brake_front <= brake_rear else "rear"
        thin = min(brake_front, brake_rear)
        _add(_Rule(
            alert_type="BRAKE_PAD_WARNING",
            severity="MEDIUM",
            title=f"{pos.title()} Brake Pads Wearing Low ({thin:.1f}mm)",
            message_customer=f"Your {pos} brake pads are wearing low. Plan a brake inspection at your next service.",
            message_dealer=f"{pos.title()} pad at {thin:.1f}mm. Schedule replacement within next 5,000 km.",
            recommended_action="Book service within 30 days.",
            cost_min=3_000,
            cost_max=12_000,
        ), confidence=0.9)

    if 10 < oil_life_pct <= 20:
        _add(_Rule(
            alert_type="ENGINE_OIL_LOW",
            severity="MEDIUM",
            title=f"Engine Oil Life Low ({oil_life_pct:.0f}%)",
            message_customer=f"Engine oil life is at {oil_life_pct:.0f}%. Schedule an oil change soon.",
            message_dealer=f"Oil life {oil_life_pct:.0f}%. Due for change within next 1,000 km.",
            recommended_action="Schedule oil change at next service opportunity.",
            cost_min=2_000,
            cost_max=6_000,
        ), confidence=0.9)

    if 90 < coolant_temp <= 105:
        _add(_Rule(
            alert_type="ENGINE_TEMP_ELEVATED",
            severity="MEDIUM",
            title=f"Engine Temperature Elevated ({coolant_temp:.0f}°C)",
            message_customer="Engine running slightly warm. Check coolant level and monitor temperature.",
            message_dealer=f"Coolant temp {coolant_temp:.0f}°C. Check coolant level, thermostat, and radiator.",
            recommended_action="Check coolant level. Schedule cooling system inspection if persistent.",
            cost_min=1_000,
            cost_max=15_000,
        ), confidence=0.7)

    if 10 <= soc < 20 and str(pwr_mode).upper() not in ("OFF", "0"):
        _add(_Rule(
            alert_type="HV_SOC_LOW",
            severity="MEDIUM",
            title=f"EV Battery Charge Low ({soc:.0f}%)",
            message_customer=f"Battery charge is at {soc:.0f}%. Find a charging station soon.",
            message_dealer=f"SOC at {soc:.0f}%. Normal low-charge event.",
            recommended_action="Navigate to nearest charging station.",
            cost_min=0,
            cost_max=0,
        ), confidence=1.0)

    if _any_low_pressure(state, 2.0):
        pos = _any_low_pressure(state, 2.0)
        _add(_Rule(
            alert_type="TYRE_PRESSURE_LOW",
            severity="MEDIUM",
            title=f"Low Tyre Pressure ({pos.upper() if pos else 'unknown'})",
            message_customer="One or more tyres are below the recommended pressure. Inflate when convenient.",
            message_dealer=f"Tyre {pos} below 2.0 bar. May indicate slow leak or temperature drop.",
            recommended_action="Inflate tyres to recommended pressure. Check for slow leak.",
            cost_min=0,
            cost_max=500,
        ), confidence=0.9)

    if 70 <= soh < 80:
        _add(_Rule(
            alert_type="HV_BATTERY_SOH_MEDIUM",
            severity="MEDIUM",
            title=f"HV Battery Health Declining ({soh:.0f}%)",
            message_customer=f"Your battery health is at {soh:.0f}%. Range may be reduced compared to new.",
            message_dealer=f"SOH {soh:.0f}%. Inform customer of expected range impact. Review warranty.",
            recommended_action="Review battery health report at next service.",
            cost_min=0,
            cost_max=50_000,
        ), confidence=0.75)

    # ── LOW rules ─────────────────────────────────────────────────────────────

    if oil_life_pct > 20 and oil_life_pct <= 30:
        _add(_Rule(
            alert_type="ENGINE_OIL_ADVISORY",
            severity="LOW",
            title=f"Engine Oil Change Advisory ({oil_life_pct:.0f}% remaining)",
            message_customer=f"Engine oil at {oil_life_pct:.0f}% life. An oil change will be needed within the next 2,500 km.",
            message_dealer=f"Oil life {oil_life_pct:.0f}%. Proactive reminder sent to customer.",
            recommended_action="Plan oil change at next routine service.",
            cost_min=2_000,
            cost_max=5_000,
        ), confidence=0.8)

    if fuel_pct < 10:
        _add(_Rule(
            alert_type="LOW_FUEL",
            severity="LOW",
            title="Low Fuel Level",
            message_customer=f"Fuel level is at {fuel_pct:.0f}%. Refuel soon to avoid running empty.",
            message_dealer=f"Fuel at {fuel_pct:.0f}%.",
            recommended_action="Refuel at nearest petrol station.",
            cost_min=0,
            cost_max=0,
        ), confidence=1.0)

    if 10 <= fuel_pct < 15:
        _add(_Rule(
            alert_type="FUEL_ADVISORY",
            severity="LOW",
            title="Fuel Level Getting Low",
            message_customer=f"Fuel at {fuel_pct:.0f}%. Consider refuelling on your next stop.",
            message_dealer="",
            recommended_action="Refuel when convenient.",
            cost_min=0,
            cost_max=0,
        ), confidence=1.0)

    if soc < 10 and str(pwr_mode).upper() not in ("OFF", "0"):
        _add(_Rule(
            alert_type="HV_SOC_CRITICAL",
            severity="LOW",
            title=f"EV Battery Charge Critical ({soc:.0f}%)",
            message_customer=f"Battery charge is at {soc:.0f}%. Find a charging station immediately.",
            message_dealer=f"SOC at {soc:.0f}%. Customer notified.",
            recommended_action="Navigate to nearest charging station immediately.",
            cost_min=0,
            cost_max=0,
        ), confidence=1.0)

    return alerts


# ── RuleEngine class ───────────────────────────────────────────────────────────

class RuleEngine:
    """
    Evaluates deterministic threshold rules against a live telemetry state dict.

    Usage:
        engine = RuleEngine()
        alerts = engine.evaluate(vin="MZ7X...", current_state_dict=telemetry_row)
    """

    def evaluate(self, vin: str, current_state_dict: dict) -> list[Alert]:
        """
        Run all threshold rules against *current_state_dict* for the given *vin*.

        Returns a list of Alert objects ordered CRITICAL → HIGH → MEDIUM → LOW.
        """
        try:
            alerts = _evaluate_rules(vin, current_state_dict)
        except Exception as exc:
            log.error("RuleEngine.evaluate failed for VIN %s: %s", vin, exc, exc_info=True)
            return []

        _SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        alerts.sort(key=lambda a: _SEVERITY_ORDER.get(a.severity, 99))
        return alerts
