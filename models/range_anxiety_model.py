"""
Range Anxiety Predictor — pure-physics range estimation for EV/PHEV.

No separate ML model. Uses fleet-calibrated efficiency from the feature store
plus real-time temperature, AC load, and driver behaviour adjustments.

Designed to be called:
  • On every SoC update from the digital twin (channels 19/20/21)
  • Via the GET /api/vehicles/{vin}/range-estimate REST endpoint
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# Physics constants (India climate calibrated)
_TEMP_COLD_LOSS_PER_C  = 0.010   # 1 % range per °C below 20 °C
_TEMP_HOT_LOSS_PER_C   = 0.005   # 0.5 % per °C above 35 °C (internal resistance rise)
_AC_PENALTY_EXTREME    = 0.18    # 18 % when AC on and > 35 °C (Nagpur/Hyderabad peak)
_AC_PENALTY_HOT        = 0.12    # 12 % when AC on and 28–35 °C
_AC_PENALTY_MILD       = 0.06    # 6 % when AC on below 28 °C
_DRIVER_EFF_FLOOR      = 0.70
_DRIVER_EFF_CAP        = 1.20
_KWH_PER_100KM_FLOOR   = 10.0    # safety floor — prevents divide-by-zero
_ANXIETY_SAFETY_MARGIN = 1.30    # alert if P10 range < 130 % of typical trip

# Heat soak physics constants
_HEAT_SOAK_AMBIENT_THRESHOLD = 32.0   # °C — below this no meaningful BMS pre-cooling draw
_HEAT_SOAK_PACK_COOL_TARGET  = 35.0   # °C — BMS target temp before / during first km
_HEAT_SOAK_TRIGGER_TEMP      = 38.0   # °C — pack must exceed this to need active cooling
_HEAT_SOAK_CELL_SHC          = 1.0    # kJ / kg·°C  specific heat of NMC cell
_HEAT_SOAK_KG_PER_KWH        = 7.0    # kg/kWh  pack mass proxy (NMC chemistry)
_HEAT_SOAK_COOLING_COP       = 0.40   # heat pump / refrigerant COP at extreme temps
_HEAT_SOAK_MAX_KWH            = 4.0   # hard cap — above this is likely a sensor fault

# Defaults used when feature_store_features are absent
_DEFAULT_SOH_PCT          = 100.0
_DEFAULT_KWH_PER_100KM    = 16.5
_DEFAULT_DRIVE_SCORE      = 70.0
_DEFAULT_EFFICIENCY_STD   = 2.0
_DEFAULT_TYPICAL_TRIP_KM  = 40.0
_DEFAULT_RATED_RANGE_KM   = 338.0
_DEFAULT_BATTERY_KWH      = 50.3


def estimate_heat_soak_energy_draw(
    ambient_temp_c: float,
    park_duration_hours: float,
    is_sunny: bool,
    battery_capacity_kwh: float,
) -> float:
    """Estimate kWh the BMS thermal system will consume in the first ~12 minutes
    after starting a hot-parked EV before useful driving begins.

    When a vehicle sits in direct sun at 44 °C for 2+ hours the pack reaches
    38–48 °C. BMS activates cooling immediately on start, consuming 1.5–3 kWh
    that the original SoC-based formula does not account for.

    Args:
        ambient_temp_c:       vehOutsideTemp at journey start (°C)
        park_duration_hours:  hours since last power-off (from twin last session end)
        is_sunny:             True if parked in direct sun (no shade).
                              Infer from vehNightDetected=0 + 09:00–17:00 + no rain.
        battery_capacity_kwh: nominal pack size from fleet_master

    Returns:
        Estimated pre-driving energy draw in kWh (0.0 if no correction needed).
    """
    if ambient_temp_c < _HEAT_SOAK_AMBIENT_THRESHOLD:
        return 0.0

    # Pack insulation slows heat ingress — full soak takes ~3.5 h
    park_factor = min(1.0, park_duration_hours / 3.5)
    pack_overheat_c = (8.0 if is_sunny else 4.0) * park_factor
    estimated_pack_temp = ambient_temp_c + pack_overheat_c

    if estimated_pack_temp <= _HEAT_SOAK_TRIGGER_TEMP:
        return 0.0

    degrees_to_cool  = max(0.0, estimated_pack_temp - _HEAT_SOAK_PACK_COOL_TARGET)
    pack_mass_kg     = battery_capacity_kwh * _HEAT_SOAK_KG_PER_KWH
    heat_to_remove_kj = pack_mass_kg * _HEAT_SOAK_CELL_SHC * degrees_to_cool
    energy_draw_kwh  = (heat_to_remove_kj / 3600.0) / _HEAT_SOAK_COOLING_COP
    return round(min(energy_draw_kwh, _HEAT_SOAK_MAX_KWH), 2)


class RangeAnxietyPredictor:
    """
    Predict instantaneous usable range and flag range anxiety.

    Parameters
    ----------
    vin                    : vehicle identifier (for logging only)
    current_soc_pct        : live SoC from telemetry (0–100)
    current_outside_temp_c : ambient temperature (°C)
    ac_is_on               : whether cabin AC is active
    feature_store_features : dict from FeatureStore (battery_hv + driver groups)
    fleet_row              : dict from fleet CSV for this VIN
    """

    def predict(
        self,
        vin: str,
        current_soc_pct: float,
        current_outside_temp_c: float,
        ac_is_on: bool,
        feature_store_features: dict,
        fleet_row: dict,
        *,
        park_duration_hours: float = 0.0,
        is_sunny: bool = False,
    ) -> dict:
        """Predict usable range with heat-soak correction.

        New keyword-only args (both default to the safe/conservative case, so
        all existing callers continue to work without modification):
            park_duration_hours: hours since last power-off.  0 = car was not parked
                                 long enough to need correction (e.g. mid-trip call).
            is_sunny:            True if parked in direct sun between 09:00–17:00
                                 with no rain detected.
        """
        fs = feature_store_features  # shorthand

        nominal_range_km  = float(fleet_row.get("rated_range_km",      _DEFAULT_RATED_RANGE_KM))
        nominal_kwh       = float(fleet_row.get("battery_capacity_kwh", _DEFAULT_BATTERY_KWH))
        soh_pct           = float(fs.get("soh_estimated",              _DEFAULT_SOH_PCT))
        kwh_per_100km     = float(fs.get("range_per_kwh_30d_trend",    _DEFAULT_KWH_PER_100KM))

        # ── Energy available at current SoC ──────────────────────────────────
        usable_soh_fraction  = min(1.0, soh_pct / 100.0)
        energy_available_kwh = nominal_kwh * usable_soh_fraction * current_soc_pct / 100.0

        # ── Heat soak deduction ───────────────────────────────────────────────
        # BMS pre-conditioning draw happens before the first km is driven and
        # is invisible to the SoC-based calculation above.
        heat_soak_kwh = estimate_heat_soak_energy_draw(
            current_outside_temp_c, park_duration_hours, is_sunny, nominal_kwh
        )
        usable_kwh_after_soak = max(0.0, energy_available_kwh - heat_soak_kwh)

        # ── Temperature efficiency adjustment ────────────────────────────────
        t = current_outside_temp_c
        if t < 20.0:
            temp_efficiency = 1.0 - max(0.0, (20.0 - t) * _TEMP_COLD_LOSS_PER_C)
        elif t > 35.0:
            temp_efficiency = 1.0 - max(0.0, (t - 35.0) * _TEMP_HOT_LOSS_PER_C)
        else:
            temp_efficiency = 1.0

        # ── AC load penalty (three-tier: extreme / hot / mild) ───────────────
        if ac_is_on and t > 35.0:
            ac_efficiency_penalty = _AC_PENALTY_EXTREME
        elif ac_is_on and t > 28.0:
            ac_efficiency_penalty = _AC_PENALTY_HOT
        elif ac_is_on:
            ac_efficiency_penalty = _AC_PENALTY_MILD
        else:
            ac_efficiency_penalty = 0.0

        # ── Driver efficiency vs fleet average (baseline = score 70) ────────
        drive_score       = float(fs.get("composite_drive_score", _DEFAULT_DRIVE_SCORE))
        driver_efficiency = drive_score / _DEFAULT_DRIVE_SCORE
        driver_efficiency = max(_DRIVER_EFF_FLOOR, min(_DRIVER_EFF_CAP, driver_efficiency))

        # ── Effective consumption for this trip ──────────────────────────────
        combined_efficiency = temp_efficiency * driver_efficiency * (1.0 - ac_efficiency_penalty)
        if combined_efficiency <= 0:
            combined_efficiency = 0.01
        effective_kwh_per_100km = kwh_per_100km / combined_efficiency
        effective_kwh_per_100km = max(effective_kwh_per_100km, _KWH_PER_100KM_FLOOR)

        # ── Predicted range (from post-soak usable energy) ───────────────────
        predicted_range_km = (usable_kwh_after_soak / effective_kwh_per_100km) * 100.0

        # ── Confidence interval from 30-day efficiency std dev ───────────────
        efficiency_std = float(fs.get("kwh_per_100km_std_30d", _DEFAULT_EFFICIENCY_STD))
        spread_fraction = efficiency_std / effective_kwh_per_100km
        range_p10_km = predicted_range_km * (1.0 - spread_fraction)
        range_p90_km = predicted_range_km * (1.0 + spread_fraction)

        # ── Range anxiety flag ────────────────────────────────────────────────
        typical_trip_km = float(fs.get("km_per_day_30d_avg", _DEFAULT_TYPICAL_TRIP_KM))
        anxiety_flag    = range_p10_km < typical_trip_km * _ANXIETY_SAFETY_MARGIN

        log.debug(
            "[%s] range=%.0f km (P10=%.0f P90=%.0f) SoC=%.1f%% temp=%.1f°C "
            "ac=%s heat_soak=%.2f kWh park=%.1fh sunny=%s anxiety=%s",
            vin, predicted_range_km, range_p10_km, range_p90_km,
            current_soc_pct, current_outside_temp_c,
            ac_is_on, heat_soak_kwh, park_duration_hours, is_sunny, anxiety_flag,
        )

        return {
            "predicted_range_km":            round(predicted_range_km, 0),
            "range_p10_km":                  round(range_p10_km, 0),
            "range_p90_km":                  round(range_p90_km, 0),
            "energy_available_kwh":          round(energy_available_kwh, 2),
            "heat_soak_energy_consumed_kwh": round(heat_soak_kwh, 2),
            "effective_kwh_per_100km":       round(effective_kwh_per_100km, 1),
            "temp_efficiency_factor":        round(temp_efficiency, 3),
            "driver_efficiency_factor":      round(driver_efficiency, 3),
            "range_anxiety_flag":            anxiety_flag,
            "anxiety_reason": (
                "Low range relative to typical daily distance" if anxiety_flag else None
            ),
        }
