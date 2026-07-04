"""
Synthetic Telemetry Generator — physically consistent 1-second vehicle telemetry.

Physical rules enforced:
 1. Driving sessions (1-4/day) with SysPwrMod state machine
 2. Speed/RPM/gear correlation via gear-ratio model
 3. Gear selection by speed thresholds
 4. Brake/accel mutual exclusion (both cannot exceed 20 simultaneously)
 5. Fuel/energy consumption physics (ICE RPM-based; EV SOC depletes + regen)
 6. Seasonal temperature + coolant exponential warmup curve
 7. 12V battery voltage tracks alternator charging / parasitic drain
 8. Tyre pressure slow leak (-0.5 kPa/day) + temperature effect
 9. HV battery SOC/SoH per charge-cycle degradation
10. Failure injection for 15% VINs with pre-failure intensity curves
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from synthetic.config import SyntheticConfig, DRIVER_ARCHETYPES

# ── Physical constants ─────────────────────────────────────────────────────

# RPM = speed_kph × factor[gear]
_GEAR_RPM_FACTOR = [0.0, 170.0, 65.0, 37.0, 26.0, 19.0, 15.0]  # idx 0=idle, 1-6

# Speed thresholds for gear changes (kph)
_GEAR_UP_SPEED   = [0, 10, 30, 60, 90, 110, 999]
_GEAR_DOWN_SPEED = [0,  5, 20, 50, 80, 100, 999]

IDLE_RPM = 820
MAX_RPM  = 6800

_AC_PENALTY   = 0.25   # fuel rate multiplier when AC is on
_REGEN_FRAC   = 0.18   # fraction of braking energy recovered by EV regen

_COOLANT_FINAL  = 91.0    # normal operating temp (°C)
_COOLANT_TAU_S  = 240     # warm-up time constant (seconds)
_OVERHEAT_FINAL = 107.0   # failure-injection target temp (°C)

_BATT_CHARGE_V   = 14.1   # V while alternator running
_BATT_DRAIN_RATE = 0.003  # V/minute when engine off

_TYRE_NORMAL  = np.array([232.0, 232.0, 226.0, 226.0])  # FL FR RL RR (kPa)
_TYRE_LEAK    = 0.5 / (24 * 60)                          # kPa/second natural leak
_TYRE_TEMP_K  = 0.10                                     # kPa/°C above 25°C baseline

# ── Driver profile parameters ──────────────────────────────────────────────

def _profile_from_archetype(name: str) -> dict:
    """Build speed-profile params from DRIVER_ARCHETYPES entry."""
    a = DRIVER_ARCHETYPES.get(name)
    if a is None:
        a = DRIVER_ARCHETYPES.get("urban_commuter", {})
    ms = a.get("max_speed_kph", 80)
    accel_tau = max(1.0, 10.0 - a.get("accel_rate_kph_per_s", 4.0))
    idle_f = a.get("idle_fraction", 0.15)
    return {
        "max_speed": ms,
        "target_bands": [(ms * 0.5, ms), (ms * 0.25, ms * 0.7), (0, ms * 0.4)],
        "accel_tau": accel_tau,
        "idle_fraction": idle_f,
        "cruise_prob": a.get("cruise_prob", 0.05),
    }


_PROFILE: dict[str, dict] = {}
for _name in DRIVER_ARCHETYPES:
    _PROFILE[_name] = _profile_from_archetype(_name)
# Legacy aliases
_PROFILE["normal"] = _PROFILE.get("urban_commuter", _profile_from_archetype("urban_commuter"))
_PROFILE["eco"] = _PROFILE.get("eco_driver", _profile_from_archetype("eco_driver"))


class TelemetryGenerator:
    """Generate physically realistic 1 Hz telemetry for every VIN in a fleet."""

    def __init__(self, cfg: SyntheticConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    # ── Public entry point ─────────────────────────────────────────────────

    def generate_all(self, fleet_df: pd.DataFrame, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        failure_plan   = self._plan_failures(fleet_df)
        manifest_rows: list[dict] = []
        combined_chunks: list[pd.DataFrame] = []

        for _, vrow in fleet_df.iterrows():
            vin   = str(vrow["vin"])
            specs = failure_plan.get(vin, [])
            print(f"  {vin} ({vrow['fuel_type']}, {vrow['driver_profile']}) ...", end=" ", flush=True)

            df = self._generate_for_vin(vrow, specs)
            csv_path = out_dir / f"telemetry_{vin}.csv"
            df.to_csv(csv_path, index=False)
            print(f"OK {len(df):,} rows -> {csv_path.name}")

            sample = df.sample(min(10_000, len(df)), random_state=42)
            combined_chunks.append(sample)

            for sp in specs:
                manifest_rows.append({
                    "vin":                vin,
                    "failure_type":       sp["ftype"],
                    "injection_date":     sp["injection_date"].date().isoformat(),
                    "failure_date":       sp["failure_date"].date().isoformat(),
                    "affected_component": sp.get("affected", ""),
                })

        combined = pd.concat(combined_chunks, ignore_index=True)
        combined.to_csv(out_dir / "telemetry_combined.csv", index=False)
        print(f"telemetry_combined.csv: {len(combined):,} rows")

        pd.DataFrame(manifest_rows).to_csv(out_dir / "failures_manifest.csv", index=False)
        print(f"failures_manifest.csv: {len(manifest_rows)} events")

    # ── Failure planning ─────────────────────────────────────────────────────

    def _plan_failures(self, fleet_df: pd.DataFrame) -> dict[str, list[dict]]:
        n_fail    = max(1, int(len(fleet_df) * self.cfg.failure_injection_rate))
        fail_vins = self.rng.choice(fleet_df["vin"].tolist(), size=n_fail, replace=False).tolist()
        start_dt  = datetime.strptime(self.cfg.start_date, "%Y-%m-%d")

        plans: dict[str, list[dict]] = {}
        for vin in fail_vins:
            ftype    = str(self.rng.choice(self.cfg.failure_types))
            fail_day = int(self.rng.integers(self.cfg.num_days // 2, self.cfg.num_days))
            pre_days = {
                "brake_degradation":      30,
                "oil_degradation":        21,
                "hv_battery_degradation": 90,
                "12v_battery_failure":    14,
                "tyre_puncture":           0,
                "overheating":             7,
            }.get(ftype, 14)
            inj_day  = max(0, fail_day - pre_days)
            affected = str(self.rng.choice(["FL", "FR", "RL", "RR"])) if ftype == "tyre_puncture" else ""
            plans[vin] = [{
                "ftype":          ftype,
                "injection_date": start_dt + timedelta(days=inj_day),
                "failure_date":   start_dt + timedelta(days=fail_day),
                "affected":       affected,
            }]
        return plans

    def _failure_intensity(self, specs: list[dict], now: datetime) -> dict:
        ctx: dict = {}
        for sp in specs:
            inj_dt  = sp["injection_date"]
            fail_dt = sp["failure_date"]
            if now < inj_dt:
                continue
            if now >= fail_dt:
                ctx[sp["ftype"]] = 1.0
            else:
                span    = (fail_dt - inj_dt).total_seconds()
                elapsed = (now - inj_dt).total_seconds()
                ctx[sp["ftype"]] = elapsed / span if span > 0 else 1.0
            if sp.get("affected"):
                ctx["_affected"] = sp["affected"]
        return ctx

    # ── Per-VIN generation ───────────────────────────────────────────────────

    def _generate_for_vin(self, vrow: pd.Series, specs: list[dict]) -> pd.DataFrame:
        fuel_type      = str(vrow["fuel_type"])
        driver_profile = str(vrow["driver_profile"])
        vin            = str(vrow["vin"])
        odometer       = float(vrow["initial_odometer"])
        home_lat       = float(vrow.get("home_lat",  19.08))
        home_long      = float(vrow.get("home_long", 72.88))

        bat_kwh      = float(vrow["battery_capacity_kwh"]) if pd.notna(vrow.get("battery_capacity_kwh")) else None
        fuel_tank_l  = float(vrow["fuel_tank_l"])          if pd.notna(vrow.get("fuel_tank_l"))          else 50.0
        base_l100km  = float(vrow["base_fuel_l100km"])     if pd.notna(vrow.get("base_fuel_l100km"))     else 9.0

        start_dt = datetime.strptime(self.cfg.start_date, "%Y-%m-%d")

        # Per-VIN persistent state
        soc_pct       = float(self.rng.uniform(70, 90)) if fuel_type in ("EV", "PHEV") else None
        soh_pct       = 98.5 if fuel_type in ("EV", "PHEV") else None
        charge_cycles = 0
        fuel_l        = fuel_tank_l * float(self.rng.uniform(0.5, 0.9))
        batt_12v      = 12.6
        tyre_kpa      = _TYRE_NORMAL.copy()
        brake_front   = 10.0
        brake_rear    = 9.5
        brake_fluid   = 100.0
        oil_life      = 100.0

        all_frames: list[pd.DataFrame] = []

        for day_idx in range(self.cfg.num_days):
            date     = start_dt + timedelta(days=day_idx)
            doy      = date.timetuple().tm_yday
            out_temp = 28.0 + 6.0 * math.sin(2 * math.pi * (doy - 60) / 365)
            fail_ctx = self._failure_intensity(
                specs, datetime.combine(date, datetime.min.time())
            )

            n_sessions    = int(self.rng.integers(1, 5))
            session_hours = sorted(float(h) for h in self.rng.uniform(6, 22, size=n_sessions))
            odo_day_start = odometer

            for s_idx, hour in enumerate(session_hours):
                session_start = date + timedelta(
                    hours=hour, minutes=int(self.rng.integers(0, 60))
                )
                duration_s = int(self.rng.integers(15 * 60, 120 * 60))

                (frame, odometer, soc_pct, fuel_l,
                 last_batt_12v, charge_cycles, soh_pct) = self._build_session(
                    vin=vin,
                    start_dt=session_start,
                    n_sec=duration_s,
                    odometer=odometer,
                    soc_pct=soc_pct,
                    soh_pct=soh_pct,
                    charge_cycles=charge_cycles,
                    fuel_l=fuel_l,
                    fuel_tank_l=fuel_tank_l,
                    base_l100km=base_l100km,
                    bat_kwh=bat_kwh,
                    batt_12v=batt_12v,
                    tyre_kpa=tyre_kpa,
                    brake_front=brake_front,
                    brake_rear=brake_rear,
                    brake_fluid=brake_fluid,
                    oil_life=oil_life,
                    out_temp=out_temp,
                    home_lat=home_lat,
                    home_long=home_long,
                    driver_profile=driver_profile,
                    fuel_type=fuel_type,
                    fail_ctx=fail_ctx,
                )
                all_frames.append(frame)
                batt_12v = last_batt_12v

                # Parasitic drain during park gap between sessions
                park_h   = (session_hours[s_idx + 1] - hour) if s_idx < n_sessions - 1 else 2.0
                batt_12v = max(11.2, batt_12v - _BATT_DRAIN_RATE * park_h * 60)

                if "tyre_puncture" in fail_ctx:
                    tyre_idx = {"FL": 0, "FR": 1, "RL": 2, "RR": 3}.get(
                        str(fail_ctx.get("_affected", "FL")), 0
                    )
                    tyre_kpa[tyre_idx] = max(
                        80.0, _TYRE_NORMAL[tyre_idx] - fail_ctx["tyre_puncture"] * 140
                    )

                if "12v_battery_failure" in fail_ctx:
                    batt_12v = min(batt_12v, 12.5 - fail_ctx["12v_battery_failure"] * 0.8)

            # Daily tyre slow leak
            tyre_kpa -= _TYRE_LEAK * 86_400
            tyre_kpa  = np.clip(tyre_kpa, 80, 280)

            # Brake wear proportional to km driven today
            day_km = odometer - odo_day_start
            arch = DRIVER_ARCHETYPES.get(driver_profile, {})
            harsh_rate = arch.get("harsh_brake_per_100km", 3.0)
            w = 0.00001 * (1 + harsh_rate / 5.0)
            if "brake_degradation" in fail_ctx:
                w *= 1.0 + 1.5 * fail_ctx["brake_degradation"]
            brake_front = max(0.5, brake_front - day_km * w * 1.1)
            brake_rear  = max(0.5, brake_rear  - day_km * w)
            brake_fluid = max(50.0, brake_fluid - day_km * 0.0001)

            oil_rate = 0.001 + (0.0008 if "oil_degradation" in fail_ctx else 0.0)
            oil_life  = max(0.0, oil_life - day_km * oil_rate)
            if oil_life <= 0:
                oil_life = 100.0  # simulated oil service

        return pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()

    # ── Session builder ──────────────────────────────────────────────────────

    def _build_session(
        self, vin, start_dt, n_sec, odometer, soc_pct, soh_pct, charge_cycles,
        fuel_l, fuel_tank_l, base_l100km, bat_kwh, batt_12v, tyre_kpa,
        brake_front, brake_rear, brake_fluid, oil_life, out_temp,
        home_lat, home_long, driver_profile, fuel_type, fail_ctx,
    ):
        rng = self.rng

        # ── Speed profile ─────────────────────────────────────────────────
        speed = _gen_speed_profile(n_sec, driver_profile, rng)

        # ── Gear selection ─────────────────────────────────────────────────
        gear  = np.zeros(n_sec, dtype=int)
        cur_g = 1
        for t in range(n_sec):
            s = speed[t]
            if s < 0.5:
                cur_g = 0
            else:
                if cur_g == 0:
                    cur_g = 1
                if cur_g < 6 and s > _GEAR_UP_SPEED[cur_g]:
                    cur_g = min(6, cur_g + 1)
                elif cur_g > 1 and s < _GEAR_DOWN_SPEED[cur_g - 1]:
                    cur_g = max(1, cur_g - 1)
            gear[t] = cur_g

        # ── RPM ───────────────────────────────────────────────────────────
        rpm_factors = np.array([_GEAR_RPM_FACTOR[g] for g in gear])
        rpm = np.where(gear > 0, speed * rpm_factors, float(IDLE_RPM))
        rpm = np.clip(rpm + rng.normal(0, 60, n_sec), 600, MAX_RPM)

        # ── SysPwrMod state machine: startup → running → shutdown ────────
        pwr_mod       = np.full(n_sec, 2, dtype=int)  # 2 = driving
        ramp_s        = min(5, n_sec)
        pwr_mod[:ramp_s] = [0, 1, 3, 3, 2][:ramp_s]
        pwr_mod[-ramp_s:] = [2, 2, 1, 1, 0][::-1][:ramp_s]

        # ── Accel and brake pedals (mutually exclusive >20) ──────────────
        dspeed    = np.gradient(speed)
        accel_pos = np.where(dspeed > 0.1, np.clip(dspeed * 8.0, 0, 100), 0.0)
        accel_pos += np.where(speed > 5,   np.clip(speed * 0.3, 5, 40), 0.0)
        accel_pos  = np.clip(accel_pos + rng.normal(0, 2, n_sec), 0, 100)

        brake_pos = np.where(dspeed < -0.3, np.clip(-dspeed * 12.0, 0, 100), 0.0)
        brake_pos = np.clip(brake_pos + rng.normal(0, 1, n_sec), 0, 100)
        brake_pos = np.where((accel_pos > 20) & (brake_pos > 20), 0.0, brake_pos)

        # Steering angle and forward acceleration
        steering = np.clip(np.cumsum(rng.normal(0, 0.5, n_sec)) * 5.0, -180, 180)
        accel_x  = np.clip(dspeed / 3.6 * 10, -200, 200)

        # ── Coolant temperature (exponential warm-up) ────────────────────
        t_final = _COOLANT_FINAL
        if "overheating" in fail_ctx:
            t_final += (_OVERHEAT_FINAL - _COOLANT_FINAL) * fail_ctx["overheating"]
        ts           = np.arange(n_sec, dtype=float)
        coolant_temp = t_final - (t_final - out_temp) * np.exp(-ts / _COOLANT_TAU_S)
        coolant_temp = np.clip(coolant_temp + rng.normal(0, 0.5, n_sec), out_temp, 115.0)
        oil_temp     = np.clip(coolant_temp - 5 + rng.normal(0, 1, n_sec), out_temp, 130.0)

        # ── Fuel consumption (ICE / PHEV) ──────────────────────────────────
        rpm_factor    = np.clip((rpm - 800) / 3000, 0, 1)
        ac_on         = float(out_temp > 26)
        fuel_rate_l_s = (
            (base_l100km / 100) * (speed / 3600) *
            (1 + 0.35 * rpm_factor) *
            (1 + _AC_PENALTY * ac_on)
        )
        fuel_rate_l_s = np.where(speed < 1, base_l100km / 100 * 0.4 / 3600, fuel_rate_l_s)
        if "oil_degradation" in fail_ctx:
            fuel_rate_l_s *= 1.0 + 0.15 * fail_ctx["oil_degradation"]

        # ── 12V battery voltage (alternator charges while running) ────────
        batt_v_arr = np.full(n_sec, _BATT_CHARGE_V) + rng.normal(0, 0.05, n_sec)
        if "12v_battery_failure" in fail_ctx:
            batt_v_arr -= fail_ctx["12v_battery_failure"] * 1.5

        # ── Tyre pressures ─────────────────────────────────────────────────
        tyre_temp = out_temp + 15 + speed * 0.1 + rng.normal(0, 3, n_sec)
        tp        = tyre_kpa.copy()
        tyre_fl   = tp[0] + _TYRE_TEMP_K * (tyre_temp - 25) - rng.uniform(0, 0.3, n_sec)
        tyre_fr   = tp[1] + _TYRE_TEMP_K * (tyre_temp - 25) - rng.uniform(0, 0.3, n_sec)
        tyre_rl   = tp[2] + _TYRE_TEMP_K * (tyre_temp - 25) - rng.uniform(0, 0.3, n_sec)
        tyre_rr   = tp[3] + _TYRE_TEMP_K * (tyre_temp - 25) - rng.uniform(0, 0.3, n_sec)
        if "tyre_puncture" in fail_ctx and fail_ctx["tyre_puncture"] > 0.8:
            arr_map = {"FL": tyre_fl, "FR": tyre_fr, "RL": tyre_rl, "RR": tyre_rr}
            pt_arr  = arr_map.get(str(fail_ctx.get("_affected", "FL")), tyre_fl)
            drop    = 120 * min(1.0, (fail_ctx["tyre_puncture"] - 0.8) / 0.2)
            mid     = n_sec // 2
            pt_arr[mid:] = np.maximum(80, pt_arr[mid:] - drop)

        # ── EV / PHEV HV battery ────────────────────────────────────────────
        hv_soc_arr  = np.full(n_sec, np.nan)
        hv_soh_arr  = np.full(n_sec, np.nan)
        hv_pack_v   = np.full(n_sec, np.nan)
        hv_pack_a   = np.full(n_sec, np.nan)
        hv_cell_max = np.full(n_sec, np.nan)
        hv_cell_min = np.full(n_sec, np.nan)
        hv_temp_arr = np.full(n_sec, np.nan)

        if fuel_type in ("EV", "PHEV") and bat_kwh and soc_pct is not None:
            v_ms      = speed / 3.6
            power_kw  = 0.10 * speed + 0.03 * np.abs(accel_x) * v_ms / 100 + ac_on * 1.5
            regen     = np.where(dspeed < -0.5, np.abs(dspeed) * v_ms * 0.8 * _REGEN_FRAC, 0.0)
            net_e_kwh = (power_kw - regen) / 3600  # per second → kWh/s

            soc_arr = np.empty(n_sec)
            cur_soc = soc_pct
            for t in range(n_sec):
                cur_soc    = float(np.clip(cur_soc - net_e_kwh[t] / bat_kwh * 100, 0, 100))
                soc_arr[t] = cur_soc
            soc_pct = float(soc_arr[-1])

            if soc_pct < 20:
                charge_cycles += 1
                soc_pct = min(90.0, soc_pct + 50.0)   # fast charge
                soh_pct = max(60.0, (soh_pct or 98.5) - 0.06)

            hv_soc_arr = soc_arr
            hv_soh_val = soh_pct or 98.5
            hv_soh_arr = np.full(n_sec, hv_soh_val)

            cell_v  = 3.3 + soc_arr / 100 * 0.9
            spread  = 0.02 + (1 - hv_soh_val / 100) * 0.6
            if "hv_battery_degradation" in fail_ctx:
                spread += 0.04 * fail_ctx["hv_battery_degradation"]
            hv_cell_max = np.clip(cell_v + spread / 2 + rng.normal(0, 0.003, n_sec), 3.0, 4.25)
            hv_cell_min = np.clip(cell_v - spread / 2 + rng.normal(0, 0.003, n_sec), 2.8, 4.20)

            nom_v       = 400 if bat_kwh > 30 else 200
            hv_pack_v   = nom_v * (soc_arr / 100 * 0.15 + 0.85) + rng.normal(0, 1, n_sec)
            hv_pack_a   = np.clip(power_kw * 1000 / (hv_pack_v + 1e-6), -200, 500)
            hv_temp_arr = np.clip(25 + power_kw / bat_kwh * 8 + rng.normal(0, 0.5, n_sec), -10, 50)

        # ── Odometer ───────────────────────────────────────────────────────
        odo_arr  = odometer + np.cumsum(speed / 3600)
        odometer = float(odo_arr[-1])

        # ── Fuel level ─────────────────────────────────────────────────────
        if fuel_type in ("ICE", "PHEV"):
            fuel_l = max(2.0, fuel_l - float(np.sum(fuel_rate_l_s)))
        fuel_pct = float(np.clip(fuel_l / fuel_tank_l * 100, 0, 100))

        # ── GPS random walk around home location ───────────────────────────
        lat_walk  = home_lat  + np.cumsum(rng.normal(0, 0.00008, n_sec))
        long_walk = home_long + np.cumsum(rng.normal(0, 0.00010, n_sec))
        gnss_head = (np.arctan2(np.gradient(long_walk), np.gradient(lat_walk)) * 180 / math.pi) % 360
        gnss_sats = np.clip(rng.integers(8, 15, n_sec), 4, 14)

        # ── Timestamps ─────────────────────────────────────────────────────
        unix_ts = [int((start_dt + timedelta(seconds=i)).timestamp()) for i in range(n_sec)]
        dates   = [(start_dt + timedelta(seconds=i)).strftime("%Y-%m-%d") for i in range(n_sec)]

        # Label: pick the first non-internal key from fail_ctx
        fail_keys = [k for k in fail_ctx if not k.startswith("_")]
        fail_label     = fail_keys[0] if fail_keys else ""
        fail_intensity = round(float(fail_ctx.get(fail_label, 0.0)), 3) if fail_label else 0.0

        # ── Assemble DataFrame ─────────────────────────────────────────────
        frame = pd.DataFrame({
            "VIN":                  vin,
            "StartTime-TimeStamp":  unix_ts,
            "StartTime-Date":       dates,
            "VehSpeed":             np.round(speed, 1),
            "VehSysPwrMod":         pwr_mod,
            "VehRPM":               np.round(rpm).astype(int),
            "VehGearPos":           gear,
            "VehSteeringAngle":     np.round(steering, 1),
            "VehBrakePos":          np.round(brake_pos, 1),
            "VehAccelPos":          np.round(accel_pos, 1),
            "VehAccelX":            np.round(accel_x, 1),
            "VehBatt":              np.round(batt_v_arr, 2),
            "VehOdo":               np.round(odo_arr, 2),
            "FuelTankLevel":        np.round(np.full(n_sec, fuel_pct), 1),
            "VehFuelConsumed":      np.round(
                np.cumsum(fuel_rate_l_s) if fuel_type in ("ICE", "PHEV") else np.zeros(n_sec), 4
            ),
            "EnginOilLifePct":      np.round(np.clip(oil_life + rng.normal(0, 0.1, n_sec), 0, 100), 1),
            "VehCoolantTemp":       np.round(coolant_temp, 1),
            "VehOutsideTemp":       np.round(out_temp + rng.normal(0, 0.3, n_sec), 1),
            "VehEngineOilTemp":     np.round(oil_temp, 1),
            "BrakePadFrontMM":      np.round(brake_front + rng.normal(0, 0.02, n_sec), 2),
            "BrakePadRearMM":       np.round(brake_rear  + rng.normal(0, 0.02, n_sec), 2),
            "BrakeFluidPct":        np.round(brake_fluid + rng.normal(0, 0.05, n_sec), 1),
            # HV battery (NaN for pure ICE)
            "BMSPackVol":           np.round(hv_pack_v, 1),
            "BMSPackCrnt":          np.round(hv_pack_a, 2),
            "BMSPackSOC":           np.round(hv_soc_arr, 1),
            "BMSPackSOH":           np.round(hv_soh_arr, 1),
            "BMSCellMaxVol":        np.round(hv_cell_max, 4),
            "BMSCellMinVol":        np.round(hv_cell_min, 4),
            "BMSCellMaxTemp":       np.round(hv_temp_arr, 1),
            "BMSCellMinTemp":       np.round(
                np.where(np.isnan(hv_temp_arr), np.nan,
                         hv_temp_arr - 3 + rng.normal(0, 0.5, n_sec)), 1
            ),
            "SOCValid":             np.where(np.isnan(hv_soc_arr), np.nan, 0),
            # Tyres
            "TyrePressureFL":       np.round(tyre_fl, 1),
            "TyrePressureFR":       np.round(tyre_fr, 1),
            "TyrePressureRL":       np.round(tyre_rl, 1),
            "TyrePressureRR":       np.round(tyre_rr, 1),
            "TyreTempFL":           np.round(tyre_temp + rng.normal(0, 1, n_sec), 1),
            "TyreTempFR":           np.round(tyre_temp + rng.normal(0, 1, n_sec), 1),
            "TyreTempRL":           np.round(tyre_temp + rng.normal(0, 1, n_sec), 1),
            "TyreTempRR":           np.round(tyre_temp + rng.normal(0, 1, n_sec), 1),
            # GNSS
            "GNSSLat":              np.round(lat_walk, 6),
            "GNSSLong":             np.round(long_walk, 6),
            "GNSSAlt":              np.round(
                10 + np.cumsum(rng.normal(0, DRIVER_ARCHETYPES.get(driver_profile, {}).get("elevation_change_m_per_km", 0) * 0.01 + 0.3, n_sec)),
            1),
            "GNSSHead":             np.round(gnss_head, 1),
            "GNSSSats":             gnss_sats,
            # Failure labels for ML training
            "_failure_type":        fail_label,
            "_failure_intensity":   fail_intensity,
        })

        last_batt_12v = float(batt_v_arr[-1])
        return frame, odometer, soc_pct, fuel_l, last_batt_12v, charge_cycles, soh_pct


# ── Speed profile generator ──────────────────────────────────────────────────

def _gen_speed_profile(n: int, profile: str, rng: np.random.Generator) -> np.ndarray:
    """Smooth physically realistic speed trace using Gaussian filter."""
    from scipy.ndimage import gaussian_filter1d

    p     = _PROFILE[profile]
    max_s = p["max_speed"]
    bands = p["target_bands"]

    speed   = np.zeros(n, dtype=float)
    v       = 0.0
    stop_cd = 0

    for i in range(n):
        if i < 15:
            v = min(v + 2.5, 25.0)
        elif i > n - 30:
            v = max(v - 2.0, 0.0)
        elif stop_cd > 0:
            v = 0.0
            stop_cd -= 1
        else:
            r = float(rng.random())
            if r < 0.008:
                stop_cd = int(rng.integers(15, 55))
                v = 0.0
            elif r < 0.025:
                band   = bands[int(rng.integers(0, len(bands)))]
                target = float(rng.uniform(band[0], band[1]))
                dv     = target - v
                v      = v + float(np.sign(dv)) * min(abs(dv), float(rng.uniform(8, 22)))
                v      = float(np.clip(v, 0, max_s))
            else:
                v = float(np.clip(v + float(rng.normal(0, 1.2)), 0, max_s))
        speed[i] = v

    speed = gaussian_filter1d(speed, sigma=p["accel_tau"])
    speed = np.clip(speed, 0, max_s)
    speed[:10]  *= np.linspace(0, 1, 10)
    speed[-20:] *= np.linspace(1, 0, 20)
    return speed


# ── CLI entry point ──────────────────────────────────────────────────────────

def generate_telemetry(
    fleet_df: pd.DataFrame | None = None,
    cfg: SyntheticConfig | None = None,
) -> None:
    cfg = cfg or SyntheticConfig()
    if fleet_df is None:
        fleet_df = pd.read_csv("data/synthetic/fleet_master.csv")
    gen = TelemetryGenerator(cfg)
    gen.generate_all(fleet_df, Path("data/synthetic"))


if __name__ == "__main__":
    generate_telemetry()
