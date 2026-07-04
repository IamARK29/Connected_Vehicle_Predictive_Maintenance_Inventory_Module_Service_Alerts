"""
Synthetic Telemetry Generator — physically consistent 1-second vehicle telemetry.

Uses ONLY real MG TBox Big Data Specification column names.
ICE vehicles: channels 1-18 + 23 (no BMS).
EV/PHEV vehicles: all channels including 19-22 (BMS signals).

Physical rules enforced:
 1. Driving sessions (1-4/day) with SysPwrMod state machine
 2. Speed/RPM/gear correlation via gear-ratio model
 3. Gear selection by speed thresholds
 4. Brake/accel mutual exclusion (both cannot exceed 20 simultaneously)
 5. Fuel/energy consumption physics (ICE RPM-based; EV SOC depletes + regen)
 6. Seasonal temperature + coolant exponential warmup curve
 7. 12V battery voltage tracks alternator charging / parasitic drain
 8. Tyre pressure slow leak (-0.5 kPa/day) + temperature effect
 9. HV battery SOC per charge-cycle (SOH derived via Coulomb counting in pipeline)
10. Failure injection for 15% VINs with pre-failure intensity curves
11. Real binary warning signals: vehOilPressureWarning, vehMILWarning, vehBrkFludLvlLow, vehABSF
12. TPMS: wheelTyreMonitorStatus = 1 when any tyre < 180 kPa
13. EV thermal runaway signals: vehBMSCMUFlt, vehBMSCellVoltFlt, vehBMSPackTemFlt
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from synthetic.config import SyntheticConfig, DRIVER_ARCHETYPES

# ── Physical constants ─────────────────────────────────────────────────────

_GEAR_RPM_FACTOR = [0.0, 170.0, 65.0, 37.0, 26.0, 19.0, 15.0]  # idx 0=idle, 1-6
_GEAR_UP_SPEED   = [0, 10, 30, 60, 90, 110, 999]
_GEAR_DOWN_SPEED = [0,  5, 20, 50, 80, 100, 999]

IDLE_RPM = 820
MAX_RPM  = 6800

_AC_PENALTY   = 0.25   # fuel rate multiplier when AC is on
_REGEN_FRAC   = 0.18   # fraction of braking energy recovered by EV regen

_COOLANT_FINAL  = 91.0
_COOLANT_TAU_S  = 240
_OVERHEAT_FINAL = 107.0

_BATT_CHARGE_V   = 14.1
_BATT_DRAIN_RATE = 0.003  # V/minute when engine off

# Tyre pressures in kPa (decoded, not encoded N values)
_TYRE_NORMAL  = np.array([232.0, 232.0, 226.0, 226.0])  # FL FR RL RR
_TYRE_LEAK    = 0.5 / (24 * 60)  # kPa/second natural leak
_TYRE_TEMP_K  = 0.10             # kPa/°C above 25°C

# EV battery constants
_EV_EFFICIENCY_KWHKM = 0.18   # typical MG ZS EV: 18 kWh/100km


# ── Driver profile parameters ──────────────────────────────────────────────

def _profile_from_archetype(name: str) -> dict:
    a = DRIVER_ARCHETYPES.get(name)
    if a is None:
        a = DRIVER_ARCHETYPES.get("urban_commuter", {})
    ms = a.get("max_speed_kph", 80)
    accel_tau = max(1.0, 10.0 - a.get("accel_rate_kph_per_s", 4.0))
    return {
        "max_speed":      ms,
        "target_bands":   [(ms * 0.5, ms), (ms * 0.25, ms * 0.7), (0, ms * 0.4)],
        "accel_tau":      accel_tau,
        "idle_fraction":  a.get("idle_fraction", 0.15),
        "cruise_prob":    a.get("cruise_prob", 0.05),
        "elevation_sigma": a.get("elevation_change_m_per_km", 0),
    }


_PROFILE: dict[str, dict] = {}
for _name in DRIVER_ARCHETYPES:
    _PROFILE[_name] = _profile_from_archetype(_name)
_PROFILE["normal"] = _PROFILE.get("urban_commuter", _profile_from_archetype("urban_commuter"))
_PROFILE["eco"]    = _PROFILE.get("eco_driver",     _profile_from_archetype("eco_driver"))


class TelemetryGenerator:
    """Generate physically realistic 1 Hz telemetry for every VIN in a fleet."""

    def __init__(self, cfg: SyntheticConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    # ── Public entry point ─────────────────────────────────────────────────

    def generate_all(self, fleet_df: pd.DataFrame, out_dir: Path, resume: bool = False) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        failure_plan = self._plan_failures(fleet_df)
        manifest_rows: list[dict] = []
        combined_chunks: list[pd.DataFrame] = []

        for _, vrow in fleet_df.iterrows():
            vin      = str(vrow["vin"])
            specs    = failure_plan.get(vin, [])
            csv_path = out_dir / f"telemetry_{vin}.csv"

            if resume and csv_path.exists():
                print(f"  {vin} ... SKIPPED (exists)")
                sample = pd.read_csv(csv_path, low_memory=False).sample(min(10_000, int(1e9)), random_state=42)
                combined_chunks.append(sample)
                for sp in specs:
                    manifest_rows.append({
                        "vin":                vin,
                        "failure_type":       sp["ftype"],
                        "injection_date":     sp["injection_date"].date().isoformat(),
                        "failure_date":       sp["failure_date"].date().isoformat(),
                        "affected_component": sp.get("affected", ""),
                    })
                continue

            print(f"  {vin} ({vrow['fuel_type']}, {vrow['driver_profile']}) ...", end=" ", flush=True)

            df = self._generate_for_vin(vrow, specs)
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
        is_ev    = fuel_type in ("EV", "PHEV")

        # Per-VIN persistent state
        soc_pct           = float(self.rng.uniform(70, 90)) if is_ev else None
        charge_cycles     = 0
        fuel_l            = fuel_tank_l * float(self.rng.uniform(0.5, 0.9))
        batt_12v          = 12.6
        tyre_kpa          = _TYRE_NORMAL.copy()
        km_since_charge   = 0.0   # EV only
        used_kwh_since_charge = 0.0  # EV only

        all_frames: list[pd.DataFrame] = []

        for day_idx in range(self.cfg.num_days):
            date     = start_dt + timedelta(days=day_idx)
            doy      = date.timetuple().tm_yday
            out_temp = 28.0 + 6.0 * math.sin(2 * math.pi * (doy - 60) / 365)
            fail_ctx = self._failure_intensity(
                specs, datetime.combine(date, datetime.min.time())
            )

            n_sessions    = int(self.rng.integers(max(1, self.cfg.sessions_per_day - 1), self.cfg.sessions_per_day + 1))
            session_hours = sorted(float(h) for h in self.rng.uniform(6, 22, size=n_sessions))

            for s_idx, hour in enumerate(session_hours):
                session_start = date + timedelta(
                    hours=hour, minutes=int(self.rng.integers(0, 60))
                )
                duration_s = int(self.rng.integers(15 * 60, 120 * 60))

                (frame, odometer, soc_pct, fuel_l, last_batt_12v,
                 charge_cycles, km_since_charge,
                 used_kwh_since_charge) = self._build_session(
                    vin=vin,
                    start_dt=session_start,
                    n_sec=duration_s,
                    step=self.cfg.sample_interval_seconds,
                    odometer=odometer,
                    soc_pct=soc_pct,
                    charge_cycles=charge_cycles,
                    fuel_l=fuel_l,
                    fuel_tank_l=fuel_tank_l,
                    base_l100km=base_l100km,
                    bat_kwh=bat_kwh,
                    batt_12v=batt_12v,
                    tyre_kpa=tyre_kpa,
                    km_since_charge=km_since_charge,
                    used_kwh_since_charge=used_kwh_since_charge,
                    out_temp=out_temp,
                    home_lat=home_lat,
                    home_long=home_long,
                    driver_profile=driver_profile,
                    fuel_type=fuel_type,
                    fail_ctx=fail_ctx,
                )
                all_frames.append(frame)
                batt_12v = last_batt_12v

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

        return pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()

    # ── Session builder ──────────────────────────────────────────────────────

    def _build_session(
        self, vin, start_dt, n_sec, odometer, soc_pct, charge_cycles,
        fuel_l, fuel_tank_l, base_l100km, bat_kwh, batt_12v, tyre_kpa,
        km_since_charge, used_kwh_since_charge,
        out_temp, home_lat, home_long, driver_profile, fuel_type, fail_ctx,
        step: int = 1,
    ):
        rng   = self.rng
        is_ev = fuel_type in ("EV", "PHEV")

        # Simulate at 'step'-second resolution: each row represents `dt` real seconds
        dt    = float(max(1, step))
        n_sec = max(10, n_sec // max(1, step))  # rows to generate

        # ── Speed profile ─────────────────────────────────────────────────
        speed = _gen_speed_profile(n_sec, driver_profile, rng)

        # ── Gear selection (vectorized speed-threshold lookup) ────────────
        if is_ev:
            gear = np.where(speed > 0.5, 3, 0).astype(int)
        else:
            gear = np.where(speed < 0.5,  0,
                   np.where(speed < 10,   1,
                   np.where(speed < 30,   2,
                   np.where(speed < 60,   3,
                   np.where(speed < 90,   4,
                   np.where(speed < 110,  5, 6)))))).astype(int)

        # ── RPM (ICE only; 0 for EV parked, motor rpm mapped for EV) ────
        rpm_factors = np.array([_GEAR_RPM_FACTOR[g] if g < 7 else 0.0 for g in gear])
        if is_ev:
            # EV motor speed scales with vehicle speed (~approx motor RPM)
            rpm = np.where(speed > 0.5, speed * 35 + rng.normal(0, 50, n_sec), 0.0)
            rpm = np.clip(rpm, 0, 12000)
        else:
            rpm = np.where(gear > 0, speed * rpm_factors, float(IDLE_RPM))
            rpm = np.clip(rpm + rng.normal(0, 60, n_sec), 600, MAX_RPM)

        # ── SysPwrMod: 0=Off, 1=ACC, 2=Run, 3=Crank ─────────────────────
        pwr_mod       = np.full(n_sec, 2, dtype=int)
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

        steering = np.clip(np.cumsum(rng.normal(0, 0.5, n_sec)) * 5.0, -180, 180)

        # ── Longitudinal, lateral, and vertical acceleration ──────────────
        # tboxAccelX: longitudinal (forward/back) in g × 1000
        accel_x = np.clip(dspeed / 3.6 * 10, -200, 200)

        # tboxAccelY: lateral G from steering rate (cornering)
        steer_rate = np.gradient(steering)
        lateral_g  = speed * np.abs(steer_rate) * 0.0005
        accel_y    = np.clip(lateral_g + rng.normal(0, 0.01, n_sec), -20.0, 20.0)

        # tboxAccelZ: vertical G from terrain elevation changes
        p    = _PROFILE[driver_profile] if driver_profile in _PROFILE else _PROFILE["normal"]
        alt_sigma = p.get("elevation_sigma", 0) * 0.01 + 0.3
        gnss_alt  = 10.0 + np.cumsum(rng.normal(0, alt_sigma, n_sec))
        alt_grad  = np.gradient(gnss_alt)
        accel_z   = np.clip(alt_grad * 0.1 + rng.normal(0, 0.05, n_sec), -20.0, 20.0)

        # ── Coolant temperature (exponential warm-up) ────────────────────
        t_final = _COOLANT_FINAL
        if "overheating" in fail_ctx:
            t_final += (_OVERHEAT_FINAL - _COOLANT_FINAL) * fail_ctx["overheating"]
        ts           = np.arange(n_sec, dtype=float) * dt
        coolant_temp = t_final - (t_final - out_temp) * np.exp(-ts / _COOLANT_TAU_S)
        coolant_temp = np.clip(coolant_temp + rng.normal(0, 0.5, n_sec), out_temp, 115.0)

        # ── HVAC signals ──────────────────────────────────────────────────
        driving_mask = pwr_mod > 0
        ac_on_arr    = np.where(driving_mask & (out_temp > 26), 1, 0)
        fan_speed_raw = max(0, int((out_temp - 26) / 2))
        ac_fan_arr   = np.where(ac_on_arr > 0, np.clip(fan_speed_raw, 1, 8), 0)
        inside_temp  = out_temp + 5 - 6 * float(out_temp > 26) + rng.normal(0, 0.5, n_sec)
        inside_temp  = np.clip(inside_temp, 15, 45)

        # ── Lighting signals ──────────────────────────────────────────────
        hour_arr      = start_dt.hour + np.arange(n_sec) / 3600
        night_arr     = ((hour_arr % 24 > 19) | (hour_arr % 24 < 7)).astype(int)
        rain_arr      = (rng.random(n_sec) < 0.08).astype(int) * rng.integers(1, 4, n_sec)
        dip_light     = np.where(driving_mask & ((night_arr > 0) | (rain_arr > 1)), 1, 0)
        main_light    = np.where(driving_mask & (night_arr > 0) & (rng.random(n_sec) < 0.15), 1, 0)
        side_light    = np.where(driving_mask, 1, 0)
        night_det     = night_arr

        # ── Fuel consumption (ICE / PHEV) ──────────────────────────────────
        rpm_factor    = np.clip((rpm - 800) / 3000, 0, 1)
        ac_scalar     = float(out_temp > 26)
        fuel_rate_l_s = (
            (base_l100km / 100) * (speed / 3600) *
            (1 + 0.35 * rpm_factor) * (1 + _AC_PENALTY * ac_scalar)
        )
        fuel_rate_l_s = np.where(speed < 1, base_l100km / 100 * 0.4 / 3600, fuel_rate_l_s)
        if not is_ev and "oil_degradation" in fail_ctx:
            fuel_rate_l_s *= 1.0 + 0.15 * fail_ctx["oil_degradation"]

        # ── 12V battery voltage ────────────────────────────────────────────
        batt_v_arr = np.full(n_sec, _BATT_CHARGE_V) + rng.normal(0, 0.05, n_sec)
        if "12v_battery_failure" in fail_ctx:
            batt_v_arr -= fail_ctx["12v_battery_failure"] * 1.5

        # ── Tyre pressures ─────────────────────────────────────────────────
        tp      = tyre_kpa.copy()
        t_delta = _TYRE_TEMP_K * (out_temp + 15 + speed * 0.1 - 25)
        tyre_fl = tp[0] + t_delta - rng.uniform(0, 0.3, n_sec)
        tyre_fr = tp[1] + t_delta - rng.uniform(0, 0.3, n_sec)
        tyre_rl = tp[2] + t_delta - rng.uniform(0, 0.3, n_sec)
        tyre_rr = tp[3] + t_delta - rng.uniform(0, 0.3, n_sec)

        if "tyre_puncture" in fail_ctx and fail_ctx["tyre_puncture"] > 0.8:
            arr_map = {"FL": tyre_fl, "FR": tyre_fr, "RL": tyre_rl, "RR": tyre_rr}
            pt_arr  = arr_map.get(str(fail_ctx.get("_affected", "FL")), tyre_fl)
            drop    = 120 * min(1.0, (fail_ctx["tyre_puncture"] - 0.8) / 0.2)
            mid     = n_sec // 2
            pt_arr[mid:] = np.maximum(80, pt_arr[mid:] - drop)

        # TPMS: 0=OK, 1=Deflation Warning, 5=Sensor not detected
        tpms_status = np.where(
            (tyre_fl < 180) | (tyre_fr < 180) | (tyre_rl < 180) | (tyre_rr < 180),
            1, 0
        ).astype(int)

        # ── Real binary warning signals ────────────────────────────────────
        # Brake fluid low: real binary signal, 1 during advanced brake degradation
        brake_degrad = fail_ctx.get("brake_degradation", 0.0)
        brkflud_warn = np.full(n_sec, int(brake_degrad > 0.75), dtype=int)

        # ABS fault: occasional spurious signal during hard braking
        abs_fault = np.where(
            (brake_pos > 175) & (np.abs(accel_x) > 100) & (rng.random(n_sec) < 0.003),
            1, 0
        ).astype(int)

        # ── ICE-only binary warnings ─────────────────────────────────────
        oil_degrad = fail_ctx.get("oil_degradation", 0.0)
        overheat   = fail_ctx.get("overheating", 0.0)
        oil_press_warn = np.full(n_sec, int(oil_degrad > 0.70), dtype=int)
        mil_warn       = np.full(n_sec, int(oil_degrad > 0.85 or overheat > 0.70), dtype=int)

        # ── Odometer ───────────────────────────────────────────────────────
        odo_arr  = odometer + np.cumsum(speed / 3600 * dt)
        odometer = float(odo_arr[-1])

        # ── Fuel level ─────────────────────────────────────────────────────
        if not is_ev:
            fuel_l = max(2.0, fuel_l - float(np.sum(fuel_rate_l_s * dt)))
        fuel_pct = float(np.clip(fuel_l / fuel_tank_l * 100, 0, 100))

        # ── GPS random walk ────────────────────────────────────────────────
        lat_walk  = home_lat  + np.cumsum(rng.normal(0, 0.00008 * dt, n_sec))
        long_walk = home_long + np.cumsum(rng.normal(0, 0.00010 * dt, n_sec))
        gnss_head = (np.arctan2(np.gradient(long_walk), np.gradient(lat_walk)) * 180 / math.pi) % 360
        gnss_sats = np.clip(rng.integers(8, 15, n_sec), 4, 14)

        # ── Timestamps ─────────────────────────────────────────────────────
        unix_ts = [int((start_dt + timedelta(seconds=int(i * dt))).timestamp()) for i in range(n_sec)]
        dates   = [(start_dt + timedelta(seconds=int(i * dt))).strftime("%Y-%m-%d") for i in range(n_sec)]

        fail_keys      = [k for k in fail_ctx if not k.startswith("_")]
        fail_label     = fail_keys[0] if fail_keys else ""
        fail_intensity = round(float(fail_ctx.get(fail_label, 0.0)), 3) if fail_label else 0.0

        # ── EV / PHEV HV battery ────────────────────────────────────────────
        hv_pack_v     = np.full(n_sec, np.nan)
        hv_pack_a     = np.full(n_sec, np.nan)
        hv_soc_arr    = np.full(n_sec, np.nan)
        hv_cell_max_v = np.full(n_sec, np.nan)
        hv_cell_min_v = np.full(n_sec, np.nan)
        hv_cell_max_t = np.full(n_sec, np.nan)
        hv_cell_min_t = np.full(n_sec, np.nan)
        hvdcdc_temp   = np.full(n_sec, np.nan)
        bms_cmu_fault = np.full(n_sec, np.nan)
        bms_cv_fault  = np.full(n_sec, np.nan)
        bms_pt_fault  = np.full(n_sec, np.nan)
        bms_status    = np.full(n_sec, np.nan)
        motor_inv_t   = np.full(n_sec, np.nan)
        motor_str_t   = np.full(n_sec, np.nan)
        gun_connected = np.full(n_sec, np.nan)
        is_charging   = np.full(n_sec, np.nan)
        dc_or_ac      = np.full(n_sec, np.nan, dtype=object)
        used_kwh_arr  = np.full(n_sec, np.nan)
        mileage_arr   = np.full(n_sec, np.nan)
        ev_range_arr  = np.full(n_sec, np.nan)
        bms_hvil      = np.full(n_sec, np.nan)
        ept_ready     = np.full(n_sec, np.nan)

        charged_this_session = False

        if is_ev and bat_kwh and soc_pct is not None:
            v_ms      = speed / 3.6
            power_kw  = 0.10 * speed + 0.03 * np.abs(accel_x) * v_ms / 100 + ac_scalar * 1.5
            regen_kw  = np.where(dspeed < -0.5, np.abs(dspeed) * v_ms * 0.8 * _REGEN_FRAC, 0.0)
            net_e_kws = power_kw - regen_kw  # kWh/s net energy use

            # Vectorised SOC: cumulative energy draw scaled by dt (seconds per step)
            soc_draw = np.cumsum(net_e_kws * dt / bat_kwh * 100)
            soc_run  = np.clip(soc_pct - soc_draw, 0, 100)
            soc_pct  = float(soc_run[-1])

            if soc_pct < 20:
                charge_cycles        += 1
                soc_pct               = min(90.0, soc_pct + 50.0)
                km_since_charge       = 0.0
                used_kwh_since_charge = 0.0
                charged_this_session  = True

            hv_soc_arr = soc_run

            # Cell voltage model
            cell_v  = 3.3 + soc_run / 100 * 0.9
            hv_flt  = fail_ctx.get("hv_battery_degradation", 0.0)
            spread  = 0.02 + hv_flt * 0.60
            hv_cell_max_v = np.clip(cell_v + spread / 2 + rng.normal(0, 0.003, n_sec), 3.0, 4.25)
            hv_cell_min_v = np.clip(cell_v - spread / 2 + rng.normal(0, 0.003, n_sec), 2.8, 4.20)

            # Pack voltage and current — SoH-aware
            nom_v   = 400 if bat_kwh > 30 else 200
            hv_flt_for_soh = fail_ctx.get("hv_battery_degradation", 0.0)
            soh_factor = 1.0 - hv_flt_for_soh * 0.25  # up to 25% capacity loss at max fault
            hv_pack_v = nom_v * soh_factor * (soc_run / 100 * 0.15 + 0.85) + rng.normal(0, 1, n_sec)
            # During charging (after SOC bump): inject negative current as proper charging session
            if charged_this_session:
                charge_start = max(0, n_sec - min(n_sec, 1800))  # last 30 min of session
                charge_a     = -(bat_kwh * 1000 / hv_pack_v[charge_start:].mean()) * 0.5  # 0.5C charge
                hv_pack_a    = np.where(
                    np.arange(n_sec) >= charge_start,
                    np.clip(charge_a + rng.normal(0, 2, n_sec), -300, 0),
                    np.clip(power_kw * 1000 / (hv_pack_v + 1e-6), -50, 500),
                )
            else:
                hv_pack_a = np.clip(power_kw * 1000 / (hv_pack_v + 1e-6), -50, 500)

            # Cell temperature: scales with power / capacity
            cell_t_mean   = np.clip(25 + power_kw / bat_kwh * 8 + rng.normal(0, 0.5, n_sec), -10, 55)
            hv_cell_max_t = np.clip(cell_t_mean + 2 + rng.normal(0, 0.3, n_sec), -10, 60)
            hv_cell_min_t = np.clip(cell_t_mean - 2 + rng.normal(0, 0.3, n_sec), -10, 55)

            # DCDC converter temperature
            hvdcdc_temp = np.clip(
                40 + power_kw / bat_kwh * 6 + rng.normal(0, 1, n_sec),
                -10, 80
            )

            # BMS fault levels: 0=None, 1=L1, 2=L2, 3=L3
            bms_cmu_fault = np.full(n_sec, (
                3 if hv_flt > 0.90 else 2 if hv_flt > 0.70 else 1 if hv_flt > 0.50 else 0
            ), dtype=int)
            bms_cv_fault = np.full(n_sec, (
                2 if hv_flt > 0.85 else 1 if hv_flt > 0.60 else 0
            ), dtype=int)
            bms_pt_fault = np.where(
                hv_cell_max_t > 50, 2, np.where(hv_cell_max_t > 45, 1, 0)
            ).astype(int)

            # BMS state: 3=Drive, 0=Off
            bms_status = np.where(pwr_mod == 0, 0, 3).astype(int)

            # Motor temperatures (scale with power demand)
            motor_inv_t = np.clip(35 + power_kw / bat_kwh * 15 + rng.normal(0, 1, n_sec), -10, 120)
            motor_str_t = np.clip(40 + power_kw / bat_kwh * 12 + rng.normal(0, 1, n_sec), -10, 150)

            # Charging signals (in this session: not charging, this is a driving session)
            gun_connected = np.zeros(n_sec, dtype=int)
            is_charging   = np.zeros(n_sec, dtype=int)
            dc_or_ac      = np.full(n_sec, "", dtype=object)

            # Cumulative energy / distance since last charge (scaled by dt)
            cum_kwh  = used_kwh_since_charge + np.cumsum(np.maximum(0, net_e_kws * dt))
            cum_km   = km_since_charge        + np.cumsum(speed / 3600 * dt)
            used_kwh_arr = np.maximum(0, cum_kwh)
            mileage_arr  = np.maximum(0, cum_km)

            # Update persistent state for next session
            if not charged_this_session:
                km_since_charge       = float(cum_km[-1])
                used_kwh_since_charge = float(cum_kwh[-1])

            # EV range: SOC × total range / 100
            base_range     = bat_kwh / _EV_EFFICIENCY_KWHKM
            ev_range_arr   = np.maximum(0, soc_run / 100 * base_range)

            bms_hvil  = np.where(pwr_mod > 0, 1, 0).astype(int)
            ept_ready = np.where(pwr_mod > 0, 1, 0).astype(int)

        # ── Assemble common columns ────────────────────────────────────────
        common = {
            "VIN":                        vin,
            "StartTime-TimeStamp":        unix_ts,
            "StartTime-Date":             dates,
            "vehSpeed":                   np.round(speed, 1),
            "vehSysPwrMod":               pwr_mod,
            "vehGearPos":                 gear,
            "vehSteeringAngle":           np.round(steering, 1),
            "vehBrakePos":                np.round(brake_pos, 1),
            "vehAccelPos":                np.round(accel_pos, 1),
            "tboxAccelX":                 np.round(accel_x, 3),
            "tboxAccelY":                 np.round(accel_y, 3),
            "tboxAccelZ":                 np.round(accel_z, 3),
            "vehBatt":                    np.round(batt_v_arr, 2),
            "vehOdo":                     np.round(odo_arr, 2),
            "vehCoolantTemp":             np.round(coolant_temp, 1),
            "vehOutsideTemp":             np.round(out_temp + rng.normal(0, 0.3, n_sec), 1),
            "vehInsideTemp":              np.round(inside_temp, 1),
            "vehAC":                      ac_on_arr,
            "vehACFanSpeed":              ac_fan_arr,
            "vehSideLight":               side_light,
            "vehDipLight":                dip_light,
            "vehMainLight":               main_light,
            "vehRainDetected":            rain_arr,
            "vehNightDetected":           night_det,
            "vehHorn":                    np.zeros(n_sec, dtype=int),
            "vehSeatBeltDrv":             np.ones(n_sec, dtype=int),
            "vehBrkFludLvlLow":           brkflud_warn,
            "vehABSF":                    abs_fault,
            "frontLeftTyrePressure":      np.round(tyre_fl, 1),
            "frontRrightTyrePressure":    np.round(tyre_fr, 1),
            "rearLeftTyrePressure":       np.round(tyre_rl, 1),
            "rearRightTyrePressure":      np.round(tyre_rr, 1),
            "wheelTyreMonitorStatus":     tpms_status,
            "gnssLat":                    np.round(lat_walk, 6),
            "gnssLong":                   np.round(long_walk, 6),
            "gnssAlt":                    np.round(gnss_alt, 1),
            "gnssHead":                   np.round(gnss_head, 1),
            "gnssSats":                   gnss_sats,
            "_failure_type":              fail_label,
            "_failure_intensity":         fail_intensity,
        }

        is_phev = fuel_type == "PHEV"

        if not is_ev or is_phev:
            # ICE or PHEV: include engine channels (vehRPM, FuelTankLevel, oil warnings)
            common.update({
                "vehRPM":                 np.round(rpm).astype(int),
                "FuelTankLevel":          np.round(np.full(n_sec, fuel_pct), 1),
                "vehFuelConsumed":        np.round(np.cumsum(fuel_rate_l_s), 4),
                "vehOilPressureWarning":  oil_press_warn,
                "vehMILWarning":          mil_warn,
            })

        if is_ev:
            # EV or PHEV: include BMS channels (19-22) — pure EV has no engine signals
            common.update({
                "vehBMSBscSta":                 bms_status,
                "vehBMSPackVol":                np.round(hv_pack_v, 1),
                "vehBMSPackCrnt":               np.round(hv_pack_a, 2),
                "vehBMSPackSOC":                np.round(hv_soc_arr, 1),
                "vehBMSPackSOCV":               np.zeros(n_sec, dtype=int),
                "vehBMSCellMaxVol":             np.round(hv_cell_max_v, 4),
                "vehBMSCellMinVol":             np.round(hv_cell_min_v, 4),
                "vehBMSCellMaxTem":             np.round(hv_cell_max_t, 1),
                "vehBMSCellMinTem":             np.round(hv_cell_min_t, 1),
                "vehHVDCDCTem":                 np.round(hvdcdc_temp, 1),
                "vehBMSCMUFlt":                 bms_cmu_fault,
                "vehBMSCellVoltFlt":            bms_cv_fault,
                "vehBMSPackTemFlt":             bms_pt_fault,
                "vehBMSHVILClsd":               bms_hvil,
                "chargingGunIsConnected":       gun_connected,
                "vehIsCharging":                is_charging,
                "dcOrAC":                       dc_or_ac,
                "usedBatterySinceLastCharge":   np.round(used_kwh_arr, 3),
                "mileageSinceLastCharge":       np.round(mileage_arr, 2),
                "vehElecRange":                 np.round(ev_range_arr, 0),
                "vehTMInvtrTem":                np.round(motor_inv_t, 1),
                "vehTMSttrTem":                 np.round(motor_str_t, 1),
                "vehEPTRdy":                    ept_ready,
            })

        frame = pd.DataFrame(common)
        last_batt_12v = float(batt_v_arr[-1])
        return frame, odometer, soc_pct, fuel_l, last_batt_12v, charge_cycles, km_since_charge, used_kwh_since_charge


# ── Speed profile generator ──────────────────────────────────────────────────

def _gen_speed_profile(n: int, profile: str, rng: np.random.Generator) -> np.ndarray:
    """Smooth physically realistic speed trace using Gaussian filter."""
    from scipy.ndimage import gaussian_filter1d

    p     = _PROFILE.get(profile, _PROFILE["normal"])
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
