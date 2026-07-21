"""
EV Charging Feature Engine

Computes charge-session-derived features for HV battery health assessment.
Designed for MG Motor India EVs/PHEVs using TBox Big Data Spec signals.

Signal mapping (pipeline stores decoded physical values):
    vehBMSPackCrnt / vehPackCrnt  → Amperes  (already decoded)
    vehBMSPackVol  / vehPackVol   → Volts    (already decoded)
    vehBMSPackSOC                 → %        (already decoded)
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MIN_SESSIONS_REQUIRED = 5          # below this, rolling SoH is unreliable
_SOH_OUTLIER_BAND_PP  = 8.0        # ±pp from session median before rejection
_GRID_VOLT_STD_MAX    = 2.0        # V std threshold for India grid instability
_GRID_DRIFT_MAX       = 3.0        # V first-half vs second-half drift limit
_SOC_WINDOW_LOW       = 20.0       # % — start of usable Coulomb-counting window
_SOC_WINDOW_HIGH      = 95.0       # % — end of usable window

# Alert thresholds used by downstream alert generators.
# NOTE: charge_acceptance_ratio alert should only fire when charger_power_limited == False.
# When charger_power_limited == True, fire a separate infra alert instead of a battery alert.
ALERT_THRESHOLDS: dict[str, tuple[str, float, str]] = {
    # feature_key: (severity, threshold, comparison)
    "charge_acceptance_ratio":       ("HIGH",   0.80, "lt"),
    "charge_duration_deviation_pct": ("MEDIUM", 25.0, "gt"),
    "avg_soc_after_overnight_ac":    ("MEDIUM", 92.0, "lt"),
    "end_voltage_deficit_v":         ("HIGH",    8.0, "gt"),
    "actual_charger_power_kw_mean":  ("MEDIUM", 37.5, "lt"),   # < 75% of 50 kW rated
}


# ── Module-level SoH functions ────────────────────────────────────────────────

def compute_soh_from_charge_session(
    session_df: pd.DataFrame,
    nominal_kwh: float,
) -> float | None:
    """
    Estimate SoH from a single charge session using Coulomb counting.

    Applies an India-specific grid stability filter to reject sessions with
    noisy or sagging grid voltage that would corrupt the energy integral.

    Returns SoH as a percentage (0–100), or None if session is rejected.
    """
    required = {"vehPackCrnt", "vehPackVol", "vehBMSPackSOC", "timestamp"}
    if not required.issubset(session_df.columns):
        return None
    if len(session_df) < 10:
        return None

    df = session_df.sort_values("timestamp").copy()
    df["current_a"]  = df["vehPackCrnt"]    # already decoded amps
    df["voltage_v"]  = df["vehPackVol"]     # already decoded volts
    df["soc_pct"]    = df["vehBMSPackSOC"]  # already decoded percent

    # ── Grid stability filter ───────────────────────────────────────────────
    volt_std = df["voltage_v"].std()
    if volt_std > _GRID_VOLT_STD_MAX:
        log.debug("Session rejected: voltage std=%.2f V > %.1f V", volt_std, _GRID_VOLT_STD_MAX)
        return None

    mid = len(df) // 2
    first_half_mean  = df["voltage_v"].iloc[:mid].mean()
    second_half_mean = df["voltage_v"].iloc[mid:].mean()
    if first_half_mean > second_half_mean + _GRID_DRIFT_MAX:
        log.debug(
            "Session rejected: voltage drift first=%.2fV second=%.2fV",
            first_half_mean, second_half_mean,
        )
        return None

    # ── Narrow to 20–95 % SoC window ───────────────────────────────────────
    window = df[(df["soc_pct"] >= _SOC_WINDOW_LOW) & (df["soc_pct"] <= _SOC_WINDOW_HIGH)]
    if len(window) < 5:
        return None

    dt_s = window["timestamp"].diff().dt.total_seconds().fillna(0).clip(lower=0, upper=300)
    power_w = window["voltage_v"] * window["current_a"].abs()
    energy_kwh = (power_w * dt_s).sum() / 3_600_000.0

    soc_delta_pct = window["soc_pct"].iloc[-1] - window["soc_pct"].iloc[0]
    if soc_delta_pct <= 0:
        return None

    # Extrapolate full-cycle energy and normalise to rated window capacity
    full_cycle_kwh = energy_kwh * ((_SOC_WINDOW_HIGH - _SOC_WINDOW_LOW) / soc_delta_pct)
    # 0.75 derate: rated capacity at 100% SoC translates to ~75% usable
    soh = (full_cycle_kwh / (nominal_kwh * 0.75)) * 100.0
    return round(float(np.clip(soh, 0.0, 110.0)), 2)


def compute_soh_rolling(
    sessions: Sequence[dict],
    nominal_kwh: float,
) -> float | None:
    """
    Rolling SoH estimate from a list of per-session SoH dicts.

    Each dict must have key ``soh_pct`` (already computed per-session).
    Applies ±8 pp outlier filter around the session median before averaging.

    Returns None if fewer than MIN_SESSIONS_REQUIRED sessions survive filtering.
    """
    raw = [s["soh_pct"] for s in sessions if s.get("soh_pct") is not None]
    if len(raw) < MIN_SESSIONS_REQUIRED:
        return None

    arr = np.array(raw, dtype=float)
    median = float(np.median(arr))
    filtered = arr[np.abs(arr - median) <= _SOH_OUTLIER_BAND_PP]

    if len(filtered) < MIN_SESSIONS_REQUIRED:
        return None

    return round(float(filtered.mean()), 2)


# ── Charge acceptance from actual delivered power ─────────────────────────────

def compute_charge_acceptance_features(
    charge_sessions_df: pd.DataFrame,
    telemetry_df: pd.DataFrame,
    fleet_row: dict,
) -> dict:
    """Compute DC charge acceptance against actual delivered power, not rated power.

    DC fast chargers in India regularly underdeliver their rated output on hot
    afternoons when grid supply is constrained.  Comparing battery acceptance
    against rated capacity attributes charger underperformance to the battery.
    This function compares against the power the charger actually delivered.

    Returns:
        charge_acceptance_ratio        — accepted / offered power (None if no DC sessions)
        charger_power_limited          — True if charger delivered < 75% of fleet-rated DC kW
        actual_charger_power_kw_mean   — mean peak power delivered across DC sessions
        charge_duration_deviation_pct  — actual vs expected duration given actual power
    """
    nominal_kwh  = float(fleet_row.get("battery_capacity_kwh") or 50.0)
    rated_dc_kw  = float(fleet_row.get("rated_dc_charge_kw")  or 50.0)

    _null = {
        "charge_acceptance_ratio":       None,
        "charger_power_limited":         None,
        "actual_charger_power_kw_mean":  None,
        "charge_duration_deviation_pct": None,
    }

    if charge_sessions_df.empty or telemetry_df.empty:
        return _null

    # Identify DC sessions (charge_type == "DC" in normalised schema)
    ct_col = next((c for c in ("charge_type", "dc_or_ac") if c in charge_sessions_df.columns), None)
    if ct_col is None:
        return _null
    if ct_col == "charge_type":
        dc_mask = charge_sessions_df[ct_col].str.upper() == "DC"
    else:
        dc_mask = charge_sessions_df[ct_col] == 1
    dc_sessions = charge_sessions_df[dc_mask].copy()
    if dc_sessions.empty:
        return _null

    # Resolve HV current/voltage columns (raw TBox names or normalised)
    crnt_col = next((c for c in ("vehBMSPackCrnt", "bms_pack_crnt", "vehPackCrnt") if c in telemetry_df.columns), None)
    volt_col = next((c for c in ("vehBMSPackVol",  "bms_pack_vol",  "vehPackVol")  if c in telemetry_df.columns), None)
    ts_col   = next((c for c in ("timestamp", "StartTime-TimeStamp") if c in telemetry_df.columns), None)
    vin_col  = "vin" if "vin" in telemetry_df.columns else None

    if crnt_col is None or volt_col is None or ts_col is None:
        return _null

    tel = telemetry_df.copy()
    tel[ts_col] = pd.to_datetime(tel[ts_col], errors="coerce", utc=True)

    start_col = next((c for c in ("start_ts", "session_start") if c in dc_sessions.columns), None)
    end_col   = next((c for c in ("end_ts",   "session_end")   if c in dc_sessions.columns), None)
    if start_col is None or end_col is None:
        return _null

    session_powers: list[float]     = []
    acceptance_ratios: list[float]  = []
    duration_deviations: list[float] = []

    for _, sr in dc_sessions.iterrows():
        t_start = pd.to_datetime(sr.get(start_col), utc=True, errors="coerce")
        t_end   = pd.to_datetime(sr.get(end_col),   utc=True, errors="coerce")
        if pd.isna(t_start) or pd.isna(t_end):
            continue

        filt = (tel[ts_col] >= t_start) & (tel[ts_col] <= t_end)
        if vin_col and "vin" in sr.index:
            filt &= tel[vin_col] == sr["vin"]
        session_tel = tel[filt]
        if len(session_tel) < 30:
            continue

        current_a  = session_tel[crnt_col]  # already decoded amps
        voltage_v  = session_tel[volt_col]   # already decoded volts
        power_kw   = (current_a.abs() * voltage_v) / 1000.0

        actual_peak_kw = float(power_kw.quantile(0.95))
        session_powers.append(actual_peak_kw)

        # First 5 min = charger negotiation ramp; captures max offered power
        ramp_win = session_tel.head(300)
        if len(ramp_win) < 30:
            continue
        charger_offer_kw = float(
            (ramp_win[crnt_col].abs() * ramp_win[volt_col] / 1000).max()
        )

        # Steady state after ramp-up: how much the battery actually accepted
        steady = session_tel.iloc[300:]
        if len(steady) < 30:
            continue
        accepted_kw = float(
            (steady[crnt_col].abs() * steady[volt_col] / 1000).quantile(0.90)
        )
        if charger_offer_kw > 5:
            acceptance_ratios.append(accepted_kw / charger_offer_kw)

        # Duration deviation: actual vs expected given the actual charger power
        soc_start = sr.get("soc_start_pct")
        soc_end   = sr.get("soc_end_pct")
        dur_min   = sr.get("duration_min")
        if actual_peak_kw <= 0 or pd.isna(soc_start) or pd.isna(soc_end) or pd.isna(dur_min):
            continue
        delta_soc = (float(soc_end) - float(soc_start)) / 100.0
        expected_min = delta_soc * nominal_kwh / actual_peak_kw * 60
        if expected_min > 5:
            duration_deviations.append(float(dur_min) / expected_min)

    actual_power_mean     = float(np.mean(session_powers))     if session_powers     else 0.0
    acceptance_ratio_mean = float(np.mean(acceptance_ratios))  if acceptance_ratios  else None
    charger_power_limited = actual_power_mean < rated_dc_kw * 0.75 if session_powers else None

    duration_dev_pct = (
        round((float(np.mean(duration_deviations)) - 1.0) * 100, 1)
        if duration_deviations else 0.0
    )

    return {
        "charge_acceptance_ratio":       round(acceptance_ratio_mean, 3) if acceptance_ratio_mean is not None else None,
        "charger_power_limited":         charger_power_limited,
        "actual_charger_power_kw_mean":  round(actual_power_mean, 1),
        "charge_duration_deviation_pct": duration_dev_pct,
    }


# ── Feature Engine ────────────────────────────────────────────────────────────

class EVChargingFeatureEngine:
    """
    Computes 9 EV charging features for the battery_hv feature group.

    Usage
    -----
    engine = EVChargingFeatureEngine()
    features = engine.compute(charge_sessions_df, telemetry_df, vin, fleet_row)
    """

    # Expected nominal pack capacity if not present in fleet_row
    _DEFAULT_CAPACITY_KWH = 50.0

    def compute(
        self,
        charge_sessions_df: pd.DataFrame,
        telemetry_df: pd.DataFrame,
        vin: str,
        fleet_row: dict,
    ) -> dict[str, float | None]:
        """
        Parameters
        ----------
        charge_sessions_df : rows for this VIN, columns include:
            session_id, start_ts, end_ts, charge_type ('AC'/'DC'),
            soc_start_pct, soc_end_pct, duration_min, energy_kwh,
            end_voltage_v, expected_end_voltage_v,
            + raw TBox columns for per-session SoH calc
        telemetry_df : raw TBox telemetry, columns include:
            timestamp, vehPackCrnt, vehPackVol, vehBMSPackSOC
        fleet_row : dict from fleet CSV row for this VIN
        """
        nominal_kwh = float(fleet_row.get("battery_capacity_kwh") or self._DEFAULT_CAPACITY_KWH)
        now = pd.Timestamp.now(tz="UTC")
        cutoff_30d = now - pd.Timedelta(days=30)

        cs = charge_sessions_df.copy()
        if cs.empty:
            return self._null_features()

        # Normalise timestamps
        for col in ("start_ts", "end_ts"):
            if col in cs.columns:
                cs[col] = pd.to_datetime(cs[col], utc=True, errors="coerce")

        cs30 = cs[cs["start_ts"] >= cutoff_30d] if "start_ts" in cs.columns else cs

        # ── Feature 1 & 2: Charge acceptance ratio + trend ─────────────────
        charge_acceptance_ratio = self._charge_acceptance_ratio(cs30)
        charge_acceptance_trend_30d = self._charge_acceptance_trend(cs, now)

        # ── Feature 3: Duration deviation ──────────────────────────────────
        charge_duration_deviation_pct = self._duration_deviation(cs30)

        # ── Feature 4: End-voltage deficit ─────────────────────────────────
        end_voltage_deficit_v = self._end_voltage_deficit(cs30)

        # ── Feature 5: DC fraction ─────────────────────────────────────────
        dc_fraction_30d = self._dc_fraction(cs30)

        # ── Feature 6: Avg SoC at charge start ─────────────────────────────
        avg_soc_at_charge_start = self._avg_soc_at_start(cs30)

        # ── Feature 7: Avg SoC after overnight AC ──────────────────────────
        avg_soc_after_overnight_ac = self._avg_soc_after_overnight_ac(cs30)

        # ── Features 8 & 9: Session counts ─────────────────────────────────
        total_charge_sessions_30d = len(cs30)
        dc_charge_sessions_30d = int(
            (cs30["charge_type"].str.upper() == "DC").sum()
            if "charge_type" in cs30.columns else 0
        )

        # ── Feature 10–12: actual delivered power + charger limit flag ─────
        dc_power_features = compute_charge_acceptance_features(cs30, telemetry_df, fleet_row)

        # Use power-based acceptance ratio when DC telemetry is available,
        # otherwise keep the SoC-delta based ratio as fallback.
        final_acceptance_ratio = (
            dc_power_features["charge_acceptance_ratio"]
            if dc_power_features["charge_acceptance_ratio"] is not None
            else charge_acceptance_ratio
        )
        final_duration_dev = (
            dc_power_features["charge_duration_deviation_pct"]
            if dc_power_features["charge_duration_deviation_pct"] is not None
            else charge_duration_deviation_pct
        )

        return {
            "charge_acceptance_ratio":       final_acceptance_ratio,
            "charge_acceptance_trend_30d":   charge_acceptance_trend_30d,
            "charge_duration_deviation_pct": final_duration_dev,
            "end_voltage_deficit_v":         end_voltage_deficit_v,
            "dc_fraction_30d":               dc_fraction_30d,
            "avg_soc_at_charge_start":       avg_soc_at_charge_start,
            "avg_soc_after_overnight_ac":    avg_soc_after_overnight_ac,
            "total_charge_sessions_30d":     total_charge_sessions_30d,
            "dc_charge_sessions_30d":        dc_charge_sessions_30d,
            "charger_power_limited":         dc_power_features["charger_power_limited"],
            "actual_charger_power_kw_mean":  dc_power_features["actual_charger_power_kw_mean"],
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _null_features() -> dict[str, None]:
        return {
            "charge_acceptance_ratio":       None,
            "charge_acceptance_trend_30d":   None,
            "charge_duration_deviation_pct": None,
            "end_voltage_deficit_v":         None,
            "dc_fraction_30d":               None,
            "avg_soc_at_charge_start":       None,
            "avg_soc_after_overnight_ac":    None,
            "total_charge_sessions_30d":     0,
            "dc_charge_sessions_30d":        0,
            "charger_power_limited":         None,
            "actual_charger_power_kw_mean":  None,
        }

    @staticmethod
    def _charge_acceptance_ratio(cs: pd.DataFrame) -> float | None:
        """Fraction of sessions where SoC delta ≥ 90 % of expected delta."""
        if cs.empty or "soc_start_pct" not in cs.columns or "soc_end_pct" not in cs.columns:
            return None
        valid = cs.dropna(subset=["soc_start_pct", "soc_end_pct"])
        if valid.empty:
            return None
        actual_delta   = (valid["soc_end_pct"] - valid["soc_start_pct"]).clip(lower=0)
        max_possible   = (100.0 - valid["soc_start_pct"]).clip(lower=1.0)
        ratio_per_sess = (actual_delta / max_possible).clip(upper=1.0)
        return round(float(ratio_per_sess.mean()), 4)

    @staticmethod
    def _charge_acceptance_trend(
        cs: pd.DataFrame, now: pd.Timestamp
    ) -> float | None:
        """Slope of charge_acceptance_ratio over two 15-day halves (pp/day)."""
        if "start_ts" not in cs.columns or cs.empty:
            return None
        cutoff_30d  = now - pd.Timedelta(days=30)
        cutoff_15d  = now - pd.Timedelta(days=15)
        cs30 = cs[cs["start_ts"] >= cutoff_30d].dropna(subset=["soc_start_pct", "soc_end_pct"])
        if cs30.empty:
            return None
        older = cs30[cs30["start_ts"] < cutoff_15d]
        newer = cs30[cs30["start_ts"] >= cutoff_15d]

        def _ratio(df: pd.DataFrame) -> float | None:
            if df.empty:
                return None
            d = (df["soc_end_pct"] - df["soc_start_pct"]).clip(lower=0)
            m = (100.0 - df["soc_start_pct"]).clip(lower=1.0)
            return float((d / m).clip(upper=1.0).mean())

        r_old = _ratio(older)
        r_new = _ratio(newer)
        if r_old is None or r_new is None:
            return None
        return round((r_new - r_old) / 15.0, 6)

    @staticmethod
    def _duration_deviation(cs: pd.DataFrame) -> float | None:
        """Mean % deviation of actual charge duration vs expected."""
        needed = {"duration_min", "soc_start_pct", "soc_end_pct", "charge_type"}
        if not needed.issubset(cs.columns) or cs.empty:
            return None
        valid = cs.dropna(subset=list(needed))
        if valid.empty:
            return None
        # Simple expected: assume AC=~1% SoC/min, DC=~3% SoC/min
        rates = valid["charge_type"].str.upper().map({"AC": 1.0, "DC": 3.0}).fillna(1.0)
        soc_delta = (valid["soc_end_pct"] - valid["soc_start_pct"]).clip(lower=0)
        expected_min = (soc_delta / rates).clip(lower=1.0)
        pct_dev = ((valid["duration_min"] - expected_min) / expected_min * 100.0).abs()
        return round(float(pct_dev.mean()), 2)

    @staticmethod
    def _end_voltage_deficit(cs: pd.DataFrame) -> float | None:
        """Mean gap between expected and actual end-of-charge pack voltage (V)."""
        needed = {"end_voltage_v", "expected_end_voltage_v"}
        if not needed.issubset(cs.columns) or cs.empty:
            return None
        valid = cs.dropna(subset=list(needed))
        if valid.empty:
            return None
        deficit = (valid["expected_end_voltage_v"] - valid["end_voltage_v"]).clip(lower=0)
        return round(float(deficit.mean()), 2)

    @staticmethod
    def _dc_fraction(cs: pd.DataFrame) -> float | None:
        """Fraction of sessions that used DC fast charging."""
        if "charge_type" not in cs.columns or cs.empty:
            return None
        return round(float((cs["charge_type"].str.upper() == "DC").mean()), 4)

    @staticmethod
    def _avg_soc_at_start(cs: pd.DataFrame) -> float | None:
        if "soc_start_pct" not in cs.columns or cs.empty:
            return None
        return round(float(cs["soc_start_pct"].dropna().mean()), 2)

    @staticmethod
    def _avg_soc_after_overnight_ac(cs: pd.DataFrame) -> float | None:
        """
        Mean end SoC for AC sessions that started between 22:00–06:00 and lasted ≥ 4 h.
        Proxy for overnight charge completeness.
        """
        needed = {"start_ts", "duration_min", "soc_end_pct", "charge_type"}
        if not needed.issubset(cs.columns) or cs.empty:
            return None
        ac = cs[cs["charge_type"].str.upper() == "AC"].dropna(subset=list(needed))
        if ac.empty:
            return None
        hour = ac["start_ts"].dt.hour
        overnight = ac[(hour >= 22) | (hour < 6)]
        long_sessions = overnight[overnight["duration_min"] >= 240]
        if long_sessions.empty:
            return None
        return round(float(long_sessions["soc_end_pct"].mean()), 2)
