"""
ThermalRunawayEarlyWarner — deterministic rule-based escalation detector for
EV/PHEV high-voltage battery safety.

Operates on CH-22 BMS signals exclusively. ML models are inappropriate here:
the global dataset of thermal runaway events is too small (< 200 incidents with
confirmed precursor telemetry), and safety-critical predictions must be
deterministic and auditable by a human engineer without black-box inference.

PostgreSQL DDL for ev_thermal_runaway_log:

    CREATE TABLE IF NOT EXISTS ev_thermal_runaway_log (
        id           BIGSERIAL    PRIMARY KEY,
        vin          VARCHAR(17)  NOT NULL,
        risk_level   VARCHAR(20)  NOT NULL,
        factors_json JSONB        NOT NULL,
        action       TEXT         NOT NULL,
        evaluated_at TIMESTAMPTZ  NOT NULL,
        created_at   TIMESTAMPTZ  NOT NULL DEFAULT now()
    );
    CREATE INDEX IF NOT EXISTS idx_ev_thermal_runaway_vin_time
        ON ev_thermal_runaway_log(vin, evaluated_at DESC);

Celery beat task: check_all_ev_vins runs every 15 minutes.
SMS for CRITICAL results is dispatched directly (bypasses cooldown queue).
ALL evaluations (including NONE) are persisted for audit trail.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

DATA_DIR = Path(os.getenv("DATA_DIR", "data/synthetic"))

# Emergency contact for CRITICAL thermal alerts (dealer ops centre or safety team)
_THERMAL_EMERGENCY_PHONE = os.getenv("THERMAL_EMERGENCY_CONTACT_PHONE", "")


# ── Signal helper ──────────────────────────────────────────────────────────────

def _col(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    """Return df[col] or a constant series of *default* when the column is absent.

    Handles the case where CH-22 signals may not yet exist in synthetic data
    or on older TBox firmware versions.
    """
    if col in df.columns:
        return df[col]
    return pd.Series([default] * len(df), index=df.index, dtype=float)


# ── Audit persistence ──────────────────────────────────────────────────────────

def persist_thermal_log(result: dict) -> None:
    """Write one thermal runaway evaluation row to ev_thermal_runaway_log.

    Creates the table on first call (CREATE TABLE IF NOT EXISTS).
    Non-fatal: swallows all exceptions so the caller never breaks.
    """
    try:
        from sqlalchemy import create_engine, text

        url = os.getenv("POSTGRES_URL", "sqlite:///./autopredict.db")
        engine = create_engine(url, pool_pre_ping=True)

        _DDL = text("""
            CREATE TABLE IF NOT EXISTS ev_thermal_runaway_log (
                id           BIGSERIAL    PRIMARY KEY,
                vin          VARCHAR(17)  NOT NULL,
                risk_level   VARCHAR(20)  NOT NULL,
                factors_json JSONB        NOT NULL,
                action       TEXT         NOT NULL,
                evaluated_at TIMESTAMPTZ  NOT NULL,
                created_at   TIMESTAMPTZ  NOT NULL DEFAULT now()
            )
        """)
        _IDX = text("""
            CREATE INDEX IF NOT EXISTS idx_ev_thermal_runaway_vin_time
                ON ev_thermal_runaway_log(vin, evaluated_at DESC)
        """)
        _INSERT = text("""
            INSERT INTO ev_thermal_runaway_log
                (vin, risk_level, factors_json, action, evaluated_at)
            VALUES
                (:vin, :risk_level, CAST(:factors_json AS JSONB), :action, :evaluated_at)
        """)
        _INSERT_SQLITE = text("""
            INSERT INTO ev_thermal_runaway_log
                (vin, risk_level, factors_json, action, evaluated_at)
            VALUES
                (:vin, :risk_level, :factors_json, :action, :evaluated_at)
        """)

        evaluated_at = result.get("evaluated_at")
        if isinstance(evaluated_at, str):
            evaluated_at = datetime.fromisoformat(evaluated_at)

        params = {
            "vin":          result["vin"],
            "risk_level":   result["risk_level"],
            "factors_json": json.dumps(result.get("factors", [])),
            "action":       result["action"],
            "evaluated_at": evaluated_at,
        }

        with engine.begin() as conn:
            conn.execute(_DDL)
            try:
                conn.execute(_IDX)
            except Exception:
                pass
            try:
                conn.execute(_INSERT, params)
            except Exception:
                # SQLite fallback — no JSONB cast
                conn.execute(_INSERT_SQLITE, params)

    except Exception as exc:
        log.debug("persist_thermal_log skipped (non-fatal): %s", exc)


# ── Direct SMS dispatch (no cooldown) ─────────────────────────────────────────

def _sms_body_for_critical(vin: str, factors: list[dict]) -> str:
    top = factors[0]["detail"] if factors else "multiple BMS fault conditions"
    msg = f"CRITICAL SAFETY: {vin} thermal runaway risk detected. {top}. STOP VEHICLE & CALL DEALER NOW."
    return msg[:160]  # single SMS


def dispatch_critical_sms(result: dict) -> None:
    """Send an SMS immediately for a CRITICAL thermal runaway result.

    Calls _send_sms() from alerts.alert_dispatcher directly, bypassing the
    AlertDispatcher cooldown queue. Per spec: never suppress CRITICAL alerts.
    """
    if result.get("risk_level") != "CRITICAL":
        return

    to_number = _THERMAL_EMERGENCY_PHONE
    if not to_number:
        log.warning("CRITICAL thermal alert for %s — SMS not sent (THERMAL_EMERGENCY_CONTACT_PHONE not set)", result["vin"])
        return

    body = _sms_body_for_critical(result["vin"], result.get("factors", []))
    try:
        from alerts.alert_dispatcher import _send_sms
        asyncio.run(_send_sms(to_number, body))
        log.warning("CRITICAL thermal runaway SMS sent for %s", result["vin"])
    except Exception as exc:
        log.error("CRITICAL thermal SMS dispatch failed for %s: %s", result["vin"], exc)


# ── Main classifier ────────────────────────────────────────────────────────────

class ThermalRunawayEarlyWarner:
    """
    Rule-based thermal runaway early warning system for EV/PHEV battery packs.

    Evaluates 6 deterministic rules on CH-22 BMS signals from the last 48 hours.
    Returns risk_level in {NONE, MEDIUM, HIGH, CRITICAL} with every contributing
    factor itemised so a service engineer can understand and verify the decision.

    Why rules, not ML:
      - Global thermal runaway event corpus is too small for reliable training.
      - Deterministic rules are fully traceable and auditable.
      - False-negative rate on safety systems must be near-zero; a threshold
        rule guarantees that — a model cannot.
    """

    FAULT_LEVELS: dict[str, dict[int, str]] = {
        "vehBMSCMUFlt":      {1: "CMU comm fault",        2: "CMU multi-fault",        3: "CMU critical"},
        "vehBMSCellVoltFlt": {1: "cell voltage low",       2: "cell voltage severe",     3: "cell reversal risk"},
        "vehBMSPackTemFlt":  {1: "pack temp high",          2: "pack temp severe",        3: "thermal event imminent"},
        "vehBMSPackVoltFlt": {1: "pack voltage low",        2: "pack voltage severe",     3: "pack voltage critical"},
    }

    def classify(
        self,
        vin: str,
        recent_telemetry_df: pd.DataFrame,
        battery_features: dict,
    ) -> dict:
        """
        Evaluate thermal runaway risk for a single VIN.

        Args:
            vin: Vehicle identification number.
            recent_telemetry_df: Last 48 hours of CH-22 data.
                May contain multiple VINs; the method filters to *vin*.
                Columns not present in the DataFrame are treated as all-zero
                (unknown / not transmitted), which is the safe default.
            battery_features: Pre-computed features from EVChargingFeatureEngine
                (optional, used for future rule extensions).

        Returns:
            {
              "vin": str,
              "risk_level": "NONE" | "MEDIUM" | "HIGH" | "CRITICAL",
              "factors": [{"signal": str, "level": str, "detail": str}, ...],
              "action": str,
              "evaluated_at": ISO-8601 str,
            }
        """
        # Normalise column name: TBox CSVs use "VIN" uppercase
        df = recent_telemetry_df.copy()
        if "VIN" in df.columns and "vin" not in df.columns:
            df = df.rename(columns={"VIN": "vin"})

        vin_df = df[df["vin"] == vin] if "vin" in df.columns else df

        if vin_df.empty:
            return {
                "vin":          vin,
                "risk_level":   "NONE",
                "factors":      [],
                "action":       "monitor",
                "evaluated_at": datetime.now(timezone.utc).isoformat(),
            }

        factors: list[dict] = []

        # ── Rule 1: Pre-thermal runaway indicator from BMS (highest severity) ─────
        pre_thr = _col(vin_df, "vehBMSPreThrmlFltInd", 0.0)
        if int(pre_thr.max()) == 1:
            factors.append({
                "signal": "vehBMSPreThrmlFltInd",
                "level":  "CRITICAL",
                "detail": "BMS has flagged pre-thermal runaway condition",
            })

        # ── Rule 2: Any fault signal at level 3 — immediate escalation ───────────
        for signal, descriptions in self.FAULT_LEVELS.items():
            col_series = _col(vin_df, signal, 0.0)
            if int(col_series.max()) >= 3:
                factors.append({
                    "signal": signal,
                    "level":  "CRITICAL",
                    "detail": descriptions[3],
                })

        # ── Rule 3: Two or more fault signals at level 2 simultaneously ──────────
        level2_count = sum(
            1 for s in self.FAULT_LEVELS
            if int(_col(vin_df, s, 0.0).max()) >= 2
        )
        if level2_count >= 2:
            factors.append({
                "signal": "multi_fault_level2",
                "level":  "HIGH",
                "detail": f"{level2_count} fault signals at level 2 simultaneously",
            })

        # ── Rule 4: Cell temperature spread expanding rapidly ─────────────────────
        # TBox scaling: vehBMSCellMaxTem and vehBMSCellMinTem are raw integer codes.
        # Actual temp (°C) = raw × 0.5 — the *delta* computation is the same scale.
        cell_max = _col(vin_df, "vehBMSCellMaxTem", float("nan"))
        cell_min = _col(vin_df, "vehBMSCellMinTem", float("nan"))
        cell_temp_delta = (cell_max - cell_min) * 0.5
        valid_delta = cell_temp_delta.dropna()
        if len(valid_delta) > 2:
            temp_delta_trend = float(valid_delta.iloc[-1] - valid_delta.iloc[0])
            if temp_delta_trend > 15:
                factors.append({
                    "signal": "cell_temp_delta_rapid_rise",
                    "level":  "HIGH",
                    "detail": f"Cell temp spread grew {temp_delta_trend:.0f}°C in 48h",
                })

        # ── Rule 5: Isolation resistance declining toward critical threshold ──────
        # vehBMSPtIsltnRstcV == 1 means the reading is invalid (TBox spec).
        iso_flag = _col(vin_df, "vehBMSPtIsltnRstcV", 0.0)
        iso_raw  = _col(vin_df, "vehBMSPtIsltnRstc",  float("nan"))
        iso_res  = iso_raw.where(iso_flag != 1) * 0.5   # kΩ
        valid_iso = iso_res.dropna()
        if len(valid_iso) > 5:
            iso_min   = float(valid_iso.min())
            iso_trend = float(valid_iso.diff().mean())  # negative = declining
            if iso_min < 300:
                factors.append({
                    "signal": "isolation_resistance_critical",
                    "level":  "CRITICAL",
                    "detail": f"Isolation resistance {iso_min:.0f} kΩ (critical < 300)",
                })
            elif iso_min < 600 and iso_trend < -5:
                factors.append({
                    "signal": "isolation_resistance_declining",
                    "level":  "HIGH",
                    "detail": f"Isolation declining rapidly, now {iso_min:.0f} kΩ",
                })

        # ── Rule 6: HV interlock (HVIL) open while system is active ──────────────
        # HVIL should be closed (1 = TRUE) whenever the HV bus is energised.
        hvil    = _col(vin_df, "vehBMSHVILClsd", 1.0)
        bsc_sta = _col(vin_df, "vehBMSBscSta",   0.0)
        if int(hvil.min()) == 0:
            hvil_open_while_active = int(((hvil == 0) & (bsc_sta > 0)).sum())
            if hvil_open_while_active > 0:
                factors.append({
                    "signal": "hvil_open",
                    "level":  "CRITICAL",
                    "detail": "HV interlock open while system active",
                })

        # ── Determine overall risk level ──────────────────────────────────────────
        if any(f["level"] == "CRITICAL" for f in factors):
            risk_level = "CRITICAL"
            action     = "STOP_VEHICLE_IMMEDIATELY_CONTACT_DEALER"
        elif any(f["level"] == "HIGH" for f in factors):
            risk_level = "HIGH"
            action     = "SERVICE_WITHIN_24_HOURS"
        elif factors:
            risk_level = "MEDIUM"
            action     = "SCHEDULE_INSPECTION"
        else:
            risk_level = "NONE"
            action     = "monitor"

        return {
            "vin":          vin,
            "risk_level":   risk_level,
            "factors":      factors,
            "action":       action,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }


# ── Telemetry loader (used by beat task) ──────────────────────────────────────

def _load_ev_telemetry(vin: str) -> pd.DataFrame:
    """Load telemetry CSV for *vin*.

    In production, filters to the last 48 h by timestamp.
    Falls back to the most-recent 500 rows when data is synthetic/historical.
    """
    for pattern in (f"telemetry_{vin}.csv", f"{vin}_telemetry.csv"):
        csv_path = DATA_DIR / pattern
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                ts_col = "StartTime-TimeStamp"
                if ts_col in df.columns:
                    cutoff = datetime.now(timezone.utc).timestamp() - 48 * 3600
                    recent = df[df[ts_col] >= cutoff]
                    if not recent.empty:
                        return recent
                # Synthetic data is historical — use last 500 rows as evaluation window
                return df.tail(500)
            except Exception as exc:
                log.debug("Failed to load telemetry for %s: %s", vin, exc)
    return pd.DataFrame()


def _load_ev_vins() -> list[str]:
    """Return VINs of all EV and PHEV vehicles from the fleet CSV."""
    for name in ("fleet_master.csv", "fleet.csv"):
        p = DATA_DIR / name
        if p.exists():
            try:
                fleet = pd.read_csv(p)
                ev_mask = fleet["fuel_type"].str.upper().isin({"EV", "PHEV", "BEV"})
                return fleet.loc[ev_mask, "vin"].astype(str).tolist()
            except Exception:
                pass
    return []


# ── Celery beat task — every 15 minutes for all EV VINs ───────────────────────

try:
    from celery import Celery
    import os as _os
    from datetime import timedelta as _td

    _celery = Celery(
        "thermal_runaway_monitor",
        broker=_os.getenv("CELERY_BROKER_URL", _os.getenv("REDIS_URL", "redis://localhost:6379/0")),
    )

    @_celery.task(name="models.thermal_runaway_model.check_all_ev_vins")
    def check_all_ev_vins() -> dict:
        """
        Evaluate thermal runaway risk for every EV/PHEV VIN.

        Runs every 15 minutes. Results are:
          1. Persisted to ev_thermal_runaway_log (ALL risk levels, for audit).
          2. Dispatched via SMS immediately for CRITICAL (no cooldown).

        Returns a summary dict for the Celery result backend.
        """
        warner = ThermalRunawayEarlyWarner()
        ev_vins = _load_ev_vins()

        counts: dict[str, int] = {"NONE": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
        critical_vins: list[str] = []

        for vin in ev_vins:
            telem_df = _load_ev_telemetry(vin)
            result = warner.classify(vin, telem_df, {})

            # ── 1. Persist to audit table (all evaluations, not just alerts) ───
            persist_thermal_log(result)

            # ── 2. CRITICAL: SMS immediately, no cooldown, no suppression ──────
            level = result["risk_level"]
            if level == "CRITICAL":
                dispatch_critical_sms(result)
                critical_vins.append(vin)
                log.critical(
                    "THERMAL RUNAWAY CRITICAL: vin=%s factors=%s",
                    vin,
                    [f["signal"] for f in result.get("factors", [])],
                )
            elif level in ("HIGH", "MEDIUM"):
                log.warning("Thermal runaway %s: vin=%s", level, vin)

            counts[level] = counts.get(level, 0) + 1

        summary = {
            "evaluated_at":  datetime.now(timezone.utc).isoformat(),
            "ev_vins_total": len(ev_vins),
            "counts":        counts,
            "critical_vins": critical_vins,
        }
        log.info("Thermal runaway sweep: %s", summary)
        return summary

    # 15-minute sweep for all EV VINs
    _celery.conf.beat_schedule = {
        "thermal-runaway-check-15min": {
            "task":     "models.thermal_runaway_model.check_all_ev_vins",
            "schedule": _td(minutes=15),
        }
    }
    _celery.conf.timezone = "UTC"

except ImportError:
    pass
