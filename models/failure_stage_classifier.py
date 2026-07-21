"""
FailureStageClassifier — maps ensemble probability + RUL + rule flags
to a FailureStage enum for each failure type.

PostgreSQL DDL for vehicle_health_stages:

    CREATE TABLE IF NOT EXISTS vehicle_health_stages (
        id               BIGSERIAL PRIMARY KEY,
        vin              VARCHAR(17)  NOT NULL,
        failure_type     VARCHAR(30)  NOT NULL,
        stage            INTEGER      NOT NULL,
        stage_name       VARCHAR(20)  NOT NULL,
        classified_at    TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
        ensemble_prob    FLOAT,
        rul_days         FLOAT,
        features_json    JSONB
    );
    CREATE INDEX IF NOT EXISTS idx_vhs_vin_type_time
        ON vehicle_health_stages (vin, failure_type, classified_at DESC);
"""
from __future__ import annotations

from enum import IntEnum
from typing import Any


class FailureStage(IntEnum):
    HEALTHY    = 0
    EARLY      = 1
    NOTICEABLE = 2
    HIGH_RISK  = 3
    IMMINENT   = 4
    CRITICAL   = 5


STAGE_LABELS: dict[int, str] = {
    0: "No issues detected",
    1: "Early degradation — monitor",
    2: "Confirmed — pre-order part",
    3: "Schedule within 30 days",
    4: "Book within 2 weeks — service required",
    5: "IMMEDIATE attention — safety risk",
}

CRITICAL_RULE_FLAGS: dict[str, list[str]] = {
    "brake":       ["vehBrkFludLvlLow", "vehABSF"],
    "oil":         ["vehOilPressureWarning"],
    "hv_battery":  ["vehBMSPackTemFlt_3", "vehBMSCMUFlt_3", "vehBMSPreThrmlFltInd"],
    "12v_battery": [],
    "tyre":        ["tpms_deflation_flag"],
    "overheating": ["vehOilPressureWarning"],
}

_FAILURE_TYPES = ["brake", "oil", "hv_battery", "12v_battery", "tyre", "overheating"]


class FailureStageClassifier:
    """Classify each failure type into a FailureStage."""

    _thresholds: dict[str, dict[str, float]] = {
        "brake":       {"harsh_brake_rate_30d": 3.0, "brake_stress_cumulative": 2000.0},
        "oil":         {"oil_degradation_index": 50.0, "cold_start_count_30d": 10.0},
        "hv_battery":  {"cell_voltage_spread": 0.03, "soh_trend_slope_90d": -0.005},
        "12v_battery": {"resting_voltage_trend_14d": -0.005},
        "tyre":        {"pressure_drop_rate_fl": -0.8},
        "overheating": {"coolant_overtemp_count_30d": 3.0},
    }

    def classify(
        self,
        failure_type: str,
        ensemble_prob: float,
        rul_days: float | None,
        rule_flags: dict[str, Any],
        features: dict[str, Any],
        *,
        thermal_runaway_risk: str = "NONE",
    ) -> FailureStage:
        """
        Classify a single (failure_type, VIN) into a FailureStage.

        Priority order: thermal_runaway_override > CRITICAL > IMMINENT > HIGH_RISK > NOTICEABLE > EARLY > HEALTHY

        thermal_runaway_risk: result from ThermalRunawayEarlyWarner.classify()["risk_level"].
            When "CRITICAL" and failure_type == "hv_battery", this overrides all other
            signals and returns FailureStage.CRITICAL unconditionally.
        """
        # Safety override: ThermalRunawayEarlyWarner CRITICAL always wins for HV battery.
        # This check runs before ensemble probability so it cannot be suppressed by model output.
        if failure_type == "hv_battery" and thermal_runaway_risk == "CRITICAL":
            return FailureStage.CRITICAL

        flags = CRITICAL_RULE_FLAGS.get(failure_type, [])

        # CRITICAL: active safety flag OR very high probability OR imminent failure
        if (
            any(rule_flags.get(f) for f in flags)
            or ensemble_prob > 0.85
            or (rul_days is not None and rul_days < 3)
        ):
            return FailureStage.CRITICAL

        # IMMINENT
        if ensemble_prob > 0.65 or (rul_days is not None and rul_days < 14):
            return FailureStage.IMMINENT

        # HIGH_RISK
        if ensemble_prob > 0.40 or (rul_days is not None and rul_days < 30):
            return FailureStage.HIGH_RISK

        # NOTICEABLE
        if ensemble_prob > 0.20:
            return FailureStage.NOTICEABLE

        # EARLY: any key feature above absolute threshold
        if self._any_feature_elevated(failure_type, features):
            return FailureStage.EARLY

        return FailureStage.HEALTHY

    def _any_feature_elevated(self, failure_type: str, features: dict[str, Any]) -> bool:
        """Return True if any monitored feature exceeds its early-warning threshold."""
        for feat, threshold in self._thresholds.get(failure_type, {}).items():
            v = features.get(feat)
            if v is not None:
                try:
                    if float(v) > threshold:
                        return True
                except (TypeError, ValueError):
                    pass
        return False

    def classify_all(
        self,
        vin: str,
        ensemble_probs: dict[str, float],
        rul_dict: dict[str, float | None],
        rule_flags: dict[str, Any],
        features: dict[str, Any],
        *,
        thermal_runaway_risk: str = "NONE",
    ) -> dict[str, FailureStage]:
        """
        Classify all 6 failure types for a VIN in one call.

        thermal_runaway_risk is forwarded only to the hv_battery call.
        Returns {failure_type: FailureStage}.
        """
        return {
            ft: self.classify(
                ft,
                ensemble_probs.get(ft, 0.0),
                rul_dict.get(ft),
                rule_flags,
                features,
                thermal_runaway_risk=thermal_runaway_risk if ft == "hv_battery" else "NONE",
            )
            for ft in _FAILURE_TYPES
        }

    def stage_label(self, stage: FailureStage) -> str:
        return STAGE_LABELS.get(int(stage), "Unknown")


# ── PostgreSQL persistence helper ──────────────────────────────────────────────

def persist_stages(
    vin: str,
    stages: dict[str, FailureStage],
    ensemble_probs: dict[str, float],
    rul_dict: dict[str, float | None],
    features: dict[str, Any],
    db_url: str | None = None,
) -> None:
    """
    Upsert classified stages into vehicle_health_stages table.

    Skips gracefully if SQLAlchemy / database is unavailable.
    """
    import json
    from datetime import datetime, timezone

    try:
        from sqlalchemy import create_engine, text
        import os
        url = db_url or os.getenv("POSTGRES_URL", "sqlite:///./autopredict.db")
        engine = create_engine(url)
        now = datetime.now(timezone.utc)

        rows = []
        for ft, stage in stages.items():
            rows.append({
                "vin":           vin,
                "failure_type":  ft,
                "stage":         int(stage),
                "stage_name":    stage.name,
                "classified_at": now,
                "ensemble_prob": float(ensemble_probs.get(ft, 0.0)),
                "rul_days":      rul_dict.get(ft),
                "features_json": json.dumps({k: v for k, v in features.items()
                                             if isinstance(v, (int, float, str, bool, type(None)))}),
            })

        insert_sql = text("""
            INSERT INTO vehicle_health_stages
                (vin, failure_type, stage, stage_name, classified_at,
                 ensemble_prob, rul_days, features_json)
            VALUES
                (:vin, :failure_type, :stage, :stage_name, :classified_at,
                 :ensemble_prob, :rul_days, CAST(:features_json AS JSONB))
        """)

        with engine.begin() as conn:
            for row in rows:
                try:
                    conn.execute(insert_sql, row)
                except Exception:
                    # SQLite fallback (no JSONB cast)
                    fallback = text("""
                        INSERT INTO vehicle_health_stages
                            (vin, failure_type, stage, stage_name, classified_at,
                             ensemble_prob, rul_days, features_json)
                        VALUES
                            (:vin, :failure_type, :stage, :stage_name, :classified_at,
                             :ensemble_prob, :rul_days, :features_json)
                    """)
                    conn.execute(fallback, row)

    except Exception as exc:
        import logging
        logging.getLogger(__name__).debug("persist_stages skipped: %s", exc)
