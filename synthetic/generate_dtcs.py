"""
Synthetic DTC (Diagnostic Trouble Code) event generator.

Injects realistic DTC sequences for each failure type in the failures manifest,
plus background noise (false positives) for model training diversity.

Output: data/synthetic/dtc_events.csv
"""
from __future__ import annotations

import logging
import os
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DATA_DIR = Path(os.getenv("DATA_DIR", "data/synthetic"))

# ── Freeze-frame value distributions ──────────────────────────────────────────
_FF_SPEED    = (0, 140)   # kph
_FF_RPM      = (700, 3500)
_FF_COOLANT  = (75, 105)
_FF_SOC      = (20, 95)


def _ff_row(rng: random.Random) -> dict:
    """Generate realistic freeze-frame data."""
    return {
        "freeze_frame_speed_kph":         round(rng.uniform(*_FF_SPEED), 1),
        "freeze_frame_rpm":               rng.randint(*_FF_RPM),
        "freeze_frame_coolant_temp_c":    rng.randint(*_FF_COOLANT),
        "freeze_frame_soc_pct":           round(rng.uniform(*_FF_SOC), 1),
    }


def _event(
    vin: str,
    event_date: datetime,
    source_system: str,
    dtc_code: str,
    serious_level: int,
    *,
    is_confirmed: bool = False,
    is_pending: bool = False,
    warning_indicator: bool = False,
    occurrence_count: int = 1,
    first_seen_date: datetime | None = None,
    cleared_date: datetime | None = None,
    rng: random.Random,
) -> dict:
    ff = _ff_row(rng)
    return {
        "vin":                        vin,
        "event_date":                 event_date.strftime("%Y-%m-%d"),
        "source_system":              source_system,
        "dtc_code":                   dtc_code,
        "serious_level":              serious_level,
        "is_confirmed":               is_confirmed,
        "is_pending":                 is_pending,
        "warning_indicator":          warning_indicator,
        "occurrence_count":           occurrence_count,
        "first_seen_date":            (first_seen_date or event_date).strftime("%Y-%m-%d"),
        "cleared_date":               cleared_date.strftime("%Y-%m-%d") if cleared_date else "",
        **ff,
    }


class DTCGenerator:

    # ── Failure-type injection sequences ──────────────────────────────────────

    _SEQUENCES: dict[str, list[dict]] = {
        "brake_degradation": [
            dict(offset=0,  dtc="C0040", level=1, pending=True,  confirmed=False, occurrences=1),
            dict(offset=21, dtc="C0040", level=2, pending=False, confirmed=True,  occurrences=3),
            dict(offset=28, dtc="C0110", level=1, pending=True,  confirmed=False, occurrences=1),
        ],
        "12v_battery_failure": [
            dict(offset=0,  dtc="P0562", level=1, pending=True,  confirmed=False, occurrences=1),
            dict(offset=8,  dtc="P0562", level=2, pending=False, confirmed=True,  occurrences=4),
            dict(offset=13, dtc="U0100", level=3, pending=False, confirmed=True,  occurrences=1),
        ],
        "hv_battery_degradation": [
            dict(offset=0,  dtc="P0A80", level=1, pending=True,  confirmed=False, warning=False, occurrences=1),
            dict(offset=45, dtc="P1E00", level=2, pending=False, confirmed=True,  warning=False, occurrences=2),
            dict(offset=85, dtc="P0A80", level=3, pending=False, confirmed=True,  warning=True,  occurrences=5),
        ],
        "engine_overheating": [
            dict(offset=0, dtc="P0117", level=1, pending=True,  confirmed=False, occurrences=1),
            dict(offset=4, dtc="P0300", level=1, pending=True,  confirmed=False, occurrences=2),
            dict(offset=6, dtc="P0172", level=2, pending=False, confirmed=True,  occurrences=3),
        ],
        "tyre_puncture": [
            dict(offset=0, dtc="C0775", level=3, pending=False, confirmed=True, occurrences=1),
            dict(offset=0, dtc="C0776", level=3, pending=False, confirmed=True, occurrences=1),
            dict(offset=0, dtc="C0777", level=3, pending=False, confirmed=True, occurrences=1),
            dict(offset=0, dtc="C0778", level=3, pending=False, confirmed=True, occurrences=1),
        ],
    }

    _SYSTEM_MAP = {
        "C0040": "TC", "C0035": "TC", "C0036": "TC", "C0045": "TC",
        "C0046": "TC", "C0110": "TC",
        "C0775": "TC", "C0776": "TC", "C0777": "TC", "C0778": "TC",
        "P0562": "BCM", "P0300": "ECM", "P0117": "ECM", "P0172": "ECM",
        "P0101": "ECM", "P0340": "ECM", "P0420": "ECM",
        "P0A80": "BMS", "P1E00": "BMS", "P0A09": "BMS",
        "P0B32": "BMS", "P0C16": "TCM", "P0AA6": "BMS",
        "U0100": "ECM", "U0155": "BCM",
    }

    def _source(self, dtc_code: str) -> str:
        return self._SYSTEM_MAP.get(dtc_code, "ECM")

    # ── Main generation method ─────────────────────────────────────────────────

    def generate(
        self,
        fleet_df: pd.DataFrame,
        failures_manifest_df: pd.DataFrame,
        num_days: int = 90,
    ) -> pd.DataFrame:
        """
        Generate DTC events for all VINs.

        Args:
            fleet_df:              DataFrame with at least 'VIN' column
            failures_manifest_df:  DataFrame with columns: VIN, failure_type, failure_date
            num_days:              Dataset horizon in days

        Returns:
            DataFrame saved to data/synthetic/dtc_events.csv
        """
        rng = random.Random(42)
        np_rng = np.random.default_rng(42)
        rows: list[dict] = []

        all_vins = list(fleet_df["VIN"].unique() if "VIN" in fleet_df.columns else fleet_df["vin"].unique())
        base_date = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        ) - timedelta(days=num_days)

        # ── 1. Failure-driven DTCs ─────────────────────────────────────────
        for _, mrow in failures_manifest_df.iterrows():
            vin          = str(mrow.get("VIN") or mrow.get("vin", ""))
            failure_type = str(mrow.get("failure_type", ""))
            raw_date     = mrow.get("failure_date") or mrow.get("date")

            try:
                failure_date = pd.to_datetime(raw_date).to_pydatetime()
                if failure_date.tzinfo is None:
                    failure_date = failure_date.replace(tzinfo=timezone.utc)
            except Exception:
                failure_date = base_date + timedelta(days=rng.randint(20, num_days - 5))

            seq = self._SEQUENCES.get(failure_type, [])
            for step in seq:
                event_date = failure_date + timedelta(days=step["offset"])
                rows.append(_event(
                    vin=vin,
                    event_date=event_date,
                    source_system=self._source(step["dtc"]),
                    dtc_code=step["dtc"],
                    serious_level=step["level"],
                    is_confirmed=step.get("confirmed", False),
                    is_pending=step.get("pending", False),
                    warning_indicator=step.get("warning", False),
                    occurrence_count=step.get("occurrences", 1),
                    rng=rng,
                ))

        # ── 2. Background noise (false positives) ──────────────────────────
        p562_vins = rng.sample(all_vins, k=max(1, int(len(all_vins) * 0.30)))
        for vin in p562_vins:
            day   = rng.randint(0, num_days - 5)
            start = base_date + timedelta(days=day)
            rows.append(_event(
                vin=vin, event_date=start, source_system="BCM",
                dtc_code="P0562", serious_level=1,
                is_pending=True, confirmed=False, occurrences=1,
                first_seen_date=start,
                cleared_date=start + timedelta(days=2),
                rng=rng,
            ))

        u155_vins = rng.sample(all_vins, k=max(1, int(len(all_vins) * 0.20)))
        for vin in u155_vins:
            day   = rng.randint(0, num_days - 5)
            start = base_date + timedelta(days=day)
            rows.append(_event(
                vin=vin, event_date=start, source_system="BCM",
                dtc_code="U0155", serious_level=1,
                is_pending=True, confirmed=False, occurrences=1,
                first_seen_date=start,
                cleared_date=start + timedelta(days=3),
                rng=rng,
            ))

        df = pd.DataFrame(rows)
        if df.empty:
            df = pd.DataFrame(columns=[
                "vin", "event_date", "source_system", "dtc_code", "serious_level",
                "is_confirmed", "is_pending", "warning_indicator", "occurrence_count",
                "first_seen_date", "cleared_date",
                "freeze_frame_speed_kph", "freeze_frame_rpm",
                "freeze_frame_coolant_temp_c", "freeze_frame_soc_pct",
            ])

        out_path = DATA_DIR / "dtc_events.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        log.info("DTC events written: %d rows -> %s", len(df), out_path)
        return df


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    fleet_path    = DATA_DIR / "fleet.csv"
    manifest_path = DATA_DIR / "failures_manifest.csv"

    if not fleet_path.exists():
        sys.exit(f"Fleet CSV not found: {fleet_path}  (run generate_fleet.py first)")
    if not manifest_path.exists():
        sys.exit(f"Failures manifest not found: {manifest_path}  (run generate_telemetry.py first)")

    fleet_df    = pd.read_csv(fleet_path)
    manifest_df = pd.read_csv(manifest_path)
    df = DTCGenerator().generate(fleet_df, manifest_df)
    print(f"Generated {len(df)} DTC events -> {DATA_DIR / 'dtc_events.csv'}")
