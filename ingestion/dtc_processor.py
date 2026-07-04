"""
DTC (Diagnostic Trouble Code) processor for AutoPredict.

DTC records arrive as Base64-encoded 7-byte strings in CH-21 fields:
  DTCInfomationBMS, DTCInfomationTC, DTCInfomationTCM, DTCInfomationECM, DTCInfomationBCM

Byte layout (Big Data Spec):
  Byte 0  bits[7-4]: serious_level (0=info 1=low 2=medium 3=high)
  Byte 2:  DTC high byte
  Byte 3:  DTC low byte
  Byte 4:  DTC failure type byte
  Byte 5  bit 7: warningIndicatorRequested
  Byte 5  bit 3: confirmedDTC
  Byte 5  bit 2: pendingDTC
  Byte 5  bit 1: testFailedThisOperationCycle
  Byte 5  bit 0: testFailed
  Byte 6:  DTC type
"""
from __future__ import annotations

import base64
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

_CATALOGUE_PATH  = Path(__file__).resolve().parents[1] / "data" / "reference" / "dtc_catalogue.csv"
_TAXONOMY_PATH   = Path(__file__).resolve().parents[1] / "data" / "reference" / "failure_taxonomy.json"

# Map high nibble of DTC high byte → letter prefix
_PREFIX_MAP = {0: "P", 1: "P", 2: "C", 3: "B"}


@dataclass
class DTCRecord:
    source_system:     str    # "BMS" | "TC" | "TCM" | "ECM" | "BCM"
    dtc_code:          str    # e.g. "P0562"
    serious_level:     int    # 0-3
    is_confirmed:      bool
    is_pending:        bool
    warning_indicator: bool
    test_failed:       bool
    raw_bytes:         bytes


class DTCProcessor:

    SAFETY_CRITICAL_DTCS = {"P0A80", "P1E00", "C0040", "C0035", "P0562", "U0100", "P0A09"}

    # ── Decoding ──────────────────────────────────────────────────────────────

    def decode(self, base64_string: str, source_system: str) -> DTCRecord | None:
        """Decode a single Base64-encoded DTC payload (7 bytes)."""
        try:
            # Pad to valid base64 length
            padding = (4 - len(base64_string) % 4) % 4
            raw = base64.b64decode(base64_string + "=" * padding)
        except Exception as exc:
            log.debug("DTC base64 decode error (%s): %s", source_system, exc)
            return None

        if len(raw) < 7:
            log.debug("DTC payload too short: %d bytes", len(raw))
            return None

        try:
            serious_level = (raw[0] >> 4) & 0x0F

            high_nibble   = (raw[2] >> 4) & 0x0F
            prefix        = _PREFIX_MAP.get(high_nibble, "U")
            code_number   = ((raw[2] & 0x3F) << 8) | raw[3]
            dtc_code      = f"{prefix}{code_number:04X}"

            status_byte          = raw[5]
            warning_indicator    = bool(status_byte & 0x80)
            confirmed_dtc        = bool(status_byte & 0x08)
            pending_dtc          = bool(status_byte & 0x04)
            test_failed          = bool(status_byte & 0x01)

            return DTCRecord(
                source_system=source_system,
                dtc_code=dtc_code,
                serious_level=min(serious_level, 3),
                is_confirmed=confirmed_dtc,
                is_pending=pending_dtc,
                warning_indicator=warning_indicator,
                test_failed=test_failed,
                raw_bytes=bytes(raw),
            )
        except Exception as exc:
            log.debug("DTC byte parsing error: %s", exc)
            return None

    # ── Classification ────────────────────────────────────────────────────────

    def get_fault_system(self, dtc_code: str) -> str:
        if not dtc_code:
            return "Unknown"
        prefix = dtc_code[0].upper()
        return {
            "P": "Powertrain",
            "C": "Chassis",
            "B": "Body",
            "U": "Network",
        }.get(prefix, "Unknown")

    def is_safety_critical(self, dtc: DTCRecord) -> bool:
        return dtc.serious_level >= 2 or dtc.dtc_code in self.SAFETY_CRITICAL_DTCS

    # ── Parts mapping ─────────────────────────────────────────────────────────

    def map_to_parts(self, dtc_code: str) -> list[str]:
        """Return part names whose related_dtcs includes dtc_code."""
        if not _TAXONOMY_PATH.exists():
            return []
        try:
            taxonomy = json.loads(_TAXONOMY_PATH.read_text())
            part_list = taxonomy if isinstance(taxonomy, list) else taxonomy.get("parts", [])
            parts = []
            for part in part_list:
                if dtc_code in part.get("related_dtcs", []):
                    parts.append(part["part_name"])
            return parts
        except Exception as exc:
            log.debug("Taxonomy lookup error for %s: %s", dtc_code, exc)
            return []

    # ── Catalogue lookup ─────────────────────────────────────────────────────

    def lookup_catalogue(self, dtc_code: str) -> dict | None:
        """Return catalogue row for a DTC code, or None if not found."""
        if not _CATALOGUE_PATH.exists():
            return None
        try:
            with _CATALOGUE_PATH.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row["dtc_code"] == dtc_code:
                        return dict(row)
        except Exception:
            pass
        return None
