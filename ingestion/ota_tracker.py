"""
OTA firmware update tracker.

Records OTA events and maintains current software versions per VIN.
Provides OTA-derived features for the HV battery model.

PostgreSQL DDL:

    CREATE TABLE ota_events (
      id SERIAL PRIMARY KEY, vin VARCHAR(17) NOT NULL,
      ota_campaign_id VARCHAR(30), component VARCHAR(10),
      from_version VARCHAR(20), to_version VARCHAR(20),
      ota_start_time TIMESTAMP, ota_complete_time TIMESTAMP,
      ota_success BOOLEAN DEFAULT FALSE
    );

    CREATE TABLE vehicle_software_versions (
      vin VARCHAR(17) PRIMARY KEY,
      tbox_version VARCHAR(20), bms_version VARCHAR(20),
      vcu_version VARCHAR(20), mcu_version VARCHAR(20), adas_version VARCHAR(20),
      last_updated TIMESTAMP DEFAULT NOW()
    );
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

log = logging.getLogger(__name__)


class OTATracker:

    def _get_engine(self):
        try:
            from sqlalchemy import create_engine
            return create_engine(os.getenv("POSTGRES_URL", "sqlite:///./autopredict.db"))
        except Exception:
            return None

    def ensure_tables(self) -> None:
        engine = self._get_engine()
        if engine is None:
            return
        try:
            from sqlalchemy import text
            with engine.begin() as conn:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS ota_events (
                        id SERIAL PRIMARY KEY,
                        vin VARCHAR(17) NOT NULL,
                        ota_campaign_id VARCHAR(30),
                        component VARCHAR(10),
                        from_version VARCHAR(20),
                        to_version VARCHAR(20),
                        ota_start_time TIMESTAMP,
                        ota_complete_time TIMESTAMP,
                        ota_success BOOLEAN DEFAULT FALSE
                    )
                """))
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS vehicle_software_versions (
                        vin VARCHAR(17) PRIMARY KEY,
                        tbox_version VARCHAR(20),
                        bms_version VARCHAR(20),
                        vcu_version VARCHAR(20),
                        mcu_version VARCHAR(20),
                        adas_version VARCHAR(20),
                        last_updated TIMESTAMP DEFAULT NOW()
                    )
                """))
        except Exception as exc:
            log.debug("OTA table creation skipped: %s", exc)
            self._ensure_tables_sqlite()

    def _ensure_tables_sqlite(self) -> None:
        import sqlite3
        db = sqlite3.connect(os.getenv("SQLITE_DB", "autopredict.db"))
        db.execute("""
            CREATE TABLE IF NOT EXISTS ota_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vin TEXT NOT NULL, ota_campaign_id TEXT,
                component TEXT, from_version TEXT, to_version TEXT,
                ota_start_time TEXT, ota_complete_time TEXT,
                ota_success INTEGER DEFAULT 0
            )
        """)
        db.execute("""
            CREATE TABLE IF NOT EXISTS vehicle_software_versions (
                vin TEXT PRIMARY KEY,
                tbox_version TEXT, bms_version TEXT,
                vcu_version TEXT, mcu_version TEXT, adas_version TEXT,
                last_updated TEXT
            )
        """)
        db.commit()
        db.close()

    def record_ota_event(
        self,
        vin: str,
        campaign_id: str,
        component: str,
        from_v: str,
        to_v: str,
        start_t: str,
        end_t: str,
        success: bool,
    ) -> None:
        engine = self._get_engine()
        if engine is None:
            return
        try:
            from sqlalchemy import text
            with engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO ota_events
                        (vin, ota_campaign_id, component, from_version, to_version,
                         ota_start_time, ota_complete_time, ota_success)
                    VALUES (:vin, :cid, :comp, :fv, :tv, :st, :et, :ok)
                """), {
                    "vin": vin, "cid": campaign_id, "comp": component,
                    "fv": from_v, "tv": to_v, "st": start_t, "et": end_t,
                    "ok": success,
                })
                if success:
                    col = f"{component.lower()}_version"
                    if col in ("tbox_version", "bms_version", "vcu_version", "mcu_version", "adas_version"):
                        conn.execute(text(f"""
                            INSERT INTO vehicle_software_versions (vin, {col}, last_updated)
                            VALUES (:vin, :ver, :now)
                            ON CONFLICT (vin) DO UPDATE SET {col} = :ver, last_updated = :now
                        """), {"vin": vin, "ver": to_v, "now": datetime.now(timezone.utc).isoformat()})
        except Exception:
            self._record_sqlite(vin, campaign_id, component, from_v, to_v, start_t, end_t, success)

    def _record_sqlite(self, vin, cid, comp, fv, tv, st, et, success):
        import sqlite3
        try:
            db = sqlite3.connect(os.getenv("SQLITE_DB", "autopredict.db"))
            db.execute(
                "INSERT INTO ota_events VALUES (NULL,?,?,?,?,?,?,?,?)",
                (vin, cid, comp, fv, tv, st, et, int(success)),
            )
            if success:
                col = f"{comp.lower()}_version"
                if col in ("tbox_version", "bms_version", "vcu_version", "mcu_version", "adas_version"):
                    db.execute(f"""
                        INSERT OR REPLACE INTO vehicle_software_versions
                            (vin, {col}, last_updated)
                        VALUES (?, ?, ?)
                    """, (vin, tv, datetime.now(timezone.utc).isoformat()))
            db.commit()
            db.close()
        except Exception as exc:
            log.debug("SQLite OTA insert failed: %s", exc)

    def get_ota_features(self, vin: str) -> dict[str, Any]:
        """OTA-derived features for the HV battery model."""
        result: dict[str, Any] = {
            "days_since_last_bms_ota": None,
            "bms_ota_count_90d": 0,
            "post_bms_ota_efficiency_delta": 0.0,
        }
        try:
            import pandas as pd
            engine = self._get_engine()
            if engine is None:
                return result
            from sqlalchemy import text
            with engine.connect() as conn:
                rows = conn.execute(text("""
                    SELECT ota_complete_time, ota_success
                    FROM ota_events
                    WHERE vin = :vin AND component = 'BMS' AND ota_success = TRUE
                    ORDER BY ota_complete_time DESC
                """), {"vin": vin}).fetchall()

            if rows:
                latest = pd.to_datetime(rows[0][0])
                now = pd.Timestamp.now(tz="UTC")
                result["days_since_last_bms_ota"] = int((now - latest).days)
                cutoff = now - pd.Timedelta(days=90)
                result["bms_ota_count_90d"] = sum(
                    1 for r in rows if pd.to_datetime(r[0]) >= cutoff
                )
                result["post_bms_ota_efficiency_delta"] = -3.0 if result["days_since_last_bms_ota"] <= 30 else 0.0
        except Exception as exc:
            log.debug("OTA feature fetch failed for %s: %s", vin, exc)
        return result
