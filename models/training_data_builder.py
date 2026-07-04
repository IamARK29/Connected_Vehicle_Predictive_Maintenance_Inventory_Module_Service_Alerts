"""
Training data builder — creates temporally-split train/val/test datasets
from service history labels and the offline feature store.

Feature cutoff is always set to service_date - 1 day so no future features
can leak into the training rows.
"""
from __future__ import annotations

import json
import logging
import os
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_DATA_DIR = Path(os.getenv("DATA_DIR", "data/synthetic"))

# ── Part-keyword maps for each model ─────────────────────────────────────────

def _build_part_keywords() -> dict[str, list[str]]:
    """Build PART_KEYWORDS from failure_taxonomy.json, with hardcoded fallback."""
    _ML_MODEL_TO_KEY = {
        "brake_wear_model":     "brake_wear",
        "engine_oil_model":     "engine_oil",
        "hv_battery_soh_model": "hv_battery",
        "battery_12v_model":    "battery_12v",
        "tyre_wear_model":      "tyre_wear",
    }
    fallback = {
        "brake_wear":      ["BRAKE PAD", "BRAKE DISC", "ROTOR", "CALIPER", "BRAKE-PAD"],
        "engine_oil":      ["ENGINE OIL", "OIL FILTER", "OIL CHANGE", "5W-30", "5W30"],
        "hv_battery":      ["HV BATTERY", "BATTERY PACK", "BMS", "BATTERY HEALTH"],
        "battery_12v":     ["12V BATTERY", "BATTERY REPLACEMENT", "LEAD ACID"],
        "tyre_wear":       ["TYRE", "TIRE", "GOODYEAR", "MICHELIN", "APOLLO"],
        "engine_overheat": ["COOLANT FLUSH", "THERMOSTAT", "RADIATOR", "COOLANT"],
    }
    try:
        import json
        from pathlib import Path
        taxonomy_path = Path(__file__).resolve().parents[1] / "data" / "reference" / "failure_taxonomy.json"
        if not taxonomy_path.exists():
            return fallback
        taxonomy = json.loads(taxonomy_path.read_text())
        part_list = taxonomy if isinstance(taxonomy, list) else taxonomy.get("parts", [])
        result: dict[str, list[str]] = {}
        for part in part_list:
            ml = part.get("ml_model")
            if ml and ml in _ML_MODEL_TO_KEY:
                key = _ML_MODEL_TO_KEY[ml]
                result.setdefault(key, []).extend(part.get("part_codes", []))
        # Deduplicate and add fallback entries not covered by taxonomy
        for key in result:
            result[key] = list(dict.fromkeys(result[key]))
        for key, codes in fallback.items():
            if key not in result:
                result[key] = codes
        return result
    except Exception:
        return fallback

PART_KEYWORDS: dict[str, list[str]] = _build_part_keywords()

# Model → feature group consumed
_MODEL_GROUP: dict[str, str] = {
    "brake_wear":       "brake",
    "engine_oil":       "engine",
    "hv_battery_soh":   "battery_hv",
    "battery_12v":      "battery_12v",
    "tyre_wear":        "tyre",
    "driver_score":     "driver",
    "fuel_anomaly":     "engine",
}

# Alias model names to PART_KEYWORDS keys
_MODEL_PART_KEY: dict[str, str] = {
    "brake_wear":     "brake_wear",
    "engine_oil":     "engine_oil",
    "hv_battery_soh": "hv_battery",
    "battery_12v":    "battery_12v",
    "tyre_wear":      "tyre_wear",
    "fuel_anomaly":   "engine_oil",
    "driver_score":   "engine_oil",   # no direct service label; reuse oil
}


class TrainingDataBuilder:

    def build(
        self,
        model_name: str,
        service_history_path: str | Path,
        feature_store_dir: str | Path,
        output_path: str | Path,
        train_cutoff: str = "2024-06-01",
        val_cutoff:   str = "2024-09-01",
    ) -> dict[str, pd.DataFrame]:
        """Build train / val / test splits. Returns {"train": df, "val": df, "test": df}."""
        from features.feature_store import FeatureStore

        service_history_path = Path(service_history_path)
        feature_store_dir    = Path(feature_store_dir)
        output_path          = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        tc = pd.Timestamp(train_cutoff).date()
        vc = pd.Timestamp(val_cutoff).date()

        store = FeatureStore(offline_root=feature_store_dir)
        group = _MODEL_GROUP.get(model_name, "brake")

        # ── Step 1: Extract label events from service history ──────────────
        svc = self._load_service_history(service_history_path)
        part_key = _MODEL_PART_KEY.get(model_name, model_name)
        keywords = PART_KEYWORDS.get(part_key, [])

        if svc.empty or not keywords:
            log.warning("No service history or keywords for model %s", model_name)
            label_events = pd.DataFrame(columns=["vin", "service_date"])
        else:
            pattern = "|".join(keywords)
            desc_col = next(
                (c for c in ["DescriptionOne", "description", "service_desc"] if c in svc.columns),
                None,
            )
            if desc_col:
                mask = svc[desc_col].str.contains(pattern, case=False, na=False)
                label_events = svc[mask].copy()
            else:
                label_events = pd.DataFrame(columns=["vin", "service_date"])

        # Ensure date column
        date_col = next((c for c in ["ServiceDate", "service_date", "date"] if c in label_events.columns), None)
        if date_col and date_col != "service_date":
            label_events = label_events.rename(columns={date_col: "service_date"})
        if "service_date" in label_events.columns:
            label_events["service_date"] = pd.to_datetime(label_events["service_date"]).dt.date
        vin_col = next((c for c in ["vin", "VIN"] if c in label_events.columns), None)
        if vin_col and vin_col != "vin":
            label_events = label_events.rename(columns={vin_col: "vin"})

        # ── Steps 2–4: Build positive rows ────────────────────────────────
        pos_rows: list[dict] = []
        for _, ev in label_events.iterrows():
            vin          = str(ev["vin"])
            service_date = ev["service_date"]
            if not isinstance(service_date, date):
                try:
                    service_date = pd.Timestamp(service_date).date()
                except Exception:
                    continue

            feature_cutoff = service_date - timedelta(days=1)   # Step 2
            feats = store.get_offline(vin, group, as_of_date=feature_cutoff)
            if feats is None:
                continue   # Step 3: skip if no feature data
            row = {
                "vin":             vin,
                "feature_cutoff":  feature_cutoff,
                "service_date":    service_date,
                "label_binary":    1,
                "label_days":      (service_date - feature_cutoff).days,
                **feats,
            }
            pos_rows.append(row)

        # ── Step 5: Generate negatives ─────────────────────────────────────
        neg_rows: list[dict] = []
        rng = random.Random(42)

        # Collect all label dates per VIN
        label_dates_by_vin: dict[str, list[date]] = {}
        for row in pos_rows:
            label_dates_by_vin.setdefault(row["vin"], []).append(row["service_date"])

        all_vins = list(label_dates_by_vin.keys())
        for vin in all_vins:
            label_dates = label_dates_by_vin[vin]

            # Date range: 2 years before earliest label to latest label
            earliest = min(label_dates) - timedelta(days=730)
            latest   = max(label_dates)

            attempts = 0
            neg_count = 0
            while neg_count < 5 and attempts < 100:
                attempts += 1
                offset = rng.randint(0, max((latest - earliest).days, 1))
                sample_date = earliest + timedelta(days=offset)
                feature_cutoff = sample_date

                # Not within 60 days before any label event
                too_close = any(
                    0 <= (ld - sample_date).days <= 60 for ld in label_dates
                )
                if too_close:
                    continue

                feats = store.get_offline(vin, group, as_of_date=feature_cutoff)
                if feats is None:
                    continue

                row = {
                    "vin":            vin,
                    "feature_cutoff": feature_cutoff,
                    "service_date":   None,
                    "label_binary":   0,
                    "label_days":     999,
                    **feats,
                }
                neg_rows.append(row)
                neg_count += 1

        all_rows = pos_rows + neg_rows
        if not all_rows:
            log.warning("No training rows generated for model %s", model_name)
            empty = pd.DataFrame()
            for split in ["train", "val", "test"]:
                empty.to_parquet(output_path / f"{model_name}_{split}.parquet", index=False)
            return {"train": empty, "val": empty, "test": empty}

        df = pd.DataFrame(all_rows)
        df["feature_cutoff"] = pd.to_datetime(df["feature_cutoff"]).dt.date

        # ── Step 6: Temporal split ─────────────────────────────────────────
        train_df = df[df["feature_cutoff"] <  tc].copy()
        val_df   = df[(df["feature_cutoff"] >= tc) & (df["feature_cutoff"] < vc)].copy()
        test_df  = df[df["feature_cutoff"] >= vc].copy()

        # ── Step 7: Save ───────────────────────────────────────────────────
        for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            path = output_path / f"{model_name}_{split_name}.parquet"
            split_df.to_parquet(path, index=False)
            log.info("Saved %s: %d rows -> %s", split_name, len(split_df), path)

        summary = {
            "model":       model_name,
            "train_rows":  len(train_df),
            "val_rows":    len(val_df),
            "test_rows":   len(test_df),
            "pos_rows":    len(pos_rows),
            "neg_rows":    len(neg_rows),
            "train_cutoff": str(tc),
            "val_cutoff":   str(vc),
        }
        (output_path / f"{model_name}_split_summary.json").write_text(
            json.dumps(summary, indent=2, default=str)
        )
        log.info("Split summary: %s", summary)
        return {"train": train_df, "val": val_df, "test": test_df}

    # ── Helper ────────────────────────────────────────────────────────────────

    def _load_service_history(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            log.warning("Service history not found: %s", path)
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except Exception as exc:
            log.error("Failed to read service history %s: %s", path, exc)
            return pd.DataFrame()
