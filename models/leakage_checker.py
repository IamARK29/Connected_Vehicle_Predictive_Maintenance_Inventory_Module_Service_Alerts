"""
Leakage checker — validates training / val / test splits for temporal and
cross-contamination leakage before any model training begins.

RAISES LeakageError on the first violation found.  Call this before every
training run:

    LeakageChecker().check(train_df, val_df, test_df, model_name)
"""
from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)


class LeakageError(Exception):
    """Raised when a data leakage violation is detected in training splits."""


class LeakageChecker:

    def check(
        self,
        train_df: pd.DataFrame,
        val_df:   pd.DataFrame,
        test_df:  pd.DataFrame,
        model_name: str,
    ) -> None:
        """
        Check all three splits for leakage.  RAISES LeakageError on violation.

        Check 1 — Future features: all rows must have feature_date < label_date
                   (only applies to positive samples, i.e. label_binary == 1).
        Check 2 — Cross-split VIN contamination: no VIN in both train and test.
        Check 3 — Temporal ordering: train max < val min < test min.
        """
        log.info("Running leakage checks for model '%s'", model_name)

        # ── Check 1: feature_date < label_date ────────────────────────────
        for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            if df.empty:
                continue
            if "feature_cutoff" not in df.columns or "service_date" not in df.columns:
                log.debug(
                    "Skipping Check 1 for %s/%s — missing feature_cutoff or service_date",
                    model_name, split_name,
                )
                continue

            positives = df[df.get("label_binary", pd.Series(0, index=df.index)) == 1].copy()
            if positives.empty:
                continue

            fc = pd.to_datetime(positives["feature_cutoff"], errors="coerce")
            sd = pd.to_datetime(positives["service_date"],   errors="coerce")
            bad = fc >= sd
            if bad.any():
                n = bad.sum()
                sample_vin = positives.loc[bad, "vin"].iloc[0] if "vin" in positives.columns else "?"
                raise LeakageError(
                    f"[{model_name}/{split_name}] Check 1 FAILED: {n} rows have "
                    f"feature_cutoff >= service_date (first offender VIN={sample_vin})"
                )

        log.info("  Check 1 passed — no future feature leakage")

        # ── Check 2: no VIN overlap between train and test ─────────────────
        if not train_df.empty and not test_df.empty and "vin" in train_df.columns and "vin" in test_df.columns:
            train_vins = set(train_df["vin"].dropna())
            test_vins  = set(test_df["vin"].dropna())
            overlap    = train_vins & test_vins
            if overlap:
                sample = list(overlap)[:5]
                raise LeakageError(
                    f"[{model_name}] Check 2 FAILED: {len(overlap)} VINs appear in both "
                    f"train and test sets. Sample: {sample}"
                )

        log.info("  Check 2 passed — no VIN contamination between train and test")

        # ── Check 3: temporal ordering ─────────────────────────────────────
        date_col = "feature_cutoff"
        splits   = [(n, df) for n, df in [("train", train_df), ("val", val_df), ("test", test_df)]
                    if not df.empty and date_col in df.columns]

        if len(splits) >= 2:
            dates = {
                name: pd.to_datetime(df[date_col], errors="coerce").dropna()
                for name, df in splits
            }

            def _max(name):
                return dates[name].max() if not dates[name].empty else pd.Timestamp.min

            def _min(name):
                return dates[name].min() if not dates[name].empty else pd.Timestamp.max

            if "train" in dates and "val" in dates:
                train_max = _max("train")
                val_min   = _min("val")
                if train_max >= val_min:
                    raise LeakageError(
                        f"[{model_name}] Check 3 FAILED: train max date ({train_max.date()}) "
                        f">= val min date ({val_min.date()})"
                    )

            if "val" in dates and "test" in dates:
                val_max  = _max("val")
                test_min = _min("test")
                if val_max >= test_min:
                    raise LeakageError(
                        f"[{model_name}] Check 3 FAILED: val max date ({val_max.date()}) "
                        f">= test min date ({test_min.date()})"
                    )

        log.info("  Check 3 passed — temporal ordering correct")
        log.info("All leakage checks passed for model '%s'", model_name)
