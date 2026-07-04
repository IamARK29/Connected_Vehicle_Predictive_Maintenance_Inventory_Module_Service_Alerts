"""
Monitoring & A/B testing endpoints.

GET /api/monitoring/ab-results   → per-model champion vs challenger AUC comparison
GET /api/monitoring/drift        → latest drift report per model
GET /api/monitoring/model-status → which models are trained and healthy
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import get_current_user

log = logging.getLogger(__name__)

router = APIRouter(prefix="/monitoring", tags=["Monitoring"])


# ── A/B results ───────────────────────────────────────────────────────────────

@router.get(
    "/ab-results",
    response_model=list[dict],
    summary="Champion vs challenger AUC comparison for all A/B experiments",
    responses={
        200: {
            "description": "Per-model A/B summary",
            "content": {"application/json": {"example": [{
                "model_name":       "brake_wear",
                "champion_auc":     0.87,
                "challenger_auc":   0.89,
                "traffic_split":    0.05,
                "recommendation":   "promote_challenger",
                "experiment_start": "2026-06-01",
            }]}},
        }
    },
)
async def get_ab_results(
    current_user: Annotated[dict, Depends(get_current_user)],
    model_name: str | None = Query(None, description="Filter to single model"),
):
    """
    Returns champion vs challenger AUC for every running A/B experiment.

    recommendation values:
      promote_challenger — challenger AUC exceeds champion by ≥ 2%
      keep_champion      — challenger did not beat champion
      insufficient_data  — not enough logged outcomes to judge
      no_experiment      — no challenger is configured
    """
    results = _query_ab_results(model_name)
    if not results:
        # Fall back to registry-level configs when DB is empty
        results = _ab_results_from_registry(model_name)
    return results


@router.get(
    "/drift",
    response_model=list[dict],
    summary="Latest drift report per model",
)
async def get_drift_reports(
    current_user: Annotated[dict, Depends(get_current_user)],
    model_name:  str | None = Query(None),
    days_back:   int        = Query(7, ge=1, le=90),
):
    return _query_drift_reports(model_name, days_back)


@router.get(
    "/model-status",
    response_model=list[dict],
    summary="Trained model status and metadata",
)
async def get_model_status(
    current_user: Annotated[dict, Depends(get_current_user)],
):
    try:
        from models.model_registry import ModelRegistry
        registry = ModelRegistry()
        return registry.get_model_metadata()
    except Exception as exc:
        log.error("Model status query failed: %s", exc)
        return []


# ── Internal helpers ──────────────────────────────────────────────────────────

def _query_ab_results(model_name_filter: str | None) -> list[dict]:
    """Query ab_experiment_log for per-model champion vs challenger outcome stats."""
    try:
        import psycopg2
        conn = psycopg2.connect(os.getenv("DATABASE_URL", ""))
        cur  = conn.cursor()
        where = "WHERE model_name = %s" if model_name_filter else ""
        args  = (model_name_filter,) if model_name_filter else ()
        cur.execute(f"""
            SELECT
                model_name,
                variant,
                AVG(outcome_correct::int) AS accuracy,
                COUNT(*) FILTER (WHERE outcome_correct IS NOT NULL) AS labelled_count,
                COUNT(*) AS total_count,
                MIN(prediction_date) AS experiment_start
            FROM ab_experiment_log
            {where}
            GROUP BY model_name, variant
            ORDER BY model_name, variant
        """, args)
        rows = cur.fetchall()
        conn.close()

        # Pivot to per-model rows
        by_model: dict[str, dict] = {}
        for model, variant, acc, n_labelled, n_total, exp_start in rows:
            if model not in by_model:
                by_model[model] = {
                    "model_name":       model,
                    "champion_auc":     None,
                    "challenger_auc":   None,
                    "traffic_split":    None,
                    "recommendation":   "insufficient_data",
                    "experiment_start": str(exp_start) if exp_start else None,
                }
            if variant == "champion":
                by_model[model]["champion_auc"] = round(float(acc or 0), 4)
            elif variant == "challenger":
                by_model[model]["challenger_auc"] = round(float(acc or 0), 4)
                by_model[model]["n_challenger"]   = int(n_total)

        # Compute recommendation
        for m, row in by_model.items():
            champ = row.get("champion_auc")
            chal  = row.get("challenger_auc")
            if champ is None or chal is None:
                row["recommendation"] = "insufficient_data"
            elif chal > champ + 0.02:
                row["recommendation"] = "promote_challenger"
            else:
                row["recommendation"] = "keep_champion"

        # Add traffic split from registry configs
        try:
            from models.model_registry import _AB_CONFIGS
            for m, row in by_model.items():
                cfg = _AB_CONFIGS.get(m)
                if cfg:
                    row["traffic_split"] = cfg.challenger_traffic_pct
        except Exception:
            pass

        return list(by_model.values())

    except Exception as exc:
        log.debug("A/B DB query failed: %s", exc)
        return []


def _ab_results_from_registry(model_name_filter: str | None) -> list[dict]:
    """Fall back to in-memory ModelServingConfig when no DB rows exist."""
    try:
        from models.model_registry import _AB_CONFIGS
        rows = []
        for name, cfg in _AB_CONFIGS.items():
            if model_name_filter and name != model_name_filter:
                continue
            rows.append({
                "model_name":       name,
                "champion_version": cfg.champion_version,
                "challenger_version": cfg.challenger_version,
                "champion_auc":     None,
                "challenger_auc":   None,
                "traffic_split":    cfg.challenger_traffic_pct,
                "recommendation":   "no_experiment" if cfg.challenger_version is None else "insufficient_data",
                "experiment_start": None,
            })
        return rows
    except Exception:
        return []


def _query_drift_reports(model_name_filter: str | None, days_back: int) -> list[dict]:
    try:
        import psycopg2
        conn = psycopg2.connect(os.getenv("DATABASE_URL", ""))
        cur  = conn.cursor()
        where = "AND model_name = %s" if model_name_filter else ""
        args  = (days_back, model_name_filter) if model_name_filter else (days_back,)
        cur.execute(f"""
            SELECT DISTINCT ON (model_name)
                model_name, report_date, feature_psi_json, action, fpr, fnr, created_at
            FROM drift_reports
            WHERE report_date >= CURRENT_DATE - INTERVAL '%s days'
            {where}
            ORDER BY model_name, report_date DESC
        """, args)
        rows = []
        for model, rdate, psi_json, action, fpr, fnr, created in cur.fetchall():
            rows.append({
                "model_name":      model,
                "report_date":     str(rdate),
                "feature_psi":     json.loads(psi_json) if psi_json else {},
                "action":          action,
                "fpr":             round(float(fpr), 4) if fpr is not None else None,
                "fnr":             round(float(fnr), 4) if fnr is not None else None,
                "created_at":      str(created),
            })
        conn.close()
        return rows
    except Exception as exc:
        log.debug("Drift report DB query failed: %s", exc)
        return []
