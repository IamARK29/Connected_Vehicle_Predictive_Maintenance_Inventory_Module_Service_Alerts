"""
Persistent training metrics store.

Writes and reads a single JSON file: models/saved/model_metrics.json
Every model's train() call should invoke save_model_metrics() after training.
The OEM API reads from this file — no hardcoded values allowed.

Schema per model entry:
{
  "model_name": str,
  "algorithm": str,
  "target": str,
  "training_samples": int,
  "feature_names": list[str],
  "feature_importances": dict[str, float],   # name -> importance (0-1, sums to ~1)
  "metrics": dict[str, float],               # e.g. cv_rmse, cv_auc, cox_concordance_index
  "trained_at": ISO8601 str,
  "status": "trained" | "skipped" | "failed",
  "notes": str
}
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_METRICS_PATH = Path("models/saved/model_metrics.json")


def _load_store() -> dict[str, Any]:
    if _METRICS_PATH.exists():
        try:
            return json.loads(_METRICS_PATH.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("Could not read model_metrics.json: %s", exc)
    return {}


def _write_store(store: dict[str, Any]) -> None:
    _METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _METRICS_PATH.write_text(json.dumps(store, indent=2, default=str), encoding="utf-8")


def save_model_metrics(
    model_name: str,
    algorithm: str,
    target: str,
    training_samples: int,
    feature_names: list[str],
    metrics: dict[str, float],
    feature_importances: dict[str, float] | None = None,
    status: str = "trained",
    notes: str = "",
) -> None:
    """
    Persist training metrics for one model.
    Merges into existing store — other models' entries are preserved.
    """
    store = _load_store()

    # Normalise importances to sum to 1
    fi = feature_importances or {}
    total = sum(fi.values())
    if total > 0:
        fi = {k: round(v / total, 6) for k, v in fi.items()}

    # Sort by importance descending, keep top 15
    fi = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True)[:15])

    store[model_name] = {
        "model_name":          model_name,
        "algorithm":           algorithm,
        "target":              target,
        "training_samples":    training_samples,
        "feature_names":       feature_names[:30],    # cap for readability
        "feature_importances": fi,
        "metrics":             {k: round(float(v), 6) for k, v in metrics.items()
                                if v is not None and not (isinstance(v, float) and __import__("math").isnan(v))},
        "trained_at":          datetime.now(timezone.utc).isoformat(),
        "status":              status,
        "notes":               notes,
    }

    _write_store(store)
    log.info("Saved metrics for %s to %s", model_name, _METRICS_PATH)


def load_all_metrics() -> dict[str, dict]:
    """Return all persisted model metrics keyed by model_name."""
    return _load_store()


def load_model_metrics(model_name: str) -> dict | None:
    """Return metrics for a single model, or None if not found."""
    return _load_store().get(model_name)


def mark_model_failed(model_name: str, error: str) -> None:
    store = _load_store()
    store[model_name] = {
        "model_name":  model_name,
        "status":      "failed",
        "error":       str(error),
        "trained_at":  datetime.now(timezone.utc).isoformat(),
    }
    _write_store(store)


def mark_model_skipped(model_name: str, reason: str) -> None:
    store = _load_store()
    existing = store.get(model_name, {})
    existing.update({
        "model_name": model_name,
        "status":     "skipped",
        "skip_reason": reason,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    })
    store[model_name] = existing
    _write_store(store)


# ── RUL model concordance extraction ──────────────────────────────────────────

def extract_rul_concordance(model_name: str, save_dir: Path | None = None) -> float | None:
    """
    Load a trained RUL model (WeibullAFT or CoxPH) from disk and
    return its concordance_index_ attribute, or None if unavailable.
    """
    from pathlib import Path as _Path
    save_dir = save_dir or _Path("models/saved")
    path = save_dir / f"{model_name}.joblib"
    if not path.exists():
        return None
    try:
        import joblib
        state = joblib.load(path)
        model_obj = state.get("model") if isinstance(state, dict) else state
        if model_obj is None:
            return None
        fitted = state.get("fitted", True) if isinstance(state, dict) else True
        if not fitted:
            return None
        ci = getattr(model_obj, "concordance_index_", None)
        if ci is not None:
            return round(float(ci), 4)
    except Exception as exc:
        log.debug("Could not extract concordance from %s: %s", model_name, exc)
    return None
