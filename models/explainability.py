"""
Model explainability via SHAP TreeExplainer.

ExplanationResult  — dataclass holding SHAP values, top-3 drivers, and NL summary.
ModelExplainer     — wraps a tree model; produces ExplanationResult per feature vector.
FEATURE_LABELS     — human-readable names for every feature-store field.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# ── Human-readable feature labels ─────────────────────────────────────────────

FEATURE_LABELS: dict[str, str] = {
    "brake_stress_cumulative":       "cumulative brake stress since last service",
    "harsh_brake_rate_30d":          "harsh braking events per 100km (30 days)",
    "harsh_brake_rate_7d":           "harsh braking events per 100km (7 days)",
    "high_speed_stop_count_30d":     "emergency stops from high speed",
    "km_since_last_brake_service":   "km driven since last brake service",
    "brake_thermal_stress":          "cumulative brake thermal stress",
    "regen_fraction":                "regenerative braking fraction (EV)",
    "oil_degradation_index":         "engine oil degradation index (0-100)",
    "cold_start_count_30d":          "cold engine starts in 30 days",
    "km_since_oil_change":           "km since last oil change",
    "coolant_overtemp_count_30d":    "coolant temperature exceedances",
    "fuel_consumption_deviation_pct": "fuel consumption above baseline (%)",
    "soh_estimated":                 "battery state of health (%)",
    "cell_voltage_spread":           "battery cell voltage imbalance (V)",
    "soh_trend_slope_90d":           "battery health decline rate (90 days)",
    "dc_charge_fraction_30d":        "DC fast charge fraction",
    "isolation_resistance_min_30d":  "battery isolation resistance (kOhm)",
    "resting_voltage_7d_avg":        "12V battery resting voltage (V)",
    "resting_voltage_trend_14d":     "12V battery voltage trend (V/day)",
    "cranking_voltage_dip_avg":      "12V voltage during engine start (V)",
    "pressure_fl_7d_avg":            "front-left tyre pressure (kPa)",
    "pressure_drop_rate_fl":         "front-left tyre pressure decline (kPa/day)",
    "tyre_stress_cumulative":        "cumulative tyre stress index",
    "axle_imbalance_front":          "front axle pressure imbalance (kPa)",
    # Additional common features
    "km_since_last_tyre_service":    "km driven since last tyre service",
    "tyre_pressure_min_7d":          "minimum tyre pressure (7 days)",
    "composite_drive_score":         "composite driver behaviour score (0-100)",
    "peer_percentile":               "driver score percentile vs fleet",
    "soc_mean_7d":                   "battery state of charge (7-day avg %)",
    "soh_mean":                      "battery state of health (avg %)",
    "resting_voltage_trend_7d":      "12V battery voltage trend (7 days)",
    "battery_12v_health_score":      "12V battery composite health score",
    "overnight_drop_avg_7d":         "overnight 12V voltage drop (avg V)",
    "idle_hours_30d":                "engine idle hours (30 days)",
    "short_trip_fraction_30d":       "fraction of trips under 8km (30 days)",
    "avg_speed_30d":                 "average driving speed (30 days, km/h)",
    "high_rpm_stress_index":         "high-RPM engine stress index",
}


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class ExplanationResult:
    shap_values:  list[float]
    top3:         list[dict]   # [{feature, shap_value, direction, contribution_pct}]
    nl_summary:   str

    def to_dict(self) -> dict:
        return {
            "shap_values": self.shap_values,
            "top3":        self.top3,
            "nl_summary":  self.nl_summary,
        }


_EMPTY_EXPLANATION = ExplanationResult(shap_values=[], top3=[], nl_summary="Explanation unavailable.")


# ── Explainer class ────────────────────────────────────────────────────────────

class ModelExplainer:
    """SHAP TreeExplainer wrapper for XGBoost / LightGBM models."""

    def __init__(
        self,
        model: Any,
        feature_names: list[str],
        model_name: str = "",
    ) -> None:
        self.feature_names = feature_names
        self.model_name    = model_name
        self._explainer    = None
        try:
            import shap
            self._explainer = shap.TreeExplainer(model)
        except Exception as exc:
            log.warning("SHAP TreeExplainer init failed for %s: %s", model_name, exc)

    def explain(self, feature_vector: np.ndarray) -> ExplanationResult:
        """
        Compute SHAP values for *feature_vector* and return an ExplanationResult.

        feature_vector: 1-D array of shape (n_features,)
        """
        if self._explainer is None:
            return _EMPTY_EXPLANATION

        try:
            raw = self._explainer.shap_values(feature_vector.reshape(1, -1))
            # TreeExplainer on classifiers may return list[array] (one per class)
            if isinstance(raw, list):
                shap_vals = raw[1][0] if len(raw) > 1 else raw[0][0]
            else:
                shap_vals = raw[0]

            top3_idx  = np.argsort(np.abs(shap_vals))[-3:][::-1]
            total_abs = float(np.abs(shap_vals).sum()) + 1e-9

            top3 = []
            for i in top3_idx:
                sv  = float(shap_vals[i])
                top3.append({
                    "feature":          self.feature_names[i],
                    "shap_value":       round(sv, 6),
                    "direction":        "higher" if sv > 0 else "lower",
                    "contribution_pct": round(abs(sv) / total_abs * 100, 1),
                })

            parts = ["Risk elevated due to:"]
            for item in top3:
                label = FEATURE_LABELS.get(item["feature"], item["feature"].replace("_", " "))
                adj   = "elevated" if item["direction"] == "higher" else "reduced"
                parts.append(f"  • {label} is {adj} ({item['contribution_pct']}% of risk)")

            return ExplanationResult(
                shap_values=shap_vals.tolist(),
                top3=top3,
                nl_summary="\n".join(parts),
            )

        except Exception as exc:
            log.debug("SHAP explain failed for %s: %s", self.model_name, exc)
            return _EMPTY_EXPLANATION


# ── Registry-level helper ──────────────────────────────────────────────────────

def try_explain(
    model: Any,
    feature_names: list[str],
    feature_vector: np.ndarray | None,
    model_name: str = "",
) -> ExplanationResult:
    """
    One-shot helper: build a ModelExplainer and call explain().
    Returns _EMPTY_EXPLANATION on any failure.
    """
    if feature_vector is None or len(feature_vector) == 0:
        return _EMPTY_EXPLANATION
    explainer = ModelExplainer(model, feature_names, model_name)
    return explainer.explain(np.asarray(feature_vector, dtype=float))
