"""
AutoPredict ML Alert Engine.

Class MLAlertEngine.evaluate(vin, features_dict) → List[Alert]

Uses ModelRegistry.predict_all(vin) to get live predictions from all
trained models, then converts each prediction to an Alert based on
confidence thresholds.

Threshold matrix:
  remaining_life_pct < 30% → MEDIUM
  remaining_life_pct < 20% → HIGH
  remaining_life_pct < 10% → CRITICAL

  brake_replacement_prob_30d > 0.80 → HIGH; > 0.50 → MEDIUM
  oil_change_prob_14d        > 0.80 → HIGH; > 0.50 → MEDIUM
  soh_below_80_within_90d    > 0.70 → MEDIUM
  no_start_within_7_days     > 0.80 → HIGH;  > 0.50 → MEDIUM
  tyre_replacement_prob_30d  > 0.70 → HIGH;  > 0.40 → MEDIUM
  fuel anomaly_score         < -0.1 → LOW (FUEL_ANOMALY)
  driver score               < 40   → HIGH; < 60  → MEDIUM
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from alerts.rule_engine import Alert

log = logging.getLogger(__name__)

_SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4}

# FailureStage → alert severity mapping
_STAGE_SEVERITY: dict[int, str | None] = {
    5: "CRITICAL",   # CRITICAL
    4: "HIGH",       # IMMINENT
    3: "MEDIUM",     # HIGH_RISK
    2: "LOW",        # NOTICEABLE
    1: "INFO",       # EARLY
    0: None,         # HEALTHY — no alert
}


class MLAlertEngine:
    """
    Translates trained ML model predictions into Alert objects.

    Usage:
        engine = MLAlertEngine()
        alerts = engine.evaluate(vin="MZ7X...", features_dict={})
    """

    def __init__(self) -> None:
        self._registry = None

    def _get_registry(self):
        if self._registry is None:
            try:
                from models.model_registry import ModelRegistry
                self._registry = ModelRegistry()
            except Exception as exc:
                log.warning("ModelRegistry unavailable: %s", exc)
        return self._registry

    def evaluate(self, vin: str, features_dict: dict | None = None) -> list[Alert]:
        """
        Run all models for *vin* and return generated alerts, sorted by severity.

        *features_dict* is currently unused (ModelRegistry fetches features
        from InfluxDB via each model's predict_single). Kept for API symmetry.
        """
        registry = self._get_registry()
        if registry is None:
            return []

        try:
            predictions = registry.predict_all(vin)
        except Exception as exc:
            log.error("predict_all failed for VIN %s: %s", vin, exc)
            return []

        alerts: list[Alert] = []
        snap = json.dumps({"vin": vin, "source": "ml_engine"})

        for model_name, pred in predictions.items():
            if "error" in pred or pred.get("severity") == "error":
                continue
            raw = pred.get("raw") or {}
            try:
                expl_text, expl_top3 = _get_explanation(vin, model_name)
                new = _convert(vin, model_name, pred, raw, snap,
                               explanation_text=expl_text, top3_features=expl_top3)
                alerts.extend(new)
            except Exception as exc:
                log.debug("Alert conversion failed for %s/%s: %s", vin, model_name, exc)

        alerts.sort(key=lambda a: _SEVERITY_ORDER.get(a.severity, 99))
        return alerts

    def evaluate_with_stages(
        self,
        vin: str,
        features_dict: dict | None = None,
        rule_flags: dict | None = None,
        feature_store: Any | None = None,
    ) -> "tuple[list[Alert], dict]":
        """
        Run ensemble predict + FailureStageClassifier and return
        (alerts, {failure_type: FailureStage}).

        Stage-to-severity mapping:
          CRITICAL → "CRITICAL"   IMMINENT  → "HIGH"
          HIGH_RISK → "MEDIUM"   NOTICEABLE → "LOW"
          EARLY     → "INFO"     HEALTHY    → no alert
        """
        try:
            from models.failure_stage_classifier import FailureStageClassifier, FailureStage
        except Exception as exc:
            log.warning("FailureStageClassifier unavailable: %s", exc)
            return self.evaluate(vin, features_dict), {}

        registry = self._get_registry()
        if registry is None:
            return [], {}

        try:
            ensemble_probs = registry.predict_ensemble(vin, feature_store)
        except Exception as exc:
            log.error("predict_ensemble failed for VIN %s: %s", vin, exc)
            return [], {}

        clf = FailureStageClassifier()
        stages = clf.classify_all(
            vin,
            ensemble_probs,
            rul_dict={},
            rule_flags=rule_flags or {},
            features=features_dict or {},
        )

        alerts: list[Alert] = []
        snap = json.dumps({"vin": vin, "source": "ml_stage_engine"})

        for ft, stage in stages.items():
            severity = _STAGE_SEVERITY.get(int(stage))
            if severity is None:
                continue
            prob = ensemble_probs.get(ft, 0.0)
            label = clf.stage_label(stage)
            alerts.append(_make(
                vin=vin,
                alert_type=f"STAGE_{ft.upper()}",
                severity=severity,
                title=f"[{stage.name}] {ft.replace('_', ' ').title()} — {label}",
                msg_customer=label,
                msg_dealer=f"{ft} stage={stage.name} prob={prob:.2f}",
                action=_stage_action(ft, stage),
                cost_min=0,
                cost_max=0,
                confidence=prob,
                snap=snap,
            ))

        alerts.sort(key=lambda a: _SEVERITY_ORDER.get(a.severity, 99))
        return alerts, {k: int(v) for k, v in stages.items()}


def _stage_action(failure_type: str, stage: Any) -> str:
    _ACTIONS: dict[str, list[str]] = {
        "brake":       ["Monitor brake wear", "Inspect brakes soon", "Pre-order pads", "Schedule within 30d", "Book within 2 weeks", "STOP — inspect immediately"],
        "oil":         ["Normal interval", "Monitor oil life", "Plan oil change", "Oil change within 30d", "Oil change this week", "STOP — oil pressure fault"],
        "hv_battery":  ["Normal", "Monitor SoH", "Schedule HV check", "Battery assessment within 30d", "Battery service within 2 weeks", "STOP — battery fault"],
        "12v_battery": ["Normal", "Monitor voltage", "Battery test recommended", "Battery test within 30d", "Replace battery within 2 weeks", "STOP — no-start imminent"],
        "tyre":        ["Normal", "Monitor pressures", "Tyre inspection", "Schedule tyre service within 30d", "Replace tyres within 2 weeks", "STOP — tyre failure risk"],
        "overheating": ["Normal", "Monitor coolant", "Coolant system check", "Service within 30d", "Service this week", "STOP — overheating risk"],
    }
    actions = _ACTIONS.get(failure_type, ["Monitor", "Monitor", "Check", "Schedule service", "Urgent service", "STOP"])
    return actions[min(int(stage), len(actions) - 1)]


# ── Per-model alert conversion ─────────────────────────────────────────────────

def _make(
    vin: str,
    alert_type: str,
    severity: str,
    title: str,
    msg_customer: str,
    msg_dealer: str,
    action: str,
    cost_min: float,
    cost_max: float,
    confidence: float,
    snap: str,
    explanation_text: str = "",
    top3_features: list | None = None,
    rul_days_median: float | None = None,
    rul_days_p10:    float | None = None,
) -> Alert:
    # Augment snap JSON with explanation and RUL data
    try:
        snap_dict = json.loads(snap) if snap else {}
        if explanation_text:
            snap_dict["explanation_text"] = explanation_text
        if top3_features:
            snap_dict["top3_features"] = top3_features
        if rul_days_median is not None:
            snap_dict["rul_days_median"] = round(rul_days_median, 1)
        if rul_days_p10 is not None:
            snap_dict["rul_days_p10"] = round(rul_days_p10, 1)
        snap = json.dumps(snap_dict)
    except Exception:
        pass

    return Alert(
        vin=vin,
        alert_type=alert_type,
        severity=severity,
        title=title,
        message_customer=msg_customer,
        message_dealer=msg_dealer,
        recommended_action=action,
        estimated_cost_min=cost_min,
        estimated_cost_max=cost_max,
        confidence_score=round(confidence, 3),
        model_version="ml/1.0",
        triggered_at=datetime.now(timezone.utc),
        data_snapshot_json=snap,
    )


def _get_rul(vin: str, failure_type: str) -> tuple[float | None, float | None]:
    """Try to fetch rul_days_median and rul_days_p10 for an alert. Non-blocking."""
    try:
        from models.model_registry import ModelRegistry
        rul = ModelRegistry().predict_rul(vin, failure_type)
        if rul is not None:
            return rul.rul_days_median, rul.rul_days_p10
    except Exception:
        pass
    return None, None


def _get_explanation(vin: str, model_name: str) -> tuple[str, list]:
    """Try to get SHAP explanation for a model prediction. Non-blocking."""
    try:
        from models.explainability import try_explain
        from models.model_registry import _MODEL_SPECS
        import importlib

        spec = _MODEL_SPECS.get(model_name, {})
        mod = importlib.import_module(spec.get("module", ""))
        if hasattr(mod, "get_explainability_artifacts"):
            clf, fvec, fnames = mod.get_explainability_artifacts(vin)
            result = try_explain(clf, fnames, fvec, model_name)
            return result.nl_summary, result.top3
    except Exception:
        pass
    return "", []


def _convert(
    vin: str,
    model_name: str,
    pred: dict,
    raw: dict,
    snap: str,
    explanation_text: str = "",
    top3_features: list | None = None,
) -> list[Alert]:
    alerts: list[Alert] = []
    _expl = {"explanation_text": explanation_text, "top3_features": top3_features}

    # ── brake_wear ────────────────────────────────────────────────────────────
    if model_name == "brake_wear":
        prob   = float(raw.get("replacement_prob_30d", 0))
        days   = raw.get("days_to_replacement_predicted")
        days_s = f"{days:.0f}" if days is not None else "unknown"
        rul_med, rul_p10 = _get_rul(vin, "brake")

        if prob > 0.80:
            alerts.append(_make(vin, "ML_BRAKE_REPLACEMENT", "HIGH",
                "ML: Brake Replacement Required Soon",
                f"Our system predicts your brake pads need replacement in ~{days_s} days. Book a service soon.",
                f"Brake replacement probability {prob:.0%} (30-day window). Predicted in {days_s} days.",
                "Book brake service within 2 weeks.",
                3_000, 15_000, prob, snap,
                rul_days_median=rul_med, rul_days_p10=rul_p10, **_expl))
        elif prob > 0.50:
            alerts.append(_make(vin, "ML_BRAKE_WARNING", "MEDIUM",
                "ML: Brake Wear Advisory",
                f"Brake wear is accelerating. We estimate ~{days_s} days until replacement is needed.",
                f"Brake replacement probability {prob:.0%}. Monitor at next service.",
                "Inspect brakes at next service visit.",
                3_000, 12_000, prob, snap,
                rul_days_median=rul_med, rul_days_p10=rul_p10, **_expl))

    # ── engine_oil ────────────────────────────────────────────────────────────
    elif model_name == "engine_oil":
        prob = float(raw.get("oil_change_prob_14d", 0))
        km   = raw.get("km_to_oil_change")
        km_s = f"{km:.0f}" if km is not None else "unknown"
        rul_med, rul_p10 = _get_rul(vin, "oil")

        if prob > 0.80:
            alerts.append(_make(vin, "ML_OIL_CHANGE_DUE", "HIGH",
                "ML: Oil Change Due Immediately",
                f"Engine oil is highly degraded. Change required within ~{km_s} km.",
                f"Oil change probability {prob:.0%} (14-day window). ~{km_s} km remaining.",
                "Schedule oil change this week.",
                2_000, 6_000, prob, snap,
                rul_days_median=rul_med, rul_days_p10=rul_p10, **_expl))
        elif prob > 0.50:
            alerts.append(_make(vin, "ML_OIL_ADVISORY", "MEDIUM",
                "ML: Oil Change Due Soon",
                f"Engine oil is degrading. An oil change will be needed in approximately {km_s} km.",
                f"Oil degradation index: {raw.get('oil_degradation_index', 'N/A')}.",
                "Schedule oil change within 30 days.",
                2_000, 5_000, prob, snap,
                rul_days_median=rul_med, rul_days_p10=rul_p10, **_expl))

    # ── hv_battery_soh ────────────────────────────────────────────────────────
    elif model_name == "hv_battery_soh":
        prob_80  = float(raw.get("prob_soh_below_80_90d", 0))
        soh      = raw.get("predicted_soh_pct")
        soh_s    = f"{soh:.1f}" if soh is not None else "N/A"
        cell_anom = int(raw.get("cell_anomaly_detected", 0))
        arima_min = raw.get("arima_soh_90d_forecast_min")

        if prob_80 > 0.70:
            alerts.append(_make(vin, "ML_HV_SOH_DECLINE", "MEDIUM",
                f"ML: HV Battery Health Declining (SoH {soh_s}%)",
                f"Battery health at {soh_s}%. Our model predicts it may fall below 80% within 90 days.",
                f"SoH={soh_s}%, prob(below 80% in 90d)={prob_80:.0%}.",
                "Schedule battery assessment. Review warranty.",
                10_000, 2_00_000, prob_80, snap, **_expl))

        if cell_anom == 1:
            alerts.append(_make(vin, "ML_CELL_ANOMALY", "HIGH",
                "ML: Battery Cell Anomaly Detected",
                "Our system detected an unusual pattern in battery cell behaviour. A health check is recommended.",
                f"Isolation Forest flagged cell anomaly. Cell spread and temp are outside normal bounds.",
                "Schedule battery diagnostic immediately.",
                8_000, 80_000, 0.75, snap, **_expl))

        if arima_min is not None and arima_min < 80:
            alerts.append(_make(vin, "ML_ARIMA_SOH_FORECAST", "MEDIUM",
                f"ML: Battery Health Forecast Below 80% (min {arima_min:.0f}%)",
                f"Trend analysis suggests battery health could reach {arima_min:.0f}% within 90 days.",
                f"ARIMA(2,1,1) 90-day forecast min SoH: {arima_min:.1f}%.",
                "Review battery health trend at next service.",
                5_000, 1_00_000, 0.65, snap, **_expl))

    # ── battery_12v ───────────────────────────────────────────────────────────
    elif model_name == "battery_12v":
        prob = float(raw.get("no_start_probability", 0))
        days = raw.get("days_to_12v_failure")
        days_s = f"{days:.0f}" if days is not None else "unknown"
        rul_med, rul_p10 = _get_rul(vin, "12v_battery")

        if prob > 0.80:
            alerts.append(_make(vin, "ML_12V_FAILURE_RISK", "HIGH",
                "ML: High No-Start Risk (12V Battery)",
                f"Our model predicts a high risk of no-start within ~{days_s} days. Test battery now.",
                f"No-start probability: {prob:.0%}. Estimated {days_s} days to failure.",
                "Have 12V battery tested at nearest service centre.",
                4_000, 12_000, prob, snap,
                rul_days_median=rul_med, rul_days_p10=rul_p10, **_expl))
        elif prob > 0.50:
            alerts.append(_make(vin, "ML_12V_ADVISORY", "MEDIUM",
                "ML: 12V Battery Health Advisory",
                "12V battery health is declining. A battery test is recommended at your next service.",
                f"No-start probability: {prob:.0%}.",
                "Test battery at next service.",
                4_000, 10_000, prob, snap,
                rul_days_median=rul_med, rul_days_p10=rul_p10, **_expl))

    # ── tyre_wear ─────────────────────────────────────────────────────────────
    elif model_name == "tyre_wear":
        prob     = float(raw.get("tyre_replacement_prob_30d", 0))
        km       = raw.get("km_to_tyre_replacement")
        km_s     = f"{km:.0f}" if km is not None else "unknown"
        puncture = int(raw.get("puncture_alert", 0))
        rul_med, rul_p10 = _get_rul(vin, "tyre")

        if puncture:
            alerts.append(_make(vin, "ML_PUNCTURE_DETECTED", "CRITICAL",
                "ML: Rapid Tyre Pressure Loss — Possible Puncture",
                "Tyre pressure is dropping rapidly. Reduce speed and check for a puncture immediately.",
                "Pressure drop rate exceeds 5 kPa/day threshold. Immediate inspection required.",
                "Slow to <60 km/h. Find safe place to stop and inspect tyres.",
                500, 6_000, 0.9, snap, **_expl))
        elif prob > 0.70:
            alerts.append(_make(vin, "ML_TYRE_REPLACEMENT", "HIGH",
                f"ML: Tyre Replacement Needed (~{km_s} km remaining)",
                f"Our model predicts tyre replacement needed in approximately {km_s} km.",
                f"Tyre replacement probability {prob:.0%} (30-day window).",
                "Book tyre inspection and replacement within 2 weeks.",
                5_000, 20_000, prob, snap,
                rul_days_median=rul_med, rul_days_p10=rul_p10, **_expl))
        elif prob > 0.40:
            alerts.append(_make(vin, "ML_TYRE_ADVISORY", "MEDIUM",
                "ML: Tyre Wear Advisory",
                f"Tyre wear is progressing. Expect to need replacement in ~{km_s} km.",
                f"Tyre replacement probability: {prob:.0%}.",
                "Inspect tyres at next service.",
                5_000, 18_000, prob, snap,
                rul_days_median=rul_med, rul_days_p10=rul_p10, **_expl))

    # ── fuel_anomaly ──────────────────────────────────────────────────────────
    elif model_name == "fuel_anomaly":
        score    = float(raw.get("anomaly_score", 0))
        is_anom  = int(raw.get("fuel_anomaly", 0))
        anom_prob = float(raw.get("anomaly_probability", 0))

        if is_anom and score < -0.1:
            severity = "MEDIUM" if anom_prob > 0.5 else "LOW"
            alerts.append(_make(vin, "ML_FUEL_ANOMALY", severity,
                "ML: Unusual Fuel Consumption Pattern",
                "Our system detected an unusual fuel consumption pattern. This could indicate a sensor issue or driving condition change.",
                f"Isolation Forest anomaly score: {score:.4f}. Probability: {anom_prob:.0%}.",
                "Check for fuel leaks. Review driving conditions. Schedule diagnostic if persistent.",
                500, 10_000, anom_prob, snap, **_expl))

    # ── driver_score ──────────────────────────────────────────────────────────
    elif model_name == "driver_score":
        score      = float(raw.get("composite_drive_score", raw.get("driver_score", 75)))
        risk_prob  = float(raw.get("high_risk_probability", 0))

        if score < 40 or risk_prob > 0.70:
            alerts.append(_make(vin, "ML_DRIVER_HIGH_RISK", "HIGH",
                f"ML: High-Risk Driving Behaviour (Score: {score:.0f}/100)",
                f"Your driving score is {score:.0f}/100, indicating high-risk patterns. Consider safe driving tips.",
                f"Driver score {score:.0f}. High-risk prob {risk_prob:.0%}. Consider fleet coaching.",
                "Review safe driving guidelines. Fleet coaching recommended.",
                0, 0, risk_prob, snap, **_expl))
        elif score < 60 or risk_prob > 0.40:
            alerts.append(_make(vin, "ML_DRIVER_ADVISORY", "MEDIUM",
                f"ML: Driving Score Below Average (Score: {score:.0f}/100)",
                f"Driving score of {score:.0f}/100. Some improvement areas identified.",
                f"Driver score {score:.0f}. Moderate risk indicators.",
                "Review driving habits. Consider eco-driving mode.",
                0, 0, risk_prob, snap, **_expl))

    return alerts
