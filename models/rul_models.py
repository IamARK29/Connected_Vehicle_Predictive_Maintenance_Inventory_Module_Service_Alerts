"""
Remaining Useful Life (RUL) survival models.

WeibullRULModel  — lifelines.WeibullAFTFitter (parametric, extrapolates well)
CoxPHRULModel    — lifelines.CoxPHFitter     (semi-parametric, flexible covariates)

Both share the same interface:
  train(df, duration_col, event_col, covariate_cols)
  predict(features_dict) -> RULPrediction

Model assignments:
  brake_wear_rul   → WeibullRULModel
  engine_oil_rul   → CoxPHRULModel
  battery_12v_rul  → CoxPHRULModel
  tyre_wear_rul    → WeibullRULModel
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# ── Covariates per model ───────────────────────────────────────────────────────

RUL_MODEL_SPECS: dict[str, dict] = {
    "brake_wear_rul": {
        "class": "WeibullRULModel",
        "covariates": [
            "harsh_brake_rate_30d",           # events per 100km (derived from vehBrakePos + tboxAccelX)
            "km_since_last_brake_service",     # from service history DMS
            "brake_thermal_stress",            # (vehBrakePos × vehSpeed²) cumulative
            "abs_activation_rate_30d",         # from real vehABSF signal
            "downhill_brake_stress",           # from tboxAccelZ × vehBrakePos
            "regen_fraction",                  # EV-only, 0 for ICE
        ],
    },
    "engine_oil_rul": {
        "class": "CoxPHRULModel",
        "covariates": [
            "km_since_oil_change",             # from service history DMS
            "cold_start_count_30d",            # vehSysPwrMod transition at vehCoolantTemp < 40°C
            "high_rpm_stress_index",           # derived from vehRPM
            "oil_degradation_index",           # physics formula from coolant/rpm/km
            "oil_pressure_warning_active",     # real binary from vehOilPressureWarning
            "mil_warning_active",              # real binary from vehMILWarning
        ],
    },
    "battery_12v_rul": {
        "class": "CoxPHRULModel",
        "covariates": [
            "resting_voltage_trend_14d",       # slope of resting vehBatt during park
            "battery_12v_health_score",        # composite: voltage + trend + crank + age
            "cranking_voltage_dip_avg",        # min vehBatt at vehSysPwrMod=3 (Crank)
            "battery_age_years",               # from manufacture_year
            "light_on_engine_off_events_7d",   # from real vehDipLight/vehMainLight signals
        ],
    },
    "tyre_wear_rul": {
        "class": "WeibullRULModel",
        "covariates": [
            "tyre_stress_cumulative",          # harsh brakes + cornering + speed stress
            "km_since_last_tyre_service",      # from service history DMS
            "axle_imbalance_front",            # abs(frontLeft − frontRight pressure)
            "pressure_drop_rate_fl",           # kPa/day trend on front-left corner
            "tpms_deflation_count",            # wheelTyreMonitorStatus == 1 events
            "lateral_g_95th_30d",             # from real tboxAccelY signal
        ],
    },
}

# Maps RUL model name → failure_type key used in FailureStageClassifier
RUL_FAILURE_TYPE_MAP: dict[str, str] = {
    "brake_wear_rul":  "brake",
    "engine_oil_rul":  "oil",
    "battery_12v_rul": "12v_battery",
    "tyre_wear_rul":   "tyre",
}


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class RULPrediction:
    rul_days_median: float
    rul_days_p10:    float        # pessimistic (10th percentile)
    rul_days_p90:    float        # optimistic  (90th percentile)
    rul_km_median:   float        # ≈ rul_days_median × 15 km/day
    survival_30d:    float        # P(survive ≥ 30 days)
    survival_90d:    float        # P(survive ≥ 90 days)
    model_used:      str

    def to_dict(self) -> dict:
        return {
            "rul_days_median": round(self.rul_days_median, 1),
            "rul_days_p10":    round(self.rul_days_p10,    1),
            "rul_days_p90":    round(self.rul_days_p90,    1),
            "rul_km_median":   round(self.rul_km_median,   1),
            "survival_30d":    round(self.survival_30d,    4),
            "survival_90d":    round(self.survival_90d,    4),
            "model_used":      self.model_used,
        }


_DEFAULT_RUL = RULPrediction(
    rul_days_median=365.0,
    rul_days_p10=90.0,
    rul_days_p90=730.0,
    rul_km_median=5475.0,
    survival_30d=0.95,
    survival_90d=0.80,
    model_used="default",
)


# ── Shared survival-function extraction ───────────────────────────────────────

def _extract_rul(sf: Any, model_name: str) -> RULPrediction:
    """
    Extract RULPrediction from a lifelines survival function DataFrame.
    sf: DataFrame with timeline index and a single column of S(t) values.
    """
    try:
        import pandas as pd

        timeline = np.asarray(sf.index, dtype=float)
        s_vals   = sf.iloc[:, 0].values.astype(float)

        # Median: first time S(t) <= 0.5
        median_mask = s_vals <= 0.50
        rul_median  = float(timeline[median_mask][0]) if median_mask.any() else float(timeline[-1])

        # P10 (pessimistic): first time S(t) <= 0.90 (10% chance of failing by this time)
        p10_mask = s_vals <= 0.90
        rul_p10  = float(timeline[p10_mask][0]) if p10_mask.any() else rul_median * 0.4

        # P90 (optimistic): first time S(t) <= 0.10
        p90_mask = s_vals <= 0.10
        rul_p90  = float(timeline[p90_mask][0]) if p90_mask.any() else rul_median * 2.5

        # Survival at 30d and 90d
        t30_idx = np.searchsorted(timeline, 30.0)
        t90_idx = np.searchsorted(timeline, 90.0)
        surv_30 = float(s_vals[min(t30_idx, len(s_vals) - 1)])
        surv_90 = float(s_vals[min(t90_idx, len(s_vals) - 1)])

        return RULPrediction(
            rul_days_median=max(0.0, rul_median),
            rul_days_p10=   max(0.0, rul_p10),
            rul_days_p90=   max(0.0, rul_p90),
            rul_km_median=  max(0.0, rul_median * 15.0),
            survival_30d=   float(np.clip(surv_30, 0.0, 1.0)),
            survival_90d=   float(np.clip(surv_90, 0.0, 1.0)),
            model_used=     model_name,
        )
    except Exception as exc:
        log.warning("RUL extraction failed (%s): %s", model_name, exc)
        return _DEFAULT_RUL


# ── WeibullAFT model ───────────────────────────────────────────────────────────

class WeibullRULModel:
    """
    Parametric RUL model using lifelines.WeibullAFTFitter.

    Advantages: extrapolates beyond observed data; provides full parametric S(t).
    """

    def __init__(self, penalizer: float = 0.01) -> None:
        self._penalizer = penalizer
        self._model     = None
        self._covariates: list[str] = []
        self._fitted    = False

    def train(
        self,
        df: "pd.DataFrame",
        duration_col: str,
        event_col: str,
        covariate_cols: list[str],
    ) -> "WeibullRULModel":
        """Fit on *df* with the given duration/event/covariate columns."""
        try:
            from lifelines import WeibullAFTFitter
            import pandas as pd

            self._covariates = [c for c in covariate_cols if c in df.columns]
            cols = [duration_col, event_col] + self._covariates
            fit_df = df[cols].dropna()
            if len(fit_df) < 10:
                log.warning("WeibullRUL: only %d rows — model may be unreliable", len(fit_df))

            self._model = WeibullAFTFitter(penalizer=self._penalizer)
            self._model.fit(fit_df, duration_col=duration_col, event_col=event_col)
            self._fitted = True
            log.info("WeibullRUL fitted on %d rows, covariates: %s", len(fit_df), self._covariates)
        except Exception as exc:
            log.error("WeibullRUL training failed: %s", exc)
        return self

    def predict(self, features_dict: dict[str, Any]) -> RULPrediction:
        """Return RULPrediction for a single vehicle."""
        if not self._fitted or self._model is None:
            return _DEFAULT_RUL
        try:
            import pandas as pd
            row = {k: float(features_dict.get(k, 0.0) or 0.0) for k in self._covariates}
            df  = pd.DataFrame([row])
            sf  = self._model.predict_survival_function(df)
            return _extract_rul(sf, "WeibullAFT")
        except Exception as exc:
            log.debug("WeibullRUL predict failed: %s", exc)
            return _DEFAULT_RUL

    def save(self, path: str | Path) -> None:
        import joblib
        joblib.dump({"model": self._model, "covariates": self._covariates, "fitted": self._fitted}, path)

    def load(self, path: str | Path) -> "WeibullRULModel":
        import joblib
        state = joblib.load(path)
        self._model      = state["model"]
        self._covariates = state["covariates"]
        self._fitted     = state.get("fitted", True)
        return self


# ── CoxPH model ───────────────────────────────────────────────────────────────

class CoxPHRULModel:
    """
    Semi-parametric RUL model using lifelines.CoxPHFitter.

    Advantages: no distributional assumption; handles time-varying hazard.
    """

    def __init__(self, penalizer: float = 2.0) -> None:
        self._penalizer = penalizer
        self._model     = None
        self._covariates: list[str] = []
        self._fitted    = False

    def train(
        self,
        df: "pd.DataFrame",
        duration_col: str,
        event_col: str,
        covariate_cols: list[str],
    ) -> "CoxPHRULModel":
        try:
            from lifelines import CoxPHFitter
            import pandas as pd

            self._covariates = [c for c in covariate_cols if c in df.columns]
            cols   = [duration_col, event_col] + self._covariates
            fit_df = df[cols].dropna()
            if len(fit_df) < 10:
                log.warning("CoxPH: only %d rows — model may be unreliable", len(fit_df))

            # Drop near-zero-variance columns to avoid singular matrix
            cov_cols = [c for c in self._covariates if fit_df[c].std() > 1e-6]
            if len(cov_cols) < len(self._covariates):
                dropped = set(self._covariates) - set(cov_cols)
                log.warning("CoxPH: dropping zero-variance covariates: %s", dropped)
                self._covariates = cov_cols
                fit_df = fit_df[[duration_col, event_col] + cov_cols]

            # Retry with progressively higher penalizer if matrix is singular
            for pen in (self._penalizer, 5.0, 10.0, 50.0):
                try:
                    model = CoxPHFitter(penalizer=pen)
                    model.fit(fit_df, duration_col=duration_col, event_col=event_col)
                    self._model   = model
                    self._fitted  = True
                    self._penalizer = pen
                    log.info("CoxPH fitted on %d rows (penalizer=%.1f), covariates: %s",
                             len(fit_df), pen, self._covariates)
                    break
                except Exception as inner:
                    log.warning("CoxPH penalizer=%.1f failed: %s — retrying with higher", pen, inner)
            else:
                log.error("CoxPH failed at all penalizer levels — model not fitted")
        except Exception as exc:
            log.error("CoxPH training failed: %s", exc)
        return self

    def predict(self, features_dict: dict[str, Any]) -> RULPrediction:
        if not self._fitted or self._model is None:
            return _DEFAULT_RUL
        try:
            import pandas as pd
            row = {k: float(features_dict.get(k, 0.0) or 0.0) for k in self._covariates}
            df  = pd.DataFrame([row])
            sf  = self._model.predict_survival_function(df)
            return _extract_rul(sf, "CoxPH")
        except Exception as exc:
            log.debug("CoxPH predict failed: %s", exc)
            return _DEFAULT_RUL

    def save(self, path: str | Path) -> None:
        import joblib
        joblib.dump({"model": self._model, "covariates": self._covariates, "fitted": self._fitted}, path)

    def load(self, path: str | Path) -> "CoxPHRULModel":
        import joblib
        state = joblib.load(path)
        self._model      = state["model"]
        self._covariates = state["covariates"]
        self._fitted     = state.get("fitted", True)
        return self


# ── Factory ───────────────────────────────────────────────────────────────────

def build_rul_model(model_name: str) -> WeibullRULModel | CoxPHRULModel:
    spec = RUL_MODEL_SPECS.get(model_name, {})
    cls  = spec.get("class", "CoxPHRULModel")
    return WeibullRULModel() if cls == "WeibullRULModel" else CoxPHRULModel()


def load_rul_model(
    model_name: str,
    save_dir: str | Path = "models/saved",
) -> WeibullRULModel | CoxPHRULModel | None:
    path = Path(save_dir) / f"{model_name}.joblib"
    if not path.exists():
        return None
    try:
        m = build_rul_model(model_name)
        m.load(path)
        return m
    except Exception as exc:
        log.warning("Failed to load RUL model %s: %s", model_name, exc)
        return None


# ── Training helper ────────────────────────────────────────────────────────────

def train_rul_models(
    features_df: "pd.DataFrame",
    duration_col: str = "days_to_failure",
    event_col: str    = "label_binary",
    save_dir: str | Path = "models/saved",
) -> dict[str, bool]:
    """
    Train all 4 RUL models from a combined feature DataFrame.

    Expects columns: duration_col, event_col, plus each model's covariates.
    Returns {model_name: trained_ok}.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    results: dict[str, bool] = {}

    for name, spec in RUL_MODEL_SPECS.items():
        try:
            model = build_rul_model(name)
            model.train(features_df, duration_col, event_col, spec["covariates"])
            model.save(save_path / f"{name}.joblib")
            results[name] = True
            log.info("RUL model %s trained and saved", name)
        except Exception as exc:
            log.error("RUL model %s failed: %s", name, exc)
            results[name] = False

    return results
