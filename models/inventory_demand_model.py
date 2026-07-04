"""
Inventory demand forecast model — LightGBM regressor predicting
30-day part demand at the dealer × part level.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MODEL_PATH = Path("models/saved/inventory_demand_30d.joblib")


class InventoryDemandModel:

    def __init__(self) -> None:
        self.model = None
        self._try_load()

    def _try_load(self) -> None:
        if MODEL_PATH.exists():
            try:
                import joblib
                self.model = joblib.load(MODEL_PATH)
            except Exception as exc:
                log.debug("Failed to load inventory model: %s", exc)

    def train(self, feature_df: pd.DataFrame, target_col: str = "units_next_30d") -> dict:
        try:
            from lightgbm import LGBMRegressor
        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor as LGBMRegressor

        numeric_cols = [c for c in feature_df.columns if c != target_col
                        and feature_df[c].dtype.kind in "biufc"]
        X = feature_df[numeric_cols]
        y = feature_df[target_col]

        self.model = LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.05)
        self.model.fit(X, y)

        import joblib
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, MODEL_PATH)

        from sklearn.metrics import mean_absolute_error
        preds = self.model.predict(X)
        mae = float(mean_absolute_error(y, preds))
        log.info("InventoryDemandModel trained: MAE=%.2f", mae)
        return {"mae": mae, "n_rows": len(feature_df)}

    def predict(self, features_dict: dict) -> dict:
        if self.model is None:
            return self._heuristic_predict(features_dict)

        try:
            point = float(self.model.predict(pd.DataFrame([features_dict]))[0])
        except Exception:
            return self._heuristic_predict(features_dict)

        lead = features_dict.get("supplier_lead_time_days", 7)
        std = max(features_dict.get("avg_monthly_units_12m", 1) * 0.3, 0.5)
        safety_stock = 1.65 * std * (lead ** 0.5)

        return {
            "point_estimate": round(max(0, point), 1),
            "safety_stock": round(safety_stock, 1),
            "reorder_point": round(max(0, point) + safety_stock, 1),
        }

    def _heuristic_predict(self, features_dict: dict) -> dict:
        avg = features_dict.get("avg_monthly_units_12m", 0)
        trend = features_dict.get("consumption_trend_slope", 0)
        seasonal = features_dict.get("seasonal_index", 1.0)
        point = max(0, avg * seasonal + trend)
        lead = features_dict.get("supplier_lead_time_days", 7)
        std = max(avg * 0.3, 0.5)
        safety_stock = 1.65 * std * (lead ** 0.5)
        return {
            "point_estimate": round(point, 1),
            "safety_stock": round(safety_stock, 1),
            "reorder_point": round(point + safety_stock, 1),
        }
