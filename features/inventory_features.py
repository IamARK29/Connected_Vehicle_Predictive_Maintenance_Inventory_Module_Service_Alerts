"""
Inventory demand feature engine.

Computes dealer × part-level features from fleet composition,
service history, and seasonal patterns for the demand forecast model.
"""
from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class InventoryFeatureEngine:

    def compute(
        self,
        dealer_code: str,
        part_code: str,
        fleet_df: pd.DataFrame,
        service_history_df: pd.DataFrame,
        feature_store: object = None,
        supplier_lead_time_days: int = 7,
    ) -> dict:
        vin_col = "vin" if "vin" in fleet_df.columns else "VIN"
        dc_col = next((c for c in ["dealer_code", "DealerCode"] if c in fleet_df.columns), "dealer_code")
        catchment_vins = fleet_df[fleet_df[dc_col] == dealer_code][vin_col].tolist()

        # Historical demand
        svc_dc_col = next((c for c in ["DealerCode", "dealer_code"] if c in service_history_df.columns), "DealerCode")
        desc_col = next((c for c in ["DescriptionOne", "description"] if c in service_history_df.columns), "DescriptionOne")
        qty_col = next((c for c in ["OrderQuantity", "order_quantity", "qty"] if c in service_history_df.columns), "OrderQuantity")
        date_col = next((c for c in ["CreatedOn", "created_on", "date"] if c in service_history_df.columns), "CreatedOn")

        hist = service_history_df[
            (service_history_df[svc_dc_col] == dealer_code) &
            (service_history_df[desc_col].str.contains(part_code, case=False, na=False))
        ].copy()

        if qty_col not in hist.columns:
            hist[qty_col] = 1

        avg_monthly = 0.0
        trend_slope = 0.0
        lags = [0.0, 0.0, 0.0]
        seasonal_index = 1.0

        if not hist.empty and date_col in hist.columns:
            hist["_period"] = pd.to_datetime(hist[date_col]).dt.to_period("M")
            monthly = hist.groupby("_period")[qty_col].sum()
            avg_monthly = float(monthly.mean()) if len(monthly) > 0 else 0.0
            trend_slope = float(np.polyfit(range(len(monthly)), monthly.values, 1)[0]) if len(monthly) >= 3 else 0.0

            raw_lags = monthly.tail(3).tolist()
            while len(raw_lags) < 3:
                raw_lags.insert(0, 0.0)
            lags = [float(v) for v in raw_lags]

            m = datetime.now().month
            by_month = hist.groupby(pd.to_datetime(hist[date_col]).dt.month)[qty_col].mean()
            seasonal_index = float(by_month.get(m, avg_monthly) / max(avg_monthly, 0.01))

        odo_col = next((c for c in ["initial_odometer", "initial_odometer_km", "odometer_km"] if c in fleet_df.columns), None)
        fleet_avg_odo = 0.0
        if odo_col:
            catchment_df = fleet_df[fleet_df[vin_col].isin(catchment_vins)]
            if not catchment_df.empty:
                fleet_avg_odo = float(catchment_df[odo_col].mean())

        m = datetime.now().month
        return {
            "avg_monthly_units_12m": round(avg_monthly, 2),
            "consumption_trend_slope": round(trend_slope, 4),
            "month_lag_1": lags[2],
            "month_lag_2": lags[1],
            "month_lag_3": lags[0],
            "seasonal_index": round(seasonal_index, 3),
            "vehicles_in_catchment": len(catchment_vins),
            "fleet_avg_odometer_km": round(fleet_avg_odo, 1),
            "supplier_lead_time_days": supplier_lead_time_days,
            "monsoon_active": int(m in [6, 7, 8, 9]),
        }
