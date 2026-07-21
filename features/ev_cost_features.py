"""
EVCostFeatureEngine — charging cost and efficiency features for EV/PHEV.

Features land in the "driver" FeatureStore group because cost-per-km reflects
the combined effect of *where/when* the driver charges and battery degradation —
it is a driver-facing, behavioural metric.

ICE vehicles will have None for all eight features.

Reference comparison used by the AI Service Agent:
    Petrol at ₹100/L, 12 km/L (conservative city economy) = ₹8.33/km

Tariffs are reviewed quarterly. Current values reflect India 2024 rates.
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

# Petrol benchmark used for customer-facing comparisons
PETROL_PRICE_INR_PER_L  = 100.0
PETROL_KM_PER_L         = 12.0
PETROL_COST_PER_KM_INR  = PETROL_PRICE_INR_PER_L / PETROL_KM_PER_L   # 8.33


class EVCostFeatureEngine:
    """
    Computes 8 cost-related features from EV charge session data.

    Accepts charge_sessions_df in two column conventions:

    Spec-native (cost engine)          EV health / synthesised sessions
    ─────────────────────────          ────────────────────────────────
    dc_or_ac      (int 1=DC, 0=AC)    charge_type  ('DC' / 'AC')
    duration_minutes                   duration_min
    total_kwh                          energy_kwh
    charge_power_kw  (optional)        —

    Both are normalised before computation, so either source works.
    """

    # INR per kWh by charge category — update quarterly
    CHARGE_TARIFFS_INR_PER_KWH: dict[str, float] = {
        "home_ac":       7.50,   # residential rate
        "public_ac":    12.00,   # mall / commercial EVSE
        "dc_fast":      18.00,   # Tata Power, MG-branded fast charger
        "dc_ultrafast": 24.00,   # 100 kW+ DC chargers
    }

    @staticmethod
    def _empty_features() -> dict[str, Any]:
        return {
            "cost_per_km_inr":                    None,
            "kwh_per_km":                         None,
            "total_charging_cost_inr_30d":        None,
            "energy_wasted_kwh_30d":              None,
            "energy_waste_cost_inr_30d":          None,
            "dc_charge_premium_inr_30d":          None,
            "projected_cost_per_km_at_80pct_soh": None,
            "home_vs_dc_cost_difference":         None,
        }

    @staticmethod
    def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Map EV-health column names to the cost engine's native convention."""
        df = df.copy()
        if "charge_type" in df.columns and "dc_or_ac" not in df.columns:
            df["dc_or_ac"] = (df["charge_type"].str.upper() == "DC").astype(int)
        if "duration_min" in df.columns and "duration_minutes" not in df.columns:
            df = df.rename(columns={"duration_min": "duration_minutes"})
        if "energy_kwh" in df.columns and "total_kwh" not in df.columns:
            df = df.rename(columns={"energy_kwh": "total_kwh"})
        return df

    def _estimate_tariff(self, row: pd.Series) -> float:
        """Classify a single session into a tariff band from its metadata."""
        if int(row.get("dc_or_ac", 0)) == 1:
            if float(row.get("charge_power_kw", 0) or 0) > 80:
                return self.CHARGE_TARIFFS_INR_PER_KWH["dc_ultrafast"]
            return self.CHARGE_TARIFFS_INR_PER_KWH["dc_fast"]
        if float(row.get("duration_minutes", 0) or 0) > 300:   # > 5 h = likely home overnight
            return self.CHARGE_TARIFFS_INR_PER_KWH["home_ac"]
        return self.CHARGE_TARIFFS_INR_PER_KWH["public_ac"]

    def compute(
        self,
        charge_sessions_df: pd.DataFrame,
        vin: str,
        fleet_row: dict,
        feature_store_features: dict,
    ) -> dict[str, Any]:
        """
        Compute EV cost features for a single VIN.

        Parameters
        ----------
        charge_sessions_df:
            One row per charge session. Required columns: vin (or VIN),
            total_kwh (or energy_kwh). Optional: dc_or_ac / charge_type,
            duration_minutes / duration_min, charge_power_kw.
        vin:
            Vehicle identification number to filter by.
        fleet_row:
            Fleet CSV row for this VIN. Used for battery_capacity_kwh fallback.
        feature_store_features:
            Pre-computed features for this VIN.
            Consumed: soh_estimated (default 100), km_per_day_30d_avg (default 40).

        Returns
        -------
        dict with 8 keys. All values are None when no session data is available.
        """
        if charge_sessions_df is None or charge_sessions_df.empty:
            return self._empty_features()

        # Handle both "vin" and "VIN" column names
        vin_col = "VIN" if ("VIN" in charge_sessions_df.columns and "vin" not in charge_sessions_df.columns) else "vin"
        if vin_col not in charge_sessions_df.columns:
            return self._empty_features()

        sessions = charge_sessions_df[charge_sessions_df[vin_col] == vin]
        if sessions.empty:
            return self._empty_features()

        sessions = self._normalise_columns(sessions)

        if "total_kwh" not in sessions.columns:
            return self._empty_features()

        sessions = sessions.dropna(subset=["total_kwh"]).copy()
        if sessions.empty:
            return self._empty_features()

        # ── Tariff & cost per session ────────────────────────────────────────
        sessions["tariff"]   = sessions.apply(self._estimate_tariff, axis=1)
        sessions["cost_inr"] = sessions["total_kwh"] * sessions["tariff"]

        # ── Cost per km ───────────────────────────────────────────────────────
        total_kwh_30d      = float(sessions["total_kwh"].sum())
        total_cost_inr_30d = float(sessions["cost_inr"].sum())

        km_per_day   = float(feature_store_features.get("km_per_day_30d_avg") or 40)
        total_km_30d = km_per_day * 30

        cost_per_km_inr = total_cost_inr_30d / max(total_km_30d, 1.0)
        kwh_per_km      = total_kwh_30d       / max(total_km_30d, 1.0)

        # ── Charging efficiency loss (battery degradation increases waste) ────
        soh = float(feature_store_features.get("soh_estimated") or 100)
        # Combined AC/DC efficiency; degrades ~0.1% per SoH percentage point lost
        charge_efficiency                = 0.92 * (0.90 + 0.10 * soh / 100.0)
        sessions["energy_wasted_kwh"]    = sessions["total_kwh"] * (1.0 - charge_efficiency)
        energy_wasted_30d_kwh            = float(sessions["energy_wasted_kwh"].sum())
        energy_waste_cost_inr            = float((sessions["energy_wasted_kwh"] * sessions["tariff"]).sum())

        # ── Projected cost when SoH reaches 80 % ─────────────────────────────
        # Shorter range at 80 % SoH means more kWh/km; 1.03 accounts for the
        # additional round-trip efficiency loss at degraded SoH.
        range_factor_at_80pct             = (soh / 100.0) / 0.80
        projected_cost_per_km_at_80pct    = cost_per_km_inr * range_factor_at_80pct * 1.03

        # ── DC fast-charge premium vs equivalent home-AC cost ─────────────────
        if "dc_or_ac" in sessions.columns:
            dc_sessions = sessions[sessions["dc_or_ac"] == 1]
        else:
            dc_sessions = sessions.iloc[0:0]   # empty

        if not dc_sessions.empty:
            dc_kwh             = float(dc_sessions["total_kwh"].sum())
            dc_cost            = float(dc_sessions["cost_inr"].sum())
            equivalent_home    = dc_kwh * self.CHARGE_TARIFFS_INR_PER_KWH["home_ac"]
            dc_premium_inr_30d = dc_cost - equivalent_home
        else:
            dc_premium_inr_30d = 0.0

        return {
            "cost_per_km_inr":                    round(cost_per_km_inr, 2),
            "kwh_per_km":                         round(kwh_per_km, 4),
            "total_charging_cost_inr_30d":        round(total_cost_inr_30d, 0),
            "energy_wasted_kwh_30d":              round(energy_wasted_30d_kwh, 2),
            "energy_waste_cost_inr_30d":          round(energy_waste_cost_inr, 0),
            "dc_charge_premium_inr_30d":          round(dc_premium_inr_30d, 0),
            "projected_cost_per_km_at_80pct_soh": round(projected_cost_per_km_at_80pct, 2),
            "home_vs_dc_cost_difference":         round(dc_premium_inr_30d, 0),
        }
