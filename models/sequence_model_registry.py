"""
SequenceModelRegistry — unified loader for LSTM, TCN, and Transformer models.

predict_lstm(vin, feature_store) -> dict   {"brake":p, "oil":p, ...}
predict_tcn_anomaly(vin, hf_df)  -> float  max MSE across 60s windows
predict_battery_rul(vin, fs, fr) -> dict|None  {"rul_days": f, "rul_km": f}
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_SAVE_DIR = Path("models/saved")


class SequenceModelRegistry:
    """
    Lazily loads the three sequence models on first use.
    Fails gracefully if .pt files are absent.
    """

    def __init__(self, save_dir: str | Path = _SAVE_DIR) -> None:
        self._save_dir = Path(save_dir)
        self._lstm_model       = None
        self._tcn_model        = None
        self._transformer_model = None
        self._lstm_loaded       = False
        self._tcn_loaded        = False
        self._transformer_loaded = False

    # ── LSTM ─────────────────────────────────────────────────────────────────

    def _load_lstm(self) -> bool:
        if self._lstm_loaded:
            return self._lstm_model is not None
        self._lstm_loaded = True
        try:
            import torch
            from models.sequence_models.lstm_failure_predictor import (
                LSTMFailurePredictor, NORM_STATS,
            )
            pt = self._save_dir / "lstm_failure_predictor.pt"
            if not pt.exists():
                log.info("LSTM weights not found at %s — predict_lstm will return zeros", pt)
                return False
            m = LSTMFailurePredictor()
            m.load_state_dict(torch.load(str(pt), map_location="cpu"))
            m.eval()
            self._lstm_model = m
            return True
        except Exception as exc:
            log.warning("Failed to load LSTM model: %s", exc)
            return False

    def predict_lstm(
        self,
        vin: str,
        feature_store: Any,
    ) -> dict[str, float]:
        """Return per-class failure probabilities {class: prob}."""
        try:
            from models.sequence_models.lstm_failure_predictor import predict as _predict
            return _predict(vin, feature_store, save_dir=self._save_dir)
        except Exception as exc:
            log.warning("predict_lstm failed for VIN %s: %s", vin, exc)
            return {}

    # ── TCN ──────────────────────────────────────────────────────────────────

    def _load_tcn(self) -> bool:
        if self._tcn_loaded:
            return self._tcn_model is not None
        self._tcn_loaded = True
        try:
            import torch
            from models.sequence_models.tcn_anomaly_detector import TCNAutoencoder
            pt = self._save_dir / "tcn_anomaly_detector.pt"
            if not pt.exists():
                log.info("TCN weights not found at %s", pt)
                return False
            m = TCNAutoencoder()
            m.load_state_dict(torch.load(str(pt), map_location="cpu"))
            m.eval()
            self._tcn_model = m
            return True
        except Exception as exc:
            log.warning("Failed to load TCN model: %s", exc)
            return False

    def predict_tcn_anomaly(
        self,
        vin: str,
        hf_df: Any,
    ) -> float:
        """Return max MSE anomaly score across 60-second windows."""
        try:
            from models.sequence_models.tcn_anomaly_detector import predict_anomaly_score
            return predict_anomaly_score(hf_df, save_dir=self._save_dir)
        except Exception as exc:
            log.warning("predict_tcn_anomaly failed for VIN %s: %s", vin, exc)
            return 0.0

    # ── Transformer ───────────────────────────────────────────────────────────

    def _load_transformer(self) -> bool:
        if self._transformer_loaded:
            return self._transformer_model is not None
        self._transformer_loaded = True
        try:
            import torch
            from models.sequence_models.battery_soh_transformer import BatterySoHTransformer
            pt = self._save_dir / "battery_soh_transformer.pt"
            if not pt.exists():
                log.info("Transformer weights not found at %s", pt)
                return False
            m = BatterySoHTransformer()
            m.load_state_dict(torch.load(str(pt), map_location="cpu"))
            m.eval()
            self._transformer_model = m
            return True
        except Exception as exc:
            log.warning("Failed to load Transformer model: %s", exc)
            return False

    def predict_battery_rul(
        self,
        vin: str,
        feature_store: Any,
        fleet_row: dict | None = None,
    ) -> dict | None:
        """
        Return {"rul_days": float, "rul_km": float} for EV VINs, None for ICE.
        """
        if fleet_row and str(fleet_row.get("fuel_type", "ICE")) != "EV":
            return None
        try:
            from models.sequence_models.battery_soh_transformer import predict as _predict
            return _predict(vin, feature_store, fleet_row=fleet_row, save_dir=self._save_dir)
        except Exception as exc:
            log.warning("predict_battery_rul failed for VIN %s: %s", vin, exc)
            return None
