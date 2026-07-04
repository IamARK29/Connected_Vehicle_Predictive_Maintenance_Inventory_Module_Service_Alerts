"""
TCN Autoencoder for anomaly detection on 60-second high-frequency windows.

Input:  (batch, n_features=5, seq_len=600)  — 60s × 10 Hz
Output: reconstruction of same shape; anomaly_score = per-sample MSE.

Features (normalised):
  0: vehRPM      / 12750.0
  1: vehSpeed    / 4000.0    (raw TBox units)
  2: vehBrakePos / 255.0
  3: vehAccelPos / 255.0
  4: abs(tboxAccelX) / 500.0  (raw)

Training: healthy VINs only. Loss = MSELoss.
Threshold: 95th percentile of training reconstruction errors.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

FEATURE_NAMES = ["vehRPM", "vehSpeed", "vehBrakePos", "vehAccelPos", "tboxAccelX"]
NORM_FACTORS  = [12750.0, 4000.0, 255.0, 255.0, 500.0]


def normalise_window(window: np.ndarray) -> np.ndarray:
    """window: (5, 600) raw → (5, 600) normalised."""
    norm = np.zeros_like(window, dtype=np.float32)
    norm[0] = window[0] / 12750.0
    norm[1] = window[1] / 4000.0
    norm[2] = window[2] / 255.0
    norm[3] = window[3] / 255.0
    norm[4] = np.abs(window[4]) / 500.0
    return norm


# ── Model ─────────────────────────────────────────────────────────────────────

try:
    import torch
    import torch.nn as nn

    class TCNBlock(nn.Module):
        """Causal dilated conv block with residual connection."""

        def __init__(self, n_channels: int, kernel_size: int, dilation: int) -> None:
            super().__init__()
            padding = (kernel_size - 1) * dilation
            self.conv = nn.Conv1d(
                n_channels, n_channels, kernel_size,
                padding=padding, dilation=dilation,
            )
            self.relu = nn.ReLU()
            self.norm = nn.BatchNorm1d(n_channels)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            out = self.conv(x)[:, :, :x.size(2)]   # causal: trim future padding
            return self.relu(self.norm(out)) + x    # residual

    class TCNAutoencoder(nn.Module):
        def __init__(self, n_features: int = 5, hidden: int = 32) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv1d(n_features, hidden, 1),
                TCNBlock(hidden, 3, 1),
                TCNBlock(hidden, 3, 2),
                TCNBlock(hidden, 3, 4),
                TCNBlock(hidden, 3, 8),
            )
            self.decoder = nn.Sequential(
                TCNBlock(hidden, 3, 8),
                TCNBlock(hidden, 3, 4),
                TCNBlock(hidden, 3, 2),
                TCNBlock(hidden, 3, 1),
                nn.Conv1d(hidden, n_features, 1),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.decoder(self.encoder(x))

        def anomaly_score(self, x: "torch.Tensor") -> "torch.Tensor":
            """Per-sample mean squared reconstruction error. Shape: (batch,)"""
            with torch.no_grad():
                recon = self.forward(x)
                return torch.mean((x - recon) ** 2, dim=[1, 2])

    _TORCH_AVAILABLE = True

except ImportError:
    _TORCH_AVAILABLE = False
    log.warning("torch not installed — TCNAutoencoder unavailable")

    class TCNBlock:  # type: ignore[no-redef]
        def __init__(self, *a, **kw):
            raise ImportError("torch is required for TCNBlock")

    class TCNAutoencoder:  # type: ignore[no-redef]
        def __init__(self, *a, **kw):
            raise ImportError("torch is required for TCNAutoencoder")


# ── Training ───────────────────────────────────────────────────────────────────

def build_windows(
    telemetry_dir: str | Path,
    healthy_vins: list[str],
    window_sec: int = 60,
    hz: int = 10,
    max_windows_per_vin: int = 200,
) -> np.ndarray:
    """
    Build (N, 5, 600) array of normalised windows from healthy-VIN telemetry CSVs.

    Expects CSV files at telemetry_dir/telemetry_{vin}.csv with raw TBox columns.
    """
    import pandas as pd

    tdir = Path(telemetry_dir)
    seq_len = window_sec * hz
    windows: list[np.ndarray] = []

    _COL_MAP = {
        "vehRPM":       ["vehRPM",      "VehRPM",      "rpm"],
        "vehSpeed":     ["vehSpeed",    "VehSpeed",    "speed"],
        "vehBrakePos":  ["vehBrakePos", "VehBrakePos", "brake_pos"],
        "vehAccelPos":  ["vehAccelPos", "VehAccelPos", "accel_pos"],
        "tboxAccelX":   ["tboxAccelX",  "VehAccelX",   "accel_x"],
    }

    for vin in healthy_vins:
        csv_path = tdir / f"telemetry_{vin}.csv"
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path, low_memory=False)
        except Exception:
            continue

        # Resolve column names
        cols = []
        for feat in FEATURE_NAMES:
            found = next((c for c in _COL_MAP[feat] if c in df.columns), None)
            if found is None:
                break
            cols.append(found)
        if len(cols) < 5:
            continue

        arr = df[cols].fillna(0).values.T.astype(np.float32)  # (5, N_rows)
        n_windows = min(max_windows_per_vin, arr.shape[1] // seq_len)
        for i in range(n_windows):
            w = arr[:, i * seq_len: (i + 1) * seq_len]
            if w.shape[1] == seq_len:
                windows.append(normalise_window(w))

    if not windows:
        return np.zeros((0, 5, seq_len), dtype=np.float32)
    return np.stack(windows)


def train(
    telemetry_dir: str | Path = "data/telemetry",
    healthy_vins_path: str | Path = "data/healthy_vins.txt",
    save_dir: str | Path = "models/saved",
    epochs: int = 30,
    batch_size: int = 64,
) -> None:
    """Train TCN Autoencoder on healthy-VIN windows."""
    if not _TORCH_AVAILABLE:
        raise ImportError("torch is required for training")

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    healthy_vins: list[str] = []
    hvp = Path(healthy_vins_path)
    if hvp.exists():
        healthy_vins = [v.strip() for v in hvp.read_text().splitlines() if v.strip()]
    else:
        log.warning("healthy_vins file not found: %s — scanning telemetry_dir", hvp)
        tdir = Path(telemetry_dir)
        healthy_vins = [p.stem.replace("telemetry_", "") for p in tdir.glob("telemetry_*.csv")]

    log.info("Building windows from %d healthy VINs …", len(healthy_vins))
    X = build_windows(telemetry_dir, healthy_vins)

    if len(X) == 0:
        log.warning("No training windows found — skipping TCN training")
        return

    X_t = torch.tensor(X, dtype=torch.float32)
    ds  = TensorDataset(X_t)
    dl  = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model     = TCNAutoencoder()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for (xb,) in dl:
            optimizer.zero_grad()
            loss = criterion(model(xb), xb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        log.info("Epoch %d  train_loss=%.6f", epoch + 1, epoch_loss / max(len(dl), 1))

    # Compute threshold at 95th percentile of training errors
    model.eval()
    with torch.no_grad():
        all_scores = model.anomaly_score(X_t).numpy()
    threshold = float(np.percentile(all_scores, 95))
    log.info("Anomaly threshold (95th pct): %.6f", threshold)

    pt_path = save_path / "tcn_anomaly_detector.pt"
    torch.save(model.state_dict(), pt_path)
    json.dump({"threshold": threshold}, open(save_path / "tcn_anomaly_threshold.json", "w"), indent=2)
    log.info("TCN training complete. Saved to %s", pt_path)


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_anomaly_score(
    hf_df: "import pandas; pandas.DataFrame",
    save_dir: str | Path = "models/saved",
    window_sec: int = 60,
    hz: int = 10,
) -> float:
    """
    Compute max anomaly score across 60-second windows from a high-frequency DataFrame.

    hf_df: DataFrame with raw TBox columns (vehRPM, vehSpeed, vehBrakePos, vehAccelPos, tboxAccelX).
    Returns the maximum MSE across all windows, or 0.0 if model not available.
    """
    if not _TORCH_AVAILABLE:
        return 0.0

    import torch

    save_path = Path(save_dir)
    pt_path   = save_path / "tcn_anomaly_detector.pt"
    if not pt_path.exists():
        log.debug("TCN model not found at %s", pt_path)
        return 0.0

    model = TCNAutoencoder()
    model.load_state_dict(torch.load(str(pt_path), map_location="cpu"))
    model.eval()

    seq_len = window_sec * hz
    windows = build_windows.__wrapped__(hf_df, seq_len) if hasattr(build_windows, "__wrapped__") else \
        _df_to_windows(hf_df, seq_len)

    if windows is None or len(windows) == 0:
        return 0.0

    with torch.no_grad():
        scores = model.anomaly_score(torch.tensor(windows, dtype=torch.float32))
    return float(scores.max().item())


def _df_to_windows(hf_df: Any, seq_len: int = 600) -> "np.ndarray | None":
    """Convert a hf_df to normalised windows array."""
    _COL_MAP = {
        "vehRPM":      ["vehRPM",      "VehRPM",      "rpm"],
        "vehSpeed":    ["vehSpeed",    "VehSpeed",    "speed"],
        "vehBrakePos": ["vehBrakePos", "VehBrakePos", "brake_pos"],
        "vehAccelPos": ["vehAccelPos", "VehAccelPos", "accel_pos"],
        "tboxAccelX":  ["tboxAccelX",  "VehAccelX",   "accel_x"],
    }
    cols = []
    for feat in FEATURE_NAMES:
        found = next((c for c in _COL_MAP[feat] if c in hf_df.columns), None)
        if found is None:
            return None
        cols.append(found)

    arr = hf_df[cols].fillna(0).values.T.astype(np.float32)
    n_windows = arr.shape[1] // seq_len
    if n_windows == 0:
        return None

    windows = []
    for i in range(n_windows):
        w = arr[:, i * seq_len: (i + 1) * seq_len]
        windows.append(normalise_window(w))
    return np.stack(windows)


from typing import Any  # noqa: E402


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    if cmd == "train":
        telemetry_dir   = sys.argv[2] if len(sys.argv) > 2 else "data/telemetry"
        healthy_vins    = sys.argv[3] if len(sys.argv) > 3 else "data/healthy_vins.txt"
        save_dir        = sys.argv[4] if len(sys.argv) > 4 else "models/saved"
        train(telemetry_dir, healthy_vins, save_dir)
    else:
        print(f"Unknown command: {cmd}. Use: train")
        sys.exit(1)
