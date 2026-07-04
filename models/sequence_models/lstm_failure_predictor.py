"""
LSTM + Multi-Head Attention failure predictor.

Input:  (batch, seq_len=30, n_features=10)
Output: (batch, 6)  — raw logits for 6 independent failure classes.
        Apply sigmoid at inference; train with BCEWithLogitsLoss.

Feature vector (10 features, exact order and normalisation):
  0: brake_stress_cumulative     / 5000.0
  1: harsh_brake_rate_30d        / 10.0
  2: oil_degradation_index       / 100.0
  3: soh_estimated               / 100.0  (0.0 for ICE)
  4: cell_voltage_spread         / 0.1    (0.0 for ICE)
  5: resting_voltage_7d_avg      → (v - 11.5) / 3.0
  6: pressure_avg_4_tyres_kpa    / 300.0
  7: composite_drive_score       / 100.0
  8: km_since_last_service       / 50000.0
  9: coolant_overtemp_count_30d  / 30.0

Output labels (6 classes, multi-label):
  0: brake_replacement_within_30_days
  1: oil_change_due_within_14_days
  2: hv_battery_soh_below_80_within_90_days
  3: 12v_no_start_within_7_days
  4: tyre_replacement_within_30_days
  5: engine_overheating_within_14_days
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

FEATURE_NAMES = [
    "brake_stress_cumulative",
    "harsh_brake_rate_30d",
    "oil_degradation_index",
    "soh_estimated",
    "cell_voltage_spread",
    "resting_voltage_7d_avg",
    "pressure_avg_4_tyres_kpa",
    "composite_drive_score",
    "km_since_last_service",
    "coolant_overtemp_count_30d",
]

CLASS_NAMES = ["brake", "oil", "hv_battery", "12v_battery", "tyre", "overheating"]

# Normalisation: x_norm = (x - offset) / divisor
_NORM_OFFSETS   = [0.0, 0.0, 0.0, 0.0, 0.0, 11.5, 0.0, 0.0, 0.0, 0.0]
_NORM_DIVISORS  = [5000.0, 10.0, 100.0, 100.0, 0.1, 3.0, 300.0, 100.0, 50000.0, 30.0]

NORM_STATS = {
    "feature_names":  FEATURE_NAMES,
    "norm_factors":   [5000, 10, 100, 100, 0.1, None, 300, 100, 50000, 30],
    "norm_offsets":   _NORM_OFFSETS,
    "norm_divisors":  _NORM_DIVISORS,
    "class_names":    CLASS_NAMES,
}


def normalise_features(raw_vec: list[float]) -> list[float]:
    """Apply per-feature (x - offset) / divisor normalisation."""
    return [
        (v - o) / d
        for v, o, d in zip(raw_vec, _NORM_OFFSETS, _NORM_DIVISORS)
    ]


# ── Model ─────────────────────────────────────────────────────────────────────

try:
    import torch
    import torch.nn as nn

    class LSTMFailurePredictor(nn.Module):
        def __init__(
            self,
            input_size: int = 10,
            hidden_size: int = 64,
            num_layers: int = 2,
            num_classes: int = 6,
        ) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size, hidden_size, num_layers,
                batch_first=True, dropout=0.2,
            )
            self.attention = nn.MultiheadAttention(
                hidden_size, num_heads=4, batch_first=True, dropout=0.1,
            )
            self.layer_norm = nn.LayerNorm(hidden_size)
            self.classifier = nn.Linear(hidden_size, num_classes)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            lstm_out, _ = self.lstm(x)                              # (B, T, H)
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            out = self.layer_norm(lstm_out + attn_out)              # residual + norm
            return self.classifier(out[:, -1, :])                   # final timestep

    _TORCH_AVAILABLE = True

except ImportError:
    _TORCH_AVAILABLE = False
    log.warning("torch not installed — LSTMFailurePredictor unavailable")

    class LSTMFailurePredictor:  # type: ignore[no-redef]
        def __init__(self, *a, **kw):
            raise ImportError("torch is required for LSTMFailurePredictor")


# ── Training data builder ──────────────────────────────────────────────────────

def build_sequences(
    feature_store_dir: str | Path,
    failures_manifest_path: str | Path,
    seq_len: int = 30,
) -> "tuple[np.ndarray, np.ndarray, list, list]":
    """
    Build (X, y, dates, vins) from offline Parquet feature store.

    X: float32 array [N, seq_len, 10]
    y: float32 array [N, 6]
    dates: list of pd.Timestamp (feature_cutoff date for each sample)
    vins:  list of str

    Temporal split is enforced in train() — do not shuffle here.
    """
    import pandas as pd

    feature_store_dir = Path(feature_store_dir)
    fails_path = Path(failures_manifest_path)

    if not fails_path.exists():
        log.warning("failures_manifest not found: %s", fails_path)
        return np.zeros((0, seq_len, 10), dtype=np.float32), np.zeros((0, 6), dtype=np.float32), [], []

    failures = pd.read_csv(fails_path, parse_dates=["failure_date"])

    # Build label lookup: vin → set of active label windows
    _LABEL_COLS = [
        ("brake_degradation",        0, 30),
        ("oil_degradation",          1, 14),
        ("hv_battery_degradation",   2, 90),
        ("12v_battery_failure",      3,  7),
        ("tyre_wear",                4, 30),
        ("engine_overheating",       5, 14),
    ]

    X_list, y_list, dates_list, vins_list = [], [], [], []

    # Scan available Parquet files grouped by VIN
    parquet_files = sorted(feature_store_dir.rglob("*.parquet"))
    if not parquet_files:
        log.warning("No Parquet files found in %s", feature_store_dir)
        return np.zeros((0, seq_len, 10), dtype=np.float32), np.zeros((0, 6), dtype=np.float32), [], []

    # Load all feature records
    all_dfs = []
    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf)
            if "vin" in df.columns and "feature_date" in df.columns:
                all_dfs.append(df)
        except Exception as exc:
            log.debug("Skipping %s: %s", pf, exc)
    if not all_dfs:
        return np.zeros((0, seq_len, 10), dtype=np.float32), np.zeros((0, 6), dtype=np.float32), [], []

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all["feature_date"] = pd.to_datetime(df_all["feature_date"])
    df_all = df_all.sort_values(["vin", "feature_date"])

    for vin, vin_df in df_all.groupby("vin"):
        vin_df = vin_df.reset_index(drop=True)
        if len(vin_df) < seq_len:
            continue

        vin_fails = failures[failures["vin"] == vin]

        for end_idx in range(seq_len - 1, len(vin_df)):
            window = vin_df.iloc[end_idx - seq_len + 1: end_idx + 1]
            feature_cutoff = window["feature_date"].iloc[-1]

            # Build feature matrix [seq_len, 10]
            seq = np.zeros((seq_len, 10), dtype=np.float32)
            for row_i, (_, row) in enumerate(window.iterrows()):
                raw_vec = [
                    float(row.get("brake_stress_cumulative",    0.0) or 0.0),
                    float(row.get("harsh_brake_rate_30d",       0.0) or 0.0),
                    float(row.get("oil_degradation_index",      0.0) or 0.0),
                    float(row.get("soh_estimated",              0.0) or 0.0),
                    float(row.get("cell_voltage_spread",        0.0) or 0.0),
                    float(row.get("resting_voltage_7d_avg",    12.6) or 12.6),
                    float(row.get("pressure_avg_4_tyres_kpa", 230.0) or 230.0),
                    float(row.get("composite_drive_score",      70.0) or 70.0),
                    float(row.get("km_since_last_service",      0.0) or 0.0),
                    float(row.get("coolant_overtemp_count_30d", 0.0) or 0.0),
                ]
                seq[row_i] = normalise_features(raw_vec)

            # Build label vector [6]
            label = np.zeros(6, dtype=np.float32)
            for fail_type, idx, window_days in _LABEL_COLS:
                mask = (vin_fails["failure_type"] == fail_type)
                vf = vin_fails[mask]
                if vf.empty:
                    continue
                fail_dt = pd.to_datetime(vf["failure_date"].iloc[0])
                days_to = (fail_dt - feature_cutoff).days
                if 0 < days_to <= window_days:
                    label[idx] = 1.0

            X_list.append(seq)
            y_list.append(label)
            dates_list.append(feature_cutoff)
            vins_list.append(str(vin))

    if not X_list:
        return np.zeros((0, seq_len, 10), dtype=np.float32), np.zeros((0, 6), dtype=np.float32), [], []

    return (
        np.stack(X_list).astype(np.float32),
        np.stack(y_list).astype(np.float32),
        dates_list,
        vins_list,
    )


# ── Training loop ──────────────────────────────────────────────────────────────

def train(
    feature_store_dir: str | Path = "data/feature_store",
    failures_manifest_path: str | Path = "data/failures_manifest.csv",
    save_dir: str | Path = "models/saved",
    epochs: int = 50,
    batch_size: int = 32,
    patience_limit: int = 10,
) -> None:
    """Train LSTMFailurePredictor with temporal split and early stopping."""
    if not _TORCH_AVAILABLE:
        raise ImportError("torch is required for training")

    import pandas as pd
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    log.info("Building sequences from %s …", feature_store_dir)
    X, y, dates, vins = build_sequences(feature_store_dir, failures_manifest_path)

    if len(X) == 0:
        log.warning("No training sequences found — skipping LSTM training")
        _save_norm_stats(save_path)
        return

    date_arr = np.array(dates)
    train_mask = date_arr < pd.Timestamp("2024-06-01")
    val_mask   = (date_arr >= pd.Timestamp("2024-06-01")) & (date_arr < pd.Timestamp("2024-09-01"))

    X_tr, y_tr = X[train_mask], y[train_mask]
    X_v,  y_v  = X[val_mask],  y[val_mask]

    if len(X_tr) == 0:
        log.warning("No training samples after temporal split — skipping")
        _save_norm_stats(save_path)
        return

    train_ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)

    model = LSTMFailurePredictor()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0] * 6))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float("inf")
    patience  = 0
    pt_path   = save_path / "lstm_failure_predictor.pt"

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_dl:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        val_loss = 0.0
        if len(X_v) > 0:
            with torch.no_grad():
                Xv_t = torch.tensor(X_v, dtype=torch.float32)
                yv_t = torch.tensor(y_v, dtype=torch.float32)
                val_loss = criterion(model(Xv_t), yv_t).item()
        else:
            val_loss = epoch_loss / max(len(train_dl), 1)

        if val_loss < best_loss:
            best_loss = val_loss
            patience  = 0
            torch.save(model.state_dict(), pt_path)
            log.info("Epoch %d - val_loss=%.4f (best)", epoch + 1, val_loss)
        else:
            patience += 1
            log.info("Epoch %d — val_loss=%.4f (patience %d/%d)", epoch + 1, val_loss, patience, patience_limit)
            if patience >= patience_limit:
                log.info("Early stopping at epoch %d", epoch + 1)
                break

        scheduler.step()

    _save_norm_stats(save_path)
    log.info("LSTM training complete. Model saved to %s", pt_path)


def _save_norm_stats(save_path: Path) -> None:
    json.dump(NORM_STATS, open(save_path / "lstm_norm_stats.json", "w"), indent=2)


# ── Inference ─────────────────────────────────────────────────────────────────

def predict(
    vin: str,
    feature_store: Any,
    save_dir: str | Path = "models/saved",
    seq_len: int = 30,
) -> dict[str, float]:
    """
    Return per-class failure probabilities for *vin* using the last seq_len days
    from the offline feature store.

    Returns {class_name: probability} or empty dict if model/data unavailable.
    """
    if not _TORCH_AVAILABLE:
        return {}

    import torch

    save_path = Path(save_dir)
    pt_path   = save_path / "lstm_failure_predictor.pt"
    if not pt_path.exists():
        log.debug("LSTM model not found at %s", pt_path)
        return {}

    try:
        norm_stats = json.load(open(save_path / "lstm_norm_stats.json"))
    except FileNotFoundError:
        norm_stats = NORM_STATS

    # Build sequence from offline feature store
    seq = _build_sequence_for_vin(vin, feature_store, seq_len)
    if seq is None:
        return {}

    model = LSTMFailurePredictor()
    model.load_state_dict(torch.load(str(pt_path), map_location="cpu"))
    model.eval()

    with torch.no_grad():
        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)  # (1, T, 10)
        logits = model(x)
        probs  = torch.sigmoid(logits).squeeze().tolist()

    class_names = norm_stats.get("class_names", CLASS_NAMES)
    if isinstance(probs, float):
        probs = [probs]
    return dict(zip(class_names, probs))


def _build_sequence_for_vin(
    vin: str,
    feature_store: Any,
    seq_len: int = 30,
) -> "np.ndarray | None":
    """Fetch last seq_len daily records from offline store and normalise."""
    import pandas as pd

    offsets   = _NORM_OFFSETS
    divisors  = _NORM_DIVISORS

    default_vals = [0.0, 0.0, 0.0, 0.0, 0.0, 12.6, 230.0, 70.0, 0.0, 0.0]
    feat_keys = FEATURE_NAMES

    rows = []
    today = pd.Timestamp.utcnow().normalize()
    for days_back in range(seq_len - 1, -1, -1):
        date = today - pd.Timedelta(days=days_back)
        try:
            # Merge feature groups available for this VIN/date
            merged: dict[str, float] = {}
            for group in ["brake", "engine", "battery_hv", "battery_12v", "tyre", "driver", "vehicle_state"]:
                rec = feature_store.get_offline(vin, group, date)
                if rec:
                    merged.update(rec)
            vec = [
                (float(merged.get(k, d) or d) - o) / dv
                for k, d, o, dv in zip(feat_keys, default_vals, offsets, divisors)
            ]
        except Exception:
            vec = [(d - o) / dv for d, o, dv in zip(default_vals, offsets, divisors)]
        rows.append(vec)

    if not any(any(v != 0 for v in row) for row in rows):
        return None
    return np.array(rows, dtype=np.float32)


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    if cmd == "train":
        feature_store_dir = sys.argv[2] if len(sys.argv) > 2 else "data/feature_store"
        failures_path     = sys.argv[3] if len(sys.argv) > 3 else "data/failures_manifest.csv"
        save_dir          = sys.argv[4] if len(sys.argv) > 4 else "models/saved"
        train(feature_store_dir, failures_path, save_dir)
    else:
        print(f"Unknown command: {cmd}. Use: train")
        sys.exit(1)
