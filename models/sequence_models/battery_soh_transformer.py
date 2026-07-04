"""
Transformer encoder for battery Remaining Useful Life (RUL) prediction.

EV VINs only. Predicts [rul_days, rul_km] as continuous positive outputs.

Input:  (batch, seq_len=90, n_features=6)  — 90 daily snapshots
Output: (batch, 2)  — [rul_days, rul_km], both ≥ 0 (ReLU applied)

Features:
  0: soh_estimated                 / 100.0
  1: cell_voltage_spread           / 0.1
  2: dc_charge_fraction_30d        (0–1, no normalisation)
  3: (avg_cell_temp - 20) / 40.0
  4: isolation_resistance_min_30d  / 8000.0
  5: range_per_kwh_30d_trend       / 10.0   (slope, can be negative)

Labels from failures_manifest where failure_type == "hv_battery_degradation":
  rul_days = (failure_date - feature_cutoff_date).days
  rul_km   = failure_odometer - current_odometer
Loss: HuberLoss
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

FEATURE_NAMES = [
    "soh_estimated",
    "cell_voltage_spread",
    "dc_charge_fraction_30d",
    "avg_cell_temp",
    "isolation_resistance_min_30d",
    "range_per_kwh_30d_trend",
]

# Normalisation: x_norm = (x - offset) / divisor
_NORM_OFFSETS  = [0.0, 0.0, 0.0, 20.0, 0.0, 0.0]
_NORM_DIVISORS = [100.0, 0.1, 1.0, 40.0, 8000.0, 10.0]


def normalise_vec(raw_vec: list[float]) -> list[float]:
    return [(v - o) / d for v, o, d in zip(raw_vec, _NORM_OFFSETS, _NORM_DIVISORS)]


# ── Model ─────────────────────────────────────────────────────────────────────

try:
    import torch
    import torch.nn as nn

    class BatterySoHTransformer(nn.Module):
        def __init__(
            self,
            n_features: int = 6,
            d_model: int = 64,
            nhead: int = 4,
            num_layers: int = 3,
            seq_len: int = 90,
        ) -> None:
            super().__init__()
            self.input_proj  = nn.Linear(n_features, d_model)
            self.pos_embed   = nn.Embedding(seq_len, d_model)
            enc_layer = nn.TransformerEncoderLayer(
                d_model, nhead, dim_feedforward=128, dropout=0.1, batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers)
            self.head = nn.Sequential(
                nn.Linear(d_model, 32),
                nn.ReLU(),
                nn.Linear(32, 2),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
            h   = self.input_proj(x) + self.pos_embed(pos)
            enc = self.transformer(h)
            return torch.relu(self.head(enc[:, -1, :]))   # RUL ≥ 0

    _TORCH_AVAILABLE = True

except ImportError:
    _TORCH_AVAILABLE = False
    log.warning("torch not installed — BatterySoHTransformer unavailable")

    class BatterySoHTransformer:  # type: ignore[no-redef]
        def __init__(self, *a, **kw):
            raise ImportError("torch is required for BatterySoHTransformer")


# ── Training data ──────────────────────────────────────────────────────────────

def build_sequences(
    feature_store_dir: str | Path,
    failures_manifest_path: str | Path,
    seq_len: int = 90,
) -> "tuple[np.ndarray, np.ndarray, list, list]":
    """
    Build (X, y, dates, vins) for EV VINs with hv_battery_degradation labels.

    X: float32 [N, 90, 6]
    y: float32 [N, 2]  — [rul_days, rul_km]
    """
    import pandas as pd

    fails_path = Path(failures_manifest_path)
    if not fails_path.exists():
        log.warning("failures_manifest not found: %s", fails_path)
        return np.zeros((0, seq_len, 6), dtype=np.float32), np.zeros((0, 2), dtype=np.float32), [], []

    failures = pd.read_csv(fails_path, parse_dates=["failure_date"])
    hv_fails = failures[failures["failure_type"] == "hv_battery_degradation"].copy()
    if hv_fails.empty:
        log.info("No hv_battery_degradation events in manifest")
        return np.zeros((0, seq_len, 6), dtype=np.float32), np.zeros((0, 2), dtype=np.float32), [], []

    fsd = Path(feature_store_dir)
    parquet_files = sorted(fsd.rglob("*.parquet"))
    if not parquet_files:
        return np.zeros((0, seq_len, 6), dtype=np.float32), np.zeros((0, 2), dtype=np.float32), [], []

    all_dfs = []
    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf)
            if "vin" in df.columns and "feature_date" in df.columns:
                all_dfs.append(df)
        except Exception:
            pass
    if not all_dfs:
        return np.zeros((0, seq_len, 6), dtype=np.float32), np.zeros((0, 2), dtype=np.float32), [], []

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all["feature_date"] = pd.to_datetime(df_all["feature_date"])
    df_all = df_all.sort_values(["vin", "feature_date"])

    ev_vins = set(hv_fails["vin"].unique())
    X_list, y_list, dates_list, vins_list = [], [], [], []

    for vin, vin_df in df_all.groupby("vin"):
        if vin not in ev_vins:
            continue
        vin_df = vin_df.reset_index(drop=True)
        if len(vin_df) < seq_len:
            continue

        vin_fails = hv_fails[hv_fails["vin"] == vin]
        if vin_fails.empty:
            continue

        fail_row   = vin_fails.iloc[0]
        fail_date  = pd.to_datetime(fail_row["failure_date"])
        fail_odo   = float(fail_row.get("failure_odometer", np.nan))

        for end_idx in range(seq_len - 1, len(vin_df)):
            window = vin_df.iloc[end_idx - seq_len + 1: end_idx + 1]
            feature_cutoff = window["feature_date"].iloc[-1]

            if feature_cutoff >= fail_date:
                continue  # no future leakage

            rul_days = float((fail_date - feature_cutoff).days)
            cur_odo  = float(window.get("current_odometer_km", pd.Series([np.nan])).iloc[-1]
                             if "current_odometer_km" in window.columns else np.nan)
            rul_km   = float(fail_odo - cur_odo) if not np.isnan(fail_odo) and not np.isnan(cur_odo) else rul_days * 50.0

            seq = np.zeros((seq_len, 6), dtype=np.float32)
            for row_i, (_, row) in enumerate(window.iterrows()):
                raw_vec = [
                    float(row.get("soh_estimated",              85.0) or 85.0),
                    float(row.get("cell_voltage_spread",         0.02) or 0.02),
                    float(row.get("dc_charge_fraction_30d",      0.1)  or 0.1),
                    float(row.get("avg_cell_temp",              30.0) or 30.0),
                    float(row.get("isolation_resistance_min_30d", 5000.0) or 5000.0),
                    float(row.get("range_per_kwh_30d_trend",     0.0)  or 0.0),
                ]
                seq[row_i] = normalise_vec(raw_vec)

            X_list.append(seq)
            y_list.append([rul_days, max(0.0, rul_km)])
            dates_list.append(feature_cutoff)
            vins_list.append(str(vin))

    if not X_list:
        return np.zeros((0, seq_len, 6), dtype=np.float32), np.zeros((0, 2), dtype=np.float32), [], []

    return (
        np.stack(X_list).astype(np.float32),
        np.array(y_list, dtype=np.float32),
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
    """Train BatterySoHTransformer with temporal split and early stopping."""
    if not _TORCH_AVAILABLE:
        raise ImportError("torch is required for training")

    import pandas as pd
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    log.info("Building battery RUL sequences …")
    X, y, dates, vins = build_sequences(feature_store_dir, failures_manifest_path)

    if len(X) == 0:
        log.warning("No EV training sequences found — skipping Transformer training")
        return

    date_arr   = np.array(dates)
    train_mask = date_arr < pd.Timestamp("2024-06-01")
    val_mask   = (date_arr >= pd.Timestamp("2024-06-01")) & (date_arr < pd.Timestamp("2024-09-01"))

    X_tr, y_tr = X[train_mask], y[train_mask]
    X_v,  y_v  = X[val_mask],  y[val_mask]

    if len(X_tr) == 0:
        log.warning("No training samples after temporal split")
        return

    train_ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)

    model     = BatterySoHTransformer()
    criterion = nn.HuberLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float("inf")
    patience  = 0
    pt_path   = save_path / "battery_soh_transformer.pt"

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_loss = 0.0
        if len(X_v) > 0:
            with torch.no_grad():
                val_loss = criterion(
                    model(torch.tensor(X_v, dtype=torch.float32)),
                    torch.tensor(y_v, dtype=torch.float32),
                ).item()
        else:
            model.eval()

        if val_loss < best_loss:
            best_loss = val_loss
            patience  = 0
            torch.save(model.state_dict(), pt_path)
            log.info("Epoch %d — val_loss=%.4f (best)", epoch + 1, val_loss)
        else:
            patience += 1
            if patience >= patience_limit:
                log.info("Early stopping at epoch %d", epoch + 1)
                break
        scheduler.step()

    log.info("Transformer training complete. Saved to %s", pt_path)


# ── Inference ─────────────────────────────────────────────────────────────────

def predict(
    vin: str,
    feature_store: object,
    fleet_row: dict | None = None,
    save_dir: str | Path = "models/saved",
    seq_len: int = 90,
) -> "dict | None":
    """
    Return {"rul_days": float, "rul_km": float} for EV VINs, or None for ICE.

    fleet_row: dict with at least {"fuel_type": "EV"/"ICE"}.
    """
    if not _TORCH_AVAILABLE:
        return None

    import torch

    if fleet_row and str(fleet_row.get("fuel_type", "ICE")) != "EV":
        return None

    save_path = Path(save_dir)
    pt_path   = save_path / "battery_soh_transformer.pt"
    if not pt_path.exists():
        log.debug("Transformer model not found at %s", pt_path)
        return None

    seq = _build_sequence_for_vin(vin, feature_store, seq_len)
    if seq is None:
        return None

    model = BatterySoHTransformer()
    model.load_state_dict(torch.load(str(pt_path), map_location="cpu"))
    model.eval()

    with torch.no_grad():
        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
        out = model(x).squeeze().tolist()

    if isinstance(out, float):
        out = [out, out * 50.0]

    return {"rul_days": round(float(out[0]), 1), "rul_km": round(float(out[1]), 1)}


def _build_sequence_for_vin(
    vin: str,
    feature_store: object,
    seq_len: int = 90,
) -> "np.ndarray | None":
    """Fetch last seq_len days of battery features and normalise."""
    import pandas as pd

    default_vals = [85.0, 0.02, 0.1, 30.0, 5000.0, 0.0]
    feat_keys    = [
        "soh_estimated", "cell_voltage_spread", "dc_charge_fraction_30d",
        "avg_cell_temp", "isolation_resistance_min_30d", "range_per_kwh_30d_trend",
    ]

    rows = []
    today = pd.Timestamp.utcnow().normalize()
    for days_back in range(seq_len - 1, -1, -1):
        date = today - pd.Timedelta(days=days_back)
        try:
            rec = feature_store.get_offline(vin, "battery_hv", date) or {}
            raw_vec = [float(rec.get(k, d) or d) for k, d in zip(feat_keys, default_vals)]
        except Exception:
            raw_vec = list(default_vals)
        rows.append(normalise_vec(raw_vec))

    if not any(any(abs(v) > 1e-6 for v in row) for row in rows):
        return None
    return np.array(rows, dtype=np.float32)


# ── CLI ────────────────────────────────────────────────────────────────────────

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
