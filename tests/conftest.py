"""Shared pytest fixtures for AutoPredict test suite."""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# ── Project root on sys.path ───────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Stub out heavy optional services so tests don't require running infra
os.environ.setdefault("POSTGRES_URL",   "sqlite:///./test_autopredict.db")
os.environ.setdefault("INFLUXDB_URL",   "http://localhost:8086")
os.environ.setdefault("INFLUXDB_TOKEN", "autopredict-dev-token")
os.environ.setdefault("INFLUXDB_ORG",   "autopredict")
os.environ.setdefault("REDIS_URL",      "redis://localhost:6379/0")
os.environ.setdefault("SECRET_KEY",     "test-secret-key-not-for-prod")

SAMPLE_VINS = [
    "MH01MZ7X0001",
    "MH01MZ7X0002",
    "MH01MZ7X0003",
    "MH01MZ7X0004",
    "MH01MZ7X0005",
]


def _make_telemetry_df(vin: str = SAMPLE_VINS[0], n_rows: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    now = datetime.now(timezone.utc)
    timestamps = [now - timedelta(hours=i) for i in range(n_rows - 1, -1, -1)]

    return pd.DataFrame({
        "timestamp":       timestamps,
        "vin":             vin,
        "speed":           rng.uniform(0, 120, n_rows),
        "brake_pos":       rng.uniform(0, 100, n_rows),
        "accel_x":         rng.uniform(-100, 100, n_rows),
        "accel_y":         rng.uniform(-50, 50, n_rows),
        "accel_z":         rng.uniform(-50, 50, n_rows),
        "odometer":        np.cumsum(rng.uniform(0.5, 2.0, n_rows)) + 50_000,
        "brake_front_mm":  rng.uniform(3, 12, n_rows),
        "brake_rear_mm":   rng.uniform(3, 12, n_rows),
        "brake_fluid_pct": rng.uniform(80, 100, n_rows),
        "batt_12v":        rng.uniform(12.0, 14.5, n_rows),
        "coolant_temp":    rng.uniform(80, 100, n_rows),
        "oil_life_pct":    rng.uniform(30, 100, n_rows),
        "rpm":             rng.uniform(700, 3000, n_rows),
        "gear_pos":        rng.integers(0, 6, n_rows).astype(float),
        "accel_pos":       rng.uniform(0, 100, n_rows),
        "steering_angle":  rng.uniform(-30, 30, n_rows),
        # HV battery
        "soc":             rng.uniform(20, 90, n_rows),
        "soh":             rng.uniform(85, 100, n_rows),
        "cell_max_temp":   rng.uniform(25, 40, n_rows),
        "cell_min_temp":   rng.uniform(20, 35, n_rows),
        "cell_max_vol":    rng.uniform(3.6, 3.8, n_rows),
        "cell_min_vol":    rng.uniform(3.5, 3.7, n_rows),
        "pack_voltage":    rng.uniform(300, 400, n_rows),
        "pack_current":    rng.uniform(-10, 50, n_rows),
        # Tyres (raw 0-255 per TBox; ~2.2 bar nominal → raw ≈ 44 at ×0.05)
        "tyre_fl":         rng.uniform(35, 55, n_rows),
        "tyre_fr":         rng.uniform(35, 55, n_rows),
        "tyre_rl":         rng.uniform(35, 55, n_rows),
        "tyre_rr":         rng.uniform(35, 55, n_rows),
        "tyre_temp_fl":    rng.uniform(25, 50, n_rows),
        "tyre_temp_fr":    rng.uniform(25, 50, n_rows),
        "tyre_temp_rl":    rng.uniform(25, 50, n_rows),
        "tyre_temp_rr":    rng.uniform(25, 50, n_rows),
        # driver behaviour proxy
        "fuel_economy":    rng.uniform(10, 20, n_rows),
        "idle_time_s":     rng.uniform(0, 60, n_rows),
    })


@pytest.fixture(scope="session")
def sample_vins() -> list[str]:
    return SAMPLE_VINS


@pytest.fixture(scope="session")
def telemetry_df() -> pd.DataFrame:
    return _make_telemetry_df()


@pytest.fixture(scope="session")
def multi_vin_df() -> pd.DataFrame:
    frames = [_make_telemetry_df(vin=v) for v in SAMPLE_VINS]
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="session")
def api_client() -> TestClient:
    from api.main import app
    return TestClient(app)


@pytest.fixture(scope="session")
def auth_headers(api_client: TestClient) -> dict[str, str]:
    resp = api_client.post("/api/auth/token", json={"username": "admin", "password": "admin123"})
    assert resp.status_code == 200, f"Auth failed: {resp.text}"
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(scope="session")
def dealer_headers(api_client: TestClient) -> dict[str, str]:
    resp = api_client.post("/api/auth/token", json={"username": "dealer", "password": "dealer123"})
    assert resp.status_code == 200
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
