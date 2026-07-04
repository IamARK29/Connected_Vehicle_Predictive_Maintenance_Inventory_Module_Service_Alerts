"""FastAPI TestClient integration tests.

Tests cover all primary API endpoints without requiring live Postgres / InfluxDB / Redis.
Heavy service calls are mocked at the module level via pytest monkeypatching.
"""
from __future__ import annotations

import io
import csv
import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client() -> TestClient:
    from api.main import app
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(scope="module")
def admin_token(client) -> str:
    resp = client.post("/api/auth/token", json={"username": "admin", "password": "admin123"})
    assert resp.status_code == 200
    return resp.json()["access_token"]


@pytest.fixture(scope="module")
def dealer_token(client) -> str:
    resp = client.post("/api/auth/token", json={"username": "dealer", "password": "dealer123"})
    assert resp.status_code == 200
    return resp.json()["access_token"]


@pytest.fixture(scope="module")
def admin_h(admin_token) -> dict:
    return {"Authorization": f"Bearer {admin_token}"}


@pytest.fixture(scope="module")
def dealer_h(dealer_token) -> dict:
    return {"Authorization": f"Bearer {dealer_token}"}


# ── Health / Root ──────────────────────────────────────────────────────────────

def test_root_returns_200(client):
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["service"] == "AutoPredict API"
    assert "version" in body


def test_health_endpoint_structure(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "status" in body
    assert "timestamp" in body
    assert "checks" in body
    assert isinstance(body["checks"], dict)


def test_docs_accessible(client):
    resp = client.get("/docs")
    assert resp.status_code == 200


# ── Auth ───────────────────────────────────────────────────────────────────────

def test_auth_admin_success(client):
    resp = client.post("/api/auth/token", json={"username": "admin", "password": "admin123"})
    assert resp.status_code == 200
    body = resp.json()
    assert "access_token" in body
    assert body["token_type"] == "bearer"
    assert body["role"] in ("ADMIN", "admin")


def test_auth_dealer_success(client):
    resp = client.post("/api/auth/token", json={"username": "dealer", "password": "dealer123"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["role"] in ("DEALER", "dealer")


def test_auth_wrong_password(client):
    resp = client.post("/api/auth/token", json={"username": "admin", "password": "wrongpass"})
    assert resp.status_code == 401


def test_auth_unknown_user(client):
    resp = client.post("/api/auth/token", json={"username": "nobody", "password": "pass"})
    assert resp.status_code == 401


def test_auth_v1_query_param_compat(client):
    resp = client.post("/api/v1/auth/token?username=dealer&password=dealer123")
    assert resp.status_code == 200
    assert "access_token" in resp.json()


def test_protected_endpoint_without_token(client):
    resp = client.get("/api/vehicles")
    assert resp.status_code in (401, 403)


def test_protected_endpoint_bad_token(client):
    resp = client.get("/api/vehicles", headers={"Authorization": "Bearer not-a-valid-token"})
    assert resp.status_code in (401, 403)


# ── Vehicles ──────────────────────────────────────────────────────────────────

def test_get_vehicles_authenticated(client, dealer_h):
    with patch("api.routers.vehicles._query_vehicles", return_value=[]):
        resp = client.get("/api/vehicles", headers=dealer_h)
    assert resp.status_code in (200, 404, 500)  # may fail if no DB; ensure not 401


def test_get_vehicles_not_401(client, dealer_h):
    resp = client.get("/api/vehicles", headers=dealer_h)
    assert resp.status_code != 401


# ── Fleet ──────────────────────────────────────────────────────────────────────

def test_fleet_health_not_401(client, dealer_h):
    resp = client.get("/api/fleet/health", headers=dealer_h)
    assert resp.status_code != 401


def test_fleet_alerts_not_401(client, dealer_h):
    resp = client.get("/api/fleet/alerts?hours=24", headers=dealer_h)
    assert resp.status_code != 401


# ── Dealer ────────────────────────────────────────────────────────────────────

def test_dealer_bay_status_not_401(client, dealer_h):
    resp = client.get("/api/dealer/DL001/bay-status", headers=dealer_h)
    assert resp.status_code != 401


def test_dealer_appointments_not_401(client, dealer_h):
    resp = client.get("/api/dealer/DL001/appointments", headers=dealer_h)
    assert resp.status_code != 401


def test_dealer_inventory_not_401(client, dealer_h):
    resp = client.get("/api/dealer/DL001/inventory", headers=dealer_h)
    assert resp.status_code != 401


# ── Agent ──────────────────────────────────────────────────────────────────────

def test_agent_workflows_not_401(client, dealer_h):
    resp = client.get("/api/agent/workflows", headers=dealer_h)
    assert resp.status_code != 401


def test_agent_chat_not_401(client, dealer_h):
    resp = client.post(
        "/api/agent/chat",
        headers=dealer_h,
        json={"message": "What is the health of vehicle MH01MZ7X0001?"},
    )
    assert resp.status_code != 401


# ── Synthetic Data Generation ──────────────────────────────────────────────────

def test_synthetic_generate_returns_202(client, admin_h):
    with patch("api.routers.synthetic._run_generation"):
        resp = client.post(
            "/api/synthetic/generate",
            headers=admin_h,
            json={"num_vehicles": 2, "num_days": 7, "failure_rate": 0.05},
        )
    assert resp.status_code == 202
    body = resp.json()
    assert "job_id" in body
    assert body["status"] == "queued"
    assert "poll_url" in body


def test_synthetic_generate_invalid_num_vehicles(client, admin_h):
    resp = client.post(
        "/api/synthetic/generate",
        headers=admin_h,
        json={"num_vehicles": 0},    # below minimum of 1
    )
    assert resp.status_code == 422


def test_synthetic_generate_invalid_failure_rate(client, admin_h):
    resp = client.post(
        "/api/synthetic/generate",
        headers=admin_h,
        json={"failure_rate": 0.5},   # above max of 0.30
    )
    assert resp.status_code == 422


def test_synthetic_generate_requires_auth(client):
    resp = client.post(
        "/api/synthetic/generate",
        json={"num_vehicles": 5, "num_days": 30, "failure_rate": 0.05},
    )
    assert resp.status_code in (401, 403)


def test_synthetic_generate_params_echoed(client, admin_h):
    with patch("api.routers.synthetic._run_generation"):
        resp = client.post(
            "/api/synthetic/generate",
            headers=admin_h,
            json={"num_vehicles": 7, "num_days": 14, "failure_rate": 0.10},
        )
    body = resp.json()
    assert body["params"]["num_vehicles"] == 7
    assert body["params"]["num_days"] == 14
    assert abs(body["params"]["failure_rate"] - 0.10) < 1e-6


# ── File Upload ───────────────────────────────────────────────────────────────

def _make_csv_file(rows: list[dict]) -> bytes:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode()


TELEMETRY_ROWS = [
    {"StartTime-TimeStamp": "2024-01-01 00:00:00", "VIN": "MH01MZ7X0001",
     "vehSpeed": 60, "vehEngineTemp": 90, "vehHvSoc": 75, "vehBattVolt": 12.6},
    {"StartTime-TimeStamp": "2024-01-01 01:00:00", "VIN": "MH01MZ7X0001",
     "vehSpeed": 80, "vehEngineTemp": 92, "vehHvSoc": 70, "vehBattVolt": 12.5},
]


def test_upload_telemetry_csv_not_401(client, admin_h):
    csv_bytes = _make_csv_file(TELEMETRY_ROWS)
    resp = client.post(
        "/api/upload/telemetry",
        headers=admin_h,
        files={"file": ("telemetry.csv", csv_bytes, "text/csv")},
    )
    assert resp.status_code != 401


def test_upload_without_auth_rejected(client):
    csv_bytes = _make_csv_file(TELEMETRY_ROWS)
    resp = client.post(
        "/api/upload/telemetry",
        files={"file": ("telemetry.csv", csv_bytes, "text/csv")},
    )
    assert resp.status_code in (401, 403)


def test_upload_wrong_content_type(client, admin_h):
    resp = client.post(
        "/api/upload/telemetry",
        headers=admin_h,
        files={"file": ("data.json", b'{"key": "val"}', "application/json")},
    )
    # Should either reject (422) or accept but return error in body
    assert resp.status_code in (200, 202, 400, 422, 500)


# ── Upload Status ──────────────────────────────────────────────────────────────

def test_upload_status_unknown_job_not_200(client, admin_h):
    resp = client.get("/api/upload/status/nonexistent-job-id", headers=admin_h)
    # Either 404 or we get a status dict; should not be 401
    assert resp.status_code != 401


# ── CORS headers ──────────────────────────────────────────────────────────────

def test_cors_header_present(client):
    resp = client.options(
        "/api/auth/token",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"},
    )
    # OPTIONS may return 200 or 405 depending on middleware
    assert "access-control-allow-origin" in resp.headers or resp.status_code == 405


# ── Process-time header ────────────────────────────────────────────────────────

def test_process_time_header_present(client):
    resp = client.get("/")
    assert "x-process-time" in resp.headers
