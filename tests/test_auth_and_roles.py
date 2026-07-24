"""
Role-based access control and JWT normalization tests.

Covers the bugs previously found and fixed:
  - require_oem() must accept lowercase roles ("oem", "admin")
  - get_current_user() always returns lowercase role
  - Dealer endpoints 403 when accessing another dealer's data
  - OEM endpoints 403 for DEALER role
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("POSTGRES_URL",   "sqlite:///./test_autopredict.db")
os.environ.setdefault("INFLUXDB_URL",   "http://localhost:8086")
os.environ.setdefault("INFLUXDB_TOKEN", "test-token")
os.environ.setdefault("INFLUXDB_ORG",   "autopredict")
os.environ.setdefault("REDIS_URL",      "redis://localhost:6379/0")
os.environ.setdefault("SECRET_KEY",     "test-secret-key-not-for-prod")


@pytest.fixture(scope="module")
def client() -> TestClient:
    from api.main import app
    return TestClient(app, raise_server_exceptions=False)


def _token(client: TestClient, username: str, password: str) -> str:
    resp = client.post("/api/auth/token", json={"username": username, "password": password})
    assert resp.status_code == 200, f"Login failed for {username}: {resp.text}"
    return resp.json()["access_token"]


def _h(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ── JWT role normalization ─────────────────────────────────────────────────────

def test_get_current_user_returns_lowercase_role_oem(client):
    """JWT payload role is uppercased in _BUILT_IN_USERS; get_current_user must lowercase it."""
    from api.dependencies import get_current_user, create_access_token
    from fastapi.security import HTTPAuthorizationCredentials

    token = create_access_token({"sub": "oem", "role": "OEM", "dealer_code": "ALL"})
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
    user = get_current_user(creds)
    assert user["role"] == "oem", f"Expected 'oem', got {user['role']!r}"


def test_get_current_user_returns_lowercase_role_admin(client):
    from api.dependencies import get_current_user, create_access_token
    from fastapi.security import HTTPAuthorizationCredentials

    token = create_access_token({"sub": "admin", "role": "ADMIN", "dealer_code": "ALL"})
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
    user = get_current_user(creds)
    assert user["role"] == "admin"


def test_get_current_user_returns_lowercase_role_dealer(client):
    from api.dependencies import get_current_user, create_access_token
    from fastapi.security import HTTPAuthorizationCredentials

    token = create_access_token({"sub": "dealer", "role": "DEALER", "dealer_code": "DL001"})
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
    user = get_current_user(creds)
    assert user["role"] == "dealer"
    assert user["dealer_code"] == "DL001"


# ── require_oem guard ──────────────────────────────────────────────────────────

def test_oem_fleet_overview_allowed_for_oem(client):
    """OEM user must reach the fleet-overview endpoint (not 403)."""
    tok = _token(client, "oem", "oem123")
    resp = client.get("/api/oem/fleet-overview", headers=_h(tok))
    assert resp.status_code != 403, "OEM user got 403 — role guard is broken"
    assert resp.status_code in (200, 204, 500)


def test_oem_fleet_overview_allowed_for_admin(client):
    tok = _token(client, "admin", "admin123")
    resp = client.get("/api/oem/fleet-overview", headers=_h(tok))
    assert resp.status_code != 403, "Admin user got 403 on OEM endpoint"
    assert resp.status_code in (200, 204, 500)


def test_oem_fleet_overview_blocked_for_dealer(client):
    """Dealer must NOT access OEM portal endpoints."""
    tok = _token(client, "dealer", "dealer123")
    resp = client.get("/api/oem/fleet-overview", headers=_h(tok))
    assert resp.status_code == 403, (
        f"Dealer should get 403 on OEM endpoint, got {resp.status_code}"
    )


def test_oem_model_health_allowed_for_oem(client):
    tok = _token(client, "oem", "oem123")
    resp = client.get("/api/oem/model-health", headers=_h(tok))
    assert resp.status_code != 403
    assert resp.status_code in (200, 204, 500)


def test_oem_model_health_blocked_for_dealer(client):
    tok = _token(client, "dealer", "dealer123")
    resp = client.get("/api/oem/model-health", headers=_h(tok))
    assert resp.status_code == 403


def test_oem_retrain_trigger_allowed_for_oem(client):
    tok = _token(client, "oem", "oem123")
    resp = client.post("/api/oem/retrain/trigger", headers=_h(tok), json={"model": "brake_wear"})
    assert resp.status_code != 403
    assert resp.status_code in (200, 202, 204, 422, 500)


def test_oem_retrain_trigger_blocked_for_dealer(client):
    tok = _token(client, "dealer", "dealer123")
    resp = client.post("/api/oem/retrain/trigger", headers=_h(tok), json={"model": "brake_wear"})
    assert resp.status_code == 403


def test_oem_eda_blocked_for_dealer(client):
    tok = _token(client, "dealer", "dealer123")
    resp = client.get("/api/oem/eda", headers=_h(tok))
    assert resp.status_code == 403


def test_oem_whatif_blocked_for_dealer(client):
    tok = _token(client, "dealer", "dealer123")
    resp = client.post("/api/oem/whatif", headers=_h(tok), json={})
    assert resp.status_code in (403, 422)


def test_oem_endpoint_requires_auth(client):
    """Unauthenticated request to any OEM endpoint must return 401/403."""
    resp = client.get("/api/oem/fleet-overview")
    assert resp.status_code in (401, 403)


# ── Dealer scoping ─────────────────────────────────────────────────────────────

def test_dealer_can_access_own_dealer_code(client):
    """DL001 dealer can access DL001 endpoints."""
    tok = _token(client, "dealer", "dealer123")
    resp = client.get("/api/dealer/DL001/bay-status", headers=_h(tok))
    assert resp.status_code != 403, "Dealer got 403 accessing own dealer_code"


def test_dealer_blocked_from_other_dealer_code(client):
    """DL001 dealer must get 403 trying to access DL002 data."""
    tok = _token(client, "dealer", "dealer123")
    resp = client.get("/api/dealer/DL002/bay-status", headers=_h(tok))
    assert resp.status_code == 403, (
        f"Expected 403 for cross-dealer access, got {resp.status_code}"
    )


def test_admin_can_access_any_dealer_code(client):
    """Admin bypasses dealer scoping and can reach any dealer_code."""
    tok = _token(client, "admin", "admin123")
    resp = client.get("/api/dealer/DL001/bay-status", headers=_h(tok))
    assert resp.status_code != 403
    resp2 = client.get("/api/dealer/DL002/bay-status", headers=_h(tok))
    assert resp2.status_code != 403


def test_dealer2_blocked_from_dealer1_data(client):
    """dealer2 (DL002) must not reach DL001 endpoints."""
    tok = _token(client, "dealer2", "dealer123")
    resp = client.get("/api/dealer/DL001/bay-status", headers=_h(tok))
    assert resp.status_code == 403


def test_dealer2_can_access_own_data(client):
    tok = _token(client, "dealer2", "dealer123")
    resp = client.get("/api/dealer/DL002/bay-status", headers=_h(tok))
    assert resp.status_code != 403


# ── Demand forecast scoping (dealer endpoint) ──────────────────────────────────

def test_demand_forecast_dealer_accesses_own_data(client):
    """Dealer can access demand forecast for their own dealer_code."""
    tok = _token(client, "dealer", "dealer123")
    resp = client.get("/api/dealer/DL001/demand-forecast", headers=_h(tok))
    assert resp.status_code not in (401, 403), (
        f"Dealer got {resp.status_code} accessing own demand forecast"
    )


def test_demand_forecast_dealer_blocked_from_other(client):
    """Dealer (DL001) cannot access demand forecast for DL002."""
    tok = _token(client, "dealer", "dealer123")
    resp = client.get("/api/dealer/DL002/demand-forecast", headers=_h(tok))
    assert resp.status_code == 403


def test_demand_forecast_oem_can_access_any_dealer(client):
    """OEM can pull demand forecast for any specific dealer_code."""
    tok = _token(client, "oem", "oem123")
    resp = client.get("/api/dealer/DL001/demand-forecast", headers=_h(tok))
    # OEM is not guarded by _assert_own_dealer — must not 403
    assert resp.status_code != 403
    assert resp.status_code in (200, 204, 500)


def test_demand_forecast_admin_unrestricted(client):
    """Admin can access any dealer's demand forecast."""
    tok = _token(client, "admin", "admin123")
    for dc in ("DL001", "DL002", "DL003"):
        resp = client.get(f"/api/dealer/{dc}/demand-forecast", headers=_h(tok))
        assert resp.status_code != 403, f"Admin blocked on {dc} demand forecast"


# ── Role case tolerance ────────────────────────────────────────────────────────

def test_require_oem_accepts_mixed_case_from_old_tokens(client):
    """require_oem must work even if an old token stored uppercase role."""
    from api.dependencies import create_access_token
    # Simulate a token minted with the old uppercase role
    token = create_access_token({"sub": "oem", "role": "OEM", "dealer_code": "ALL"})
    resp = client.get("/api/oem/fleet-overview", headers=_h(token))
    # get_current_user lowercases → require_oem sees "oem" → should not 403
    assert resp.status_code != 403, (
        "OEM endpoint should work with uppercase role in JWT token payload"
    )
