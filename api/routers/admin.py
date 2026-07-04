"""
Admin-only endpoints for user management.

Users are stored in data/users.json so they survive server restarts.
Falls back to built-in DEMO_USERS if the file is missing.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.dependencies import get_current_user

log = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["Admin"])

USERS_FILE = Path("data/users.json")

_BUILT_IN_USERS: dict[str, dict] = {
    "admin":   {"password": "admin123",  "role": "ADMIN",  "dealer_code": "ALL"},
    "oem":     {"password": "oem123",    "role": "OEM",    "dealer_code": "ALL"},
    "dealer":  {"password": "dealer123", "role": "DEALER", "dealer_code": "DL001"},
    "dealer2": {"password": "dealer123", "role": "DEALER", "dealer_code": "DL002"},
}


def require_admin(current_user: Annotated[dict, Depends(get_current_user)]) -> dict:
    if current_user.get("role") not in ("ADMIN",):
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


def _load_users() -> dict[str, dict]:
    if USERS_FILE.exists():
        try:
            return json.loads(USERS_FILE.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("Could not read users.json: %s", exc)
    return dict(_BUILT_IN_USERS)


def _save_users(users: dict[str, dict]) -> None:
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    USERS_FILE.write_text(json.dumps(users, indent=2), encoding="utf-8")


@router.get("/users")
async def list_users(current_user: Annotated[dict, Depends(require_admin)]):
    users = _load_users()
    return {
        "users": [
            {
                "username": uname,
                "role": info["role"],
                "dealer_code": info["dealer_code"],
            }
            for uname, info in users.items()
        ]
    }


class CreateUserRequest(BaseModel):
    username: str
    password: str
    role: str       # ADMIN | OEM | DEALER
    dealer_code: str = "ALL"


@router.post("/users", status_code=201)
async def create_user(
    body: CreateUserRequest,
    current_user: Annotated[dict, Depends(require_admin)],
):
    if body.role not in ("ADMIN", "OEM", "DEALER"):
        raise HTTPException(status_code=422, detail="role must be ADMIN, OEM, or DEALER")
    if not body.username or len(body.username) < 3:
        raise HTTPException(status_code=422, detail="username must be at least 3 characters")
    if not body.password or len(body.password) < 6:
        raise HTTPException(status_code=422, detail="password must be at least 6 characters")

    dealer_code = body.dealer_code
    if body.role in ("ADMIN", "OEM"):
        dealer_code = "ALL"

    users = _load_users()
    if body.username in users:
        raise HTTPException(status_code=409, detail=f"User '{body.username}' already exists")

    users[body.username] = {
        "password": body.password,
        "role": body.role,
        "dealer_code": dealer_code,
    }
    _save_users(users)
    log.info("Admin created user '%s' (role=%s)", body.username, body.role)
    return {"username": body.username, "role": body.role, "dealer_code": dealer_code}


@router.delete("/users/{username}", status_code=200)
async def delete_user(
    username: str,
    current_user: Annotated[dict, Depends(require_admin)],
):
    if username == "admin":
        raise HTTPException(status_code=403, detail="Cannot delete the built-in admin account")
    users = _load_users()
    if username not in users:
        raise HTTPException(status_code=404, detail=f"User '{username}' not found")
    del users[username]
    _save_users(users)
    return {"deleted": username}
