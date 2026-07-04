"""AutoPredict FastAPI application entry point."""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from api.routers import vehicles, fleet, dealer, agent, upload, synthetic, monitoring, oem, admin, inventory as inventory_router
from api.ws import telemetry_stream
from api.dependencies import create_access_token
from api.schemas import TokenRequest

# Legacy routers kept for backwards-compatibility
_legacy_routers: list = []
for _mod_name in ("telemetry", "predictions", "alerts", "maintenance", "analytics"):
    try:
        import importlib
        _m = importlib.import_module(f"api.routers.{_mod_name}")
        _legacy_routers.append(_m.router)
    except Exception:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("AutoPredict API starting up...")
    try:
        from ingestion.tbox_receiver import TBoxMQTTReceiver
        TBoxMQTTReceiver().start_background()
        print("MQTT TBox receiver started")
    except Exception as exc:
        print(f"MQTT receiver not started (broker unavailable): {exc}")
    yield
    print("AutoPredict API shutting down.")


limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])

app = FastAPI(
    title="AutoPredict API",
    description=(
        "Automotive Predictive Maintenance Platform — vehicle health monitoring, "
        "ML predictions, alert dispatch, and service scheduling for MG Motor India fleet."
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "Health",    "description": "Service health and version"},
        {"name": "Auth",      "description": "JWT authentication"},
        {"name": "Vehicles",  "description": "Per-vehicle telemetry, predictions, and history"},
        {"name": "Fleet",     "description": "Fleet-wide aggregated views"},
        {"name": "Dealer",    "description": "Dealer bay, inventory, and appointment management"},
        {"name": "AI Agent",  "description": "AI service workflow agent and conversational interface"},
        {"name": "Upload",    "description": "Bulk data ingestion (CSV / MQTT)"},
        {"name": "WebSocket",   "description": "Real-time telemetry and alert streams"},
        {"name": "Monitoring",  "description": "Drift reports, A/B experiment results, model health"},
    ],
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time"] = f"{(time.perf_counter() - start) * 1000:.2f}ms"
    return response


# ── Primary routers ────────────────────────────────────────────────────────────

app.include_router(vehicles.router,         prefix="/api")
app.include_router(fleet.router,            prefix="/api")
app.include_router(dealer.router,           prefix="/api")
app.include_router(agent.router,            prefix="/api")
app.include_router(upload.router,           prefix="/api")
app.include_router(synthetic.router,        prefix="/api")
app.include_router(monitoring.router,       prefix="/api")
app.include_router(oem.router,              prefix="/api")
app.include_router(admin.router,            prefix="/api")
app.include_router(inventory_router.router, prefix="/api")
app.include_router(telemetry_stream.router)   # WebSocket — no /api prefix (ws://)

# Legacy v1 routers (mounted under /api/v1 for backwards-compat)
for _lr in _legacy_routers:
    app.include_router(_lr, prefix="/api/v1")

# TBox ingestion router (Mode A HTTP path)
try:
    from ingestion.tbox_receiver import router as tbox_router
    app.include_router(tbox_router, prefix="/api/v1")
except Exception:
    pass


# ── Auth endpoint ──────────────────────────────────────────────────────────────

@app.post(
    "/api/auth/token",
    tags=["Auth"],
    summary="Issue JWT bearer token",
    responses={
        200: {
            "content": {"application/json": {"example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type":   "bearer",
                "role":         "dealer",
            }}}
        },
        401: {"content": {"application/json": {"example": {"detail": "Invalid credentials"}}}},
    },
)
async def get_token(payload: TokenRequest):
    """
    Issue a JWT for API access.

    - **username**: `admin` / `oem` / `dealer` / `dealer2`
    - **password**: `admin123` / `oem123` / `dealer123`
    """
    from api.routers.admin import _load_users
    users = _load_users()
    info = users.get(payload.username)
    if not info or info["password"] != payload.password:
        return JSONResponse(status_code=401, content={"detail": "Invalid credentials"})
    token = create_access_token({"sub": payload.username, "role": info["role"], "dealer_code": info["dealer_code"]})
    return {"access_token": token, "token_type": "bearer", "role": info["role"], "dealer_code": info["dealer_code"]}


# Backwards-compat alias (query-param form)
@app.post("/api/v1/auth/token", tags=["Auth"], include_in_schema=False)
async def get_token_v1(username: str, password: str):
    from api.routers.admin import _load_users
    users = _load_users()
    info = users.get(username)
    if not info or info["password"] != password:
        return JSONResponse(status_code=401, content={"detail": "Invalid credentials"})
    token = create_access_token({"sub": username, "role": info["role"], "dealer_code": info["dealer_code"]})
    return {"access_token": token, "token_type": "bearer", "role": info["role"], "dealer_code": info["dealer_code"]}


# ── Health endpoints ───────────────────────────────────────────────────────────

@app.get(
    "/",
    tags=["Health"],
    responses={
        200: {
            "content": {"application/json": {"example": {
                "service": "AutoPredict API",
                "version": "2.0.0",
                "status":  "running",
                "docs":    "/docs",
            }}}
        }
    },
)
async def root():
    return {
        "service": "AutoPredict API",
        "version": "2.0.0",
        "status":  "running",
        "docs":    "/docs",
        "ws_endpoints": ["/ws/live/{vin}", "/ws/alerts"],
    }


@app.get(
    "/health",
    tags=["Health"],
    responses={200: {"content": {"application/json": {"example": {"status": "healthy", "timestamp": 1750000000.0}}}}},
)
async def health_check():
    import socket
    checks: dict[str, str] = {}

    def _port_open(host: str, port: int) -> bool:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.3)
        ok = s.connect_ex((host, port)) == 0
        s.close()
        return ok

    checks["postgres"] = "ok" if _port_open("localhost", 5432) else "unavailable"
    checks["influxdb"] = "ok" if _port_open("localhost", 8086) else "unavailable"
    checks["redis"]    = "ok" if _port_open("localhost", 6379) else "unavailable"

    return {
        "status":    "healthy",
        "timestamp": time.time(),
        "checks":    checks,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
