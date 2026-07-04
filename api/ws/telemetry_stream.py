"""
WebSocket endpoints for real-time telemetry and alert broadcasting.

WS /ws/live/{vin}   — per-VIN telemetry push (polls InfluxDB every 2s)
WS /ws/alerts       — fleet-wide alert broadcast
WS /ws/upload/{job_id} — upload progress (mounted separately from upload router)

Connection Manager pattern: one room per VIN + a global alerts room.
Any component can call `broadcast_alert(alert)` to push to all /ws/alerts clients.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

log = logging.getLogger(__name__)

router = APIRouter(tags=["WebSocket"])

DATA_DIR    = os.getenv("DATA_DIR", "data/synthetic")
POLL_SECS   = float(os.getenv("WS_POLL_SECONDS", "2"))


# ── Connection manager ─────────────────────────────────────────────────────────

class _ConnectionManager:
    """Thread-safe in-process WebSocket room manager."""

    def __init__(self) -> None:
        # vin → set of WebSocket connections
        self._vin_rooms:   dict[str, set[WebSocket]] = {}
        # global alert broadcast set
        self._alert_conns: set[WebSocket] = set()

    async def connect_vin(self, vin: str, ws: WebSocket) -> None:
        await ws.accept()
        self._vin_rooms.setdefault(vin, set()).add(ws)
        log.debug("WS connected: VIN=%s total_rooms=%d", vin, len(self._vin_rooms))

    def disconnect_vin(self, vin: str, ws: WebSocket) -> None:
        room = self._vin_rooms.get(vin, set())
        room.discard(ws)
        if not room:
            self._vin_rooms.pop(vin, None)

    async def send_vin(self, vin: str, data: dict) -> None:
        room = list(self._vin_rooms.get(vin, set()))
        dead: list[WebSocket] = []
        for ws in room:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect_vin(vin, ws)

    async def connect_alerts(self, ws: WebSocket) -> None:
        await ws.accept()
        self._alert_conns.add(ws)

    def disconnect_alerts(self, ws: WebSocket) -> None:
        self._alert_conns.discard(ws)

    async def broadcast_alert(self, data: dict) -> None:
        dead: list[WebSocket] = []
        for ws in list(self._alert_conns):
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._alert_conns.discard(ws)

    def vin_count(self, vin: str) -> int:
        return len(self._vin_rooms.get(vin, set()))

    def total_alert_subscribers(self) -> int:
        return len(self._alert_conns)


manager = _ConnectionManager()


# ── Public API for other modules to broadcast alerts ──────────────────────────

async def broadcast_alert(alert_dict: dict) -> None:
    """Call from AlertDispatcher or alert evaluation to push to WS clients."""
    await manager.broadcast_alert({
        "type":  "alert",
        "data":  alert_dict,
        "ts":    datetime.now(timezone.utc).isoformat(),
    })


# ── Telemetry fetcher ──────────────────────────────────────────────────────────

async def _fetch_latest_telemetry(vin: str) -> dict[str, Any] | None:
    """Fetch the single most-recent telemetry row for *vin*."""
    # Try InfluxDB
    try:
        from influxdb_client import InfluxDBClient
        url    = os.getenv("INFLUXDB_URL", "http://localhost:8086")
        token  = os.getenv("INFLUXDB_TOKEN", "autopredict-dev-token")
        org    = os.getenv("INFLUXDB_ORG", "autopredict")
        bucket = os.getenv("INFLUXDB_BUCKET", "telemetry")
        client = InfluxDBClient(url=url, token=token, org=org)
        qapi   = client.query_api()
        flux   = f'''
            from(bucket: "{bucket}")
            |> range(start: -2m)
            |> filter(fn: (r) => r["vin"] == "{vin}")
            |> pivot(rowKey:["_time"], columnKey:["_field"], valueColumn:"_value")
            |> last()
        '''
        tables = qapi.query(flux)
        for table in tables:
            for rec in table.records:
                client.close()
                return {"timestamp": rec.get_time().isoformat(),
                        **{k: v for k, v in rec.values.items() if not k.startswith("_")}}
        client.close()
    except Exception:
        pass

    # CSV fallback — read last row
    try:
        import pathlib, pandas as pd
        tdir = pathlib.Path(DATA_DIR) / "telemetry"
        csv  = (tdir / f"{vin}_telemetry.csv") if tdir.exists() else \
               (pathlib.Path(DATA_DIR) / f"{vin}_telemetry.csv")
        if csv.exists():
            df  = pd.read_csv(csv, parse_dates=["StartTime-TimeStamp"])
            row = df.iloc[-1].to_dict()
            return {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in row.items()}
    except Exception:
        pass

    return None


async def _run_rules_on_telemetry(vin: str, tel: dict) -> list[dict]:
    """Non-fatal rule evaluation for WS push."""
    try:
        from alerts.rule_engine import RuleEngine
        alerts = RuleEngine().evaluate(vin, tel)
        return [a.to_dict() for a in alerts]
    except Exception:
        return []


# ── WebSocket endpoints ────────────────────────────────────────────────────────

@router.websocket("/ws/live/{vin}")
async def telemetry_stream(websocket: WebSocket, vin: str):
    """
    Real-time telemetry push for a single VIN.

    Protocol:
      → CLIENT  sends JSON: {"cmd": "ping"} to keep alive
      ← SERVER  sends every POLL_SECS seconds:
        {
          "type": "telemetry",
          "vin":  "MZ7X...",
          "ts":   "2026-06-27T...",
          "data": { ...telemetry fields... },
          "alerts": [ ...rule-engine alerts... ]
        }
      ← SERVER  sends on disconnect/error:
        {"type": "error", "message": "..."}
    """
    await manager.connect_vin(vin, websocket)
    log.info("WS /ws/live/%s connected (subscribers=%d)", vin, manager.vin_count(vin))

    # Send a "connected" handshake
    await websocket.send_json({
        "type": "connected",
        "vin":  vin,
        "ts":   datetime.now(timezone.utc).isoformat(),
        "poll_seconds": POLL_SECS,
    })

    try:
        while True:
            # Race: either client sends something or poll timer fires
            try:
                client_msg = await asyncio.wait_for(
                    websocket.receive_json(), timeout=POLL_SECS
                )
                if client_msg.get("cmd") == "disconnect":
                    break
            except asyncio.TimeoutError:
                pass    # normal — proceed to fetch and push
            except WebSocketDisconnect:
                break

            tel = await _fetch_latest_telemetry(vin)
            if tel:
                alerts = await _run_rules_on_telemetry(vin, tel)
                await websocket.send_json({
                    "type":   "telemetry",
                    "vin":    vin,
                    "ts":     datetime.now(timezone.utc).isoformat(),
                    "data":   tel,
                    "alerts": alerts,
                })
                # Also push any new alerts to the global alert broadcast
                for a in alerts:
                    asyncio.ensure_future(manager.broadcast_alert({
                        "type": "alert", "data": a, "ts": datetime.now(timezone.utc).isoformat()
                    }))
            else:
                await websocket.send_json({
                    "type": "no_data",
                    "vin":  vin,
                    "ts":   datetime.now(timezone.utc).isoformat(),
                })

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        log.error("WS /ws/live/%s error: %s", vin, exc)
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        manager.disconnect_vin(vin, websocket)
        log.info("WS /ws/live/%s disconnected", vin)


@router.websocket("/ws/alerts")
async def alerts_stream(websocket: WebSocket):
    """
    Fleet-wide alert broadcast.

    Pushes every new alert to all subscribed clients.

    Protocol:
      ← SERVER sends on new alert:
        {
          "type":  "alert",
          "data":  { ...Alert.to_dict()... },
          "ts":    "2026-06-27T..."
        }
      ← SERVER sends heartbeat every 30s:
        {"type": "heartbeat", "ts": "..."}
      → CLIENT can send {"cmd": "ping"} to keep alive

    Subscribe from JavaScript:
        const ws = new WebSocket("ws://localhost:8000/ws/alerts");
        ws.onmessage = e => console.log(JSON.parse(e.data));
    """
    await manager.connect_alerts(websocket)
    log.info("WS /ws/alerts connected (subscribers=%d)", manager.total_alert_subscribers())

    await websocket.send_json({
        "type": "connected",
        "ts":   datetime.now(timezone.utc).isoformat(),
        "subscribers": manager.total_alert_subscribers(),
    })

    try:
        while True:
            # Heartbeat every 30 seconds; client can keep-alive with pings
            try:
                client_msg = await asyncio.wait_for(
                    websocket.receive_json(), timeout=30.0
                )
                if client_msg.get("cmd") == "disconnect":
                    break
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({
                    "type": "heartbeat",
                    "ts":   datetime.now(timezone.utc).isoformat(),
                    "subscribers": manager.total_alert_subscribers(),
                })
            except WebSocketDisconnect:
                break

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        log.error("WS /ws/alerts error: %s", exc)
    finally:
        manager.disconnect_alerts(websocket)
        log.info("WS /ws/alerts disconnected (remaining=%d)", manager.total_alert_subscribers())
