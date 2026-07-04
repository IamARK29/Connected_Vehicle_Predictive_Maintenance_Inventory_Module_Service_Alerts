"""
AutoPredict Alert Dispatcher.

Class AlertDispatcher.dispatch(alert, vehicle_profile, customer_prefs)

Priority routing:
  CRITICAL → FCM push + SMS + email (all simultaneously, asyncio.gather)
  HIGH     → FCM push + SMS
  MEDIUM   → FCM push + email
  LOW      → FCM push only

Cooldown: Redis SETNX with 24-hour TTL per (VIN, alert_type) key.
          Same alert type will not be re-dispatched for the same VIN
          within 24 hours.

Audit log: Every dispatch attempt is written to the PostgreSQL
           alerts_log table (async, non-fatal).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

import httpx

log = logging.getLogger(__name__)

# ── Channel credentials (from environment) ─────────────────────────────────────

FCM_SERVER_KEY = os.getenv("FCM_SERVER_KEY", "")
FCM_URL        = "https://fcm.googleapis.com/fcm/send"

TWILIO_SID     = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_TOKEN   = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM    = os.getenv("TWILIO_FROM_NUMBER", "")

SENDGRID_KEY   = os.getenv("SENDGRID_API_KEY", "")
SENDGRID_FROM  = os.getenv("SENDGRID_FROM_EMAIL", "alerts@autopredict.io")

REDIS_URL      = os.getenv("REDIS_URL", "redis://localhost:6379/0")
PG_DSN         = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/autopredict")

_COOLDOWN_SECONDS = 86_400  # 24 hours


# ── Low-level channel sends ────────────────────────────────────────────────────

async def _send_push(device_token: str, title: str, body: str, data: dict) -> bool:
    if not FCM_SERVER_KEY or not device_token:
        log.debug("FCM push skipped — no key or token")
        return False
    payload = {
        "to": device_token,
        "notification": {"title": title, "body": body, "sound": "default"},
        "data": {k: str(v) for k, v in data.items()},
        "priority": "high",
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                FCM_URL,
                json=payload,
                headers={"Authorization": f"key={FCM_SERVER_KEY}",
                         "Content-Type": "application/json"},
            )
        ok = resp.status_code == 200
        if not ok:
            log.warning("FCM returned %d: %s", resp.status_code, resp.text[:200])
        return ok
    except Exception as exc:
        log.error("FCM push failed: %s", exc)
        return False


async def _send_sms(to_number: str, body: str) -> bool:
    if not TWILIO_SID or not TWILIO_TOKEN or not to_number:
        log.debug("SMS skipped — Twilio not configured or no number")
        return False
    # Truncate to 160 chars for single SMS
    body = body[:160]
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_SID}/Messages.json",
                data={"To": to_number, "From": TWILIO_FROM, "Body": body},
                auth=(TWILIO_SID, TWILIO_TOKEN),
            )
        ok = resp.status_code == 201
        if not ok:
            log.warning("Twilio SMS returned %d: %s", resp.status_code, resp.text[:200])
        return ok
    except Exception as exc:
        log.error("SMS dispatch failed: %s", exc)
        return False


async def _send_email(to_email: str, subject: str, html_body: str) -> bool:
    if not SENDGRID_KEY or not to_email:
        log.debug("Email skipped — SendGrid not configured or no address")
        return False
    payload = {
        "personalizations": [{"to": [{"email": to_email}]}],
        "from": {"email": SENDGRID_FROM, "name": "AutoPredict Alerts"},
        "subject": subject,
        "content": [{"type": "text/html", "value": html_body}],
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                "https://api.sendgrid.com/v3/mail/send",
                json=payload,
                headers={"Authorization": f"Bearer {SENDGRID_KEY}",
                         "Content-Type": "application/json"},
            )
        ok = resp.status_code == 202
        if not ok:
            log.warning("SendGrid returned %d: %s", resp.status_code, resp.text[:200])
        return ok
    except Exception as exc:
        log.error("Email dispatch failed: %s", exc)
        return False


# ── Cooldown (Redis) ──────────────────────────────────────────────────────────

async def _check_and_set_cooldown(vin: str, alert_type: str) -> bool:
    """
    Returns True if the alert is within cooldown (should NOT dispatch).
    Sets the cooldown key if not already set.
    Uses redis.asyncio (from the redis package).
    """
    try:
        import redis.asyncio as aioredis
        r = aioredis.from_url(REDIS_URL, decode_responses=True)
        key = f"alert_cooldown:{vin}:{alert_type}"
        set_result = await r.set(key, "1", nx=True, ex=_COOLDOWN_SECONDS)
        await r.aclose()
        # set_result is None if key already existed (cooldown active)
        return set_result is None
    except Exception as exc:
        log.debug("Redis cooldown check failed (proceeding): %s", exc)
        return False  # fail open — dispatch if Redis unavailable


# ── PostgreSQL audit log ──────────────────────────────────────────────────────

async def _log_to_postgres(alert: Any, channels_sent: dict[str, bool], dispatch_at: datetime) -> None:
    """Write one row to alerts_log. Non-fatal."""
    try:
        import asyncpg
        conn = await asyncpg.connect(PG_DSN)
        await conn.execute(
            """
            INSERT INTO alerts_log (
                vin, alert_type, severity, title,
                message_customer, recommended_action,
                estimated_cost_min, estimated_cost_max,
                confidence_score, model_version,
                triggered_at, dispatched_at,
                channels_json, data_snapshot_json
            ) VALUES (
                $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14
            )
            """,
            alert.vin,
            alert.alert_type,
            alert.severity,
            alert.title,
            alert.message_customer,
            alert.recommended_action,
            float(alert.estimated_cost_min),
            float(alert.estimated_cost_max),
            float(alert.confidence_score),
            alert.model_version,
            alert.triggered_at,
            dispatch_at,
            json.dumps(channels_sent),
            alert.data_snapshot_json,
        )
        await conn.close()
    except Exception as exc:
        log.debug("PostgreSQL alert log failed (non-fatal): %s", exc)


# ── HTML email template ────────────────────────────────────────────────────────

def _html_email(alert: Any, vehicle_profile: dict) -> str:
    plate = vehicle_profile.get("license_plate", alert.vin)
    model = vehicle_profile.get("model_name", "")
    color_map = {"CRITICAL": "#cc0000", "HIGH": "#e65100", "MEDIUM": "#f9a825", "LOW": "#1565c0"}
    color = color_map.get(alert.severity, "#333333")
    cost_str = (
        f"₹{alert.estimated_cost_min:,.0f} – ₹{alert.estimated_cost_max:,.0f}"
        if alert.estimated_cost_max > 0 else "No immediate cost"
    )
    return f"""
    <html><body style="font-family:Arial,sans-serif;max-width:600px;margin:auto;padding:20px">
    <div style="background:{color};color:white;padding:15px 20px;border-radius:8px 8px 0 0">
        <h2 style="margin:0">{alert.severity}: {alert.title}</h2>
    </div>
    <div style="border:1px solid #ddd;border-top:none;padding:20px;border-radius:0 0 8px 8px">
        <p><strong>Vehicle:</strong> {model} — {plate}</p>
        <p style="font-size:16px">{alert.message_customer}</p>
        <div style="background:#f5f5f5;padding:15px;border-radius:6px;margin:15px 0">
            <strong>Recommended Action</strong>
            <p style="margin:8px 0 0">{alert.recommended_action}</p>
        </div>
        <p><strong>Estimated Service Cost:</strong> {cost_str}</p>
        <p style="color:#888;font-size:12px">
            Alert generated: {alert.triggered_at.strftime('%d %b %Y %H:%M UTC')}<br>
            Confidence: {alert.confidence_score:.0%}
        </p>
    </div>
    <p style="color:#aaa;font-size:11px;text-align:center;margin-top:15px">
        AutoPredict by MG Motor India · You are receiving this because you opted in to vehicle health alerts.
    </p>
    </body></html>
    """


# ── AlertDispatcher ────────────────────────────────────────────────────────────

class AlertDispatcher:
    """
    Routes an Alert to the appropriate notification channels based on severity,
    respects per-(VIN, alert_type) cooldown, and logs every dispatch to PostgreSQL.

    Usage (async context):
        dispatcher = AlertDispatcher()
        result = await dispatcher.dispatch(alert, vehicle_profile, customer_prefs)

    For synchronous callers:
        result = asyncio.run(dispatcher.dispatch(alert, vehicle_profile, customer_prefs))

    vehicle_profile keys: vin, license_plate, model_name, fuel_type, color
    customer_prefs  keys: device_token, phone_number, email, language_preference,
                          alert_opt_in (bool), sms_opt_in (bool), email_opt_in (bool)
    """

    async def dispatch(
        self,
        alert: Any,
        vehicle_profile: dict,
        customer_prefs: dict,
    ) -> dict[str, Any]:
        """
        Dispatch a single alert.

        Returns:
            {
              "dispatched": bool,
              "reason": str,            # "sent" | "cooldown" | "opt_out"
              "channels": {             # per-channel success/skip
                "push":  bool | None,
                "sms":   bool | None,
                "email": bool | None,
              }
            }
        """
        vin        = alert.vin
        sev        = alert.severity          # CRITICAL | HIGH | MEDIUM | LOW
        alert_type = alert.alert_type

        # ── Cooldown check ────────────────────────────────────────────────────
        in_cooldown = await _check_and_set_cooldown(vin, alert_type)
        if in_cooldown and sev != "CRITICAL":
            log.debug("Alert %s/%s suppressed by cooldown", vin, alert_type)
            return {"dispatched": False, "reason": "cooldown", "channels": {}}

        # ── Customer opt-in ───────────────────────────────────────────────────
        if not customer_prefs.get("alert_opt_in", True):
            return {"dispatched": False, "reason": "opt_out", "channels": {}}

        device_token = customer_prefs.get("device_token")
        phone_number = customer_prefs.get("phone_number") if customer_prefs.get("sms_opt_in", True) else None
        email        = customer_prefs.get("email") if customer_prefs.get("email_opt_in", True) else None

        # ── Compose messages ──────────────────────────────────────────────────
        push_title = f"[{sev}] {alert.title}"
        push_body  = alert.message_customer[:200]
        sms_body   = f"AutoPredict [{vin}]: {alert.title}. {alert.message_customer}"
        email_subj = f"[AutoPredict {sev}] {alert.title} — {vehicle_profile.get('license_plate', vin)}"
        html_body  = _html_email(alert, vehicle_profile)
        push_data  = {
            "vin": vin,
            "alert_type": alert_type,
            "severity": sev,
            "cost_min": str(alert.estimated_cost_min),
            "cost_max": str(alert.estimated_cost_max),
        }

        # ── Channel selection per severity ────────────────────────────────────
        channels: dict[str, Any] = {"push": None, "sms": None, "email": None}

        if sev == "CRITICAL":
            # All channels simultaneously
            tasks = [
                _send_push(device_token or "", push_title, push_body, push_data),
                _send_sms(phone_number or "", sms_body),
                _send_email(email or "", email_subj, html_body),
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            channels["push"]  = bool(results[0]) if not isinstance(results[0], Exception) else False
            channels["sms"]   = bool(results[1]) if not isinstance(results[1], Exception) else False
            channels["email"] = bool(results[2]) if not isinstance(results[2], Exception) else False

        elif sev == "HIGH":
            tasks = [
                _send_push(device_token or "", push_title, push_body, push_data),
                _send_sms(phone_number or "", sms_body),
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            channels["push"] = bool(results[0]) if not isinstance(results[0], Exception) else False
            channels["sms"]  = bool(results[1]) if not isinstance(results[1], Exception) else False

        elif sev == "MEDIUM":
            tasks = [
                _send_push(device_token or "", push_title, push_body, push_data),
                _send_email(email or "", email_subj, html_body),
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            channels["push"]  = bool(results[0]) if not isinstance(results[0], Exception) else False
            channels["email"] = bool(results[1]) if not isinstance(results[1], Exception) else False

        else:  # LOW
            channels["push"] = await _send_push(device_token or "", push_title, push_body, push_data)

        # ── Audit log ─────────────────────────────────────────────────────────
        dispatch_at = datetime.now(timezone.utc)
        asyncio.ensure_future(_log_to_postgres(alert, channels, dispatch_at))

        dispatched_any = any(v is True for v in channels.values())
        log.info(
            "Alert dispatched — VIN=%s type=%s severity=%s channels=%s",
            vin, alert_type, sev,
            {k: v for k, v in channels.items() if v is not None},
        )

        return {
            "dispatched": dispatched_any,
            "reason": "sent",
            "channels": channels,
        }

    def dispatch_sync(
        self,
        alert: Any,
        vehicle_profile: dict,
        customer_prefs: dict,
    ) -> dict[str, Any]:
        """Synchronous wrapper for non-async callers."""
        return asyncio.run(self.dispatch(alert, vehicle_profile, customer_prefs))
