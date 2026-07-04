"""
TBox Real-Time Receiver — Mode A ingestion.

Handles two transport paths simultaneously:
  HTTP:  FastAPI router  POST /api/v1/ingest/tbox/{channel_id}
  MQTT:  paho-mqtt subscriber  vehicle/{vin}/channel/{channel_id}

Both paths validate via TelemetryValidator and publish to Kafka:
  "telemetry.raw"      — valid records
  "telemetry.rejected" — invalid records with error details
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Annotated, Any

import paho.mqtt.client as mqtt
from confluent_kafka import Producer
from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import get_current_user
from ingestion.validators import CHANNEL_MEASUREMENT, TelemetryValidator
from ingestion.signal_registry import SignalDecoder
from ingestion.telemetry_tier_router import TelemetryTierRouter

log = logging.getLogger(__name__)

MQTT_HOST  = os.getenv("MQTT_BROKER_HOST", "localhost")
MQTT_PORT  = int(os.getenv("MQTT_BROKER_PORT", "1883"))
KAFKA_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

# MQTT topic: vehicle/{vin}/channel/{channel_id}
MQTT_TOPIC_PATTERN = "vehicle/+/channel/+"

_validator = TelemetryValidator()
_router    = TelemetryTierRouter()
_producer: Producer | None = None


def _get_producer() -> Producer:
    global _producer
    if _producer is None:
        _producer = Producer({"bootstrap.servers": KAFKA_SERVERS, "acks": "all"})
    return _producer


def _delivery_report(err, msg) -> None:
    if err:
        log.error("Kafka delivery failed: %s", err)


def _publish(topic: str, vin: str, channel_id: int, payload: dict[str, Any]) -> None:
    envelope = {
        "vin": vin,
        "channel_id": channel_id,
        "measurement": CHANNEL_MEASUREMENT.get(channel_id, "tbox_unknown"),
        "received_at": datetime.now(timezone.utc).isoformat(),
        "payload": payload,
    }
    try:
        _get_producer().produce(
            topic,
            key=vin.encode(),
            value=json.dumps(envelope).encode(),
            callback=_delivery_report,
        )
        _get_producer().poll(0)
    except Exception as exc:
        log.error("Kafka publish error [%s]: %s", topic, exc)


def process_tbox_message(vin: str, channel_id: int, raw_payload: dict[str, Any]) -> dict:
    """Validate, decode (scale/offset), and route a TBox channel message."""
    is_valid, cleaned, errors, warnings = _validator.validate(channel_id, raw_payload)

    if not is_valid:
        _publish("telemetry.rejected", vin, channel_id, {
            "original": raw_payload, "errors": errors,
        })
        return {"accepted": False, "vin": vin, "channel_id": channel_id, "errors": errors}

    # Apply SignalDecoder to convert raw→physical and detect invalid sentinels
    decoded = SignalDecoder.decode_row(cleaned)

    # Any signal that decoded to None → publish original to rejected with context
    for signal_name, physical_val in decoded.items():
        if physical_val is None and signal_name in cleaned and cleaned[signal_name] is not None:
            _publish("telemetry.rejected", vin, channel_id, {
                "vin": vin, "channel_id": channel_id,
                "signal_name": signal_name,
                "raw_value": cleaned[signal_name],
                "reason": "invalid_sentinel_or_out_of_range",
            })
            warnings.append(f"{signal_name}: raw={cleaned[signal_name]} rejected by SignalDecoder")

    # Route to correct Kafka tier topics
    for topic in _router.route(channel_id):
        _publish(topic, vin, channel_id, decoded)

    return {"accepted": True, "vin": vin, "channel_id": channel_id, "warnings": warnings}


# ── FastAPI Router ──────────────────────────────────────────────────────────

router = APIRouter(prefix="/ingest/tbox", tags=["TBox Ingestion"])


@router.post("/{channel_id}", status_code=status.HTTP_202_ACCEPTED)
async def ingest_tbox_http(
    channel_id: int,
    body: dict[str, Any],
    current_user: Annotated[dict, Depends(get_current_user)],
):
    """Ingest a single TBox channel payload over HTTP POST."""
    if channel_id not in range(1, 24):
        raise HTTPException(status_code=400, detail=f"channel_id must be 1-23, got {channel_id}")

    vin = body.pop("vin", None) or body.pop("VIN", None)
    if not vin:
        raise HTTPException(status_code=422, detail="Field 'vin' is required in the payload body")

    result = process_tbox_message(vin, channel_id, body)
    if not result["accepted"]:
        raise HTTPException(status_code=422, detail={"validation_errors": result["errors"]})
    return result


@router.post("/batch", status_code=status.HTTP_202_ACCEPTED)
async def ingest_tbox_batch(
    records: list[dict[str, Any]],
    current_user: Annotated[dict, Depends(get_current_user)],
):
    """Ingest a batch of multi-channel TBox records."""
    accepted = rejected = 0
    for rec in records:
        vin = rec.pop("vin", rec.pop("VIN", None))
        channel_id = rec.pop("channel_id", None)
        if not vin or not channel_id:
            rejected += 1
            continue
        result = process_tbox_message(vin, int(channel_id), rec)
        if result["accepted"]:
            accepted += 1
        else:
            rejected += 1
    return {"accepted": accepted, "rejected": rejected, "total": accepted + rejected}


# ── MQTT Subscriber ────────────────────────────────────────────────────────

class TBoxMQTTReceiver:
    """Subscribes to MQTT and routes channel messages through process_tbox_message."""

    def __init__(self) -> None:
        self._client = mqtt.Client(client_id="autopredict-tbox-mqtt", protocol=mqtt.MQTTv5)
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        self._client.on_disconnect = self._on_disconnect

    def _on_connect(self, client, userdata, flags, rc, properties=None) -> None:
        log.info("MQTT connected (rc=%s) — subscribing to %s", rc, MQTT_TOPIC_PATTERN)
        client.subscribe(MQTT_TOPIC_PATTERN, qos=1)

    def _on_disconnect(self, client, userdata, rc, properties=None) -> None:
        log.warning("MQTT disconnected (rc=%s)", rc)

    def _on_message(self, client, userdata, msg) -> None:
        # Topic pattern: vehicle/{vin}/channel/{channel_id}
        try:
            parts = msg.topic.split("/")
            if len(parts) != 4:
                return
            vin = parts[1]
            channel_id = int(parts[3])
            payload = json.loads(msg.payload.decode("utf-8"))
            result = process_tbox_message(vin, channel_id, payload)
            if not result["accepted"]:
                log.warning("Rejected MQTT message VIN=%s CH=%d: %s", vin, channel_id, result.get("errors"))
        except Exception as exc:
            log.error("MQTT message processing error on %s: %s", msg.topic, exc)

    def start_background(self) -> threading.Thread:
        username = os.getenv("MQTT_USERNAME")
        password = os.getenv("MQTT_PASSWORD")
        if username:
            self._client.username_pw_set(username, password)
        self._client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)

        thread = threading.Thread(target=self._client.loop_forever, daemon=True)
        thread.start()
        log.info("MQTT subscriber started (background thread) — %s:%s", MQTT_HOST, MQTT_PORT)
        return thread

    def stop(self) -> None:
        self._client.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    TBoxMQTTReceiver().start_background().join()
