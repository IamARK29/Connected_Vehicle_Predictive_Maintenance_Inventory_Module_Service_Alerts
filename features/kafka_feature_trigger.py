"""
Kafka feature trigger — consumes "telemetry.standard" and fires feature
refresh for a VIN when specific signal conditions are detected.

Trigger conditions:
  vehBrkFludLvlLow==1, vehOilPressureWarning==1, vehMILWarning==1,
  vehSysPwrMod transitions 2→0 (trip ended),
  vehIsCharging transitions 1→0 (charge session ended).

Debounce: Redis key "trigger_cooldown:{vin}" TTL=300 s.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import sys
from typing import Any

log = logging.getLogger(__name__)

_KAFKA_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
_REDIS_URL     = os.getenv("REDIS_URL",               "redis://localhost:6379/0")
_TOPIC         = "telemetry.standard"
_COOLDOWN_TTL  = 300   # seconds


class KafkaFeatureTrigger:

    def __init__(self) -> None:
        self._prev_state: dict[str, dict[str, Any]] = {}   # vin → {pwr_mod, is_charging}
        self._redis: Any = None
        self._running = False

    # ── Redis helpers ────────────────────────────────────────────────────────

    def _get_redis(self):
        if self._redis is None:
            import redis
            self._redis = redis.Redis.from_url(_REDIS_URL, decode_responses=True)
        return self._redis

    def _should_trigger(self, vin: str) -> bool:
        """Return True if cooldown has expired (NX → True means key was absent)."""
        try:
            result = self._get_redis().set(
                f"trigger_cooldown:{vin}", 1, ex=_COOLDOWN_TTL, nx=True
            )
            return bool(result)
        except Exception as exc:
            log.warning("Redis cooldown check failed for VIN %s: %s", vin, exc)
            return True   # fire anyway if Redis is unavailable

    # ── Signal condition checks ──────────────────────────────────────────────

    def _check_triggers(self, vin: str, decoded: dict) -> list[str]:
        triggers: list[str] = []
        prev = self._prev_state.get(vin, {})

        if decoded.get("vehBrkFludLvlLow")       == 1:  triggers.append("brake_fluid_low")
        if decoded.get("vehOilPressureWarning")   == 1:  triggers.append("oil_pressure_warning")
        if decoded.get("vehMILWarning")           == 1:  triggers.append("mil_warning")

        pwr = decoded.get("vehSysPwrMod")
        if prev.get("vehSysPwrMod") == 2 and pwr == 0:
            triggers.append("trip_ended")

        charging = decoded.get("vehIsCharging")
        if prev.get("vehIsCharging") == 1 and charging == 0:
            triggers.append("charge_session_ended")

        # Update state
        self._prev_state[vin] = {
            "vehSysPwrMod":  pwr,
            "vehIsCharging": charging,
        }
        return triggers

    # ── Main consume loop ────────────────────────────────────────────────────

    def run(self) -> None:
        try:
            from confluent_kafka import Consumer
        except ImportError:
            log.error("confluent-kafka not installed; cannot start trigger consumer")
            return

        try:
            from ingestion.signal_registry import SignalDecoder
        except ImportError:
            SignalDecoder = None  # type: ignore[assignment]

        consumer = Consumer({
            "bootstrap.servers": _KAFKA_SERVERS,
            "group.id":          "autopredict-feature-trigger",
            "auto.offset.reset": "latest",
        })
        consumer.subscribe([_TOPIC])
        log.info("Feature trigger consumer subscribed to %s", _TOPIC)

        self._running = True
        try:
            while self._running:
                msg = consumer.poll(timeout=1.0)
                if msg is None:
                    continue
                if msg.error():
                    log.error("Kafka error: %s", msg.error())
                    continue

                try:
                    envelope = json.loads(msg.value().decode("utf-8"))
                    vin     = envelope.get("vin", "UNKNOWN")
                    payload = envelope.get("payload", {})

                    # Decode signals if SignalDecoder is available
                    decoded = SignalDecoder.decode_row(payload) if SignalDecoder else payload

                    triggers = self._check_triggers(vin, decoded)
                    if not triggers:
                        continue

                    log.info("Trigger conditions for VIN %s: %s", vin, triggers)

                    if not self._should_trigger(vin):
                        log.debug("Debounced trigger for VIN %s (cooldown active)", vin)
                        continue

                    self._fire_refresh(vin)

                except Exception as exc:
                    log.error("Message processing error: %s", exc)

        finally:
            consumer.close()
            log.info("Feature trigger consumer stopped")

    def _fire_refresh(self, vin: str) -> None:
        """Dispatch async Celery task; fall back to synchronous if Celery absent."""
        try:
            from features.feature_refresh_job import refresh_single_vin_task
            # Celery .delay() if available
            if hasattr(refresh_single_vin_task, "delay"):
                refresh_single_vin_task.delay(vin)
                log.info("Queued async feature refresh for VIN %s", vin)
            else:
                refresh_single_vin_task(vin)
                log.info("Ran synchronous feature refresh for VIN %s", vin)
        except Exception as exc:
            log.error("Failed to fire refresh for VIN %s: %s", vin, exc)

    def stop(self) -> None:
        self._running = False


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    trigger = KafkaFeatureTrigger()

    def _handle_signal(signum, frame):
        log.info("Received signal %s — shutting down", signum)
        trigger.stop()

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    trigger.run()


if __name__ == "__main__":
    main()
