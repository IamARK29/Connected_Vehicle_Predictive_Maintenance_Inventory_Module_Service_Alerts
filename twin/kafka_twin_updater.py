"""
Kafka consumer that keeps VehicleTwin in sync with all event streams.

Subscribes to:
  telemetry.standard    → update_from_telemetry()
  predictions.generated → update_predictions()
  alerts.generated      → update_alerts()
  dtc.events            → update_dtcs()
  features.daily        → update_degradation()
"""
from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any

log = logging.getLogger(__name__)

KAFKA_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
GROUP_ID = "twin-updater"

TOPICS = [
    "telemetry.standard",
    "predictions.generated",
    "alerts.generated",
    "dtc.events",
    "features.daily",
]


class KafkaTwinUpdater:
    """Consumes Kafka topics and routes messages to TwinManager update methods."""

    def __init__(self) -> None:
        self._consumer = None
        self._running = False

    def _get_consumer(self):
        if self._consumer is None:
            from confluent_kafka import Consumer
            self._consumer = Consumer({
                "bootstrap.servers": KAFKA_SERVERS,
                "group.id": GROUP_ID,
                "auto.offset.reset": "latest",
                "enable.auto.commit": True,
            })
            self._consumer.subscribe(TOPICS)
        return self._consumer

    def run(self) -> None:
        from twin.vehicle_twin import TwinManager

        mgr = TwinManager()
        consumer = self._get_consumer()
        self._running = True
        log.info("KafkaTwinUpdater started — topics: %s", TOPICS)

        while self._running:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                log.warning("Kafka consumer error: %s", msg.error())
                continue

            try:
                topic = msg.topic()
                payload = json.loads(msg.value().decode("utf-8"))
                self._dispatch(mgr, topic, payload)
            except Exception as exc:
                log.debug("Twin update failed for %s: %s", msg.topic(), exc)

        consumer.close()
        log.info("KafkaTwinUpdater stopped")

    def _dispatch(self, mgr: Any, topic: str, payload: dict) -> None:
        vin = payload.get("vin", "")
        if not vin:
            return

        if topic == "telemetry.standard":
            channel_id = int(payload.get("channel_id", 0))
            decoded = payload.get("payload", payload)
            mgr.update_from_telemetry(vin, channel_id, decoded)

        elif topic == "predictions.generated":
            mgr.update_predictions(
                vin,
                model_results=payload.get("failure_probs", {}),
                stages=payload.get("failure_stages", {}),
                rul=payload.get("rul_days", {}),
            )

        elif topic == "alerts.generated":
            alerts = payload.get("alerts", [payload])
            mgr.update_alerts(vin, alerts)

        elif topic == "dtc.events":
            dtcs = payload.get("dtcs", [payload])
            mgr.update_dtcs(vin, dtcs)

        elif topic == "features.daily":
            features = payload.get("features", payload)
            mgr.update_degradation(vin, features)

    def stop(self) -> None:
        self._running = False

    def start_background(self) -> threading.Thread:
        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()
        return thread
