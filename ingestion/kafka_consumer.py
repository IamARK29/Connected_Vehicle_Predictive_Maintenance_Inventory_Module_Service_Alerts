"""
Kafka Consumer — Mode A processing pipeline.

Consumes from "telemetry.raw", routes by channel_id to the correct
InfluxDB measurement, batches writes, and updates PostgreSQL vehicle state.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import time
from collections import defaultdict
from datetime import datetime, timezone

from confluent_kafka import Consumer, KafkaError, KafkaException
from influxdb_client import Point, WritePrecision

from ingestion.db_writer import write_influx_batch, write_postgres_state
from ingestion.validators import CHANNEL_MEASUREMENT

log = logging.getLogger(__name__)

BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
RAW_TOPIC         = "telemetry.raw"
CONSUMER_GROUP    = "autopredict-ingestion"
BATCH_SIZE        = 500
FLUSH_INTERVAL_S  = 1.0


def _build_point(envelope: dict) -> Point:
    vin     = envelope["vin"]
    meas    = envelope.get("measurement", "tbox_unknown")
    ts      = envelope.get("received_at", datetime.now(timezone.utc).isoformat())
    payload = envelope.get("payload", {})

    point = Point(meas).tag("vin", vin).time(ts, WritePrecision.SECONDS)

    for k, v in payload.items():
        if k.endswith("_invalid"):
            continue  # skip invalidity markers
        if isinstance(v, bool):
            point = point.field(k, v)
        elif isinstance(v, (int, float)):
            point = point.field(k, float(v))
        # strings (imei, dtcCodes etc.) are skipped from InfluxDB numeric fields

    return point


class TelemetryKafkaConsumer:
    def __init__(self) -> None:
        self._running = True
        self._consumer = Consumer({
            "bootstrap.servers": BOOTSTRAP_SERVERS,
            "group.id": CONSUMER_GROUP,
            "auto.offset.reset": "latest",
            "enable.auto.commit": False,
            "fetch.min.bytes": 1,
            "fetch.wait.max.ms": 500,
        })
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

        self._batch: list[Point] = []
        # vin → merged latest values (for PG state upsert)
        self._state_buffer: dict[str, dict] = defaultdict(dict)
        self._last_flush = time.monotonic()

    def _shutdown(self, signum, frame) -> None:
        log.info("Shutdown signal — flushing and stopping")
        self._running = False

    def run(self) -> None:
        self._consumer.subscribe([RAW_TOPIC])
        log.info("Consuming from: %s (group=%s)", RAW_TOPIC, CONSUMER_GROUP)

        try:
            while self._running:
                msg = self._consumer.poll(timeout=0.5)
                if msg is not None:
                    if msg.error():
                        if msg.error().code() != KafkaError._PARTITION_EOF:
                            raise KafkaException(msg.error())
                    else:
                        self._handle(msg)

                elapsed = time.monotonic() - self._last_flush
                if len(self._batch) >= BATCH_SIZE or elapsed >= FLUSH_INTERVAL_S:
                    self._flush()
                    self._consumer.commit(asynchronous=False)
        finally:
            self._flush()
            self._consumer.close()
            log.info("Kafka consumer closed")

    def _handle(self, msg) -> None:
        try:
            envelope = json.loads(msg.value().decode("utf-8"))
            vin = envelope.get("vin", "UNKNOWN")
            channel_id = envelope.get("channel_id", 0)

            point = _build_point(envelope)
            self._batch.append(point)

            # Merge into PG state buffer (flat key = channel:field)
            for k, v in envelope.get("payload", {}).items():
                if not k.endswith("_invalid"):
                    self._state_buffer[vin][k] = v

        except Exception as exc:
            log.error("Message processing error (offset=%s): %s", msg.offset(), exc)

    def _flush(self) -> None:
        if self._batch:
            write_influx_batch(self._batch)
            log.debug("Flushed %d InfluxDB points", len(self._batch))
            self._batch.clear()

        for vin, values in self._state_buffer.items():
            write_postgres_state(vin, values)
        self._state_buffer.clear()

        self._last_flush = time.monotonic()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    TelemetryKafkaConsumer().run()
