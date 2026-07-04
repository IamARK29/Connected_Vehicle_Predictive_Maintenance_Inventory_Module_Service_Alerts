"""
Telemetry Tier Router — maps TBox channel IDs to Kafka topics and InfluxDB buckets.

Tiers:
  hf       → channels 3, 4          (10 Hz drive data)
  standard → channels 5-7, 9-16, 19-23 (1 Hz vehicle state)
  lf       → channels 1, 2, 8, 18   (low-frequency / config)
  dtc      → channel 17              (diagnostic trouble codes)
"""
from __future__ import annotations


class TelemetryTierRouter:

    TIER_MAP: dict[str, set[int]] = {
        "hf":       {3, 4},
        "standard": {5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 19, 20, 21, 22, 23},
        "lf":       {1, 2, 8, 18},
        "dtc":      {17},
    }

    # CH-17 also goes to standard so the alert engine can see airbag events
    _EXTRA_TOPICS: dict[int, list[str]] = {
        17: ["telemetry.standard"],
    }

    _BUCKET_MAP: dict[str, str] = {
        "hf":       "tbox_hf",
        "standard": "tbox_standard",
        "lf":       "tbox_lf",
        "dtc":      "tbox_dtc",
    }

    def _tier_for(self, channel_id: int) -> str | None:
        for tier, channels in self.TIER_MAP.items():
            if channel_id in channels:
                return tier
        return None

    def route(self, channel_id: int) -> list[str]:
        """
        Return Kafka topic list for a channel.

        CH-17 routes to both telemetry.dtc AND telemetry.standard.
        Unknown channels fall back to telemetry.standard.
        """
        tier = self._tier_for(channel_id)
        if tier is None:
            return ["telemetry.standard"]

        topics = [f"telemetry.{tier}"]
        extra  = self._EXTRA_TOPICS.get(channel_id, [])
        for t in extra:
            if t not in topics:
                topics.append(t)
        return topics

    def get_influx_bucket(self, channel_id: int) -> str:
        tier = self._tier_for(channel_id)
        if tier is None:
            return "tbox_standard"
        return self._BUCKET_MAP[tier]
