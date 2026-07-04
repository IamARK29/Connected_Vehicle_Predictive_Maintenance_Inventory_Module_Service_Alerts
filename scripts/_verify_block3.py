import sys
sys.path.insert(0, ".")

from ingestion.telemetry_tier_router import TelemetryTierRouter

r = TelemetryTierRouter()

assert "telemetry.hf" in r.route(3),        f"CH3: {r.route(3)}"
assert "telemetry.hf" in r.route(4),        f"CH4: {r.route(4)}"
assert "telemetry.standard" in r.route(15), f"CH15: {r.route(15)}"
assert "telemetry.lf" in r.route(1),        f"CH1: {r.route(1)}"
assert r.get_influx_bucket(3) == "tbox_hf",          f"bucket CH3: {r.get_influx_bucket(3)}"
assert r.get_influx_bucket(15) == "tbox_standard",   f"bucket CH15: {r.get_influx_bucket(15)}"

print("BLOCK 3 PASS — all TelemetryTierRouter assertions OK")
