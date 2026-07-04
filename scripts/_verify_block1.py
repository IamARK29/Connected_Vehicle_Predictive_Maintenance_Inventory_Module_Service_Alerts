import sys
sys.path.insert(0, ".")

from ingestion.signal_registry import SignalDecoder

v = SignalDecoder.decode("vehSpeed", 600)
assert v == 60.0, f"vehSpeed got {v}"

v = SignalDecoder.decode("vehBatt", 145)
assert v == 14.5, f"vehBatt got {v}"

v = SignalDecoder.decode("tboxAccelX", -75)
assert v == -0.3, f"tboxAccelX got {v}"

v = SignalDecoder.decode("vehBMSCellMaxTem", 87)
assert v == 3.5, f"vehBMSCellMaxTem got {v}"

raw = 10004
expected = round(10004 * 0.05 - 1000, 6)
v = SignalDecoder.decode("vehBMSPackCrnt", raw)
assert v == expected, f"vehBMSPackCrnt got {v}, expected {expected}"

v = SignalDecoder.decode("frontLeftTyrePressure", 128)
assert v is None, f"frontLeftTyrePressure sentinel got {v}"

v = SignalDecoder.decode("vehSpeed", 9999)
assert v is None, f"vehSpeed OOR got {v}"

row = {"vehBMSPackSOC": 800, "vehBMSPackSOCV": 1}
decoded = SignalDecoder.decode_row(row)
assert decoded["vehBMSPackSOC"] is None, f"validity gate: {decoded}"

v = SignalDecoder.decode("vehEPTTrInptShaftToq", 1696)
assert v == 0.0, f"torque got {v}"

print("BLOCK 1 PASS — all SignalDecoder assertions OK")
