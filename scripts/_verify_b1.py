import sys
sys.path.insert(0, ".")

from features.derived_utils import (
    detect_harsh_brake, compute_brake_stress_index,
    compute_battery_12v_health_score, correct_tyre_pressure_for_temp,
)
import pandas as pd

# BSI formula test
df = pd.DataFrame({
    "vin":        ["V1"],
    "timestamp":  [pd.Timestamp("2024-01-01")],
    "vehBrakePos":[175],
    "tboxAccelX": [-75],
    "vehSpeed":   [800],
})
df = compute_brake_stress_index(df)
expected = 70.0 * (80.0 / 100.0) ** 2
assert abs(df["bsi"].iloc[0] - expected) < 0.01, f"BSI wrong: {df['bsi'].iloc[0]}"

# 12V health score extremes
v = compute_battery_12v_health_score(12.6, 0.0, 11.5, 1.0)
assert v > 80, f"Healthy 12V score should be > 80, got {v}"

v2 = compute_battery_12v_health_score(11.7, -0.02, 9.0, 5.5)
assert v2 < 20, f"Degraded 12V score should be < 20, got {v2}"

# Pressure temperature correction
c = correct_tyre_pressure_for_temp(240.0, 40.0, 25.0)
assert c < 240.0, f"Hot tyre pressure should correct downward, got {c}"

print("derived_utils: ALL PASS")
