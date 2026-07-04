import sys
sys.path.insert(0, ".")

from ingestion.dtc_processor import DTCProcessor

d = DTCProcessor()
v = d.get_fault_system("P0562")
assert v == "Powertrain", f"P0562 got {v}"

v = d.get_fault_system("C0040")
assert v == "Chassis", f"C0040 got {v}"

v = d.get_fault_system("U0100")
assert v == "Network", f"U0100 got {v}"

print("BLOCK 2 PASS — all DTCProcessor assertions OK")
