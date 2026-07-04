"""
AutoPredict Synthetic Data Orchestrator.

Runs all four generators in sequence:
  1. fleet      → data/synthetic/fleet_master.csv
  2. telemetry  → data/synthetic/telemetry_{vin}.csv  (per VIN)
                  data/synthetic/telemetry_combined.csv
                  data/synthetic/failures_manifest.csv
  3. trips      → data/synthetic/trips.csv
  4. service    → data/synthetic/service_history.csv

Usage:
    python -m synthetic
    python -m synthetic --vehicles 100 --days 365
    python -m synthetic --skip fleet trips
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from synthetic.config import SyntheticConfig


def _banner(title: str) -> None:
    print(f"\n{'-' * 60}")
    print(f"  {title}")
    print(f"{'-' * 60}")


def run(cfg: SyntheticConfig, skip: set[str], data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)

    fleet_df = None

    # ── 1. Fleet ─────────────────────────────────────────────────────────────
    if "fleet" not in skip:
        _banner("Step 1 / 4 — Fleet master")
        t0 = time.perf_counter()
        from synthetic.generate_fleet import generate_fleet
        fleet_df = generate_fleet(cfg)
        print(f"  Done in {time.perf_counter() - t0:.1f}s")
    else:
        fleet_path = data_dir / "fleet_master.csv"
        if fleet_path.exists():
            import pandas as pd
            fleet_df = pd.read_csv(fleet_path)
            print(f"[fleet] Skipped — loaded {len(fleet_df)} vehicles from {fleet_path.name}")
        else:
            print("[fleet] Skipped but fleet_master.csv not found — aborting")
            return

    # ── 2. Telemetry ──────────────────────────────────────────────────────────
    if "telemetry" not in skip:
        _banner("Step 2 / 4 — Telemetry generation")
        t0 = time.perf_counter()
        from synthetic.generate_telemetry import TelemetryGenerator
        gen = TelemetryGenerator(cfg)
        gen.generate_all(fleet_df, data_dir)
        print(f"  Done in {time.perf_counter() - t0:.1f}s")
    else:
        print("[telemetry] Skipped")

    # ── 3. Trips ──────────────────────────────────────────────────────────────
    if "trips" not in skip:
        _banner("Step 3 / 4 — Trip aggregation")
        t0 = time.perf_counter()
        from synthetic.generate_trips import generate_trips
        generate_trips(fleet_df=fleet_df, cfg=cfg, data_dir=data_dir)
        print(f"  Done in {time.perf_counter() - t0:.1f}s")
    else:
        print("[trips] Skipped")

    # ── 4. Service history ────────────────────────────────────────────────────
    if "service" not in skip:
        _banner("Step 4 / 4 — Service history")
        t0 = time.perf_counter()
        from synthetic.generate_service_history import generate_service_history
        generate_service_history(fleet_df=fleet_df, cfg=cfg, data_dir=data_dir)
        print(f"  Done in {time.perf_counter() - t0:.1f}s")
    else:
        print("[service] Skipped")

    _banner("Complete")
    print(f"  Output: {data_dir.resolve()}")
    for f in sorted(data_dir.glob("*.csv")):
        size_mb = f.stat().st_size / 1_048_576
        print(f"    {f.name:<45} {size_mb:>7.2f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AutoPredict Synthetic Data Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--vehicles", type=int, default=50,    help="Number of vehicles")
    parser.add_argument("--days",     type=int, default=180,   help="Simulation days")
    parser.add_argument("--start",    type=str, default="2024-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--seed",     type=int, default=42,    help="Random seed")
    parser.add_argument("--data-dir", type=str, default="data/synthetic", help="Output directory")
    parser.add_argument(
        "--skip", nargs="*",
        choices=["fleet", "telemetry", "trips", "service"],
        default=[],
        help="Steps to skip (re-uses existing CSVs)",
    )
    args = parser.parse_args()

    cfg = SyntheticConfig(
        num_vehicles=args.vehicles,
        num_days=args.days,
        start_date=args.start,
        seed=args.seed,
    )

    print(f"\nAutoPredict Synthetic Data Generator")
    print(f"  Vehicles : {cfg.num_vehicles}")
    print(f"  Days     : {cfg.num_days}")
    print(f"  Start    : {cfg.start_date}")
    print(f"  Seed     : {cfg.seed}")
    print(f"  Output   : {Path(args.data_dir).resolve()}")
    if args.skip:
        print(f"  Skipping : {', '.join(args.skip)}")

    run(cfg, skip=set(args.skip or []), data_dir=Path(args.data_dir))


if __name__ == "__main__":
    main()
