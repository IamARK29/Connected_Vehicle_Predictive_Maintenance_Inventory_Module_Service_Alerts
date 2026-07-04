"""
Synthetic Data Loader — Mode C auto-load.

Scans data/synthetic/ for generated CSV files and loads them into the
platform using FileIngestor. Designed to run after synthetic generation.

CLI usage:
    python -m ingestion.synthetic_loader
    python -m ingestion.synthetic_loader --data-dir data/synthetic/
    python -m ingestion.synthetic_loader --data-dir data/synthetic/ --types telemetry trips
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)


def load_synthetic_to_db(
    data_dir: str | Path = "data/synthetic/",
    types: list[str] | None = None,
    verbose: bool = True,
) -> dict:
    """
    Scan data_dir for synthetic CSV files and load them into InfluxDB/PostgreSQL.

    File name patterns:
      telemetry_*.csv  → InfluxDB via ingest_telemetry_csv
      trips_*.csv      → PostgreSQL trips table
      service_*.csv    → PostgreSQL service_history table

    Returns a summary dict with counts per type.
    """
    from ingestion.file_ingestor import FileIngestor

    data_path = Path(data_dir)
    if not data_path.exists():
        log.error("Data directory does not exist: %s", data_path)
        return {"error": f"Directory not found: {data_path}"}

    types_to_load = set(types or ["telemetry", "trips", "service"])
    ingestor = FileIngestor()
    summary: dict[str, dict] = {}

    patterns = {
        "telemetry": ("telemetry_*.csv", ingestor.ingest_telemetry_csv),
        "trips":     ("trips_*.csv",     ingestor.ingest_trip_csv),
        "service":   ("service_*.csv",   ingestor.ingest_service_history_csv),
    }

    for dtype, (glob_pat, ingest_fn) in patterns.items():
        if dtype not in types_to_load:
            continue

        files = sorted(data_path.glob(glob_pat))
        if not files:
            log.info("[%s] No files found matching %s in %s", dtype, glob_pat, data_path)
            summary[dtype] = {"files": 0, "uploaded": 0, "failed": 0}
            continue

        total_up = total_fail = 0
        for f in files:
            if verbose:
                print(f"  Loading {dtype}: {f.name} ...", end=" ", flush=True)
            result = ingest_fn(f)
            total_up   += result.get("uploaded", 0)
            total_fail += result.get("failed", 0)
            if verbose:
                print(f"OK {result.get('uploaded', 0)} rows  ({result.get('failed', 0)} failed)")
            if result.get("errors"):
                for err in result["errors"][:5]:
                    log.warning("  %s", err)

        summary[dtype] = {"files": len(files), "uploaded": total_up, "failed": total_fail}

    return summary


def _run_cli() -> None:
    parser = argparse.ArgumentParser(description="AutoPredict Synthetic Data Loader")
    parser.add_argument("--data-dir", default="data/synthetic/", help="Directory containing synthetic CSV files")
    parser.add_argument(
        "--types", nargs="+", choices=["telemetry", "trips", "service"],
        default=["telemetry", "trips", "service"],
        help="Which file types to load (default: all)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress per-file output")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print(f"\nAutoPredict Synthetic Data Loader")
    print(f"  Source : {Path(args.data_dir).resolve()}")
    print(f"  Types  : {', '.join(args.types)}\n")

    summary = load_synthetic_to_db(args.data_dir, args.types, verbose=not args.quiet)

    print("\n-- Summary ----------------------------------")
    for dtype, counts in summary.items():
        if "error" in counts:
            print(f"  {dtype:<12} ERROR: {counts['error']}")
        else:
            print(f"  {dtype:<12} files={counts['files']}  loaded={counts['uploaded']}  failed={counts['failed']}")
    print()


if __name__ == "__main__":
    _run_cli()
