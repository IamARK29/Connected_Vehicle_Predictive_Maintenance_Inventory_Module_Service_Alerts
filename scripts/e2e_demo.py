#!/usr/bin/env python3
"""
AutoPredict End-to-End Demo
===========================
Walks through all 10 platform capabilities in sequence against a live API server.

Usage:
    python scripts/e2e_demo.py [--base-url http://localhost:8000]

Prerequisites:
    - docker compose up -d (or API + backing services running locally)
    - pip install requests websockets
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from typing import Any

try:
    import requests
except ImportError:
    sys.exit("Install requests first:  pip install requests")

# ── Colour output ──────────────────────────────────────────────────────────────

RESET  = "\033[0m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"


def _ok(msg: str)   -> None: print(f"  {GREEN}✓{RESET} {msg}")
def _warn(msg: str) -> None: print(f"  {YELLOW}⚠{RESET} {msg}")
def _err(msg: str)  -> None: print(f"  {RED}✗{RESET} {msg}")
def _hdr(n: int, title: str) -> None:
    print(f"\n{BOLD}{CYAN}Step {n:02d}/{TOTAL_STEPS}: {title}{RESET}")
    print("─" * 60)


TOTAL_STEPS = 10
DEMO_VIN    = "MH01MZ7X0001"


class AutoPredictDemo:

    def __init__(self, base_url: str) -> None:
        self.base = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        self.token: str | None = None
        self.job_id: str | None = None
        self.results: list[dict] = []

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _get(self, path: str, **kwargs) -> requests.Response:
        return self.session.get(f"{self.base}{path}", **kwargs)

    def _post(self, path: str, data: Any = None, **kwargs) -> requests.Response:
        return self.session.post(f"{self.base}{path}", json=data, **kwargs)

    def _record(self, step: int, name: str, passed: bool, detail: str = "") -> None:
        self.results.append({"step": step, "name": name, "passed": passed, "detail": detail})
        if passed:
            _ok(f"{name}: {detail}" if detail else name)
        else:
            _err(f"{name}: {detail}" if detail else name)

    # ── Steps ──────────────────────────────────────────────────────────────────

    def step01_health_check(self) -> None:
        _hdr(1, "Health Check")
        try:
            r = self._get("/health", timeout=5)
            body = r.json()
            self._record(1, "HTTP 200", r.status_code == 200, f"status={body.get('status')}")
            for svc, status in body.get("checks", {}).items():
                if status == "ok":
                    _ok(f"{svc}: reachable")
                else:
                    _warn(f"{svc}: {status}")
        except Exception as exc:
            self._record(1, "Health check", False, str(exc))

    def step02_authenticate(self) -> None:
        _hdr(2, "JWT Authentication")
        try:
            r = self._post("/api/auth/token", {"username": "admin", "password": "admin123"})
            body = r.json()
            ok = r.status_code == 200 and "access_token" in body
            self._record(2, "Admin login", ok, f"role={body.get('role')}")
            if ok:
                self.token = body["access_token"]
                self.session.headers["Authorization"] = f"Bearer {self.token}"

            # Also test dealer
            r2 = self._post("/api/auth/token", {"username": "dealer", "password": "dealer123"})
            b2 = r2.json()
            self._record(2, "Dealer login", r2.status_code == 200, f"role={b2.get('role')}")
        except Exception as exc:
            self._record(2, "Authentication", False, str(exc))

    def step03_generate_synthetic_data(self) -> None:
        _hdr(3, "Generate Synthetic Fleet Dataset")
        try:
            r = self._post("/api/synthetic/generate", {
                "num_vehicles": 5,
                "num_days":     30,
                "failure_rate": 0.10,
            })
            body = r.json()
            ok = r.status_code == 202 and "job_id" in body
            self._record(3, "POST /api/synthetic/generate", ok, f"job_id={body.get('job_id', 'N/A')[:8]}…")
            if ok:
                self.job_id = body["job_id"]
                print(f"    Poll URL : {self.base}{body.get('poll_url')}")
                print(f"    WS URL   : ws://…{body.get('ws')}")

                # Poll for completion (max 120 s)
                poll_url = f"/api/upload/status/{self.job_id}"
                deadline = time.time() + 120
                while time.time() < deadline:
                    pr = self._get(poll_url, timeout=5)
                    if pr.status_code == 200:
                        pb = pr.json()
                        pct = pb.get("pct", 0)
                        msg = pb.get("message", "")
                        print(f"    [{pct:3d}%] {msg}", end="\r", flush=True)
                        if pct >= 100:
                            print()
                            _ok("Generation complete")
                            break
                    time.sleep(3)
                else:
                    _warn("Generation still running (exceeded 120 s poll window)")
        except Exception as exc:
            self._record(3, "Synthetic generation", False, str(exc))

    def step04_fleet_health_overview(self) -> None:
        _hdr(4, "Fleet Health Overview")
        try:
            r = self._get("/api/fleet/health", timeout=5)
            body = r.json()
            ok = r.status_code == 200
            self._record(4, "GET /api/fleet/health", ok)
            if ok and isinstance(body, dict):
                total  = body.get("total_vehicles", "?")
                online = body.get("online_now", "?")
                crit   = body.get("active_alerts_critical", "?")
                score  = body.get("fleet_avg_health_score", "?")
                print(f"    Total vehicles : {total}")
                print(f"    Online now     : {online}")
                print(f"    Critical alerts: {crit}")
                print(f"    Avg health score: {score}")
        except Exception as exc:
            self._record(4, "Fleet health", False, str(exc))

    def step05_vehicle_telemetry_and_predictions(self) -> None:
        _hdr(5, f"Vehicle Telemetry & ML Predictions ({DEMO_VIN})")
        try:
            # vehicle details
            r = self._get(f"/api/vehicles/{DEMO_VIN}", timeout=5)
            self._record(5, f"GET /api/vehicles/{DEMO_VIN}", r.status_code in (200, 404))

            # ML predictions
            r2 = self._get(f"/api/vehicles/{DEMO_VIN}/predictions", timeout=10)
            ok = r2.status_code in (200, 404)
            self._record(5, "GET /api/vehicles/{vin}/predictions", ok)
            if r2.status_code == 200:
                preds = r2.json()
                if isinstance(preds, dict):
                    for model, result in list(preds.items())[:3]:
                        sev = result.get("severity", "?")
                        print(f"    {model:<20s}: {sev}")
        except Exception as exc:
            self._record(5, "Vehicle telemetry", False, str(exc))

    def step06_rule_based_alerts(self) -> None:
        _hdr(6, "Rule-Based Alert Engine (Direct)")
        try:
            from alerts.rule_engine import RuleEngine
            engine = RuleEngine()

            # Inject a thermal runaway fault
            state = {
                "vehBMSPackTemFlt": 3,
                "VehBatt": 12.6,
                "VehCoolantTemp": 90,
                "BrakePadFrontMM": 8.0,
                "BrakePadRearMM": 8.0,
                "BrakeFluidPct": 95,
            }
            alerts = engine.evaluate(DEMO_VIN, state)
            critical = [a for a in alerts if a.severity == "CRITICAL"]
            self._record(6, "THERMAL_RUNAWAY fires", bool(critical), f"{len(critical)} CRITICAL alert(s)")
            for a in critical[:2]:
                print(f"    [{a.severity}] {a.alert_type}: {a.title}")

            # Clean state → no CRITICAL
            clean_alerts = engine.evaluate(DEMO_VIN, {"VehBatt": 12.6, "VehCoolantTemp": 90})
            no_crit = not any(a.severity == "CRITICAL" for a in clean_alerts)
            self._record(6, "Clean state has no CRITICAL", no_crit)
        except Exception as exc:
            self._record(6, "Rule engine", False, str(exc))

    def step07_dealer_operations(self) -> None:
        _hdr(7, "Dealer Portal Operations")
        endpoints = [
            ("/api/dealer/DL001/bay-status",    "Bay status"),
            ("/api/dealer/DL001/appointments",  "Appointments"),
            ("/api/dealer/DL001/inventory",     "Parts inventory"),
            ("/api/dealer/DL001/demand-forecast", "Demand forecast"),
        ]
        for path, label in endpoints:
            try:
                r = self._get(path, timeout=5)
                self._record(7, label, r.status_code not in (401, 403), f"HTTP {r.status_code}")
            except Exception as exc:
                self._record(7, label, False, str(exc))

    def step08_ai_agent_workflow(self) -> None:
        _hdr(8, "AI Agent Workflow")
        try:
            # Trigger a workflow
            r = self._post(f"/api/agent/trigger/{DEMO_VIN}", timeout=10)
            self._record(8, f"POST /api/agent/trigger/{DEMO_VIN}", r.status_code not in (401, 403), f"HTTP {r.status_code}")

            # List active workflows
            r2 = self._get("/api/agent/workflows", timeout=5)
            ok2 = r2.status_code == 200
            self._record(8, "GET /api/agent/workflows", ok2)
            if ok2:
                data = r2.json()
                wfs  = data.get("workflows", data) if isinstance(data, dict) else data
                count = len(wfs) if isinstance(wfs, list) else "?"
                print(f"    Active workflows: {count}")

            # Chat
            r3 = self._post("/api/agent/chat", {
                "message": f"What is the battery health of vehicle {DEMO_VIN}?",
                "vin": DEMO_VIN,
            }, timeout=30)
            self._record(8, "POST /api/agent/chat", r3.status_code not in (401, 403), f"HTTP {r3.status_code}")
        except Exception as exc:
            self._record(8, "AI agent", False, str(exc))

    def step09_data_ingestion_csv_upload(self) -> None:
        _hdr(9, "CSV Upload Ingestion")
        import io, csv as _csv
        rows = [
            {"StartTime-TimeStamp": "2024-06-01 08:00:00", "VIN": DEMO_VIN,
             "vehSpeed": 60, "vehEngineTemp": 90, "vehHvSoc": 75, "vehBattVolt": 12.6},
            {"StartTime-TimeStamp": "2024-06-01 09:00:00", "VIN": DEMO_VIN,
             "vehSpeed": 80, "vehEngineTemp": 92, "vehHvSoc": 70, "vehBattVolt": 12.5},
        ]
        buf = io.StringIO()
        writer = _csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        csv_bytes = buf.getvalue().encode()

        try:
            r = self.session.post(
                f"{self.base}/api/upload/telemetry",
                files={"file": ("demo_telemetry.csv", csv_bytes, "text/csv")},
                timeout=15,
            )
            self._record(9, "POST /api/upload/telemetry (CSV)", r.status_code not in (401, 403), f"HTTP {r.status_code}")
        except Exception as exc:
            self._record(9, "CSV upload", False, str(exc))

    def step10_driver_scores_and_maintenance_calendar(self) -> None:
        _hdr(10, "Driver Scores + Maintenance Calendar")
        try:
            r = self._get("/api/fleet/driver-scores", timeout=5)
            self._record(10, "GET /api/fleet/driver-scores", r.status_code not in (401, 403), f"HTTP {r.status_code}")
            if r.status_code == 200:
                data = r.json()
                drivers = data.get("drivers", data) if isinstance(data, dict) else data
                if isinstance(drivers, list) and drivers:
                    top = drivers[0]
                    print(f"    Top driver: {top.get('vin', '?')} — score {top.get('score', '?'):.1f}" if isinstance(top.get("score"), float) else f"    Top entry: {top}")

            r2 = self._get("/api/fleet/maintenance-calendar?days=90", timeout=5)
            self._record(10, "GET /api/fleet/maintenance-calendar", r2.status_code not in (401, 403), f"HTTP {r2.status_code}")
        except Exception as exc:
            self._record(10, "Driver scores / calendar", False, str(exc))

    # ── Run all steps ──────────────────────────────────────────────────────────

    def run(self) -> bool:
        print(f"\n{BOLD}AutoPredict — End-to-End Demo{RESET}")
        print(f"Target : {self.base}")
        print(f"Time   : {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n")

        self.step01_health_check()
        self.step02_authenticate()
        self.step03_generate_synthetic_data()
        self.step04_fleet_health_overview()
        self.step05_vehicle_telemetry_and_predictions()
        self.step06_rule_based_alerts()
        self.step07_dealer_operations()
        self.step08_ai_agent_workflow()
        self.step09_data_ingestion_csv_upload()
        self.step10_driver_scores_and_maintenance_calendar()

        # ── Summary ────────────────────────────────────────────────────────────
        passed = sum(1 for r in self.results if r["passed"])
        total  = len(self.results)
        pct    = int(100 * passed / total) if total else 0
        colour = GREEN if pct >= 80 else YELLOW if pct >= 60 else RED

        print(f"\n{'═' * 60}")
        print(f"{BOLD}Demo Summary{RESET}")
        print(f"{'═' * 60}")
        print(f"  Checks passed : {colour}{passed}/{total}{RESET} ({pct}%)")

        failed = [r for r in self.results if not r["passed"]]
        if failed:
            print(f"\n  {RED}Failed checks:{RESET}")
            for r in failed:
                print(f"    Step {r['step']:02d}: {r['name']} — {r['detail']}")

        print()
        return pct == 100


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoPredict end-to-end demo")
    parser.add_argument("--base-url", default="http://localhost:8000",
                        help="API base URL (default: http://localhost:8000)")
    args = parser.parse_args()

    demo = AutoPredictDemo(args.base_url)
    success = demo.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
