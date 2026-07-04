# AutoPredict — Automotive Predictive Maintenance Platform

> AI-powered vehicle health monitoring, fault prediction, and service orchestration.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                     │
│   [Vehicle TBox] ──MQTT──► [Kafka Topic: tbox.telemetry]                    │
│   [CSV Upload]  ──HTTP──►  [FastAPI Upload API]                              │
│   [Synthetic]   ──Script►  [CSV → InfluxDB + PostgreSQL]                    │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
              ┌────────────────────▼───────────────────────┐
              │             FastAPI Backend (v2.0)          │
              │  ┌─────────┐ ┌──────────┐ ┌────────────┐   │
              │  │Vehicles │ │  Fleet   │ │   Dealer   │   │
              │  │  /api   │ │  /api    │ │   /api     │   │
              │  └────┬────┘ └────┬─────┘ └─────┬──────┘   │
              │       │           │              │           │
              │  ┌────▼───────────▼──────────────▼──────┐   │
              │  │         Core Services Layer            │   │
              │  │  ┌──────────────┐ ┌────────────────┐  │   │
              │  │  │ Rule Engine  │ │ ML Predictions │  │   │
              │  │  │ (24 rules)   │ │ (7 models)     │  │   │
              │  │  └──────┬───────┘ └───────┬────────┘  │   │
              │  │         │                 │            │   │
              │  │  ┌──────▼─────────────────▼──────┐    │   │
              │  │  │     Feature Engineering        │    │   │
              │  │  │  brake│engine│HVbatt│12V│tyre  │    │   │
              │  │  └──────────────────────────────┘    │   │
              │  └────────────────────────────────────────┘   │
              │                                               │
              │  ┌────────────────────────────────────────┐   │
              │  │       AI Agent Workflow Engine          │   │
              │  │  Trigger → Diagnose → Schedule → Notify│   │
              │  └────────────────────────────────────────┘   │
              │                                               │
              │  WebSocket: /ws/live/{vin} │ /ws/alerts       │
              └───────────────────────────────────────────────┘
                         │                   │
          ┌──────────────▼──┐      ┌─────────▼──────────┐
          │   InfluxDB 2.x  │      │   PostgreSQL 15     │
          │  (time-series   │      │ (vehicles, alerts,  │
          │   telemetry)    │      │  appointments, inv) │
          └─────────────────┘      └────────────────────┘
                         │
          ┌──────────────▼──────────────────────┐
          │         Dealer Portal (React)         │
          │  Dashboard │ VehicleDetail │ Alerts   │
          │  ServiceBay│ Inventory     │ Workflows │
          │  DriverScores │ Upload              │
          └──────────────────────────────────────┘
```

---

## Quickstart

### Prerequisites

- Docker 24+ and Docker Compose v2
- Python 3.13 (for local development)
- Node.js 20+ (for dealer portal development)

### 1. Clone and configure

```bash
git clone https://github.com/autopredict.git
cd autopredict
cp .env.example .env
# Edit .env — set SECRET_KEY, GEMINI_API_KEY (for agent), and optionals
```

### 2. Start all services

```bash
make up
# or: docker compose up -d
```

Services started:
- `http://localhost:8000` — FastAPI backend + docs at `/docs`
- `http://localhost:8086` — InfluxDB (UI)
- `http://localhost:5432` — PostgreSQL
- `http://localhost:6379` — Redis
- `http://localhost:5000` — MLflow
- `http://localhost:5173` — Dealer portal (dev server)

### 3. Generate synthetic data

```bash
make generate
# or: python scripts/e2e_demo.py --step generate
```

### 4. Train models

```bash
make train
# or: python models/train_all.py
```

### 5. Run the e2e demo

```bash
make demo
# Walks through all 10 platform capabilities with live output
```

### 6. Run tests

```bash
make test
# or: pytest tests/ -v
```

---

## Data Ingestion Modes

### Mode A — TBox MQTT (Real-Time)

The TBox unit in each vehicle publishes JSON telemetry to an MQTT broker every 30 seconds.

```
Topic pattern : autopredict/telemetry/{VIN}/{channel_id}
Broker default: localhost:1883
```

The API server auto-subscribes on startup via `ingestion/tbox_receiver.py`. Configure the broker in `.env`:

```env
MQTT_BROKER_HOST=mqtt.-motor.in
MQTT_BROKER_PORT=1883
MQTT_USERNAME=tbox_ingest
MQTT_PASSWORD=<secret>
```

### Mode B — CSV Bulk Upload

Upload historical telemetry, trip, or service history files via the dealer portal or API:

```bash
curl -X POST http://localhost:8000/api/upload/telemetry \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@data/fleet_telemetry.csv"
```

Expected columns (telemetry): `StartTime-TimeStamp`, `VIN`, `vehSpeed`, `vehEngineTemp`, `vehHvSoc`, `vehBattVolt`, …

### Mode C — Synthetic Data Generation

For development and testing, generate a synthetic fleet dataset:

```bash
# Via API (async, tracked by job_id):
curl -X POST http://localhost:8000/api/synthetic/generate \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"num_vehicles": 20, "num_days": 90, "failure_rate": 0.08}'

# Via CLI:
python -m synthetic.generate_fleet --num-vehicles 20
python -m synthetic.generate_telemetry --num-days 90 --failure-rate 0.08
```

---

## Channel Validators

The platform validates 23 TBox telemetry channels on ingestion:

| CH | Description | Key Fields |
|----|-------------|------------|
| 1 | GNSS Position | gnssTime, gnssLong, gnssLat |
| 2 | GNSS Quality | altitude, gnssSats, hdop, gpsStatus |
| 3 | Vehicle Dynamics | vehSpeed (÷10=kph), vehRPM, vehSysPwrMod |
| 4 | Accelerometers + Pedals | tboxAccelX/Y/Z, vehAccelPos, vehBrakePos |
| 5 | Doors | driverDoorAjar, passDoorAjar, … |
| 6 | Windows | driverWindowPos, sunroofPos, … |
| 7 | Vehicle Status | ignitionStatus, seatbelt*, parkingBrake |
| 8 | Cruise Control | cruiseActive, cruiseSetSpeed |
| 9 | Temperatures | ambientAirTemp, coolantTemp, engineOilTemp |
| 10 | HVAC | hvacMode, hvacFanSpeed, hvacSetTemp |
| 11 | Lights | headLightsOn, highBeamOn, fogFront/Rear |
| 12 | Rain / Night | rainSensorActive, wiperFrontSpeed |
| 13 | Vehicle General | odometer, fuelLevel, battVoltage12V |
| 14 | Horn | hornActive |
| 15 | MIL + Safety | milActive, milDtcCodes, absActive, tcsActive |
| 16 | Seat Belts (detail) | seatbelt* × 5 positions |
| 17 | Airbag | airbagDeployedAny, airbagDriver/Pass/Side* |
| 18 | Network | imei, iccid, networkType, signalStrength |
| 19 | HV Battery Pack | vehPackVol (÷0.25=V), vehPackSOC, vehPackSOH |
| 20 | Charging | chargePlugConnected, chargeMode, chargeVoltage |
| 21 | EV RVM / BMS + Motor | vehBMSPack*, vehTMSpeed/Torque/Temp, validity flags |
| 22 | Thermal Runaway | thermalRunawayLevel (0-3), thermalRunawayActive |
| 23 | Tyres | tyrePressureFL/FR/RL/RR (1-128), wheelTyreMonitorStatus |

---

## ML Models

| Model | Algorithm | Target | Trigger |
|-------|-----------|--------|---------|
| `brake_wear` | XGBoost + Ridge | Days to brake replacement | brake_stress, harsh_brake_rate |
| `engine_oil` | XGBoost + IsolationForest | Oil change urgency | oil_life_pct, rpm_mean |
| `hv_battery_soh` | Linear + LogReg + IsoForest | SOH% decline | soh_trend, cell_temp_max |
| `battery_12v` | Linear + XGBoost | 12V failure probability | voltage_drop_rate, off_voltage |
| `tyre_wear` | LightGBM + LogReg | Tyre replacement days | pressure_trend, temp_variance |
| `fuel_anomaly` | IsolationForest | Fuel theft / leak | fuel_drop_rate, idle_consumption |
| `driver_score` | Ridge + LogReg | Score 0-100, risk class | harsh_brake/accel rates, speeding |

### Train all models

```bash
make train
# Individual:
python models/train_all.py --model brake_wear
python models/train_all.py --model engine_oil
```

---

## API Reference

Base URL: `http://localhost:8000`  
Interactive docs: `http://localhost:8000/docs`

### Authentication

```http
POST /api/auth/token
Content-Type: application/json

{"username": "admin", "password": "admin123"}
```

Response: `{"access_token": "...", "token_type": "bearer", "role": "ADMIN"}`

All subsequent requests require: `Authorization: Bearer <token>`

Demo users:
- `admin / admin123` (role: ADMIN)
- `dealer / dealer123` (role: DEALER)

### Key Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service health (postgres, influxdb, redis) |
| GET | `/api/vehicles` | List vehicles (filter by dealer_code, fuel_type) |
| GET | `/api/vehicles/{vin}` | Vehicle detail + latest telemetry |
| GET | `/api/vehicles/{vin}/predictions` | All ML model predictions |
| GET | `/api/vehicles/{vin}/alerts` | Active alerts for VIN |
| GET | `/api/vehicles/{vin}/service-history` | Past service records |
| GET | `/api/fleet/health` | Fleet-wide KPI summary |
| GET | `/api/fleet/alerts` | Fleet alerts (filter by severity, hours) |
| GET | `/api/fleet/driver-scores` | Driver behaviour leaderboard |
| GET | `/api/fleet/maintenance-calendar` | Upcoming service schedule |
| GET | `/api/dealer/{code}/bay-status` | Workshop bay occupancy |
| GET | `/api/dealer/{code}/appointments` | Upcoming appointments |
| POST | `/api/dealer/{code}/appointments` | Book appointment |
| GET | `/api/dealer/{code}/inventory` | Parts inventory |
| GET | `/api/dealer/{code}/demand-forecast` | 30/90-day parts demand |
| POST | `/api/agent/trigger/{vin}` | Trigger AI service workflow |
| GET | `/api/agent/workflows` | List active workflows |
| POST | `/api/agent/chat` | Conversational AI query |
| POST | `/api/upload/telemetry` | Upload telemetry CSV |
| POST | `/api/upload/trips` | Upload trip CSV |
| POST | `/api/upload/service-history` | Upload service history CSV |
| GET | `/api/upload/status/{job_id}` | Job progress (0-100%) |
| POST | `/api/synthetic/generate` | Generate synthetic fleet dataset |
| WS | `/ws/live/{vin}` | Real-time telemetry stream (2s poll) |
| WS | `/ws/alerts` | Global alert broadcast |

---

## WebSocket Streams

### Live Telemetry: `ws://localhost:8000/ws/live/{vin}`

Pushes every 2 seconds:

```json
{
  "type": "telemetry",
  "vin": "MH01MZ7X0001",
  "ts": "2024-06-01T10:00:00Z",
  "data": {"speed": 65.2, "soc": 74.1, "engineTemp": 91.0, "voltage12v": 12.6},
  "alerts": []
}
```

### Alert Stream: `ws://localhost:8000/ws/alerts`

Sends when any vehicle triggers a rule alert:

```json
{
  "type": "alert",
  "vin": "MH01MZ7X0002",
  "alert_type": "THERMAL_RUNAWAY",
  "severity": "CRITICAL",
  "title": "HV Battery Thermal Runaway Risk",
  "triggered_at": "2024-06-01T10:05:30Z"
}
```

---

## Alert Severity Levels

| Severity | SLA | Typical Alerts |
|----------|-----|----------------|
| CRITICAL | < 30 seconds | Thermal runaway, brake failure, engine overtemp, 12V critical |
| HIGH | < 1 hour | TPMS deflation, BMS temp warning, SOH < 70%, cell imbalance |
| MEDIUM | Next business slot | Brake pad wear, oil change due, engine temp elevated, low SOC |
| LOW | In-app batch | Oil advisory, low fuel, fuel advisory, SOC critical (EV) |

Alert costs are quoted in INR. Warranty coverage: age < 36 months AND odometer < 100,000 km → 80% parts cost covered.

---

## Environment Variables

Copy `.env.example` to `.env` and configure:

```env
# Database
POSTGRES_URL=postgresql://autopredict:autopredict@localhost:5432/autopredict_db
DATABASE_URL=postgresql://autopredict:autopredict@localhost:5432/autopredict_db

# InfluxDB
INFLUXDB_URL=http://localhost:8086
INFLUXDB_TOKEN=autopredict-dev-token
INFLUXDB_ORG=autopredict
INFLUXDB_BUCKET=telemetry

# Redis
REDIS_URL=redis://localhost:6379/0

# Auth
SECRET_KEY=change-me-in-production-use-32-char-random-string

# MQTT (Mode A ingestion)
MQTT_BROKER_HOST=localhost
MQTT_BROKER_PORT=1883
MQTT_TOPIC_PREFIX=autopredict/telemetry

# AI Agent
GEMINI_API_KEY=your-gemini-api-key-here

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# Data directories
DATA_DIR=data/synthetic
MODEL_DIR=models/saved

# CORS (comma-separated origins or *)
CORS_ORIGINS=http://localhost:5173,http://localhost:3000

# Rate limiting
DEFAULT_RATE_LIMIT=200/minute
```

---

## Project Structure

```
autopredict/
├── api/
│   ├── main.py                  # FastAPI app, auth, health
│   ├── dependencies.py          # JWT, DB session, rate limiter
│   ├── schemas.py               # Pydantic request/response models
│   ├── routers/
│   │   ├── vehicles.py          # Per-vehicle endpoints
│   │   ├── fleet.py             # Fleet-wide aggregates
│   │   ├── dealer.py            # Bay, appointments, inventory
│   │   ├── agent.py             # AI workflow + chat
│   │   ├── upload.py            # CSV / file ingestion
│   │   └── synthetic.py         # Synthetic data generation
│   └── ws/
│       └── telemetry_stream.py  # WebSocket endpoints
├── alerts/
│   ├── rule_engine.py           # 24 deterministic threshold rules
│   └── dispatch.py              # Redis-backed cooldown + notification
├── features/
│   ├── base_pipeline.py         # FeaturePipeline ABC
│   ├── brake_features.py
│   ├── battery_12v_features.py
│   ├── battery_hv_features.py
│   ├── engine_features.py
│   ├── tyre_features.py
│   └── driver_behaviour_features.py
├── models/
│   ├── model_registry.py        # Unified load/infer interface
│   ├── brake_wear_model.py
│   ├── engine_oil_model.py
│   ├── hv_battery_soh_model.py
│   ├── battery_12v_model.py
│   ├── tyre_wear_model.py
│   ├── fuel_anomaly_model.py
│   ├── driver_score_model.py
│   ├── train_all.py
│   └── saved/                   # Trained .joblib artifacts
├── ingestion/
│   ├── validators.py            # 23-channel TBox validator
│   ├── tbox_receiver.py         # MQTT subscriber (Mode A)
│   ├── csv_loader.py            # Bulk CSV loader (Mode B)
│   └── synthetic_loader.py      # Synthetic data DB loader
├── synthetic/
│   ├── generate_fleet.py
│   ├── generate_telemetry.py
│   ├── generate_trips.py
│   └── generate_service_history.py
├── dealer_portal/               # React TypeScript frontend
│   ├── src/
│   │   ├── api/                 # Axios client + React Query hooks
│   │   ├── components/          # HealthGauge, AlertBadge, TelemetryChart, …
│   │   └── pages/               # Dashboard, VehicleDetail, Alerts, …
│   └── vite.config.ts
├── tests/
│   ├── conftest.py
│   ├── test_validators.py
│   ├── test_features.py
│   ├── test_models.py
│   ├── test_alert_engine.py
│   └── test_api.py
├── scripts/
│   └── e2e_demo.py
├── Dockerfile
├── Dockerfile.worker
├── docker-compose.yml
├── Makefile
└── .env.example
```

---

## Development

### Backend

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

### Dealer Portal

```bash
cd dealer_portal
npm install
npm run dev    # http://localhost:5173
npm run build  # production bundle → dist/
```

### Tests

```bash
pytest tests/ -v                  # all tests
pytest tests/test_validators.py   # validator tests only
pytest tests/test_api.py -v       # API tests with output
pytest --tb=short -q              # concise output
```

### Linting

```bash
ruff check .
mypy api/ alerts/ features/ models/ ingestion/
```

---

## Docker

```bash
# Start all services
docker compose up -d

# View logs
docker compose logs -f api

# Stop everything
docker compose down

# Full rebuild
docker compose build --no-cache && docker compose up -d
```

---

## Makefile Targets

```bash
make up          # docker compose up -d
make down        # docker compose down
make build       # docker compose build
make demo        # python scripts/e2e_demo.py
make train       # python models/train_all.py
make test        # pytest tests/ -v
make generate    # generate synthetic data via API
make logs        # docker compose logs -f api
make clean       # remove __pycache__, .pytest_cache, *.pyc
```

---

## License.
