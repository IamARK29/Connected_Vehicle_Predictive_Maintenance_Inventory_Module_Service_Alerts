.PHONY: up down build demo train test generate logs clean lint format help

PYTHON     := python
API_URL    := http://localhost:8000
TOKEN_FILE := .dev_token

help:
	@echo ""
	@echo "AutoPredict — Available targets"
	@echo "────────────────────────────────────────"
	@echo "  make up        Start all Docker services"
	@echo "  make down      Stop all Docker services"
	@echo "  make build     Rebuild Docker images"
	@echo "  make demo      Run end-to-end demo script"
	@echo "  make train     Train all ML models"
	@echo "  make test      Run pytest suite"
	@echo "  make generate  Generate synthetic fleet data"
	@echo "  make logs      Tail API logs"
	@echo "  make clean     Remove build artifacts"
	@echo "  make lint      Run ruff linter"
	@echo "  make format    Run ruff formatter"
	@echo ""

# ── Docker ──────────────────────────────────────────────────────────────────

up:
	docker compose up -d
	@echo ""
	@echo "Services started:"
	@echo "  API       : $(API_URL)"
	@echo "  API Docs  : $(API_URL)/docs"
	@echo "  InfluxDB  : http://localhost:8086"
	@echo "  MLflow    : http://localhost:5000"
	@echo "  Portal    : http://localhost:5173"

down:
	docker compose down

build:
	docker compose build --no-cache

logs:
	docker compose logs -f api

# ── Application ──────────────────────────────────────────────────────────────

demo:
	$(PYTHON) scripts/e2e_demo.py --base-url $(API_URL)

train:
	$(PYTHON) models/train_all.py

generate:
	@echo "Obtaining token..."
	@curl -s -X POST $(API_URL)/api/auth/token \
	  -H "Content-Type: application/json" \
	  -d '{"username":"admin","password":"admin123"}' \
	  | python -c "import sys,json; print(json.load(sys.stdin)['access_token'])" > $(TOKEN_FILE)
	@echo "Triggering synthetic data generation (20 vehicles, 90 days)..."
	@curl -s -X POST $(API_URL)/api/synthetic/generate \
	  -H "Authorization: Bearer $$(cat $(TOKEN_FILE))" \
	  -H "Content-Type: application/json" \
	  -d '{"num_vehicles":20,"num_days":90,"failure_rate":0.08}' | python -m json.tool
	@rm -f $(TOKEN_FILE)

# ── Tests ───────────────────────────────────────────────────────────────────

test:
	pytest tests/ -v --tb=short

test-validators:
	pytest tests/test_validators.py -v

test-features:
	pytest tests/test_features.py -v

test-models:
	pytest tests/test_models.py -v

test-alerts:
	pytest tests/test_alert_engine.py -v

test-api:
	pytest tests/test_api.py -v

test-cov:
	pytest tests/ --cov=. --cov-report=html --cov-report=term-missing

# ── Code quality ─────────────────────────────────────────────────────────────

lint:
	ruff check api/ alerts/ features/ models/ ingestion/ synthetic/ scripts/

format:
	ruff format api/ alerts/ features/ models/ ingestion/ synthetic/ scripts/

# ── Cleanup ──────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.pyo" -delete 2>/dev/null || true
	rm -rf htmlcov/ .coverage $(TOKEN_FILE)
