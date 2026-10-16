.PHONY: help setup dev test lint format build docker-up docker-down migrate download-weights fetch-data demo wipe-data

PYTHON ?= python
NPM ?= npm

help:
	@echo "OphthalmoAI Build and Development Automation"
	@echo "Targets:"
	@echo "  setup            Install backend and frontend dependencies"
	@echo "  dev              Start local backend development server"
	@echo "  test             Run all backend and frontend tests"
	@echo "  lint             Run ruff and frontend linter"
	@echo "  format           Format codebase using ruff"
	@echo "  build            Build production frontend assets"
	@echo "  docker-up        Launch system using Docker Compose v2"
	@echo "  docker-down      Stop Docker Compose containers"
	@echo "  migrate          Apply database migrations"
	@echo "  download-weights Verify and download model weight checkpoints"
	@echo "  fetch-data       Verify dataset manifest and sources"
	@echo "  demo             Run end-to-end API smoke test verification"
	@echo "  wipe-data        Wipe local SQLite database fixtures"

setup:
	$(PYTHON) -m pip install -e ".[dev]"
	cd frontend && $(NPM) install

dev:
	uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

test:
	$(PYTHON) -m pytest tests/
	cd frontend && $(NPM) test -- --run

lint:
	$(PYTHON) -m ruff check backend
	$(PYTHON) -m mypy backend
	cd frontend && $(NPM) run lint

format:
	$(PYTHON) -m ruff format backend

build:
	cd frontend && $(NPM) run build

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down

migrate:
	alembic upgrade head

download-weights:
	$(PYTHON) scripts/download_weights.py

fetch-data:
	$(PYTHON) scripts/fetch_dataset.py

demo:
	$(PYTHON) scripts/smoke_test_api.py

wipe-data:
	$(PYTHON) -c "import os; [os.remove(f) for f in ('ophthalmoai.db', 'test_ophthalmoai.db') if os.path.exists(f)]; print('Local development databases wiped.')"
