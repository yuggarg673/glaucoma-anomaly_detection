# Linux / macOS shortcuts. On Windows use the python commands shown in README.md instead.
PY ?= python3
export PYTHONPATH := src:.

.PHONY: help setup setup-frontend quick train data splits test test-ui doctor website serve-backend serve-frontend frontend-build

help:
	@echo "make setup | quick | train | test | test-ui | doctor | website"
	@echo "optional React version (never built, not needed): setup-frontend | serve-frontend | frontend-build"

setup:
	$(PY) -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

setup-frontend:   # only for the optional React version in frontend/ (needs Node 18+)
	cd frontend && npm install

website:          # the website and API on http://localhost:5000 (no Node, no build; needs trained models, see README)
	$(PY) -m backend.app --open

quick:            # whole pipeline on synthetic data (code check only, numbers are meaningless)
	$(PY) scripts/run_pipeline.py --quick

data:
	$(PY) scripts/download_data.py
	$(PY) scripts/inspect_data.py

splits:
	$(PY) scripts/make_splits.py

train:            # real data
	$(PY) scripts/run_pipeline.py

test:
	$(PY) -m unittest discover -s tests -v

test-ui:          # the website in a real browser. Once: pip install playwright && python -m playwright install chromium
	$(PY) -m unittest tests.test_ui_browser -v

doctor:
	$(PY) scripts/doctor.py

serve-backend:
	$(PY) -m backend.app

serve-frontend:
	cd frontend && npm run dev

frontend-build:
	cd frontend && npm run build
