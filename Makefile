# Linux / macOS shortcuts. On Windows use the python commands shown in README.md instead.
PY ?= python3
export PYTHONPATH := src:.

.PHONY: help setup quick train data splits test doctor serve-backend serve-frontend frontend-build

help:
	@echo "make setup | quick | train | test | doctor | serve-backend | serve-frontend"

setup:
	$(PY) -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt
	cd frontend && npm install

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

doctor:
	$(PY) scripts/doctor.py

serve-backend:
	$(PY) -m backend.app

serve-frontend:
	cd frontend && npm run dev

frontend-build:
	cd frontend && npm run build
