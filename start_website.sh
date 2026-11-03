#!/usr/bin/env bash
# Starts the website. Mac/Linux:  bash start_website.sh        (Windows: double-click start_website.bat)
# The first run creates a virtual environment (.venv) and installs the requirements. That happens once and takes a
# few minutes because PyTorch is a large download. After that it starts in a few seconds.
# Extra options are passed on, for example:  bash start_website.sh --port 5001
set -e
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"

if ! command -v "$PY" >/dev/null 2>&1; then
  echo "Python was not found. Install Python 3.10 or newer from https://www.python.org/downloads/ and try again."
  exit 1
fi
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo "Python 3.10 or newer is needed. You have: $("$PY" --version)"
  exit 1
fi

if [ ! -x .venv/bin/python ]; then
  echo "Creating a virtual environment in .venv ..."
  "$PY" -m venv .venv
fi

if [ -z "$GLAUCOMA_SKIP_INSTALL" ] && [ ! -f .venv/.requirements-installed ]; then
  echo "Installing the requirements (first run only, a few minutes) ..."
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
  touch .venv/.requirements-installed
fi

echo "Starting the website. Press Ctrl+C to stop it."
exec .venv/bin/python -m backend.app --open "$@"
