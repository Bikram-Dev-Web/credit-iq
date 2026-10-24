#!/usr/bin/env bash
# CreditIQ launcher for macOS / Linux:  bash run_app.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
if [ ! -x ".venv/bin/python" ]; then
  echo "Creating virtual environment in .venv ..."
  "$PY" -m venv .venv
fi
source .venv/bin/activate
if [ ! -f ".venv/.creditiq_installed" ]; then
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  python -m pip install -r requirements-optional.txt || echo "Optional extras failed to install; the app will use its fallbacks."
  touch .venv/.creditiq_installed
fi
echo "Starting CreditIQ at http://localhost:8501 (Ctrl+C to stop)"
exec python -m streamlit run app.py
