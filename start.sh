#!/usr/bin/env bash
set -euo pipefail

# Move to project root
cd "$(dirname "$0")"

# Activate .venv if no virtualenv is active
if [ -z "${VIRTUAL_ENV:-}" ] && [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# Load .env if present (KEY=VALUE format)
if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

# Add current dir to PYTHONPATH
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# Run FastAPI app
exec uvicorn app.main:app --host "${UVICORN_HOST:-0.0.0.0}" --port "${UVICORN_PORT:-8000}" --reload

