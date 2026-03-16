#!/usr/bin/env bash
set -euo pipefail

# プロジェクトルートに移動
cd "$(dirname "$0")"

# .env があれば読み込む（KEY=VALUE 形式想定）
if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

# PYTHONPATH にカレントを追加
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# FastAPIアプリ起動
exec uvicorn app.main:app --host "${UVICORN_HOST:-0.0.0.0}" --port "${UVICORN_PORT:-8000}" --reload

