#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export EVAL_HOST="${EVAL_HOST:-127.0.0.1}"
export EVAL_PORT="${EVAL_PORT:-9000}"

if [[ -x "$REPO_DIR/.venv/bin/python" ]]; then
  exec "$REPO_DIR/.venv/bin/python" -m uvicorn eval_server_two_sources:app --host "$EVAL_HOST" --port "$EVAL_PORT" --log-level info
fi

exec uv run python -m uvicorn eval_server_two_sources:app --host "$EVAL_HOST" --port "$EVAL_PORT" --log-level info
