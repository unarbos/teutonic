#!/usr/bin/env bash
# Start/reload the PM2 daily Lium benchmark service.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
ENV_FILE="${TEUTONIC_KING_BENCH_ENV_FILE:-$HOME/.teutonic-king-bench-lium.env}"

if [ -f "$ENV_FILE" ]; then
  echo "[king-bench-lium] sourcing $ENV_FILE"
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi

cd "$REPO_ROOT"
mkdir -p logs runs/king-benchmark-daily runs/lium-rentals

if pm2 describe teutonic-king-bench-lium >/dev/null 2>&1; then
  echo "[king-bench-lium] reloading existing pm2 app"
  pm2 reload scripts/king_benchmark_daily/ecosystem.config.js --update-env
else
  echo "[king-bench-lium] starting pm2 app"
  pm2 start scripts/king_benchmark_daily/ecosystem.config.js
fi

pm2 save
echo "[king-bench-lium] latest JSON: $REPO_ROOT/runs/king-benchmark-daily/latest.json"
echo "[king-bench-lium] logs: $REPO_ROOT/logs/king-bench-lium.{out,err}.log"
