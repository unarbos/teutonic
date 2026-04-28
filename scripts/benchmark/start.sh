#!/usr/bin/env bash
# Boot the teutonic-bench pm2 app on the GPU host.
#
# Looks for credentials in (in order of precedence):
#   1. The current process environment (e.g. doppler run -- start.sh)
#   2. ~/.teutonic-bench.env (a plain `KEY=VALUE` file, gitignored)
#
# Then starts/reloads the pm2 app from scripts/benchmark/ecosystem.config.js.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
ENV_FILE="${TEUTONIC_BENCH_ENV_FILE:-$HOME/.teutonic-bench.env}"

if [ -f "$ENV_FILE" ]; then
  echo "[start.sh] sourcing $ENV_FILE"
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi

: "${TEUTONIC_HIPPIUS_ACCESS_KEY:?TEUTONIC_HIPPIUS_ACCESS_KEY (or HIPPIUS_ACCESS_KEY) must be set}"
: "${TEUTONIC_HIPPIUS_SECRET_KEY:?TEUTONIC_HIPPIUS_SECRET_KEY (or HIPPIUS_SECRET_KEY) must be set}"

cd "$REPO_ROOT"
mkdir -p logs

if pm2 describe teutonic-bench >/dev/null 2>&1; then
  echo "[start.sh] reloading existing teutonic-bench"
  pm2 reload scripts/benchmark/ecosystem.config.js --update-env
else
  echo "[start.sh] starting teutonic-bench"
  pm2 start scripts/benchmark/ecosystem.config.js
fi
pm2 save
echo "[start.sh] done. logs at $REPO_ROOT/logs/bench.{out,err}.log"
