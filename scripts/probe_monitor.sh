#!/usr/bin/env bash
# Quick health-check for the 5-layer trainability probe in production.
# Run anytime to get a snapshot of probe activity, rejections by reason,
# and recent verdicts.
#
# Usage:
#   bash scripts/probe_monitor.sh          # tail eval-server log + tally
#   bash scripts/probe_monitor.sh --watch  # follow live (Ctrl-C to exit)
#
# Env (override per call):
#   GPU_HOST   wrk-... ssh target (default: from tunnel.sh)
#   N_LINES    how many recent eval-server log lines to scan (default 2000)

set -euo pipefail

GPU_HOST="${GPU_HOST:-wrk-0638a6gucc7t@ssh.deployments.targon.com}"
N_LINES="${N_LINES:-2000}"

SSH_OPTS=(-o ConnectTimeout=5 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)

if [ "${1:-}" = "--watch" ]; then
  ssh "${SSH_OPTS[@]}" "$GPU_HOST" \
    "tail -F /tmp/eval_server.log" \
    | grep --line-buffered -E 'trainability probe|REJECTED|REFUSING|untrainable|king audit|status='
  exit 0
fi

echo "=== probe activity (last $N_LINES eval-server log lines) ==="
ssh "${SSH_OPTS[@]}" "$GPU_HOST" "tail -n $N_LINES /tmp/eval_server.log" > /tmp/_eval_log.$$ 2>/dev/null

echo
echo "--- probe verdicts (ok=True vs ok=False) ---"
grep -E 'trainability probe for' /tmp/_eval_log.$$ \
  | sed -E 's/.*ok=(True|False).*/ok=\1/' \
  | sort | uniq -c | sort -rn

echo
echo "--- last 10 probe verdicts ---"
grep -E 'trainability probe for' /tmp/_eval_log.$$ \
  | tail -n 10 \
  | sed -E 's/.* eval_server INFO //'

echo
echo "--- rejection reasons (last $N_LINES lines) ---"
grep -E 'REJECTED|REFUSING' /tmp/_eval_log.$$ \
  | sed -E 's/.*: //' \
  | sort | uniq -c | sort -rn \
  || echo "(no rejections in window)"

echo
echo "--- king audit / probe endpoint hits ---"
grep -E 'probe: |king audit' /tmp/_eval_log.$$ \
  | tail -n 5 \
  | sed -E 's/.* eval_server INFO //'

echo
echo "=== current king (from R2) ==="
if command -v doppler >/dev/null; then
  doppler run -p arbos -c dev -- bash -c '
    export AWS_ACCESS_KEY_ID=$R2_ACCESS_KEY_ID
    export AWS_SECRET_ACCESS_KEY=$R2_SECRET_ACCESS_KEY
    aws s3 cp --quiet --endpoint-url $R2_URL s3://$R2_BUCKET_NAME/king/current.json - \
      | python3 -c "import sys, json; d=json.load(sys.stdin); print(json.dumps({k:v for k,v in d.items() if k!=\"previous_king\"}, indent=2))"
  ' 2>/dev/null || echo "(could not fetch king state)"
fi

rm -f /tmp/_eval_log.$$
