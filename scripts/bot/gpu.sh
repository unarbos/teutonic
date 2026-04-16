#!/usr/bin/env bash
# Check eval server health
set -euo pipefail

EVAL_SERVER="${TEUTONIC_EVAL_SERVER:-http://localhost:9000}"

response=$(curl -s --max-time 5 "${EVAL_SERVER}/health" 2>/dev/null) || {
    echo '{"error": "eval server unreachable", "url": "'"${EVAL_SERVER}"'"}'
    exit 0
}

echo "$response" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(json.dumps({
        'status': d.get('status'),
        'gpus': d.get('gpus'),
        'king_loaded': d.get('king_loaded'),
        'url': '${EVAL_SERVER}',
    }, indent=2))
except:
    print(json.dumps({'raw': sys.stdin.read(), 'url': '${EVAL_SERVER}'}))
"
