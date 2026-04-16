#!/usr/bin/env bash
# Returns teutonic-validator status from pm2 as JSON
set -euo pipefail

if ! command -v pm2 &>/dev/null; then
    echo '{"error": "pm2 not installed"}'
    exit 0
fi

pm2 jlist 2>/dev/null | python3 -c "
import sys, json
try:
    procs = json.load(sys.stdin)
    out = []
    for p in procs:
        if 'teutonic' in p.get('name', '').lower():
            env = p.get('pm2_env', {})
            out.append({
                'name': p.get('name'),
                'status': env.get('status'),
                'pid': p.get('pid'),
                'uptime_ms': env.get('pm_uptime'),
                'restarts': env.get('restart_time', 0),
                'memory_mb': round(p.get('monit', {}).get('memory', 0) / 1048576, 1),
                'cpu': p.get('monit', {}).get('cpu', 0),
            })
    if out:
        print(json.dumps(out, indent=2))
    else:
        print(json.dumps({'error': 'no teutonic processes found in pm2'}))
except Exception as e:
    print(json.dumps({'error': str(e)}))
"
