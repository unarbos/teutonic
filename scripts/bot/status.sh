#!/usr/bin/env bash
# Returns teutonic-validator status from pm2 as JSON
set -euo pipefail

HF_CACHE_DIR="${HOME}/.cache/huggingface"
MIN_FREE_MB="${MIN_FREE_MB:-2048}"   # trigger cleanup if available space < 2 GiB

json_escape() {
    python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))'
}

get_avail_mb() {
    df -Pm "$HOME" | awk 'NR==2 {print $4}'
}

cleanup_triggered=false
cleanup_error=""
disk_free_before_mb="$(get_avail_mb || echo 0)"
disk_free_after_mb="$disk_free_before_mb"

if [ "${disk_free_before_mb:-0}" -lt "$MIN_FREE_MB" ]; then
    if [ -d "$HF_CACHE_DIR" ]; then
        if rm -rf "$HF_CACHE_DIR" 2>/dev/null; then
            cleanup_triggered=true
        else
            cleanup_triggered=true
            cleanup_error="failed to remove ${HF_CACHE_DIR}"
        fi
    else
        cleanup_triggered=true
        cleanup_error="${HF_CACHE_DIR} does not exist"
    fi
    disk_free_after_mb="$(get_avail_mb || echo "$disk_free_before_mb")"
fi

if ! command -v pm2 >/dev/null 2>&1; then
    python3 - <<PY
import json
print(json.dumps({
    "error": "pm2 not installed",
    "disk_free_before_mb": int("${disk_free_before_mb}"),
    "disk_free_after_mb": int("${disk_free_after_mb}"),
    "disk_cleanup_threshold_mb": int("${MIN_FREE_MB}"),
    "emergency_cleanup_triggered": ${cleanup_triggered},
    "cleanup_error": ${cleanup_error:+${cleanup_error@Q}} if False else ${cleanup_error@Q}
}))
PY
    exit 0
fi

PM2_JSON="$(pm2 jlist 2>/dev/null || echo '[]')"

python3 - <<PY
import json

raw = """${PM2_JSON}"""
cleanup_error = ${cleanup_error@Q}

try:
    procs = json.loads(raw)
except Exception as e:
    print(json.dumps({
        "error": f"failed to parse pm2 output: {e}",
        "disk_free_before_mb": int("${disk_free_before_mb}"),
        "disk_free_after_mb": int("${disk_free_after_mb}"),
        "disk_cleanup_threshold_mb": int("${MIN_FREE_MB}"),
        "emergency_cleanup_triggered": ${cleanup_triggered},
        "cleanup_error": cleanup_error or None
    }, indent=2))
    raise SystemExit(0)

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

result = {
    "disk_free_before_mb": int("${disk_free_before_mb}"),
    "disk_free_after_mb": int("${disk_free_after_mb}"),
    "disk_cleanup_threshold_mb": int("${MIN_FREE_MB}"),
    "emergency_cleanup_triggered": ${cleanup_triggered},
    "cleanup_error": cleanup_error or None,
    "processes": out if out else [],
}

if not out:
    result["error"] = "no teutonic processes found in pm2"

print(json.dumps(result, indent=2))
PY
