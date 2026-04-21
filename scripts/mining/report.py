#!/usr/bin/env python3
"""Write a status report on the current Teutonic mining run + drop it
into the Arbos outbox so it lands in the Telegram topic.

Designed to be called from heartbeats/cron. Idempotent — same content
twice in a row is suppressed.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[3]
OUTBOX = Path("/home/const/workspace/.arbos/outbox")
REPORTS = ROOT / "reports" / "teutonic-mining"
DASHBOARD = "https://s3.hippius.com/teutonic-sn3/dashboard.json"
LAST_HASH_FILE = REPORTS / ".last_report_hash"


def fetch_json(url):
    try:
        with urlopen(url, timeout=15) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}


def remote_status():
    """Best-effort poll of the GPU box training session."""
    out = {}
    cmd = [
        "ssh", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=no",
        "wrk-nlapfgb9asmx@ssh.deployments.targon.com",
        "tmux has-session -t teutonic-miner 2>/dev/null && echo TMUX_ALIVE || echo TMUX_DEAD; "
        "tail -n 30 /root/teutonic-mining/work/train.log 2>/dev/null; "
        "echo ---SEP---; "
        "cat /root/teutonic-mining/work/verdict.json 2>/dev/null",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        out["raw"] = r.stdout[-4000:]
        out["session_alive"] = "TMUX_ALIVE" in r.stdout
    except Exception as e:
        out["error"] = str(e)
    return out


def render(report):
    lines = []
    lines.append("# Teutonic mining status")
    lines.append(f"_generated: {dt.datetime.utcnow().isoformat()}Z_")
    lines.append("")
    k = report["king"].get("king", {})
    lines.append(f"**King:** `{k.get('hf_repo','?')}` (reign #{k.get('reign_number','?')}) "
                 f"hotkey `{(k.get('hotkey') or '')[:16]}...`")
    lines.append("")
    rs = report["king"].get("reign_summary") or {}
    if rs:
        lines.append(f"- accepted dethrones: {rs.get('accepted',0)}")
        lines.append(f"- total evals seen: {rs.get('total_evals','?')}")

    lines.append("")
    lines.append("## Training run")
    rem = report["remote"]
    if rem.get("session_alive"):
        lines.append("- tmux: **alive**")
    else:
        lines.append("- tmux: **not running**")
    raw = rem.get("raw", "")
    sep = raw.split("---SEP---")
    log_tail = sep[0].strip().splitlines()[-15:] if sep else []
    if log_tail:
        lines.append("```")
        lines.extend(log_tail)
        lines.append("```")
    if len(sep) > 1 and sep[1].strip():
        try:
            v = json.loads(sep[1].split("\n",1)[-1].strip()) if "{" in sep[1] else None
        except Exception:
            v = None
        if v and v.get("best"):
            b = v["best"]
            lines.append("")
            lines.append(f"**Latest verdict:** mu_hat=`{b['mu_hat']:.6f}` "
                         f"lcb=`{b['lcb']:.6f}` delta=`{b['delta']}` "
                         f"accepted=**{b['accepted']}**")

    lines.append("")
    lines.append("## Notes")
    lines.append("- Validator delta = 0.01 nats/token. Offline target_mu = 0.05 (5x cushion).")
    lines.append("- Iterates up to 3 attempts; uploads only if accepted.")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None,
                    help="Override outbox file path; default writes to outbox.")
    ap.add_argument("--force", action="store_true",
                    help="Send even if content is identical to previous report.")
    ap.add_argument("--no-send", action="store_true",
                    help="Just print, do not write to outbox.")
    args = ap.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)
    report = {
        "king": fetch_json(DASHBOARD),
        "remote": remote_status(),
        "ts": time.time(),
    }
    text = render(report)

    h = hashlib.sha256(text.encode()).hexdigest()
    if not args.force and LAST_HASH_FILE.exists() and LAST_HASH_FILE.read_text().strip() == h:
        print("[report] no change since last run; skipping send", file=sys.stderr)
        return

    print(text)
    if args.no_send:
        return

    OUTBOX.mkdir(parents=True, exist_ok=True)
    fname = f"teutonic-mining-{int(time.time())}.md"
    target = OUTBOX / fname
    target.write_text(text)
    print(f"[report] wrote {target}", file=sys.stderr)
    LAST_HASH_FILE.write_text(h)


if __name__ == "__main__":
    main()
