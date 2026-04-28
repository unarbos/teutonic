#!/usr/bin/env python3
"""One-shot operator script: walk the previous_king chain via /probe, find the
first clean ancestor (or fall back to SEED), and patch R2 state.

Use case: deployed a stricter trainability_probe; the live king now fails it,
and the validator's audit-loop dethrone path would take 2+ hours per dirty
king. This script does the cascade in one go and lets the validator restart
on a clean state.

Run on the validator host (has R2 doppler creds + tunnel access). Validator
should be stopped first to avoid racing on state writes.
"""
import argparse
import json
import os
import sys
import time

import boto3
import httpx
from botocore.config import Config as BotoConfig
from huggingface_hub import HfApi

EVAL_SERVER = os.environ.get("TEUTONIC_EVAL_SERVER", "http://localhost:9000")
PROBE_TIMEOUT_S = int(os.environ.get("TEUTONIC_PROBE_TIMEOUT_S", "600"))
SEED_REPO = os.environ.get("TEUTONIC_SEED_REPO", "unconst/Teutonic-VIII")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

R2_ENDPOINT = os.environ["TEUTONIC_R2_ENDPOINT"]
R2_BUCKET = os.environ["TEUTONIC_R2_BUCKET"]
R2_ACCESS = os.environ["TEUTONIC_R2_ACCESS_KEY"]
R2_SECRET = os.environ["TEUTONIC_R2_SECRET_KEY"]


def r2_client():
    return boto3.client(
        "s3", endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS, aws_secret_access_key=R2_SECRET,
        region_name="auto",
        config=BotoConfig(connect_timeout=15, read_timeout=45,
                           retries={"max_attempts": 3, "mode": "adaptive"}),
    )


def r2_get_json(c, key):
    try:
        return json.loads(c.get_object(Bucket=R2_BUCKET, Key=key)["Body"].read())
    except Exception as e:
        print(f"[r2] get {key} FAILED: {e}")
        return None


def r2_put_json(c, key, data):
    c.put_object(Bucket=R2_BUCKET, Key=key,
                 Body=json.dumps(data, default=str).encode(),
                 ContentType="application/json")
    print(f"[r2] put {key} OK")


def probe_repo(repo, revision):
    print(f"[probe] {repo}@{(revision or 'HEAD')[:12]} ... ", end="", flush=True)
    t0 = time.time()
    try:
        with httpx.Client(timeout=httpx.Timeout(PROBE_TIMEOUT_S)) as client:
            for attempt in range(20):
                resp = client.post(f"{EVAL_SERVER}/probe",
                                    json={"repo": repo, "revision": revision or ""})
                if resp.status_code == 409:
                    print(f"[busy, retrying in 30s] ", end="", flush=True)
                    time.sleep(30)
                    continue
                resp.raise_for_status()
                v = resp.json()
                ok = v.get("ok")
                reason = v.get("reason")
                print(f"ok={ok} reason={reason or '-'} "
                      f"(max_norm_w={v.get('max_norm_weight')} "
                      f"global_grad={v.get('global_grad_norm')}) "
                      f"({time.time() - t0:.1f}s)")
                return v
            print("[/probe stayed busy after 20 attempts]")
            return {"ok": False, "reason": "probe_endpoint_busy"}
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        return {"ok": False, "reason": f"probe_call_failed:{e}"}


def resolve_revision(repo):
    try:
        info = HfApi(token=HF_TOKEN or None).model_info(repo)
        return info.sha
    except Exception as e:
        print(f"[hf] cannot resolve {repo}: {e}")
        return ""


def walk_chain(king):
    """Yield (repo, revision, hotkey, snapshot_dict) for king then ancestors."""
    cur = king
    seen = set()
    while cur and cur.get("hf_repo"):
        repo = cur["hf_repo"]
        if repo in seen:
            print(f"[chain] cycle detected at {repo}, stopping")
            return
        seen.add(repo)
        yield (repo, cur.get("king_revision", ""), cur.get("hotkey", ""), cur)
        cur = cur.get("previous_king") or {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Probe the chain but do not write R2.")
    ap.add_argument("--max-depth", type=int, default=20,
                    help="Stop walking after N ancestors (safety).")
    args = ap.parse_args()

    c = r2_client()
    king = r2_get_json(c, "king/current.json")
    if not king:
        sys.exit("no king/current.json found in R2")
    print(f"[main] current king: {king.get('hf_repo')} "
          f"@{(king.get('king_revision') or '')[:12]} "
          f"hotkey={king.get('hotkey', '')[:16]}")

    chain = list(walk_chain(king))[:args.max_depth]
    print(f"[main] chain length: {len(chain)}")

    failed_repos = []
    clean = None
    for repo, rev, hotkey, snapshot in chain:
        verdict = probe_repo(repo, rev)
        if verdict.get("ok"):
            clean = (repo, rev, hotkey, snapshot, verdict)
            break
        failed_repos.append({"repo": repo, "revision": rev,
                              "hotkey": hotkey, "reason": verdict.get("reason")})

    if clean is None:
        print(f"[main] entire chain dirty, falling back to SEED_REPO={SEED_REPO}")
        seed_rev = resolve_revision(SEED_REPO)
        if not seed_rev:
            sys.exit("could not resolve seed revision")
        seed_v = probe_repo(SEED_REPO, seed_rev)
        if not seed_v.get("ok"):
            sys.exit(f"SEED also failed probe: {seed_v.get('reason')} -- aborting")
        # Build a fresh king record for the seed.
        new_king = {
            "hotkey": king.get("hotkey", ""),
            "hf_repo": SEED_REPO,
            "king_hash": "seed",
            "king_revision": seed_rev,
            "reign_number": int(king.get("reign_number", 0)) + 1,
            "crowned_at": __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc).isoformat(),
            "crowned_block": int(king.get("crowned_block", 0)),
            "challenge_id": "cascade_dethrone_seed",
            "previous_king": None,
        }
    else:
        repo, rev, hotkey, snapshot, verdict = clean
        print(f"[main] FOUND CLEAN ANCESTOR: {repo}@{(rev or '')[:12]} "
              f"hotkey={hotkey[:16]}")
        new_king = dict(snapshot)
        new_king["challenge_id"] = "cascade_dethrone_revert"
        new_king["crowned_at"] = __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat()
        # Bump reign_number so the new "reign" is distinct in history.
        new_king["reign_number"] = int(king.get("reign_number", 0)) + 1
        # Drop previous_king pointer to a fresh deep copy of itself's chain.
        # (Keep its own previous_king; that ancestor was already reached as
        # part of this chain walk and either probed clean or is the SEED.)

    print("\n[main] proposed new king:")
    print(json.dumps(new_king, default=str, indent=2))
    print(f"\n[main] dethroning {len(failed_repos)} kings:")
    for fr in failed_repos:
        print(f"  - {fr['repo']}@{(fr['revision'] or '')[:12]} "
              f"hotkey={fr['hotkey'][:16]} reason={fr['reason']}")

    if args.dry_run:
        print("\n[main] --dry-run, NOT writing R2")
        return

    # Patch state.
    print("\n[main] writing king/current.json ...")
    r2_put_json(c, "king/current.json", new_king)

    # Update validator_state.json: clear queue + reset score window + add audit.
    state = r2_get_json(c, "state/validator_state.json") or {}
    state["king"] = new_king
    state["score_window"] = {
        "window_id": (state.get("score_window") or {}).get("window_id", "window-0000"),
        "started_at": new_king["crowned_at"],
        "started_block": int(new_king.get("crowned_block", 0)),
        "accepted_by_hotkey": {},
        "topk": [],
        "last_weight_set": (state.get("score_window") or {}).get("last_weight_set"),
    }
    state["king_audit"] = {
        "last_at": new_king["crowned_at"],
        "last_status": "ok",
        "consecutive_fails": 0,
        "last_verdict": None,
        "last_reason": "cascade_dethrone_clean_state",
    }
    state["updated_at"] = new_king["crowned_at"]
    r2_put_json(c, "state/validator_state.json", state)

    # Append history events for transparency.
    print("[main] appending history events ...")
    try:
        existing = c.get_object(Bucket=R2_BUCKET,
                                 Key="state/history.jsonl")["Body"].read()
    except Exception:
        existing = b""
    lines = [
        json.dumps({
            "timestamp": new_king["crowned_at"],
            "event": "cascade_dethrone",
            "lost_repos": failed_repos,
            "reverted_to": new_king.get("hf_repo"),
            "reverted_to_revision": new_king.get("king_revision"),
            "trigger": "operator_script:cascade_dethrone_now",
            "reason": "trainability_probe_rewrite",
        })
    ]
    body = existing + ("\n".join(lines) + "\n").encode()
    c.put_object(Bucket=R2_BUCKET, Key="state/history.jsonl",
                 Body=body, ContentType="application/x-ndjson")

    # Reset queue (it's full of stale challenger entries).
    c.put_object(Bucket=R2_BUCKET, Key="state/queue.json",
                 Body=json.dumps({"pending": [], "updated_at": new_king["crowned_at"]}).encode(),
                 ContentType="application/json")
    print("[r2] reset state/queue.json")

    print("\n[main] DONE. Restart the validator to pick up the new state.")


if __name__ == "__main__":
    main()
