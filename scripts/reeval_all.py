#!/usr/bin/env python3
"""Re-evaluate all cached challengers against the current king at 102400 samples.

Results are written to reeval_results.json as they complete. The script
retries on 409 (eval_server busy with live validator eval) with backoff.
Run in tmux/screen — takes ~35 hours for 63 models.

Usage:
    python3 reeval_script.py [--dry-run] [--n 102400] [--resume]
"""
import json
import os
import sys
import time
import hashlib
import httpx

EVAL_SERVER = os.environ.get("TEUTONIC_EVAL_SERVER", "http://localhost:9000")
KING_REPO = "teutonic-miner/teutonic-q3-4b-5g6x3hrj-top3"
KING_DIGEST = "sha256:80693d7ee5bfcfe8eb4cad422380fba635d0422dbedf706cd3a1791a20749319"
RESULTS_FILE = "reeval_results.json"
EVAL_N = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 102400
DRY_RUN = "--dry-run" in sys.argv
RESUME = "--resume" in sys.argv

MODELS = {}  # repo -> digest, populated from GPU box cache listing


def load_models_from_cache_listing(path="reeval_models.txt"):
    """Load model list from a pre-generated file (one 'repo|digest' per line)."""
    models = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or "|" not in line:
                continue
            repo_raw, digest = line.split("|", 1)
            repo = repo_raw.replace("--", "/", 1)
            if repo == KING_REPO:
                continue
            models[repo] = digest
    return models


def load_existing_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {}


def save_results(results):
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)


def submit_eval(challenger_repo, challenger_digest):
    seed = hashlib.sha256(
        f"reeval:{challenger_repo}:{EVAL_N}".encode()
    ).hexdigest()
    payload = {
        "king_repo": KING_REPO,
        "challenger_repo": challenger_repo,
        "block_hash": f"0x{seed}",
        "hotkey": "5Freeval00000000000000000000000000000000000000",
        "shard_key": "",
        "king_digest": KING_DIGEST,
        "challenger_digest": challenger_digest,
        "delta_threshold": 0.0025,
        "n_public": EVAL_N,
        "n_private": 0,
        "n_bootstrap": 10000,
        "alpha": 0.001,
        "seq_len": 2048,
        "batch_size": 512,
    }
    r = httpx.post(f"{EVAL_SERVER}/eval", json=payload, timeout=30)
    return r


def wait_for_result(eval_id):
    while True:
        r = httpx.get(f"{EVAL_SERVER}/eval/{eval_id}", timeout=30)
        d = r.json()
        state = d.get("state")
        if state == "completed":
            return d.get("verdict")
        if state == "failed":
            return {"error": d.get("error"), "accepted": None}
        p = d.get("progress", {})
        done = p.get("done", "?")
        total = p.get("total", "?")
        sps = p.get("seqs_per_sec", "?")
        print(f"  ... {done}/{total} @ {sps} seq/s", flush=True)
        time.sleep(30)


def main():
    if not os.path.exists("reeval_models.txt"):
        print("ERROR: reeval_models.txt not found. Generate it with:")
        print("  ssh GPU_BOX 'find /tmp/teutonic/hippius_models ...' > reeval_models.txt")
        sys.exit(1)

    models = load_models_from_cache_listing()
    print(f"Loaded {len(models)} models to re-evaluate at N={EVAL_N}")

    results = load_existing_results() if RESUME else {}
    done_keys = set(results.keys())

    for i, (repo, digest) in enumerate(sorted(models.items())):
        key = f"{repo}@{digest[:19]}"
        if RESUME and key in done_keys:
            print(f"[{i+1}/{len(models)}] SKIP (already done): {repo}")
            continue

        print(f"\n[{i+1}/{len(models)}] {repo} ({digest[:19]})")
        if DRY_RUN:
            print("  (dry run, skipping)")
            continue

        # Submit with retry on 409
        for attempt in range(60):
            r = submit_eval(repo, digest)
            if r.status_code == 200:
                break
            if r.status_code == 409:
                print(f"  eval server busy, waiting 60s (attempt {attempt+1})")
                time.sleep(60)
            else:
                print(f"  ERROR: {r.status_code} {r.text}")
                time.sleep(10)
                break
        else:
            print("  FAILED: eval server busy for 60 attempts, skipping")
            results[key] = {"error": "server_busy_timeout", "accepted": None}
            save_results(results)
            continue

        eval_id = r.json().get("eval_id")
        print(f"  eval_id={eval_id}, waiting...")

        verdict = wait_for_result(eval_id)
        results[key] = {
            "repo": repo,
            "digest": digest,
            "eval_n": EVAL_N,
            "mu_hat": verdict.get("mu_hat"),
            "lcb": verdict.get("lcb"),
            "accepted": verdict.get("accepted"),
            "avg_king_loss": verdict.get("avg_king_loss"),
            "avg_challenger_loss": verdict.get("avg_challenger_loss"),
            "wall_time_s": verdict.get("wall_time_s"),
            "seqs_per_sec": verdict.get("seqs_per_sec"),
            "error": verdict.get("error"),
        }
        save_results(results)

        v = results[key]
        status = "ACCEPTED" if v["accepted"] else "REJECTED"
        print(f"  {status} mu_hat={v['mu_hat']} lcb={v['lcb']} "
              f"king_loss={v['avg_king_loss']} chall_loss={v['avg_challenger_loss']} "
              f"wall={v['wall_time_s']}s")

    print(f"\nDone. Results in {RESULTS_FILE}")


if __name__ == "__main__":
    main()
