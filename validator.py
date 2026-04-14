#!/usr/bin/env python3
"""Teutonic validator — single-file king-of-the-hill evaluator.

Polls Bittensor chain for challenger submissions, runs N=10000 alpha=0.001
sign-test evaluations using two remote H200 boxes as vLLM inference servers,
manages king lifecycle on HuggingFace, persists all state to R2.
"""
import asyncio
import hashlib
import io
import json
import logging
import os
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone

import bittensor as bt
import boto3
import httpx
import numpy as np
from botocore.config import Config as BotoConfig
from huggingface_hub import HfApi, snapshot_download
from scipy.stats import binom

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EVAL_N = 10_000
EVAL_ALPHA = 0.001
SEQ_LEN = 2048
CONCURRENCY = 8
POLL_INTERVAL = 30
NETUID = int(os.environ.get("TEUTONIC_NETUID", "3"))
NETWORK = os.environ.get("TEUTONIC_NETWORK", "finney")
KING_REPO = os.environ.get("TEUTONIC_KING_REPO", "unconst/Teutonic-I")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
SSH_KING = os.environ.get("TEUTONIC_SSH_KING", "")
SSH_CHALLENGER = os.environ.get("TEUTONIC_SSH_CHALLENGER", "")
WALLET_NAME = os.environ.get("BT_WALLET_NAME", "teutonic")
WALLET_HOTKEY = os.environ.get("BT_WALLET_HOTKEY", "default")
KING_PORT = 9001
CHALLENGER_PORT = 9002
MODEL_PATH = "/models/model"
VLLM_PORT = 8000

R2_ENDPOINT = os.environ.get("TEUTONIC_R2_ENDPOINT", "")
R2_BUCKET = os.environ.get("TEUTONIC_R2_BUCKET", "")
R2_ACCESS_KEY = os.environ.get("TEUTONIC_R2_ACCESS_KEY", "")
R2_SECRET_KEY = os.environ.get("TEUTONIC_R2_SECRET_KEY", "")

REPO_PATTERN = r"^[^/]+/Teutonic-I-.+$"

log = logging.getLogger("teutonic")

# ---------------------------------------------------------------------------
# R2
# ---------------------------------------------------------------------------

class R2:
    def __init__(self):
        self.client = boto3.client(
            "s3", endpoint_url=R2_ENDPOINT,
            aws_access_key_id=R2_ACCESS_KEY, aws_secret_access_key=R2_SECRET_KEY,
            region_name="auto",
            config=BotoConfig(retries={"max_attempts": 3, "mode": "adaptive"}),
        )

    def put(self, key, data):
        self.client.put_object(
            Bucket=R2_BUCKET, Key=key,
            Body=json.dumps(data, default=str).encode(),
            ContentType="application/json",
        )

    def get(self, key):
        try:
            return json.loads(
                self.client.get_object(Bucket=R2_BUCKET, Key=key)["Body"].read()
            )
        except Exception:
            return None

    def append_jsonl(self, key, record):
        line = json.dumps(record, default=str) + "\n"
        existing = b""
        try:
            existing = self.client.get_object(Bucket=R2_BUCKET, Key=key)["Body"].read()
        except Exception:
            pass
        self.client.put_object(
            Bucket=R2_BUCKET, Key=key,
            Body=existing + line.encode(),
            ContentType="application/x-ndjson",
        )

    def put_raw(self, key, body, content_type):
        self.client.put_object(
            Bucket=R2_BUCKET, Key=key, Body=body, ContentType=content_type,
        )

    def range_get(self, key, start, end):
        return self.client.get_object(
            Bucket=R2_BUCKET, Key=key, Range=f"bytes={start}-{end}"
        )["Body"].read()


# ---------------------------------------------------------------------------
# SSH
# ---------------------------------------------------------------------------

SSH_OPTS = ["-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=no",
            "-o", "LogLevel=ERROR"]

def ssh(host, cmd, timeout=600):
    r = subprocess.run(
        ["ssh"] + SSH_OPTS + [host, cmd],
        capture_output=True, text=True, timeout=timeout,
    )
    if r.returncode != 0 and r.returncode != 143:
        log.warning("ssh %s exit=%d stderr=%s", host, r.returncode, r.stderr[:200])
    return r.stdout.strip()


_tunnels = {}

def ensure_tunnel(host, local_port):
    key = (host, local_port)
    proc = _tunnels.get(key)
    if proc and proc.poll() is None:
        return
    if proc:
        proc.kill()
    p = subprocess.Popen(
        ["ssh"] + SSH_OPTS + ["-f", "-N", "-L", f"{local_port}:localhost:{VLLM_PORT}", host],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    p.wait(timeout=15)
    _tunnels[key] = p
    time.sleep(1)
    log.info("tunnel %s -> localhost:%d", host, local_port)


def gpu_clean(host):
    """Kill everything holding GPU memory on a remote host. Retries until free."""
    for attempt in range(5):
        pids = ssh(host,
            "nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null",
            timeout=10)
        if not pids.strip():
            log.info("gpu clean on %s (no GPU processes, attempt %d)", host, attempt + 1)
            return
        for pid in pids.strip().split("\n"):
            pid = pid.strip()
            if pid.isdigit():
                ssh(host, f"kill -9 {pid} 2>/dev/null", timeout=5)
        time.sleep(3)
        used = ssh(host, "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits", timeout=10)
        try:
            used_mb = int(used.strip().split()[0])
            if used_mb < 500:
                log.info("gpu clean on %s (%dMB used, attempt %d)", host, used_mb, attempt + 1)
                return
        except (ValueError, IndexError):
            pass
    log.warning("gpu_clean: could not free GPU on %s after 5 attempts", host)


def deploy_model(host, hf_repo, local_port):
    """Kill vLLM, download model, patch config, restart vLLM on remote host."""
    log.info("deploying %s on %s", hf_repo, host)
    gpu_clean(host)

    dl_cmd = (
        f"HF_TOKEN={HF_TOKEN} python3 -c \""
        f"from huggingface_hub import snapshot_download; "
        f"snapshot_download('{hf_repo}', local_dir='{MODEL_PATH}')"
        f"\""
    )
    ssh(host, dl_cmd, timeout=300)

    patch_cmd = (
        "python3 -c \""
        "import json, os; "
        f"cfg=json.load(open('{MODEL_PATH}/config.json')); "
        "rp=cfg.get('rope_parameters',{}); "
        "rp.setdefault('rope_type','default') if isinstance(rp,dict) else None; "
        f"json.dump(cfg,open('{MODEL_PATH}/config.json','w'),indent=2); "
        f"tc_path='{MODEL_PATH}/tokenizer_config.json'; "
        "tc=json.load(open(tc_path)) if os.path.exists(tc_path) else None; "
        "exec('tc[\\\"extra_special_tokens\\\"]={}') if tc and isinstance(tc.get('extra_special_tokens'),list) else None; "
        "json.dump(tc,open(tc_path,'w'),indent=2) if tc else None"
        "\""
    )
    ssh(host, patch_cmd, timeout=30)

    serve_cmd = (
        f"TRITON_CACHE_DIR=/root/.triton_cache "
        f"nohup vllm serve {MODEL_PATH} --host 0.0.0.0 --port {VLLM_PORT} "
        f"--enforce-eager > /tmp/vllm.log 2>&1 &"
    )
    ssh(host, serve_cmd, timeout=15)

    ensure_tunnel(host, local_port)
    wait_for_server(f"http://localhost:{local_port}")
    _model_name_cache.pop(f"http://localhost:{local_port}", None)
    log.info("vLLM ready on %s (localhost:%d)", host, local_port)


# ---------------------------------------------------------------------------
# vLLM inference
# ---------------------------------------------------------------------------

async def wait_for_server_async(url, timeout=600):
    async with httpx.AsyncClient() as c:
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                r = await c.get(f"{url}/health", timeout=5)
                if r.status_code == 200:
                    return True
            except Exception:
                pass
            await asyncio.sleep(3)
    raise TimeoutError(f"{url} not ready after {timeout}s")


def wait_for_server(url, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = httpx.get(f"{url}/health", timeout=5)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(3)
    raise TimeoutError(f"{url} not ready after {timeout}s")


_model_name_cache = {}

async def get_model_name(client, url):
    if url in _model_name_cache:
        return _model_name_cache[url]
    r = await client.get(f"{url}/v1/models", timeout=10)
    r.raise_for_status()
    name = r.json()["data"][0]["id"]
    _model_name_cache[url] = name
    return name


async def compute_loss(client, url, tokens):
    model = await get_model_name(client, url)
    r = await client.post(
        f"{url}/v1/completions",
        json={
            "model": model, "prompt": tokens, "max_tokens": 1,
            "temperature": 0.0, "logprobs": 1, "echo": True,
        },
        timeout=120.0,
    )
    r.raise_for_status()
    lps = r.json()["choices"][0]["logprobs"]["token_logprobs"]
    valid = [lp for lp in lps[1:] if lp is not None]
    return -sum(valid) / len(valid) if valid else float("nan")


# ---------------------------------------------------------------------------
# Dataset (range-request from R2)
# ---------------------------------------------------------------------------

def fetch_sequences(r2, shard_key, indices, seq_len):
    header = r2.range_get(shard_key, 0, 1023)
    buf = io.BytesIO(header)
    buf.read(6)  # magic
    ver = struct.unpack("BB", buf.read(2))
    hl = struct.unpack("<H" if ver[0] == 1 else "<I", buf.read(2 if ver[0] == 1 else 4))[0]
    buf.read(hl)
    data_offset = buf.tell()

    bps = seq_len * 4
    sorted_idx = sorted(set(indices))
    groups, gs, ge = [], sorted_idx[0], sorted_idx[0]
    for i in sorted_idx[1:]:
        if i - ge <= 64:
            ge = i
        else:
            groups.append((gs, ge)); gs = ge = i
    groups.append((gs, ge))

    result = {}
    for gs, ge in groups:
        chunk = r2.range_get(shard_key, data_offset + gs * bps, data_offset + (ge + 1) * bps - 1)
        for idx in range(gs, ge + 1):
            if idx in set(indices):
                off = (idx - gs) * bps
                result[idx] = np.frombuffer(chunk[off:off + bps], dtype="<u4").tolist()
    return result


def get_shard_info(r2, shard_key):
    header = r2.range_get(shard_key, 0, 1023)
    buf = io.BytesIO(header)
    buf.read(6)
    ver = struct.unpack("BB", buf.read(2))
    hl = struct.unpack("<H" if ver[0] == 1 else "<I", buf.read(2 if ver[0] == 1 else 4))[0]
    hdr = eval(buf.read(hl).decode("latin1").strip())
    n = 1
    for s in hdr["shape"]:
        n *= s
    return n


# ---------------------------------------------------------------------------
# Sign test
# ---------------------------------------------------------------------------

async def run_sign_test(r2, king_url, challenger_url, shard_key, challenge_id,
                        block_hash, hotkey, on_progress=None):
    N = EVAL_N
    K = int(binom.isf(EVAL_ALPHA, N, 0.5))
    log.info("sign test: N=%d K=%d alpha=%s", N, K, EVAL_ALPHA)

    n_tokens = get_shard_info(r2, shard_key)
    n_sequences = n_tokens // SEQ_LEN
    actual_N = min(N, n_sequences)

    seed_material = f"{block_hash}:{hotkey}".encode()
    seed = int.from_bytes(hashlib.blake2b(seed_material, digest_size=8).digest(), "little")
    rng = np.random.Generator(np.random.PCG64(seed))
    eval_indices = rng.choice(n_sequences, size=actual_N, replace=False).tolist()

    log.info("fetching %d sequences from %s", actual_N, shard_key)
    seq_cache = fetch_sequences(r2, shard_key, eval_indices, SEQ_LEN)

    s, n, n_ties = 0, 0, 0
    king_sum, chall_sum = 0.0, 0.0
    t0 = time.time()
    outcomes_key = f"eval/{challenge_id}/outcomes.jsonl"
    batch_buf = []

    client = httpx.AsyncClient(timeout=120.0)
    sem = asyncio.Semaphore(CONCURRENCY)

    for i, idx in enumerate(eval_indices):
        tokens = seq_cache[idx]

        async with sem:
            kl, cl = await asyncio.gather(
                compute_loss(client, king_url, tokens),
                compute_loss(client, challenger_url, tokens),
            )

        king_sum += kl
        chall_sum += cl
        outcome = {"seq_idx": idx, "king_loss": round(kl, 6), "challenger_loss": round(cl, 6)}

        if kl == cl:
            n_ties += 1
            outcome["win"] = None
        else:
            n += 1
            win = cl < kl
            if win:
                s += 1
            outcome["win"] = 1 if win else 0

        outcome.update({"s": s, "n": n, "N": actual_N})
        batch_buf.append(outcome)

        if len(batch_buf) >= 100:
            for rec in batch_buf:
                r2.append_jsonl(outcomes_key, rec)
            batch_buf = []
            if on_progress:
                total_done = i + 1
                on_progress(total_done, actual_N, s, n, king_sum, chall_sum, n_ties)

        if n > 0:
            if s >= K:
                log.info("EARLY STOP challenger wins s=%d >= K=%d", s, K)
                break
            remaining = actual_N - (n + n_ties)
            if s + remaining < K:
                log.info("EARLY STOP king holds s=%d remaining=%d", s, remaining)
                break

        if (i + 1) % 500 == 0:
            log.info("progress %d/%d s=%d n=%d wr=%.3f", i + 1, actual_N, s, n, s / n if n else 0)

    for rec in batch_buf:
        r2.append_jsonl(outcomes_key, rec)

    await client.aclose()
    elapsed = time.time() - t0
    total = n + n_ties
    accepted = s >= K

    verdict = {
        "accepted": accepted,
        "verdict": "challenger" if accepted else "king",
        "S_N": s, "K": K, "N": actual_N,
        "n_evaluated": n, "n_ties": n_ties,
        "win_rate": round(s / n, 6) if n else 0,
        "alpha": EVAL_ALPHA,
        "early_stopped": total < actual_N,
        "avg_king_loss": round(king_sum / total, 6) if total else 0,
        "avg_challenger_loss": round(chall_sum / total, 6) if total else 0,
        "wall_time_s": round(elapsed, 1),
        "challenge_id": challenge_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    r2.put(f"eval/{challenge_id}/verdict.json", verdict)
    log.info("verdict: %s (s=%d K=%d wr=%.4f %.1fs)", verdict["verdict"], s, K, verdict["win_rate"], elapsed)
    return verdict


# ---------------------------------------------------------------------------
# Chain
# ---------------------------------------------------------------------------

import re
_REPO_RE = re.compile(REPO_PATTERN)

def scan_reveals(subtensor, netuid, seen):
    try:
        all_reveals = subtensor.get_all_revealed_commitments(netuid)
    except Exception:
        log.exception("failed to fetch reveals")
        return []
    if not all_reveals:
        return []

    new = []
    for hotkey, entries in all_reveals.items():
        if hotkey in seen or not entries:
            continue
        block, data = max(entries, key=lambda e: e[0])
        parts = data.split(":", 2)
        if len(parts) != 3:
            continue
        king_hash, hf_repo, model_hash = parts
        if not _REPO_RE.match(hf_repo.strip()):
            continue
        seen.add(hotkey)
        new.append({
            "hotkey": hotkey, "block": block,
            "king_hash": king_hash.strip(), "hf_repo": hf_repo.strip(),
            "model_hash": model_hash.strip(),
        })
    new.sort(key=lambda x: x["block"])
    return new


def set_weights(subtensor, wallet, netuid, king_hotkey):
    try:
        meta = subtensor.metagraph(netuid)
        if king_hotkey in meta.hotkeys:
            uid = meta.hotkeys.index(king_hotkey)
            subtensor.set_weights(wallet=wallet, netuid=netuid, uids=[uid], weights=[1.0])
            log.info("weights set 100%% to uid=%d (%s)", uid, king_hotkey[:16])
        else:
            log.warning("king hotkey %s not in metagraph", king_hotkey[:16])
    except Exception:
        log.exception("failed to set weights")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def _now():
    return datetime.now(timezone.utc).isoformat()


class State:
    def __init__(self, r2):
        self.r2 = r2
        self.king = {}
        self.queue = []
        self.seen = set()
        self.stats = {"challenges": 0, "accepted": 0, "rejected": 0}
        self.counter = 0
        self.current_eval = None
        self.history = []

    def load(self):
        k = self.r2.get("king/current.json")
        if k:
            self.king = k
        q = self.r2.get("state/queue.json")
        if q:
            self.queue = q.get("pending", [])
        s = self.r2.get("state/seen_hotkeys.json")
        if s:
            self.seen = set(s.get("hotkeys", []))
        st = self.r2.get("state/validator_state.json")
        if st:
            self.stats = st.get("stats", self.stats)
            self.counter = st.get("counter", 0)
        h = self.r2.get("state/dashboard_history.json")
        if h:
            self.history = h.get("history", [])
        log.info("loaded state: king=%s queue=%d seen=%d",
                 self.king.get("king_hash", "none")[:16], len(self.queue), len(self.seen))

    def flush(self):
        self.r2.put("state/validator_state.json", {
            "king": self.king, "queue": self.queue,
            "stats": self.stats, "counter": self.counter, "updated_at": _now(),
        })
        self.r2.put("state/queue.json", {"pending": self.queue, "updated_at": _now()})
        self.r2.put("king/current.json", self.king)
        self.r2.put("state/seen_hotkeys.json", {
            "hotkeys": sorted(self.seen), "updated_at": _now(),
        })

    def event(self, data):
        data.setdefault("timestamp", _now())
        self.r2.append_jsonl("state/history.jsonl", data)

    def next_id(self):
        self.counter += 1
        return f"eval-{self.counter:04d}"

    def enqueue(self, reveal):
        cid = self.next_id()
        entry = {"challenge_id": cid, **reveal, "queued_at": _now()}
        self.queue.append(entry)
        self.stats["challenges"] += 1
        self.flush()
        self.flush_dashboard()
        self.event({"event": "queued", **entry})
        return cid

    def set_king(self, hotkey, hf_repo, king_hash, block, challenge_id="seed"):
        reign = self.king.get("reign_number", 0) + (0 if challenge_id == "seed" else 1)
        self.king = {
            "hotkey": hotkey, "hf_repo": hf_repo, "king_hash": king_hash,
            "reign_number": reign, "crowned_at": _now(),
            "crowned_block": block, "challenge_id": challenge_id,
        }
        self.flush()
        self.flush_dashboard()
        self.event({"event": "king_changed", "hotkey": hotkey, "reign": reign,
                     "challenge_id": challenge_id})

    def record_verdict(self, verdict, challenger_repo, hotkey):
        self.history.insert(0, {
            "challenge_id": verdict["challenge_id"],
            "hotkey": hotkey,
            "challenger_repo": challenger_repo,
            "accepted": verdict["accepted"],
            "verdict": verdict["verdict"],
            "win_rate": verdict["win_rate"],
            "avg_king_loss": verdict["avg_king_loss"],
            "avg_challenger_loss": verdict["avg_challenger_loss"],
            "wall_time_s": verdict["wall_time_s"],
            "timestamp": verdict["timestamp"],
        })
        self.history = self.history[:50]
        self.r2.put("state/dashboard_history.json", {"history": self.history})

    def flush_dashboard(self):
        self.r2.put("dashboard.json", {
            "updated_at": _now(),
            "king": self.king,
            "stats": self.stats,
            "current_eval": self.current_eval,
            "queue": [{"challenge_id": e.get("challenge_id"), "hotkey": e.get("hotkey"),
                        "hf_repo": e.get("hf_repo"), "queued_at": e.get("queued_at")}
                       for e in self.queue],
            "history": self.history,
        })


# ---------------------------------------------------------------------------
# King management
# ---------------------------------------------------------------------------

def sha256_dir(path):
    """SHA256 over sorted safetensors files (matches validation.py)."""
    import hashlib as hl
    h = hl.sha256()
    from pathlib import Path
    for p in sorted(Path(path).glob("*.safetensors")):
        with open(p, "rb") as f:
            while chunk := f.read(1 << 20):
                h.update(chunk)
    return h.hexdigest()


def fork_winner(challenger_repo, king_hash, hotkey, challenge_id):
    """Upload challenger weights to king repo (creates git history)."""
    api = HfApi(token=HF_TOKEN)
    api.create_repo(KING_REPO, exist_ok=True, private=False)
    tmp = "/tmp/teutonic/fork"
    snapshot_download(challenger_repo, local_dir=tmp, token=HF_TOKEN or None,
                      allow_patterns=["*.safetensors"],
                      ignore_patterns=["*.bin", "*.pt", "__pycache__/*"])
    api.upload_folder(
        folder_path=tmp, repo_id=KING_REPO,
        commit_message=f"King #{king_hash[:8]} dethroned by {hotkey[:16]} ({challenge_id})",
        allow_patterns=["*.safetensors"],
    )
    new_hash = sha256_dir(tmp)
    log.info("forked %s -> %s hash=%s", challenger_repo, KING_REPO, new_hash[:16])
    return new_hash


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def process_challenge(state, r2, entry, subtensor, wallet):
    cid = entry["challenge_id"]
    hotkey = entry["hotkey"]
    hf_repo = entry["hf_repo"]
    log.info("processing %s from %s repo=%s", cid, hotkey[:16], hf_repo)

    current_hash = state.king.get("king_hash", "")
    if not current_hash.startswith(entry["king_hash"][:len(entry["king_hash"])]):
        log.info("stale %s: king changed", cid)
        state.event({"event": "stale", "challenge_id": cid, "hotkey": hotkey})
        return

    deploy_model(SSH_CHALLENGER, hf_repo, CHALLENGER_PORT)

    # Deterministic shard selection
    block_hash = "default"
    try:
        block_hash = subtensor.get_block_hash(entry["block"]) or "default"
    except Exception:
        pass

    manifest = r2.get("dataset/v1/manifest.json")
    if not manifest:
        log.error("no dataset manifest")
        return
    n_shards = manifest["total_shards"]
    seed_mat = f"{block_hash}:{hotkey}".encode()
    shard_idx = int.from_bytes(hashlib.blake2b(seed_mat, digest_size=8).digest(), "little") % n_shards
    shard_key = manifest["shards"][shard_idx]["key"]

    king_url = f"http://localhost:{KING_PORT}"
    chall_url = f"http://localhost:{CHALLENGER_PORT}"

    r2.put(f"eval/{cid}/meta.json", {
        "challenge_id": cid, "king_repo": KING_REPO,
        "challenger_repo": hf_repo, "hotkey": hotkey,
        "N": EVAL_N, "alpha": EVAL_ALPHA, "shard": shard_key,
    })

    state.current_eval = {
        "challenge_id": cid, "challenger_repo": hf_repo, "hotkey": hotkey,
        "progress": 0, "total": EVAL_N, "s": 0, "n": 0,
        "win_rate": 0, "avg_king_loss": 0, "avg_challenger_loss": 0,
        "started_at": _now(),
    }
    state.flush_dashboard()

    def _on_progress(done, total, s, n, king_sum, chall_sum, n_ties):
        evaluated = n + n_ties
        state.current_eval.update({
            "progress": done, "total": total, "s": s, "n": n,
            "win_rate": round(s / n, 6) if n else 0,
            "avg_king_loss": round(king_sum / evaluated, 6) if evaluated else 0,
            "avg_challenger_loss": round(chall_sum / evaluated, 6) if evaluated else 0,
        })
        state.flush_dashboard()

    verdict = await run_sign_test(r2, king_url, chall_url, shard_key, cid,
                                  block_hash, hotkey, on_progress=_on_progress)

    state.current_eval = None
    state.record_verdict(verdict, hf_repo, hotkey)

    accepted = verdict.get("accepted", False)
    if accepted:
        state.stats["accepted"] += 1
    else:
        state.stats["rejected"] += 1

    state.flush_dashboard()
    state.event({"event": "eval_completed", "challenge_id": cid,
                 "hotkey": hotkey, "accepted": accepted, **verdict})

    if accepted:
        log.info("DETHRONE! %s wins via %s", hotkey[:16], cid)
        old_hash = state.king.get("king_hash", "")
        new_hash = fork_winner(hf_repo, old_hash, hotkey, cid)
        deploy_model(SSH_KING, KING_REPO, KING_PORT)
        state.set_king(hotkey, KING_REPO, new_hash, entry.get("block", 0), cid)
        set_weights(subtensor, wallet, NETUID, hotkey)

    state.flush()


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if not SSH_KING or not SSH_CHALLENGER:
        log.error("set TEUTONIC_SSH_KING and TEUTONIC_SSH_CHALLENGER")
        sys.exit(1)

    r2 = R2()
    state = State(r2)
    state.load()
    state.flush_dashboard()

    # Upload dashboard HTML to R2
    html_path = os.path.join(os.path.dirname(__file__) or ".", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "rb") as f:
            r2.put_raw("index.html", f.read(), "text/html")
        log.info("uploaded dashboard to R2")

    wallet = bt.wallet(name=WALLET_NAME, hotkey=WALLET_HOTKEY)
    subtensor = bt.subtensor(network=NETWORK)

    # Ensure king is running
    if not state.king:
        king_hash = "seed"
        try:
            tmp = "/tmp/teutonic/king_seed"
            snapshot_download(KING_REPO, local_dir=tmp, token=HF_TOKEN or None,
                              allow_patterns=["*.safetensors"])
            king_hash = sha256_dir(tmp)
        except Exception:
            pass
        state.set_king(wallet.hotkey.ss58_address, KING_REPO, king_hash, subtensor.block)

    def cleanup():
        log.info("cleaning up GPU on both boxes")
        gpu_clean(SSH_KING)
        gpu_clean(SSH_CHALLENGER)
        for proc in _tunnels.values():
            try:
                proc.kill()
            except Exception:
                pass
        log.info("cleanup done")

    import signal
    def _on_signal(sig, frame):
        log.info("received signal %d", sig)
        cleanup()
        sys.exit(0)
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    deploy_model(SSH_KING, KING_REPO, KING_PORT)

    log.info("validator running | king=%s | poll=%ds", state.king.get("king_hash", "")[:16], POLL_INTERVAL)

    try:
        while True:
            try:
                reveals = scan_reveals(subtensor, NETUID, state.seen)
                if reveals:
                    state.flush()
                    for rev in reveals:
                        cid = state.enqueue(rev)
                        log.info("queued %s from %s", cid, rev["hotkey"][:16])

                while state.queue:
                    entry = state.queue.pop(0)
                    state.flush()
                    await process_challenge(state, r2, entry, subtensor, wallet)

            except KeyboardInterrupt:
                break
            except Exception:
                log.exception("tick error")

            await asyncio.sleep(POLL_INTERVAL)
    finally:
        cleanup()


def main_sync():
    asyncio.run(main())


if __name__ == "__main__":
    main_sync()
