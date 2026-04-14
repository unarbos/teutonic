#!/usr/bin/env python3
"""Teutonic validator — single-file king-of-the-hill evaluator.

Polls Bittensor chain for challenger submissions, dispatches evaluations
to a remote eval server (eval_server.py on a GPU box), manages king
lifecycle on HuggingFace, persists all state to R2.
"""
import asyncio
import hashlib
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone

import bittensor as bt
import boto3
import httpx
from botocore.config import Config as BotoConfig
from huggingface_hub import HfApi

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EVAL_N = 10_000
EVAL_ALPHA = 0.001
EVAL_DELTA = float(os.environ.get("TEUTONIC_EVAL_DELTA", "0.01"))
SEQ_LEN = 2048
POLL_INTERVAL = 30
WEIGHT_INTERVAL = 300
NETUID = int(os.environ.get("TEUTONIC_NETUID", "3"))
NETWORK = os.environ.get("TEUTONIC_NETWORK", "finney")
SEED_REPO = os.environ.get("TEUTONIC_SEED_REPO", "unconst/Teutonic-I")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
EVAL_SERVER_URL = os.environ.get("TEUTONIC_EVAL_SERVER", "http://localhost:9000")
WALLET_NAME = os.environ.get("BT_WALLET_NAME", "teutonic")
WALLET_HOTKEY = os.environ.get("BT_WALLET_HOTKEY", "default")

R2_ENDPOINT = os.environ.get("TEUTONIC_R2_ENDPOINT", "")
R2_BUCKET = os.environ.get("TEUTONIC_R2_BUCKET", "")
R2_ACCESS_KEY = os.environ.get("TEUTONIC_R2_ACCESS_KEY", "")
R2_SECRET_KEY = os.environ.get("TEUTONIC_R2_SECRET_KEY", "")
TMC_API_KEY = os.environ.get("TMC_API_KEY", "")

REPO_PATTERN = r"^[^/]+/Teutonic-I-.+$"

TMC_BASE = "https://api.taomarketcap.com/public/v1"

log = logging.getLogger("teutonic")


# ---------------------------------------------------------------------------
# TaoMarketCap
# ---------------------------------------------------------------------------

async def fetch_tmc_data() -> dict | None:
    """Fetch TAO price, SN3 alpha price, and registration burn from TMC API."""
    if not TMC_API_KEY:
        return None
    headers = {"Authorization": TMC_API_KEY}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
            market_resp, subnet_resp, burn_resp = await asyncio.gather(
                client.get(f"{TMC_BASE}/market/market-data/", headers=headers),
                client.get(f"{TMC_BASE}/subnets/{NETUID}/", headers=headers),
                client.get(f"{TMC_BASE}/subnets/burn/{NETUID}/", headers=headers),
            )
        m = market_resp.json()
        s = subnet_resp.json()
        b = burn_resp.json()
        asp = float(s["latest_snapshot"]["alpha_sqrt_price"])
        tao_price = m["current_price"]
        alpha_tao = asp ** 2
        return {
            "tao_price_usd": tao_price,
            "tao_change_24h": m["usd_quote"]["percent_change_24h"],
            "sn3_alpha_price_tao": alpha_tao,
            "sn3_alpha_price_usd": alpha_tao * tao_price,
            "sn3_reg_burn_tao": b[0]["burn"] / 1e9,
        }
    except Exception:
        log.warning("TMC fetch failed", exc_info=True)
        return None

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

    def append_jsonl_batch(self, key, records):
        lines = "".join(json.dumps(r, default=str) + "\n" for r in records)
        existing = b""
        try:
            existing = self.client.get_object(Bucket=R2_BUCKET, Key=key)["Body"].read()
        except Exception:
            pass
        self.client.put_object(
            Bucket=R2_BUCKET, Key=key,
            Body=existing + lines.encode(),
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
# Challenger validation
# ---------------------------------------------------------------------------

_king_config: dict | None = None
_king_config_key: str | None = None

def get_king_config(king_repo: str, king_revision: str = ""):
    """Fetch and cache the king model's config.json from HuggingFace."""
    global _king_config, _king_config_key
    cache_key = f"{king_repo}@{king_revision}"
    if _king_config is not None and _king_config_key == cache_key:
        return _king_config
    try:
        api = HfApi(token=HF_TOKEN or None)
        cfg_path = api.hf_hub_download(king_repo, "config.json",
                                        token=HF_TOKEN or None,
                                        revision=king_revision or None)
        with open(cfg_path) as f:
            _king_config = json.load(f)
            _king_config_key = cache_key
    except Exception:
        log.warning("could not fetch king config.json from %s@%s",
                    king_repo, (king_revision or "HEAD")[:12])
        _king_config = {}
        _king_config_key = cache_key
    return _king_config


def validate_challenger_config(hf_repo: str, king_repo: str = "",
                                king_revision: str = "") -> str | None:
    """Check challenger config.json matches king architecture before deploying.

    Returns None if OK, or a human-readable rejection reason.
    """
    king_cfg = get_king_config(king_repo or SEED_REPO, king_revision)
    if not king_cfg:
        return None

    try:
        api = HfApi(token=HF_TOKEN or None)
        cfg_path = api.hf_hub_download(hf_repo, "config.json", token=HF_TOKEN or None)
        with open(cfg_path) as f:
            challenger_cfg = json.load(f)
    except Exception as e:
        return f"cannot fetch config.json: {e}"

    king_arch = king_cfg.get("architectures", [])
    chall_arch = challenger_cfg.get("architectures", [])
    if king_arch and chall_arch and king_arch != chall_arch:
        return f"architecture mismatch: king={king_arch} challenger={chall_arch}"

    for key in ("vocab_size", "hidden_size", "num_hidden_layers",
                "num_attention_heads", "num_key_value_heads", "head_dim",
                "intermediate_size", "model_type"):
        king_val = king_cfg.get(key)
        chall_val = challenger_cfg.get(key)
        if king_val is not None and chall_val is not None and king_val != chall_val:
            return f"{key} mismatch: king={king_val} challenger={chall_val}"

    st_files = [s for s in api.list_repo_files(hf_repo, token=HF_TOKEN or None)
                if s.endswith(".safetensors")]
    if not st_files:
        return "no .safetensors files in repo"

    return None


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


def set_weights(subtensor, wallet, netuid, king_hotkey) -> bool:
    """Set 100% weight to *king_hotkey*. Returns True on success."""
    try:
        meta = subtensor.metagraph(netuid)
        if king_hotkey in meta.hotkeys:
            uid = meta.hotkeys.index(king_hotkey)
            subtensor.set_weights(wallet=wallet, netuid=netuid, uids=[uid], weights=[1.0])
            log.info("weights set 100%% to uid=%d (%s)", uid, king_hotkey[:16])
            return True
        else:
            log.warning("king hotkey %s not in metagraph", king_hotkey[:16])
            return False
    except Exception:
        log.exception("failed to set weights")
        return False


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
        self.failed_repos: set[str] = set()
        self.evaluated_repos: set[str] = set()
        self.stats = {"queued": 0, "accepted": 0, "rejected": 0, "failed": 0}
        self.counter = 0
        self.current_eval = None
        self.history = []
        self.last_weight_block = 0
        self.last_winner_hotkey: str | None = None
        self.market: dict | None = None
        self.uid_map: dict[str, int] = {}

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
            loaded = st.get("stats", self.stats)
            if "challenges" in loaded and "queued" not in loaded:
                loaded["queued"] = loaded.pop("challenges")
            loaded.setdefault("failed", 0)
            loaded.setdefault("queued", 0)
            self.stats = loaded
            self.counter = st.get("counter", 0)
            self.last_weight_block = st.get("last_weight_block", 0)
            self.last_winner_hotkey = st.get("last_winner_hotkey")
        h = self.r2.get("state/dashboard_history.json")
        if h:
            self.history = h.get("history", [])
        log.info("loaded state: king=%s queue=%d seen=%d",
                 self.king.get("king_hash", "none")[:16], len(self.queue), len(self.seen))

    def flush(self):
        self.r2.put("state/validator_state.json", {
            "king": self.king, "queue": self.queue,
            "stats": self.stats, "counter": self.counter,
            "last_weight_block": self.last_weight_block,
            "last_winner_hotkey": self.last_winner_hotkey,
            "updated_at": _now(),
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
        repo = reveal.get("hf_repo", "")
        hotkey = reveal.get("hotkey", "")
        king_hotkey = self.king.get("hotkey", "")
        if king_hotkey and hotkey == king_hotkey:
            log.info("skipping enqueue: hotkey %s is the current king", hotkey[:16])
            return None
        for existing in self.queue:
            if existing.get("hf_repo") == repo:
                log.info("skipping duplicate repo: %s already queued", repo)
                return None
        if repo in self.evaluated_repos:
            log.info("skipping %s: already evaluated this cycle", repo)
            return None
        cid = self.next_id()
        entry = {"challenge_id": cid, **reveal, "queued_at": _now()}
        self.queue.append(entry)
        self.stats["queued"] += 1
        self.flush()
        self.flush_dashboard()
        self.event({"event": "queued", **entry})
        return cid

    def set_king(self, hotkey, hf_repo, king_hash, block, challenge_id="seed",
                  king_revision=""):
        global _king_config, _king_config_key
        _king_config = None
        _king_config_key = None
        self.failed_repos.clear()
        self.evaluated_repos.clear()
        reign = self.king.get("reign_number", 0) + (0 if challenge_id == "seed" else 1)
        prev = self.king.copy() if self.king else None
        if prev:
            prev.pop("previous_king", None)
        self.king = {
            "hotkey": hotkey, "hf_repo": hf_repo, "king_hash": king_hash,
            "king_revision": king_revision,
            "reign_number": reign, "crowned_at": _now(),
            "crowned_block": block, "challenge_id": challenge_id,
            "previous_king": prev,
        }
        self.flush()
        self.flush_dashboard()
        self.event({"event": "king_changed", "hotkey": hotkey, "reign": reign,
                     "challenge_id": challenge_id})

    def record_verdict(self, verdict, challenger_repo, hotkey):
        king_loss = verdict["avg_king_loss"]
        chall_loss = verdict["avg_challenger_loss"]
        self.history.insert(0, {
            "challenge_id": verdict["challenge_id"],
            "hotkey": hotkey,
            "uid": self.uid_map.get(hotkey, "?"),
            "challenger_repo": challenger_repo,
            "accepted": verdict["accepted"],
            "verdict": verdict["verdict"],
            "mu_hat": verdict.get("mu_hat", 0),
            "lcb": verdict.get("lcb", 0),
            "delta": verdict.get("delta", 0),
            "avg_king_loss": king_loss,
            "avg_challenger_loss": chall_loss,
            "best_loss": min(king_loss, chall_loss),
            "wall_time_s": verdict["wall_time_s"],
            "timestamp": verdict["timestamp"],
        })
        self.r2.put("state/dashboard_history.json", {"history": self.history})

    def refresh_uid_map(self, subtensor, netuid):
        try:
            meta = subtensor.metagraph(netuid)
            self.uid_map = {hk: uid for uid, hk in enumerate(meta.hotkeys)}
        except Exception:
            log.warning("failed to refresh uid_map", exc_info=True)

    def flush_dashboard(self):
        payload = {
            "updated_at": _now(),
            "king": self.king,
            "stats": self.stats,
            "current_eval": self.current_eval,
            "queue": [{"challenge_id": e.get("challenge_id"), "hotkey": e.get("hotkey"),
                        "uid": self.uid_map.get(e.get("hotkey", ""), "?"),
                        "hf_repo": e.get("hf_repo"), "queued_at": e.get("queued_at"),
                        "block": e.get("block")}
                       for e in self.queue],
            "history": self.history,
        }
        if self.market:
            payload["market"] = self.market
        self.r2.put("dashboard.json", payload)



# ---------------------------------------------------------------------------
# King liveness
# ---------------------------------------------------------------------------

def check_king_alive(state):
    """Verify king repo is still accessible at pinned revision. Auto-dethrone if not."""
    repo = state.king.get("hf_repo", "")
    rev = state.king.get("king_revision", "")
    if not repo or not rev:
        return True
    try:
        HfApi(token=HF_TOKEN or None).model_info(repo, revision=rev)
        return True
    except Exception:
        log.warning("KING REPO UNAVAILABLE: %s@%s — auto-dethroning", repo, rev[:12])
        prev = state.king.get("previous_king")
        if prev and prev.get("hf_repo"):
            log.info("reverting to previous king: %s@%s",
                     prev["hf_repo"], prev.get("king_revision", "?")[:12])
            state.king = prev
            state.flush()
            state.flush_dashboard()
            state.event({"event": "king_dethroned_absent",
                         "lost_repo": repo, "lost_revision": rev[:12],
                         "reverted_to": prev.get("hf_repo")})
        else:
            log.error("no previous king to revert to — king repo is gone")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def process_challenge(state, r2, entry, subtensor, wallet, *, check_stale=True):
    cid = entry["challenge_id"]
    hotkey = entry["hotkey"]
    hf_repo = entry["hf_repo"]
    log.info("processing %s from %s repo=%s", cid, hotkey[:16], hf_repo)

    king_hotkey = state.king.get("hotkey", "")
    if king_hotkey and hotkey == king_hotkey:
        log.info("skipping %s: challenger hotkey %s is the current king", cid, hotkey[:16])
        return

    if hf_repo in state.failed_repos:
        log.info("skipping %s: repo %s previously failed", cid, hf_repo)
        return

    if hf_repo in state.evaluated_repos:
        log.info("skipping %s: repo %s already evaluated this cycle", cid, hf_repo)
        return

    rejection = validate_challenger_config(
        hf_repo,
        king_repo=state.king.get("hf_repo", ""),
        king_revision=state.king.get("king_revision", ""),
    )
    if rejection:
        log.warning("rejecting %s (%s): %s", cid, hf_repo, rejection)
        state.failed_repos.add(hf_repo)
        state.event({"event": "config_rejected", "challenge_id": cid,
                     "hf_repo": hf_repo, "reason": rejection})
        return

    if check_stale:
        current_hash = state.king.get("king_hash", "")
        entry_king_hash = entry.get("king_hash", "")
        if current_hash and entry_king_hash and not current_hash.startswith(entry_king_hash[:len(entry_king_hash)]):
            log.info("stale %s: king changed (entry=%s current=%s)", cid, entry_king_hash[:16], current_hash[:16])
            state.event({"event": "stale", "challenge_id": cid, "hotkey": hotkey})
            return

    try:
        challenger_info = HfApi(token=HF_TOKEN or None).model_info(hf_repo, revision="main")
        challenger_revision = challenger_info.sha
        log.info("challenger %s pinned at revision %s", hf_repo, challenger_revision[:12])
    except Exception:
        log.warning("cannot get commit SHA for %s, skipping", hf_repo)
        state.failed_repos.add(hf_repo)
        return

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

    king_repo = state.king.get("hf_repo", SEED_REPO)
    king_revision = state.king.get("king_revision", "")

    r2.put(f"eval/{cid}/meta.json", {
        "challenge_id": cid, "king_repo": king_repo,
        "king_revision": king_revision,
        "challenger_repo": hf_repo, "challenger_revision": challenger_revision,
        "hotkey": hotkey,
        "N": EVAL_N, "alpha": EVAL_ALPHA, "delta": EVAL_DELTA, "shard": shard_key,
    })

    state.current_eval = {
        "challenge_id": cid, "challenger_repo": hf_repo, "hotkey": hotkey,
        "progress": 0, "total": EVAL_N, "mu_hat": 0,
        "avg_king_loss": 0, "avg_challenger_loss": 0,
        "started_at": _now(),
    }
    state.flush_dashboard()

    verdict = None
    async with httpx.AsyncClient(timeout=httpx.Timeout(1800.0, connect=30.0)) as client:
        resp = await client.post(f"{EVAL_SERVER_URL}/eval", json={
            "king_repo": king_repo,
            "challenger_repo": hf_repo,
            "block_hash": block_hash,
            "hotkey": hotkey,
            "shard_key": shard_key,
            "king_hash": state.king.get("king_hash", ""),
            "king_revision": king_revision,
            "challenger_revision": challenger_revision,
            "eval_n": EVAL_N,
            "alpha": EVAL_ALPHA,
            "delta": EVAL_DELTA,
            "seq_len": SEQ_LEN,
        })
        resp.raise_for_status()
        eval_id = resp.json()["eval_id"]
        log.info("eval %s dispatched to eval server as %s", cid, eval_id)

        async with client.stream("GET", f"{EVAL_SERVER_URL}/eval/{eval_id}/stream",
                                  timeout=httpx.Timeout(1800.0)) as stream:
            async for line in stream.aiter_lines():
                if not line.startswith("data: "):
                    continue
                event = json.loads(line[6:])

                if event["type"] == "progress":
                    d = event["data"]
                    state.current_eval.update({
                        "progress": d.get("done", 0),
                        "total": d.get("total", EVAL_N),
                        "mu_hat": d.get("mu_hat", 0),
                        "avg_king_loss": d.get("avg_king_loss", 0),
                        "avg_challenger_loss": d.get("avg_challenger_loss", 0),
                    })
                    state.flush_dashboard()

                elif event["type"] == "verdict":
                    verdict = event["data"]
                    verdict["challenge_id"] = cid
                    break

                elif event["type"] == "error":
                    raise RuntimeError(f"eval server error: {event['data']}")

    if not verdict:
        raise RuntimeError("eval stream ended without verdict")

    r2.put(f"eval/{cid}/verdict.json", verdict)
    log.info("verdict: %s (mu_hat=%.6f lcb=%.6f delta=%.6f %.1fs)",
             verdict["verdict"], verdict.get("mu_hat", 0), verdict.get("lcb", 0),
             verdict.get("delta", 0), verdict["wall_time_s"])

    state.current_eval = None
    state.evaluated_repos.add(hf_repo)
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
        log.info("DETHRONE! %s wins via %s (repo=%s rev=%s)",
                 hotkey[:16], cid, hf_repo, challenger_revision[:12])
        state.set_king(hotkey, hf_repo, entry.get("model_hash", ""),
                       entry.get("block", 0), cid,
                       king_revision=challenger_revision)
        state.last_winner_hotkey = hotkey
        if set_weights(subtensor, wallet, NETUID, hotkey):
            state.last_weight_block = subtensor.block

    state.flush()


async def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if not EVAL_SERVER_URL:
        log.error("set TEUTONIC_EVAL_SERVER")
        sys.exit(1)

    r2 = R2()
    state = State(r2)
    state.load()
    if not args.seen:
        state.seen.clear()
        log.info("--no-seen: will continuously re-evaluate all challengers")
    state.flush_dashboard()

    html_path = os.path.join(os.path.dirname(__file__) or ".", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "rb") as f:
            r2.put_raw("index.html", f.read(), "text/html")
        log.info("uploaded dashboard to R2")

    wallet = bt.wallet(name=WALLET_NAME, hotkey=WALLET_HOTKEY)
    subtensor = bt.subtensor(network=NETWORK)

    if not state.king:
        seed_revision = ""
        try:
            seed_info = HfApi(token=HF_TOKEN or None).model_info(SEED_REPO)
            seed_revision = seed_info.sha
            log.info("seed king %s at revision %s", SEED_REPO, seed_revision[:12])
        except Exception:
            log.warning("could not get seed king revision from %s", SEED_REPO)
        state.set_king(wallet.hotkey.ss58_address, SEED_REPO, "seed",
                       subtensor.block, king_revision=seed_revision)

    # Verify eval server is reachable
    try:
        r = httpx.get(f"{EVAL_SERVER_URL}/health", timeout=10)
        r.raise_for_status()
        health = r.json()
        log.info("eval server healthy: %s", health)
    except Exception:
        log.warning("eval server at %s not reachable at startup (will retry on eval)", EVAL_SERVER_URL)

    def _on_signal(sig, frame):
        log.info("received signal %d, shutting down", sig)
        sys.exit(0)
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    log.info("validator running | king=%s@%s | eval_server=%s | poll=%ds",
             state.king.get("hf_repo", "?"),
             state.king.get("king_revision", "?")[:12],
             EVAL_SERVER_URL, POLL_INTERVAL)

    while True:
        try:
            if not check_king_alive(state):
                log.warning("king repo check failed, skipping this tick")
                await asyncio.sleep(POLL_INTERVAL)
                continue

            state.refresh_uid_map(subtensor, NETUID)

            tmc = await fetch_tmc_data()
            if tmc:
                state.market = tmc

            # Phase 1: process new (unseen) challengers first
            reveals = scan_reveals(subtensor, NETUID, state.seen)
            if reveals:
                state.flush()
                for rev in reveals:
                    cid = state.enqueue(rev)
                    if cid:
                        log.info("queued %s from %s (new)", cid, rev["hotkey"][:16])

            while state.queue:
                entry = state.queue.pop(0)
                state.current_eval = {
                    "challenge_id": entry.get("challenge_id", "?"),
                    "challenger_repo": entry.get("hf_repo", ""),
                    "hotkey": entry.get("hotkey", ""),
                    "progress": 0, "total": EVAL_N, "mu_hat": 0,
                    "avg_king_loss": 0, "avg_challenger_loss": 0,
                    "loading": True,
                    "started_at": _now(),
                }
                state.flush_dashboard()
                state.flush()
                try:
                    await process_challenge(state, r2, entry, subtensor, wallet,
                                            check_stale=args.seen)
                except Exception:
                    log.exception("eval failed: %s", entry.get("challenge_id"))
                    state.stats["failed"] += 1
                    state.current_eval = None
                    state.flush_dashboard()

                fresh = scan_reveals(subtensor, NETUID, state.seen)
                if fresh:
                    state.flush()
                    for rev in fresh:
                        cid = state.enqueue(rev)
                        if cid:
                            log.info("queued %s from %s (new, mid-cycle)", cid, rev["hotkey"][:16])

            state.current_eval = None
            state.flush_dashboard()

            # Phase 2: re-evaluate already-seen challengers when none are unseen
            if args.seen and not state.queue:
                log.info("all miners seen — starting re-evaluation cycle")
                state.evaluated_repos.clear()
                throwaway_seen = set()
                reeval_reveals = scan_reveals(subtensor, NETUID, throwaway_seen)
                king_hk = state.king.get("hotkey", "")
                reeval_reveals = [r for r in reeval_reveals
                                  if r["hotkey"] != king_hk
                                  and r["hf_repo"] not in state.failed_repos]
                for rev in reeval_reveals:
                    cid = state.enqueue(rev)
                    if cid:
                        log.info("queued %s from %s (re-eval)", cid, rev["hotkey"][:16])

                while state.queue:
                    entry = state.queue.pop(0)
                    state.current_eval = {
                        "challenge_id": entry.get("challenge_id", "?"),
                        "challenger_repo": entry.get("hf_repo", ""),
                        "hotkey": entry.get("hotkey", ""),
                        "progress": 0, "total": EVAL_N, "mu_hat": 0,
                        "avg_king_loss": 0, "avg_challenger_loss": 0,
                        "loading": True,
                        "started_at": _now(),
                    }
                    state.flush_dashboard()
                    state.flush()
                    try:
                        await process_challenge(state, r2, entry, subtensor, wallet,
                                                check_stale=False)
                    except Exception:
                        log.exception("eval failed: %s", entry.get("challenge_id"))
                        state.stats["failed"] += 1
                        state.current_eval = None
                        state.flush_dashboard()

                    fresh = scan_reveals(subtensor, NETUID, state.seen)
                    if fresh:
                        log.info("new miners appeared during re-eval, prioritizing them")
                        for rev in fresh:
                            cid = state.enqueue(rev)
                            if cid:
                                log.info("queued %s from %s (new, interrupt)", cid, rev["hotkey"][:16])
                        break

                state.current_eval = None
                state.flush_dashboard()

            if not args.seen:
                state.seen.clear()
                state.evaluated_repos.clear()
                state.flush()

            try:
                current_block = subtensor.block
                if current_block - state.last_weight_block >= WEIGHT_INTERVAL:
                    winner = state.last_winner_hotkey
                    if winner:
                        log.info("periodic weight set at block %d (last=%d) to winner %s",
                                 current_block, state.last_weight_block, winner[:16])
                        if set_weights(subtensor, wallet, NETUID, winner):
                            state.last_weight_block = current_block
                            state.flush()
                    else:
                        log.info("skipping periodic weight set: no duel winner yet")
            except Exception:
                log.exception("periodic weight-set failed")

        except KeyboardInterrupt:
            break
        except Exception:
            log.exception("tick error")

        await asyncio.sleep(POLL_INTERVAL)


def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seen", action=argparse.BooleanOptionalAction, default=True,
                   help="Track seen hotkeys to avoid re-evaluating (default: True). "
                        "Use --no-seen to continuously cycle all challengers.")
    return p.parse_args()


def main_sync():
    asyncio.run(main())


if __name__ == "__main__":
    main_sync()
