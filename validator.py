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

HIPPIUS_ENDPOINT = os.environ.get("TEUTONIC_HIPPIUS_ENDPOINT", "https://s3.hippius.com")
HIPPIUS_BUCKET = os.environ.get("TEUTONIC_HIPPIUS_BUCKET", "teutonic-sn3")
HIPPIUS_ACCESS_KEY = os.environ.get("TEUTONIC_HIPPIUS_ACCESS_KEY", "")
HIPPIUS_SECRET_KEY = os.environ.get("TEUTONIC_HIPPIUS_SECRET_KEY", "")

DS_ENDPOINT = os.environ.get("TEUTONIC_DS_ENDPOINT", "")
DS_BUCKET = os.environ.get("TEUTONIC_DS_BUCKET", "")
DS_ACCESS_KEY = os.environ.get("TEUTONIC_DS_ACCESS_KEY", "")
DS_SECRET_KEY = os.environ.get("TEUTONIC_DS_SECRET_KEY", "")

TMC_API_KEY = os.environ.get("TMC_API_KEY", "")

DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "")
DISCORD_CHANNEL_ID = os.environ.get("DISCORD_CHANNEL_ID", "")

BATCH_MAX = int(os.environ.get("TEUTONIC_BATCH_MAX", "5"))

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
# Discord notifications
# ---------------------------------------------------------------------------

async def notify_new_king(king_info: dict, verdict: dict | None = None):
    """Post a message to Discord when a new king is crowned."""
    if not DISCORD_BOT_TOKEN or not DISCORD_CHANNEL_ID:
        return
    repo = king_info.get("hf_repo", "?")
    hotkey = king_info.get("hotkey", "?")
    reign = king_info.get("reign_number", 0)
    revision = king_info.get("king_revision", "")[:12]

    lines = [
        f"**New King of Subnet 3!**",
        f"**Repo:** `{repo}`" + (f" (`{revision}`)" if revision else ""),
        f"**Hotkey:** `{hotkey[:16]}...`",
        f"**Reign:** #{reign}",
    ]
    if verdict:
        mu = verdict.get("mu_hat", 0)
        king_loss = verdict.get("avg_king_loss", 0)
        chall_loss = verdict.get("avg_challenger_loss", 0)
        wall = verdict.get("wall_time_s", 0)
        lines.append(f"**Eval:** challenger loss {chall_loss:.4f} vs king loss {king_loss:.4f} (μ̂={mu:.6f}, {wall:.0f}s)")
    prev = king_info.get("previous_king")
    if prev and prev.get("hf_repo"):
        lines.append(f"**Dethroned:** `{prev['hf_repo']}`")

    embed = {
        "title": "👑 New King Crowned",
        "description": "\n".join(lines),
        "color": 0xFFD700,
    }

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            resp = await client.post(
                f"https://discord.com/api/v10/channels/{DISCORD_CHANNEL_ID}/messages",
                headers={"Authorization": f"Bot {DISCORD_BOT_TOKEN}",
                         "Content-Type": "application/json"},
                json={"embeds": [embed]},
            )
            if resp.status_code < 300:
                log.info("discord notification sent for reign #%d", reign)
            else:
                log.warning("discord notification failed: %d %s", resp.status_code, resp.text[:200])
    except Exception:
        log.warning("discord notification error", exc_info=True)


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
        if HIPPIUS_ACCESS_KEY and HIPPIUS_SECRET_KEY:
            self._hippius = boto3.client(
                "s3", endpoint_url=HIPPIUS_ENDPOINT,
                aws_access_key_id=HIPPIUS_ACCESS_KEY,
                aws_secret_access_key=HIPPIUS_SECRET_KEY,
                region_name="decentralized",
                config=BotoConfig(
                    signature_version="s3v4",
                    retries={"max_attempts": 3, "mode": "adaptive"},
                    s3={"addressing_style": "path"},
                ),
            )
        else:
            self._hippius = None

        if DS_ACCESS_KEY and DS_SECRET_KEY and DS_ENDPOINT:
            self._ds_client = boto3.client(
                "s3", endpoint_url=DS_ENDPOINT,
                aws_access_key_id=DS_ACCESS_KEY,
                aws_secret_access_key=DS_SECRET_KEY,
                region_name="decentralized",
                config=BotoConfig(
                    signature_version="s3v4",
                    retries={"max_attempts": 3, "mode": "adaptive"},
                    s3={"addressing_style": "path"},
                ),
            )
            self._ds_bucket = DS_BUCKET
            log.info("dataset store: %s bucket=%s", DS_ENDPOINT, DS_BUCKET)
        else:
            self._ds_client = None
            self._ds_bucket = None

    def put_dashboard(self, key, data):
        body = json.dumps(data, default=str).encode()
        ct = "application/json"
        if self._hippius:
            self._hippius.put_object(
                Bucket=HIPPIUS_BUCKET, Key=key, Body=body, ContentType=ct,
            )
        else:
            self.client.put_object(
                Bucket=R2_BUCKET, Key=key, Body=body, ContentType=ct,
            )

    def put_dashboard_raw(self, key, body, content_type):
        if self._hippius:
            self._hippius.put_object(
                Bucket=HIPPIUS_BUCKET, Key=key, Body=body,
                ContentType=content_type,
            )
        else:
            self.client.put_object(
                Bucket=R2_BUCKET, Key=key, Body=body,
                ContentType=content_type,
            )

    def put(self, key, data):
        try:
            self.client.put_object(
                Bucket=R2_BUCKET, Key=key,
                Body=json.dumps(data, default=str).encode(),
                ContentType="application/json",
            )
        except Exception:
            log.warning("R2 put failed for %s (non-fatal)", key)

    def get(self, key):
        try:
            return json.loads(
                self.client.get_object(Bucket=R2_BUCKET, Key=key)["Body"].read()
            )
        except Exception:
            return None

    def ds_get(self, key):
        """Read JSON from the dataset store (Hippius), falling back to R2."""
        if self._ds_client:
            try:
                return json.loads(
                    self._ds_client.get_object(
                        Bucket=self._ds_bucket, Key=key
                    )["Body"].read()
                )
            except Exception:
                pass
        return self.get(key)

    def append_jsonl(self, key, record):
        try:
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
        except Exception:
            log.warning("R2 append_jsonl failed for %s (non-fatal)", key)

    def append_jsonl_batch(self, key, records):
        try:
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
        except Exception:
            log.warning("R2 append_jsonl_batch failed for %s (non-fatal)", key)

    def put_raw(self, key, body, content_type):
        try:
            self.client.put_object(
                Bucket=R2_BUCKET, Key=key, Body=body, ContentType=content_type,
            )
        except Exception:
            log.warning("R2 put_raw failed for %s (non-fatal)", key)

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

    def replenish_reeval(self, subtensor, netuid):
        """Fill queue with re-eval candidates so the dashboard never shows empty."""
        self.evaluated_repos.clear()
        throwaway_seen = set()
        reeval_reveals = scan_reveals(subtensor, netuid, throwaway_seen)
        king_hk = self.king.get("hotkey", "")
        reeval_reveals = [r for r in reeval_reveals
                          if r["hotkey"] != king_hk
                          and r["hf_repo"] not in self.failed_repos]
        count = 0
        for rev in reeval_reveals:
            rev["reeval"] = True
            cid = self.enqueue(rev)
            if cid:
                count += 1
                log.info("queued %s from %s (re-eval)", cid, rev["hotkey"][:16])
        if count:
            log.info("replenished queue with %d re-eval candidates", count)
        return count

    def flush_dashboard(self):
        payload = {
            "updated_at": _now(),
            "king": self.king,
            "stats": self.stats,
            "current_eval": self.current_eval,
            "queue": [{"challenge_id": e.get("challenge_id"), "hotkey": e.get("hotkey"),
                        "uid": self.uid_map.get(e.get("hotkey", ""), "?"),
                        "hf_repo": e.get("hf_repo"), "queued_at": e.get("queued_at"),
                        "block": e.get("block"), "reeval": e.get("reeval", False)}
                       for e in self.queue],
            "history": self.history,
        }
        if self.market:
            payload["market"] = self.market
        self.r2.put_dashboard("dashboard.json", payload)



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

    manifest = r2.ds_get("dataset/v2/manifest.json")
    if not manifest:
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
        await notify_new_king(state.king, verdict)
        if set_weights(subtensor, wallet, NETUID, hotkey):
            state.last_weight_block = subtensor.block

    state.flush()


async def process_batch(state, r2, batch_entries, subtensor, wallet):
    """Evaluate a batch of challengers against shared king baseline, crown the best."""

    # --- Phase 0: Validate and prepare entries ---
    valid_entries = []
    for entry in batch_entries:
        cid = entry["challenge_id"]
        hotkey = entry["hotkey"]
        hf_repo = entry["hf_repo"]

        king_hotkey = state.king.get("hotkey", "")
        if king_hotkey and hotkey == king_hotkey:
            log.info("batch: skipping %s: hotkey is current king", cid)
            continue
        if hf_repo in state.failed_repos:
            log.info("batch: skipping %s: repo %s previously failed", cid, hf_repo)
            continue
        if hf_repo in state.evaluated_repos:
            log.info("batch: skipping %s: repo %s already evaluated", cid, hf_repo)
            continue

        rejection = validate_challenger_config(
            hf_repo,
            king_repo=state.king.get("hf_repo", ""),
            king_revision=state.king.get("king_revision", ""),
        )
        if rejection:
            log.warning("batch: rejecting %s (%s): %s", cid, hf_repo, rejection)
            state.failed_repos.add(hf_repo)
            state.event({"event": "config_rejected", "challenge_id": cid,
                         "hf_repo": hf_repo, "reason": rejection})
            continue

        try:
            info = HfApi(token=HF_TOKEN or None).model_info(hf_repo, revision="main")
            entry["challenger_revision"] = info.sha
            log.info("batch: %s pinned at %s", hf_repo, info.sha[:12])
        except Exception:
            log.warning("batch: cannot get SHA for %s, skipping", hf_repo)
            state.failed_repos.add(hf_repo)
            continue

        valid_entries.append(entry)

    if not valid_entries:
        log.info("batch: no valid entries after validation")
        return

    log.info("batch: %d valid challengers", len(valid_entries))

    # --- Phase 1: Compute batch-level shard + seed ---
    ref_block = valid_entries[0]["block"]
    block_hash = "default"
    try:
        block_hash = subtensor.get_block_hash(ref_block) or "default"
    except Exception:
        pass

    king_hash = state.king.get("king_hash", "")
    seed_str = f"{block_hash}:{king_hash}"

    manifest = r2.get("dataset/v1/manifest.json")
    if not manifest:
        log.error("batch: no dataset manifest")
        return
    n_shards = manifest["total_shards"]
    shard_idx = int.from_bytes(
        hashlib.blake2b(seed_str.encode(), digest_size=8).digest(), "little"
    ) % n_shards
    shard_key = manifest["shards"][shard_idx]["key"]

    king_repo = state.king.get("hf_repo", SEED_REPO)
    king_revision = state.king.get("king_revision", "")

    # --- Phase 2: Compute king baseline ---
    log.info("batch: computing king baseline on shard %s", shard_key)
    state.current_eval = {
        "challenge_id": "batch-king-baseline",
        "challenger_repo": "(king baseline)",
        "hotkey": "",
        "progress": 0, "total": EVAL_N, "mu_hat": 0,
        "avg_king_loss": 0, "avg_challenger_loss": 0,
        "loading": True,
        "started_at": _now(),
        "batch_total": len(valid_entries),
    }
    state.flush_dashboard()

    async with httpx.AsyncClient(timeout=httpx.Timeout(1800.0, connect=30.0)) as client:
        resp = await client.post(f"{EVAL_SERVER_URL}/eval/king-baseline", json={
            "king_repo": king_repo,
            "shard_key": shard_key,
            "seed_str": seed_str,
            "king_hash": king_hash,
            "king_revision": king_revision,
            "eval_n": EVAL_N,
            "seq_len": SEQ_LEN,
        })
        resp.raise_for_status()
        eval_id = resp.json()["eval_id"]
        log.info("batch: king baseline dispatched as %s", eval_id)

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
                        "avg_king_loss": d.get("avg_king_loss", 0),
                    })
                    state.flush_dashboard()
                elif event["type"] == "verdict":
                    baseline_result = event["data"]
                    log.info("batch: king baseline complete: avg_king_loss=%.6f (%.1fs)",
                             baseline_result["avg_king_loss"], baseline_result["wall_time_s"])
                    break
                elif event["type"] == "error":
                    raise RuntimeError(f"king baseline error: {event['data']}")

    # --- Phase 3: Evaluate each challenger ---
    verdicts = []

    for i, entry in enumerate(valid_entries):
        cid = entry["challenge_id"]
        hotkey = entry["hotkey"]
        hf_repo = entry["hf_repo"]
        challenger_revision = entry["challenger_revision"]

        log.info("batch: evaluating challenger %d/%d: %s (%s)",
                 i + 1, len(valid_entries), cid, hf_repo)

        r2.put(f"eval/{cid}/meta.json", {
            "challenge_id": cid, "king_repo": king_repo,
            "king_revision": king_revision,
            "challenger_repo": hf_repo, "challenger_revision": challenger_revision,
            "hotkey": hotkey, "batch_mode": True,
            "N": EVAL_N, "alpha": EVAL_ALPHA, "delta": EVAL_DELTA, "shard": shard_key,
        })

        state.current_eval = {
            "challenge_id": cid, "challenger_repo": hf_repo, "hotkey": hotkey,
            "progress": 0, "total": EVAL_N, "mu_hat": 0,
            "avg_king_loss": 0, "avg_challenger_loss": 0,
            "started_at": _now(),
            "batch_index": i + 1,
            "batch_total": len(valid_entries),
        }
        state.flush_dashboard()

        verdict = None
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(1800.0, connect=30.0)) as client:
                resp = await client.post(f"{EVAL_SERVER_URL}/eval/challenger", json={
                    "challenger_repo": hf_repo,
                    "challenger_revision": challenger_revision,
                    "alpha": EVAL_ALPHA,
                    "delta": EVAL_DELTA,
                })
                resp.raise_for_status()
                eval_id = resp.json()["eval_id"]

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

        except Exception:
            log.exception("batch: eval failed for %s", cid)
            state.stats["failed"] += 1
            state.current_eval = None
            state.flush_dashboard()
            continue

        if not verdict:
            log.error("batch: no verdict for %s", cid)
            state.stats["failed"] += 1
            continue

        r2.put(f"eval/{cid}/verdict.json", verdict)
        log.info("batch: %s verdict: %s (mu_hat=%.6f lcb=%.6f)",
                 cid, verdict["verdict"], verdict.get("mu_hat", 0), verdict.get("lcb", 0))

        state.evaluated_repos.add(hf_repo)
        state.record_verdict(verdict, hf_repo, hotkey)

        if verdict.get("accepted", False):
            state.stats["accepted"] += 1
            verdicts.append((entry, verdict))
        else:
            state.stats["rejected"] += 1

        state.event({"event": "eval_completed", "challenge_id": cid,
                     "hotkey": hotkey, "accepted": verdict.get("accepted", False),
                     **verdict})

    # --- Phase 4: Select best ---
    state.current_eval = None

    if not verdicts:
        log.info("batch: no challengers accepted (%d evaluated)", len(valid_entries))
        state.flush_dashboard()
        state.flush()
        return

    best_entry, best_verdict = max(verdicts, key=lambda x: x[1].get("mu_hat", 0))
    best_cid = best_entry["challenge_id"]
    best_hotkey = best_entry["hotkey"]
    best_repo = best_entry["hf_repo"]
    best_revision = best_entry["challenger_revision"]

    log.info("BATCH WINNER: %s from %s (mu_hat=%.6f, %d/%d accepted)",
             best_cid, best_hotkey[:16], best_verdict.get("mu_hat", 0),
             len(verdicts), len(valid_entries))

    state.set_king(best_hotkey, best_repo, best_entry.get("model_hash", ""),
                   best_entry.get("block", 0), best_cid,
                   king_revision=best_revision)
    state.last_winner_hotkey = best_hotkey
    await notify_new_king(state.king, best_verdict)
    if set_weights(subtensor, wallet, NETUID, best_hotkey):
        state.last_weight_block = subtensor.block

    state.flush()
    state.flush_dashboard()
    state.event({
        "event": "batch_completed",
        "batch_size": len(valid_entries),
        "accepted_count": len(verdicts),
        "winner_challenge_id": best_cid,
        "winner_hotkey": best_hotkey,
        "winner_mu_hat": best_verdict.get("mu_hat", 0),
    })


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
    if args.seen:
        log.info("--seen: will re-evaluate old challengers when queue is empty")
    else:
        log.info("new-only mode: will idle when all hotkeys have been seen")
    state.flush_dashboard()

    html_path = os.path.join(os.path.dirname(__file__) or ".", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "rb") as f:
            html_bytes = f.read()
        r2.put_dashboard_raw("index.html", html_bytes, "text/html")
        log.info("uploaded dashboard to Hippius")

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

    log.info("validator running | king=%s@%s | eval_server=%s | poll=%ds | batch_max=%d",
             state.king.get("hf_repo", "?"),
             state.king.get("king_revision", "?")[:12],
             EVAL_SERVER_URL, POLL_INTERVAL, BATCH_MAX)

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

            reveals = scan_reveals(subtensor, NETUID, state.seen)
            if reveals:
                state.flush()
                for rev in reveals:
                    cid = state.enqueue(rev)
                    if cid:
                        log.info("queued %s from %s (new)", cid, rev["hotkey"][:16])

            if BATCH_MAX <= 1:
                # Legacy single-eval mode (unchanged)
                while state.queue:
                    entry = state.queue.pop(0)
                    is_reeval = entry.get("reeval", False)
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
                                                check_stale=not is_reeval and args.seen)
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
                        new_items = [e for e in state.queue if not e.get("reeval")]
                        reeval_items = [e for e in state.queue if e.get("reeval")]
                        state.queue = new_items + reeval_items
            else:
                # Batch mode: evaluate all pending, crown the best
                while state.queue:
                    new_items = [e for e in state.queue if not e.get("reeval")]
                    reeval_items = [e for e in state.queue if e.get("reeval")]
                    batch = new_items[:BATCH_MAX]
                    remaining_slots = BATCH_MAX - len(batch)
                    if remaining_slots > 0:
                        batch.extend(reeval_items[:remaining_slots])

                    batch_ids = {e["challenge_id"] for e in batch}
                    state.queue = [e for e in state.queue if e["challenge_id"] not in batch_ids]

                    log.info("batch: collected %d challengers (max=%d, queue_remaining=%d)",
                             len(batch), BATCH_MAX, len(state.queue))

                    try:
                        await process_batch(state, r2, batch, subtensor, wallet)
                    except Exception:
                        log.exception("batch eval failed")
                        state.stats["failed"] += len(batch)
                        state.current_eval = None
                        state.flush_dashboard()

                    fresh = scan_reveals(subtensor, NETUID, state.seen)
                    if fresh:
                        state.flush()
                        for rev in fresh:
                            cid = state.enqueue(rev)
                            if cid:
                                log.info("queued %s from %s (new, mid-cycle)", cid, rev["hotkey"][:16])
                        new_items = [e for e in state.queue if not e.get("reeval")]
                        reeval_items = [e for e in state.queue if e.get("reeval")]
                        state.queue = new_items + reeval_items

            state.current_eval = None

            if args.seen and not state.queue:
                state.replenish_reeval(subtensor, NETUID)

            if not args.seen and not state.queue:
                log.info("idle: all hotkeys seen, waiting for new submissions")

            state.flush_dashboard()

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
    p.add_argument("--seen", action=argparse.BooleanOptionalAction, default=False,
                   help="When idle, replenish queue with re-eval candidates (default: off). "
                        "Without --seen, only genuinely new hotkeys are evaluated and the "
                        "validator idles when the queue is empty.")
    return p.parse_args()


def main_sync():
    asyncio.run(main())


if __name__ == "__main__":
    main_sync()
