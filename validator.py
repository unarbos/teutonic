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
import math
import os
import signal
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone

# Disable hf-xet BEFORE any huggingface_hub import. The xet runtime (Rust /
# Tokio) has been observed to abort the entire Python process during snapshot
# downloads of the dethrone-target king repo (see incidents 2026-04-26 09:52
# and 11:04 — process printed "Cancellation requested; stopping current tasks"
# and was then SIGKILLed by PM2 because it became unresponsive). The legacy
# HTTP downloader is robust enough for our 15 GB safetensors files.
# `HF_HUB_DISABLE_XET` is read at huggingface_hub import time, so it MUST be
# set before the `from huggingface_hub import HfApi` line below.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import bittensor as bt
import boto3
import httpx
import numpy as np
from botocore.config import Config as BotoConfig
from huggingface_hub import HfApi

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EVAL_N = 20_000
EVAL_ALPHA = 0.001
EVAL_DELTA = float(os.environ.get("TEUTONIC_EVAL_DELTA", "0.01"))
SEQ_LEN = 2048
POLL_INTERVAL = 30
WEIGHT_INTERVAL = 300
NETUID = int(os.environ.get("TEUTONIC_NETUID", "3"))

# Watchdogs / anti-stuckness safeguards.
TICK_WARN_AFTER = int(os.environ.get("TEUTONIC_TICK_WARN_AFTER", "120"))
TICK_RESTART_AFTER = int(os.environ.get("TEUTONIC_TICK_RESTART_AFTER", "1800"))
STREAM_IDLE_WARN_AFTER = int(os.environ.get("TEUTONIC_STREAM_IDLE_WARN_AFTER", "180"))
STREAM_IDLE_TIMEOUT = int(os.environ.get("TEUTONIC_STREAM_IDLE_TIMEOUT", "420"))
HEALTHCHECK_INTERVAL = int(os.environ.get("TEUTONIC_HEALTHCHECK_INTERVAL", "60"))
STATE_FLUSH_INTERVAL = int(os.environ.get("TEUTONIC_STATE_FLUSH_INTERVAL", "60"))
MAX_CONSECUTIVE_TICK_ERRORS = int(os.environ.get("TEUTONIC_MAX_CONSECUTIVE_TICK_ERRORS", "10"))
NETWORK = os.environ.get("TEUTONIC_NETWORK", "finney")
SEED_REPO = os.environ.get("TEUTONIC_SEED_REPO", "unconst/Teutonic-VIII")
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

REPO_PATTERN = r"^[^/]+/Teutonic-VIII-.+$"

# Magnitude/ratio sanity checks have been removed in favor of a single
# first-principles trainability probe that runs on the eval server (see
# eval_torch.trainability_probe). The probe takes one SGD step on the
# candidate and rejects models whose loss explodes — the actual property
# we care about. See DESIGN.md / commit history for the analysis.

# Periodic incumbent-king reprobe. The same trainability probe used to gate
# challengers is re-applied to the sitting king on a slow cadence; after
# KING_AUDIT_FAILS_BEFORE_DETHRONE consecutive failures, the validator
# reverts to `previous_king` and bans the dead repo.
KING_AUDIT_INTERVAL_S = int(os.environ.get("TEUTONIC_KING_AUDIT_INTERVAL_S", "3600"))
KING_AUDIT_FAILS_BEFORE_DETHRONE = int(os.environ.get("TEUTONIC_KING_AUDIT_FAILS", "2"))
KING_AUDIT_TIMEOUT_S = int(os.environ.get("TEUTONIC_KING_AUDIT_TIMEOUT_S", "600"))

TMC_BASE = "https://api.taomarketcap.com/public/v1"

log = logging.getLogger("teutonic")


class _EvalInnerError(Exception):
    """Wraps any exception raised inside process_challenge so we can tell it
    apart from asyncio.wait_for's own asyncio.TimeoutError sentinel.

    Without this, a TimeoutError raised by the stream-idle watchdog inside
    process_challenge would be caught by `except asyncio.TimeoutError` (since
    Python 3.11 unified them) and mis-classified as a 1800s wall-clock kill
    instead of the transient infra-side hiccup it actually is.
    """

    def __init__(self, original: BaseException):
        super().__init__(repr(original))
        self.original = original


# Rolling per-reign emission window. Each of the most recent KING_CHAIN_DEPTH
# kingships earns one slot worth `1/KING_CHAIN_DEPTH` of total emission. The
# newest valid (accepted-dethrone) king is pushed onto the head; the oldest is
# evicted. Same-hotkey wins stack — a hotkey that wins N of the last
# KING_CHAIN_DEPTH reigns receives N * (1/KING_CHAIN_DEPTH) of emissions. The
# per-hotkey aggregation is applied at the chain RPC boundary
# (`aggregate_chain_weights` -> `set_weights`).
TOPK_WEIGHTS = [0.2, 0.2, 0.2, 0.2, 0.2]
KING_CHAIN_DEPTH = len(TOPK_WEIGHTS)

# Bittensor emits one block every 12 seconds. Used to convert per-block
# emission rates from `meta.emission` into per-hour rates for the dashboard.
BLOCKS_PER_HOUR = 300

# Dashboard/Hippius writes are non-critical presentation updates. Keep them
# off the hot path and fail open when the public endpoint is degraded.
DASHBOARD_FLUSH_MIN_INTERVAL = float(os.environ.get("TEUTONIC_DASHBOARD_FLUSH_MIN_INTERVAL", "5"))
HIPPIUS_COOLDOWN_SECONDS = int(os.environ.get("TEUTONIC_HIPPIUS_COOLDOWN_SECONDS", "300"))

# Transient infra-side failures should not lose queue priority. If an eval
# fails because the eval server/stream/watchdog got wedged, requeue the same
# challenge at the front a bounded number of times before falling back to a
# normal recorded failure.
MAX_TRANSIENT_EVAL_RETRIES = int(os.environ.get("TEUTONIC_MAX_TRANSIENT_EVAL_RETRIES", "3"))


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
        snap = s["latest_snapshot"]
        asp = float(snap["alpha_sqrt_price"])
        tao_price = m["current_price"]
        alpha_tao = asp ** 2
        # `subnet_alpha_out_emission` is the per-block alpha emission to
        # miners (server side) on this subnet, in nanoalpha (1e-9). This is
        # the "going rate" that gets carved up by validator weights — much
        # more useful than the chain's lagging post-Yuma per-uid `emission`
        # tensor for an at-a-glance dashboard. Falls back to 0 if absent.
        try:
            sn3_alpha_per_block = float(snap.get("subnet_alpha_out_emission", 0)) / 1e9
        except Exception:
            sn3_alpha_per_block = 0.0
        return {
            "tao_price_usd": tao_price,
            "tao_change_24h": m["usd_quote"]["percent_change_24h"],
            "sn3_alpha_price_tao": alpha_tao,
            "sn3_alpha_price_usd": alpha_tao * tao_price,
            "sn3_reg_burn_tao": b[0]["burn"] / 1e9,
            "sn3_alpha_per_block": sn3_alpha_per_block,
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
        # Hippius is currently flaky on long-lived reads — bound every S3
        # call so a single hung connection cannot wedge the validator's
        # main async loop for minutes (botocore default is 60s + retries).
        _s3_cfg = dict(
            connect_timeout=15,
            read_timeout=45,
            retries={"max_attempts": 3, "mode": "adaptive"},
        )
        self.client = boto3.client(
            "s3", endpoint_url=R2_ENDPOINT,
            aws_access_key_id=R2_ACCESS_KEY, aws_secret_access_key=R2_SECRET_KEY,
            region_name="auto",
            config=BotoConfig(**_s3_cfg),
        )
        if HIPPIUS_ACCESS_KEY and HIPPIUS_SECRET_KEY:
            self._hippius = boto3.client(
                "s3", endpoint_url=HIPPIUS_ENDPOINT,
                aws_access_key_id=HIPPIUS_ACCESS_KEY,
                aws_secret_access_key=HIPPIUS_SECRET_KEY,
                region_name="decentralized",
                config=BotoConfig(
                    signature_version="s3v4",
                    s3={"addressing_style": "path"},
                    **_s3_cfg,
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
                    s3={"addressing_style": "path"},
                    **_s3_cfg,
                ),
            )
            self._ds_bucket = DS_BUCKET
            log.info("dataset store: %s bucket=%s", DS_ENDPOINT, DS_BUCKET)
        else:
            self._ds_client = None
            self._ds_bucket = None

    def _hippius_available(self):
        if not self._hippius:
            return False
        retry_after = getattr(self, "_hippius_retry_after", 0.0)
        return time.monotonic() >= retry_after

    def _mark_hippius_failure(self, key, exc):
        self._hippius_retry_after = time.monotonic() + HIPPIUS_COOLDOWN_SECONDS
        log.warning(
            "Hippius dashboard write failed for %s; cooling down for %ss and falling back to R2: %s",
            key,
            HIPPIUS_COOLDOWN_SECONDS,
            exc,
        )

    def _put_dashboard_bytes(self, key, body, content_type):
        if self._hippius_available():
            try:
                self._hippius.put_object(
                    Bucket=HIPPIUS_BUCKET,
                    Key=key,
                    Body=body,
                    ContentType=content_type,
                )
                return
            except Exception as exc:
                self._mark_hippius_failure(key, exc)

        try:
            self.client.put_object(
                Bucket=R2_BUCKET,
                Key=key,
                Body=body,
                ContentType=content_type,
            )
        except Exception:
            log.warning("dashboard fallback put failed for %s (non-fatal)", key, exc_info=True)

    def put_dashboard(self, key, data):
        body = json.dumps(data, default=str).encode()
        self._put_dashboard_bytes(key, body, "application/json")

    def put_dashboard_raw(self, key, body, content_type):
        self._put_dashboard_bytes(key, body, content_type)

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


def validate_challenger_config(hf_repo: str, challenger_revision: str,
                                king_repo: str = "",
                                king_revision: str = "") -> str | None:
    """Check challenger config.json matches king architecture before deploying.

    All HF API calls use the pinned challenger_revision to prevent TOCTOU
    attacks where a miner swaps safetensors for a malicious pickle between
    validation and evaluation.

    Returns None if OK, or a human-readable rejection reason.
    """
    king_cfg = get_king_config(king_repo or SEED_REPO, king_revision)
    if not king_cfg:
        return None

    try:
        api = HfApi(token=HF_TOKEN or None)
        cfg_path = api.hf_hub_download(hf_repo, "config.json",
                                        token=HF_TOKEN or None,
                                        revision=challenger_revision)
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

    st_files = [s for s in api.list_repo_files(hf_repo, token=HF_TOKEN or None,
                                                revision=challenger_revision)
                if s.endswith(".safetensors")]
    if not st_files:
        return "no .safetensors files in repo"

    # Trainability gate runs on the eval server (eval_torch.trainability_probe),
    # not here. The validator's job is just structural/architecture matching.
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


def _rank_sort_key(entry: dict):
    return (
        -(entry.get("mu_hat") or 0),
        -(entry.get("lcb") or 0),
        entry.get("avg_challenger_loss") or float("inf"),
        entry.get("timestamp") or "",
    )


def _normalize_weights(base_weights, n):
    weights = list(base_weights[:n])
    if not weights:
        return []
    total = sum(weights)
    if total <= 0:
        return []
    return [w / total for w in weights]


def set_weights(subtensor, wallet, netuid, ranked_hotkeys, ranked_weights) -> bool:
    """Set weights across one or more ranked hotkeys. Returns True on success.

    Defensively collapses repeated hotkeys before hitting the subtensor RPC
    (subtensor.set_weights behavior on duplicate uids is undefined). Per-reign
    stacking is intentional upstream — see `aggregate_chain_weights` — but the
    aggregation is repeated here as belt-and-suspenders.
    """
    try:
        meta = subtensor.metagraph(netuid)
        hotkeys = list(getattr(meta, "hotkeys", []))
        agg_uid: dict[int, tuple[float, str]] = {}
        missing: list[str] = []
        for hotkey, weight in zip(ranked_hotkeys, ranked_weights):
            if hotkey not in hotkeys:
                missing.append(hotkey)
                continue
            uid = hotkeys.index(hotkey)
            prev_w, _ = agg_uid.get(uid, (0.0, hotkey))
            agg_uid[uid] = (prev_w + float(weight), hotkey)
        if missing:
            log.warning("weight-set dropping missing hotkeys: %s",
                        ", ".join(h[:16] for h in missing))
        chosen = [(uid, w, hk) for uid, (w, hk) in agg_uid.items() if w > 0]
        if not chosen:
            log.warning("no eligible ranked hotkeys present in metagraph for weight set")
            return False
        total = sum(weight for _, weight, _ in chosen)
        if total <= 0:
            log.warning("ranked weight total is zero")
            return False
        uids = [uid for uid, _, _ in chosen]
        weights = [weight / total for _, weight, _ in chosen]
        subtensor.set_weights(wallet=wallet, netuid=netuid, uids=uids, weights=weights)
        log.info("weights set to %s",
                 ", ".join(f"uid={uid}:{weight:.6f}:{hotkey[:16]}" for (uid, _raw, hotkey), weight in zip(chosen, weights)))
        return True
    except Exception:
        log.exception("failed to set weights")
        return False


def aggregate_chain_weights(chain, uid_map):
    """Per-reign aggregator for the rolling 5-king emission window.

    Each entry in `chain` (newest first) earns one slot worth `1/KING_CHAIN_DEPTH`
    of total emission. Slots whose hotkey is not currently registered in
    `uid_map` are dropped (the missing weight is implicitly forfeit; the
    set_weights chain RPC renormalizes what remains). Repeats stack — a hotkey
    that wins kingship N times in the last KING_CHAIN_DEPTH reigns receives
    N * (1/KING_CHAIN_DEPTH) of total emission.

    Returns `[(hotkey, weight)]` in first-seen-newest-first order.
    """
    slot_weight = 1.0 / KING_CHAIN_DEPTH
    agg: dict[str, float] = {}
    order: list[str] = []
    for entry in chain:
        hk = entry.get("hotkey") if isinstance(entry, dict) else None
        if not hk or hk not in uid_map:
            continue
        if hk not in agg:
            agg[hk] = 0.0
            order.append(hk)
        agg[hk] += slot_weight
    return [(hk, agg[hk]) for hk in order]


def maybe_set_weights(subtensor, wallet, state, *, force: bool = False,
                      reason: str = "") -> bool:
    """Push subnet weights for the rolling last-`KING_CHAIN_DEPTH` reigns.

    Each of the most recent KING_CHAIN_DEPTH dethrones earns one equal slot
    (1/KING_CHAIN_DEPTH of total emission). Same-hotkey repeats stack at the
    per-hotkey aggregation step (see `aggregate_chain_weights`). The current
    king always sits at slot 0.
    """
    fallback_hotkey = state.king.get("hotkey") if state.king else None
    if not fallback_hotkey:
        if force:
            log.info("skipping weight set (%s): no king yet", reason or "forced")
        return False
    try:
        current_block = subtensor.block
    except Exception:
        log.exception("failed to read current block for weight-set")
        return False
    if not force and current_block - state.last_weight_block < WEIGHT_INTERVAL:
        return False

    chain = state.recent_king_chain(KING_CHAIN_DEPTH)
    pairs = aggregate_chain_weights(chain, state.uid_map)
    if not pairs:
        ranked_hotkeys = [fallback_hotkey]
        ranked_weights = [1.0]
    else:
        ranked_hotkeys = [hk for hk, _ in pairs]
        ranked_weights = [w for _, w in pairs]

    log.info("setting weights at block %d (last=%d, %s) to %s",
             current_block, state.last_weight_block,
             reason or ("forced" if force else "interval"),
             ", ".join(f"{hk[:16]}:{w:.6f}" for hk, w in zip(ranked_hotkeys, ranked_weights)))
    if set_weights(subtensor, wallet, NETUID, ranked_hotkeys, ranked_weights):
        state.last_weight_block = current_block
        state.last_winner_hotkey = ranked_hotkeys[0]
        # Kings are crowned only by accepted dethrones in process_challenge.
        # The weight-set tick no longer touches state.king or reign_number.
        state.note_weight_set(current_block, ranked_hotkeys, ranked_weights, reason or ("forced" if force else "interval"))
        try:
            state.reset_score_window(current_block)
            state.flush()
            state.flush_dashboard()
        except Exception:
            log.exception("failed to flush state after weight set")
        return True
    return False


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def _now():
    return datetime.now(timezone.utc).isoformat()


def _monotonic_now() -> float:
    return time.monotonic()


def _safe_block(subtensor) -> int:
    """Best-effort current block; returns 0 if the chain call raises so the
    dethrone path can still record a king transition without losing state."""
    try:
        return int(subtensor.block)
    except Exception:
        return 0


def _age_seconds(ts: str | None) -> float | None:
    if not ts:
        return None
    try:
        return max(0.0, datetime.now(timezone.utc).timestamp() - datetime.fromisoformat(ts).timestamp())
    except Exception:
        return None


_SEED_KING_HASH_SUBPROCESS = r"""
# Runs in an isolated child process so any native crash (hf-xet abort, OOM,
# segfault) only kills the child, not the parent validator. The parent reads
# the digest from stdout. On any failure mode, parent falls back to "seed".
import hashlib, os, sys, tempfile
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import snapshot_download

repo = sys.argv[1]
revision = sys.argv[2] or None
token = os.environ.get("HF_TOKEN") or None

with tempfile.TemporaryDirectory(prefix="seed_king_") as tmp:
    snapshot_download(repo, local_dir=tmp, token=token, revision=revision,
                      allow_patterns=["*.safetensors"])
    h = hashlib.sha256()
    for p in sorted(Path(tmp).glob("*.safetensors")):
        with open(p, "rb") as f:
            while chunk := f.read(1 << 20):
                h.update(chunk)
    sys.stdout.write(h.hexdigest())
"""


def _seed_king_hash(repo: str, revision: str) -> str:
    """Compute sha256 over the king repo's safetensors so it matches the
    `king_hash` miners encode in their on-chain commits (see miner.sha256_dir).

    Runs the download + hash in an isolated subprocess so a native crash in
    the huggingface downloader cannot take the whole validator with it (see
    2026-04-26 incidents where xet aborts SIGKILLed the parent mid-dethrone).

    Falls back to the literal string "seed" if the subprocess fails — the live
    state can be patched later with scripts/patch_seed_king_hash.py, or by the
    placeholder-recompute path in State.load().
    """
    import subprocess
    timeout_s = int(os.environ.get("TEUTONIC_KING_HASH_TIMEOUT_S", "1200"))
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _SEED_KING_HASH_SUBPROCESS, repo, revision or ""],
            capture_output=True, text=True, timeout=timeout_s,
            env={**os.environ, "HF_HUB_DISABLE_XET": "1"},
        )
        if proc.returncode == 0 and proc.stdout.strip():
            digest = proc.stdout.strip()
            log.info("seed king_hash for %s@%s = %s",
                     repo, (revision or "latest")[:12], digest[:16])
            return digest
        log.warning("seed king_hash subprocess failed for %s rc=%s "
                    "stderr=%s — using 'seed'",
                    repo, proc.returncode, (proc.stderr or "")[-400:])
        return "seed"
    except subprocess.TimeoutExpired:
        log.warning("seed king_hash subprocess timed out (%ds) for %s — using 'seed'",
                    timeout_s, repo)
        return "seed"
    except Exception as exc:
        log.warning("seed king_hash subprocess raised for %s: %s — using 'seed'",
                    repo, exc)
        return "seed"


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
        # Per-hotkey alpha emission per block, captured from
        # `subtensor.metagraph(netuid).emission` in `refresh_uid_map`. Used by
        # `flush_dashboard` to express each king's incentive in alpha/hour and
        # USD/hour. Empty until the first metagraph refresh succeeds.
        self.uid_emission_per_block: dict[str, float] = {}
        self.score_window = {
            "window_id": "window-0000",
            "started_at": _now(),
            "started_block": 0,
            "accepted_by_hotkey": {},
            "topk": [],
            "last_weight_set": None,
        }
        self.known_revisions: dict[str, dict[str, str]] = {}
        self.watchdog = {
            "started_at": _now(),
            "last_tick_started_at": None,
            "last_tick_completed_at": None,
            "last_progress_at": None,
            "last_state_flush_at": None,
            "last_dashboard_flush_at": None,
            "phase": "startup",
            "phase_since": _now(),
            "current_challenge_id": None,
            "current_eval_id": None,
            "consecutive_tick_errors": 0,
            "restart_requested": False,
            "restart_reason": "",
            "notes": "",
        }
        self.king_audit: dict = {}

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
            loaded_window = st.get("score_window")
            if loaded_window:
                loaded_window.setdefault("window_id", "window-0000")
                loaded_window.setdefault("started_at", _now())
                loaded_window.setdefault("started_block", self.last_weight_block)
                loaded_window.setdefault("accepted_by_hotkey", {})
                loaded_window.setdefault("topk", [])
                loaded_window.setdefault("last_weight_set", None)
                self.score_window = loaded_window
            self.known_revisions = st.get("known_revisions", {})
            self.king_audit = st.get("king_audit", {}) or {}
        h = self.r2.get("state/dashboard_history.json")
        if h:
            self.history = h.get("history", [])
        wd = self.r2.get("state/watchdog.json")
        if wd:
            self.watchdog.update(wd)
        log.info("loaded state: king=%s queue=%d seen=%d",
                 self.king.get("king_hash", "none")[:16], len(self.queue), len(self.seen))

        if (self.king
                and self.king.get("king_hash") == "seed"
                and self.king.get("hf_repo")):
            log.warning("loaded king has placeholder king_hash='seed'; "
                        "recomputing from %s@%s ...",
                        self.king["hf_repo"],
                        (self.king.get("king_revision") or "latest")[:12])
            real_hash = _seed_king_hash(self.king["hf_repo"],
                                        self.king.get("king_revision", ""))
            if real_hash and real_hash != "seed":
                self.king["king_hash"] = real_hash
                self.flush()
                self.flush_dashboard(force=True)
                log.info("upgraded king_hash to %s and persisted to R2", real_hash[:16])

        # Reconcile the king chain against the duel history. If a dethrone
        # crashed between record_verdict (line ~1862) and set_king (line
        # ~1905) — the exact failure mode that ate uid=0 and uid=56's wins
        # on 2026-04-26 — the verdict is in history but the chain never
        # advanced. This rebuild puts the chain back in sync.
        self._reconcile_chain_from_history()

    def _reconcile_chain_from_history(self):
        """Rebuild the top KING_CHAIN_DEPTH slots of the king chain from
        the most recent accepted-challenger verdicts in `self.history`.

        Idempotent: when the chain already matches history, this is a
        no-op. When history disagrees (a crashed dethrone), the chain is
        rewritten so the rolling-5 weight-set credits the right hotkeys.

        Preserves king_hash/king_revision/crowned_block for every entry
        that already lives in the existing chain (keyed by challenge_id);
        only synthesizes placeholder entries for the lost dethrones.
        """
        if not self.king or not self.history:
            return

        wins_newest_first = []
        seen_cids = set()
        for h in self.history:
            if h.get("verdict") != "challenger" or not h.get("accepted", False):
                continue
            cid = h.get("challenge_id")
            if cid in seen_cids:
                continue
            seen_cids.add(cid)
            wins_newest_first.append(h)
            if len(wins_newest_first) >= KING_CHAIN_DEPTH:
                break

        if not wins_newest_first:
            return

        # Sanity: top of history must match current king. If not, history
        # is corrupt or king/state diverged some other way — bail loudly
        # rather than rewriting state from a bad source.
        top_hk = wins_newest_first[0].get("hotkey", "")
        if top_hk != self.king.get("hotkey", ""):
            log.warning("chain reconcile: history top hk=%s != king hk=%s; "
                        "skipping reconcile",
                        top_hk[:16], (self.king.get("hotkey") or "")[:16])
            return

        # Index existing chain by challenge_id so we preserve real
        # king_hash + king_revision + crowned_block for entries that
        # actually got crowned.
        existing_by_cid = {}
        node = self.king
        while node:
            cid = node.get("challenge_id")
            if cid:
                existing_by_cid[cid] = node
            node = node.get("previous_king")

        current_reign = int(self.king.get("reign_number") or 0)
        rebuilt = []
        any_new = False
        for i, w in enumerate(wins_newest_first):
            cid = w.get("challenge_id", "")
            ts = w.get("timestamp", _now())
            reign_for_this = current_reign - i
            existing = existing_by_cid.get(cid)
            if existing:
                rebuilt.append({
                    "hotkey": existing.get("hotkey") or w.get("hotkey", ""),
                    "hf_repo": existing.get("hf_repo") or w.get("challenger_repo", ""),
                    "king_hash": existing.get("king_hash", "dethrone"),
                    "king_revision": existing.get("king_revision", ""),
                    "reign_number": reign_for_this,
                    "crowned_at": existing.get("crowned_at", ts),
                    "crowned_block": existing.get("crowned_block", 0),
                    "challenge_id": cid,
                })
            else:
                any_new = True
                rebuilt.append({
                    "hotkey": w.get("hotkey", ""),
                    "hf_repo": w.get("challenger_repo", ""),
                    "king_hash": "dethrone",
                    "king_revision": "",
                    "reign_number": reign_for_this,
                    "crowned_at": ts,
                    "crowned_block": 0,
                    "challenge_id": cid,
                })

        # Compare cid order to detect whether the existing chain already
        # matches; if so we don't bother rewriting.
        existing_cids = []
        node = self.king
        while node and len(existing_cids) < KING_CHAIN_DEPTH:
            existing_cids.append(node.get("challenge_id"))
            node = node.get("previous_king")
        rebuilt_cids = [r.get("challenge_id") for r in rebuilt]
        if not any_new and existing_cids[: len(rebuilt_cids)] == rebuilt_cids:
            return  # already consistent

        # Link list newest -> oldest.
        new_king = None
        for entry in reversed(rebuilt):
            entry = dict(entry)
            entry["previous_king"] = new_king
            new_king = entry

        log.warning("chain reconcile: rewriting king chain from history "
                    "(was: %s; now: %s)",
                    [c[:8] if c else "?" for c in existing_cids],
                    [c[:8] if c else "?" for c in rebuilt_cids])
        self.king = new_king
        self.flush()
        self.flush_dashboard(force=True)
        self.event({"event": "chain_reconciled",
                     "previous_cids": existing_cids,
                     "rebuilt_cids": rebuilt_cids})

    def flush(self):
        now = _now()
        self.watchdog["last_state_flush_at"] = now
        self.r2.put("state/validator_state.json", {
            "king": self.king, "queue": self.queue,
            "stats": self.stats, "counter": self.counter,
            "last_weight_block": self.last_weight_block,
            "last_winner_hotkey": self.last_winner_hotkey,
            "score_window": self.score_window,
            "known_revisions": self.known_revisions,
            "king_audit": self.king_audit,
            "updated_at": now,
        })
        self.r2.put("state/queue.json", {"pending": self.queue, "updated_at": now})
        self.r2.put("king/current.json", self.king)
        self.r2.put("state/seen_hotkeys.json", {
            "hotkeys": sorted(self.seen), "updated_at": now,
        })
        self.r2.put("state/watchdog.json", self.watchdog)

    def event(self, data):
        data.setdefault("timestamp", _now())
        self.r2.append_jsonl("state/history.jsonl", data)

    def next_id(self):
        self.counter += 1
        return f"eval-{self.counter:04d}"

    def enqueue(self, reveal, defer_flush: bool = False):
        """Add a reveal to the queue. Each enqueue normally triggers ~4 R2
        sync writes (state, queue, dashboard, history-jsonl), so when the
        caller is enqueueing a batch (replenish_reeval, mid-cycle scans),
        pass defer_flush=True and call self.flush()/self.flush_dashboard()
        once at the end. Otherwise replenishing 100+ items can stall the
        eval pipeline for 10+ minutes while flushes run sequentially."""
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
        entry = {"challenge_id": cid, **reveal, "queued_at": _now(), "retry_count": int(reveal.get("retry_count", 0))}
        self.queue.append(entry)
        self.stats["queued"] += 1
        if not defer_flush:
            self.flush()
            self.flush_dashboard(force=True)
            self.event({"event": "queued", **entry})
        return cid

    def requeue_front(self, entry, *, reason: str, error_code: str = "", error_detail: str = ""):
        """Requeue an existing challenge at the front for transient infra failures.

        Keeps challenge_id and original repo/hotkey, increments retry_count,
        refreshes queued_at, and avoids duplicating the same repo if it's already
        pending elsewhere in the queue.
        """
        repo = entry.get("hf_repo", "")
        retry_count = int(entry.get("retry_count", 0)) + 1
        new_entry = {**entry, "retry_count": retry_count, "queued_at": _now(), "reeval": False}

        deduped = []
        for existing in self.queue:
            if existing.get("hf_repo") == repo:
                continue
            deduped.append(existing)
        self.queue = [new_entry] + deduped
        self.current_eval = None
        self.flush()
        self.flush_dashboard(force=True)
        self.event({
            "event": "requeued_front",
            "challenge_id": entry.get("challenge_id", "?"),
            "hotkey": entry.get("hotkey", ""),
            "hf_repo": repo,
            "retry_count": retry_count,
            "reason": reason,
            "error_code": error_code,
            "error_detail": str(error_detail),
        })
        log.warning("re-queued %s at front (retry %d/%d) due to %s: %s",
                    entry.get("challenge_id", "?"), retry_count,
                    MAX_TRANSIENT_EVAL_RETRIES, reason, error_detail)
        return retry_count

    def remember_revision(self, hotkey, repo, revision):
        if not hotkey:
            return
        self.known_revisions[hotkey] = {
            "repo": repo,
            "revision": revision,
            "updated_at": _now(),
        }

    def best_known_revision(self, hotkey, repo=""):
        info = self.known_revisions.get(hotkey, {})
        if repo and info.get("repo") and info.get("repo") != repo:
            return ""
        return info.get("revision", "")

    def _best_of(self, entries):
        return sorted(entries, key=_rank_sort_key)[0] if entries else None

    def recompute_topk(self):
        accepted = list(self.score_window.get("accepted_by_hotkey", {}).values())
        ranked = sorted(accepted, key=_rank_sort_key)
        self.score_window["topk"] = ranked[: len(TOPK_WEIGHTS)]
        return self.score_window["topk"]

    def record_accepted_result(self, verdict, challenger_repo, hotkey, block=0):
        entry = {
            "challenge_id": verdict["challenge_id"],
            "hotkey": hotkey,
            "uid": self.uid_map.get(hotkey, "?"),
            "challenger_repo": challenger_repo,
            "challenger_revision": verdict.get("challenger_revision", self.best_known_revision(hotkey, challenger_repo)),
            "mu_hat": verdict.get("mu_hat", 0),
            "lcb": verdict.get("lcb", 0),
            "avg_challenger_loss": verdict.get("avg_challenger_loss", 0),
            "avg_king_loss": verdict.get("avg_king_loss", 0),
            "verdict": verdict.get("verdict", "accepted"),
            "accepted": True,
            "timestamp": verdict.get("timestamp", _now()),
            "block": block,
        }
        existing = self.score_window.setdefault("accepted_by_hotkey", {}).get(hotkey)
        if existing is None or _rank_sort_key(entry) < _rank_sort_key(existing):
            self.score_window["accepted_by_hotkey"][hotkey] = entry
        return self.recompute_topk()

    def topk_for_weight_set(self):
        """DEPRECATED — kept only for backward-compatible R2/dashboard reads.

        Real emission weights now come from `aggregate_chain_weights` over the
        rolling per-reign king chain (see `maybe_set_weights`). This method
        ranks the in-progress score-window's accepted verdicts and is no
        longer used to drive `subtensor.set_weights`.
        """
        ranked = []
        seen_hotkeys = set()
        for entry in sorted(self.score_window.get("accepted_by_hotkey", {}).values(), key=_rank_sort_key):
            hotkey = entry.get("hotkey")
            if not hotkey or hotkey in seen_hotkeys:
                continue
            if hotkey not in self.uid_map:
                continue
            ranked.append(entry)
            seen_hotkeys.add(hotkey)
            if len(ranked) >= len(TOPK_WEIGHTS):
                break
        return ranked

    def note_weight_set(self, block, ranked_hotkeys, ranked_weights, reason):
        self.score_window["last_weight_set"] = {
            "timestamp": _now(),
            "block": block,
            "reason": reason,
            "ranked_hotkeys": ranked_hotkeys,
            "ranked_weights": ranked_weights,
        }

    def reset_score_window(self, current_block):
        prev_num = 0
        wid = self.score_window.get("window_id", "window-0000")
        try:
            prev_num = int(str(wid).split("-")[-1])
        except Exception:
            prev_num = 0
        self.score_window = {
            "window_id": f"window-{prev_num + 1:04d}",
            "started_at": _now(),
            "started_block": current_block,
            "accepted_by_hotkey": {},
            "topk": [],
            "last_weight_set": self.score_window.get("last_weight_set"),
        }
        self.evaluated_repos.clear()

    @staticmethod
    def _truncate_king_chain(node, max_depth):
        """Return a deep-ish copy of `node` whose `previous_king` chain is
        capped at `max_depth` total layers (including the node itself).
        Avoids unbounded growth as dethrones accumulate."""
        if not node or max_depth <= 0:
            return None
        head = dict(node)
        cur = head
        remaining = max_depth - 1
        while remaining > 0:
            child = cur.get("previous_king")
            if not child:
                cur["previous_king"] = None
                return head
            child_copy = dict(child)
            cur["previous_king"] = child_copy
            cur = child_copy
            remaining -= 1
        cur["previous_king"] = None
        return head

    def recent_king_chain(self, depth):
        """Walk king + previous_king chain up to `depth` layers; return list of
        per-reign dicts in order [current, prev, prev2, ...].

        Repeats are intentional: if the same hotkey wins consecutively, each
        win occupies its own slot. The per-hotkey aggregation lives one layer
        up in `maybe_set_weights`, where slot weights are summed before the
        subtensor RPC.
        """
        out = []
        node = self.king
        while node and len(out) < depth:
            if node.get("hotkey"):
                out.append(node)
            node = node.get("previous_king")
        return out

    def set_king(self, hotkey, hf_repo, king_hash, block, challenge_id="seed",
                  king_revision=""):
        global _king_config, _king_config_key
        _king_config = None
        _king_config_key = None
        self.failed_repos.clear()
        self.evaluated_repos.clear()
        reign = self.king.get("reign_number", 0) + (0 if challenge_id == "seed" else 1)
        prev_full = self.king.copy() if self.king else None
        # Always preserve the prior reign in the chain — even if the same hotkey
        # wins consecutively. Per-reign stacking is intentional: each successful
        # dethrone earns a fresh slot in the rolling 5-reign emission window
        # (see TOPK_WEIGHTS). `_truncate_king_chain` caps the recursive
        # previous_king pointer at KING_CHAIN_DEPTH-1 so the new head + chain
        # contains at most KING_CHAIN_DEPTH reigns.
        chain_root = self._truncate_king_chain(prev_full, KING_CHAIN_DEPTH - 1)
        self.king = {
            "hotkey": hotkey, "hf_repo": hf_repo, "king_hash": king_hash,
            "king_revision": king_revision,
            "reign_number": reign, "crowned_at": _now(),
            "crowned_block": block, "challenge_id": challenge_id,
            "previous_king": chain_root,
        }
        self.flush()
        self.flush_dashboard(force=True)
        self.event({"event": "king_changed", "hotkey": hotkey, "reign": reign,
                     "challenge_id": challenge_id})

    def record_verdict(self, verdict, challenger_repo, hotkey):
        # Probe-rejected verdicts (eval_server.py probe-fail path) lack the
        # full bootstrap-test fields (avg_*_loss / wall_time_s / timestamp).
        # Default everything so a partial verdict still records cleanly.
        king_loss = verdict.get("avg_king_loss", 0)
        chall_loss = verdict.get("avg_challenger_loss", 0)
        entry = {
            "challenge_id": verdict.get("challenge_id"),
            "hotkey": hotkey,
            "uid": self.uid_map.get(hotkey, "?"),
            "challenger_repo": challenger_repo,
            "accepted": verdict.get("accepted", False),
            "verdict": verdict.get("verdict", "unknown"),
            "mu_hat": verdict.get("mu_hat", 0),
            "lcb": verdict.get("lcb", 0),
            "delta": verdict.get("delta", 0),
            "avg_king_loss": king_loss,
            "avg_challenger_loss": chall_loss,
            "best_loss": min(king_loss, chall_loss) if (king_loss or chall_loss) else 0,
            "wall_time_s": verdict.get("wall_time_s", 0),
            "timestamp": verdict.get("timestamp", _now()),
        }
        if verdict.get("rejection_reason"):
            entry["rejection_reason"] = verdict["rejection_reason"]
        self.history.insert(0, entry)
        self.r2.put("state/dashboard_history.json", {"history": self.history})

    def record_failure(self, entry, error_code, error_detail=""):
        self.history.insert(0, {
            "challenge_id": entry.get("challenge_id", "?"),
            "hotkey": entry.get("hotkey", ""),
            "uid": self.uid_map.get(entry.get("hotkey", ""), "?"),
            "challenger_repo": entry.get("hf_repo", ""),
            "accepted": False,
            "verdict": "error",
            "error_code": error_code,
            "error_detail": str(error_detail),
            "mu_hat": 0,
            "lcb": 0,
            "delta": 0,
            "avg_king_loss": 0,
            "avg_challenger_loss": 0,
            "best_loss": 0,
            "wall_time_s": 0,
            "timestamp": _now(),
        })
        self.r2.put("state/dashboard_history.json", {"history": self.history})

    def refresh_uid_map(self, subtensor, netuid):
        try:
            meta = subtensor.metagraph(netuid)
            self.uid_map = {hk: uid for uid, hk in enumerate(meta.hotkeys)}
            em = getattr(meta, "emission", None)
            if em is not None:
                emissions = em.tolist() if hasattr(em, "tolist") else list(em)
            else:
                emissions = []
            self.uid_emission_per_block = {
                hk: (float(emissions[uid]) if uid < len(emissions) else 0.0)
                for hk, uid in self.uid_map.items()
            }
        except Exception:
            log.warning("failed to refresh uid_map", exc_info=True)

    def replenish_reeval(self, subtensor, netuid):
        """Fill queue with re-eval candidates so the dashboard never shows empty.

        Defers per-item R2 flushes; one summary flush at the end. With ~150
        re-eval candidates and ~5s per flush previously, this loop used to
        block the eval pipeline for ~12-15 minutes."""
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
            cid = self.enqueue(rev, defer_flush=True)
            if cid:
                count += 1
                log.info("queued %s from %s (re-eval)", cid, rev["hotkey"][:16])
        if count:
            self.flush()
            self.flush_dashboard(force=True)
            self.event({"event": "replenish_reeval", "count": count})
            log.info("replenished queue with %d re-eval candidates", count)
        return count

    def flush_dashboard(self, *, force: bool = False):
        # Dashboard flush is presentational: it MUST NEVER raise into the main
        # eval loop. A Hippius/R2 outage here used to propagate up through
        # process_challenge and get logged as `eval failed:` even though the
        # eval verdict had already been recorded. Catch absolutely everything.
        try:
            now_monotonic = _monotonic_now()
            last_flush = getattr(self, "_last_dashboard_flush_monotonic", 0.0)
            if not force and (now_monotonic - last_flush) < DASHBOARD_FLUSH_MIN_INTERVAL:
                return False

            self._last_dashboard_flush_monotonic = now_monotonic
            self.watchdog["last_dashboard_flush_at"] = _now()
            chain = self.recent_king_chain(KING_CHAIN_DEPTH)
            chain_pairs = aggregate_chain_weights(chain, self.uid_map)
            mkt = self.market or {}
            alpha_tao = float(mkt.get("sn3_alpha_price_tao") or 0.0)
            alpha_usd = float(mkt.get("sn3_alpha_price_usd") or 0.0)
            # Forward-looking subnet miner emission rate: this is what the
            # network will pay to miners per block, divided across our
            # validator-set weights. We renormalize the per-hotkey weights
            # over the kings actually present in the metagraph (matches the
            # set_weights renormalization) so the dashboard sums to 1.0.
            sn3_alpha_per_block = float(mkt.get("sn3_alpha_per_block") or 0.0)
            weight_total = sum(w for _, w in chain_pairs) or 1.0

            # Per-hotkey aggregated incentive (used by `set_weights` and shown
            # in the dashboard's "ROLLING REIGN WINDOW" table). emission_per_block
            # is chain-truth from `meta.emission[uid]` and lags weight changes
            # by one tempo; alpha_per_hour is the forward-looking projection
            # using the subnet's current per-block miner emission rate.
            king_chain_weights = []
            per_hotkey_money: dict[str, dict] = {}
            for hk, w in chain_pairs:
                em_per_block = float(self.uid_emission_per_block.get(hk, 0.0))
                share = (w / weight_total) if weight_total > 0 else 0.0
                projected_alpha_per_block = sn3_alpha_per_block * share
                alpha_per_hour = projected_alpha_per_block * BLOCKS_PER_HOUR
                tao_per_hour = alpha_per_hour * alpha_tao
                usd_per_hour = alpha_per_hour * alpha_usd
                entry = {
                    "hotkey": hk,
                    "uid": self.uid_map.get(hk, "?"),
                    "weight": round(w, 6),
                    "weight_share": round(share, 6),
                    "emission_per_block": round(em_per_block, 9),
                    "projected_alpha_per_block": round(projected_alpha_per_block, 9),
                    "alpha_per_hour": round(alpha_per_hour, 6),
                    "tao_per_hour": round(tao_per_hour, 6),
                    "usd_per_hour": round(usd_per_hour, 4),
                }
                king_chain_weights.append(entry)
                per_hotkey_money[hk] = entry

            king_chain = []
            for e in chain:
                hk = e.get("hotkey", "")
                money = per_hotkey_money.get(hk, {})
                king_chain.append({
                    "hotkey": hk,
                    "uid": self.uid_map.get(hk, "?"),
                    "hf_repo": e.get("hf_repo"),
                    "king_revision": e.get("king_revision"),
                    "reign_number": e.get("reign_number"),
                    "crowned_at": e.get("crowned_at"),
                    "crowned_block": e.get("crowned_block"),
                    "challenge_id": e.get("challenge_id"),
                    # Per-reign rows carry the per-hotkey AGGREGATED incentive
                    # so a hotkey that holds two of the last 5 slots shows the
                    # combined dollar number on each of its rows.
                    "weight": money.get("weight"),
                    "alpha_per_hour": money.get("alpha_per_hour"),
                    "tao_per_hour": money.get("tao_per_hour"),
                    "usd_per_hour": money.get("usd_per_hour"),
                })
            payload = {
                "updated_at": _now(),
                "king": self.king,
                "king_chain": king_chain,
                "king_chain_weights": king_chain_weights,
                "stats": self.stats,
                "current_eval": self.current_eval,
                "watchdog": self.watchdog,
                "score_window": self.score_window,
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
            return True
        except Exception:
            log.warning("flush_dashboard failed (non-fatal, eval continues)", exc_info=True)
            return False

    def set_phase(self, phase: str, *, challenge_id: str | None = None,
                  eval_id: str | None = None, notes: str = ""):
        now = _now()
        self.watchdog.update({
            "phase": phase,
            "phase_since": now,
            "notes": notes,
        })
        if challenge_id is not None:
            self.watchdog["current_challenge_id"] = challenge_id
        if eval_id is not None:
            self.watchdog["current_eval_id"] = eval_id

    def note_progress(self, *, notes: str = ""):
        now = _now()
        self.watchdog["last_progress_at"] = now
        if notes:
            self.watchdog["notes"] = notes

    def begin_tick(self):
        now = _now()
        self.watchdog["last_tick_started_at"] = now
        self.set_phase("tick", notes="validator tick started")

    def complete_tick(self):
        now = _now()
        self.watchdog["last_tick_completed_at"] = now
        self.watchdog["consecutive_tick_errors"] = 0
        self.set_phase("sleep", notes="validator tick completed")

    def fail_tick(self, reason: str):
        self.watchdog["consecutive_tick_errors"] = self.watchdog.get("consecutive_tick_errors", 0) + 1
        self.set_phase("tick_error", notes=reason)

    def request_restart(self, reason: str):
        self.watchdog["restart_requested"] = True
        self.watchdog["restart_reason"] = reason
        self.set_phase("restart_requested", notes=reason)
        self.event({"event": "watchdog_restart_requested", "reason": reason})

    def clear_restart_request(self):
        self.watchdog["restart_requested"] = False
        self.watchdog["restart_reason"] = ""



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


async def audit_incumbent_king(state, subtensor):
    """Reprobe the sitting king out-of-band via eval_server's /probe endpoint.

    On each call we reach the eval server, ask it to load the current king
    repo+revision on a single GPU, run `trainability_probe`, and return the
    verdict. Successes reset `consecutive_fails`; failures increment it.
    Once it hits `KING_AUDIT_FAILS_BEFORE_DETHRONE` we revert to
    `previous_king` (mirroring `check_king_alive`) and ban the dead repo
    from being re-queued as a challenger this cycle.

    State is persisted on `state.king_audit`:
      {
        "last_at": iso8601,
        "last_status": "ok" | "failed" | "error" | "skipped",
        "consecutive_fails": int,
        "last_verdict": {...},
        "last_reason": str,
      }
    """
    repo = state.king.get("hf_repo", "")
    rev = state.king.get("king_revision", "")
    if not repo:
        return

    state.king_audit.setdefault("consecutive_fails", 0)

    log.info("king audit: probing %s@%s", repo, (rev or "HEAD")[:12])
    state.set_phase("king_audit", notes=f"reprobing king {repo}")

    verdict: dict | None = None
    error: str | None = None
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(KING_AUDIT_TIMEOUT_S)) as client:
            resp = await client.post(
                f"{EVAL_SERVER_URL}/probe",
                json={"repo": repo, "revision": rev or ""},
            )
            if resp.status_code == 409:
                # Eval lock held by a live eval — try again next interval.
                log.info("king audit: eval server busy (409), will retry next interval")
                state.king_audit["last_at"] = _now()
                state.king_audit["last_status"] = "skipped"
                state.king_audit["last_reason"] = "eval_server_busy"
                state.flush()
                return
            resp.raise_for_status()
            verdict = resp.json()
    except Exception as exc:
        error = str(exc)
        log.warning("king audit: probe call failed: %s", error)

    state.king_audit["last_at"] = _now()

    if error is not None or verdict is None:
        state.king_audit["last_status"] = "error"
        state.king_audit["last_reason"] = error or "no verdict returned"
        state.flush()
        return

    state.king_audit["last_verdict"] = {
        "ok": verdict.get("ok"),
        "reason": verdict.get("reason"),
        "max_ratio": verdict.get("max_ratio"),
        "max_grad_norm": verdict.get("max_grad_norm"),
        "min_loss_before": verdict.get("min_loss_before"),
        "max_loss_after": verdict.get("max_loss_after"),
        "norm_quantization": verdict.get("norm_quantization"),
        "n_seeds": verdict.get("n_seeds"),
        "n_steps_per_seed": verdict.get("n_steps_per_seed"),
        "warnings": verdict.get("warnings", []),
    }

    if verdict.get("ok"):
        prev_fails = state.king_audit.get("consecutive_fails", 0)
        state.king_audit["consecutive_fails"] = 0
        state.king_audit["last_status"] = "ok"
        state.king_audit["last_reason"] = None
        log.info("king audit: %s passed (max_ratio=%.3f max_grad=%.2e norm_quant=%s)",
                 repo,
                 verdict.get("max_ratio", float("nan")),
                 verdict.get("max_grad_norm", float("nan")),
                 verdict.get("norm_quantization"))
        if prev_fails:
            state.event({"event": "king_audit_recovered",
                         "hf_repo": repo, "previous_fails": prev_fails})
        state.flush()
        return

    fails = state.king_audit.get("consecutive_fails", 0) + 1
    state.king_audit["consecutive_fails"] = fails
    state.king_audit["last_status"] = "failed"
    state.king_audit["last_reason"] = verdict.get("reason", "unknown")
    log.warning("king audit FAILED %d/%d for %s: %s",
                fails, KING_AUDIT_FAILS_BEFORE_DETHRONE,
                repo, verdict.get("reason"))
    state.event({
        "event": "king_audit_failed",
        "hf_repo": repo,
        "king_revision": rev,
        "consecutive_fails": fails,
        "threshold": KING_AUDIT_FAILS_BEFORE_DETHRONE,
        "reason": verdict.get("reason"),
        "max_ratio": verdict.get("max_ratio"),
        "max_grad_norm": verdict.get("max_grad_norm"),
        "norm_quantization": verdict.get("norm_quantization"),
    })

    if fails < KING_AUDIT_FAILS_BEFORE_DETHRONE:
        state.flush()
        state.flush_dashboard()
        return

    # Threshold reached — dethrone.
    prev = state.king.get("previous_king")
    if not (prev and prev.get("hf_repo")):
        log.error("king audit: %s failed %d times but no previous_king to "
                  "revert to — operator must hand-restore", repo, fails)
        state.event({
            "event": "king_audit_dethrone_blocked",
            "hf_repo": repo, "reason": "no_previous_king",
        })
        state.flush()
        state.flush_dashboard()
        return

    log.warning("king audit: dethroning %s after %d consecutive probe failures; "
                "reverting to %s@%s",
                repo, fails, prev["hf_repo"],
                (prev.get("king_revision") or "?")[:12])

    dead_repo = repo
    dethroned_block = _safe_block(subtensor)
    state.failed_repos.add(dead_repo)
    state.king = prev
    state.king_audit = {
        "last_at": _now(),
        "last_status": "ok",
        "consecutive_fails": 0,
        "last_verdict": None,
        "last_reason": "reverted_to_previous_king",
        "dethroned_repo": dead_repo,
        "dethroned_at_block": dethroned_block,
    }
    state.flush()
    state.flush_dashboard()
    state.event({
        "event": "king_dethroned_untrainable",
        "lost_repo": dead_repo,
        "reverted_to": prev.get("hf_repo"),
        "reverted_to_revision": (prev.get("king_revision") or "")[:12],
        "consecutive_fails": fails,
        "block": dethroned_block,
    })

    try:
        await notify_king_dethroned_untrainable(dead_repo, prev, verdict)
    except Exception:
        log.warning("discord notification for untrainable dethrone failed",
                    exc_info=True)


async def notify_king_dethroned_untrainable(dead_repo: str, reverted_to: dict,
                                             verdict: dict):
    """Discord notification: incumbent king demoted by trainability audit."""
    if not DISCORD_BOT_TOKEN or not DISCORD_CHANNEL_ID:
        return
    rv_repo = reverted_to.get("hf_repo", "?")
    rv_rev = (reverted_to.get("king_revision") or "")[:12]
    lines = [
        "**King Demoted — Trainability Audit Failed**",
        f"**Demoted:** `{dead_repo}`",
        f"**Reason:** {verdict.get('reason', 'unknown')}",
        f"**max_ratio:** {verdict.get('max_ratio')}  "
        f"**max_grad_norm:** {verdict.get('max_grad_norm')}",
        f"**norm_quantization:** {verdict.get('norm_quantization')}",
        f"**Reverted to:** `{rv_repo}`" + (f" (`{rv_rev}`)" if rv_rev else ""),
    ]
    embed = {
        "title": "⚠️ Incumbent King Dethroned",
        "description": "\n".join(lines),
        "color": 0xCC3333,
    }
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            await client.post(
                f"https://discord.com/api/v10/channels/{DISCORD_CHANNEL_ID}/messages",
                headers={"Authorization": f"Bot {DISCORD_BOT_TOKEN}",
                         "Content-Type": "application/json"},
                json={"embeds": [embed]},
            )
    except Exception:
        log.warning("discord untrainable dethrone notify failed", exc_info=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def _stream_events_with_idle_watchdog(stream, state, cid):
    # IMPORTANT: bind the iterator ONCE outside the loop. httpx's aiter_lines()
    # returns iterators that share the underlying stream — calling it more than
    # once on the same response raises StreamConsumed on subsequent iterators.
    #
    # We also must NOT cancel the in-flight __anext__() on a healthcheck
    # timeout: cancelling httpx's read mid-flight closes the underlying
    # iterator, so the *next* __anext__() returns StopAsyncIteration and the
    # validator misreports "eval stream ended without verdict".
    #
    # Instead, we keep a single long-lived task per __anext__() call and only
    # spawn a new one once the previous resolved. asyncio.wait() with a
    # timeout does NOT cancel pending tasks (unlike wait_for), so the read
    # task survives across multiple healthcheck windows.
    line_iter = stream.aiter_lines()
    last_event_monotonic = _monotonic_now()
    warned = False
    pending_task: asyncio.Task | None = None
    try:
        while True:
            if pending_task is None:
                pending_task = asyncio.ensure_future(line_iter.__anext__())
            done, _pending = await asyncio.wait(
                {pending_task}, timeout=HEALTHCHECK_INTERVAL,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                idle = _monotonic_now() - last_event_monotonic
                if idle >= STREAM_IDLE_TIMEOUT:
                    raise TimeoutError(f"{cid}: eval stream idle for {idle:.0f}s")
                if idle >= STREAM_IDLE_WARN_AFTER and not warned:
                    warned = True
                    log.warning("%s: eval stream idle for %.0fs", cid, idle)
                    state.event({"event": "eval_stream_idle_warning", "challenge_id": cid,
                                 "idle_s": round(idle, 1)})
                    state.set_phase("eval_stream_idle", challenge_id=cid,
                                    notes=f"idle {idle:.0f}s waiting for eval stream")
                    state.flush_dashboard()
                continue
            try:
                line = pending_task.result()
            except StopAsyncIteration:
                return
            finally:
                pending_task = None
            last_event_monotonic = _monotonic_now()
            warned = False
            yield line
    finally:
        if pending_task is not None and not pending_task.done():
            pending_task.cancel()
            try:
                await pending_task
            except (asyncio.CancelledError, StopAsyncIteration, Exception):
                pass


def _is_transient_eval_error(exc: Exception | str) -> tuple[bool, str]:
    text = str(exc).lower()
    transient_markers = (
        "eval server error",
        "internal error",
        "stream idle",
        "watchdog timeout",
        "timed out",
        "timeout",
        "server disconnected",
        "connection reset",
        "connecterror",
        "readerror",
        "remoteprotocolerror",
        "streamconsumed",
        "streamclosed",
        "streamerror",
        "503",
        "502",
        "504",
    )
    for marker in transient_markers:
        if marker in text:
            return True, marker
    return False, ""


async def process_challenge(state, r2, entry, subtensor, wallet, *, check_stale=True):
    cid = entry["challenge_id"]
    hotkey = entry["hotkey"]
    hf_repo = entry["hf_repo"]
    log.info("processing %s from %s repo=%s", cid, hotkey[:16], hf_repo)
    state.set_phase("process_challenge", challenge_id=cid, notes=f"processing {hf_repo}")
    state.note_progress(notes=f"started processing {cid}")

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

    if check_stale:
        current_hash = state.king.get("king_hash", "")
        entry_king_hash = entry.get("king_hash", "")
        if current_hash and entry_king_hash and not current_hash.startswith(entry_king_hash[:len(entry_king_hash)]):
            log.info("stale %s: king changed (entry=%s current=%s)", cid, entry_king_hash[:16], current_hash[:16])
            state.event({"event": "stale", "challenge_id": cid, "hotkey": hotkey})
            return

    try:
        state.set_phase("hf_metadata", challenge_id=cid, notes=f"resolving {hf_repo}@main")
        challenger_info = HfApi(token=HF_TOKEN or None).model_info(hf_repo, revision="main")
        challenger_revision = challenger_info.sha
        state.remember_revision(hotkey, hf_repo, challenger_revision)
        log.info("challenger %s pinned at revision %s", hf_repo, challenger_revision[:12])
    except Exception as exc:
        log.warning("cannot get commit SHA for %s, skipping", hf_repo)
        state.failed_repos.add(hf_repo)
        state.record_failure(entry, "hf_metadata_error", str(exc))
        return

    state.set_phase("validate_config", challenge_id=cid, notes=f"validating {hf_repo}")
    rejection = validate_challenger_config(
        hf_repo, challenger_revision,
        king_repo=state.king.get("hf_repo", ""),
        king_revision=state.king.get("king_revision", ""),
    )
    if rejection:
        log.warning("rejecting %s (%s): %s", cid, hf_repo, rejection)
        state.failed_repos.add(hf_repo)
        state.record_failure(entry, "config_rejected", rejection)
        state.event({"event": "config_rejected", "challenge_id": cid,
                     "hf_repo": hf_repo, "reason": rejection})
        return

    block_hash = "default"
    try:
        eval_block = subtensor.block
        block_hash = subtensor.get_block_hash(eval_block) or "default"
    except Exception:
        pass

    state.set_phase("dataset_manifest", challenge_id=cid, notes="fetching dataset manifest")
    manifest = None
    manifest_attempts = 4
    for attempt in range(manifest_attempts):
        manifest = r2.ds_get("dataset/v2/manifest.json")
        if not manifest:
            manifest = r2.get("dataset/v1/manifest.json")
        if manifest:
            break
        if attempt < manifest_attempts - 1:
            backoff = 2 ** attempt
            log.warning("manifest fetch failed (attempt %d/%d), retrying in %ds",
                        attempt + 1, manifest_attempts, backoff)
            await asyncio.sleep(backoff)
    if not manifest:
        log.error("no dataset manifest after %d attempts; re-queuing %s",
                  manifest_attempts, cid)
        state.queue.insert(0, entry)
        state.flush()
        return
    n_shards = manifest["total_shards"]
    seed_mat = f"{block_hash}:{hotkey}".encode()
    shard_idx = int.from_bytes(hashlib.blake2b(seed_mat, digest_size=8).digest(), "little") % n_shards
    shard_key = manifest["shards"][shard_idx]["key"]

    king_repo = state.king.get("hf_repo", SEED_REPO)
    king_revision = state.king.get("king_revision", "")

    state.set_phase("dispatch_eval", challenge_id=cid, notes=f"dispatching {cid} to eval server")
    r2.put(f"eval/{cid}/meta.json", {
        "challenge_id": cid, "king_repo": king_repo,
        "king_revision": king_revision,
        "challenger_repo": hf_repo, "challenger_revision": challenger_revision,
        "hotkey": hotkey,
        "N": EVAL_N, "alpha": EVAL_ALPHA, "delta": EVAL_DELTA, "shard": shard_key,
        "eval_block": eval_block, "block_hash": block_hash,
    })

    state.current_eval = {
        "challenge_id": cid, "challenger_repo": hf_repo, "hotkey": hotkey,
        "progress": 0, "total": EVAL_N, "mu_hat": 0,
        "avg_king_loss": 0, "avg_challenger_loss": 0,
        "started_at": _now(),
    }
    state.flush_dashboard(force=True)

    verdict = None
    async with httpx.AsyncClient(timeout=httpx.Timeout(1800.0, connect=30.0)) as client:
        eval_payload = {
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
        }

        max_busy_retries = 20
        for attempt in range(max_busy_retries):
            state.set_phase("eval_dispatch_wait", challenge_id=cid,
                            notes=f"dispatch attempt {attempt + 1}/{max_busy_retries}")
            resp = await client.post(f"{EVAL_SERVER_URL}/eval", json=eval_payload)
            if resp.status_code != 409:
                break
            log.warning("%s: eval server busy (attempt %d/%d), waiting 30s",
                        cid, attempt + 1, max_busy_retries)
            await asyncio.sleep(30)
        else:
            log.error("%s: eval server still busy after %d attempts, re-queuing",
                      cid, max_busy_retries)
            state.queue.insert(0, entry)
            state.current_eval = None
            state.flush()
            state.flush_dashboard(force=True)
            return

        resp.raise_for_status()
        eval_id = resp.json()["eval_id"]
        state.set_phase("eval_stream", challenge_id=cid, eval_id=eval_id,
                        notes=f"streaming eval {eval_id}")
        state.note_progress(notes=f"eval {eval_id} started")
        state.flush_dashboard(force=True)
        log.info("eval %s dispatched to eval server as %s", cid, eval_id)

        async with client.stream("GET", f"{EVAL_SERVER_URL}/eval/{eval_id}/stream",
                                  timeout=httpx.Timeout(1800.0)) as stream:
            async for line in _stream_events_with_idle_watchdog(stream, state, cid):
                if not line.startswith("data: "):
                    continue
                event = json.loads(line[6:])

                if event["type"] == "progress":
                    d = event["data"]
                    state.note_progress(notes=f"eval {eval_id} progress {d.get('done', 0)}/{d.get('total', EVAL_N)}")
                    state.current_eval.update({
                        "progress": d.get("done", 0),
                        "total": d.get("total", EVAL_N),
                        "mu_hat": d.get("mu_hat", 0),
                        "avg_king_loss": d.get("avg_king_loss", 0),
                        "avg_challenger_loss": d.get("avg_challenger_loss", 0),
                    })
                    state.flush_dashboard()

                elif event["type"] == "verdict":
                    state.note_progress(notes=f"eval {eval_id} produced verdict")
                    verdict = event["data"]
                    verdict["challenge_id"] = cid
                    verdict["challenger_revision"] = challenger_revision
                    break

                elif event["type"] == "error":
                    raise RuntimeError(f"eval server error: {event['data']}")

    if not verdict:
        raise RuntimeError("eval stream ended without verdict")

    r2.put(f"eval/{cid}/verdict.json", verdict)
    log.info("verdict: %s (mu_hat=%.6f lcb=%.6f delta=%.6f %.1fs)",
             verdict.get("verdict", "unknown"), verdict.get("mu_hat", 0), verdict.get("lcb", 0),
             verdict.get("delta", 0), verdict.get("wall_time_s", 0))

    state.current_eval = None
    state.set_phase("post_eval", challenge_id=cid, notes="recording verdict")
    state.evaluated_repos.add(hf_repo)
    state.record_verdict(verdict, hf_repo, hotkey)

    accepted = verdict.get("accepted", False)
    if accepted:
        state.stats["accepted"] += 1
    else:
        state.stats["rejected"] += 1

    state.flush_dashboard(force=True)
    state.event({"event": "eval_completed", "challenge_id": cid,
                 "hotkey": hotkey, "accepted": accepted, **verdict})

    if accepted:
        topk_before = deepcopy(state.score_window.get("topk", []))
        state.record_accepted_result(verdict, hf_repo, hotkey, entry.get("block", 0))
        topk_after = state.score_window.get("topk", [])
        top1 = topk_after[0] if topk_after else None
        prev_top1_hotkey = topk_before[0].get("hotkey") if topk_before else None
        became_new_top1 = bool(
            top1 and top1.get("hotkey") == hotkey
            and top1.get("hotkey") != prev_top1_hotkey
        )
        if top1 and top1.get("hotkey") != prev_top1_hotkey:
            log.info("top scorer changed to %s via %s (repo=%s rev=%s)",
                     top1.get("hotkey", hotkey)[:16], cid, hf_repo, challenger_revision[:12])
            state.last_winner_hotkey = top1.get("hotkey", hotkey)

        if became_new_top1:
            # Promote immediately so the next challenger in this weight-set
            # interval is evaluated against the new frontier (not the old
            # n-2 king). Previously set_king only fired inside maybe_set_weights
            # at WEIGHT_INTERVAL boundaries, which let multiple challengers all
            # win against the same starting king within an interval.
            prev_king_snapshot = state.king.copy() if state.king else None
            try:
                new_king_hash = _seed_king_hash(hf_repo, challenger_revision)
            except Exception:
                log.warning("failed to compute new king hash for %s@%s; "
                            "falling back to verdict marker",
                            hf_repo, (challenger_revision or "")[:12],
                            exc_info=True)
                new_king_hash = "dethrone"
            dethrone_block = entry.get("block", 0) or _safe_block(subtensor)
            state.set_king(
                hotkey, hf_repo, new_king_hash,
                dethrone_block,
                challenge_id=cid,
                king_revision=challenger_revision,
            )
            # Score-window entries were measured against the OLD king and are
            # now stale — reset before forcing a weight-set so the inner
            # set_king path inside maybe_set_weights is a no-op (top1 will be
            # None, fallback_hotkey == new king hotkey).
            state.reset_score_window(dethrone_block)
            try:
                maybe_set_weights(subtensor, wallet, state,
                                   force=True, reason="dethrone")
            except Exception:
                log.exception("force weight-set after dethrone failed")
            await notify_new_king({
                "hotkey": hotkey,
                "hf_repo": hf_repo,
                "reign_number": state.king.get("reign_number", 0),
                "king_revision": challenger_revision,
                "previous_king": prev_king_snapshot,
            }, verdict)

    state.flush()
    state.flush_dashboard(force=True)


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

    wallet = bt.wallet(name=WALLET_NAME, hotkey=WALLET_HOTKEY)
    subtensor = bt.subtensor(network=NETWORK)
    state.refresh_uid_map(subtensor, NETUID)
    state.flush_dashboard(force=True)

    html_path = os.path.join(os.path.dirname(__file__) or ".", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "rb") as f:
            html_bytes = f.read()
        r2.put_dashboard_raw("index.html", html_bytes, "text/html")
        log.info("uploaded dashboard to Hippius")

    if not state.king:
        seed_revision = ""
        try:
            seed_info = HfApi(token=HF_TOKEN or None).model_info(SEED_REPO)
            seed_revision = seed_info.sha
            log.info("seed king %s at revision %s", SEED_REPO, seed_revision[:12])
        except Exception:
            log.warning("could not get seed king revision from %s", SEED_REPO)
        seed_king_hash = _seed_king_hash(SEED_REPO, seed_revision)
        state.set_king(wallet.hotkey.ss58_address, SEED_REPO, seed_king_hash,
                       subtensor.block, king_revision=seed_revision)

    state.clear_restart_request()
    maybe_set_weights(subtensor, wallet, state, force=True, reason="startup")

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
        tick_started_monotonic = _monotonic_now()
        state.begin_tick()
        try:
            if not check_king_alive(state):
                log.warning("king repo check failed, skipping this tick")
                await asyncio.sleep(POLL_INTERVAL)
                continue

            audit_age = _age_seconds(state.king_audit.get("last_at"))
            if audit_age is None or audit_age >= KING_AUDIT_INTERVAL_S:
                try:
                    await audit_incumbent_king(state, subtensor)
                except Exception:
                    log.exception("king audit failed (non-fatal)")

            if _monotonic_now() - tick_started_monotonic > TICK_WARN_AFTER:
                log.warning("tick already running for %.0fs before uid refresh",
                            _monotonic_now() - tick_started_monotonic)
            state.set_phase("refresh_uid_map", notes="refreshing metagraph uid map")
            state.refresh_uid_map(subtensor, NETUID)

            state.set_phase("fetch_market_data", notes="fetching TaoMarketCap data")
            tmc = await fetch_tmc_data()
            if tmc:
                state.market = tmc

            state.set_phase("scan_reveals", notes="polling chain for reveals")
            reveals = scan_reveals(subtensor, NETUID, state.seen)
            if reveals:
                queued_count = 0
                for rev in reveals:
                    cid = state.enqueue(rev, defer_flush=True)
                    if cid:
                        queued_count += 1
                        log.info("queued %s from %s (new)", cid, rev["hotkey"][:16])
                if queued_count:
                    state.flush()
                    state.flush_dashboard()
                    state.event({"event": "queued_batch", "count": queued_count, "kind": "new"})

            while state.queue:
                # Per-eval watchdog: reset timer for each queue item so we only
                # restart on a single stuck/hung eval, not on legitimately processing
                # a large queue back-to-back.
                eval_started_monotonic = _monotonic_now()
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
                state.note_progress(notes=f"starting queue item {entry.get('challenge_id', '?')}")
                # Distinguish "hard wall-clock kill" (asyncio.wait_for) from any
                # TimeoutError raised inside process_challenge (e.g. the stream-idle
                # watchdog), since TimeoutError == asyncio.TimeoutError in py3.11+
                # and we don't want to confuse a 423s stream-idle event with a
                # 1800s hard kill.
                async def _bounded_eval():
                    try:
                        await process_challenge(state, r2, entry, subtensor, wallet,
                                                check_stale=not is_reeval and args.seen)
                    except BaseException as inner:
                        # Re-raise inner exceptions wrapped so they don't collide
                        # with asyncio.wait_for's own TimeoutError sentinel.
                        raise _EvalInnerError(inner) from inner
                try:
                    await asyncio.wait_for(_bounded_eval(), timeout=TICK_RESTART_AFTER)
                except asyncio.TimeoutError:
                    # Hard wall-clock kill from asyncio.wait_for.
                    eval_elapsed = _monotonic_now() - eval_started_monotonic
                    reason = (f"single-eval hard timeout: {entry.get('challenge_id')} "
                              f"exceeded {TICK_RESTART_AFTER}s wall clock "
                              f"(elapsed {eval_elapsed:.0f}s)")
                    log.error(reason)
                    state.set_phase("eval_timeout", challenge_id=entry.get("challenge_id"),
                                    notes=reason)
                    state.stats["failed"] += 1
                    state.record_failure(entry, "eval_hard_timeout", reason)
                    state.current_eval = None
                    state.flush_dashboard()
                    state.flush()
                    # Don't restart -- a single bad model shouldn't kill the validator.
                    # Continue to next queue item.
                    continue
                except _EvalInnerError as wrapped:
                    exc = wrapped.original
                    log.exception("eval failed: %s", entry.get("challenge_id"),
                                  exc_info=exc)
                    is_transient, transient_reason = _is_transient_eval_error(exc)
                    retry_count = int(entry.get("retry_count", 0))
                    if is_transient and retry_count < MAX_TRANSIENT_EVAL_RETRIES:
                        state.set_phase("eval_retrying", challenge_id=entry.get("challenge_id"),
                                        notes=str(exc))
                        state.requeue_front(
                            entry,
                            reason=transient_reason or "transient_eval_error",
                            error_code="eval_error_transient",
                            error_detail=str(exc),
                        )
                    else:
                        state.stats["failed"] += 1
                        state.record_failure(entry, "eval_error", str(exc))
                        state.current_eval = None
                        state.set_phase("eval_failed", challenge_id=entry.get("challenge_id"),
                                        notes=str(exc))
                        state.flush_dashboard()

                fresh = scan_reveals(subtensor, NETUID, state.seen)
                if fresh:
                    queued_count = 0
                    for rev in fresh:
                        cid = state.enqueue(rev, defer_flush=True)
                        if cid:
                            queued_count += 1
                            log.info("queued %s from %s (new, mid-cycle)", cid, rev["hotkey"][:16])
                    new_items = [e for e in state.queue if not e.get("reeval")]
                    reeval_items = [e for e in state.queue if e.get("reeval")]
                    state.queue = new_items + reeval_items
                    if queued_count:
                        state.flush()
                        state.flush_dashboard()
                        state.event({"event": "queued_batch",
                                     "count": queued_count, "kind": "mid-cycle"})

                try:
                    maybe_set_weights(subtensor, wallet, state,
                                      reason="in-queue interval")
                except Exception:
                    log.exception("in-queue weight-set failed")

            state.current_eval = None

            if args.seen and not state.queue:
                state.replenish_reeval(subtensor, NETUID)

            if not args.seen and not state.queue:
                log.info("idle: all hotkeys seen, waiting for new submissions")

            state.complete_tick()
            state.flush_dashboard()

            try:
                maybe_set_weights(subtensor, wallet, state,
                                  reason="periodic interval")
            except Exception:
                log.exception("periodic weight-set failed")

        except KeyboardInterrupt:
            break
        except Exception as exc:
            state.fail_tick(str(exc))
            log.exception("tick error")
            if state.watchdog.get("consecutive_tick_errors", 0) >= MAX_CONSECUTIVE_TICK_ERRORS:
                reason = (f"too many consecutive tick errors: "
                          f"{state.watchdog.get('consecutive_tick_errors', 0)}")
                log.error(reason)
                state.request_restart(reason)
                state.flush()
                state.flush_dashboard()
                raise RuntimeError(reason)
        finally:
            tick_elapsed = _monotonic_now() - tick_started_monotonic
            if tick_elapsed >= TICK_WARN_AFTER:
                log.warning("tick duration %.1fs exceeded warn threshold %ss",
                            tick_elapsed, TICK_WARN_AFTER)
                state.event({"event": "tick_slow", "duration_s": round(tick_elapsed, 1)})
            flush_age = _age_seconds(state.watchdog.get("last_state_flush_at"))
            if flush_age is not None and flush_age >= STATE_FLUSH_INTERVAL:
                state.flush()
                state.flush_dashboard()

        await asyncio.sleep(POLL_INTERVAL)


def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seen", action=argparse.BooleanOptionalAction, default=True,
                   help="When idle, replenish queue with re-eval candidates (default: on). "
                        "Use --no-seen to only evaluate genuinely new hotkeys and idle "
                        "when the queue is empty.")
    return p.parse_args()


def main_sync():
    asyncio.run(main())


if __name__ == "__main__":
    main_sync()