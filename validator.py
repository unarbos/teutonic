#!/usr/bin/env python3
"""Teutonic validator — single-file king-of-the-hill evaluator.

Polls Bittensor chain for challenger submissions, dispatches evaluations
to a remote eval server (eval_server.py on a GPU box), manages king
lifecycle on Hippius Hub, persists all state to R2.
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

# Make the repo root importable regardless of cwd / how the script is invoked
# (PM2 runs from the repo root; ad-hoc ssh-and-run from any cwd should also
# work). chain_config + the vendored archs/ tree sit next to validator.py.
_repo_root = os.path.dirname(os.path.abspath(__file__))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Register the active vendored arch with AutoConfig / AutoModelForCausalLM so
# downstream transformers dispatch (config inspection, model loading) resolves the
# king checkpoint without trust_remote_code. The seed checkpoint strips
# auto_map and we deny *.py uploads in validate_challenger_config; loading the
# arch here is what makes that policy actually workable. The arch module is
# selected by chain.toml -> [arch].module.
import chain_config  # noqa: E402
from model_store import (  # noqa: E402
    DIGEST_RE,
    ModelRef,
    list_remote_files,
    list_snapshot_files,
    materialize_model,
    parse_reveal_v4,
    parse_reveal_v3,
    snapshot_size,
)
from startup_policy import should_seed_king  # noqa: E402

chain_config.load_arch()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EVAL_N = int(os.environ.get("TEUTONIC_EVAL_N", "5000"))
# Public + private split per §6.1. Default 50/50 of EVAL_N. Operators without
# a populated private holdout pool should set TEUTONIC_EVAL_N_PRIVATE=0; the
# whole-corpus-overfit defense is then disabled (public-only mode).
EVAL_N_PRIVATE = int(os.environ.get("TEUTONIC_EVAL_N_PRIVATE", str(EVAL_N // 2)))
EVAL_N_PUBLIC = int(os.environ.get("TEUTONIC_EVAL_N_PUBLIC", str(EVAL_N - EVAL_N_PRIVATE)))
EVAL_ALPHA = 0.001
SEQ_LEN = 2048
POLL_INTERVAL = 30
WEIGHT_INTERVAL = 300
NETUID = int(os.environ.get("TEUTONIC_NETUID", "3"))

# Weight policy: equal-share across the current king plus up to four prior
# distinct kings that are still registered. If none are available, fall back to
# BURN_UID (default 0 = subnet-owner burn slot) so emission still leaves the
# subnet rather than stalling.
BURN_UID = int(os.environ.get("TEUTONIC_BURN_UID", "0"))

# Watchdogs / anti-stuckness safeguards.
TICK_WARN_AFTER = int(os.environ.get("TEUTONIC_TICK_WARN_AFTER", "120"))
TICK_RESTART_AFTER = int(os.environ.get("TEUTONIC_TICK_RESTART_AFTER", "1800"))
STREAM_IDLE_WARN_AFTER = int(os.environ.get("TEUTONIC_STREAM_IDLE_WARN_AFTER", "180"))
STREAM_IDLE_TIMEOUT = int(os.environ.get("TEUTONIC_STREAM_IDLE_TIMEOUT", "420"))
HEALTHCHECK_INTERVAL = int(os.environ.get("TEUTONIC_HEALTHCHECK_INTERVAL", "60"))
STATE_FLUSH_INTERVAL = int(os.environ.get("TEUTONIC_STATE_FLUSH_INTERVAL", "60"))
MAX_CONSECUTIVE_TICK_ERRORS = int(os.environ.get("TEUTONIC_MAX_CONSECUTIVE_TICK_ERRORS", "10"))
NETWORK = os.environ.get("TEUTONIC_NETWORK", "finney")
SEED_REPO = os.environ.get("TEUTONIC_SEED_REPO", chain_config.SEED_REPO)
SEED_DIGEST = os.environ.get("TEUTONIC_SEED_DIGEST", getattr(chain_config, "SEED_DIGEST", ""))
# NOTE: TEUTONIC_FORCE_SEED_KING is intentionally ignored here. PM2 can keep stale env vars across restarts, and honoring this flag would reseed the king on every validator boot. Re-enable only with a one-shot startup policy.
FORCE_SEED_KING = False
EVAL_SERVER_URL = os.environ.get("TEUTONIC_EVAL_SERVER", "http://localhost:9000")
EVAL_DATASET_MODE = os.environ.get("TEUTONIC_EVAL_DATASET_MODE", "")
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

DISCORD_BOT_TOKEN = ""#os.environ.get("DISCORD_BOT_TOKEN", "")
DISCORD_CHANNEL_ID = ""#os.environ.get("DISCORD_CHANNEL_ID", "")

# Anti-impersonation: miners must include the first N ss58 chars of their
# coldkey somewhere in their Hippius repo name. Two miners trying to claim the
# same checkpoint would have to advertise different coldkey prefixes, so
# only the legit owner can submit a repo whose name embeds *their* coldkey.
# (Also forces miners to host under their own Hippius namespace: the basename or
# namespace must contain the prefix; case-insensitive substring check.)
# 8 chars of ss58 ≈ 40 bits of entropy after the universal "5" prefix.
COLDKEY_PREFIX_LEN = int(os.environ.get("TEUTONIC_COLDKEY_PREFIX_LEN", "8"))

# Trainability and reparam-symmetry defenses run on the eval server's /eval
# (validate_challenger_sanity + the on-GPU trainability_probe); see DESIGN.md
# §7 for the threat model.

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


# Bittensor emits one block every 12 seconds. Used to convert per-block
# emission rates from `meta.emission` into per-hour rates for the dashboard.
BLOCKS_PER_HOUR = 300

# Dashboard/Hippius writes are non-critical presentation updates. Keep them
# off the hot path and fail open when the public endpoint is degraded.
DASHBOARD_FLUSH_MIN_INTERVAL = float(os.environ.get("TEUTONIC_DASHBOARD_FLUSH_MIN_INTERVAL", "5"))
HIPPIUS_COOLDOWN_SECONDS = int(os.environ.get("TEUTONIC_HIPPIUS_COOLDOWN_SECONDS", "300"))

# Number of most-recent kings that share equal weight and appear in the Reigns table.
KING_CHAIN_SIZE = int(os.environ.get("TEUTONIC_KING_CHAIN_SIZE", "5"))

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
        # `subnet_alpha_out_emission` is the GROSS per-block alpha emission
        # to this subnet (in nanoalpha, 1e-9). It's split between three pots
        # before reaching miners:
        #   - pending_server_emission    -> miners (server side)
        #   - pending_validator_emission -> validators
        #   - pending_owner_cut          -> subnet owner
        # The dashboard cares about the MINER share, so we scale the gross
        # rate by `pending_server / (server + validator + owner)`. Earlier
        # versions used the gross number unscaled, which double-counted by
        # ~2x (miners actually get ~40% on SN3, not 100%). Falls back to a
        # 50/50 split if any pending field is missing or zero.
        try:
            gross_apb = float(snap.get("subnet_alpha_out_emission", 0)) / 1e9
        except Exception:
            gross_apb = 0.0
        try:
            pend_srv = float(snap.get("pending_server_emission", 0))
            pend_val = float(snap.get("pending_validator_emission", 0))
            pend_own = float(snap.get("pending_owner_cut", 0))
            pend_total = pend_srv + pend_val + pend_own
            miner_share = (pend_srv / pend_total) if pend_total > 0 else 0.5
        except Exception:
            miner_share = 0.5
        sn3_alpha_per_block = gross_apb * miner_share
        return {
            "tao_price_usd": tao_price,
            "tao_change_24h": m["usd_quote"]["percent_change_24h"],
            "sn3_alpha_price_tao": alpha_tao,
            "sn3_alpha_price_usd": alpha_tao * tao_price,
            "sn3_reg_burn_tao": b[0]["burn"] / 1e9,
            "sn3_alpha_per_block": sn3_alpha_per_block,
            "sn3_miner_share": miner_share,
            "sn3_alpha_per_block_gross": gross_apb,
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
    repo = king_info.get("model_repo", "?")
    hotkey = king_info.get("hotkey", "?")
    reign = king_info.get("reign_number", 0)
    revision = king_info.get("king_digest", "")[:12]

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
    prev_repo = king_info.get("previous_repo") or ""
    if prev_repo:
        lines.append(f"**Dethroned:** `{prev_repo}`")

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
            retries={"max_attempts": 10, "mode": "adaptive"},
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

    def _put_dashboard_bytes(self, key, body, content_type, cache_control=None):
        extra = {"CacheControl": cache_control} if cache_control else {}
        if self._hippius_available():
            try:
                self._hippius.put_object(
                    Bucket=HIPPIUS_BUCKET,
                    Key=key,
                    Body=body,
                    ContentType=content_type,
                    **extra,
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
                **extra,
            )
        except Exception:
            log.warning("dashboard fallback put failed for %s (non-fatal)", key, exc_info=True)

    def put_dashboard(self, key, data):
        body = json.dumps(data, default=str).encode()
        self._put_dashboard_bytes(key, body, "application/json")

    def put_dashboard_raw(self, key, body, content_type, cache_control=None):
        self._put_dashboard_bytes(key, body, content_type, cache_control=cache_control)

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

# ---------------------------------------------------------------------------
# Challenger validation
# ---------------------------------------------------------------------------

_king_config: dict | None = None
_king_config_key: str | None = None


def get_king_config(king_repo: str, king_digest: str = ""):
    """Fetch and cache the king model's config.json from a Hippius digest snapshot."""
    global _king_config, _king_config_key
    cache_key = f"{king_repo}@{king_digest}"
    if _king_config is not None and _king_config_key == cache_key:
        return _king_config
    try:
        ref = ModelRef(king_repo, king_digest)
        snapshot = materialize_model(ref, max_workers=4, config_only=True)
        with open(os.path.join(snapshot, "config.json")) as f:
            _king_config = json.load(f)
            _king_config_key = cache_key
    except Exception:
        log.warning("could not fetch king config.json from %s@%s",
                    king_repo, (king_digest or "missing")[:19])
        _king_config = {}
        _king_config_key = cache_key
    return _king_config


def validate_challenger_config(model_repo: str, challenger_digest: str,
                                king_repo: str = "",
                                king_digest: str = "") -> str | None:
    """Architecture/shape lock + repo hygiene gate (§8 submission format).

    Defends against: (a) tokenizer-shift cheats where a remapped vocab inflates
    measured CE difference for free; (b) custom modeling via auto_map /
    trust_remote_code, which would let a challenger execute arbitrary code in
    the eval server; (c) oversized uploads exhausting the validator's disk.
    The per-weight reparam-symmetry defenses (RMSNorm * Linear and SwiGLU
    invariances) live in eval_server's validate_challenger_sanity and the
    on-GPU trainability_probe — they remain unchanged per §7.
    """
    king_cfg = get_king_config(king_repo or SEED_REPO, king_digest)
    if not king_cfg:
        return None

    try:
        ref = ModelRef(model_repo, challenger_digest)
        # config-only fetch: pulls just config.json + tokenizer + index, not the
        # ~8GB safetensors. The eval server downloads the full snapshot when it
        # actually loads the model. Without this the validator's local disk
        # fills after ~50 challengers (Qwen3-4B is 8GB; no eviction logic).
        snapshot = materialize_model(ref, max_workers=4, config_only=True)
        with open(os.path.join(snapshot, "config.json")) as f:
            challenger_cfg = json.load(f)
        # Snapshot lists only the config-only subset, so use the remote
        # manifest/file-tree to verify safetensors are actually in the repo.
        repo_files = list_remote_files(ref)
    except Exception as e:
        return f"cannot materialize Hippius model snapshot: {e}"

    king_arch = king_cfg.get("architectures", [])
    chall_arch = challenger_cfg.get("architectures", [])
    if king_arch and chall_arch and king_arch != chall_arch:
        return f"architecture mismatch: king={king_arch} challenger={chall_arch}"

    _generic_lock = (
        "vocab_size", "hidden_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads", "head_dim",
        "intermediate_size", "model_type",
        "tie_word_embeddings", "rope_theta", "max_position_embeddings",
        "max_seq_len",
    )
    # Per-key compare: absent-in-king must mean absent-in-challenger too. A
    # missing key on the king side previously short-circuited the check (so
    # challengers could declare e.g. rope_scaling=YARN against a king that
    # had no rope_scaling field), allowing silent arch divergence.
    _SENTINEL = object()
    for key in _generic_lock + chain_config.EXTRA_LOCK_KEYS:
        king_val = king_cfg.get(key, _SENTINEL)
        chall_val = challenger_cfg.get(key, _SENTINEL)
        if king_val != chall_val:
            return f"{key} mismatch: king={king_val if king_val is not _SENTINEL else '<absent>'} challenger={chall_val if chall_val is not _SENTINEL else '<absent>'}"

    if "auto_map" in challenger_cfg:
        return "auto_map present in config.json (custom modeling code is not allowed)"

    py_files = [f for f in repo_files if f.endswith(".py")]
    if py_files:
        return f"repo ships *.py files (not allowed): {py_files[:3]}"

    st_files = [f for f in repo_files if f.endswith(".safetensors")]
    if not st_files:
        return "no .safetensors files in repo"

    has_single = "model.safetensors" in repo_files
    has_index = "model.safetensors.index.json" in repo_files
    has_shards = any(_SAFETENSORS_SHARD_RE.match(f) for f in st_files)
    if not (has_single or (has_index and has_shards)):
        if has_shards and not has_index:
            return (f"missing `model.safetensors.index.json` for sharded "
                    f"safetensors layout (found {sum(1 for f in st_files if _SAFETENSORS_SHARD_RE.match(f))} "
                    f"`model-NNNNN-of-NNNNN.safetensors` shards but no index file)")
        return (f"safetensors files present but none match the canonical transformers layout; "
                f"got {st_files[:3]}")

    total_st_bytes = snapshot_size(snapshot, st_files)
    if total_st_bytes > 0:
        size_gb = total_st_bytes / 1e9
        max_gb = float(os.environ.get("TEUTONIC_MAX_CHALLENGER_SAFETENSORS_GB", "200"))
        if size_gb > max_gb:
            return (f"oversized: {size_gb:.1f} GB of .safetensors > {max_gb:.0f} GB cap "
                    f"(check for fp32 weights, duplicated shards, or extra optimizer state)")

    return None

# ---------------------------------------------------------------------------
# Chain
# ---------------------------------------------------------------------------

import re
_SAFETENSORS_SHARD_RE = re.compile(r"^model-\d{5}-of-\d{5}\.safetensors$")


def _decode_commitment_pair(pair):
    """Return (hotkey_ss58, [(block, payload), ...]) for one RevealedCommitments row.

    Depending on the substrate client path, the payload may arrive as either a hex-serialized SCALE byte string (`0x...`) or raw commitment bytes wrapped in a Python str via latin-1. We normalize both shapes to bytes, strip the SCALE compact-length prefix, and decode the rest as UTF-8.
    """
    key, data = pair
    if not isinstance(key, str):
        raise ValueError(f"unexpected commitment key type {type(key).__name__}")
    out = []
    for entry in data:
        text, block = entry
        if not isinstance(text, str):
            raise ValueError(f"unexpected commitment payload type {type(text).__name__}")
        if text.startswith(("0x", "0X")):
            raw = bytes.fromhex(text[2:])
        else:
            raw = text.encode("latin-1")
        if not raw:
            raise ValueError("empty commitment payload")
        mode = raw[0] & 0b11
        offset = 1 if mode == 0 else 2 if mode == 1 else 4
        out.append((block, raw[offset:].decode("utf-8", errors="ignore")))
    return key, out


def scan_reveals(subtensor, netuid, completed_repos, seen_hotkeys):
    """Pull v4 reveals; return latest per hotkey not previously enqueued.

    v4 format: `v4|<challenger_repo>|<challenger_digest>|<author_hotkey>`.
    Any legacy reveal that still embeds a king digest is dropped at intake.

    Per-pair decode via _decode_commitment_pair instead of bittensor's
    `decode_revealed_commitment_with_hotkey`, which (a) raises on any single
    bad legacy row and poisons the whole scan, and (b) assumes hex-encoded
    payloads in bt 10.3 even though substrate returns raw bytes. Both bugs
    have a single fix: decode it ourselves.
    """
    try:
        query = subtensor.query_map(module="Commitments", name="RevealedCommitments", params=[netuid])
    except Exception:
        log.exception("query_map RevealedCommitments failed")
        return []
    all_reveals = {}
    bad = 0
    for pair in query:
        try:
            hotkey_ss58, commitment_msg = _decode_commitment_pair(pair)
            all_reveals[hotkey_ss58] = commitment_msg
        except Exception:
            bad += 1
    if bad:
        log.warning("scan_reveals: skipped %d undecodable on-chain commitments", bad)
    if not all_reveals:
        return []

    new = []
    for hotkey, entries in all_reveals.items():
        if not entries or hotkey in seen_hotkeys:
            continue
        block, data = max(entries, key=lambda e: e[0])
        try:
            ref, author_hotkey = parse_reveal_v4(data)
        except ValueError:
            try:
                legacy_king_digest, _legacy_ref, _legacy_author_hotkey = parse_reveal_v3(data)
            except ValueError:
                continue
            log.warning("dropping legacy king-bound reveal from %s at block %s "
                        "(king_digest=%s); resubmit as v4 without king binding",
                        hotkey[:16], block, legacy_king_digest[:19])
            continue
        if author_hotkey != hotkey:
            # Chain side is the source of truth (commit-reveal keys reveals by
            # signer). Payload mismatch means the miner stuffed the wrong ss58
            # in their reveal — log and trust the chain key.
            log.warning("v4 author_hotkey %s mismatches chain key %s; trusting chain",
                        author_hotkey[:16], hotkey[:16])
        if ref.immutable_ref in completed_repos:
            continue
        new.append({
            "hotkey": hotkey,
            "block": block,
            "model_repo": ref.repo,
            "model_digest": ref.digest,
        })
    new.sort(key=lambda x: x["block"])
    return new


async def maybe_set_weights(subtensor, wallet, state, *, force: bool = False,
                            reason: str = "") -> bool:
    """Push equal-share weight to the current king plus recent prior kings.
    Falls back to BURN_UID if no king is set or none of the tracked king
    hotkeys are on the metagraph.

    Async — the underlying `set_weights` call blocks for inclusion +
    finalization (~25-50s) so it runs in a thread executor to keep the event
    loop responsive. Routes through commit-reveal v4 when SN3 has CR enabled
    (asserted at startup). Rate-limited per `WEIGHT_INTERVAL`.
    """
    try:
        current_block = subtensor.block
    except Exception:
        log.exception("failed to read current block for weight-set")
        return False
    if not force and current_block - state.last_weight_block < WEIGHT_INTERVAL:
        return False

    # Collect up to KING_CHAIN_SIZE distinct hotkeys: current king first, then past kings.
    all_king_hks: list[str] = []
    king_hotkey = (state.king or {}).get("hotkey", "")
    if king_hotkey:
        all_king_hks.append(king_hotkey)
    for e in (state.king_chain or []):
        hk = e.get("hotkey", "")
        if hk and hk not in all_king_hks:
            all_king_hks.append(hk)

    target_uids = [int(state.uid_map[hk]) for hk in all_king_hks if hk in state.uid_map]
    if not target_uids:
        target_uids = [BURN_UID]
        weights_list = [1.0]
        winner_label = f"burn:uid={BURN_UID}"
        log_target = f"burn uid={BURN_UID} (no kings registered)"
    else:
        w = round(1.0 / len(target_uids), 9)
        weights_list = [w] * len(target_uids)
        winner_label = king_hotkey or "multi"
        log_target = f"uids={target_uids} weight={w:.4f} each ({len(target_uids)} kings)"

    log.info("set_weights at block %d (last=%d, %s) -> %s",
             current_block, state.last_weight_block,
             reason or ("forced" if force else "interval"), log_target)
    loop = asyncio.get_running_loop()
    try:
        resp = await loop.run_in_executor(
            None,
            lambda: subtensor.set_weights(
                wallet=wallet, netuid=NETUID, uids=target_uids, weights=weights_list
            ),
        )
    except Exception:
        log.exception("failed to set weights")
        return False
    if not resp.success:
        # bt's internal rate-limit guard returns success=False with no message
        # when blocks_since_last_update <= weights_rate_limit. Treat as no-op
        # and advance last_weight_block so we don't hammer every tick.
        if not resp.message:
            log.info("set_weights rate-limited (no-op); advancing last_weight_block")
            state.last_weight_block = current_block
        else:
            log.error("set_weights failed: %s", resp.message)
        return False
    state.last_weight_block = current_block
    state.last_winner_hotkey = winner_label
    try:
        state.flush()
        state.flush_dashboard()
    except Exception:
        log.exception("failed to flush state after weight set")
    return True


# TODO: operator must enable Yuma3 liquid alpha for this subnet via the
# subnet-owner `sudo_set_liquid_alpha_enabled` extrinsic (admin_utils pallet).
# Validators do not hold this privilege. §9 specifies liquid alpha enabled with
# default settings — there is nothing for the validator process to do here, but
# leaving the marker so the deploy checklist surfaces it.


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


def _model_key(repo: str, digest: str = "") -> str:
    return f"{repo}@{digest}" if digest else repo


class State:
    def __init__(self, r2):
        self.r2 = r2
        self.king = {}
        self.queue = []
        self.seen = set()
        self.failed_repos: set[str] = set()
        self.evaluated_repos: set[str] = set()
        self.completed_repos: set[str] = set()
        self.stats = {"queued": 0, "accepted": 0, "rejected": 0, "failed": 0}
        self.counter = 0
        self.current_eval = None
        self.history = []
        self.last_weight_block = 0
        self.last_winner_hotkey: str | None = None
        self.market: dict | None = None
        self.uid_map: dict[str, int] = {}
        self.uid_emission_per_block: dict[str, float] = {}
        self.hotkey_coldkey: dict[str, str] = {}
        self.known_digests: dict[str, dict[str, str]] = {}
        self.king_chain: list[dict] = []
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
        cr = self.r2.get("state/completed_repos.json")
        if cr:
            self.completed_repos = set(cr.get("repos", []))
        st = self.r2.get("state/validator_state.json")
        if st:
            self.stats = st.get("stats", self.stats)
            self.counter = st.get("counter", 0)
            self.last_weight_block = st.get("last_weight_block", 0)
            self.last_winner_hotkey = st.get("last_winner_hotkey")
            self.known_digests = st.get("known_digests", {})
        h = self.r2.get("state/dashboard_history.json")
        if h:
            self.history = h.get("history", [])
        kc = self.r2.get("state/king_chain.json")
        if kc:
            self.king_chain = kc.get("chain", [])
        wd = self.r2.get("state/watchdog.json")
        if wd:
            self.watchdog.update(wd)

        log.info("loaded state: king=%s@%s queue=%d seen=%d completed=%d",
                 self.king.get("model_repo", "none"),
                 (self.king.get("king_digest") or "")[:12],
                 len(self.queue), len(self.seen), len(self.completed_repos))

    def flush(self):
        now = _now()
        self.watchdog["last_state_flush_at"] = now
        self.r2.put("state/validator_state.json", {
            "king": self.king, "queue": self.queue,
            "stats": self.stats, "counter": self.counter,
            "last_weight_block": self.last_weight_block,
            "last_winner_hotkey": self.last_winner_hotkey,
            "known_digests": self.known_digests,
            "updated_at": now,
        })
        self.r2.put("state/queue.json", {"pending": self.queue, "updated_at": now})
        self.r2.put("king/current.json", self.king)
        self.r2.put("state/seen_hotkeys.json", {
            "hotkeys": sorted(self.seen), "updated_at": now,
        })
        self.r2.put("state/completed_repos.json", {
            "repos": sorted(self.completed_repos), "updated_at": now,
        })
        self.r2.put("state/watchdog.json", self.watchdog)
        self.r2.put("state/king_chain.json", {"chain": self.king_chain})

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
        repo = reveal.get("model_repo", "")
        digest = reveal.get("model_digest", "")
        model_key = _model_key(repo, digest)
        hotkey = reveal.get("hotkey", "")
        king_hotkey = self.king.get("hotkey", "")
        if king_hotkey and hotkey == king_hotkey:
            log.info("skipping enqueue: hotkey %s is the current king", hotkey[:16])
            return None
        # 1-hotkey-1-eval: this is the policy gate. `scan_reveals` already
        # filters by `seen` at the chain-intake layer; this is the
        # belt-and-suspenders check for any direct enqueue path. A hotkey
        # that's already been enqueued (whether the eval succeeded, failed,
        # or was lost to a validator crash) cannot submit again — the miner
        # must register a fresh hotkey on subnet.
        if hotkey and hotkey in self.seen:
            log.info("skipping enqueue: hotkey %s already used its 1-eval slot "
                     "(must re-register for another shot)", hotkey[:16])
            return None
        for existing in self.queue:
            if existing.get("model_repo") == repo:
                log.info("skipping duplicate repo: %s already queued", repo)
                return None
        if model_key in self.evaluated_repos:
            log.info("skipping %s: already evaluated this cycle", repo)
            return None
        cid = self.next_id()
        entry = {"challenge_id": cid, **reveal, "queued_at": _now(), "retry_count": int(reveal.get("retry_count", 0))}
        entry.pop("reeval", None)
        self.queue.append(entry)
        self.stats["queued"] += 1
        # Burn the hotkey now (at enqueue, not at verdict). Per policy this
        # is intentional: one reveal = one shot. If the validator crashes
        # between enqueue and verdict the eval is lost and the miner must
        # register a fresh hotkey to retry. Mirroring the burn into
        # `completed_repos` keeps the per-repo idempotency check effective
        # in case a hotkey is somehow ever revived (operator override).
        if hotkey:
            self.seen.add(hotkey)
        if repo:
            self.completed_repos.add(model_key)
        if not defer_flush:
            self.flush()
            self.flush_dashboard(force=True)
        return cid

    def requeue_front(self, entry, *, reason: str, error_code: str = "", error_detail: str = ""):
        """Requeue an existing challenge at the front for transient infra failures.

        Keeps challenge_id and original repo/hotkey, increments retry_count,
        refreshes queued_at, and avoids duplicating the same repo if it's already
        pending elsewhere in the queue.
        """
        repo = entry.get("model_repo", "")
        retry_count = int(entry.get("retry_count", 0)) + 1
        new_entry = {**entry, "retry_count": retry_count, "queued_at": _now()}
        new_entry.pop("reeval", None)

        deduped = []
        for existing in self.queue:
            if existing.get("model_repo") == repo:
                continue
            deduped.append(existing)
        self.queue = [new_entry] + deduped
        self.current_eval = None
        self.flush()
        self.flush_dashboard(force=True)
        log.warning("re-queued %s at front (retry %d/%d) due to %s: %s",
                    entry.get("challenge_id", "?"), retry_count,
                    MAX_TRANSIENT_EVAL_RETRIES, reason, error_detail)
        return retry_count

    def remember_digest(self, hotkey, repo, digest):
        if not hotkey:
            return
        self.known_digests[hotkey] = {
            "repo": repo,
            "digest": digest,
            "updated_at": _now(),
        }

    def best_known_digest(self, hotkey, repo=""):
        info = self.known_digests.get(hotkey, {})
        if repo and info.get("repo") and info.get("repo") != repo:
            return ""
        return info.get("digest", "")

    def set_king(self, hotkey, model_repo, block, challenge_id="seed", king_digest=""):
        global _king_config, _king_config_key
        _king_config = None
        _king_config_key = None
        self.failed_repos.clear()
        self.evaluated_repos.clear()
        reign = self.king.get("reign_number", 0) + (0 if challenge_id == "seed" else 1)
        prev_repo = self.king.get("model_repo") if self.king else ""
        if self.king and challenge_id != "seed":
            past = {**self.king,
                    "uid": self.uid_map.get(self.king.get("hotkey", "")),
                    "coldkey": self.coldkey_for(self.king.get("hotkey", ""))}
            self.king_chain.insert(0, past)
            self.king_chain = self.king_chain[:KING_CHAIN_SIZE - 1]
        self.king = {
            "hotkey": hotkey, "model_repo": model_repo,
            "king_digest": king_digest,
            "reign_number": reign, "crowned_at": _now(),
            "crowned_block": block, "challenge_id": challenge_id,
            "previous_repo": prev_repo,
        }
        self.flush()
        self.flush_dashboard(force=True)

    def record_verdict(self, verdict, challenger_repo, hotkey):
        # Probe-rejected verdicts (eval_server.py probe-fail path) lack the
        # full bootstrap-test fields (avg_*_loss / wall_time_s / timestamp).
        # Default everything so a partial verdict still records cleanly.
        king_loss = verdict.get("avg_king_loss", 0)
        chall_loss = verdict.get("avg_challenger_loss", 0)
        delta = verdict.get("delta", verdict.get("delta_threshold", 0))
        entry = {
            "challenge_id": verdict.get("challenge_id"),
            "hotkey": hotkey,
            "uid": self.uid_map.get(hotkey),
            "coldkey": self.coldkey_for(hotkey),
            "challenger_repo": challenger_repo,
            "challenger_digest": verdict.get("challenger_digest", ""),
            "accepted": verdict.get("accepted", False),
            "verdict": verdict.get("verdict", "unknown"),
            "mu_hat": verdict.get("mu_hat", 0),
            "lcb": verdict.get("lcb", 0),
            "delta": delta,
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
        hk = entry.get("hotkey", "")
        self.history.insert(0, {
            "challenge_id": entry.get("challenge_id", "?"),
            "hotkey": hk,
            "uid": self.uid_map.get(hk),
            "coldkey": self.coldkey_for(hk) if hk else None,
            "challenger_repo": entry.get("model_repo", ""),
            "challenger_digest": entry.get("model_digest", ""),
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
            cks = list(getattr(meta, "coldkeys", []) or [])
            self.hotkey_coldkey = {
                hk: cks[uid]
                for hk, uid in self.uid_map.items()
                if uid < len(cks) and cks[uid]
            }
        except Exception:
            log.warning("failed to refresh uid_map", exc_info=True)

    def coldkey_for(self, hotkey: str) -> str | None:
        return self.hotkey_coldkey.get(hotkey) or None

    def expected_coldkey_prefix(self, hotkey: str) -> str | None:
        """First N chars of the miner's coldkey ss58, used to gate Hippius repo
        names. Returns None when the metagraph hasn't surfaced this hotkey
        yet — callers should treat that as "skip the check, retry later".
        """
        ck = self.coldkey_for(hotkey)
        if not ck:
            return None
        return ck[:COLDKEY_PREFIX_LEN]

    def _with_fresh_uid(self, entry):
        """Return a copy of `entry` whose `uid` and `coldkey` are re-derived
        from the current metagraph. Insert-time uids can go stale (deregistration,
        hotkey re-registration under a new uid) and old payloads from before
        these fields existed had `uid="?"` / no coldkey at all. We project at
        flush time so the dashboard is always consistent with the latest
        `refresh_uid_map` snapshot, and so the dashboard hotkey -> coldkey
        link always points at the *current* coldkey for a hotkey rather than
        whatever coldkey was on file when the duel was recorded.
        """
        hk = entry.get("hotkey") if isinstance(entry, dict) else None
        if not hk:
            return entry
        # Prefer freshly resolved coldkey; fall back to the persisted value
        # only if the hotkey has been deregistered out of the metagraph.
        ck = self.coldkey_for(hk) or entry.get("coldkey")
        return {**entry, "uid": self.uid_map.get(hk), "coldkey": ck}

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
            mkt = self.market or {}
            alpha_tao = float(mkt.get("sn3_alpha_price_tao") or 0.0)
            alpha_usd = float(mkt.get("sn3_alpha_price_usd") or 0.0)
            sn3_alpha_per_block = float(mkt.get("sn3_alpha_per_block") or 0.0)
            
            # Compute equal-share payout across all registered kings in the chain.
            all_king_hks: list[str] = []
            if self.king:
                all_king_hks.append(self.king.get("hotkey", ""))
            for e in self.king_chain:
                hk = e.get("hotkey", "")
                if hk and hk not in all_king_hks:
                    all_king_hks.append(hk)
            registered_kings = [hk for hk in all_king_hks if hk in self.uid_map]
            n_kings = max(len(registered_kings), 1)
            alpha_per_hour_total = sn3_alpha_per_block * BLOCKS_PER_HOUR
            equal_alpha = round(alpha_per_hour_total / n_kings, 6)
            equal_usd = round(equal_alpha * alpha_usd, 4)
            equal_weight = round(1.0 / n_kings, 9)

            king_hk = self.king.get("hotkey") if self.king else None
            if king_hk and king_hk in self.uid_map:
                em_per_block = float(self.uid_emission_per_block.get(king_hk, 0.0))
                alpha_per_hour = sn3_alpha_per_block * BLOCKS_PER_HOUR
                tao_per_hour = alpha_per_hour * alpha_tao
                usd_per_hour = alpha_per_hour * alpha_usd
                king_payout = {
                    "hotkey": king_hk,
                    "uid": self.uid_map.get(king_hk),
                    "coldkey": self.coldkey_for(king_hk),
                    "weight": 1.0,
                    "weight_share": 1.0,
                    "emission_per_block": round(em_per_block, 9),
                    "projected_alpha_per_block": round(sn3_alpha_per_block, 9),
                    "alpha_per_hour": round(alpha_per_hour, 6),
                    "tao_per_hour": round(tao_per_hour, 6),
                    "usd_per_hour": round(usd_per_hour, 4),
                }
            else:
                king_payout = None


            # Build king_chain for dashboard: current king first, then past kings.
            # Field names follow the schema: model_repo, king_revision (mapped from
            # internal model_repo / king_digest), plus per-king payout fields.
            def _chain_entry(e, hk):
                registered = hk in self.uid_map
                aw = equal_alpha if registered else None
                uw = equal_usd if registered else None
                tw = round(aw * alpha_tao, 6) if aw is not None else None
                return {
                    "challenge_id":  e.get("challenge_id"),
                    "reign_number":  e.get("reign_number"),
                    "hotkey":        hk,
                    "uid":           self.uid_map.get(hk),
                    "coldkey":       self.coldkey_for(hk),
                    "model_repo":       e.get("model_repo", e.get("model_repo", "")),
                    "king_revision": e.get("king_digest", e.get("king_revision", "")),
                    "crowned_at":    e.get("crowned_at"),
                    "crowned_block": e.get("crowned_block"),
                    "weight":        equal_weight if registered else None,
                    "alpha_per_hour": aw,
                    "tao_per_hour":  tw,
                    "usd_per_hour":  uw,
                }
            dashboard_king_chain = []
            if self.king:
                dashboard_king_chain.append(_chain_entry(self.king, king_hk or ""))
            for e in self.king_chain:
                dashboard_king_chain.append(_chain_entry(e, e.get("hotkey", "")))

            king_chain_weights = [
                {
                    "hotkey":                   hk,
                    "uid":                      self.uid_map.get(hk),
                    "coldkey":                  self.coldkey_for(hk),
                    "weight":                   equal_weight,
                    "weight_share":             equal_weight,
                    "emission_per_block":       round(float(self.uid_emission_per_block.get(hk, 0.0)), 9),
                    "projected_alpha_per_block": round(sn3_alpha_per_block / n_kings, 9),
                    "alpha_per_hour":           equal_alpha,
                    "tao_per_hour":             round(equal_alpha * alpha_tao, 6),
                    "usd_per_hour":             equal_usd,
                }
                for hk in all_king_hks
                if hk in self.uid_map
            ]
            payload = {
                "updated_at": _now(),
                "chain": {
                    "name": chain_config.NAME,
                    "seed_repo": chain_config.SEED_REPO,
                    "seed_digest": SEED_DIGEST,
                },
                "king": self.king,
                "king_payout": king_payout,
                "king_chain": dashboard_king_chain,
                "king_chain_weights": king_chain_weights,
                "stats": self.stats,
                "current_eval": self.current_eval,
                "watchdog": self.watchdog,
                "queue": [{"challenge_id": e.get("challenge_id"), "hotkey": e.get("hotkey"),
                            "uid": self.uid_map.get(e.get("hotkey", "")),
                            "coldkey": self.coldkey_for(e.get("hotkey", "")),
                            "model_repo": e.get("model_repo"),
                            "model_digest": e.get("model_digest"),
                            "queued_at": e.get("queued_at"),
                            "block": e.get("block")}
                           for e in self.queue],
                "history": [self._with_fresh_uid(h) for h in self.history],
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

    def clear_restart_request(self):
        self.watchdog["restart_requested"] = False
        self.watchdog["restart_reason"] = ""



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
    # asyncio.CancelledError fires when the validator itself is shutting down
    # (SIGINT/SIGTERM from `pm2 restart`, deploys, etc.). The eval is still
    # running on the eval server, and the miner did nothing wrong, so this
    # should re-queue rather than record a permanent failure in duel history.
    # str(CancelledError()) is "" so the marker scan below would mis-classify
    # it as fatal — short-circuit on type instead.
    if isinstance(exc, asyncio.CancelledError):
        return True, "validator_cancelled"
    text = str(exc).lower()
    # Eval-server-reported failures are categorically PERMANENT for that model:
    # the eval-server already gave it a full prefetch budget / GPU window, so
    # retrying just burns another 30-min budget on the same dead CDN (or the
    # same OOM, or the same rejected config). Caught by the explicit
    # "prefetch ... exceeded ... (likely stuck CDN)" string the eval-server
    # raises, plus any error structured as `eval server error: {...}` coming
    # via the SSE error event. Genuine infra hiccups (network reset, stream
    # truncation, CancelledError above) still match the remaining
    # transient_markers below and are retried as today. Pre-2026-05-13 these
    # both fell into the "eval server error" wildcard marker and got 3 retry
    # attempts each — observed live with jenny08311 v5.13 (2026-05-12, ~90
    # min wasted) and ClarenceDan A5518/A5519 (2026-05-13, ~3 hours wasted
    # back-to-back).
    if ("stuck cdn" in text) or ("prefetch" in text and "exceeded" in text):
        return False, "prefetch_exhausted"
    if text.startswith("eval server error") or "'eval server error'" in text:
        return False, "eval_server_reported"
    transient_markers = (
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
        # Eval-server SSE stream getting truncated (eval-server restart, tunnel
        # blip, k8s pod cycle). These are infrastructure and should never go
        # in the miner's duel-history as a permanent failure.
        "peer closed connection",
        "incomplete chunked",
        "incompleteread",
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
    model_repo = entry["model_repo"]
    log.info("processing %s from %s repo=%s", cid, hotkey[:16], model_repo)
    state.set_phase("process_challenge", challenge_id=cid, notes=f"processing {model_repo}")
    state.note_progress(notes=f"started processing {cid}")

    king_hotkey = state.king.get("hotkey", "")
    if king_hotkey and hotkey == king_hotkey:
        log.info("skipping %s: challenger hotkey %s is the current king", cid, hotkey[:16])
        return

    model_key = _model_key(model_repo, entry.get("model_digest", ""))
    if model_key in state.failed_repos:
        log.info("skipping %s: repo %s previously failed", cid, model_repo)
        return

    if model_key in state.evaluated_repos:
        log.info("skipping %s: repo %s already evaluated this cycle", cid, model_repo)
        return

    # Any queued entry that still carries a king digest predates the v4 fork.
    # Drop it rather than evaluating a payload shape we no longer accept.
    legacy_king_digest = (entry.get("king_digest_at_reveal") or "").strip()
    if legacy_king_digest:
        log.warning("rejecting %s: legacy reveal pinned king %s", cid, legacy_king_digest[:19])
        state.failed_repos.add(model_key)
        state.record_failure(entry, "legacy_reveal_version",
                              "submission included king_digest; resubmit as "
                              "v4|repo|challenger_digest|author_hotkey")
        return

    expected_ck_prefix = state.expected_coldkey_prefix(hotkey)
    if expected_ck_prefix:
        # Case-insensitive substring match anywhere in the full repo (so the
        # miner can put their coldkey prefix in the Hippius namespace OR the
        # model basename — whichever they prefer).
        if expected_ck_prefix.lower() not in model_repo.lower():
            reason = (f"Hippius repo '{model_repo}' must contain miner coldkey prefix "
                      f"'{expected_ck_prefix}' (first {COLDKEY_PREFIX_LEN} chars "
                      f"of the coldkey ss58); rename your Hippius namespace or model "
                      f"to embed it, then re-reveal on chain")
            log.warning("rejecting %s (%s): %s", cid, model_repo, reason)
            state.failed_repos.add(model_key)
            state.record_failure(entry, "coldkey_required", reason)
            return
    else:
        # Metagraph hasn't surfaced this hotkey's coldkey yet (fresh
        # registration, refresh_uid_map staleness). Skip the check and let
        # the next tick retry — we don't want to penalize miners for our
        # own metagraph latency.
        log.info("%s: coldkey for %s not in metagraph yet, skipping coldkey check",
                 cid, hotkey[:16])

    _ = check_stale  # parameter retained for back-compat; v4 removed stale-parent binding

    # Pin to the OCI digest that the miner committed on-chain. Every reveal
    # entry MUST have a `model_digest` field; `scan_reveals` rejects everything
    # else at the chain-intake layer. We still verify the committed digest
    # exists on Hippius Hub (catches typos / forged commits); a missing digest is
    # a `digest_not_found`
    # failure rather than the legacy "fall back to HEAD" behavior.
    #
    # NOTE: any in-flight queue entry from before the fork lacks `model_digest`
    # — those were purged from R2 state at deploy time so they shouldn't
    # exist; if one slips through we fail it explicitly rather than silently
    # falling back to a mutable tag (which would re-open the exploit).
    challenger_digest = entry.get("model_digest", "").strip()
    if not challenger_digest:
        log.warning("eval %s: legacy queue entry without committed digest "
                    "(repo=%s) — failing rather than falling back to HEAD",
                    cid, model_repo)
        state.failed_repos.add(model_key)
        state.record_failure(entry, "legacy_format",
                              "queue entry predates the revision-pinned hard fork; "
                              "miner must resubmit with the new miner.py")
        return
    if not DIGEST_RE.match(challenger_digest):
        log.warning("eval %s: digest %r is not a valid digest "
                    "(expected sha256:<64hex> or hf:<40hex>)",
                    cid, challenger_digest[:32])
        state.failed_repos.add(model_key)
        state.record_failure(entry, "digest_malformed",
                              f"on-chain digest {challenger_digest!r} is not a valid "
                              f"digest (expected sha256:<64hex> or hf:<40hex>)")
        return
    try:
        state.set_phase("hippius_metadata", challenge_id=cid,
                         notes=f"verifying {model_repo}@{challenger_digest[:19]}")
        ref = ModelRef(model_repo, challenger_digest)
        materialize_model(ref, max_workers=4, config_only=True)
        state.remember_digest(hotkey, model_repo, challenger_digest)
        log.info("challenger %s pinned at digest %s (committed on-chain)",
                 model_repo, challenger_digest[:19])
    except Exception as exc:
        log.warning("cannot resolve committed digest %s of %s, skipping",
                    challenger_digest[:19], model_repo)
        state.failed_repos.add(model_key)
        state.record_failure(entry, "digest_not_found",
                              f"Hippius returned no metadata for {model_repo}@{challenger_digest[:19]}: {exc}")
        return

    state.set_phase("validate_config", challenge_id=cid, notes=f"validating {model_repo}")
    rejection = validate_challenger_config(
        model_repo, challenger_digest,
        king_repo=state.king.get("model_repo", ""),
        king_digest=state.king.get("king_digest", ""),
    )
    if rejection:
        log.warning("rejecting %s (%s): %s", cid, model_repo, rejection)
        state.failed_repos.add(model_key)
        state.record_failure(entry, "config_rejected", rejection)
        return

    # §6.1 + §10: holdout seed material is `block_hash_at_reveal || hotkey`,
    # so the pinned block_hash MUST be the reveal-commit block's hash (which
    # an external auditor can fetch from the chain) — NOT the eval-time block,
    # which is arbitrary minutes/hours later. Use the entry's reveal block;
    # only fall back to the eval block if scan didn't capture one.
    eval_block = _safe_block(subtensor)
    try:
        eval_block = subtensor.block
    except Exception:
        pass
    reveal_block = int(entry.get("block", 0) or eval_block)
    block_hash = "default"
    try:
        block_hash = subtensor.get_block_hash(reveal_block) or "default"
    except Exception:
        pass

    if EVAL_DATASET_MODE.lower() in {"raw", "raw_hippius", "fineweb", "fineweb_edu"}:
        state.set_phase("dataset_raw", challenge_id=cid, notes="using raw Hippius dataset")
        shard_key = "raw:hippius:fineweb-edu"
    else:
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

    king_repo = state.king.get("model_repo", SEED_REPO)
    king_digest = state.king.get("king_digest", "")

    state.set_phase("dispatch_eval", challenge_id=cid, notes=f"dispatching {cid} to eval server")
    r2.put(f"eval/{cid}/meta.json", {
        "challenge_id": cid, "king_repo": king_repo,
        "king_digest": king_digest,
        "challenger_repo": model_repo, "challenger_digest": challenger_digest,
        "hotkey": hotkey,
        "N": EVAL_N, "alpha": EVAL_ALPHA, "shard": shard_key,
        "eval_block": eval_block, "block_hash": block_hash,
    })

    state.current_eval = {
        "challenge_id": cid, "challenger_repo": model_repo, "hotkey": hotkey,
        "progress": 0, "total": EVAL_N, "mu_hat": 0,
        "avg_king_loss": 0, "avg_challenger_loss": 0,
        "stage": "dispatching",
        "stage_started_at": _now(),
        "stage_elapsed_s": 0,
        "started_at": _now(),
    }
    state.flush_dashboard(force=True)

    verdict = None
    async with httpx.AsyncClient(timeout=httpx.Timeout(1800.0, connect=30.0)) as client:
        eval_payload = {
            "king_repo": king_repo,
            "challenger_repo": model_repo,
            "block_hash": block_hash,
            "hotkey": hotkey,
            "shard_key": shard_key,
            "king_digest": king_digest,
            "challenger_digest": challenger_digest,
            "delta_threshold": 0.0025,
            "n_public": EVAL_N_PUBLIC,
            "n_private": EVAL_N_PRIVATE,
            "n_bootstrap": 10_000,
            "alpha": EVAL_ALPHA,
            "seq_len": SEQ_LEN,
        }

        max_busy_retries = 30
        for attempt in range(max_busy_retries):
            state.set_phase("eval_dispatch_wait", challenge_id=cid,
                            notes=f"dispatch attempt {attempt + 1}/{max_busy_retries}")
            state.current_eval["stage"] = "waiting_for_slot"
            state.current_eval["stage_started_at"] = _now()
            state.current_eval["stage_extra"] = {
                "attempt": attempt + 1, "max_attempts": max_busy_retries,
            }
            state.flush_dashboard(force=True)
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
                        "stage": "bootstrap_running",
                        "stage_started_at": _now(),
                        "stage_elapsed_s": 0,
                    })
                    state.flush_dashboard()

                elif event["type"] == "stage":
                    d = event["data"]
                    stage_name = d.get("name", "?")
                    extra = {k: v for k, v in d.items() if k not in ("name", "ts")}
                    state.current_eval["stage"] = stage_name
                    state.current_eval["stage_started_at"] = _now()
                    state.current_eval["stage_elapsed_s"] = 0
                    state.current_eval["stage_extra"] = extra
                    state.set_phase(f"eval_{stage_name}", challenge_id=cid,
                                    eval_id=eval_id, notes=stage_name)
                    state.note_progress(notes=f"stage {stage_name}")
                    state.flush_dashboard(force=True)

                elif event["type"] == "heartbeat":
                    d = event["data"]
                    state.current_eval["stage"] = d.get("stage", state.current_eval.get("stage", "?"))
                    state.current_eval["stage_elapsed_s"] = d.get("elapsed_s", 0)
                    state.note_progress(notes=f"stage {d.get('stage','?')} {d.get('elapsed_s',0):.0f}s")
                    state.flush_dashboard()

                elif event["type"] == "verdict":
                    state.note_progress(notes=f"eval {eval_id} produced verdict")
                    verdict = event["data"]
                    verdict["challenge_id"] = cid
                    verdict["challenger_digest"] = challenger_digest
                    break

                elif event["type"] == "error":
                    raise RuntimeError(f"eval server error: {event['data']}")

    if not verdict:
        raise RuntimeError("eval stream ended without verdict")

    r2.put(f"eval/{cid}/verdict.json", verdict)
    verdict_delta = verdict.get("delta", verdict.get("delta_threshold", 0))
    log.info("verdict: %s (mu_hat=%.6f lcb=%.6f delta=%.6f %.1fs)",
             verdict.get("verdict", "unknown"), verdict.get("mu_hat", 0), verdict.get("lcb", 0),
             verdict_delta, verdict.get("wall_time_s", 0))

    state.current_eval = None
    state.set_phase("post_eval", challenge_id=cid, notes="recording verdict")
    state.evaluated_repos.add(model_key)
    state.record_verdict(verdict, model_repo, hotkey)

    accepted = verdict.get("accepted", False)
    if accepted:
        state.stats["accepted"] += 1
    else:
        state.stats["rejected"] += 1

    state.flush_dashboard(force=True)

    if accepted:
        prev_repo = state.king.get("model_repo") if state.king else ""
        dethrone_block = entry.get("block", 0) or _safe_block(subtensor)
        state.set_king(hotkey, model_repo, dethrone_block,
                       challenge_id=cid, king_digest=challenger_digest)
        state.last_winner_hotkey = hotkey
        try:
            await maybe_set_weights(subtensor, wallet, state,
                                    force=True, reason="dethrone")
        except Exception:
            log.exception("force weight-set after dethrone failed")
        await notify_new_king({
            "hotkey": hotkey,
            "model_repo": model_repo,
            "reign_number": state.king.get("reign_number", 0),
            "king_digest": challenger_digest,
            "previous_repo": prev_repo,
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

    wallet = bt.Wallet(name=WALLET_NAME, hotkey=WALLET_HOTKEY)
    subtensor = bt.Subtensor(network=NETWORK)

    # §9: commit-reveal weights is the single load-bearing defense against
    # parallel-validator weight-copying. If SN3 doesn't have CR enabled,
    # set_weights silently degrades to unencrypted set_mechanism_weights and
    # the design's trust assumption is gone. Refuse to start.
    if not subtensor.commit_reveal_enabled(NETUID):
        log.error("commit-reveal NOT enabled on netuid %d. "
                  "Run sudo_set_commit_reveal_weights_enabled from the subnet-owner "
                  "key before starting this validator.", NETUID)
        sys.exit(2)

    state.refresh_uid_map(subtensor, NETUID)
    state.flush_dashboard(force=True)

    html_path = os.path.join(os.path.dirname(__file__) or ".", "website", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "rb") as f:
            html_bytes = f.read()
        # Stamp a build id derived from the source bytes so long-lived browser
        # tabs can detect a deploy and reload themselves (see checkVersion in
        # website/index.html). The placeholder must be replaced before upload
        # or no version check ever fires.
        build_id = hashlib.sha256(html_bytes).hexdigest()[:12]
        html_bytes = html_bytes.replace(b"__BUILD_ID__", build_id.encode())
        # no-cache forces browsers to revalidate every load via ETag, so a
        # deploy lands on the next refresh instead of after max-age expires.
        r2.put_dashboard_raw(
            "index.html",
            html_bytes,
            "text/html; charset=utf-8",
            cache_control="no-cache, must-revalidate",
        )
        log.info("uploaded dashboard to Hippius (build=%s)", build_id)

    # Was: if should_seed_king(FORCE_SEED_KING, state.king):
    if not state.king:
        if not SEED_DIGEST:
            log.error("set TEUTONIC_SEED_DIGEST for the initial seed king")
            sys.exit(1)
        if state.king and FORCE_SEED_KING:
            log.warning(
                "TEUTONIC_FORCE_SEED_KING enabled: overriding persisted king %s@%s with seed %s@%s",
                state.king.get("model_repo", "?"),
                (state.king.get("king_digest") or "")[:19],
                SEED_REPO,
                SEED_DIGEST[:19],
            )
        seed_ref = ModelRef(SEED_REPO, SEED_DIGEST)
        materialize_model(seed_ref, max_workers=4, config_only=True)
        log.info("seed king %s", seed_ref.immutable_ref)
        state.set_king(wallet.hotkey.ss58_address, seed_ref.repo,
                       subtensor.block, king_digest=seed_ref.digest)

    state.clear_restart_request()
    await maybe_set_weights(subtensor, wallet, state, force=True, reason="startup")

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
             state.king.get("model_repo", "?"),
             state.king.get("king_digest", "?")[:19],
             EVAL_SERVER_URL, POLL_INTERVAL)

    while True:
        tick_started_monotonic = _monotonic_now()
        state.begin_tick()
        try:
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
            reveals = scan_reveals(subtensor, NETUID, state.completed_repos, state.seen)
            if reveals:
                # If any newly-revealed hotkey isn't in the uid_map snapshot we
                # just refreshed (registration happened *between* refresh and
                # this scan, or we're fresh out of startup), refresh once more
                # so the dashboard shows the miner's UID immediately rather
                # than rendering "--" until the next tick (~5-10 min away when
                # an eval is running).
                if any(r["hotkey"] not in state.uid_map for r in reveals):
                    try:
                        state.refresh_uid_map(subtensor, NETUID)
                    except Exception:
                        log.warning("uid_map refresh after scan failed (non-fatal)", exc_info=True)
                queued_count = 0
                for rev in reveals:
                    cid = state.enqueue(rev, defer_flush=True)
                    if cid:
                        queued_count += 1
                        log.info("queued %s from %s (new)", cid, rev["hotkey"][:16])
                if queued_count:
                    state.flush()
                    state.flush_dashboard()

            while state.queue:
                # Per-eval watchdog: reset timer for each queue item so we only
                # restart on a single stuck/hung eval, not on legitimately processing
                # a large queue back-to-back.
                eval_started_monotonic = _monotonic_now()
                entry = state.queue.pop(0)
                state.current_eval = {
                    "challenge_id": entry.get("challenge_id", "?"),
                    "challenger_repo": entry.get("model_repo", ""),
                    "hotkey": entry.get("hotkey", ""),
                    "progress": 0, "total": EVAL_N, "mu_hat": 0,
                    "avg_king_loss": 0, "avg_challenger_loss": 0,
                    "stage": "queued",
                    "stage_started_at": _now(),
                    "stage_elapsed_s": 0,
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
                #
                # CancelledError is special: asyncio.wait_for cancels the inner
                # task when the timer fires, then re-raises as asyncio.TimeoutError
                # *iff* the CancelledError propagates up unmodified. Wrapping it
                # in _EvalInnerError(CancelledError) breaks that path — the outer
                # `except asyncio.TimeoutError` never fires and we fall through
                # to the transient-error retry branch (validator_cancelled), which
                # then burns the full TICK_RESTART_AFTER * MAX_TRANSIENT_EVAL_RETRIES
                # = 30 min × 3 = 90 min on every wall-clocked eval. Observed live
                # 2026-05-13 with eval-0479 burning ~2 hours in 4 retries before
                # being noticed. Re-raise CancelledError directly so
                # asyncio.wait_for can do its job; SIGTERM cancellations during
                # `pm2 restart` also ride this path and propagate cleanly out of
                # the main loop (the entry was already popped from the queue, so
                # losing it on hard-kill matches the previous behavior anyway —
                # the prior `validator_cancelled` retry path didn't actually
                # rescue SIGTERM-cancelled evals because the process exited via
                # sys.exit(0) in the signal handler before the requeue ran).
                async def _bounded_eval():
                    try:
                        await process_challenge(state, r2, entry, subtensor, wallet)
                    except asyncio.CancelledError:
                        raise
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

                fresh = scan_reveals(subtensor, NETUID, state.completed_repos, state.seen)
                if fresh:
                    # See same comment in tick scan above: refresh uid_map
                    # eagerly so the dashboard never shows uid="--" for a
                    # miner that's actually registered on chain right now.
                    if any(r["hotkey"] not in state.uid_map for r in fresh):
                        try:
                            state.refresh_uid_map(subtensor, NETUID)
                        except Exception:
                            log.warning("uid_map refresh after mid-cycle scan failed (non-fatal)", exc_info=True)
                    queued_count = 0
                    for rev in fresh:
                        cid = state.enqueue(rev, defer_flush=True)
                        if cid:
                            queued_count += 1
                            log.info("queued %s from %s (new, mid-cycle)", cid, rev["hotkey"][:16])
                    if queued_count:
                        state.flush()
                        state.flush_dashboard()

                try:
                    await maybe_set_weights(subtensor, wallet, state,
                                            reason="in-queue interval")
                except Exception:
                    log.exception("in-queue weight-set failed")

            state.current_eval = None

            # Re-eval is permanently disabled. Two layered gates enforce
            # 1-hotkey-1-eval in scan_reveals:
            #   1. hotkey -> in `seen`: the primary policy. A miner gets
            #      exactly one shot per hotkey registration, period.
            #   2. model_repo -> in `completed_repos`: belt-and-suspenders to
            #      prevent the "wait until king is weak then re-eval the
            #      same checkpoint" replay even if a hotkey leaks through.
            # Miners who want another shot must register a fresh hotkey
            # on subnet (which costs alpha) and reveal from that new key.
            if not state.queue:
                log.info("idle: all known reveals processed, waiting for new submissions")

            state.complete_tick()
            state.flush_dashboard()

            try:
                await maybe_set_weights(subtensor, wallet, state,
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
            flush_age = _age_seconds(state.watchdog.get("last_state_flush_at"))
            if flush_age is not None and flush_age >= STATE_FLUSH_INTERVAL:
                state.flush()
                state.flush_dashboard()

        await asyncio.sleep(POLL_INTERVAL)


def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    # --seen / --no-seen are retained as no-ops for callers that still pass
    # them. Re-eval is permanently disabled (one eval per model_repo, ever); the
    # flag has no effect either way.
    p.add_argument("--seen", action=argparse.BooleanOptionalAction, default=True,
                   help=argparse.SUPPRESS)
    return p.parse_args()


def main_sync():
    asyncio.run(main())


if __name__ == "__main__":
    main_sync()
