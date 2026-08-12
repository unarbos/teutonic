#!/usr/bin/env python3
"""Teutonic validator — king-of-the-hill evaluation coordinator.

Polls Bittensor chain for challenger submissions, dispatches evaluations
to the GPU eval service, manages king
lifecycle on Hippius Hub, persists all state to R2.
"""
import asyncio
import hashlib
import json
import logging
import os
import signal
import shutil
import sys
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import bittensor as bt
import boto3
import httpx
from botocore.config import Config as BotoConfig

_repo_root = os.path.dirname(os.path.abspath(__file__))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Register the architecture selected by chain.toml without trust_remote_code.
import chain_config  # noqa: E402
from model_store import (  # noqa: E402
    DIGEST_RE,
    ModelRef,
    list_remote_files,
    materialize_model,
    parse_reveal_v4,
    parse_reveal_v3,
    snapshot_size,
    _resolve_hub_token,
)

chain_config.load_arch()

EVAL_N = int(os.environ.get("TEUTONIC_EVAL_N", "10000"))
EVAL_ALPHA = 0.001
EVAL_DELTA_THRESHOLD = 0.0015
EVAL_BOOTSTRAP_SAMPLES = 10_000
SEQ_LEN = 2048
POLL_INTERVAL = 30
WEIGHT_INTERVAL = 300
NETUID = int(os.environ.get("TEUTONIC_NETUID", "3"))
MIN_SUBMISSION_BLOCK = int(os.environ.get("TEUTONIC_MIN_SUBMISSION_BLOCK", "8377970"))

BURN_UID = int(os.environ.get("TEUTONIC_BURN_UID", "0"))

TICK_WARN_AFTER = int(os.environ.get("TEUTONIC_TICK_WARN_AFTER", "120"))
TICK_RESTART_AFTER = int(os.environ.get("TEUTONIC_TICK_RESTART_AFTER", "2700"))
STREAM_IDLE_WARN_AFTER = int(os.environ.get("TEUTONIC_STREAM_IDLE_WARN_AFTER", "180"))
STREAM_IDLE_TIMEOUT = int(os.environ.get("TEUTONIC_STREAM_IDLE_TIMEOUT", "420"))
HEALTHCHECK_INTERVAL = int(os.environ.get("TEUTONIC_HEALTHCHECK_INTERVAL", "60"))
STATE_FLUSH_INTERVAL = int(os.environ.get("TEUTONIC_STATE_FLUSH_INTERVAL", "60"))
MAX_CONSECUTIVE_TICK_ERRORS = int(os.environ.get("TEUTONIC_MAX_CONSECUTIVE_TICK_ERRORS", "10"))
NETWORK = os.environ.get("TEUTONIC_NETWORK", "finney")
SEED_REPO = os.environ.get("TEUTONIC_SEED_REPO", chain_config.SEED_REPO)
SEED_DIGEST = os.environ.get("TEUTONIC_SEED_DIGEST", getattr(chain_config, "SEED_DIGEST", ""))
EVAL_SERVER_URL = os.environ.get("TEUTONIC_EVAL_SERVER", "http://localhost:9000")
WALLET_NAME = os.environ.get("BT_WALLET_NAME", "teutonic")
WALLET_HOTKEY = os.environ.get("BT_WALLET_HOTKEY", "default")

R2_ENDPOINT = os.environ.get("TEUTONIC_R2_ENDPOINT", "")
R2_BUCKET = os.environ.get("TEUTONIC_R2_BUCKET", "")
R2_ACCESS_KEY = os.environ.get("TEUTONIC_R2_ACCESS_KEY", "")
R2_SECRET_KEY = os.environ.get("TEUTONIC_R2_SECRET_KEY", "")
R2_DRY_RUN = os.environ.get("TEUTONIC_R2_DRY_RUN", "").lower() in ("1", "true", "yes", "on")
SIDE_EFFECT_DRY_RUN = os.environ.get("TEUTONIC_SIDE_EFFECT_DRY_RUN", "").lower() in ("1", "true", "yes", "on")

HIPPIUS_ENDPOINT = os.environ.get("TEUTONIC_HIPPIUS_ENDPOINT", "https://s3.hippius.com")
HIPPIUS_BUCKET = os.environ.get("TEUTONIC_HIPPIUS_BUCKET", "teutonic-sn3")
HIPPIUS_ACCESS_KEY = os.environ.get("TEUTONIC_HIPPIUS_ACCESS_KEY", "")
HIPPIUS_SECRET_KEY = os.environ.get("TEUTONIC_HIPPIUS_SECRET_KEY", "")

TMC_API_KEY = os.environ.get("TMC_API_KEY", "")

DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "")
DISCORD_CHANNEL_ID = os.environ.get("DISCORD_CHANNEL_ID", "")

# Miner repositories must contain a token derived from the owner's coldkey.
COLDKEY_PREFIX_LEN = int(os.environ.get("TEUTONIC_COLDKEY_PREFIX_LEN", "5"))
COLDKEY_SUFFIX_LEN = int(os.environ.get("TEUTONIC_COLDKEY_SUFFIX_LEN", "5"))

# Quasar permits only these hash-matched local model files.
CUSTOM_CODE_POLICY = os.environ.get("TEUTONIC_CUSTOM_CODE_POLICY", "").strip().lower()
QUASAR_CODE_POLICY_ENV = os.environ.get("TEUTONIC_ALLOW_QUASAR_CUSTOM_CODE", "").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
QUASAR_ALLOWED_CODE_FILES = {
    "configuration_qwen3_5.py",
    "modeling_qwen3_5.py",
}
QUASAR_EXPECTED_AUTO_MAP = {
    "AutoConfig": "configuration_qwen3_5.QuasarConfig",
    "AutoModelForCausalLM": "modeling_qwen3_5.QuasarForCausalLM",
}

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


BLOCKS_PER_HOUR = 300

DASHBOARD_FLUSH_MIN_INTERVAL = float(os.environ.get("TEUTONIC_DASHBOARD_FLUSH_MIN_INTERVAL", "5"))
HIPPIUS_COOLDOWN_SECONDS = int(os.environ.get("TEUTONIC_HIPPIUS_COOLDOWN_SECONDS", "300"))
S3_CONNECT_TIMEOUT = int(os.environ.get("TEUTONIC_S3_CONNECT_TIMEOUT", "5"))
S3_READ_TIMEOUT = int(os.environ.get("TEUTONIC_S3_READ_TIMEOUT", "15"))
S3_MAX_ATTEMPTS = int(os.environ.get("TEUTONIC_S3_MAX_ATTEMPTS", "3"))

KING_CHAIN_SIZE = int(os.environ.get("TEUTONIC_KING_CHAIN_SIZE", "5"))

MAX_TRANSIENT_EVAL_RETRIES = int(os.environ.get("TEUTONIC_MAX_TRANSIENT_EVAL_RETRIES", "3"))

async def fetch_tmc_data() -> dict | None:
    """Fetch TAO price and SN3 alpha price from TMC public API."""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
            market_resp, subnet_resp = await asyncio.gather(
                client.get(f"{TMC_BASE}/market/market-data/"),
                client.get(f"{TMC_BASE}/subnets/{NETUID}/"),
            )
        m = market_resp.json()
        s = subnet_resp.json()
        snap = s["latest_snapshot"]
        asp = float(snap["alpha_sqrt_price"])
        tao_price = m["current_price"]
        alpha_tao = asp ** 2
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
            "sn3_alpha_per_block": sn3_alpha_per_block,
            "sn3_miner_share": miner_share,
            "sn3_alpha_per_block_gross": gross_apb,
        }
    except Exception:
        log.warning("TMC fetch failed", exc_info=True)
        return None

async def notify_new_king(king_info: dict, verdict: dict | None = None):
    """Post a message to Discord when a new king is crowned."""
    if SIDE_EFFECT_DRY_RUN:
        log.info("side-effect dry-run: skipping Discord new-king notification")
        return
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
    if verdict and verdict.get("verdict") == "crown_earlier_commit":
        chall_ts  = verdict.get("challenger_committed_at") or "?"
        king_ts   = verdict.get("king_committed_at") or "?"
        chall_src = verdict.get("challenger_timestamp_source") or "?"
        king_src  = verdict.get("king_timestamp_source") or "?"
        lines.append(f"**Method:** identical weights — earlier upload displaced copy (no eval)")
        lines.append(f"**Challenger upload:** `{chall_ts}` via `{chall_src}`")
        lines.append(f"**King upload:** `{king_ts}` via `{king_src}`")
    elif verdict:
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


class R2:
    def __init__(self):
        _s3_cfg = dict(
            connect_timeout=S3_CONNECT_TIMEOUT,
            read_timeout=S3_READ_TIMEOUT,
            retries={"max_attempts": S3_MAX_ATTEMPTS, "mode": "standard"},
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
        if R2_DRY_RUN:
            log.info("R2 dry-run: skip dashboard put %s (%d bytes)", key, len(body))
            return
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
        if R2_DRY_RUN:
            try:
                size = len(json.dumps(data, default=str).encode())
            except Exception:
                size = -1
            log.info("R2 dry-run: skip put %s (%d bytes)", key, size)
            return
        try:
            self.client.put_object(
                Bucket=R2_BUCKET, Key=key,
                Body=json.dumps(data, default=str).encode(),
                ContentType="application/json",
            )
        except Exception:
            log.warning("R2 put failed for %s (non-fatal)", key)

    def get(self, key):
        started = time.monotonic()
        log.info("R2 get start: %s", key)
        try:
            body = self.client.get_object(Bucket=R2_BUCKET, Key=key)["Body"].read()
            data = json.loads(body)
            log.info("R2 get ok: %s (%.1fs, %d bytes)", key, time.monotonic() - started, len(body))
            return data
        except Exception as exc:
            log.warning("R2 get failed: %s (%.1fs): %s", key, time.monotonic() - started, exc)
            return None

_king_config: dict | None = None
_king_config_key: str | None = None
_code_hash_cache: dict[tuple[str, str, tuple[str, ...]], dict[str, str]] = {}


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


def _quasar_custom_code_allowed(king_cfg: dict, challenger_cfg: dict) -> bool:
    if QUASAR_CODE_POLICY_ENV or CUSTOM_CODE_POLICY in {"quasar", "quasar_qwen3_5"}:
        return True
    if chain_config.ARCH_MODULE.endswith(".quasar"):
        return True
    return (
        king_cfg.get("model_type") == "quasar_text"
        or challenger_cfg.get("model_type") == "quasar_text"
    )


def _validate_quasar_auto_map(auto_map: dict | None) -> str | None:
    if not isinstance(auto_map, dict):
        return "quasar config must provide auto_map"
    if set(auto_map) != set(QUASAR_EXPECTED_AUTO_MAP):
        return (
            "quasar auto_map keys mismatch: "
            f"expected={sorted(QUASAR_EXPECTED_AUTO_MAP)} got={sorted(auto_map)}"
        )
    for key, expected in QUASAR_EXPECTED_AUTO_MAP.items():
        value = auto_map.get(key)
        if value != expected:
            return f"quasar auto_map[{key!r}] mismatch: expected={expected!r} got={value!r}"
        module = expected.rsplit(".", 1)[0]
        if "--" in module or "/" in module:
            return f"quasar auto_map[{key!r}] must be local, got {value!r}"
    return None


def _code_cache_dir(ref: ModelRef, files: set[str]) -> Path:
    digest = (ref.digest or "latest").replace(":", "-")
    material = f"{ref.repo}@{digest}|{','.join(sorted(files))}".encode()
    suffix = hashlib.sha256(material).hexdigest()[:16]
    return Path(os.environ.get("TEUTONIC_CODE_CACHE_DIR", "/tmp/teutonic/validator_code")) / (
        ref.repo.replace("/", "--") + "--" + digest + "--" + suffix
    )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _download_code_files(ref: ModelRef, files: set[str]) -> Path:
    target = _code_cache_dir(ref, files)
    if all((target / name).exists() for name in files):
        return target
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    allow_patterns = sorted(files)

    if ref.digest.startswith("hf:"):
        from huggingface_hub import snapshot_download as hf_snapshot_download

        hf_snapshot_download(
            repo_id=ref.repo,
            revision=ref.digest[3:],
            local_dir=str(target),
            allow_patterns=allow_patterns,
            max_workers=4,
            token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY"),
        )
        return target

    from hippius_hub import snapshot_download

    snapshot_download(
        repo_id=ref.repo,
        revision=ref.digest,
        local_dir=str(target),
        allow_patterns=allow_patterns,
        max_workers=4,
        token=_resolve_hub_token(f"Downloading code files for {ref.immutable_ref}"),
    )
    return target


def _remote_code_hashes(ref: ModelRef, files: set[str]) -> dict[str, str]:
    key = (ref.repo, ref.digest or "", tuple(sorted(files)))
    if key in _code_hash_cache:
        return dict(_code_hash_cache[key])
    root = _download_code_files(ref, files)
    missing = sorted(name for name in files if not (root / name).exists())
    if missing:
        raise FileNotFoundError(f"{ref.immutable_ref} missing code files: {missing}")
    hashes = {name: _sha256_file(root / name) for name in sorted(files)}
    _code_hash_cache[key] = hashes
    return dict(hashes)


def validate_custom_code_policy(
    *,
    model_ref: ModelRef,
    challenger_cfg: dict,
    repo_files: list[str],
    king_repo: str,
    king_digest: str,
    king_cfg: dict,
) -> str | None:
    auto_map = challenger_cfg.get("auto_map")
    py_files = sorted(f for f in repo_files if f.endswith(".py"))

    if not _quasar_custom_code_allowed(king_cfg, challenger_cfg):
        if auto_map:
            return "auto_map present in config.json (custom modeling code is not allowed)"
        if py_files:
            return f"repo ships *.py files (not allowed): {py_files[:3]}"
        return None

    unexpected_py = sorted(set(py_files) - QUASAR_ALLOWED_CODE_FILES)
    if unexpected_py:
        return f"repo ships non-Quasar *.py files (not allowed): {unexpected_py[:3]}"

    if auto_map:
        rejection = _validate_quasar_auto_map(auto_map)
        if rejection:
            return rejection
        required = set(QUASAR_ALLOWED_CODE_FILES)
        missing_py = sorted(required - set(py_files))
        if missing_py:
            return f"quasar auto_map requires missing code files: {missing_py}"

        try:
            king_ref = ModelRef(king_repo or SEED_REPO, king_digest or SEED_DIGEST)
            king_hashes = _remote_code_hashes(king_ref, required)
            challenger_hashes = _remote_code_hashes(model_ref, required)
        except Exception as exc:
            return f"could not verify Quasar custom code hashes: {exc}"

        mismatches = [
            name
            for name in sorted(required)
            if challenger_hashes.get(name) != king_hashes.get(name)
        ]
        if mismatches:
            details = {
                name: {
                    "king": king_hashes.get(name, "")[:16],
                    "challenger": challenger_hashes.get(name, "")[:16],
                }
                for name in mismatches
            }
            return f"quasar custom code hash mismatch: {details}"
        return None

    if py_files:
        return "quasar *.py files are allowed only with the exact approved auto_map"
    return None


_COPY_CHECK_SAMPLE_N     = int(os.environ.get("TEUTONIC_COPY_CHECK_SAMPLE_N", "12"))
_COPY_CHECK_SAMPLE_BYTES = int(os.environ.get("TEUTONIC_COPY_CHECK_SAMPLE_BYTES", "65536"))


def _parse_registry_timestamp(ts: str | None):
    if not ts:
        return None
    if isinstance(ts, datetime):
        dt = ts
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        try:
            dt = parsedate_to_datetime(ts)
        except Exception:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _format_registry_timestamp(dt) -> str | None:
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).isoformat()


def _first_registry_timestamp(candidates: list[tuple[str, str | None]]) -> tuple[str | None, str | None]:
    for source, value in candidates:
        dt = _parse_registry_timestamp(value)
        if dt is not None:
            return _format_registry_timestamp(dt), source
    return None, None


def _normal_blob_digest(value: str | None) -> str:
    return (value or "").lower().removeprefix("sha256:")


def _trusted_timestamp_source(source: str | None) -> bool:
    return bool(source) and not str(source).startswith("untrusted:")


def _copy_info_from_siblings(siblings, timestamp_candidates: list[tuple[str, object]]) -> dict | None:
    safetensor_layers: dict[str, str] = {}
    safetensor_dates = []
    for item in siblings or []:
        title = getattr(item, "path", "") or getattr(item, "rfilename", "")
        if not title.endswith(".safetensors"):
            continue
        lfs = getattr(item, "lfs", None)
        digest = getattr(lfs, "sha256", None) or getattr(item, "blob_id", None)
        if digest:
            safetensor_layers[title] = _normal_blob_digest(digest)
        last_commit = getattr(item, "last_commit", None)
        dt = _parse_registry_timestamp(getattr(last_commit, "date", None))
        if dt is not None:
            safetensor_dates.append(dt)

    if safetensor_dates:
        timestamp_candidates.insert(0, ("untrusted:huggingface_safetensor.last_commit", max(safetensor_dates)))
    committed_at, timestamp_source = _first_registry_timestamp(timestamp_candidates)
    return {
        "safetensor_layers": safetensor_layers,
        "committed_at": committed_at,
        "timestamp_source": timestamp_source,
    }


def _fetch_hf_model_info(repo_id: str, hf_digest: str) -> dict | None:
    try:
        from huggingface_hub import HfApi

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")
        api = HfApi(token=token)
        revision = hf_digest[3:]
        siblings = list(api.list_repo_tree(repo_id, recursive=True, expand=True, revision=revision))
        result = _copy_info_from_siblings(siblings, [])
        result.update({
            "_hf_repo": repo_id,
            "_hf_revision": revision,
            "_hf_token": token,
        })
        return result
    except Exception:
        log.debug("could not fetch HF info for %s@%s (copy check skipped)",
                  repo_id, hf_digest[:19], exc_info=True)
        return None


def _fetch_hippius_model_info(repo_id: str, oci_digest: str) -> dict | None:
    try:
        import hippius_hub

        token = _resolve_hub_token(f"copy-check model info {repo_id}")
        for revision in (oci_digest, "main"):
            try:
                info = hippius_hub.model_info(
                    repo_id,
                    revision=revision,
                    files_metadata=True,
                    token=token,
                )
                break
            except Exception:
                if revision != "main":
                    continue
                raise
        if _normal_blob_digest(getattr(info, "sha", "")) != _normal_blob_digest(oci_digest):
            return None

        timestamp_candidates = [
            ("hippius_model.created_at", getattr(info, "created_at", None)),
            ("hippius_model.last_modified", getattr(info, "last_modified", None)),
        ]
        return _copy_info_from_siblings(getattr(info, "siblings", None), timestamp_candidates)
    except Exception:
        log.debug("could not fetch Hippius model info for %s@%s (copy check skipped)",
                  repo_id, oci_digest[:19], exc_info=True)
        return None


def _fetch_model_oci_info(repo_id: str, oci_digest: str) -> dict | None:
    """Fetch model file digests and server-observed timestamp metadata.

    Returns None when the check cannot be performed, so callers can fail open
    rather than blocking valid submissions.

    Return shape::

        {
            "safetensor_layers": {"model-00001-of-00004.safetensors": "...", ...},
            "committed_at": "2026-06-08T15:45:59.489795+00:00",   # may be None
            "timestamp_source": "harbor_artifact.push_time",      # may be None
        }
    """
    if oci_digest.startswith("hf:"):
        return _fetch_hf_model_info(repo_id, oci_digest)
    try:
        from hippius_hub._harbor import harbor_get_artifact, split_repo_id
        from hippius_hub._oci import manifest_url, oci_headers
        from hippius_hub.auth import (
            get_oci_bearer_token,
            resolve_auth_header,
            resolve_token_value,
        )
        from hippius_hub.constants import resolve_registry
        from hippius_hub.file_download import _oci_repo_path

        registry = resolve_registry(None)
        oci_repo = _oci_repo_path(repo_id, None)
        raw_token = _resolve_hub_token(f"copy-check manifest {repo_id}")
        oci_token = get_oci_bearer_token(oci_repo, resolve_token_value(raw_token), push=False)

        resp = httpx.get(
            manifest_url(registry, oci_repo, oci_digest),
            headers=oci_headers(oci_token),
            timeout=httpx.Timeout(15.0),
        )
        if resp.status_code == 404:
            return _fetch_hippius_model_info(repo_id, oci_digest)
        resp.raise_for_status()
        manifest = resp.json()

        safetensor_layers: dict[str, str] = {}
        index_json_digest: str | None = None
        for layer in manifest.get("layers", []):
            title = layer.get("annotations", {}).get("org.opencontainers.image.title", "")
            if title.endswith(".safetensors") and "digest" in layer:
                safetensor_layers[title] = _normal_blob_digest(layer["digest"])
            elif title == "model.safetensors.index.json" and "digest" in layer:
                index_json_digest = layer["digest"]  # full "sha256:..." form for blob URL

        artifact = None
        auth_header = resolve_auth_header(raw_token)
        if auth_header:
            try:
                project, repo = split_repo_id(oci_repo)
                artifact = harbor_get_artifact(
                    auth_header,
                    project,
                    repo,
                    oci_digest,
                    endpoint=None,
                )
            except Exception:
                log.debug("could not fetch Harbor artifact metadata for %s@%s",
                          repo_id, oci_digest[:19], exc_info=True)

        timestamp_candidates: list[tuple[str, str | None]] = []
        if isinstance(artifact, dict):
            timestamp_candidates.append(("harbor_artifact.push_time", artifact.get("push_time")))
        timestamp_candidates.append(("manifest_last_modified", resp.headers.get("Last-Modified")))

        committed_at, timestamp_source = _first_registry_timestamp(timestamp_candidates)
        return {
            "safetensor_layers": safetensor_layers,
            "committed_at": committed_at,
            "timestamp_source": timestamp_source,
            "index_json_digest": index_json_digest,
            "_registry": registry,
            "_oci_repo": oci_repo,
            "_oci_token": oci_token,
        }
    except Exception:
        log.debug("could not fetch OCI info for %s@%s (copy check skipped)",
                  repo_id, oci_digest[:19], exc_info=True)
        return _fetch_hippius_model_info(repo_id, oci_digest)


def _fetch_weight_map(info: dict) -> dict[str, str] | None:
    """Return {tensor_name: shard_file} from model.safetensors.index.json.

    Works for both Hippius (OCI blob) and HuggingFace (resolve URL).
    """
    try:
        if "_registry" in info:
            idx_digest = info.get("index_json_digest")
            if not idx_digest:
                return None
            r = httpx.get(
                f"{info['_registry']}/v2/{info['_oci_repo']}/blobs/{idx_digest}",
                headers={"Authorization": f"Bearer {info['_oci_token']}"},
                timeout=30.0,
                follow_redirects=True,
            )
        elif "_hf_repo" in info:
            repo     = info["_hf_repo"]
            revision = info["_hf_revision"]
            token    = info.get("_hf_token")
            hdrs     = {"Authorization": f"Bearer {token}"} if token else {}
            r = httpx.get(
                f"https://huggingface.co/{repo}/resolve/{revision}/model.safetensors.index.json",
                headers=hdrs, timeout=30.0, follow_redirects=True,
            )
        else:
            return None
        if r.status_code != 200:
            return None
        return r.json().get("weight_map", {})
    except Exception:
        log.debug("_fetch_weight_map failed", exc_info=True)
        return None


def _fetch_tensor_fingerprint(
    info: dict,
    tensor_names: list[str],
    sample_bytes: int = _COPY_CHECK_SAMPLE_BYTES,
    *,
    _weight_map: dict[str, str] | None = None,
) -> dict[str, str] | None:
    """Download `sample_bytes` from each named tensor and return {name: sha256_hex}.

    Handles both Hippius OCI (via registry blob Range requests) and HuggingFace
    (via resolve URL Range requests). `_weight_map` can be passed from a
    pre-fetched index.json to avoid a redundant network round-trip.
    Returns None on failure so callers fail open.
    """
    import struct
    import concurrent.futures

    is_hippius = "_registry" in info
    is_hf      = "_hf_repo" in info
    if not is_hippius and not is_hf:
        return None

    client: httpx.Client | None = None
    try:
        client = httpx.Client(timeout=httpx.Timeout(30.0))

        def _range_get(url: str, headers: dict, start: int, end: int) -> bytes | None:
            r = client.get(url, headers={**headers, "Range": f"bytes={start}-{end}"},
                           follow_redirects=True)
            return r.content if r.status_code in (200, 206) else None

        if is_hippius:
            registry  = info["_registry"]
            oci_repo  = info["_oci_repo"]
            oci_token = info["_oci_token"]
            auth      = {"Authorization": f"Bearer {oci_token}"}

            def _blob_url(shard_file: str) -> str | None:
                d = info["safetensor_layers"].get(shard_file, "")
                if not d:
                    return None
                blob = d if d.startswith("sha256:") else f"sha256:{d}"
                return f"{registry}/v2/{oci_repo}/blobs/{blob}"

        else:  # HuggingFace
            repo     = info["_hf_repo"]
            revision = info["_hf_revision"]
            token    = info.get("_hf_token")
            auth     = {"Authorization": f"Bearer {token}"} if token else {}
            base     = f"https://huggingface.co/{repo}/resolve/{revision}"

            def _blob_url(shard_file: str) -> str | None:
                return f"{base}/{shard_file}"

        weight_map = _weight_map if _weight_map is not None else _fetch_weight_map(info)
        if not weight_map:
            return None

        shard_to_tensors: dict[str, list[str]] = {}
        for tname in tensor_names:
            shard = weight_map.get(tname)
            if shard:
                shard_to_tensors.setdefault(shard, []).append(tname)

        def _fetch_shard_meta(shard_file: str):
            url = _blob_url(shard_file)
            if not url:
                return None
            hdr8 = _range_get(url, auth, 0, 7)
            if not hdr8 or len(hdr8) < 8:
                return None
            hdr_len = struct.unpack_from("<Q", hdr8, 0)[0]
            hdr_raw = _range_get(url, auth, 8, 7 + hdr_len)
            if not hdr_raw:
                return None
            tensors = {
                k: v for k, v in json.loads(hdr_raw[:hdr_len]).items()
                if k != "__metadata__"
            }
            return shard_file, {"url": url, "hdr_len": hdr_len, "tensors": tensors}

        shard_meta: dict[str, dict] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            for result in pool.map(_fetch_shard_meta, list(shard_to_tensors)):
                if result:
                    shard_meta[result[0]] = result[1]

        def _fetch_one_tensor(args: tuple) -> tuple[str, str] | None:
            tname, shard_file = args
            meta  = shard_meta.get(shard_file)
            if not meta:
                return None
            tinfo = meta["tensors"].get(tname)
            if not tinfo:
                return None
            offsets   = tinfo["data_offsets"]
            abs_start = 8 + meta["hdr_len"] + offsets[0]
            abs_end   = 8 + meta["hdr_len"] + min(offsets[1], offsets[0] + sample_bytes) - 1
            data = _range_get(meta["url"], auth, abs_start, abs_end)
            if data is None:
                return None
            return tname, hashlib.sha256(data).hexdigest()

        work = [(t, sf) for sf, tensors in shard_to_tensors.items() for t in tensors]
        fingerprints: dict[str, str] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            for result in pool.map(_fetch_one_tensor, work):
                if result:
                    fingerprints[result[0]] = result[1]

        return fingerprints or None

    except Exception:
        log.debug("tensor fingerprint check failed", exc_info=True)
        return None
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass


def _check_tensor_copy(
    challenger_info: dict,
    king_info: dict,
    challenger_digest: str,
    king_digest: str,
) -> dict | None:
    """Tensor-level copy check for models whose shard count differs.

    Samples _COPY_CHECK_SAMPLE_N tensors deterministically, downloads
    _COPY_CHECK_SAMPLE_BYTES from each, and compares sha256 fingerprints.
    Returns None when models differ or the check cannot be completed (fail-open).
    """
    import random as _random

    king_has_ctx = "_registry" in king_info or "_hf_repo" in king_info
    chall_has_ctx = "_registry" in challenger_info or "_hf_repo" in challenger_info
    if not king_has_ctx or not chall_has_ctx:
        log.debug("tensor copy check skipped: auth context missing on one or both models")
        return None

    king_weight_map = _fetch_weight_map(king_info)
    if not king_weight_map:
        return None
    all_tensors = sorted(king_weight_map.keys())
    if not all_tensors:
        return None

    seed = int(hashlib.blake2b(
        (challenger_digest + king_digest).encode(), digest_size=8,
    ).hexdigest(), 16)
    rng     = _random.Random(seed)
    sample_n = min(_COPY_CHECK_SAMPLE_N, len(all_tensors))
    sampled  = rng.sample(all_tensors, sample_n)

    log.info(
        "tensor copy check: sampling %d/%d tensors @ %d B each",
        sample_n, len(all_tensors), _COPY_CHECK_SAMPLE_BYTES,
    )

    king_fp  = _fetch_tensor_fingerprint(king_info,       sampled, _weight_map=king_weight_map)
    chall_fp = _fetch_tensor_fingerprint(challenger_info, sampled)

    if not chall_fp or not king_fp:
        log.debug("tensor copy check: fingerprint fetch failed on at least one model")
        return None

    common = set(chall_fp) & set(king_fp)
    if len(common) < max(1, sample_n // 2):
        log.debug(
            "tensor copy check: only %d/%d tensors fingerprinted — cannot conclude",
            len(common), sample_n,
        )
        return None

    mismatches = [t for t in common if chall_fp[t] != king_fp[t]]
    if mismatches:
        log.debug(
            "tensor copy check: %d/%d tensors differ → not a copy",
            len(mismatches), len(common),
        )
        return None

    n_chall_shards   = len(challenger_info["safetensor_layers"])
    n_king_shards    = len(king_info["safetensor_layers"])
    challenger_ts    = challenger_info.get("committed_at")
    king_ts          = king_info.get("committed_at")
    challenger_source = challenger_info.get("timestamp_source")
    king_source       = king_info.get("timestamp_source")

    challenger_dt = _parse_registry_timestamp(challenger_ts)
    king_dt       = _parse_registry_timestamp(king_ts)

    base_reason = (
        f"all {len(common)} sampled tensors have identical byte content "
        f"(shard layout differs: {n_chall_shards} vs {n_king_shards} shards); "
        f"challenger pushed_at={challenger_ts} source={challenger_source}, "
        f"king pushed_at={king_ts} source={king_source}"
    )
    result_meta = {
        "challenger_committed_at": challenger_ts,
        "king_committed_at": king_ts,
        "challenger_timestamp_source": challenger_source,
        "king_timestamp_source": king_source,
    }

    if (
        challenger_dt is None
        or king_dt is None
        or not _trusted_timestamp_source(challenger_source)
    ):
        return {
            "action": "reject",
            "reason": (
                f"model is a tensor-level copy (trusted challenger timestamp unavailable): "
                f"{base_reason}"
            ),
            **result_meta,
        }

    if challenger_dt < king_dt:
        return {
            "action": "crown_earlier",
            "reason": (
                f"model is tensor-identical to the king but has an earlier registry-observed "
                f"push time ({challenger_ts} < {king_ts}); displacing king with original author. "
                f"{base_reason}"
            ),
            **result_meta,
        }

    return {
        "action": "reject",
        "reason": f"model is a tensor-level copy (different shard layout, not earlier): {base_reason}",
        **result_meta,
    }


def check_model_copy(
    challenger_repo: str,
    challenger_digest: str,
    king_repo: str,
    king_digest: str,
) -> dict | None:
    """Check whether the challenger is a weight-for-weight copy of the king.

    Returns None when models differ or the check cannot be performed.

    When an exact copy is detected, returns a dict with an ``action`` key:

    * ``"reject"`` — challenger committed *after* the king; it is a copy and
      should be rejected.
    * ``"crown_earlier"`` — challenger was committed *before* the king; the
      challenger is the original and should displace the king without an eval.

    The dict also carries ``reason``, ``challenger_committed_at``, and
    ``king_committed_at`` for logging / dashboard display.
    """
    if not king_repo or not king_digest:
        return None
    if challenger_repo == king_repo and challenger_digest == king_digest:
        return {
            "action": "reject",
            "reason": (
                f"challenger is identical to the current king "
                f"(same repo {challenger_repo!r} and digest {challenger_digest[:19]})"
            ),
            "challenger_committed_at": None,
            "king_committed_at": None,
        }

    challenger_info = _fetch_model_oci_info(challenger_repo, challenger_digest)
    if not challenger_info:
        return None

    king_info = _fetch_model_oci_info(king_repo, king_digest)
    if not king_info:
        return None

    challenger_layers = challenger_info["safetensor_layers"]
    king_layers = king_info["safetensor_layers"]

    if not challenger_layers or len(challenger_layers) != len(king_layers):
        return _check_tensor_copy(
            challenger_info, king_info, challenger_digest, king_digest,
        )

    mismatches = [
        title for title, digest in challenger_layers.items()
        if king_layers.get(title) != digest
    ]
    if mismatches:
        return None

    n = len(challenger_layers)
    challenger_ts = challenger_info.get("committed_at")
    king_ts = king_info.get("committed_at")
    challenger_source = challenger_info.get("timestamp_source")
    king_source = king_info.get("timestamp_source")

    challenger_dt = _parse_registry_timestamp(challenger_ts)
    king_dt = _parse_registry_timestamp(king_ts)

    base_reason = (
        f"all {n} .safetensors layers have identical blob digests; "
        f"challenger pushed_at={challenger_ts} source={challenger_source}, "
        f"king pushed_at={king_ts} source={king_source}"
    )

    result_meta = {
        "challenger_committed_at": challenger_ts,
        "king_committed_at": king_ts,
        "challenger_timestamp_source": challenger_source,
        "king_timestamp_source": king_source,
    }

    if (
        challenger_dt is None
        or king_dt is None
        or not _trusted_timestamp_source(challenger_source)
    ):
        return {
            "action": "reject",
            "reason": f"model is a copy of the king (trusted challenger timestamp unavailable): {base_reason}",
            **result_meta,
        }

    if challenger_dt < king_dt:
        return {
            "action": "crown_earlier",
            "reason": (
                f"model is identical to the king but has an earlier registry-observed push time "
                f"({challenger_ts} < {king_ts}); displacing king with original author. "
                f"{base_reason}"
            ),
            **result_meta,
        }

    return {
        "action": "reject",
        "reason": f"model is a copy of the king (not earlier than king): {base_reason}",
        **result_meta,
    }


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
        snapshot = materialize_model(ref, max_workers=4, config_only=True)
        with open(os.path.join(snapshot, "config.json")) as f:
            challenger_cfg = json.load(f)
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
    _SENTINEL = object()
    for key in _generic_lock + chain_config.EXTRA_LOCK_KEYS:
        king_val = king_cfg.get(key, _SENTINEL)
        chall_val = challenger_cfg.get(key, _SENTINEL)
        if king_val != chall_val:
            return f"{key} mismatch: king={king_val if king_val is not _SENTINEL else '<absent>'} challenger={chall_val if chall_val is not _SENTINEL else '<absent>'}"

    custom_code_rejection = validate_custom_code_policy(
        model_ref=ref,
        challenger_cfg=challenger_cfg,
        repo_files=repo_files,
        king_repo=king_repo,
        king_digest=king_digest,
        king_cfg=king_cfg,
    )
    if custom_code_rejection:
        return custom_code_rejection

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

import re
_SAFETENSORS_SHARD_RE = re.compile(r"^model-\d{5}-of-\d{5}\.safetensors$")


def _verdict_shards_used(verdict: dict) -> list[dict]:
    shards = verdict.get("shards_used")
    if shards:
        return shards
    dataset = verdict.get("dataset") if isinstance(verdict.get("dataset"), dict) else {}
    return dataset.get("shards_used") or []


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


def _resolve_chain_coldkey(subtensor, hotkey: str, reveal_block: int = 0) -> str | None:
    attempts = [(None, "chain head")]
    if reveal_block > 0:
        attempts.append((reveal_block, f"reveal block {reveal_block}"))

    for block, label in attempts:
        try:
            owner = subtensor.get_hotkey_owner(hotkey, block=block)
        except Exception:
            log.warning("coldkey lookup for %s failed at %s", hotkey[:16], label,
                        exc_info=True)
            continue
        if owner:
            return str(owner)
    return None


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
        if int(block or 0) <= MIN_SUBMISSION_BLOCK:
            continue
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
            log.warning("v4 author_hotkey %s mismatches chain key %s; trusting chain",
                        author_hotkey[:16], hotkey[:16])
        coldkey = _resolve_chain_coldkey(subtensor, hotkey, int(block or 0))
        if not coldkey:
            log.warning("skipping reveal from %s at block %s: chain owner unavailable",
                        hotkey[:16], block)
            continue
        new.append({
            "hotkey": hotkey,
            "coldkey": coldkey,
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
    if SIDE_EFFECT_DRY_RUN:
        log.info("side-effect dry-run: skipping set_weights (%s)", reason or "no reason")
        return False
    try:
        current_block = subtensor.block
    except Exception:
        log.exception("failed to read current block for weight-set")
        return False
    if not force and current_block - state.last_weight_block < WEIGHT_INTERVAL:
        return False

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
        # Bittensor reports rate limiting as an empty failure.
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


def _completed_digests(completed_repos: set[str]) -> set[str]:
    return {item.split("@", 1)[1] for item in completed_repos if "@" in item}


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
        block = int(reveal.get("block", 0) or 0)
        if block <= MIN_SUBMISSION_BLOCK:
            log.info("skipping enqueue: submission from %s at block %s is not over %s",
                     hotkey[:16], block, MIN_SUBMISSION_BLOCK)
            return None
        king_hotkey = self.king.get("hotkey", "")
        if king_hotkey and hotkey == king_hotkey:
            log.info("skipping enqueue: hotkey %s is the current king", hotkey[:16])
            return None
        # Each registered hotkey gets one evaluation.
        if hotkey and hotkey in self.seen:
            log.info("skipping enqueue: hotkey %s already used its 1-eval slot "
                     "(must re-register for another shot)", hotkey[:16])
            return None
        if digest and digest in _completed_digests(self.completed_repos):
            cid = self.next_id()
            entry = {"challenge_id": cid, **reveal, "queued_at": _now(), "retry_count": int(reveal.get("retry_count", 0))}
            entry.pop("reeval", None)
            reason = f"model digest {digest[:19]} was already submitted/evaluated before"
            log.warning("rejecting %s: %s", cid, reason)
            if hotkey:
                self.seen.add(hotkey)
            if repo:
                self.completed_repos.add(model_key)
            self.failed_repos.add(model_key)
            self.record_failure(entry, "digest_already_completed", reason)
            if not defer_flush:
                self.flush()
                self.flush_dashboard(force=True)
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

    def set_king(self, hotkey, model_repo, block, challenge_id="seed", king_digest="",
                 *, displace_in_place=False):
        global _king_config, _king_config_key
        _king_config = None
        _king_config_key = None
        self.failed_repos.clear()
        self.evaluated_repos.clear()
        prev_repo = self.king.get("model_repo") if self.king else ""
        if displace_in_place:
            # Identical earlier weights reclaim the existing reign slot.
            displaced_digest = self.king.get("king_digest", "")
            if displaced_digest:
                self.king_chain = [
                    e for e in self.king_chain
                    if e.get("king_digest") != displaced_digest
                ]
            reign = self.king.get("reign_number", 0) if self.king else 1
        else:
            reign = self.king.get("reign_number", 0) + (0 if challenge_id == "seed" else 1)
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
        if verdict.get("challenger_committed_at") is not None:
            entry["challenger_committed_at"] = verdict["challenger_committed_at"]
        if verdict.get("king_committed_at") is not None:
            entry["king_committed_at"] = verdict["king_committed_at"]
        if verdict.get("challenger_timestamp_source"):
            entry["challenger_timestamp_source"] = verdict["challenger_timestamp_source"]
        if verdict.get("king_timestamp_source"):
            entry["king_timestamp_source"] = verdict["king_timestamp_source"]
        if verdict.get("source_scores"):
            entry["source_scores"] = verdict["source_scores"]
        shards_used = _verdict_shards_used(verdict)
        if shards_used:
            dataset = verdict.get("dataset") if isinstance(verdict.get("dataset"), dict) else {}
            entry["shards_used"] = shards_used
            entry["dataset"] = {
                "source": verdict.get("dataset_source") or dataset.get("source"),
                "shards_used": shards_used,
            }
        if verdict.get("early_stopped"):
            entry["early_stopped"] = True
            entry["n_sequences"] = verdict.get("n_sequences")
            entry["n_sequences_evaluated"] = verdict.get("n_sequences_evaluated")
        self.history.insert(0, entry)
        self.r2.put("state/dashboard_history.json", {"history": self.history})

    def record_failure(self, entry, error_code, error_detail="", extra: dict | None = None):
        hk = entry.get("hotkey", "")
        record = {
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
        }
        if extra:
            record.update(extra)
        self.history.insert(0, record)
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

    def expected_coldkey_token(self, hotkey: str) -> str | None:
        ck = self.coldkey_for(hotkey)
        if not ck:
            return None
        return ck[:COLDKEY_PREFIX_LEN] + ck[-COLDKEY_SUFFIX_LEN:]

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
        ck = self.coldkey_for(hk) or entry.get("coldkey")
        return {**entry, "uid": self.uid_map.get(hk), "coldkey": ck}

    def flush_dashboard(self, *, force: bool = False):
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
                            "coldkey": (self.coldkey_for(e.get("hotkey", ""))
                                        or e.get("coldkey")),
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

async def _stream_events_with_idle_watchdog(stream, state, cid):
    # Keep one pending read alive; cancelling it closes httpx's stream.
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
    if isinstance(exc, asyncio.CancelledError):
        return True, "validator_cancelled"
    text = str(exc).lower()
    if ("stuck cdn" in text) or ("prefetch" in text and "exceeded" in text):
        return False, "prefetch_exhausted"
    if (
        "failed to download shard" in text
        or "s3 shard download failed" in text
        or "retriesexceeded" in text
        or "max retries exceeded" in text
    ):
        return True, "dataset_shard_download"
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

    reveal_block = int(entry.get("block", 0) or 0)
    if reveal_block <= MIN_SUBMISSION_BLOCK:
        log.info("skipping %s: submission block %s is not over %s",
                 cid, reveal_block, MIN_SUBMISSION_BLOCK)
        return

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

    challenger_coldkey = str(entry.get("coldkey") or "").strip()
    if not challenger_coldkey:
        challenger_coldkey = _resolve_chain_coldkey(subtensor, hotkey, reveal_block) or ""
    if not challenger_coldkey:
        reason = (f"could not resolve the on-chain owner of challenger hotkey {hotkey} "
                  f"at the chain head or reveal block {reveal_block}")
        log.warning("rejecting %s: %s", cid, reason)
        state.failed_repos.add(model_key)
        state.record_failure(entry, "coldkey_unresolved", reason)
        return
    entry["coldkey"] = challenger_coldkey
    state.hotkey_coldkey[hotkey] = challenger_coldkey

    legacy_king_digest = (entry.get("king_digest_at_reveal") or "").strip()
    if legacy_king_digest:
        log.warning("rejecting %s: legacy reveal pinned king %s", cid, legacy_king_digest[:19])
        state.failed_repos.add(model_key)
        state.record_failure(entry, "legacy_reveal_version",
                              "submission included king_digest; resubmit as "
                              "v4|repo|challenger_digest|author_hotkey")
        return

    expected_ck_token = state.expected_coldkey_token(hotkey)
    if expected_ck_token:
        if expected_ck_token.lower() not in model_repo.lower():
            reason = (f"Hippius repo '{model_repo}' must contain miner coldkey token "
                      f"'{expected_ck_token}' (first {COLDKEY_PREFIX_LEN} + last "
                      f"{COLDKEY_SUFFIX_LEN} chars of the coldkey ss58, concatenated); "
                      f"rename your Hippius namespace or model "
                      f"to embed it, then re-reveal on chain")
            log.warning("rejecting %s (%s): %s", cid, model_repo, reason)
            state.failed_repos.add(model_key)
            state.record_failure(entry, "coldkey_required", reason)
            return
    else:
        log.info("%s: coldkey for %s unavailable, skipping coldkey check",
                 cid, hotkey[:16])

    _ = check_stale  # parameter retained for back-compat; v4 removed stale-parent binding

    # Never fall back from the committed immutable digest to a mutable tag.
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

    copy_result = check_model_copy(
        model_repo, challenger_digest,
        king_repo=state.king.get("model_repo", ""),
        king_digest=state.king.get("king_digest", ""),
    )
    if copy_result is not None:
        action = copy_result["action"]
        reason = copy_result["reason"]
        if action == "reject":
            log.warning("rejecting %s (%s): %s", cid, model_repo, reason)
            state.failed_repos.add(model_key)
            state.record_failure(entry, "model_copy", reason, extra={
                "challenger_committed_at": copy_result.get("challenger_committed_at"),
                "king_committed_at": copy_result.get("king_committed_at"),
                "challenger_timestamp_source": copy_result.get("challenger_timestamp_source"),
                "king_timestamp_source": copy_result.get("king_timestamp_source"),
            })
            return
        if action == "crown_earlier":
            log.warning(
                "%s (%s): identical weights but earlier registry push time; "
                "displacing king. %s",
                cid, model_repo, reason,
            )
            state.set_phase("crown_earlier_commit", challenge_id=cid,
                            notes=f"crowning {model_repo} as original author")
            prev_repo = state.king.get("model_repo") if state.king else ""
            dethrone_block = entry.get("block", 0) or _safe_block(subtensor)

            rejection = validate_challenger_config(
                model_repo, challenger_digest,
                king_repo=state.king.get("model_repo", ""),
                king_digest=state.king.get("king_digest", ""),
            )
            if rejection:
                log.warning("crown_earlier %s (%s) blocked by config check: %s",
                            cid, model_repo, rejection)
                state.failed_repos.add(model_key)
                state.record_failure(entry, "config_rejected", rejection)
                return

            synthetic_verdict = {
                "accepted": True,
                "verdict": "crown_earlier_commit",
                "challenge_id": cid,
                "challenger_digest": challenger_digest,
                "rejection_reason": None,
                "mu_hat": 0.0,
                "lcb": 0.0,
                "delta": 0.0,
                "avg_king_loss": 0.0,
                "avg_challenger_loss": 0.0,
                "wall_time_s": 0.0,
                "timestamp": _now(),
                "challenger_committed_at": copy_result.get("challenger_committed_at"),
                "king_committed_at": copy_result.get("king_committed_at"),
                "challenger_timestamp_source": copy_result.get("challenger_timestamp_source"),
                "king_timestamp_source": copy_result.get("king_timestamp_source"),
            }
            state.stats["accepted"] += 1
            state.record_verdict(synthetic_verdict, model_repo, hotkey)
            state.set_king(hotkey, model_repo, dethrone_block,
                           challenge_id=cid, king_digest=challenger_digest,
                           displace_in_place=True)
            state.last_winner_hotkey = hotkey
            state.flush_dashboard(force=True)
            try:
                await maybe_set_weights(subtensor, wallet, state,
                                        force=True, reason="crown_earlier_commit")
            except Exception:
                log.exception("force weight-set after crown_earlier_commit failed")
            await notify_new_king({
                "hotkey": hotkey,
                "model_repo": model_repo,
                "reign_number": state.king.get("reign_number", 0),
                "king_digest": challenger_digest,
                "previous_repo": prev_repo,
            }, synthetic_verdict)
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

    # Sampling is bound to the reveal block, not the later evaluation block.
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

    state.set_phase(
        "dataset_multi_source",
        challenge_id=cid,
        notes="eval server selects the configured multi-source sample",
    )
    shard_key = "multi_source_npy"

    king_repo = state.king.get("model_repo", SEED_REPO)
    king_digest = state.king.get("king_digest", "")

    state.set_phase("dispatch_eval", challenge_id=cid, notes=f"dispatching {cid} to eval server")
    r2.put(f"eval/{cid}/meta.json", {
        "challenge_id": cid, "king_repo": king_repo,
        "king_digest": king_digest,
        "challenger_repo": model_repo, "challenger_digest": challenger_digest,
        "hotkey": hotkey, "coldkey": challenger_coldkey,
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
    async with httpx.AsyncClient(timeout=httpx.Timeout(2700.0, connect=30.0)) as client:
        eval_payload = {
            "king_repo": king_repo,
            "challenger_repo": model_repo,
            "block_hash": block_hash,
            "hotkey": hotkey,
            "coldkey": challenger_coldkey,
            "shard_key": shard_key,
            "king_digest": king_digest,
            "challenger_digest": challenger_digest,
            "delta_threshold": EVAL_DELTA_THRESHOLD,
            "n": EVAL_N,
            "n_bootstrap": EVAL_BOOTSTRAP_SAMPLES,
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
                                  timeout=httpx.Timeout(2700.0)) as stream:
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if not EVAL_SERVER_URL:
        log.error("set TEUTONIC_EVAL_SERVER")
        sys.exit(1)

    log.info("startup: constructing storage clients")
    r2 = R2()
    state = State(r2)
    log.info("startup: loading persisted state")
    state.load()
    log.info("startup: persisted state loaded")

    log.info("startup: opening wallet %s/%s", WALLET_NAME, WALLET_HOTKEY)
    wallet = bt.Wallet(name=WALLET_NAME, hotkey=WALLET_HOTKEY)
    log.info("startup: connecting subtensor network=%s", NETWORK)
    subtensor = bt.Subtensor(network=NETWORK)
    log.info("startup: subtensor connected")

    # Refuse to publish unencrypted weights.
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
        build_id = hashlib.sha256(html_bytes).hexdigest()[:12]
        html_bytes = html_bytes.replace(b"__BUILD_ID__", build_id.encode())
        r2.put_dashboard_raw(
            "index.html",
            html_bytes,
            "text/html; charset=utf-8",
            cache_control="no-cache, must-revalidate",
        )
        log.info("uploaded dashboard to Hippius (build=%s)", build_id)

    if not state.king:
        if not SEED_DIGEST:
            log.error("set TEUTONIC_SEED_DIGEST for the initial seed king")
            sys.exit(1)
        seed_ref = ModelRef(SEED_REPO, SEED_DIGEST)
        materialize_model(seed_ref, max_workers=4, config_only=True)
        log.info("seed king %s", seed_ref.immutable_ref)
        state.set_king(wallet.hotkey.ss58_address, seed_ref.repo,
                       subtensor.block, king_digest=seed_ref.digest)

    state.clear_restart_request()
    await maybe_set_weights(subtensor, wallet, state, force=True, reason="startup")

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
                # Let wait_for receive cancellation directly; wrap only inner errors.
                async def _bounded_eval():
                    try:
                        await process_challenge(state, r2, entry, subtensor, wallet)
                    except asyncio.CancelledError:
                        raise
                    except BaseException as inner:
                        raise _EvalInnerError(inner) from inner
                try:
                    await asyncio.wait_for(_bounded_eval(), timeout=TICK_RESTART_AFTER)
                except asyncio.TimeoutError:
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


def main_sync():
    asyncio.run(main())


if __name__ == "__main__":
    main_sync()
