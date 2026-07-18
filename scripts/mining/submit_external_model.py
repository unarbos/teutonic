#!/usr/bin/env python3
"""Submit an external HF model as a challenger to Teutonic SN3.

Unlike miner.py (which fine-tunes from the king), this submits a vanilla
HuggingFace model as-is — useful for first-deployment smoke testing where the
operator wants to exercise the full submission pipeline with a real model.

What it does:
  1. (re)uses a local staging dir (from `huggingface_hub.snapshot_download`)
  2. patches config.json fields that need to match the king's arch lock
  3. validates the patched config matches the king
  4. uploads to Hippius Hub under the operator's namespace
  5. forms a v4 reveal commitment and submits it on chain

Coldkey-token gate: the validator REJECTS any Hippius repo whose name does
NOT contain the miner's coldkey token (first 5 + last 5 ss58 chars of the
coldkey, concatenated). The repo name this script generates embeds the
token automatically.

Usage:
    HF_TOKEN=... python scripts/mining/submit_external_model.py \
        --hf Qwen/Qwen3-4B-Base \
        --hippius-namespace Arbos \
        --wallet default --hotkey teutonic-miner

Hippius Hub token is read from $HIPPIUS_HUB_TOKEN or
$HOME/.cache/hippius/hub/{token,api_token}; see model_store._resolve_hub_token.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import bittensor as bt
import huggingface_hub

# chain_config + model_store sit at the repo root.
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
import chain_config  # noqa: E402
from model_store import (  # noqa: E402
    DIGEST_RE,
    ModelRef,
    build_reveal_v4,
    upload_model_folder,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("submit-external")


# Fields that the validator's validate_challenger_config compares between
# king and challenger. Must match exactly; absent-in-king must match
# absent-in-challenger. Mirror of the lock list in validator.py.
_GENERIC_LOCK = (
    "vocab_size", "hidden_size", "num_hidden_layers",
    "num_attention_heads", "num_key_value_heads", "head_dim",
    "intermediate_size", "model_type",
    "tie_word_embeddings", "rope_theta", "max_position_embeddings",
    "max_seq_len",
)


def _load_king_config(king_repo: str, king_digest: str) -> dict:
    """Pull just the king's config.json — works for hf:<rev> and sha256:<oci>."""
    if king_digest.startswith("hf:"):
        path = huggingface_hub.hf_hub_download(
            king_repo, "config.json",
            revision=king_digest[3:],
            token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY"),
        )
        with open(path) as f:
            return json.load(f)
    # Hippius OCI digest — go through materialize_model config_only path.
    from model_store import materialize_model
    snap = materialize_model(ModelRef(king_repo, king_digest),
                             max_workers=4, config_only=True)
    with open(Path(snap) / "config.json") as f:
        return json.load(f)


def _patch_config_to_match(chall_cfg: dict, king_cfg: dict) -> list[str]:
    """Mutate chall_cfg in place so it matches king_cfg on all lock keys.

    Returns the list of patched key names. Only patches numeric / scalar
    fields — refuses to patch lock keys that would require touching weights
    (vocab_size, hidden_size, num_*, intermediate_size, model_type, head_dim).
    """
    WEIGHTS_BOUND = {
        "vocab_size", "hidden_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads", "head_dim",
        "intermediate_size", "model_type",
    }
    patched = []
    for key in _GENERIC_LOCK + chain_config.EXTRA_LOCK_KEYS:
        king_val = king_cfg.get(key)
        chall_val = chall_cfg.get(key)
        if king_val == chall_val:
            continue
        if key in WEIGHTS_BOUND:
            raise SystemExit(
                f"can't reconcile {key}: king={king_val} challenger={chall_val} "
                f"(would require changing weights — refusing)"
            )
        log.info("patching config[%s]: %s -> %s", key, chall_val, king_val)
        if king_val is None:
            chall_cfg.pop(key, None)
        else:
            chall_cfg[key] = king_val
        patched.append(key)
    return patched


def _validate_local(king_cfg: dict, chall_cfg: dict, chall_dir: str) -> str | None:
    if king_cfg.get("architectures") != chall_cfg.get("architectures"):
        return f"architectures mismatch: king={king_cfg.get('architectures')} chall={chall_cfg.get('architectures')}"
    _SENTINEL = object()
    for key in _GENERIC_LOCK + chain_config.EXTRA_LOCK_KEYS:
        a = king_cfg.get(key, _SENTINEL)
        b = chall_cfg.get(key, _SENTINEL)
        if a != b:
            return f"{key} mismatch: king={a} chall={b}"
    if not list(Path(chall_dir).glob("*.safetensors")):
        return "no .safetensors files"
    if list(Path(chall_dir).rglob("*.py")):
        return "challenger contains *.py files (validator rejects custom modeling code)"
    if "auto_map" in chall_cfg:
        return "config.json has auto_map (validator rejects trust_remote_code path)"
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hf", required=True,
                   help="HuggingFace repo id (e.g. Qwen/Qwen3-4B-Base)")
    p.add_argument("--hippius-namespace", required=True,
                   help="Hippius account namespace to upload under (e.g. Arbos)")
    p.add_argument("--wallet", default="default", help="bt wallet name")
    p.add_argument("--hotkey", required=True, help="bt hotkey name")
    p.add_argument("--netuid", type=int, default=3)
    p.add_argument("--network", default="finney")
    p.add_argument("--workdir", default="/tmp/qwen3-4b-base-stage",
                   help="Local staging dir (snapshot_download target)")
    p.add_argument("--suffix", default=None,
                   help="Repo name suffix (default: <coldkey5+5>-<hfname>)")
    p.add_argument("--dry-run", action="store_true",
                   help="Stop after upload — print payload, don't submit on chain")
    p.add_argument("--skip-download", action="store_true",
                   help="Trust the workdir is already populated (e.g. you ran "
                        "snapshot_download in the background)")
    args = p.parse_args()

    # Chain config — king is what the validator's chain.toml says.
    king_repo = chain_config.SEED_REPO
    king_digest = getattr(chain_config, "SEED_DIGEST", "")
    if not king_digest:
        log.error("chain.toml has no seed_digest")
        sys.exit(1)
    log.info("targeting king %s @ %s", king_repo, king_digest[:24])

    # Wallet — needed up front so we can compute the coldkey token for the repo name.
    wallet = bt.Wallet(name=args.wallet, hotkey=args.hotkey)
    ck_ss58 = wallet.coldkeypub.ss58_address
    ck_token = ck_ss58[:5] + ck_ss58[-5:]
    hk_ss58 = wallet.hotkey.ss58_address
    log.info("wallet: cold=%s (token %s) hot=%s...", ck_ss58, ck_token, hk_ss58[:8])

    suffix = args.suffix or f"{ck_token}-{args.hf.rsplit('/', 1)[-1].replace('_', '-')}"
    # Docker registry repo names must be all-lowercase per the distribution spec.
    repo = f"{args.hippius_namespace}/{chain_config.NAME}-{suffix}".lower()
    log.info("target Hippius repo: %s", repo)

    # Pre-flight: hotkey registered on netuid
    sub = bt.Subtensor(network=args.network)
    meta = sub.metagraph(args.netuid)
    if hk_ss58 not in meta.hotkeys:
        log.error("hotkey %s not registered on netuid %d", hk_ss58, args.netuid)
        log.error("register first: btcli subnet register --wallet.name %s --wallet.hotkey %s --netuid %d",
                  args.wallet, args.hotkey, args.netuid)
        sys.exit(2)

    # Pre-flight: hotkey doesn't already have a reveal on chain.
    revealed = sub.get_revealed_commitment_by_hotkey(args.netuid, hk_ss58)
    if revealed:
        log.warning("hotkey already has %d reveal(s) on chain. The validator "
                    "may de-dupe and skip this submission. Continuing.", len(revealed))

    # Stage the model.
    work = Path(args.workdir)
    if not args.skip_download or not (work / "config.json").exists():
        if work.exists():
            shutil.rmtree(work)
        log.info("downloading %s -> %s ...", args.hf, work)
        huggingface_hub.snapshot_download(
            args.hf,
            local_dir=str(work),
            allow_patterns=["*.safetensors", "*.json", "tokenizer*",
                            "special_tokens*", "*.model", "*.txt"],
            max_workers=8,
            token=os.environ.get("HF_TOKEN") or
                  open(os.path.expanduser("~/.cache/huggingface/token")).read().strip(),
        )
    else:
        log.info("using pre-staged %s (skip-download)", work)

    # Surgical config patch + scrub.
    chall_cfg_path = work / "config.json"
    chall_cfg = json.loads(chall_cfg_path.read_text())
    king_cfg = _load_king_config(king_repo, king_digest)
    patched = _patch_config_to_match(chall_cfg, king_cfg)
    if patched:
        chall_cfg_path.write_text(json.dumps(chall_cfg, indent=2))
        log.info("patched %d config field(s): %s", len(patched), patched)

    # Scrub things the validator rejects (auto_map, *.py — Qwen3 ships neither
    # but defense-in-depth).
    if "auto_map" in chall_cfg:
        chall_cfg.pop("auto_map", None)
        chall_cfg_path.write_text(json.dumps(chall_cfg, indent=2))
        log.info("stripped auto_map from config.json")
    for py in work.rglob("*.py"):
        log.info("removing modeling file: %s", py.name)
        py.unlink()

    # Final local validation — exact same check the validator runs.
    rej = _validate_local(king_cfg, chall_cfg, str(work))
    if rej:
        log.error("local validation failed: %s", rej)
        sys.exit(3)
    log.info("local validation passed")

    size_gb = sum(f.stat().st_size for f in work.glob("*.safetensors")) / (1 << 30)
    log.info("safetensors bytes: %.2f GiB across %d files",
             size_gb, len(list(work.glob('*.safetensors'))))

    # Upload to Hippius.
    log.info("uploading to Hippius Hub: %s ...", repo)
    ref = upload_model_folder(str(work), repo, revision=suffix,
                              commit_message=f"submit {args.hf} as challenger")
    log.info("uploaded: %s", ref.immutable_ref)
    if not DIGEST_RE.match(ref.digest):
        log.error("upload returned invalid digest: %r", ref.digest)
        sys.exit(4)

    # Form v4 reveal.
    payload = build_reveal_v4(ref, hk_ss58)
    log.info("v4 reveal payload (%d chars): %s", len(payload), payload)

    if args.dry_run:
        log.info("[dry-run] not submitting on chain. Run again without --dry-run "
                 "to commit.")
        return

    # Commit.
    log.info("submitting set_reveal_commitment (blocks_until_reveal=3) ...")
    resp = sub.set_reveal_commitment(
        wallet=wallet, netuid=args.netuid,
        data=payload, blocks_until_reveal=3,
        wait_for_revealed_execution=False,
    )
    if resp.success:
        log.info("submit ok: %s", resp.message)
        log.info("validator will pick this up after ~3-block reveal delay (~36s)")
    else:
        log.error("submit failed: %s", resp.message)
        sys.exit(5)


if __name__ == "__main__":
    main()
