"""CLI entry points for the KOTH system."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-20s %(levelname)-7s %(message)s",
)


def _load_config():
    """Load KOTHConfig from environment variables or a JSON file."""
    from .config import (
        KOTHConfig, EvalConfig, BoundingBoxConfig, R2Config,
        KingConfig, ChainConfig, PodConfig,
    )

    return KOTHConfig(
        eval=EvalConfig(
            N=int(os.getenv("KOTH_EVAL_N", "10000")),
            alpha=float(os.getenv("KOTH_EVAL_ALPHA", "0.01")),
            sequence_length=int(os.getenv("KOTH_SEQUENCE_LENGTH", "2048")),
        ),
        bounding_box=BoundingBoxConfig(
            max_linf=float(os.getenv("KOTH_BBOX_MAX_LINF", "0.5")),
            max_l2_global=float(os.getenv("KOTH_BBOX_MAX_L2_GLOBAL", "0")) or None,
        ),
        r2=R2Config(
            endpoint_url=os.getenv("KOTH_R2_ENDPOINT", ""),
            bucket_name=os.getenv("KOTH_R2_BUCKET", ""),
            access_key_id=os.getenv("KOTH_R2_ACCESS_KEY", ""),
            secret_access_key=os.getenv("KOTH_R2_SECRET_KEY", ""),
        ),
        king=KingConfig(
            hf_repo=os.getenv("KOTH_KING_REPO", ""),
            hf_token=os.getenv("HF_TOKEN", ""),
            local_cache_dir=os.getenv("KOTH_CACHE_DIR", "/tmp/koth/king"),
        ),
        chain=ChainConfig(
            netuid=int(os.getenv("KOTH_NETUID", "3")),
            network=os.getenv("KOTH_NETWORK", "finney"),
            wallet_name=os.getenv("KOTH_WALLET_NAME", "default"),
            wallet_hotkey=os.getenv("KOTH_WALLET_HOTKEY", "default"),
        ),
        pod=PodConfig(
            gpu_type=os.getenv("KOTH_GPU_TYPE", "rtx4090"),
        ),
        poll_interval_s=int(os.getenv("KOTH_POLL_INTERVAL", "12")),
    )


def run_validator():
    """Run the KOTH validator coordinator."""
    from .validator import Validator

    config = _load_config()
    validator = Validator(config)
    validator.run()


def run_miner():
    """Run the KOTH reference miner."""
    parser = argparse.ArgumentParser(description="KOTH Reference Miner")
    parser.add_argument("--king-repo", required=True, help="HF repo of the current king")
    parser.add_argument("--miner-repo", required=True, help="Your HF repo to upload the challenger")
    parser.add_argument("--dataset", default="", help="Path to tokenized .npy dataset shard")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--steps", type=int, default=100, help="Training steps")
    parser.add_argument("--seq-len", type=int, default=2048, help="Sequence length")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--netuid", type=int, default=3)
    parser.add_argument("--network", default="finney")
    parser.add_argument("--wallet-name", default="default")
    parser.add_argument("--wallet-hotkey", default="default")
    args = parser.parse_args()

    from .miner import Miner

    miner = Miner(
        king_repo=args.king_repo,
        miner_repo=args.miner_repo,
        netuid=args.netuid,
        network=args.network,
        wallet_name=args.wallet_name,
        wallet_hotkey=args.wallet_hotkey,
        hf_token=os.getenv("HF_TOKEN", ""),
        learning_rate=args.lr,
        train_steps=args.steps,
        sequence_length=args.seq_len,
        batch_size=args.batch_size,
        dataset_path=args.dataset,
    )
    miner.run()


def seed_king():
    """Upload an initial seed model to the king repo."""
    parser = argparse.ArgumentParser(description="Seed the KOTH king model")
    parser.add_argument("--model-dir", required=True, help="Path to the seed model directory")
    parser.add_argument("--king-repo", required=True, help="HF repo for the king")
    args = parser.parse_args()

    from .king import KingManager

    mgr = KingManager(
        hf_repo=args.king_repo,
        cache_dir="/tmp/koth/king",
        hf_token=os.getenv("HF_TOKEN", ""),
    )
    king_hash = mgr.upload_seed(args.model_dir)
    print(f"Seed king uploaded to {args.king_repo}")
    print(f"King hash: {king_hash}")


def show_state():
    """Show the current validator state from R2."""
    config = _load_config()

    from .r2 import R2Client

    r2 = R2Client(config.r2)

    state = r2.get_json("state/validator_state.json")
    if state:
        print(json.dumps(state, indent=2))
    else:
        print("No validator state found in R2.")

    king = r2.get_json("king/current.json")
    if king:
        print("\nCurrent King:")
        print(json.dumps(king, indent=2))


def main():
    parser = argparse.ArgumentParser(description="King of the Hill")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("validator", help="Run the validator coordinator")
    sub.add_parser("miner", help="Run the reference miner")
    sub.add_parser("seed", help="Upload initial seed king model")
    sub.add_parser("state", help="Show current validator state from R2")

    args, remaining = parser.parse_known_args()

    if args.command == "validator":
        sys.argv = [sys.argv[0]] + remaining
        run_validator()
    elif args.command == "miner":
        sys.argv = [sys.argv[0]] + remaining
        run_miner()
    elif args.command == "seed":
        sys.argv = [sys.argv[0]] + remaining
        seed_king()
    elif args.command == "state":
        sys.argv = [sys.argv[0]] + remaining
        show_state()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
