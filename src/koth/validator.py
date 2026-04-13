"""Main validator coordinator loop for King of the Hill.

The coordinator runs on the validator host (no GPU needed). It:
1. Polls the Bittensor chain for newly revealed miner commits
2. Queues challenges in block order (each hotkey evaluated at most once)
3. For each challenge:
   a. Checks staleness (king_hash match)
   b. Spins up an ephemeral Lium pod
   c. Deploys the eval worker script
   d. Polls R2 for the verdict
   e. If accepted: forks winner into king repo, sets weights
   f. Tears down the pod
4. Updates state in R2 for the dashboard
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import bittensor as bt

from .commit import RevealScanner
from .config import KOTHConfig
from .king import KingManager
from .orchestrator import PodOrchestrator, poll_for_verdict
from .r2 import R2Client
from .state import ValidatorState

logger = logging.getLogger(__name__)


class Validator:
    """King of the Hill validator coordinator."""

    def __init__(self, config: KOTHConfig):
        self.config = config
        self.r2 = R2Client(config.r2)
        self.state = ValidatorState(self.r2)
        self.king_mgr = KingManager(
            hf_repo=config.king.hf_repo,
            cache_dir=config.king.local_cache_dir,
            hf_token=config.king.hf_token,
        )
        self.orchestrator = PodOrchestrator(config.pod, config.r2)
        self._challenge_counter = 0

        # Bittensor
        self.wallet = bt.wallet(
            name=config.chain.wallet_name,
            hotkey=config.chain.wallet_hotkey,
        )
        self.subtensor = bt.subtensor(network=config.chain.network)
        self.scanner: RevealScanner | None = None

    def _next_challenge_id(self) -> str:
        self._challenge_counter += 1
        return f"eval-{self._challenge_counter:04d}"

    def initialize(self) -> None:
        """Initialize the validator: load state, download king, wire up scanner."""
        logger.info("Initializing KOTH validator...")
        self.state.load()

        self.scanner = RevealScanner(
            self.subtensor,
            self.config.chain.netuid,
            seen_hotkeys=self.state.seen_hotkeys,
        )

        king_dir = self.king_mgr.download_king()
        logger.info("King loaded: hash=%s", self.king_mgr.king_hash[:16])

        if not self.state.king:
            self.state.set_initial_king(
                hotkey=self.wallet.hotkey.ss58_address,
                hf_repo=self.config.king.hf_repo,
                king_hash=self.king_mgr.king_hash,
                block=self.subtensor.block,
            )

    def run(self) -> None:
        """Main loop: poll for commits, evaluate challengers, update king."""
        self.initialize()
        logger.info("KOTH validator running. Polling every %ds.", self.config.poll_interval_s)

        while True:
            try:
                self._tick()
            except KeyboardInterrupt:
                logger.info("Shutting down.")
                break
            except Exception:
                logger.exception("Error in main loop tick")

            time.sleep(self.config.poll_interval_s)

    def _tick(self) -> None:
        """One iteration: scan for revealed commits, process queue."""
        new_reveals = self.scanner.scan()

        if new_reveals:
            # Scanner already added hotkeys to its seen set; persist to R2
            self.state.seen_hotkeys = self.scanner.seen_hotkeys
            self.state._flush_seen_hotkeys()

        for reveal in new_reveals:
            challenge_id = self._next_challenge_id()
            self.state.enqueue_challenge(
                challenge_id=challenge_id,
                hotkey=reveal.hotkey,
                hf_repo=reveal.hf_repo,
                commit_block=reveal.block,
                king_hash=reveal.king_hash,
                model_hash=reveal.model_hash,
            )
            logger.info(
                "Queued reveal %s from %s (block %d, repo %s)",
                challenge_id, reveal.hotkey[:16], reveal.block, reveal.hf_repo,
            )

        while True:
            entry = self.state.dequeue()
            if entry is None:
                break
            self._process_challenge(entry)

    def _process_challenge(self, entry: dict) -> None:
        """Process a single challenge from the queue."""
        challenge_id = entry["challenge_id"]
        hotkey = entry["hotkey"]
        hf_repo = entry["hf_repo"]
        king_hash = entry["king_hash"]
        model_hash = entry["model_hash"]

        logger.info("Processing challenge %s from %s", challenge_id, hotkey[:16])

        # Staleness check
        current_hash = self.king_mgr.king_hash
        if not current_hash.startswith(king_hash[:len(king_hash)]):
            logger.info("Stale submission %s: king changed", challenge_id)
            self.state.submission_stale(challenge_id, hotkey)
            return

        # Spin up eval pod
        pod_name = f"koth-eval-{challenge_id}"
        pod = None
        try:
            pod = self.orchestrator.start_pod(pod_name)
            self.state.eval_started(challenge_id, hotkey, hf_repo, pod_name)

            # Pick a dataset shard for eval
            dataset_shard_key = self._select_dataset_shard()

            # Deploy and run eval
            self.orchestrator.deploy_and_run_eval(
                pod=pod,
                challenge_id=challenge_id,
                king_repo=self.config.king.hf_repo,
                challenger_repo=hf_repo,
                eval_cfg=self.config.eval,
                bbox_cfg=self.config.bounding_box,
                dataset_shard_key=dataset_shard_key,
                hf_token=self.config.king.hf_token,
            )

            # Poll for verdict
            verdict = poll_for_verdict(
                self.r2,
                challenge_id,
                timeout_s=self.config.pod.eval_timeout_s,
                interval_s=15,
            )

            if verdict is None:
                logger.error("Eval %s timed out", challenge_id)
                self.state.submission_rejected(challenge_id, hotkey, "eval_timeout")
                return

            # Check for validation rejection (not a sign test result)
            if verdict.get("reason") in (
                "key_mismatch", "shape_mismatch", "dtype_mismatch",
                "linf_violation", "l2_tensor_violation", "l2_global_violation",
                "frozen_modified", "hash_mismatch",
            ):
                self.state.submission_rejected(
                    challenge_id, hotkey,
                    verdict["reason"], verdict.get("detail", ""),
                )
                return

            accepted = verdict.get("accepted", False)
            self.state.eval_completed(challenge_id, hotkey, accepted, verdict)

            if accepted:
                self._crown_new_king(challenge_id, hotkey, hf_repo, entry.get("commit_block", 0), verdict)

        except Exception:
            logger.exception("Error processing challenge %s", challenge_id)
            self.state.submission_rejected(challenge_id, hotkey, "eval_error")
        finally:
            if pod:
                try:
                    self.orchestrator.stop_pod(pod)
                except Exception:
                    logger.exception("Failed to stop pod %s", pod_name)

    def _crown_new_king(
        self, challenge_id: str, hotkey: str, hf_repo: str, block: int, verdict: dict
    ) -> None:
        """Crown a new king: fork weights, update state, set weights."""
        logger.info("Crowning new king: %s from %s", challenge_id, hotkey[:16])

        # Download the winner's model (or use cached from eval)
        winner_dir = self.king_mgr.download_challenger(hf_repo)

        # Fork into king repo
        new_hash = self.king_mgr.fork_winner(winner_dir, hotkey, challenge_id)

        # Update state
        self.state.king_changed(
            challenge_id=challenge_id,
            new_hotkey=hotkey,
            new_hf_repo=self.config.king.hf_repo,
            new_king_hash=new_hash,
            block=block,
            avg_loss=verdict.get("avg_challenger_loss", 0.0),
        )

        # Set weights on chain: 100% to the new king
        self._set_weights(hotkey)

    def _set_weights(self, king_hotkey: str) -> None:
        """Set 100% of weights to the reigning king's UID."""
        try:
            metagraph = self.subtensor.metagraph(self.config.chain.netuid)
            if king_hotkey in metagraph.hotkeys:
                king_uid = metagraph.hotkeys.index(king_hotkey)
                self.subtensor.set_weights(
                    wallet=self.wallet,
                    netuid=self.config.chain.netuid,
                    uids=[king_uid],
                    weights=[1.0],
                )
                logger.info("Weights set: 100%% to UID %d (%s)", king_uid, king_hotkey[:16])
            else:
                logger.warning("King hotkey %s not in metagraph", king_hotkey[:16])
        except Exception:
            logger.exception("Failed to set weights")

    def _select_dataset_shard(self) -> str:
        """Select a dataset shard key for evaluation.

        Uses one of the anneal shards on R2.
        """
        shards = [
            "anneal/anneal_000000.npy",
            "anneal/anneal_000002.npy",
            "anneal/anneal_000004.npy",
            "anneal/anneal_000005.npy",
        ]
        # Round-robin through shards
        idx = self._challenge_counter % len(shards)
        return shards[idx]
