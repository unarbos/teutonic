#!/usr/bin/env python3
"""Replay the exact evaluated prefix from a cached production evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np

from teutonic.evaluation import EvaluationRequestV2
from teutonic.evaluator import engine, sources
from teutonic.evaluator.long_documents import sequence_digest as ragged_sequence_digest


def snapshot_path(cache_dir: Path, digest: str) -> Path:
    path = cache_dir / digest
    if not (path / "config.json").is_file() or not any(path.glob("*.safetensors")):
        raise FileNotFoundError(f"cached model snapshot is incomplete: {path}")
    return path


def parse_gpu_ids(value: str) -> list[int]:
    gpu_ids = [int(item) for item in value.split(",")]
    if len(gpu_ids) != 8 or len(set(gpu_ids)) != 8:
        raise argparse.ArgumentTypeError("exactly eight distinct GPU IDs are required")
    return gpu_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", required=True, type=Path)
    parser.add_argument("--model-cache", required=True, type=Path)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu-ids", type=parse_gpu_ids, default=list(range(8)))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--long-document-config", type=Path, help="append documents using a pinned config; score the full sample")
    parser.add_argument("--verify-packing", action="store_true", help="compare cached-model losses in mixed batches versus one sample at a time")
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "flash_attention_4"),
        default="flash_attention_4",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    cached_record = json.loads(args.record.read_text())
    protocol_request = EvaluationRequestV2.from_mapping(cached_record["request"])
    reference = cached_record["verdict"]
    evaluated_count = int(reference.get("n_sequences_evaluated", reference["n_sequences"]))

    king_snapshot = snapshot_path(args.model_cache, protocol_request.king.expected_digest)
    challenger_snapshot = snapshot_path(
        args.model_cache,
        protocol_request.challenger.expected_digest,
    )
    base_request = engine.internal_request_from_v2(
        protocol_request,
        str(king_snapshot),
        str(challenger_snapshot),
    )
    request = engine.EvalRequest(
        **{
            **base_request.model_dump(),
            "attn_implementation": args.attn_implementation,
            "batch_size": args.batch_size,
            "early_stop_enabled": False,
            "king_digest": protocol_request.king.expected_digest,
            "challenger_digest": protocol_request.challenger.expected_digest,
        }
    )
    # Production sets vocab_size before sampling. It controls the oversample
    # count (and thus RNG consumption), even when every token is in vocabulary.
    king_config = json.loads((king_snapshot / "config.json").read_text())
    challenger_config = json.loads((challenger_snapshot / "config.json").read_text())
    request.vocab_size = int(king_config["vocab_size"])
    request.max_document_tokens = min(int(king_config["max_position_embeddings"]), int(challenger_config["max_position_embeddings"]))
    if args.long_document_config:
        request.long_documents = json.loads(args.long_document_config.read_text())

    model_ready_times = []

    def progress(event: dict) -> None:
        phase = event.get("phase")
        if phase == "model_worker_ready" and event.get("ready") == event.get("total_workers"):
            model_ready_times.append(time.perf_counter())
        if phase in {"model_worker_ready", "model_workers_loading", "heartbeat"} or (
            phase == "eval_progress" and int(event.get("done", 0)) % 100 == 0
        ):
            print(json.dumps(event, sort_keys=True), flush=True)

    all_sequences, dataset_meta = sources.sample_eval_sequences(request, on_phase=progress)
    if args.long_document_config:
        evaluated_count = len(all_sequences)
    if evaluated_count > len(all_sequences):
        raise RuntimeError(
            f"reference evaluated {evaluated_count} of only {len(all_sequences)} reconstructed sequences"
        )
    sequences = all_sequences[:evaluated_count]
    labels = dataset_meta["_source_labels"][:evaluated_count]
    saved_samples = reference.get("sample_results")
    if saved_samples is not None and not args.long_document_config:
        provenance = dataset_meta["_sample_provenance"][:evaluated_count]
        keys = ["shard_group_index", "shard_index", "shard_sequence_index"]
        if request.long_documents:
            for key, source_key in (("document_row", "document_row"), ("document_dataset", "dataset"), ("document_category", "category")):
                if [p.get(source_key) for p in provenance] != saved_samples[key][:evaluated_count]:
                    raise RuntimeError(f"reconstructed documents differ from saved evaluation: {key}")
        for key in keys:
            if [sample[key] for sample in provenance] != saved_samples[key][:evaluated_count]:
                raise RuntimeError(f"reconstructed samples differ from the saved evaluation: {key}")
    sequence_digest = (ragged_sequence_digest(sequences) if request.long_documents else
                       hashlib.sha256(np.asarray(sequences, dtype=np.int64).tobytes()).hexdigest())

    if args.prepare_only:
        report = {
            "status": "prepared",
            "reference_evaluation_id": protocol_request.evaluation_id,
            "reference_request_sha256": protocol_request.request_sha256,
            "requested_sequences": len(all_sequences),
            "evaluated_prefix_sequences": evaluated_count,
            "full_sequence_digest": dataset_meta["digest"],
            "evaluated_prefix_digest": sequence_digest,
            "long_documents": dataset_meta.get("long_documents"),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        return

    packing_check = None
    started = time.perf_counter()
    try:
        king_losses, challenger_losses, worker_meta = engine.score_with_model_workers(
            sequences,
            request,
            str(king_snapshot),
            str(challenger_snapshot),
            args.gpu_ids,
            progress,
        )
        scoring_elapsed = time.perf_counter() - started
        if args.verify_packing:
            # Real cached weights only. Include each long bucket and an ordinary window.
            chosen = {}
            for i, p in enumerate(dataset_meta["_sample_provenance"][:evaluated_count]):
                key = p.get("length_bucket", "windows")
                if key not in chosen:
                    chosen[key] = sequences[i]
            examples = list(chosen.values())
            batched = engine.score_with_model_workers(examples, request, str(king_snapshot), str(challenger_snapshot), args.gpu_ids, progress)
            single_request = request.model_copy(update={"batch_size": 1})
            individual = engine.score_with_model_workers(examples, single_request, str(king_snapshot), str(challenger_snapshot), args.gpu_ids, progress)
            max_diff = max(abs(a - b) for side in range(2) for a, b in zip(batched[side], individual[side], strict=True))
            packing_check = {"max_abs_loss_difference": max_diff, "lengths": [len(s) for s in examples]}
            if not np.isfinite(max_diff) or max_diff > 1e-4:
                raise RuntimeError(f"packed versus individual scoring differs: {packing_check}")
    finally:
        if engine._model_worker_pool is not None:
            engine._model_worker_pool.close(force=True)
            engine._model_worker_pool = None
    elapsed = scoring_elapsed

    counts = worker_meta["scored_tokens"]
    verdict = engine.bootstrap_verdict(king_losses, challenger_losses, request, counts)
    verdict["source_scores"] = engine._compute_source_scores(
        king_losses,
        challenger_losses,
        labels,
        counts,
    )
    if request.long_documents:
        provenance = dataset_meta["_sample_provenance"][:evaluated_count]
        verdict["component_scores"] = engine._compute_source_scores(king_losses, challenger_losses, [p.get("component", "windows") for p in provenance], counts)
        verdict["long_document_scores"] = engine.long_document_scores(king_losses, challenger_losses, provenance, counts)
        verdict["sample_results"] = engine._build_sample_results(king_losses, challenger_losses, provenance, counts)
    comparison = {
        key: float(verdict[key]) - float(reference[key])
        for key in ("avg_king_loss", "avg_challenger_loss", "mu_hat", "lcb")
    }
    report = {
        "status": "complete",
        "reference_record": str(args.record),
        "reference_evaluation_id": protocol_request.evaluation_id,
        "reference_request_sha256": protocol_request.request_sha256,
        "king_digest": protocol_request.king.expected_digest,
        "challenger_digest": protocol_request.challenger.expected_digest,
        "attention_implementation": args.attn_implementation,
        "scoring_policy": engine.MASKED_POLICY_VERSION,
        "reference_scoring_policy": protocol_request.versions["evaluation_policy"],
        "added_long_document_sample": bool(args.long_document_config),
        "long_documents": dataset_meta.get("long_documents"),
        "packing_check": packing_check,
        "scored_tokens": counts,
        "batch_size": args.batch_size,
        "n_sequences": evaluated_count,
        "sequence_digest": sequence_digest,
        "elapsed_seconds": elapsed,
        "model_load_seconds": model_ready_times[0] - started,
        "scoring_seconds": elapsed - (model_ready_times[0] - started),
        "sequences_per_second": evaluated_count / elapsed,
        "reference": {
            key: reference[key] for key in ("avg_king_loss", "avg_challenger_loss", "mu_hat", "lcb")
        },
        "replay": verdict,
        "replay_minus_reference": comparison,
        "worker_metadata": worker_meta,
        "king_losses": king_losses,
        "challenger_losses": challenger_losses,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "n_sequences": evaluated_count,
                "sequence_digest": sequence_digest,
                "sequences_per_second": report["sequences_per_second"],
                "replay_minus_reference": comparison,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
