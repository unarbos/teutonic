#!/usr/bin/env python3
"""Read-only GPU smoke/replay of a packed-text mix. Writes local results only."""

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mix", type=Path, required=True)
    parser.add_argument("--king-snapshot", type=Path, required=True)
    parser.add_argument("--challenger-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-budget", type=int, default=100000)
    parser.add_argument("--block-hash", default="0x" + "a" * 64)
    parser.add_argument("--hotkey", default="local-gpu-smoke")
    parser.add_argument("--self-duel", action="store_true", help="require the same snapshot and a zero reward")
    args = parser.parse_args()
    from teutonic.evaluation.packed_text import validate_mix
    from teutonic.evaluator import engine
    from teutonic.evaluator.document_duel import run_document_duel
    from teutonic.evaluator.horizon import HorizonDownload

    mix = validate_mix(json.loads(args.mix.read_text()))
    mix = validate_mix({**mix, "token_budget": args.token_budget})
    if args.self_duel and args.king_snapshot.resolve() != args.challenger_snapshot.resolve():
        parser.error("--self-duel requires identical snapshot paths")
    engine._gpu_ids = engine.parse_gpu_ids()
    engine._horizon_downloader = HorizonDownload.from_env(engine.MODEL_CACHE_DIR)
    engine._horizon_downloader.prepare()
    req = engine.EvalRequest(king_repo=str(args.king_snapshot), challenger_repo=str(args.challenger_snapshot),
                             packed_dataset=mix, block_hash=args.block_hash, hotkey=args.hotkey,
                             n_bootstrap=1000, delta_threshold=.0015)
    configs = [engine.load_model_config(str(path), req, role)[0]
               for path, role in [(args.king_snapshot, "king"), (args.challenger_snapshot, "challenger")]]
    if engine.compare_model_configs(*configs):
        raise ValueError("model contracts differ")
    req.max_document_tokens = min(c.max_position_embeddings for c in configs)
    req.vocab_size = configs[0].vocab_size
    for config, role in zip(configs, ["king", "challenger"], strict=True):
        engine.validate_and_report_attention_config(config, role)
    def progress(info):
        print(json.dumps(info), flush=True)
    try:
        verdict, metadata, workers = run_document_duel(engine, req, str(args.king_snapshot),
                                                       str(args.challenger_snapshot), progress)
        args.output.write_text(json.dumps({"verdict": verdict, "dataset": metadata, "workers": workers}, indent=2))
        if args.self_duel and (abs(verdict["mu_hat"]) > 1e-6 or verdict["accepted"]):
            raise ValueError("self-duel failed numerical repeatability")
    finally:
        if engine._model_worker_pool is not None:
            engine._model_worker_pool.close()


if __name__ == "__main__":
    main()
