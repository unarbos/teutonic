#!/usr/bin/env python3
"""Eight-replica local MiMo self-duel smoke test; no R2 or model promotion."""
import argparse
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from teutonic.evaluation.masking import DOCUMENT_EOS_TOKEN_ID
    from teutonic.evaluator import engine
    from teutonic.evaluator.text_pipeline import TextTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[32, 64, 128, 256, 512, 1024, 2048, 4096])
    parser.add_argument("--embedded-eos", action="store_true")
    parser.add_argument("--additional-embedded-eos", action="store_true")
    args = parser.parse_args()
    snapshot = str(args.snapshot.resolve())
    req = engine.EvalRequest(king_repo=snapshot, challenger_repo=snapshot, batch_size=4,
                             batch_token_budget=8192, long_documents={"packed_text": True})
    config, _ = engine.load_model_config(snapshot, req, "king")
    req.vocab_size = config.vocab_size
    req.max_document_tokens = config.max_position_embeddings
    engine.validate_and_report_attention_config(config, "king")
    engine.configure_deterministic_execution()
    gpu_ids = engine.parse_gpu_ids()
    tokenizer = TextTokenizer(args.snapshot, eos_token_id=DOCUMENT_EOS_TOKEN_ID)
    base = tokenizer.encode(["The sky is blue. Zażółć gęślą jaźń. 你好世界. 2 + 2 = 4.\n"])[0][:-1]
    sequences = [(base * ((length - 1) // len(base) + 1))[:length - 1] + [tokenizer.eos_token_id]
                 for length in args.lengths]
    if args.embedded_eos:
        for sequence in sequences:
            if len(sequence) > 3:
                sequence[len(sequence) // 2] = tokenizer.eos_token_id
    if any(not 2 <= len(seq) <= req.max_document_tokens for seq in sequences):
        raise ValueError("invalid smoke-test sequence lengths")
    started = time.monotonic()
    try:
        with engine.loading_model_workers(req, snapshot, snapshot, gpu_ids,
                                           lambda event: print(json.dumps(event), flush=True)) as (pool, ready):
            generation = ready.result()
            loaded = time.monotonic()
            king, challenger, metadata = pool.score(sequences, generation, req,
                lambda event: print(json.dumps(event), flush=True))
            elapsed = time.monotonic() - loaded
            embedded = None
            if args.additional_embedded_eos:
                extra = [list(sequence) for sequence in sequences if len(sequence) > 3]
                for sequence in extra:
                    sequence[len(sequence) // 2] = tokenizer.eos_token_id
                extra_started = time.monotonic()
                ek, ec, em = pool.score(extra, generation, req,
                    lambda event: print(json.dumps(event), flush=True))
                if max(abs(a-b) for a, b in zip(ek, ec, strict=True)) > 1e-6:
                    raise AssertionError("embedded-EOS self-duel mismatch")
                if em["scored_tokens"] != [len(sequence)-1 for sequence in extra]:
                    raise AssertionError("embedded-EOS target accounting differs")
                embedded = {"king": ek, "challenger": ec, "metadata": em,
                            "seconds": time.monotonic() - extra_started}
        errors = [abs(a-b) for a, b in zip(king, challenger, strict=True)]
        if max(errors) > 1e-6:
            raise AssertionError(f"self-duel mismatch: {errors}")
        if metadata["scored_tokens"] != [len(seq) - 1 for seq in sequences]:
            raise AssertionError("whole-document scored-token accounting differs")
        report = {"snapshot": snapshot, "input": "synthetic repeated multilingual text",
                  "lengths": args.lengths, "king": king, "challenger": challenger,
                  "embedded_eos": args.embedded_eos,
                  "additional_embedded_eos": embedded,
                  "attention": req.attn_implementation,
                  "max_absolute_error": max(errors), "load_seconds": loaded - started,
                  "score_seconds": elapsed, "metadata": metadata}
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"report": str(args.report), "max_absolute_error": max(errors),
                          "score_seconds": elapsed}), flush=True)
    finally:
        if engine._model_worker_pool is not None:
            engine._model_worker_pool.close(force=True)


if __name__ == "__main__":
    main()
