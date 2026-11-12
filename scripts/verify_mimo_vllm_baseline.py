#!/usr/bin/env python3
"""Save current paired MiMo scorer results on existing local CodeIO fixtures."""
import argparse
import hashlib
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

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--fixtures", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--synthetic-lengths", type=int, nargs="+", default=[127, 128, 129, 512, 16384])
    args = p.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    snapshot = str(args.snapshot.resolve())
    tokenizer = TextTokenizer(args.snapshot, eos_token_id=DOCUMENT_EOS_TOKEN_ID)
    fixtures = json.loads(args.fixtures.read_text())
    texts = [r["text"] for r in fixtures]
    tokens = tokenizer.encode(texts)
    rows = [{"id": hashlib.sha256(t.encode()).hexdigest(), "tokens": ids,
             "scored_tokens": len(ids)-1, "byte_length": len(t.encode()), "kind": "codeio"}
            for t, ids in zip(texts, tokens, strict=True)]
    # Synthetic boundary/embedded-EOS cases supplement, not replace, real data.
    seed = tokenizer.encode(["The sky is blue. Zażółć gęślą jaźń. 你好世界. 2 + 2 = 4.\n"])[0][:-1]
    for length in args.synthetic_lengths:
        ids = (seed * ((length - 1)//len(seed) + 1))[:length-1] + [DOCUMENT_EOS_TOKEN_ID]
        ids[length//2] = DOCUMENT_EOS_TOKEN_ID
        rows.append({"id": f"synthetic-embedded-eos-{length}", "tokens": ids,
                     "scored_tokens": len(ids)-1, "byte_length": None, "kind": "synthetic"})
    if max(len(r["tokens"]) for r in rows) > 16384:
        raise ValueError("fixture outside intended 16K validation range")
    (args.output/"inputs-unscored.json").write_text(json.dumps(rows))
    req = engine.EvalRequest(king_repo=snapshot, challenger_repo=snapshot, batch_size=4,
                            batch_token_budget=8192, long_documents={"packed_text": True},
                            attn_implementation="flash_attention_4")
    config, _ = engine.load_model_config(snapshot, req, "king")
    req.vocab_size = config.vocab_size
    req.max_document_tokens = config.max_position_embeddings
    engine.validate_and_report_attention_config(config, "king")
    engine.configure_deterministic_execution()
    ids = [r["tokens"] for r in rows]
    emit = lambda event: print(json.dumps(event), flush=True)
    start = time.monotonic()
    try:
        with engine.loading_model_workers(req, snapshot, snapshot, engine.parse_gpu_ids(), emit) as (pool, ready):
            generation = ready.result()
            loaded = time.monotonic()
            king, challenger, metadata = pool.score(ids, generation, req, emit)
            scored = time.monotonic()
            if king != challenger:
                raise AssertionError("current scorer self-duel is not exact")
            if metadata["scored_tokens"] != [r["scored_tokens"] for r in rows]:
                raise AssertionError("target counts differ")
            for row, loss in zip(rows, king, strict=True):
                row["baseline_nats_per_token"] = loss
                row["baseline_nll"] = loss * row["scored_tokens"]
            (args.output/"inputs.json").write_text(json.dumps(rows))
            # Warm same-input timing and repeatability with all workers resident.
            again = time.monotonic()
            rk, rc, rm = pool.score(ids, generation, req, emit)
            repeat_seconds = time.monotonic() - again
            report = {"snapshot": snapshot, "attention": req.attn_implementation, "documents": len(rows),
                      "scored_tokens": sum(r["scored_tokens"] for r in rows),
                      "load_seconds": loaded-start, "score_seconds": scored-loaded,
                      "repeat_seconds": repeat_seconds, "repeat_exact": rk == king and rc == challenger,
                      "king": king, "challenger": challenger, "metadata": metadata,
                      "repeat_metadata": rm, "tokenizer": tokenizer.status(),
                      "config_sha256": hashlib.sha256((args.snapshot/"config.json").read_bytes()).hexdigest()}
            (args.output/"report.json").write_text(json.dumps(report, indent=2))
            if not report["repeat_exact"]:
                raise AssertionError("current scorer repeat mismatch")
            emit({"phase": "baseline_complete", **{k:v for k,v in report.items() if k not in
                  {"metadata", "repeat_metadata", "king", "challenger"}}})
    finally:
        if engine._model_worker_pool is not None:
            engine._model_worker_pool.close(force=True)


if __name__ == "__main__":
    main()
