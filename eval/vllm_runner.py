"""vLLM-backed evaluator for paired per-token CE on pretokenized shards.

Drop-in replacement for `eval_torch.MultiGPUEvaluator` + `compute_paired_multi_gpu`
when both king and challenger are vLLM-supported architectures (e.g. Qwen3,
Qwen3-Next, Llama, Mixtral, DeepSeek-V3). Reuses `eval_torch`'s shard plumbing
and bootstrap math verbatim (`fetch_sequences`, `download_shard`,
`get_shard_info`, `extract_sequences`) so semantics match production:

  per-sequence loss = -mean_{i=1..seq_len-1} log p_model(token_i | token_<i)
  d_i               = king_loss_i - challenger_loss_i
  accept iff lcb(d, alpha) > EVAL_DELTA  (default 0.0025 nats/token)

The forward pass is a vLLM prefill-only request (`max_tokens=1`) with
`prompt_logprobs=0`, which causes vLLM to return the model's logprob of every
prompt token. We sum `-logprob` from position 1 onward, divide by `seq_len-1`,
and the result is identical to `compute_batch_losses`'s output up to numerics.

Topology assumed by the server: each `VllmEvaluator` runs in its own child
process with `CUDA_VISIBLE_DEVICES` pinned to its slice of GPUs, and the two
evaluators score the same batches concurrently from the parent thread.
"""
from __future__ import annotations

import hashlib
import logging
import multiprocessing as mp
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

from model_store import ModelRef, materialize_model
from .torch_runner import (
    EVAL_DELTA,
    download_shard,
    extract_sequences,
    get_shard_info,
    validate_sequence_cache,
)
from .raw_dataset import load_raw_sequences, raw_dataset_enabled
import chain_config

log = logging.getLogger("eval_vllm")


VLLM_DTYPE = os.environ.get("EVAL_VLLM_DTYPE", "bfloat16")
VLLM_GPU_MEM_UTIL = float(os.environ.get("EVAL_VLLM_GPU_MEM_UTIL", "0.90"))
VLLM_ENFORCE_EAGER = os.environ.get("EVAL_VLLM_ENFORCE_EAGER", "0") == "1"
VLLM_MAX_BATCH = int(os.environ.get("EVAL_VLLM_MAX_BATCH", "256"))
VLLM_TRUST_REMOTE_CODE = os.environ.get("EVAL_VLLM_TRUST_REMOTE_CODE", "1") == "1"


# ---------------------------------------------------------------------------
# Worker — runs in a child process with CUDA_VISIBLE_DEVICES pinned
# ---------------------------------------------------------------------------

@dataclass
class _Cmd:
    op: str
    payload: Any = None


def _worker_main(
    repo: str,
    digest: str | None,
    gpu_ids: list[int],
    seq_len: int,
    label: str,
    cmd_q: "mp.Queue[_Cmd]",
    res_q: "mp.Queue[Any]",
) -> None:
    """Long-lived worker: load one vLLM engine, then handle scoring requests."""
    visible = ",".join(str(g) for g in gpu_ids)
    os.environ["CUDA_VISIBLE_DEVICES"] = visible

    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s eval_vllm[{label}:%(process)d] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    wlog = logging.getLogger(f"eval_vllm.{label}")

    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        res_q.put({"ok": False, "error": f"vllm not installed: {exc}"})
        return

    try:
        wlog.info(
            "loading %s@%s on GPUs %s tp=%d",
            repo, (digest or "missing")[:19], gpu_ids, len(gpu_ids),
        )
        t0 = time.time()
        local_model = materialize_model(ModelRef(repo, digest or ""), max_workers=16)
        llm = LLM(
            model=local_model,
            tensor_parallel_size=len(gpu_ids),
            dtype=VLLM_DTYPE,
            trust_remote_code=VLLM_TRUST_REMOTE_CODE,
            max_model_len=seq_len,
            gpu_memory_utilization=VLLM_GPU_MEM_UTIL,
            enforce_eager=VLLM_ENFORCE_EAGER,
            disable_log_stats=True,
            seed=0,
        )
        load_s = time.time() - t0
        wlog.info("engine ready in %.1fs", load_s)
        res_q.put({"ok": True, "load_s": load_s})
    except Exception as exc:
        wlog.exception("engine load failed")
        res_q.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        return

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        prompt_logprobs=0,
        detokenize=False,
    )

    while True:
        cmd = cmd_q.get()
        if cmd.op == "shutdown":
            wlog.info("shutdown received")
            res_q.put({"ok": True})
            return

        if cmd.op == "score":
            seqs: list[list[int]] = cmd.payload["seqs"]
            try:
                losses = _score_seqs(llm, sampling, seqs)
                res_q.put({"ok": True, "losses": losses})
            except Exception as exc:
                wlog.exception("score failed")
                res_q.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            continue

        res_q.put({"ok": False, "error": f"unknown op: {cmd.op}"})


def _score_seqs(llm, sampling, seqs: list[list[int]]) -> list[float]:
    """Run a prefill-only generate, extract per-seq mean nats from prompt_logprobs.

    Mirrors `eval_torch.compute_batch_losses`: mean over positions 1..seq_len-1
    of `-log p_model(token_i | token_<i)`. Returns one float per input sequence.
    """
    if not seqs:
        return []

    prompts = [{"prompt_token_ids": list(s)} for s in seqs]
    outputs = llm.generate(prompts, sampling, use_tqdm=False)

    out_by_idx = {int(o.request_id): o for o in outputs} if all(
        getattr(o, "request_id", None) is not None and str(o.request_id).isdigit()
        for o in outputs
    ) else None

    losses: list[float] = []
    for i, seq in enumerate(seqs):
        out = outputs[i] if out_by_idx is None else out_by_idx.get(i, outputs[i])
        plp = out.prompt_logprobs
        if plp is None or len(plp) != len(seq):
            raise RuntimeError(
                f"prompt_logprobs missing/length mismatch: got "
                f"{None if plp is None else len(plp)}, expected {len(seq)}"
            )

        nll_sum = 0.0
        n_pos = 0
        for pos in range(1, len(seq)):
            entry = plp[pos]
            if entry is None:
                continue
            tok = int(seq[pos])
            lp_obj = entry.get(tok)
            if lp_obj is None:
                continue
            nll_sum += -float(lp_obj.logprob)
            n_pos += 1

        if n_pos == 0:
            raise RuntimeError("no valid prompt_logprobs positions for sequence")
        losses.append(nll_sum / n_pos)

    return losses


# ---------------------------------------------------------------------------
# Parent-side handle
# ---------------------------------------------------------------------------

class VllmEvaluator:
    """Handle for a child-process vLLM engine pinned to specific GPUs.

    Mirrors the public surface of `eval_torch.MultiGPUEvaluator` enough to be
    used in the same paired bootstrap loop:
      - `gpu_ids`: GPU indices reserved for this engine
      - `compute_losses(token_batches)`: list of mean nats per sequence
      - `shutdown()`: terminate the worker

    `token_batches` is a list of lists-of-batches as in eval_torch
    (`compute_paired_multi_gpu`'s contract): each element is a list of
    `seq_len`-long token id lists (one batch). Returns the concatenated
    per-sequence loss list in input order.
    """

    def __init__(
        self,
        repo: str,
        gpu_ids: list[int],
        seq_len: int,
        label: str = "model",
        digest: str | None = None,
        on_phase=None,
    ) -> None:
        self.repo = repo
        self.digest = digest
        self.gpu_ids = list(gpu_ids)
        self.seq_len = seq_len
        self.label = label

        ctx = mp.get_context("spawn")
        self._cmd_q: mp.Queue[_Cmd] = ctx.Queue()
        self._res_q: mp.Queue[Any] = ctx.Queue()
        self._proc = ctx.Process(
            target=_worker_main,
            args=(repo, digest, self.gpu_ids, seq_len, label,
                  self._cmd_q, self._res_q),
            name=f"vllm-{label}-{repo[:24]}",
            daemon=False,
        )

        if on_phase:
            try:
                on_phase({"phase": f"{label}_load_start", "repo": repo,
                          "gpus": self.gpu_ids})
            except Exception:
                log.warning("on_phase callback raised (non-fatal)", exc_info=True)

        t0 = time.time()
        self._proc.start()
        ready = self._res_q.get()
        if not ready.get("ok"):
            self._proc.join(timeout=5)
            raise RuntimeError(f"vllm worker failed to start: {ready.get('error')}")
        log.info("vllm %s evaluator ready (%.1fs wall, %.1fs in worker)",
                 label, time.time() - t0, ready.get("load_s", 0.0))

        if on_phase:
            try:
                on_phase({"phase": f"{label}_load_done", "repo": repo,
                          "gpus": self.gpu_ids})
            except Exception:
                log.warning("on_phase callback raised (non-fatal)", exc_info=True)

    def compute_losses(self, token_batches: list[list[list[int]]]) -> list[float]:
        """Process a list of batches, return one loss per sequence in flat order.

        Each entry in `token_batches` is a batch of sequences (lists of int
        token ids). Inside this method we further chunk to `VLLM_MAX_BATCH`
        for engine-side memory hygiene, but the input order is preserved.
        """
        if not token_batches:
            return []
        flat: list[list[int]] = []
        for b in token_batches:
            flat.extend(list(s) for s in b)

        results: list[float] = []
        for i in range(0, len(flat), VLLM_MAX_BATCH):
            chunk = flat[i : i + VLLM_MAX_BATCH]
            self._cmd_q.put(_Cmd(op="score", payload={"seqs": chunk}))
            res = self._res_q.get()
            if not res.get("ok"):
                raise RuntimeError(f"vllm score failed: {res.get('error')}")
            results.extend(res["losses"])

        return results

    def shutdown(self) -> None:
        if not self._proc.is_alive():
            return
        try:
            self._cmd_q.put(_Cmd(op="shutdown"))
            try:
                self._res_q.get(timeout=30)
            except Exception:
                pass
            self._proc.join(timeout=30)
        finally:
            if self._proc.is_alive():
                log.warning("vllm %s worker still alive after shutdown, killing",
                            self.label)
                self._proc.kill()
                self._proc.join(timeout=10)


# ---------------------------------------------------------------------------
# Concurrent paired scoring across two evaluators
# ---------------------------------------------------------------------------

def compute_paired_vllm(
    king: VllmEvaluator,
    chall: VllmEvaluator,
    token_batches: list[list[list[int]]],
) -> tuple[list[float], list[float]]:
    """Score the same `token_batches` on king and challenger concurrently.

    Equivalent to `eval_torch.compute_paired_multi_gpu` but each side runs in
    its own subprocess against its own pinned GPU group. Returns
    (king_losses, challenger_losses), both flat in the input sequence order.
    """
    if not token_batches:
        return [], []
    if king is chall:
        losses = king.compute_losses(token_batches)
        return losses, list(losses)

    with ThreadPoolExecutor(max_workers=2) as pool:
        fk = pool.submit(king.compute_losses, token_batches)
        fc = pool.submit(chall.compute_losses, token_batches)
        return fk.result(), fc.result()


# ---------------------------------------------------------------------------
# Bootstrap test (mirrors eval_torch.run_bootstrap_test, vLLM-flavored)
# ---------------------------------------------------------------------------

def run_bootstrap_test_vllm(
    king_eval: VllmEvaluator,
    challenger_eval: VllmEvaluator,
    r2,
    shard_key: str,
    eval_n: int,
    alpha: float,
    seq_len: int,
    batch_size: int,
    seed_str: str,
    n_bootstrap: int = 10000,
    on_progress=None,
) -> dict:
    """Paired bootstrap test on per-token log-loss differences (vLLM backend).

    Identical statistical contract to `eval_torch.run_bootstrap_test`:
      - same shard, same per-(block_hash, hotkey) seed -> indices
      - same per-token mean nats over (seq_len - 1) positions
      - same paired LCB at level `alpha`
      - same accept rule: lcb > EVAL_DELTA (default 0.0025 nats/token)

    The only difference is the inner forward pass: vLLM prefill in two
    subprocess engines instead of HF teacher forcing on resident models.
    """
    delta = EVAL_DELTA

    seed = int.from_bytes(
        hashlib.blake2b(seed_str.encode(), digest_size=8).digest(), "little",
    )
    raw_meta = None
    if raw_dataset_enabled():
        log.info("bootstrap[vllm] using raw Hippius dataset mode")
        raw_sequences, raw_meta = load_raw_sequences(
            r2, eval_n, seq_len, seed_str, chain_config.SEED_TOKENIZER_REPO,
        )
        actual_N = len(raw_sequences)
        eval_indices = list(range(actual_N))
        seq_cache = dict(enumerate(raw_sequences))
        log.info(
            "bootstrap[vllm] N=%d actual_N=%d alpha=%s delta=%.6f B=%d dataset=%s",
            eval_n, actual_N, alpha, delta, n_bootstrap, raw_meta.get("prefix"),
        )
    else:
        n_tokens = get_shard_info(r2, shard_key)
        n_sequences = n_tokens // seq_len
        actual_N = min(eval_n, n_sequences)
        log.info("bootstrap[vllm] N=%d actual_N=%d alpha=%s delta=%.6f B=%d",
                 eval_n, actual_N, alpha, delta, n_bootstrap)

        rng = np.random.Generator(np.random.PCG64(seed))
        eval_indices = rng.choice(n_sequences, size=actual_N, replace=False).tolist()

        log.info("downloading shard %s ...", shard_key)
        data_offset, shard_data = download_shard(r2, shard_key)

        log.info("extracting %d sequences", actual_N)
        seq_cache = extract_sequences(shard_data, data_offset, eval_indices, seq_len)
        validate_sequence_cache(seq_cache, seq_len)
        log.info("extracted %d sequences", len(seq_cache))

    batches = [
        eval_indices[i : i + batch_size]
        for i in range(0, len(eval_indices), batch_size)
    ]

    all_diffs: list[float] = []
    king_sum = 0.0
    chall_sum = 0.0
    total_done = 0
    t0 = time.time()

    same_evaluator = king_eval is challenger_eval

    for bi, batch_indices in enumerate(batches):
        token_batch = [seq_cache[idx] for idx in batch_indices]
        if same_evaluator:
            k_losses = king_eval.compute_losses([token_batch])
            c_losses = list(k_losses)
        else:
            k_losses, c_losses = compute_paired_vllm(
                king_eval, challenger_eval, [token_batch],
            )

        for k_loss, c_loss in zip(k_losses, c_losses):
            total_done += 1
            king_sum += k_loss
            chall_sum += c_loss
            all_diffs.append(k_loss - c_loss)

        elapsed = time.time() - t0
        seqs_per_sec = total_done / elapsed if elapsed > 0 else 0.0
        mu_hat = float(np.mean(all_diffs)) if all_diffs else 0.0
        log.info(
            "batch %d/%d | done=%d/%d | mu_hat=%.6f | %.1f seq/s",
            bi + 1, len(batches), total_done, actual_N, mu_hat, seqs_per_sec,
        )
        if on_progress:
            on_progress({
                "done": total_done, "total": actual_N,
                "mu_hat": round(mu_hat, 6),
                "avg_king_loss": round(king_sum / total_done, 6),
                "avg_challenger_loss": round(chall_sum / total_done, 6),
                "seqs_per_sec": round(seqs_per_sec, 1),
            })

    elapsed = time.time() - t0
    d = np.array(all_diffs)
    mu_hat = float(d.mean())

    boot_rng = np.random.Generator(np.random.PCG64(seed ^ 0xB007))
    boot_means = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = boot_rng.integers(0, len(d), size=len(d))
        boot_means[b] = d[idx].mean()
    lcb = float(np.quantile(boot_means, alpha))

    accepted = lcb > delta
    log.info("bootstrap[vllm] mu_hat=%.6f lcb=%.6f delta=%.6f accepted=%s",
             mu_hat, lcb, delta, accepted)

    verdict = {
        "accepted": accepted,
        "verdict": "challenger" if accepted else "king",
        "mu_hat": round(mu_hat, 6),
        "lcb": round(lcb, 6),
        "delta": delta,
        "alpha": alpha,
        "n_bootstrap": n_bootstrap,
        "N": actual_N,
        "avg_king_loss": round(king_sum / total_done, 6) if total_done else 0,
        "avg_challenger_loss": round(chall_sum / total_done, 6) if total_done else 0,
        "wall_time_s": round(elapsed, 1),
        "seqs_per_sec": round(total_done / elapsed, 1) if elapsed > 0 else 0,
        "backend": "vllm",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if raw_meta is not None:
        verdict["dataset"] = raw_meta
    return verdict
