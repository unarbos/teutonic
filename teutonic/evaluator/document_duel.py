"""Document-only orchestration with disk-cached reference losses and bounded inputs."""

import math
import os
import shutil
import sqlite3
import tempfile
import time
from contextlib import nullcontext
from dataclasses import asdict
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from teutonic.evaluation.masking import DOCUMENT_EOS_TOKEN_ID
from teutonic.evaluation.packed_text import DatasetReader, PackedDataset, digest, read_client
from teutonic.evaluation.reference_policy import reference_verdict
from teutonic.evaluator.horizon import REVISION, file_hash
from teutonic.evaluator.horizon_pipeline import BACKEND as REFERENCE_BACKEND
from teutonic.evaluator.horizon_pipeline import MAX_REQUEST_DOCUMENTS, MAX_REQUEST_TOKENS
from teutonic.evaluator.horizon_pipeline_runtime import HorizonWorker
from teutonic.evaluator.horizon_pipeline_runtime import (
    runtime_identity as reference_runtime_identity,
)
from teutonic.evaluator.mimo_pool import MiMoPool
from teutonic.evaluator.text_pipeline import TextTokenizer, prepared_batches
from teutonic.evaluator.text_sampling import sample_documents
from teutonic.evaluator.vllm_runtime import runtime_identity, wait_for_gpu_release
from teutonic.evaluator.vllm_score import BACKEND as MIMO_BACKEND

# Canonical text plus native EOS; first raw token unscored, EOS scored.
TEXT_POLICY = "raw-text-native-eos-no-bos-first-unscored"


def reference_batches(dataset, documents, tokenizer):
    """Enough micro-batches to fill eight stages, bounded by text and token sizes."""
    for group, sequences in prepared_batches(dataset, documents, tokenizer,
                                             batch_documents=MAX_REQUEST_DOCUMENTS, max_tokens=524287):
        chunk_docs, chunk_tokens, total = [], [], 0
        for doc, tokens in zip(group, sequences, strict=True):
            if chunk_docs and total + len(tokens) > MAX_REQUEST_TOKENS:
                yield chunk_docs, chunk_tokens
                chunk_docs, chunk_tokens, total = [], [], 0
            if len(tokens) > MAX_REQUEST_TOKENS:
                raise ValueError("Horizon document exceeds request bound; no truncation")
            chunk_docs.append(doc)
            chunk_tokens.append(tokens)
            total += len(tokens)
        if chunk_docs:
            yield chunk_docs, chunk_tokens


class ReferenceCache:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS scores (key TEXT PRIMARY KEY, nll REAL NOT NULL, tokens INTEGER NOT NULL)")

    def get(self, key):
        row = self.db.execute("SELECT nll,tokens FROM scores WHERE key=?", (key,)).fetchone()
        if row is None:
            return None
        if not math.isfinite(row[0]) or row[0] < 0 or row[1] < 1:
            raise ValueError("invalid cached reference score")
        return row

    def put(self, key, nll, tokens):
        if not math.isfinite(nll) or nll < 0 or type(tokens) is not int or tokens < 1:
            raise ValueError("invalid reference score")
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO scores VALUES (?,?,?)", (key, nll, tokens))

    def close(self):
        self.db.close()


@lru_cache(maxsize=1)
def backend_versions():
    versions = {}
    for package in ("torch", "transformers", "accelerate", "tokenizers", "gigatoken"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "unavailable"
    return versions


def reference_key(doc, tokenizer_hash, code_hash, runtime=None):
    return digest({"checkpoint": REVISION, "tokenizer": tokenizer_hash, "text": doc.text_sha256,
                   "bytes": doc.byte_length, "text_policy": TEXT_POLICY,
                   "backend": REFERENCE_BACKEND, "scorer_code": code_hash,
                   "dependencies": backend_versions(), "runtime": runtime})


def run_document_duel(base, req, king_snapshot, challenger_snapshot, on_phase):
    king_tokenizer = file_hash(Path(king_snapshot) / "tokenizer.json")
    if king_tokenizer != file_hash(Path(challenger_snapshot) / "tokenizer.json"):
        raise ValueError("king/challenger tokenizer mismatch")
    # One selected corpus on local disk: range-fetch once across the two GPU phases.
    # Automatic cleanup also applies when preparation or inference fails.
    with tempfile.TemporaryDirectory(prefix="duel-text-", dir=base._horizon_downloader.root) as cache_dir:
        try:
            return _run_document_duel(base, req, king_snapshot, challenger_snapshot, on_phase, cache_dir)
        except BaseException:
            # Preserve diagnostic logs, but never retain corpus text after a duel.
            destination = base._horizon_downloader.root / "failed-workers" / Path(cache_dir).name
            for log in Path(cache_dir).glob("*.log"):
                destination.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(log, destination / log.name)
            raise


def _run_document_duel(base, req, king_snapshot, challenger_snapshot, on_phase, cache_dir):
    reader = DatasetReader(read_client())
    mix = req.packed_dataset
    dataset = PackedDataset(reader, mix["dataset"], text_cache=cache_dir)
    catalog = reader.json(mix["sampling"])
    sampling_started = time.monotonic()
    documents, metadata = sample_documents(dataset, catalog, mix, base.dataset_seed(req))
    metadata["sampling_seconds"] = time.monotonic() - sampling_started
    on_phase({"phase": "documents_selected", "total": len(documents), "digest": metadata["digest"]})
    horizon = base._horizon_downloader.wait_ready(on_phase)
    tokenizer_hash = file_hash(horizon / "tokenizer.json")
    runtime = runtime_identity()
    reference_runtime = reference_runtime_identity()
    if len(base._gpu_ids) != 8 or len(set(base._gpu_ids)) != 8:
        raise ValueError("document evaluation requires eight distinct GPUs")
    scorer_hash = digest({"runtime": reference_runtime,
                          "tokenization": file_hash(Path(__file__).with_name("text_pipeline.py"))})
    runtime_hash = digest(reference_runtime)
    keys = [reference_key(d, tokenizer_hash, scorer_hash, runtime_hash) for d in documents]
    # Retire any baseline workers before allocating the vLLM engines.
    if base._model_worker_pool is not None:
        base._model_worker_pool.close()
        base._model_worker_pool = None
    resident = getattr(base, "_idle_horizon", None)
    if resident is None:
        wait_for_gpu_release(base._gpu_ids)
    cache = ReferenceCache(base._horizon_downloader.root / "reference-losses.sqlite")
    reports = {}
    tok = None
    reference_started = time.monotonic()
    try:
        # Identical text has the same reference key; score it only once, retaining
        # every sampled document independently in the final paired verdict.
        missing = {}
        for doc, key in zip(documents, keys, strict=True):
            if cache.get(key) is None:
                missing.setdefault(key, doc)
        if missing:
            on_phase({"phase": "reference_prepare", "missing_documents": len(missing)})
            tok = TextTokenizer(horizon, backend=os.environ.get("TEUTONIC_REFERENCE_TOKENIZER_BACKEND", "gigatoken"))
            key_by_id = {doc.doc_id: key for key, doc in missing.items()}
            try:
                lease = (nullcontext(resident.acquire(horizon, reference_runtime, on_phase)) if resident is not None
                         else HorizonWorker("horizon", horizon, base._gpu_ids, cache_dir, reference_runtime, on_phase))
                with lease as worker:
                    on_phase({"phase": "reference_scoring", "missing_documents": len(missing)})
                    for group, sequences in reference_batches(dataset, list(missing.values()), tok):
                        for row in worker.score(group, sequences):
                            cache.put(key_by_id[row["id"]], row["nll"], row["scored_tokens"])
                    reports["horizon"] = dict(worker.report)
            except BaseException:
                if resident is not None:
                    log_path = resident.directory / "horizon.log"
                    if log_path.exists():
                        shutil.copyfile(log_path, Path(cache_dir) / "horizon.log")
                raise
        reference = [cache.get(key)[0] for key in keys]
    finally:
        cache.close()
        # Also unload a preloaded Horizon when all reference scores were cached.
        if resident is not None:
            resident.release()
        wait_for_gpu_release(base._gpu_ids)
    metadata["reference_phase_seconds"] = time.monotonic() - reference_started
    tokenizer = TextTokenizer(king_snapshot, eos_token_id=DOCUMENT_EOS_TOKEN_ID,
                              backend=os.environ.get("TEUTONIC_TOKENIZER_BACKEND", "gigatoken"))
    king, challenger, counts = [], [], []
    maximum_length = 0
    started = time.monotonic()
    with MiMoPool(king_snapshot, challenger_snapshot, base._gpu_ids,
                  cache_dir, runtime, on_phase) as pool:
        batches = prepared_batches(dataset, documents, tokenizer, batch_documents=128,
                                   max_tokens=req.max_document_tokens - 1)
        for group, sequences in batches:
            maximum_length = max(maximum_length, max(map(len, sequences)))
            k, c = pool.score(group, sequences)
            king.extend(row["nll"] for row in k)
            challenger.extend(row["nll"] for row in c)
            counts.extend(len(t) - 1 for t in sequences)
            on_phase({"phase": "paired_scoring", "done": len(king), "total": len(documents)})
    reports["mimo_replicas"] = pool.reports
    combined_runtime = {"horizon": reference_runtime, "mimo": runtime}
    workers = {"backend": {"horizon": REFERENCE_BACKEND, "mimo": MIMO_BACKEND}, "runtime": combined_runtime, "roles": reports,
               "model_parallel_strategy": "replicas_with_large_document_tp2",
               "scored_tokens": counts}
    verdict = reference_verdict(king, challenger, reference, [d.byte_length for d in documents],
                                bootstrap_seed=req.bootstrap_seed, n_bootstrap=req.n_bootstrap,
                                alpha=req.alpha, delta_threshold=req.delta_threshold)
    # Preserve ordinary token-weighted NLL for the historical loss chart.
    # These diagnostics never enter the Horizon-deficit reward/bootstrap.
    verdict.update(avg_king_token_loss=math.fsum(king) / sum(counts),
                   avg_challenger_token_loss=math.fsum(challenger) / sum(counts))
    metadata.update(reference_revision=REVISION, reference_misses=len(missing),
                    reference_backend=REFERENCE_BACKEND, dependencies=backend_versions(), scorer_runtime=combined_runtime,
                    tokenization={"mimo": tokenizer.status(), "horizon": tok.status() if tok else {"cached": True}},
                    text_policy=TEXT_POLICY, max_seq_len=maximum_length, paired_scoring_seconds=time.monotonic()-started)
    verdict["sample_results"] = {"format": "packed-document-losses", "documents": [asdict(d) for d in documents],
                                 "king_nll": king, "challenger_nll": challenger,
                                 "reference_nll": reference, "scored_tokens": counts}
    return verdict, metadata, workers
