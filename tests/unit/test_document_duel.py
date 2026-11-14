import hashlib
import json
import math
import threading
from types import SimpleNamespace

import pytest
from test_packed_text import fixture

from teutonic.evaluation.packed_text import canonical
from teutonic.evaluator import document_duel as duel
from teutonic.evaluator import idle_horizon, mimo_pool
from teutonic.evaluator.model_downloads import ModelDownloads


@pytest.mark.parametrize("failure", [None, "horizon", "challenger"])
@pytest.mark.parametrize("preloaded", [False, True])
@pytest.mark.parametrize("deferred", [False, True])
def test_cache_miss_hit_and_shared_text_with_read_only_r2(tmp_path, monkeypatch, failure, preloaded, deferred):
    reference_gate, download_started, prepared = threading.Event(), threading.Event(), threading.Event()
    client, _, catalog, mix = fixture()
    raw = canonical(catalog)
    client.objects[mix['sampling']['key']] = raw
    mix['sampling']['sha256'] = hashlib.sha256(raw).hexdigest()
    snapshot = tmp_path / 'model'
    snapshot.mkdir()
    (snapshot / 'tokenizer.json').write_text('{}')
    monkeypatch.setattr(duel, 'read_client', lambda **kwargs: client)
    original_batches = duel.prepared_batches
    def small_batches(*args, **kwargs):
        return original_batches(*args, **{**kwargs, 'batch_documents': 2})
    monkeypatch.setattr(duel, 'prepared_batches', small_batches)
    events = []
    tokenizer_options = []
    class Tokenizer:
        def __init__(self, *args, **kwargs):
            tokenizer_options.append(kwargs)
            self.horizon = 'eos_token_id' not in kwargs
        def status(self):
            return {"backend": "fixture"}
        def encode(self, texts):
            return [[3, 4, 5, 6, 7] if self.horizon else [3, 4, 5] for text in texts]
    monkeypatch.setattr(duel, 'TextTokenizer', Tokenizer)
    subprocess_calls, closed = [], []
    mimo_identity = {'backend': 'fixture-vllm'}
    monkeypatch.setattr(duel, 'runtime_identity', lambda: mimo_identity)
    reference_identity = {'backend': 'fixture-pipeline'}
    monkeypatch.setattr(duel, 'reference_runtime_identity', lambda: reference_identity)
    monkeypatch.setattr(duel, 'wait_for_gpu_release', lambda ids: None)
    class ModelWorker:
        def __init__(self, role, snapshot, ids, directory, identity, on_phase, **kwargs):
            assert ids == list(range(8)) if role == 'horizon' else (len(ids) == 1 and ids[0] in (range(4) if role == 'king' else range(4, 8)))
            assert identity == (reference_identity if role == 'horizon' else mimo_identity)
            assert old_pool.closed
            if deferred and role != "horizon":
                assert prepared.is_set()
            if role == 'horizon':
                subprocess_calls.append(role)
            self.role = role
            self.identity = identity
            self.closed = False
            self.process = SimpleNamespace(poll=lambda: 0 if self.closed else None)
            self.report = {'role': role}
            if role == failure:
                from pathlib import Path
                (Path(directory) / f"{role}.log").write_text("fixture scorer failure")
        def score(self, documents, sequences):
            if deferred and self.role == "horizon":
                assert download_started.is_set()
                reference_gate.set()  # A real resolver thread was blocked until inference.
            if self.role == failure:
                raise RuntimeError("fixture scorer failure")
            return [{'id': d.doc_id, 'nll': 4.0 if self.role == 'king' else 2.0,
                     'scored_tokens': len(t) - 1} for d, t in zip(documents, sequences, strict=True)]
        def wait_ready(self):
            pass
        def close(self):
            self.closed = True
            closed.append(self.role)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            closed.append(self.role)
    monkeypatch.setattr(duel, 'HorizonWorker', ModelWorker)
    monkeypatch.setattr(mimo_pool, 'Worker', ModelWorker)
    monkeypatch.setattr(mimo_pool, 'wait_for_gpu_release', lambda ids: None)
    class Pool:
        closed = False
        def close(self):
            self.closed = True
    old_pool = Pool()
    base = SimpleNamespace(_model_worker_pool=old_pool, _gpu_ids=list(range(8)),
                           _horizon_downloader=SimpleNamespace(root=tmp_path, wait_ready=lambda callback: snapshot),
                           dataset_seed=lambda req: 123)
    def preload():
        if preloaded:
            base._idle_horizon.preload()
            for future in base._idle_horizon._preloads:
                future.result(timeout=5)
            assert base._idle_horizon.status()['state'] == 'ready'
    if preloaded:
        old_pool.close()
        monkeypatch.setattr(idle_horizon, 'HorizonWorker', ModelWorker)
        monkeypatch.setattr(idle_horizon, 'wait_for_gpu_release', lambda ids: None)
        base._idle_horizon = idle_horizon.IdleHorizon(base._horizon_downloader, base._gpu_ids, reference_identity)
        # Any wait for empty GPUs during a duel must happen AFTER resident unload.
        def released(ids):
            assert base._idle_horizon._worker is None
        monkeypatch.setattr(duel, 'wait_for_gpu_release', released)
    req = SimpleNamespace(packed_dataset=mix, batch_size=4, max_document_tokens=16384,
                          bootstrap_seed=0, n_bootstrap=100, alpha=.05, delta_threshold=.0015)
    def run():
        events.clear()
        if not deferred:
            return duel.run_document_duel(base, req, snapshot, snapshot, events.append)
        prepared.clear()
        def resolve(artifact):
            download_started.set()
            assert reference_gate.wait(5), "reference inference did not overlap downloads"
            return snapshot
        with ModelDownloads(SimpleNamespace(resolve=resolve), "king", "challenger") as downloads:
            assert download_started.wait(5)
            def ready():
                paths = downloads.result()
                assert closed.count("horizon") == len(subprocess_calls)
                prepared.set()
                return paths["king"], paths["challenger"]
            return duel.run_document_duel(base, req, None, None, events.append,
                                          prepare_models=ready, check_preparation=downloads.check)

    if failure:
        preload()
        with pytest.raises(RuntimeError, match="fixture scorer failure"):
            run()
        assert not list(tmp_path.glob('duel-text-*'))
        assert failure in closed
        if failure == "challenger":
            assert "king" in closed
        logs = list((tmp_path / "failed-workers").rglob("*.log"))
        assert len(logs) == 1 and logs[0].read_text() == "fixture scorer failure"
        if preloaded:
            base._idle_horizon.close()
        return
    previous_samples = None
    for expected_misses in [True, False]:
        preload()
        client.calls.clear()
        verdict, metadata, _ = run()
        assert verdict['accepted']
        # Fixture documents have two scored targets: raw NLL stays separate from BPB.
        assert verdict['avg_king_token_loss'] == 2.0
        assert verdict['avg_challenger_token_loss'] == 1.0
        assert verdict['loss_unit'] == 'bits/byte'
        samples = verdict['sample_results']
        size = len(samples['documents'])
        assert size == metadata['n_documents']
        assert samples['scoring_policy'] == verdict['scoring_policy'] == 'horizon-deficit-active-bpb'
        assert samples['sample_filter'] == 'exclude-both-zero-deficits'
        assert sum(samples['scoring_included']) == verdict['n_scoring_sequences']
        assert [e['percent'] for e in events] == sorted(e['percent'] for e in events)
        reference_events = [e for e in events if e['phase'] == 'reference_scoring']
        assert reference_events[-1]['percent'] == 50
        assert reference_events[-1]['done'] == size
        if not expected_misses:
            assert reference_events[0]['done'] == size
        checkpoints = [e for e in events if e['phase'] == 'paired_scoring' and 'provisional_lcb' in e]
        assert len(checkpoints) > 1
        assert checkpoints[-1]['percent'] == 100
        assert checkpoints[-1]['provisional_lcb'] == verdict['lcb']
        assert checkpoints[-1]['provisional_mu_hat'] == verdict['mu_hat']
        for event in checkpoints:
            n = event['provisional_n_sequences']
            expected = duel.reference_verdict(
                samples['king_nll'][:n], samples['challenger_nll'][:n], samples['reference_nll'][:n],
                [d['byte_length'] for d in samples['documents'][:n]],
                bootstrap_seed=req.bootstrap_seed, n_bootstrap=req.n_bootstrap,
                alpha=req.alpha, delta_threshold=req.delta_threshold,
            )
            assert event['provisional_lcb'] == expected['lcb']
            assert event['provisional_n_bootstrap'] == req.n_bootstrap
        assert samples['scored_tokens'] == [2] * size
        assert samples['reference_scored_tokens'] == [4] * size
        assert samples['reference_model'] == {'repository': duel.REPO, 'revision': duel.REVISION}
        assert samples['units'] == {'nll': 'nats', 'token_loss': 'nats/token', 'bits_per_byte': 'bits/byte'}
        for role, nll, token_loss in [('king', 4.0, 2.0), ('challenger', 2.0, 1.0), ('reference', 2.0, 0.5)]:
            assert samples[f'{role}_nll'] == [nll] * size
            assert samples[f'{role}_token_loss'] == [token_loss] * size
            assert samples[f'{role}_bits_per_byte'] == pytest.approx([
                nll / (math.log(2) * doc['byte_length']) for doc in samples['documents']
            ])
            # The byte-weighted sample diagnostics reproduce the verdict's BPB.
            total_bytes = sum(doc['byte_length'] for doc in samples['documents'])
            weighted = math.fsum(bpb * doc['byte_length'] for bpb, doc in
                                 zip(samples[f'{role}_bits_per_byte'], samples['documents'], strict=True))
            assert weighted / total_bytes == pytest.approx(verdict[f'avg_{role}_loss'])
        assert json.loads(json.dumps(samples, allow_nan=False)) == samples
        if previous_samples is not None:
            assert samples == previous_samples, 'cached Horizon must preserve every sample diagnostic'
        previous_samples = samples
        assert bool(metadata['reference_misses']) == expected_misses
        assert len([call for call in client.calls if call[0].endswith('.utf8')]) == metadata['n_documents']
        assert not list(tmp_path.glob('duel-text-*'))
    assert len(subprocess_calls) == (2 if preloaded else 1)
    assert old_pool.closed
    assert closed.count("horizon") == (2 if preloaded else 1)
    assert closed.count("king") == closed.count("challenger") == 8
    assert sum(options.get('eos_token_id') == 151645 for options in tokenizer_options) == 2
    # MiMo changes must not invalidate cached reference inference; Horizon changes must.
    mimo_identity['build'] = 'updated-mimo'
    preload()
    _, metadata, _ = run()
    assert metadata['reference_misses'] == 0
    assert len(subprocess_calls) == (3 if preloaded else 1)
    reference_identity['build'] = 'updated-pipeline'
    preload()
    _, metadata, _ = run()
    assert metadata['reference_misses'] > 0
    assert len(subprocess_calls) == (4 if preloaded else 2)
    if preloaded:
        base._idle_horizon.close()


def test_mismatched_tokenizers_rejected_before_dataset_reads(tmp_path):
    a, b = tmp_path/'a', tmp_path/'b'
    a.mkdir(); b.mkdir()
    (a/'tokenizer.json').write_text('a')
    (b/'tokenizer.json').write_text('b')
    with pytest.raises(ValueError, match='tokenizer mismatch'):
        duel.run_document_duel(None, None, a, b, None)
