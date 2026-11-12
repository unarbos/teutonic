from types import SimpleNamespace

import pytest

from teutonic.evaluator import mimo_pool


def test_replica_dispatch_and_large_only_transition_restore_original_order(monkeypatch, tmp_path):
    created, scored, released = [], [], []
    class Worker:
        def __init__(self, role, snapshot, ids, directory, identity, callback, **options):
            self.role, self.ids = role, ids
            self.report = {"role": role, "gpu_ids": ids}
            created.append((role, ids, options["max_model_len"]))
        def wait_ready(self):
            pass
        def score(self, docs, sequences):
            scored.extend((self.role, d.doc_id, len(t), self.ids) for d, t in zip(docs, sequences, strict=True))
            return [{"id": d.doc_id, "nll": int(d.doc_id) + (0 if self.role == "king" else .5)} for d in docs]
        def close(self):
            released.append(self.ids)
    monkeypatch.setattr(mimo_pool, "Worker", Worker)
    monkeypatch.setattr(mimo_pool, "wait_for_gpu_release", lambda ids: None)
    docs = [SimpleNamespace(doc_id=str(i)) for i in range(9)]
    tokens = [[7] * n for n in [4, 5, 130000, 6, 8, 131000, 3, 9, 7]]
    with mimo_pool.MiMoPool("king", "challenger", list(range(8)), tmp_path, {}, lambda e: None) as pool:
        king, challenger = pool.score(docs, tokens)
        assert [r["nll"] for r in king] == list(range(9))
        assert [r["nll"] for r in challenger] == [i + .5 for i in range(9)]
    assert len(created) == 12 and len(released) == 12
    assert len(scored) == 18 and len({(r, d) for r, d, _, _ in scored}) == 18
    for role, _, length, ids in scored:
        assert len(ids) == (1 if length <= mimo_pool.SINGLE_GPU_MAX_TOKENS else 2)
        assert all(i in (range(4) if role == "king" else range(4, 8)) for i in ids)
    assert [ids for _, ids, _ in created[:8]] == [[i] for i in range(8)]
    assert [ids for _, ids, _ in created[8:]] == [[0, 1], [2, 3], [4, 5], [6, 7]]


def test_balancing_is_deterministic_and_keeps_every_document():
    sequences = [[0] * n for n in [11, 4, 9, 3, 7, 2, 5, 1]]
    groups = mimo_pool.balanced_groups(range(8), sequences, 4)
    assert groups == mimo_pool.balanced_groups(reversed(range(8)), sequences, 4)
    assert sorted(i for group in groups for i in group) == list(range(8))
    assert all(group == sorted(group) for group in groups)


def test_partial_loading_failure_closes_started_replicas(monkeypatch, tmp_path):
    closed = []
    class Worker:
        def __init__(self, role, snapshot, ids, *a, **kw):
            self.report = {"gpu_ids": ids}
            self.ids = ids
        def wait_ready(self):
            if self.ids == [3]:
                raise RuntimeError("load failure")
        def close(self):
            closed.append(self.ids)
    monkeypatch.setattr(mimo_pool, "Worker", Worker)
    monkeypatch.setattr(mimo_pool, "wait_for_gpu_release", lambda ids: None)
    with pytest.raises(RuntimeError, match="load failure"), \
            mimo_pool.MiMoPool("k", "c", list(range(8)), tmp_path, {}, lambda e: None) as pool:
        pool.score([SimpleNamespace(doc_id="0")], [[1, 151645]])
    assert sorted(closed) == [[i] for i in range(8)]
