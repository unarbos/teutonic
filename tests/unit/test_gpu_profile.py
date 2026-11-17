from types import SimpleNamespace

import pytest

from teutonic.evaluator import (
    gpu_profile,
    horizon_pipeline,
    horizon_pipeline_runtime,
    mimo_pool,
    vllm_runtime,
    vllm_score,
)


def test_profiles_are_explicit_and_worker_environments_preserve_them(monkeypatch, tmp_path):
    monkeypatch.delenv(gpu_profile.ENVIRONMENT_KEY, raising=False)
    assert gpu_profile.mimo_tensor_parallel_size() == 1
    assert horizon_pipeline.settings()['batch_tokens'] == 131072
    monkeypatch.setenv(gpu_profile.ENVIRONMENT_KEY, 'h200')
    monkeypatch.setenv('TEUTONIC_HORIZON_KERNEL_CACHE_DIR', str(tmp_path))
    monkeypatch.setenv('TEUTONIC_VLLM_TRITON_CACHE_DIR', str(tmp_path / 'triton'))
    assert gpu_profile.mimo_tensor_parallel_size() == 2
    assert horizon_pipeline.settings()['batch_tokens'] == 32768
    assert horizon_pipeline.settings()['min_batch_tokens'] == 8192
    assert horizon_pipeline.settings()['max_model_len'] == 524288
    assert vllm_score.settings('king')['tensor_parallel_size'] == 2
    assert vllm_score.settings('king')['kv_cache_memory_bytes'] == 16 * 1024**3
    for env in (vllm_runtime.worker_env('king', [0, 1]), horizon_pipeline_runtime.worker_env(range(8))):
        assert env[gpu_profile.ENVIRONMENT_KEY] == 'h200'
    king_cache = vllm_runtime.worker_env('king', [0, 1])['TRITON_CACHE_DIR']
    assert king_cache == str(tmp_path / 'triton' / 'king-gpus-0-1')
    assert vllm_runtime.worker_env('king', [2, 3])['TRITON_CACHE_DIR'] != king_cache
    assert vllm_runtime.worker_env('challenger', [0, 1])['TRITON_CACHE_DIR'] != king_cache
    with pytest.raises(ValueError, match='two GPUs'):
        vllm_score.settings('king', 1)
    monkeypatch.setenv(gpu_profile.ENVIRONMENT_KEY, 'typo')
    with pytest.raises(ValueError, match='b300 or h200'):
        horizon_pipeline.settings()


def test_h200_keeps_all_document_sizes_on_four_tp2_replicas(monkeypatch, tmp_path):
    monkeypatch.setenv(gpu_profile.ENVIRONMENT_KEY, 'h200')
    created, closed = [], []

    class Worker:
        def __init__(self, role, snapshot, ids, *args, **kwargs):
            self.report = {'role': role, 'gpu_ids': ids}
            created.append((role, ids, kwargs['max_model_len']))

        def wait_ready(self):
            pass

        def score(self, docs, sequences):
            return [{'id': d.doc_id, 'nll': len(s)} for d, s in zip(docs, sequences, strict=True)]

        def close(self):
            closed.append(self.report)

    monkeypatch.setattr(mimo_pool, 'Worker', Worker)
    monkeypatch.setattr(mimo_pool, 'wait_for_gpu_release', lambda _: None)
    docs = [SimpleNamespace(doc_id=str(i)) for i in range(4)]
    sequences = [[1]*n for n in (2, 16384, 130000, 5)]
    with mimo_pool.MiMoPool('k', 'c', list(range(8)), tmp_path, {}, lambda _: None) as pool:
        k, c = pool.score(docs, sequences)
        assert k == c
        assert [r['nll'] for r in k] == [len(s) for s in sequences]
        pool.score(docs[:1], sequences[:1])
        assert len(created) == 4
        # A longer document grows context without truncation or falling back to TP1.
        longer = [[1] * 140000]
        k, c = pool.score(docs[:1], longer)
        assert k == c and k[0]['nll'] == 140000
        assert len(created) == 8 and len(closed) == 4
        pool.score(docs[:1], sequences[:1])
        assert len(created) == 8
    assert [ids for _, ids, _ in created] == [[0, 1], [2, 3], [4, 5], [6, 7]] * 2
    assert [context for _, _, context in created] == [131073] * 4 + [140001] * 4
    assert len(closed) == 8


def test_incremental_expert_packing_preserves_weights_exactly():
    torch = pytest.importorskip('torch')
    experts = [SimpleNamespace(gate_proj=torch.nn.Linear(8, 4, bias=False),
                               up_proj=torch.nn.Linear(8, 4, bias=False),
                               down_proj=torch.nn.Linear(4, 8, bias=False)) for _ in range(3)]
    expected13 = torch.stack([torch.cat([e.gate_proj.weight, e.up_proj.weight]) for e in experts])
    expected2 = torch.stack([e.down_proj.weight for e in experts])
    module = SimpleNamespace(experts=experts)
    horizon_pipeline.pack_experts_incrementally(module)
    assert torch.equal(module.w13, expected13)
    assert torch.equal(module.w2, expected2)
    assert module.w13.is_contiguous() and module.w2.is_contiguous()
    assert all(p.weight.numel() == 0 for e in experts for p in (e.gate_proj, e.up_proj, e.down_proj))
