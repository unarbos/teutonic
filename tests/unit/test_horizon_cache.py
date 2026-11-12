import hashlib
from types import SimpleNamespace

import pytest
from test_packed_text import fixture

from teutonic.evaluator import horizon
from teutonic.evaluator.document_duel import ReferenceCache, reference_key


def test_verified_download_reuse_corruption_and_revision(tmp_path, monkeypatch):
    files = {"config.json": b'{"model_type":"k2_horizon"}', "tokenizer.json": b'{}',
             "model.safetensors.index.json": b'{"weight_map":{"weight":"part.safetensors"}}',
             "part.safetensors": b"weights"}
    monkeypatch.setattr(horizon, "CONFIG_SHA256", hashlib.sha256(files["config.json"]).hexdigest())
    monkeypatch.setattr(horizon, "INDEX_SHA256", hashlib.sha256(files["model.safetensors.index.json"]).hexdigest())
    info = SimpleNamespace(sha=horizon.REVISION, siblings=[
        SimpleNamespace(rfilename=name, size=len(raw),
                        lfs=SimpleNamespace(sha256=hashlib.sha256(raw).hexdigest()))
        for name, raw in files.items()])
    class API:
        def model_info(self, repo, revision, files_metadata):
            assert revision == horizon.REVISION and files_metadata
            return info
    calls = []
    def download(repo_id, filename, revision, local_dir, force_download):
        from pathlib import Path
        calls.append(filename)
        (Path(local_dir)/filename).write_bytes(files[filename])
    task = horizon.HorizonDownload(tmp_path, api=API(), download=download)
    task.prepare()
    assert task.require_ready() == task.snapshot
    assert len(calls) == len(files)
    again = horizon.HorizonDownload(tmp_path, api=API(), download=download)
    again.prepare()
    assert len(calls) == len(files)
    (task.snapshot/"part.safetensors").write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="changed"):
        again.require_ready()
    again.prepare()
    assert calls.count("part.safetensors") == 2
    info.sha = "other"
    with pytest.raises(ValueError, match="revision"):
        again.prepare()


def test_reference_cache_identity_and_invalid_scores(tmp_path):
    _, dataset, _, _ = fixture()
    doc = dataset.document("index/test/a/part-0.documents.npy", 0)
    key = reference_key(doc, "tok1", "code")
    assert key != reference_key(doc, "tok2", "code")
    assert key != reference_key(doc, "tok1", "newcode")
    assert key != reference_key(doc, "tok1", "code", {"backend": "vllm"})
    assert reference_key(doc, "tok1", "code", {"kv": 32}) != reference_key(doc, "tok1", "code", {"kv": 64})
    cache = ReferenceCache(tmp_path/"losses.sqlite")
    assert cache.get(key) is None
    cache.put(key, 3.0, 4)
    assert cache.get(key) == (3.0, 4)
    with pytest.raises(ValueError):
        cache.put(key, float("nan"), 4)
    cache.close()
