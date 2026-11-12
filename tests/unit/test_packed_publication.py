import hashlib
import json

import pytest
from botocore.exceptions import ClientError
from test_packed_text import fixture

from scripts.prepare_packed_evaluation import prepare
from scripts.publish_packed_evaluation import run
from scripts.stage_packed_configuration import stage


def bundle(tmp_path):
    client, dataset, _, _ = fixture()
    prepare(dataset, {"main": {"categories": {"a": 1}, "long_document_tokens": 35000000}},
            {"a": {"dataset": "test"}}, tmp_path)
    writes = []
    def head(*, Bucket, Key):
        assert Bucket == "dataset"
        if Key not in client.objects:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        return {}
    def put(*, Bucket, Key, Body, IfNoneMatch, **kwargs):
        assert Bucket == "dataset" and IfNoneMatch == "*" and Key not in client.objects
        writes.append(Key)
        client.objects[Key] = Body
    client.head_object, client.put_object = head, put
    return client, writes


def test_dry_run_is_read_only_and_default_publication_is_immutable(tmp_path):
    client, writes = bundle(tmp_path)
    assert run(tmp_path, client, publish=False)["state"] == "validated-read-only"
    assert not writes
    result = run(tmp_path, client)
    assert result["budgets"] == {"main": 35000000}
    assert len(writes) == 2 and "/sampling/" in writes[0] and "/splits/" in writes[1]
    run(tmp_path, client, publish=True)
    assert len(writes) == 2
    client.objects[writes[-1]] = b"collision"
    with pytest.raises(ValueError, match="immutable object checksum mismatch"):
        run(tmp_path, client, publish=True)
    assert client.objects[writes[-1]] == b"collision" and len(writes) == 2


@pytest.mark.parametrize("mutation", ["hash", "bucket", "key", "dataset_pin"])
def test_invalid_bundle_fails_before_any_upload(tmp_path, mutation):
    client, writes = bundle(tmp_path)
    path = tmp_path / "pending-publication.json"
    entries = json.loads(path.read_text())
    if mutation == "hash":
        (tmp_path / "main.json").write_bytes(b"bad")
    elif mutation == "bucket":
        entries[-1]["bucket"] = "datasets"
    elif mutation == "key":
        entries[-1]["key"] = "teutonic-text/splits/main/latest.json"
    else:
        value = json.loads((tmp_path / "main.json").read_text())
        value["dataset"]["sha256"] = "c" * 64
        raw = json.dumps(value).encode()
        (tmp_path / "main.json").write_bytes(raw)
        sha = hashlib.sha256(raw).hexdigest()
        entries[-1].update(sha256=sha, key=f"teutonic-text/splits/main/{sha}.json")
    path.write_text(json.dumps(entries))
    with pytest.raises(ValueError):
        run(tmp_path, client, publish=True)
    assert not writes


def test_budget_change_is_explicit_in_staged_sql(tmp_path):
    bundle(tmp_path)
    sql = stage(tmp_path, netuid=999, generation="fixture", threshold=.0015,
                previous_token_budget=70000000)
    assert "::bigint=70000000" in sql
    assert '"token_budget":35000000' in sql
    assert "Drain active evaluations" in sql
    assert "DO UPDATE SET enabled=false" in sql
    assert "DELETE" not in sql
