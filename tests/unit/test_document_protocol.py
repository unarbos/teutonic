import copy
import hashlib
import json

import pytest
from test_packed_text import fixture

from scripts.prepare_packed_evaluation import prepare
from scripts.stage_packed_configuration import stage
from teutonic.evaluation.configuration import DatasetManifestSnapshot, EvaluationSettings
from teutonic.evaluation.packed_text import canonical
from teutonic.evaluation.protocol_v3 import (
    PROTOCOL_VERSION,
    EvaluationRequestV3,
    ProtocolValidationError,
)
from teutonic.evaluation.reference_policy import POLICY_VERSION
from teutonic.validator.runtime import evaluation_policy_from_env


def request():
    _, _, _, mix = fixture()
    artifact = lambda sha: {"kind": "r2-prefix", "bucket": "teutonic-models-enam",
                            "prefix": f"models/sha256/{sha}/", "expected_digest": sha}
    return {"protocol_version": PROTOCOL_VERSION, "evaluation_id": "test", "attempt_number": 1,
            "king": artifact("a"*64), "challenger": artifact("b"*64),
            "miner": {"hotkey": "hot", "coldkey": "cold", "uid": 1, "netuid": 3},
            "versions": {"evaluation_policy": POLICY_VERSION, "dataset": "fixture",
                         "code": "fixture", "evaluator": "document-evaluator-v3"},
            "sampling": {"seed": 1, "bootstrap_seed": 3, "block_hash": "0x" + "a"*64},
            "limits": {"batch_token_budget": 81920, "batch_size": 40, "n_bootstrap": 100,
                       "alpha": .05, "delta_threshold": .0015}, "dataset": mix}


def test_document_request_and_legacy_rejection():
    payload = request()
    parsed = EvaluationRequestV3.from_mapping(payload)
    assert parsed.dataset["dataset"]["bucket"] == "dataset"
    assert parsed.request_sha256 == EvaluationRequestV3.from_mapping(copy.deepcopy(payload)).request_sha256
    for mutate in [lambda p: p.update(protocol_version="teutonic-evaluator-v2"),
                   lambda p: p["limits"].update(n=0),
                   lambda p: p["limits"].update(seq_len=2048),
                   lambda p: p["limits"].update(delta_threshold=float("nan")),
                   lambda p: p["dataset"]["dataset"].update(bucket="datasets"),
                   lambda p: p["dataset"].update(source="pretokenized_npy")]:
        bad = copy.deepcopy(payload)
        mutate(bad)
        with pytest.raises(ProtocolValidationError):
            EvaluationRequestV3.from_mapping(bad)


def test_packed_database_settings_to_validator_request():
    payload = request()
    mix = payload["dataset"]
    sha = hashlib.sha256(canonical(mix)).hexdigest()
    snapshot = DatasetManifestSnapshot("main", f"s3://dataset/teutonic-text/splits/main/{sha}.json", sha, 1, mix)
    settings = EvaluationSettings("c"*64, "main", 0, .0015, (snapshot,), 1, mix)
    policy = evaluation_policy_from_env({"TEUTONIC_EVALUATION_POLICY_VERSION": POLICY_VERSION,
                                         "TEUTONIC_EVALUATOR_CODE_VERSION": "fixture",
                                         "TEUTONIC_EVALUATOR_VERSION": "document-evaluator-v3"}, settings=settings)
    payload["limits"] = policy.thresholds
    payload["dataset"] = policy.dataset_request(block_hash="0x"+"a"*64, hotkey="hot")
    assert EvaluationRequestV3.from_mapping(payload).dataset == mix
    assert "n" not in payload["limits"] and "seq_len" not in payload["limits"]


def test_local_migration_bundle_preserves_weights_and_never_writes_r2(tmp_path):
    client, dataset, _, _ = fixture()
    entries = prepare(dataset, {"main": {"categories": {"a": 1}, "long_document_tokens": 40000}},
                      {"a": {"dataset": "test"}}, tmp_path)
    assert len(entries) == 2
    for entry in entries:
        assert entry["bucket"] == "dataset"
        assert hashlib.sha256((tmp_path/entry["local"]).read_bytes()).hexdigest() == entry["sha256"]
    sql = stage(tmp_path, netuid=3, generation="test", threshold=.0015)
    assert "category_weights" in sql and "category_sources" in sql
    assert "AND active FOR UPDATE" in sql
    assert "DELETE" not in sql
    assert "s3://dataset/teutonic-text/splits/main/" in sql
    assert "COMMIT" in sql
    # The fake client exposes no upload/delete methods: preparation can only read.
    assert client.calls
    assert json.loads((tmp_path/"main.json").read_text())["categories"][0]["weight"] == 1


def test_result_requires_complete_reference_objective_and_consistent_decision():
    from teutonic.evaluation.protocol_v3 import result_provenance, validate_result_v3
    from teutonic.evaluation.reference_policy import reference_verdict
    parsed = EvaluationRequestV3.from_mapping(request())
    verdict = reference_verdict([3.]*4, [2.]*4, [1.]*4, [10]*4,
        bootstrap_seed=0, n_bootstrap=100, alpha=.05, delta_threshold=.0015)
    verdict.update(result_provenance(parsed, started_at='start', completed_at='end',
        requested_sequences=4, completed_sequences=4, early_stopped=False, hardware={}))
    verdict.update(wall_time_s=1, result_artifact_sha256='a'*64)
    validate_result_v3(verdict, parsed)
    for field, value in [('accepted',False),('lcb',float('nan')),('delta_threshold',.5),
                          ('retention_gates',True),('scoring_policy','legacy')]:
        with pytest.raises(ProtocolValidationError):
            validate_result_v3({**verdict,field:value}, parsed)
    broken = copy.deepcopy(verdict)
    broken['sampling']['completed_sequences'] = 3
    with pytest.raises(ProtocolValidationError):
        validate_result_v3(broken, parsed)


def test_packed_dashboard_manifest_and_version():
    from datetime import UTC, datetime

    from test_dashboard_view import dataset_manifest, payload

    from teutonic.dashboard.contracts import (
        canonical_dashboard_json,
        canonical_dataset_manifest_json,
    )
    from teutonic.dashboard.projection import _dataset_sources, _dataset_versions

    mix = request()["dataset"]
    row = {"manifest_json": mix, "name": "main", "sample_proportion": 1,
           "manifest_url": "s3://dataset/teutonic-text/splits/main/test.json",
           "manifest_sha256": "a" * 64, "config_version": "b" * 64,
           "dataset_label": "packed main", "eval_n": 0, "delta_threshold": .0015,
           "config_created_at": datetime.now(UTC)}
    sources = _dataset_sources([row])
    assert len(sources) == len(mix["categories"])
    assert sum(source["proportion"] for source in sources) == pytest.approx(1)
    assert all(source["total_tokens"] is None for source in sources)
    manifest = {**dataset_manifest(), "sources": sources, "eval_n": 0,
                "token_budget": mix["token_budget"], "loss_unit": "bits/byte"}
    canonical_dataset_manifest_json(manifest)
    dashboard = payload()
    dashboard["dataset_versions"] = _dataset_versions([row])
    canonical_dashboard_json(dashboard)
    assert dashboard["dataset_versions"][0]["token_budget"] == mix["token_budget"]


def test_packed_category_scores_use_bytes_and_reference_deficits():
    import math

    from teutonic.evaluation.category_scores import category_scores_from_samples

    samples = {"format": "packed-document-losses",
               "documents": [{"dataset": "test", "category": "a", "byte_length": n}
                             for n in (10, 30)],
               "king_nll": [10 * math.log(2), 90 * math.log(2)],
               "challenger_nll": [15 * math.log(2), 60 * math.log(2)],
               "reference_nll": [20 * math.log(2), 30 * math.log(2)]}
    score = category_scores_from_samples(samples, {})["test/a"]
    assert score["n_bytes"] == 40 and score["n_sequences"] == 2
    assert score["avg_king_loss"] == 2.5
    assert score["avg_challenger_loss"] == 1.875
    assert score["mu_hat"] == .75  # The first document stays below Horizon and is neutral.
    samples["reference_nll"].pop()
    assert category_scores_from_samples(samples, {}) == {}
