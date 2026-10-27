from collections import OrderedDict
from contextlib import nullcontext
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from psycopg.errors import QueryCanceled

from teutonic.dashboard import projection
from teutonic.evaluation.category_scores import category_scores_from_samples


def audit():
    return {
        "samples": {
            "n_samples": 3,
            "king_loss": [2, 6, 8],
            "challenger_loss": [1, 3, 5],
            "scored_tokens": [1, 3, 2],
            "shard_group_index": [0, 0, 0],
            "shard_index": [0, 0, 1],
        },
        "dataset_identity": {
            "sources": [
                {
                    "name": "mix06",
                    "shards": [
                        {
                            "url": "https://datasets.example/private/pretrain-behaviors__games--a.npy?secret=never"
                        },
                        {
                            "url": "https://datasets.example/private/math-reasoning__math-dialogue--b.npy"
                        },
                    ],
                }
            ]
        },
    }


def test_token_weighted_category_losses_use_actual_sample_counts():
    scores = category_scores_from_samples(**audit())
    assert scores == {
        "math-reasoning/math-dialogue": {
            "n_sequences": 1,
            "n_scored_tokens": 2,
            "avg_king_loss": 8,
            "avg_challenger_loss": 5,
            "mu_hat": 3,
        },
        "pretrain-behaviors/games": {
            "n_sequences": 2,
            "n_scored_tokens": 4,
            "avg_king_loss": 5,
            "avg_challenger_loss": 2.5,
            "mu_hat": 2.5,
        },
    }
    assert "private" not in str(scores) and "secret" not in str(scores)


def test_early_stopped_prefix_and_legacy_shard_references():
    item = audit()
    for key, value in item["samples"].items():
        item["samples"][key] = value[:2] if isinstance(value, list) else 2
    groups = [
        {
            "source": "mix06",
            "refs": [
                shard["url"] for shard in item.pop("dataset_identity")["sources"][0]["shards"]
            ],
        }
    ]
    scores = category_scores_from_samples(**item, dataset_identity=None, shards_used=groups)
    assert list(scores) == ["pretrain-behaviors/games"]
    assert scores["pretrain-behaviors/games"]["n_sequences"] == 2


def test_windows_and_whole_documents_in_one_category_share_token_weights():
    item = audit()
    item["samples"].update(
        component=["windows", "long_documents", "windows"],
        document_dataset=[None, "pretrain-behaviors", None],
        document_category=[None, "games", None],
        shard_index=[0, -1, 1],
    )
    assert category_scores_from_samples(**item)["pretrain-behaviors/games"] == {
        "n_sequences": 2,
        "n_scored_tokens": 4,
        "avg_king_loss": 5,
        "avg_challenger_loss": 2.5,
        "mu_hat": 2.5,
    }


def test_dclm_is_a_single_flat_category():
    item = audit()
    item["dataset_identity"]["sources"][0]["shards"][0]["url"] = (
        "https://datasets.example/dclm-baseline-1.0__global-shard.npy"
    )
    assert category_scores_from_samples(**item)["dclm-baseline-1.0"]["n_sequences"] == 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("king_loss", [1]),
        ("king_loss", [2, float("nan"), 8]),
        ("challenger_loss", [True, 3, 5]),
        ("scored_tokens", [1, 0, 2]),
        ("scored_tokens", [1, 1.5, 2]),
        ("shard_group_index", [-1, 0, 0]),
        ("shard_index", [True, 0, 1]),
        ("shard_index", [9, 0, 1]),
        ("n_samples", 20),
        ("component", ["unknown"] * 3),
    ],
)
def test_incomplete_or_invalid_audits_are_unavailable(field, value):
    item = audit()
    item["samples"][field] = value
    assert category_scores_from_samples(**item) == {}


def test_unclassified_samples_are_never_inferred_from_mix_totals_or_query_strings():
    item = audit()
    item["dataset_identity"]["sources"][0]["shards"][0]["url"] = (
        "https://datasets.example/unlabelled.npy?name=pretrain-behaviors__games--a.npy"
    )
    assert category_scores_from_samples(**item) == {}
    assert category_scores_from_samples(None, None) == {}


class AuditCursor:
    def __init__(self, audits, *, timeout=False):
        self.audits = audits
        self.reads = []
        self.settings = []
        self.timeout = timeout
        self.connection = SimpleNamespace(transaction=nullcontext)

    def execute(self, sql, params=None):
        if sql == "SHOW statement_timeout":
            return SimpleNamespace(fetchone=lambda: {"statement_timeout": "7s"})
        if "set_config" in sql:
            self.settings.append(params[0])
        if "FROM control_plane.evaluations" in sql:
            self.reads.append(params[0])
            if self.timeout:
                raise QueryCanceled("slow audit read")
            return SimpleNamespace(fetchall=lambda: [self.audits[key] for key in params[0]])
        return self


def metadata(count):
    now = datetime(2026, 10, 27, tzinfo=UTC)
    return [
        {
            "evaluation_id": str(index),
            "challenge_id": "same-upload",
            "completed_at": now - timedelta(seconds=index),
            "updated_at": now,
            "result_artifact_sha256": "digest-" + str(index),
        }
        for index in range(count)
    ]


@pytest.fixture(autouse=True)
def isolated_score_cache(monkeypatch):
    monkeypatch.setattr(projection, "_CATEGORY_SCORE_CACHE", OrderedDict())


def test_thousands_of_history_rows_fetch_only_two_audits_then_reuse_cache():
    rows = metadata(2000)
    audits = {
        row["evaluation_id"]: {"evaluation_id": row["evaluation_id"], **audit()} for row in rows
    }
    cursor = AuditCursor(audits)
    first = projection._category_score_rows(cursor, rows)
    assert cursor.reads == [["0", "1"]]
    assert len(first) == 2
    assert cursor.settings == ["1000", "7s"]
    second = projection._category_score_rows(cursor, rows)
    assert cursor.reads == [["0", "1"], ["2", "3"]]
    assert len(second) == 4
    assert first[("same-upload", rows[0]["completed_at"])][0]["source"] == (
        "math-reasoning/math-dialogue"
    )
    projection._category_score_rows(cursor, rows[:4])
    assert len(cursor.reads) == 2, "Warm aggregates must not refetch private sample arrays"
    changed = deepcopy(rows[:1])
    changed[0]["updated_at"] += timedelta(seconds=1)
    projection._category_score_rows(cursor, changed)
    assert cursor.reads[-1] == ["0"]


def test_timeout_keeps_cached_scores_and_leaves_missing_scores_pending():
    rows = metadata(3)
    audits = {
        row["evaluation_id"]: {"evaluation_id": row["evaluation_id"], **audit()} for row in rows
    }
    cursor = AuditCursor(audits)
    projection._category_score_rows(cursor, rows[:1])
    cursor.timeout = True
    scores = projection._category_score_rows(cursor, rows)
    assert list(scores) == [("same-upload", rows[0]["completed_at"])]
    assert len(projection._CATEGORY_SCORE_CACHE) == 1


def test_unavailable_records_are_cached_and_cache_has_a_memory_bound(monkeypatch):
    monkeypatch.setattr(projection, "_CATEGORY_SCORE_CACHE_LIMIT", 2)
    rows = metadata(4)
    cursor = AuditCursor(
        {row["evaluation_id"]: {"evaluation_id": row["evaluation_id"]} for row in rows}
    )
    assert all(not scores for scores in projection._category_score_rows(cursor, rows[:2]).values())
    projection._category_score_rows(cursor, rows[:2])
    assert len(cursor.reads) == 1
    projection._category_score_rows(cursor, rows)
    assert len(projection._CATEGORY_SCORE_CACHE) == 2
