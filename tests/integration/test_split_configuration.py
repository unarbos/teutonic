import hashlib
import io
import os
import tomllib
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import psycopg
import pytest
from psycopg.rows import dict_row

import chain_config
from scripts import configure_evaluation as admin
from teutonic.evaluation.configuration import (
    DatasetManifestSnapshot,
    canonical_manifest_bytes,
    store_evaluation_configuration,
)

DATABASE_URL = os.environ.get("TEUTONIC_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DATABASE_URL, reason="PostgreSQL integration database required")


LEGACY_CONFIG = tomllib.loads((Path(__file__).parents[1] / "fixtures/legacy_evaluation.toml").read_text())

def cleanup(connection):
    connection.execute("UPDATE control_plane.competitions SET current_reign_id=NULL,reward_reign_id=NULL WHERE netuid=998")
    connection.execute("DELETE FROM control_plane.weight_publications WHERE competition_id IN (SELECT competition_id FROM control_plane.competitions WHERE netuid=998)")
    connection.execute("DELETE FROM control_plane.king_reigns WHERE competition_id IN (SELECT competition_id FROM control_plane.competitions WHERE netuid=998)")
    connection.execute(
        "DELETE FROM control_plane.dataset_manifests WHERE evaluation_config_id IN (SELECT ec.evaluation_config_id FROM control_plane.evaluation_configs ec JOIN control_plane.competitions c USING(competition_id) WHERE c.netuid=998)"
    )
    connection.execute(
        "DELETE FROM control_plane.evaluation_configs WHERE competition_id IN (SELECT competition_id FROM control_plane.competitions WHERE netuid=998)"
    )
    connection.execute(
        "DELETE FROM control_plane.evaluation_early_stopping_policies WHERE competition_id IN (SELECT competition_id FROM control_plane.competitions WHERE netuid=998)"
    )
    connection.execute(
        "DELETE FROM control_plane.competitions WHERE netuid=998 AND competition_key<>'main'"
    )
    connection.execute("DELETE FROM control_plane.competitions WHERE netuid=998")


@pytest.fixture
def database():
    with psycopg.connect(DATABASE_URL, autocommit=True, row_factory=dict_row) as connection:
        assert (
            connection.execute("SELECT current_database() AS name").fetchone()["name"]
            == "teutonic_test"
        )
        cleanup(connection)
        connection.execute(
            "INSERT INTO control_plane.competitions(netuid,chain_generation,name) VALUES (998,'split-admin-test','existing-main')"
        )
        try:
            yield connection
        finally:
            cleanup(connection)


def fixture_fetch(**source):
    name = source["name"]
    weights = {
        category: spec["weight"]
        for category, spec in LEGACY_CONFIG["splits"][name]["categories"].items()
    }
    manifest = {
        "format": "teutonic-split-v1",
        "split": name,
        "category_weights": weights,
        "shards": [
            {
                "url": f"https://datasets.example/{name}/{category}.npy",
                "category": category,
                "sha256": "a" * 64,
                "n_tokens": 30000 * 2048,
                "size_bytes": 30000 * 2048 * 4 + 128,
            }
            for category in weights
        ],
    }
    return DatasetManifestSnapshot(
        **source,
        manifest_sha256=hashlib.sha256(canonical_manifest_bytes(manifest)).hexdigest(),
        manifest=manifest,
    )


def seed_main(database):
    main = replace(
        fixture_fetch(name="math", manifest_url="https://datasets.example/main.json", proportion=1),
        name="original-main",
    )
    store_evaluation_configuration(
        database,
        netuid=998,
        chain_generation="split-admin-test",
        competition="existing-main",
        dataset_label="original",
        n=30000,
        delta_threshold=0.012,
        manifests=[main],
        shards_per_dataset=4,
    )


def run_admin(*args):
    with (
        patch.dict(
            os.environ,
            {
                "TEUTONIC_DATABASE_URL": DATABASE_URL,
                "TEUTONIC_NETUID": "998",
                "TEUTONIC_CHAIN_GENERATION": "split-admin-test",
                "TEUTONIC_COMPETITION": "existing-main",
            },
        ),
        redirect_stdout(io.StringIO()),
    ):
        return admin.main(list(args))


def policies(database):
    return database.execute(
        "SELECT c.competition_key, ec.config_version, ec.delta_threshold FROM control_plane.competitions c JOIN control_plane.evaluation_configs ec USING(competition_id) WHERE c.netuid=998 AND ec.active ORDER BY c.competition_key"
    ).fetchall()


def category_fixtures():
    from teutonic.evaluation.index_manifest import (
        INDEX_FORMAT,
        TOKEN_FORMAT,
        index_paths,
        pin_index_manifest,
    )
    from teutonic.evaluation.mixes import build_mix_manifests
    sources, datasets, files = [], {}, {}
    for name, spec in chain_config.CATEGORY_DATASETS.items():
        categories = [c for c, entry in chain_config.CATEGORY_REGISTRY.items() if entry["dataset"] == name]
        manifest = {"shards": [
            {"key": f"shards/{name}__{category}--part-{i}.npy", "sha256": "a" * 64,
             "n_tokens": 100_000 * 2048, "size_bytes": 100_000 * 2048 * 4 + 128}
            for category in categories for i in range(5)]}
        sha = hashlib.sha256(canonical_manifest_bytes(manifest)).hexdigest()
        sources.append(DatasetManifestSnapshot(name, spec["manifest_url"], sha, 1, manifest))
        datasets[name] = {"categories": categories, "source_manifest": {"url": spec["manifest_url"], "sha256": sha}}
        for category in categories:
            for path in index_paths(name, category).values():
                files[path] = {"etag": '"abcd"', "size_bytes": 100, "sha256": "b" * 64}
    raw = canonical_manifest_bytes({"format": INDEX_FORMAT, "token_format": TOKEN_FORMAT, "datasets": datasets, "files": files})
    return build_mix_manifests(sources), pin_index_manifest(raw, "https://datasets.example/doc_index/new.json")


def seed_rewards(database):
    main = database.execute("SELECT competition_id FROM control_plane.competitions WHERE netuid=998").fetchone()["competition_id"]
    reign = database.execute(
        """INSERT INTO control_plane.king_reigns(competition_id,reign_number,model_digest,public_bucket,public_prefix,
               hotkey,uid,crowned_at,crowned_finalized_block,operator_provenance)
           VALUES (%s,0,%s,'public',%s,'latest-main',1,now(),100,'test') RETURNING reign_id""",
        (main, "a" * 64, f"models/sha256/{'a' * 64}/"),
    ).fetchone()["reign_id"]
    database.execute("UPDATE control_plane.competitions SET current_reign_id=%s WHERE competition_id=%s", (reign, main))
    database.execute(
        """INSERT INTO control_plane.weight_publications(competition_id,source_reign_id,policy_version,policy_hotkeys,
               policy_weights,target_hotkeys,target_uids,normalized_weights,payload_sha256,mapping_finalized_block,idempotency_key,state)
           VALUES (%s,%s,'legacy',ARRAY['latest-main','old-winner'],ARRAY[0.6,0.4],
               ARRAY['latest-main','old-winner'],ARRAY[1,2],ARRAY[0.6,0.4],%s,100,'category-cutover-test','requested')""",
        (main, reign, "b" * 64),
    )
    database.execute("INSERT INTO control_plane.competitions(netuid,chain_generation,name,competition_key,main_competition_id) VALUES (998,'split-admin-test','code','code',%s)", (main,))


def test_initialize_dry_run_targeted_update_and_atomic_all(database, tmp_path):
    from teutonic.evaluation.mixes import write_bundle
    from teutonic.weights.policy import replacement_rewards
    seed_main(database)
    seed_rewards(database)
    original = policies(database)
    outputs, index = category_fixtures()
    write_bundle(outputs, tmp_path)
    arguments = ("--all", "--initialize", "--replace-definition", "--cutover", "--bundle-dir", str(tmp_path))
    with patch.object(admin, "fetch_index_manifest", return_value=index):
        assert run_admin(*arguments, "--dry-run") == 0
        assert policies(database) == original
        assert database.execute("SELECT count(*) AS n FROM control_plane.competitions WHERE netuid=998").fetchone()["n"] == 2
        real_store = admin.store_evaluation_configuration

        def fail_last(connection, **kwargs):
            if kwargs["competition"] == "mix06":
                raise RuntimeError("simulated activation failure")
            return real_store(connection, **kwargs)

        with patch.object(admin, "store_evaluation_configuration", side_effect=fail_last), pytest.raises(RuntimeError, match="simulated"):
            run_admin(*arguments)
        assert policies(database) == original
        assert database.execute("SELECT reward_transition FROM control_plane.competitions WHERE netuid=998 AND competition_key='main'").fetchone()["reward_transition"] is None
        assert run_admin(*arguments) == 0
        with pytest.raises(ValueError, match="already initialized"):
            run_admin(*arguments)
    rows = policies(database)
    assert len(rows) == 7
    summaries = database.execute("SELECT manifest_json FROM control_plane.dashboard_dataset_manifests WHERE netuid=998").fetchall()
    assert len(summaries) == 7
    assert all("shards" not in r["manifest_json"] and "_shards_zlib_base64" not in r["manifest_json"] for r in summaries)
    assert all(r["delta_threshold"] == chain_config.COMPETITION_DEFAULTS[r["competition_key"]]["delta_threshold"] for r in rows)
    states = database.execute("SELECT competition_key,active,current_reign_id,reward_transition FROM control_plane.competitions WHERE netuid=998").fetchall()
    assert {r["competition_key"] for r in states if r["active"]} == set(admin.COMPETITIONS)
    assert all(r["current_reign_id"] is None for r in states if r["competition_key"].startswith("mix"))
    state = next(r["reward_transition"] for r in states if r["competition_key"] == "main")
    assert dict(zip(*replacement_rewards(state))) == pytest.approx({"latest-main": .6, "old-winner": .4})
    budgets = database.execute("SELECT ec.eval_n,ec.long_documents FROM control_plane.evaluation_configs ec JOIN control_plane.competitions c USING(competition_id) WHERE c.netuid=998 AND ec.active").fetchall()
    assert all(r["eval_n"] == 20000 and r["long_documents"]["token_budget"] == 30000000 for r in budgets)
    with patch.object(admin, "fetch_dataset_manifest", side_effect=AssertionError("unexpected network request")):
        assert run_admin("--competition", "MIX02", "--delta-threshold", ".004") == 0
    changed = policies(database)
    assert [r for r in changed if r["competition_key"] != "mix02"] == [r for r in rows if r["competition_key"] != "mix02"]
    assert next(r for r in changed if r["competition_key"] == "mix02")["delta_threshold"] == .004


def test_reward_snapshot_orders_reigns_by_position_when_hotkeys_overlap(database):
    from teutonic.weights.transition import snapshot_transition
    seed_rewards(database)
    main_id = database.execute("SELECT competition_id FROM control_plane.competitions WHERE netuid=998 AND competition_key='main'").fetchone()["competition_id"]
    database.execute("UPDATE control_plane.king_reigns SET reign_number=1 WHERE competition_id=%s", (main_id,))
    database.execute("UPDATE control_plane.competitions SET reward_main_hotkeys=ARRAY['latest-main','old-winner'],next_reign_number=2 WHERE competition_id=%s", (main_id,))
    for key, hotkey, block in (("main", "old-winner", 10), ("math", "math-winner", 50), ("text", "text-winner", 70), ("code", "old-winner", 200)):
        if key == "main":
            cid = main_id
        else:
            cid = database.execute("INSERT INTO control_plane.competitions(netuid,chain_generation,name,competition_key,main_competition_id) VALUES (998,'split-admin-test',%s,%s,%s) ON CONFLICT(netuid,chain_generation,name) DO UPDATE SET active=true RETURNING competition_id", (key, key, main_id)).fetchone()["competition_id"]
        reign = database.execute("""INSERT INTO control_plane.king_reigns(competition_id,reign_number,model_digest,public_bucket,public_prefix,hotkey,uid,crowned_at,crowned_finalized_block,operator_provenance,ended_at,replacement_reason)
            VALUES (%s,0,%s,'public',%s,%s,2,now(),%s,'test',CASE WHEN %s THEN now() END,CASE WHEN %s THEN 'replaced' END) RETURNING reign_id""",
            (cid, "c" * 64, f"models/sha256/{'c' * 64}/", hotkey, block, key == "main", key == "main")).fetchone()["reign_id"]
        if key != "main":
            database.execute("UPDATE control_plane.competitions SET current_reign_id=%s WHERE competition_id=%s", (reign, cid))
    hotkeys, weights = ["latest-main", "old-winner", "math-winner", "text-winner"], [.4, .3, .15, .15]
    database.execute("UPDATE control_plane.weight_publications SET policy_hotkeys=%s,policy_weights=%s,target_hotkeys=%s,target_uids=ARRAY[1,2,3,4],normalized_weights=%s WHERE competition_id=%s", (hotkeys, weights, hotkeys, weights, main_id))
    main = database.execute("SELECT * FROM control_plane.competitions WHERE competition_id=%s", (main_id,)).fetchone()
    state = snapshot_transition(database, main)
    assert [entry["id"] for entry in state["legacy"]] == ["main-history:1", "math", "text", "code"]
    assert [entry["crowned_block"] for entry in state["legacy"]] == [10, 50, 70, 200]


def test_cutover_refuses_running_intake_before_configuration_writes(database):
    from teutonic.access.repository import CONTROLLER_ADVISORY_LOCK_ID
    seed_rewards(database)
    database.execute("SELECT pg_advisory_lock(%s)", (CONTROLLER_ADVISORY_LOCK_ID,))
    try:
        with pytest.raises(ValueError, match="stop the access controller"):
            run_admin("--all", "--initialize", "--replace-definition", "--cutover", "--dry-run")
        assert database.execute("SELECT count(*) AS n FROM control_plane.competitions WHERE netuid=998").fetchone()["n"] == 2
    finally:
        database.execute("SELECT pg_advisory_unlock(%s)", (CONTROLLER_ADVISORY_LOCK_ID,))
