import json
import os

import psycopg
import pytest
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from scripts.retire_mix_competitions import main
from teutonic.weights.contracts import weight_payload_digest
from teutonic.weights.policy import replacement_rewards


@pytest.mark.skipif(not os.environ.get("TEUTONIC_TEST_DATABASE_URL"), reason="disposable PostgreSQL required")
def test_read_only_preview_atomic_apply_and_idempotence(tmp_path, monkeypatch):
    url = os.environ["TEUTONIC_TEST_DATABASE_URL"]
    with psycopg.connect(url, autocommit=True, row_factory=dict_row) as db:
        assert db.execute("SELECT current_database() AS name").fetchone()["name"] == "teutonic_test"
        db.execute("TRUNCATE control_plane.competitions CASCADE")
        keys = ("main", "mix01", "mix02", "mix03", "mix05", "mix06", "mix07")
        state = {"version": "mix-successor-v2", "main": {"id": "main", "hotkey": "main", "weight": .25},
                 "legacy": [], "mixes": {k: k for k in keys[1:]}, "successors": {"mix07": "mix04"}}
        main_id = main_reign = None
        for i, key in enumerate(keys):
            cid = db.execute(
                """INSERT INTO control_plane.competitions
                   (netuid,chain_generation,name,competition_key,main_competition_id,reward_transition)
                   VALUES (999,'reward-test',%s,%s,%s,%s) RETURNING competition_id""",
                ("Teutonic-II-110B" if key == "main" else key,
                 key, main_id, Jsonb(state) if key == "main" else None),
            ).fetchone()["competition_id"]
            digest = f"{i + 1:064x}"
            reign = db.execute(
                """INSERT INTO control_plane.king_reigns
                   (competition_id,reign_number,model_digest,public_bucket,public_prefix,hotkey,uid,
                    crowned_at,crowned_finalized_block,operator_provenance)
                   VALUES (%s,0,%s,'fixture',%s,%s,%s,now(),100,'fixture') RETURNING reign_id""",
                (cid, digest, f"models/sha256/{digest}/", key, i),
            ).fetchone()["reign_id"]
            db.execute("UPDATE control_plane.competitions SET current_reign_id=%s WHERE competition_id=%s", (reign, cid))
            if key == "main":
                main_id, main_reign = cid, reign
        h, w = replacement_rewards(state)
        uids = list(range(len(h)))
        db.execute(
            """INSERT INTO control_plane.weight_publications
               (competition_id,source_reign_id,policy_version,policy_hotkeys,policy_weights,
                target_hotkeys,target_uids,normalized_weights,payload_sha256,mapping_finalized_block,
                idempotency_key,state,last_attempted_block,next_due_block)
               VALUES (%s,%s,'fixture',%s,%s,%s,%s,%s,%s,100,%s,'requested',100,201)""",
            (main_id, main_reign, list(h), list(w), list(h), uids, list(w),
             weight_payload_digest(uids, w), f"publish-weights:{main_reign}"),
        )
        monkeypatch.setenv("TEUTONIC_DATABASE_URL", url)
        monkeypatch.setenv("TEUTONIC_NETUID", "999")
        monkeypatch.setenv("TEUTONIC_CHAIN_GENERATION", "reward-test")

        def run(name, apply=False):
            path = tmp_path / name
            monkeypatch.setattr("sys.argv", ["retire_mix_competitions", "--output", str(path), *(["--apply"] if apply else [])])
            main()
            return json.loads(path.read_text())

        before = db.execute("SELECT * FROM control_plane.weight_publications").fetchone()
        preview = run("preview.json")
        assert preview["plan"]["policy_weights"] == [.4, .2, .2, .2]
        assert db.execute("SELECT * FROM control_plane.weight_publications").fetchone() == before
        assert db.execute("SELECT count(*) AS n FROM control_plane.competitions WHERE active").fetchone()["n"] == 7
        run("backup.json", True)
        active = db.execute("SELECT competition_key AS name FROM control_plane.competitions WHERE active ORDER BY name").fetchall()
        assert [r["name"] for r in active] == ["main", "mix01", "mix06", "mix07"]
        after = db.execute("SELECT * FROM control_plane.weight_publications").fetchone()
        assert after["policy_weights"] == after["normalized_weights"] == [.4, .2, .2, .2]
        assert after["payload_revision"] == 2 and after["next_due_block"] == 201
        assert db.execute("SELECT count(*) AS n FROM control_plane.king_reigns").fetchone()["n"] == 7
        assert not run("replay.json", True)["plan"]["changed"]
        assert db.execute("SELECT * FROM control_plane.weight_publications").fetchone() == after
