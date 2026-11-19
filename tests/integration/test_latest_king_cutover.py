import json
import os

import psycopg
import pytest
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from scripts.activate_latest_king_bonus import main
from teutonic.weights.contracts import weight_payload_digest
from teutonic.weights.policy import FOUR_COMPETITION_POLICY, replacement_rewards


@pytest.mark.skipif(not os.environ.get("TEUTONIC_TEST_DATABASE_URL"), reason="disposable PostgreSQL required")
def test_bonus_preview_apply_rollback_and_idempotence(tmp_path, monkeypatch):
    url = os.environ["TEUTONIC_TEST_DATABASE_URL"]
    with psycopg.connect(url, autocommit=True, row_factory=dict_row) as db:
        assert db.execute("SELECT current_database() AS name").fetchone()["name"] == "teutonic_test"
        db.execute("TRUNCATE control_plane.competitions CASCADE")
        keys = ("main", "mix01", "mix06", "mix07")
        state = {"version": FOUR_COMPETITION_POLICY,
                 "main": {"id": "main", "hotkey": "main", "weight": .4},
                 "mixes": {k: k for k in keys[1:]}, "legacy": []}
        main_id = None
        reigns = {}
        for i, key in enumerate(keys):
            cid = db.execute(
                """INSERT INTO control_plane.competitions
                   (netuid,chain_generation,name,competition_key,main_competition_id,reward_transition)
                   VALUES (999,'bonus-test',%s,%s,%s,%s) RETURNING competition_id""",
                ("Teutonic-II-110B" if key == "main" else key,
                 key, main_id, Jsonb(state) if key == "main" else None),
            ).fetchone()["competition_id"]
            digest = f"{i + 1:064x}"
            reigns[key] = db.execute(
                """INSERT INTO control_plane.king_reigns
                   (competition_id,reign_number,model_digest,public_bucket,public_prefix,hotkey,uid,
                    crowned_at,crowned_finalized_block,operator_provenance)
                   VALUES (%s,0,%s,'fixture',%s,%s,%s,now(),100,'fixture') RETURNING reign_id""",
                (cid, digest, f"models/sha256/{digest}/", key, i),
            ).fetchone()["reign_id"]
            db.execute("UPDATE control_plane.competitions SET current_reign_id=%s WHERE competition_id=%s", (reigns[key], cid))
            if key == "main": main_id = cid
        # The reward pointer, not row order or the king's name, identifies the
        # last crown. MIX07 was inserted after MIX06 in this synthetic fixture.
        latest = reigns["mix06"]
        db.execute("UPDATE control_plane.competitions SET reward_reign_id=%s WHERE competition_id=%s", (latest, main_id))
        h, w = replacement_rewards(state)
        db.execute(
            """INSERT INTO control_plane.weight_publications
               (competition_id,source_reign_id,policy_version,policy_hotkeys,policy_weights,
                target_hotkeys,target_uids,normalized_weights,payload_sha256,mapping_finalized_block,
                idempotency_key,state,last_attempted_block,next_due_block)
               VALUES (%s,%s,'fixture',%s,%s,%s,%s,%s,%s,100,%s,'requested',100,201)""",
            (main_id, latest, list(h), list(w), list(h), [0, 1, 2, 3], list(w),
             weight_payload_digest([0, 1, 2, 3], w), f"publish-weights:{latest}"),
        )
        monkeypatch.setenv("TEUTONIC_DATABASE_URL", url)
        monkeypatch.setenv("TEUTONIC_NETUID", "999")
        monkeypatch.setenv("TEUTONIC_CHAIN_GENERATION", "bonus-test")

        def run(name, apply=False):
            path = tmp_path / name
            monkeypatch.setattr("sys.argv", ["activate_latest_king_bonus", "--output", str(path), *(["--apply"] if apply else [])])
            main()
            return json.loads(path.read_text())

        before = db.execute("SELECT * FROM control_plane.weight_publications").fetchone()
        kings_before = db.execute("SELECT * FROM control_plane.king_reigns ORDER BY reign_id").fetchall()
        preview = run("preview.json")
        assert preview["plan"]["policy_weights"] == [.2, .2, .4, .2]
        assert db.execute("SELECT * FROM control_plane.weight_publications").fetchone() == before
        # Failure to preserve the recovery file aborts the database transaction.
        with pytest.raises(FileExistsError): run("preview.json", True)
        assert db.execute("SELECT * FROM control_plane.weight_publications").fetchone() == before
        assert db.execute("SELECT reward_transition FROM control_plane.competitions WHERE competition_id=%s", (main_id,)).fetchone()["reward_transition"] == state
        run("before.json", True)
        after = db.execute("SELECT * FROM control_plane.weight_publications").fetchone()
        assert after["policy_weights"] == after["normalized_weights"] == [.2, .2, .4, .2]
        assert after["payload_revision"] == 2 and after["next_due_block"] == 201
        assert after["source_reign_id"] == latest
        assert db.execute("SELECT count(*) AS n FROM control_plane.competitions WHERE active").fetchone()["n"] == 4
        assert db.execute("SELECT * FROM control_plane.king_reigns ORDER BY reign_id").fetchall() == kings_before
        assert not run("replay.json", True)["plan"]["changed"]
        assert db.execute("SELECT * FROM control_plane.weight_publications").fetchone() == after
