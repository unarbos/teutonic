"""Exercise the staged configuration migration in a disposable PostgreSQL database."""
import hashlib
import json
import os
from pathlib import Path

import psycopg
import pytest
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

DATABASE_URL = os.environ.get('TEUTONIC_TEST_DATABASE_URL')
pytestmark = pytest.mark.skipif(not DATABASE_URL, reason='disposable PostgreSQL required')
RELEASE = Path(__file__).resolve().parents[2] / 'deployment/packed-text-20261112'


def test_published_release_cutover_guards_idempotence_and_rollback():
    """Rehearse the actual release SQL and exact published mix bytes, without R2."""
    import chain_config

    sql = (RELEASE / 'cutover.sql').read_text()
    mixes = {name: json.loads((RELEASE / f'{name}.json').read_text())
             for name in ('main', 'mix01', 'mix06', 'mix07')}
    receipt = json.loads((RELEASE / 'publication.json').read_text())
    for entry in receipt['objects']:
        if entry['local'] != 'sampling.json':
            assert hashlib.sha256((RELEASE / entry['local']).read_bytes()).hexdigest() == entry['sha256']
    with psycopg.connect(DATABASE_URL, autocommit=True, row_factory=dict_row) as db:
        assert db.execute('SELECT current_database() AS name').fetchone()['name'] == 'teutonic_test'
        db.execute('TRUNCATE control_plane.competitions, control_plane.registrations CASCADE')
        before = {}
        for number, (name, mix) in enumerate(mixes.items(), 1):
            cid = db.execute('INSERT INTO control_plane.competitions(netuid,chain_generation,name,competition_key) '
                             'VALUES (3,%s,%s,%s) RETURNING competition_id',
                             (chain_config.CHAIN_GENERATION,
                              'Teutonic-II-110B' if name == 'main' else name, name)).fetchone()['competition_id']
            old = {'category_weights': {r['category']: r['weight'] for r in mix['categories']},
                   'category_sources': {r['category']: r['dataset'] for r in mix['categories']}}
            eid = db.execute('INSERT INTO control_plane.evaluation_configs '
                             '(competition_id,config_version,dataset_label,eval_n,delta_threshold,long_documents,active) '
                             'VALUES (%s,%s,%s,0,.0015,%s,true) RETURNING evaluation_config_id',
                             (cid, f'{number:064x}', name, Jsonb({'token_budget':70000000}))).fetchone()['evaluation_config_id']
            db.execute('INSERT INTO control_plane.dataset_manifests '
                       '(evaluation_config_id,position,name,manifest_url,manifest_sha256,manifest_json,sample_proportion) '
                       'VALUES (%s,0,%s,%s,%s,%s,1)',
                       (eid,name,f'https://fixture.invalid/{name}.json','a'*64,Jsonb(old)))
            db.execute('INSERT INTO control_plane.evaluation_early_stopping_policies(competition_id,enabled) VALUES (%s,true)',(cid,))
            before[name] = (cid, eid, old)
        def rollback():
            db.execute('UPDATE control_plane.evaluation_configs SET active=false WHERE active')
            for cid, eid, _ in before.values():
                db.execute('UPDATE control_plane.evaluation_configs SET active=true WHERE evaluation_config_id=%s',(eid,))
                db.execute('UPDATE control_plane.evaluation_early_stopping_policies SET enabled=true WHERE competition_id=%s',(cid,))
        def active():
            return db.execute('SELECT c.competition_key AS name,ec.evaluation_config_id,ec.long_documents,p.enabled '
                              'FROM control_plane.competitions c JOIN control_plane.evaluation_configs ec USING(competition_id) '
                              'JOIN control_plane.evaluation_early_stopping_policies p USING(competition_id) WHERE ec.active ORDER BY c.name').fetchall()
        try:
            db.execute((RELEASE.parents[1] / 'scripts/db/allow_packed_text_manifests.sql').read_text())
            # Historical retries must remain unchanged once a later attempt finished.
            # A retry without that evidence must still block the migration.
            cid = before['main'][0]
            registration = 'e' * 64
            db.execute("INSERT INTO control_plane.registrations "
                       "(registration_id,netuid,chain_generation,uid,hotkey,first_seen_finalized_block,"
                       "last_seen_finalized_block,model_prefix,state) VALUES (%s,3,%s,0,'fixture',1,1,%s,'active')",
                       (registration, chain_config.CHAIN_GENERATION, f'models/registrations/{registration}/'))
            db.execute("INSERT INTO control_plane.r2_parent_tokens "
                       "(registration_id,cloudflare_token_id,access_key_id,state,activated_at) "
                       "VALUES (%s,'cutover-fixture','cutover-fixture','active',now())", (registration,))
            upload = db.execute("INSERT INTO control_plane.uploads "
                                "(registration_id,chain_generation,state,signalling_hotkey,ready_payload,"
                                "ready_finalized_block,ready_extrinsic_index,ready_event_index,manifest_sha256,ready_at) "
                                "VALUES (%s,%s,'retry_pending','fixture','fixture',1,0,0,%s,now()) RETURNING upload_id",
                                (registration, chain_config.CHAIN_GENERATION, 'a' * 64)).fetchone()['upload_id']
            db.execute("INSERT INTO control_plane.verified_uploads "
                       "(upload_id,immutable_bucket,immutable_prefix,model_digest,manifest_sha256,"
                       "object_count,total_size_bytes,verified_at) VALUES (%s,'fixture',%s,%s,%s,1,1,now())",
                       (upload, f'models/registrations/{registration}/', 'a' * 64, 'a' * 64))
            reign = db.execute("INSERT INTO control_plane.king_reigns "
                               "(competition_id,reign_number,model_digest,public_bucket,public_prefix,hotkey,uid,"
                               "crowned_at,crowned_finalized_block,operator_provenance) "
                               "VALUES (%s,0,%s,'fixture',%s,'fixture',0,now(),1,'fixture') RETURNING reign_id",
                               (cid, 'a' * 64, f'models/sha256/{"a" * 64}/')).fetchone()['reign_id']
            def attempt(number, state):
                return db.execute("INSERT INTO control_plane.evaluations "
                                  "(upload_id,competition_id,attempt_number,claimed_king_reign_id,state,policy_version,"
                                  "code_version,dataset_version,sampling_seed,bootstrap_seed,thresholds) "
                                  "VALUES (%s,%s,%s,%s,%s,'old','old','old',0,0,'{}') RETURNING evaluation_id",
                                  (upload, cid, number, reign, state)).fetchone()['evaluation_id']
            retry = attempt(1, 'retryable_failure')
            retry_before = db.execute('SELECT * FROM control_plane.evaluations WHERE evaluation_id=%s', (retry,)).fetchone()
            for upload_state in ('retry_pending', 'rejected'):
                db.execute('UPDATE control_plane.uploads SET state=%s WHERE upload_id=%s', (upload_state, upload))
                with pytest.raises(psycopg.errors.RaiseException, match='Drain active evaluations'):
                    db.execute(sql)
                db.execute('ROLLBACK')
            attempt(2, 'terminal_failure')
            for _ in range(2):
                db.execute(sql)
                assert db.execute('SELECT count(*) AS n FROM control_plane.evaluation_configs').fetchone()['n'] == 8
                assert all(r['long_documents'] == mixes[r['name']] and not r['enabled'] for r in active())
                assert db.execute('SELECT * FROM control_plane.evaluations WHERE evaluation_id=%s', (retry,)).fetchone() == retry_before
            rollback()
            original = active()
            assert all(r['evaluation_config_id'] == before[r['name']][1] and r['enabled'] for r in original)
            # A mismatch in the LAST competition must roll back preceding updates too.
            cid, eid, old = before['mix07']
            for broken in ('budget','weights','sources'):
                if broken == 'budget':
                    db.execute('UPDATE control_plane.evaluation_configs SET long_documents=%s WHERE evaluation_config_id=%s',
                               (Jsonb({'token_budget':69000000}),eid))
                else:
                    changed = json.loads(json.dumps(old))
                    field = 'category_weights' if broken == 'weights' else 'category_sources'
                    changed[field][next(iter(changed[field]))] = .9 if broken == 'weights' else 'wrong-source'
                    db.execute('UPDATE control_plane.dataset_manifests SET manifest_json=%s WHERE evaluation_config_id=%s',(Jsonb(changed),eid))
                with pytest.raises(psycopg.errors.RaiseException, match='Active category weights, sources or budget differ'):
                    db.execute(sql)
                db.execute('ROLLBACK')
                assert all(r['evaluation_config_id'] == before[r['name']][1] and r['enabled'] for r in active())
                db.execute('UPDATE control_plane.evaluation_configs SET long_documents=%s WHERE evaluation_config_id=%s',
                           (Jsonb({'token_budget':70000000}),eid))
                db.execute('UPDATE control_plane.dataset_manifests SET manifest_json=%s WHERE evaluation_config_id=%s',(Jsonb(old),eid))
            db.execute(sql)
            assert all(r['long_documents'] == mixes[r['name']] for r in active())
        finally:
            db.execute('ROLLBACK')
            db.execute('TRUNCATE control_plane.competitions, control_plane.registrations CASCADE')


def test_60m_budget_update_preserves_pins_and_rejects_drift():
    release = RELEASE.with_name('packed-text-60m-20261113')
    scope = json.loads((release / 'scope.json').read_text())
    configs = json.loads((release / 'previous-configs.json').read_text())
    sql = (release / 'cutover.sql').read_text()
    entries = json.loads((release / 'pending-publication.json').read_text())
    mixes = {name: json.loads((release / f'{name}.json').read_text()) for name in configs}
    for entry in entries:
        if entry['local'] != 'sampling.json':
            assert hashlib.sha256((release / entry['local']).read_bytes()).hexdigest() == entry['sha256']
    with psycopg.connect(DATABASE_URL, autocommit=True, row_factory=dict_row) as db:
        assert db.execute('SELECT current_database() AS name').fetchone()['name'] == 'teutonic_test'
        db.execute('TRUNCATE control_plane.competitions, control_plane.registrations CASCADE')
        before = {}
        try:
            for name, mix in mixes.items():
                old = {**mix, 'token_budget': 35_000_000}
                cid = db.execute('INSERT INTO control_plane.competitions(netuid,chain_generation,name,competition_key) '
                                 'VALUES (%s,%s,%s,%s) RETURNING competition_id',
                                 (scope['netuid'], scope['chain_generation'], name, name)).fetchone()['competition_id']
                eid = db.execute('INSERT INTO control_plane.evaluation_configs '
                                 '(competition_id,config_version,dataset_label,eval_n,delta_threshold,shards_per_dataset,long_documents,active) '
                                 'VALUES (%s,%s,%s,0,.00075,1,%s,true) RETURNING evaluation_config_id',
                                 (cid, configs[name], name, Jsonb(old))).fetchone()['evaluation_config_id']
                db.execute('INSERT INTO control_plane.dataset_manifests '
                           '(evaluation_config_id,position,name,manifest_url,manifest_sha256,manifest_json,sample_proportion) '
                           'VALUES (%s,0,%s,%s,%s,%s,1)',
                           (eid, name, f'https://fixture.invalid/{name}', 'a'*64, Jsonb(old)))
                db.execute('INSERT INTO control_plane.evaluation_early_stopping_policies(competition_id,enabled) VALUES (%s,true)', (cid,))
                before[name] = (eid, old)
            # Last competition drift must roll back all preceding updates.
            eid, old = before['mix07']
            for field, value in [('token_budget', 36_000_000),
                                 ('dataset', {**old['dataset'], 'sha256': 'b'*64}),
                                 ('categories', [{**c, 'weight': c['weight'] / 2} for c in old['categories']])]:
                changed = {**old, field: value}
                db.execute('UPDATE control_plane.evaluation_configs SET long_documents=%s WHERE evaluation_config_id=%s', (Jsonb(changed), eid))
                db.execute('UPDATE control_plane.dataset_manifests SET manifest_json=%s WHERE evaluation_config_id=%s', (Jsonb(changed), eid))
                with pytest.raises(psycopg.errors.RaiseException, match='Active category weights'):
                    db.execute(sql)
                db.execute('ROLLBACK')
                assert db.execute('SELECT count(*) AS n FROM control_plane.evaluation_configs').fetchone()['n'] == 4
            db.execute('UPDATE control_plane.evaluation_configs SET long_documents=%s WHERE evaluation_config_id=%s', (Jsonb(old), eid))
            db.execute('UPDATE control_plane.dataset_manifests SET manifest_json=%s WHERE evaluation_config_id=%s', (Jsonb(old), eid))
            for _ in range(2):
                db.execute(sql)
                active = db.execute('SELECT c.competition_key,ec.long_documents FROM control_plane.competitions c '
                                    'JOIN control_plane.evaluation_configs ec USING(competition_id) WHERE ec.active').fetchall()
                assert {r['competition_key']: r['long_documents'] for r in active} == mixes
                assert db.execute('SELECT count(*) AS n FROM control_plane.evaluation_configs').fetchone()['n'] == 8
                assert all(r['enabled'] for r in db.execute('SELECT enabled FROM control_plane.evaluation_early_stopping_policies'))
                for eid, old in before.values():
                    assert db.execute('SELECT long_documents FROM control_plane.evaluation_configs WHERE evaluation_config_id=%s', (eid,)).fetchone()['long_documents'] == old
        finally:
            db.execute('ROLLBACK')
            db.execute('TRUNCATE control_plane.competitions, control_plane.registrations CASCADE')
