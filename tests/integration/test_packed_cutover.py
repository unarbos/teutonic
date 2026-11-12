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
        db.execute('TRUNCATE control_plane.competitions CASCADE')
        before = {}
        for number, (name, mix) in enumerate(mixes.items(), 1):
            cid = db.execute('INSERT INTO control_plane.competitions(netuid,chain_generation,name) '
                             'VALUES (3,%s,%s) RETURNING competition_id',
                             (chain_config.CHAIN_GENERATION, name)).fetchone()['competition_id']
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
            return db.execute('SELECT c.name,ec.evaluation_config_id,ec.long_documents,p.enabled '
                              'FROM control_plane.competitions c JOIN control_plane.evaluation_configs ec USING(competition_id) '
                              'JOIN control_plane.evaluation_early_stopping_policies p USING(competition_id) WHERE ec.active ORDER BY c.name').fetchall()
        try:
            db.execute((RELEASE.parents[1] / 'scripts/db/allow_packed_text_manifests.sql').read_text())
            for _ in range(2):
                db.execute(sql)
                assert db.execute('SELECT count(*) AS n FROM control_plane.evaluation_configs').fetchone()['n'] == 8
                assert all(r['long_documents'] == mixes[r['name']] and not r['enabled'] for r in active())
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
            db.execute('TRUNCATE control_plane.competitions CASCADE')
