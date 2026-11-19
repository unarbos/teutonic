#!/usr/bin/env python3
"""Restore retired reward slots before the first MIX08/MIX09 crown; PostgreSQL only."""
import argparse
import json
import math
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psycopg
from psycopg.rows import dict_row
from scripts.retire_mix_competitions import apply_plan
from teutonic.weights.contracts import weight_payload_digest
from teutonic.weights.policy import (
    REFINECODE_POLICY, REFINECODE_COMPETITIONS, begin_refinecode_replacement,
    mapped_weight_plan, replacement_rewards,
)
from teutonic.weights.transition import lock_drained_services


def prepare(rows, publication, uid_by_hotkey):
    by_key = {r['competition_key']: r for r in rows}
    if len(by_key) != len(rows) or {k for k, r in by_key.items() if r['active']} != set(REFINECODE_COMPETITIONS):
        raise ValueError('requires the four active RefineCode competitions')
    root = by_key['main']
    old = root['reward_transition']
    if old.get('version') != REFINECODE_POLICY:
        raise ValueError('requires the immediate RefineCode policy before replacement activation')
    for key in REFINECODE_COMPETITIONS:
        row = by_key[key]
        if row['main_competition_id'] != root['competition_id'] or row['king_hotkey'] != old['kings'][key]:
            raise ValueError('starting/current king differs from existing reward position')
    if any(by_key[k]['current_reign_id'] is not None for k in ('mix08', 'mix09')):
        raise ValueError('cannot restore retired slots after a MIX08/MIX09 crown')
    if not publication or publication['source_reign_id'] != root['reward_reign_id']:
        raise ValueError('missing current reward publication')
    expected = dict(zip(*replacement_rewards(old), strict=True))
    actual = {}
    for h, w in zip(publication['policy_hotkeys'], publication['policy_weights'], strict=True):
        if not math.isfinite(w) or w < 0:
            raise ValueError('invalid current reward weight')
        actual[h] = actual.get(h, 0) + w
    if set(actual) != set(expected) or any(abs(actual[h] - expected[h]) > 1e-9 for h in expected):
        raise ValueError('current publication differs from reward policy')
    retired = []
    for key in ('mix01', 'mix06'):
        row = by_key[key]
        if row['active'] or not row['current_reign_id'] or row['main_competition_id'] != root['competition_id']:
            raise ValueError('missing retired competition king')
        retired.append({'id': key, 'hotkey': row['king_hotkey'], 'reign_id': str(row['current_reign_id']),
                        'crowned_block': row['king_crowned_block']})
    state = begin_refinecode_replacement(old, {k: old['kings'][k] for k in ('main2', 'mix07')}, retired)
    hotkeys, shares = replacement_rewards(state)
    if not set(hotkeys) <= set(uid_by_hotkey):
        raise ValueError('reward recipient missing from finalized metagraph')
    targets, uids, weights = mapped_weight_plan(hotkeys, shares, uid_by_hotkey, burn_uid=0)
    return {'changed': True, 'transition': state, 'retired_ids': [],
            'policy_hotkeys': list(hotkeys), 'policy_weights': list(shares),
            'target_hotkeys': list(targets), 'target_uids': list(uids),
            'normalized_weights': list(weights), 'payload_sha256': weight_payload_digest(uids, weights)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='new before-state/preview file')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    scope = (int(os.environ['TEUTONIC_NETUID']), os.environ['TEUTONIC_CHAIN_GENERATION'])
    with psycopg.connect(os.environ['TEUTONIC_DATABASE_URL'], row_factory=dict_row) as db:
        if not args.apply:
            db.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY')
        rows = db.execute('''SELECT c.*, k.hotkey AS king_hotkey,
            k.crowned_finalized_block AS king_crowned_block FROM control_plane.competitions c
            LEFT JOIN control_plane.king_reigns k ON k.reign_id=coalesce(c.current_reign_id,c.starting_reign_id)
            WHERE c.netuid=%s AND c.chain_generation=%s ORDER BY c.competition_key'''
            + (' FOR UPDATE OF c' if args.apply else ''), scope).fetchall()
        root = next(r for r in rows if r['competition_key'] == 'main')
        if args.apply:
            lock_drained_services(db, root)
        publication = db.execute('SELECT * FROM control_plane.weight_publications WHERE source_reign_id=%s'
            + (' FOR UPDATE' if args.apply else ''), (root['reward_reign_id'],)).fetchone()
        assignments = db.execute('''SELECT a.hotkey,a.uid FROM control_plane.metagraph_uid_assignments a
            WHERE a.snapshot_id=(SELECT snapshot_id FROM control_plane.metagraph_snapshots
            WHERE netuid=%s AND chain_generation=%s AND is_complete ORDER BY finalized_block DESC LIMIT 1)''', scope).fetchall()
        plan = prepare(rows, publication, {a['hotkey']: a['uid'] for a in assignments})
        pending = db.execute('''SELECT 1 FROM control_plane.weight_submission_attempts
            WHERE weight_publication_id=%s AND state IN
            ('claimed','submitting','submitted','included','retry_pending') LIMIT 1''',
            (publication['weight_publication_id'],)).fetchone()
        if pending:
            raise ValueError('resolve outstanding weight attempt before activation')
        with args.output.open('x') as stream:
            json.dump({'before': {'competitions': rows, 'publication': publication},
                       'plan': plan, 'apply_requested': args.apply}, stream, indent=2, default=str)
        if args.apply:
            apply_plan(db, root, publication, plan)
    print('Replacement policy committed; no chain writes.' if args.apply else 'Read-only preview saved.')


if __name__ == '__main__':
    main()
