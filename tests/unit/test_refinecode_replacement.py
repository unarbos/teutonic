from copy import deepcopy
import pytest
from teutonic.weights.policy import (
    FOUR_COMPETITION_POLICY, REFINECODE_POLICY, REFINECODE_REPLACEMENT_POLICY,
    begin_refinecode_replacement, replacement_rewards, replacement_after_win, validate_transition,
)


def initial(shared=False):
    old = {"version": FOUR_COMPETITION_POLICY, "main": {"hotkey": "main"},
           "mixes": {"mix01": "one", "mix06": "one" if shared else "six", "mix07": "seven"}, "legacy": []}
    # Reversed input order makes the FIFO ordering check independent of names.
    retired = [{"id": k, "hotkey": old["mixes"][k], "reign_id": k, "crowned_block": n}
               for k, n in (("mix06", 200), ("mix01", 100))]
    return begin_refinecode_replacement(old, {"main2": "main", "mix07": "seven"}, retired)


def rewards(state):
    return dict(zip(*replacement_rewards(state)))


@pytest.mark.parametrize('first,second', [('mix08','mix09'), ('mix09','mix08')])
def test_first_crowns_displace_oldest_once_and_later_wins_only_replace_own_share(first, second):
    state = initial()
    assert rewards(state) == {'main': .4, 'seven': .2, 'one': .2, 'six': .2}
    saved = deepcopy(state)
    first_state = replacement_after_win(state, first, 'winner1')
    assert state == saved
    assert rewards(first_state) == {'main': .4, 'seven': .2, 'winner1': .2, 'six': .2}
    assert first_state['evictions'][0]['entry']['id'] == 'mix01'
    repeat = replacement_after_win(first_state, first, 'winner2')
    assert repeat['legacy'] == first_state['legacy']
    assert repeat['evictions'] == first_state['evictions']
    final = replacement_after_win(repeat, second, 'winner3')
    assert rewards(final) == {'main': .4, 'seven': .2, 'winner2': .2, 'winner3': .2}
    assert final['legacy'] == []
    assert [e['entry']['id'] for e in final['evictions']] == ['mix01', 'mix06']


def test_main2_and_mix07_wins_do_not_displace_retired_positions():
    state = initial()
    for key in ('main2', 'mix07'):
        state = replacement_after_win(state, key, key)
    assert len(state['legacy']) == 2 and state['evictions'] == []
    assert rewards(state) == {'main2': .4, 'mix07': .2, 'one': .2, 'six': .2}


def test_shared_hotkeys_merge_but_reward_slots_are_still_displaced():
    state = initial(shared=True)
    assert rewards(state)['one'] == pytest.approx(.4)
    state = replacement_after_win(state, 'mix08', 'one')
    assert rewards(state)['one'] == pytest.approx(.4)
    state = replacement_after_win(state, 'mix09', 'main')
    assert rewards(state) == pytest.approx({'main': .6, 'seven': .2, 'one': .2})


def test_initial_opponents_do_not_receive_slots_and_retired_competitions_cannot_win():
    state = initial()
    assert set(state['kings']) == {'main2', 'mix07'}
    for key in ('main','mix01','mix06','mix10'):
        with pytest.raises(ValueError, match='retired'):
            replacement_after_win(state, key, 'winner')
    with pytest.raises(ValueError):
        replacement_after_win(state, 'mix08', '')
    with pytest.raises(ValueError):
        begin_refinecode_replacement(state, state['kings'], state['legacy'])


@pytest.mark.parametrize('mutation', ['reverse','missing','duplicate','unknown','wrong-eviction'])
def test_corrupt_slot_accounting_is_rejected(mutation):
    state = initial()
    if mutation == 'reverse': state['legacy'].reverse()
    elif mutation == 'missing': state['legacy'].pop()
    elif mutation == 'duplicate': state['legacy'][1] = deepcopy(state['legacy'][0])
    elif mutation == 'unknown': state['kings']['mix10'] = 'bad'
    else: state['evictions'] = [{'competition':'mix08','entry':state['legacy'][0]}]
    with pytest.raises(ValueError): validate_transition(state)


def test_v5_history_keeps_immediate_starting_rewards_but_can_be_explicitly_migrated():
    old = {'version': REFINECODE_POLICY, 'kings': {'main2':'main','mix07':'seven','mix08':'v9','mix09':'v9'}}
    assert rewards(old) == {'main':.4,'seven':.2,'v9':.4}
    new = begin_refinecode_replacement(old, {'main2':'main','mix07':'seven'}, initial()['legacy'])
    assert new['version'] == REFINECODE_REPLACEMENT_POLICY
    assert 'v9' not in rewards(new)


def repair_fixture():
    state = {'version':REFINECODE_POLICY,'kings':{'main2':'main','mix07':'seven','mix08':'v9','mix09':'v9'}}
    rows = []
    for key, hotkey in {'main':'main','main2':'main','mix07':'seven','mix08':'v9','mix09':'v9','mix01':'one','mix06':'six'}.items():
        rows.append({'competition_key':key,'competition_id':key,'active':key in state['kings'],
                     'main_competition_id':None if key=='main' else 'main',
                     'current_reign_id':None if key in ('main2','mix08','mix09') else key+'-reign',
                     'king_hotkey':hotkey,'king_crowned_block':100 if key=='mix06' else 200,
                     'reward_transition':state,'reward_reign_id':'main-reign'})
    h,w=replacement_rewards(state)
    publication={'source_reign_id':'main-reign','policy_hotkeys':h,'policy_weights':w}
    return rows,publication,{'main':1,'seven':2,'one':3,'six':4}


def test_repair_preserves_pins_and_restores_slots_from_actual_retired_crown_order():
    from scripts.activate_refinecode_replacement import prepare
    args=repair_fixture();saved=deepcopy(args)
    plan=prepare(*args)
    assert args==saved
    assert plan['target_uids']==[1,2,4,3]
    assert plan['normalized_weights']==pytest.approx([.4,.2,.2,.2])
    assert [e['id'] for e in plan['transition']['legacy']]==['mix06','mix01']
    assert plan['retired_ids']==[]


@pytest.mark.parametrize('bad',['crowned','wrong-pin','publication','missing-uid','active-retired'])
def test_repair_refuses_unsafe_live_state(bad):
    from scripts.activate_refinecode_replacement import prepare
    rows,pub,mapping=repair_fixture();by={r['competition_key']:r for r in rows}
    if bad=='crowned': by['mix08']['current_reign_id']='winner'
    elif bad=='wrong-pin': by['mix09']['king_hotkey']='someone-else'
    elif bad=='publication': pub['policy_weights']=[.5,.2,.3]
    elif bad=='missing-uid': mapping.pop('one')
    else: by['mix01']['active']=True
    with pytest.raises(ValueError):prepare(rows,pub,mapping)
