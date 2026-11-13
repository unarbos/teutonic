import pytest

from teutonic.evaluation.reference_policy import ACTIVE_POLICY_VERSION, reference_verdict


def score(k, c, r, weights, **kwargs):
    return reference_verdict(k, c, r, weights, bootstrap_seed=42, n_bootstrap=200,
                             alpha=.05, delta_threshold=.00075, **kwargs)


def test_both_zero_documents_are_removed_from_denominator_and_bootstrap():
    # Last two samples are already at/better than Horizon, including exact parity.
    k, c, r, weights = [5., 3., 1., 2.], [3., 2., 2., 2.], [2.]*4, [10, 30, 100, 200]
    active = score(k, c, r, weights, skip_both_zero=True)
    subset = score(k[:2], c[:2], r[:2], weights[:2])
    old = score(k, c, r, weights, skip_both_zero=False)
    assert active['mu_hat'] == subset['mu_hat']
    assert active['lcb'] == subset['lcb']
    assert active['mu_hat'] == pytest.approx(old['mu_hat'] * 340 / 40)
    assert active['n_sequences'] == 4
    assert active['n_scoring_sequences'] == active['n_skipped_sequences'] == 2
    assert active['scoring_bytes'] == 40
    assert active['skipped_bytes'] == 300
    for role in ('king', 'challenger', 'reference'):
        assert active[f'avg_{role}_loss'] == old[f'avg_{role}_loss']
    assert active['scoring_policy'] == ACTIVE_POLICY_VERSION
    default = score(k, c, r, weights)
    assert default['scoring_policy'] == active['scoring_policy']
    assert default['mu_hat'] == active['mu_hat']
    assert default['lcb'] == active['lcb']


def test_equal_positive_deficits_and_regressions_are_kept():
    # Equal positive deficits have zero improvement but must stay in the sample.
    result = score([3., 1.], [3., 4.], [2., 2.], [10, 10], skip_both_zero=True)
    old = score([3., 1.], [3., 4.], [2., 2.], [10, 10])
    assert result['n_scoring_sequences'] == 2
    assert result['mu_hat'] == old['mu_hat'] < 0
    assert result['lcb'] == old['lcb']
    assert not result['accepted']


def test_no_eligible_documents_keeps_king_without_nan():
    result = score([1., 2.], [2., 1.], [2., 2.], [10, 20], skip_both_zero=True)
    assert result['n_scoring_sequences'] == result['scoring_bytes'] == 0
    assert result['n_skipped_sequences'] == 2
    assert result['mu_hat'] == result['lcb'] == 0
    assert not result['accepted']


def test_filter_does_not_hide_invalid_samples():
    with pytest.raises(ValueError, match='finite'):
        score([float('nan')], [1.], [2.], [10], skip_both_zero=True)
