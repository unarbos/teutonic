import numpy as np
import pytest

from teutonic.evaluation.early_stopping import EarlyStoppingPolicy, challenger_futility_decision
from teutonic.evaluation.masking import DOCUMENT_EOS_TOKEN_ID as EOS
from teutonic.evaluation.masking import scored_token_counts
from teutonic.evaluation.policy import paired_bootstrap_verdict, provisional_paired_bootstrap


def test_targets_include_terminal_eos_but_never_cross_a_boundary():
    assert scored_token_counts([[1, 2, EOS, 3, 4, EOS], [EOS, EOS, 1, 2, EOS, 3]]) == [4, 2]
    assert scored_token_counts([[1, 2, 3]]) == [2]
    with pytest.raises(ValueError, match="at least one"):
        scored_token_counts([[EOS, EOS, EOS]])


def test_bootstrap_resamples_windows_and_recomputes_token_weighted_ratio():
    king, challenger, counts = [2.0, 1.0, 3.0], [1.0, 2.0, 1.0], [1, 9, 2]
    result = paired_bootstrap_verdict(
        king,
        challenger,
        scored_tokens=counts,
        bootstrap_seed=7,
        n_bootstrap=100,
        alpha=0.1,
        delta_threshold=0.0,
    )
    # Independent sum/count oracle. Sampling tokens or averaging window means
    # would produce a different uncertainty interval and point estimate.
    differences = np.array(king) - challenger
    rng = np.random.default_rng(7)
    draws = []
    for _ in range(100):
        indices = rng.integers(0, 3, size=3)
        draws.append(
            sum(differences[i] * counts[i] for i in indices) / sum(counts[i] for i in indices)
        )
    assert result["mu_hat"] == round(-4 / 12, 6)
    assert result["avg_king_loss"] == round(17 / 12, 6)
    assert result["lcb"] == round(float(np.quantile(draws, 0.1)), 6)
    assert not result["accepted"]
    progress = provisional_paired_bootstrap(
        king,
        challenger,
        scored_tokens=counts,
        bootstrap_seed=7,
        n_bootstrap=100,
        alpha=0.1,
        delta_threshold=0.0,
    )
    assert progress["provisional_mu_hat"] == result["mu_hat"]
    assert progress["provisional_lcb"] == result["lcb"]


@pytest.mark.parametrize("counts", [[0, 1], [1], [1, -1], [1, float("nan")], [1, 1.5]])
def test_invalid_weights_are_rejected(counts):
    with pytest.raises(ValueError, match="scored token counts"):
        paired_bootstrap_verdict(
            [1.0, 2.0],
            [2.0, 1.0],
            scored_tokens=counts,
            bootstrap_seed=0,
            n_bootstrap=10,
            alpha=0.1,
            delta_threshold=0,
        )


def test_equal_counts_reproduce_original_bootstrap():
    kwargs = {
        "bootstrap_seed": 8,
        "n_bootstrap": 100,
        "alpha": 0.1,
        "delta_threshold": 0.01,
        "now": lambda: "fixed",
    }
    losses = ([1.25, 2.0, 4.5], [1.0, 3.0, 4.0])
    assert paired_bootstrap_verdict(*losses, **kwargs) == paired_bootstrap_verdict(
        *losses, **kwargs, scored_tokens=[8, 8, 8]
    )


def test_futility_projects_remaining_tokens_not_remaining_windows():
    decision = challenger_futility_decision(
        [2.0, 0.0],
        [0.0, 1.0],
        total_sequences=3,
        scored_tokens=[1, 9],
        total_scored_tokens=11,
        delta_threshold=0,
        policy=EarlyStoppingPolicy(enabled=True, advantage_quantile=1.0),
    )
    assert decision["mu_hat"] == -0.7
    assert decision["mu_hat_upper_bound"] == -5 / 11
    assert (
        challenger_futility_decision(
            [2.0, 0.0],
            [0.0, 1.0],
            total_sequences=3,
            scored_tokens=[1, 9],
            total_scored_tokens=30,
            delta_threshold=0,
            policy=EarlyStoppingPolicy(enabled=True, advantage_quantile=1.0),
        )
        is None
    )
