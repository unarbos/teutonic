"""The accept/reject rule itself.

`bootstrap_verdict` decides who holds the subnet's emissions. Its contract:
accept iff the alpha-quantile lower confidence bound on the paired loss
improvement clears `delta_threshold` — being merely better is not enough.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from eval_server_quasar_pair import bootstrap_verdict


def _req(*, alpha=0.001, delta=0.0015, n_bootstrap=2000, seed=0xB007):
    return SimpleNamespace(
        alpha=alpha,
        delta_threshold=delta,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=seed,
    )


def _losses(rng, n, king_mean, challenger_mean, spread=0.05):
    """Paired losses with shared per-sequence difficulty, as real evals produce."""
    difficulty = rng.normal(0.0, spread, size=n)
    king = king_mean + difficulty + rng.normal(0.0, 0.001, size=n)
    challenger = challenger_mean + difficulty + rng.normal(0.0, 0.001, size=n)
    return king.tolist(), challenger.tolist()


def test_identical_models_are_rejected():
    losses = [3.0 + 0.01 * i for i in range(500)]
    v = bootstrap_verdict(losses, list(losses), _req())
    assert v["accepted"] is False
    assert v["verdict"] == "king"
    assert v["mu_hat"] == pytest.approx(0.0, abs=1e-9)


def test_clearly_better_challenger_is_accepted():
    rng = np.random.default_rng(0)
    king, challenger = _losses(rng, 2000, 3.0, 3.0 - 0.02)
    v = bootstrap_verdict(king, challenger, _req())
    assert v["accepted"] is True
    assert v["verdict"] == "challenger"
    assert v["lcb"] > v["delta_threshold"]


def test_worse_challenger_is_rejected():
    rng = np.random.default_rng(1)
    king, challenger = _losses(rng, 2000, 3.0, 3.0 + 0.02)
    v = bootstrap_verdict(king, challenger, _req())
    assert v["accepted"] is False
    assert v["mu_hat"] < 0


def test_improvement_below_delta_is_rejected():
    """Better, but not by the required margin — the anti-noise moat."""
    rng = np.random.default_rng(2)
    king, challenger = _losses(rng, 4000, 3.0, 3.0 - 0.0005)
    v = bootstrap_verdict(king, challenger, _req(delta=0.0015))
    assert v["mu_hat"] > 0
    assert v["accepted"] is False


def test_accept_is_exactly_lcb_greater_than_delta():
    rng = np.random.default_rng(3)
    king, challenger = _losses(rng, 1000, 3.0, 3.0 - 0.01)
    v = bootstrap_verdict(king, challenger, _req())
    assert v["accepted"] == (v["lcb"] > v["delta_threshold"])


def test_verdict_is_deterministic_for_a_fixed_seed():
    rng = np.random.default_rng(4)
    king, challenger = _losses(rng, 800, 3.0, 3.0 - 0.01)
    a = bootstrap_verdict(king, challenger, _req(seed=1234))
    b = bootstrap_verdict(king, challenger, _req(seed=1234))
    assert a["lcb"] == b["lcb"]
    assert a["accepted"] == b["accepted"]


def test_different_seeds_do_not_flip_a_clear_verdict():
    rng = np.random.default_rng(5)
    king, challenger = _losses(rng, 2000, 3.0, 3.0 - 0.02)
    seeds = [1, 2, 3, 99, 0xB007]
    assert all(bootstrap_verdict(king, challenger, _req(seed=s))["accepted"] for s in seeds)


def test_tighter_alpha_yields_a_lower_bound():
    """Smaller alpha = more conservative = harder to dethrone the king."""
    rng = np.random.default_rng(6)
    king, challenger = _losses(rng, 1500, 3.0, 3.0 - 0.01)
    strict = bootstrap_verdict(king, challenger, _req(alpha=0.001))
    loose = bootstrap_verdict(king, challenger, _req(alpha=0.05))
    assert strict["lcb"] <= loose["lcb"]


def test_reported_averages_match_inputs():
    rng = np.random.default_rng(7)
    king, challenger = _losses(rng, 300, 3.0, 2.98)
    v = bootstrap_verdict(king, challenger, _req())
    assert v["avg_king_loss"] == pytest.approx(float(np.mean(king)), abs=1e-6)
    assert v["avg_challenger_loss"] == pytest.approx(float(np.mean(challenger)), abs=1e-6)
    assert v["n_sequences"] == 300
