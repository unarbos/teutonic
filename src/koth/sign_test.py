"""One-sided binomial sign test with early stopping.

Tests whether Pr(challenger beats king on a random sequence) > 0.5
using a fixed-sample sign test with deterministic early termination.
"""

from __future__ import annotations

from dataclasses import dataclass

from scipy.stats import binom


def compute_threshold(N: int, alpha: float = 0.01) -> int:
    """Smallest K such that Pr(Binomial(N, 0.5) >= K) <= alpha.

    This is the acceptance threshold: challenger must accumulate at least K wins
    out of N evaluated sequences for us to reject the null hypothesis that it is
    no better than the king.
    """
    return int(binom.isf(alpha, N, 0.5))


@dataclass
class SignTestState:
    """Mutable state of a running sign test."""

    N: int
    K: int
    alpha: float
    s: int = 0  # wins so far
    n: int = 0  # non-tie sequences evaluated
    n_ties: int = 0
    finished: bool = False
    verdict: str | None = None  # "challenger" | "king"
    early_stopped: bool = False

    @classmethod
    def create(cls, N: int, alpha: float = 0.01) -> SignTestState:
        K = compute_threshold(N, alpha)
        return cls(N=N, K=K, alpha=alpha)

    def record(self, challenger_loss: float, king_loss: float) -> dict:
        """Record one sequence outcome. Returns the outcome dict."""
        if self.finished:
            raise RuntimeError("Test already finished")

        if challenger_loss == king_loss:
            self.n_ties += 1
            return {"win": None, "s": self.s, "n": self.n}

        self.n += 1
        win = challenger_loss < king_loss
        if win:
            self.s += 1

        self._check_stopping()
        return {"win": 1 if win else 0, "s": self.s, "n": self.n}

    def _check_stopping(self) -> None:
        if self.s >= self.K:
            self.finished = True
            self.verdict = "challenger"
            self.early_stopped = True
        elif self.s + (self.N - self.n) < self.K:
            self.finished = True
            self.verdict = "king"
            self.early_stopped = True
        elif self.n >= self.N:
            self.finished = True
            self.verdict = "challenger" if self.s >= self.K else "king"
            self.early_stopped = False

    @property
    def win_rate(self) -> float:
        return self.s / self.n if self.n > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "S_N": self.s,
            "K": self.K,
            "N": self.N,
            "n_evaluated": self.n,
            "n_ties": self.n_ties,
            "win_rate": round(self.win_rate, 6),
            "alpha": self.alpha,
            "finished": self.finished,
            "verdict": self.verdict,
            "early_stopped": self.early_stopped,
        }
