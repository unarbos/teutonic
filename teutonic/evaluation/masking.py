"""Identity and target accounting for isolated fragments of sampled windows."""

from collections.abc import Sequence

import numpy as np

MASKED_POLICY_VERSION = "document-masked-token-bootstrap-v1"
DOCUMENT_EOS_TOKEN_ID = 151645


def scored_token_counts(sequences: Sequence[Sequence[int]]) -> list[int]:
    """Keep EOS targets, exclude each window/fragment's first token.

    EOS belongs to the fragment it terminates. No context is restored from
    outside a sampled window, and no synthetic BOS/EOS is inserted.
    """
    counts = [sum(token != DOCUMENT_EOS_TOKEN_ID for token in row[:-1]) for row in sequences]
    if not counts or any(count == 0 for count in counts):
        raise ValueError("masked evaluation requires at least one scored target in every window")
    return counts


def token_weights(counts: Sequence[int], size: int) -> np.ndarray:
    weights = np.asarray(counts, dtype=np.float64)
    if (
        weights.shape != (size,)
        or size == 0
        or not np.isfinite(weights).all()
        or (weights <= 0).any()
        or (weights != np.floor(weights)).any()
    ):
        raise ValueError("scored token counts must be positive integers aligned with windows")
    return weights
