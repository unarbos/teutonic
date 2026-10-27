"""Category diagnostics from saved paired samples; never used for verdicts."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from functools import lru_cache
from typing import Any
from urllib.parse import unquote, urlsplit

from .categories import DEFAULT_RULES_PATH, FLAT, category_of, load_rules

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


@lru_cache(maxsize=1)
def _rules():
    return load_rules(DEFAULT_RULES_PATH)


def _shard_category(source: str, reference: Any) -> str | None:
    if not isinstance(reference, str):
        return None
    try:
        path = unquote(urlsplit(reference).path)
    except ValueError:
        return None
    for dataset, rule in _rules().items():
        category = category_of(rule, path)
        if category != FLAT and _NAME.fullmatch(category):
            return f"{dataset}/{category}"
    if source == "dclm-baseline-1.0" or "dclm-baseline-1.0__" in path:
        return "dclm-baseline-1.0"
    return None


def category_scores_from_samples(
    samples: Any, dataset_identity: Any, shards_used: Any = None
) -> dict[str, dict[str, int | float]]:
    """Aggregate actual scored tokens, including early-stopped prefixes.

    Unknown categories or incomplete audit arrays return no breakdown rather
    than guessing from mix weights or silently omitting evaluated samples.
    """
    if not isinstance(samples, Mapping):
        return {}
    columns = [
        samples.get(key)
        for key in (
            "king_loss",
            "challenger_loss",
            "scored_tokens",
            "shard_group_index",
            "shard_index",
        )
    ]
    if any(not isinstance(column, list) for column in columns):
        return {}
    size = len(columns[0])
    if not size or any(len(column) != size for column in columns):
        return {}
    if samples.get("n_samples", size) != size:
        return {}
    kings, challengers, tokens, group_indices, shard_indices = columns
    sources = dataset_identity.get("sources") if isinstance(dataset_identity, Mapping) else None
    if not isinstance(sources, list):
        sources = shards_used if isinstance(shards_used, list) else []
    categories = []
    for source in sources:
        if not isinstance(source, Mapping):
            return {}
        name = source.get("name", source.get("source", ""))
        shards = source.get("shards", source.get("refs", source.get("names", [])))
        if not isinstance(shards, list):
            return {}
        categories.append(
            [
                _shard_category(name, shard.get("url") if isinstance(shard, Mapping) else shard)
                for shard in shards
            ]
        )

    components = samples.get("component", ["windows"] * size)
    datasets = samples.get("document_dataset", [None] * size)
    document_categories = samples.get("document_category", [None] * size)
    if any(
        not isinstance(column, list) or len(column) != size
        for column in (components, datasets, document_categories)
    ):
        return {}
    totals: dict[str, list[float]] = {}
    for index in range(size):
        king, challenger, weight = kings[index], challengers[index], tokens[index]
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in (king, challenger, weight)
        ):
            return {}
        if king < 0 or challenger < 0 or weight <= 0 or weight != int(weight):
            return {}
        if components[index] == "long_documents":
            dataset, category = datasets[index], document_categories[index]
            if not all(
                isinstance(value, str) and _NAME.fullmatch(value) for value in (dataset, category)
            ):
                return {}
            label = dataset if dataset == category else f"{dataset}/{category}"
        elif components[index] == "windows":
            group, shard = group_indices[index], shard_indices[index]
            if any(
                isinstance(value, bool) or not isinstance(value, int) for value in (group, shard)
            ):
                return {}
            if not 0 <= group < len(categories) or not 0 <= shard < len(categories[group]):
                return {}
            label = categories[group][shard]
            if label is None:
                return {}
        else:
            return {}
        total = totals.setdefault(label, [0, 0, 0, 0, 0])
        total[0] += 1
        total[1] += weight
        total[2] += king * weight
        total[3] += challenger * weight
        total[4] += (king - challenger) * weight
    return {
        label: {
            "n_sequences": int(count),
            "n_scored_tokens": int(weight),
            "avg_king_loss": round(king / weight, 6),
            "avg_challenger_loss": round(challenger / weight, 6),
            "mu_hat": round(delta / weight, 6),
        }
        for label, (count, weight, king, challenger, delta) in sorted(totals.items())
    }
