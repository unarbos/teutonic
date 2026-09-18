"""Category-stratified shard planning for evaluation sampling.

A dataset manifest is a flat shard list, but the corpora behind these datasets
are internally split into categories (subsets): different source corpora, task
types or synthetic formats. The manifest never labels them -- the category is
only recoverable from each shard's key, which is what the rules in
the bundled ``dataset_categories.json`` rules describe.

Without this, a source's target is filled from whichever shards the shuffle
happened to surface, so a 7-category source can be evaluated on one or two of
its categories. That is a between-run variance source that more shards alone
does not remove: spreading across 4 shards of the same category still measures
one category.

Apportionment is by SHARD COUNT within the live manifest, not by the shares
declared in the rules file. The pipeline packs shards to a fixed token size, so
the two agree closely, but shard count stays correct if that packing changes and
needs nothing from the rules file beyond the regex. The declared shares are
informational -- four of the five are raw-parquet-byte proxies taken before
tokenization.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

FLAT = "*"
DEFAULT_RULES_PATH = Path(__file__).with_name("dataset_categories.json")


def category_seed(base_seed: int, key: str) -> int:
    """Deterministic per-(seed, key) integer seed, matching the source-level one."""
    digest = hashlib.blake2b(f"{base_seed}:{key}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def apportion(total: int, count: int, weights: Sequence[float] | None) -> list[int]:
    """Split `total` into `count` integers by `weights`: floor + largest remainder."""
    if count <= 0:
        return []
    if not weights or len(weights) != count or sum(weights) <= 0:
        base, remainder = divmod(total, count)
        return [base + (1 if index < remainder else 0) for index in range(count)]
    total_weight = sum(weights)
    raw = [total * weight / total_weight for weight in weights]
    targets = [int(value) for value in raw]
    remainder = total - sum(targets)
    order = sorted(range(count), key=lambda index: -(raw[index] - targets[index]))
    for index in order[:remainder]:
        targets[index] += 1
    return targets


def raise_to_min(targets: Sequence[int], min_each: int) -> list[int]:
    """Give every entry at least `min_each`, paying for it out of the largest ones.

    Pure largest-remainder lets a small category round to zero: `codeio` is 0.7%
    of code-reasoning's shards and reaches 0 sequences as soon as the source
    target falls to ~50, silently dropping a whole category from the evaluation.
    The sum is preserved exactly.
    """
    values = list(targets)
    count = len(values)
    total = sum(values)
    if min_each <= 0 or count == 0:
        return values
    if total < count * min_each:
        order = sorted(range(count), key=lambda index: (-values[index], index))
        out = [0] * count
        left = total
        for index in order:
            if left <= 0:
                break
            give = min(min_each, left)
            out[index] = give
            left -= give
        return out
    out = [max(value, min_each) for value in values]
    deficit = sum(out) - total
    while deficit > 0:
        spare = [index for index in range(count) if out[index] > min_each]
        if not spare:
            break
        index = max(spare, key=lambda position: out[position])
        take = min(deficit, out[index] - min_each)
        out[index] -= take
        deficit -= take
    return out


def load_rules(path: str | Path | None) -> dict[str, dict[str, Any]]:
    """{dataset name: rule} from a subsets JSON, or {} when no path is configured.

    Opt-in: with no path every source is treated as one flat category. A path
    that is set but missing is an error -- evaluating on a different mix than the
    operator configured is worse than refusing to build the request.
    """
    if not path or not str(path).strip():
        return {}
    resolved = Path(str(path).strip())
    if not resolved.is_file():
        raise FileNotFoundError(
            f"dataset subsets file {resolved} does not exist; unset it to treat "
            f"every source as flat, or fix the path"
        )
    try:
        raw = json.loads(resolved.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"{resolved} is not readable JSON: {exc}") from exc
    rules: dict[str, dict[str, Any]] = {}
    for name, rule in (raw.get("datasets") or {}).items():
        pattern = str((rule or {}).get("regex") or "").strip()
        if not pattern:
            continue
        try:
            compiled = re.compile(pattern)
            if compiled.groups != 1:
                raise ValueError(f"{resolved}: dataset {name} regex needs exactly one capture group")
            rules[name] = {"re": compiled, "raw": pattern}
        except re.error as exc:
            raise ValueError(
                f"{resolved}: dataset {name} has an invalid regex {pattern!r}: {exc}"
            ) from exc
    return rules


def category_of(rule: Mapping[str, Any] | None, reference: str | None) -> str:
    """Category for one shard, or FLAT when there is no rule or no match.

    An unmatched shard is not an error: it lands in a catch-all category so a
    manifest that grows a new layout still evaluates, just unstratified.
    """
    if not rule or not reference:
        return FLAT
    match = rule["re"].search(str(reference))
    return match.group(1) if match and match.lastindex else FLAT


def plan_source_shards(
    shards: Sequence[Any],
    *,
    target: int,
    shard_budget: int,
    seed: int,
    source_name: str,
    rule: Mapping[str, Any] | None,
    reference_of: Callable[[Any], str],
    capacity_of: Callable[[Any], int],
    min_sequences: int = 1,
) -> list[tuple[Any, int]]:
    """Plan [(shard, sequences to draw)] for one source, stratified by category.

    The source target is split across categories by shard count, every present
    category is guaranteed at least `min_sequences`, and each category's share is
    then spread across several of its own shards so no single shard carries a
    category. `shard_budget` is a floor, not a cap: a source with more categories
    than the budget gets one shard per category, because dropping a category
    biases the mix in a way that extra shards do not.
    """
    if target <= 0 or not shards:
        return []

    groups: dict[str, list[Any]] = {}
    for shard in shards:
        groups.setdefault(category_of(rule, reference_of(shard)), []).append(shard)
    names = sorted(groups)
    if target < len(names) * min_sequences:
        raise ValueError(
            f"dataset {source_name!r} needs at least {len(names) * min_sequences} "
            f"sequences to cover all {len(names)} categories; got {target}"
        )

    weights = [float(len(groups[name])) for name in names]
    sequence_targets = apportion(target, len(names), weights if len(names) > 1 else None)
    if len(names) > 1:
        sequence_targets = raise_to_min(sequence_targets, min_sequences)

    # Only categories that will actually contribute sequences earn a shard, so a
    # target too small to cover every category does not download unused shards.
    active = [index for index, count in enumerate(sequence_targets) if count > 0]
    shard_total = max(int(shard_budget), len(active))
    shard_counts = [0] * len(names)
    if active:
        allocated = apportion(
            shard_total, len(active), [float(sequence_targets[i]) for i in active]
        )
        allocated = raise_to_min(allocated, 1)
        for position, index in enumerate(active):
            shard_counts[index] = allocated[position]

    planned: list[tuple[Any, int]] = []
    for index, name in enumerate(names):
        remaining = sequence_targets[index]
        if remaining <= 0:
            continue
        ordered = list(groups[name])
        random.Random(
            category_seed(seed, f"{source_name}|{name}")
        ).shuffle(ordered)
        quotas = apportion(remaining, max(1, min(shard_counts[index], len(ordered))), None)
        assigned: list[tuple[Any, int]] = []
        overflow = 0
        for position, shard in enumerate(ordered):
            if position >= len(quotas) and overflow <= 0:
                break
            want = (quotas[position] if position < len(quotas) else 0) + overflow
            if want <= 0:
                continue
            room = max(0, capacity_of(shard))
            take = min(want, room)
            overflow = want - take
            if take > 0:
                assigned.append((shard, take))
        planned.extend(assigned)
    return planned
