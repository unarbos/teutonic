"""Single source of truth for the active king chain.

Reads `chain.toml` at the repo root and exposes constants used by the evaluator
and architecture checks. To switch architectures, edit `chain.toml` and add
`teutonic/archs/<new>/` when required.

Override knob: `TEUTONIC_CHAIN_OVERRIDE` env var, when set, points at
an alternate TOML (relative to repo root or absolute path). Used by local
testing and archived alternate chain configs so the default `chain.toml`
can stay pointed at the live chain.
"""
from __future__ import annotations

import importlib
import math
import os
import pathlib
import re
import tomllib
from types import MappingProxyType
from types import ModuleType
from typing import Mapping

_REPO_ROOT = pathlib.Path(__file__).resolve().parent
_OVERRIDE = os.environ.get("TEUTONIC_CHAIN_OVERRIDE", "").strip()
if _OVERRIDE:
    _candidate = pathlib.Path(_OVERRIDE)
    _TOML_PATH = _candidate if _candidate.is_absolute() else (_REPO_ROOT / _candidate)
else:
    _TOML_PATH = _REPO_ROOT / "chain.toml"

with open(_TOML_PATH, "rb") as _f:
    _doc = tomllib.load(_f)

CONFIG_PATH: pathlib.Path = _TOML_PATH.resolve()

_chain = _doc.get("chain", {})
_arch = _doc.get("arch", {})
_seed = _doc.get("seed", {})
_evaluation = _doc.get("evaluation", {})

_VALID_SEED_REPO_BACKENDS = {"hf"}

NAME: str = _chain["name"]
SEED_REPO: str = _chain["seed_repo"]
REPO_PATTERN: str = _chain.get("repo_pattern") or rf"^[^/]+/{re.escape(NAME)}-.+$"

ARCH_MODULE: str = _arch.get("module", "")
EXTRA_LOCK_KEYS: tuple[str, ...] = tuple(_arch.get("extra_lock_keys", []))

SEED_TOKENIZER_REPO: str = _seed.get("tokenizer_repo", "")
SEED_DIGEST: str = _seed.get("seed_digest", "")
SEED_REPO_BACKEND: str = (_seed.get("repo_backend") or "hf").strip().lower()
SEED_HOTKEY: str = _seed.get("genesis_hotkey", "").strip()
SEED_INITIAL_WEIGHT_UIDS: tuple[int, ...] = tuple(
    int(value) for value in _seed.get("initial_weight_uids", ())
)
_contract_files = _seed.get("contract_files", {})
if not isinstance(_contract_files, dict):
    raise RuntimeError("chain.toml [seed.contract_files] must be a table")
GENESIS_CONTRACT_FILES: Mapping[str, str] = MappingProxyType(
    {
        str(path): str(digest).strip().lower()
        for path, digest in sorted(_contract_files.items())
    }
)
EVALUATION_DATASET_LABEL: str = str(_evaluation.get("dataset_label") or "").strip()
EVALUATION_N: int = int(_evaluation.get("n") or 0)
try:
    EVALUATION_DELTA_THRESHOLD: float = float(_evaluation.get("delta_threshold"))
except (TypeError, ValueError) as exc:
    raise RuntimeError(
        "chain.toml [evaluation].delta_threshold must be numeric"
    ) from exc
EVALUATION_DATASETS: tuple[dict[str, object], ...] = tuple(
    {
        "name": str(item.get("name") or "").strip(),
        "manifest_url": str(item.get("manifest_url") or "").strip(),
        "proportion": float(item.get("proportion") or 0),
    }
    for item in _evaluation.get("datasets", ())
)
EVALUATION_SHARDS_PER_DATASET: int = int(_evaluation.get("shards_per_dataset") or 0)
CHAIN_GENERATION: str = str(_chain.get("generation") or "").strip() or (
    f"{NAME}-{SEED_DIGEST.replace(':', '-')}"
)
if SEED_REPO_BACKEND not in _VALID_SEED_REPO_BACKENDS:
    raise RuntimeError(
        f"chain.toml [seed].repo_backend must be one of "
        f"{sorted(_VALID_SEED_REPO_BACKENDS)}, got {SEED_REPO_BACKEND!r}"
    )
if not SEED_HOTKEY:
    raise RuntimeError("chain.toml [seed].genesis_hotkey is required")
if (
    len(SEED_INITIAL_WEIGHT_UIDS) != 5
    or len(set(SEED_INITIAL_WEIGHT_UIDS)) != len(SEED_INITIAL_WEIGHT_UIDS)
    or any(uid < 0 for uid in SEED_INITIAL_WEIGHT_UIDS)
):
    raise RuntimeError(
        "chain.toml [seed].initial_weight_uids must contain five distinct non-negative UIDs"
    )
if not GENESIS_CONTRACT_FILES:
    raise RuntimeError("chain.toml [seed.contract_files] requires at least one file")
if any(
    not path
    or pathlib.PurePosixPath(path).is_absolute()
    or ".." in pathlib.PurePosixPath(path).parts
    or str(pathlib.PurePosixPath(path)) != path
    or not re.fullmatch(r"[0-9a-f]{64}", digest)
    for path, digest in GENESIS_CONTRACT_FILES.items()
):
    raise RuntimeError("chain.toml contains an invalid genesis contract file lock")
if not EVALUATION_DATASET_LABEL:
    raise RuntimeError("chain.toml [evaluation].dataset_label is required")
if EVALUATION_N < 1:
    raise RuntimeError("chain.toml [evaluation].n must be positive")
if not math.isfinite(EVALUATION_DELTA_THRESHOLD) or EVALUATION_DELTA_THRESHOLD < 0:
    raise RuntimeError("chain.toml [evaluation].delta_threshold must be finite and non-negative")
if not EVALUATION_DATASETS:
    raise RuntimeError("chain.toml [[evaluation.datasets]] requires at least one dataset")
if EVALUATION_SHARDS_PER_DATASET < 1:
    raise RuntimeError("chain.toml [evaluation].shards_per_dataset must be positive")
if any(
    not item["name"]
    or not str(item["manifest_url"]).startswith("https://")
    or not 0 < float(item["proportion"]) <= 1
    for item in EVALUATION_DATASETS
):
    raise RuntimeError("chain.toml contains an invalid evaluation dataset")
if not math.isclose(
    sum(float(item["proportion"]) for item in EVALUATION_DATASETS),
    1.0,
    rel_tol=0.0,
    abs_tol=1e-9,
):
    raise RuntimeError("chain.toml evaluation dataset proportions must sum to 1")
if not CHAIN_GENERATION or "|" in CHAIN_GENERATION or len(CHAIN_GENERATION) > 128:
    raise RuntimeError("chain.toml produces an invalid chain generation")

# HF namespace inferred from the seed repo. Miners default their challenger
# repo to "<namespace>/<NAME>-<suffix>" though they can override to publish
# under their own account.
SEED_NAMESPACE: str = SEED_REPO.split("/", 1)[0] if "/" in SEED_REPO else ""


def load_arch() -> ModuleType:
    """Import the configured architecture module.

    The arch package's import side effect is to register its config + model
    classes with HuggingFace `AutoConfig` / `AutoModelForCausalLM` so any
    downstream `from_pretrained` resolves the king without trust_remote_code.
    """
    if not ARCH_MODULE:
        raise RuntimeError("chain.toml is missing [arch].module")
    return importlib.import_module(ARCH_MODULE)


__all__ = [
    "CONFIG_PATH",
    "NAME",
    "SEED_REPO",
    "REPO_PATTERN",
    "ARCH_MODULE",
    "EXTRA_LOCK_KEYS",
    "SEED_TOKENIZER_REPO",
    "SEED_DIGEST",
    "EVALUATION_DATASET_LABEL",
    "EVALUATION_N",
    "EVALUATION_DELTA_THRESHOLD",
    "EVALUATION_DATASETS",
    "EVALUATION_SHARDS_PER_DATASET",
    "SEED_REPO_BACKEND",
    "SEED_HOTKEY",
    "SEED_INITIAL_WEIGHT_UIDS",
    "GENESIS_CONTRACT_FILES",
    "CHAIN_GENERATION",
    "SEED_NAMESPACE",
    "load_arch",
]
