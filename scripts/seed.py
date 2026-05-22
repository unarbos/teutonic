#!/usr/bin/env python3
"""Seed the genesis king for the active chain.

Thin chain-agnostic driver: imports `<chain.toml [arch].module>.seed` and
delegates to its `main()`. The arch package owns the model-specific build
(architecture, dimensions, tokenizer); chain.toml controls *which* arch and
*what* repo/backend to push to.
"""
import importlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chain_config

if not chain_config.ARCH_MODULE:
    raise SystemExit("chain.toml is missing [arch].module")

seed_mod_name = f"{chain_config.ARCH_MODULE}.seed"
try:
    seed_mod = importlib.import_module(seed_mod_name)
except ModuleNotFoundError as e:
    raise SystemExit(
        f"arch '{chain_config.ARCH_MODULE}' has no seed module ({seed_mod_name}); "
        f"add a seed.py with a main() function to the arch package"
    ) from e

if __name__ == "__main__":
    seed_mod.main()
