"""Qwen3 dense arch shim.

Vanilla `Qwen3ForCausalLM` ships in `transformers >= 4.51` and is already
self-registered with `AutoConfig` / `AutoModelForCausalLM`, so importing this
package is enough to make `chain_config.load_arch()` resolve `model_type=qwen3`
without `trust_remote_code`. We deliberately do not vendor modeling code —
upstream transformers is the source of truth.

`seed.py` downloads the chosen Qwen3 checkpoint from HF and pushes it to
Hippius as the genesis king for this chain.
"""
from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: F401

__all__ = ["Qwen3Config", "Qwen3ForCausalLM"]
