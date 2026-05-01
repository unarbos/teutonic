"""Vendored Quasar architecture for Teutonic-IX.

Importing this package registers QuasarConfig and QuasarForCausalLM with the
HuggingFace Auto* APIs. Once imported, AutoModelForCausalLM.from_pretrained
loads any Teutonic-IX checkpoint without trust_remote_code.

The plan is enforced upstream (validator rejects challenger uploads that ship
*.py or set auto_map). The seed checkpoint also strips auto_map before push so
no client of this package is ever asked to execute remote code.
"""

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_quasar import QuasarConfig
from .modeling_quasar import (
    QuasarCausalLMOutputWithPast,
    QuasarForCausalLM,
    QuasarModel,
    QuasarModelOutputWithPast,
    QuasarPreTrainedModel,
)


def _register():
    # AutoConfig.register raises ValueError if already registered; same for the
    # AutoModel families. Make registration idempotent so re-importing this
    # module (in tests, eval workers, training scripts) does not crash.
    try:
        AutoConfig.register("quasar", QuasarConfig)
    except ValueError:
        pass
    try:
        AutoModel.register(QuasarConfig, QuasarModel)
    except ValueError:
        pass
    try:
        AutoModelForCausalLM.register(QuasarConfig, QuasarForCausalLM)
    except ValueError:
        pass


_register()


__all__ = [
    "QuasarConfig",
    "QuasarPreTrainedModel",
    "QuasarModel",
    "QuasarForCausalLM",
    "QuasarModelOutputWithPast",
    "QuasarCausalLMOutputWithPast",
]
