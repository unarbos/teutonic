"""Pinned vLLM 0.33 adapter for Teutonic's MiMo shared experts.

Loaded through vLLM's general-plugin entry point in the dedicated worker environment.
The immutable checkpoint and its config are not modified.
"""


def register():
    from vllm import ModelRegistry
    ModelRegistry.register_model("MiMoV2ForCausalLM",
                                 "teutonic.evaluator.mimo_vllm:TeutonicMiMoForCausalLM")


# Lazy plugin registration must not import the model implementation in processes
# which only enumerate plugins. vLLM resolves this attribute when loading it.
def __getattr__(name):
    if name != "TeutonicMiMoForCausalLM":
        raise AttributeError(name)
    from importlib.metadata import version

    from vllm.model_executor.models import mimo_v2

    if version("vllm") != "0.33.0":
        raise RuntimeError("this adapter is pinned to vLLM 0.33.0")
    original_moe = mimo_v2.MiMoV2MoE

    class SharedExpertMoE(original_moe):
        def __init__(self, vllm_config, prefix="", is_nextn=False):
            super().__init__(vllm_config, prefix=prefix, is_nextn=is_nextn)
            config = vllm_config.model_config.hf_text_config
            count = getattr(config, "n_shared_experts", None) or 0
            width = getattr(config, "shared_expert_inter_dim", None) or config.moe_intermediate_size
            self.shared_experts = (
                mimo_v2.MiMoV2MLP(config.hidden_size, count * width, config.hidden_act,
                                quant_config=vllm_config.quant_config, reduce_results=True,
                                prefix=f"{prefix}.shared_experts") if count else None
            )

        def forward(self, hidden_states):
            routed = super().forward(hidden_states)
            if self.shared_experts is not None:
                routed = routed + self.shared_experts(hidden_states)
            return routed

    mimo_v2.MiMoV2MoE = SharedExpertMoE

    class TeutonicMiMoForCausalLM(mimo_v2.MiMoV2ForCausalLM):
        def load_weights(self, weights):
            shared_seen = set()

            def tracked():
                for key, value in weights:
                    if ".shared_experts." in key:
                        shared_seen.add(key)
                    yield key, value

            result = super().load_weights(tracked())
            expected = set()
            for prefix, module in self.named_modules():
                if isinstance(module, SharedExpertMoE) and module.shared_experts is not None:
                    expected.update(f"{prefix}.shared_experts.{part}.weight"
                                    for part in ("gate_proj", "up_proj", "down_proj"))
            if missing := expected - shared_seen:
                raise RuntimeError(f"missing shared-expert checkpoint tensors: {sorted(missing)}")
            print(f"Teutonic MiMo adapter loaded {len(expected)} shared-expert checkpoint tensors", flush=True)
            return result

    globals()[name] = TeutonicMiMoForCausalLM
    return TeutonicMiMoForCausalLM
