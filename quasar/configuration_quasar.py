"""Quasar model configuration — HuggingFace compatible.

"""

from transformers.configuration_utils import PreTrainedConfig


class QuasarConfig(PreTrainedConfig):
    model_type = "quasar"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        # Core dimensions
        vocab_size: int = 248320,
        d_model: int = 1536,
        n_layers: int = 24,
        n_heads: int = 12,
        d_ff: int = 4096,
        head_dim: int = 128,
        max_seq_len: int = 16384,
        dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        # HF aliases (set automatically)
        # hidden_size = d_model, num_hidden_layers = n_layers, etc.
        # Hybrid layer config
        quasar_layers: int = 4,
        gated_layers: int = 2,
        use_gla_first: bool = False,
        # QuasarAttention params
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        allow_neg_eigval: bool = False,
        attn_mode: str = "chunk",
        # GLA params
        expand_k: float = 0.5,
        expand_v: float = 1.0,
        gla_mode: str = "chunk",
        # Latent Memory params
        memory_slots: int = 128,
        memory_dim: int = 128,
        # MoE params
        moe_type: str = "bigmac",
        num_shared_experts: int = 1,
        num_routed_experts: int = 64,
        top_k: int = 4,
        shared_expert_size: int = 3072,
        routed_expert_size: int = 256,
        dense_input_layers: int = 4,
        bigmac_r: float = 0.25,
        # MoE stability (SMEBU)
        moe_z_loss_coeff: float = 1e-4,
        moe_aux_loss_coeff: float = 1e-4,
        smebu_kappa: float = 2.0,
        smebu_lambda: float = 2e-3,
        smebu_beta: float = 0.5,
        # Looped transformer
        num_loops: int = 1,
        use_looped_injection: bool = False,
        looped_injection_init: float = 0.1,
        # RoPE
        rope_theta: float = 1000000.0,
        # Training
        gradient_checkpointing: bool = False,
        residual_scale: float = 0.1,
        # FLA compatibility
        fuse_norm: bool = True,
        fuse_swiglu: bool = True,
        fuse_cross_entropy: bool = True,
        use_l2warp: bool = False,
        hidden_act: str = "silu",
        hidden_ratio: int | None = 4,
        # Token ids
        pad_token_id: int | None = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.hidden_size = d_model
        self.n_layers = n_layers
        self.num_hidden_layers = n_layers
        self.n_heads = n_heads
        self.num_attention_heads = n_heads
        self.num_heads = n_heads  # FLA alias
        self.d_ff = d_ff
        self.intermediate_size = d_ff
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.max_position_embeddings = max_seq_len
        self.dropout = dropout
        self.rms_norm_eps = rms_norm_eps
        self.norm_eps = rms_norm_eps  # FLA alias
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings

        # Hybrid layer config
        self.quasar_layers = quasar_layers
        self.gated_layers = gated_layers
        self.use_gla_first = use_gla_first

        # layer_types uses HF-allowed values only (for validation)
        # hybrid_layer_types stores the actual quasar/gla distinction
        # Always force layer_types to HF-safe values, even if quasar/gla passed in
        self.hybrid_layer_types = self._build_hybrid_layer_types()
        self.layer_types = ["linear_attention"] * self.n_layers

        # QuasarAttention params
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.allow_neg_eigval = allow_neg_eigval
        self.attn_mode = attn_mode

        # GLA params
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.gla_mode = gla_mode

        # Latent Memory
        self.memory_slots = memory_slots
        self.memory_dim = memory_dim

        # MoE
        self.moe_type = moe_type
        self.num_shared_experts = num_shared_experts
        self.num_routed_experts = num_routed_experts
        self.top_k = top_k
        self.shared_expert_size = shared_expert_size
        self.routed_expert_size = routed_expert_size
        self.dense_input_layers = dense_input_layers
        self.bigmac_r = bigmac_r

        # SMEBU
        self.moe_z_loss_coeff = moe_z_loss_coeff
        self.moe_aux_loss_coeff = moe_aux_loss_coeff
        self.smebu_kappa = smebu_kappa
        self.smebu_lambda = smebu_lambda
        self.smebu_beta = smebu_beta

        # Looped transformer
        self.num_loops = num_loops
        self.use_looped_injection = use_looped_injection
        self.looped_injection_init = looped_injection_init

        # RoPE
        self.rope_theta = rope_theta

        # Training
        self.gradient_checkpointing = gradient_checkpointing
        self.residual_scale = residual_scale

        # FLA compatibility
        self.fuse_norm = fuse_norm
        self.fuse_swiglu = fuse_swiglu
        self.fuse_cross_entropy = fuse_cross_entropy
        self.use_l2warp = use_l2warp
        self.hidden_act = hidden_act
        self.hidden_ratio = hidden_ratio

        # KV heads (for HF compatibility)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", n_heads)
        self.num_v_heads = kwargs.get("num_v_heads", None)

        # Pop layer_types from kwargs to prevent PreTrainedConfig from overriding
        # our HF-safe value with quasar/gla from config.json
        kwargs.pop("layer_types", None)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _build_hybrid_layer_types(self) -> list[str]:
        """Internal quasar/gla distinction — not validated by HF."""
        cycle_len = self.quasar_layers + self.gated_layers
        types = []
        for i in range(self.n_layers):
            pos_in_cycle = i % cycle_len
            if self.use_gla_first:
                is_quasar = pos_in_cycle >= self.gated_layers
            else:
                is_quasar = pos_in_cycle < self.quasar_layers
            types.append("quasar" if is_quasar else "gla")
        return types


__all__ = ["QuasarConfig"]
