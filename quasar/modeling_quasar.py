"""Quasar hybrid transformer — HuggingFace compatible.

"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers import GenerationMixin
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .configuration_quasar import QuasarConfig

logger = logging.get_logger(__name__)

# FLA layer imports — required
from fla.layers.quasar import QuasarAttention
from fla.layers.gla import GatedLinearAttention
from fla.models.utils import Cache as FlaCache, FLAGenerationMixin


# ===================================================================
# RMSNorm (standalone — weight name: .weight, no bias)
# ===================================================================
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# ===================================================================
# Rotary Embedding (persistent inv_freq to match checkpoint)
# ===================================================================
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=4096, base=100000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # Pre-compute cos/sin cache
        t = torch.arange(max_position_embeddings + 1, device=device, dtype=inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("_cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("_sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len is not None and seq_len > self._cos_cached.shape[2]:
            t = torch.arange(seq_len + 1024, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.register_buffer("_cos_cached", emb.cos()[None, None, :, :].to(self._cos_cached.dtype), persistent=False)
            self.register_buffer("_sin_cached", emb.sin()[None, None, :, :].to(self._sin_cached.dtype), persistent=False)
        return (
            self._cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self._sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


# ===================================================================
# Latent Memory Module (use_triton=False — PyTorch bmm is faster)
# ===================================================================
class LatentMemoryModule(nn.Module):
    """Persistent Latent Parameter Memory — weight names match checkpoint."""

    def __init__(self, hidden_size, memory_slots=128, memory_dim=128, use_triton=False):
        super().__init__()
        self.K = memory_slots
        self.D = memory_dim

        self.W_eta = nn.Linear(hidden_size, 1, bias=True)
        nn.init.zeros_(self.W_eta.weight)
        nn.init.constant_(self.W_eta.bias, -5.0)

        self.segment_len = 64
        self.summary_query = nn.Parameter(torch.randn(1, 1, memory_dim))
        self.summary_proj = nn.Linear(hidden_size, memory_dim, bias=True)
        self.eta_channels = nn.Parameter(torch.ones(1, 1, memory_dim))
        self.temperature = nn.Parameter(torch.ones(1))
        self.hidden_size = hidden_size
        self.use_triton = False
        self.input_norm = nn.LayerNorm(hidden_size)
        self.compress_z = nn.Sequential(
            nn.Linear(hidden_size, memory_dim * 2, bias=False),
            nn.SiLU(),
            nn.Linear(memory_dim * 2, memory_dim, bias=False),
        )
        self.W_qkv_mem = nn.Linear(hidden_size, memory_dim * 3, bias=False)
        self.scale = 1.0 / math.sqrt(memory_dim)

    def get_diversity_loss(self, M):
        B, K, D = M.shape
        M_norm = F.normalize(M, p=2, dim=-1)
        sim = torch.bmm(M_norm, M_norm.transpose(1, 2))
        mask = torch.eye(K, device=M.device).unsqueeze(0)
        sim = sim * (1 - mask)
        return sim.pow(2).mean()

    def write_memory(self, H, M, chunk_idx=0):
        H = self.input_norm(H)
        B, T, _ = H.shape
        H_mem = self.summary_proj(H)
        eta_tokens = self.W_eta(H).squeeze(-1)

        L = self.segment_len
        if T % L != 0:
            pad_len = L - (T % L)
            H_padded = F.pad(H_mem, (0, 0, 0, pad_len))
            eta_padded = F.pad(eta_tokens, (0, pad_len), value=-10.0)
        else:
            H_padded = H_mem
            eta_padded = eta_tokens

        T_pad = H_padded.shape[1]
        num_segments = T_pad // L
        H_segs = H_padded.view(B * num_segments, L, self.D)

        summary_scores = torch.bmm(
            self.summary_query.expand(B * num_segments, -1, -1),
            H_segs.transpose(1, 2),
        )
        summary_weights = F.softmax(summary_scores * self.scale, dim=-1)
        Z_seg = torch.bmm(summary_weights, H_segs).view(B, num_segments, self.D)

        eta_raw_sig = torch.sigmoid(eta_tokens)
        eta_seg_sig = torch.max(
            torch.sigmoid(eta_padded.view(B, num_segments, L)), dim=-1, keepdim=True
        )[0]

        scores = torch.bmm(Z_seg, M.transpose(-1, -2)) * self.scale * torch.exp(self.temperature)
        A = F.softmax(scores, dim=-1)
        DeltaM_seg = torch.bmm(A.transpose(1, 2), Z_seg * eta_seg_sig)
        eta_avg = eta_seg_sig.mean(dim=1, keepdim=True)
        gate = eta_avg * torch.sigmoid(self.eta_channels)
        M_new = (1.0 - gate) * M + DeltaM_seg / num_segments
        norm_sq = torch.sum(DeltaM_seg ** 2) / num_segments
        div_loss = self.get_diversity_loss(M_new)
        return M_new, norm_sq * 0.01 + div_loss * 0.1, eta_raw_sig

    def read_memory(self, H, M, memory_scale=1.0):
        H = self.input_norm(H)
        qkv_mem = self.W_qkv_mem(H)
        _, _, Q_r = torch.split(qkv_mem, [self.D, self.D, self.D], dim=-1)
        scores = torch.bmm(Q_r, M.transpose(-1, -2))
        if M.shape[1] > 1024:
            top_k = 64
            top_vals, top_idx = torch.topk(scores, top_k, dim=-1)
            mask = torch.full_like(scores, float('-inf'))
            mask.scatter_(-1, top_idx, top_vals)
            scores = mask
        A = F.softmax(scores * 2.0, dim=-1)
        C = torch.bmm(A, M)
        return C * memory_scale


# ===================================================================
# FFN Components
# ===================================================================
class SwiGLUBlock(nn.Module):
    """Dense FFN — weight names: gate.weight, up.weight, down.weight"""

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class SigmoidRouter(nn.Module):
    """Router with router_weights Parameter — weight name: router.router_weights"""

    def __init__(self, d_model, num_experts):
        super().__init__()
        self.router_weights = nn.Parameter(torch.zeros(num_experts, d_model))
        nn.init.kaiming_uniform_(self.router_weights, a=math.sqrt(5))

    def forward(self, x):
        logits = F.linear(x, self.router_weights)
        scores = torch.sigmoid(logits)
        return scores, logits


class BigMacMoE(nn.Module):
    """BigMac MoE with DCCA bottleneck — matches checkpoint weight names exactly.

    Weights: w_down_proj, w_up_proj, experts_w12, experts_w3,
             router.router_weights, shared_experts.{i}.{gate,up,down}.weight,
             max_vio
    """

    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.d_model = config.d_model
        self.bigmac_r = getattr(config, 'bigmac_r', 0.25)
        self.bottle_dim = int(self.d_model * self.bigmac_r)

        self.num_shared_experts = getattr(config, 'num_shared_experts', 1)
        self.num_routed_experts = getattr(config, 'num_routed_experts', 64)
        self.top_k = getattr(config, 'top_k', 4)

        default_routed_size = int(getattr(config, 'routed_expert_size', 768) / self.bigmac_r)
        self.routed_expert_size = getattr(config, 'bigmac_expert_size', default_routed_size)
        self.shared_expert_size = getattr(config, 'shared_expert_size', config.d_ff)
        self.layer_idx = layer_idx

        self.shared_experts = nn.ModuleList([
            SwiGLUBlock(self.d_model, self.shared_expert_size)
            for _ in range(self.num_shared_experts)
        ])

        # BigMac DCCA Projections
        self.w_down_proj = nn.Linear(self.d_model, self.bottle_dim, bias=False)
        self.w_up_proj = nn.Linear(self.bottle_dim, self.d_model, bias=False)

        # BigMac Experts (fused gate+up W12, down W3)
        self.experts_w12 = nn.Parameter(torch.zeros(self.num_routed_experts, self.bottle_dim, 2 * self.routed_expert_size))
        self.experts_w3 = nn.Parameter(torch.zeros(self.num_routed_experts, self.routed_expert_size, self.bottle_dim))

        self.router = SigmoidRouter(self.d_model, self.num_routed_experts)

        self.expert_bias = None
        self.expert_momentum = None
        self.smebu_kappa = getattr(config, 'smebu_kappa', 2.0)
        self.smebu_lambda = getattr(config, 'smebu_lambda', 2e-3)
        self.smebu_beta = getattr(config, 'smebu_beta', 0.5)

        self.z_loss_weight = getattr(config, 'moe_z_loss_coeff', 1e-4)
        self.aux_loss_weight = getattr(config, 'moe_aux_loss_coeff', 1e-4)
        self.register_buffer("max_vio", torch.tensor(0.0))
        self.route_scale = math.sqrt(self.top_k)
        self.moe_scale = 1.0 / (1.0 + float(self.num_shared_experts > 0))

        # Buffers for padded BMM dispatch
        self.register_buffer("_dummy_token", torch.zeros(1, self.bottle_dim, dtype=torch.bfloat16), persistent=False)
        self.register_buffer("_dummy_out", torch.zeros(1, self.bottle_dim, dtype=torch.bfloat16), persistent=False)
        self._cached_N = -1
        self._cached_K = -1
        self._cached_indices = None

    def _init_weights(self, std=0.011):
        nn.init.normal_(self.w_down_proj.weight, std=std)
        nn.init.normal_(self.w_up_proj.weight, std=std)
        nn.init.normal_(self.experts_w12, std=std)
        nn.init.normal_(self.experts_w3, std=std)
        for expert in self.shared_experts:
            nn.init.normal_(expert.gate.weight, std=std)
            nn.init.normal_(expert.up.weight, std=std)
            nn.init.normal_(expert.down.weight, std=std)

    def forward(self, x, expert_bias=None):
        batch_size, seq_len, d_model = x.shape
        hidden_states = x.view(-1, d_model)
        N, D = hidden_states.shape
        K = self.top_k
        num_tokens_total = N * K

        # 1. Routing & Gating
        with torch.autocast(device_type=x.device.type, dtype=torch.float32):
            scores, logits = self.router(hidden_states)
            z_loss = torch.mean(logits.nan_to_num() ** 2) * self.z_loss_weight

            bias = expert_bias if expert_bias is not None else torch.zeros(self.num_routed_experts, device=x.device)
            selection_scores = scores + bias
            _, topk_indices = torch.topk(selection_scores, K, dim=-1)
            topk_indices = topk_indices.clamp(0, logits.shape[1] - 1)

            topk_logits = torch.gather(logits, 1, topk_indices)
            gating_scores = F.softmax(topk_logits, dim=-1).to(torch.bfloat16)

        # 2. Aux loss
        if self.training:
            flat_topk_idx = topk_indices.view(-1)
            expert_counts = torch.bincount(flat_topk_idx, minlength=self.num_routed_experts)
            fi = expert_counts.float() / num_tokens_total
            Pi = scores.nan_to_num().mean(dim=0)
            aux_loss = torch.sum(fi * Pi) * self.aux_loss_weight
        else:
            aux_loss = torch.tensor(0.0, device=x.device)
            expert_counts = None

        # 3. Shared experts
        shared_out = 0
        if self.num_shared_experts > 0:
            for expert in self.shared_experts:
                shared_out = shared_out + expert(hidden_states)

        # 4. Bottleneck projection
        down_proj_hidden = self.w_down_proj(hidden_states)

        # 5. Routed experts (padded BMM dispatch)
        flat_topk_idx = topk_indices.view(-1).clamp(0, self.num_routed_experts - 1)
        sorted_experts, permutation = torch.sort(flat_topk_idx)

        if self._cached_N == N and self._cached_K == K:
            token_indices, global_rel_idx = self._cached_indices
        else:
            token_indices = torch.arange(N, device=x.device).repeat_interleave(K)
            global_rel_idx = torch.arange(num_tokens_total, device=x.device)
            self._cached_N, self._cached_K = N, K
            self._cached_indices = (token_indices, global_rel_idx)

        max_load = ((num_tokens_total // self.num_routed_experts) // 8 + 6) * 8
        used_counts = expert_counts if expert_counts is not None else torch.bincount(flat_topk_idx, minlength=self.num_routed_experts)
        expert_ptr = torch.cumsum(used_counts, dim=0) - used_counts

        local_idx = global_rel_idx - expert_ptr.index_select(0, sorted_experts)
        capacity_mask = local_idx < max_load
        valid_slots = sorted_experts[capacity_mask] * max_load + local_idx[capacity_mask]
        num_slots = self.num_routed_experts * max_load

        hidden_with_dummy = torch.cat([down_proj_hidden, self._dummy_token], dim=0)
        reverse_map = torch.full((num_slots,), N, device=x.device, dtype=torch.long)
        reverse_map.scatter_(0, valid_slots.long(), token_indices[permutation][capacity_mask])

        padding = hidden_with_dummy.index_select(0, reverse_map).view(self.num_routed_experts, max_load, self.bottle_dim)

        h12 = torch.bmm(padding, self.experts_w12)
        h1, h2 = h12.chunk(2, dim=-1)
        padded_out = torch.bmm(F.silu(h1) * h2, self.experts_w3)

        padded_out_flat = padded_out.view(-1, self.bottle_dim)
        padded_out_with_dummy = torch.cat([padded_out_flat, self._dummy_out], dim=0)

        gather_map = torch.full((num_tokens_total,), num_slots, device=x.device, dtype=torch.long)
        gather_map.scatter_(0, permutation[capacity_mask], valid_slots)

        gathered_out = padded_out_with_dummy.index_select(0, gather_map).view(N, K, self.bottle_dim)

        routed_out_bottle = torch.bmm(gating_scores.to(gathered_out.dtype).unsqueeze(1), gathered_out).squeeze(1)
        routed_out = self.w_up_proj(routed_out_bottle)

        if self.training:
            mean_load = num_tokens_total / self.num_routed_experts
            self._pending_violation = (mean_load - used_counts.float()) / (mean_load + 1e-6)

        route_scale = math.sqrt(self.top_k) if self.training else 1.0
        out = (shared_out + routed_out * route_scale) * self.moe_scale
        out = out.view(batch_size, seq_len, d_model).to(x.dtype)

        return out, z_loss + aux_loss

    def update_bias(self, counts, num_tokens):
        expert_counts = counts.float()
        mean_load = num_tokens * self.top_k / self.num_routed_experts
        violation = (mean_load - expert_counts) / (mean_load + 1e-6)
        clamped_update = torch.tanh(self.smebu_kappa * violation)
        delta_bi = self.smebu_lambda * clamped_update
        delta_bi = delta_bi - delta_bi.mean()
        self.expert_momentum.data = self.smebu_beta * self.expert_momentum.data + (1 - self.smebu_beta) * delta_bi
        self.expert_bias.data = (self.expert_bias.data + self.expert_momentum.data).nan_to_num_().clamp(-10.0, 10.0)
        self.expert_bias.data -= self.expert_bias.data.mean()
        current_max_vio = -violation.min()
        self.max_vio.copy_(0.99 * self.max_vio + 0.01 * current_max_vio)


class GroupedMoE(nn.Module):
    """Grouped MoE fallback — for non-BigMac configs."""

    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.d_model = config.d_model
        self.num_shared_experts = getattr(config, 'num_shared_experts', 1)
        self.num_routed_experts = getattr(config, 'num_routed_experts', 64)
        self.top_k = getattr(config, 'top_k', 6)
        self.shared_expert_size = getattr(config, 'shared_expert_size', config.d_ff)
        self.routed_expert_size = getattr(config, 'routed_expert_size', 1408)
        self.layer_idx = layer_idx

        self.shared_experts = nn.ModuleList([
            SwiGLUBlock(self.d_model, self.shared_expert_size)
            for _ in range(self.num_shared_experts)
        ])
        self.experts_w12 = nn.Parameter(torch.zeros(self.num_routed_experts, self.d_model, 2 * self.routed_expert_size))
        self.experts_w3 = nn.Parameter(torch.zeros(self.num_routed_experts, self.routed_expert_size, self.d_model))
        self.router = nn.Linear(config.d_model, config.num_routed_experts, bias=False)
        with torch.no_grad():
            nn.init.normal_(self.router.weight, std=0.01)
        self.z_loss_weight = getattr(config, 'moe_z_loss_coeff', 1e-6)
        self.aux_loss_weight = getattr(config, 'moe_aux_loss_coeff', 1e-4)
        self.smebu_kappa = getattr(config, 'smebu_kappa', 2.0)
        self.smebu_lambda = getattr(config, 'smebu_lambda', 5e-4)
        self.smebu_beta = getattr(config, 'smebu_beta', 0.5)
        self.register_buffer("max_vio", torch.tensor(0.0))
        self.moe_scale = 1.0 / (1.0 + float(self.num_shared_experts > 0))

    def _init_weights(self, std=0.011):
        nn.init.normal_(self.experts_w12, std=std)
        nn.init.normal_(self.experts_w3, std=std)
        for expert in self.shared_experts:
            nn.init.normal_(expert.gate.weight, std=std)
            nn.init.normal_(expert.up.weight, std=std)
            nn.init.normal_(expert.down.weight, std=std)

    def forward(self, x, expert_bias=None):
        batch_size, seq_len, d_model = x.shape
        hidden_states = x.view(-1, d_model)
        N, D = hidden_states.shape
        K = self.top_k

        with torch.autocast(device_type=x.device.type, dtype=torch.float32):
            logits = self.router(hidden_states)
            scores = torch.sigmoid(logits)
            z_loss = torch.mean(logits.nan_to_num() ** 2) * self.z_loss_weight
            bias = expert_bias if expert_bias is not None else torch.zeros(self.num_routed_experts, device=x.device)
            selection_scores = scores + bias
            _, topk_indices = torch.topk(selection_scores, K, dim=-1)
            topk_indices = topk_indices.clamp(0, logits.shape[1] - 1)
            topk_logits = torch.gather(logits, 1, topk_indices)
            gating_scores = F.softmax(topk_logits, dim=-1).to(torch.bfloat16)

        if self.training:
            flat_topk_idx = topk_indices.view(-1)
            expert_counts = torch.bincount(flat_topk_idx, minlength=self.num_routed_experts)
            fi = expert_counts.float() / (N * K)
            Pi = scores.nan_to_num().mean(dim=0)
            aux_loss = torch.sum(fi * Pi) * self.aux_loss_weight
            self._pending_violation = fi.detach() - (1.0 / self.num_routed_experts)
        else:
            aux_loss = torch.tensor(0.0, device=x.device)
            expert_counts = None
            self._pending_violation = torch.zeros(self.num_routed_experts, device=x.device)

        shared_out = 0
        if self.num_shared_experts > 0:
            for expert in self.shared_experts:
                shared_out = shared_out + expert(hidden_states)

        # Padded BMM dispatch
        num_experts = self.num_routed_experts
        flat_topk_idx = topk_indices.view(-1)
        tokens_per_expert = torch.bincount(flat_topk_idx, minlength=num_experts)
        max_tokens = tokens_per_expert.max().item()

        if max_tokens == 0:
            out = shared_out * self.moe_scale
            return out.view(batch_size, seq_len, d_model).to(x.dtype), aux_loss

        sorted_indices = torch.argsort(flat_topk_idx)
        token_indices = torch.arange(N, device=x.device).repeat_interleave(K)[sorted_indices]
        grouped_x = hidden_states[token_indices]
        padded_x = torch.zeros(num_experts, max_tokens, D, device=x.device, dtype=x.dtype)
        expert_starts = torch.cat([torch.tensor([0], device=x.device), tokens_per_expert[:-1].cumsum(0)])
        intra_offsets = torch.arange(N * K, device=x.device) - expert_starts.repeat_interleave(tokens_per_expert)
        expert_idx = flat_topk_idx[sorted_indices]
        padded_x_flat = padded_x.view(-1, D)
        flat_dest_indices = expert_idx * max_tokens + intra_offsets
        padded_x_flat.index_put_((flat_dest_indices,), grouped_x)
        h12 = torch.bmm(padded_x, self.experts_w12)
        h1, h2 = h12.chunk(2, dim=-1)
        h = F.silu(h1) * h2
        expert_out_padded = torch.bmm(h, self.experts_w3)
        full_expert_out = expert_out_padded.view(-1, D)[flat_dest_indices]
        gating_flat = gating_scores.view(-1)
        sorted_gating = gating_flat[sorted_indices].unsqueeze(1)
        weighted_out = full_expert_out * sorted_gating
        routed_out = torch.zeros_like(hidden_states)
        routed_out.index_add_(0, token_indices, weighted_out)

        route_scale = math.sqrt(self.top_k) if self.training else 1.0
        out = (shared_out + routed_out * route_scale) * self.moe_scale
        out = out.view(batch_size, seq_len, d_model).to(x.dtype)
        return out, z_loss + aux_loss


# ===================================================================
# HybridBlock — one transformer layer
# Weight names: ln1.weight, ln1_out.weight, ln2.weight, ln2_out.weight,
#              attn.*, memory.*, W_alpha.*, C_to_hidden.*,
#              ffn.*, injection_gate
# ===================================================================
class HybridBlock(nn.Module):
    def __init__(self, config: QuasarConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.d_model
        self.layer_idx = layer_idx
        self.n_layers = config.n_layers
        self.config = config
        self.gradient_checkpointing = False

        # Looped Transformer injection gate (checkpoint always has it)
        self.use_looped_injection = config.use_looped_injection
        self.injection_gate = nn.Parameter(torch.tensor([-2.197]))

        # Determine layer type (use hybrid_layer_types for quasar/gla distinction)
        self.layer_type = config.hybrid_layer_types[layer_idx]

        # Attention layer
        if self.layer_type == "quasar":
            self.attn = QuasarAttention(
                mode=config.attn_mode,
                hidden_size=config.d_model,
                expand_v=config.expand_v,
                head_dim=config.head_dim,
                num_heads=config.n_heads,
                num_v_heads=config.num_v_heads,
                use_short_conv=config.use_short_conv,
                allow_neg_eigval=config.allow_neg_eigval,
                conv_size=config.conv_size,
                norm_eps=config.rms_norm_eps,
                layer_idx=layer_idx,
            )
        elif self.layer_type == "gla":
            self.attn = GatedLinearAttention(
                mode=config.gla_mode,
                hidden_size=config.d_model,
                expand_k=config.expand_k,
                expand_v=config.expand_v,
                num_heads=config.n_heads,
                layer_idx=layer_idx,
            )
            # Latent Memory Module
            self.memory = LatentMemoryModule(
                hidden_size=config.d_model,
                memory_slots=config.memory_slots,
                memory_dim=config.memory_dim,
                use_triton=False,
            )
            nn.init.constant_(self.memory.W_eta.bias, -1.0)
            self.W_alpha = nn.Linear(config.d_model, 1)
            self.C_to_hidden = nn.Linear(config.memory_dim, config.d_model, bias=False)
        else:
            raise ValueError(f"Unknown layer_type: {self.layer_type}")

        # Sandwich norms
        self.ln1 = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.ln1_out = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.ln2 = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.ln2_out = RMSNorm(config.d_model, eps=config.rms_norm_eps)

        # FFN vs MoE
        dense_layers = config.dense_input_layers
        num_routed = config.num_routed_experts

        if layer_idx < dense_layers or num_routed == 0:
            self.is_moe = False
            self.ffn = SwiGLUBlock(config.d_model, config.d_ff)
        else:
            self.is_moe = True
            if config.moe_type == "bigmac":
                self.ffn = BigMacMoE(config, layer_idx=layer_idx)
            elif config.moe_type == "deepseek":
                # DeepSeekMoE could be added here if needed
                self.ffn = BigMacMoE(config, layer_idx=layer_idx)
            else:
                self.ffn = GroupedMoE(config, layer_idx=layer_idx)

        self.dropout = nn.Dropout(config.dropout)
        self.scale_factor = 1.0 / math.sqrt(2 * self.n_layers)
        self.residual_scale = config.residual_scale

        self._init_weights()

    def _init_weights(self):
        trinity_std = 0.5 / math.sqrt(self.hidden_size)

        if self.layer_type == "gla":
            nn.init.constant_(self.W_alpha.bias, -10.0)
            nn.init.zeros_(self.W_alpha.weight)
            nn.init.normal_(self.C_to_hidden.weight, std=trinity_std)

        def apply_deep_init(m):
            if hasattr(m, 'down') and isinstance(m.down, nn.Linear):
                nn.init.normal_(m.down.weight, mean=0.0, std=trinity_std * self.scale_factor)

        if not self.is_moe:
            nn.init.normal_(self.ffn.gate.weight, mean=0.0, std=trinity_std)
            nn.init.normal_(self.ffn.up.weight, mean=0.0, std=trinity_std)
            apply_deep_init(self.ffn)
        else:
            self.ffn._init_weights(std=trinity_std)
            for expert in self.ffn.shared_experts:
                apply_deep_init(expert)
            nn.init.normal_(self.ffn.experts_w3, mean=0.0, std=trinity_std)

        nn.init.constant_(self.ln1_out.weight, 1.0)
        nn.init.constant_(self.ln2_out.weight, 1.0)

        if hasattr(self.attn, 'o_proj') and isinstance(self.attn.o_proj, nn.Linear):
            nn.init.normal_(self.attn.o_proj.weight, mean=0.0, std=trinity_std * self.scale_factor)
        for proj_name in ['q_proj', 'k_proj', 'v_proj', 'g_proj']:
            if hasattr(self.attn, proj_name):
                m = getattr(self.attn, proj_name)
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=trinity_std)
                elif isinstance(m, nn.Sequential):
                    for subm in m:
                        if isinstance(subm, nn.Linear):
                            nn.init.normal_(subm.weight, mean=0.0, std=trinity_std)

    def forward(self, x, cos=None, sin=None, expert_bias=None,
                memory_state=None, lambda_reg=0.01, **kwargs):
        if self.use_looped_injection:
            P = kwargs.get('P')
            if P is not None:
                x = x + (torch.sigmoid(self.injection_gate) * P)

        if self.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, cos, sin, expert_bias, memory_state, lambda_reg,
                use_reentrant=False, **kwargs,
            )
        return self._forward(x, cos, sin, expert_bias, memory_state, lambda_reg, **kwargs)

    def _forward(self, x, cos=None, sin=None, expert_bias=None,
                 memory_state=None, lambda_reg=0.01, **kwargs):
        # 1. Attention block
        residual = x
        x = self.ln1(x)

        # Build attention kwargs
        attn_kwargs = {}
        if cos is not None and sin is not None:
            attn_kwargs['cos'] = cos
            attn_kwargs['sin'] = sin

        # Pass past_key_values for FLA cache support
        if 'past_key_values' in kwargs and kwargs['past_key_values'] is not None:
            attn_kwargs['past_key_values'] = kwargs['past_key_values']
        if 'use_cache' in kwargs:
            attn_kwargs['use_cache'] = kwargs['use_cache']

        attn_out = self.attn(x, **attn_kwargs)
        if isinstance(attn_out, tuple):
            attn_out = attn_out[0]

        new_memory_state = None
        mem_loss = torch.tensor(0.0, device=x.device)

        # GLA layers: read/write latent memory
        if self.layer_type == "gla" and memory_state is not None:
            new_memory_state, total_mem_loss, _ = self.memory.write_memory(x, memory_state)
            C = self.memory.read_memory(x, new_memory_state)
            alpha = torch.sigmoid(self.W_alpha(x))
            C_proj = self.C_to_hidden(C)
            attn_out = attn_out + (alpha * C_proj)
            mem_loss = total_mem_loss

        # Sandwich norm + residual scaling
        x = residual + self.residual_scale * self.dropout(self.ln1_out(attn_out))

        # 2. FFN / MoE block
        residual = x
        x = self.ln2(x)
        if self.is_moe:
            block_out, aux_loss = self.ffn(x, expert_bias=expert_bias)
        else:
            block_out = self.ffn(x)
            aux_loss = torch.tensor(0.0, device=x.device)

        x = residual + self.residual_scale * self.dropout(self.ln2_out(block_out))
        return x, aux_loss, new_memory_state, mem_loss


# ===================================================================
# Output dataclasses
# ===================================================================
@dataclass
class QuasarModelOutputWithPast(BaseModelOutputWithPast):
    memory_states: dict | None = None
    memory_loss: torch.Tensor | None = None
    aux_loss: torch.Tensor | None = None


@dataclass
class QuasarCausalLMOutputWithPast(CausalLMOutputWithPast):
    memory_states: dict | None = None
    memory_loss: torch.Tensor | None = None
    aux_loss: torch.Tensor | None = None


# ===================================================================
# PreTrainedModel base
# ===================================================================
class QuasarPreTrainedModel(PreTrainedModel):
    config_class = QuasarConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["HybridBlock"]
    _supports_cache_class = True

    def _init_weights(self, module):
        std = getattr(self.config, "initializer_range", 0.02)
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


# ===================================================================
# QuasarModel — base transformer (no LM head)
# Weight prefix: model.* (embed_tokens, embed_norm, layers, norm, rotary_emb, all_moe_*)
# ===================================================================
class QuasarModel(QuasarPreTrainedModel):
    config: QuasarConfig

    def __init__(self, config: QuasarConfig):
        super().__init__(config)
        self.config = config
        d_model = config.d_model
        n_heads = config.n_heads
        n_layers = config.n_layers
        vocab_size = config.vocab_size
        max_seq_len = config.max_seq_len

        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.embed_norm = RMSNorm(d_model, eps=config.rms_norm_eps)
        self.layers = nn.ModuleList([
            HybridBlock(config, i) for i in range(n_layers)
        ])
        self.norm = RMSNorm(d_model, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(
            d_model // n_heads, max_seq_len, base=config.rope_theta,
        )

        # SMEBU global buffers — sized [num_moe, num_experts] to match checkpoint
        self.moe_layer_ffns = [l.ffn for l in self.layers if getattr(l, 'is_moe', False)]
        self.num_moe = len(self.moe_layer_ffns)
        num_experts = config.num_routed_experts
        if self.num_moe > 0 and num_experts > 0:
            self.register_buffer("all_moe_bias", torch.zeros(self.num_moe, num_experts))
            self.register_buffer("all_moe_momentum", torch.zeros(self.num_moe, num_experts))
            self.register_buffer("all_moe_max_vio", torch.zeros(self.num_moe))

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def init_memory(self, batch_size, device, dtype=torch.float32):
        memory_states = {}
        for i, layer in enumerate(self.layers):
            if layer.layer_type == "gla":
                m = torch.zeros(batch_size, layer.memory.K, layer.memory.D, device=device, dtype=dtype)
                memory_states[i] = m
        return memory_states

    def reset_state(self):
        # Latent memory is reinitialized every call when memory_states=None, and SMEBU bias buffers
        # only mutate when self.training is True. Eval (.eval()) is therefore stateless across
        # forwards. This is a contract anchor for the eval server: callers can invoke before each
        # paired sequence to make the stateless guarantee explicit and survive future changes.
        return

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        memory_states: dict | None = None,
        lambda_reg: float = 0.01,
        **kwargs,
    ):
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Embed norm for stability
        hidden_states = self.embed_norm(inputs_embeds)
        batch_size, seq_len, _ = hidden_states.shape

        # Position ids
        if position_ids is None:
            past_seen_tokens = 0
            if past_key_values is not None:
                try:
                    past_seen_tokens = past_key_values.get_seq_length()
                except Exception:
                    past_seen_tokens = 0
            position_ids = torch.arange(past_seen_tokens, past_seen_tokens + seq_len, device=hidden_states.device)

        # RoPE
        max_pos = int(position_ids.max().item() + 1) if position_ids.numel() > 0 else seq_len
        cos_full, sin_full = self.rotary_emb(hidden_states, seq_len=max_pos)
        if position_ids.dim() == 1:
            cos = cos_full[:, :, position_ids]
            sin = sin_full[:, :, position_ids]
        else:
            cos = cos_full[:, :, position_ids[0]]
            sin = sin_full[:, :, position_ids[0]]

        # Memory states
        if memory_states is None:
            memory_states = self.init_memory(batch_size, hidden_states.device, hidden_states.dtype)

        all_hidden_states = () if output_hidden_states else None
        aux_losses = []
        mem_losses = []
        new_memory_states = {}

        # Looped transformer anchor
        P = hidden_states
        num_loops = self.config.num_loops
        current_memory_states = memory_states

        # Snapshot expert bias for gradient checkpointing consistency
        if self.num_moe > 0:
            bias_snapshot = self.all_moe_bias.detach().clone()
        else:
            bias_snapshot = None

        for loop_idx in range(num_loops):
            moe_idx = 0
            iteration_new_memory_states = {}
            for layer in self.layers:
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                bias = bias_snapshot[moe_idx] if (getattr(layer, 'is_moe', False) and bias_snapshot is not None) else None

                layer_out = layer(
                    hidden_states,
                    cos=cos, sin=sin,
                    expert_bias=bias,
                    memory_state=current_memory_states.get(layer.layer_idx),
                    lambda_reg=lambda_reg,
                    P=P if self.config.use_looped_injection else None,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    **kwargs,
                )
                hidden_states, aux_loss, new_m, m_loss = layer_out
                if new_m is not None:
                    iteration_new_memory_states[layer.layer_idx] = new_m
                    mem_losses.append(m_loss)
                if bias is not None:
                    moe_idx += 1
                aux_losses.append(aux_loss)

            current_memory_states = iteration_new_memory_states
            new_memory_states = iteration_new_memory_states

        # SMEBU bias update (no_grad to avoid checkpointing issues)
        if self.training and self.num_moe > 0:
            with torch.no_grad():
                self._update_all_moe_biases()

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        total_aux = torch.stack(aux_losses).sum() if aux_losses else torch.tensor(0.0, device=hidden_states.device)
        total_mem = torch.stack(mem_losses).sum() if mem_losses else torch.tensor(0.0, device=hidden_states.device)

        return QuasarModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            memory_states=new_memory_states,
            memory_loss=total_mem,
            aux_loss=total_aux,
        )

    def _update_all_moe_biases(self):
        violations = torch.stack([m._pending_violation for m in self.moe_layer_ffns])
        m0 = self.moe_layer_ffns[0]
        kappa, lamb, beta = m0.smebu_kappa, m0.smebu_lambda, m0.smebu_beta
        clamped_update = torch.tanh(kappa * violations)
        delta_bi = lamb * clamped_update
        delta_bi = delta_bi - delta_bi.mean(dim=-1, keepdim=True)
        self.all_moe_momentum.mul_(beta).add_(delta_bi, alpha=1 - beta)
        self.all_moe_bias.add_(self.all_moe_momentum).nan_to_num_().clamp_(-10.0, 10.0)
        self.all_moe_bias.sub_(self.all_moe_bias.mean(dim=-1, keepdim=True))
        current_max_vios = -violations.min(dim=-1).values
        self.all_moe_max_vio.mul_(0.99).add_(current_max_vios, alpha=0.01)
        for i, moe in enumerate(self.moe_layer_ffns):
            moe.max_vio.copy_(self.all_moe_max_vio[i])
            del moe._pending_violation


# ===================================================================
# QuasarForCausalLM — with LM head + generation support
# Weight prefix: lm_head.* (top-level), model.* (from QuasarModel)
# ===================================================================
class QuasarForCausalLM(QuasarPreTrainedModel, FLAGenerationMixin):
    config: QuasarConfig
    _tied_weights_keys = {}

    def __init__(self, config: QuasarConfig):
        super().__init__(config)
        self.model = QuasarModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def tie_weights(self, missing_keys=None, recompute_mapping=False):
        pass  # Don't tie — crashes FSDP

    def reset_state(self):
        self.model.reset_state()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        memory_states: dict | None = None,
        lambda_reg: float = 0.01,
        return_dict: bool | None = None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        model_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            memory_states=memory_states,
            lambda_reg=lambda_reg,
            **kwargs,
        )

        hidden_states = model_outputs.last_hidden_state
        total_aux = model_outputs.aux_loss if model_outputs.aux_loss is not None else torch.tensor(0.0, device=hidden_states.device)

        loss = None
        if labels is not None:
            shift_hidden = hidden_states[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            flat_hidden = shift_hidden.view(-1, self.config.d_model)
            flat_labels = shift_labels.view(-1)
            mask = flat_labels != -100
            if mask.any():
                active_hidden = flat_hidden[mask]
                active_labels = flat_labels[mask]
                chunk_size = 256
                total_loss = 0.0
                total_tokens = active_labels.numel()
                for i in range(0, total_tokens, chunk_size):
                    end = min(i + chunk_size, total_tokens)
                    chunk_logits = self.lm_head(active_hidden[i:end])
                    chunk_loss = F.cross_entropy(chunk_logits.float(), active_labels[i:end], reduction='sum')
                    total_loss += chunk_loss
                loss = total_loss / total_tokens
                loss = loss + total_aux + model_outputs.memory_loss
            else:
                loss = torch.tensor(0.0, device=hidden_states.device, requires_grad=True)
            logits = None
        else:
            logits = self.lm_head(hidden_states)

        if not return_dict:
            output = (logits,) + model_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return QuasarCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=model_outputs.past_key_values,
            hidden_states=model_outputs.hidden_states,
            memory_states=model_outputs.memory_states,
            memory_loss=model_outputs.memory_loss,
            aux_loss=total_aux,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        memory_states=None,
        cache_position=None,
        use_cache=True,
        **kwargs,
    ):
        if past_key_values is not None:
            if input_ids is not None:
                input_ids = input_ids[:, -1:]
            if inputs_embeds is not None:
                inputs_embeds = inputs_embeds[:, -1:]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        if memory_states is None and past_key_values is not None:
            memory_states = getattr(past_key_values, "memory_states", None)

        model_inputs.update({
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "memory_states": memory_states,
        })
        return model_inputs

    def update_model_kwargs_for_generation(self, outputs, model_kwargs, is_seq2seq=False, num_new_tokens=1):
        model_kwargs = super().update_model_kwargs_for_generation(
            outputs=outputs, model_kwargs=model_kwargs,
            is_seq2seq=is_seq2seq, num_new_tokens=num_new_tokens,
        )
        if getattr(outputs, "memory_states", None) is not None:
            model_kwargs["memory_states"] = outputs.memory_states
        return model_kwargs

    def _reorder_cache(self, past_key_values, beam_idx):
        if past_key_values is None:
            return None
        return past_key_values.reorder_cache(beam_idx)


__all__ = [
    "QuasarConfig",
    "QuasarPreTrainedModel",
    "QuasarModel",
    "QuasarForCausalLM",
    "QuasarModelOutputWithPast",
    "QuasarCausalLMOutputWithPast",
]
