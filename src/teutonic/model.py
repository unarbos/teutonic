"""Tiny LLaMA model for local testing and the production model config."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LlamaConfig:
    vocab_size: int = 32000
    hidden_dim: int = 128
    intermediate_dim: int = 384
    n_layers: int = 4
    n_heads: int = 4
    seq_len: int = 512
    rope_base: float = 10000.0


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def _precompute_freqs(dim: int, seq_len: int, base: float) -> torch.Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def _apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    B, n_heads, T, head_dim = x.shape
    xc = torch.view_as_complex(x.float().reshape(B, n_heads, T, head_dim // 2, 2))
    freqs = freqs[:T].unsqueeze(0).unsqueeze(0)
    out = torch.view_as_real(xc * freqs).reshape(B, n_heads, T, head_dim)
    return out.type_as(x)


class Attention(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.hidden_dim // cfg.n_heads
        self.wq = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        self.wk = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        self.wv = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        self.wo = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q = _apply_rope(q, freqs)
        k = _apply_rope(k, freqs)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(out.transpose(1, 2).reshape(B, T, -1))


class FeedForward(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.hidden_dim, cfg.intermediate_dim, bias=False)
        self.w2 = nn.Linear(cfg.intermediate_dim, cfg.hidden_dim, bias=False)
        self.w3 = nn.Linear(cfg.hidden_dim, cfg.intermediate_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.attention = Attention(cfg)
        self.feed_forward = FeedForward(cfg)
        self.attention_norm = RMSNorm(cfg.hidden_dim)
        self.ffn_norm = RMSNorm(cfg.hidden_dim)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), freqs)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class TinyLlama(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        self.layers = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.norm = RMSNorm(cfg.hidden_dim)
        self.output = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)
        self.register_buffer(
            "freqs",
            _precompute_freqs(
                cfg.hidden_dim // cfg.n_heads, cfg.seq_len, cfg.rope_base
            ),
            persistent=False,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.tok_embeddings(tokens)
        for layer in self.layers:
            x = layer(x, self.freqs)
        x = self.norm(x)
        return self.output(x)
