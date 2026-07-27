"""Butter: a hybrid Gated-DeltaNet / attention language model built on popcorn kernels.

HF-style modeling file. Every fusable op is a popcorn kernel; `ButterConfig.dispatch`
selects between tuned dispatch (default) and the kernels' plain-torch references —
same weights, same call sites, no monkey patching.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from popcorn.kernels import gated_delta_rule, linear_cross_entropy, rms_norm, rope, swiglu


@dataclass
class ButterConfig:
    hidden_size: int = 2048
    num_heads: int = 16
    num_layers: int = 4
    intermediate_size: int = 5632
    vocab_size: int = 32768
    max_position_embeddings: int = 4096
    rope_theta: float = 10000.0
    layer_types: tuple[str, ...] = ("delta", "attn")  # cycled across layers
    dispatch: bool = True  # False: call the plain-torch reference of every kernel

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads


def _impl(op, config: ButterConfig):
    return op if config.dispatch else op.reference


class ButterAttention(nn.Module):
    def __init__(self, config: ButterConfig):
        super().__init__()
        self.rope = _impl(rope, config)
        self.num_heads, self.head_dim = config.num_heads, config.head_dim
        self.qkv_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        positions = torch.arange(config.max_position_embeddings)
        inv_freq = config.rope_theta ** (-torch.arange(0, config.head_dim, 2) / config.head_dim)
        angles = torch.cat((table := torch.outer(positions, inv_freq), table), -1)[None]
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x):
        batch, seq, hidden = x.shape
        q, k, v = (t.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2) for t in self.qkv_proj(x).chunk(3, -1))
        q, k = self.rope(q, k, self.cos[:, :seq], self.sin[:, :seq])
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(o.transpose(1, 2).reshape(batch, seq, hidden))


class ButterDeltaMixer(nn.Module):
    def __init__(self, config: ButterConfig):
        super().__init__()
        self.gated_delta_rule = _impl(gated_delta_rule, config)
        self.num_heads, self.head_dim = config.num_heads, config.head_dim
        self.qkv_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, 2 * config.num_heads, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, x):
        batch, seq, hidden = x.shape
        q, k, v = (t.view(batch, seq, self.num_heads, self.head_dim) for t in self.qkv_proj(x).chunk(3, -1))
        g, beta = self.gate_proj(x).chunk(2, -1)
        o = self.gated_delta_rule(q, F.normalize(k, dim=-1), v, F.logsigmoid(g), beta.sigmoid())
        return self.o_proj(o.reshape(batch, seq, hidden))


class ButterMLP(nn.Module):
    def __init__(self, config: ButterConfig):
        super().__init__()
        self.swiglu = _impl(swiglu, config)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(self.swiglu(self.gate_proj(x), self.up_proj(x)))


class ButterDecoderLayer(nn.Module):
    def __init__(self, config: ButterConfig, layer_type: str):
        super().__init__()
        self.rms_norm = _impl(rms_norm, config)
        self.mixer = ButterDeltaMixer(config) if layer_type == "delta" else ButterAttention(config)
        self.mlp = ButterMLP(config)
        self.input_norm = nn.Parameter(torch.ones(config.hidden_size))
        self.post_norm = nn.Parameter(torch.ones(config.hidden_size))

    def forward(self, x):
        x = x + self.mixer(self.rms_norm(x, self.input_norm))
        return x + self.mlp(self.rms_norm(x, self.post_norm))


class ButterForCausalLM(nn.Module):
    def __init__(self, config: ButterConfig):
        super().__init__()
        self.config = config
        self.rms_norm = _impl(rms_norm, config)
        self.loss = _impl(linear_cross_entropy, config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            ButterDecoderLayer(config, config.layer_types[i % len(config.layer_types)]) for i in range(config.num_layers)
        )
        self.norm = nn.Parameter(torch.ones(config.hidden_size))
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(0.0, 0.02)

    def forward(self, input_ids, labels=None):
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h)
        h = self.rms_norm(h, self.norm)
        if labels is None:
            return self.lm_head(h)
        return self.loss(h.reshape(-1, self.config.hidden_size), self.lm_head.weight, labels.reshape(-1))
