"""Butter: a hybrid Wall-Attention / attention language model in plain PyTorch.

HF-style modeling file with no popcorn dependency. The fusable ops live in the
`OPS` namespace the model takes at construction, so a drop-in with the same names
(such as `popcorn.kernels`) can replace them without touching the modeling code.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

RCP_LN2 = 1.4426950408889634


@dataclass
class ButterConfig:
    hidden_size: int = 2048
    num_heads: int = 16
    num_layers: int = 4
    intermediate_size: int = 5632
    vocab_size: int = 32768
    max_position_embeddings: int = 4096
    rope_theta: float = 10000.0
    layer_types: tuple[str, ...] = ("wall", "attn")  # cycled across layers

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads


def rms_norm(x, weight, eps=1e-6):
    h = x.float()
    return (h * torch.rsqrt(h.square().mean(-1, keepdim=True) + eps)).to(x.dtype) * weight


def _rotate_half(x):
    a, b = x.chunk(2, -1)
    return torch.cat((-b, a), -1)


def rope(q, k, cos, sin):  # q, k: [batch, heads, seq, head_dim]; cos, sin: [1, seq, head_dim]
    cos, sin = cos[:, None], sin[:, None]
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


def swiglu(a, b):
    return F.silu(a) * b


def wall_attn(q, k, v, g):  # [batch, seq, heads, dim]; g is log-decay per channel
    """Causal attention with per-channel decay on every logit (materializes scores)."""
    scale = q.shape[-1] ** -0.5 * RCP_LN2
    q32, k32, v32, g32 = (t.float() for t in (q, k, v, g))
    prefix = g32.cumsum(1) * RCP_LN2
    scores = torch.einsum("bihc,bjhc->bhij", q32 * torch.exp2(prefix), k32 * torch.exp2(-prefix)) * scale
    idx = torch.arange(q.shape[1], device=q.device)
    scores = scores.masked_fill(idx[:, None] < idx[None, :], float("-inf"))
    m = scores.amax(-1, keepdim=True)
    m = torch.where(torch.isfinite(m), m, 0.0)
    p = torch.exp2(scores - m)
    return torch.einsum("bhij,bjhc->bihc", p / p.sum(-1, keepdim=True), v32).to(q.dtype)


def linear_cross_entropy(x, weight, labels):  # materializes the [tokens, vocab] logits
    return F.cross_entropy(x @ weight.T, labels)


OPS = SimpleNamespace(
    rms_norm=rms_norm,
    rope=rope,
    swiglu=swiglu,
    wall_attn=wall_attn,
    linear_cross_entropy=linear_cross_entropy,
)


class ButterAttention(nn.Module):
    def __init__(self, config: ButterConfig, ops):
        super().__init__()
        self.rope = ops.rope
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


class ButterWallMixer(nn.Module):
    def __init__(self, config: ButterConfig, ops):
        super().__init__()
        self.wall_attn = ops.wall_attn
        self.num_heads, self.head_dim = config.num_heads, config.head_dim
        self.qkv_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.g_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, x):
        batch, seq, hidden = x.shape
        q, k, v = (t.view(batch, seq, self.num_heads, self.head_dim) for t in self.qkv_proj(x).chunk(3, -1))
        g = F.logsigmoid(self.g_proj(x).view(batch, seq, self.num_heads, self.head_dim))
        return self.o_proj(self.wall_attn(q, k, v, g).reshape(batch, seq, hidden))


class ButterMLP(nn.Module):
    def __init__(self, config: ButterConfig, ops):
        super().__init__()
        self.swiglu = ops.swiglu
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(self.swiglu(self.gate_proj(x), self.up_proj(x)))


class ButterDecoderLayer(nn.Module):
    def __init__(self, config: ButterConfig, layer_type: str, ops):
        super().__init__()
        self.rms_norm = ops.rms_norm
        self.mixer = ButterWallMixer(config, ops) if layer_type == "wall" else ButterAttention(config, ops)
        self.mlp = ButterMLP(config, ops)
        self.input_norm = nn.Parameter(torch.ones(config.hidden_size))
        self.post_norm = nn.Parameter(torch.ones(config.hidden_size))

    def forward(self, x):
        x = x + self.mixer(self.rms_norm(x, self.input_norm))
        return x + self.mlp(self.rms_norm(x, self.post_norm))


class ButterForCausalLM(nn.Module):
    def __init__(self, config: ButterConfig, ops=OPS):
        super().__init__()
        self.config = config
        self.rms_norm, self.loss = ops.rms_norm, ops.linear_cross_entropy
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            ButterDecoderLayer(config, config.layer_types[i % len(config.layer_types)], ops) for i in range(config.num_layers)
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
