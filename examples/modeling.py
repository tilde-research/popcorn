"""A tiny hybrid Wall-Attention language model in plain PyTorch."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelConfig:
    hidden_size: int = 2048
    num_heads: int = 32
    num_layers: int = 4
    intermediate_size: int = 5632
    vocab_size: int = 32768
    max_position_embeddings: int = 4096
    layer_types: tuple[str, ...] = ("wall", "attn")


def rms_norm(x, weight, eps=1e-6):
    h = x.float()
    return (h * torch.rsqrt(h.square().mean(-1, keepdim=True) + eps)).to(x.dtype) * weight


def rope(q, k, cos, sin):
    def rotate(x):
        a, b = x.chunk(2, -1)
        return x * cos[:, None] + torch.cat((-b, a), -1) * sin[:, None]

    return rotate(q), rotate(k)


def swiglu(a, b):
    return F.silu(a) * b


def wall_attn(q, k, v, g):
    dtype = q.dtype
    q, k, v, g = (tensor.float() for tensor in (q, k, v, g))
    prefix = g.cumsum(1)
    output = []
    for i in range(q.shape[1]):
        decay = (prefix[:, i, None] - prefix[:, : i + 1]).exp()
        scores = torch.einsum("bhd,bjhd,bjhd->bhj", q[:, i], k[:, : i + 1], decay) * q.shape[-1] ** -0.5
        output.append(torch.einsum("bhj,bjhd->bhd", scores.softmax(-1), v[:, : i + 1]))
    return torch.stack(output, 1).to(dtype)


def linear_cross_entropy(x, weight, labels):
    return F.cross_entropy(x @ weight.T, labels)


class Block(nn.Module):
    def __init__(self, config: ModelConfig, layer_type: str, rms_norm, rope, swiglu, wall_attn):
        super().__init__()
        self.wall = layer_type == "wall"
        self.op = wall_attn if self.wall else rope
        self.heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.mix_out = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        if self.wall:
            # Bias the gate open so the cumulative decay stays inside fp32 range over long sequences.
            self.gate = nn.Linear(config.hidden_size, config.hidden_size)
            nn.init.constant_(self.gate.bias, 4.0)
        else:
            positions = torch.arange(config.max_position_embeddings)
            frequencies = 10000 ** (-torch.arange(0, self.head_dim, 2) / self.head_dim)
            angles = torch.outer(positions, frequencies)
            angles = torch.cat((angles, angles), -1)[None]
            self.register_buffer("cos", angles.cos(), persistent=False)
            self.register_buffer("sin", angles.sin(), persistent=False)

        self.norm, self.swiglu = rms_norm, swiglu
        self.norm_weights = nn.Parameter(torch.ones(2, config.hidden_size))
        self.mlp_in = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.mlp_out = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def mix(self, x):
        batch, seq, hidden = x.shape
        q, k, v = (part.view(batch, seq, self.heads, self.head_dim) for part in self.qkv(x).chunk(3, -1))
        if self.wall:
            g = F.logsigmoid(self.gate(x).view(batch, seq, self.heads, self.head_dim))
            y = self.op(q, k, v, g)
        else:
            q, k = self.op(q.transpose(1, 2), k.transpose(1, 2), self.cos[:, :seq], self.sin[:, :seq])
            y = F.scaled_dot_product_attention(q, k, v.transpose(1, 2), is_causal=True).transpose(1, 2)
        return self.mix_out(y.reshape(batch, seq, hidden))

    def forward(self, x):
        x = x + self.mix(self.norm(x, self.norm_weights[0]))
        gate, value = self.mlp_in(self.norm(x, self.norm_weights[1])).chunk(2, -1)
        return x + self.mlp_out(self.swiglu(gate, value))


class CausalLM(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        ops=None,
        *,
        rms_norm=rms_norm,
        rope=rope,
        swiglu=swiglu,
        wall_attn=wall_attn,
        linear_cross_entropy=linear_cross_entropy,
    ):
        super().__init__()
        if ops is not None:
            rms_norm = ops.rms_norm
            rope = ops.rope
            swiglu = ops.swiglu
            wall_attn = ops.wall_attn
            linear_cross_entropy = ops.linear_cross_entropy
        self.rms_norm, self.loss = rms_norm, linear_cross_entropy
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            Block(
                config,
                config.layer_types[i % len(config.layer_types)],
                rms_norm,
                rope,
                swiglu,
                wall_attn,
            )
            for i in range(config.num_layers)
        )
        self.norm = nn.Parameter(torch.ones(config.hidden_size))
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, labels=None):
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h)
        h = self.rms_norm(h, self.norm)
        if labels is None:
            return self.lm_head(h)
        return self.loss(h.flatten(0, 1), self.lm_head.weight, labels.flatten())
