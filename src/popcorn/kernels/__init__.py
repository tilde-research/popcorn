"""Kernel registry: importing this package registers every op and backend."""

# ruff: noqa: E402

from popcorn.core.sources import declare_backend

declare_backend("fa3", package="flash-attn-3", min_version="3.0.0", max_version="3.0.0", extra="fa3")
declare_backend("fla", package="flash-linear-attention", min_version="0.4.0", max_version="0.4.2", extra="fla")
declare_backend("liger", package="liger-kernel", min_version="0.6.0", max_version="0.7.0", extra="liger")
declare_backend("popcorn", package="triton", min_version="3.6.0", max_version="3.6.0")
declare_backend("quack", package="quack-kernels", min_version="0.5.0", max_version="0.5.0", extra="quack")

from popcorn.kernels.abc import abc
from popcorn.kernels.add_rms_norm import add_rms_norm
from popcorn.kernels.addmm import addmm
from popcorn.kernels.attn import attn
from popcorn.kernels.attn_varlen import attn_varlen
from popcorn.kernels.based import based
from popcorn.kernels.bias_gelu import bias_gelu
from popcorn.kernels.bit_linear import bit_linear
from popcorn.kernels.chunk_global_cumsum import chunk_global_cumsum
from popcorn.kernels.chunk_local_cumsum import chunk_local_cumsum
from popcorn.kernels.comba import comba
from popcorn.kernels.cross_entropy import cross_entropy
from popcorn.kernels.delta_rule import delta_rule
from popcorn.kernels.deltaformer import deltaformer
from popcorn.kernels.dyt import dyt
from popcorn.kernels.embedding import embedding
from popcorn.kernels.forgetting_attn import forgetting_attn
from popcorn.kernels.gated_delta_product import gated_delta_product
from popcorn.kernels.gated_delta_rule import gated_delta_rule
from popcorn.kernels.gated_oja_rule import gated_oja_rule
from popcorn.kernels.geglu import geglu
from popcorn.kernels.gelu import gelu
from popcorn.kernels.gla import gla
from popcorn.kernels.group_norm import group_norm
from popcorn.kernels.group_norm_linear import group_norm_linear
from popcorn.kernels.grpo import grpo
from popcorn.kernels.grpo_offpolicy import grpo_offpolicy
from popcorn.kernels.gsa import gsa
from popcorn.kernels.hadamard_transform import hadamard_transform
from popcorn.kernels.hgrn import hgrn
from popcorn.kernels.int8_int2_matmul import int8_int2_matmul
from popcorn.kernels.iplr_delta_rule import iplr_delta_rule
from popcorn.kernels.jsd import jsd
from popcorn.kernels.kda import kda
from popcorn.kernels.kda_gate import kda_gate
from popcorn.kernels.kda_gate_cumsum import kda_gate_cumsum
from popcorn.kernels.kl_div import kl_div
from popcorn.kernels.l2_norm import l2_norm
from popcorn.kernels.layer_norm import layer_norm
from popcorn.kernels.layer_norm_gated import layer_norm_gated
from popcorn.kernels.layer_norm_linear import layer_norm_linear
from popcorn.kernels.layer_norm_linear_quant import layer_norm_linear_quant
from popcorn.kernels.layer_norm_swish_linear import layer_norm_swish_linear
from popcorn.kernels.lightning_attn import lightning_attn
from popcorn.kernels.linear_attn import linear_attn
from popcorn.kernels.linear_cross_entropy import linear_cross_entropy
from popcorn.kernels.linear_jsd import linear_jsd
from popcorn.kernels.linear_kl_div import linear_kl_div
from popcorn.kernels.llama4_rope import llama4_rope
from popcorn.kernels.log_linear_attn import log_linear_attn
from popcorn.kernels.log_sigmoid import log_sigmoid
from popcorn.kernels.logsumexp import logsumexp
from popcorn.kernels.matmul import matmul
from popcorn.kernels.mean_pooling import mean_pooling
from popcorn.kernels.mesa_net import mesa_net
from popcorn.kernels.mesa_net_decode import mesa_net_decode
from popcorn.kernels.momoe import momoe
from popcorn.kernels.multi_token_attention import multi_token_attention
from popcorn.kernels.neighborhood_attn import neighborhood_attn
from popcorn.kernels.nsa import nsa
from popcorn.kernels.nsa_compression import nsa_compression
from popcorn.kernels.path_attn import path_attn
from popcorn.kernels.poly_norm import poly_norm
from popcorn.kernels.qwen2vl_mrope import qwen2vl_mrope
from popcorn.kernels.rebased import rebased
from popcorn.kernels.retention import retention
from popcorn.kernels.rms_norm import rms_norm
from popcorn.kernels.rms_norm_gated import rms_norm_gated
from popcorn.kernels.rms_norm_linear import rms_norm_linear
from popcorn.kernels.rms_norm_linear_quant import rms_norm_linear_quant
from popcorn.kernels.rms_norm_swish_linear import rms_norm_swish_linear
from popcorn.kernels.rope import rope
from popcorn.kernels.rotary_embedding import rotary_embedding
from popcorn.kernels.rwkv4 import rwkv4
from popcorn.kernels.rwkv6 import rwkv6
from popcorn.kernels.rwkv7 import rwkv7
from popcorn.kernels.rwkv7_addcmul import rwkv7_addcmul
from popcorn.kernels.rwkv7_channel_mixing import rwkv7_channel_mixing
from popcorn.kernels.rwkv7_gate_output import rwkv7_gate_output
from popcorn.kernels.rwkv7_k_update import rwkv7_k_update
from popcorn.kernels.selective_log_softmax import selective_log_softmax
from popcorn.kernels.sigmoid import sigmoid
from popcorn.kernels.simple_gla import simple_gla
from popcorn.kernels.softmax import softmax
from popcorn.kernels.solve_tril import solve_tril
from popcorn.kernels.sparsemax import sparsemax
from popcorn.kernels.sqrelu import sqrelu
from popcorn.kernels.swiglu import swiglu
from popcorn.kernels.swiglu_linear import swiglu_linear
from popcorn.kernels.swiglu_mlp import swiglu_mlp
from popcorn.kernels.swish import swish
from popcorn.kernels.titans_linear import titans_linear
from popcorn.kernels.token_shift import token_shift
from popcorn.kernels.ttt import ttt
from popcorn.kernels.tvd import tvd
from popcorn.kernels.wall_attn import wall_attn

__all__ = [
    "abc",
    "add_rms_norm",
    "addmm",
    "attn",
    "attn_varlen",
    "based",
    "bias_gelu",
    "bit_linear",
    "chunk_global_cumsum",
    "chunk_local_cumsum",
    "comba",
    "cross_entropy",
    "delta_rule",
    "deltaformer",
    "dyt",
    "embedding",
    "forgetting_attn",
    "gated_delta_product",
    "gated_delta_rule",
    "gated_oja_rule",
    "geglu",
    "gelu",
    "gla",
    "group_norm",
    "group_norm_linear",
    "grpo",
    "grpo_offpolicy",
    "gsa",
    "hadamard_transform",
    "hgrn",
    "int8_int2_matmul",
    "iplr_delta_rule",
    "jsd",
    "kda",
    "kda_gate",
    "kda_gate_cumsum",
    "kl_div",
    "l2_norm",
    "layer_norm",
    "layer_norm_gated",
    "layer_norm_linear",
    "layer_norm_linear_quant",
    "layer_norm_swish_linear",
    "lightning_attn",
    "linear_attn",
    "linear_cross_entropy",
    "linear_jsd",
    "linear_kl_div",
    "llama4_rope",
    "log_linear_attn",
    "log_sigmoid",
    "logsumexp",
    "matmul",
    "mean_pooling",
    "mesa_net",
    "mesa_net_decode",
    "momoe",
    "multi_token_attention",
    "neighborhood_attn",
    "nsa",
    "nsa_compression",
    "path_attn",
    "poly_norm",
    "qwen2vl_mrope",
    "rebased",
    "retention",
    "rms_norm",
    "rms_norm_gated",
    "rms_norm_linear",
    "rms_norm_linear_quant",
    "rms_norm_swish_linear",
    "rope",
    "rotary_embedding",
    "rwkv4",
    "rwkv6",
    "rwkv7",
    "rwkv7_addcmul",
    "rwkv7_channel_mixing",
    "rwkv7_gate_output",
    "rwkv7_k_update",
    "selective_log_softmax",
    "sigmoid",
    "simple_gla",
    "softmax",
    "solve_tril",
    "sparsemax",
    "sqrelu",
    "swiglu",
    "swiglu_linear",
    "swiglu_mlp",
    "swish",
    "titans_linear",
    "token_shift",
    "ttt",
    "tvd",
    "wall_attn",
]
