"""The canonical dim vocabulary and each name's default grid pool.

Like tags, this single table is what prevents near-duplicate names: reuse
before inventing, torch's names where torch has them (`normalized_shape`,
`vocab`, `out_channels`), no abbreviations. A size determined by another dim is
an expression in the annotation (`response+1`, `head_dim/2`,
`seq*num_householder`), never a name. `heads` when an op has one head count,
q_heads/kv_heads when query and kv counts differ. Plurals are counts, singulars
are extents.

Pools are production-derived: dense and adversarial on the short end (tile±1,
powers of two), then sparse through lengths and widths that show up in real
checkpoints (Llama 3.1, Qwen3, Mistral/Mixtral, Gemma, DeepSeek) and today's
long-context windows through 1M tokens. The `"..."` entry is
the leading extent for variadic annotations (rank-1 in the grid for now).
Per-op pool overrides are not supported — validity is learned from report rows
via `bench map`.
"""

from popcorn.core.spaces import Range, Space

# Sequence / token extents: fine below 4K (training chunks, decode steps, tile
# edges), then the shipped context ladders: 8K–128K common and 256K–1M frontier.
_SEQ = Range(2, 4096) | {
    8192,
    16384,
    32768,
    65536,
    131_072,
    262_144,
    524_288,
    1_048_576,
}

# Hidden / FFN widths: small models and norms through Llama-405B (16384) /
# Command-R+ (12288). Intermediate is typically ~2.7–3.5× hidden (SwiGLU).
_WIDTH = Range(8, 512) | {
    768,
    1024,
    1536,
    2048,
    2560,
    3072,
    4096,
    5120,
    6144,
    8192,
    10240,
    12288,
    14336,
    16384,
    18432,
    22016,
    24576,
    28672,
    32768,
    53248,
}

# Attention heads: MQA (kv=1), GQA groups (2/4/8), and MHA (kv == q).
# Llama 3.1: 32/8, 64/8, 128/8; Qwen3: 16/8 … 64/8; Mixtral: 32/8. Every one of those
# divides, which the pools cannot say on their own, so an op declaring both counts gets
# its `kv_heads` snapped onto a divisor of `q_heads` when the grid pairs them (see
# `bench.grid._repair`); that is also why a case can carry a `kv_heads` not listed here.
# The same pass repairs every other cross-axis relation (rotary `half` and `mrope_section`
# against `head_dim`, `top_k` against `experts`, `cos_batch` against `batch`, `levels`
# against `seq`, `mini_batch_size` against `seq`), so repaired cases can carry values
# these pools never list.
_Q_HEADS = {1, 2, 4, 6, 8, 12, 16, 18, 20, 24, 28, 32, 36, 40, 48, 64, 72, 80, 96, 128}
_KV_HEADS = {1, 2, 4, 8, 16, 32, 40, 64, 128}

# Head / state dims: 64 and 128 dominate; 256 appears (Gemma 2); 16–32 show up
# in linear-attn / NSA key spaces; 80/96 for a few multimodal stacks.
_HEAD_DIM = {16, 32, 48, 64, 80, 96, 128, 192, 256}

# Vocab: GPT-2-class 50K, Llama/Mistral 32K–128K, Qwen ~152K, Gemma 256K.
_VOCAB = Range(2, 4096) | {32_000, 32_768, 50_257, 50_304, 128_256, 151_936, 152_064, 256_000}

# Production anchor per dim for one-dimensional benchmark ladders: the value every other
# dim is pinned to while one axis sweeps its pool (see `bench.grid.ladder_cases`). Values
# sketch a Llama-3-class decoder (32 q / 8 kv heads of 128, hidden 4096, GQA vocab 128K)
# and are snapped to the nearest pool member at grid build, so the table can stay round
# while pools evolve. Dims absent here anchor on their pool median.
ANCHORS: dict[str, int] = {
    "...": 0,
    "batch": 8,
    "seq": 4096,
    "tokens": 4096,
    "total": 4096,
    "boundaries": 33,
    "first": 64,
    "second": 64,
    "heads": 32,
    "q_heads": 32,
    "kv_heads": 8,
    "head_dim": 128,
    "key_dim": 128,
    "value_dim": 128,
    "blocks": 8,
    "block_size": 64,
    "slots": 64,
    "levels": 8,
    "gate_dim": 256,
    "cos_batch": 1,
    "half": 64,
    "hidden": 4096,
    "hadamard_dim": 4096,
    "intermediate": 14336,
    "out_features": 4096,
    "embedding_dim": 4096,
    "normalized_shape": 4096,
    "vocab": 128_256,
    "channels": 4096,
    "out_channels": 1024,
    "width": 256,
    "kernel_size": 3,
    "experts": 8,
    "rows": 4096,
    "cols": 4096,
    "inner": 4096,
    "response": 1023,
    "target_hidden": 4096,
    "teacher_hidden": 4096,
}

# Long-context curves keep the axis being measured on its full pool, but pin the
# orthogonal work to a small, valid model. This is deliberately an override,
# not a second copy of `ANCHORS`: production remains the source of truth for
# every dim not named here.
LONG_CONTEXT_ANCHORS: dict[str, int] = {
    "...": 1,
    "batch": 1,
    "heads": 1,
    "q_heads": 1,
    "kv_heads": 1,
    "head_dim": 64,
    "key_dim": 64,
    "value_dim": 64,
    "blocks": 1,
    "slots": 16,
    "gate_dim": 64,
    "hidden": 256,
    "hadamard_dim": 256,
    "intermediate": 512,
    "out_features": 256,
    "embedding_dim": 256,
    "normalized_shape": 256,
    "vocab": 4096,
    "channels": 256,
    "out_channels": 256,
    "width": 64,
    "rows": 256,
    "cols": 256,
    "inner": 256,
    "target_hidden": 256,
    "teacher_hidden": 256,
}

DIMS: dict[str, Space | set[int] | None] = {
    # Variadic leading extent for annotation `...`. Rank-1 for now: grid emits
    # `batch=()` when 0, else `batch=(n,)`. Keep it batch-sized; temporal axes
    # have explicit names and their own long-context curves.
    "...": {0, 1, 2, 4, 8, 16, 32, 64},
    # data layout
    "batch": {1, 2, 4, 8, 16, 32, 64},
    "seq": _SEQ,
    "tokens": _SEQ,
    "total": _SEQ,
    "boundaries": Range(2, 17) | {33, 65, 129, 257, 513, 1025},
    "first": {9, 16, 64, 128},
    "second": {9, 16, 64, 128},
    # attention
    "heads": _Q_HEADS,
    "q_heads": _Q_HEADS,
    "kv_heads": _KV_HEADS,
    "head_dim": _HEAD_DIM,
    "key_dim": _HEAD_DIM,
    "value_dim": _HEAD_DIM,
    "blocks": {1, 2, 4, 8, 16, 32},
    "block_size": {16, 32, 64, 128, 256},
    "slots": {16, 32, 64, 128, 256},
    "levels": {4, 5, 6, 7, 8, 10, 12, 16},
    "gate_dim": {32, 64, 128, 256, 512, 1024, 2048, 4096},
    # rope
    "cos_batch": {1, 2, 4, 8},
    "half": {8, 16, 32, 40, 48, 64, 96, 128},
    # feature widths
    "hidden": _WIDTH,
    # Sylvester Hadamard transforms exist only at powers of two, which no shared width
    # pool can promise, so the op sweeps its own dim.
    "hadamard_dim": {256, 512, 1024, 2048, 4096, 8192, 16384, 32768},
    "intermediate": _WIDTH,
    "out_features": _WIDTH,
    "embedding_dim": {64, 128, 256, 512, 768, 1024, 2048, 4096},
    "normalized_shape": _WIDTH,
    "vocab": _VOCAB,
    "channels": _WIDTH,
    "out_channels": {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096},
    "width": {8, 16, 32, 64, 128, 256, 512, 1024},
    "kernel_size": {1, 3, 5, 7, 9, 11},
    # MoE: Mixtral 8, Qwen-MoE 60–128, DeepSeek-V3 160+
    "experts": {1, 2, 4, 8, 16, 32, 60, 64, 128, 160, 256},
    # matmul (keep a few non-aligned sizes for packing / tiling bugs)
    "rows": {1, 2, 3, 8, 16, 17, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192},
    "cols": {1, 2, 3, 8, 16, 19, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192},
    "inner": {16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192},
    # losses / RL: completion lengths, plus teacher/student hidden widths
    "response": {15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191},
    "target_hidden": _WIDTH,
    "teacher_hidden": _WIDTH,
}
