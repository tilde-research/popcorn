"""The kernel taxonomy, in one place. Add a tag here and only here, after
checking that no existing one fits -- a single file is what keeps near
duplicates ("linear", "linear kernel") from creeping in. Values are the
human-readable card labels. Every tag must be worn by at least one kernel and
every kernel wears at least one tag, both enforced by the conventions tests."""

import enum


@enum.unique
class Tag(enum.StrEnum):
    SEQUENCE_MIXER = "sequence mixer"  # moves information across positions
    FEATURE_MIXER = "feature mixer"  # mixes within a position, across channels
    ATTENTION = "attention"  # softmax attention family
    LINEAR_ATTENTION = "linear attention"  # linear/recurrent scans over the sequence
    ACTIVATION = "activation"
    NORMALIZATION = "normalization"
    LINEAR = "linear"  # built around a matmul / projection
    FUSED = "fused"  # fuses ops that are separate in torch
    LOSS = "loss"
    QUANTIZED = "quantized"
    POSITIONAL = "positional"  # position-dependent transforms (rope family)
    VARLEN = "varlen"  # packed cu_seqlens layouts
    REDUCTION = "reduction"  # softmax, logsumexp, cumsums, pooling
