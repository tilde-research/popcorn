"""Scalar-arg policy: what splits validity strata, what drives speed, what's continuous.

Dims are always structural. Scalars split into:

- **discrete** (bool / int / str / None / tuples): exact keys in the validity stratum
- **continuous** (float): fitted as `Real` bands like dims, never exact stratum keys

Among both, only `SPEED_ARGS` participate in nearest-neighbor timing. Everything in
`NEUTRAL_ARGS` is ignored for speed (same work, different immediates / labels).
New kernel kwargs must land in one of the two frozensets — the vocabulary test
enforces it, same idea as `DIMS`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# Structural / algorithmic — changes work, tiling, or kernel path.
SPEED_ARGS = frozenset(
    {
        "activation",
        "approximate",
        "block_count",
        "block_size",
        "causal",
        "chunk_size",
        "dilation",
        "head_first",
        "interleaved",
        "kernel_size",
        "max_cg_iterations",
        "mini_batch_size",
        "mrope_section",
        "normalize",
        "num_groups",
        "num_householder",
        "output_final_state",
        "padding",
        "reduction",
        "reverse",
        "sparse",
        "top_k",
        "use_norm",
        "use_normalize",
        "use_scale",
        "window_size",
    }
)

# Constants / labels — same launch shape; ignored when picking a timed neighbor.
NEUTRAL_ARGS = frozenset(
    {
        "alpha",
        "beta",
        "dim",
        "eps",
        "eps_high",
        "eps_low",
        "ignore_index",
        "label_smoothing",
        "layer_idx",
        "log_target",
        "lower_bound",
        "num_layers",
        "offset",
        "scale",
        "softmax_scale",
        "temperature",
    }
)

KNOWN_ARGS = SPEED_ARGS | NEUTRAL_ARGS


def is_continuous(value: Any) -> bool:
    """True for real floats; bools/ints/None/str/tuples stay discrete."""
    return isinstance(value, float) and not isinstance(value, bool)


def partition_args(args: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
    """Split a call's scalars into (discrete, continuous)."""
    discrete: dict[str, Any] = {}
    continuous: dict[str, float] = {}
    for name, value in args.items():
        if is_continuous(value):
            continuous[name] = float(value)
        else:
            discrete[name] = value
    return discrete, continuous


def speed_discrete(args: Mapping[str, Any]) -> dict[str, Any]:
    """Discrete SPEED_ARGS subset — must match exactly for timing lookup."""
    discrete, _ = partition_args(args)
    return {name: value for name, value in discrete.items() if name in SPEED_ARGS}


def speed_continuous(args: Mapping[str, Any]) -> dict[str, float]:
    """Continuous SPEED_ARGS subset — log-distance like dims for timing lookup."""
    _, continuous = partition_args(args)
    return {name: value for name, value in continuous.items() if name in SPEED_ARGS}
