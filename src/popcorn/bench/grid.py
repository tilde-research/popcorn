"""Test-case grids and randomized inputs derived from kernel signatures."""

import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch

from popcorn.bench.model import Case
from popcorn.core.annotations import token_size
from popcorn.core.dims import ANCHORS, DIMS, LONG_CONTEXT_ANCHORS
from popcorn.core.spaces import space

# `op` stays `Any` here: the layering test forbids this module from importing
# the dispatcher, even for annotations.

# Seeds fed into Space.grid: adversarial shorts (tile±1), powers of two, then
# a few mid/long context lengths so Range-based pools don't only hit endpoints.
GRID = (
    1,
    2,
    3,
    8,
    16,
    17,
    33,
    64,
    65,
    128,
    129,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    32768,
    131072,
    1048576,
)
DTYPES = (torch.float32, torch.float16, torch.bfloat16)
ELLIPSIS = "..."

# Refuse to materialize a full cartesian product past this; pass `limit=`.
_MAX_FULL_GRID = 100_000
_CURVE_PROFILES = ("production", "long-context")
_LONG_CONTEXT_AXES = ("seq", "tokens", "total", "response")
_PRODUCTION_CONTEXT_MAX = 32_768
_LONG_CONTEXT_MIN = {"seq": 8192, "tokens": 8192, "total": 8192, "response": 4095}
_SMOKE_TEMPORAL_MAX = 512
_FP16_BACKWARD_ACCUMULATION_LIMIT = 32_768
_TEMPORAL_DIMS = frozenset(_LONG_CONTEXT_AXES)
_PRIMARY_AXES = (
    *_LONG_CONTEXT_AXES,
    "hidden",
    "normalized_shape",
    "head_dim",
    "key_dim",
    "value_dim",
    "channels",
    "vocab",
    ELLIPSIS,
)


@dataclass(frozen=True, slots=True)
class CaseSeries:
    """A named planning unit; its metadata is never written to report rows."""

    name: str
    kind: Literal["coverage", "curve", "adaptive"]
    cases: tuple[Case, ...]
    profile: str | None = None
    axis: str | None = None
    fixed_dims: tuple[tuple[str, int], ...] = ()
    batch: tuple[int, ...] | None = None
    dtype: torch.dtype | None = None
    args: tuple[tuple[str, Any], ...] = ()
    present: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.kind == "curve" and (self.profile is None or self.axis is None or self.dtype is None):
            raise ValueError("curve series require profile, axis, and dtype")


@dataclass(frozen=True, slots=True)
class CasePlan:
    """Ordered case series with first-seen deduplication at the execution edge."""

    series: tuple[CaseSeries, ...]

    def flatten(self) -> list[Case]:
        found: dict[str, Case] = {}
        for series in self.series:
            for case in series.cases:
                found.setdefault(case.case_id, case)
        return list(found.values())

    def cases(self) -> list[Case]:
        """Compatibility spelling for callers that prefer a noun."""
        return self.flatten()

    def __iter__(self):
        return iter(self.flatten())

    def __len__(self) -> int:
        return len(self.flatten())


def _dim_pool(_op: Any, name: str) -> list[int]:
    """Pool for `name` from the global `DIMS` vocabulary (unknown → `GRID`)."""
    spec = DIMS.get(name)
    if spec is None:
        return list(GRID)
    return space(spec).grid(seeds=GRID)


def _repair_grouped_query(values: dict[str, Any], _args: dict[str, Any]) -> None:
    q, kv = values.get("q_heads"), values.get("kv_heads")
    if q is not None and kv is not None and not (kv <= q and q % kv == 0):
        values["kv_heads"] = max(group for group in range(1, min(q, kv) + 1) if q % group == 0)


def _repair_rotary_half(values: dict[str, Any], _args: dict[str, Any]) -> None:
    half, head_dim = values.get("half"), values.get("head_dim")
    if half is not None and head_dim is not None:
        values["half"] = min(half, head_dim // 2)


def _repair_moe_top_k(values: dict[str, Any], args: dict[str, Any]) -> None:
    top_k, experts = args.get("top_k"), values.get("experts")
    if top_k is not None and experts is not None and top_k > experts:
        args["top_k"] = experts


def _repair_mrope_section(values: dict[str, Any], args: dict[str, Any]) -> None:
    section, head_dim = args.get("mrope_section"), values.get("head_dim")
    if section is not None and head_dim is not None and sum(section) != head_dim // 2:
        target = head_dim // 2
        args["mrope_section"] = [target - 2 * (target // 4), target // 4, target // 4]


def _repair_rotary_batch(values: dict[str, Any], _args: dict[str, Any]) -> None:
    cos_batch, batch = values.get("cos_batch"), values.get("batch")
    if cos_batch is not None and batch is not None and cos_batch not in (1, batch):
        values["cos_batch"] = 1


def _repair_log_linear_levels(values: dict[str, Any], _args: dict[str, Any]) -> None:
    levels, seq = values.get("levels"), values.get("seq")
    if levels is not None and seq is not None:
        values["levels"] = math.ceil(math.log2(seq)) + 1 if seq > 1 else 1


def _repair_ttt_mini_batch(values: dict[str, Any], args: dict[str, Any]) -> None:
    mini_batch, seq = args.get("mini_batch_size"), values.get("seq")
    if mini_batch is not None and seq is not None and seq % mini_batch:
        args["mini_batch_size"] = max(size for size in range(1, min(seq, mini_batch) + 1) if seq % size == 0)


def _repair_group_count(values: dict[str, Any], args: dict[str, Any]) -> None:
    num_groups, channels = args.get("num_groups"), values.get("channels")
    if num_groups is not None and channels is not None and channels % num_groups:
        args["num_groups"] = max(count for count in range(1, min(channels, num_groups) + 1) if channels % count == 0)


def _repair_gate_dim(values: dict[str, Any], _args: dict[str, Any]) -> None:
    heads, key_dim = values.get("heads"), values.get("key_dim")
    if "gate_dim" in values and heads is not None and key_dim is not None:
        values["gate_dim"] = heads * key_dim


def _repair_varlen_boundaries(values: dict[str, Any], _args: dict[str, Any]) -> None:
    boundaries, total = values.get("boundaries"), values.get("total")
    if boundaries is not None and total is not None and boundaries > total + 1:
        values["boundaries"] = total + 1


def _repair_padded_conv(values: dict[str, Any], args: dict[str, Any]) -> None:
    kernel_size, seq, padding = values.get("kernel_size"), values.get("seq"), args.get("padding")
    if kernel_size is not None and seq is not None and padding is not None:
        values["kernel_size"] = max(1, min(kernel_size, seq + 2 * padding))


def _repair_sparse_selection(values: dict[str, Any], args: dict[str, Any]) -> None:
    """Keep top-k block selection deterministic across reference and kernels."""
    seq, block_count, block_size = values.get("seq"), args.get("block_count"), args.get("block_size")
    if seq is not None and block_count is not None and block_size is not None:
        values["seq"] = min(seq, block_count * block_size)


def _repair(values: dict[str, Any], args: dict[str, Any]) -> None:
    """Snap independent pools onto the cross-axis relations real layers satisfy."""
    _repair_grouped_query(values, args)
    _repair_rotary_half(values, args)
    _repair_moe_top_k(values, args)
    _repair_mrope_section(values, args)
    _repair_rotary_batch(values, args)
    _repair_log_linear_levels(values, args)
    _repair_ttt_mini_batch(values, args)
    _repair_group_count(values, args)
    _repair_gate_dim(values, args)
    _repair_varlen_boundaries(values, args)
    _repair_padded_conv(values, args)
    _repair_sparse_selection(values, args)


def _unrank(index: int, sizes: Sequence[int]) -> list[int]:
    coords = []
    for size in reversed(sizes):
        coords.append(index % size)
        index //= size
    coords.reverse()
    return coords


def _pairwise_indices(sizes: Sequence[int]) -> list[tuple[int, ...]]:
    """Deterministic covering array: every value pair across every two axes."""
    if not sizes or any(size < 1 for size in sizes):
        return []
    order = sorted(range(len(sizes)), key=lambda index: (-sizes[index], index))
    rows = [[value] for value in range(sizes[order[0]])]
    for position, axis in enumerate(order[1:], start=1):
        size = sizes[axis]
        prior_sizes = [sizes[index] for index in order[:position]]
        uncovered = {
            (prior, prior_value, value)
            for prior, prior_size in enumerate(prior_sizes)
            for prior_value in range(prior_size)
            for value in range(size)
        }

        # Extend existing rows with the value that covers the most new pairs.
        for row_index, row in enumerate(rows):
            scores = [sum((prior, row[prior], value) in uncovered for prior in range(position)) for value in range(size)]
            best = max(scores)
            choices = [value for value, score in enumerate(scores) if score == best]
            value = choices[row_index % len(choices)]
            row.append(value)
            for prior in range(position):
                uncovered.discard((prior, row[prior], value))

        # Rare pairs left after horizontal growth each seed one additional row.
        while uncovered:
            prior, prior_value, value = min(uncovered)
            row = [0] * position + [value]
            row[prior] = prior_value
            for other, prior_size in enumerate(prior_sizes):
                if other == prior:
                    continue
                choices = [candidate for candidate in range(prior_size) if (other, candidate, value) in uncovered]
                row[other] = choices[0] if choices else 0
            for other in range(position):
                uncovered.discard((other, row[other], value))
            rows.append(row)

    restored = []
    for row in rows:
        coordinates = [0] * len(sizes)
        for position, axis in enumerate(order):
            coordinates[axis] = row[position]
        restored.append(tuple(coordinates))
    return restored


def _pairwise_rows(axes: Mapping[str, Sequence[Any]], limit: int | None = None) -> list[dict[str, Any]]:
    """Map the shared covering-array coordinates back onto named value pools."""
    names = sorted(axes)
    if not names:
        return []
    rows = _pairwise_indices([len(axes[name]) for name in names])
    if limit is not None:
        rows = rows[:limit]
    return [{name: axes[name][row[index]] for index, name in enumerate(names)} for row in rows]


def _missing_args(op: Any) -> list[str]:
    return [name for name, pool in op.arg_pools.items() if pool is None]


def _dim_names(op: Any) -> list[str]:
    has_ellipsis = any(... in spec.tokens for spec in op.specs)
    return sorted(op._dims | ({ELLIPSIS} if has_ellipsis else set()))


def _dtypes(op: Any) -> tuple[torch.dtype, ...]:
    return DTYPES if any("float" in dtype for spec in op.specs for dtype in spec.dtypes) else (torch.float32,)


def _build_case(
    values: Mapping[str, int],
    dtype: torch.dtype,
    args: Mapping[str, Any],
    present: frozenset[str],
) -> Case:
    repaired_values, repaired_args = dict(values), dict(args)
    _repair(repaired_values, repaired_args)
    leading = repaired_values.pop(ELLIPSIS, None)
    batch = () if leading is None or leading == 0 else (leading,)
    return Case(
        tuple(sorted(repaired_values.items())),
        batch,
        dtype,
        tuple(sorted(repaired_args.items())),
        present,
    )


def covering_cases(op: Any) -> list[Case]:
    """Pairwise-cover every declared dim and case axis, smallest shapes first.

    Full products reach hundreds of millions of cases. A covering array keeps
    every value and every two-axis interaction while leaving higher-order
    combinations to adaptive follow-up. Ordering by dimension rank lets an
    online runner discover an OOM boundary before reaching dominated shapes.
    """
    if missing := _missing_args(op):
        raise TypeError(f"{op.name}: arguments {missing} need test_args, a default, or a Literal annotation")
    dim_names = _dim_names(op)
    dim_pools = [_dim_pool(op, name) for name in dim_names]
    arg_names = [name for name, pool in sorted(op.arg_pools.items()) if pool is not None]
    arg_pools = [list(op.arg_pools[name]) for name in arg_names]
    opt_names = sorted({spec.param for spec in op.specs if spec.optional})
    dtypes = list(_dtypes(op))
    pools: list[list[Any]] = [*dim_pools, dtypes, *arg_pools, *([[False, True]] * len(opt_names))]
    sizes = [len(pool) for pool in pools]
    if any(size == 0 for size in sizes):
        return []

    coordinates = [
        (0,) * len(sizes),
        *_pairwise_indices(sizes),
        tuple(size - 1 for size in sizes),
    ]
    dim_count = len(dim_names)
    coordinates.sort(key=lambda row: (sum(row[:dim_count]), max(row[:dim_count], default=0), row))

    found: list[Case] = []
    seen: set[str] = set()
    for row in coordinates:
        cursor = 0
        values = {name: pool[row[cursor + offset]] for offset, (name, pool) in enumerate(zip(dim_names, dim_pools))}
        cursor += dim_count
        dtype = dtypes[row[cursor]]
        cursor += 1
        args = {name: pool[row[cursor + offset]] for offset, (name, pool) in enumerate(zip(arg_names, arg_pools))}
        cursor += len(arg_names)
        present = frozenset(name for offset, name in enumerate(opt_names) if row[cursor + offset])
        case = _build_case(values, dtype, args, present)
        if case.case_id not in seen:
            seen.add(case.case_id)
            found.append(case)
    return found


def _anchor(pool: Sequence[int], target: int | None) -> int:
    """The pool value nearest the anchor target, or the pool median when none is declared."""
    if target is None:
        return pool[len(pool) // 2]
    return min(pool, key=lambda value: (abs(value - target), value))


def _profile_targets(profile: str) -> Mapping[str, int]:
    if profile == "production":
        return ANCHORS
    if profile == "long-context":
        return ANCHORS | LONG_CONTEXT_ANCHORS
    raise ValueError(f"unknown curve profile {profile!r}; expected one of {list(_CURVE_PROFILES)}")


def _profile_anchors(dim_pools: Mapping[str, Sequence[int]], profile: str) -> dict[str, int]:
    targets = _profile_targets(profile)
    return {name: _anchor(pool, targets.get(name)) for name, pool in dim_pools.items()}


def _curve_variants(op: Any) -> list[tuple[str, dict[str, Any], frozenset[str]]]:
    defaults = {name: list(pool)[0] for name, pool in sorted(op.arg_pools.items()) if pool}
    variants = [("base", defaults, frozenset())]
    for name, pool in sorted(op.arg_pools.items()):
        if pool is None:
            continue
        for index, value in enumerate(list(pool)[1:], start=1):
            variants.append((f"arg-{name}-{index}", defaults | {name: value}, frozenset()))
    for name in sorted({spec.param for spec in op.specs if spec.optional}):
        variants.append((f"present-{name}", dict(defaults), frozenset({name})))
    return variants


def _primary_axis(axes: Sequence[str]) -> str:
    return next((name for name in _PRIMARY_AXES if name in axes), axes[0])


def _curve_axes(dim_names: Sequence[str], profile: str) -> list[str]:
    if profile == "production":
        return list(dim_names)
    return [name for name in _LONG_CONTEXT_AXES if name in dim_names]


def _curve_values(axis: str, pool: Sequence[int], profile: str) -> Sequence[int]:
    """Keep full-model curves practical and reserve the reduced context for the frontier."""
    if axis not in _LONG_CONTEXT_AXES:
        return pool
    if profile == "production":
        selected = [value for value in pool if value <= _PRODUCTION_CONTEXT_MAX]
    else:
        selected = [value for value in pool if value >= _LONG_CONTEXT_MIN[axis]]
    return selected if len(selected) >= 2 else pool


def curve_series(op: Any, profile: str) -> tuple[CaseSeries, ...]:
    """Named one-axis curves for one fixed profile.

    The base scalar/presence context gets a curve for every profile axis. Each
    non-default scalar and present optional also gets a curve on the profile's
    primary axis, so semantic branches are represented by more than one point.
    """
    if missing := _missing_args(op):
        raise TypeError(f"{op.name}: arguments {missing} need test_args, a default, or a Literal annotation")
    _profile_targets(profile)
    dim_names = _dim_names(op)
    dim_pools = {name: _dim_pool(op, name) for name in dim_names}
    axes = _curve_axes(dim_names, profile)
    if not axes or any(not pool for pool in dim_pools.values()) or any(pool == [] for pool in op.arg_pools.values()):
        return ()
    anchors = _profile_anchors(dim_pools, profile)
    variants = _curve_variants(op)
    primary = _primary_axis(axes)
    planned: list[CaseSeries] = []
    for dtype in _dtypes(op):
        dtype_name = str(dtype).removeprefix("torch.")
        for axis in axes:
            for variant, args, present in variants if axis == primary else variants[:1]:
                found: dict[str, Case] = {}
                for value in _curve_values(axis, dim_pools[axis], profile):
                    case = _build_case(anchors | {axis: value}, dtype, args, present)
                    found.setdefault(case.case_id, case)
                if len(found) < 2:
                    continue
                context = _build_case(anchors, dtype, args, present)
                fixed_dims = tuple((name, value) for name, value in context.dims if name != axis)
                planned.append(
                    CaseSeries(
                        f"curve:{profile}:{axis}:{dtype_name}:{variant}",
                        "curve",
                        tuple(found.values()),
                        profile=profile,
                        axis=axis,
                        fixed_dims=fixed_dims,
                        batch=None if axis == ELLIPSIS else context.batch,
                        dtype=dtype,
                        args=context.args,
                        present=present,
                    )
                )
    return tuple(planned)


def coverage_series(op: Any) -> CaseSeries:
    return CaseSeries("coverage", "coverage", tuple(covering_cases(op)))


def ladder_cases(op: Any, profile: str = "production") -> list[Case]:
    """Compatibility API for base-context curves in one named profile."""
    base = tuple(series for series in curve_series(op, profile) if series.name.endswith(":base"))
    return CasePlan(base).flatten()


def case_plan(op: Any) -> CasePlan:
    """Coverage breadth plus production and reduced-shape long-context curves."""
    return CasePlan(
        (
            coverage_series(op),
            *curve_series(op, "production"),
            *curve_series(op, "long-context"),
        )
    )


def grid_cases(op: Any) -> list[Case]:
    """Flatten the shared case plan, preserving the established public API."""
    return case_plan(op).flatten()


def _omitted_extent(tokens: Sequence[Any], represented: set[Any], case: Case) -> int:
    dims, args = dict(case.dims), dict(case.args)
    extent = 1
    for token in tokens:
        if getattr(token, "base", token) in represented:
            continue
        extent *= math.prod(case.batch) if token is ... else token_size(token, dims, args, "return")
    return extent


def backward_safe(op: Any, case: Case) -> bool:
    """Whether fp16 gradients avoid an oversized reduction into any input."""
    if str(case.dtype) != "torch.float16":
        return True
    dims = dict(case.dims)
    temporal = _TEMPORAL_DIMS & dims.keys()
    for spec in op.specs:
        if spec.optional and spec.param not in case.present:
            continue
        if not any("float" in dtype for dtype in spec.dtypes):
            continue
        represented = {getattr(token, "base", token) for token in spec.tokens}
        extent = math.prod(dims[name] for name in temporal if name not in represented)
        if ... not in spec.tokens:
            extent *= math.prod(case.batch)
        if extent >= _FP16_BACKWARD_ACCUMULATION_LIMIT:
            return False
        for output in op.output_specs:
            try:
                extent = _omitted_extent(output.tokens, represented, case)
            except KeyError:
                continue
            if extent >= _FP16_BACKWARD_ACCUMULATION_LIMIT:
                return False
    return True


def estimated_bytes(op: Any, case: Case) -> int:
    """Device-resident input bytes for a case, from annotations alone, nothing allocated.

    Planners prune cases whose inputs could never leave room to run (gradients, outputs,
    and an upcast ground truth all join the inputs on device) and order the rest cheapest
    first, so a budget frontier meets its boundary before paying for anything beyond it.
    """
    dims = dict(case.dims)
    args = dict(case.args)
    total = 0
    for spec in op.specs:
        if spec.optional and spec.param not in case.present:
            continue
        shape: list[int] = []
        for token in spec.tokens:
            if token is ...:
                shape.extend(case.batch)
            else:
                shape.append(token_size(token, dims, args, spec.param))
        floating = any("float" in dtype for dtype in spec.dtypes)
        total += math.prod(shape) * (case.dtype.itemsize if floating else 8)
    return total


def smoke_cases(
    op: Any,
    limit: int,
    max_bytes: int = 2 * 2**20,
    allowed: Callable[[Case], bool] | None = None,
) -> list[Case]:
    """Deterministic, bounded sample of the shared plan for backend smoke tests."""
    eligible = [
        case
        for case in case_plan(op).flatten()
        if estimated_bytes(op, case) <= max_bytes
        and all(value <= _SMOKE_TEMPORAL_MAX for name, value in case.dims if name in _TEMPORAL_DIMS)
        and (allowed is None or allowed(case))
    ]
    eligible.sort(key=lambda case: (estimated_bytes(op, case), case.case_id))
    return eligible[:limit]


def sample_cases(op: Any, limit: int | None = None) -> list[Case]:
    """Sample the legacy dim × arg × presence × dtype Cartesian product.

    Never builds the full cartesian list when `limit` is set — ranks into the
    product space instead — so wide `DIMS` pools stay cheap. Annotation `...`
    is one leading extent from `DIMS["..."]` (0 → `batch=()`, else `(n,)`).
    """
    if missing := _missing_args(op):
        raise TypeError(f"{op.name}: arguments {missing} need test_args, a default, or a Literal annotation")
    dim_names = _dim_names(op)
    dim_pools = [_dim_pool(op, name) for name in dim_names]
    arg_names = [name for name, pool in sorted(op.arg_pools.items()) if pool is not None]
    arg_pools = [list(op.arg_pools[name]) for name in arg_names]
    opt_names = sorted(spec.param for spec in op.specs if spec.optional)
    dtypes = list(_dtypes(op))
    sizes = [len(pool) for pool in dim_pools] + [len(pool) for pool in arg_pools] + [2] * len(opt_names) + [len(dtypes)]
    if any(size == 0 for size in sizes):
        return []
    total = math.prod(sizes)
    if limit is None and total > _MAX_FULL_GRID:
        raise ValueError(f"{op.name}: full grid has {total} cases; pass limit=")
    rng = random.Random(0)
    indices = range(total) if limit is None or total <= limit else rng.sample(range(total), limit)

    def build(index: int) -> Case:
        coords = _unrank(index, sizes)
        cursor = 0
        values = {name: pool[coords[cursor + offset]] for offset, (name, pool) in enumerate(zip(dim_names, dim_pools))}
        cursor += len(dim_names)
        args = {name: pool[coords[cursor + offset]] for offset, (name, pool) in enumerate(zip(arg_names, arg_pools))}
        cursor += len(arg_names)
        present = frozenset(name for offset, name in enumerate(opt_names) if coords[cursor + offset])
        cursor += len(opt_names)
        dtype = dtypes[coords[cursor]]
        return _build_case(values, dtype, args, present)

    # Repairs can collapse two draws onto one case, so `limit` is a ceiling.
    found: dict[str, Case] = {}
    for index in indices:
        case = build(index)
        found.setdefault(case.case_id, case)
    return list(found.values())


def cases(op: Any, limit: int | None = None) -> list[Case]:
    """Compatibility alias for :func:`sample_cases`."""
    return sample_cases(op, limit)


def _transform(
    op: Any,
    param: str,
    tensor: torch.Tensor,
    dims: Mapping[str, int],
    generator: torch.Generator,
) -> torch.Tensor:
    transform = op.test_inputs.get(param)
    if transform is None:
        return tensor
    return transform(tensor, dims, generator) if param in op._contextual_inputs else transform(tensor)


def make_inputs(
    op: Any,
    case: Case,
    device: torch.device | str = "cuda",
    seed: int = 0,
    grad: bool = True,
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    dims = dict(case.dims)
    inputs: dict[str, Any] = dict(case.args)
    target = torch.device(device)
    budget = torch.cuda.get_device_properties(target).total_memory if target.type == "cuda" else None
    for spec in op.specs:
        if spec.optional and spec.param not in case.present:
            inputs[spec.param] = None
            continue
        shape = []
        for token in spec.tokens:
            if token is ...:
                shape.extend(case.batch)
            else:
                shape.append(token_size(token, dims, inputs, spec.param))
        floating = any("float" in dtype for dtype in spec.dtypes)
        # Tensors are drawn on the host first; an absurd case (e.g. seq=1M x hidden=28K,
        # ~1TB) would get the process OOM-killed there before any exception can surface.
        nbytes = math.prod(shape) * (4 if floating else 8)
        if budget is not None and nbytes > budget:
            raise torch.OutOfMemoryError(
                f"{spec.param} needs {nbytes / 2**30:.0f} GiB, over device capacity {budget / 2**30:.0f} GiB"
            )
        if floating:
            tensor = torch.randn(shape, generator=generator)
        else:
            other_dims = [value for name, value in dims.items() if name not in spec.tokens]
            tensor = torch.randint(0, min(other_dims) if other_dims else 8, shape, generator=generator)
        tensor = _transform(op, spec.param, tensor, dims, generator)
        tensor = tensor.to(case.dtype) if floating else tensor
        tensor = tensor.to(device)
        if floating:
            tensor.requires_grad_(grad)
        inputs[spec.param] = tensor
    return inputs
