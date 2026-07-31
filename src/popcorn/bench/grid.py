"""Test-case grids and randomized inputs derived from kernel signatures."""

import math
import random
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from popcorn.bench.model import Case
from popcorn.core.annotations import token_size
from popcorn.core.dims import DIMS
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


def _dim_pool(_op: Any, name: str) -> list[int]:
    """Pool for `name` from the global `DIMS` vocabulary (unknown → `GRID`)."""
    spec = DIMS.get(name)
    if spec is None:
        return list(GRID)
    return space(spec).grid(seeds=GRID)


def _regroup(values: dict[str, Any]) -> None:
    """Snap `kv_heads` onto a divisor of `q_heads`, in place.

    Grouped-query attention folds `q_heads // kv_heads` query heads onto each kv head, so
    counts that do not divide describe no attention layer an op can express: the reference
    builds an empty tensor and every implementation raises. The two pools are declared
    independently and over half of their pairings are that shape, so this would otherwise
    be the single largest source of wasted cases. Repairing the row rather than dropping
    it keeps whatever the row was covering on every other axis.
    """
    q, kv = values.get("q_heads"), values.get("kv_heads")
    if q is None or kv is None or (kv <= q and q % kv == 0):
        return
    values["kv_heads"] = max(group for group in range(1, min(q, kv) + 1) if q % group == 0)


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


def covering_cases(op: Any) -> list[Case]:
    """Pairwise-cover every declared dim and case axis, smallest shapes first.

    Full products reach hundreds of millions of cases. A covering array keeps
    every value and every two-axis interaction while leaving higher-order
    combinations to adaptive follow-up. Ordering by dimension rank lets an
    online runner discover an OOM boundary before reaching dominated shapes.
    """
    if missing := [name for name, pool in op.arg_pools.items() if pool is None]:
        raise TypeError(f"{op.name}: arguments {missing} need test_args, a default, or a Literal annotation")
    has_ellipsis = any(... in spec.tokens for spec in op.specs)
    dim_names = sorted(op._dims | ({ELLIPSIS} if has_ellipsis else set()))
    dim_pools = [_dim_pool(op, name) for name in dim_names]
    arg_names = [name for name, pool in sorted(op.arg_pools.items()) if pool is not None]
    arg_pools = [list(op.arg_pools[name]) for name in arg_names]
    opt_names = sorted({spec.param for spec in op.specs if spec.optional})
    dtypes = list(DTYPES if any("float" in dtype for spec in op.specs for dtype in spec.dtypes) else (torch.float32,))
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
        _regroup(values)
        leading = values.pop(ELLIPSIS, None)
        batch = () if leading is None or leading == 0 else (leading,)
        dtype = dtypes[row[cursor]]
        cursor += 1
        args = tuple((name, pool[row[cursor + offset]]) for offset, (name, pool) in enumerate(zip(arg_names, arg_pools)))
        cursor += len(arg_names)
        present = frozenset(name for offset, name in enumerate(opt_names) if row[cursor + offset])
        case = Case(tuple(sorted(values.items())), batch, dtype, args, present)
        if case.case_id not in seen:
            seen.add(case.case_id)
            found.append(case)
    return found


def cases(op: Any, limit: int | None = None) -> list[Case]:
    """Sample the dim (incl. `...` when present) × arg × presence × dtype product.

    Never builds the full cartesian list when `limit` is set — ranks into the
    product space instead — so wide `DIMS` pools stay cheap. Annotation `...`
    is one leading extent from `DIMS["..."]` (0 → `batch=()`, else `(n,)`).
    """
    if missing := [name for name, pool in op.arg_pools.items() if pool is None]:
        raise TypeError(f"{op.name}: arguments {missing} need test_args, a default, or a Literal annotation")
    has_ellipsis = any(... in spec.tokens for spec in op.specs)
    dim_names = sorted(op._dims | ({ELLIPSIS} if has_ellipsis else set()))
    dim_pools = [_dim_pool(op, name) for name in dim_names]
    arg_names = [name for name, pool in sorted(op.arg_pools.items()) if pool is not None]
    arg_pools = [list(op.arg_pools[name]) for name in arg_names]
    opt_names = sorted(spec.param for spec in op.specs if spec.optional)
    dtypes = list(DTYPES if any("float" in dtype for spec in op.specs for dtype in spec.dtypes) else (torch.float32,))
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
        _regroup(values)
        leading = values.pop(ELLIPSIS, None)
        batch = () if leading is None or leading == 0 else (leading,)
        dims = [(name, values[name]) for name in sorted(values)]
        args = [(name, pool[coords[cursor + offset]]) for offset, (name, pool) in enumerate(zip(arg_names, arg_pools))]
        cursor += len(arg_names)
        present = frozenset(name for offset, name in enumerate(opt_names) if coords[cursor + offset])
        cursor += len(opt_names)
        dtype = dtypes[coords[cursor]]
        return Case(tuple(dims), batch, dtype, tuple(args), present)

    # Repairing head counts can collapse two draws onto one case, so `limit` is a ceiling.
    found: dict[str, Case] = {}
    for index in indices:
        case = build(index)
        found.setdefault(case.case_id, case)
    return list(found.values())


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
