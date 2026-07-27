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
BATCHES = ((), (2, 3), (2, 2048))
DTYPES = (torch.float32, torch.float16, torch.bfloat16)

# Refuse to materialize a full cartesian product past this; pass `limit=`.
_MAX_FULL_GRID = 100_000


def _dim_pool(_op: Any, name: str) -> list[int]:
    """Pool for `name` from the global `DIMS` vocabulary (unknown → `GRID`)."""
    spec = DIMS.get(name)
    if spec is None:
        return list(GRID)
    return space(spec).grid(seeds=GRID)


def _unrank(index: int, sizes: Sequence[int]) -> list[int]:
    coords = []
    for size in reversed(sizes):
        coords.append(index % size)
        index //= size
    coords.reverse()
    return coords


def cases(op: Any, limit: int | None = None) -> list[Case]:
    """Sample the dim × arg × presence × dtype × batch product.

    Never builds the full cartesian list when `limit` is set — ranks into the
    product space instead — so wide `DIMS` pools stay cheap.
    """
    if missing := [name for name, pool in op.arg_pools.items() if pool is None]:
        raise TypeError(f"{op.name}: arguments {missing} need test_args, a default, or a Literal annotation")
    dim_names = sorted(op._dims)
    dim_pools = [_dim_pool(op, name) for name in dim_names]
    arg_names = [name for name, pool in sorted(op.arg_pools.items()) if pool is not None]
    arg_pools = [list(op.arg_pools[name]) for name in arg_names]
    opt_names = sorted(spec.param for spec in op.specs if spec.optional)
    dtypes = list(DTYPES if any("float" in dtype for spec in op.specs for dtype in spec.dtypes) else (torch.float32,))
    batches = list(BATCHES if any(... in spec.tokens for spec in op.specs) else ((),))
    sizes = [len(pool) for pool in dim_pools] + [len(pool) for pool in arg_pools] + [2] * len(opt_names) + [len(dtypes), len(batches)]
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
        dims = [(name, pool[coords[cursor + offset]]) for offset, (name, pool) in enumerate(zip(dim_names, dim_pools))]
        cursor += len(dim_names)
        args = [(name, pool[coords[cursor + offset]]) for offset, (name, pool) in enumerate(zip(arg_names, arg_pools))]
        cursor += len(arg_names)
        present = frozenset(name for offset, name in enumerate(opt_names) if coords[cursor + offset])
        cursor += len(opt_names)
        dtype = dtypes[coords[cursor]]
        batch = batches[coords[cursor + 1]]
        return Case(tuple(dims), batch, dtype, tuple(args), present)

    return [build(index) for index in indices]


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
