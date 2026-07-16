"""Test-case grids and randomized inputs derived from kernel signatures."""

import itertools
import random
from collections.abc import Mapping
from typing import Any

import torch

from popcorn.bench.model import Case
from popcorn.core.constraints import Range, satisfies

# `op` stays `Any` here: the layering test forbids this module from importing
# the dispatcher, even for annotations.

GRID = (1, 2, 3, 8, 16, 33, 64, 128, 1024, 4096)
BATCHES = ((), (2, 3), (2, 2048))
DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _dim_pool(spec: Any) -> list[int]:
    candidates = set(GRID)
    for atom in spec if isinstance(spec, tuple) else (spec,):
        match atom:
            case Range(lo, hi):
                candidates |= {lo, hi}
            case set() | frozenset():
                candidates = set(atom)
    pool = sorted(value for value in candidates if spec is None or satisfies(spec, value))
    if not pool:
        raise ValueError(f"test_shapes spec {spec} admits no sizes")
    return pool


def cases(op: Any, limit: int | None = None) -> list[Case]:
    if missing := [name for name, pool in op.arg_pools.items() if pool is None]:
        raise TypeError(f"{op.name}: arguments {missing} need test_args, a default, or a Literal annotation")
    dim_axes = [[(name, value) for value in _dim_pool(op.test_shapes.get(name))] for name in sorted(op._dims)]
    arg_axes = [[(name, value) for value in pool] for name, pool in sorted(op.arg_pools.items()) if pool is not None]
    presence_axes = [[(name, False), (name, True)] for name in sorted(spec.param for spec in op.specs if spec.optional)]
    dtypes = DTYPES if any("float" in dtype for spec in op.specs for dtype in spec.dtypes) else (torch.float32,)
    batches = BATCHES if any(... in spec.tokens for spec in op.specs) else ((),)
    product = itertools.product(
        itertools.product(*dim_axes),
        itertools.product(*arg_axes),
        itertools.product(*presence_axes),
        dtypes,
        batches,
    )
    grid = [
        Case(dims, batch, dtype, args, frozenset(name for name, enabled in presence if enabled))
        for dims, args, presence, dtype, batch in product
    ]
    return random.Random(0).sample(grid, limit) if limit and len(grid) > limit else grid


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
    for spec in op.specs:
        if spec.optional and spec.param not in case.present:
            inputs[spec.param] = None
            continue
        shape = []
        for token in spec.tokens:
            if token is ...:
                shape.extend(case.batch)
            else:
                shape.append(token if isinstance(token, int) else dims[token])
        floating = any("float" in dtype for dtype in spec.dtypes)
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
