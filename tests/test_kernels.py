import os

import pytest
import torch

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

# Documented in ISSUES.md; these cannot pass an exactness harness.
KNOWN_ISSUES = {
    "mesa_net-fla": "approximate CG solver",
    "bit_linear-fla": "int8 quantization boundary chatter",
    "layer_norm_linear_quant-fla": "int8 quantization boundary chatter",
    "rms_norm_linear_quant-fla": "int8 quantization boundary chatter",
    "tvd-liger": "subgradient tie-breaking at p == q",
}


def backend_params():
    import popcorn.kernels  # noqa: F401
    from popcorn import KERNELS

    params = []
    for op in KERNELS.values():
        for backend in op.available_backends():
            if backend == "torch":
                continue
            name = f"{op.name}-{backend}"
            marks = [pytest.mark.xfail(reason=KNOWN_ISSUES[name], strict=False)] if name in KNOWN_ISSUES else []
            params.append(pytest.param(op, backend, id=name, marks=marks))
    shard = os.getenv("POPCORN_TEST_SHARD")
    if shard is None:
        return params
    index, count = map(int, shard.split("/"))
    return params[index::count]


def test_dim_names_are_canonical():
    """Every dim name is vocabulary and every entry is used: new and retired
    names both land in core/dims.py, one reviewable place."""
    import popcorn.kernels  # noqa: F401
    from popcorn import KERNELS
    from popcorn.core.dims import DIMS

    used = set().union(*(op._dims for op in KERNELS.values()))
    if any(... in spec.tokens for op in KERNELS.values() for spec in op.specs):
        used.add("...")
    assert used - DIMS.keys() == set(), f"undeclared dim names: {sorted(used - DIMS.keys())}"
    assert DIMS.keys() - used == set(), f"unused vocabulary entries: {sorted(DIMS.keys() - used)}"


def test_scalar_args_are_classified():
    """Every kernel scalar is SPEED (timing) or NEUTRAL (ignored for timing)."""
    import popcorn.kernels  # noqa: F401
    from popcorn import KERNELS
    from popcorn.core.args import KNOWN_ARGS, NEUTRAL_ARGS, SPEED_ARGS

    used = set().union(*(op.arg_pools for op in KERNELS.values()))
    assert used - KNOWN_ARGS == set(), f"unclassified scalar args: {sorted(used - KNOWN_ARGS)}"
    assert SPEED_ARGS - used == set(), f"unused SPEED_ARGS: {sorted(SPEED_ARGS - used)}"
    assert NEUTRAL_ARGS - used == set(), f"unused NEUTRAL_ARGS: {sorted(NEUTRAL_ARGS - used)}"


def test_adapters_are_import_free():
    """Adapters map arguments only; library code arrives through `source=`, so
    it stays lazy and fingerprinted. Imports inside a body dodge both."""
    import dis
    from types import CodeType

    import popcorn.kernels  # noqa: F401
    from popcorn import KERNELS

    def imports(code):
        if any(instruction.opname in ("IMPORT_NAME", "IMPORT_FROM") for instruction in dis.get_instructions(code)):
            return True
        return any(imports(const) for const in code.co_consts if isinstance(const, CodeType))

    offenders = [
        f"{op.name}:{impl.name}"
        for op in KERNELS.values()
        for impl in op._impls
        if impl.adapter is not None and imports(impl.adapter.__code__)
    ]
    assert not offenders, f"function-body imports in adapters: {offenders}"


@requires_cuda
@pytest.mark.parametrize("op,backend", backend_params())
def test_backend_matches_reference(op, backend):
    """Smoke slice of the grid; the full 10-rep benchmark grid runs via `python -m popcorn.bench`."""
    from popcorn.bench.grid import cases

    trials = int(os.getenv("POPCORN_TEST_TRIALS", "2"))
    limit = int(os.getenv("POPCORN_TEST_LIMIT", "24"))
    records = op.bench.run_backend(backend, trials=trials, limit=limit)
    grid = {case.case_id: case for case in cases(op, limit=limit)}
    records = [
        op.bench.run_case(backend, grid[record.case_id], trials=10, benchmark=False)
        if record.result.status == "fail"
        else record
        for record in records
    ]
    bad = [record for record in records if record.result.status in ("fail", "crash", "error") or record.result.bench_error]
    assert not bad, "\n".join(
        f"[{record.result.status}] {record.case}: {record.result.reason or 'benchmark: ' + record.result.bench_error}"
        for record in bad
    )
