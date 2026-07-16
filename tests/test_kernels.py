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


@requires_cuda
@pytest.mark.parametrize("op,backend", backend_params())
def test_backend_matches_reference(op, backend):
    """Smoke slice of the grid; the full 10-rep benchmark grid runs via `python -m popcorn.bench`."""
    from popcorn.bench.grid import cases

    trials = int(os.getenv("POPCORN_TEST_TRIALS", "2"))
    limit = int(os.getenv("POPCORN_TEST_LIMIT", "24"))
    records = op.bench.run_backend(backend, trials=trials, limit=limit)
    grid = {case.case_id: case for case in cases(op)}
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
