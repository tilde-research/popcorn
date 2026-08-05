import multiprocessing
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


def _smoke_worker(connection, op_name, backend, trials, limit):
    """Run one adapter in a fresh CUDA process so a bad kernel cannot poison the suite."""
    try:
        import popcorn.kernels  # noqa: F401
        from popcorn import KERNELS
        from popcorn.bench.grid import make_inputs, smoke_cases
        from popcorn.bench.store import static_allowed

        op = KERNELS[op_name]
        impl = next(candidate for candidate in op._impls if candidate.name == backend)
        candidates = smoke_cases(
            op,
            max(128, limit),
            allowed=lambda case: static_allowed(op, impl, case.dtype, dict(case.args), case.present),
        )
        records = []
        for case in candidates:
            inputs = make_inputs(op, case, device="cpu", grad=not impl.forward_only)
            if impl.rejects({}, inputs):
                continue
            record = op.bench.run_case(backend, case, trials=trials, benchmark=False)
            if record.result.status == "skip":
                continue
            if record.result.status == "fail":
                record = op.bench.run_case(backend, case, trials=10, benchmark=False)
            records.append(record)
            if len(records) == limit:
                break
        connection.send(
            (
                "records",
                [(record.result.status, record.case, record.result.reason, record.result.bench_error) for record in records],
            )
        )
    except BaseException as error:
        connection.send(("error", f"{type(error).__name__}: {error}"))
    finally:
        connection.close()


def _isolated_smoke(op_name, backend, trials, limit):
    timeout = int(os.getenv("POPCORN_TEST_TIMEOUT", "300"))
    receive, send = multiprocessing.get_context("spawn").Pipe(duplex=False)
    process = multiprocessing.get_context("spawn").Process(
        target=_smoke_worker,
        args=(send, op_name, backend, trials, limit),
    )
    process.start()
    send.close()
    process.join(timeout)
    if process.is_alive():
        process.kill()
        process.join()
        pytest.fail(f"smoke worker exceeded {timeout} seconds")
    try:
        payload = receive.recv() if receive.poll() else ("error", f"worker exited {process.exitcode} without a result")
    except EOFError:
        payload = ("error", f"worker exited {process.exitcode} without a result")
    receive.close()
    if payload[0] == "error":
        pytest.fail(payload[1])
    return payload[1]


def test_matrix_backend_is_available():
    """The isolated CI environment contains exactly the backend named by its matrix row."""
    expected = os.getenv("POPCORN_TEST_BACKEND")
    if expected is None:
        pytest.skip("CI matrix contract")

    import popcorn.kernels  # noqa: F401
    from popcorn import KERNELS
    from popcorn.core.sources import _BACKENDS, available, installed_version

    external = {name for name, spec in _BACKENDS.items() if spec.extra is not None}
    installed = {name for name in external if available(name)}
    if expected == "none":
        assert not installed
        return

    allowed = {"fla"} if expected == "popcorn" else {expected}
    assert not installed - allowed, f"unexpected backends installed with {expected}: {sorted(installed - allowed)}"
    registered = {impl.name for op in KERNELS.values() for impl in op._impls if impl.name.split(":", 1)[0] == expected}
    assert registered, f"no implementations registered for {expected}"
    assert available(expected), f"{expected} is registered but its package is unavailable"
    assert installed_version(expected)


@requires_cuda
def test_cudnn_attention_executes_forward_and_backward():
    if os.getenv("POPCORN_TEST_BACKEND") != "cudnn":
        pytest.skip("cuDNN matrix contract")

    from popcorn.bench.model import Case
    from popcorn.kernels import attn

    case = Case(
        (("batch", 2), ("head_dim", 64), ("kv_heads", 2), ("q_heads", 8), ("seq", 64)),
        (),
        torch.bfloat16,
        (("causal", True), ("softmax_scale", None)),
        frozenset(),
    )
    record = attn.bench.run_case("cudnn", case, device="cuda", trials=2, grad=True, benchmark=False)
    assert record.result.status == "pass", record.result.reason


@requires_cuda
def test_transformer_engine_softmax_executes_forward_and_backward():
    if os.getenv("POPCORN_TEST_BACKEND") != "transformer_engine":
        pytest.skip("Transformer Engine matrix contract")

    from popcorn.bench.model import Case
    from popcorn.kernels import softmax

    case = Case((("hidden", 128),), (4, 8), torch.bfloat16, (), frozenset())
    record = softmax.bench.run_case("transformer_engine", case, device="cuda", trials=2, grad=True, benchmark=False)
    assert record.result.status == "pass", record.result.reason


def backend_params():
    """Every declared pair, with the ones whose library is absent marked skipped.

    This tier runs once per installed backend, so most pairs are unavailable in any given
    run. The harness records an unavailable backend as `skip`, which would pass the
    assertion below and report a green test that executed nothing — the count has to say
    which backend the job actually covered.
    """
    import popcorn.kernels  # noqa: F401
    from popcorn import KERNELS
    from popcorn.core.sources import unavailable_reason

    params = []
    expected = os.getenv("POPCORN_TEST_BACKEND")
    for op in KERNELS.values():
        for backend in op.available_backends():
            if backend == "torch":
                continue
            if expected and expected != "none" and backend.split(":", 1)[0] != expected:
                continue
            name = f"{op.name}-{backend}"
            if reason := unavailable_reason(backend):
                marks = [pytest.mark.skip(reason=reason)]
            elif name in KNOWN_ISSUES:
                marks = [pytest.mark.xfail(reason=KNOWN_ISSUES[name], strict=False)]
            else:
                marks = []
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
    trials = int(os.getenv("POPCORN_TEST_TRIALS", "2"))
    limit = int(os.getenv("POPCORN_TEST_LIMIT", "1"))
    records = _isolated_smoke(op.name, backend, trials, limit)
    if not records and os.getenv("POPCORN_TEST_BACKEND") in (None, "none"):
        pytest.skip("optional adapter dependencies unavailable")
    bad = [record for record in records if record[0] in ("fail", "crash", "error") or record[3]]
    assert not bad, "\n".join(
        f"[{status}] {case}: {reason or 'benchmark: ' + bench_error}" for status, case, reason, bench_error in bad
    )
    assert any(status == "pass" for status, *_ in records), "\n".join(
        f"[{status}] {case}: {reason or 'benchmark: ' + bench_error}" for status, case, reason, bench_error in records
    )
