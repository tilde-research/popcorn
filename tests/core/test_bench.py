import ast
import importlib
import json
import math
import subprocess
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch
from jaxtyping import Float
from torch import Tensor

import popcorn
from popcorn import KERNELS
from popcorn.bench import Case, Gauge, compare, hub, store
from popcorn.bench.__main__ import (
    _fill_pending,
    _impl,
    _isolated,
    _record_index,
    _require_clean,
    _slice_work,
    _target_device,
    _usable_records,
    _work,
    cmd_fill,
    cmd_merge,
    cmd_publish,
    cmd_run,
    cmd_submit,
)
from popcorn.bench.grid import ELLIPSIS, _dim_pool, case_plan, cases, covering_cases, make_inputs, sample_cases
from popcorn.bench.model import Environment, Record, Result
from popcorn.bench.readme import badges, matrix
from popcorn.bench.service import INCOMPLETE
from popcorn.bench.store import PUBLISHED, Store, bundled_path, read, read_file, unmeasured, write
from popcorn.bench.viewer import render
from popcorn.core import Dispatcher
from popcorn.core.config import call_config, config_id, device_name, make_config
from popcorn.core.sources import installed_version
from popcorn.core.tuning import Tuner
from popcorn.kernels._utils import default_scale, rms, upcast

comparison = importlib.import_module("popcorn.bench.compare")


def _record(status="pass", case_id="case", ts="2026-01-01T00:00:00+00:00"):
    result = Result(status=status, grad=False)
    environment = Environment("cpu", torch.__version__, None, ts)
    config = {"dims": {"D": 4}, "batch": [], "dtype": "float32", "args": {}, "present": []}
    return Record("op", "fast", "D=4", case_id, config, environment, result)


def _op():
    def reference(x: Float[Tensor, "D"]):
        return x + 1

    op = Dispatcher(reference)
    op.register("alt")(lambda x: x + 1)
    return op


def test_legacy_missing_hash_does_not_validate_hashable_code():
    assert store.matching(None, None)
    assert store.matching(None, "recorded")
    assert store.matching("current", "current")
    assert not store.matching("current", None)
    assert not store.matching("current", "stale")


def test_readme_counts_reference_as_an_implementation():
    row = badges([], {"op": _op()})
    assert "implementations-2-blue" in row
    assert "backends-1-blue" in row


def test_compare_is_callable_first():
    result = compare(
        lambda x, scale: x * scale,
        lambda x, scale: x * scale,
        {"x": torch.randn(8), "scale": 2},
        benchmark=False,
        repeats=2,
    )
    assert result.status == "pass"
    assert result.reps == 2


def test_cached_reference_timing_skips_only_the_benchmark_leg(monkeypatch):
    reference_calls = 0
    measured = []

    def reference(x):
        nonlocal reference_calls
        reference_calls += 1
        return x + 1

    def measure(fn, device, repeats, warmup):
        measured.append(fn)
        fn()
        return {"ms": 2.0, "mem_mb": 3.0}

    monkeypatch.setattr(comparison, "_measure", measure)
    result = comparison.compare_inputs(
        lambda x: x + 1,
        reference,
        [{"x": torch.ones(4)}],
        backward=False,
        repeats=1,
        warmup=0,
        reference_bench={"ref_fwd_ms": 7.0, "ref_fwd_mem_mb": 11.0},
    )

    assert result.status == "pass"
    assert reference_calls == 2  # float64 truth and native-dtype budget still run
    assert len(measured) == 1  # only implementation timing is repeated
    assert result.bench["fwd_ms"] == 2.0
    assert result.bench["ref_fwd_ms"] == 7.0
    assert result.bench["ref_fwd_mem_mb"] == 11.0


def test_reference_subject_is_timed_once(monkeypatch):
    measured = []

    def measure(fn, device, repeats, warmup):
        measured.append(fn)
        fn()
        return {"ms": 2.0}

    monkeypatch.setattr(comparison, "_measure", measure)
    result = comparison.compare_inputs(
        lambda x: x,
        lambda x: x,
        [{"x": torch.ones(1)}],
        backward=False,
        repeats=1,
        reference_bench={},
    )
    assert len(measured) == 1
    assert result.bench["fwd_ms"] == result.bench["ref_fwd_ms"] == 2.0


def test_store_returns_only_fresh_exact_torch_timing(tmp_path):
    config = {"dims": {"D": 4}, "batch": [], "dtype": "float32", "args": {}, "present": []}
    result = Result(
        status="pass",
        grad=False,
        benchmarked=True,
        bench={"fwd_ms": 1.25, "fwd_q20_ms": 1.0},
    )
    row = Record(
        "op",
        "torch",
        "D=4",
        "case",
        config,
        Environment("cpu", torch.__version__, None, "2026-01-01T00:00:00+00:00", "reference-hash", "reference-hash"),
        result,
    )
    cache = Store(bundled=tmp_path / "reports", user=tmp_path / "reports")
    cache.write_user([row])

    assert cache.reference_bench("op", "case", config, "cpu", torch.__version__, False, "reference-hash") == {
        "ref_fwd_ms": 1.25,
        "ref_fwd_q20_ms": 1.0,
    }
    assert cache.reference_bench("op", "case", config, "cpu", torch.__version__, False, "stale-hash") is None
    assert cache.reference_bench("op", "case", config, "other-device", torch.__version__, False, "reference-hash") is None


def test_compare_grades_forward_and_backward():
    x = torch.randn(8)
    result = compare(lambda x: x.square(), lambda x: x.square(), {"x": x}, benchmark=False, repeats=2)
    assert result.status == "pass"
    assert result.fwd and result.bwd

    failed = compare(lambda x: x + 1, lambda x: x, {"x": x}, benchmark=False, repeats=1)
    assert failed.status == "fail" and "out0" in failed.reason

    precise = torch.ones(4, dtype=torch.float64)
    failed = compare(lambda x: x + 1e-8, lambda x: x, {"x": precise}, benchmark=False, repeats=1)
    assert failed.status == "fail"


def test_compare_separates_kernel_and_reference_errors():
    def broken(x):
        raise RuntimeError("broken")

    mine = compare(broken, lambda x: x, {"x": torch.ones(1)}, benchmark=False, repeats=1)
    reference = compare(lambda x: x, broken, {"x": torch.ones(1)}, benchmark=False, repeats=1)
    assert mine.status == "crash" and mine.reason == "RuntimeError: broken"
    assert reference.status == "error" and reference.reason == "reference: RuntimeError: broken"


def test_correctness_precedes_timing(monkeypatch):
    called = False

    def timer(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(comparison, "_benchmark", timer)
    result = compare(lambda x: x + 1, lambda x: x, {"x": torch.ones(1)}, repeats=1)
    assert result.status == "fail" and not called


def test_benchmark_error_preserves_correctness(monkeypatch):
    def timer(*args, **kwargs):
        raise RuntimeError("timer failed")

    monkeypatch.setattr(comparison, "_benchmark", timer)
    result = compare(lambda x: x, lambda x: x, {"x": torch.ones(1)}, repeats=1)
    assert result.status == "pass"
    assert result.bench_error == "RuntimeError: timer failed"
    assert not result.benchmarked


def test_report_formats_bench_and_error(capsys):
    from popcorn.bench import report
    from popcorn.bench.model import Gauge, Result

    result = Result(
        status="pass",
        benchmarked=True,
        bench={"fwd_ms": 1.0, "ref_fwd_ms": 4.0, "fwd_mem_mb": 512.0, "ref_fwd_mem_mb": 1024.0},
        fwd={"out0": Gauge(err=0.0, scale=1.0)},
    )
    text = report(result, "forward", mine="popcorn", reference="baseline", assert_rel=0.02)
    assert "4.0 ms" in text and "1.0 ms" in text and "(4.0x)" in text
    assert "0.50 GB" in text and "1.00 GB" in text
    assert "err 0.00e+00" in text
    assert "forward" in capsys.readouterr().out


def test_non_finite_errors_fail():
    assert comparison._error(torch.tensor([float("nan")]), torch.tensor([float("nan")])) == 0
    assert math.isinf(comparison._error(torch.tensor([float("nan")]), torch.tensor([1.0])))
    assert Gauge(err=float("inf")).verdict(1e-3) == "err is non-finite"


def test_reference_precision_helpers():
    x = torch.ones(2, dtype=torch.float64)
    assert upcast(x).dtype == torch.float64
    assert upcast(x.half()).dtype == torch.float32
    assert rms(x, 1e-6).dtype == torch.float64
    assert default_scale(0.0, 64) == 0.0


def test_grid_is_deterministic_and_uses_canonical_config(monkeypatch):
    from popcorn.core import dims as dims_mod

    monkeypatch.setitem(dims_mod.DIMS, "D", {4})
    op = _op()
    grid = cases(op)
    assert len(grid) == 3  # D={4} × 3 dtypes × batch=()
    assert [case.case_id for case in grid] == [case.case_id for case in cases(op)]
    case = grid[0]
    inputs = make_inputs(op, case, "cpu")
    call = call_config(op, op._values(inputs), inputs)
    assert call.config == case.config()
    assert config_id(call.config) == case.case_id


def test_covering_grid_uses_every_canonical_dim_value():
    for op in KERNELS.values():
        grid = covering_cases(op)
        # Repaired axes deviate from their declared pool (see bench.grid._repair): snapped
        # axes (`kv_heads`, `half`) keep every declared value reachable somewhere, while
        # derived axes are recomputed from the dimensions or scalar arguments they follow.
        snapped = set()
        if {"q_heads", "kv_heads"} <= op._dims:
            snapped.add("kv_heads")
        if {"half", "head_dim"} <= op._dims:
            snapped.add("half")
        if {"boundaries", "total"} <= op._dims:
            snapped.add("boundaries")
        if {"kernel_size", "seq"} <= op._dims:
            snapped.add("kernel_size")
        derived: dict[str, set[int]] = {}
        if {"levels", "seq"} <= op._dims:
            derived["levels"] = {math.ceil(math.log2(seq)) + 1 if seq > 1 else 1 for seq in _dim_pool(op, "seq")}
        if {"cos_batch", "batch"} <= op._dims:
            derived["cos_batch"] = {1} | (set(_dim_pool(op, "cos_batch")) & set(_dim_pool(op, "batch")))
        if {"gate_dim", "heads", "key_dim"} <= op._dims:
            products = {h * k for h in _dim_pool(op, "heads") for k in _dim_pool(op, "key_dim")}
            derived["gate_dim"] = products
        if {"block_count", "block_size"} <= op.arg_pools.keys() and "seq" in op._dims:
            derived["seq"] = {
                min(seq, count * size)
                for seq in _dim_pool(op, "seq")
                for count in op.arg_pools["block_count"] or ()
                for size in op.arg_pools["block_size"] or ()
            }
        for name in op._dims:
            seen, pool = {dict(case.dims)[name] for case in grid}, set(_dim_pool(op, name))
            if name in snapped:
                assert pool <= seen and all(value <= max(pool) for value in seen), f"{op.name}:{name}"
            elif name in derived:
                assert seen <= derived[name], f"{op.name}:{name}"
            else:
                assert seen == pool, f"{op.name}:{name}"
        if any(... in spec.tokens for spec in op.specs):
            leading = {case.batch[0] if case.batch else 0 for case in grid}
            assert leading == set(_dim_pool(op, ELLIPSIS)), f"{op.name}:{ELLIPSIS}"


def test_config_values_are_json_stable():
    config = make_config({"D": 4}, (), torch.float32, {"sections": (1, {3, 2})}, ())
    assert config["args"] == {"sections": [1, [2, 3]]}


def test_registered_service_uses_compare_without_persistence():
    op = _op()
    case = Case((("D", 4),), (), torch.float32, (), frozenset())
    record = op.bench.run_case("alt", case, device="cpu", trials=1, benchmark=False, grad=False)
    assert record.result.status == "pass"
    assert not record.result.benchmarked
    assert record.case_id == case.case_id


def test_service_stamps_code_fingerprints():
    op = _op()
    case = Case((("D", 4),), (), torch.float32, (), frozenset())
    record = op.bench.run_case("alt", case, device="cpu", trials=1, benchmark=False, grad=False)
    assert record.environment.ref_hash == op.fingerprint is not None
    assert record.environment.impl_hash == op["alt"].fingerprint is not None
    revived = Record.from_dict(record.to_dict())
    assert revived.environment.ref_hash == record.environment.ref_hash
    assert revived.environment.impl_hash == record.environment.impl_hash


def test_forward_only_backends_get_forward_grid_coverage():
    op = _op()
    op.register("fwd", forward_only=True)(lambda x: x + 1)
    record = op.bench.run_case("fwd", cases(op)[0], device="cpu", trials=1, benchmark=False)
    assert record.result.status == "pass"
    assert record.result.grad is False


def test_input_allocation_oom_is_recorded_as_oom(monkeypatch):
    op = _op()

    def out_of_memory(*args, **kwargs):
        raise torch.OutOfMemoryError("test capacity")

    monkeypatch.setattr("popcorn.bench.service.make_inputs", out_of_memory)
    record = op.bench.run_case("alt", cases(op, 1)[0], device="cpu", trials=1, benchmark=False)
    assert record.result.status == "oom"
    assert record.result.reason == "inputs: test capacity"


class _OOMBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        raise torch.OutOfMemoryError("test capacity")


def test_backward_oom_is_recorded_as_oom_not_error():
    oom_on_backward = lambda x: _OOMBackward.apply(x)  # noqa: E731 - `apply` takes no keywords
    result = comparison.compare_inputs(
        oom_on_backward,
        oom_on_backward,
        [{"x": torch.ones(4)}],
        backward=True,
        benchmark=False,
        repeats=1,
        warmup=0,
    )
    assert result.status == "oom"


def test_comparison_oom_outside_either_side_is_recorded_as_oom(monkeypatch):
    """The float64 ground truth allocates too, and a shape that does not fit still prunes."""
    op = _op()

    def out_of_memory(*args, **kwargs):
        raise torch.OutOfMemoryError("test capacity")

    monkeypatch.setattr("popcorn.bench.service.compare_inputs", out_of_memory)
    record = op.bench.run_case("alt", cases(op, 1)[0], device="cpu", trials=1, benchmark=False)
    assert record.result.status == "oom"
    assert record.result.reason == "harness: test capacity"


def test_run_case_reuses_torch_timing_from_the_shared_store(monkeypatch, tmp_path):
    op = _op()
    case = cases(op, 1)[0]
    reference = Record(
        op.name,
        "torch",
        str(case),
        case.case_id,
        case.config(),
        Environment(
            device_name("cpu"),
            torch.__version__,
            None,
            "2026-01-01T00:00:00+00:00",
            op.fingerprint,
            op["torch"].fingerprint,
        ),
        Result(status="pass", grad=True, benchmarked=True, bench={"fwd_ms": 4.0}),
    )
    cache = Store(bundled=tmp_path / "reports", user=tmp_path / "reports")
    cache.write_user([reference])
    op.bench.store = cache
    seen = {}

    def compare_inputs(*args, **kwargs):
        seen.update(kwargs)
        return Result(status="pass", grad=True)

    monkeypatch.setattr("popcorn.bench.service.compare_inputs", compare_inputs)
    op.bench.run_case("alt", case, device="cpu", trials=1)
    assert seen["reference_bench"] == {"ref_fwd_ms": 4.0}


def test_torch_reference_rows_do_not_become_tuner_samples(tmp_path):
    op = _op()
    case = cases(op, 1)[0]
    result = Result(status="pass", grad=True, benchmarked=True, bench={"fwd_ms": 1.0, "ref_fwd_ms": 2.0})
    records = [
        Record(
            op.name,
            impl,
            str(case),
            case.case_id,
            case.config(),
            Environment(
                device_name("cpu"),
                torch.__version__,
                None,
                "2026-01-01T00:00:00+00:00",
                op.fingerprint,
                op[impl].fingerprint,
            ),
            result,
        )
        for impl in ("torch", "alt")
    ]
    reports = tmp_path / "reports"
    cache = Store(bundled=reports, user=reports)
    cache.write_user(records)

    assert [sample.impl for sample in Tuner(op, cache).samples] == ["alt"]


def test_non_floating_outputs_are_value_checked():
    x = torch.arange(4)
    passed = compare(lambda x: x.clone(), lambda x: x, {"x": x}, benchmark=False, repeats=1, backward=False)
    failed = compare(lambda x: x + 1, lambda x: x, {"x": x}, benchmark=False, repeats=1, backward=False)
    assert passed.status == "pass"
    assert failed.status == "fail" and "out0" in failed.reason


def test_compare_leaves_global_rng_alone():
    x = torch.randn(3, requires_grad=True)
    torch.manual_seed(7)
    expected = torch.randn(4)
    torch.manual_seed(7)
    compare(lambda x: x.square(), lambda x: x.square(), {"x": x}, benchmark=False, repeats=2)
    assert torch.equal(torch.randn(4), expected)


def test_store_upserts_and_preserves_conclusive_rows(tmp_path):
    write([_record()], tmp_path)
    write([_record("fail")], tmp_path)
    assert read(tmp_path)[0].result.status == "fail"

    failure = _record("fail", ts="2026-01-01T00:00:00+00:00")
    harness = _record("error", ts="2026-01-02T00:00:00+00:00")
    write([failure], tmp_path)
    write([harness], tmp_path)
    assert read(tmp_path)[0].result.status == "fail"


def test_store_rejects_unversioned_rows(tmp_path):
    path = tmp_path / "op.jsonl"
    path.write_text('{"op": "op"}\n')
    with pytest.raises(ValueError, match="unsupported report schema"):
        read(tmp_path)


def test_concurrent_store_upserts_are_atomic(tmp_path):
    records = [_record(case_id=str(index)) for index in range(16)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda record: write([record], tmp_path), records))
    assert sorted((record.case_id for record in read(tmp_path)), key=int) == [str(index) for index in range(16)]
    assert not list(tmp_path.glob("*.tmp"))


def test_published_rows_round_trip_through_parquet(tmp_path):
    records = [_record(), _record("fail", "other")]
    (tmp_path / store.FETCHED).write_text("old-pin\n")
    write(records, tmp_path, PUBLISHED)
    assert not (tmp_path / store.FETCHED).exists()
    assert (tmp_path / f"op{PUBLISHED}").exists()
    assert [record.to_dict() for record in read(tmp_path)] == [record.to_dict() for record in records]


def test_local_published_edits_keep_the_tracked_revision_marker(tmp_path):
    (tmp_path / "REVISION").write_text("base-pin\n")
    incomplete = _record("error", "incomplete")
    incomplete.result.reason = INCOMPLETE
    write([_record(), incomplete], tmp_path, PUBLISHED)

    assert (tmp_path / store.FETCHED).read_text() == "base-pin\n"
    assert store.scrub(tmp_path, lambda record: record.result.reason == INCOMPLETE) == 1
    assert (tmp_path / store.FETCHED).read_text() == "base-pin\n"


def test_parquet_keeps_columns_typed_when_a_field_is_always_absent(tmp_path):
    """backend_version is None for every first-party row; it must still read back as text."""
    write([_record()], tmp_path, PUBLISHED)
    assert read(tmp_path)[0].environment.backend_version is None


def test_published_parquet_is_preferred_over_scratch_jsonl(tmp_path):
    write([_record("fail")], tmp_path, PUBLISHED)
    write([_record("pass")], tmp_path)
    assert bundled_path(tmp_path, "op").suffix == PUBLISHED
    assert read_file(bundled_path(tmp_path, "op"))[0].result.status == "fail"


def test_store_says_so_when_it_holds_no_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "_warned", False)
    with pytest.warns(RuntimeWarning, match="torch reference"):
        assert Store(bundled=tmp_path, user=tmp_path / "cache").merged("op") == ()


def test_pull_copies_the_pinned_revision_into_the_reports_directory(tmp_path, monkeypatch):
    published = tmp_path / "snapshot"
    published.mkdir()
    write([_record()], published, PUBLISHED)
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / hub.REVISION).write_text("deadbeef\n")
    (reports / "stale.parquet").write_bytes(b"old")
    asked = {}

    def snapshot(repo_id, repo_type, revision, allow_patterns):
        asked.update(repo_id=repo_id, repo_type=repo_type, revision=revision)
        return str(published)

    monkeypatch.setattr(hub, "snapshot_download", snapshot)
    assert hub.pull(directory=reports) == (1, "deadbeef")
    assert asked == {"repo_id": hub.REPO, "repo_type": "dataset", "revision": "deadbeef"}
    assert not (reports / "stale.parquet").exists()
    assert (reports / hub.FETCHED).read_text().strip() == "deadbeef"
    assert read(reports)[0].result.status == "pass"


def test_publish_records_the_commit_it_created_as_the_new_pin(tmp_path, monkeypatch):
    write([_record()], tmp_path, PUBLISHED)
    uploaded = {}

    class Api:
        def create_repo(self, **kwargs):
            uploaded["created"] = kwargs["repo_id"]

        def upload_folder(self, **kwargs):
            uploaded.update(kwargs)
            return Namespace(oid="cafe1234")

    monkeypatch.setattr(hub, "HfApi", Api)
    assert hub.publish(directory=tmp_path) == "cafe1234"
    assert (tmp_path / hub.REVISION).read_text().strip() == "cafe1234"
    assert (tmp_path / hub.FETCHED).read_text().strip() == "cafe1234"
    assert uploaded["created"] == hub.REPO
    assert uploaded["allow_patterns"] == [f"*{PUBLISHED}"]


def test_publish_refuses_when_there_is_nothing_to_upload(tmp_path):
    with pytest.raises(SystemExit, match="nothing to publish"):
        hub.publish(directory=tmp_path)


def test_publish_command_forwards_the_repository_and_message(monkeypatch, capsys):
    called = {}

    def publish(**kwargs):
        called.update(kwargs)
        return "cafe1234"

    monkeypatch.setattr(hub, "publish", publish)
    monkeypatch.setitem(cmd_publish.__globals__, "_require_passing_reports", lambda: None)
    cmd_publish(Namespace(repo="org/reports", message="Fresh grid"))
    assert called == {"repo": "org/reports", "message": "Fresh grid"}
    assert "org/reports at cafe1234" in capsys.readouterr().out


def test_publish_rejects_unsuccessful_rows():
    with pytest.raises(SystemExit, match="refusing to publish.*error=1"):
        _require_clean([_record("error")])
    _require_clean([_record("pass"), _record("skip"), _record("oom"), _record("timeout")])


def test_unmeasured_wants_a_timing_even_when_correctness_already_passed():
    timed = _record()
    timed.result.benchmarked, timed.result.bench = True, {"fwd_ms": 1.0}
    assert not unmeasured(timed)
    assert unmeasured(None)  # never run here
    assert unmeasured(_record())  # passed but untimed: dispatch cannot rank it
    assert not unmeasured(_record("fail"))  # a verdict rerunning will not change
    assert unmeasured(_record("error"))  # the harness failed, correctness unknown
    assert not unmeasured(timed, benchmark=False)


def test_fill_retains_stable_skip_and_exhausted_evidence():
    assert not _fill_pending(_record("skip"))
    assert not _fill_pending(_record("oom"))
    assert not _fill_pending(_record("timeout"))
    assert _fill_pending(_record("error"))


def test_fill_explains_which_backend_extra_is_missing(monkeypatch):
    op = _op()
    case = Case((("D", 4),), (), torch.float32, (), frozenset())
    monkeypatch.setitem(cmd_fill.__globals__, "Store", object)
    monkeypatch.setitem(cmd_fill.__globals__, "_work", lambda *args: [(op, "fla", case)])
    monkeypatch.setitem(cmd_fill.__globals__, "available", lambda impl: False)
    monkeypatch.setitem(
        cmd_fill.__globals__,
        "unavailable_reason",
        lambda impl: "implementation 'fla' needs 'flash-linear-attention'; install popcorn[fla]",
    )

    with pytest.raises(SystemExit, match=r"install popcorn\[fla\]"):
        cmd_fill(
            Namespace(
                ops=[],
                backend=None,
                limit=None,
                device="cpu",
                hardware=None,
            )
        )


def test_fill_prunes_dominated_oom_shapes_but_runs_mixed_tradeoffs(monkeypatch, tmp_path):
    op = _op()

    def case(x, y, z):
        return Case((("X", x), ("Y", y), ("Z", z)), (), torch.float32, (), frozenset())

    oom = case(4, 8, 16)
    dominated = case(8, 8, 16)
    mixed = case(2, 8, 32)
    work = [(op, "alt", candidate) for candidate in (oom, dominated, mixed)]
    calls = []
    written = []

    class FakeStore:
        user = tmp_path

        def write_user(self, records):
            written.extend(records)

    def run_case(impl, candidate, *args, **kwargs):
        calls.append(candidate)
        status = "oom" if candidate == oom else "pass"
        return Record(
            op.name,
            impl,
            str(candidate),
            candidate.case_id,
            candidate.config(),
            Environment("cpu", torch.__version__, None, "2026-01-01T00:00:00+00:00"),
            Result(status=status, grad=True, benchmarked=status == "pass"),
        )

    monkeypatch.setitem(cmd_fill.__globals__, "Store", FakeStore)
    monkeypatch.setitem(cmd_fill.__globals__, "_work", lambda *args: work)
    monkeypatch.setitem(cmd_fill.__globals__, "_usable_records", lambda *args: [])
    monkeypatch.setitem(cmd_fill.__globals__, "available", lambda impl: True)
    monkeypatch.setattr(op.bench, "run_case", run_case)
    cmd_fill(
        Namespace(
            ops=[],
            backend=None,
            limit=None,
            device="cpu",
            hardware=None,
            force=False,
            in_process=True,
            timeout=30,
            reps=1,
            flush=10,
        )
    )

    assert calls == [oom, mixed]
    assert [record.result.status for record in written] == ["oom", "pass"]


def test_fill_counts_only_rows_this_machine_could_have_produced(tmp_path):
    """Evidence from another GPU or Torch build says nothing about what to run here."""
    op = KERNELS["rms_norm"]
    case = cases(op, 1)[0]
    environment = Environment(
        "TestDevice",
        torch.__version__,
        installed_version("fla"),
        "2026-01-01T00:00:00+00:00",
        op.fingerprint,
        _impl(op, "fla").fingerprint,
    )
    grad = not _impl(op, "fla").forward_only
    result = Result(status="pass", grad=grad, benchmarked=True, bench={"fwd_ms": 1.0})
    removed = Environment("TestDevice", torch.__version__, None, environment.ts, op.fingerprint, "old")
    write(
        [
            Record(op.name, "fla", str(case), case.case_id, case.config(), environment, result),
            Record(op.name, "removed", str(case), case.case_id, case.config(), removed, result),
        ],
        tmp_path,
    )

    store = Store(bundled=tmp_path, user=tmp_path / "user")
    work = [(op, "fla", case)]
    cached = _record_index(_usable_records(store, work, "TestDevice"))
    assert not unmeasured(cached[(op.name, "fla", case.case_id, grad)])
    assert _usable_records(store, work, "OtherDevice") == []


def test_fill_re_measures_only_what_a_fingerprint_change_invalidated(tmp_path):
    """A workflow that benchmarks changed code leans on this: edit one adapter and its rows,
    and only its rows, stop counting as cached."""
    op = KERNELS["rms_norm"]
    case = cases(op, 1)[0]
    grad = not _impl(op, "fla").forward_only
    result = Result(status="pass", grad=grad, benchmarked=True, bench={"fwd_ms": 1.0})

    def row(impl: str, impl_hash: str | None) -> Record:
        environment = Environment(
            "TestDevice", torch.__version__, installed_version(impl), "2026-01-01T00:00:00+00:00", op.fingerprint, impl_hash
        )
        return Record(op.name, impl, str(case), case.case_id, case.config(), environment, result)

    write([row("fla", _impl(op, "fla").fingerprint), row("liger", "stale")], tmp_path)
    usable = _usable_records(
        Store(bundled=tmp_path, user=tmp_path / "user"),
        [(op, "fla", case), (op, "liger", case)],
        "TestDevice",
    )
    cached = _record_index(usable)
    assert (op.name, "fla", case.case_id, grad) in cached  # untouched adapter stays cached
    assert (op.name, "liger", case.case_id, grad) not in cached  # edited adapter is measured again


def test_the_build_hook_names_the_same_dataset_as_the_hub():
    """setup.py cannot import popcorn (no torch in the build env), so it repeats the name."""
    source = (Path(__file__).parents[2] / "setup.py").read_text()
    assert f'REPO = "{hub.REPO}"' in source
    assert "POPCORN_SKIP_REPORTS" in source


def test_reports_and_viewer_consume_typed_records():
    records = [_record(), _record("skip", "other")]
    assert "| op | fast | cpu | 1 | 1 |" in matrix(records)
    assert "No report rows." in render([])
    assert '"schema":3' in render(records)


def test_empty_cli_work_is_an_error(tmp_path):
    with pytest.raises(SystemExit, match="no report rows"):
        cmd_merge(Namespace(sources=[tmp_path / "missing.jsonl"], expect=None, check=False, scrub_incomplete=False))
    with pytest.raises(SystemExit, match="expected 2 shard files"):
        cmd_merge(Namespace(sources=[tmp_path / "one.jsonl"], expect=2, check=False, scrub_incomplete=False))
    with pytest.raises(SystemExit, match="needs shard files"):
        cmd_merge(Namespace(sources=[], expect=None, check=False, scrub_incomplete=False))
    with pytest.raises(SystemExit, match="pass no sources"):
        cmd_merge(Namespace(sources=[tmp_path / "one.jsonl"], expect=None, check=True, scrub_incomplete=False))
    with pytest.raises(SystemExit, match="pass no sources or --check"):
        cmd_merge(Namespace(sources=[tmp_path / "one.jsonl"], expect=None, check=False, scrub_incomplete=True))
    with pytest.raises(SystemExit, match="no matching"):
        cmd_run(
            Namespace(
                ops=["rms_norm"],
                backend="torch",
                limit=1,
                shard=None,
                reps=1,
                device="cpu",
                hardware=None,
                out=None,
                timeout=30,
                in_process=True,
                only=None,
            )
        )


def test_scrub_drops_only_selected_rows_and_removes_emptied_files(tmp_path):
    keep, junk = _record(), _record("error", "other")
    junk.result.reason = INCOMPLETE
    write([keep, junk], tmp_path, PUBLISHED)
    orphan = _record("error", "doomed")
    orphan.op = "gone"
    orphan.result.reason = INCOMPLETE
    write([orphan], tmp_path)
    (tmp_path / store.FETCHED).write_text("cafe")

    dropped = store.scrub(tmp_path, lambda record: record.result.reason == INCOMPLETE)

    assert dropped == 2
    assert [record.case_id for record in read(tmp_path)] == ["case"]
    assert not (tmp_path / f"gone{store.SCRATCH}").exists()
    assert not (tmp_path / store.FETCHED).exists()
    assert store.scrub(tmp_path, lambda record: record.result.reason == INCOMPLETE) == 0


def _selector(path, op, impl, *grid_cases):
    """A JSONL of report rows, as `run --only` consumes it."""
    rows = []
    for case in grid_cases:
        record = _record(case_id=case.case_id)
        record.op, record.impl, record.config = op, impl, case.config()
        rows.append(json.dumps(record.to_dict(), sort_keys=True))
    path.write_text("\n".join(rows) + "\n")
    return str(path)


def test_work_uses_the_shared_plan_and_only_samples_when_limited():
    op = popcorn.KERNELS["rms_norm"]
    backend = next(name for name in op.available_backends() if name != "torch")
    planned = [case for _, _, case in _work([op.name], backend=backend)]
    sampled = [case for _, _, case in _work([op.name], backend=backend, limit=4)]
    assert planned == case_plan(op).flatten()
    assert sampled == sample_cases(op, 4)


def test_slice_work_expands_one_exact_context_over_its_production_axis():
    op = popcorn.KERNELS["rms_norm"]
    backend = next(name for name in op.available_backends() if name != "torch")
    source = case_plan(op).flatten()[0]
    axis = "normalized_shape"
    fixed = {name: value for name, value in source.dims if name != axis}
    spec = [
        axis,
        f"dtype={str(source.dtype).removeprefix('torch.')}",
        *([f"...={json.dumps(source.batch)}"] if source.batch else []),
        *(f"{name}={json.dumps(value)}" for name, value in sorted(fixed.items())),
        *(f"{name}={json.dumps(value)}" for name, value in source.args),
        *(f"+{name}" for name in sorted(source.present)),
    ]

    selected = [case for _, _, case in _slice_work([op.name], backend, spec)]

    assert len(selected) > 1
    assert all(case.batch == source.batch for case in selected)
    assert all(case.dtype == source.dtype for case in selected)
    assert all(case.args == source.args and case.present == source.present for case in selected)
    assert all({name: value for name, value in case.dims if name != axis} == fixed for case in selected)


def test_only_rebuilds_exactly_the_recorded_cases(tmp_path):
    op = popcorn.KERNELS["swiglu"]
    chosen, skipped = cases(op, limit=4)[:2]
    selector = _selector(tmp_path / "selected.jsonl", op.name, "popcorn", chosen)

    work = _work([op.name], backend="popcorn", only=selector)
    assert [case.case_id for _, _, case in work] == [chosen.case_id]
    assert skipped.case_id not in {case.case_id for _, _, case in work}


def test_only_rebuilds_cases_the_current_grid_no_longer_samples(tmp_path):
    """A recorded row carries its own config, so a shifted grid sample cannot drop it."""
    op = popcorn.KERNELS["swiglu"]
    unsampled = cases(op, limit=400)[-1]
    assert unsampled.case_id not in {case.case_id for case in cases(op, limit=4)}
    selector = _selector(tmp_path / "selected.jsonl", op.name, "popcorn", unsampled)

    work = _work([op.name], backend="popcorn", only=selector)
    assert [case.case_id for _, _, case in work] == [unsampled.case_id]


def test_only_reports_unrunnable_selectors(tmp_path, capsys):
    op = popcorn.KERNELS["swiglu"]
    case = cases(op, limit=4)[0]
    selector = _selector(tmp_path / "selected.jsonl", op.name, "not-an-impl", case)

    assert _work([op.name], only=selector) == []
    out = capsys.readouterr().out
    assert "0 of 1" in out and "unavailable" in out  # never silently dropped


def test_only_rejects_a_limit(tmp_path):
    with pytest.raises(SystemExit, match="--limit does not apply"):
        _work([], limit=4, only=str(tmp_path / "selected.jsonl"))


def test_only_still_honours_a_named_op(tmp_path):
    op = popcorn.KERNELS["swiglu"]
    selector = _selector(tmp_path / "selected.jsonl", op.name, "popcorn", cases(op, limit=4)[0])
    assert _work(["rms_norm"], only=selector) == []
    assert len(_work(["swiglu"], only=selector)) == 1


def test_isolated_case_records_a_timeout_without_killing_the_run(monkeypatch):
    op = popcorn.KERNELS["rms_norm"]
    case = cases(op, limit=1)[0]
    seen = []

    def hang(command, **kwargs):
        seen.extend(command)
        raise subprocess.TimeoutExpired(cmd="run-case", timeout=kwargs.get("timeout", 0))

    monkeypatch.setattr(subprocess, "run", hang)
    record = _isolated(op, "torch", case, "cpu", reps=1, timeout=7, grad=False)
    assert record.result.status == "timeout"
    assert record.result.grad is False
    assert record.result.reason == "no result within 7s"
    assert "--no-grad" in seen


def test_isolated_case_records_a_crash_when_the_worker_dies(monkeypatch):
    op = popcorn.KERNELS["rms_norm"]
    case = cases(op, limit=1)[0]

    def die(*args, **kwargs):
        return subprocess.CompletedProcess(args=[], returncode=-11, stdout="", stderr="Segmentation fault")

    monkeypatch.setattr(subprocess, "run", die)
    record = _isolated(op, "torch", case, "cpu", reps=1, timeout=7)
    assert record.result.status == "crash"
    assert "worker exit -11" in record.result.reason
    assert "Segmentation fault" in record.result.reason


def test_target_device_gates_on_the_live_hardware_name():
    with pytest.raises(SystemExit, match="does not match"):
        _target_device(Namespace(device="cpu", hardware="H100"))
    with pytest.raises(SystemExit, match="invalid --device"):
        _target_device(Namespace(device="not-a-device", hardware=None))
    assert _target_device(Namespace(device="cpu", hardware=None)) == "cpu"
    if torch.cuda.is_available():
        assert _target_device(Namespace(device="cuda", hardware=None)) == f"cuda:{torch.cuda.current_device()}"


def test_slurm_submit_dry_run_builds_a_sharded_script(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cmd_submit(
        Namespace(
            ops=["rms_norm"],
            array=2,
            reps=1,
            limit=1,
            qos=None,
            hardware=None,
            time="00:10:00",
            dry_run=True,
            timeout=30,
            only=None,
        )
    )
    script = (tmp_path / "logs" / "popcorn_bench.slurm").read_text()
    assert "#SBATCH --array=0-1" in script
    assert "--qos" not in script  # unset QoS leaves the cluster default
    assert 'TRITON_CACHE_DIR="/tmp/triton_' in script
    assert '--shard "$SLURM_ARRAY_TASK_ID/2"' in script
    assert "--hardware" not in script  # unset gate leaves array tasks unconstrained
    assert "--timeout 30" in script  # array tasks isolate each case behind a deadline
    assert "--only" not in script


def test_slurm_submit_asks_the_merge_job_to_publish(tmp_path, monkeypatch):
    """The array writes shards; only the dependent merge job uploads, and only when asked."""
    monkeypatch.chdir(tmp_path)
    wrapped = []
    submitted = []

    def fake_sbatch(command, **kwargs):
        submitted.append(command)
        wrapped.extend(argument for argument in command if argument.startswith("--wrap="))
        return subprocess.CompletedProcess(command, 0, stdout="Submitted batch job 1\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_sbatch)
    arguments = {
        "ops": ["rms_norm"],
        "array": 1,
        "reps": 1,
        "limit": 1,
        "qos": None,
        "hardware": None,
        "time": "00:10:00",
        "dry_run": False,
        "timeout": 30,
        "only": None,
    }
    cmd_submit(Namespace(**arguments, publish=True))
    assert "merge --expect 1 --publish " in wrapped[0]
    assert "--dependency=afterok:1" in submitted[1]
    wrapped.clear()
    submitted.clear()
    cmd_submit(Namespace(**arguments, publish=False))
    assert "--publish" not in wrapped[0]
    assert "--dependency=afterany:1" in submitted[1]


def test_slurm_submit_includes_qos_when_given(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cmd_submit(
        Namespace(
            ops=["rms_norm"],
            array=1,
            reps=1,
            limit=1,
            qos="batch",
            hardware=None,
            time="00:10:00",
            dry_run=True,
            timeout=30,
            only=None,
        )
    )
    script = (tmp_path / "logs" / "popcorn_bench.slurm").read_text()
    assert "#SBATCH --qos=batch\n" in script


def test_slurm_submit_forwards_the_hardware_gate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cmd_submit(
        Namespace(
            ops=["rms_norm"],
            array=1,
            reps=1,
            limit=1,
            qos=None,
            hardware="H100",
            time="1:00",
            dry_run=True,
            timeout=30,
            only=None,
        )
    )
    assert "--hardware H100" in (tmp_path / "logs" / "popcorn_bench.slurm").read_text()


def test_production_benchmark_modules_have_no_local_imports():
    root = Path(popcorn.__file__).parent
    paths = [
        *sorted((root / "bench").glob("*.py")),
        root / "core" / "config.py",
        root / "core" / "dispatcher.py",
        root / "core" / "tuning.py",
    ]
    for path in paths:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            imports = [child for child in ast.walk(node) if isinstance(child, (ast.Import, ast.ImportFrom))]
            assert not imports, f"{path.relative_to(root)}:{node.lineno} contains a function-local import"


def test_import_layers_have_no_back_edges():
    root = Path(popcorn.__file__).parent
    forbidden = {
        "bench/compare.py": ("popcorn.core.dispatcher", "popcorn.core.tuning", "popcorn.bench.service"),
        "bench/grid.py": ("popcorn.core.dispatcher", "popcorn.core.tuning", "popcorn.bench.service"),
        "bench/store.py": ("popcorn.core.dispatcher", "popcorn.core.tuning", "popcorn.bench.service"),
        "core/tuning.py": ("popcorn.bench.compare", "popcorn.bench.grid", "popcorn.bench.service"),
    }
    for relative, names in forbidden.items():
        text = (root / relative).read_text()
        assert not any(name in text for name in names)
