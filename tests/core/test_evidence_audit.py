"""Release evidence gate tests."""

from datetime import datetime, timezone

import torch
from jaxtyping import Float, Float32
from torch import Tensor

from popcorn.bench.grid import CasePlan, CaseSeries
from popcorn.bench.model import Case, Environment, Record, Result
from popcorn.bench.store import audit_evidence, audit_outcomes, write
from popcorn.core.dispatcher import Dispatcher


def _op():
    def evidence_op(x: Float[Tensor, "tokens hidden"]):
        return x

    op = Dispatcher(evidence_op)

    @op.register("popcorn")
    def implementation(x):
        return x

    return op


def _float32_op():
    def evidence_op(x: Float[Tensor, "tokens hidden"]):
        return x

    op = Dispatcher(evidence_op)

    @op.register("popcorn")
    def implementation(x: Float32[Tensor, "tokens hidden"]):
        return x

    return op


def _record(op, case, *, status="pass", benchmarked=True, bench_error="", grad=True):
    impl = next(impl for impl in op._impls if impl.name == "popcorn")
    environment = Environment(
        "Test Device",
        torch.__version__,
        None,
        datetime.now(timezone.utc).isoformat(),
        op.fingerprint,
        impl.fingerprint,
    )
    result = Result(
        status=status,
        grad=grad,
        benchmarked=benchmarked,
        bench={"fwd_ms": 1.0} if benchmarked else {},
        bench_error=bench_error,
    )
    return Record(op.name, impl.name, str(case), case.case_id, case.config(), environment, result)


def test_evidence_audit_checks_timing_curves_and_bad_outcomes(monkeypatch, tmp_path):
    op = _op()
    cases = tuple(Case((("hidden", hidden), ("tokens", 8)), (), torch.float32, (), frozenset()) for hidden in (8, 16, 32))
    series = CaseSeries(
        "production:hidden:float32",
        "curve",
        cases[:2],
        profile="production",
        axis="hidden",
        fixed_dims=(("tokens", 8),),
        dtype=torch.float32,
    )
    monkeypatch.setattr("popcorn.bench.store.case_plan", lambda _op: CasePlan((series,)))
    write([_record(op, cases[0]), _record(op, cases[1]), _record(op, cases[2], status="error")], tmp_path)

    counts, problems = audit_evidence([op], tmp_path)

    assert counts == {"pairs": 1, "timed": 1, "curves": 1, "complete_curves": 1}
    assert problems["bad_status"] == [f"{op.name}:popcorn (error=1)"]


def test_evidence_audit_reports_missing_timing_and_curve_depth(monkeypatch, tmp_path):
    op = _op()
    cases = tuple(Case((("hidden", hidden), ("tokens", 8)), (), torch.float32, (), frozenset()) for hidden in (8, 16))
    series = CaseSeries(
        "production:hidden:float32",
        "curve",
        cases,
        profile="production",
        axis="hidden",
        fixed_dims=(("tokens", 8),),
        dtype=torch.float32,
    )
    monkeypatch.setattr("popcorn.bench.store.case_plan", lambda _op: CasePlan((series,)))
    write([_record(op, cases[0], benchmarked=False)], tmp_path)

    counts, problems = audit_evidence([op], tmp_path)

    assert counts["timed"] == counts["complete_curves"] == 0
    assert problems["no_timing"] == [f"{op.name}:popcorn"]
    assert problems["incomplete_curve"] == [f"{op.name}:popcorn production:hidden:float32 (0/2 timed points; 2 planned)"]


def test_outcome_audit_checks_only_new_rows(tmp_path):
    op = _op()
    case = Case((("hidden", 8), ("tokens", 8)), (), torch.float32, (), frozenset())
    other = Case((("hidden", 16), ("tokens", 8)), (), torch.float32, (), frozenset())
    write(
        [
            _record(op, other, status="timeout"),
            _record(op, case, status="pass", bench_error="timing failed"),
        ],
        tmp_path,
    )

    problems = audit_outcomes([tmp_path])

    assert "bad_status" not in problems
    assert problems["bench_error"] == [f"{op.name}:popcorn (1 rows)"]


def test_evidence_audit_excludes_statically_gated_curves(monkeypatch, tmp_path):
    op = _float32_op()
    cases = tuple(Case((("hidden", hidden), ("tokens", 8)), (), torch.float16, (), frozenset()) for hidden in (8, 16))
    series = CaseSeries(
        "production:hidden:float16",
        "curve",
        cases,
        profile="production",
        axis="hidden",
        fixed_dims=(("tokens", 8),),
        dtype=torch.float16,
    )
    monkeypatch.setattr("popcorn.bench.store.case_plan", lambda _op: CasePlan((series,)))

    counts, problems = audit_evidence([op], tmp_path)

    assert counts["curves"] == counts["complete_curves"] == 0
    assert "incomplete_curve" not in problems


def test_evidence_audit_excludes_fully_attempted_skip_curves(monkeypatch, tmp_path):
    op = _op()
    cases = tuple(Case((("hidden", hidden), ("tokens", 8)), (), torch.float32, (), frozenset()) for hidden in (8, 16))
    series = CaseSeries(
        "production:hidden:float32",
        "curve",
        cases,
        profile="production",
        axis="hidden",
        fixed_dims=(("tokens", 8),),
        dtype=torch.float32,
    )
    monkeypatch.setattr("popcorn.bench.store.case_plan", lambda _op: CasePlan((series,)))
    write([_record(op, case, status="skip") for case in cases], tmp_path)

    counts, problems = audit_evidence([op], tmp_path)

    assert counts["curves"] == counts["complete_curves"] == 0
    assert "incomplete_curve" not in problems


def test_evidence_audit_accepts_forward_only_points_at_the_fp16_backward_frontier(monkeypatch, tmp_path):
    def evidence_op(x: Float[Tensor, "tokens hidden"], weight: Float[Tensor, "hidden"]):
        return x * weight

    op = Dispatcher(evidence_op)
    op.register("popcorn")(lambda x, weight: x * weight)
    cases = tuple(
        Case((("hidden", 8), ("tokens", tokens)), (), torch.float16, (), frozenset()) for tokens in (65_536, 131_072)
    )
    series = CaseSeries(
        "long-context:tokens:float16",
        "curve",
        cases,
        profile="long-context",
        axis="tokens",
        fixed_dims=(("hidden", 8),),
        dtype=torch.float16,
    )
    monkeypatch.setattr("popcorn.bench.store.case_plan", lambda _op: CasePlan((series,)))
    write([_record(op, case, grad=False) for case in cases], tmp_path)

    counts, problems = audit_evidence([op], tmp_path)

    assert counts["curves"] == counts["complete_curves"] == 1
    assert "incomplete_curve" not in problems
