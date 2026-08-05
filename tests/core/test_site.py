"""GPU-free contract tests for the generated site evidence catalog."""

import importlib.util
from pathlib import Path

import torch
from jaxtyping import Float
from torch import Tensor

from popcorn.bench.model import Case, Environment, Record, Result
from popcorn.core.dispatcher import Dispatcher


def _script():
    path = Path(__file__).parents[2] / "scripts" / "update_site.py"
    spec = importlib.util.spec_from_file_location("update_site_script", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_evidence_catalog_separates_curves_samples_and_frontiers(monkeypatch):
    from popcorn.core import dims as dims_mod

    def reference(x: Float[Tensor, "seq hidden"], causal: bool = False):
        return x

    op = Dispatcher(reference)

    @op.register("fast")
    def fast(x, causal):
        return x

    monkeypatch.setitem(dims_mod.DIMS, "seq", {8, 16})
    monkeypatch.setitem(dims_mod.DIMS, "hidden", {4})
    monkeypatch.setitem(dims_mod.ANCHORS, "seq", 8)
    monkeypatch.setitem(dims_mod.ANCHORS, "hidden", 4)
    module = _script()
    plan = module.case_plan(op)
    curve = next(series for series in plan.series if series.name == "curve:production:seq:float32:base")
    implementation = next(impl for impl in op._impls if impl.name == "fast")

    def row(case, status, *, ref_hash=op.fingerprint, ts="2026-08-03T00:00:00+00:00"):
        bench = {"fwd_ms": 1.0, "ref_fwd_ms": 2.0} if status == "pass" else {}
        return Record(
            op.name,
            "fast",
            str(case),
            case.case_id,
            case.config(),
            Environment(
                "test device",
                torch.__version__,
                "1.0",
                ts,
                ref_hash,
                implementation.fingerprint,
            ),
            Result(status=status, grad=True, bench=bench, benchmarked=bool(bench)),
        )

    first, second = curve.cases
    historical = Case((("hidden", 4), ("seq", 7)), (), torch.float32, (("causal", False),), frozenset())
    stale_case = Case((("hidden", 4), ("seq", 9)), (), torch.float32, (("causal", False),), frozenset())
    stale = row(stale_case, "pass", ref_hash="old")
    newer_error = row(first, "error", ts="2026-08-04T00:00:00+00:00")
    catalog = module.evidence(
        op,
        [row(first, "pass"), newer_error, row(second, "oom"), row(historical, "pass"), stale],
    )

    assert catalog["freshness"]["stale_results_omitted"] == 1
    assert historical.case_id in catalog["samples"]
    assert first.case_id not in catalog["samples"]
    exported = next(item for item in catalog["curves"] if item["id"] == curve.name)
    assert exported["fixed"] == {
        "dims": {"hidden": 4},
        "batch": [],
        "args": {"causal": False},
        "present": [],
    }
    assert catalog["cases"][first.case_id]["results"][0]["fwd_ms"] == 1.0
    assert catalog["cases"][first.case_id]["results"][0]["status"] == "pass"
    frontier = next(item for item in catalog["frontiers"] if item["curve_id"] == curve.name and item["impl"] == "fast")
    assert frontier["pass_max"] == 8
    assert frontier["terminal"] == {"x": 16, "status": "oom"}
    assert catalog["coverage"]["statuses"] == {"oom": 1, "pass": 2}
