"""GPU-free tests for Space algebra, region fitting, and the probe planner."""

import random

import torch

from popcorn.bench.fit import Region, fit, stratum
from popcorn.bench.model import Case, Environment, Record, Result
from popcorn.bench.plan import plan
from popcorn.core.args import partition_args
from popcorn.core.dispatcher import Dispatcher
from popcorn.core.spaces import Div, Pow2, Range, Real, Space, contains, space
from jaxtyping import Float
from torch import Tensor


class TestSpace:
    def test_range_membership_and_text(self):
        band = Range(2, 4)
        assert 3 in band and 5 not in band and 2 in band
        assert str(band) == "[2, 4]"
        assert band == Range(2, 4)

    def test_operators(self):
        assert 16 in Range(1, 64) % 8 and 12 not in Range(1, 64) % 8
        assert 64 in Pow2() and 48 not in Pow2()
        assert 16 in Div(8) and 12 not in Div(8)
        assert 3 in (Range(1, 2) | {3, 4})
        assert 2 in (Range(1, 4) - {1, 3})
        assert 1 in (Range(1, 2) ^ Range(2, 3)) and 2 not in (Range(1, 2) ^ Range(2, 3))
        assert 4 in ({2, 4} | Range(8, 8))

    def test_grid_and_sample(self):
        values = Range(2, 128).grid(seeds=(1, 2, 3, 8, 16, 33, 64, 128, 1024))
        assert values[0] == 2 and values[-1] == 128
        assert 3 in values and 1024 not in values
        drawn = (Range(1, 64) % 8).sample(5, random.Random(0))
        assert drawn and all(value % 8 == 0 for value in drawn)
        # Curated sets and sparse union ladders survive log-subsampling.
        assert space({8, 16, 32, 64}).grid() == [8, 16, 32, 64]
        sparse = (Range(2, 64) | {1024, 4096}).grid(seeds=(2, 16))
        assert 1024 in sparse and 4096 in sparse and 16 in sparse

    def test_contains_polymorphic(self):
        assert contains(Range(2, 4), 3)
        assert contains({16, 32}, 16) and not contains({16, 32}, 8)
        assert contains(None, None) and not contains(None, torch.ones(1))
        assert contains(lambda v: v > 0, 1)
        assert not contains(5, torch.tensor(5))

    def test_set_lift(self):
        lifted = space({16, 32})
        assert isinstance(lifted, Space) and 16 in lifted and 8 not in lifted


def _row(backend, dims, status, *, device="cpu", grad=False, dtype="float32"):
    return Record(
        "op",
        backend,
        "",
        f"{backend}-{dims}",
        {"dims": dims, "batch": [], "dtype": dtype, "args": {"flag": False}, "present": []},
        Environment(device, torch.__version__, None, "2026-01-01T00:00:00+00:00"),
        Result(status=status, grad=grad),
    )


class TestFit:
    def test_induces_range_modulus_and_drops(self):
        rows = [
            *(_row("fast", {"D": n}, "pass") for n in (8, 16, 24, 32)),
            _row("fast", {"D": 12}, "fail"),
            _row("fast", {"D": 20}, "fail"),
            _row("fast", {"D": 28}, "fail"),
        ]
        regions = fit(rows, ["D"])
        region = next(iter(regions.values()))
        assert isinstance(region, Region)
        assert 16 in region.spaces["D"] and 12 not in region.spaces["D"]
        assert region.contains({"D": 24})
        assert region.reject({"D": 12})

    def test_oom_caps_without_failing(self):
        rows = [
            _row("fast", {"D": 8}, "pass"),
            _row("fast", {"D": 64}, "pass"),
            _row("fast", {"D": 256}, "oom"),
        ]
        region = next(iter(fit(rows, ["D"]).values()))
        assert 64 in region.spaces["D"] and 256 not in region.spaces["D"]

    def test_continuous_args_fit_as_reals_not_stratum(self):
        def row(eps, status):
            record = _row("fast", {"D": 8}, status)
            record.config = {**record.config, "args": {"flag": False, "eps": eps}}
            return record

        regions = fit([row(1e-6, "pass"), row(1e-4, "pass"), row(1e-5, "fail")], ["D"])
        assert len(regions) == 1  # floats do not split the stratum
        region = next(iter(regions.values()))
        assert isinstance(region.reals["eps"], Real)
        assert region.contains({"D": 8}, {"flag": False, "eps": 5e-5})
        assert region.reject({"D": 8}, {"flag": False, "eps": 1e-5})
        # Same discrete args → same stratum even when eps differs.
        assert stratum(row(1e-6, "pass")) == stratum(row(1e-3, "pass"))

    def test_partition_args(self):
        discrete, continuous = partition_args({"causal": True, "eps": 1e-5, "scale": None, "beta": 0.5})
        assert discrete == {"causal": True, "scale": None}
        assert continuous == {"eps": 1e-5, "beta": 0.5}


class TestPlan:
    def test_emits_screen_points_then_fixpoints(self, monkeypatch):
        from popcorn.core import dims as dims_mod

        def reference(x: Float[Tensor, "... D"], flag: bool = False):
            return x

        monkeypatch.setitem(dims_mod.DIMS, "D", {2, 4, 8})
        op = Dispatcher(reference)
        first = plan(op, [], "fast", effort="quick", device="cpu", grad=False)
        assert first and all(isinstance(case, Case) for case in first)
        # Label every planned case as pass; next plan should shrink.
        rows = [
            Record(
                "op",
                "fast",
                str(case),
                case.case_id,
                case.config(),
                Environment("cpu", torch.__version__, None, "2026-01-01T00:00:00+00:00"),
                Result(status="pass", grad=False),
            )
            for case in first
        ]
        # Force config dtype/args to match planner stratum.
        for record, case in zip(rows, first):
            record.config = case.config()
        second = plan(op, rows, "fast", effort="quick", device="cpu", grad=False)
        assert isinstance(second, list)
