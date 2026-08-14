"""GPU-free tests for Space algebra, region fitting, and the probe planner."""

import random
from itertools import combinations, product
from math import prod

import torch

from popcorn.bench.fit import Region, fit, stratum
from popcorn.bench.grid import (
    CasePlan,
    CaseSeries,
    _anchor,
    _repair,
    case_plan,
    cases,
    covering_cases,
    estimated_bytes,
    grid_cases,
    ladder_cases,
    sample_cases,
    slice_cases,
    smoke_cases,
)
from popcorn.bench.model import Case, Environment, Record, Result
from popcorn.bench.plan import EXHAUSTED, BudgetFrontier, adaptive_plan, dominates, plan
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
    def test_pairwise_grid_covers_every_declared_dim_and_case_axis(self, monkeypatch):
        from popcorn.core import dims as dims_mod

        def reference(
            x: Float[Tensor, "X Y Z"],
            bias: Float[Tensor, "Z"] | None = None,
            flag: bool = False,
        ):
            return x if bias is None or not flag else x + bias

        monkeypatch.setitem(dims_mod.DIMS, "X", {2, 4})
        monkeypatch.setitem(dims_mod.DIMS, "Y", {3, 6, 9})
        monkeypatch.setitem(dims_mod.DIMS, "Z", {5, 10})
        grid = covering_cases(Dispatcher(reference))
        axes = {
            "X": [2, 4],
            "Y": [3, 6, 9],
            "Z": [5, 10],
            "dtype": ["float32", "float16", "bfloat16"],
            "flag": [False, True],
            "bias": [False, True],
        }

        def coordinates(case):
            dims = dict(case.dims)
            return {
                **dims,
                "dtype": str(case.dtype).removeprefix("torch."),
                "flag": dict(case.args)["flag"],
                "bias": "bias" in case.present,
            }

        rows = [coordinates(case) for case in grid]
        for left, right in combinations(axes, 2):
            assert {(row[left], row[right]) for row in rows} == set(product(axes[left], axes[right]))
        assert len(grid) < prod(map(len, axes.values()))
        assert dict(grid[0].dims) == {"X": 2, "Y": 3, "Z": 5}

    def test_repair_snaps_kv_heads_onto_the_largest_divisor_of_q_heads(self):
        for q_heads, kv_heads, expected in ((6, 4, 3), (2, 8, 2), (128, 40, 32), (32, 8, 8), (6, 6, 6), (1, 128, 1)):
            values = {"q_heads": q_heads, "kv_heads": kv_heads}
            _repair(values, {})
            assert values["kv_heads"] == expected

    def test_repair_caps_rotary_half_at_half_the_head_dim(self):
        for half, head_dim, expected in ((128, 128, 64), (48, 80, 40), (64, 128, 64), (8, 256, 8)):
            values = {"half": half, "head_dim": head_dim}
            _repair(values, {})
            assert values["half"] == expected

    def test_repair_caps_moe_top_k_at_the_expert_count(self):
        values, args = {"experts": 2}, {"top_k": 6}
        _repair(values, args)
        assert args == {"top_k": 2}
        _repair(values, args)
        assert args == {"top_k": 2}

    def test_repair_resplits_mrope_sections_to_sum_to_half_the_head(self):
        for head_dim, expected in ((32, [8, 4, 4]), (16, [4, 2, 2]), (80, [20, 10, 10]), (48, [12, 6, 6])):
            args = {"mrope_section": [8, 4, 4]}
            _repair({"head_dim": head_dim}, args)
            assert args["mrope_section"] == expected and sum(expected) == head_dim // 2

    def test_repair_collapses_unbroadcastable_cos_batch_to_shared_tables(self):
        for cos_batch, batch, expected in ((2, 8, 1), (8, 8, 8), (1, 64, 1)):
            values = {"cos_batch": cos_batch, "batch": batch}
            _repair(values, {})
            assert values["cos_batch"] == expected

    def test_repair_derives_log_linear_levels_from_the_sequence(self):
        for seq, expected in ((1, 1), (128, 8), (129, 9), (4096, 13)):
            values = {"levels": 8, "seq": seq}
            _repair(values, {})
            assert values["levels"] == expected

    def test_repair_snaps_the_ttt_mini_batch_onto_a_divisor_of_seq(self):
        for seq, mini_batch, expected in ((1290, 16, 15), (4096, 16, 16), (7, 16, 7), (33, 16, 11)):
            args = {"mini_batch_size": mini_batch}
            _repair({"seq": seq}, args)
            assert args["mini_batch_size"] == expected

    def test_ttt_fla_rejects_shapes_outside_its_safe_triton_region(self):
        from popcorn.kernels.ttt import _fla_ready

        safe = torch.empty(8, 4096, 64, 128, device="meta")
        assert not _fla_ready(q=safe, mini_batch_size=15)
        assert _fla_ready(q=safe, mini_batch_size=16)
        assert not _fla_ready(q=torch.empty(1, 16, 1, 192, device="meta"), mini_batch_size=16)
        assert not _fla_ready(q=torch.empty(8, 4096, 72, 128, device="meta"), mini_batch_size=16)
        assert not _fla_ready(q=torch.empty(2, 1_048_576, 1, 32, device="meta"), mini_batch_size=16)

    def test_repair_snaps_num_groups_onto_a_divisor_of_channels(self):
        for channels, groups, expected in ((6144, 4, 4), (2560, 3, 2), (7, 4, 1)):
            args = {"num_groups": groups}
            _repair({"channels": channels}, args)
            assert args["num_groups"] == expected

    def test_repair_derives_the_kda_gate_dim_from_heads_and_key_dim(self):
        values = {"gate_dim": 512, "heads": 16, "key_dim": 192}
        _repair(values, {})
        assert values["gate_dim"] == 16 * 192

    def test_repair_caps_varlen_boundaries_at_total_plus_one(self):
        values = {"boundaries": 33, "total": 8}
        _repair(values, {})
        assert values["boundaries"] == 9
        values = {"boundaries": 33, "total": 4096}
        _repair(values, {})
        assert values["boundaries"] == 33

    def test_repair_caps_the_conv_kernel_at_the_padded_sequence(self):
        values = {"kernel_size": 9, "seq": 7}
        _repair(values, {"padding": 0})
        assert values["kernel_size"] == 7
        values = {"kernel_size": 11, "seq": 3}
        _repair(values, {"padding": 2})
        assert values["kernel_size"] == 7

    def test_repair_caps_sparse_selection_at_its_deterministic_block_span(self):
        values = {"seq": 8192}
        _repair(values, {"block_count": 16, "block_size": 32})
        assert values["seq"] == 512

    def test_repair_leaves_axes_alone_unless_a_relation_pairs_them(self):
        for values, args in (
            ({"heads": 8, "kv_heads": 32}, {}),
            ({"q_heads": 2}, {}),
            ({"seq": 16}, {"top_k": 6}),
            ({"half": 128}, {}),
            ({"channels": 2560}, {}),
            ({"gate_dim": 512, "heads": 16}, {}),
            ({"boundaries": 33}, {}),
            # neighborhood windows saturate on short sequences, so bare kernel_size + seq
            # stays untouched; only the padded-conv op (padding argument) caps it.
            ({"kernel_size": 9, "seq": 7}, {}),
        ):
            unchanged, unchanged_args = dict(values), dict(args)
            _repair(values, args)
            assert (values, args) == (unchanged, unchanged_args)

    def test_grouped_query_grids_only_emit_expressible_head_counts(self, monkeypatch):
        from popcorn.core import dims as dims_mod

        def reference(q: Float[Tensor, "seq q_heads dim"], k: Float[Tensor, "seq kv_heads dim"]):
            return q + k.repeat_interleave(q.shape[1] // k.shape[1], 1)

        monkeypatch.setitem(dims_mod.DIMS, "q_heads", {2, 6})
        monkeypatch.setitem(dims_mod.DIMS, "kv_heads", {1, 4, 8})
        monkeypatch.setitem(dims_mod.DIMS, "seq", {8})
        monkeypatch.setitem(dims_mod.DIMS, "dim", {16})
        op = Dispatcher(reference)

        grid = covering_cases(op)
        sampled = cases(op, limit=32)
        assert grid and sampled
        for case in [*grid, *sampled]:
            dims = dict(case.dims)
            assert dims["q_heads"] % dims["kv_heads"] == 0

        # Repair snaps down to the nearest divisor rather than collapsing every row to MQA.
        assert {dims["kv_heads"] for dims in map(dict, (case.dims for case in grid))} == {1, 2, 3, 6}

    def test_ladders_sweep_one_axis_at_a_time_through_the_anchors(self, monkeypatch):
        from popcorn.core import dims as dims_mod

        def reference(x: Float[Tensor, "X Y"], flag: bool = False):
            return x

        monkeypatch.setitem(dims_mod.DIMS, "X", {2, 4, 8})
        monkeypatch.setitem(dims_mod.DIMS, "Y", {3, 6, 9})
        monkeypatch.setitem(dims_mod.ANCHORS, "X", 4)
        monkeypatch.setitem(dims_mod.ANCHORS, "Y", 7)  # snaps onto the nearest pool value, 6
        ladder = ladder_cases(Dispatcher(reference))

        rows = {(dict(case.dims)["X"], dict(case.dims)["Y"], case.dtype) for case in ladder}
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            assert {(x, 6, dtype) for x in (2, 4, 8)} <= rows
            assert {(4, y, dtype) for y in (3, 6, 9)} <= rows
        # Per dtype: three X rungs plus three Y rungs sharing the anchor point itself.
        assert len(ladder) == 3 * 5
        assert all(dict(case.args) == {"flag": False} for case in ladder)

    def test_anchor_falls_back_to_the_pool_median(self):
        assert _anchor([2, 4, 8], None) == 4
        assert _anchor([2, 4, 8], 5) == 4
        assert _anchor([2, 4, 8], 6) == 4  # ties break toward the smaller value
        assert _anchor([2, 4, 8], 100) == 8

    def test_ladders_respect_grouped_query_repair(self, monkeypatch):
        from popcorn.core import dims as dims_mod

        def reference(q: Float[Tensor, "seq q_heads dim"], k: Float[Tensor, "seq kv_heads dim"]):
            return q + k.repeat_interleave(q.shape[1] // k.shape[1], 1)

        monkeypatch.setitem(dims_mod.DIMS, "q_heads", {2, 6})
        monkeypatch.setitem(dims_mod.DIMS, "kv_heads", {1, 4, 8})
        monkeypatch.setitem(dims_mod.DIMS, "seq", {8})
        monkeypatch.setitem(dims_mod.DIMS, "dim", {16})
        for case in ladder_cases(Dispatcher(reference)):
            dims = dict(case.dims)
            assert dims["q_heads"] % dims["kv_heads"] == 0

    def test_slice_cases_keep_the_exact_context_and_omit_repairs_that_leave_it(self, monkeypatch):
        from popcorn.core import dims as dims_mod

        def reference(q: Float[Tensor, "... seq q_heads dim"], k: Float[Tensor, "... seq kv_heads dim"]):
            return q + k.repeat_interleave(q.shape[-2] // k.shape[-2], -2)

        monkeypatch.setitem(dims_mod.DIMS, "q_heads", {2, 4, 6, 8})
        op = Dispatcher(reference)
        context = Case(
            (("dim", 16), ("kv_heads", 4), ("seq", 8)),
            (2, 3),
            torch.float16,
            (),
            frozenset(),
        )

        selected = slice_cases(op, "q_heads", context)

        assert {dict(case.dims)["q_heads"] for case in selected} == {4, 8}
        assert all(dict(case.dims)["kv_heads"] == 4 for case in selected)
        assert all(case.batch == (2, 3) for case in selected)

    def test_grid_cases_union_covering_and_ladders_without_duplicates(self, monkeypatch):
        from popcorn.core import dims as dims_mod

        def reference(x: Float[Tensor, "X Y"]):
            return x

        monkeypatch.setitem(dims_mod.DIMS, "X", {2, 4})
        monkeypatch.setitem(dims_mod.DIMS, "Y", {3, 6})
        op = Dispatcher(reference)
        grid = grid_cases(op)
        identifiers = [case.case_id for case in grid]
        assert len(identifiers) == len(set(identifiers))
        assert {case.case_id for case in covering_cases(op)} <= set(identifiers)
        assert {case.case_id for case in ladder_cases(op)} <= set(identifiers)

    def test_case_plan_names_profiles_and_primary_axis_variants(self, monkeypatch):
        from popcorn.core import dims as dims_mod

        def reference(
            x: Float[Tensor, "... seq hidden"],
            bias: Float[Tensor, "hidden"] | None = None,
            causal: bool = False,
        ):
            return x if bias is None or causal else x + bias

        monkeypatch.setitem(dims_mod.DIMS, "...", {0, 1, 8})
        monkeypatch.setitem(dims_mod.DIMS, "seq", {8, 1024, 131_072})
        monkeypatch.setitem(dims_mod.DIMS, "hidden", {64, 256, 4096})
        op = Dispatcher(reference)
        planned = case_plan(op)

        assert isinstance(planned, CasePlan)
        assert planned.series[0].name == "coverage" and planned.series[0].kind == "coverage"
        assert {series.profile for series in planned.series if series.kind == "curve"} == {
            "production",
            "long-context",
        }
        long_base = next(
            series
            for series in planned.series
            if series.profile == "long-context"
            and series.axis == "seq"
            and series.dtype == torch.float32
            and series.name.endswith(":base")
        )
        assert long_base.batch == (1,)
        assert dict(long_base.fixed_dims)["hidden"] == 256
        assert {dict(case.dims)["seq"] for case in long_base.cases} == {8, 1024, 131_072}
        assert all(dict(case.dims)["hidden"] == 256 for case in long_base.cases)

        causal = next(
            series
            for series in planned.series
            if series.profile == "production"
            and series.axis == "seq"
            and series.dtype == torch.float32
            and dict(series.args)["causal"] is True
        )
        with_bias = next(
            series
            for series in planned.series
            if series.profile == "production"
            and series.axis == "seq"
            and series.dtype == torch.float32
            and series.present == frozenset({"bias"})
        )
        assert {dict(case.dims)["seq"] for case in causal.cases} == {8, 1024}
        assert all(dict(case.args)["causal"] is True for case in causal.cases)
        assert all("bias" in case.present for case in with_bias.cases)
        assert [case.case_id for case in grid_cases(op)] == [case.case_id for case in planned.flatten()]

    def test_context_profiles_split_full_model_and_frontier_lengths(self, monkeypatch):
        from popcorn.core import dims as dims_mod

        def reference(x: Float[Tensor, "seq hidden"]):
            return x

        monkeypatch.setitem(dims_mod.DIMS, "seq", {1024, 8192, 32_768, 65_536, 131_072})
        monkeypatch.setitem(dims_mod.DIMS, "hidden", {256, 4096})
        curves = case_plan(Dispatcher(reference)).series
        production = next(
            series
            for series in curves
            if series.profile == "production"
            and series.axis == "seq"
            and series.dtype == torch.float32
            and series.name.endswith(":base")
        )
        frontier = next(
            series
            for series in curves
            if series.profile == "long-context"
            and series.axis == "seq"
            and series.dtype == torch.float32
            and series.name.endswith(":base")
        )

        assert {dict(case.dims)["seq"] for case in production.cases} == {1024, 8192, 32_768}
        assert {dict(case.dims)["seq"] for case in frontier.cases} == {8192, 32_768, 65_536, 131_072}
        assert dims_mod.DIMS["..."] == {0, 1, 2, 4, 8, 16, 32, 64}

    def test_case_plan_flatten_deduplicates_in_first_series_order(self):
        first = Case((("X", 2),), (), torch.float32, (), frozenset())
        second = Case((("X", 4),), (), torch.float32, (), frozenset())
        planned = CasePlan(
            (
                CaseSeries("coverage", "coverage", (first,)),
                CaseSeries("adaptive:screen:X", "adaptive", (first, second), axis="X"),
            )
        )
        assert planned.cases() == [first, second]
        assert list(planned) == [first, second]

    def test_sample_cases_keeps_the_legacy_cases_alias(self, monkeypatch):
        from popcorn.core import dims as dims_mod

        def reference(x: Float[Tensor, "X"]):
            return x

        monkeypatch.setitem(dims_mod.DIMS, "X", {2, 4, 8})
        op = Dispatcher(reference)
        assert sample_cases(op, 4) == cases(op, 4)

    def test_smoke_cases_sample_only_bounded_cases_from_the_shared_plan(self, monkeypatch):
        from popcorn.core import dims as dims_mod

        def reference(x: Float[Tensor, "seq hidden"]):
            return x

        monkeypatch.setitem(dims_mod.DIMS, "seq", {8, 1024, 1_048_576})
        monkeypatch.setitem(dims_mod.DIMS, "hidden", {8, 1024})
        op = Dispatcher(reference)
        selected = smoke_cases(op, 4, max_bytes=64 * 2**10)

        assert 0 < len(selected) <= 4
        assert all(estimated_bytes(op, case) <= 64 * 2**10 for case in selected)
        assert all(dict(case.dims)["seq"] <= 512 for case in selected)
        assert {case.case_id for case in selected} <= {case.case_id for case in case_plan(op).flatten()}

    def test_smoke_cases_choose_the_smallest_allowed_cases(self, monkeypatch):
        from popcorn.core import dims as dims_mod

        def reference(x: Float[Tensor, "seq hidden"]):
            return x

        monkeypatch.setitem(dims_mod.DIMS, "seq", {8, 64, 512, 1024})
        monkeypatch.setitem(dims_mod.DIMS, "hidden", {8, 64})
        op = Dispatcher(reference)
        selected = smoke_cases(op, 4, allowed=lambda case: dict(case.dims)["seq"] >= 64)

        assert len(selected) == 4
        assert all(dict(case.dims)["seq"] >= 64 for case in selected)
        assert [estimated_bytes(op, case) for case in selected] == sorted(estimated_bytes(op, case) for case in selected)

    def test_estimated_bytes_prices_only_the_tensors_a_case_presents(self):
        def reference(
            x: Float[Tensor, "... seq hidden"],
            bias: Float[Tensor, "hidden"] | None = None,
        ):
            return x if bias is None else x + bias

        op = Dispatcher(reference)
        base = Case((("hidden", 8), ("seq", 4)), (2,), torch.float16, (), frozenset())
        assert estimated_bytes(op, base) == 2 * 4 * 8 * 2
        with_bias = Case((("hidden", 8), ("seq", 4)), (2,), torch.float16, (), frozenset({"bias"}))
        assert estimated_bytes(op, with_bias) == (2 * 4 * 8 + 8) * 2

    def test_budget_frontier_prunes_only_coordinate_wise_larger_shapes(self):
        def case(x, y, z, *, dtype=torch.float32, batch=(2,), flag=False):
            return Case(
                (("X", x), ("Y", y), ("Z", z)),
                batch,
                dtype,
                (("flag", flag),),
                frozenset(),
            )

        oom = case(4, 8, 16)
        frontier = BudgetFrontier([oom])
        assert dominates(oom, case(8, 8, 16))
        assert frontier.blocker(case(4, 16, 16)) == oom
        assert frontier.blocker(case(4, 8, 32)) == oom
        assert frontier.blocker(case(2, 8, 32)) is None
        assert frontier.blocker(case(8, 4, 32)) is None
        assert frontier.blocker(case(8, 8, 16, batch=(4,))) == oom
        assert frontier.blocker(case(8, 8, 16, dtype=torch.float16)) is None
        assert frontier.blocker(case(8, 8, 16, flag=True)) is None

        smaller = case(2, 4, 8)
        frontier.add(smaller)
        assert frontier.blocker(oom) == smaller

    def test_only_the_monotone_budgets_prune(self):
        # A timeout bounds larger shapes exactly as an OOM does, and rediscovering one costs the
        # whole per-case budget. A crash carries no such guarantee and must stay measurable.
        assert set(EXHAUSTED) == {"oom", "timeout"}

    def test_emits_screen_points_then_fixpoints(self, monkeypatch):
        from popcorn.core import dims as dims_mod

        def reference(x: Float[Tensor, "... D"], flag: bool = False):
            return x

        monkeypatch.setitem(dims_mod.DIMS, "D", {2, 4, 8})
        op = Dispatcher(reference)
        first_plan = adaptive_plan(op, [], "fast", effort="quick", device="cpu", grad=False)
        first = first_plan.flatten()
        assert all(series.kind == "adaptive" for series in first_plan.series)
        assert any(series.name == "adaptive:screen:D" and series.axis == "D" for series in first_plan.series)
        assert all(case.batch == () for case in first)  # production `...` anchor comes from dims.ANCHORS
        assert first == plan(op, [], "fast", effort="quick", device="cpu", grad=False)
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
