#!/usr/bin/env python
"""Export the kernel evidence catalog to site/public/data.

Usage:
    uv run python scripts/update_site.py [--out DIRECTORY] [--check]
"""

import argparse
import inspect
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch

import popcorn.kernels  # noqa: F401  # populates KERNELS
from popcorn import KERNELS
from popcorn.bench.compare import FLOORS
from popcorn.bench.grid import CasePlan, case_plan
from popcorn.bench.model import Case, Record
from popcorn.bench.store import BUNDLED_REPORTS, bundled_path, matching, prefer, read_file

BENCH_KEYS = (
    "fwd_ms",
    "bwd_ms",
    "ref_fwd_ms",
    "ref_bwd_ms",
    "fwd_mem_mb",
    "bwd_mem_mb",
    "ref_fwd_mem_mb",
    "ref_bwd_mem_mb",
)


def errors(record: Record) -> dict[str, float]:
    """Worst abs/rel error per pass plus the validation cutoff, as in Gauge.verdict.

    Non-finite values (e.g. an infinite budget when the low-precision reference
    overflows) are dropped: bare Infinity is not valid JSON.
    """
    floor = FLOORS.get(getattr(torch, record.config["dtype"], None), 2e-5)
    out = {}
    for pass_, gauges in (("fwd", record.result.fwd), ("bwd", record.result.bwd)):
        if not gauges:
            continue
        out[f"{pass_}_err"] = max(g.err for g in gauges.values())
        out[f"{pass_}_cut"] = max(max(2 * g.budget, floor * g.scale) for g in gauges.values())
        out[f"{pass_}_rel"] = max(g.err / g.scale for g in gauges.values())
        out[f"{pass_}_rel_cut"] = max(max(2 * g.budget, floor * g.scale) / g.scale for g in gauges.values())
    return {key: value for key, value in out.items() if math.isfinite(value)}


def _latest(records: Iterable[Record]) -> list[Record]:
    """Best row per implementation, device, case, and gradient mode."""
    latest: dict[tuple[str, str, str, bool], Record] = {}
    for record in records:
        key = (record.impl, record.environment.device, record.case_id, record.result.grad)
        if key not in latest or prefer(record, latest[key]):
            latest[key] = record
    return list(latest.values())


def _current(op: Any, records: Iterable[Record]) -> tuple[list[Record], int]:
    """Newest current-fingerprint rows and the number of stale rows omitted."""
    implementations = {impl.name: impl for impl in op._impls}
    records = list(records)
    current = []
    for record in records:
        impl = implementations.get(record.impl)
        if (
            impl is not None
            and matching(op.fingerprint, record.environment.ref_hash)
            and matching(impl.fingerprint, record.environment.impl_hash)
        ):
            current.append(record)
    return _latest(current), len(records) - len(current)


def _case(case_id: str, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": case_id,
        **{name: config[name] for name in ("dims", "batch", "dtype", "args", "present")},
        "results": [],
    }


def _result(record: Record) -> dict[str, Any]:
    bench = record.result.bench
    return {
        "impl": record.impl,
        "device": record.environment.device,
        "grad": record.result.grad,
        "status": record.result.status,
        "benchmarked": record.result.benchmarked,
        **({"bench_error": True} if record.result.bench_error else {}),
        **{key: bench[key] for key in BENCH_KEYS if key in bench and math.isfinite(bench[key])},
        **errors(record),
    }


def _x(case: Case, axis: str) -> int:
    if axis == "...":
        return case.batch[0] if case.batch else 0
    return dict(case.dims)[axis]


def _fixed(series: Any) -> dict[str, Any]:
    """Properties that are genuinely invariant after relation repairs."""
    cases = series.cases
    first_dims = dict(cases[0].dims)
    dims = {
        name: value
        for name, value in first_dims.items()
        if name != series.axis and all(dict(case.dims).get(name) == value for case in cases)
    }
    first_args = dict(cases[0].args)
    args = {name: value for name, value in first_args.items() if all(dict(case.args).get(name) == value for case in cases)}
    batch = list(cases[0].batch) if series.axis != "..." and all(case.batch == cases[0].batch for case in cases) else None
    return {"dims": dims, "batch": batch, "args": args, "present": sorted(series.present)}


def _curves(plan: CasePlan) -> list[dict[str, Any]]:
    curves = []
    for series in plan.series:
        if series.kind != "curve" or series.axis is None or series.dtype is None or not series.cases:
            continue
        curves.append(
            {
                "id": series.name,
                "profile": series.profile,
                "axis": series.axis,
                "dtype": str(series.dtype).removeprefix("torch."),
                "variant": series.name.rsplit(":", 1)[-1],
                "fixed": _fixed(series),
                "case_ids": [case.case_id for case in series.cases],
            }
        )
    return curves


def _frontiers(curves: list[dict[str, Any]], planned: dict[str, Case], records: list[Record]) -> list[dict[str, Any]]:
    """Compact per-implementation curve endpoints, including the first terminal result."""
    by_case: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        by_case[record.case_id].append(record)
    summaries = []
    for curve in curves:
        points = {case_id: _x(planned[case_id], curve["axis"]) for case_id in curve["case_ids"]}
        groups: dict[tuple[str, str, bool], list[tuple[int, Record]]] = defaultdict(list)
        for case_id, x in points.items():
            for record in by_case.get(case_id, ()):
                groups[(record.impl, record.environment.device, record.result.grad)].append((x, record))
        for (impl, device, grad), rows in sorted(groups.items()):
            rows.sort(key=lambda item: item[0])
            passes = [x for x, record in rows if record.result.status == "pass" and not record.result.bench_error]
            pass_max = max(passes, default=None)
            terminal = next(
                (
                    (x, record)
                    for x, record in rows
                    if (pass_max is None or x > pass_max) and (record.result.status != "pass" or record.result.bench_error)
                ),
                None,
            )
            summaries.append(
                {
                    "curve_id": curve["id"],
                    "impl": impl,
                    "device": device,
                    "grad": grad,
                    "observed": len(rows),
                    "timed": sum(
                        record.result.status == "pass" and record.result.benchmarked and "fwd_ms" in record.result.bench
                        for _, record in rows
                    ),
                    "observed_max": max(x for x, _ in rows),
                    "pass_max": pass_max,
                    **(
                        {
                            "terminal": {
                                "x": terminal[0],
                                "status": "bench_error" if terminal[1].result.status == "pass" else terminal[1].result.status,
                            }
                        }
                        if terminal is not None
                        else {}
                    ),
                }
            )
    return summaries


def evidence(op: Any, records: Iterable[Record]) -> dict[str, Any]:
    """Join schema-3 rows to the current plan without changing persisted records."""
    records, stale = _current(op, records)
    plan = case_plan(op)
    coverage = next(series for series in plan.series if series.kind == "coverage")
    curves = _curves(plan)
    curve_ids = {case_id for curve in curves for case_id in curve["case_ids"]}
    planned = {case.case_id: case for series in plan.series if series.kind == "curve" for case in series.cases}
    cases = {case_id: _case(case_id, case.config()) for case_id, case in planned.items()}
    for record in records:
        cases.setdefault(record.case_id, _case(record.case_id, record.config))
        cases[record.case_id]["results"].append(_result(record))
    for item in cases.values():
        item["results"].sort(key=lambda result: (result["impl"], result["device"], result["grad"]))

    observed = {record.case_id for record in records}
    coverage_ids = {case.case_id for case in coverage.cases}
    samples = sorted((observed & coverage_ids) - curve_ids | (observed - coverage_ids - curve_ids))
    statuses = Counter(record.result.status for record in records)
    by_impl = {}
    for impl in sorted({record.impl for record in records}):
        impl_records = [record for record in records if record.impl == impl]
        by_impl[impl] = {
            "results": len(impl_records),
            "cases": len({record.case_id for record in impl_records}),
            "timed": sum(
                record.result.status == "pass" and record.result.benchmarked and "fwd_ms" in record.result.bench
                for record in impl_records
            ),
            "statuses": dict(sorted(Counter(record.result.status for record in impl_records).items())),
        }
    return {
        "cases": dict(sorted(cases.items())),
        "curves": curves,
        "samples": samples,
        "frontiers": _frontiers(curves, planned, records),
        "coverage": {
            "planned_samples": len(coverage_ids),
            "planned_curve_cases": len(curve_ids),
            "observed_cases": len(observed),
            "results": len(records),
            "timed": sum(
                record.result.status == "pass" and record.result.benchmarked and "fwd_ms" in record.result.bench
                for record in records
            ),
            "statuses": dict(sorted(statuses.items())),
            "by_impl": by_impl,
        },
        "freshness": {
            "latest": max((record.environment.ts for record in records), default=None),
            "current_results": len(records),
            "stale_results_omitted": stale,
            "ref_hash": op.fingerprint,
            "impl_hashes": {impl.name: impl.fingerprint for impl in op._impls},
        },
    }


def card(op: Any) -> dict[str, Any]:
    return {
        "name": op.name,
        "summary": op.summary,
        "math": op.math,
        "citations": [{"label": label, "url": url} for label, url in op.citations],
        "tags": sorted(op.tags),
        "params": [
            {"name": p.name} | ({} if p.default is inspect.Parameter.empty else {"default": repr(p.default)})
            for p in inspect.signature(op.reference).parameters.values()
        ],
        "impls": [
            {
                "name": b.name,
                "source": b.source,
                "forward_only": b.forward_only,
            }
            for b in op._impls
        ],
    }


def payloads() -> dict[str, str]:
    """Every file the site consumes, keyed by filename: one card per kernel plus the index."""
    files, index = {}, []
    for name, op in sorted(KERNELS.items()):
        records = read_file(bundled_path(BUNDLED_REPORTS, name))
        data = card(op) | {"evidence": evidence(op, records)}
        files[f"{name}.json"] = json.dumps(data, separators=(",", ":"))
        index.append(
            {
                "name": name,
                "summary": op.summary,
                "tags": sorted(op.tags),
                "impls": [b.name for b in op._impls],
                "rows": data["evidence"]["coverage"]["results"],
                "cases": data["evidence"]["coverage"]["observed_cases"],
                "curves": len(data["evidence"]["curves"]),
            }
        )
    files["index.json"] = json.dumps(index, separators=(",", ":"))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).parents[1] / "site" / "public" / "data")
    parser.add_argument("--check", action="store_true", help="fail instead of writing when generated data is stale")
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    files = payloads()
    stale = [name for name, text in files.items() if not (out / name).exists() or (out / name).read_text() != text]
    extra = sorted(path.name for path in out.glob("*.json") if path.name not in files)
    if args.check:
        if stale or extra:
            detail = [*(f"stale: {name}" for name in stale), *(f"extra: {name}" for name in extra)]
            raise SystemExit("site data is stale; run scripts/update_site.py\n  " + "\n  ".join(detail))
        print(f"{len(files) - 1} kernels current in {out}")
        return
    for name, text in files.items():
        (out / name).write_text(text)
    for name in extra:
        (out / name).unlink()
    print(f"{len(files) - 1} kernels -> {out}")


if __name__ == "__main__":
    main()
