"""Export kernel cards and slimmed benchmark rows as JSON for the docs site.

Writes site/public/data/index.json (card grid + search) and one
<kernel>.json per op (card page + plots). Run from the repo root:

    python scripts/export_site_data.py [--out site/public/data]
"""

import argparse
import inspect
import json
import math
from pathlib import Path

import torch

import popcorn.kernels  # noqa: F401  # populates KERNELS
from popcorn import KERNELS
from popcorn.bench.compare import FLOORS
from popcorn.bench.store import BUNDLED_REPORTS, read_file

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


def errors(record) -> dict:
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


def rows(name: str) -> list[dict]:
    """Newest benchmarked row per (backend, device, case, grad), slimmed to plot fields."""
    latest = {}
    for record in read_file(BUNDLED_REPORTS / f"{name}.jsonl"):
        key = (record.backend, record.environment.device, record.case_id, record.result.grad)
        if key not in latest or record.environment.ts > latest[key].environment.ts:
            latest[key] = record
    slim = [
        {
            "backend": record.backend,
            "device": record.environment.device,
            "dtype": record.config["dtype"],
            "grad": record.result.grad,
            **{k: record.config[k] for k in ("dims", "batch", "args", "present")},
            **{k: bench[k] for k in BENCH_KEYS if k in bench},
            **errors(record),
        }
        for record in latest.values()
        if (bench := record.result.bench).get("fwd_ms")
    ]
    return sorted(slim, key=lambda row: json.dumps(row, sort_keys=True))


def card(op) -> dict:
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
        "backends": [
            {
                "name": b.name,
                "source": b.source,
                "forward_only": b.forward_only,
                "supports": {dim: str(constraint) for dim, constraint in b.supports.items()},
            }
            for b in op._backends
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).parents[1] / "site" / "public" / "data")
    out = parser.parse_args().out
    out.mkdir(parents=True, exist_ok=True)

    index = []
    for name, op in sorted(KERNELS.items()):
        data = card(op) | {"rows": rows(name)}
        (out / f"{name}.json").write_text(json.dumps(data, separators=(",", ":")))
        index.append(
            {
                "name": name,
                "summary": op.summary,
                "tags": sorted(op.tags),
                "backends": [b.name for b in op._backends],
                "rows": len(data["rows"]),
            }
        )
    (out / "index.json").write_text(json.dumps(index, separators=(",", ":")))
    print(f"{len(index)} kernels -> {out}")


if __name__ == "__main__":
    main()
