"""Markdown tables, badges, and README refresh derived from report records."""

from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from statistics import median
from typing import Any

from popcorn.bench.model import Record
from popcorn.bench.store import BUNDLED_REPORTS, read

PACKAGE = Path(__file__).parents[1]
BADGE_MARKERS = ("<!-- popcorn:badges -->", "<!-- /popcorn:badges -->")


def _readme() -> Path:
    """The repository README: next to the package in a flat checkout, above `src/` in a src layout."""
    for parent in (PACKAGE, *PACKAGE.parents):
        candidate = parent / "README.md"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("no README.md found above the popcorn package; pass refresh(readme=...)")


def matrix(records: Iterable[Record]) -> str:
    header = (
        "op",
        "impl",
        "device",
        "pass",
        "skip",
        "fail",
        "crash",
        "error",
        "bench error",
        "max fwd err",
        "max grad err",
        "fwd speedup",
        "bwd speedup",
        "peak mem",
    )
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    groups = defaultdict(list)
    for record in records:
        groups[record.op, record.impl, record.environment.device].append(record)
    for key in sorted(groups):
        group = groups[key]
        cells: list = list(key)
        cells += [
            sum(record.result.status == status for record in group) for status in ("pass", "skip", "fail", "crash", "error")
        ]
        cells.append(sum(bool(record.result.bench_error) for record in group))
        for direction in ("fwd", "bwd"):
            error = max(
                (gauge.err for record in group for gauge in getattr(record.result, direction).values()),
                default=None,
            )
            cells.append(f"{error:.1e}" if error is not None else "-")
        for direction in ("fwd", "bwd"):
            ratios = [
                result.bench[f"ref_{direction}_ms"] / result.bench[f"{direction}_ms"]
                for record in group
                if (result := record.result).bench.get(f"{direction}_ms") and result.bench.get(f"ref_{direction}_ms")
            ]
            cells.append(f"{median(ratios):.2f}x" if ratios else "-")
        peaks = [
            (
                max(bench.get("fwd_mem_mb", 0), bench.get("bwd_mem_mb", 0)),
                max(bench.get("ref_fwd_mem_mb", 0), bench.get("ref_bwd_mem_mb", 0)),
            )
            for record in group
            if (bench := record.result.bench).get("fwd_mem_mb") is not None
        ]
        ratios = [reference / mine for mine, reference in peaks if mine]
        cells.append(f"{max(mine for mine, _ in peaks):.1f}MB ({median(ratios):.2f}x)" if ratios else "-")
        lines.append("| " + " | ".join(str(cell) for cell in cells) + " |")
    return "\n".join(lines)


def badges(records: Iterable[Record], ops: Mapping[str, Any]) -> str:
    """Shields.io badge row: registered counts plus the recorded grid size."""
    names = [impl.name for op in ops.values() for impl in op._impls]
    counts = (
        ("kernels", str(len(ops))),
        ("backends", str(len({name.split(":")[0] for name in names if name != "torch"}))),
        ("implementations", str(len(names))),
        ("grid rows", f"{sum(1 for _ in records):,}".replace(",", "%2C")),
    )
    images = "\n".join(
        f'  <img src="https://img.shields.io/badge/{label.replace(" ", "%20")}-{value}-blue" alt="{label}"/>'
        for label, value in counts
    )
    return f'<p align="center">\n{images}\n</p>'


def _replaced(text: str, markers: tuple[str, str], body: str) -> str:
    start, end = markers
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    return head + f"{start}\n{body}\n{end}" + tail


def update_readme(path: Path | str, badge_row: str, *, write: bool = True) -> bool:
    """Splice the badge row between the markers. True when the file is, or would be, changed."""
    path = Path(path)
    text = path.read_text()
    if not all(marker in text for marker in BADGE_MARKERS):
        return False
    updated = _replaced(text, BADGE_MARKERS, badge_row)
    if updated == text:
        return False
    if write:
        path.write_text(updated)
    return True


def refresh(
    records: Iterable[Record] | None = None,
    ops: Mapping[str, Any] = {},
    readme: Path | str | None = None,
    *,
    write: bool = True,
) -> tuple[str, bool]:
    """Regenerate the README badges from the bundled reports.

    Returns the detail matrix and whether the badge block is, or would be, changed.
    `write=False` reports staleness without touching the file.
    """
    rows = read(BUNDLED_REPORTS) if records is None else list(records)
    stale = update_readme(readme if readme is not None else _readme(), badges(rows, ops), write=write)
    return matrix(rows), stale
