"""Markdown tables, badges, and README refresh derived from report records."""

from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from statistics import median
from typing import Any

from popcorn.bench.model import Record
from popcorn.bench.store import BUNDLED_REPORTS, read

PACKAGE = Path(__file__).parents[1]
MARKERS = ("<!-- popcorn:matrix -->", "<!-- /popcorn:matrix -->")
BADGE_MARKERS = ("<!-- popcorn:badges -->", "<!-- /popcorn:badges -->")


def _readme() -> Path:
    """The repository README: next to the package in a flat checkout, above `src/` in a src layout."""
    for parent in (PACKAGE, *PACKAGE.parents):
        candidate = parent / "README.md"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("no README.md found above the popcorn package; pass refresh(readme=...)")


def support(records: Iterable[Record], ops: Iterable[str] = ()) -> str:
    records = list(records)
    ops = sorted(set(ops) | {record.op for record in records})
    backends = sorted({record.backend for record in records})
    grouped = defaultdict(list)
    for record in records:
        grouped[record.op, record.backend].append(record)
    columns = ["kernel", "torch", *backends]
    lines = ["| " + " | ".join(columns) + " |", "|---|" + ":---:|" * (len(columns) - 1)]
    for op in ops:
        cells = ["✔"]
        for backend in backends:
            group = grouped[op, backend]
            passed = sum(record.result.status == "pass" for record in group)
            cells.append("✘" if not passed else "✔" if passed == len(group) else "✔*")
        lines.append(f"| `{op}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def matrix(records: Iterable[Record]) -> str:
    header = (
        "op",
        "backend",
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
        groups[record.op, record.backend, record.environment.device].append(record)
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
    backends = {backend.name for op in ops.values() for backend in op._backends}
    implementations = sum(backend.name != "torch" for op in ops.values() for backend in op._backends)
    counts = (
        ("kernels", str(len(ops))),
        ("backends", str(len(backends))),
        ("implementations", str(implementations)),
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


def update_readme(path: Path | str, table: str, badge_row: str | None = None) -> None:
    path = Path(path)
    text = path.read_text() if path.exists() else "# popcorn\n"
    if all(marker in text for marker in MARKERS):
        text = _replaced(text, MARKERS, table)
    else:
        text = text.rstrip() + "\n\n## Verified backends\n\n" + f"{MARKERS[0]}\n{table}\n{MARKERS[1]}" + "\n"
    if badge_row is not None and all(marker in text for marker in BADGE_MARKERS):
        text = _replaced(text, BADGE_MARKERS, badge_row)
    path.write_text(text)


def refresh(records: Iterable[Record] | None = None, ops: Iterable[str] = (), readme: Path | str | None = None) -> str:
    """Regenerate the README support matrix and badges from the bundled reports; returns the detail matrix."""
    rows = read(BUNDLED_REPORTS) if records is None else list(records)
    badge_row = badges(rows, ops) if isinstance(ops, Mapping) else None
    update_readme(readme if readme is not None else _readme(), support(rows, ops), badge_row)
    return matrix(rows)
