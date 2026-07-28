"""Public benchmarking API: the comparison harness and its data model."""

from popcorn.bench.compare import compare, report
from popcorn.bench.grid import cases, make_inputs
from popcorn.bench.model import Case, Gauge, Record, Result

__all__ = ["compare", "report", "Case", "Gauge", "Record", "Result", "cases", "make_inputs"]
