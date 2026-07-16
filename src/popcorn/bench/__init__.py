"""Public benchmarking API: the comparison harness and its data model."""

from popcorn.bench.compare import compare
from popcorn.bench.grid import cases, make_inputs
from popcorn.bench.model import Case, Gauge, Result

__all__ = ["compare", "Case", "Gauge", "Result", "cases", "make_inputs"]
