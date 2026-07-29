"""Integration smoke tests for the FLA and Liger backend pair."""

import os

import pytest
import torch

import popcorn.kernels  # noqa: F401  # populates KERNELS
from popcorn import KERNELS
from popcorn.core.sources import available, installed_version, resolve

PAIR = ("fla", "liger")

pytestmark = pytest.mark.skipif(
    not all(available(name) for name in PAIR),
    reason=f"needs {' and '.join(PAIR)} installed together: uv pip install -e '.[{','.join(PAIR)}]'",
)
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def shared() -> list:
    """Ops both backends implement: where the two can actually contradict each other."""
    return [op for op in KERNELS.values() if set(PAIR) <= set(op.available_backends())]


def test_the_pair_overlaps_on_the_ops_it_is_chosen_for():
    """A pair that shares no op would install two libraries and integrate nothing."""
    names = [op.name for op in shared()]
    assert names, f"{PAIR} no longer share an op; pick a different pair for this tier"
    assert "rms_norm" in names  # the anchor the CUDA checks below use


def test_both_libraries_resolve_in_one_interpreter():
    """The collision check: every source path of both backends imports and resolves
    with the other backend present. A shadowed module or a version skew that only
    appears alongside its neighbour shows up here and nowhere else."""
    broken = {}
    for op in shared():
        for impl in op._impls:
            if impl.name not in PAIR or impl.source is None:
                continue
            try:
                assert resolve(impl.source) is not None
            except Exception as error:
                broken[f"{op.name}:{impl.name}"] = f"{type(error).__name__}: {error}"
    assert not broken, f"source paths that fail with both backends installed: {broken}"


def test_both_backends_report_a_version():
    assert all(installed_version(name) for name in PAIR), {name: installed_version(name) for name in PAIR}


@requires_cuda
@pytest.mark.parametrize("backend", PAIR)
def test_each_backend_matches_the_reference_with_the_other_installed(backend):
    """The kernel tier runs this per backend in isolation; here both are present, so a
    failure means the neighbour changed the answer."""
    limit = int(os.getenv("POPCORN_TEST_LIMIT", "8"))
    bad = []
    for op in shared():
        for record in op.bench.run_backend(backend, trials=2, limit=limit):
            if record.result.status in ("fail", "crash", "error") or record.result.bench_error:
                bad.append(f"[{record.result.status}] {op.name}:{backend} {record.case}: {record.result.reason}")
    assert not bad, "\n".join(bad)


@requires_cuda
def test_dispatch_chooses_between_the_two_and_stays_correct():
    """Unforced dispatch with two competing implementations installed: whichever the
    policy admits and ranks first must still answer like the reference. Ranking is
    otherwise only exercised against synthetic records in tests/core."""
    op = KERNELS["rms_norm"]
    x = torch.randn(4, 128, 512, device="cuda", dtype=torch.float16)
    weight = torch.ones(512, device="cuda", dtype=torch.float16)
    torch.testing.assert_close(op(x, weight), op.reference(x, weight), rtol=1e-2, atol=1e-2)
