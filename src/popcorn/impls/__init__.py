"""First-party kernels, registered as the `popcorn` backend.

One module per op, named after it. Each module exposes a callable named
exactly like the op with the reference signature (autograd handled inside),
so it registers source-only: `op.register("popcorn", source="popcorn.impls.<op>.<op>")`.
CUDA-backed impls also pass `predicate=cuda_toolkit`.

This package init stays import-light; kernel registrations import it eagerly.
"""

import functools
import os
import shutil
from pathlib import Path


@functools.cache
def _nvcc() -> bool:
    home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    return bool(home or shutil.which("nvcc") or Path("/usr/local/cuda/bin/nvcc").exists())


def cuda_toolkit(**_: object) -> bool:
    return _nvcc()
