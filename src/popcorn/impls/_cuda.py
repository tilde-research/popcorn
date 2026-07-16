"""JIT harness for first-party CUDA kernels.

Sources live next to their python module as `impls/<op>.cu` and compile on
first use into torch's extension cache, so a build happens once per source
change and torch/CUDA version. This module is only imported when an impl
resolves at first dispatch; eligibility is `impls.cuda_toolkit`.
"""

import functools
from pathlib import Path
from types import ModuleType

from torch.utils import cpp_extension


@functools.cache
def load(op: str) -> ModuleType:
    return cpp_extension.load(
        name=f"popcorn_{op}",
        sources=[str(Path(__file__).parent / f"{op}.cu")],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )
