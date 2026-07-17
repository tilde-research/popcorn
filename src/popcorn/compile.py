"""torch.compile integration: inductor pattern matching that rewrites eager
subgraphs computing a popcorn kernel into its `torch.ops.popcorn` binding, so
compiled models route through the dispatcher without source changes.

Patterns are traced from each kernel's reference over grid-derived example
inputs, one variant per dtype, optional-tensor presence, and non-float scalar
value, on the device graphs will come from. Inference graphs match
forward-only patterns; training graphs match joint forward+backward ones, where
the rewrite lands the binding's autograd pair (`popcorn::<name>` forward,
`popcorn::<name>_backward` replay). A rewrite is declined when no backend
beyond the reference could serve the matched call, and a miss simply leaves
the graph to inductor. References whose graph topology depends on shape (a
python loop over sequence length) can never match: call the op directly
instead.

    import popcorn.compile
    popcorn.compile.enable()          # or enable(ops=["rms_norm", "swiglu"])
    model = torch.compile(model)
"""

from __future__ import annotations

import contextlib
import inspect
import itertools
import logging
import signal
import threading
import warnings
from collections import defaultdict
from collections.abc import Iterable, Iterator
from typing import Any, cast

import torch
from torch._inductor import config as inductor_config
from torch._inductor.custom_graph_pass import CustomGraphPass, get_hash_for_files
from torch._inductor.pattern_matcher import PatternMatcherPass, fwd_only, joint_fwd_bwd, register_replacement

import popcorn.core.library
from popcorn.bench import grid
from popcorn.bench.model import Case
from popcorn.core.dispatcher import KERNELS, Dispatcher

__all__ = ["enable", "disable"]

log = logging.getLogger(__name__)

PATTERNS = PatternMatcherPass(pass_name="popcorn")
DTYPES = (torch.bfloat16, torch.float16, torch.float32)
MAX_VARIANTS = 12
TRACE_BUDGET = 10.0  # seconds per pattern trace; loopy references unroll into graphs not worth matching
_MAGIC = 0.8377134043  # float scalars are traced with values nothing else uses
_TRACED: set[tuple[str, torch.dtype]] = set()


class _Expired(BaseException):
    """Raised when a _deadline lapses. A BaseException so that no broad
    `except Exception` -- the per-trace handler below, or torch's own inside
    the tracers -- can swallow it on the way up."""


@contextlib.contextmanager
def _deadline(seconds: float) -> Iterator[None]:
    """_Expired after `seconds` (main thread only; a no-op elsewhere)."""
    if not hasattr(signal, "SIGALRM") or threading.current_thread() is not threading.main_thread():
        yield
        return

    def expire(signum: Any, frame: Any) -> None:
        raise _Expired(f"trace exceeded {seconds:.0f}s")

    previous = signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _size(case: Case) -> int:
    # Degenerate dims trace degenerate patterns (a mean over one element folds
    # away), so sizes below 8 are heavily penalized rather than preferred.
    return sum(value if value >= 8 else value + 10_000 for _, value in case.dims) + sum(case.batch)


def _variants(op: Dispatcher, dtypes: frozenset[torch.dtype]) -> list[Case]:
    """Smallest case per (dtype, present optionals, non-float scalars),
    interleaved across dtypes so the cap starves none of them. Batched ops are
    traced with a rank-2 batch only: the matcher re-traces at the matched
    shapes, which generalizes across ranks, while rank-1 examples trace
    degenerate backwards (no batch reduction) that poison replacements."""
    batch = (2, 3) if any(... in spec.tokens for spec in op.specs) else ()
    best: dict[Any, Case] = {}
    for case in grid.cases(op):
        if case.dtype not in dtypes or case.batch != batch:
            continue
        scalars = tuple((name, value) for name, value in case.args if type(value) is not float)
        key = (case.dtype, case.present, scalars)
        if key not in best or _size(case) < _size(best[key]):
            best[key] = case
    by_dtype: dict[torch.dtype, list[Case]] = defaultdict(list)
    for case in best.values():
        by_dtype[case.dtype].append(case)
    interleaved = itertools.chain.from_iterable(itertools.zip_longest(*by_dtype.values()))
    return [case for case in interleaved if case is not None][:MAX_VARIANTS]


def _pattern_fns(op: Dispatcher, bound: dict[str, Any], symbolic: bool) -> tuple[Any, Any, list[Any], dict[str, float]]:
    """Search/replace functions over the case's free arguments: present tensors
    stay placeholders and everything else (absent optionals, ints, bools,
    strings) is baked in. Float scalars become magic-valued keyword args when
    `symbolic`, matching any user value -- but a magic value can steer a branchy
    reference (label smoothing, dropout) into ops the user's graph never runs,
    so a second variant bakes the floats as literals. `bound` is not mutated."""
    tensors = {spec.param for spec in op.specs}
    values = dict(bound)
    workaround: dict[str, float] = {}
    free: list[str] = []
    for index, (name, value) in enumerate(values.items()):
        if isinstance(value, torch.Tensor):
            free.append(name)
        elif symbolic and name not in tensors and type(value) is float:
            workaround[name] = values[name] = round(_MAGIC + index * 0.0137, 10)
            free.append(name)

    def full(args: tuple[Any, ...]) -> list[Any]:
        merged = dict(values, **dict(zip(free, args)))
        return [merged[name] for name in op._params]

    def search(*args: Any) -> Any:
        return op.reference(*full(args))

    def replace(*args: Any) -> Any:
        return op.torch_op(*full(args))

    signature = inspect.Signature([inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in free])
    search.__signature__ = replace.__signature__ = signature  # type: ignore[attr-defined]
    example = [values[name] for name in free]
    return search, replace, example, workaround


def _eligibility_check(op: Dispatcher, bound: dict[str, Any], grad: bool) -> Any:
    """Decline a rewrite unless a non-reference backend could serve the matched
    call: routing back to the reference through the custom-op boundary only
    adds overhead. Decided on the matched shapes; the runtime pick stays the
    tuner's."""

    def check(match: Any) -> bool:
        try:
            values = dict(bound)
            for name, node in match.kwargs.items():  # the pattern's free arguments, matched
                values[name] = node.meta["val"] if isinstance(node, torch.fx.Node) else node
            arguments = {name: values[name] for name in op._params}
            # Fake tensors do not carry requires_grad, so _eligible cannot see
            # grad-ness; filter forward-only backends by the pattern variant.
            candidates = op._eligible(op._values(arguments), arguments)
            return any(backend.name != "torch" and not (grad and backend.forward_only) for backend in candidates)
        except Exception as error:
            log.debug("%s: eligibility check failed, allowing rewrite: %s", op.name, error)
            return True

    return check


def _register(op: Dispatcher, case: Case, index: int) -> int:
    # Trace on the device graphs will come from: decompositions differ (CUDA
    # sdpa lowers to flash nodes, CPU to the math path), and a pattern traced
    # on the wrong device never matches.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    count = 0
    for trace_fn, grad, tag in ((fwd_only, False, "fwd"), (joint_fwd_bwd, True, "joint")):
        bound = op._bind(**grid.make_inputs(op, case, device=device, grad=grad))
        if grad and not any(isinstance(v, torch.Tensor) and v.requires_grad for v in bound.values()):
            continue
        for symbolic in (True, False):
            search, replace, example, workaround = _pattern_fns(op, bound, symbolic)
            if symbolic and not workaround:
                continue  # no float scalars: the baked variant is the only one
            search.__name__ = f"{op.name}_{index}_{tag}" + ("_sym" if symbolic else "")
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # failed traces spam anomaly-mode hints
                    registered = register_replacement(
                        search,
                        replace,
                        example,
                        trace_fn,
                        # torch 2.13's _PassDictsType protocol names the
                        # __getitem__ param `k`; PatternMatcherPass uses `item`,
                        # so it fails the protocol on spelling alone.
                        cast(Any, PATTERNS),
                        extra_check=_eligibility_check(op, bound, grad),
                        scalar_workaround=workaround or None,
                        skip_duplicates=True,
                    )
                count += bool(registered)
            except Exception as error:
                log.debug("%s: %s pattern for %s failed: %s", op.name, tag, case, error)
    return count


class _PopcornPass(CustomGraphPass):
    """Applies popcorn's replacements before inductor's own joint-graph passes,
    whose folding rewrites graphs away from the traced patterns; wraps whatever
    pass the user already had installed."""

    def __init__(self, inner: Any = None) -> None:
        self.inner = inner

    def __call__(self, graph: torch.fx.Graph) -> None:
        from torch._inductor.fx_passes.post_grad import remove_noop_ops

        if self.inner is not None:
            self.inner(graph)
        remove_noop_ops(graph)  # patterns are traced noop-free; align the graph before matching
        PATTERNS.apply(graph)

    def uuid(self) -> Any:
        stamps = "|".join(
            f"{name}:{dtype}:{KERNELS[name].fingerprint}" for name, dtype in sorted(_TRACED, key=str) if name in KERNELS
        )
        mine = get_hash_for_files((__file__, popcorn.core.library.__file__), extra=stamps)
        if self.inner is None:
            return mine
        inner = self.inner.uuid() if isinstance(self.inner, CustomGraphPass) else None
        return None if inner is None else (mine, inner)


def enable(ops: Iterable[str] | None = None, dtypes: Iterable[torch.dtype] = DTYPES) -> int:
    """Trace and register replacement patterns, then install the inductor pass.
    Returns the number of newly registered patterns. Idempotent per op; call
    again with more `ops` to widen coverage."""
    import popcorn.kernels  # noqa: F401  # populates KERNELS

    selected = [KERNELS[name] for name in ops] if ops is not None else list(KERNELS.values())
    total = 0
    for op in selected:
        missing = frozenset(dtype for dtype in dtypes if (op.name, dtype) not in _TRACED)
        if not missing or op.torch_op is None:
            continue
        _TRACED.update((op.name, dtype) for dtype in missing)  # marked even on failure; do not retry pathological traces
        registered = 0
        try:
            with _deadline(TRACE_BUDGET):
                for index, case in enumerate(_variants(op, missing)):
                    registered += _register(op, case, index)
        except _Expired:
            log.info("%s: compile patterns truncated after %.0fs", op.name, TRACE_BUDGET)
        except Exception as error:
            log.info("%s: no compile patterns: %s", op.name, error)
        log.debug("%s: %d patterns registered", op.name, registered)
        total += registered
    if not isinstance(inductor_config.joint_custom_pre_pass, _PopcornPass):
        inductor_config.joint_custom_pre_pass = _PopcornPass(inductor_config.joint_custom_pre_pass)
    return total


def disable() -> None:
    """Uninstall the pass; registered patterns are kept for a later enable()."""
    installed = inductor_config.joint_custom_pre_pass
    if isinstance(installed, _PopcornPass):
        inductor_config.joint_custom_pre_pass = installed.inner
