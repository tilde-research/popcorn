"""Exactness and timing comparison between an implementation and its reference."""

import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from itertools import chain, repeat
from statistics import median
from typing import Any

import torch

from popcorn.bench.model import Gauge, Record, Result

FLOORS = {torch.float64: 1e-10, torch.float32: 2e-5, torch.float16: 1e-3, torch.bfloat16: 2e-2}

try:
    from triton.testing import do_bench as _do_bench
except ImportError:  # CPU-only / no triton: fall back to the hand-rolled timer
    _do_bench = None


def _clone(inputs: Mapping[str, Any], dtype: torch.dtype | None = None, grad: bool = True) -> dict[str, Any]:
    def clone(value: Any) -> Any:
        if not isinstance(value, torch.Tensor):
            return value
        tensor = value.detach().clone()
        if dtype is not None and tensor.is_floating_point():
            tensor = tensor.to(dtype)
        return tensor.requires_grad_(grad and tensor.is_floating_point())

    return {name: clone(value) for name, value in inputs.items()}


def _outputs(value: Any) -> tuple[Any, ...]:
    return value if isinstance(value, tuple) else (value,)


def _graded(outputs: Sequence[Any]) -> list[int]:
    return [index for index, output in enumerate(outputs) if isinstance(output, torch.Tensor) and output.is_floating_point()]


def _structure(outputs: Sequence[Any]) -> tuple[Any, ...]:
    return tuple(
        (tuple(output.shape), str(output.dtype)) if isinstance(output, torch.Tensor) else type(output).__name__
        for output in outputs
    )


def _error(value: torch.Tensor, truth: torch.Tensor) -> float:
    actual, expected = value.detach().double(), truth.detach()
    equal = (actual == expected) | (actual.isnan() & expected.isnan())
    diff = torch.where(equal, 0.0, (actual - expected).abs())
    error = diff.max().item() if diff.numel() else 0.0
    return error if math.isfinite(error) else float("inf")


def _cotangents(outputs: Sequence[torch.Tensor], seed: int) -> list[torch.Tensor]:
    generator = torch.Generator(outputs[0].device).manual_seed(seed)
    return [torch.empty_like(output).normal_(generator=generator) for output in outputs]


def _grads(
    outputs: Sequence[torch.Tensor],
    inputs: Mapping[str, Any],
    cotangents: Sequence[torch.Tensor],
    retain: bool = False,
) -> dict[str, torch.Tensor | None]:
    tensors = {name: value for name, value in inputs.items() if isinstance(value, torch.Tensor) and value.requires_grad}
    grads = torch.autograd.grad(
        outputs,
        tuple(tensors.values()),
        grad_outputs=cotangents,
        allow_unused=True,
        retain_graph=retain,
    )
    return dict(zip(tensors, grads))


def _timed(fn: Callable[[], Any], device: torch.device | None) -> tuple[float, float | None]:
    if device is not None:
        with torch.cuda.device(device):
            base = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize(device)
            return start.elapsed_time(end), (torch.cuda.max_memory_allocated(device) - base) / 2**20
    started = time.perf_counter()
    fn()
    return (time.perf_counter() - started) * 1e3, None


def _measure(fn: Callable[[], Any], device: torch.device | None, repeats: int, warmup: int) -> dict[str, float]:
    """Time `fn`; on CUDA prefer Triton's do_bench (L2 clear, quantiles)."""
    memory: list[float] = []
    if device is not None and _do_bench is not None:
        try:
            with torch.cuda.device(device):
                base = torch.cuda.memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)
                quantiles = _do_bench(
                    fn,
                    warmup=max(5, warmup * 5),
                    rep=max(10, repeats * 10),
                    quantiles=[0.2, 0.5, 0.8],
                )
                peak = (torch.cuda.max_memory_allocated(device) - base) / 2**20
            raw = quantiles if isinstance(quantiles, (list, tuple)) else (quantiles, quantiles, quantiles)
            q20, mid, q80 = raw[0], raw[1], raw[2]
            if q20 is None or mid is None or q80 is None:
                raise TypeError("do_bench returned None quantile")
            return {
                "ms": round(float(mid), 6),
                "q20_ms": round(float(q20), 6),
                "q80_ms": round(float(q80), 6),
                "mem_mb": round(peak, 3),
            }
        except Exception:
            pass
    times = []
    for index in range(warmup + repeats):
        elapsed, peak = _timed(fn, device)
        if index >= warmup:
            times.append(elapsed)
            if peak is not None:
                memory.append(peak)
    result = {"ms": round(median(times), 6)}
    if memory:
        result["mem_mb"] = round(max(memory), 3)
    return result


def _benchmark(
    mine: Callable[..., Any],
    reference: Callable[..., Any],
    inputs: Mapping[str, Any],
    backward: bool,
    repeats: int,
    warmup: int,
    reference_bench: Mapping[str, float] | None = None,
) -> dict[str, float]:
    tensor = next((value for value in inputs.values() if isinstance(value, torch.Tensor)), None)
    device = tensor.device if tensor is not None and tensor.device.type == "cuda" else None
    bench: dict[str, float] = {}
    forwards = (("", mine),) if reference_bench is not None else (("", mine), ("ref_", reference))
    for prefix, forward in forwards:
        side = _clone(inputs, grad=backward)
        measured = _measure(lambda: forward(**side), device, repeats, warmup)
        bench[f"{prefix}fwd_ms"] = measured["ms"]
        if "q20_ms" in measured:
            bench[f"{prefix}fwd_q20_ms"], bench[f"{prefix}fwd_q80_ms"] = measured["q20_ms"], measured["q80_ms"]
        if "mem_mb" in measured:
            bench[f"{prefix}fwd_mem_mb"] = measured["mem_mb"]
        if not backward:
            continue
        outputs = _outputs(forward(**side))
        if not (graded := _graded(outputs)):
            continue
        picked = [outputs[index] for index in graded]
        cotangents = _cotangents(picked, 0)
        measured = _measure(lambda: _grads(picked, side, cotangents, retain=True), device, repeats, warmup)
        bench[f"{prefix}bwd_ms"] = measured["ms"]
        if "q20_ms" in measured:
            bench[f"{prefix}bwd_q20_ms"], bench[f"{prefix}bwd_q80_ms"] = measured["q20_ms"], measured["q80_ms"]
        if "mem_mb" in measured:
            bench[f"{prefix}bwd_mem_mb"] = measured["mem_mb"]
    if reference_bench is not None:
        cached = reference_bench or {
            f"ref_{name}": value for name, value in bench.items() if name.startswith(("fwd_", "bwd_"))
        }
        bench.update({name: value for name, value in cached.items() if name.startswith("ref_")})
    return bench


def _halt(result: Result, status: str, reason: str) -> Result:
    result.status, result.reason = status, reason
    return result


def compare_inputs(
    mine: Callable[..., Any],
    reference: Callable[..., Any],
    trials: Iterable[Mapping[str, Any]],
    *,
    backward: bool = True,
    benchmark: bool = True,
    repeats: int = 10,
    warmup: int = 2,
    reference_bench: Mapping[str, float] | None = None,
) -> Result:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    iterator = iter(trials)
    try:
        first = next(iterator)
    except StopIteration:
        raise ValueError("compare needs at least one input mapping")
    result = Result(grad=backward)

    for index, inputs in enumerate(chain((first,), iterator)):
        truth_inputs = _clone(inputs, torch.float64, backward)
        budget_inputs = _clone(inputs, grad=backward)
        mine_inputs = _clone(inputs, grad=backward)
        try:
            truth_outputs = _outputs(reference(**truth_inputs))
            budget_outputs = _outputs(reference(**budget_inputs))
        except torch.OutOfMemoryError as error:
            return _halt(result, "oom", f"reference: {error}")
        except Exception as error:
            return _halt(result, "error", f"reference: {type(error).__name__}: {error}")
        try:
            mine_outputs = _outputs(mine(**mine_inputs))
        except torch.OutOfMemoryError as error:
            return _halt(result, "oom", str(error))
        except Exception as error:
            return _halt(result, "crash", f"{type(error).__name__}: {error}")

        if _structure(mine_outputs) != _structure(budget_outputs):
            return _halt(
                result, "fail", f"output structure {_structure(mine_outputs)} != reference {_structure(budget_outputs)}"
            )
        for out, (ours, target) in enumerate(zip(mine_outputs, budget_outputs)):
            if isinstance(target, torch.Tensor):
                if not target.is_floating_point() and not torch.equal(ours, target):
                    return _halt(result, "fail", f"out{out}: values differ from reference")
            elif ours != target:
                return _halt(result, "fail", f"out{out}: {ours!r} != reference {target!r}")

        graded = _graded(budget_outputs)
        for out in graded:
            result.fwd.setdefault(f"out{out}", Gauge()).note(
                _error(mine_outputs[out], truth_outputs[out]),
                _error(budget_outputs[out], truth_outputs[out]),
                truth_outputs[out],
            )
        if not graded or not backward:
            continue

        cotangents = _cotangents([truth_outputs[out] for out in graded], index)

        def pull(outputs: Sequence[Any], side_inputs: Mapping[str, Any]) -> dict[str, torch.Tensor | None]:
            cast = [cotangent.to(outputs[out].dtype) for cotangent, out in zip(cotangents, graded)]
            return _grads([outputs[out] for out in graded], side_inputs, cast)

        try:
            truth_grads = _grads([truth_outputs[out] for out in graded], truth_inputs, cotangents)
            budget_grads = pull(budget_outputs, budget_inputs)
        except Exception as error:
            return _halt(result, "error", f"reference backward: {type(error).__name__}: {error}")
        try:
            mine_grads = pull(mine_outputs, mine_inputs)
        except Exception as error:
            return _halt(result, "crash", f"backward: {type(error).__name__}: {error}")

        for name, truth_grad in truth_grads.items():
            budget_grad, mine_grad = budget_grads.get(name), mine_grads.get(name)
            if truth_grad is None:
                if mine_grad is not None and torch.count_nonzero(mine_grad).item():
                    return _halt(result, "fail", f"grad {name}: unexpected")
                continue
            if budget_grad is None:
                return _halt(result, "error", f"reference grad {name}: missing at native dtype")
            if mine_grad is None:
                return _halt(result, "fail", f"grad {name}: missing")
            result.bwd.setdefault(name, Gauge()).note(
                _error(mine_grad, truth_grad),
                _error(budget_grad, truth_grad),
                truth_grad,
            )

    floor = max(
        (
            FLOORS.get(value.dtype, FLOORS[torch.float32])
            for value in first.values()
            if isinstance(value, torch.Tensor) and value.is_floating_point()
        ),
        default=FLOORS[torch.float32],
    )
    failures = [
        f"{prefix}{name}: {reason}"
        for prefix, gauges in (("", result.fwd), ("grad ", result.bwd))
        for name, gauge in gauges.items()
        if (reason := gauge.verdict(floor)) is not None
    ]
    if failures:
        return _halt(result, "fail", "; ".join(failures))

    result.reps = index + 1
    if benchmark:
        try:
            result.bench = _benchmark(mine, reference, first, backward, repeats, warmup, reference_bench)
            result.benchmarked = True
        except Exception as error:
            result.bench_error = f"{type(error).__name__}: {error}"
    return result


def compare(
    mine: Callable[..., Any],
    reference: Callable[..., Any],
    inputs: Mapping[str, Any],
    *,
    backward: bool = True,
    benchmark: bool = True,
    repeats: int = 10,
    warmup: int = 2,
) -> Result:
    return compare_inputs(
        mine,
        reference,
        repeat(inputs, repeats),
        backward=backward,
        benchmark=benchmark,
        repeats=repeats,
        warmup=warmup,
    )


def report(
    subject: Record | Result,
    label: str = "",
    *,
    mine: str | None = None,
    reference: str | None = None,
    assert_rel: float | None = None,
) -> str:
    """Pretty-print a `compare` result: latency, peak memory, and forward error.

    >>> report(compare(fast, slow, inputs), "forward", mine="popcorn", reference="baseline")
    forward    baseline  237.5 ms -> popcorn    6.1 ms   (39.1x)
               peak        0.77 GB ->      0.77 GB
               err 0.00e+00 on scale 10.8

    A `Record` from `op.validate` or `op.benchmark` names itself after its implementation.
    """
    if isinstance(subject, Record):
        result, label = subject.result, label or subject.impl
        mine, reference = mine or subject.impl, reference or "torch"
    else:
        result, mine, reference = subject, mine or "mine", reference or "ref"
    width = max(len(label), 8)
    lines: list[str] = []
    bench = result.bench
    if result.benchmarked and "fwd_ms" in bench and "ref_fwd_ms" in bench:
        speed = bench["ref_fwd_ms"] / bench["fwd_ms"] if bench["fwd_ms"] else float("inf")
        lines.append(
            f"{label:{width}s} {reference} {bench['ref_fwd_ms']:8.1f} ms -> {mine} {bench['fwd_ms']:6.1f} ms   ({speed:.1f}x)"
        )
        if "fwd_mem_mb" in bench and "ref_fwd_mem_mb" in bench:
            lines.append(
                f"{'':{width}s} peak     {bench['ref_fwd_mem_mb'] / 1024:8.2f} GB -> {bench['fwd_mem_mb'] / 1024:9.2f} GB"
            )
        if "bwd_ms" in bench and "ref_bwd_ms" in bench:
            bwd_speed = bench["ref_bwd_ms"] / bench["bwd_ms"] if bench["bwd_ms"] else float("inf")
            lines.append(
                f"{'':{width}s} bwd      {bench['ref_bwd_ms']:8.1f} ms -> {bench['bwd_ms']:6.1f} ms   ({bwd_speed:.1f}x)"
            )
    elif result.bench_error:
        lines.append(f"{label:{width}s} {result.status}: {result.bench_error}")
    else:
        lines.append(f"{label:{width}s} {result.status}" + (f": {result.reason}" if result.reason else ""))

    gauge = result.fwd.get("out0")
    if gauge is not None:
        lines.append(f"{'':{width}s} err {gauge.err:.2e} on scale {gauge.scale:.1f}")
        if assert_rel is not None and gauge.scale > 0:
            rel = gauge.err / gauge.scale
            if rel >= assert_rel:
                raise AssertionError(f"{label or 'compare'}: relative error {rel:.2e} >= {assert_rel:.2e}")

    text = "\n".join(lines)
    print(text)
    return text
