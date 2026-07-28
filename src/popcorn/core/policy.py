"""Pluggable dispatch policy: who is eligible, and who wins among them.

Assign a custom instance to `op.tuner.policy` (or subclass `Policy`) to change
admission / ranking without touching the tuner. The default policy is safe:
outside a fitted region is blocked. `unsafe=True` (call kwarg or
`POPCORN_UNSAFE=1`) extrapolates past the region for backends that already have
a proven range, then picks the nearest timed neighbor — so a backend only
proven up to N/2 loses to ones proven up to N when the call is at 2N.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from statistics import median
from typing import TYPE_CHECKING, Any, Protocol

from popcorn.bench.fit import Region
from popcorn.core.args import speed_continuous, speed_discrete
from popcorn.core.errors import DispatchError

if TYPE_CHECKING:
    from popcorn.bench.model import Record
    from popcorn.core.config import Call
    from popcorn.core.dispatcher import Implementation
    from popcorn.core.tuning import Sample

INDIFFERENCE = 0.03


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").lower() in {"1", "true", "yes", "on"}


class DecisionContext(Protocol):
    """Read-only view the policy needs from the tuner."""

    def exact(self, impl: Implementation, call: Call) -> Record | None: ...

    def region_for(self, impl: Implementation, call: Call) -> Region | None: ...

    def mapped(self, impl: str) -> bool: ...

    @property
    def samples(self) -> tuple[Sample, ...]: ...

    @property
    def op_name(self) -> str: ...


@dataclass(frozen=True, slots=True)
class Policy:
    """Default admission + nearest-timing ranking.

    Override `blocked` / `choose` on a subclass, or replace fields, then set
    `tuner.policy = ...`.
    """

    unsafe: bool = False
    indifference: float = INDIFFERENCE

    def bind(self, unsafe: bool | None = None) -> Policy:
        """Apply a per-call / env override without mutating the base policy."""
        if unsafe is True:
            return replace(self, unsafe=True)
        if unsafe is False:
            return replace(self, unsafe=False)
        if _env_flag("POPCORN_UNSAFE"):
            return replace(self, unsafe=True)
        return self

    def blocked(self, impl: Implementation, call: Call | None, forced: str | None, ctx: DecisionContext) -> bool:
        """True → drop from eligible. Forced failures raise `DispatchError`."""
        if impl.name == "torch" or call is None:
            return False
        record = ctx.exact(impl, call)
        status = record.result.status if record else None
        if status in ("fail", "crash"):
            if forced == impl.name:
                reason = (record.result.reason if record else "") or status
                raise DispatchError(f"{ctx.op_name}: backend {impl.name!r} failed validation: {reason}")
            return True
        if status == "pass":
            return False
        region = ctx.region_for(impl, call)
        if region is not None:
            if region.contains(call.config["dims"], call.config["args"]):
                return False
            # Outside the fitted envelope.
            if self.unsafe:
                return False  # extrapolate; ranking keeps nearer proven backends ahead
            if forced == impl.name:
                raise DispatchError(
                    f"{ctx.op_name}: backend {impl.name!r} is outside its validity region: "
                    f"{region.reject(call.config['dims'], call.config['args'])}"
                )
            return True
        # No region on this stratum.
        if forced == impl.name:
            return False
        if self.unsafe:
            # Auto-unsafe only trusts implementations that have proven *some* range.
            return not ctx.mapped(impl.name)
        # Safe: unmapped → registration-order eligible; other strata mapped → block.
        return ctx.mapped(impl.name)

    def choose(
        self,
        eligible: Sequence[Implementation],
        call: Call | None,
        forced: str | None,
        ctx: DecisionContext,
    ) -> Implementation | None:
        """Pick a winner among `eligible`, or `None` to fall back to registration order."""
        if forced is not None or call is None:
            return None
        return self.nearest(eligible, call, ctx.samples)

    def nearest(self, candidates: Sequence[Implementation], call: Call, samples: Sequence[Sample]) -> Implementation | None:
        names = {impl.name for impl in candidates}
        config = call.config
        want = speed_discrete(config["args"])
        pool = [
            sample
            for sample in samples
            if sample.impl in names
            and sample.device == call.device_name
            and sample.grad == call.grad
            and sample.dtype == config["dtype"]
            and speed_discrete(sample.args) == want
            and sample.present == frozenset(config["present"])
        ]
        if not pool:
            return None
        nearest = min(self.distance(config, sample) for sample in pool)
        winner = self.fastest([sample for sample in pool if self.distance(config, sample) == nearest])
        return next((impl for impl in candidates if impl.name == winner), None)

    @staticmethod
    def log_gap(left: float, right: float) -> float:
        if left == right:
            return 0.0
        if left > 0 and right > 0:
            return abs(math.log2(left) - math.log2(right))
        if left < 0 and right < 0:
            return abs(math.log2(-left) - math.log2(-right))
        return 10.0

    @classmethod
    def distance(cls, config: Mapping[str, Any], sample: Sample) -> float:
        dims = config["dims"]
        if dims.keys() != sample.dims.keys():
            return math.inf
        distance = sum(cls.log_gap(max(1, dims[name]), max(1, sample.dims[name])) for name in dims)
        continuous = speed_continuous(config["args"])
        sample_continuous = speed_continuous(sample.args)
        if continuous.keys() != sample_continuous.keys():
            return math.inf
        distance += sum(cls.log_gap(continuous[name], sample_continuous[name]) for name in continuous)
        return distance + 10 * (tuple(config["batch"]) != sample.batch)

    def fastest(self, samples: Sequence[Sample]) -> str:
        times = {"torch": median(sample.time[1] for sample in samples)}
        for impl in {sample.impl for sample in samples}:
            times[impl] = median(sample.time[0] for sample in samples if sample.impl == impl)
        ordered = sorted(times, key=lambda name: times[name])
        best, runner = ordered[0], ordered[1] if len(ordered) > 1 else ordered[0]
        if runner != best and times[runner] <= times[best] * (1 + self.indifference):
            return "torch" if "torch" in times else best
        return best
