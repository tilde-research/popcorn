"""Canonical call configurations: the JSON shape of a case and its stable id."""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from popcorn.core.annotations import batch_shape

if TYPE_CHECKING:
    from popcorn.core.dispatcher import Dispatcher


def dtype_name(dtype: torch.dtype | str | None) -> str | None:
    return None if dtype is None else str(dtype).removeprefix("torch.")


def _json_value(value: Any) -> Any:
    match value:
        case dict():
            return {str(name): _json_value(item) for name, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
        case list() | tuple():
            return [_json_value(item) for item in value]
        case set() | frozenset():
            return sorted(map(_json_value, value), key=repr)
        case None | bool() | int() | float() | str():
            return value
        case _:
            return str(value)


def make_config(
    dims: Mapping[str, int],
    batch: Iterable[int],
    dtype: torch.dtype | str | None,
    args: Mapping[str, Any],
    present: Iterable[str],
) -> dict[str, Any]:
    return {
        "dims": dict(sorted(dims.items())),
        "batch": list(batch),
        "dtype": dtype_name(dtype),
        "args": {name: _json_value(value) for name, value in sorted(args.items())},
        "present": sorted(present),
    }


def config_id(config: Mapping[str, Any]) -> str:
    dtype = config["dtype"]
    dtype = dtype if str(dtype).startswith("torch.") else f"torch.{dtype}"
    payload = [
        tuple(sorted(config["dims"].items())),
        tuple(config["batch"]),
        dtype,
        tuple(sorted(config["args"].items())),
        sorted(config["present"]),
    ]
    return hashlib.sha1(json.dumps(payload, default=str).encode()).hexdigest()[:12]


def device_name(device: torch.device | str | int | None = None) -> str:
    try:
        target = None if device is None else torch.device(device)
    except (TypeError, RuntimeError, ValueError):
        return str(device)
    if torch.cuda.is_available() and (target is None or target.type == "cuda"):
        return torch.cuda.get_device_name(target)
    return platform.processor() or "cpu"


@dataclass(frozen=True, slots=True)
class Call:
    """One dispatchable call: its case config, gradient need, and device."""

    config: dict
    grad: bool
    device: torch.device
    device_name: str


def call_config(op: Dispatcher, values: Mapping[str, Any], arguments: Mapping[str, Any]) -> Call | None:
    dims = {name: values[name] for name in op._dims if name in values}
    if dims.keys() != op._dims:
        return None
    variadic = any(... in spec.tokens for spec in op.specs)
    batch = batch_shape(op.specs, arguments) if variadic else ()
    if batch is None:
        return None
    dtype = dtype_name(values.get("dtype"))
    if dtype is None:
        if any("float" in kind for spec in op.specs for kind in spec.dtypes):
            return None
        dtype = "float32"
    devices = {value.device for value in arguments.values() if isinstance(value, torch.Tensor)}
    if len(devices) > 1:
        return None
    device = next(iter(devices), torch.device("cpu"))
    config = make_config(
        dims,
        batch,
        dtype,
        {name: arguments[name] for name in op.arg_pools},
        (spec.param for spec in op.specs if spec.optional and arguments[spec.param] is not None),
    )
    grad = torch.is_grad_enabled() and any(
        isinstance(value, torch.Tensor) and value.requires_grad for value in arguments.values()
    )
    return Call(config, grad, device, device_name(device))
