"""Device introspection shared by triton impls."""

import functools

import torch
import triton


@functools.cache
def sm_count(device_index: int) -> int:
    return triton.runtime.driver.active.utils.get_device_properties(device_index)["multiprocessor_count"]


@functools.cache
def capability(device_index: int) -> tuple[int, int]:
    """Compute capability as a comparable tuple, e.g. (9, 0) for Hopper."""
    return torch.cuda.get_device_capability(device_index)


@functools.cache
def device_type(device_index: int) -> str:
    """The triton backend for the device: "cuda" or "hip"."""
    with torch.cuda.device(device_index):
        return triton.runtime.driver.active.get_current_target().backend
