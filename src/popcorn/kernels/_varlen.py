"""Helpers for packed variable-length (cu_seqlens) ops."""

from collections.abc import Mapping

import torch


def cuts(t: torch.Tensor, dims: Mapping[str, int], generator: torch.Generator) -> torch.Tensor:
    """Random cu_seqlens: `boundaries - 1` non-empty segments covering `total`."""
    total, segments = dims["total"], t.shape[-1] - 1
    assert total >= segments, f"total={total} cannot host {segments} non-empty segments"
    points = torch.randperm(total - 1, generator=generator)[: segments - 1].sort().values + 1
    return torch.cat([points.new_zeros(1), points, points.new_full((1,), total)]).int()
