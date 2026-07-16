import inspect

import pytest
import torch
from jaxtyping import Float, Int
from torch import Tensor

from popcorn.core.annotations import dim_names, extract, plans
from popcorn.core.errors import DispatchError


def signature_of(fn):
    return plans(inspect.signature(fn))


def test_plans_and_dims():
    def op(
        x: Float[Tensor, "... D"],
        w: Float[Tensor, "D Dp"],
        idx: Int[Tensor, "N"],
        bias: Float[Tensor, "Dp"] | None = None,
        eps: float = 1e-6,
    ):
        pass

    specs = signature_of(op)
    assert [s.param for s in specs] == ["x", "w", "idx", "bias"]
    assert dim_names(specs) == {"D", "Dp", "N"}
    assert next(s for s in specs if s.param == "bias").optional
    assert "float32" in next(s for s in specs if s.param == "x").dtypes


def test_extract_with_ellipsis_and_literals():
    def op(x: Float[Tensor, "... D"], w: Float[Tensor, "D 4"]):
        pass

    specs = signature_of(op)
    dims = extract(specs, {"x": torch.zeros(2, 3, 8), "w": torch.zeros(8, 4)})
    assert dims == {"D": 8}
    with pytest.raises(DispatchError, match="expected dim of 4"):
        extract(specs, {"x": torch.zeros(2, 8), "w": torch.zeros(8, 5)})
    with pytest.raises(DispatchError, match="other arguments have"):
        extract(specs, {"x": torch.zeros(2, 3), "w": torch.zeros(8, 4)})
    with pytest.raises(DispatchError, match="expected 2 dims"):
        extract(specs, {"x": torch.zeros(2, 8), "w": torch.zeros(4)})


def test_optional_none_skipped():
    def op(x: Float[Tensor, "D"], bias: Float[Tensor, "D"] | None = None):
        pass

    assert extract(signature_of(op), {"x": torch.zeros(4), "bias": None}) == {"D": 4}


def test_unsupported_tokens_rejected():
    def op(x: Float[Tensor, "*batch D"]):
        pass

    with pytest.raises(TypeError, match="unsupported dim token"):
        signature_of(op)
