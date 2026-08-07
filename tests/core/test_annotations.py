import inspect

import pytest
import torch
from jaxtyping import Float, Int
from torch import Tensor

from popcorn.core.annotations import dim_names, extract, plans, return_plans
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


def test_return_plans_support_single_and_tuple_outputs():
    def single(x: Float[Tensor, "... D"]) -> Float[Tensor, "... D"]:
        return x

    def pair(x: Float[Tensor, "batch D"]) -> tuple[Float[Tensor, "batch D"], Float[Tensor, "D"]]:
        return x, x[0]

    assert return_plans(inspect.signature(single))[0].tokens == (..., "D")
    outputs = return_plans(inspect.signature(pair))
    assert [spec.param for spec in outputs] == ["out0", "out1"]
    assert [spec.tokens for spec in outputs] == [("batch", "D"), ("D",)]


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


def test_derived_dim_binds_and_validates():
    def op(logits: Float[Tensor, "batch response+1 vocab"], ref: Float[Tensor, "batch response"]):
        pass

    specs = signature_of(op)
    assert dim_names(specs) == {"batch", "response", "vocab"}
    # logits comes first: the derived occurrence binds the base by inversion.
    dims = extract(specs, {"logits": torch.zeros(2, 32, 7), "ref": torch.zeros(2, 31)})
    assert dims == {"batch": 2, "response": 31, "vocab": 7}
    with pytest.raises(DispatchError, match="ref has 31, other arguments have 30"):
        extract(specs, {"logits": torch.zeros(2, 31, 7), "ref": torch.zeros(2, 31)})

    def base_first(ref: Float[Tensor, "batch response"], logits: Float[Tensor, "batch response+1 vocab"]):
        pass

    with pytest.raises(DispatchError, match=r"logits has 31, expected 32 \(response=31\)"):
        extract(signature_of(base_first), {"ref": torch.zeros(2, 31), "logits": torch.zeros(2, 31, 7)})


def test_derived_dim_scalar_operand():
    def op(q: Float[Tensor, "batch seq"], k: Float[Tensor, "batch seq*num_householder"], num_householder: int = 1):
        pass

    specs = signature_of(op)
    dims = extract(specs, {"q": torch.zeros(2, 8), "k": torch.zeros(2, 24), "num_householder": 3})
    assert dims == {"batch": 2, "seq": 8}
    with pytest.raises(DispatchError, match=r"k has 25, expected 24 \(seq=8\)"):
        extract(specs, {"q": torch.zeros(2, 8), "k": torch.zeros(2, 25), "num_householder": 3})
    with pytest.raises(DispatchError, match="needs a positive int"):
        extract(specs, {"q": torch.zeros(2, 8), "k": torch.zeros(2, 24), "num_householder": None})

    def derived_first(k: Float[Tensor, "seq*num_householder"], num_householder: int = 1):
        pass

    # No free occurrence at all: the inversion must both bind and gatekeep.
    assert extract(signature_of(derived_first), {"k": torch.zeros(24), "num_householder": 3}) == {"seq": 8}
    with pytest.raises(DispatchError, match="size 25 is not a multiple of 3"):
        extract(signature_of(derived_first), {"k": torch.zeros(25), "num_householder": 3})


def test_derived_dim_division():
    def op(x: Float[Tensor, "batch head_dim"], table: Float[Tensor, "batch head_dim/2"]):
        pass

    specs = signature_of(op)
    assert extract(specs, {"x": torch.zeros(2, 32), "table": torch.zeros(2, 16)}) == {"batch": 2, "head_dim": 32}
    with pytest.raises(DispatchError, match="head_dim=33 is not divisible by 2"):
        extract(specs, {"x": torch.zeros(2, 33), "table": torch.zeros(2, 16)})
    with pytest.raises(DispatchError, match=r"table has 16, expected 17 \(head_dim=34\)"):
        extract(specs, {"x": torch.zeros(2, 34), "table": torch.zeros(2, 16)})


def test_derived_operand_must_be_scalar_parameter():
    def unknown(x: Float[Tensor, "seq*groups"]):
        pass

    with pytest.raises(TypeError, match="not a parameter"):
        signature_of(unknown)

    def tensor_operand(x: Float[Tensor, "seq*w"], w: Float[Tensor, "seq"]):
        pass

    with pytest.raises(TypeError, match="must be a scalar parameter"):
        signature_of(tensor_operand)
