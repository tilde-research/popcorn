"""Annotation checks for dispatch: Literal, Union/Optional, None, plain classes.

Stdlib typing covers this vocabulary; if it grows, beartype is the upgrade path
(`beartype.door.is_subhint` generalizes `narrows`, `die_if_unbearable` replaces
`matches`). typeguard is slower, pydantic coerces values -- both wrong fits.
"""

import inspect
import types
import typing
from typing import Any

_UNIONS = (typing.Union, types.UnionType)


def _norm(annotation: Any) -> Any:
    return type(None) if annotation is None else annotation


def matches(value: Any, annotation: Any) -> bool:
    annotation = _norm(annotation)
    if annotation is inspect.Parameter.empty or annotation is typing.Any:
        return True
    origin = typing.get_origin(annotation)
    if origin is typing.Literal:
        return any(value is a or (type(value) is type(a) and value == a) for a in typing.get_args(annotation))
    if origin in _UNIONS:
        return any(matches(value, a) for a in typing.get_args(annotation))
    if isinstance(annotation, type):
        return isinstance(value, annotation)
    return True


def narrows(sub: Any, sup: Any) -> bool:
    """Is `sub` a subset of the type `sup`?"""
    sub, sup = _norm(sub), _norm(sup)
    if sup is inspect.Parameter.empty or sup is typing.Any or sub == sup:
        return True
    if typing.get_origin(sub) is typing.Literal:
        return all(matches(v, sup) for v in typing.get_args(sub))
    if typing.get_origin(sub) in _UNIONS:
        return all(narrows(a, sup) for a in typing.get_args(sub))
    if typing.get_origin(sup) in _UNIONS:
        return any(narrows(sub, a) for a in typing.get_args(sup))
    sub_dims, sup_dims = getattr(sub, "dim_str", None), getattr(sup, "dim_str", None)
    if sub_dims is not None and sup_dims is not None:  # jaxtyping annotations compare structurally
        return sub_dims == sup_dims and set(getattr(sub, "dtypes", ())) <= set(getattr(sup, "dtypes", ()))
    if isinstance(sub, type) and isinstance(sup, type):
        try:
            return issubclass(sub, sup)
        except TypeError:
            return False
    return False
