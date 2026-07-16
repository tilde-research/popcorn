"""Normalized code fingerprints, stable across comments, docstrings, and formatting.

A fingerprint hashes a callable's AST and, recursively, every kernel-code
function or class it references, so recorded benchmarks stop matching when the
code that produced them changes. Recursion stays within `SCOPES`: harness and
core changes must not invalidate records, and external backends are already
pinned by `backend_version`.
"""

import ast
import hashlib
import inspect
import sys
import textwrap
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import FunctionType
from typing import Any

SCOPES = ("popcorn.kernels.", "popcorn.impls.")

Target = FunctionType | type


def _unwrap(value: Any) -> Target | None:
    """The plain function or class behind common wrappers (partial, wraps, triton jit)."""
    for _ in range(8):
        if isinstance(value, (FunctionType, type)):
            return value
        for attr in ("__wrapped__", "__func__", "func", "fn"):
            inner = getattr(value, attr, None)
            if callable(inner):
                value = inner
                break
        else:
            return None
    return None


def _scoped(value: Target) -> bool:
    return (getattr(value, "__module__", None) or "").startswith(SCOPES)


def _normalized(target: Target) -> ast.Module:
    """Dedented source parsed with docstrings dropped; comments never reach the AST."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(target)))
    for node in ast.walk(tree):
        body = getattr(node, "body", None)  # Lambda/IfExp bodies are single expressions, not lists
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            if isinstance(body[0].value.value, str):
                del body[0]
    return tree


def _namespace(target: Target) -> dict[str, Any]:
    if isinstance(target, type):
        module = sys.modules.get(target.__module__)
        return vars(module) if module else {}
    space = dict(target.__globals__)
    for name, cell in zip(target.__code__.co_freevars, target.__closure__ or ()):
        try:
            space[name] = cell.cell_contents
        except ValueError:
            pass
    return space


def _cuda_source(target: Target) -> Path | None:
    """CUDA sibling by convention: `impls/<op>_cu.py` launches `impls/<op>.cu` (see `_cuda.load`)."""
    module = sys.modules.get(getattr(target, "__module__", None) or "")
    path = Path(file) if (file := getattr(module, "__file__", None)) else None
    if path is None or not path.stem.endswith("_cu"):
        return None
    sibling = path.with_name(path.stem.removesuffix("_cu") + ".cu")
    return sibling if sibling.exists() else None


def _references(tree: ast.AST, space: Mapping[str, Any]) -> Iterator[Any]:
    """Values a body may call: loaded names and one-level attributes (module.helper)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            yield space.get(node.id)
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            yield getattr(space.get(node.value.id), node.attr, None)


def fingerprint(*callables: Any) -> str | None:
    """Digest of the callables' normalized code, or None when nothing is hashable."""
    seen: set[Target] = set()
    queue: list[Target] = []
    for value in callables:
        entry = _unwrap(value)
        if entry is not None and entry not in seen:
            seen.add(entry)
            queue.append(entry)
    hasher, hashed, index, cuda = hashlib.sha256(), False, 0, set()
    while index < len(queue):
        target, index = queue[index], index + 1
        if (sibling := _cuda_source(target)) is not None:
            cuda.add(sibling)
        try:
            tree = _normalized(target)
        except (OSError, TypeError, SyntaxError):
            continue
        hasher.update(ast.dump(tree, include_attributes=False).encode())
        hashed = True
        space = _namespace(target)
        for value in _references(tree, space):
            dependency = _unwrap(value)
            if dependency is not None and dependency not in seen and _scoped(dependency):
                seen.add(dependency)
                queue.append(dependency)
    for sibling in sorted(cuda, key=lambda path: path.name):
        hasher.update(sibling.read_bytes())
        hashed = True
    return hasher.hexdigest()[:12] if hashed else None
