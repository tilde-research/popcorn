"""Normalized code fingerprints, stable across comments, docstrings, and formatting.

A fingerprint hashes a callable's normalized source and, recursively, every
kernel-code function or class it references. First-party `source=` modules are
hashed statically with their local implementation imports, so missing optional
libraries cannot disable staleness checks. The serializer emulates Python
3.12's `ast.dump`, preserving callable hashes while filling the `type_params`
field absent on 3.11. Recursion stays within `SCOPES`: harness and core changes
must not invalidate records, and external backends are pinned separately by
`backend_version`.
"""

import ast
import hashlib
import importlib.util
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
    """Dedented source parsed with docstrings and decorators dropped; comments
    never reach the AST. Decorators carry registration metadata (test grids,
    sources, tags), not the code the benchmark measured."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(target)))
    return _clean(tree)


def _clean(tree: ast.Module) -> ast.Module:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            node.decorator_list.clear()
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


def _stable_dump(node: ast.AST) -> str:
    """Python 3.12's compact AST dump, including empty definition type params."""

    def format_value(value: Any) -> tuple[str, bool]:
        if isinstance(value, ast.AST):
            cls = type(value)
            fields = list(value._fields)
            if isinstance(value, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and "type_params" not in fields:
                fields.append("type_params")
            parts = []
            simple = True
            for name in fields:
                field = [] if name == "type_params" and not hasattr(value, name) else getattr(value, name)
                if field is None and getattr(cls, name, ...) is None:
                    continue
                rendered, child_simple = format_value(field)
                simple = simple and child_simple
                parts.append(f"{name}={rendered}")
            if simple and len(parts) <= 3:
                return f"{value.__class__.__name__}({', '.join(parts)})", not parts
            return f"{value.__class__.__name__}({', '.join(parts)})", False
        if isinstance(value, list):
            if not value:
                return "[]", True
            return f"[{', '.join(format_value(item)[0] for item in value)}]", False
        return repr(value), True

    return format_value(node)[0]


def _source_modules(source: str) -> list[tuple[str, ast.Module, Path]]:
    """Local source module and local implementation modules it imports."""
    root = source.rsplit(".", 1)[0]
    queued = [root]
    found: dict[str, tuple[ast.Module, Path]] = {}
    while queued:
        module = queued.pop()
        if module in found:
            continue
        try:
            spec = importlib.util.find_spec(module)
        except (AttributeError, ImportError, ValueError):
            continue
        path = Path(spec.origin) if spec is not None and spec.origin else None
        if path is None or path.suffix != ".py":
            continue
        tree = _clean(ast.parse(path.read_text()))
        found[module] = (tree, path)
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                relative = "." * node.level + (node.module or "")
                try:
                    package = module.rpartition(".")[0]
                    base = importlib.util.resolve_name(relative, package) if node.level else relative
                    names = [base, *(f"{base}.{alias.name}" for alias in node.names if alias.name != "*")]
                except (ImportError, ValueError):
                    names = []
            queued.extend(name for name in names if name.startswith(SCOPES))
    return [(module, *found[module]) for module in sorted(found)]


def fingerprint(*callables: Any, source: str | None = None) -> str | None:
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
        hasher.update(_stable_dump(tree).encode())
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
    if source is not None:
        hasher.update(source.encode())
        for module, tree, path in _source_modules(source):
            hasher.update(module.encode())
            hasher.update(_stable_dump(tree).encode())
            if path.stem.endswith("_cu"):
                sibling = path.with_name(path.stem.removesuffix("_cu") + ".cu")
                if sibling.exists():
                    hasher.update(sibling.read_bytes())
            hashed = True
    return hasher.hexdigest()[:12] if hashed else None
