import importlib
import inspect
import json
import logging
import sys
import threading
import tomllib
import warnings
from pathlib import Path
from typing import Literal

import pytest
import torch
from jaxtyping import Float, Int
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from torch import Tensor

import popcorn.core.sources
from popcorn.bench.model import Case, Record
from popcorn.bench.store import write
from popcorn import (
    BackendUnavailableError,
    BackendVersionError,
    DispatchError,
    Dispatcher,
    Div,
    Pow2,
    Range,
    UnvalidatedWarning,
    declare_backend,
    kernel,
)
from popcorn.core.constraints import check, satisfies
from popcorn.core.config import call_config, config_id, device_name
from popcorn.core.fingerprint import fingerprint
from popcorn.core.typecheck import matches, narrows

pytestmark = pytest.mark.filterwarnings("ignore::popcorn.core.errors.UnvalidatedWarning")
comparison = importlib.import_module("popcorn.bench.compare")


def make_op(**dispatcher_kwargs):
    def op(
        x: Float[Tensor, "... D"],
        weight: Float[Tensor, "D"] | None = None,
        flag: bool = False,
    ):
        return x + 1

    return Dispatcher(op, **dispatcher_kwargs)


class TestConstraints:
    def test_specs(self):
        assert satisfies(Range(2, 4), 3) and not satisfies(Range(2, 4), 5)
        assert satisfies(Div(8), 16) and not satisfies(Div(8), 12)
        assert satisfies(Pow2(), 64) and not satisfies(Pow2(), 48)
        assert satisfies({16, 32}, 16) and not satisfies({16, 32}, 8)
        assert satisfies(None, None) and not satisfies(None, torch.ones(1))
        assert satisfies((Range(2, 64), Div(4)), 8) and not satisfies((Range(2, 64), Div(4)), 6)
        assert satisfies(lambda v: v > 0, 1)

    def test_tensor_never_equals_literal(self):
        assert not satisfies(5, torch.tensor(5))

    def test_check_reports_first_failure(self):
        message = check({"D": Range(2, 4)}, {"D": 8})
        assert message == "D=8 is not in [2, 4]"
        assert check({"D": Range(2, 4)}, {"other": 1}) is None


class TestTypecheck:
    def test_matches(self):
        assert matches(False, Literal[False]) and not matches(True, Literal[False])
        assert not matches(1, Literal[True])  # no bool/int conflation
        assert matches("sum", Literal["mean", "sum"])
        assert matches(None, None) and not matches(0, None)
        assert matches(1.5, float) and not matches("x", float)
        assert matches(None, bool | None) and matches(True, bool | None)

    def test_narrows(self):
        assert narrows(Literal[False], bool)
        assert narrows(Literal["mean"], Literal["mean", "sum"])
        assert narrows(None, Tensor | None)
        assert narrows(bool, int) and not narrows(int, bool)
        assert not narrows(str, bool)


class TestFingerprint:
    @staticmethod
    def _load(tmp_path, name, text, monkeypatch=None):
        path = tmp_path / f"{name}.py"
        path.write_text(text)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        if monkeypatch is not None:  # classes resolve their source through sys.modules
            monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    def test_ignores_comments_docstrings_and_formatting(self, tmp_path):
        noisy = self._load(tmp_path, "noisy", 'def f(x):\n    """doc"""\n    # comment\n    return (x\n        + 1)\n')
        clean = self._load(tmp_path, "clean", "def f(x):\n    return x + 1\n")
        changed = self._load(tmp_path, "changed", "def f(x):\n    return x + 2\n")
        assert fingerprint(noisy.f) == fingerprint(clean.f) is not None
        assert fingerprint(clean.f) != fingerprint(changed.f)
        assert fingerprint(len) is None

    def test_recurses_into_kernel_scoped_dependencies(self, tmp_path, monkeypatch):
        text = "def helper(x):\n    return x {op} 1\n\ndef f(x):\n    return helper(x)\n"
        one = self._load(tmp_path, "one", text.format(op="+"))
        two = self._load(tmp_path, "two", text.format(op="-"))
        assert fingerprint(one.f) == fingerprint(two.f)  # helpers outside kernel scope are opaque
        one = self._load(tmp_path, "popcorn.kernels.fake_one", text.format(op="+"), monkeypatch)
        two = self._load(tmp_path, "popcorn.kernels.fake_two", text.format(op="-"), monkeypatch)
        assert fingerprint(one.f) != fingerprint(two.f)

    def test_hashes_cuda_siblings_by_content(self, tmp_path, monkeypatch):
        cu = tmp_path / "popcorn.impls.fake.cu"
        cu.write_text("__global__ void k() {}\n")
        module = self._load(tmp_path, "popcorn.impls.fake_cu", "def f(x):\n    return x\n", monkeypatch)
        first = fingerprint(module.f)
        cu.write_text("__global__ void k() { return; }\n")
        assert first is not None
        assert fingerprint(module.f) != first

    def test_sees_through_wrappers_and_classes(self, tmp_path, monkeypatch):
        text = (
            "import functools\n\n"
            "class C:\n    @staticmethod\n    def run(x):\n        return x {op} 1\n\n"
            "@functools.cache\ndef cached(x):\n    return C.run(x)\n\n"
            "def f(x):\n    return cached(x)\n"
        )
        plus = self._load(tmp_path, "popcorn.impls.fake_plus", text.format(op="+"), monkeypatch)
        minus = self._load(tmp_path, "popcorn.impls.fake_minus", text.format(op="-"), monkeypatch)
        assert fingerprint(plus.f) != fingerprint(minus.f)


class TestDispatch:
    def test_reference_is_fallback(self):
        op = make_op()
        assert op.available_backends() == ("torch",)
        assert torch.equal(op(torch.zeros(3, 4)), torch.ones(3, 4))

    def test_supports_gates_resolution(self):
        op = make_op()

        @op.register("fast", supports={"D": Pow2()})
        def fast(x, weight, flag):
            return x - 1

        assert torch.equal(op(torch.zeros(3, 4)), -torch.ones(3, 4))
        assert torch.equal(op(torch.zeros(3, 5)), torch.ones(3, 5))  # D=5 -> torch

    def test_annotation_gates_resolution(self):
        op = make_op()

        @op.register("fast")
        def fast(x, weight, flag: Literal[False]):
            return x - 1

        assert torch.equal(op(torch.zeros(2)), -torch.ones(2))
        assert torch.equal(op(torch.zeros(2), flag=True), torch.ones(2))
        with pytest.raises(DispatchError, match="does not satisfy"):
            op(torch.zeros(2), flag=True, backend="fast")

    def test_registration_order(self):
        op = make_op()
        op.register("first")(lambda x, weight, flag: x + 10)
        op.register("second")(lambda x, weight, flag: x + 20)
        assert op.available_backends() == ("first", "second", "torch")
        assert torch.equal(op(torch.zeros(2)), torch.full((2,), 10.0))

    def test_explicit_backend_violation_raises(self):
        op = make_op()
        op.register("fast", supports={"D": Pow2()})(lambda x, weight, flag: x)
        with pytest.raises(DispatchError, match="a power of two"):
            op(torch.zeros(3, 5), backend="fast")
        with pytest.raises(DispatchError, match="unknown backend"):
            op(torch.zeros(3), backend="nope")

    def test_scope(self):
        op = make_op()
        op.register("a")(lambda x, weight, flag: x + 10)
        op.register("b")(lambda x, weight, flag: x + 20)

        def probe():
            return op(torch.zeros(2))[0].item()

        assert probe() == 10
        with op["b"] as scoped:
            assert scoped is op
            assert probe() == 20
            with op["a"]:
                assert probe() == 10
            assert probe() == 20
        assert probe() == 10

    def test_scopes_are_thread_local(self):
        op = make_op()
        op.register("a")(lambda x, weight, flag: x + 10)
        op.register("b")(lambda x, weight, flag: x + 20)
        a_entered, b_entered = threading.Event(), threading.Event()
        exit_a, exit_b = threading.Event(), threading.Event()
        errors = []

        def run(backend, entered, ready, exit_):
            try:
                with op[backend]:
                    entered.set()
                    ready.wait()
                    exit_.wait()
            except Exception as error:
                errors.append(error)

        first = threading.Thread(target=run, args=("a", a_entered, b_entered, exit_a))
        second = threading.Thread(target=run, args=("b", b_entered, a_entered, exit_b))
        first.start()
        a_entered.wait()
        second.start()
        b_entered.wait()
        exit_a.set()
        first.join()
        exit_b.set()
        second.join()

        assert not errors

    def test_getitem(self):
        op = make_op()
        op.register("alt")(lambda x, weight, flag: x - 1)
        assert torch.equal(op["alt"](torch.zeros(2)), -torch.ones(2))
        with pytest.raises(DispatchError):
            op["nope"]

    def test_backward_flows_through_source_adapter(self):
        def square(x: Float[Tensor, "... D"]):
            return x * x

        op = Dispatcher(square)

        @op.register("alt", source="torch.mul")
        def alt(x):
            return kernel(x, x)

        reference_x = torch.randn(4, requires_grad=True)
        backend_x = reference_x.detach().clone().requires_grad_()
        op(reference_x, backend="torch").sum().backward()
        op(backend_x, backend="alt").sum().backward()
        assert torch.allclose(backend_x.grad, reference_x.grad)

    def test_dim_consistency_checked(self):
        op = make_op()
        with pytest.raises(DispatchError, match="dim 'D'"):
            op(torch.zeros(3, 4), weight=torch.zeros(5))

    def test_predicate(self):
        op = make_op()
        op.register("alt", predicate=lambda x, weight, flag: x.ndim == 1)(lambda x, weight, flag: x - 1)
        assert torch.equal(op(torch.zeros(2)), -torch.ones(2))
        assert torch.equal(op(torch.zeros(2, 2)), torch.ones(2, 2))

    def test_selection_cache_respects_current_eligibility(self):
        # Same tensor metadata, different content: a content-sensitive predicate
        # must not be bypassed by the tuner's memoized selection.
        op = make_op()
        op.register("alt", predicate=lambda x, weight, flag: bool(x.sum() > 0))(lambda x, weight, flag: x - 1)
        assert torch.equal(op(torch.ones(2)), torch.zeros(2))
        assert torch.equal(op(-torch.ones(2)), torch.zeros(2))


class TestRegistration:
    def test_kernel_registry_conventions(self):
        import popcorn.kernels as kernels
        from popcorn import KERNELS

        assert set(kernels.__all__) == set(KERNELS)
        modules = {path.stem for path in Path(kernels.__file__).parent.glob("[!_]*.py")}
        assert set(KERNELS) == modules
        for path in (Path(kernels.__file__).parents[1] / "impls").glob("[!_]*.py"):
            op_name, _, tech = path.stem.rpartition("_")
            assert tech in {"tl", "cu"}, f"impl module {path.name} must end in _tl or _cu"
            backend = next(b for b in KERNELS[op_name]._backends if b.name == "popcorn")
            assert backend.source == f"popcorn.impls.{path.stem}.{op_name}"
        for op in KERNELS.values():
            if "softmax_scale" in op.arg_pools:
                assert len(op.arg_pools["softmax_scale"]) > 1
            backends = [backend for backend in op._backends if backend.name != "torch"]
            assert [backend.name for backend in backends] == sorted(backend.name for backend in backends)
            for backend in backends:
                if backend.adapter is not None:
                    assert backend.adapter.__name__ == f"{op.name}_{backend.name.replace(':', '_')}"

    def test_reference_signature_conventions(self):
        import popcorn.kernels  # noqa: F401
        from popcorn import KERNELS
        from popcorn.core.annotations import _unwrap_optional

        empty = inspect.Parameter.empty
        for op in KERNELS.values():
            for name, param in op._signature.parameters.items():
                assert param.annotation is not empty, f"{op.name}.{name}: parameter must be annotated"
                annotation, _ = _unwrap_optional(param.annotation)
                if isinstance(annotation, type) and issubclass(annotation, Tensor):
                    assert hasattr(annotation, "dim_str"), f"{op.name}.{name}: tensor params use jaxtyping shapes"
            returned = op._signature.return_annotation
            assert returned is not empty, f"{op.name}: return must be annotated"
            if isinstance(returned, type) and issubclass(returned, Tensor):
                assert hasattr(returned, "dim_str"), f"{op.name}: tensor returns use jaxtyping shapes"
            controls = list(op.__signature__.parameters.values())[len(op._signature.parameters) :]
            assert [(control.name, control.kind, control.default) for control in controls] == [
                (name, inspect.Parameter.KEYWORD_ONLY, None) for name in ("backend", "bench", "validate")
            ], f"{op.name}: __signature__ must expose the reference params plus the dispatch controls"

    def test_readme_matrix_covers_every_kernel(self):
        import re

        import popcorn.kernels  # noqa: F401
        from popcorn import KERNELS

        text = (Path(__file__).parents[1] / "README.md").read_text()
        block = text.split("<!-- popcorn:matrix -->", 1)[1].split("<!-- /popcorn:matrix -->", 1)[0]
        rows = set(re.findall(r"^\| `(\w+)` \|", block, re.M))
        assert rows == set(KERNELS), (
            "README matrix out of date; run scripts/update_readme.py "
            f"(missing: {sorted(set(KERNELS) - rows)}, stale: {sorted(rows - set(KERNELS))})"
        )

    def test_params_must_match_reference(self):
        op = make_op()
        with pytest.raises(TypeError, match="parameters must be exactly"):
            op.register("bad")(lambda x, w, flag: x)
        with pytest.raises(TypeError, match="parameters must be exactly"):
            op.register("bad")(lambda x, weight: x)

    def test_defaults_belong_to_reference(self):
        op = make_op()
        with pytest.raises(TypeError, match="must not declare defaults"):
            op.register("bad")(lambda x, weight, flag=False: x)

    def test_annotation_must_narrow(self):
        op = make_op()
        with pytest.raises(TypeError, match="does not narrow"):

            @op.register("bad")
            def bad(x, weight, flag: str):
                return x

        assert op.available_backends() == ("torch",)

    def test_duplicate_and_invalid_names(self):
        op = make_op()
        op.register("alt")(lambda x, weight, flag: x)
        with pytest.raises(ValueError, match="already registered"):
            op.register("alt")
        with pytest.raises(ValueError, match="invalid backend name"):
            op.register("Bad Name")

    def test_constraint_keys_are_shapes_only(self):
        op = make_op()
        with pytest.raises(TypeError, match="unknown supports keys"):
            op.register("alt", supports={"flag": False})
        with pytest.raises(TypeError, match="unknown supports keys"):
            op.register("alt", supports={"dtype": {torch.float16}})
        with pytest.raises(TypeError, match="unknown supports keys"):
            op.register("alt", supports={"x.dtype": {torch.float16}})
        with pytest.raises(TypeError, match="unknown test_shapes keys"):
            make_op(test_shapes={"flag": {True}})
        with pytest.raises(TypeError, match="unknown test_shapes keys"):
            make_op(test_shapes={"nope": Range(1, 2)})

    def test_test_args_validated(self):
        with pytest.raises(TypeError, match="not scalar arguments"):
            make_op(test_args={"x": [1]})
        with pytest.raises(TypeError, match="does not satisfy"):
            make_op(test_args={"flag": ["nope"]})
        op = make_op(test_args={"flag": [True]})
        assert op.arg_pools["flag"] == [True]

    def test_source_only_conforming(self):
        def op(input, other=3):
            return input * other

        d = Dispatcher(op)
        d.register("stub", source="torch.mul")
        assert d(torch.ones(2), 3).tolist() == [3.0, 3.0]

    def test_source_only_mismatch_needs_adapter(self):
        def op(x, y=None):
            return x

        d = Dispatcher(op)
        d.register("stub", source="operator.add")
        with pytest.raises(TypeError, match="register an adapter"):
            d(torch.ones(2))

    def test_empty_registration_rejected(self):
        op = make_op()
        op.register("empty")
        with pytest.raises(DispatchError, match="neither adapter nor source"):
            op(torch.zeros(2))


class TestEnvelope:
    def test_arg_pools_derived(self):
        op = make_op()
        assert op.arg_pools == {"flag": [False, True]}

    def test_test_shapes_are_grid_metadata_only(self):
        op = make_op(test_shapes={"D": Range(2, 8)})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            op(torch.zeros(3, 16))

    def test_test_args_are_grid_metadata_only(self):
        def op(x, mode: str = "a"):
            return x

        d = Dispatcher(op, test_args={"mode": ["a", "b"]})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            d(torch.zeros(1), mode="c")

    def test_contextual_test_input(self):
        from popcorn.bench.grid import cases, make_inputs

        def op(x: Float[Tensor, "rows"], offsets: Int[Tensor, "boundaries"]):
            return x

        d = Dispatcher(
            op,
            test_shapes={"rows": {4}, "boundaries": {3}},
            test_inputs={"offsets": lambda _, dims, generator: torch.tensor([0, 1, dims["rows"]], dtype=torch.int32)},
        )
        inputs = make_inputs(d, cases(d)[0], device="cpu")
        assert inputs["offsets"].tolist() == [0, 1, 4]

    def test_test_inputs_must_be_callable(self):
        with pytest.raises(TypeError, match="must be callable"):
            make_op(test_inputs={"x": object()})
        with pytest.raises(TypeError, match="one or three positional arguments"):
            make_op(test_inputs={"x": lambda tensor, dims: tensor})

    def test_grid_is_full_product(self):
        from popcorn.bench.grid import cases

        op = make_op(test_shapes={"D": {2, 4}})
        grid = cases(op)
        assert len(grid) == 2 * 2 * 2 * 3 * 3  # D x flag x bias presence x dtypes x batches
        assert len({c.case_id for c in grid}) == len(grid)
        assert cases(op, limit=10)


class TestTuning:
    @pytest.fixture(autouse=True)
    def _isolate(self, tmp_path, monkeypatch):
        from popcorn.bench import store

        self.reports = tmp_path / "bundled"
        self.reports.mkdir()
        monkeypatch.setattr(store, "BUNDLED_REPORTS", self.reports)
        monkeypatch.setenv("POPCORN_CACHE_DIR", str(tmp_path / "cache"))
        monkeypatch.delenv("POPCORN_VALIDATE", raising=False)
        monkeypatch.delenv("POPCORN_BENCH", raising=False)
        self.old_cache = tmp_path / "cache" / "reports"
        self.cache = tmp_path / "cache" / "v2" / "reports"
        self.device = device_name("cpu")

    def _record(self, *rows):
        full = [
            {
                "schema": 2,
                "op": "op",
                "backend": backend,
                "device": self.device,
                "case": "",
                "case_id": f"{backend}-{dims['D']}",
                "status": "pass",
                "reason": "",
                "grad": False,
                "torch": torch.__version__,
                "backend_version": None,
                "ts": "2026-01-01T00:00:00+00:00",
                "config": {"dims": dims, "batch": [], "dtype": "float32", "args": {"flag": False}, "present": []},
                "bench": {"fwd_ms": ms, "ref_fwd_ms": ref_ms},
                "benchmarked": True,
                "bench_error": "",
                "fwd": {},
                "bwd": {},
                "reps": 1,
            }
            for backend, dims, ms, ref_ms in rows
        ]
        (self.reports / "op.jsonl").write_text("\n".join(json.dumps(r) for r in full) + "\n")

    def _row(self, op, x, *, backend="alt", status="pass", reason="", ts="2026-01-01T00:00:00+00:00", bench=None, **kwargs):
        from popcorn.core.sources import installed_version

        bound = op._signature.bind(x, **kwargs)
        bound.apply_defaults()
        arguments = bound.arguments
        call = call_config(op, op._values(arguments), arguments)
        return {
            "schema": 2,
            "op": op.name,
            "backend": backend,
            "device": call.device_name,
            "case": "exact",
            "case_id": config_id(call.config),
            "status": status,
            "reason": reason,
            "grad": call.grad,
            "torch": torch.__version__,
            "backend_version": installed_version(backend),
            "ts": ts,
            "config": call.config,
            "fwd": {},
            "bwd": {},
            "bench": bench or {},
            "benchmarked": bench is not None,
            "bench_error": "",
            "reps": 1,
        }

    @staticmethod
    def _store(directory, *rows):
        write([Record.from_dict(row) for row in rows], directory)

    def test_picks_winner_of_closest_configuration(self):
        op = make_op()
        op.register("alt")(lambda x, weight, flag: x - 1)
        self._record(("alt", {"D": 8}, 1.0, 2.0), ("alt", {"D": 1024}, 3.0, 1.0))
        assert torch.equal(op(torch.zeros(3, 16)), -torch.ones(3, 16))  # near D=8: alt wins
        assert torch.equal(op(torch.zeros(3, 2048)), torch.ones(3, 2048))  # near D=1024: torch wins
        assert len(op.tuner._selected) == 2
        op(torch.zeros(3, 16))
        assert len(op.tuner._selected) == 2  # cached, decided once

    def test_compares_each_backend_without_cross_backend_infinities(self):
        op = make_op()
        op.register("fast")(lambda x, weight, flag: x - 1)
        op.register("slow")(lambda x, weight, flag: x - 2)
        self._record(
            ("fast", {"D": 4}, 0.1, 1.0),
            ("slow", {"D": 4}, 0.5, 1.0),
        )
        assert torch.equal(op(torch.zeros(2, 4)), -torch.ones(2, 4))

    def test_picks_winner_for_matching_batch_shape(self):
        op = make_op()
        op.register("alt")(lambda x, weight, flag: x - 1)

        def row(batch, fwd, ref):
            return {
                "schema": 2,
                "op": "op",
                "backend": "alt",
                "device": self.device,
                "case": "",
                "case_id": str(batch),
                "status": "pass",
                "reason": "",
                "grad": False,
                "torch": torch.__version__,
                "backend_version": None,
                "ts": "2026-01-01T00:00:00+00:00",
                "config": {
                    "dims": {"D": 4},
                    "batch": list(batch),
                    "dtype": "float32",
                    "args": {"flag": False},
                    "present": [],
                },
                "bench": {"fwd_ms": fwd, "ref_fwd_ms": ref},
                "benchmarked": True,
                "bench_error": "",
                "fwd": {},
                "bwd": {},
                "reps": 1,
            }

        (self.reports / "op.jsonl").write_text("\n".join(json.dumps(r) for r in (row((), 0.1, 1.0), row((2, 3), 3.0, 1.0))))
        assert torch.equal(op(torch.zeros(2, 3, 4)), torch.ones(2, 3, 4))

    def test_falls_back_to_registration_order_without_data(self):
        op = make_op()
        op.register("alt")(lambda x, weight, flag: x - 1)
        assert torch.equal(op(torch.zeros(2, 4)), -torch.ones(2, 4))

    def test_stale_fingerprint_rows_are_ignored_for_routing(self):
        op = make_op()
        op.register("alt")(lambda x, weight, flag: x - 1)
        x = torch.zeros(2, 4)
        slow = self._row(op, x, bench={"fwd_ms": 3.0, "ref_fwd_ms": 1.0})
        slow["impl_hash"] = op["alt"].fingerprint
        self._store(self.reports, slow)
        assert torch.equal(op(x), torch.ones_like(x))  # trusted "alt is slow": torch wins

        slow["impl_hash"] = "dead"
        self._store(self.reports, slow)
        op.tuner.forget()
        assert torch.equal(op(x), -torch.ones_like(x))  # stale row ignored: registration order wins

    def test_stale_fingerprint_failures_stop_blocking(self):
        op = make_op()
        op.register("alt")(lambda x, weight, flag: x - 1)
        x = torch.zeros(2, 4)
        failed = self._row(op, x, status="fail", reason="old failure")
        failed["ref_hash"] = op.fingerprint
        self._store(self.reports, failed)
        with pytest.raises(DispatchError, match="old failure"):
            op(x, backend="alt")

        failed["ref_hash"] = "dead"
        self._store(self.reports, failed)
        op.tuner.forget()
        assert torch.equal(op(x, backend="alt"), -torch.ones_like(x))  # reference changed: failure no longer applies

    def test_unknown_warns_once_per_exact_call(self):
        op = make_op()
        op.register("alt")(lambda x, weight, flag: x - 1)
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always", UnvalidatedWarning)
            op(torch.zeros(2, 4))
            op(torch.zeros(2, 4))
            op(torch.zeros(3, 4))
        assert len([warning for warning in seen if warning.category is UnvalidatedWarning]) == 2

    def test_exact_failure_matches_every_call_dimension(self):
        op = make_op()
        op.register("alt")(lambda x, weight, flag: x - 1)
        x = torch.zeros(2, 4)
        self._store(self.reports, self._row(op, x, status="fail", reason="numeric mismatch"))

        assert torch.equal(op(x), torch.ones_like(x))
        with pytest.raises(DispatchError, match="numeric mismatch"):
            op(x, backend="alt")

        assert torch.equal(op(torch.zeros(3, 4), backend="alt"), -torch.ones(3, 4))
        assert torch.equal(op(torch.zeros(2, 5), backend="alt"), -torch.ones(2, 5))
        assert torch.equal(op(x, flag=True, backend="alt"), -torch.ones_like(x))
        assert torch.equal(op(x, weight=torch.ones(4), backend="alt"), -torch.ones_like(x))
        assert op(x.double(), backend="alt").dtype == torch.float64
        assert op(x.requires_grad_(), backend="alt").requires_grad

    def test_known_failure_applies_to_every_forcing_api(self):
        op = make_op()
        op.register("alt")(lambda x, weight, flag: x - 1)
        failed = self._row(op, torch.zeros(2, 4), status="fail", reason="known failure")
        measured = self._row(
            op,
            torch.zeros(2, 8),
            bench={"fwd_ms": 0.1, "ref_fwd_ms": 1.0},
        )
        self._store(self.reports, failed, measured)

        with pytest.raises(DispatchError, match="known failure"):
            op(torch.zeros(2, 4), backend="alt")
        with pytest.raises(DispatchError, match="known failure"):
            op["alt"](torch.zeros(2, 4))
        with op["alt"]:
            with pytest.raises(DispatchError, match="known failure"):
                op(torch.zeros(2, 4))
        best = op.tuner.best(device="cpu", grad=False, D={8})
        with pytest.raises(DispatchError, match="known failure"):
            best(torch.zeros(2, 4))

    def test_implementation_crash_blocks_but_harness_error_does_not(self):
        x = torch.zeros(2, 4)
        crashed = make_op(name="crashed")
        crashed.register("alt")(lambda x, weight, flag: x - 1)
        self._store(self.reports, self._row(crashed, x, status="crash", reason="kernel crashed"))
        with pytest.raises(DispatchError, match="kernel crashed"):
            crashed(x, backend="alt")

        errored = make_op(name="errored")
        errored.register("alt")(lambda x, weight, flag: x - 1)
        self._store(self.reports, self._row(errored, x, status="error", reason="worker failed"))
        assert torch.equal(errored(x, backend="alt"), -torch.ones_like(x))

    @pytest.mark.parametrize(
        ("field", "value"),
        [("device", "other device"), ("torch", "0.0.0"), ("backend_version", "0.0.0")],
    )
    def test_validation_is_environment_specific(self, field, value):
        op = make_op()
        op.register("alt")(lambda x, weight, flag: x - 1)
        row = self._row(op, torch.zeros(2, 4), status="fail", reason="wrong environment")
        row[field] = value
        self._store(self.reports, row)
        assert torch.equal(op(torch.zeros(2, 4), backend="alt"), -torch.ones(2, 4))

    def test_unversioned_cache_directory_is_ignored(self):
        op = make_op()
        op.register("alt")(lambda x, weight, flag: x - 1)
        x = torch.zeros(2, 4)
        self._store(self.old_cache, self._row(op, x, status="fail", reason="stale cache"))
        assert torch.equal(op(x, backend="alt"), -torch.ones_like(x))

    @pytest.mark.parametrize(
        ("bundled_status", "bundled_ts", "user_status", "user_ts", "fails"),
        [
            ("fail", "2026-01-01T00:00:00+00:00", "pass", "2026-01-02T00:00:00+00:00", False),
            ("fail", "2026-01-02T00:00:00+00:00", "pass", "2026-01-01T00:00:00+00:00", True),
        ],
    )
    def test_newest_cache_row_wins(self, bundled_status, bundled_ts, user_status, user_ts, fails):
        op = make_op()
        op.register("alt")(lambda x, weight, flag: x - 1)
        x = torch.zeros(2, 4)
        bundled = self._row(op, x, status=bundled_status, reason="bundled", ts=bundled_ts)
        user = self._row(op, x, status=user_status, reason="user", ts=user_ts)
        self._store(self.reports, bundled)
        self._store(self.cache, user)
        if fails:
            with pytest.raises(DispatchError, match="bundled"):
                op(x, backend="alt")
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("error", UnvalidatedWarning)
                assert torch.equal(op(x, backend="alt"), -torch.ones_like(x))

    def test_newer_harness_error_cannot_erase_known_failure(self):
        op = make_op()
        op.register("alt")(lambda x, weight, flag: x - 1)
        x = torch.zeros(2, 4)
        failure = self._row(op, x, status="fail", reason="known failure", ts="2026-01-01T00:00:00+00:00")
        error = self._row(op, x, status="error", reason="worker failed", ts="2026-01-02T00:00:00+00:00")
        self._store(self.reports, failure)
        self._store(self.cache, error)
        with pytest.raises(DispatchError, match="known failure"):
            op(x, backend="alt")

    def test_validate_mode_reruns_stale_fingerprint_rows(self, monkeypatch):
        op = make_op()
        op.register("alt")(lambda x, weight, flag: x - 1)
        x = torch.zeros(2, 4)
        stale = self._row(op, x, status="pass", bench={"fwd_ms": 0.1, "ref_fwd_ms": 1.0})
        stale["impl_hash"] = "dead"
        self._store(self.reports, stale)
        monkeypatch.setenv("POPCORN_VALIDATE", "1")
        assert torch.equal(op(x), torch.ones_like(x))  # re-validated: alt now fails, torch serves the call
        rows = [json.loads(line) for line in (self.cache / "op.jsonl").read_text().splitlines()]
        assert [row["impl_hash"] for row in rows] == [op["alt"].fingerprint]
        assert [row["status"] for row in rows] == ["fail"]

    def test_recorded_results_are_logged(self, caplog):
        op = make_op()
        op.register("alt")(lambda x, weight, flag: x + 1)
        with caplog.at_level(logging.INFO, logger="popcorn.bench"):
            results = op.validate(torch.zeros(2, 4))
        assert [result.status for result in results] == ["pass"]
        assert any("op:alt [pass]" in message for message in caplog.messages)
        assert any("recorded 1 row(s)" in message for message in caplog.messages)

    def test_validate_records_and_invalidates_selection(self):
        op = make_op()
        op.register("alt")(lambda x, weight, flag: x - 1)
        x = torch.zeros(2, 4)
        op(x)
        assert op.tuner._selected
        [result] = op.validate(x, backend="alt")
        assert result.status == "fail"
        assert not op.tuner._selected
        assert (self.cache / "op.jsonl").exists()
        assert torch.equal(op(x), torch.ones_like(x))

    def test_benchmark_records_correctness_and_timings(self):
        op = make_op()
        op.register("alt")(lambda x, weight, flag: x + 1)
        [result] = op.benchmark(torch.zeros(2, 4), backend="alt")
        assert result.status == "pass" and result.benchmarked
        assert result.bench["fwd_ms"] > 0 and result.bench["ref_fwd_ms"] > 0

    def test_validate_mode_routes_around_new_failure(self, monkeypatch):
        op = make_op()
        op.register("alt")(lambda x, weight, flag: x - 1)
        monkeypatch.setenv("POPCORN_VALIDATE", "1")
        with warnings.catch_warnings():
            warnings.simplefilter("error", UnvalidatedWarning)
            assert torch.equal(op(torch.zeros(2, 4)), torch.ones(2, 4))
        rows = [json.loads(line) for line in (self.cache / "op.jsonl").read_text().splitlines()]
        assert rows[0]["status"] == "fail" and not rows[0]["benchmarked"]

    def test_bench_mode_records_timings(self, monkeypatch):
        op = make_op()
        op.register("alt")(lambda x, weight, flag: x + 1)
        monkeypatch.setenv("POPCORN_VALIDATE", "1")
        monkeypatch.setenv("POPCORN_BENCH", "1")
        with warnings.catch_warnings():
            warnings.simplefilter("error", UnvalidatedWarning)
            assert torch.equal(op(torch.zeros(2, 4)), torch.ones(2, 4))
        rows = [json.loads(line) for line in (self.cache / "op.jsonl").read_text().splitlines()]
        assert rows[0]["status"] == "pass" and rows[0]["benchmarked"]
        assert rows[0]["bench"]["fwd_ms"] > 0 and rows[0]["bench"]["ref_fwd_ms"] > 0

    def test_bench_error_does_not_block_dispatch(self, monkeypatch):
        calls = 0
        op = make_op()

        @op.register("alt")
        def alt(x, weight, flag):
            nonlocal calls
            calls += 1
            return x + 1

        def fail(*args, **kwargs):
            raise RuntimeError("timer failed")

        monkeypatch.setattr(comparison, "_benchmark", fail)
        monkeypatch.setenv("POPCORN_BENCH", "1")
        with warnings.catch_warnings():
            warnings.simplefilter("error", UnvalidatedWarning)
            assert torch.equal(op(torch.zeros(2, 4), backend="alt"), torch.ones(2, 4))
            assert torch.equal(op(torch.zeros(2, 4)), torch.ones(2, 4))
        assert calls == 22
        [row] = [json.loads(line) for line in (self.cache / "op.jsonl").read_text().splitlines()]
        assert row["status"] == "pass" and row["bench_error"] == "RuntimeError: timer failed"

    def test_cache_accepts_unhashable_arguments(self):
        def reference(x, sections: list):
            return x

        op = Dispatcher(reference, test_args={"sections": [[1, 2]]})
        op.register("alt")(lambda x, sections: x + sum(sections))
        assert torch.equal(op(torch.zeros(1), [1, 2]), torch.tensor([3.0]))
        assert len(op.tuner._selected) == 1

    def test_best_across_range(self):
        from popcorn import Range

        op = make_op()
        op.register("alt")(lambda x, weight, flag: x - 1)
        self._record(("alt", {"D": 8}, 1.0, 2.0), ("alt", {"D": 16}, 1.0, 2.0), ("alt", {"D": 1024}, 3.0, 1.0))
        assert op.tuner.best(device="cpu", grad=False, D=Range(2, 32)).name == "alt"
        assert op.tuner.best(device=self.device, grad=False, D=Range(2, 32)).name == "alt"
        assert op.tuner.best(device=self.device, grad=False, D={1024}).name == "torch"
        assert op.tuner.best(device=self.device, grad=False, D=Range(2, 32), flag=False).name == "alt"
        with op.tuner.best(device=self.device, grad=False, D=Range(2, 32)):
            assert torch.equal(op(torch.zeros(2, 4)), -torch.ones(2, 4))  # scoped to alt
        with pytest.raises(TypeError, match="unknown filter keys"):
            op.tuner.best(nope=Range(1, 2))
        with pytest.raises(LookupError):
            op.tuner.best(device=self.device, D={7})


class TestForwardOnly:
    def test_declines_calls_that_need_grad(self):
        op = make_op()
        op.register("fwd", forward_only=True)(lambda x, weight, flag: x - 1)
        x = torch.zeros(2, requires_grad=True)
        assert torch.equal(op(x), torch.ones(2))  # needs grad: falls through to torch
        assert torch.equal(op(torch.zeros(2)), -torch.ones(2))
        with torch.no_grad():
            assert torch.equal(op(x), -torch.ones(2))
        with pytest.raises(DispatchError, match="no backward"):
            op(x, backend="fwd")
        assert "fwd-only" in repr(op)


class TestTorchOp:
    def setup_method(self):
        import popcorn.kernels  # noqa: F401
        from popcorn import KERNELS

        self.op = KERNELS["rms_norm"]

    def test_eager_matches_reference(self):
        x, w = torch.randn(4, 8), torch.randn(8)
        with self.op["torch"]:
            out = torch.ops.popcorn.rms_norm(x, w)
        assert torch.allclose(out, self.op.reference(x, w))

    def test_fake_from_reference(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            out = torch.ops.popcorn.rms_norm(torch.empty(4, 8, dtype=torch.bfloat16), torch.empty(8, dtype=torch.bfloat16))
        assert tuple(out.shape) == (4, 8) and out.dtype == torch.bfloat16

    def test_grad_replays_through_dispatcher(self):
        x = torch.randn(3, 8, requires_grad=True)
        w = torch.randn(8, requires_grad=True)
        with self.op["torch"]:
            torch.ops.popcorn.rms_norm(x, w).square().sum().backward()
        xr, wr = (t.detach().clone().requires_grad_() for t in (x, w))
        self.op.reference(xr, wr).square().sum().backward()
        assert torch.allclose(x.grad, xr.grad, atol=1e-6)
        assert torch.allclose(w.grad, wr.grad, atol=1e-6)


class TestKernelAmbient:
    def test_adapter_calls_source(self):
        def op(x, scale=2):
            return x * scale

        d = Dispatcher(op, test_args={"scale": [2, 3]})

        @d.register("stub", source="torch.mul")
        def adapter(x, scale):
            return kernel(x, scale) + 100

        assert d(torch.ones(1), 3).tolist() == [103.0]

    def test_kernel_outside_dispatch_raises(self):
        with pytest.raises(RuntimeError, match="only valid inside"):
            kernel(1)

    def test_nesting_restores_outer_source(self):
        def op(x):
            return x

        inner, outer = Dispatcher(op, name="inner"), Dispatcher(op, name="outer")

        @inner.register("stub", source="torch.mul")
        def inner_adapter(x):
            return kernel(x, 2)

        @outer.register("stub", source="torch.add")
        def outer_adapter(x):
            before = inner(x)
            return before + kernel(x, 0)

        assert outer(torch.ones(1)).tolist() == [3.0]


class TestVersionGate:
    @pytest.fixture(autouse=True)
    def restore_backends(self):
        before = popcorn.core.sources._BACKENDS.copy()
        yield
        popcorn.core.sources._BACKENDS.clear()
        popcorn.core.sources._BACKENDS.update(before)

    def test_backend_extras_match_declarations(self):
        import popcorn.kernels  # noqa: F401

        extras = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())["project"]["optional-dependencies"]
        assert list(popcorn.core.sources._BACKENDS) == sorted(popcorn.core.sources._BACKENDS)
        requirements = {}
        for name, declared in popcorn.core.sources._BACKENDS.items():
            if declared.extra is None:  # first-party: the compiler ships with torch, no extra
                assert name not in extras
                continue
            assert declared.extra == name
            requirement = Requirement(extras[name][0])
            assert requirement.name == declared.package
            assert requirement.specifier == SpecifierSet(f">={declared.min_version},<={declared.max_version}")
            requirements[name] = requirement.name
        assert {Requirement(item).name for item in extras["all"]} == set(requirements.values())

    def test_missing_package_is_skipped_not_fatal(self):
        op = make_op()
        declare_backend("ghost", package="popcorn-definitely-not-installed")
        op.register("ghost")(lambda x, weight, flag: x - 1)
        case = Case((("D", 4),), (), torch.float32, (("flag", False),), frozenset())
        assert op.bench.incomplete("ghost", case, "cpu").environment.backend_version is None
        assert torch.equal(op(torch.zeros(2)), torch.ones(2))  # auto: falls through to torch
        with pytest.raises(BackendUnavailableError, match="install"):
            op(torch.zeros(2), backend="ghost")
        with pytest.raises(BackendUnavailableError):
            op["ghost"]

    def test_version_out_of_range(self):
        op = make_op()
        declare_backend("aged", package="pytest", max_version="0.1")
        op.register("aged")(lambda x, weight, flag: x)
        assert torch.equal(op(torch.zeros(2)), torch.ones(2))
        with pytest.raises(BackendVersionError, match="outside the supported range"):
            op(torch.zeros(2), backend="aged")

    def test_undeclared_backend_skips_gate(self):
        op = make_op()
        op.register("local")(lambda x, weight, flag: x - 1)
        assert torch.equal(op(torch.zeros(2)), -torch.ones(2))
