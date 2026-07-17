"""popcorn.compile: pattern registration on CPU, graph rewriting on CUDA."""

import pytest
import torch

import popcorn.compile
from popcorn import KERNELS
from popcorn.compile import PATTERNS, _PopcornPass

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(autouse=True)
def _restore_pass():
    yield
    popcorn.compile.disable()


def _patterns() -> int:
    return sum(len(entries) for entries in PATTERNS.patterns.values())


class TestRegistration:
    def test_enable_registers_and_installs(self):
        before = _patterns()
        registered = popcorn.compile.enable(ops=["rms_norm"], dtypes=(torch.float16,))
        installed = torch._inductor.config.joint_custom_pre_pass
        assert isinstance(installed, _PopcornPass)
        assert registered > 0 and _patterns() - before == registered
        assert installed.uuid() is not None

    def test_enable_is_idempotent_and_disable_restores(self):
        popcorn.compile.enable(ops=["rms_norm"], dtypes=(torch.float32,))
        count = _patterns()
        assert popcorn.compile.enable(ops=["rms_norm"], dtypes=(torch.float32,)) == 0  # already traced
        assert _patterns() == count
        prior = torch._inductor.config.joint_custom_pre_pass.inner
        popcorn.compile.disable()
        assert torch._inductor.config.joint_custom_pre_pass is prior

    def test_wraps_existing_pass(self):
        calls = []
        sentinel = lambda graph: calls.append(1)  # noqa: E731
        torch._inductor.config.joint_custom_pre_pass = sentinel
        popcorn.compile.enable(ops=["rms_norm"], dtypes=(torch.float32,))
        installed = torch._inductor.config.joint_custom_pre_pass
        assert isinstance(installed, _PopcornPass) and installed.inner is sentinel
        assert installed.uuid() is None  # opaque inner pass disables caching
        popcorn.compile.disable()
        assert torch._inductor.config.joint_custom_pre_pass is sentinel
        torch._inductor.config.joint_custom_pre_pass = None


class TestTorchOpAutograd:
    def test_grads_match_reference(self):
        op = KERNELS["rms_norm"]
        x, w, b = (
            torch.randn(3, 8, requires_grad=True),
            torch.randn(8, requires_grad=True),
            torch.randn(8, requires_grad=True),
        )
        with op["torch"]:
            torch.ops.popcorn.rms_norm(x, w, b, 1e-5).square().sum().backward()
        xr, wr, br = (t.detach().clone().requires_grad_() for t in (x, w, b))
        op.reference(xr, wr, br, 1e-5).square().sum().backward()
        for got, want in ((x, xr), (w, wr), (b, br)):
            assert torch.allclose(got.grad, want.grad, atol=1e-6)

    def test_partial_requires_grad(self):
        op = KERNELS["rms_norm"]
        x, w = torch.randn(3, 8), torch.randn(8, requires_grad=True)
        with op["torch"]:
            torch.ops.popcorn.rms_norm(x, w).sum().backward()
        assert w.grad is not None and x.grad is None


class HandwrittenRMSNorm(torch.nn.Module):
    """The reference idiom written out by hand, as a model author would."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(dim, dtype=torch.bfloat16))

    def forward(self, x):
        h = x.float()
        h = h * torch.rsqrt(h.square().mean(-1, keepdim=True) + 1e-6)
        return h.to(x.dtype) * self.weight


@requires_cuda
class TestCompiledGraphs:
    """Asserts on inductor's generated code, not on match counters: an
    FX-cache hit legitimately skips every pass (the counter never moves), and
    inductor's own pattern matches share the global counter (which can mask a
    popcorn miss). Caches are disabled per test so code is always generated."""

    @pytest.fixture(autouse=True)
    def _fresh_compile(self):
        prior = torch._inductor.config.force_disable_caches
        torch._inductor.config.force_disable_caches = True
        torch._dynamo.reset()
        popcorn.compile.enable(ops=["rms_norm"], dtypes=(torch.bfloat16,))
        yield
        torch._inductor.config.force_disable_caches = prior

    def test_training_graph_rewrites_and_matches_eager(self):
        from torch._inductor.utils import run_and_get_code

        model = HandwrittenRMSNorm(64).cuda()
        x = torch.randn(8, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)

        eager = model(x)
        eager.float().square().sum().backward()
        gx, gw = x.grad.clone(), model.weight.grad.clone()
        x.grad = model.weight.grad = None

        compiled = torch.compile(model)

        def step(x: torch.Tensor) -> torch.Tensor:
            out = compiled(x)
            out.float().square().sum().backward()
            return out

        out, code = run_and_get_code(step, x)
        assert any("popcorn.rms_norm" in source for source in code), "no popcorn rewrite in the training graph"
        assert any("rms_norm_backward" in source for source in code), "backward not served by popcorn"
        assert torch.allclose(out, eager, atol=3e-2, rtol=3e-2)
        assert torch.allclose(x.grad, gx, atol=3e-2, rtol=3e-2)
        assert torch.allclose(model.weight.grad, gw, atol=3e-2, rtol=3e-2)

    def test_inference_graph_rewrites(self):
        from torch._inductor.utils import run_and_get_code

        model = HandwrittenRMSNorm(64).cuda()
        x = torch.randn(8, 64, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            out, code = run_and_get_code(torch.compile(model), x)
        assert any("popcorn.rms_norm" in source for source in code), "no popcorn rewrite in the inference graph"
        assert torch.allclose(out, model(x), atol=3e-2, rtol=3e-2)
