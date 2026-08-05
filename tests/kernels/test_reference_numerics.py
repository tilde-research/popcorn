import torch

from popcorn.bench.grid import make_inputs
from popcorn.bench.model import Case
from popcorn.kernels.cross_entropy import cross_entropy
from popcorn.kernels.grpo import grpo
from popcorn.kernels.grpo_offpolicy import grpo_offpolicy
from popcorn.kernels.jsd import generalized_jsd, jsd
from popcorn.kernels.kl_div import kl_div
from popcorn.kernels.linear_cross_entropy import linear_cross_entropy
from popcorn.kernels.linear_jsd import linear_jsd
from popcorn.kernels.linear_kl_div import linear_kl_div
from popcorn.kernels.retention import _causal_decay
from popcorn.kernels.swiglu_mlp import swiglu_mlp
from popcorn.kernels.tvd import tvd


def test_kl_div_handles_zero_probability_in_float16():
    x = torch.tensor([[0.0, -1.0]], dtype=torch.float16, requires_grad=True)
    target = torch.tensor([[1.0, 0.0]], dtype=torch.float16)

    loss = kl_div.reference(x, target)
    loss.backward()

    assert torch.isfinite(loss)
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_cross_entropy_upcasts_float16_label_smoothing_reduction():
    logits = torch.zeros((2, 4096), dtype=torch.float16, requires_grad=True)
    labels = torch.tensor([0, 1])

    loss = cross_entropy.reference(logits, labels, label_smoothing=0.1)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_linear_cross_entropy_upcasts_projection_and_reduction():
    x = torch.zeros((2, 1), dtype=torch.float16, requires_grad=True)
    weight = torch.zeros((4096, 1), dtype=torch.float16, requires_grad=True)
    labels = torch.tensor([0, 1])

    loss = linear_cross_entropy.reference(x, weight, labels, label_smoothing=0.1)
    loss.backward()

    assert torch.isfinite(loss)
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert weight.grad is not None and torch.isfinite(weight.grad).all()


def test_grpo_draws_finite_log_probabilities_for_float16():
    case = Case(
        (("batch", 2), ("response", 31), ("vocab", 1024)),
        (),
        torch.float16,
        (("beta", 0.1),),
        frozenset(),
    )
    inputs = make_inputs(grpo, case, device="cpu", grad=True)

    loss = grpo.reference(**inputs)
    loss.backward(torch.randn_like(loss))

    assert (inputs["ref_logp"] < 0).all()
    assert torch.isfinite(loss).all()
    assert inputs["logits"].grad is not None and torch.isfinite(inputs["logits"].grad).all()


def test_grpo_offpolicy_draws_finite_log_probabilities_for_float16():
    case = Case(
        (("batch", 2), ("response", 31), ("vocab", 1024)),
        (),
        torch.float16,
        (),
        frozenset(),
    )
    inputs = make_inputs(grpo_offpolicy, case, device="cpu", grad=True)

    loss, kl, clipped = grpo_offpolicy.reference(**inputs)
    loss.backward(torch.randn_like(loss))

    assert (inputs["old_logp"] < 0).all() and (inputs["ref_logp"] < 0).all()
    assert torch.isfinite(loss).all()
    assert kl is not None and torch.isfinite(kl).all()
    assert torch.isfinite(clipped).all()
    assert inputs["logits"].grad is not None and torch.isfinite(inputs["logits"].grad).all()


def test_linear_kl_div_upcasts_projection_and_averages_before_reduction_overflows():
    hidden = 4096
    x = torch.ones((1, hidden), dtype=torch.float16, requires_grad=True)
    target_x = torch.ones_like(x)
    weight = torch.stack((torch.full((hidden,), 20.0), torch.zeros(hidden))).half().requires_grad_()
    target_weight = torch.stack((torch.zeros(hidden), torch.full((hidden,), 20.0))).half()

    loss = linear_kl_div.reference(x, target_x, weight, target_weight)
    loss.backward()

    assert torch.isfinite(loss)
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert weight.grad is not None and torch.isfinite(weight.grad).all()


def test_linear_jsd_upcasts_float16_projections():
    hidden = 4096
    student = torch.ones((1, hidden), dtype=torch.float16, requires_grad=True)
    teacher = torch.ones_like(student)
    student_weight = torch.stack((torch.full((hidden,), 20.0), torch.zeros(hidden))).half().requires_grad_()
    teacher_weight = torch.stack((torch.zeros(hidden), torch.full((hidden,), 20.0))).half()

    loss = linear_jsd.reference(student, teacher, student_weight, teacher_weight)
    loss.backward()

    assert torch.isfinite(loss)
    assert student.grad is not None and torch.isfinite(student.grad).all()
    assert student_weight.grad is not None and torch.isfinite(student_weight.grad).all()


def test_swiglu_mlp_scales_weights_and_upcasts_float16_intermediates():
    case = Case(
        (("hidden", 4096), ("intermediate", 64)),
        (),
        torch.float16,
        (),
        frozenset(),
    )
    inputs = make_inputs(swiglu_mlp, case, device="cpu", grad=True)

    output = swiglu_mlp.reference(**inputs)
    output.backward(torch.randn_like(output))

    assert torch.isfinite(output).all()
    assert all(tensor.grad is not None and torch.isfinite(tensor.grad).all() for tensor in inputs.values())


def test_jsd_handles_jointly_underflowed_probability_mass():
    log_p = torch.tensor([[-1000.0, 0.0]], requires_grad=True)
    log_q = torch.tensor([[-1000.0, 0.0]])

    loss = generalized_jsd(log_p, log_q, beta=0.5).sum()
    loss.backward()

    assert torch.isfinite(loss)
    assert log_p.grad is not None and torch.isfinite(log_p.grad).all()


def test_jsd_averages_before_float16_reduction_overflows():
    tokens = 4096
    log_p = torch.tensor([[0.0, -20.0]], dtype=torch.float16).repeat(tokens, 1).requires_grad_()
    log_q = torch.tensor([[-20.0, 0.0]], dtype=torch.float16).repeat(tokens, 1)

    loss = jsd.reference(log_p, log_q, beta=1.0)
    loss.backward()

    assert torch.isfinite(loss)
    assert log_p.grad is not None and torch.isfinite(log_p.grad).all()


def test_retention_masks_future_positions_before_exponentiation():
    steps = torch.tensor([[-3000.0, 0.0], [1.0, 3000.0]])
    gamma = torch.tensor([1 - 2**-5])

    decay = _causal_decay(steps, gamma)

    assert torch.isfinite(decay).all()
    assert decay[0, 0, 0] == 0


def test_tvd_upcasts_large_float16_reduction():
    p = torch.tensor([[1.0, 0.0]], dtype=torch.float16).repeat(65_536, 1).requires_grad_()
    q = torch.tensor([[0.0, 1.0]], dtype=torch.float16).repeat(65_536, 1)

    loss = tvd.reference(p, q, reduction="sum")
    loss.backward()

    assert torch.isfinite(loss)
    assert p.grad is not None and torch.isfinite(p.grad).all()
