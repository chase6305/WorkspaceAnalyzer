"""Value and derivative regressions for differentiable Torch dexterity metrics."""

from pathlib import Path

import numpy as np
import pytest

from workspace_analyzer import create_solver
from workspace_analyzer.metrics import _dexterity_from_jacobian

FIXTURE = Path(__file__).parent / "fixtures/cartesian_stage.urdf"


def _solver(dtype="float64", backend="torch"):
    if backend == "torch":
        pytest.importorskip("torch")
    return create_solver(FIXTURE, backend=backend, device="cpu", dtype=dtype)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("metric", ["isotropy", "condition_number"])
def test_mixed_singular_batches_have_finite_independent_gradients(dtype, metric):
    torch = pytest.importorskip("torch")
    solver = _solver(dtype)
    precision = getattr(torch, dtype)
    jacobian = torch.zeros((3, 6, 3), dtype=precision)
    jacobian[1, :3] = torch.diag(torch.tensor([2.0, 1.0, 0.0], dtype=precision))
    jacobian[2, :3] = torch.diag(torch.tensor([3.0, 2.0, 1.0], dtype=precision))
    jacobian.requires_grad_()
    result = _dexterity_from_jacobian(
        solver, torch.zeros((3, 3), dtype=precision), jacobian, task="position"
    )
    values = getattr(result, metric)
    # This is a common loss pattern: exclude undefined condition numbers.
    loss = torch.where(torch.isfinite(values), values, 0).sum()
    loss.backward()
    assert torch.isfinite(jacobian.grad).all()
    expected = np.array([0, 0, 1 / 3]) if metric == "isotropy" else [np.inf, np.inf, 3]
    np.testing.assert_allclose(values.detach().numpy(), expected, rtol=1e-6)
    np.testing.assert_array_equal(jacobian.grad[0].numpy(), 0)
    if metric == "condition_number":
        np.testing.assert_array_equal(jacobian.grad[1].numpy(), 0)
    # Distinct positive diagonal entries: differentiate min/max or max/min.
    expected_gradient = np.zeros((6, 3))
    expected_gradient[0, 0] = -1 / 9 if metric == "isotropy" else 1
    expected_gradient[2, 2] = 1 / 3 if metric == "isotropy" else -3
    np.testing.assert_allclose(jacobian.grad[2].numpy(), expected_gradient, rtol=1e-6)
    reference = _dexterity_from_jacobian(
        _solver(dtype, "numpy"),
        np.zeros((3, 3), dtype=dtype),
        jacobian.detach().numpy(),
        task="position",
    )
    np.testing.assert_allclose(
        values.detach().numpy(), getattr(reference, metric), rtol=1e-6
    )


@pytest.mark.parametrize("metric", ["isotropy", "condition_number"])
@pytest.mark.parametrize("task", ["position", "rotation", "pose"])
def test_regular_metric_derivatives_pass_finite_difference_check(metric, task):
    torch = pytest.importorskip("torch")
    solver = _solver()
    # Separate singular values avoid the SVD's nondifferentiable degeneracies.
    jacobian = torch.zeros((1, 6, 3), dtype=torch.float64)
    jacobian[0, :3] = torch.diag(torch.tensor([3.0, 2.0, 1.0]))
    jacobian[0, 3:] = torch.diag(torch.tensor([0.9, 0.5, 0.2]))
    jacobian.requires_grad_()

    def evaluate(value):
        result = _dexterity_from_jacobian(
            solver,
            torch.zeros((1, 3), dtype=torch.float64),
            value,
            task=task,
            weights=[0.7, 1.1, 1.3] * (2 if task == "pose" else 1),
        )
        return getattr(result, metric)

    assert torch.autograd.gradcheck(evaluate, (jacobian,))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_near_singular_condition_mask_matches_numpy_and_blocks_gradient(dtype):
    torch = pytest.importorskip("torch")
    precision = getattr(torch, dtype)
    eps = torch.finfo(precision).eps
    jacobian = torch.zeros((2, 6, 3), dtype=precision)
    jacobian[0, :3] = torch.diag(torch.tensor([1, 0.5, eps], dtype=precision))
    jacobian[1, :3] = torch.diag(torch.tensor([1, 0.5, eps * 4], dtype=precision))
    jacobian.requires_grad_()
    values = _dexterity_from_jacobian(
        _solver(dtype),
        torch.zeros((2, 3), dtype=precision),
        jacobian,
        task="position",
    ).condition_number
    assert torch.isinf(values[0]) and torch.isfinite(values[1])
    torch.where(torch.isfinite(values), values, 0).sum().backward()
    assert torch.isfinite(jacobian.grad).all()
    np.testing.assert_array_equal(jacobian.grad[0].numpy(), 0)
    reference = _dexterity_from_jacobian(
        _solver(dtype, "numpy"),
        np.zeros((2, 3), dtype=dtype),
        jacobian.detach().numpy(),
        task="position",
    ).condition_number
    np.testing.assert_allclose(values.detach().numpy(), reference, rtol=1e-6)


@pytest.mark.parametrize("metric", ["isotropy", "condition_number"])
def test_public_dexterity_zero_rotation_task_does_not_poison_joint_gradient(metric):
    torch = pytest.importorskip("torch")
    joints = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float64, requires_grad=True)
    values = getattr(_solver().dexterity(joints, task="rotation"), metric)
    torch.where(torch.isfinite(values), values, 0).sum().backward()
    assert joints.grad is not None
    np.testing.assert_array_equal(joints.grad.numpy(), 0)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("weights", [[1e100] * 3, [1e-100] * 3, [-1e-100, 1, 1]])
def test_weight_precision_failure_precedes_jacobian(backend, weights, monkeypatch):
    solver = _solver("float32", backend)

    def unexpected(*args, **kwargs):
        pytest.fail("invalid weights must be rejected before Jacobian computation")

    monkeypatch.setattr(solver, "_jacobian_batch", unexpected)
    with np.errstate(over="raise", invalid="raise"):
        with pytest.raises(ValueError, match="weights"):
            solver.dexterity([0, 0, 0], weights=weights)
