"""Independent spatial-Jacobian and damped posture-projection checks."""

from pathlib import Path

import numpy as np
import pytest

from workspace_analyzer import create_solver
from workspace_analyzer.kinematics import _biased_dls, _dls

FIXTURE = Path(__file__).parent / "fixtures/mixed_chain.urdf"


def _numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else value


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_full_spatial_jacobian_matches_pose_derivatives(backend, dtype):
    if backend == "torch":
        pytest.importorskip("torch")
    solver = create_solver(FIXTURE, backend=backend, device="cpu", dtype=dtype)
    reference = create_solver(FIXTURE, backend="numpy", dtype="float64")
    q = np.array([[0.3, 0.2, 3.4], [-0.7, -0.1, -2.1]])
    pose, jacobian = map(_numpy, solver.forward_with_jacobian(q))
    expected_pose = reference.forward(q)
    expected_jacobian = np.empty((2, 6, 3))
    epsilon = 1e-6
    for joint in range(solver.dof):
        plus, minus = q.copy(), q.copy()
        plus[:, joint] += epsilon
        minus[:, joint] -= epsilon
        derivative = (reference.forward(plus) - reference.forward(minus)) / (
            2 * epsilon
        )
        # Spatial angular velocity: skew(omega) = R_dot R^T.
        skew = derivative[:, :3, :3] @ expected_pose[:, :3, :3].swapaxes(1, 2)
        expected_jacobian[:, :3, joint] = derivative[:, :3, 3]
        expected_jacobian[:, 3:, joint] = np.stack(
            (skew[:, 2, 1], skew[:, 0, 2], skew[:, 1, 0]), axis=-1
        )
    tolerance = 4e-7 if dtype == "float32" else 1e-9
    np.testing.assert_allclose(pose, expected_pose, atol=tolerance, rtol=tolerance)
    np.testing.assert_allclose(
        jacobian, expected_jacobian, atol=tolerance, rtol=tolerance
    )
    np.testing.assert_array_equal(jacobian[:, 3:, 1], 0)


@pytest.mark.parametrize(
    "backend,device", [("numpy", "cpu"), ("torch", "cpu"), ("torch", "cuda")]
)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("rows", [3, 6])
@pytest.mark.parametrize("columns", [1, 2, 3, 7])
def test_biased_dls_matches_explicit_damped_projection(
    backend, device, dtype, rows, columns
):
    rng = np.random.default_rng(26)
    jacobian = rng.normal(size=(3, rows, columns)).astype(dtype)
    jacobian[1, -1] = jacobian[1, 0]  # Rank-deficient task.
    jacobian[2] = 0  # Bias should survive a task with no authority.
    error = rng.normal(size=(3, rows)).astype(dtype)
    bias = rng.normal(size=(3, columns)).astype(dtype)
    expected = []
    for j, e, b in zip(jacobian.astype(float), error, bias):
        inverse = np.linalg.solve(j @ j.T + 0.1**2 * np.eye(rows), j).T
        expected.append(inverse @ e + (np.eye(columns) - inverse @ j) @ b)
    inputs = [jacobian, error, bias]
    if backend == "torch":
        torch = pytest.importorskip("torch")
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA device is unavailable")
        inputs = [torch.as_tensor(value, device=device) for value in inputs]
    actual = _biased_dls(inputs[0], inputs[1], 0.1, backend, inputs[2])
    tolerance = 3e-4 if dtype == "float32" else 2e-12
    np.testing.assert_allclose(_numpy(actual), expected, atol=tolerance, rtol=tolerance)
    np.testing.assert_array_equal(_numpy(actual)[2], bias[2])


@pytest.mark.parametrize("rows,columns", [(3, 2), (6, 3), (3, 7)])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_biased_dls_derivatives_match_finite_differences(rows, columns, device):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device is unavailable")
    rng = np.random.default_rng(26)
    inputs = tuple(
        torch.tensor(
            rng.normal(size=shape),
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        for shape in [(2, rows, columns), (2, rows), (2, columns)]
    )
    assert torch.autograd.gradcheck(
        lambda j, e, b: _biased_dls(j, e, 0.1, "torch", b), inputs
    )


@pytest.mark.parametrize("damping", [1e-3, 1e-4])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize(
    "backend,device", [("numpy", "cpu"), ("torch", "cpu"), ("torch", "cuda")]
)
def test_dls_retains_damping_at_rank_deficient_configurations(
    damping, dtype, backend, device
):
    s = np.dtype(dtype).type(np.sqrt(0.5))
    jacobian = np.array(
        [[[-2 * s, -s], [2 * s, s], [0, 0]], np.zeros((3, 2))],
        dtype=dtype,
    )
    error = np.array([[0.1, -0.2, 0.3], [0.1, 0.2, 0.3]], dtype=dtype)
    # Independent augmented least-squares reference avoids forming J @ J.T.
    expected = np.stack(
        [
            np.linalg.lstsq(
                np.vstack((j.astype(float), damping * np.eye(2))),
                np.r_[e, [0, 0]],
                rcond=None,
            )[0]
            for j, e in zip(jacobian, error, strict=True)
        ]
    )
    if backend == "torch":
        torch = pytest.importorskip("torch")
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA device is unavailable")
        jacobian = torch.as_tensor(jacobian, device=device)
        error = torch.as_tensor(error, device=device)
    actual = _dls(jacobian, error, damping, backend)
    assert actual.dtype == jacobian.dtype
    np.testing.assert_allclose(_numpy(actual), expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "backend,device", [("numpy", "cpu"), ("torch", "cpu"), ("torch", "cuda")]
)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("damping", [1e-6, 1e-8])
def test_dls_rank_one_step_matches_svd_with_small_damping(
    backend, device, dtype, damping
):
    # Zero-position Jacobian of a planar arm with 0.3 m and 0.7 m links.
    jacobian = np.array(
        [[[0, 0], [1, 0.7], [0, 0]], [[1, 0], [0, 1], [0, 0]]], dtype=dtype
    )
    error = np.array([[0.4, 1, 0.2], [-0.2, 0.4, 0.1]], dtype=dtype)
    expected = []
    for matrix, residual in zip(jacobian.astype(float), error):
        left, singular, right = np.linalg.svd(matrix, full_matrices=False)
        expected.append(
            right.T @ (singular / (singular**2 + damping**2) * (left.T @ residual))
        )
    if backend == "torch":
        torch = pytest.importorskip("torch")
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA device is unavailable")
        jacobian = torch.as_tensor(jacobian, device=device)
        error = torch.as_tensor(error, device=device)
    actual = _dls(jacobian, error, damping, backend)
    assert actual.dtype == jacobian.dtype
    tolerance = 2e-7 if dtype == "float32" else 1e-12
    np.testing.assert_allclose(_numpy(actual), expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize(
    "backend,device", [("numpy", "cpu"), ("torch", "cpu"), ("torch", "cuda")]
)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("position_only", [True, False])
def test_ik_solves_reachable_target_with_small_damping(
    backend, device, dtype, position_only
):
    if backend == "torch":
        torch = pytest.importorskip("torch")
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA device is unavailable")
    solver = create_solver(
        FIXTURE.parent / "two_link.urdf",
        backend=backend,
        device=device,
        dtype=dtype,
        damping=1e-4,
    )
    target = solver.forward([np.pi / 4, 0.1])
    seed = np.array([np.pi / 4, 0], dtype=dtype)
    if backend == "torch":
        seed = torch.tensor(seed, device=device, requires_grad=True)
    result = solver.inverse(target, seed=seed, position_only=position_only)
    assert bool(result.success)
    np.testing.assert_allclose(
        _numpy(solver.forward(result.positions))[:3, 3],
        _numpy(target)[:3, 3],
        atol=solver.config.tolerance,
    )
    if not position_only:
        np.testing.assert_allclose(
            _numpy(solver.forward(result.positions))[:3, :3],
            _numpy(target)[:3, :3],
            atol=solver.config.tolerance,
        )
    if backend == "torch":
        result.positions.sum().backward()
        assert bool(torch.isfinite(seed.grad).all())


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize(
    "centering,posture", [(0.0, 0.0), (0.2, 0.0), (0.0, 0.3), (0.2, 0.3)]
)
def test_one_iteration_matches_separate_posture_projections(
    backend, centering, posture, monkeypatch
):
    if backend == "torch":
        pytest.importorskip("torch")
    import workspace_analyzer.kinematics as kinematics

    solver = create_solver(
        FIXTURE,
        backend=backend,
        device="cpu",
        max_iterations=1,
        damping=0.1,
        joint_centering_gain=centering,
    )
    q = np.array([[0.3, 0.2, 3.0], [-0.7, -0.1, -3.0]])
    reference = np.array([[0.1, -0.2, -3.0], [0.3, 0.1, 3.0]])
    target = _numpy(solver.forward(q)).copy()
    target[:, :3, 3] += [0.02, -0.01, 0.03]
    jacobian = _numpy(solver.jacobian(q))[:, :3]
    limits = solver.joint_limits
    center_bias = (limits.mean(axis=1) - q) / (np.diff(limits, axis=1)[:, 0] / 2)
    center_bias[:, 2] = 0
    delta = reference - q
    delta[:, 2] = (delta[:, 2] + np.pi) % (2 * np.pi) - np.pi
    posture_bias = delta / np.diff(limits, axis=1)[:, 0]
    expected = []
    for j, c, p in zip(jacobian, center_bias, posture_bias):
        inverse = np.linalg.solve(j @ j.T + 0.1**2 * np.eye(3), j).T
        projection = np.eye(3) - inverse @ j
        expected.append(
            inverse @ [0.02, -0.01, 0.03]
            + centering * (projection @ c)
            + posture * (projection @ p)
        )
    expected = q + 0.8 * np.clip(expected, -0.35, 0.35)
    expected = np.clip(expected, limits[:, 0], limits[:, 1])
    expected[:, 2] = (expected[:, 2] + np.pi) % (2 * np.pi) - np.pi
    original = kinematics._dls
    calls = []

    def count_solve(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(kinematics, "_dls", count_solve)
    result = solver.inverse(
        target,
        seed=q,
        position_only=True,
        posture_reference=reference,
        posture_gain=posture,
    )
    np.testing.assert_allclose(_numpy(result.positions), expected, atol=1e-12)
    assert len(calls) == 1
    residual = np.linalg.norm(
        _numpy(solver.forward(result.positions))[:, :3, 3] - target[:, :3, 3], axis=-1
    )
    np.testing.assert_allclose(_numpy(result.residual), residual, atol=1e-12)
