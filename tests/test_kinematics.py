from pathlib import Path

import numpy as np
import pytest

from workspace_analyzer import (
    CartesianConfig,
    KinematicsSolver,
    RobotModel,
    SamplingConfig,
    SolverConfig,
    WorkspaceAnalyzer,
)

URDF = Path(__file__).parent / "fixtures/two_link.urdf"


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_fk_and_automatic_chain(backend):
    if backend == "torch":
        pytest.importorskip("torch")
    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF), config=SolverConfig(backend=backend)
    )
    assert solver.joint_names == ("shoulder", "elbow")
    pose = solver.forward([0, 0])
    if hasattr(pose, "detach"):
        pose = pose.cpu().numpy()
    np.testing.assert_allclose(pose[:3, 3], [2, 0, 0], atol=1e-9)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_batched_fk_jacobian_and_position_ik(backend):
    if backend == "torch":
        pytest.importorskip("torch")
    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF),
        config=SolverConfig(backend=backend, max_iterations=300),
    )
    q = np.array([[0.3, 0.7], [-0.4, 0.5]])
    targets = solver.forward(q)
    jac = solver.jacobian(q)
    assert tuple(jac.shape) == (2, 6, 2)
    result = solver.inverse(targets, seed=q + 0.1, position_only=True)
    success = (
        result.success.detach().cpu().numpy()
        if hasattr(result.success, "detach")
        else result.success
    )
    assert success.all()


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_analytic_position_jacobian_matches_finite_difference(backend):
    if backend == "torch":
        pytest.importorskip("torch")
    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF), config=SolverConfig(backend=backend)
    )
    q = np.array([[0.3, 0.7], [-0.4, 0.5]])
    jacobian = solver.jacobian(q)
    if hasattr(jacobian, "detach"):
        jacobian = jacobian.detach().cpu().numpy()
    epsilon = 1e-7
    for joint in range(solver.dof):
        plus, minus = q.copy(), q.copy()
        plus[:, joint] += epsilon
        minus[:, joint] -= epsilon
        p_plus, p_minus = solver.forward(plus), solver.forward(minus)
        if hasattr(p_plus, "detach"):
            p_plus, p_minus = p_plus.cpu().numpy(), p_minus.cpu().numpy()
        finite_difference = (p_plus[:, :3, 3] - p_minus[:, :3, 3]) / (2 * epsilon)
        np.testing.assert_allclose(jacobian[:, :3, joint], finite_difference, atol=1e-8)


def test_ik_multistart_preserves_target_batch():
    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF),
        config=SolverConfig(backend="numpy", max_iterations=300),
    )
    target = solver.forward([[0.3, 0.7], [-0.4, 0.5]])
    result = solver.inverse(target, position_only=True, restarts=3)
    assert result.positions.shape == (2, 2)
    assert result.success.shape == (2,)
    assert result.success.all()


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_ik_multistart_handles_half_turn_orientation(backend):
    if backend == "torch":
        pytest.importorskip("torch")
    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF),
        config=SolverConfig(backend=backend, max_iterations=300),
    )
    target = solver.forward([np.pi, 0.0])
    result = solver.inverse(target, seed=[0.0, 0.0], restarts=4)
    success = (
        result.success.item() if hasattr(result.success, "item") else result.success
    )
    assert success
    assert float(result.residual) < solver.config.tolerance


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_cartesian_analysis_classifies_targets(backend):
    if backend == "torch":
        pytest.importorskip("torch")
    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF),
        config=SolverConfig(backend=backend, max_iterations=200),
    )
    result = WorkspaceAnalyzer(solver).analyze_cartesian(
        CartesianConfig(
            bounds=np.array([[-2.5, 2.5], [-2.5, 2.5], [-1e-6, 1e-6]]),
            sampling=SamplingConfig(num_samples=64, batch_size=32, seed=7),
            position_only=True,
            restarts=2,
            reference_joints=np.array([0.1, -0.1]),
        )
    )
    assert result.points.shape == (64, 3)
    assert result.joint_positions.shape == (64, 2)
    assert result.reachable.shape == (64,)
    assert result.residual.shape == (64,)
    assert result.reachable.any()
    assert (~result.reachable).any()
    assert result.metadata["mode"] == "cartesian"
    assert result.metadata["reference_joints"] == [0.1, -0.1]
