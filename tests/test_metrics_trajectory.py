from pathlib import Path

import numpy as np
import pytest

from workspace_analyzer import KinematicsSolver, RobotModel, SolverConfig

URDF = Path(__file__).parent / "fixtures/two_link.urdf"


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_dexterity_metrics_are_batched_and_bounded(backend):
    if backend == "torch":
        pytest.importorskip("torch")
    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF), config=SolverConfig(backend=backend)
    )
    metrics = solver.dexterity([[0.0, 0.0], [0.3, 0.8]])
    singular = _numpy(metrics.singular_values)
    isotropy = _numpy(metrics.isotropy)
    margin = _numpy(metrics.joint_limit_margin)
    assert singular.shape == (2, 2)
    assert np.all((isotropy >= 0) & (isotropy <= 1))
    assert np.all((margin >= 0) & (margin <= 1))
    assert isotropy[0] == pytest.approx(0.0)
    assert isotropy[1] > isotropy[0]


def test_dexterity_rejects_unknown_task():
    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF), config=SolverConfig(backend="numpy")
    )
    with pytest.raises(ValueError, match="task"):
        solver.dexterity([0.0, 0.0], task="invalid")


@pytest.mark.parametrize("weights", [[1.0, 2.0], [1.0, -1.0, 1.0], [0.0, 0.0, 0.0]])
def test_dexterity_rejects_invalid_weights(weights):
    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF), config=SolverConfig(backend="numpy")
    )
    with pytest.raises(ValueError, match="weights"):
        solver.dexterity([0.2, 0.7], weights=weights)


def test_directional_dexterity_weights_change_singular_values():
    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF), config=SolverConfig(backend="numpy")
    )
    unweighted = solver.dexterity([0.2, 0.7])
    weighted = solver.dexterity([0.2, 0.7], weights=[1.0, 0.25, 1.0])
    assert not np.allclose(weighted.singular_values, unweighted.singular_values)


def test_trajectory_ik_warm_start_and_continuity_metrics():
    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF),
        config=SolverConfig(backend="numpy", max_iterations=300),
    )
    expected = np.column_stack((np.linspace(0.2, 0.5, 8), np.linspace(0.7, 0.4, 8)))
    targets = solver.forward(expected)
    result = solver.solve_trajectory(
        targets,
        seed=expected[0],
        position_only=True,
        failure_restarts=4,
        dt=0.05,
    )
    assert result.success.all()
    assert result.target_poses.shape == (8, 4, 4)
    assert result.positions.shape == expected.shape
    assert result.joint_delta.shape == (7, 2)
    assert result.velocity.shape == (7, 2)
    assert result.acceleration.shape == (6, 2)
    assert result.velocity_violation.shape == (7, 2)
    assert result.minimum_singular_value.shape == (8,)
    assert result.joint_limit_margin.shape == (8,)
    assert result.max_joint_jump < 0.1
    summary = result.summary()
    assert summary["success_rate"] == 1.0
    assert summary["velocity_violations"] == int(result.velocity_violation.sum())
    assert summary["max_normalized_joint_jump"] > 0
    assert summary["joint_path_length"] > 0
    assert summary["max_joint_velocity"] > 0
    assert summary["rms_joint_velocity"] > 0
    assert summary["max_joint_acceleration"] >= 0


def test_failed_trajectory_frame_invalidates_adjacent_segments():
    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF),
        config=SolverConfig(backend="numpy", max_iterations=20),
    )
    reachable = solver.forward([0.2, 0.7])
    unreachable = reachable.copy()
    unreachable[:3, 3] = [10.0, 0.0, 0.0]
    result = solver.solve_trajectory(
        np.stack((reachable, unreachable, reachable)),
        seed=[0.2, 0.7],
        position_only=True,
        failure_restarts=2,
        dt=0.1,
    )
    np.testing.assert_array_equal(result.success, [True, False, True])
    assert np.isnan(result.joint_delta).all()
    assert np.isnan(result.minimum_singular_value[1])
    assert np.isnan(result.joint_limit_margin[1])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"failure_restarts": 0},
        {"dt": 0.0},
        {"jump_repair_threshold": 0.0},
        {"posture_gain": -0.1},
    ],
)
def test_trajectory_ik_rejects_invalid_options(kwargs):
    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF), config=SolverConfig(backend="numpy")
    )
    with pytest.raises(ValueError):
        solver.solve_trajectory(np.eye(4)[None], position_only=True, **kwargs)


def test_trajectory_accepts_nonuniform_timestamps():
    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF), config=SolverConfig(backend="numpy")
    )
    q = np.array([[0.2, 0.7], [0.21, 0.69], [0.23, 0.67]])
    targets = solver.forward(q)
    result = solver.solve_trajectory(
        targets,
        seed=q[0],
        position_only=True,
        timestamps=[0.0, 0.1, 0.3],
    )
    assert result.velocity.shape == (2, 2)
    assert result.acceleration.shape == (1, 2)
    with pytest.raises(ValueError, match="either dt or timestamps"):
        solver.solve_trajectory(
            targets,
            position_only=True,
            dt=0.1,
            timestamps=[0.0, 0.1, 0.3],
        )
    with pytest.raises(ValueError, match="strictly increase"):
        solver.solve_trajectory(targets, position_only=True, timestamps=[0.0, 0.1, 0.1])


def test_periodic_trajectory_closes_joint_path_after_cartesian_reprojection():
    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF),
        config=SolverConfig(backend="numpy", max_iterations=300),
    )
    q = np.array([[0.2, 0.7], [0.3, 0.6], [0.4, 0.5], [0.2, 0.7]])
    targets = solver.forward(q)
    result = solver.solve_trajectory(
        targets,
        seed=q[0],
        position_only=True,
        posture_gain=0.005,
    )
    assert result.success.all()
    np.testing.assert_allclose(result.positions[-1], result.positions[0], atol=1e-12)
    assert result.summary()["normalized_closure_joint_error"] == pytest.approx(0.0)


def _numpy(value):
    return (
        value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    )


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("direction", [1, -1])
def test_continuous_trajectory_crosses_wrap_boundary_without_restarts(
    backend, direction
):
    if backend == "torch":
        pytest.importorskip("torch")
    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF.with_name("continuous.urdf")),
        config=SolverConfig(backend=backend, device="cpu"),
    )
    q = direction * np.linspace(3.0, 3.4, 9)[:, None]
    targets = solver.forward(q)
    result = solver.solve_trajectory(
        targets,
        seed=q[0],
        failure_restarts=1,
        dt=0.1,
        enforce_loop_closure=False,
    )
    assert result.success.all()
    np.testing.assert_allclose(result.positions, q, atol=1e-5)
    np.testing.assert_allclose(result.velocity, direction * 0.5, atol=1e-4)
    assert not result.velocity_violation.any()
    np.testing.assert_allclose(result.joint_limit_margin, 1.0)
    np.testing.assert_allclose(
        _numpy(solver.forward(result.positions)), _numpy(targets), atol=1e-5
    )


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_continuous_multiturn_seed_preserves_equivalent_pose(backend):
    if backend == "torch":
        pytest.importorskip("torch")
    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF.with_name("continuous.urdf")),
        config=SolverConfig(backend=backend, max_iterations=1),
    )
    seed = [10 * np.pi + 0.4]
    target = solver.forward(seed)
    result = solver.inverse(target, seed=seed)
    assert bool(result.success)
    np.testing.assert_allclose(_numpy(result.positions), [0.4], atol=1e-12)
    np.testing.assert_allclose(_numpy(solver.forward(result.positions)), _numpy(target))


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_continuous_full_revolution_remains_continuous_after_loop_refinement(backend):
    if backend == "torch":
        pytest.importorskip("torch")
    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF.with_name("continuous.urdf")),
        config=SolverConfig(backend=backend),
    )
    q = np.linspace(0, 2 * np.pi, 33)[:, None]
    targets = _numpy(solver.forward(q))
    targets[-1] = targets[0]
    result = solver.solve_trajectory(targets, seed=q[0], failure_restarts=1)
    assert result.success.all()
    assert result.max_joint_jump < 0.2
    np.testing.assert_allclose(result.positions, q, atol=1e-5)


def test_nearly_closed_trajectory_keeps_last_target_and_accurate_residual():
    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF),
        config=SolverConfig(backend="numpy", tolerance=1e-11, max_iterations=300),
    )
    q = [[0.2, 0.7], [0.3, 0.6], [0.2 + 2e-9, 0.7]]
    targets = solver.forward(q)
    result = solver.solve_trajectory(
        targets, seed=q[0], position_only=True, posture_gain=0.0
    )
    assert result.success.all()
    errors = np.linalg.norm(
        solver.forward(result.positions)[:, :3, 3] - targets[:, :3, 3], axis=1
    )
    np.testing.assert_allclose(result.residual, errors, atol=1e-14)
    assert errors[-1] <= solver.config.tolerance


def test_failed_frames_do_not_insert_turns_into_continuous_joint_path():
    from workspace_analyzer.trajectory import _unwrap_continuous

    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF.with_name("continuous.urdf")),
        config=SolverConfig(backend="numpy"),
    )
    q = np.array([[0.0], [2.0], [-2.0], [0.1], [0.2]])
    success = np.array([True, False, False, True, True])
    unwrapped = _unwrap_continuous(solver, q, success)
    np.testing.assert_array_equal(unwrapped[success], q[success])


def test_trajectory_cancellation_propagates_to_closed_loop_refinement(monkeypatch):
    import threading

    from workspace_analyzer import AnalysisCancelled

    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF), config=SolverConfig(backend="numpy")
    )
    q = np.array([[0.2, 0.7], [0.3, 0.6], [0.2, 0.7]])
    targets = solver.forward(q)
    cancel = threading.Event()
    original = solver.inverse
    saw_batch = False

    def cancel_refinement(target, **kwargs):
        nonlocal saw_batch
        assert kwargs.get("cancel_event") is cancel
        if np.asarray(target).ndim == 3:
            saw_batch = True
            cancel.set()
        return original(target, **kwargs)

    monkeypatch.setattr(solver, "inverse", cancel_refinement)
    with pytest.raises(AnalysisCancelled):
        solver.solve_trajectory(targets, seed=q[0], cancel_event=cancel)
    assert saw_batch


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_trajectory_reuses_successful_restart_repair(backend, monkeypatch):
    if backend == "torch":
        pytest.importorskip("torch")
    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF),
        config=SolverConfig(backend=backend, max_iterations=8),
    )
    target = solver.forward([[1.4343916796455298, 0.7894721162374556]])
    original = solver.inverse
    calls = []

    def record(target, **kwargs):
        calls.append(kwargs.get("restarts", 1))
        return original(target, **kwargs)

    monkeypatch.setattr(solver, "inverse", record)
    result = solver.solve_trajectory(
        target,
        seed=[0, 0],
        position_only=True,
        failure_restarts=8,
        enforce_loop_closure=False,
    )
    assert result.success.all()
    assert calls == [1, 8]
    np.testing.assert_allclose(
        _numpy(solver.forward(result.positions))[:, :3, 3],
        _numpy(target)[:, :3, 3],
        atol=solver.config.tolerance,
    )


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_trajectory_refinement_and_metrics_respect_batch_size(
    backend, batch_size, monkeypatch
):
    if backend == "torch":
        pytest.importorskip("torch")
    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF), config=SolverConfig(backend=backend)
    )
    phase = np.linspace(0, 2 * np.pi, 9)
    q = np.column_stack((0.3 + 0.03 * np.sin(phase), 0.5 + 0.04 * np.sin(phase)))
    targets = solver.forward(q)
    reference = solver.solve_trajectory(targets, seed=q[0])
    inverse = solver.inverse
    dexterity = solver.dexterity
    refinement_batches, metric_batches = [], []

    def record_inverse(target, **kwargs):
        if len(target.shape) == 3:
            refinement_batches.append(len(target))
        return inverse(target, **kwargs)

    def record_metrics(q, **kwargs):
        metric_batches.append(len(q))
        return dexterity(q, **kwargs)

    monkeypatch.setattr(solver, "inverse", record_inverse)
    monkeypatch.setattr(solver, "dexterity", record_metrics)
    result = solver.solve_trajectory(targets, seed=q[0], batch_size=batch_size)
    assert sum(refinement_batches) == sum(metric_batches) == 9
    assert max(refinement_batches) <= batch_size
    assert max(metric_batches) <= batch_size
    np.testing.assert_allclose(result.positions, reference.positions, atol=1e-10)
    np.testing.assert_allclose(result.residual, reference.residual, atol=1e-10)
    np.testing.assert_allclose(
        result.minimum_singular_value, reference.minimum_singular_value, atol=1e-10
    )


def test_failed_trajectory_frames_skip_dexterity_computation(monkeypatch):
    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF),
        config=SolverConfig(backend="numpy", max_iterations=1),
    )
    targets = np.repeat(np.eye(4)[None], 5, axis=0)
    targets[:, :3, 3] = [10, 0, 0]

    def unexpected(*args, **kwargs):
        pytest.fail("failed frames should not compute dexterity")

    monkeypatch.setattr(solver, "dexterity", unexpected)
    result = solver.solve_trajectory(targets, failure_restarts=1, batch_size=2)
    assert not result.success.any()
    assert np.isnan(result.minimum_singular_value).all()
    assert np.isnan(result.joint_limit_margin).all()


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_trajectory_rejects_invalid_batch_sizes(batch_size):
    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF), config=SolverConfig(backend="numpy")
    )
    with pytest.raises(ValueError, match="batch_size"):
        solver.solve_trajectory(solver.forward([[0.2, 0.3]]), batch_size=batch_size)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_condition_number_is_invariant_to_uniform_task_scaling(backend, dtype):
    if backend == "torch":
        pytest.importorskip("torch")
    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF), config=SolverConfig(backend=backend, dtype=dtype)
    )
    q = [[0.2, 0.7]]
    reference = solver.dexterity(q)
    for scale in [1e-12, 1e12]:
        scaled = solver.dexterity(q, weights=[scale] * 3)
        np.testing.assert_allclose(
            _numpy(scaled.condition_number),
            _numpy(reference.condition_number),
            rtol=2e-6,
        )


def test_invalid_weights_are_rejected_before_jacobian_computation(monkeypatch):
    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF), config=SolverConfig(backend="numpy")
    )

    def unexpected(*args, **kwargs):
        pytest.fail("invalid weights should be rejected before computing Jacobians")

    monkeypatch.setattr(solver, "_jacobian_batch", unexpected)
    with pytest.raises(ValueError, match="weights"):
        solver.dexterity([[0.2, 0.7]], weights=[0, 0, 0])


def test_failed_refinement_batch_keeps_entire_original_trajectory(monkeypatch):
    from workspace_analyzer import IKResult
    from workspace_analyzer.trajectory import _refine_closed_loop

    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF), config=SolverConfig(backend="numpy")
    )
    q = np.array([[0.2, 0.7], [0.21, 0.69], [0.22, 0.68], [0.2, 0.7]])
    targets = solver.forward(q)
    success = np.ones(4, dtype=bool)
    residual = np.zeros(4)
    original = solver.inverse
    calls = 0

    def fail_second_batch(targets, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            return IKResult(np.zeros((2, 2)), np.zeros(2, dtype=bool), np.ones(2), 1)
        return original(targets, **kwargs)

    monkeypatch.setattr(solver, "inverse", fail_second_batch)
    refined, solved, errors = _refine_closed_loop(
        solver,
        targets,
        q,
        success,
        residual,
        position_only=True,
        posture_gain=0.005,
        batch_size=2,
    )
    assert calls == 2
    np.testing.assert_array_equal(refined, q)
    np.testing.assert_array_equal(solved, success)
    np.testing.assert_array_equal(errors, residual)


def test_trajectory_metric_batches_honor_cancellation(monkeypatch):
    import threading

    from workspace_analyzer import AnalysisCancelled

    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF), config=SolverConfig(backend="numpy")
    )
    q = np.tile([0.2, 0.7], (5, 1))
    targets = solver.forward(q)
    cancel = threading.Event()
    original = solver.dexterity
    calls = []

    def cancel_after_metrics(q, **kwargs):
        calls.append(len(q))
        result = original(q, **kwargs)
        cancel.set()
        return result

    monkeypatch.setattr(solver, "dexterity", cancel_after_metrics)
    with pytest.raises(AnalysisCancelled):
        solver.solve_trajectory(
            targets,
            seed=q[0],
            enforce_loop_closure=False,
            cancel_event=cancel,
            batch_size=2,
        )
    assert calls == [2]
