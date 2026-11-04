import threading
import warnings
from pathlib import Path

import numpy as np
import pytest

from workspace_analyzer import (
    AnalysisCancelled,
    CartesianConfig,
    KinematicsSolver,
    RobotModel,
    SamplingConfig,
    SolverConfig,
    WorkspaceAnalyzer,
    WorkspaceConfig,
)
from workspace_analyzer.kinematics import _quaternion_rotation_error, _select_best

URDF = Path(__file__).parent / "fixtures/two_link.urdf"


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("container", ["array", "list"])
@pytest.mark.parametrize("imaginary", [0, 0.5])
def test_complex_kinematic_inputs_are_rejected_before_cast(
    backend, dtype, container, imaginary
):
    if backend == "torch":
        pytest.importorskip("torch")
    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF), config=SolverConfig(backend=backend, dtype=dtype)
    )
    configuration = np.array([0.3 + imaginary * 1j, 0.7])
    target = np.eye(4, dtype=complex)
    target[0, 3] = imaginary * 1j
    if container == "list":
        configuration, target = configuration.tolist(), target.tolist()
    # Even zero-imaginary complex values must not silently cast to real inputs.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for operation in ("forward", "jacobian", "forward_with_jacobian", "dexterity"):
            with pytest.raises(ValueError, match="real values"):
                getattr(solver, operation)(configuration)
        for arguments in (
            {"target": target},
            {"target": np.eye(4), "seed": configuration},
            {
                "target": np.eye(4),
                "posture_reference": configuration,
                "posture_gain": 0.1,
            },
        ):
            with pytest.raises(ValueError, match="real values"):
                solver.inverse(**arguments)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("imaginary", [0, 0.5])
def test_complex_tensor_inputs_are_rejected_without_device_transfer(
    backend, dtype, device, imaginary, monkeypatch
):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device is unavailable")
    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF),
        config=SolverConfig(
            backend=backend,
            device=device if backend == "torch" else "cpu",
            dtype="float32" if dtype == "complex64" else "float64",
        ),
    )
    configuration = torch.tensor(
        [0.3 + imaginary * 1j, 0.7],
        dtype=getattr(torch, dtype),
        device=device,
        requires_grad=True,
    )
    target = torch.eye(4, dtype=getattr(torch, dtype), device=device)
    target[0, 3] = imaginary * 1j

    def unexpected(*args, **kwargs):
        pytest.fail("complex rejection must only inspect dtype metadata")

    monkeypatch.setattr(torch.Tensor, "cpu", unexpected)
    monkeypatch.setattr(torch.Tensor, "numpy", unexpected)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for operation in ("forward", "jacobian", "forward_with_jacobian", "dexterity"):
            with pytest.raises(ValueError, match="real values"):
                getattr(solver, operation)(configuration)
        for arguments in (
            {"target": target},
            {"target": np.eye(4), "seed": configuration},
            {
                "target": np.eye(4),
                "posture_reference": configuration,
                "posture_gain": 0.1,
            },
        ):
            with pytest.raises(ValueError, match="real values"):
                solver.inverse(**arguments)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_real_tensor_conversion_preserves_fk_gradient_and_device(dtype, device):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device is unavailable")
    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF),
        config=SolverConfig(backend="torch", device=device, dtype=dtype),
    )
    configuration = torch.tensor(
        [0.3, 0.7], dtype=getattr(torch, dtype), device=device, requires_grad=True
    )
    pose, jacobian = solver.forward_with_jacobian(configuration)
    pose[:3, 3].sum().backward()
    torch.testing.assert_close(configuration.grad, jacobian[:3].sum(dim=0))
    assert pose.dtype == configuration.dtype
    assert pose.device == configuration.device
    np.testing.assert_allclose(
        _to_numpy(pose), _to_numpy(solver.forward([0.3, 0.7])), rtol=1e-6, atol=1e-7
    )


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
    combined_pose, combined_jac = solver.forward_with_jacobian(q)
    assert tuple(jac.shape) == (2, 6, 2)
    if hasattr(targets, "detach"):
        np.testing.assert_allclose(combined_pose.cpu(), targets.cpu())
        np.testing.assert_allclose(combined_jac.cpu(), jac.cpu())
    else:
        np.testing.assert_allclose(combined_pose, targets)
        np.testing.assert_allclose(combined_jac, jac)
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


def test_multistart_can_prefer_nearest_successful_seed():
    q = np.array([[0.1, 0.1], [2.0, 2.0]])
    success = np.array([True, True])
    residual = np.array([9e-6, 1e-8])
    selected, selected_success, selected_residual = _select_best(
        q,
        success,
        residual,
        restarts=2,
        targets=1,
        backend="numpy",
        reference=np.zeros((1, 2)),
        limits=np.array([[-np.pi, np.pi], [-np.pi, np.pi]]),
        continuous=np.array([False, False]),
    )
    np.testing.assert_allclose(selected[0], [0.1, 0.1])
    assert selected_success[0]
    assert selected_residual[0] == pytest.approx(9e-6)


def test_ik_rescue_preserves_shapes_and_is_deterministic():
    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF),
        config=SolverConfig(backend="numpy", max_iterations=50),
    )
    targets = solver.forward([[0.3, 0.7], [-0.4, 0.5]])
    first = solver.inverse(targets, restarts=1, rescue_restarts=3)
    second = solver.inverse(targets, restarts=1, rescue_restarts=3)
    assert first.positions.shape == (2, 2)
    np.testing.assert_allclose(first.positions, second.positions)
    np.testing.assert_array_equal(first.success, second.success)


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


def test_joint_workspace_includes_dexterity_metrics():
    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF), config=SolverConfig(backend="numpy")
    )
    result = WorkspaceAnalyzer(
        solver,
        WorkspaceConfig(SamplingConfig(num_samples=32, batch_size=7)),
    ).analyze()
    assert set(result.metrics) == {
        "minimum_singular_value",
        "isotropy",
        "joint_limit_margin",
    }
    for values in result.metrics.values():
        assert values.shape == (32,)
        assert np.isfinite(values).all()


def test_cartesian_analysis_progress_and_cancellation():
    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF), config=SolverConfig(backend="numpy")
    )
    analyzer = WorkspaceAnalyzer(solver)
    config = CartesianConfig(
        bounds=np.array([[-1.0, 1.0], [-1.0, 1.0], [-1e-6, 1e-6]]),
        sampling=SamplingConfig(num_samples=12, batch_size=4),
        rescue_restarts=0,
    )
    progress = []
    analyzer.analyze_cartesian(config, progress_callback=progress.append)
    assert progress == [pytest.approx(1 / 3), pytest.approx(2 / 3), 1.0]
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(AnalysisCancelled):
        analyzer.analyze_cartesian(config, cancel_event=cancelled)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bounds": np.array([[np.nan, 1.0], [0.0, 1.0], [0.0, 1.0]])},
        {
            "bounds": np.array([[0.0, 1.0]] * 3),
            "reference_pose": np.full((4, 4), np.nan),
        },
        {
            "bounds": np.array([[0.0, 1.0]] * 3),
            "reference_joints": np.array([0.0, np.inf]),
        },
    ],
)
def test_cartesian_config_rejects_nonfinite_values(kwargs):
    with pytest.raises(ValueError):
        CartesianConfig(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"backend": "invalid"}, "backend"),
        ({"dtype": "float16"}, "dtype"),
        ({"max_iterations": 0}, "max_iterations"),
        ({"tolerance": 0.0}, "tolerance"),
        ({"damping": float("nan")}, "damping"),
    ],
)
def test_solver_config_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SolverConfig(**kwargs)


def test_solver_rejects_invalid_inputs_and_clamps_seed():
    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF),
        config=SolverConfig(backend="numpy", max_iterations=300),
    )
    with pytest.raises(ValueError, match="joint positions"):
        solver.forward(np.zeros((1, 1, 2)))
    with pytest.raises(ValueError, match="finite"):
        solver.forward([np.nan, 0.0])
    target = np.eye(4)
    target[3, 3] = 2.0
    with pytest.raises(ValueError, match="homogeneous"):
        solver.inverse(target)
    target = np.eye(4)
    target[0, 0] = 2.0
    with pytest.raises(ValueError, match="orthonormal"):
        solver.inverse(target)
    # Position-only IK intentionally ignores target rotation.
    solver.inverse(target, position_only=True)
    reachable = solver.forward([0.2, 0.3])
    result = solver.inverse(reachable, seed=[100.0, -100.0], restarts=4)
    assert result.success
    assert np.all(result.positions >= solver.joint_limits[:, 0])
    assert np.all(result.positions <= solver.joint_limits[:, 1])


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_fk_jacobian_and_ik_preserve_dtype_and_singleton_batch(backend, dtype):
    if backend == "torch":
        pytest.importorskip("torch")
    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF),
        config=SolverConfig(backend=backend, dtype=dtype),
    )
    q = np.array([[0.3, 0.7]])
    pose, jacobian = solver.forward_with_jacobian(q)
    links = solver.forward(q, all_links=True)
    result = solver.inverse(pose, seed=q, position_only=True, restarts=2)
    assert result.positions.shape == (1, 2)
    assert result.success.shape == result.residual.shape == (1,)
    assert bool(result.success.all())
    for value in (pose, jacobian, result.positions, result.residual, *links.values()):
        assert str(value.dtype).endswith(dtype)
    single = solver.inverse(pose[0], seed=q[0], position_only=True)
    assert single.positions.shape == (2,)
    assert single.success.shape == single.residual.shape == ()
    # Retaining per-link transforms must not let mutation affect another link.
    links[solver.tip_link][...] = 0
    assert float(links[solver.base_link][0, 3, 3]) == 1.0


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("restarts,rescue_rounds", [(1, 0), (3, 0), (2, 2)])
@pytest.mark.parametrize("position_only", [True, False])
def test_ik_iteration_limit_residual_matches_returned_pose(
    backend, restarts, rescue_rounds, position_only
):
    if backend == "torch":
        pytest.importorskip("torch")
    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF),
        config=SolverConfig(backend=backend, max_iterations=1),
    )
    target = solver.forward([[0.3, 0.7], [-0.4, 0.5]])
    result = solver.inverse(
        target,
        seed=[[0.4, 0.8], [-0.3, 0.6]],
        position_only=position_only,
        restarts=restarts,
        rescue_restarts=2,
        rescue_rounds=rescue_rounds,
    )
    actual = _to_numpy(solver.forward(result.positions))
    target = _to_numpy(target)
    squared_error = np.sum((actual[:, :3, 3] - target[:, :3, 3]) ** 2, axis=1)
    if not position_only:
        relative = target[:, :3, :3] @ actual[:, :3, :3].swapaxes(1, 2)
        # The rotation error uses 2*sin(angle/2), whose square is 3-trace(R).
        squared_error += 3 - np.trace(relative, axis1=1, axis2=2)
    residual = _to_numpy(result.residual)
    np.testing.assert_allclose(residual, np.sqrt(squared_error), atol=1e-12)
    np.testing.assert_array_equal(
        _to_numpy(result.success), residual <= solver.config.tolerance
    )


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_ik_recognizes_convergence_on_final_step(backend):
    if backend == "torch":
        pytest.importorskip("torch")
    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF),
        config=SolverConfig(backend=backend, max_iterations=1, tolerance=0.01),
    )
    result = solver.inverse(
        solver.forward([0.3, 0.7]), seed=[0.31, 0.7], position_only=True
    )
    assert bool(result.success)
    assert float(result.residual) <= 0.01


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("position_only,damping", [(True, 1e-8), (False, 1e-6)])
def test_small_damping_ik_handles_a_singular_seed_in_a_batch(
    backend, dtype, position_only, damping
):
    if backend == "torch":
        pytest.importorskip("torch")
    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF),
        config=SolverConfig(
            backend=backend, device="cpu", dtype=dtype, damping=damping
        ),
    )
    targets = solver.forward([[0.2, 0.4], [0.31, 0.51]])
    result = solver.inverse(
        targets, seed=[[0, 0], [0.3, 0.5]], position_only=position_only
    )
    assert bool(result.success.all())
    assert str(result.positions.dtype).endswith(dtype)
    achieved = _to_numpy(solver.forward(result.positions))
    targets = _to_numpy(targets)
    residual = np.linalg.norm(achieved[:, :3, 3] - targets[:, :3, 3], axis=1)
    if not position_only:
        rotation_chord = np.linalg.norm(
            achieved[:, :3, :3] - targets[:, :3, :3], axis=(1, 2)
        ) / np.sqrt(2)
        residual = np.hypot(residual, rotation_chord)
    np.testing.assert_allclose(
        _to_numpy(result.residual),
        residual,
        atol=2e-7 if dtype == "float32" else 1e-12,
    )


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("count,batch", [(1, 4), (5, 2), (3, 1)])
def test_cartesian_analysis_handles_singleton_batches(backend, count, batch):
    if backend == "torch":
        pytest.importorskip("torch")
    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF),
        config=SolverConfig(backend=backend, max_iterations=20),
    )
    result = WorkspaceAnalyzer(solver).analyze_cartesian(
        CartesianConfig(
            bounds=np.array([[-2, 2], [-2, 2], [-1e-6, 1e-6]]),
            sampling=SamplingConfig(num_samples=count, batch_size=batch),
            restarts=2,
            rescue_restarts=2,
            rescue_rounds=1,
        )
    )
    assert result.joint_positions.shape == (count, 2)
    assert result.reachable.shape == result.residual.shape == (count,)
    pose = _to_numpy(solver.forward(result.joint_positions))
    expected = np.linalg.norm(pose[:, :3, 3] - result.points, axis=1)
    np.testing.assert_allclose(result.residual, expected, atol=1e-12)


@pytest.mark.parametrize("compute_jacobians", [True, False])
@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_workspace_batches_agree_with_direct_computation(
    compute_jacobians, backend, dtype
):
    if backend == "torch":
        pytest.importorskip("torch")
    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF), config=SolverConfig(backend=backend, dtype=dtype)
    )
    result = WorkspaceAnalyzer(
        solver,
        WorkspaceConfig(
            SamplingConfig(num_samples=10, batch_size=3), compute_jacobians
        ),
    ).analyze()
    np.testing.assert_allclose(
        result.points, _to_numpy(solver.forward(result.joint_positions))[:, :3, 3]
    )
    if compute_jacobians:
        quality = solver.dexterity(result.joint_positions)
        np.testing.assert_allclose(
            result.manipulability, _to_numpy(quality.manipulability)
        )
        for name, values in result.metrics.items():
            np.testing.assert_allclose(values, _to_numpy(getattr(quality, name)))
    else:
        assert result.manipulability is result.metrics is None


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_workspace_isotropy_is_invariant_to_link_scale(backend, dtype, tmp_path):
    if backend == "torch":
        pytest.importorskip("torch")
    robot = tmp_path / "small-arm.urdf"
    robot.write_text(URDF.read_text().replace('xyz="1 0 0"', 'xyz="1e-34 0 0"'))
    model = RobotModel.from_urdf(URDF)
    small_model = RobotModel.from_urdf(robot)
    config = WorkspaceConfig(SamplingConfig(num_samples=10, batch_size=3))
    solver_config = SolverConfig(backend=backend, dtype=dtype)
    normal = WorkspaceAnalyzer(KinematicsSolver(model, config=solver_config), config)
    small = WorkspaceAnalyzer(
        KinematicsSolver(small_model, config=solver_config), config
    )
    np.testing.assert_allclose(
        small.analyze().metrics["isotropy"],
        normal.analyze().metrics["isotropy"],
        rtol=2e-5,
    )


def test_cancelled_final_batch_is_not_published_or_cached(tmp_path):
    from workspace_analyzer import ResultCache

    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF),
        config=SolverConfig(backend="numpy", max_iterations=1),
    )
    cancelled = threading.Event()
    config = CartesianConfig(
        bounds=np.array([[-2, 2], [-2, 2], [-1e-6, 1e-6]]),
        sampling=SamplingConfig(num_samples=2, batch_size=2),
        rescue_restarts=0,
    )
    with pytest.raises(AnalysisCancelled):
        WorkspaceAnalyzer(solver).analyze_cartesian(
            config,
            cache=ResultCache(tmp_path),
            cancel_event=cancelled,
            progress_callback=lambda _: cancelled.set(),
        )
    assert not list(tmp_path.rglob("*.npz"))


@pytest.mark.parametrize("value", [1.5, True, float("nan")])
def test_solver_rejects_noninteger_iteration_and_restart_counts(value):
    with pytest.raises(ValueError, match="max_iterations"):
        SolverConfig(max_iterations=value)
    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF), config=SolverConfig(backend="numpy")
    )
    for name in ("restarts", "rescue_restarts", "rescue_rounds", "random_seed"):
        with pytest.raises(ValueError, match=name):
            solver.inverse(np.eye(4), **{name: value})


def _to_numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else value


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize(
    "backend,device", [("numpy", "cpu"), ("torch", "cpu"), ("torch", "cuda")]
)
def test_mixed_axis_chain_matches_spatial_finite_differences(backend, device, dtype):
    if backend == "torch":
        torch = pytest.importorskip("torch")
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA device is unavailable")
    model = RobotModel.from_urdf(URDF.parent / "mixed_chain.urdf")
    solver = KinematicsSolver(
        model, config=SolverConfig(backend=backend, device=device, dtype=dtype)
    )
    reference = KinematicsSolver(model, config=SolverConfig(backend="numpy"))
    configurations = np.array([[0.3, 0.2, -0.7], [-0.4, -0.1, 1.2]], dtype=dtype)
    pose, jacobian = solver.forward_with_jacobian(configurations)
    links = solver.forward(configurations, all_links=True)
    tolerance = 2e-6 if dtype == "float32" else 1e-9
    np.testing.assert_allclose(
        _to_numpy(pose), reference.forward(configurations), atol=tolerance
    )
    np.testing.assert_allclose(_to_numpy(links[solver.tip_link]), _to_numpy(pose))
    rotation = reference.forward(configurations)[:, :3, :3]
    epsilon = 1e-6
    for column in range(solver.dof):
        plus, minus = configurations.astype(float), configurations.astype(float)
        plus[:, column] += epsilon
        minus[:, column] -= epsilon
        derivative = (reference.forward(plus) - reference.forward(minus)) / (
            2 * epsilon
        )
        skew = derivative[:, :3, :3] @ rotation.swapaxes(1, 2)
        angular = np.stack((skew[:, 2, 1], skew[:, 0, 2], skew[:, 1, 0]), axis=1)
        expected = np.concatenate((derivative[:, :3, 3], angular), axis=1)
        np.testing.assert_allclose(
            _to_numpy(jacobian)[:, :, column], expected, atol=tolerance
        )
    if backend == "torch":
        configuration = torch.tensor(
            configurations[0],
            dtype=getattr(torch, dtype),
            device=device,
            requires_grad=True,
        )
        derivative = torch.autograd.functional.jacobian(solver.forward, configuration)
        torch.testing.assert_close(
            derivative[:3, 3],
            solver.jacobian(configuration)[:3],
            atol=tolerance,
            rtol=tolerance,
        )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize(
    "backend,device", [("numpy", "cpu"), ("torch", "cpu"), ("torch", "cuda")]
)
def test_solver_geometry_is_a_consistent_snapshot_of_mutable_model(
    backend, device, dtype
):
    if backend == "torch":
        torch = pytest.importorskip("torch")
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA device is unavailable")
    model = RobotModel.from_urdf(URDF)
    solver = KinematicsSolver(
        model, config=SolverConfig(backend=backend, device=device, dtype=dtype)
    )
    configuration = np.array([0.4, 0.7])
    original_pose, original_jacobian = solver.forward_with_jacobian(configuration)
    model.joints[0].axis[:] = [1, 0, 0]
    model.joints[0].origin[:3, 3] = [0.4, -0.3, 0.2]
    pose, jacobian = solver.forward_with_jacobian(configuration)
    np.testing.assert_array_equal(_to_numpy(pose), _to_numpy(original_pose))
    np.testing.assert_array_equal(_to_numpy(jacobian), _to_numpy(original_jacobian))

    epsilon = 1e-3 if dtype == "float32" else 1e-6
    tolerance = 2e-4 if dtype == "float32" else 1e-9
    for column in range(solver.dof):
        plus, minus = configuration.copy(), configuration.copy()
        plus[column] += epsilon
        minus[column] -= epsilon
        derivative = (
            _to_numpy(solver.forward(plus)) - _to_numpy(solver.forward(minus))
        ) / (2 * epsilon)
        skew = derivative[:3, :3] @ _to_numpy(pose)[:3, :3].T
        expected = np.r_[derivative[:3, 3], skew[2, 1], skew[0, 2], skew[1, 0]]
        np.testing.assert_allclose(
            _to_numpy(jacobian)[:, column], expected, atol=tolerance
        )


def _precision_stage(tmp_path, lower, upper, *, origin=0, axis="1 0 0"):
    source = tmp_path / "precision.urdf"
    source.write_text(
        '<robot name="precision"><link name="base"/><link name="tcp"/>'
        '<joint name="slide" type="prismatic"><parent link="base"/>'
        f'<child link="tcp"/><origin xyz="{origin} 0 0"/>'
        f'<axis xyz="{axis}"/><limit lower="{lower}" upper="{upper}"/>'
        "</joint></robot>"
    )
    return RobotModel.from_urdf(source)


@pytest.mark.parametrize(
    "lower,upper",
    [
        (1, 1.00000001),
        (0, 1e-50),
        (0, 1.401298464324817e-45),
        (1e39, 2e39),
        (-3e38, 3e38),
        (2e38, 3e38),
    ],
)
@pytest.mark.parametrize(
    "backend,device", [("numpy", "cpu"), ("torch", "cpu"), ("torch", "cuda")]
)
def test_precision_degenerate_limits_fail_before_solver_computation(
    tmp_path, lower, upper, backend, device
):
    if backend == "torch":
        torch = pytest.importorskip("torch")
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA device is unavailable")
    model = _precision_stage(tmp_path, lower, upper)
    with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
        with pytest.raises(ValueError, match="slide.*float32"):
            KinematicsSolver(
                model,
                config=SolverConfig(backend=backend, device=device, dtype="float32"),
            )
    solver = KinematicsSolver(
        model,
        config=SolverConfig(
            backend=backend,
            device=device,
            dtype="float64",
            max_iterations=1,
            joint_centering_gain=0.01,
        ),
    )
    seed = solver.joint_limits.mean(axis=1)
    assert np.isfinite(_to_numpy(solver.dexterity(seed).joint_limit_margin)).all()
    result = solver.inverse(np.eye(4), seed=seed, position_only=True)
    assert np.isfinite(_to_numpy(result.positions)).all()
    assert np.isfinite(_to_numpy(result.residual)).all()


@pytest.mark.parametrize(
    "backend,device", [("numpy", "cpu"), ("torch", "cpu"), ("torch", "cuda")]
)
def test_precision_overflowed_geometry_fails_at_construction(tmp_path, backend, device):
    if backend == "torch":
        torch = pytest.importorskip("torch")
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA device is unavailable")
    model = _precision_stage(tmp_path, -1, 1, origin=1e39)
    with np.errstate(over="raise", invalid="raise"):
        with pytest.raises(ValueError, match="slide.*float32"):
            KinematicsSolver(
                model,
                config=SolverConfig(backend=backend, device=device, dtype="float32"),
            )
    solver = KinematicsSolver(
        model, config=SolverConfig(backend=backend, device=device, dtype="float64")
    )
    assert np.isfinite(_to_numpy(solver.forward([0]))).all()


@pytest.mark.parametrize(
    "backend,device", [("numpy", "cpu"), ("torch", "cpu"), ("torch", "cuda")]
)
def test_normal_float32_limit_rounding_and_normalized_axes_remain_supported(
    tmp_path, backend, device
):
    if backend == "torch":
        torch = pytest.importorskip("torch")
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA device is unavailable")
    model = _precision_stage(tmp_path, 0, 1.1, axis="1e300 0 0")
    solver = KinematicsSolver(
        model,
        config=SolverConfig(backend=backend, device=device, dtype="float32"),
    )
    target = np.eye(4)
    target[0, 3] = 1.1
    result = solver.inverse(target, seed=[2], position_only=True)
    assert bool(result.success)
    assert float(result.positions[0]) == float(np.float32(1.1))
    assert float(result.residual) == 0


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_exact_full_pose_seed_converges_without_moving(backend, dtype):
    if backend == "torch":
        pytest.importorskip("torch")
    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF),
        config=SolverConfig(backend=backend, dtype=dtype, max_iterations=1),
    )
    q = np.random.default_rng(12).uniform(-np.pi, np.pi, (32, 2)).astype(dtype)
    result = solver.inverse(solver.forward(q), seed=q)
    assert bool(result.success.all())
    np.testing.assert_array_equal(_to_numpy(result.positions), q)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("angle", [1e-7, 0.2, np.pi - 1e-7, np.pi, np.pi + 1e-7])
def test_rotation_error_is_stable_for_mixed_axes_and_half_turns(backend, angle):
    axis = np.array([1.0, -2.0, 3.0]) / np.sqrt(14.0)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    rotation = (np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew))[
        None
    ]
    if backend == "torch":
        torch = pytest.importorskip("torch")
        rotation = torch.as_tensor(rotation)
    error = _to_numpy(_quaternion_rotation_error(rotation, backend))[0]
    expected = 2 * np.sin(angle / 2) * axis
    if angle > np.pi:
        expected = -expected
    if angle == np.pi:
        # Both signs describe the same shortest half turn.
        if np.dot(error, expected) < 0:
            expected = -expected
    np.testing.assert_allclose(error, expected, atol=1e-12)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("active", [[], [0], [0, 1]])
def test_cartesian_fixed_axes_match_analytic_stage(backend, active, tmp_path):
    from workspace_analyzer import AnalysisResult, ResultCache

    if backend == "torch":
        pytest.importorskip("torch")
    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF.parent / "cartesian_stage.urdf"),
        config=SolverConfig(backend=backend, device="cpu", max_iterations=30),
    )
    bounds = np.array([[0.25, 0.25], [-0.25, -0.25], [0.5, 0.5]])
    bounds[active] = [-0.5, 0.5]
    config = CartesianConfig(
        bounds=bounds,
        sampling=SamplingConfig("uniform", num_samples=9, batch_size=4),
        restarts=1,
        rescue_restarts=0,
    )
    analyzer = WorkspaceAnalyzer(solver)
    cache = ResultCache(tmp_path / "cache")
    result = analyzer.analyze_cartesian(config, cache=cache)
    assert result.reachable.all()
    np.testing.assert_allclose(result.joint_positions, result.points, atol=1e-5)
    fixed = np.ones(3, dtype=bool)
    fixed[active] = False
    np.testing.assert_array_equal(
        result.points[:, fixed], np.broadcast_to(bounds[fixed, 0], (9, fixed.sum()))
    )
    if active:
        assert len(np.unique(result.points, axis=0)) == 9
    replay = analyzer.analyze_cartesian(config, cache=cache)
    assert replay.metadata["cache_hit"]
    np.testing.assert_array_equal(replay.points, result.points)
    result.save(tmp_path / "plane.npz")
    np.testing.assert_array_equal(
        AnalysisResult.load(tmp_path / "plane.npz").points, result.points
    )
