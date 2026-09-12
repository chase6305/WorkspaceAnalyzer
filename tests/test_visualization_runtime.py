import socket
import threading
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("viser")
pytest.importorskip("trimesh")

from workspace_analyzer import (  # noqa: E402
    CartesianConfig,
    KinematicsSolver,
    ReachabilityConfig,
    RobotModel,
    SamplingConfig,
    SolverConfig,
    WorkspaceAnalyzer,
    WorkspaceConfig,
)
from workspace_analyzer.visualization import (
    ViserWorkspace,  # noqa: E402
    _display_indices,  # noqa: E402
    _metric_colors,  # noqa: E402
    _zero_tree_transforms,  # noqa: E402
)

URDF = Path(__file__).parent / "fixtures/two_link.urdf"


def test_display_indices_are_deterministic_and_bounded():
    np.testing.assert_array_equal(_display_indices(3, 5), [0, 1, 2])
    indices = _display_indices(100, 7)
    assert len(indices) == 7
    assert indices[0] == 0
    assert indices[-1] == 99
    assert np.all(np.diff(indices) > 0)
    colors = _metric_colors(np.array([0.0, 0.5, 1.0, np.nan]))
    assert colors.shape == (4, 3)
    np.testing.assert_array_equal(colors[-1], [128, 128, 128])


def test_zero_full_tree_matches_chain_fk():
    model = RobotModel.from_urdf(URDF)
    solver = KinematicsSolver(model, config=SolverConfig(backend="numpy"))
    neutral = _zero_tree_transforms(model)
    expected = solver.forward(np.zeros(solver.dof), all_links=True)
    for link, transform in expected.items():
        np.testing.assert_allclose(neutral[link], transform, atol=1e-12)


def test_viewer_reports_quality_failure_separately_from_ik():
    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF), config=SolverConfig(backend="numpy")
    )
    result = WorkspaceAnalyzer(solver).analyze_targets(
        [[2, 0, 0]],
        ReachabilityConfig(restarts=1, rescue_restarts=0, minimum_isotropy=0.1),
    )
    with ViserWorkspace(solver, port=_free_port(), load_robot_visuals=False) as viewer:
        viewer.add_workspace(result)
        assert "Quality accepted: **0**" in viewer._handles["workspace_info"].content
        assert (
            "IK succeeded, below quality gates: **1**"
            in viewer._handles["workspace_info"].content
        )
        assert len(viewer._handles["reachable_splats"].centers) == 1


def _free_port():
    try:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]
    except PermissionError:
        pytest.skip("the test environment does not allow local sockets")


def test_viser_cartesian_controls_and_recompute():
    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF),
        config=SolverConfig(backend="numpy", max_iterations=80),
    )
    analyzer = WorkspaceAnalyzer(solver)
    config = CartesianConfig(
        bounds=np.array([[-2.2, 2.2], [-2.2, 2.2], [-1e-6, 1e-6]]),
        sampling=SamplingConfig(num_samples=16, batch_size=8),
        restarts=2,
        rescue_restarts=2,
        rescue_rounds=1,
        reference_pose=solver.forward([0.0, 0.0]),
        reference_joints=np.zeros(2),
    )
    result = analyzer.analyze_cartesian(config)
    with ViserWorkspace(solver, port=_free_port(), load_robot_visuals=False) as viewer:
        assert not viewer._handles["workspace"].visible
        assert "reference_frames" in viewer._handles
        joint_result = WorkspaceAnalyzer(
            solver,
            WorkspaceConfig(SamplingConfig(num_samples=12, batch_size=6)),
        ).analyze()
        viewer.add_workspace(joint_result, color_metric="isotropy")
        assert "joint_limit_margin" in viewer._metric_selector.options
        before = viewer._handles["workspace_cloud"].colors.copy()
        viewer._set_workspace_color_metric("joint_limit_margin")
        assert np.any(viewer._handles["workspace_cloud"].colors != before)
        path_q = np.array([[0.1, 0.5], [0.15, 0.45], [0.2, 0.4]])
        trajectory = solver.solve_trajectory(
            solver.forward(path_q),
            seed=path_q[0],
            position_only=True,
            dt=0.1,
        )
        viewer.add_trajectory(trajectory)
        assert len(viewer._handles["trajectory_targets"].points) == 3
        assert len(viewer._handles["trajectory_path"].points) == 2
        assert "trajectory_frames" in viewer._handles
        assert "current_target" in viewer._handles
        assert "pose_error" in viewer._handles
        assert "position error" in viewer._trajectory_frame_info.content
        assert "orientation error" in viewer._trajectory_frame_info.content
        assert viewer.trajectory_root.startswith(viewer.root + "/")
        assert not viewer.trajectory_root.startswith(viewer.workspace_root + "/")
        viewer.show_trajectory_frame(2)
        np.testing.assert_allclose(viewer._q, trajectory.positions[2])
        assert "IK success" in viewer._trajectory_info.content

        viewer.add_workspace(result)
        assert len(viewer._handles["reachable_splats"].centers) == int(
            result.reachable.sum()
        )
        viewer._workspace_point_size = 0.012
        viewer._reachable_opacity = 0.4
        viewer._schedule_splat_refresh()
        time.sleep(0.2)
        handle = viewer._handles["reachable_splats"]
        assert handle.covariances[0, 0, 0] == pytest.approx(0.012**2, rel=1e-3)
        assert handle.opacities[0, 0] == pytest.approx(round(0.4 * 255), abs=1)

        viewer._q[:] = [0.1, -0.2]
        reference = viewer.capture_reference_pose()
        np.testing.assert_allclose(reference, solver.forward([0.1, -0.2]))
        display_from_base = np.eye(4)
        display_from_base[:3, 3] = [0.4, -0.2, 0.7]
        viewer._display_base_transform = display_from_base
        viewer._update_reference_display()
        expected_display = display_from_base @ reference
        np.testing.assert_allclose(
            viewer._handles["reference_tool"].position,
            expected_display[:3, 3],
        )
        assert "Solver-base XYZ" in viewer._reference_info.content
        assert "Robot-world XYZ" in viewer._reference_info.content
        viewer.configure_cartesian_recompute(analyzer, config)
        viewer.recompute_cartesian_async()
        deadline = time.monotonic() + 5.0
        while viewer._recompute_running and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not viewer._recompute_running
        assert viewer.last_cartesian_result.metadata["samples"] == 16

        shared = ViserWorkspace(
            solver,
            server=viewer.server,
            label="second_robot",
            workspace_root="/second_workspace",
            gui_label="Second robot",
            scene_offset=(1.0, 0.0, 0.0),
            load_robot_visuals=False,
        )
        shared.add_workspace(joint_result, color_metric="minimum_singular_value")
        assert not shared._owns_server
        assert shared.workspace_root == "/second_workspace"
        shared.close()


def test_viewer_close_cancels_worker_and_does_not_publish_late_results(monkeypatch):
    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF), config=SolverConfig(backend="numpy")
    )
    config = CartesianConfig(bounds=np.array([[-1, 1]] * 3))
    started = threading.Event()
    completed = threading.Event()

    class FinishingAnalyzer:
        calls = 0

        def analyze_cartesian(self, config, *, cancel_event, progress_callback):
            self.calls += 1
            started.set()
            if not cancel_event.wait(timeout=2):
                raise RuntimeError("viewer did not cancel the worker")
            progress_callback(1.0)
            completed.set()
            # Simulate completion racing with the close request.
            return object()

    analyzer = FinishingAnalyzer()
    published = []
    viewer = ViserWorkspace(solver, port=_free_port(), load_robot_visuals=False)
    try:
        monkeypatch.setattr(viewer, "add_workspace", published.append)
        viewer.configure_cartesian_recompute(analyzer, config)
        viewer.recompute_cartesian_async()
        assert started.wait(timeout=2)
        viewer.recompute_cartesian_async()
        assert analyzer.calls == 1
        viewer.close()
        assert completed.is_set()
        assert not viewer._analysis_thread.is_alive()
        assert not viewer._recompute_running
        assert published == []
        viewer.close()
        viewer.recompute_cartesian_async()
        assert analyzer.calls == 1
    finally:
        viewer.close()


def test_viewer_close_stops_playback_and_prevents_restart():
    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF), config=SolverConfig(backend="numpy")
    )
    q = np.array([[0.2, 0.7], [0.21, 0.69], [0.22, 0.68]])
    trajectory = solver.solve_trajectory(solver.forward(q), seed=q[0])
    viewer = ViserWorkspace(solver, port=_free_port(), load_robot_visuals=False)
    try:
        viewer.add_trajectory(trajectory)
        viewer.play_trajectory()
        worker = viewer._playback_thread
        viewer.close()
        assert not worker.is_alive()
        viewer.play_trajectory()
        assert viewer._playback_thread is worker
    finally:
        viewer.close()


def test_workspace_starts_visible_and_reset_keeps_initial_pose():
    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF), config=SolverConfig(backend="numpy")
    )
    result = WorkspaceAnalyzer(
        solver, WorkspaceConfig(SamplingConfig(num_samples=8))
    ).analyze()
    initial = np.array([0.2, -0.3])
    with ViserWorkspace(
        solver, port=_free_port(), initial_q=initial, load_robot_visuals=False
    ) as viewer:
        viewer.add_workspace(result)
        assert viewer._workspace_root.visible
        assert viewer._visibility_controls["workspace"].value
        viewer._q[:] = 0
        viewer.reset_pose()
        np.testing.assert_array_equal(viewer._q, initial)
        viewer.add_workspace(result, visible=False)
        viewer.add_workspace(result)
        assert not viewer._workspace_root.visible
        viewer.frame_view()
        if hasattr(viewer.server, "initial_camera"):
            assert np.isfinite(viewer.server.initial_camera.position).all()
