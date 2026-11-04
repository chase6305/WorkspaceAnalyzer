import gc
import socket
import threading
import time
import weakref
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("viser")
pytest.importorskip("trimesh")

from workspace_analyzer import (  # noqa: E402
    CartesianConfig,
    KinematicsSolver,
    PlaneConfig,
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

        joint_result = WorkspaceAnalyzer(
            solver, WorkspaceConfig(SamplingConfig(num_samples=4))
        ).analyze()
        viewer.add_workspace(joint_result)
        assert "workspace_info" not in viewer._handles
        viewer.add_workspace(result)
        assert "Quality accepted: **0**" in viewer._handles["workspace_info"].content


def _free_port():
    try:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]
    except PermissionError:
        pytest.skip("the test environment does not allow local sockets")


def _viewer_control(viewer, label):
    for control in viewer.server.gui._gui_input_handle_from_uuid.values():
        if getattr(control, "label", None) != label:
            continue
        for callback in control._impl.update_cb:
            if any(
                cell.cell_contents is viewer
                for cell in getattr(callback, "__closure__", None) or ()
            ):
                return control
    raise AssertionError(f"missing viewer control: {label}")


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


@pytest.mark.parametrize(
    "backend,device", [("numpy", "cpu"), ("torch", "cpu"), ("torch", "cuda")]
)
@pytest.mark.parametrize("angle", [0, 1e-3])
def test_float32_trajectory_orientation_display_preserves_small_angles(
    backend, device, angle
):
    if backend == "torch":
        torch = pytest.importorskip("torch")
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA device is unavailable")
    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF),
        config=SolverConfig(backend=backend, device=device, dtype="float32"),
    )
    configuration = [-0.3246581216898804, 0.38049788130940476]
    target = solver.forward(configuration)
    if hasattr(target, "detach"):
        target = target.detach().cpu().numpy()
    trajectory = solver.solve_trajectory(target[None], seed=configuration)
    rotation_offset = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    trajectory.target_poses[0, :3, :3] = rotation_offset @ target[:3, :3]
    with ViserWorkspace(
        solver, port=_free_port(), initial_q=configuration, load_robot_visuals=False
    ) as viewer:
        viewer.add_trajectory(trajectory)
        # A float32 rotation is only approximately orthogonal. Trace-only acos
        # can report hundredths of a degree even for an identical target.
        expected = f"orientation error: **{np.degrees(angle):.4f} deg**"
        assert expected in viewer._trajectory_frame_info.content


def test_playback_thread_start_failure_restores_controls_and_allows_retry(monkeypatch):
    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF), config=SolverConfig(backend="numpy")
    )
    joints = np.array([[0.2, 0.7], [0.21, 0.69], [0.22, 0.68]])
    trajectory = solver.solve_trajectory(solver.forward(joints), seed=joints[0])
    with ViserWorkspace(solver, port=_free_port(), load_robot_visuals=False) as viewer:
        viewer.add_trajectory(trajectory)
        original_start = threading.Thread.start

        def fail_playback_start(thread, *args, **kwargs):
            target = getattr(thread, "_target", None)
            if target and target.__qualname__.endswith(
                ".play_trajectory.<locals>.work"
            ):
                raise RuntimeError("playback thread unavailable")
            return original_start(thread, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(threading.Thread, "start", fail_playback_start)
            viewer.play_trajectory()
        assert viewer._playback_thread is None
        assert not viewer._play_button.disabled
        assert "playback thread unavailable" in viewer._trajectory_frame_info.content
        viewer.play_trajectory()
        assert viewer._playback_thread.ident is not None


def test_callbacks_after_close_do_not_change_display():
    from workspace_analyzer import PlaneConfig

    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF), config=SolverConfig(backend="numpy")
    )
    joints = np.array([[0.2, 0.7], [0.3, 0.6], [0.4, 0.5]])
    trajectory = solver.solve_trajectory(solver.forward(joints), seed=joints[0])
    viewer = ViserWorkspace(solver, port=_free_port(), load_robot_visuals=False)
    try:
        viewer.add_trajectory(trajectory)
        previous_joints = viewer._q.copy()
        target_handle = viewer._handles["current_target"]
        previous_target = target_handle.position.copy()
        previous_info = viewer._trajectory_frame_info.content
        viewer.close()
        viewer.show_trajectory_frame(2)
        np.testing.assert_array_equal(viewer._q, previous_joints)
        np.testing.assert_array_equal(
            viewer._handles["current_target"].position, previous_target
        )
        assert viewer._trajectory_frame_info.content == previous_info
        viewer.add_trajectory(trajectory)
        assert viewer._handles["current_target"] is target_handle
        viewer.configure_plane_recompute(PlaneConfig(plane="yz"))
        assert viewer.plane_panel.plane.value == "xy"
    finally:
        viewer.close()


@pytest.mark.parametrize(
    "label,value",
    [
        ("shoulder", 1.1),
        ("Capture current FK pose", True),
        ("Reset to initial pose", True),
        ("Frame robot and workspace", True),
        ("Workspace point size", 0.025),
        ("Trajectory target size", 0.025),
        ("Trajectory line width", 7.0),
        ("Workspace color metric", "joint_limit_margin"),
        ("Robot visuals", False),
        ("Workspace", False),
    ],
)
def test_shared_viewer_close_ignores_callbacks_waiting_for_lock(
    label, value, monkeypatch
):
    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF), config=SolverConfig(backend="numpy")
    )
    with ViserWorkspace(solver, port=_free_port(), load_robot_visuals=False) as peer:
        viewer = ViserWorkspace(
            solver,
            server=peer.server,
            label="closing",
            gui_label="Closing",
            workspace_root="/closing_workspace",
            initial_q=[0.2, 0.7],
            load_robot_visuals=False,
        )
        try:
            joints = np.array([[0.8, 0.4], [0.85, 0.35], [0.9, 0.3]])
            trajectory = solver.solve_trajectory(solver.forward(joints), seed=joints[0])
            viewer.add_trajectory(trajectory)
            workspace = WorkspaceAnalyzer(
                solver, WorkspaceConfig(SamplingConfig(num_samples=16))
            ).analyze()
            viewer.add_workspace(workspace)
            camera = peer.server.initial_camera
            camera.position = (0, 0, 3)
            control = _viewer_control(viewer, label)
            display_lock = viewer._lock
            waiting = threading.Event()
            errors = []

            class ObservedLock:
                def __enter__(self):
                    if threading.current_thread() is worker:
                        waiting.set()
                    display_lock.acquire()

                def __exit__(self, *args):
                    display_lock.release()

            def deliver_update():
                try:
                    control.value = value
                except Exception as exc:
                    errors.append(exc)

            def state():
                return {
                    "joints": viewer._q.copy(),
                    "reference_joints": viewer.reference_joints.copy(),
                    "reference_pose": viewer.reference_pose.copy(),
                    "tool": viewer._handles["tool"].position.copy(),
                    "reference_info": viewer._reference_info.content,
                    "camera": np.array(camera.position),
                    "cloud_colors": viewer._handles["workspace_cloud"].colors.copy(),
                    "workspace_size": viewer._workspace_point_size,
                    "trajectory_size": viewer._handles["trajectory_targets"].point_size,
                    "trajectory_width": viewer._handles["trajectory_path"].thickness,
                    "robot_visible": viewer._robot_root.visible,
                    "workspace_visible": viewer._workspace_root.visible,
                }

            worker = threading.Thread(target=deliver_update)
            monkeypatch.setattr(viewer, "_lock", ObservedLock())
            with viewer._lock:
                worker.start()
                assert waiting.wait(3)
                viewer.close()
                frozen = state()
            worker.join(timeout=3)
            assert not worker.is_alive()
            assert not errors
            observed = state()
            for name, expected in frozen.items():
                np.testing.assert_array_equal(observed[name], expected, err_msg=name)
            captured = viewer.capture_reference_pose()
            np.testing.assert_array_equal(captured, frozen["reference_pose"])
            captured[:] = 0
            np.testing.assert_array_equal(
                viewer.reference_pose, frozen["reference_pose"]
            )
            assert not peer._closed
            peer._sliders[0].value = 0.95
            assert peer._q[0] == 0.95
            peer.frame_view()
            assert not np.array_equal(camera.position, frozen["camera"])
        finally:
            viewer.close()


def test_replacing_trajectory_stops_old_playback(monkeypatch):
    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF), config=SolverConfig(backend="numpy")
    )
    joints = np.array([[0.2, 0.7], [0.3, 0.6], [0.4, 0.5]])
    original = solver.solve_trajectory(solver.forward(joints), seed=joints[0])
    replacement = solver.solve_trajectory(solver.forward(joints[::-1]), seed=joints[-1])
    with ViserWorkspace(solver, port=_free_port(), load_robot_visuals=False) as viewer:
        viewer.add_trajectory(original)
        viewer._fps_slider.value = 1
        first_frame = threading.Event()
        show_frame = viewer.show_trajectory_frame

        def show(index, **kwargs):
            show_frame(index, **kwargs)
            if threading.current_thread() is viewer._playback_thread:
                first_frame.set()

        monkeypatch.setattr(viewer, "show_trajectory_frame", show)
        viewer.play_trajectory()
        worker = viewer._playback_thread
        assert first_frame.wait(5)
        viewer.add_trajectory(replacement)
        worker.join(timeout=1)
        assert not worker.is_alive()
        assert viewer._trajectory is replacement
        np.testing.assert_allclose(viewer._q, replacement.positions[0])
        assert not viewer._play_button.disabled


def test_replacing_trajectory_cancels_playback_waiting_for_display_lock(monkeypatch):
    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF), config=SolverConfig(backend="numpy")
    )
    joints = np.array([[0.2, 0.7], [0.3, 0.6], [0.4, 0.5]])
    original = solver.solve_trajectory(solver.forward(joints), seed=joints[0])
    replacement = solver.solve_trajectory(solver.forward(joints[::-1]), seed=joints[-1])
    with ViserWorkspace(solver, port=_free_port(), load_robot_visuals=False) as viewer:
        viewer.add_trajectory(original)
        viewer.show_trajectory_frame(2)
        viewer._frame_slider.value = 2
        display_lock = viewer._lock
        waiting = threading.Event()

        class ObservedLock:
            def __enter__(self):
                if threading.current_thread() is viewer._playback_thread:
                    waiting.set()
                display_lock.acquire()

            def __exit__(self, *args):
                display_lock.release()

        monkeypatch.setattr(viewer, "_lock", ObservedLock())
        # Hold the display lock until the old worker has passed its outer stop
        # check. Replacing the trajectory must invalidate that pending frame.
        with viewer._lock:
            viewer.play_trajectory()
            worker = viewer._playback_thread
            assert waiting.wait(3)
            viewer.add_trajectory(replacement)
        worker.join(timeout=3)
        assert not worker.is_alive()
        assert viewer._frame_slider.value == 0
        np.testing.assert_allclose(viewer._q, replacement.positions[0])
        assert not viewer._play_button.disabled


@pytest.mark.parametrize(
    "backend,device", [("numpy", "cpu"), ("torch", "cpu"), ("torch", "cuda")]
)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_robot_geometry_shares_the_same_display_transform(backend, device, dtype):
    from trimesh.transformations import quaternion_matrix

    from workspace_analyzer.model import _rpy_matrix

    if backend == "torch":
        torch = pytest.importorskip("torch")
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA is unavailable")
    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF.parent / "mixed_chain.urdf"),
        config=SolverConfig(backend=backend, device=device, dtype=dtype),
    )
    configuration = np.array([0.4, 0.2, -0.3])
    with ViserWorkspace(solver, port=_free_port(), load_robot_visuals=False) as viewer:
        display_from_base = _rpy_matrix([0.3, -0.2, 0.1]).astype(dtype)
        display_from_base[:3, 3] = [0.4, -0.3, 0.8]
        viewer._display_base_transform = display_from_base
        viewer.update_robot(configuration)
        expected_poses = solver.forward(configuration, all_links=True)
        viewer._update_robot_transforms(dict(reversed(tuple(expected_poses.items()))))
        tolerance = 2e-7 if dtype == "float32" else 1e-12
        expected_points = []
        for name, pose in expected_poses.items():
            if hasattr(pose, "detach"):
                pose = pose.detach().cpu().numpy()
            expected = display_from_base @ pose
            expected_points.append(expected[:3, 3])
            np.testing.assert_allclose(
                viewer._link_frames[name].position, expected[:3, 3], atol=1e-7
            )
            quaternion = viewer._link_frames[name].wxyz
            np.testing.assert_allclose(
                np.linalg.norm(quaternion), 1.0, atol=tolerance, rtol=0
            )
            np.testing.assert_allclose(
                quaternion_matrix(quaternion)[:3, :3],
                expected[:3, :3],
                atol=tolerance,
                rtol=0,
            )
        expected_points = np.asarray(expected_points)
        np.testing.assert_allclose(
            viewer._handles["joints"].points, expected_points, atol=1e-7
        )
        np.testing.assert_allclose(
            viewer._handles["skeleton"].points,
            np.stack((expected_points[:-1], expected_points[1:]), axis=1),
            atol=1e-7,
        )
        np.testing.assert_allclose(
            viewer._handles["tool"].position, expected_points[-1], atol=1e-7
        )
        np.testing.assert_array_equal(
            viewer._handles["tool"].wxyz,
            viewer._link_frames[solver.tip_link].wxyz,
        )


@pytest.mark.parametrize(
    "key", ["trajectory_targets", "trajectory_frames", "trajectory_path"]
)
def test_replacing_trajectory_preserves_visibility_choice(key):
    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF), config=SolverConfig(backend="numpy")
    )
    q = np.array([[0.2, 0.7], [0.21, 0.69], [0.22, 0.68]])
    trajectory = solver.solve_trajectory(solver.forward(q), seed=q[0])
    with ViserWorkspace(solver, port=_free_port(), load_robot_visuals=False) as viewer:
        viewer.add_trajectory(trajectory)
        control = viewer._visibility_controls[key]
        control.value = False
        deadline = time.monotonic() + 2.0
        while viewer._handles[key].visible and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not viewer._handles[key].visible
        viewer.add_trajectory(trajectory)
        assert not control.value
        assert not viewer._handles[key].visible


def test_failed_trajectory_frame_hides_error_until_successful_frame():
    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF), config=SolverConfig(backend="numpy")
    )
    q = np.array([[0.2, 0.7], [0.3, 0.6], [0.4, 0.5]])
    targets = solver.forward(q)
    targets[1, :3, 3] = [10, 10, 10]
    trajectory = solver.solve_trajectory(
        targets, seed=q[0], position_only=True, failure_restarts=1
    )
    np.testing.assert_array_equal(trajectory.success, [True, False, True])
    with ViserWorkspace(solver, port=_free_port(), load_robot_visuals=False) as viewer:
        viewer._display_base_transform[:3, 3] = [0.4, -0.2, 0.7]
        viewer.add_trajectory(trajectory)
        error_line = viewer._handles["pose_error"]
        previous = error_line.points.copy()
        assert error_line.visible

        viewer.show_trajectory_frame(1)
        assert not error_line.visible
        display_target = viewer._display_base_transform @ targets[1]
        np.testing.assert_allclose(
            viewer._handles["current_target"].position, display_target[:3, 3]
        )
        assert "IK: **failed**" in viewer._trajectory_frame_info.content

        viewer.show_trajectory_frame(2)
        assert error_line.visible
        display_actual = viewer._display_base_transform @ solver.forward(
            trajectory.positions[2]
        )
        display_target = viewer._display_base_transform @ targets[2]
        np.testing.assert_allclose(
            error_line.points,
            [[display_actual[:3, 3], display_target[:3, 3]]],
            atol=1e-7,
        )
        assert not np.array_equal(error_line.points, previous)
        assert "IK: **success**" in viewer._trajectory_frame_info.content


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


def test_plane_controls_compute_export_and_change_orientation(tmp_path):
    from workspace_analyzer import AnalysisResult, PlaneConfig

    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF.parent / "cartesian_stage.urdf"),
        config=SolverConfig(backend="numpy", max_iterations=15),
    )
    with ViserWorkspace(solver, port=_free_port(), load_robot_visuals=False) as viewer:
        viewer.configure_plane_recompute(
            PlaneConfig(plane="xz", center=(0, 0.4, 0), span=3, resolution=5),
            ReachabilityConfig(batch_size=7, restarts=1, rescue_restarts=0),
            output=tmp_path / "plane.npz",
            report=tmp_path / "plane.json",
        )
        panel = viewer.plane_panel
        assert panel.result is None
        assert "25 targets" in panel.info.content
        assert "plane_preview" in viewer._handles
        assert not panel.compute_button.disabled
        panel.compute()
        viewer._analysis_thread.join(timeout=10)
        assert not viewer._recompute_running
        assert "Complete: 9/25" in panel.info.content
        result = AnalysisResult.load(tmp_path / "plane.npz")
        np.testing.assert_array_equal(
            result.reachable, np.all(np.abs(result.points) <= 1, axis=1)
        )
        assert not panel.download_npz.disabled
        assert not panel.download_json.disabled
        assert (tmp_path / "plane.json").is_file()
        panel.full_pose.value = True
        panel.rpy.value = (0.3, 0, 0)
        panel.compute()
        viewer._analysis_thread.join(timeout=10)
        assert "Complete: 0/25" in panel.info.content
        assert panel.result.metadata["plane_scan"]["task"] == "fixed_orientation"
        assert panel.result.target_poses is not None


def test_plane_invalid_settings_do_not_start_worker():
    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF), config=SolverConfig(backend="numpy")
    )
    with ViserWorkspace(solver, port=_free_port(), load_robot_visuals=False) as viewer:
        panel = viewer.plane_panel
        panel.span.value = -1
        panel.compute()
        assert "Invalid plane settings" in panel.info.content
        assert not viewer._recompute_running
        assert viewer._analysis_thread is None
        assert not panel.compute_button.disabled


def test_plane_cancel_keeps_previous_result_and_unlocks_controls(monkeypatch):
    import workspace_analyzer.plane_visualization as plane_view
    from workspace_analyzer import PlaneConfig

    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF.parent / "cartesian_stage.urdf"),
        config=SolverConfig(backend="numpy"),
    )
    with ViserWorkspace(solver, port=_free_port(), load_robot_visuals=False) as viewer:
        viewer.configure_plane_recompute(
            PlaneConfig(resolution=3), ReachabilityConfig(restarts=1, rescue_restarts=0)
        )
        panel = viewer.plane_panel
        panel.compute()
        viewer._analysis_thread.join(timeout=10)
        original = panel.result
        assert original is not None
        started = threading.Event()

        def delayed(*args, cancel_event, **kwargs):
            started.set()
            assert cancel_event.wait(timeout=5)
            return original  # A late completion must not be published.

        monkeypatch.setattr(plane_view, "analyze_plane", delayed)
        panel.compute()
        assert started.wait(timeout=5)
        assert panel.compute_button.disabled
        panel.cancel()
        viewer._analysis_thread.join(timeout=5)
        assert "Cancelled" in panel.info.content
        assert panel.result is original
        assert not viewer._recompute_running
        assert not panel.compute_button.disabled
        assert not panel.center.disabled


def test_plane_random_sampling_plot_and_download(tmp_path):
    from types import SimpleNamespace

    from workspace_analyzer import PlaneConfig
    from workspace_analyzer.plane_plot import plane_svg

    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF.parent / "cartesian_stage.urdf"),
        config=SolverConfig(backend="numpy"),
    )
    with ViserWorkspace(solver, port=_free_port(), load_robot_visuals=False) as viewer:
        viewer.configure_plane_recompute(
            PlaneConfig(span=0.4, resolution=3),
            ReachabilityConfig(restarts=1, rescue_restarts=0),
            plot=tmp_path / "plane.svg",
        )
        panel = viewer.plane_panel
        panel.sampling_method.value = "random"
        panel.sample_count.value = 17
        panel.compute()
        viewer._analysis_thread.join(timeout=10)
        assert "Complete: 17/17" in panel.info.content
        assert panel.result.metadata["plane_scan"]["sampling_strategy"] == "random"
        assert panel.resolution.disabled
        assert not panel.sample_count.disabled
        assert not panel.download_svg.disabled
        expected = plane_svg(panel.result)
        assert (tmp_path / "plane.svg").read_text() == expected
        if panel.plot_view is not None:
            assert panel.plot_view.content == expected
        downloads = []
        client = SimpleNamespace(
            send_file_download=lambda *args: downloads.append(args)
        )
        panel.download(SimpleNamespace(client=client), "svg")
        assert downloads == [("plane.svg", expected.encode())]
        panel.sampling_method.value = "grid"
        panel.compute()
        viewer._analysis_thread.join(timeout=10)
        assert "Complete: 9/9" in panel.info.content
        assert not panel.resolution.disabled
        assert panel.sample_count.disabled


def test_plane_displays_independent_sampling_and_ik_seeds():
    from workspace_analyzer import PlaneConfig

    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF.parent / "cartesian_stage.urdf"),
        config=SolverConfig(backend="numpy"),
    )
    original = PlaneConfig(span=0.4, sampling=SamplingConfig(num_samples=11, seed=17))
    with ViserWorkspace(solver, port=_free_port(), load_robot_visuals=False) as viewer:
        viewer.configure_plane_recompute(
            original, ReachabilityConfig(random_seed=23, restarts=1, rescue_restarts=0)
        )
        panel = viewer.plane_panel
        assert panel.sampling_seed.value == 17
        assert panel.seed.value == 23
        panel.compute()
        viewer._analysis_thread.join(timeout=10)
        np.testing.assert_array_equal(panel.result.points, original.points())
        panel.seed.value = 31
        panel.compute()
        viewer._analysis_thread.join(timeout=10)
        np.testing.assert_array_equal(panel.result.points, original.points())
        assert panel.result.metadata["plane_scan"]["seed"] == 31
        panel.sampling_seed.value = 19
        panel.compute()
        viewer._analysis_thread.join(timeout=10)
        assert not np.array_equal(panel.result.points, original.points())
        assert panel.result.metadata["plane_scan"]["sampling_seed"] == 19
        assert panel.result.metadata["plane_scan"]["seed"] == 31


@pytest.mark.parametrize("frame", ["base", "world", "mount", "tcp"])
@pytest.mark.parametrize("rotated", [False, True])
def test_center_plane_uses_current_tcp_and_sampling_frame(frame, rotated, tmp_path):
    from workspace_analyzer import PlaneConfig

    urdf = tmp_path / "mounted-stage.urdf"
    urdf.write_text(
        (URDF.parent / "cartesian_stage.urdf")
        .read_text()
        .replace(
            '<robot name="cartesian_stage">',
            """<robot name="mounted">
        <link name="world"/><link name="mount"/>
        <joint name="mounting" type="fixed"><parent link="world"/><child link="mount"/>
          <origin xyz="0.5 -0.2 0.7" rpy="0 0 1.5707963267948966"/></joint>
        <joint name="base_fixed" type="fixed"><parent link="mount"/><child link="base"/>
          <origin xyz="0.2 0 0"/></joint>""",
        )
    )
    solver = KinematicsSolver(
        RobotModel.from_urdf(urdf),
        base_link="base",
        tip_link="tcp",
        config=SolverConfig(backend="numpy"),
    )
    with ViserWorkspace(solver, port=_free_port(), load_robot_visuals=False) as viewer:
        viewer.configure_plane_recompute(
            PlaneConfig(resolution=3, span=0.2),
            ReachabilityConfig(restarts=1, rescue_restarts=0),
        )
        panel = viewer.plane_panel
        assert set(solver.model.links) <= set(panel.frame.options)
        panel.frame.value = frame
        if rotated:
            panel.frame_rpy.value = (0, 0, np.pi / 2)
        viewer._q[:] = [0.1, 0.2, 0.3]
        original_reference = viewer.reference_joints.copy()
        panel.center_at_current_tcp()
        expected = {
            "base": [0.1, 0.2, 0.3],
            "world": [0.3, 0.1, 1.0],
            "mount": [0.3, 0.2, 0.3],
            "tcp": [0.1, 0.2, 0.3],
        }[frame]
        if rotated:
            expected = [expected[1], -expected[0], expected[2]]
        np.testing.assert_allclose(panel.center.value, expected, atol=1e-12)
        np.testing.assert_array_equal(viewer.reference_joints, original_reference)
        assert panel.show_preview.value
        panel.compute()
        viewer._analysis_thread.join(timeout=10)
        assert panel.result.reachable.all()
        np.testing.assert_allclose(panel.result.points[4], [0.1, 0.2, 0.3], atol=1e-12)


def test_plane_export_validation_preserves_previous_configuration(tmp_path):
    from workspace_analyzer import PlaneConfig

    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF), config=SolverConfig(backend="numpy")
    )
    with ViserWorkspace(solver, port=_free_port(), load_robot_visuals=False) as viewer:
        panel = viewer.plane_panel
        before = panel.configuration()
        with pytest.raises(ValueError, match="distinct paths"):
            panel.configure(
                PlaneConfig(plane="yz"),
                output=tmp_path / "same",
                report=tmp_path / "same.npz",
            )
        assert panel.configuration() == before
        assert not list(tmp_path.iterdir())


def test_plane_save_failure_keeps_completed_result_and_retry_avoids_ik(
    tmp_path, monkeypatch
):
    import workspace_analyzer.plane_visualization as plane_view
    from workspace_analyzer import AnalysisResult, PlaneConfig

    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF.parent / "cartesian_stage.urdf"),
        config=SolverConfig(backend="numpy"),
    )
    with ViserWorkspace(solver, port=_free_port(), load_robot_visuals=False) as viewer:
        report_path = tmp_path / "plane.json"
        viewer.configure_plane_recompute(
            PlaneConfig(resolution=3),
            ReachabilityConfig(restarts=1, rescue_restarts=0),
            output=tmp_path / "plane.npz",
            report=report_path,
            plot=tmp_path / "plane.svg",
        )
        # The destination becomes unusable after settings were validated.
        report_path.mkdir()
        panel = viewer.plane_panel
        panel.compute()
        viewer._analysis_thread.join(timeout=10)
        assert panel.result is not None
        assert panel.result.reachable.all()
        assert "Complete: 9/9" in panel.info.content
        assert "Save failed" in panel.export_info.content
        assert "plane.json" in panel.export_info.content
        assert not panel.download_npz.disabled
        assert not panel.retry_save.disabled
        saved = AnalysisResult.load(tmp_path / "plane.npz")
        np.testing.assert_array_equal(saved.points, panel.result.points)
        assert (tmp_path / "plane.svg").is_file()
        monkeypatch.setattr(
            plane_view, "analyze_plane", lambda *a, **k: pytest.fail("retry ran IK")
        )
        report_path.rmdir()
        monkeypatch.setattr(
            AnalysisResult,
            "save",
            lambda *a, **k: pytest.fail("retry rewrote successful NPZ"),
        )
        # Editing the next task must not change the result being saved.
        panel.center.value = (0.2, 0, 0)
        panel.retry_exports()
        assert panel.retry_save.disabled
        assert "Save failed" not in panel.export_info.content
        import json

        assert json.loads(report_path.read_text())["plane_center"] == [0, 0, 0]


@pytest.mark.parametrize("job", ["cartesian", "plane"])
def test_switching_display_cancels_old_job_and_rejects_late_result(job, monkeypatch):
    import workspace_analyzer.plane_visualization as plane_view
    from workspace_analyzer import PlaneConfig

    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF.parent / "cartesian_stage.urdf"),
        config=SolverConfig(backend="numpy"),
    )
    analyzer = WorkspaceAnalyzer(solver, WorkspaceConfig(SamplingConfig(num_samples=7)))
    replacement = analyzer.analyze()
    started, release = threading.Event(), threading.Event()

    def delayed(*args, cancel_event, progress_callback, **kwargs):
        started.set()
        assert release.wait(5)
        progress_callback(1.0)
        # Simulate a solver that finishes just as cancellation is requested.
        return replacement

    with ViserWorkspace(solver, port=_free_port(), load_robot_visuals=False) as viewer:
        if job == "cartesian":
            monkeypatch.setattr(analyzer, "analyze_cartesian", delayed)
            viewer.configure_cartesian_recompute(
                analyzer, CartesianConfig(bounds=np.array([[-1, 1]] * 3))
            )
            viewer.recompute_cartesian_async()
        else:
            monkeypatch.setattr(plane_view, "analyze_plane", delayed)
            viewer.configure_plane_recompute(PlaneConfig(resolution=3))
            viewer.plane_panel.compute()
        assert started.wait(5)
        try:
            viewer.add_workspace(replacement)
            assert viewer._analysis_cancel.is_set()
            viewer.configure_cartesian_recompute(None, None)
            release.set()
            viewer._analysis_thread.join(timeout=5)
            assert viewer._workspace_result is replacement
            assert viewer.last_cartesian_result is None
            assert viewer.plane_panel.result is None
            assert viewer._recompute_button.disabled
            assert not viewer._recompute_running
        finally:
            release.set()


@pytest.mark.parametrize("max_points", [1, 3, 25])
def test_plane_uses_discrete_quality_colors_and_restores_volume_controls(max_points):
    from workspace_analyzer import PlaneConfig, analyze_plane, create_solver

    solver = create_solver(URDF.parent / "cartesian_stage.urdf", backend="numpy")
    assessment = ReachabilityConfig(
        restarts=1, rescue_restarts=0, minimum_joint_limit_margin=0.3
    )
    plane = analyze_plane(solver, PlaneConfig(resolution=5, span=3), assessment)
    volume = WorkspaceAnalyzer(solver).analyze_targets([[0, 0, 0]], assessment)
    with ViserWorkspace(solver, port=_free_port(), load_robot_visuals=False) as viewer:
        viewer.add_workspace(volume)
        assert all(not item.disabled for item in viewer._opacity_controls)
        viewer.add_workspace(plane, max_points=max_points)
        cloud = viewer._handles["workspace_cloud"]
        assert len(cloud.points) == min(max_points, 25)
        np.testing.assert_allclose(
            cloud.points, plane.points[viewer._workspace_indices]
        )
        assert "reachable_splats" not in viewer._handles
        assert all(item.disabled for item in viewer._opacity_controls)
        if max_points >= 3:
            assert set(map(tuple, cloud.colors)) == {
                (22, 132, 106),
                (217, 155, 24),
                (206, 70, 83),
            }
        assert "Amber: below quality gates" in viewer._handles["workspace_info"].content
        viewer.add_workspace(volume)
        assert "workspace_cloud" not in viewer._handles
        assert len(viewer._handles["reachable_splats"].centers) == 1
        assert all(not item.disabled for item in viewer._opacity_controls)


@pytest.mark.parametrize("kind", ["cartesian", "plane"])
@pytest.mark.parametrize("changed", [False, True])
def test_reference_capture_invalidates_only_changed_inflight_jobs(
    kind, changed, monkeypatch, tmp_path
):
    import workspace_analyzer.plane_visualization as plane_view
    from workspace_analyzer import PlaneConfig

    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF.parent / "cartesian_stage.urdf"),
        config=SolverConfig(backend="numpy"),
    )
    analyzer = WorkspaceAnalyzer(solver)
    config = CartesianConfig(
        bounds=np.array([[-0.2, 0.2]] * 3),
        sampling=SamplingConfig(num_samples=5),
        restarts=1,
        rescue_restarts=0,
    )
    previous = analyzer.analyze_cartesian(config)
    started, release = threading.Event(), threading.Event()
    with ViserWorkspace(solver, port=_free_port(), load_robot_visuals=False) as viewer:
        viewer.add_workspace(previous)
        if kind == "cartesian":
            compute = analyzer.analyze_cartesian

            def delayed(*args, **kwargs):
                result = compute(*args, **kwargs)
                started.set()
                assert release.wait(timeout=5)
                return result

            monkeypatch.setattr(analyzer, "analyze_cartesian", delayed)
            viewer.configure_cartesian_recompute(analyzer, config)
            begin = viewer.recompute_cartesian_async
        else:
            viewer.configure_plane_recompute(
                PlaneConfig(resolution=3, span=0.4),
                ReachabilityConfig(restarts=1, rescue_restarts=0),
                output=tmp_path / "plane.npz",
            )
            viewer.plane_panel.compute()
            viewer._analysis_thread.join(timeout=5)
            previous = viewer.plane_panel.result
            previous_bytes = (tmp_path / "plane.npz").read_bytes()
            compute = plane_view.analyze_plane

            def delayed(*args, **kwargs):
                result = compute(*args, **kwargs)
                started.set()
                assert release.wait(timeout=5)
                return result

            monkeypatch.setattr(plane_view, "analyze_plane", delayed)
            begin = viewer.plane_panel.compute
        try:
            begin()
            assert started.wait(timeout=5)
            cancellation = viewer._analysis_cancel
            if changed:
                with viewer._lock:
                    viewer._q[:] = [0.1, -0.1, 0.2]
            reference = viewer.capture_reference_pose()
            assert cancellation.is_set() == changed
            np.testing.assert_allclose(reference, solver.forward(viewer._q))
        finally:
            release.set()
            viewer._analysis_thread.join(timeout=5)
        assert not viewer._recompute_running
        if changed:
            assert viewer._workspace_result is previous
            if kind == "cartesian":
                assert viewer.last_cartesian_result is None
                assert "cancelled" in viewer._reference_info.content
            else:
                assert viewer.plane_panel.result is previous
                assert (tmp_path / "plane.npz").read_bytes() == previous_bytes
                assert "Cancelled" in viewer.plane_panel.info.content
        else:
            assert viewer._workspace_result is not previous
        begin()
        viewer._analysis_thread.join(timeout=5)
        assert not viewer._recompute_running
        result = (
            viewer.last_cartesian_result
            if kind == "cartesian"
            else viewer.plane_panel.result
        )
        metadata = (
            result.metadata if kind == "cartesian" else result.metadata["plane_scan"]
        )
        np.testing.assert_array_equal(
            metadata["reference_joints"], viewer.reference_joints
        )


@pytest.mark.parametrize("kind", ["cartesian", "plane"])
def test_analysis_thread_start_failure_restores_controls_and_allows_retry(
    kind, monkeypatch, tmp_path
):
    from workspace_analyzer import PlaneConfig

    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF.parent / "cartesian_stage.urdf"),
        config=SolverConfig(backend="numpy"),
    )
    analyzer = WorkspaceAnalyzer(solver)
    config = CartesianConfig(
        bounds=[[-0.2, 0.2]] * 3,
        sampling=SamplingConfig(num_samples=5),
        restarts=1,
        rescue_restarts=0,
    )
    with ViserWorkspace(solver, port=_free_port(), load_robot_visuals=False) as viewer:
        viewer.configure_cartesian_recompute(analyzer, config)
        viewer.configure_plane_recompute(
            PlaneConfig(resolution=3, span=0.4),
            ReachabilityConfig(restarts=1, rescue_restarts=0),
            output=tmp_path / "plane.npz",
        )
        begin = (
            viewer.recompute_cartesian_async
            if kind == "cartesian"
            else viewer.plane_panel.compute
        )
        begin()
        viewer._analysis_thread.join(timeout=5)
        previous = viewer._workspace_result
        original_start = threading.Thread.start

        def fail_analysis_start(thread, *args, **kwargs):
            target = getattr(thread, "_target", None)
            if target and target.__qualname__.endswith(
                (".compute.<locals>.work", ".recompute_cartesian_async.<locals>.work")
            ):
                raise RuntimeError("cannot start new thread")
            return original_start(thread, *args, **kwargs)

        try:
            with monkeypatch.context() as patch:
                patch.setattr(threading.Thread, "start", fail_analysis_start)
                begin()
            assert not viewer._recompute_running
            assert viewer._analysis_cancel is None
            assert viewer._workspace_result is previous
            assert not viewer._recompute_button.disabled
            assert viewer._cancel_button.disabled
            panel = viewer.plane_panel
            assert not panel.compute_button.disabled
            assert not panel.center_on_tcp.disabled
            assert panel.cancel_button.disabled
            status = (
                viewer._reference_info.content
                if kind == "cartesian"
                else panel.info.content
            )
            assert "cannot start new thread" in status
            begin()
            viewer._analysis_thread.join(timeout=5)
            assert not viewer._recompute_running
            assert viewer._workspace_result is not previous
        finally:
            if (
                viewer._analysis_thread is not None
                and viewer._analysis_thread.ident is None
            ):
                viewer._analysis_thread = None


def test_invalid_cartesian_configuration_keeps_previous_job():
    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF), config=SolverConfig(backend="numpy")
    )
    analyzer = WorkspaceAnalyzer(solver)
    config = CartesianConfig(bounds=[[-1, 1]] * 3)
    with ViserWorkspace(solver, port=_free_port(), load_robot_visuals=False) as viewer:
        viewer.configure_cartesian_recompute(analyzer, config)
        previous = viewer._cartesian_job
        with pytest.raises(ValueError, match="CartesianConfig"):
            viewer.configure_cartesian_recompute(analyzer, {"bounds": [[-1, 1]] * 3})
        assert viewer._cartesian_job is previous
        assert not viewer._recompute_button.disabled


def test_invalid_captured_reference_does_not_lock_recompute():
    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF), config=SolverConfig(backend="numpy")
    )
    analyzer = WorkspaceAnalyzer(solver)
    config = CartesianConfig(
        bounds=[[1, 1], [0, 0], [0, 0]],
        sampling=SamplingConfig(num_samples=1),
        restarts=2,
        rescue_restarts=0,
    )
    with ViserWorkspace(solver, port=_free_port(), load_robot_visuals=False) as viewer:
        viewer.configure_cartesian_recompute(analyzer, config)
        reference = viewer.reference_pose.copy()
        viewer.reference_pose[0, 3] = np.nan
        viewer.recompute_cartesian_async()
        assert not viewer._recompute_running
        assert viewer._analysis_cancel is None
        assert not viewer._recompute_button.disabled
        assert viewer._cancel_button.disabled
        assert "failed" in viewer._reference_info.content.lower()
        viewer.reference_pose = reference
        viewer.recompute_cartesian_async()
        viewer._analysis_thread.join(timeout=5)
        assert viewer.last_cartesian_result is not None


@pytest.mark.parametrize(
    "options",
    [
        {"port": -1},
        {"port": 65536},
        {"port": True},
        {"port": 8080.5},
        {"port": "8080"},
        {"workspace_root": "relative/path"},
        {"scene_offset": [0, np.nan, 0]},
        {"initial_q": [0]},
        {"initial_q": [np.nan, 0]},
        {"reference_active_link_index": 0},
        {"reference_active_link_index": 1.5},
        {"reference_active_link_index": True},
    ],
)
def test_invalid_viewer_configuration_does_not_start_server(options, monkeypatch):
    import viser

    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF), config=SolverConfig(backend="numpy")
    )

    def unexpected_server(*args, **kwargs):
        pytest.fail("invalid configuration must fail before starting a server")

    monkeypatch.setattr(viser, "ViserServer", unexpected_server)
    with pytest.raises(ValueError):
        ViserWorkspace(solver, load_robot_visuals=False, **options)


@pytest.mark.parametrize(
    "options",
    [
        {"color_metric": "missing"},
        {"max_points": 0},
        {"max_points": 2.5},
        {"max_points": True},
        {"max_points": np.nan},
        {"point_size": -0.1},
        {"point_size": 0},
        {"point_size": np.nan},
    ],
)
def test_invalid_workspace_display_keeps_previous_result_and_job(options):
    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF), config=SolverConfig(backend="numpy")
    )
    result = WorkspaceAnalyzer(
        solver, WorkspaceConfig(SamplingConfig(num_samples=12))
    ).analyze()
    with ViserWorkspace(solver, port=_free_port(), load_robot_visuals=False) as viewer:
        viewer.add_workspace(result)
        cloud = viewer._handles["workspace_cloud"]
        previous_points = cloud.points.copy()
        indices = viewer._workspace_indices
        cancel = viewer._analysis_cancel = threading.Event()
        with pytest.raises(ValueError):
            viewer.add_workspace(result, **options)
        assert not cancel.is_set()
        assert viewer._workspace_result is result
        assert viewer._workspace_indices is indices
        assert viewer._handles["workspace_cloud"] is cloud
        np.testing.assert_array_equal(cloud.points, previous_points)


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("stage", ["controls", "visuals", "plane"])
def test_failed_viewer_start_stops_only_its_owned_server(shared, stage, monkeypatch):
    import viser

    from workspace_analyzer.plane_visualization import PlaneReachabilityPanel

    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF), config=SolverConfig(backend="numpy")
    )
    server_class = viser.ViserServer
    original_stop = server_class.stop
    created, stopped, baseline_resources = [], [], []

    def create_server(*args, **kwargs):
        server = server_class(*args, **kwargs)
        created.append(server)
        baseline_resources.append(_server_resource_ids(server))
        return server

    def stop(server):
        stopped.append(server)
        original_stop(server)

    monkeypatch.setattr(server_class, "stop", stop)
    monkeypatch.setattr(viser, "ViserServer", create_server)
    owner, method = {
        "controls": (ViserWorkspace, "_add_controls"),
        "visuals": (ViserWorkspace, "load_urdf_visuals"),
        "plane": (PlaneReachabilityPanel, "__init__"),
    }[stage]
    initialize = getattr(owner, method)

    def interrupted(*args, **kwargs):
        initialize(*args, **kwargs)
        raise RuntimeError("GUI initialization failed")

    monkeypatch.setattr(owner, method, interrupted)
    server = create_server(port=_free_port()) if shared else None
    try:
        with pytest.raises(RuntimeError, match="GUI initialization failed"):
            ViserWorkspace(
                solver,
                server=server,
                port=_free_port(),
                load_robot_visuals=stage == "visuals",
            )
        assert len(created) == 1
        assert stopped == ([] if shared else created)
        assert _server_resource_ids(created[0]) == baseline_resources[0]
        if shared:
            with socket.create_connection(("127.0.0.1", server.get_port()), timeout=2):
                pass
    finally:
        for server in created:
            if server not in stopped:
                original_stop(server)


def _server_resource_ids(server):
    gui = server.gui
    return (
        frozenset(gui._gui_input_handle_from_uuid),
        frozenset(gui._container_handle_from_uuid),
        frozenset(
            (parent, child)
            for parent, container in gui._container_handle_from_uuid.items()
            for child in container._children
        ),
        frozenset(server.scene._handle_from_node_name),
    )


@pytest.mark.parametrize("reuse_paths", [False, True])
def test_shared_viewer_close_releases_resources_and_callback_references(reuse_paths):
    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF), config=SolverConfig(backend="numpy")
    )
    result = WorkspaceAnalyzer(solver).analyze_targets(
        [[2, 0, 0]], ReachabilityConfig(restarts=1, rescue_restarts=0)
    )
    with ViserWorkspace(
        solver,
        port=_free_port(),
        label="peer",
        workspace_root="/peer_workspace",
        load_robot_visuals=False,
    ) as peer:
        baseline = _server_resource_ids(peer.server)
        references = []
        for index in range(3):
            label = "repeat" if reuse_paths else f"robot_{index}"
            viewer = ViserWorkspace(
                solver,
                server=peer.server,
                label=label,
                workspace_root=f"/{label}_workspace",
                load_robot_visuals=False,
            )
            viewer.add_workspace(result)
            viewer.add_dashboard("Analysis summary")
            references.append(weakref.ref(viewer))
            if index == 1:
                viewer._gui_folders[0].remove()
            viewer.close()
            viewer.close()
            assert _server_resource_ids(peer.server) == baseline
            del viewer
            gc.collect()
            assert all(reference() is None for reference in references)
        peer._sliders[0].value = 0.35
        assert peer._q[0] == 0.35
        assert not peer._scene_root._impl.removed


def test_closing_superseded_viewer_preserves_replacement_scene():
    import viser

    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF), config=SolverConfig(backend="numpy")
    )
    server = viser.ViserServer(port=_free_port())
    baseline = _server_resource_ids(server)
    old = replacement = None
    try:
        old = ViserWorkspace(solver, server=server, load_robot_visuals=False)
        replacement = ViserWorkspace(solver, server=server, load_robot_visuals=False)
        assert old._scene_root._impl.removed
        assert old._workspace_root._impl.removed
        old.close()
        assert not replacement._scene_root._impl.removed
        assert not replacement._workspace_root._impl.removed
        assert not replacement._frame_slider._impl.removed
        assert server.scene._handle_from_node_name[replacement.root] is (
            replacement._scene_root
        )
        replacement._sliders[0].value = 0.35
        assert replacement._q[0] == 0.35
        replacement.update_robot(replacement._q)
        replacement.close()
        assert _server_resource_ids(server) == baseline
    finally:
        if old is not None:
            old.close()
        if replacement is not None:
            replacement.close()
        server.stop()


@pytest.mark.parametrize("stage", ["controls", "plane"])
def test_partial_folder_failure_preserves_shared_peer(stage, monkeypatch):
    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF), config=SolverConfig(backend="numpy")
    )
    with ViserWorkspace(
        solver,
        port=_free_port(),
        label="peer",
        workspace_root="/peer_workspace",
        load_robot_visuals=False,
    ) as peer:
        baseline = _server_resource_ids(peer.server)

        def fail(*args, **kwargs):
            raise RuntimeError("GUI element creation failed")

        method = "add_slider" if stage == "controls" else "add_number"
        monkeypatch.setattr(peer.server.gui, method, fail)
        with pytest.raises(RuntimeError, match="GUI element creation failed"):
            ViserWorkspace(solver, server=peer.server, load_robot_visuals=False)
        assert _server_resource_ids(peer.server) == baseline
        peer._sliders[0].value = 0.35
        assert peer._q[0] == 0.35


def test_slow_shared_plane_export_does_not_restore_closed_viewer_resources(
    tmp_path, monkeypatch
):
    import workspace_analyzer.plane_visualization as plane_view

    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF), config=SolverConfig(backend="numpy")
    )
    with ViserWorkspace(
        solver,
        port=0,
        label="peer",
        workspace_root="/peer_workspace",
        load_robot_visuals=False,
    ) as peer:
        baseline = _server_resource_ids(peer.server)
        with ViserWorkspace(
            solver, server=peer.server, port=-1, load_robot_visuals=False
        ) as viewer:
            panel = viewer.plane_panel
            report = tmp_path / "plane.json"
            panel.configure(
                PlaneConfig(resolution=3),
                ReachabilityConfig(restarts=1, rescue_restarts=0),
                report=report,
            )
            entered, release = threading.Event(), threading.Event()
            real_writer = plane_view.save_result_artifacts

            def write(*args, **kwargs):
                entered.set()
                assert release.wait(5)
                return real_writer(*args, **kwargs)

            monkeypatch.setattr(plane_view, "save_result_artifacts", write)
            panel.compute()
            try:
                assert entered.wait(3)
                viewer.close()
                assert viewer._analysis_thread.is_alive()
                assert _server_resource_ids(peer.server) == baseline
                assert panel.info._impl.removed
                frozen_info = panel.info.content, panel.export_info.content
                peer._sliders[0].value = 0.35
                assert peer._q[0] == 0.35
            finally:
                release.set()
                viewer._analysis_thread.join(timeout=5)
            assert not viewer._analysis_thread.is_alive()
            assert report.is_file()
            assert not panel._export_running
            assert (panel.info.content, panel.export_info.content) == frozen_info
            assert _server_resource_ids(peer.server) == baseline
