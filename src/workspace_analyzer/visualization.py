"""Viser viewer for URDF visuals, workspace metrics, and live joint control."""

from __future__ import annotations

import threading
import time
import warnings
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import numpy as np

from .analyzer import AnalysisCancelled, AnalysisResult
from .kinematics import KinematicsSolver
from .presets import default_reference_joints


class ViserWorkspace:
    """Interactive robot and workspace viewer backed by Viser's web client."""

    def __init__(
        self,
        solver: KinematicsSolver,
        *,
        port: int = 8080,
        label: str = "robot",
        load_robot_visuals: bool = True,
        initial_q=None,
        server=None,
        workspace_root: str = "/workspace",
        gui_label: str | None = None,
        scene_offset=(0.0, 0.0, 0.0),
        load_full_robot: bool = False,
        reference_active_link_index: int = 1,
    ):
        try:
            import viser
        except ImportError as exc:
            raise ImportError(
                "Viser support requires workspace-analyzer[viser]"
            ) from exc
        self.solver = solver
        self._owns_server = server is None
        self.server = (
            viser.ViserServer(port=port, label="WorkspaceAnalyzer")
            if server is None
            else server
        )
        self.root = f"/{label}"
        self.workspace_root = workspace_root.rstrip("/")
        self.trajectory_root = self.root + "/overlays/trajectory"
        if not self.workspace_root.startswith("/") or not self.workspace_root:
            raise ValueError("workspace_root must be an absolute scene path")
        self._gui_prefix = "" if gui_label is None else f"{gui_label} · "
        scene_offset = np.asarray(scene_offset, dtype=float)
        if scene_offset.shape != (3,) or not np.isfinite(scene_offset).all():
            raise ValueError("scene_offset must contain three finite values")
        self._lock = threading.RLock()
        self._closed = False
        self._analysis_thread = None
        self._handles: dict[str, object] = {}
        self._link_frames: dict[str, object] = {}
        self._sliders = []
        self._q = np.asarray(
            default_reference_joints(solver) if initial_q is None else initial_q,
            dtype=float,
        ).copy()
        if self._q.shape != (solver.dof,):
            raise ValueError(f"initial_q must have shape ({solver.dof},)")
        self._initial_q = self._q.copy()
        self.reference_pose = _numpy(solver.forward(self._q)).copy()
        self.reference_joints = self._q.copy()
        self._cartesian_job = None
        self._recompute_running = False
        self._analysis_cancel = None
        self._display_base_transform = np.eye(4)
        self._load_full_robot = load_full_robot
        if reference_active_link_index < 1:
            raise ValueError("reference_active_link_index must be positive")
        self._reference_active_link_index = reference_active_link_index
        self._trajectory = None
        self._playback_stop = threading.Event()
        self._playback_thread = None
        self._playback_frame_update = False
        self._scene_root = self.server.scene.add_frame(
            self.root, show_axes=False, position=scene_offset
        )
        self._robot_root = self.server.scene.add_frame(
            self.root + "/links", show_axes=False
        )
        self._workspace_root = self.server.scene.add_frame(
            self.workspace_root, show_axes=False, position=scene_offset, visible=False
        )
        self._handles["workspace"] = self._workspace_root
        self._workspace_point_size = 0.006
        self._trajectory_point_size = 0.012
        self._trajectory_line_width = 3.0
        self._reachable_opacity = 0.65
        self._unreachable_opacity = 0.18
        self._workspace_result = None
        self._workspace_indices = None
        self._splat_update_timer = None
        self._add_controls()
        self.visual_count = 0
        if load_robot_visuals:
            self.visual_count = self.load_urdf_visuals(full_robot=load_full_robot)
        self._add_reference_frames()
        # Full-robot loading may change world_from_solver_base after GUI creation.
        self._update_reference_display()
        self.update_robot(self._q)
        if self._owns_server:
            self.frame_view()

    def _framing_points(self):
        poses = self.solver.forward(self._q, all_links=True)
        points = _transform_points(
            self._display_base_transform,
            np.array([_numpy(pose)[:3, 3] for pose in poses.values()]),
        )
        if self._workspace_result is not None and self._workspace_root.visible:
            cloud = _transform_points(
                self._display_base_transform,
                self._workspace_result.points[self._workspace_indices],
            )
            points = np.vstack((points, cloud))
        return points + np.asarray(self._scene_root.position)

    def frame_view(self, additional_viewers=()):
        """Frame robots and visible workspaces in the shared display coordinates."""
        points = np.vstack(
            [viewer._framing_points() for viewer in (self, *additional_viewers)]
        )
        points = points[np.isfinite(points).all(axis=1)]
        if not len(points):
            return
        lower, upper = points.min(axis=0), points.max(axis=0)
        center = (lower + upper) * 0.5
        radius = max(float(np.linalg.norm(upper - lower)) * 0.5, 0.25)
        direction = np.array([1.5, -2.0, 1.1])
        position = center + direction / np.linalg.norm(direction) * radius * 2.4
        cameras = [client.camera for client in self.server.get_clients().values()]
        initial = getattr(self.server, "initial_camera", None)
        if initial is not None:
            cameras.append(initial)
        for camera in cameras:
            camera.up_direction = (0.0, 0.0, 1.0)
            camera.position = tuple(position)
            camera.look_at = tuple(center)

    def reset_pose(self):
        """Restore the startup posture, including an explicitly supplied initial_q."""
        with self._lock:
            self._q[:] = self._initial_q
            for value, slider in zip(self._q, self._sliders):
                slider.value = float(value)
            self.update_robot(self._q)

    def _add_controls(self) -> None:
        with self.server.gui.add_folder(
            self._gui_prefix + "Display", expand_by_default=False
        ):
            frame = self.server.gui.add_button("Frame robot and workspace")

            @frame.on_click
            def _frame(_event):
                self.frame_view()

            controls = {
                "robot": self.server.gui.add_checkbox("Robot visuals", True),
                "skeleton": self.server.gui.add_checkbox("Skeleton", True),
                "joints": self.server.gui.add_checkbox("Joint points", False),
                "tool": self.server.gui.add_checkbox("TCP frame", True),
                "workspace": self.server.gui.add_checkbox("Workspace", False),
                "reference_frames": self.server.gui.add_checkbox(
                    "World/structural-base reference frames", True
                ),
                "trajectory_targets": self.server.gui.add_checkbox(
                    "Trajectory targets", True
                ),
                "trajectory_path": self.server.gui.add_checkbox(
                    "Trajectory path", True
                ),
                "trajectory_frames": self.server.gui.add_checkbox(
                    "Trajectory coordinate frames", True
                ),
            }
            self._visibility_controls = controls

            @controls["robot"].on_update
            def _robot_visibility(event):
                self._robot_root.visible = bool(event.target.value)

            for key in (
                "skeleton",
                "joints",
                "tool",
                "workspace",
                "reference_frames",
                "trajectory_targets",
                "trajectory_path",
                "trajectory_frames",
            ):
                control = controls[key]

                @control.on_update
                def _visibility(event, handle_key=key):
                    handle = self._handles.get(handle_key)
                    if handle is not None:
                        handle.visible = bool(event.target.value)

            point_size = self.server.gui.add_slider(
                "Workspace point size",
                min=0.001,
                max=0.03,
                step=0.001,
                initial_value=self._workspace_point_size,
            )
            reachable_opacity = self.server.gui.add_slider(
                "Reachable opacity",
                min=0.0,
                max=1.0,
                step=0.05,
                initial_value=self._reachable_opacity,
            )
            unreachable_opacity = self.server.gui.add_slider(
                "Unreachable opacity",
                min=0.0,
                max=1.0,
                step=0.05,
                initial_value=self._unreachable_opacity,
            )
            metric_selector = self.server.gui.add_dropdown(
                "Workspace color metric",
                ("manipulability",),
                initial_value="manipulability",
                disabled=True,
            )
            self._metric_selector = metric_selector
            trajectory_point_size = self.server.gui.add_slider(
                "Trajectory target size",
                min=0.002,
                max=0.04,
                step=0.001,
                initial_value=self._trajectory_point_size,
            )
            trajectory_line_width = self.server.gui.add_slider(
                "Trajectory line width",
                min=1.0,
                max=10.0,
                step=0.5,
                initial_value=self._trajectory_line_width,
            )
            self._metric_info = self.server.gui.add_markdown("")
            self.server.gui.add_markdown(
                "**Axes:** X red · Y green · Z blue  \n"
                "World/structural-base axes are parallel; target axes are desired TCP; "
                "TCP frame is FK actual."
            )

            @point_size.on_update
            def _point_size(event):
                self._workspace_point_size = float(event.target.value)
                cloud = self._handles.get("workspace_cloud")
                if cloud is not None:
                    cloud.point_size = self._workspace_point_size
                self._schedule_splat_refresh()

            @reachable_opacity.on_update
            def _reachable_opacity(event):
                self._reachable_opacity = float(event.target.value)
                self._schedule_splat_refresh()

            @unreachable_opacity.on_update
            def _unreachable_opacity(event):
                self._unreachable_opacity = float(event.target.value)
                self._schedule_splat_refresh()

            @metric_selector.on_update
            def _metric(event):
                self._set_workspace_color_metric(str(event.target.value))

            @trajectory_point_size.on_update
            def _trajectory_points(event):
                self._trajectory_point_size = float(event.target.value)
                handle = self._handles.get("trajectory_targets")
                if handle is not None:
                    handle.point_size = self._trajectory_point_size

            @trajectory_line_width.on_update
            def _trajectory_width(event):
                self._trajectory_line_width = float(event.target.value)
                handle = self._handles.get("trajectory_path")
                if handle is not None:
                    handle.thickness = self._trajectory_line_width

        with self.server.gui.add_folder(self._gui_prefix + "Joint control"):
            reset = self.server.gui.add_button("Reset to initial pose")

            @reset.on_click
            def _reset(_event):
                self.reset_pose()

            for i, (name, limit) in enumerate(
                zip(self.solver.joint_names, self.solver.joint_limits)
            ):
                slider = self.server.gui.add_slider(
                    name,
                    min=float(limit[0]),
                    max=float(limit[1]),
                    step=max(float(limit[1] - limit[0]) / 1000, 1e-4),
                    initial_value=float(self._q[i]),
                )

                @slider.on_update
                def _update(event, index=i):
                    with self._lock:
                        self._q[index] = event.target.value
                        self.update_robot(self._q)

                self._sliders.append(slider)

        with self.server.gui.add_folder(self._gui_prefix + "Reference pose"):
            capture = self.server.gui.add_button("Capture current FK pose")
            recompute = self.server.gui.add_button(
                "Recompute Cartesian reachability", disabled=True
            )
            cancel = self.server.gui.add_button("Cancel recompute", disabled=True)
            self._reference_info = self.server.gui.add_markdown("")
            self._recompute_button = recompute
            self._cancel_button = cancel

            @capture.on_click
            def _capture(_event):
                with self._lock:
                    self.capture_reference_pose()

            @recompute.on_click
            def _recompute(_event):
                self.recompute_cartesian_async()

            @cancel.on_click
            def _cancel(_event):
                if self._analysis_cancel is not None:
                    self._analysis_cancel.set()
                    self._set_reference_status("cancelling...")

        self._update_reference_display()

        with self.server.gui.add_folder(self._gui_prefix + "Trajectory playback"):
            self._play_button = self.server.gui.add_button("Play", disabled=True)
            self._stop_button = self.server.gui.add_button("Stop", disabled=True)
            self._frame_slider = self.server.gui.add_slider(
                "Frame", min=0, max=1, step=1, initial_value=0, disabled=True
            )
            self._fps_slider = self.server.gui.add_slider(
                "FPS", min=1, max=60, step=1, initial_value=20
            )
            self._loop_playback = self.server.gui.add_checkbox("Loop", True)
            self._trajectory_info = self.server.gui.add_markdown(
                "No trajectory loaded."
            )
            self._trajectory_frame_info = self.server.gui.add_markdown("")

            @self._play_button.on_click
            def _play(_event):
                self.play_trajectory()

            @self._stop_button.on_click
            def _stop(_event):
                self.stop_trajectory()

            @self._frame_slider.on_update
            def _frame(event):
                if self._trajectory is not None and not self._playback_frame_update:
                    self.show_trajectory_frame(int(event.target.value))

    def capture_reference_pose(self) -> np.ndarray:
        """Capture FK at the current sliders as the Cartesian reference pose."""
        self.reference_pose = _numpy(self.solver.forward(self._q)).copy()
        self.reference_joints = self._q.copy()
        self._update_reference_display()
        return self.reference_pose.copy()

    def _update_reference_display(self) -> None:
        pose = self.reference_pose
        display_pose = self._display_base_transform @ pose
        frame = self._handles.get("reference_tool")
        if frame is None:
            frame = self.server.scene.add_frame(
                self.root + "/overlays/reference_tool",
                axes_length=0.13,
                axes_radius=0.004,
            )
            self._handles["reference_tool"] = frame
        frame.wxyz = _wxyz(display_pose[:3, :3])
        frame.position = display_pose[:3, 3]
        local_xyz = ", ".join(f"{value:.4f}" for value in pose[:3, 3])
        world_xyz = ", ".join(f"{value:.4f}" for value in display_pose[:3, 3])
        joints = ", ".join(f"{value:.4f}" for value in self.reference_joints)
        self._reference_info.content = (
            "### Captured FK reference\n"
            f"- Solver-base XYZ: `[{local_xyz}]`\n"
            f"- Robot-world XYZ: `[{world_xyz}]`\n"
            f"- joints: `[{joints}]`"
        )

    def configure_cartesian_recompute(self, analyzer, config) -> None:
        """Enable recomputation using the orientation captured from the sliders."""
        self._cartesian_job = (analyzer, config)
        self._recompute_button.disabled = False

    def recompute_cartesian_async(self) -> None:
        """Recompute without blocking Viser's GUI callback thread."""
        with self._lock:
            if self._closed or self._cartesian_job is None or self._recompute_running:
                return
            self._recompute_running = True
            self._recompute_button.disabled = True
            self._cancel_button.disabled = False
            cancel = self._analysis_cancel = threading.Event()
            analyzer, config = self._cartesian_job
            updated = replace(
                config,
                reference_pose=self.reference_pose.copy(),
                reference_joints=self.reference_joints.copy(),
            )
            self._set_reference_status("computing 0.0%")

            def progress(value):
                with self._lock:
                    if not self._closed:
                        self._set_reference_status(f"computing {100.0 * value:.1f}%")

            def work():
                try:
                    result = analyzer.analyze_cartesian(
                        updated, cancel_event=cancel, progress_callback=progress
                    )
                    with self._lock:
                        if not self._closed and not cancel.is_set():
                            self.add_workspace(result)
                            self.last_cartesian_result = result
                            self._set_reference_status("complete")
                except AnalysisCancelled:
                    with self._lock:
                        if not self._closed:
                            self._set_reference_status("cancelled")
                except Exception as exc:  # Surface callback failures in the GUI.
                    with self._lock:
                        if not self._closed:
                            self._set_reference_status(f"failed: `{exc}`")
                finally:
                    with self._lock:
                        self._recompute_running = False
                        self._analysis_cancel = None
                        if not self._closed:
                            self._recompute_button.disabled = False
                            self._cancel_button.disabled = True

            self._analysis_thread = threading.Thread(target=work, daemon=True)
            self._analysis_thread.start()

    def _set_reference_status(self, status: str) -> None:
        self._update_reference_display()
        self._reference_info.content += f"\n- Status: **{status}**"

    def load_urdf_visuals(self, *, full_robot: bool = False) -> int:
        """Load mesh, box, cylinder, and sphere visuals for the selected chain."""
        source = self.solver.model.source
        if source is None:
            return 0
        try:
            import trimesh
        except ImportError as exc:
            raise ImportError("URDF visuals require workspace-analyzer[viser]") from exc
        root = ET.parse(source).getroot()
        if full_robot:
            neutral = _zero_tree_transforms(self.solver.model)
            self._display_base_transform = neutral[self.solver.base_link]
            for link_name, transform in neutral.items():
                frame = self._ensure_link_frame(link_name)
                frame.wxyz = _wxyz(transform[:3, :3])
                frame.position = transform[:3, 3]
            chain_links = set(self.solver.model.links)
        else:
            chain_links = {
                self.solver.base_link,
                *(joint.child for joint in self.solver.chain),
            }
        count = 0
        for link_node in root.findall("link"):
            link_name = link_node.get("name")
            if link_name not in chain_links:
                continue
            self._ensure_link_frame(link_name)
            for index, visual in enumerate(link_node.findall("visual")):
                origin = _origin_matrix(visual.find("origin"))
                scene_name = f"{self.root}/links/{link_name}/visual_{index}"
                try:
                    mesh_node = visual.find("geometry/mesh")
                    mesh_path = (
                        None
                        if mesh_node is None
                        else _resolve_mesh(mesh_node.get("filename", ""), source.parent)
                    )
                    if mesh_path is not None and mesh_path.suffix.lower() == ".glb":
                        scale = tuple(_numbers(mesh_node.get("scale"), (1.0, 1.0, 1.0)))
                        self.server.scene.add_glb(
                            scene_name,
                            glb_data=mesh_path.read_bytes(),
                            scale=scale,
                            wxyz=_wxyz(origin[:3, :3]),
                            position=origin[:3, 3],
                        )
                        count += 1
                        continue
                    mesh = _visual_mesh(visual, source.parent, trimesh)
                except (FileNotFoundError, ValueError) as exc:
                    warnings.warn(f"skipping {link_name} visual {index}: {exc}")
                    continue
                self.server.scene.add_mesh_trimesh(
                    scene_name,
                    mesh,
                    wxyz=_wxyz(origin[:3, :3]),
                    position=origin[:3, 3],
                )
                count += 1
        return count

    def _ensure_link_frame(self, link_name: str):
        handle = self._link_frames.get(link_name)
        if handle is None:
            handle = self.server.scene.add_frame(
                f"{self.root}/links/{link_name}", show_axes=False
            )
            self._link_frames[link_name] = handle
        return handle

    def update_robot(self, q) -> None:
        transforms = self.solver.forward(q, all_links=True)
        for link_name, transform_value in transforms.items():
            transform = self._display_base_transform @ _numpy(transform_value)
            frame = self._ensure_link_frame(link_name)
            frame.wxyz = _wxyz(transform[:3, :3])
            frame.position = transform[:3, 3]
        points = np.asarray(
            [
                (
                    self._display_base_transform
                    @ _numpy(transforms[self.solver.base_link])
                )[:3, 3]
            ]
            + [
                (self._display_base_transform @ _numpy(transforms[j.child]))[:3, 3]
                for j in self.solver.chain
            ]
        )
        segments = np.stack((points[:-1], points[1:]), axis=1)
        colors = np.broadcast_to(
            np.array([30, 144, 255], dtype=np.uint8), segments.shape
        )
        skeleton = self._handles.get("skeleton")
        if skeleton is None:
            skeleton = self.server.scene.add_line_segments(
                self.root + "/overlays/skeleton",
                segments,
                colors,
                thickness=4.0,
                thickness_units="screen",
            )
            self._handles["skeleton"] = skeleton
        else:
            skeleton.points = segments
        joints = self._handles.get("joints")
        if joints is None:
            joints = self.server.scene.add_point_cloud(
                self.root + "/overlays/joints",
                points=points,
                colors=np.tile([255, 165, 0], (len(points), 1)),
                point_size=0.015,
                point_shape="circle",
                visible=False,
            )
            self._handles["joints"] = joints
        else:
            joints.points = points
        tip = self._display_base_transform @ _numpy(transforms[self.solver.tip_link])
        tool = self._handles.get("tool")
        if tool is None:
            tool = self.server.scene.add_frame(
                self.root + "/overlays/tool", axes_length=0.1, axes_radius=0.003
            )
            self._handles["tool"] = tool
        tool.wxyz, tool.position = _wxyz(tip[:3, :3]), tip[:3, 3]

    def _add_reference_frames(self) -> None:
        """Show parallel comparison axes at world origin and structural link base."""
        root = self.server.scene.add_frame(
            self.root + "/overlays/reference_frames", show_axes=False
        )
        self._handles["reference_frames"] = root
        self.server.scene.add_frame(
            self.root + "/overlays/reference_frames/world",
            axes_length=0.16,
            axes_radius=0.004,
        )
        reference_from_base = np.eye(4)
        active_count = 0
        for joint in self.solver.chain:
            reference_from_base = reference_from_base @ joint.origin
            if joint.active:
                active_count += 1
                if active_count == self._reference_active_link_index:
                    break
        reference = self._display_base_transform @ reference_from_base
        # Deliberately keep identity rotation: this is the common world-axis frame
        # translated to the structural link base, not the URDF joint's local frame.
        self.server.scene.add_frame(
            self.root + "/overlays/reference_frames/structural_base_world_axes",
            axes_length=0.12,
            axes_radius=0.003,
            position=reference[:3, 3],
        )

    def add_workspace(
        self,
        result: AnalysisResult,
        *,
        point_size: float = 0.006,
        max_points: int = 250_000,
        color_metric: str = "manipulability",
        visible: bool | None = None,
    ) -> None:
        """Add a robustly metric-colored cloud with deterministic downsampling."""
        if max_points < 1:
            raise ValueError("max_points must be positive")
        indices = _display_indices(len(result.points), max_points)
        points = np.asarray(result.points[indices], dtype=np.float32)
        points = _transform_points(self._display_base_transform, points).astype(
            np.float32
        )
        first_workspace = self._workspace_result is None
        self._workspace_result = result
        self._workspace_indices = indices
        if visible is not None or first_workspace:
            # Show the first workspace by default. Recompute preserves toggles.
            show = True if visible is None else bool(visible)
            self._workspace_root.visible = show
            self._visibility_controls["workspace"].value = show
        if self._owns_server and first_workspace:
            self.frame_view()
        self._workspace_point_size = point_size
        old_cloud = self._handles.pop("workspace_cloud", None)
        if old_cloud is not None:
            old_cloud.remove()
        for key in ("reachable", "unreachable"):
            old_splats = self._handles.pop(f"{key}_splats", None)
            if old_splats is not None:
                old_splats.remove()
        if result.reachable is not None:
            self._metric_selector.disabled = True
            reachable = np.asarray(result.reachable[indices], dtype=bool)
            self._add_reachability_splats(points, reachable)
            rate = 100.0 * float(np.mean(result.reachable))
            panel = self._handles.get("workspace_info")
            content = (
                "### Cartesian reachability\n"
                f"- Targets: **{len(result.points):,}**\n"
                f"- Reachable: **{np.count_nonzero(result.reachable):,}** "
                f"(**{rate:.1f}%**)\n"
                "- Green: reachable · Red: unreachable"
            )
            assessment = result.metadata.get("assessment")
            if assessment is not None:
                content += (
                    f"\n- Quality accepted: **{assessment['quality_pass_count']:,}** "
                    f"(**{assessment['quality_pass_rate']:.1%}** of all targets)"
                    f"\n- IK succeeded, below quality gates: "
                    f"**{assessment['quality_rejected_count']:,}**"
                    f"\n- Dexterity task: **{result.metadata['dexterity_task']}**"
                    "\n- Quality gates evaluate the selected IK solution."
                )
            failure_stats = result.metadata.get("residual_summary", {}).get(
                "unreachable"
            )
            if failure_stats is not None:
                content += (
                    "\n- Unreachable residual: "
                    f"median `{failure_stats['median']:.3g}`, "
                    f"p95 `{failure_stats['p95']:.3g}`"
                )
            if panel is None:
                self._handles["workspace_info"] = self.server.gui.add_markdown(content)
            else:
                panel.content = content
            return
        available_metrics = {
            "manipulability": result.manipulability,
            **(result.metrics or {}),
        }
        if color_metric not in available_metrics:
            raise ValueError(
                f"unknown color metric {color_metric!r}; "
                f"choose from {sorted(available_metrics)}"
            )
        metric = available_metrics[color_metric]
        selectable_metrics = tuple(
            name for name, values in available_metrics.items() if values is not None
        )
        self._metric_selector.options = selectable_metrics or ("manipulability",)
        self._metric_selector.value = color_metric
        self._metric_selector.disabled = len(selectable_metrics) < 2
        if metric is None:
            colors = np.tile([50, 180, 255], (len(points), 1))
        else:
            colors = _metric_colors(np.asarray(metric[indices]))
        self._handles["workspace_cloud"] = self.server.scene.add_point_cloud(
            self.workspace_root + "/points",
            points=points,
            colors=colors.astype(np.uint8),
            point_size=point_size,
            point_shape="circle",
            precision="float32",
        )
        self._update_metric_info(color_metric, metric)

    def _set_workspace_color_metric(self, name: str) -> None:
        result, indices = self._workspace_result, self._workspace_indices
        cloud = self._handles.get("workspace_cloud")
        if result is None or indices is None or cloud is None:
            return
        metrics = {"manipulability": result.manipulability, **(result.metrics or {})}
        values = metrics.get(name)
        if values is not None:
            cloud.colors = _metric_colors(np.asarray(values[indices])).astype(np.uint8)
            self._update_metric_info(name, values)

    def _update_metric_info(self, name: str, values) -> None:
        if values is None:
            self._metric_info.content = ""
            return
        values = np.asarray(values)
        finite = values[np.isfinite(values)]
        if not len(finite):
            self._metric_info.content = f"**{name}**: no finite values"
            return
        p02, median, p98 = np.percentile(finite, (2, 50, 98))
        self._metric_info.content = (
            f"**Color: {name}** · p02 `{p02:.4g}` · "
            f"median `{median:.4g}` · p98 `{p98:.4g}`"
        )

    def add_trajectory(self, trajectory) -> None:
        """Render ordered IK targets and valid segments colored by dexterity."""
        self._trajectory = trajectory
        self._frame_slider.max = max(len(trajectory.positions) - 1, 1)
        self._frame_slider.value = 0
        self._frame_slider.disabled = False
        self._play_button.disabled = False
        self._stop_button.disabled = False
        summary = trajectory.summary()
        self._trajectory_info.content = (
            "### Trajectory metrics\n"
            f"- Frames: **{summary['samples']:,}**\n"
            f"- IK success: **{summary['success_rate']:.1%}**\n"
            f"- Max joint jump: **{summary['max_joint_jump']:.5f} rad**\n"
            "- Max normalized jump: "
            f"**{summary['max_normalized_joint_jump']:.5f}**\n"
            f"- Joint path length: **{summary['joint_path_length']:.4f} rad**\n"
            "- Normalized closure error: "
            f"**{_format_metric(summary['normalized_closure_joint_error'])}**\n"
            "- Peak / RMS velocity: "
            f"**{_format_metric(summary['max_joint_velocity'])} / "
            f"{_format_metric(summary['rms_joint_velocity'])} rad/s**\n"
            "- Minimum singular value: "
            f"**{_format_metric(summary['minimum_singular_value'])}**\n"
            "- Minimum joint-limit margin: "
            f"**{_format_metric(summary['minimum_joint_limit_margin'])}**\n"
            f"- Velocity violations: **{summary['velocity_violations']}**"
        )
        targets = np.asarray(trajectory.target_poses, dtype=float)
        success = np.asarray(trajectory.success, dtype=bool)
        points = _transform_points(
            self._display_base_transform, targets[:, :3, 3]
        ).astype(np.float32)
        point_colors = np.where(
            success[:, None],
            np.array([46, 204, 113], dtype=np.uint8),
            np.array([231, 76, 60], dtype=np.uint8),
        )
        old_points = self._handles.pop("trajectory_targets", None)
        if old_points is not None:
            old_points.remove()
        self._handles["trajectory_targets"] = self.server.scene.add_point_cloud(
            self.trajectory_root + "/targets",
            points=points,
            colors=point_colors,
            point_size=self._trajectory_point_size,
            point_shape="circle",
            precision="float32",
        )

        old_frames = self._handles.pop("trajectory_frames", None)
        if old_frames is not None:
            old_frames.remove()
        self._handles.pop("current_target", None)
        frame_root = self.server.scene.add_frame(
            self.trajectory_root + "/frames", show_axes=False
        )
        self._handles["trajectory_frames"] = frame_root
        stride = max(1, int(np.ceil(len(targets) / 24)))
        for index in range(0, len(targets), stride):
            display_pose = self._display_base_transform @ targets[index]
            self.server.scene.add_frame(
                self.trajectory_root + f"/frames/{index:04d}",
                axes_length=0.035,
                axes_radius=0.002,
                wxyz=_wxyz(display_pose[:3, :3]),
                position=display_pose[:3, 3],
            )
        current_pose = self._display_base_transform @ targets[0]
        current_target = self.server.scene.add_frame(
            self.trajectory_root + "/frames/current_target",
            axes_length=0.075,
            axes_radius=0.004,
            wxyz=_wxyz(current_pose[:3, :3]),
            position=current_pose[:3, 3],
        )
        self._handles["current_target"] = current_target
        old_error = self._handles.pop("pose_error", None)
        if old_error is not None:
            old_error.remove()
        self._handles["pose_error"] = self.server.scene.add_line_segments(
            self.trajectory_root + "/pose_error",
            points=np.stack((points[0], points[0]))[None, :, :],
            colors=np.array([[[255, 64, 64], [255, 220, 64]]], dtype=np.uint8),
            thickness=3.0,
            thickness_units="screen",
        )

        old_path = self._handles.pop("trajectory_path", None)
        if old_path is not None:
            old_path.remove()
        valid = success[:-1] & success[1:]
        if not np.any(valid):
            self.show_trajectory_frame(0)
            return
        segments = np.stack((points[:-1][valid], points[1:][valid]), axis=1)
        metric_colors = _metric_colors(trajectory.minimum_singular_value).astype(
            np.uint8
        )
        colors = np.stack((metric_colors[:-1][valid], metric_colors[1:][valid]), axis=1)
        self._handles["trajectory_path"] = self.server.scene.add_line_segments(
            self.trajectory_root + "/path",
            points=segments,
            colors=colors,
            thickness=self._trajectory_line_width,
            thickness_units="screen",
        )
        self.show_trajectory_frame(0)

    def show_trajectory_frame(self, index: int, *, update_sliders: bool = True) -> None:
        """Display one successful trajectory configuration on the robot."""
        if self._trajectory is None:
            return
        index = int(np.clip(index, 0, len(self._trajectory.positions) - 1))
        target = np.asarray(self._trajectory.target_poses[index], dtype=float)
        display_target = self._display_base_transform @ target
        target_handle = self._handles.get("current_target")
        if target_handle is not None:
            target_handle.wxyz = _wxyz(display_target[:3, :3])
            target_handle.position = display_target[:3, 3]
        if not self._trajectory.success[index]:
            self._trajectory_frame_info.content = (
                f"### Current frame {index}\n- IK: **failed**\n"
                f"- solver residual: `{self._trajectory.residual[index]:.6g}`"
            )
            return
        with self._lock:
            self._q[:] = self._trajectory.positions[index]
            if update_sliders:
                for value, slider in zip(self._q, self._sliders):
                    slider.value = float(value)
            self.update_robot(self._q)
            actual = _numpy(self.solver.forward(self._q))
            position_error = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
            relative_rotation = target[:3, :3].T @ actual[:3, :3]
            cosine = np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0)
            rotation_error = float(np.arccos(cosine))
            self._trajectory_frame_info.content = (
                f"### Current frame {index}\n"
                f"- IK: **success**\n"
                f"- position error: **{position_error * 1000:.3f} mm**\n"
                f"- orientation error: **{np.degrees(rotation_error):.4f} deg**\n"
                f"- solver residual: `{self._trajectory.residual[index]:.6g}`"
            )
            error_line = self._handles.get("pose_error")
            if error_line is not None:
                display_actual = self._display_base_transform @ actual
                error_line.points = np.array(
                    [[display_actual[:3, 3], display_target[:3, 3]]],
                    dtype=np.float32,
                )

    def play_trajectory(self) -> None:
        """Animate the loaded trajectory in a background thread."""
        with self._lock:
            if self._closed or self._trajectory is None:
                return
            if self._playback_thread is not None and self._playback_thread.is_alive():
                return
            self._playback_stop.clear()
            self._play_button.disabled = True

            def work():
                index = int(self._frame_slider.value)
                try:
                    while not self._playback_stop.is_set():
                        with self._lock:
                            if self._closed:
                                break
                            self.show_trajectory_frame(index, update_sliders=False)
                            self._playback_frame_update = True
                            try:
                                self._frame_slider.value = index
                            finally:
                                self._playback_frame_update = False
                        delay = 1.0 / max(float(self._fps_slider.value), 1.0)
                        if self._playback_stop.wait(delay):
                            break
                        index += 1
                        if index >= len(self._trajectory.positions):
                            if self._loop_playback.value:
                                index = 0
                            else:
                                break
                finally:
                    with self._lock:
                        if not self._closed:
                            self._play_button.disabled = False

            self._playback_thread = threading.Thread(target=work, daemon=True)
            self._playback_thread.start()

    def stop_trajectory(self) -> None:
        """Stop trajectory playback without changing the current frame."""
        self._playback_stop.set()

    def add_dashboard(self, markdown: str) -> None:
        """Add or update a viewer-specific analysis summary panel."""
        handle = self._handles.get("dashboard")
        if handle is None:
            handle = self.server.gui.add_markdown(markdown)
            self._handles["dashboard"] = handle
        else:
            handle.content = markdown

    def _add_reachability_splats(self, points, reachable) -> None:
        for key, mask, color, opacity in (
            ("reachable", reachable, (46, 204, 113), self._reachable_opacity),
            ("unreachable", ~reachable, (231, 76, 60), self._unreachable_opacity),
        ):
            centers = np.asarray(points[mask], dtype=np.float32)
            count = len(centers)
            covariances = _isotropic_covariances(count, self._workspace_point_size)
            colors = np.broadcast_to(
                np.asarray(color, dtype=np.uint8), (count, 3)
            ).copy()
            opacities = np.full((count, 1), opacity, dtype=np.float32)
            self._handles[f"{key}_splats"] = self.server.scene.add_gaussian_splats(
                f"{self.workspace_root}/{key}",
                centers,
                covariances,
                colors,
                opacities,
            )

    def _update_splat_covariances(self) -> None:
        for key in ("reachable", "unreachable"):
            handle = self._handles.get(f"{key}_splats")
            if handle is not None:
                handle.covariances = _isotropic_covariances(
                    len(handle.centers), self._workspace_point_size
                )

    def _update_splat_opacity(self, key: str) -> None:
        handle = self._handles.get(f"{key}_splats")
        if handle is None:
            return
        opacity = (
            self._reachable_opacity if key == "reachable" else self._unreachable_opacity
        )
        handle.opacities = np.full((len(handle.centers), 1), opacity, dtype=np.float32)

    def _schedule_splat_refresh(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._splat_update_timer is not None:
                self._splat_update_timer.cancel()
            timer = threading.Timer(0.12, self._refresh_splats)
            timer.daemon = True
            self._splat_update_timer = timer
            timer.start()

    def _refresh_splats(self) -> None:
        with self._lock:
            if self._closed:
                return
            for key, opacity in (
                ("reachable", self._reachable_opacity),
                ("unreachable", self._unreachable_opacity),
            ):
                handle = self._handles.get(f"{key}_splats")
                if handle is None:
                    continue
                centers = handle.centers.copy()
                handle.set_gaussians(
                    centers,
                    _isotropic_covariances(len(centers), self._workspace_point_size),
                    handle.rgbs.copy(),
                    np.full((len(centers), 1), opacity, dtype=np.float32),
                )
            self._splat_update_timer = None

    def wait(self) -> None:
        try:
            while not self._closed:
                time.sleep(1)
        except KeyboardInterrupt:
            self.close()

    def close(self) -> None:
        """Cancel background work before stopping an owned Viser server."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self.stop_trajectory()
            if self._analysis_cancel is not None:
                self._analysis_cancel.set()
            if self._splat_update_timer is not None:
                self._splat_update_timer.cancel()
                self._splat_update_timer = None
        # Join outside the lock so workers can run their cleanup handlers.
        for worker in (self._playback_thread, self._analysis_thread):
            if worker is not None and worker is not threading.current_thread():
                worker.join(timeout=1.0)
        if self._owns_server:
            self.server.stop()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.close()


def _visual_mesh(visual, urdf_directory: Path, trimesh):
    geometry = visual.find("geometry")
    if geometry is None:
        raise ValueError("missing <geometry>")
    mesh_node = geometry.find("mesh")
    if mesh_node is not None:
        loaded = trimesh.load(
            _resolve_mesh(mesh_node.get("filename", ""), urdf_directory),
            force="scene",
            process=False,
        )
        mesh = _flatten_scene(loaded, trimesh)
        mesh.apply_scale(_numbers(mesh_node.get("scale"), (1.0, 1.0, 1.0)))
    elif geometry.find("box") is not None:
        mesh = trimesh.creation.box(
            extents=_numbers(geometry.find("box").get("size"), (1, 1, 1))
        )
    elif geometry.find("cylinder") is not None:
        node = geometry.find("cylinder")
        mesh = trimesh.creation.cylinder(
            radius=float(node.get("radius")), height=float(node.get("length"))
        )
    elif geometry.find("sphere") is not None:
        mesh = trimesh.creation.icosphere(
            subdivisions=2, radius=float(geometry.find("sphere").get("radius"))
        )
    else:
        raise ValueError("unsupported URDF visual geometry")
    color_node = visual.find("material/color")
    if color_node is not None:
        rgba = np.clip(
            _numbers(color_node.get("rgba"), (0.7, 0.7, 0.7, 1)) * 255, 0, 255
        )
        mesh.visual.face_colors = rgba.astype(np.uint8)
    return mesh


def _flatten_scene(loaded, trimesh):
    if isinstance(loaded, trimesh.Trimesh):
        return loaded.copy()
    meshes = []
    for node_name in loaded.graph.nodes_geometry:
        transform, geometry_name = loaded.graph[node_name]
        mesh = loaded.geometry[geometry_name].copy()
        mesh.apply_transform(transform)
        meshes.append(mesh)
    if not meshes:
        raise ValueError("mesh scene contains no geometry")
    return trimesh.util.concatenate(meshes)


def _resolve_mesh(filename: str, directory: Path) -> Path:
    if not filename:
        raise ValueError("empty mesh filename")
    if filename.startswith("file://"):
        candidate = Path(filename[7:])
    elif filename.startswith("package://"):
        relative = Path(filename[len("package://") :])
        candidates = [directory / relative, directory / Path(*relative.parts[1:])]
        candidate = next((path for path in candidates if path.is_file()), candidates[0])
    else:
        candidate = Path(filename)
        if not candidate.is_absolute():
            candidate = directory / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def _origin_matrix(node) -> np.ndarray:
    transform = np.eye(4)
    if node is None:
        return transform
    xyz = _numbers(node.get("xyz"), (0.0, 0.0, 0.0))
    roll, pitch, yaw = _numbers(node.get("rpy"), (0.0, 0.0, 0.0))
    cr, sr, cp, sp, cy, sy = (
        np.cos(roll),
        np.sin(roll),
        np.cos(pitch),
        np.sin(pitch),
        np.cos(yaw),
        np.sin(yaw),
    )
    transform[:3, :3] = (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )
    transform[:3, 3] = xyz
    return transform


def _numbers(text, default):
    return np.asarray(default if text is None else [float(x) for x in text.split()])


def _turbo_like(scale):
    anchors = np.asarray(((48, 18, 130), (31, 173, 230), (172, 220, 50), (240, 53, 31)))
    position = scale * (len(anchors) - 1)
    lower = np.minimum(position.astype(int), len(anchors) - 2)
    fraction = (position - lower)[:, None]
    return anchors[lower] * (1 - fraction) + anchors[lower + 1] * fraction


def _metric_colors(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    colors = np.tile([128, 128, 128], (len(values), 1)).astype(float)
    if np.any(finite):
        low, high = np.percentile(values[finite], (2, 98))
        scale = np.clip((values[finite] - low) / max(high - low, 1e-12), 0, 1)
        colors[finite] = _turbo_like(scale)
    return colors


def _isotropic_covariances(count: int, point_size: float) -> np.ndarray:
    covariance = np.eye(3, dtype=np.float32) * float(point_size) ** 2
    return np.broadcast_to(covariance, (count, 3, 3)).copy()


def _zero_tree_transforms(model) -> dict[str, np.ndarray]:
    """Compute a full URDF tree pose with every movable joint set to zero."""
    children = {}
    for joint in model.joints:
        children.setdefault(joint.parent, []).append(joint)
    transforms = {root: np.eye(4) for root in model.root_links}
    queue = list(model.root_links)
    while queue:
        parent = queue.pop(0)
        for joint in children.get(parent, ()):
            value = 0.0
            transforms[joint.child] = (
                transforms[parent] @ joint.origin @ _joint_motion(joint, value)
            )
            queue.append(joint.child)
    if len(transforms) != len(model.links):
        raise ValueError("cannot construct a complete neutral URDF tree pose")
    return transforms


def _joint_motion(joint, value: float) -> np.ndarray:
    transform = np.eye(4)
    if joint.kind == "prismatic":
        transform[:3, 3] = joint.axis * value
    elif joint.kind in {"revolute", "continuous"}:
        axis = joint.axis
        skew = np.array(
            (
                (0.0, -axis[2], axis[1]),
                (axis[2], 0.0, -axis[0]),
                (-axis[1], axis[0], 0.0),
            )
        )
        transform[:3, :3] = (
            np.eye(3) + np.sin(value) * skew + (1.0 - np.cos(value)) * (skew @ skew)
        )
    return transform


def _format_metric(value) -> str:
    return "n/a" if value is None else f"{value:.5g}"


def _display_indices(count: int, max_points: int) -> np.ndarray:
    if count <= max_points:
        return np.arange(count)
    return np.linspace(0, count - 1, max_points, dtype=np.int64)


def _transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def _numpy(value):
    return (
        value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    )


def _wxyz(rotation):
    import trimesh

    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    return trimesh.transformations.quaternion_from_matrix(matrix)
