"""Viser viewer for URDF visuals, workspace metrics, and live joint control."""

from __future__ import annotations

import threading
import time
import warnings
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import numpy as np

from .analyzer import AnalysisResult
from .kinematics import KinematicsSolver


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
    ):
        try:
            import viser
        except ImportError as exc:
            raise ImportError(
                "Viser support requires workspace-analyzer[viser]"
            ) from exc
        self.solver = solver
        self.server = viser.ViserServer(port=port, label="WorkspaceAnalyzer")
        self.root = f"/{label}"
        self._lock = threading.RLock()
        self._handles: dict[str, object] = {}
        self._link_frames: dict[str, object] = {}
        self._sliders = []
        self._q = np.asarray(
            solver.joint_limits.mean(axis=1) if initial_q is None else initial_q,
            dtype=float,
        ).copy()
        if self._q.shape != (solver.dof,):
            raise ValueError(f"initial_q must have shape ({solver.dof},)")
        self.reference_pose = _numpy(solver.forward(self._q)).copy()
        self.reference_joints = self._q.copy()
        self._cartesian_job = None
        self._recompute_running = False
        self._robot_root = self.server.scene.add_frame(
            self.root + "/links", show_axes=False
        )
        self._workspace_root = self.server.scene.add_frame(
            "/workspace", show_axes=False
        )
        self._handles["workspace"] = self._workspace_root
        self._workspace_point_size = 0.006
        self._reachable_opacity = 0.65
        self._unreachable_opacity = 0.18
        self._add_controls()
        self.visual_count = 0
        if load_robot_visuals:
            self.visual_count = self.load_urdf_visuals()
        self.update_robot(self._q)

    def _add_controls(self) -> None:
        with self.server.gui.add_folder("Display"):
            controls = {
                "robot": self.server.gui.add_checkbox("Robot visuals", True),
                "skeleton": self.server.gui.add_checkbox("Skeleton", True),
                "joints": self.server.gui.add_checkbox("Joint points", False),
                "tool": self.server.gui.add_checkbox("TCP frame", True),
                "workspace": self.server.gui.add_checkbox("Workspace", True),
            }

            @controls["robot"].on_update
            def _robot_visibility(event):
                self._robot_root.visible = bool(event.target.value)

            for key in ("skeleton", "joints", "tool", "workspace"):
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

            @point_size.on_update
            def _point_size(event):
                self._workspace_point_size = float(event.target.value)
                cloud = self._handles.get("workspace_cloud")
                if cloud is not None:
                    cloud.point_size = self._workspace_point_size
                self._update_splat_covariances()

            @reachable_opacity.on_update
            def _reachable_opacity(event):
                self._reachable_opacity = float(event.target.value)
                self._update_splat_opacity("reachable")

            @unreachable_opacity.on_update
            def _unreachable_opacity(event):
                self._unreachable_opacity = float(event.target.value)
                self._update_splat_opacity("unreachable")

        with self.server.gui.add_folder("Joint control"):
            reset = self.server.gui.add_button("Reset to joint centers")

            @reset.on_click
            def _reset(_event):
                with self._lock:
                    self._q[:] = self.solver.joint_limits.mean(axis=1)
                    for value, slider in zip(self._q, self._sliders):
                        slider.value = float(value)
                    self.update_robot(self._q)

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

        with self.server.gui.add_folder("Reference pose"):
            capture = self.server.gui.add_button("Capture current FK pose")
            recompute = self.server.gui.add_button(
                "Recompute Cartesian reachability", disabled=True
            )
            self._reference_info = self.server.gui.add_markdown("")
            self._recompute_button = recompute

            @capture.on_click
            def _capture(_event):
                with self._lock:
                    self.capture_reference_pose()

            @recompute.on_click
            def _recompute(_event):
                self.recompute_cartesian_async()

        self._update_reference_display()

    def capture_reference_pose(self) -> np.ndarray:
        """Capture FK at the current sliders as the Cartesian reference pose."""
        self.reference_pose = _numpy(self.solver.forward(self._q)).copy()
        self.reference_joints = self._q.copy()
        self._update_reference_display()
        return self.reference_pose.copy()

    def _update_reference_display(self) -> None:
        pose = self.reference_pose
        frame = self._handles.get("reference_tool")
        if frame is None:
            frame = self.server.scene.add_frame(
                self.root + "/overlays/reference_tool",
                axes_length=0.13,
                axes_radius=0.004,
            )
            self._handles["reference_tool"] = frame
        frame.wxyz = _wxyz(pose[:3, :3])
        frame.position = pose[:3, 3]
        xyz = ", ".join(f"{value:.4f}" for value in pose[:3, 3])
        joints = ", ".join(f"{value:.4f}" for value in self._q)
        self._reference_info.content = (
            f"### Captured FK reference\n- XYZ: `[{xyz}]`\n- joints: `[{joints}]`"
        )

    def configure_cartesian_recompute(self, analyzer, config) -> None:
        """Enable recomputation using the orientation captured from the sliders."""
        self._cartesian_job = (analyzer, config)
        self._recompute_button.disabled = False

    def recompute_cartesian_async(self) -> None:
        """Recompute without blocking Viser's GUI callback thread."""
        if self._cartesian_job is None or self._recompute_running:
            return
        self._recompute_running = True
        self._recompute_button.disabled = True
        self._reference_info.content += "\n- Status: **computing...**"

        def work():
            try:
                analyzer, config = self._cartesian_job
                updated = replace(
                    config,
                    reference_pose=self.reference_pose.copy(),
                    reference_joints=self.reference_joints.copy(),
                )
                result = analyzer.analyze_cartesian(updated)
                self.add_workspace(result)
                self.last_cartesian_result = result
                self._update_reference_display()
                self._reference_info.content += "\n- Status: **complete**"
            except Exception as exc:  # Surface callback failures in the GUI.
                self._reference_info.content += f"\n- Status: **failed:** `{exc}`"
            finally:
                self._recompute_running = False
                self._recompute_button.disabled = False

        threading.Thread(target=work, daemon=True).start()

    def load_urdf_visuals(self) -> int:
        """Load mesh, box, cylinder, and sphere visuals for the selected chain."""
        source = self.solver.model.source
        if source is None:
            return 0
        try:
            import trimesh
        except ImportError as exc:
            raise ImportError("URDF visuals require workspace-analyzer[viser]") from exc
        root = ET.parse(source).getroot()
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
                try:
                    mesh = _visual_mesh(visual, source.parent, trimesh)
                except (FileNotFoundError, ValueError) as exc:
                    warnings.warn(f"skipping {link_name} visual {index}: {exc}")
                    continue
                origin = _origin_matrix(visual.find("origin"))
                self.server.scene.add_mesh_trimesh(
                    f"{self.root}/links/{link_name}/visual_{index}",
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
            transform = _numpy(transform_value)
            frame = self._ensure_link_frame(link_name)
            frame.wxyz = _wxyz(transform[:3, :3])
            frame.position = transform[:3, 3]
        points = np.asarray(
            [_numpy(transforms[self.solver.base_link])[:3, 3]]
            + [_numpy(transforms[j.child])[:3, 3] for j in self.solver.chain]
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
        tip = _numpy(transforms[self.solver.tip_link])
        tool = self._handles.get("tool")
        if tool is None:
            tool = self.server.scene.add_frame(
                self.root + "/overlays/tool", axes_length=0.1, axes_radius=0.003
            )
            self._handles["tool"] = tool
        tool.wxyz, tool.position = _wxyz(tip[:3, :3]), tip[:3, 3]

    def add_workspace(
        self,
        result: AnalysisResult,
        *,
        point_size: float = 0.006,
        max_points: int = 250_000,
    ) -> None:
        """Add a robustly metric-colored cloud with deterministic downsampling."""
        stride = max(1, int(np.ceil(len(result.points) / max_points)))
        points = np.asarray(result.points[::stride], dtype=np.float32)
        self._workspace_point_size = point_size
        old_cloud = self._handles.pop("workspace_cloud", None)
        if old_cloud is not None:
            old_cloud.remove()
        for key in ("reachable", "unreachable"):
            old_splats = self._handles.pop(f"{key}_splats", None)
            if old_splats is not None:
                old_splats.remove()
        if result.reachable is not None:
            reachable = np.asarray(result.reachable[::stride], dtype=bool)
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
            if panel is None:
                self._handles["workspace_info"] = self.server.gui.add_markdown(content)
            else:
                panel.content = content
            return
        if result.manipulability is None:
            colors = np.tile([50, 180, 255], (len(points), 1))
        else:
            score = np.asarray(result.manipulability[::stride])
            low, high = np.percentile(score, (2, 98))
            scale = np.clip((score - low) / max(high - low, 1e-12), 0, 1)
            colors = _turbo_like(scale)
        self._handles["workspace_cloud"] = self.server.scene.add_point_cloud(
            "/workspace/points",
            points=points,
            colors=colors.astype(np.uint8),
            point_size=point_size,
            point_shape="circle",
            precision="float32",
        )

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
                f"/workspace/{key}", centers, covariances, colors, opacities
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

    def wait(self) -> None:
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


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


def _isotropic_covariances(count: int, point_size: float) -> np.ndarray:
    covariance = np.eye(3, dtype=np.float32) * float(point_size) ** 2
    return np.broadcast_to(covariance, (count, 3, 3)).copy()


def _numpy(value):
    return (
        value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    )


def _wxyz(rotation):
    import trimesh

    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    return trimesh.transformations.quaternion_from_matrix(matrix)
