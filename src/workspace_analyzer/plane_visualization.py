"""Interactive plane selection, asynchronous evaluation and browser downloads."""

from __future__ import annotations

import json
import tempfile
import threading
from dataclasses import replace
from pathlib import Path

import numpy as np

from ._arrays import as_numpy as _numpy
from ._cancellation import AnalysisCancelled
from ._output_paths import validate_output_paths
from .plane import PlaneConfig, analyze_plane, plane_frame
from .plane_plot import plane_svg
from .reachability import ReachabilityConfig
from .result_export import save_result_artifacts
from .sampling import SamplingConfig, SamplingStrategy


class PlaneReachabilityPanel:
    def __init__(self, viewer):
        self.viewer = viewer
        self.result = None
        self._export_paths = {}
        self._export_errors = {}
        self._saved_paths = {}
        self._export_running = False
        self._configured = False
        folder = viewer.server.gui.add_folder(
            viewer._gui_prefix + "Plane reachability", order=-10
        )
        viewer._gui_folders.append(folder)
        with folder:
            gui = viewer.server.gui
            self.frame = gui.add_dropdown(
                "Sampling frame", ("base", "world", "reference"), initial_value="base"
            )
            self.frame_rpy = gui.add_vector3(
                "Sampling frame RPY (rad)",
                initial_value=(0.0, 0.0, 0.0),
                step=0.1,
            )
            self.plane = gui.add_dropdown(
                "Plane", ("xy", "xz", "yz"), initial_value="xy"
            )
            self.center = gui.add_vector3(
                "Plane center (m)",
                initial_value=tuple(viewer.reference_pose[:3, 3]),
                step=0.01,
            )
            self.center_on_tcp = gui.add_button("Center plane on current TCP")
            self.span = gui.add_number(
                "Plane side length (m)", initial_value=0.4, min=0.001, step=0.05
            )
            self.sampling_method = gui.add_dropdown(
                "Plane sampling",
                ("grid", *(item.value for item in SamplingStrategy)),
                initial_value="grid",
            )
            self.sample_count = gui.add_number(
                "Plane sample count", initial_value=1000, min=1, step=1
            )
            self.resolution = gui.add_number(
                "Points per axis", initial_value=21, min=2, step=1
            )
            self.full_pose = gui.add_checkbox(
                "Constrain TCP orientation", initial_value=False
            )
            self.rpy = gui.add_vector3(
                "TCP RPY (rad)", initial_value=(0.0, 0.0, 0.0), step=0.1
            )
            self.show_preview = gui.add_checkbox(
                "Show plane preview", initial_value=False
            )
            with gui.add_folder("Solver settings", expand_by_default=False):
                self.batch_size = gui.add_number(
                    "IK batch size", initial_value=256, min=1, step=1
                )
                self.sampling_seed = gui.add_number(
                    "Sampling seed", initial_value=42, min=0, step=1
                )
                self.seed = gui.add_number(
                    "IK random seed", initial_value=42, min=0, step=1
                )
                self.restarts = gui.add_number(
                    "IK starts", initial_value=4, min=1, step=1
                )
            self.compute_button = gui.add_button("Compute plane reachability")
            self.cancel_button = gui.add_button(
                "Cancel plane computation", disabled=True
            )
            self.download_json = gui.add_button(
                "Download last plane report", disabled=True
            )
            self.download_npz = gui.add_button(
                "Download last plane samples", disabled=True
            )
            self.download_svg = gui.add_button(
                "Download last plane plot", disabled=True
            )
            self.retry_save = gui.add_button(
                "Retry saving last plane result", disabled=True
            )
            self.info = gui.add_markdown("")
            self.export_info = gui.add_markdown("")
            self.plot_view = gui.add_html("") if hasattr(gui, "add_html") else None
        self.inputs = (
            self.frame,
            self.frame_rpy,
            self.plane,
            self.center,
            self.span,
            self.resolution,
            self.sampling_method,
            self.sample_count,
            self.full_pose,
            self.rpy,
            self.batch_size,
            self.seed,
            self.sampling_seed,
            self.restarts,
            self.show_preview,
        )
        for control in self.inputs:
            control.on_update(self._settings_changed)
        self.center_on_tcp.on_click(lambda _: self.center_at_current_tcp())
        self.compute_button.on_click(lambda _: self.compute())
        self.cancel_button.on_click(lambda _: self.cancel())
        self.download_json.on_click(lambda event: self.download(event, "json"))
        self.download_npz.on_click(lambda event: self.download(event, "npz"))
        self.download_svg.on_click(lambda event: self.download(event, "svg"))
        self.retry_save.on_click(lambda _: self.retry_exports())
        self.configure(
            PlaneConfig(
                center=tuple(viewer.reference_pose[:3, 3]), span=0.4, resolution=21
            ),
            ReachabilityConfig(batch_size=256),
            show_preview=False,
        )

    def configure(
        self,
        plane,
        assessment=None,
        *,
        output=None,
        report=None,
        plot=None,
        cache=None,
        show_preview=True,
    ):
        output, report, plot = validate_output_paths(
            output=output,
            report=report,
            plot=plot,
            inputs=(self.viewer.solver.model.source,),
        )
        with self.viewer._lock:
            if self.viewer._closed:
                return
            if self.viewer._recompute_running:
                raise RuntimeError("cannot configure a plane during computation")
            self._configured = False
            self.assessment = assessment or ReachabilityConfig()
            self.output, self.report, self.cache = output, report, cache
            self.plot = plot
            self.frame.options = tuple(
                dict.fromkeys(
                    (
                        "base",
                        "world",
                        "reference",
                        *sorted(self.viewer.solver.model.links),
                        plane.frame,
                    )
                )
            )
            self.frame.value = plane.frame
            self.frame_rpy.value = plane.frame_rpy
            self.plane.value = plane.plane
            self.center.value = tuple(plane.effective_center)
            self.span.value = plane.span
            self.resolution.value = plane.resolution
            self.sampling_method.value = (
                "grid"
                if plane.sampling is None
                else SamplingStrategy(plane.sampling.strategy).value
            )
            if plane.sampling is not None:
                self.sample_count.value = plane.sampling.num_samples
            self.full_pose.value = not self.assessment.position_only
            self.rpy.value = plane.rpy
            self.batch_size.value = self.assessment.batch_size
            self.seed.value = self.assessment.random_seed
            self.sampling_seed.value = (
                self.assessment.random_seed
                if plane.sampling is None
                else plane.sampling.seed
            )
            self.restarts.value = self.assessment.restarts
            self.show_preview.value = show_preview
            self._configured = True
            self._preview()

    def configuration(self):
        plane = PlaneConfig(
            plane=self.plane.value,
            frame=self.frame.value,
            frame_rpy=self.frame_rpy.value,
            center=self.center.value,
            span=self.span.value,
            resolution=self.resolution.value,
            rpy=self.rpy.value,
            sampling=None
            if self.sampling_method.value == "grid"
            else SamplingConfig(
                SamplingStrategy(self.sampling_method.value),
                self.sample_count.value,
                self.batch_size.value,
                self.sampling_seed.value,
            ),
        )
        assessment = replace(
            self.assessment,
            position_only=not self.full_pose.value,
            batch_size=self.batch_size.value,
            random_seed=self.seed.value,
            restarts=self.restarts.value,
        )
        return plane, assessment

    def _settings_changed(self, _event):
        with self.viewer._lock:
            if (
                self._configured
                and not self.viewer._closed
                and not self.viewer._recompute_running
            ):
                self._preview()

    def _preview(self):
        try:
            plane, _ = self.configuration()
            name, base_from_frame = plane_frame(
                self.viewer.solver, plane.frame, plane.frame_rpy
            )
            transform = self.viewer._display_base_transform @ base_from_frame
            corners = plane.corners() @ transform[:3, :3].T + transform[:3, 3]
            segments = np.stack((corners, np.roll(corners, -1, axis=0)), axis=1)
            handle = self.viewer.server.scene.add_line_segments(
                self.viewer.root + "/overlays/plane_preview",
                points=segments,
                colors=(240, 170, 40),
                thickness=3.0,
                thickness_units="screen",
                visible=self.show_preview.value,
            )
            self.viewer._handles["plane_preview"] = handle
            if self.show_preview.value:
                self.viewer.frame_view()
            self._update_task_controls()
            suffix = (
                " Last completed result is still displayed."
                if self.result is not None
                else ""
            )
            self.info.content = (
                f"**Ready: {plane.num_samples:,} targets** · "
                f"{plane.plane.upper()} in `{name}`.\n\n"
                "The orange outline previews the region. Click Compute to evaluate."
                + suffix
            )
        except (ValueError, TypeError) as exc:
            self.info.content = f"**Invalid plane settings:** {exc}"

    def _update_task_controls(self):
        self.rpy.disabled = not self.full_pose.value
        self.resolution.disabled = self.sampling_method.value != "grid"
        self.sample_count.disabled = self.sampling_method.value == "grid"
        self.sampling_seed.disabled = self.sampling_method.value == "grid"

    def center_at_current_tcp(self):
        """Place the plane at slider FK, expressed in the selected sampling frame."""
        viewer = self.viewer
        with viewer._lock:
            if viewer._closed or viewer._recompute_running:
                return
            try:
                _, base_from_frame = plane_frame(
                    viewer.solver, self.frame.value, self.frame_rpy.value
                )
                tcp = _numpy(viewer.solver.forward(viewer._q))[:3, 3]
                self.center.value = tuple(
                    (tcp - base_from_frame[:3, 3]) @ base_from_frame[:3, :3]
                )
                self.show_preview.value = True
                self._preview()
            except (ValueError, TypeError) as exc:
                self.info.content = f"**Cannot center plane:** {exc}"

    def compute(self):
        viewer = self.viewer
        with viewer._lock:
            if viewer._closed:
                return
            if self._export_running:
                return
            if viewer._recompute_running:
                self.info.content = (
                    "**Another computation is running.** "
                    "Cancel it or wait for completion."
                )
                return
            try:
                plane, assessment = self.configuration()
                plane_frame(viewer.solver, plane.frame, plane.frame_rpy)
            except (ValueError, TypeError) as exc:
                self.info.content = f"**Invalid plane settings:** {exc}"
                return
            seed = viewer.reference_joints.copy()
            output, report, cache = self.output, self.report, self.cache
            plot = self.plot
            cancel = viewer._analysis_cancel = threading.Event()
            viewer._recompute_running = True
            viewer._recompute_button.disabled = True
            self.compute_button.disabled = True
            self.center_on_tcp.disabled = True
            self.retry_save.disabled = True
            self.cancel_button.disabled = False
            for control in self.inputs:
                control.disabled = True
            self.info.content = f"**Computing 0.0%** · {plane.num_samples:,} targets"

            def finish():
                with viewer._lock:
                    self._export_running = False
                    viewer._recompute_running = False
                    viewer._analysis_cancel = None
                    if not viewer._closed:
                        viewer._recompute_button.disabled = (
                            viewer._cartesian_job is None
                        )
                        self.compute_button.disabled = False
                        self.center_on_tcp.disabled = False
                        self.retry_save.disabled = not bool(self._export_errors)
                        self.cancel_button.disabled = True
                        for control in self.inputs:
                            control.disabled = False
                        self._update_task_controls()

            def progress(value):
                with viewer._lock:
                    if (
                        not viewer._closed
                        and viewer._analysis_cancel is cancel
                        and not cancel.is_set()
                    ):
                        self.info.content = (
                            f"**Computing {value:.1%}** · {plane.num_samples:,} targets"
                        )

            def work():
                try:
                    result = analyze_plane(
                        viewer.solver,
                        plane,
                        assessment,
                        seed=seed,
                        cache=cache,
                        cancel_event=cancel,
                        progress_callback=progress,
                    )
                    svg = plane_svg(result, max_points=5000)
                    with viewer._lock:
                        if viewer._closed or cancel.is_set():
                            raise AnalysisCancelled()
                        viewer.add_workspace(result)
                        self.result = result
                        self.download_json.disabled = False
                        self.download_npz.disabled = False
                        self.download_svg.disabled = False
                        if self.plot_view is not None:
                            self.plot_view.content = svg
                        viewer.frame_view()
                        summary = result.metadata["plane_scan"]
                        task_label = (
                            "position only"
                            if assessment.position_only
                            else "fixed orientation"
                        )
                        self.info.content = (
                            f"**Complete: {summary['reachable']:,}/"
                            f"{summary['samples']:,} targets solved "
                            f"({summary['success_rate']:.1%})**\n\n"
                            f"{plane.plane.upper()} · "
                            f"`{summary['coordinate_frame']}` · "
                            f"{task_label}. "
                            "Plot: green passes quality gates; "
                            "amber is solved below gates; "
                            "red means IK did not find a solution. "
                            "Downloads contain this completed run."
                        )
                        self._export_paths = {
                            kind: path
                            for kind, path in (
                                ("npz", output),
                                ("json", report),
                                ("svg", plot),
                            )
                            if path is not None
                        }
                        self._saved_paths = {}
                        self._export_errors = {}
                        paths = self._export_paths.copy()
                        self._export_running = True
                        self.export_info.content = (
                            "**Saving last completed run…**" if paths else ""
                        )
                        # The measured result is now published. Cancellation
                        # applies to analysis, not these independent atomic saves.
                        viewer._analysis_cancel = None
                        self.cancel_button.disabled = True
                    self._save_exports(result, paths)
                except AnalysisCancelled:
                    with viewer._lock:
                        if not viewer._closed:
                            self.info.content = (
                                "**Cancelled.** Previous completed result is retained."
                                if self.result is not None
                                else "**Cancelled.** No plane result has completed."
                            )
                except Exception as exc:
                    with viewer._lock:
                        if not viewer._closed:
                            self.info.content = f"**Computation failed:** {exc}"
                finally:
                    finish()

            try:
                viewer._analysis_thread = threading.Thread(target=work, daemon=True)
                viewer._analysis_thread.start()
            except Exception as exc:
                viewer._analysis_thread = None
                finish()
                self.info.content = f"**Computation failed to start:** {exc}"

    def _save_exports(self, result, paths):
        """Write captured artifacts outside the viewer lock, then publish status.

        Callers set _export_running while selecting the result and paths.
        """
        viewer = self.viewer
        try:
            try:
                saved, errors = save_result_artifacts(
                    result, paths, report_format="plane"
                )
            except Exception as exc:
                saved, errors = {}, {kind: str(exc) for kind in paths}
            with viewer._lock:
                if viewer._closed or self.result is not result:
                    return
                self._saved_paths.update(saved)
                for kind in saved:
                    self._export_errors.pop(kind, None)
                self._export_errors.update(errors)
                lines = []
                if self._saved_paths:
                    lines.append(
                        "**Saved last completed run:** "
                        + ", ".join(f"`{path}`" for path in self._saved_paths.values())
                    )
                if self._export_errors:
                    lines.append("**Save failed (calculation is complete):**")
                    lines.extend(
                        f"- `{self._export_paths[kind]}`: {error}"
                        for kind, error in self._export_errors.items()
                    )
                    lines.append(
                        "The completed result remains available through downloads. "
                        "Fix the destination and click Retry saving; "
                        "this does not rerun IK."
                    )
                self.export_info.content = "\n\n".join(lines)
        finally:
            with viewer._lock:
                self._export_running = False
                if not viewer._closed:
                    self.compute_button.disabled = viewer._recompute_running
                    self.retry_save.disabled = viewer._recompute_running or not bool(
                        self._export_errors
                    )

    def retry_exports(self):
        with self.viewer._lock:
            if (
                self.viewer._closed
                or self.viewer._recompute_running
                or self._export_running
                or self.result is None
                or not self._export_errors
            ):
                return
            result = self.result
            paths = {kind: self._export_paths[kind] for kind in self._export_errors}
            self._export_running = True
            self.compute_button.disabled = True
            self.retry_save.disabled = True
            self.export_info.content = "**Saving last completed run…**"
        self._save_exports(result, paths)

    def cancel(self):
        with self.viewer._lock:
            if not self.viewer._closed and self.viewer._analysis_cancel is not None:
                self.viewer._analysis_cancel.set()
                self.info.content = "**Cancelling…**"

    def download(self, event, extension):
        with self.viewer._lock:
            if self.viewer._closed:
                return
            result = self.result
        if result is None or event.client is None:
            return
        if extension == "json":
            content = (
                json.dumps(result.metadata["plane_scan"], indent=2, allow_nan=False)
                + "\n"
            ).encode()
        elif extension == "svg":
            content = plane_svg(result).encode("utf-8")
        else:
            with tempfile.TemporaryDirectory(prefix="workspace-plane-") as directory:
                path = Path(directory) / "plane.npz"
                result.save(path)
                content = path.read_bytes()
        with self.viewer._lock:
            if not self.viewer._closed:
                event.client.send_file_download(f"plane.{extension}", content)
