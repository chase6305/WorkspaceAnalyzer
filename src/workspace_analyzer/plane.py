"""Plane sampling shared by the CLI, browser controls and evaluation demos."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from ._arrays import as_numpy as _numpy
from ._atomic import atomic_write_text
from ._cancellation import check_cancelled
from ._validation import require_integer
from .model import _rpy_matrix
from .reachability import ReachabilityConfig, analyze_targets
from .sampling import SamplingConfig, SamplingStrategy, sample

PLANE_AXES = {"xy": (0, 1, 2), "xz": (0, 2, 1), "yz": (1, 2, 0)}


@dataclass(frozen=True)
class PlaneConfig:
    """Square region in a fixed frame; constant optionally overrides the center.

    Without sampling, resolution sets points per axis, including both endpoints.
    With sampling, its count and strategy replace the regular resolution grid.
    Named-link and reference frames use the robot's zero configuration.
    """

    plane: str = "xy"
    frame: str = "base"
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    constant: float | None = None
    span: float = 0.8
    resolution: int = 41
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    sampling: SamplingConfig | None = None
    frame_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self):
        if self.plane not in PLANE_AXES:
            raise ValueError("plane must be xy, xz, or yz")
        if not isinstance(self.frame, str) or not self.frame:
            raise ValueError("frame must be base, world, reference, or a link name")
        require_integer(self.resolution, "resolution", minimum=2)
        object.__setattr__(self, "resolution", int(self.resolution))
        if not np.isfinite(self.span) or self.span <= 0:
            raise ValueError("span must be finite and positive")
        object.__setattr__(self, "span", float(self.span))
        for name in ("center", "rpy", "frame_rpy"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError(f"{name} must contain three finite values")
            object.__setattr__(self, name, tuple(float(x) for x in value))
        if self.constant is not None and not np.isfinite(self.constant):
            raise ValueError("constant must be finite")
        if self.sampling is not None and not isinstance(self.sampling, SamplingConfig):
            raise ValueError("sampling must be a SamplingConfig")
        bounds = self.bounds
        if not np.isfinite(bounds).all():
            raise ValueError("plane bounds must remain finite")
        plane_bounds = bounds[list(PLANE_AXES[self.plane][:2])]
        if np.any(plane_bounds[:, 0] >= plane_bounds[:, 1]):
            raise ValueError("span is too small for the plane center's precision")

    @property
    def effective_center(self):
        center = np.array(self.center)
        if self.constant is not None:
            center[PLANE_AXES[self.plane][2]] = self.constant
        return center

    @property
    def bounds(self):
        bounds = np.repeat(self.effective_center[:, None], 2, axis=1)
        with np.errstate(over="ignore"):
            bounds[list(PLANE_AXES[self.plane][:2])] += [-self.span / 2, self.span / 2]
        return bounds

    def points(self):
        if self.sampling is not None:
            return sample(self.bounds, self.sampling, allow_fixed=True)
        u, v, _ = PLANE_AXES[self.plane]
        values = np.linspace(-self.span / 2, self.span / 2, self.resolution)
        points = np.tile(self.effective_center, (self.resolution**2, 1))
        points[:, u] += np.repeat(values, self.resolution)
        points[:, v] += np.tile(values, self.resolution)
        return points

    @property
    def num_samples(self):
        return (
            self.resolution**2 if self.sampling is None else self.sampling.num_samples
        )

    def corners(self):
        u, v, _ = PLANE_AXES[self.plane]
        corners = np.tile(self.effective_center, (4, 1))
        corners[:, [u, v]] += (
            np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * self.span / 2
        )
        return corners


def plane_frame(solver, frame, rpy=(0.0, 0.0, 0.0)):
    """Return base_from_sampling_frame, with optional local RPY rotation.

    Rotation is applied at the selected frame's origin. Plane center and TCP RPY
    are expressed in the resulting rotated sampling frame.
    """
    angles = np.asarray(rpy, dtype=float)
    if angles.shape != (3,) or not np.isfinite(angles).all():
        raise ValueError("frame_rpy must contain three finite values")
    local_rotation = _rpy_matrix(angles)
    if frame == "base":
        return solver.base_link, local_rotation
    transforms = solver.model.zero_transforms()
    if frame == "reference":
        active = solver.active_joints
        name = active[min(1, len(active) - 1)].child
    else:
        name = frame
    if name != "world" and name not in transforms:
        raise ValueError(
            f"unknown sampling frame {frame!r}; "
            "use base, world, reference, or a URDF link"
        )
    world_from_frame = np.eye(4) if name == "world" else transforms[name]
    return name, np.linalg.inv(
        transforms[solver.base_link]
    ) @ world_from_frame @ local_rotation


def analyze_plane(
    solver,
    plane: PlaneConfig,
    config=None,
    *,
    seed=None,
    cache=None,
    cancel_event=None,
    progress_callback=None,
):
    check_cancelled(cancel_event)
    config = config or ReachabilityConfig()
    frame_name, transform = plane_frame(solver, plane.frame, plane.frame_rpy)
    points = plane.points()
    targets = np.broadcast_to(
        _rpy_matrix(np.asarray(plane.rpy)), (len(points), 4, 4)
    ).copy()
    targets[:, :3, 3] = points
    result = analyze_targets(
        solver,
        targets,
        config,
        seed=seed,
        base_from_targets=transform,
        cache=cache,
        cancel_event=cancel_event,
        progress_callback=progress_callback,
    )
    check_cancelled(cancel_event)
    u, v, normal = PLANE_AXES[plane.plane]
    report = {
        "plane": plane.plane,
        "frame": plane.frame,
        "coordinate_frame": frame_name,
        "frame_rpy_rad": list(plane.frame_rpy),
        "plane_center": plane.effective_center.tolist(),
        "base_from_plane_frame": transform.tolist(),
        "solver_base_link": solver.base_link,
        "solver_tip_link": solver.tip_link,
        "constant_coordinate_m": float(plane.effective_center[normal]),
        "span_m": plane.span,
        "resolution": plane.resolution if plane.sampling is None else None,
        "sampling_strategy": "grid"
        if plane.sampling is None
        else SamplingStrategy(plane.sampling.strategy).value,
        "sampling_seed": None if plane.sampling is None else plane.sampling.seed,
        "sampling_bounds_m": plane.bounds.tolist(),
        "batch_size": config.batch_size,
        "seed": config.random_seed,
        "ik_restarts": config.restarts,
        "ik_rescue_restarts": config.rescue_restarts,
        "ik_rescue_rounds": config.rescue_rounds,
        "ik_max_iterations": solver.config.max_iterations,
        "ik_tolerance": solver.config.tolerance,
        "reference_joints": None if seed is None else _numpy(seed).tolist(),
        "backend": solver.backend,
        "device": solver.device,
        "dtype": solver.config.dtype,
        "target_rpy_rad": list(plane.rpy),
        "task": "position_only" if config.position_only else "fixed_orientation",
        "reachable": int(result.reachable.sum()),
        "samples": len(points),
        "success_rate": float(result.reachable.mean()),
        "row_axis": "xyz"[u],
        "column_axis": "xyz"[v],
        "row_coordinates_m": None
        if plane.sampling is not None
        else np.linspace(*plane.bounds[u], plane.resolution).tolist(),
        "column_coordinates_m": None
        if plane.sampling is not None
        else np.linspace(*plane.bounds[v], plane.resolution).tolist(),
        "grid": None
        if plane.sampling is not None
        else result.reachable.reshape(plane.resolution, plane.resolution).tolist(),
        "assessment": result.metadata["assessment"],
        "quality_thresholds": result.metadata["quality_thresholds"],
        "dexterity_task": config.dexterity_task,
    }
    if plane.frame == "reference":
        report.update(
            reference_link=frame_name,
            plane_center_link2=plane.effective_center.tolist(),
        )
    result.metadata.update(
        sampling_mode="plane",
        plane=plane.plane,
        sampling_frame=frame_name,
        plane_scan=report,
    )
    return result


def save_plane_report(result, path):
    atomic_write_text(
        path,
        json.dumps(result.metadata["plane_scan"], indent=2, allow_nan=False) + "\n",
    )
