"""Workspace analysis orchestration and metrics."""

from __future__ import annotations

import json
import os
import tempfile
import zlib
from dataclasses import dataclass, field, replace
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

from ._cancellation import AnalysisCancelled as AnalysisCancelled
from ._cancellation import check_cancelled
from ._validation import require_integer
from .kinematics import (
    KinematicsSolver,
    _validate_homogeneous_rows,
    _validate_rotation_matrices,
)
from .metrics import _joint_limit_margin_numpy
from .sampling import SamplingConfig, SamplingStrategy, sample


@dataclass(frozen=True)
class WorkspaceConfig:
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    compute_jacobians: bool = True


@dataclass(frozen=True)
class CartesianConfig:
    """Cartesian target sampling and IK settings; equal bounds fix an axis."""

    bounds: np.ndarray
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    position_only: bool = True
    restarts: int = 4
    rescue_restarts: int = 16
    rescue_rounds: int = 3
    reference_pose: np.ndarray = field(default_factory=lambda: np.eye(4))
    reference_joints: np.ndarray | None = None

    def __post_init__(self):
        bounds = np.asarray(self.bounds, dtype=float)
        if (
            bounds.shape != (3, 2)
            or not np.isfinite(bounds).all()
            or np.any(bounds[:, 0] > bounds[:, 1])
        ):
            raise ValueError("Cartesian bounds must have shape (3, 2) with min <= max")
        original_pose = np.asarray(self.reference_pose)
        pose = np.array(
            original_pose,
            dtype=(original_pose.dtype if original_pose.dtype.kind == "f" else float),
            copy=True,
        )
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("reference_pose must have shape (4, 4)")
        _validate_homogeneous_rows(pose[None], "numpy")
        if not self.position_only:
            # Respect the precision of reference poses produced by float32 FK.
            rotation = pose[:3, :3]
            _validate_rotation_matrices(rotation[None], "numpy")
        require_integer(self.restarts, "restarts", minimum=1)
        require_integer(self.rescue_restarts, "rescue_restarts")
        require_integer(self.rescue_rounds, "rescue_rounds")
        if self.reference_joints is not None:
            joints = np.array(self.reference_joints, dtype=float, copy=True)
            if joints.ndim != 1 or not np.isfinite(joints).all():
                raise ValueError("reference_joints must be a one-dimensional vector")
            joints.setflags(write=False)
            object.__setattr__(self, "reference_joints", joints)
        bounds = bounds.copy()
        bounds.setflags(write=False)
        pose.setflags(write=False)
        object.__setattr__(self, "bounds", bounds)
        object.__setattr__(self, "reference_pose", pose)


@dataclass
class AnalysisResult:
    points: np.ndarray
    joint_positions: np.ndarray
    manipulability: np.ndarray | None
    metadata: dict
    reachable: np.ndarray | None = None
    residual: np.ndarray | None = None
    metrics: dict[str, np.ndarray] | None = None
    target_poses: np.ndarray | None = None

    FORMAT_VERSION = 1

    def reassess_quality(self, **kwargs) -> AnalysisResult:
        """Apply new quality gates/task weights without rerunning kinematics."""
        from .reachability import reassess_quality

        return reassess_quality(self, **kwargs)

    def orientation_coverage(
        self, position_ids, *, position_tolerance=1e-8, cancel_event=None
    ):
        """Summarize sampled full-pose coverage for each supplied position ID."""
        from .coverage import summarize_orientation_coverage

        return summarize_orientation_coverage(
            self,
            position_ids,
            position_tolerance=position_tolerance,
            cancel_event=cancel_event,
        )

    def __post_init__(self):
        points = self.points = _numeric_array(self.points, "points")
        joints = self.joint_positions = _numeric_array(
            self.joint_positions, "joint_positions"
        )
        if not isinstance(self.metadata, dict):
            raise ValueError("result metadata must be a dictionary")
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("result points must have shape (N, 3)")
        if joints.ndim != 2 or len(joints) != len(points):
            raise ValueError("result joint_positions must have shape (N, DoF)")
        if self.target_poses is not None:
            self.target_poses = _numeric_array(self.target_poses, "target_poses")
            if self.target_poses.shape != (len(points), 4, 4):
                raise ValueError("result target_poses must have shape (N, 4, 4)")
        for name in ("manipulability", "reachable", "residual"):
            value = getattr(self, name)
            if value is not None:
                value = _numeric_array(value, name)
                if value.shape != (len(points),):
                    raise ValueError(f"result {name} must have shape (N,)")
                if name == "reachable":
                    if value.dtype.kind != "b" and not np.isin(value, [0, 1]).all():
                        raise ValueError("result reachable must contain Boolean flags")
                    value = value.astype(bool, copy=False)
                setattr(self, name, value)
        if self.metrics is not None:
            self.metrics = {
                name: _numeric_array(value, f"metric {name!r}")
                for name, value in self.metrics.items()
            }
            for name, value in self.metrics.items():
                if value.shape != (len(points),):
                    raise ValueError(f"result metric {name!r} must have shape (N,)")

    def save(self, path: str | Path, *, compressed: bool = True) -> None:
        """Atomically save validated numeric arrays; optionally skip compression."""
        if not isinstance(compressed, bool):
            raise ValueError("compressed must be Boolean")
        # Public arrays may have changed after construction. Validate a shallow
        # snapshot before replacing an existing archive, without mutating self.
        snapshot = replace(self)
        target = Path(path).with_suffix(".npz")
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": np.asarray(self.FORMAT_VERSION, dtype=np.int64),
            "points": snapshot.points,
            "joint_positions": snapshot.joint_positions,
            "metadata_json": np.asarray(json.dumps(snapshot.metadata)),
        }
        for name in ("manipulability", "reachable", "residual", "target_poses"):
            value = getattr(snapshot, name)
            if value is not None:
                payload[name] = np.asarray(value)
        for name, value in (snapshot.metrics or {}).items():
            if not isinstance(name, str) or not name.replace("_", "").isalnum():
                raise ValueError(f"invalid metric name {name!r}")
            payload[f"metric__{name}"] = np.asarray(value)
        with tempfile.NamedTemporaryFile(
            dir=target.parent,
            prefix=f".{target.stem}-",
            suffix=".npz",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            writer = np.savez_compressed if compressed else np.savez
            writer(temporary_path, **payload)
            os.replace(temporary_path, target)
        finally:
            temporary_path.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | Path) -> AnalysisResult:
        """Load a versioned result without enabling pickle."""
        try:
            return cls._load_archive(path)
        except (BadZipFile, EOFError, KeyError, zlib.error) as exc:
            raise ValueError(f"invalid result archive: {exc}") from exc

    @classmethod
    def _load_archive(cls, path: str | Path) -> AnalysisResult:
        archive = np.load(Path(path), allow_pickle=False)
        if isinstance(archive, np.ndarray):
            raise ValueError("result file must be an NPZ archive, not an NPY array")
        with archive:
            version_array = archive["format_version"]
            if version_array.shape != () or version_array.dtype.kind not in "iu":
                raise ValueError("result format_version must be an integer scalar")
            version = int(version_array)
            if version != cls.FORMAT_VERSION:
                raise ValueError(
                    f"unsupported result format {version}; "
                    f"expected {cls.FORMAT_VERSION}"
                )
            return cls(
                points=archive["points"],
                joint_positions=archive["joint_positions"],
                manipulability=_optional_array(archive, "manipulability"),
                metadata=json.loads(str(archive["metadata_json"])),
                reachable=_optional_array(archive, "reachable"),
                residual=_optional_array(archive, "residual"),
                target_poses=_optional_array(archive, "target_poses"),
                metrics={
                    name[len("metric__") :]: archive[name]
                    for name in archive.files
                    if name.startswith("metric__")
                }
                or None,
            )


class WorkspaceAnalyzer:
    def __init__(self, solver: KinematicsSolver, config: WorkspaceConfig | None = None):
        self.solver, self.config = solver, config or WorkspaceConfig()

    def analyze_targets(
        self,
        targets,
        config=None,
        *,
        seed=None,
        weights=None,
        base_from_targets=None,
        cache=None,
        cancel_event=None,
        progress_callback=None,
    ) -> AnalysisResult:
        """Assess supplied points/poses, keeping IK and quality acceptance separate.

        See ``ReachabilityConfig`` for IK budgets and dexterity thresholds.
        Targets and optional non-negative task weights preserve input order.
        ``base_from_targets`` maps the input frame into this solver's base.
        """
        from .reachability import analyze_targets

        return analyze_targets(
            self.solver,
            targets,
            config,
            seed=seed,
            weights=weights,
            base_from_targets=base_from_targets,
            cache=cache,
            cancel_event=cancel_event,
            progress_callback=progress_callback,
        )

    def analyze(
        self, cache=None, cancel_event=None, progress_callback=None
    ) -> AnalysisResult:
        """Sample joints with optional per-batch progress and cancellation."""
        check_cancelled(cancel_event)
        if cache is not None:
            from .cache import analysis_cache_key

            key = analysis_cache_key(self.solver, "joint", self.config)
            cached = cache.get(key)
            if cached is not None:
                cached.metadata["cache_hit"] = True
                if progress_callback is not None:
                    progress_callback(1.0)
                check_cancelled(cancel_event)
                return cached
        result = self._analyze_joint(cancel_event, progress_callback)
        check_cancelled(cancel_event)
        if cache is not None:
            result.metadata["cache_hit"] = False
            cache.put(key, result)
        return result

    def _analyze_joint(
        self, cancel_event=None, progress_callback=None
    ) -> AnalysisResult:
        q = sample(self.solver.joint_limits, self.config.sampling)
        points = np.empty((len(q), 3), dtype=self.solver.config.dtype)
        scores = minimum_values = isotropy_values = None
        if self.config.compute_jacobians:
            scores = np.empty(len(q), dtype=self.solver.config.dtype)
            minimum_values = np.empty_like(scores)
            isotropy_values = np.empty_like(scores)
            margin_values = np.empty(len(q), dtype=q.dtype)
        batch = self.config.sampling.batch_size
        for start in range(0, len(q), batch):
            check_cancelled(cancel_event)
            qb = q[start : start + batch]
            selection = slice(start, start + len(qb))
            if self.config.compute_jacobians:
                pose, jac = self.solver.forward_with_jacobian(qb)
            else:
                pose = self.solver.forward(qb)
            points[selection] = _numpy(pose[:, :3, 3])
            if self.config.compute_jacobians:
                margin_values[selection] = _joint_limit_margin_numpy(self.solver, qb)
                if hasattr(jac, "detach"):
                    import torch

                    singular = torch.linalg.svdvals(jac[:, :3])
                    values = _numpy(
                        torch.stack(
                            (
                                torch.prod(singular, dim=1),
                                singular[:, -1],
                                singular[:, -1] / singular[:, 0].clamp_min(1e-30),
                            ),
                            dim=1,
                        )
                    )
                    scores[selection] = values[:, 0]
                    minimum_values[selection] = values[:, 1]
                    isotropy_values[selection] = values[:, 2]
                else:
                    singular = np.linalg.svd(jac[:, :3], compute_uv=False)
                    scores[selection] = np.prod(singular, axis=1)
                    minimum_values[selection] = singular[:, -1]
                    isotropy_values[selection] = np.divide(
                        singular[:, -1],
                        singular[:, 0],
                        out=np.zeros_like(singular[:, 0]),
                        where=singular[:, 0] > 0,
                    )
            check_cancelled(cancel_event)
            if progress_callback is not None:
                progress_callback((start + len(qb)) / len(q))
        check_cancelled(cancel_event)
        metrics = None
        if scores is not None:
            metrics = {
                "minimum_singular_value": minimum_values,
                "isotropy": isotropy_values,
                "joint_limit_margin": margin_values,
            }
        return AnalysisResult(
            points,
            q,
            scores,
            {
                "robot": self.solver.model.name,
                "base_link": self.solver.base_link,
                "tip_link": self.solver.tip_link,
                "joint_names": self.solver.joint_names,
                "backend": self.solver.backend,
                "device": self.solver.device,
                "samples": len(q),
                "sampling_strategy": SamplingStrategy(
                    self.config.sampling.strategy
                ).value,
                "sampling_seed": self.config.sampling.seed,
                "batch_size": self.config.sampling.batch_size,
                "compute_jacobians": self.config.compute_jacobians,
            },
            metrics=metrics,
        )

    def analyze_cartesian(
        self,
        config: CartesianConfig,
        cache=None,
        cancel_event=None,
        progress_callback=None,
    ) -> AnalysisResult:
        """Sample XYZ targets and classify reachability through batched IK."""
        check_cancelled(cancel_event)
        if cache is not None:
            from .cache import analysis_cache_key

            key = analysis_cache_key(self.solver, "cartesian", config)
            cached = cache.get(key)
            if cached is not None:
                cached.metadata["cache_hit"] = True
                if progress_callback is not None:
                    progress_callback(1.0)
                check_cancelled(cancel_event)
                return cached
        reference_joints = config.reference_joints
        if reference_joints is not None:
            reference_joints = np.array(reference_joints, dtype=float, copy=True)
            if reference_joints.shape != (self.solver.dof,):
                raise ValueError(
                    f"reference_joints must have shape ({self.solver.dof},)"
                )
        targets_xyz = sample(config.bounds, config.sampling, allow_fixed=True)
        solutions = np.empty(
            (len(targets_xyz), self.solver.dof), dtype=self.solver.config.dtype
        )
        reachable = np.empty(len(targets_xyz), dtype=bool)
        residual = np.empty(len(targets_xyz), dtype=self.solver.config.dtype)
        batch = config.sampling.batch_size
        for start in range(0, len(targets_xyz), batch):
            check_cancelled(cancel_event)
            xyz = targets_xyz[start : start + batch]
            targets = np.broadcast_to(config.reference_pose, (len(xyz), 4, 4)).copy()
            targets[:, :3, 3] = xyz
            result = self.solver.inverse(
                targets,
                seed=reference_joints,
                position_only=config.position_only,
                restarts=config.restarts,
                rescue_restarts=config.rescue_restarts,
                rescue_rounds=config.rescue_rounds,
                cancel_event=cancel_event,
            )
            check_cancelled(cancel_event)
            selection = slice(start, start + len(xyz))
            solutions[selection] = _numpy(result.positions)
            reachable[selection] = _numpy(result.success)
            residual[selection] = _numpy(result.residual)
            if progress_callback is not None:
                progress_callback(min(1.0, (start + len(xyz)) / len(targets_xyz)))
        check_cancelled(cancel_event)
        residual_summary = _residual_summary(residual, reachable)
        result = AnalysisResult(
            points=targets_xyz,
            joint_positions=solutions,
            manipulability=None,
            metadata={
                "robot": self.solver.model.name,
                "base_link": self.solver.base_link,
                "tip_link": self.solver.tip_link,
                "joint_names": self.solver.joint_names,
                "backend": self.solver.backend,
                "device": self.solver.device,
                "samples": len(targets_xyz),
                "bounds": np.asarray(config.bounds).tolist(),
                "sampling_strategy": SamplingStrategy(config.sampling.strategy).value,
                "sampling_seed": config.sampling.seed,
                "batch_size": config.sampling.batch_size,
                "mode": "cartesian",
                "position_only": config.position_only,
                "restarts": config.restarts,
                "rescue_restarts": config.rescue_restarts,
                "rescue_rounds": config.rescue_rounds,
                "reference_joints": (
                    None if reference_joints is None else reference_joints.tolist()
                ),
                "reference_pose": np.asarray(config.reference_pose).tolist(),
                "success_rate": float(np.mean(reachable)),
                "failure_count": int(np.count_nonzero(~reachable)),
                "residual_summary": residual_summary,
            },
            reachable=reachable,
            residual=residual,
        )
        if cache is not None:
            result.metadata["cache_hit"] = False
            cache.put(key, result)
        return result


def _numpy(value):
    return (
        value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    )


def _optional_array(archive, name: str):
    return archive[name] if name in archive.files else None


def _numeric_array(value, name: str):
    array = np.asarray(value)
    if array.dtype.kind not in "biuf":
        raise ValueError(f"result {name} must contain real numeric values")
    return array


def _residual_summary(residual: np.ndarray, reachable: np.ndarray) -> dict:
    def stats(values):
        if not len(values):
            return None
        return {
            "min": float(np.min(values)),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
        }

    return {
        "all": stats(residual),
        "reachable": stats(residual[reachable]),
        "unreachable": stats(residual[~reachable]),
    }
