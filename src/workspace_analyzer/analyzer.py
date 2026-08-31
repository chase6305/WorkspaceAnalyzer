"""Workspace analysis orchestration and metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .kinematics import KinematicsSolver
from .sampling import SamplingConfig, sample


@dataclass(frozen=True)
class WorkspaceConfig:
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    compute_jacobians: bool = True


@dataclass(frozen=True)
class CartesianConfig:
    """Cartesian target sampling and IK reachability settings."""

    bounds: np.ndarray
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    position_only: bool = True
    restarts: int = 4
    reference_pose: np.ndarray = field(default_factory=lambda: np.eye(4))
    reference_joints: np.ndarray | None = None

    def __post_init__(self):
        bounds = np.asarray(self.bounds, dtype=float)
        if bounds.shape != (3, 2) or np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("Cartesian bounds must have shape (3, 2) with min < max")
        pose = np.asarray(self.reference_pose, dtype=float)
        if pose.shape != (4, 4):
            raise ValueError("reference_pose must have shape (4, 4)")
        if self.restarts < 1:
            raise ValueError("restarts must be at least one")
        if self.reference_joints is not None:
            joints = np.asarray(self.reference_joints, dtype=float)
            if joints.ndim != 1:
                raise ValueError("reference_joints must be a one-dimensional vector")


@dataclass
class AnalysisResult:
    points: np.ndarray
    joint_positions: np.ndarray
    manipulability: np.ndarray | None
    metadata: dict
    reachable: np.ndarray | None = None
    residual: np.ndarray | None = None

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            points=self.points,
            joint_positions=self.joint_positions,
            manipulability=self.manipulability,
            reachable=self.reachable,
            residual=self.residual,
            metadata=np.asarray([self.metadata], dtype=object),
        )


class WorkspaceAnalyzer:
    def __init__(self, solver: KinematicsSolver, config: WorkspaceConfig | None = None):
        self.solver, self.config = solver, config or WorkspaceConfig()

    def analyze(self) -> AnalysisResult:
        q = sample(self.solver.joint_limits, self.config.sampling)
        points, scores = [], []
        batch = self.config.sampling.batch_size
        for start in range(0, len(q), batch):
            qb = q[start : start + batch]
            pose = self.solver.forward(qb)
            if hasattr(pose, "detach"):
                points.append(pose[:, :3, 3].detach().cpu().numpy())
            else:
                points.append(pose[:, :3, 3])
            if self.config.compute_jacobians:
                jac = self.solver.jacobian(qb)
                if hasattr(jac, "detach"):
                    import torch

                    singular = torch.linalg.svdvals(jac[:, :3])
                    scores.append(torch.prod(singular, dim=1).detach().cpu().numpy())
                else:
                    singular = np.linalg.svd(jac[:, :3], compute_uv=False)
                    scores.append(np.prod(singular, axis=1))
        result_points = np.concatenate(points)
        manipulability = np.concatenate(scores) if scores else None
        return AnalysisResult(
            result_points,
            q,
            manipulability,
            {
                "robot": self.solver.model.name,
                "base_link": self.solver.base_link,
                "tip_link": self.solver.tip_link,
                "joint_names": self.solver.joint_names,
                "backend": self.solver.backend,
                "device": self.solver.device,
                "samples": len(q),
            },
        )

    def analyze_cartesian(self, config: CartesianConfig) -> AnalysisResult:
        """Sample XYZ targets and classify reachability through batched IK."""
        reference_joints = config.reference_joints
        if reference_joints is not None:
            reference_joints = np.asarray(reference_joints, dtype=float)
            if reference_joints.shape != (self.solver.dof,):
                raise ValueError(
                    f"reference_joints must have shape ({self.solver.dof},)"
                )
        targets_xyz = sample(np.asarray(config.bounds, dtype=float), config.sampling)
        solutions, success_values, residual_values = [], [], []
        batch = config.sampling.batch_size
        for start in range(0, len(targets_xyz), batch):
            xyz = targets_xyz[start : start + batch]
            targets = np.broadcast_to(config.reference_pose, (len(xyz), 4, 4)).copy()
            targets[:, :3, 3] = xyz
            result = self.solver.inverse(
                targets,
                seed=reference_joints,
                position_only=config.position_only,
                restarts=config.restarts,
            )
            solutions.append(_numpy(result.positions))
            success_values.append(_numpy(result.success).astype(bool))
            residual_values.append(_numpy(result.residual))
        reachable = np.concatenate(success_values)
        residual = np.concatenate(residual_values)
        return AnalysisResult(
            points=targets_xyz,
            joint_positions=np.concatenate(solutions),
            manipulability=None,
            metadata={
                "robot": self.solver.model.name,
                "base_link": self.solver.base_link,
                "tip_link": self.solver.tip_link,
                "joint_names": self.solver.joint_names,
                "backend": self.solver.backend,
                "device": self.solver.device,
                "samples": len(targets_xyz),
                "mode": "cartesian",
                "position_only": config.position_only,
                "restarts": config.restarts,
                "reference_joints": (
                    None if reference_joints is None else reference_joints.tolist()
                ),
                "success_rate": float(np.mean(reachable)),
            },
            reachable=reachable,
            residual=residual,
        )


def _numpy(value):
    return (
        value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    )
