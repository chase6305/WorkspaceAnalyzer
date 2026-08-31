"""Automatically constructed batched FK, Jacobian, and damped-least-squares IK."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .model import RobotModel


@dataclass(frozen=True)
class SolverConfig:
    backend: Literal["auto", "numpy", "torch"] = "auto"
    device: str = "auto"
    dtype: Literal["float32", "float64"] = "float64"
    max_iterations: int = 150
    tolerance: float = 1e-5
    damping: float = 1e-3
    step_size: float = 0.8
    max_joint_step: float = 0.35
    random_seed: int = 42


@dataclass
class IKResult:
    positions: object
    success: object
    residual: object
    iterations: int


class KinematicsSolver:
    """Serial-chain solver generated from any URDF chain.

    Inputs may be a single ``(dof,)`` configuration or a batch ``(N, dof)``.
    Torch mode preserves tensors and executes the whole batch on the selected device.
    """

    def __init__(
        self,
        model: RobotModel,
        base_link: str | None = None,
        tip_link: str | None = None,
        config: SolverConfig | None = None,
    ):
        self.model, self.config = model, config or SolverConfig()
        self.chain = model.chain(base_link, tip_link)
        self.active_joints = tuple(j for j in self.chain if j.active)
        self.base_link = base_link or model._default_root()
        self.tip_link = tip_link or self.chain[-1].child
        if not self.active_joints:
            raise ValueError("selected chain has no movable joints")
        backend = self.config.backend
        if backend == "auto":
            try:
                import torch  # noqa: F401

                backend = "torch"
            except ImportError:
                backend = "numpy"
        self.backend = backend
        self.device = "cpu"
        if backend == "torch":
            import torch

            self.device = (
                ("cuda" if torch.cuda.is_available() else "cpu")
                if self.config.device == "auto"
                else self.config.device
            )
            if self.device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA was requested but torch.cuda.is_available() is false"
                )
            dtype = torch.float64 if self.config.dtype == "float64" else torch.float32
            self._torch_chain = tuple(
                (
                    torch.as_tensor(joint.origin, dtype=dtype, device=self.device),
                    torch.as_tensor(joint.axis, dtype=dtype, device=self.device),
                )
                for joint in self.chain
            )

    @property
    def dof(self) -> int:
        return len(self.active_joints)

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(j.name for j in self.active_joints)

    @property
    def joint_limits(self) -> np.ndarray:
        return np.asarray([[j.limit.lower, j.limit.upper] for j in self.active_joints])

    def _array(self, value):
        if self.backend == "numpy":
            return np.asarray(value, dtype=self.config.dtype)
        import torch

        dtype = torch.float64 if self.config.dtype == "float64" else torch.float32
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    def forward(self, q, *, all_links: bool = False):
        x = self._array(q)
        single = x.ndim == 1
        if single:
            x = x[None, :]
        if x.shape[-1] != self.dof:
            raise ValueError(f"expected {self.dof} joints, got {x.shape[-1]}")
        if self.backend == "numpy":
            result = self._forward_numpy(x, all_links)
        else:
            result = self._forward_torch(x, all_links)
        if all_links:
            return {k: v[0] if single else v for k, v in result.items()}
        return result[0] if single else result

    def _forward_numpy(self, q, all_links):
        n, active = len(q), 0
        transform = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
        links = {self.base_link: transform.copy()}
        for joint in self.chain:
            transform = transform @ joint.origin
            if joint.active:
                motion = _motion_numpy(joint.kind, joint.axis, q[:, active])
                transform = transform @ motion
                active += 1
            links[joint.child] = transform.copy()
        return links if all_links else transform

    def _forward_torch(self, q, all_links):
        import torch

        n, active = len(q), 0
        transform = torch.eye(4, dtype=q.dtype, device=q.device).expand(n, 4, 4).clone()
        links = {self.base_link: transform.clone()}
        for joint, (origin, axis) in zip(self.chain, self._torch_chain):
            transform = transform @ origin
            if joint.active:
                transform = transform @ _motion_torch(joint.kind, axis, q[:, active], q)
                active += 1
            links[joint.child] = transform.clone()
        return links if all_links else transform

    def jacobian(self, q):
        """Return geometric Jacobian(s), shaped ``(..., 6, dof)``."""
        x = self._array(q)
        single = x.ndim == 1
        if single:
            x = x[None, :]
        result = (
            _geometric_jacobian_torch(self, x)
            if self.backend == "torch"
            else _geometric_jacobian_numpy(self, x)
        )
        return result[0] if single else result

    def inverse(
        self, target, seed=None, *, position_only: bool = False, restarts: int = 1
    ) -> IKResult:
        """Solve targets concurrently, optionally choosing the best of several seeds."""
        if restarts < 1:
            raise ValueError("restarts must be at least one")
        goal = self._array(target)
        if goal.ndim == 2:
            goal = goal[None, ...]
        if goal.shape[-2:] != (4, 4):
            raise ValueError("target must have shape (..., 4, 4)")
        n = len(goal)
        initial = (
            np.tile(self.joint_limits.mean(axis=1), (n, 1)) if seed is None else seed
        )
        q = self._array(initial)
        if q.ndim == 1:
            q = q[None, :]
        if len(q) == 1 and n > 1:
            q = q.repeat(n, axis=0) if self.backend == "numpy" else q.repeat(n, 1)
        if q.shape != (n, self.dof):
            raise ValueError(f"seed must have shape ({n}, {self.dof}) or ({self.dof},)")
        if restarts > 1:
            rng = np.random.default_rng(self.config.random_seed)
            extra = rng.uniform(
                self.joint_limits[:, 0],
                self.joint_limits[:, 1],
                size=((restarts - 1) * n, self.dof),
            )
            q = _concat_rows(q, self._array(extra), self.backend)
            goal = _repeat_rows(goal, restarts, self.backend)
        limits = self._array(self.joint_limits)
        success = None
        for iteration in range(1, self.config.max_iterations + 1):
            current = self.forward(q)
            error = _pose_error(current, goal, self.backend, position_only)
            residual = _norm(error, self.backend)
            success = residual <= self.config.tolerance
            if bool(success.all()):
                break
            jac = self.jacobian(q)
            if position_only:
                jac = jac[:, :3]
            dq = _dls(jac, error, self.config.damping, self.backend)
            dq = _clip(
                dq,
                -self.config.max_joint_step,
                self.config.max_joint_step,
                self.backend,
            )
            dq = _zero_successful(dq, success, self.backend)
            q = _clip(
                q + self.config.step_size * dq, limits[:, 0], limits[:, 1], self.backend
            )
        if restarts > 1:
            q, success, residual = _select_best(
                q, success, residual, restarts, n, self.backend
            )
        return IKResult(
            q[0] if n == 1 else q,
            success[0] if n == 1 else success,
            residual[0] if n == 1 else residual,
            iteration,
        )


def create_solver(
    urdf: str,
    *,
    base_link: str | None = None,
    tip_link: str | None = None,
    backend: str = "auto",
    device: str = "auto",
    **kwargs,
) -> KinematicsSolver:
    """Parse a URDF, select its longest chain, and construct the best backend."""
    return KinematicsSolver(
        RobotModel.from_urdf(urdf),
        base_link,
        tip_link,
        SolverConfig(backend=backend, device=device, **kwargs),
    )


def _motion_numpy(kind, axis, values):
    n = len(values)
    out = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
    if kind == "prismatic":
        out[:, :3, 3] = values[:, None] * axis
    else:
        a = np.broadcast_to(axis, (n, 3))
        c, s = np.cos(values), np.sin(values)
        cross = np.zeros((n, 3, 3))
        cross[:, 0, 1], cross[:, 0, 2] = -a[:, 2], a[:, 1]
        cross[:, 1, 0], cross[:, 1, 2] = a[:, 2], -a[:, 0]
        cross[:, 2, 0], cross[:, 2, 1] = -a[:, 1], a[:, 0]
        out[:, :3, :3] = (
            np.eye(3)
            + s[:, None, None] * cross
            + (1 - c)[:, None, None] * (cross @ cross)
        )
    return out


def _motion_torch(kind, axis, values, like):
    import torch

    n = len(values)
    out = torch.eye(4, dtype=like.dtype, device=like.device).expand(n, 4, 4).clone()
    a = torch.as_tensor(axis, dtype=like.dtype, device=like.device).expand(n, 3)
    if kind == "prismatic":
        out[:, :3, 3] = values[:, None] * a
    else:
        cross = torch.zeros((n, 3, 3), dtype=like.dtype, device=like.device)
        cross[:, 0, 1], cross[:, 0, 2] = -a[:, 2], a[:, 1]
        cross[:, 1, 0], cross[:, 1, 2] = a[:, 2], -a[:, 0]
        cross[:, 2, 0], cross[:, 2, 1] = -a[:, 1], a[:, 0]
        out[:, :3, :3] = (
            torch.eye(3, dtype=like.dtype, device=like.device)
            + torch.sin(values)[:, None, None] * cross
            + (1 - torch.cos(values))[:, None, None] * (cross @ cross)
        )
    return out


def _geometric_jacobian_numpy(solver, q):
    n = len(q)
    t = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
    origins = []
    axes = []
    kinds = []
    active = 0
    for joint in solver.chain:
        t = t @ joint.origin
        if joint.active:
            origins.append(t[:, :3, 3].copy())
            axes.append(t[:, :3, :3] @ joint.axis)
            kinds.append(joint.kind)
            t = t @ _motion_numpy(joint.kind, joint.axis, q[:, active])
            active += 1
    tip = t[:, :3, 3]
    j = np.zeros((n, 6, solver.dof))
    for i, (origin, axis, kind) in enumerate(zip(origins, axes, kinds)):
        if kind == "prismatic":
            j[:, :3, i] = axis
        else:
            j[:, :3, i] = np.cross(axis, tip - origin)
            j[:, 3:, i] = axis
    return j


def _geometric_jacobian_torch(solver, q):
    import torch

    n = len(q)
    t = torch.eye(4, dtype=q.dtype, device=q.device).expand(n, 4, 4).clone()
    origins, axes, kinds = [], [], []
    active = 0
    for joint, (origin, local_axis) in zip(solver.chain, solver._torch_chain):
        t = t @ origin
        if joint.active:
            axis = t[:, :3, :3] @ local_axis
            origins.append(t[:, :3, 3].clone())
            axes.append(axis)
            kinds.append(joint.kind)
            t = t @ _motion_torch(joint.kind, local_axis, q[:, active], q)
            active += 1
    tip = t[:, :3, 3]
    columns = []
    for origin, axis, kind in zip(origins, axes, kinds):
        if kind == "prismatic":
            columns.append(torch.cat((axis, torch.zeros_like(axis)), dim=1))
        else:
            linear = torch.linalg.cross(axis, tip - origin, dim=1)
            columns.append(torch.cat((linear, axis), dim=1))
    return torch.stack(columns, dim=2)


def _pose_error(current, target, backend, position_only):
    pos = target[:, :3, 3] - current[:, :3, 3]
    if position_only:
        return pos
    relative = target[:, :3, :3] @ current[:, :3, :3].swapaxes(1, 2)
    rot = _quaternion_rotation_error(relative, backend)
    return _concat(pos, rot, backend)


def _quaternion_rotation_error(rotation, backend):
    """Continuous shortest-path error, including rotations close to 180 degrees."""
    if backend == "numpy":
        diagonal = np.diagonal(rotation, axis1=1, axis2=2)
        xyz = 0.5 * np.sqrt(
            np.maximum(
                0.0,
                np.stack(
                    (
                        1 + diagonal[:, 0] - diagonal[:, 1] - diagonal[:, 2],
                        1 - diagonal[:, 0] + diagonal[:, 1] - diagonal[:, 2],
                        1 - diagonal[:, 0] - diagonal[:, 1] + diagonal[:, 2],
                    ),
                    axis=1,
                ),
            )
        )
        signs = np.stack(
            (
                rotation[:, 2, 1] - rotation[:, 1, 2],
                rotation[:, 0, 2] - rotation[:, 2, 0],
                rotation[:, 1, 0] - rotation[:, 0, 1],
            ),
            axis=1,
        )
        xyz = np.copysign(xyz, np.where(signs == 0.0, 1.0, signs))
        w = 0.5 * np.sqrt(np.maximum(0.0, 1.0 + diagonal.sum(axis=1)))
        return 2.0 * xyz * np.where(w[:, None] < 0.0, -1.0, 1.0)
    import torch

    diagonal = torch.diagonal(rotation, dim1=1, dim2=2)
    xyz = 0.5 * torch.sqrt(
        torch.clamp(
            torch.stack(
                (
                    1 + diagonal[:, 0] - diagonal[:, 1] - diagonal[:, 2],
                    1 - diagonal[:, 0] + diagonal[:, 1] - diagonal[:, 2],
                    1 - diagonal[:, 0] - diagonal[:, 1] + diagonal[:, 2],
                ),
                dim=1,
            ),
            min=0.0,
        )
    )
    signs = torch.stack(
        (
            rotation[:, 2, 1] - rotation[:, 1, 2],
            rotation[:, 0, 2] - rotation[:, 2, 0],
            rotation[:, 1, 0] - rotation[:, 0, 1],
        ),
        dim=1,
    )
    xyz = torch.copysign(xyz, torch.where(signs == 0, torch.ones_like(signs), signs))
    return 2.0 * xyz


def _concat(a, b, backend):
    if backend == "numpy":
        return np.concatenate((a, b), axis=1)
    import torch

    return torch.cat((a, b), dim=1)


def _concat_rows(a, b, backend):
    if backend == "numpy":
        return np.concatenate((a, b), axis=0)
    import torch

    return torch.cat((a, b), dim=0)


def _repeat_rows(value, count, backend):
    if backend == "numpy":
        return np.tile(value, (count, 1, 1))
    return value.repeat(count, 1, 1)


def _select_best(q, success, residual, restarts, targets, backend):
    shaped_residual = residual.reshape(restarts, targets)
    if backend == "numpy":
        indices = np.argmin(shaped_residual, axis=0)
        columns = np.arange(targets)
        return (
            q.reshape(restarts, targets, -1)[indices, columns],
            success.reshape(restarts, targets)[indices, columns],
            shaped_residual[indices, columns],
        )
    import torch

    indices = torch.argmin(shaped_residual, dim=0)
    columns = torch.arange(targets, device=q.device)
    return (
        q.reshape(restarts, targets, -1)[indices, columns],
        success.reshape(restarts, targets)[indices, columns],
        shaped_residual[indices, columns],
    )


def _norm(x, backend):
    if backend == "numpy":
        return np.linalg.norm(x, axis=1)
    import torch

    return torch.linalg.vector_norm(x, dim=1)


def _dls(j, e, damping, backend):
    jt = j.swapaxes(1, 2)
    size = j.shape[1]
    if backend == "numpy":
        eye = np.eye(size)[None]
        return (jt @ np.linalg.solve(j @ jt + damping * damping * eye, e[..., None]))[
            ..., 0
        ]
    import torch

    eye = torch.eye(size, dtype=j.dtype, device=j.device)[None]
    return (jt @ torch.linalg.solve(j @ jt + damping * damping * eye, e[..., None]))[
        ..., 0
    ]


def _clip(q, lo, hi, backend):
    return np.clip(q, lo, hi) if backend == "numpy" else q.clamp(lo, hi)


def _zero_successful(delta, success, backend):
    if backend == "numpy":
        return np.where(success[:, None], 0.0, delta)
    return delta.masked_fill(success[:, None], 0.0)
