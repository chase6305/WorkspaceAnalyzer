"""Automatically constructed batched FK, Jacobian, and damped-least-squares IK."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ._cancellation import check_cancelled
from ._validation import require_integer
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
    joint_centering_gain: float = 0.0
    random_seed: int = 42

    def __post_init__(self):
        if self.backend not in {"auto", "numpy", "torch"}:
            raise ValueError(f"unsupported backend {self.backend!r}")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError(f"unsupported dtype {self.dtype!r}")
        require_integer(self.max_iterations, "max_iterations", minimum=1)
        require_integer(self.random_seed, "random_seed")
        for name in ("tolerance", "damping", "step_size", "max_joint_step"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.joint_centering_gain) or self.joint_centering_gain < 0:
            raise ValueError("joint_centering_gain must be finite and non-negative")


@dataclass
class IKResult:
    """IK outcome; iterations counts loop passes across initial and rescue attempts."""

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
        if not self.chain:
            raise ValueError("base and tip links must be different")
        self.active_joints = tuple(j for j in self.chain if j.active)
        self._continuous_indices = tuple(
            index
            for index, joint in enumerate(self.active_joints)
            if joint.kind == "continuous"
        )
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
        self._numpy_chain = tuple(
            (
                np.asarray(joint.origin, dtype=self.config.dtype),
                np.asarray(joint.axis, dtype=self.config.dtype),
            )
            for joint in self.chain
        )
        if backend == "torch":
            try:
                import torch
            except ImportError as exc:
                raise ImportError(
                    "Torch backend requires workspace-analyzer[torch]"
                ) from exc

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
        x = self._configuration_array(q)
        single = x.ndim == 1
        if single:
            x = x[None, :]
        if self.backend == "numpy":
            result = self._forward_numpy(x, all_links)
        else:
            result = self._forward_torch(x, all_links)
        if all_links:
            return {k: v[0] if single else v for k, v in result.items()}
        return result[0] if single else result

    def dexterity(self, q, *, task="position", weights=None):
        """Return Jacobian dexterity and joint-limit metrics."""
        from .metrics import dexterity_metrics

        return dexterity_metrics(self, q, task=task, weights=weights)

    def solve_trajectory(self, target_poses, seed=None, **kwargs):
        """Solve ordered Cartesian poses with continuity diagnostics."""
        from .trajectory import solve_trajectory

        return solve_trajectory(self, target_poses, seed, **kwargs)

    def _forward_numpy(self, q, all_links):
        n, active = len(q), 0
        transform = np.broadcast_to(np.eye(4, dtype=q.dtype), (n, 4, 4)).copy()
        links = {self.base_link: transform} if all_links else None
        for joint, (origin, axis) in zip(self.chain, self._numpy_chain):
            transform = transform @ origin
            if joint.active:
                motion = _motion_numpy(joint.kind, axis, q[:, active])
                transform = transform @ motion
                active += 1
            if all_links:
                links[joint.child] = transform
        return links if all_links else transform

    def _forward_torch(self, q, all_links):
        import torch

        n, active = len(q), 0
        transform = torch.eye(4, dtype=q.dtype, device=q.device).expand(n, 4, 4).clone()
        links = {self.base_link: transform} if all_links else None
        for joint, (origin, axis) in zip(self.chain, self._torch_chain):
            transform = transform @ origin
            if joint.active:
                transform = transform @ _motion_torch(joint.kind, axis, q[:, active], q)
                active += 1
            if all_links:
                links[joint.child] = transform
        return links if all_links else transform

    def jacobian(self, q):
        """Return geometric Jacobian(s), shaped ``(..., 6, dof)``."""
        x = self._configuration_array(q)
        single = x.ndim == 1
        if single:
            x = x[None, :]
        result = (
            _geometric_jacobian_torch(self, x)
            if self.backend == "torch"
            else _geometric_jacobian_numpy(self, x)
        )
        return result[0] if single else result

    def forward_with_jacobian(self, q):
        """Compute end pose and geometric Jacobian in one chain traversal."""
        x = self._configuration_array(q)
        single = x.ndim == 1
        if single:
            x = x[None, :]
        if self.backend == "numpy":
            pose, jacobian = _geometric_jacobian_numpy(self, x, return_pose=True)
        else:
            pose, jacobian = _geometric_jacobian_torch(self, x, return_pose=True)
        if single:
            return pose[0], jacobian[0]
        return pose, jacobian

    def inverse(
        self,
        target,
        seed=None,
        *,
        position_only: bool = False,
        restarts: int = 1,
        rescue_restarts: int = 0,
        rescue_rounds: int = 1,
        random_seed: int | None = None,
        prefer_seed: bool = False,
        posture_reference=None,
        posture_gain: float = 0.0,
        cancel_event=None,
    ) -> IKResult:
        """Solve targets concurrently, optionally choosing among several seeds.

        A single ``(4, 4)`` target returns a ``(DoF,)`` solution and scalar
        diagnostics. Batched ``(N, 4, 4)`` targets preserve N, including N=1.
        Continuous joints wrap into [-pi, pi); other joints respect their limits.
        Cancellation is checked between iterations and rescue attempts.
        """
        check_cancelled(cancel_event)
        require_integer(restarts, "restarts", minimum=1)
        require_integer(rescue_restarts, "rescue_restarts")
        require_integer(rescue_rounds, "rescue_rounds")
        if random_seed is not None:
            require_integer(random_seed, "random_seed")
        if not np.isfinite(posture_gain) or posture_gain < 0:
            raise ValueError("posture_gain must be finite and non-negative")
        goal = self._array(target)
        single = goal.ndim == 2
        if single:
            goal = goal[None, ...]
        if goal.shape[-2:] != (4, 4):
            raise ValueError("target must have shape (..., 4, 4)")
        if goal.ndim != 3 or len(goal) == 0:
            raise ValueError("target batch must contain at least one pose")
        _validate_finite(goal, "target")
        _validate_homogeneous_rows(goal, self.backend)
        if not position_only:
            _validate_rotation_matrices(goal[:, :3, :3], self.backend)
        n = len(goal)
        base_goal = goal
        initial = (
            np.tile(self.joint_limits.mean(axis=1), (n, 1)) if seed is None else seed
        )
        q = self._array(initial)
        if q.ndim not in (1, 2):
            raise ValueError("seed must have shape (DoF,) or (N, DoF)")
        if q.ndim == 1:
            q = q[None, :]
        if len(q) == 1 and n > 1:
            q = q.repeat(n, axis=0) if self.backend == "numpy" else q.repeat(n, 1)
        if q.shape != (n, self.dof):
            raise ValueError(f"seed must have shape ({n}, {self.dof}) or ({self.dof},)")
        _validate_finite(q, "seed")
        limits = self._array(self.joint_limits)
        q = self._bound_configuration(q, limits)
        base_seed = q.copy() if self.backend == "numpy" else q.clone()
        posture = None
        if posture_reference is not None:
            posture = self._array(posture_reference)
            if posture.ndim not in (1, 2):
                raise ValueError("posture_reference has incompatible shape")
            if posture.ndim == 1:
                posture = posture[None, :]
            if len(posture) == 1 and n > 1:
                posture = (
                    posture.repeat(n, axis=0)
                    if self.backend == "numpy"
                    else posture.repeat(n, 1)
                )
            if posture.shape != (n, self.dof):
                raise ValueError("posture_reference has incompatible shape")
            _validate_finite(posture, "posture_reference")
        if restarts > 1:
            rng = np.random.default_rng(
                self.config.random_seed if random_seed is None else random_seed
            )
            extra = rng.uniform(
                self.joint_limits[:, 0],
                self.joint_limits[:, 1],
                size=((restarts - 1) * n, self.dof),
            )
            q = _concat_rows(q, self._array(extra), self.backend)
            goal = _repeat_rows(goal, restarts, self.backend)
        expanded_posture = (
            _repeat_configuration(posture, restarts, self.backend)
            if posture is not None and posture_gain > 0 and restarts > 1
            else posture
        )
        if self.backend == "numpy":
            active = np.arange(len(q))
            success = np.zeros(len(q), dtype=bool)
            residual = np.empty(len(q), dtype=q.dtype)
        else:
            import torch

            active = torch.arange(len(q), device=q.device)
            success = torch.zeros(len(q), dtype=torch.bool, device=q.device)
            residual = torch.empty(len(q), dtype=q.dtype, device=q.device)
        for iteration in range(1, self.config.max_iterations + 1):
            check_cancelled(cancel_event)
            active_q = q[active]
            current, jac = (
                _geometric_jacobian_numpy(self, active_q, return_pose=True)
                if self.backend == "numpy"
                else _geometric_jacobian_torch(self, active_q, return_pose=True)
            )
            error = _pose_error(current, goal[active], self.backend, position_only)
            active_residual = _norm(error, self.backend)
            converged = active_residual <= self.config.tolerance
            residual = _put_rows(residual, active, active_residual, self.backend)
            success[active] = converged
            pending = ~converged
            active = active[pending]
            if len(active) == 0:
                break
            active_q, jac, error = active_q[pending], jac[pending], error[pending]
            if position_only:
                jac = jac[:, :3]
            dq = _dls(jac, error, self.config.damping, self.backend)
            if self.config.joint_centering_gain > 0:
                dq = dq + self.config.joint_centering_gain * _joint_centering_step(
                    jac,
                    active_q,
                    limits,
                    self.config.damping,
                    self.backend,
                    self._continuous_indices,
                )
            if posture_gain > 0 and posture is not None:
                dq = dq + posture_gain * _posture_step(
                    jac,
                    active_q,
                    expanded_posture[active],
                    limits,
                    self.config.damping,
                    self.backend,
                    self._continuous_indices,
                )
            dq = _clip(
                dq,
                -self.config.max_joint_step,
                self.config.max_joint_step,
                self.backend,
            )
            updated = self._bound_configuration(
                active_q + self.config.step_size * dq, limits
            )
            q = _put_rows(q, active, updated, self.backend)
        else:
            # Only unconverged candidates changed on the last allowed step.
            check_cancelled(cancel_event)
            error = _pose_error(
                self._forward_batch(q[active]),
                goal[active],
                self.backend,
                position_only,
            )
            active_residual = _norm(error, self.backend)
            residual = _put_rows(residual, active, active_residual, self.backend)
            success[active] = active_residual <= self.config.tolerance
        check_cancelled(cancel_event)
        if restarts > 1:
            q, success, residual = _select_best(
                q,
                success,
                residual,
                restarts,
                n,
                self.backend,
                reference=base_seed if prefer_seed else None,
                limits=self.joint_limits,
                continuous=np.asarray(
                    [joint.kind == "continuous" for joint in self.active_joints]
                ),
            )
        if rescue_restarts > 0:
            rescue_seed = (
                self.config.random_seed if random_seed is None else random_seed
            )
            for round_index in range(rescue_rounds):
                check_cancelled(cancel_event)
                failed = ~success
                if not bool(failed.any()):
                    break
                # Each attempt disables nested rescue, bounding the call stack
                # and releasing its intermediates before the next round.
                rescue = self.inverse(
                    base_goal[failed],
                    seed=base_seed[failed],
                    position_only=position_only,
                    restarts=rescue_restarts,
                    random_seed=rescue_seed + round_index + 1,
                    prefer_seed=prefer_seed,
                    posture_reference=(None if posture is None else posture[failed]),
                    posture_gain=posture_gain,
                    cancel_event=cancel_event,
                )
                iteration += rescue.iterations
                rescue_q, rescue_success, rescue_residual = _as_result_batch(
                    rescue, self.backend
                )
                q, success, residual = _merge_rescue(
                    q,
                    success,
                    residual,
                    failed,
                    rescue_q,
                    rescue_success,
                    rescue_residual,
                    self.backend,
                )
        check_cancelled(cancel_event)
        return IKResult(
            q[0] if single else q,
            success[0] if single else success,
            residual[0] if single else residual,
            iteration,
        )

    def _bound_configuration(self, q, limits=None):
        """Clip bounded joints and wrap continuous angles without imposing stops."""
        if limits is None:
            limits = self._array(self.joint_limits)
        bounded = _clip(q, limits[:, 0], limits[:, 1], self.backend)
        if self._continuous_indices:
            indices = list(self._continuous_indices)
            bounded[..., indices] = (q[..., indices] + np.pi) % (2 * np.pi) - np.pi
        return bounded

    def _configuration_array(self, q):
        value = self._array(q)
        if value.ndim not in (1, 2):
            raise ValueError("joint positions must have shape (DoF,) or (N, DoF)")
        if value.shape[-1] != self.dof:
            raise ValueError(f"expected {self.dof} joints, got {value.shape[-1]}")
        if value.ndim == 2 and len(value) == 0:
            raise ValueError("joint position batch must not be empty")
        _validate_finite(value, "joint positions")
        return value

    def _forward_batch(self, q):
        return (
            self._forward_numpy(q, False)
            if self.backend == "numpy"
            else self._forward_torch(q, False)
        )

    def _jacobian_batch(self, q):
        return (
            _geometric_jacobian_numpy(self, q)
            if self.backend == "numpy"
            else _geometric_jacobian_torch(self, q)
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
    out = np.broadcast_to(np.eye(4, dtype=values.dtype), (n, 4, 4)).copy()
    if kind == "prismatic":
        out[:, :3, 3] = values[:, None] * axis
    else:
        c, s = np.cos(values), np.sin(values)
        x, y, z = axis
        cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=values.dtype)
        out[:, :3, :3] = (
            np.eye(3, dtype=values.dtype)
            + s[:, None, None] * cross
            + (1 - c)[:, None, None] * (cross @ cross)
        )
    return out


def _motion_torch(kind, axis, values, like):
    import torch

    n = len(values)
    out = torch.eye(4, dtype=like.dtype, device=like.device).expand(n, 4, 4).clone()
    a = torch.as_tensor(axis, dtype=like.dtype, device=like.device)
    if kind == "prismatic":
        out[:, :3, 3] = values[:, None] * a
    else:
        cross = torch.zeros((3, 3), dtype=like.dtype, device=like.device)
        cross[0, 1], cross[0, 2] = -a[2], a[1]
        cross[1, 0], cross[1, 2] = a[2], -a[0]
        cross[2, 0], cross[2, 1] = -a[1], a[0]
        out[:, :3, :3] = (
            torch.eye(3, dtype=like.dtype, device=like.device)
            + torch.sin(values)[:, None, None] * cross
            + (1 - torch.cos(values))[:, None, None] * (cross @ cross)
        )
    return out


def _geometric_jacobian_numpy(solver, q, *, return_pose=False):
    n = len(q)
    t = np.broadcast_to(np.eye(4, dtype=q.dtype), (n, 4, 4)).copy()
    origins = []
    axes = []
    kinds = []
    active = 0
    for joint, (origin, local_axis) in zip(solver.chain, solver._numpy_chain):
        t = t @ origin
        if joint.active:
            origins.append(t[:, :3, 3])
            axes.append(t[:, :3, :3] @ local_axis)
            kinds.append(joint.kind)
            t = t @ _motion_numpy(joint.kind, local_axis, q[:, active])
            active += 1
    tip = t[:, :3, 3]
    j = np.zeros((n, 6, solver.dof), dtype=q.dtype)
    for i, (origin, axis, kind) in enumerate(zip(origins, axes, kinds)):
        if kind == "prismatic":
            j[:, :3, i] = axis
        else:
            j[:, :3, i] = np.cross(axis, tip - origin)
            j[:, 3:, i] = axis
    return (t, j) if return_pose else j


def _geometric_jacobian_torch(solver, q, *, return_pose=False):
    import torch

    n = len(q)
    t = torch.eye(4, dtype=q.dtype, device=q.device).expand(n, 4, 4).clone()
    origins, axes, kinds = [], [], []
    active = 0
    for joint, (origin, local_axis) in zip(solver.chain, solver._torch_chain):
        t = t @ origin
        if joint.active:
            axis = t[:, :3, :3] @ local_axis
            origins.append(t[:, :3, 3])
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
    jacobian = torch.stack(columns, dim=2)
    return (t, jacobian) if return_pose else jacobian


def _pose_error(current, target, backend, position_only):
    pos = target[:, :3, 3] - current[:, :3, 3]
    if position_only:
        return pos
    relative = target[:, :3, :3] @ current[:, :3, :3].swapaxes(1, 2)
    rot = _quaternion_rotation_error(relative, backend)
    return _concat(pos, rot, backend)


def _quaternion_rotation_error(rotation, backend):
    """Shortest-path quaternion error, stable near both zero and half turns."""
    if backend == "numpy":
        stack = lambda values: np.stack(values, axis=-1)  # noqa: E731
    else:
        import torch

        stack = lambda values: torch.stack(values, dim=-1)  # noqa: E731

    r00, r11, r22 = rotation[:, 0, 0], rotation[:, 1, 1], rotation[:, 2, 2]
    squared = stack(
        (
            1 + r00 + r11 + r22,
            1 + r00 - r11 - r22,
            1 - r00 + r11 - r22,
            1 - r00 - r11 + r22,
        )
    )
    xy = rotation[:, 0, 1] + rotation[:, 1, 0]
    xz = rotation[:, 0, 2] + rotation[:, 2, 0]
    yz = rotation[:, 1, 2] + rotation[:, 2, 1]
    wx = rotation[:, 2, 1] - rotation[:, 1, 2]
    wy = rotation[:, 0, 2] - rotation[:, 2, 0]
    wz = rotation[:, 1, 0] - rotation[:, 0, 1]
    # Each column is 4*q_component times [w, x, y, z]. Choose the
    # largest component so the divisor stays away from zero. In particular,
    # small rotations use off-diagonal differences rather than sqrt(1-Rii).
    candidates = stack(
        (
            stack((squared[:, 0], wx, wy, wz)),
            stack((wx, squared[:, 1], xy, xz)),
            stack((wy, xy, squared[:, 2], yz)),
            stack((wz, xz, yz, squared[:, 3])),
        )
    )
    if backend == "numpy":
        index = np.argmax(squared, axis=-1)
        rows = np.arange(len(rotation))
        quaternion = candidates[rows, :, index] / (
            2 * np.sqrt(squared[rows, index, None])
        )
        return (
            2
            * quaternion[:, 1:]
            * np.where(quaternion[:, :1] < 0, -1.0, 1.0).astype(rotation.dtype)
        )
    index = torch.argmax(squared, dim=-1)
    rows = torch.arange(len(rotation), device=rotation.device)
    quaternion = candidates[rows, :, index] / (
        2 * torch.sqrt(squared[rows, index, None])
    )
    return 2 * quaternion[:, 1:] * torch.where(quaternion[:, :1] < 0, -1.0, 1.0)


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


def _select_best(
    q,
    success,
    residual,
    restarts,
    targets,
    backend,
    reference=None,
    limits=None,
    continuous=None,
):
    shaped_residual = residual.reshape(restarts, targets)
    if backend == "numpy":
        indices = np.argmin(shaped_residual, axis=0)
        if reference is not None:
            shaped_success = success.reshape(restarts, targets)
            distance = _seed_distance_numpy(
                q.reshape(restarts, targets, -1), reference, limits, continuous
            )
            closest = np.argmin(np.where(shaped_success, distance, np.inf), axis=0)
            indices = np.where(np.any(shaped_success, axis=0), closest, indices)
        columns = np.arange(targets)
        return (
            q.reshape(restarts, targets, -1)[indices, columns],
            success.reshape(restarts, targets)[indices, columns],
            shaped_residual[indices, columns],
        )
    import torch

    indices = torch.argmin(shaped_residual, dim=0)
    if reference is not None:
        shaped_success = success.reshape(restarts, targets)
        distance = _seed_distance_torch(
            q.reshape(restarts, targets, -1), reference, limits, continuous
        )
        closest = torch.argmin(torch.where(shaped_success, distance, torch.inf), dim=0)
        indices = torch.where(torch.any(shaped_success, dim=0), closest, indices)
    columns = torch.arange(targets, device=q.device)
    return (
        q.reshape(restarts, targets, -1)[indices, columns],
        success.reshape(restarts, targets)[indices, columns],
        shaped_residual[indices, columns],
    )


def _seed_distance_numpy(q, reference, limits, continuous):
    delta = q - reference[None, :, :]
    delta[..., continuous] = (delta[..., continuous] + np.pi) % (2 * np.pi) - np.pi
    ranges = np.maximum(limits[:, 1] - limits[:, 0], 1e-12)
    return np.linalg.norm(delta / ranges, axis=-1)


def _seed_distance_torch(q, reference, limits, continuous):
    import torch

    delta = q - reference[None, :, :]
    mask = torch.as_tensor(continuous, dtype=torch.bool, device=q.device)
    delta[..., mask] = (
        torch.remainder(delta[..., mask] + torch.pi, 2 * torch.pi) - torch.pi
    )
    ranges = torch.as_tensor(
        limits[:, 1] - limits[:, 0], dtype=q.dtype, device=q.device
    ).clamp_min(1e-12)
    return torch.linalg.vector_norm(delta / ranges, dim=-1)


def _as_result_batch(result, backend):
    if result.positions.ndim == 2:
        return result.positions, result.success, result.residual
    if backend == "numpy":
        return (
            result.positions[None, :],
            np.asarray([result.success]),
            np.asarray([result.residual]),
        )
    return (
        result.positions[None, :],
        result.success[None],
        result.residual[None],
    )


def _merge_rescue(
    q,
    success,
    residual,
    failed,
    rescue_q,
    rescue_success,
    rescue_residual,
    backend,
):
    if backend == "numpy":
        failed_indices = np.flatnonzero(failed)
    else:
        import torch

        failed_indices = torch.nonzero(failed, as_tuple=False).flatten()
    better = rescue_residual < residual[failed]
    selected = failed_indices[better]
    q = _put_rows(q, selected, rescue_q[better], backend)
    success[selected] = rescue_success[better]
    residual = _put_rows(residual, selected, rescue_residual[better], backend)
    return q, success, residual


def _norm(x, backend):
    if backend == "numpy":
        return np.linalg.norm(x, axis=1)
    import torch

    return torch.linalg.vector_norm(x, dim=1)


def _dls(j, e, damping, backend):
    jt = j.swapaxes(1, 2)
    size = j.shape[1]
    if backend == "numpy":
        eye = np.eye(size, dtype=j.dtype)[None]
        return (jt @ np.linalg.solve(j @ jt + damping * damping * eye, e[..., None]))[
            ..., 0
        ]
    import torch

    eye = torch.eye(size, dtype=j.dtype, device=j.device)[None]
    return (jt @ torch.linalg.solve(j @ jt + damping * damping * eye, e[..., None]))[
        ..., 0
    ]


def _joint_centering_step(j, q, limits, damping, backend, continuous=()):
    """Project a normalized center-seeking gradient into the task null space."""
    center = (limits[:, 0] + limits[:, 1]) * 0.5
    half_range = (limits[:, 1] - limits[:, 0]) * 0.5
    gradient = (center - q) / half_range
    if continuous:
        gradient[..., list(continuous)] = 0.0
    task_gradient = (j @ gradient[..., None])[..., 0]
    projected_task = _dls(j, task_gradient, damping, backend)
    return gradient - projected_task


def _posture_step(j, q, reference, limits, damping, backend, continuous=()):
    """Project a normalized reference-posture gradient into the task null space."""
    ranges = limits[:, 1] - limits[:, 0]
    delta = reference - q
    if continuous:
        indices = list(continuous)
        delta[..., indices] = (delta[..., indices] + np.pi) % (2 * np.pi) - np.pi
    gradient = delta / ranges
    task_gradient = (j @ gradient[..., None])[..., 0]
    return gradient - _dls(j, task_gradient, damping, backend)


def _repeat_configuration(value, count, backend):
    return np.tile(value, (count, 1)) if backend == "numpy" else value.repeat(count, 1)


def _clip(q, lo, hi, backend):
    return np.clip(q, lo, hi) if backend == "numpy" else q.clamp(lo, hi)


def _put_rows(destination, indices, values, backend):
    if backend == "numpy":
        destination[indices] = values
        return destination
    # Out-of-place index_copy preserves autograd values needed by earlier steps.
    return destination.index_copy(0, indices, values)


def _validate_finite(value, name: str) -> None:
    if isinstance(value, np.ndarray):
        finite = np.isfinite(value).all()
    else:
        import torch

        finite = torch.isfinite(value).all().item()
    if not finite:
        raise ValueError(f"{name} must contain only finite values")


def _validate_homogeneous_rows(target, backend: str) -> None:
    expected = [0.0, 0.0, 0.0, 1.0]
    if backend == "numpy":
        valid = np.allclose(target[:, 3], expected, atol=1e-7, rtol=0.0)
    else:
        import torch

        row = torch.as_tensor(expected, dtype=target.dtype, device=target.device)
        valid = torch.allclose(
            target[:, 3], row.expand_as(target[:, 3]), atol=1e-7, rtol=0.0
        )
    if not valid:
        raise ValueError("target poses must have homogeneous last row [0, 0, 0, 1]")


def _validate_rotation_matrices(rotation, backend: str) -> None:
    tolerance = 2e-5 if str(rotation.dtype).endswith("32") else 1e-8
    if backend == "numpy":
        identity = np.eye(3)
        orthogonal = np.allclose(
            rotation @ rotation.swapaxes(1, 2), identity, atol=tolerance, rtol=0.0
        )
        proper = np.allclose(np.linalg.det(rotation), 1.0, atol=tolerance, rtol=0.0)
    else:
        import torch

        identity = torch.eye(3, dtype=rotation.dtype, device=rotation.device)
        orthogonal = torch.allclose(
            rotation @ rotation.swapaxes(1, 2), identity, atol=tolerance, rtol=0.0
        )
        proper = torch.allclose(
            torch.linalg.det(rotation),
            torch.ones(len(rotation), dtype=rotation.dtype, device=rotation.device),
            atol=tolerance,
            rtol=0.0,
        )
    if not orthogonal or not proper:
        raise ValueError("target rotations must be proper orthonormal matrices")
