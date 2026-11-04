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
    """Batched NumPy/Torch solver for a selected serial chain.

    Inputs may be a single ``(dof,)`` configuration or a batch ``(N, dof)``.
    Torch mode preserves tensors and executes the whole batch on the selected device.
    Joint transforms and axes are snapshotted at construction; rebuild after edits.
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
        # Clipping and quality calculations must retain a usable range in this dtype.
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            runtime_limits = np.asarray(self.joint_limits, dtype=self.config.dtype)
            centers = (runtime_limits[:, 0] + runtime_limits[:, 1]) * 0.5
            half_ranges = (runtime_limits[:, 1] - runtime_limits[:, 0]) * 0.5
        valid_ranges = (
            np.isfinite(runtime_limits).all(axis=1)
            & np.isfinite(centers)
            & np.isfinite(half_ranges)
            & (half_ranges > 0)
        )
        if not valid_ranges.all():
            joint = self.active_joints[np.flatnonzero(~valid_ranges)[0]]
            raise ValueError(
                f"joint {joint.name!r} limits must have finite bounds and center, "
                f"and a positive finite half-range in {self.config.dtype}"
            )
        backend = self.config.backend
        if backend == "auto":
            try:
                import torch  # noqa: F401

                backend = "torch"
            except ImportError:
                backend = "numpy"
        self.backend = backend
        self.device = "cpu"
        numpy_chain = []
        for joint in self.chain:
            with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                origin = np.array(joint.origin, dtype=self.config.dtype, copy=True)
                axis = np.array(joint.axis, dtype=self.config.dtype, copy=True)
                rotation_terms = _rotation_terms(axis, self.config.dtype)
            if not all(
                np.isfinite(value).all() for value in (origin, axis, *rotation_terms)
            ):
                raise ValueError(
                    f"joint {joint.name!r} geometry must remain finite "
                    f"in {self.config.dtype}"
                )
            numpy_chain.append((origin, axis, rotation_terms))
        self._numpy_chain = tuple(numpy_chain)
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
                    torch.as_tensor(origin, dtype=dtype, device=self.device),
                    torch.as_tensor(axis, dtype=dtype, device=self.device),
                    tuple(
                        torch.as_tensor(value, dtype=dtype, device=self.device)
                        for value in rotation_terms
                    ),
                )
                for origin, axis, rotation_terms in self._numpy_chain
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
        input_dtype = getattr(value, "dtype", None)
        if (
            getattr(input_dtype, "is_complex", False)
            or getattr(input_dtype, "kind", None) == "c"
            or (input_dtype is None and np.iscomplexobj(value))
        ):
            raise ValueError("kinematic inputs must contain only real values")
        if self.backend == "numpy":
            return np.asarray(value, dtype=self.config.dtype)
        import torch

        dtype = torch.float64 if self.config.dtype == "float64" else torch.float32
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    def forward(self, q, *, all_links: bool = False):
        configuration = self._configuration_array(q)
        single = configuration.ndim == 1
        if single:
            configuration = configuration[None, :]
        if self.backend == "numpy":
            result = self._forward_numpy(configuration, all_links)
        else:
            result = self._forward_torch(configuration, all_links)
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
        batch_size, joint_index = len(q), 0
        transform = np.broadcast_to(np.eye(4, dtype=q.dtype), (batch_size, 4, 4)).copy()
        links = {self.base_link: transform} if all_links else None
        for joint, (origin, axis, rotation_terms) in zip(self.chain, self._numpy_chain):
            transform = transform @ origin
            if joint.active:
                motion = _motion_numpy(
                    joint.kind, axis, q[:, joint_index], rotation_terms
                )
                transform = transform @ motion
                joint_index += 1
            if all_links:
                links[joint.child] = transform
        return links if all_links else transform

    def _forward_torch(self, q, all_links):
        import torch

        batch_size, joint_index = len(q), 0
        transform = (
            torch.eye(4, dtype=q.dtype, device=q.device)
            .expand(batch_size, 4, 4)
            .clone()
        )
        links = {self.base_link: transform} if all_links else None
        for joint, (origin, axis, rotation_terms) in zip(self.chain, self._torch_chain):
            transform = transform @ origin
            if joint.active:
                transform = transform @ _motion_torch(
                    joint.kind, axis, q[:, joint_index], rotation_terms
                )
                joint_index += 1
            if all_links:
                links[joint.child] = transform
        return links if all_links else transform

    def jacobian(self, q):
        """Return geometric Jacobian(s), shaped ``(..., 6, dof)``."""
        configuration = self._configuration_array(q)
        single = configuration.ndim == 1
        if single:
            configuration = configuration[None, :]
        result = (
            _geometric_jacobian_torch(self, configuration)
            if self.backend == "torch"
            else _geometric_jacobian_numpy(self, configuration)
        )
        return result[0] if single else result

    def forward_with_jacobian(self, q):
        """Compute end pose and geometric Jacobian in one chain traversal."""
        configuration = self._configuration_array(q)
        single = configuration.ndim == 1
        if single:
            configuration = configuration[None, :]
        if self.backend == "numpy":
            pose, jacobian = _geometric_jacobian_numpy(
                self, configuration, return_pose=True
            )
        else:
            pose, jacobian = _geometric_jacobian_torch(
                self, configuration, return_pose=True
            )
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
        target_count = len(goal)
        base_goal = goal
        joint_limits = self.joint_limits
        initial = (
            np.tile(joint_limits.mean(axis=1), (target_count, 1))
            if seed is None
            else seed
        )
        q = self._array(initial)
        if q.ndim not in (1, 2):
            raise ValueError("seed must have shape (DoF,) or (N, DoF)")
        if q.ndim == 1:
            q = q[None, :]
        if len(q) == 1 and target_count > 1:
            q = (
                q.repeat(target_count, axis=0)
                if self.backend == "numpy"
                else q.repeat(target_count, 1)
            )
        if q.shape != (target_count, self.dof):
            raise ValueError(
                f"seed must have shape ({target_count}, {self.dof}) or ({self.dof},)"
            )
        _validate_finite(q, "seed")
        limits = self._array(joint_limits)
        q = self._bound_configuration(q, limits)
        base_seed = q.copy() if self.backend == "numpy" else q.clone()
        posture = None
        if posture_reference is not None:
            posture = self._array(posture_reference)
            if posture.ndim not in (1, 2):
                raise ValueError("posture_reference has incompatible shape")
            if posture.ndim == 1:
                posture = posture[None, :]
            if len(posture) == 1 and target_count > 1:
                posture = (
                    posture.repeat(target_count, axis=0)
                    if self.backend == "numpy"
                    else posture.repeat(target_count, 1)
                )
            if posture.shape != (target_count, self.dof):
                raise ValueError("posture_reference has incompatible shape")
            _validate_finite(posture, "posture_reference")
        if restarts > 1:
            rng = np.random.default_rng(
                self.config.random_seed if random_seed is None else random_seed
            )
            extra = rng.uniform(
                joint_limits[:, 0],
                joint_limits[:, 1],
                size=((restarts - 1) * target_count, self.dof),
            )
            q = _concat(q, self._array(extra), self.backend, axis=0)
            goal = _repeat_rows(goal, restarts, self.backend)
        expanded_posture = (
            _repeat_rows(posture, restarts, self.backend)
            if posture is not None and posture_gain > 0 and restarts > 1
            else posture
        )
        if self.backend == "numpy":
            active_indices = np.arange(len(q))
            success = np.zeros(len(q), dtype=bool)
            residual = np.empty(len(q), dtype=q.dtype)
        else:
            import torch

            active_indices = torch.arange(len(q), device=q.device)
            success = torch.zeros(len(q), dtype=torch.bool, device=q.device)
            residual = torch.empty(len(q), dtype=q.dtype, device=q.device)
        for iteration in range(1, self.config.max_iterations + 1):
            check_cancelled(cancel_event)
            active_q = q[active_indices]
            current, jacobian = (
                _geometric_jacobian_numpy(self, active_q, return_pose=True)
                if self.backend == "numpy"
                else _geometric_jacobian_torch(self, active_q, return_pose=True)
            )
            error = _pose_error(
                current, goal[active_indices], self.backend, position_only
            )
            active_residual = _norm(error, self.backend)
            converged = active_residual <= self.config.tolerance
            residual = _put_rows(
                residual, active_indices, active_residual, self.backend
            )
            success[active_indices] = converged
            pending = ~converged
            active_indices = active_indices[pending]
            if len(active_indices) == 0:
                break
            active_q, jacobian, error = (
                active_q[pending],
                jacobian[pending],
                error[pending],
            )
            if position_only:
                jacobian = jacobian[:, :3]
            bias = None
            if self.config.joint_centering_gain > 0:
                bias = self.config.joint_centering_gain * _joint_centering_gradient(
                    active_q,
                    limits,
                    self._continuous_indices,
                )
            if posture_gain > 0 and posture is not None:
                posture_bias = posture_gain * _posture_gradient(
                    active_q,
                    expanded_posture[active_indices],
                    limits,
                    self._continuous_indices,
                )
                bias = posture_bias if bias is None else bias + posture_bias
            joint_step = _biased_dls(
                jacobian, error, self.config.damping, self.backend, bias
            )
            joint_step = _clip(
                joint_step,
                -self.config.max_joint_step,
                self.config.max_joint_step,
                self.backend,
            )
            updated = self._bound_configuration(
                active_q + self.config.step_size * joint_step, limits
            )
            moving = _moving_candidates(active_q, updated, self.backend)
            q = _put_rows(q, active_indices, updated, self.backend)
            if moving is not None:
                # An unchanged configuration is a deterministic fixed point for
                # this attempt. Keep its measured residual and allow rescue.
                active_indices = active_indices[moving]
                if len(active_indices) == 0:
                    break
        else:
            # Only unconverged candidates changed on the last allowed step.
            check_cancelled(cancel_event)
            error = _pose_error(
                self._forward_batch(q[active_indices]),
                goal[active_indices],
                self.backend,
                position_only,
            )
            active_residual = _norm(error, self.backend)
            residual = _put_rows(
                residual, active_indices, active_residual, self.backend
            )
            success[active_indices] = active_residual <= self.config.tolerance
        check_cancelled(cancel_event)
        if restarts > 1:
            q, success, residual = _select_best(
                q,
                success,
                residual,
                restarts,
                target_count,
                self.backend,
                reference=base_seed if prefer_seed else None,
                limits=joint_limits,
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
                # Slicing base_goal preserves the target batch, including N=1.
                q, success, residual = _merge_rescue(
                    q,
                    success,
                    residual,
                    failed,
                    rescue.positions,
                    rescue.success,
                    rescue.residual,
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
    """Parse a URDF, select a chain, and configure a NumPy or Torch solver.

    Joint transforms and axes are copied when the solver is constructed.
    """
    return KinematicsSolver(
        RobotModel.from_urdf(urdf),
        base_link,
        tip_link,
        SolverConfig(backend=backend, device=device, **kwargs),
    )


def _moving_candidates(before, after, backend):
    if backend == "numpy":
        return np.any(after != before, axis=1)
    # A value fixed point need not be a derivative fixed point. Preserve the
    # complete unrolled graph whenever the update participates in autograd.
    if after.requires_grad:
        return None
    return (after != before).any(dim=1)


def _rotation_terms(axis, dtype):
    """Cache the constant Rodrigues coefficients in the solver's precision."""
    x, y, z = np.asarray(axis, dtype=dtype)
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=dtype)
    return skew, skew @ skew


def _motion_numpy(kind, axis, values, rotation_terms):
    batch_size = len(values)
    motion = np.broadcast_to(np.eye(4, dtype=values.dtype), (batch_size, 4, 4)).copy()
    if kind == "prismatic":
        motion[:, :3, 3] = values[:, None] * axis
    else:
        cosine, sine = np.cos(values), np.sin(values)
        skew, skew_squared = rotation_terms
        motion[:, :3, :3] = (
            np.eye(3, dtype=values.dtype)
            + sine[:, None, None] * skew
            + (1 - cosine)[:, None, None] * skew_squared
        )
    return motion


def _motion_torch(kind, axis, values, rotation_terms):
    import torch

    batch_size = len(values)
    motion = (
        torch.eye(4, dtype=values.dtype, device=values.device)
        .expand(batch_size, 4, 4)
        .clone()
    )
    if kind == "prismatic":
        motion[:, :3, 3] = values[:, None] * axis
    else:
        skew, skew_squared = rotation_terms
        motion[:, :3, :3] = (
            torch.eye(3, dtype=values.dtype, device=values.device)
            + torch.sin(values)[:, None, None] * skew
            + (1 - torch.cos(values))[:, None, None] * skew_squared
        )
    return motion


def _geometric_jacobian_numpy(solver, q, *, return_pose=False):
    batch_size = len(q)
    transform = np.broadcast_to(np.eye(4, dtype=q.dtype), (batch_size, 4, 4)).copy()
    origins = []
    axes = []
    kinds = []
    joint_index = 0
    for joint, (origin, local_axis, rotation_terms) in zip(
        solver.chain, solver._numpy_chain
    ):
        transform = transform @ origin
        if joint.active:
            origins.append(transform[:, :3, 3])
            axes.append(transform[:, :3, :3] @ local_axis)
            kinds.append(joint.kind)
            transform = transform @ _motion_numpy(
                joint.kind, local_axis, q[:, joint_index], rotation_terms
            )
            joint_index += 1
    tip_position = transform[:, :3, 3]
    jacobian = np.zeros((batch_size, 6, solver.dof), dtype=q.dtype)
    for column, (origin, axis, kind) in enumerate(zip(origins, axes, kinds)):
        if kind == "prismatic":
            jacobian[:, :3, column] = axis
        else:
            jacobian[:, :3, column] = np.cross(axis, tip_position - origin)
            jacobian[:, 3:, column] = axis
    return (transform, jacobian) if return_pose else jacobian


def _geometric_jacobian_torch(solver, q, *, return_pose=False):
    import torch

    batch_size = len(q)
    transform = (
        torch.eye(4, dtype=q.dtype, device=q.device).expand(batch_size, 4, 4).clone()
    )
    origins, axes, kinds = [], [], []
    joint_index = 0
    for joint, (origin, local_axis, rotation_terms) in zip(
        solver.chain, solver._torch_chain
    ):
        transform = transform @ origin
        if joint.active:
            axis = transform[:, :3, :3] @ local_axis
            origins.append(transform[:, :3, 3])
            axes.append(axis)
            kinds.append(joint.kind)
            transform = transform @ _motion_torch(
                joint.kind, local_axis, q[:, joint_index], rotation_terms
            )
            joint_index += 1
    tip_position = transform[:, :3, 3]
    columns = []
    for origin, axis, kind in zip(origins, axes, kinds):
        if kind == "prismatic":
            columns.append(torch.cat((axis, torch.zeros_like(axis)), dim=1))
        else:
            linear = torch.linalg.cross(axis, tip_position - origin, dim=1)
            columns.append(torch.cat((linear, axis), dim=1))
    jacobian = torch.stack(columns, dim=2)
    return (transform, jacobian) if return_pose else jacobian


def _pose_error(current, target, backend, position_only):
    position_error = target[:, :3, 3] - current[:, :3, 3]
    if position_only:
        return position_error
    relative = target[:, :3, :3] @ current[:, :3, :3].swapaxes(1, 2)
    rotation_error = _quaternion_rotation_error(relative, backend)
    return _concat(position_error, rotation_error, backend)


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


def _concat(a, b, backend, *, axis=1):
    if backend == "numpy":
        return np.concatenate((a, b), axis=axis)
    import torch

    return torch.cat((a, b), dim=axis)


def _repeat_rows(value, count, backend):
    repeats = (count,) + (1,) * (value.ndim - 1)
    if backend == "numpy":
        return np.tile(value, repeats)
    return value.repeat(*repeats)


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


def _dls(jacobian, error, damping, backend):
    # Float64 normal systems reduce damping loss from float32 rounding.
    # Preserve the caller's dtype on return.
    dtype = jacobian.dtype
    task_dimension = jacobian.shape[1]
    if backend == "numpy":
        jacobian = jacobian.astype(np.float64, copy=False)
        error = error.astype(np.float64, copy=False)
        transposed = jacobian.swapaxes(1, 2)
        identity = np.eye(task_dimension, dtype=jacobian.dtype)
        solved_error = np.linalg.solve(
            jacobian @ transposed + damping * damping * identity, error[..., None]
        )
        return (transposed @ solved_error)[..., 0].astype(dtype, copy=False)
    import torch

    if dtype == torch.float32 and jacobian.device.type in {"cpu", "cuda"}:
        jacobian, error = jacobian.double(), error.double()
    transposed = jacobian.swapaxes(1, 2)
    identity = torch.eye(task_dimension, dtype=jacobian.dtype, device=jacobian.device)
    solved_error = torch.linalg.solve(
        jacobian @ transposed + damping * damping * identity, error[..., None]
    )
    return (transposed @ solved_error)[..., 0].to(dtype)


def _biased_dls(jacobian, error, damping, backend, bias=None):
    """Combine task correction and damped posture projection in one solve.

    J# e + (I - J# J) bias = bias + J# (e - J bias).
    The damped projection only approximately preserves the primary task.
    """
    if bias is None:
        return _dls(jacobian, error, damping, backend)
    task_bias = (jacobian @ bias[..., None])[..., 0]
    return bias + _dls(jacobian, error - task_bias, damping, backend)


def _joint_centering_gradient(q, limits, continuous=()):
    """Return the normalized center-seeking joint gradient."""
    center = (limits[:, 0] + limits[:, 1]) * 0.5
    half_range = (limits[:, 1] - limits[:, 0]) * 0.5
    gradient = (center - q) / half_range
    if continuous:
        gradient[..., list(continuous)] = 0.0
    return gradient


def _posture_gradient(q, reference, limits, continuous=()):
    """Return the normalized shortest-path reference-posture gradient."""
    ranges = limits[:, 1] - limits[:, 0]
    delta = reference - q
    if continuous:
        indices = list(continuous)
        delta[..., indices] = (delta[..., indices] + np.pi) % (2 * np.pi) - np.pi
    return delta / ranges


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
