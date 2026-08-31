"""Reproducible FK/Jacobian/IK benchmark for one Marvin M6 arm."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from workspace_analyzer import create_solver

DEFAULT_URDF = Path(
    "/home/ubuntu/workspace/chase/HumanoidAssets/Marvin_M6_S_CCS_696_V4.0/robot.urdf"
)


def _numpy(value):
    return (
        value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    parser.add_argument("--backend", choices=("numpy", "torch"), default="numpy")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--ik-targets", type=int, default=256)
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--seed", type=int, default=12)
    args = parser.parse_args()

    solver = create_solver(
        str(args.urdf),
        base_link="base_link",
        tip_link=f"{args.arm}_ee",
        backend=args.backend,
        device=args.device,
        dtype=args.dtype,
    )
    rng = np.random.default_rng(args.seed)
    lower, upper = solver.joint_limits[:, 0], solver.joint_limits[:, 1]
    q = rng.uniform(lower, upper, (args.batch_size, solver.dof))

    solver.forward(q)
    solver.jacobian(q)
    started = time.perf_counter()
    solver.forward(q)
    _synchronize(solver)
    fk_seconds = time.perf_counter() - started
    started = time.perf_counter()
    solver.jacobian(q)
    _synchronize(solver)
    jacobian_seconds = time.perf_counter() - started

    ik_q = q[: args.ik_targets]
    targets = solver.forward(ik_q)
    started = time.perf_counter()
    result = solver.inverse(targets, restarts=args.restarts)
    _synchronize(solver)
    ik_seconds = time.perf_counter() - started
    residual = _numpy(result.residual)
    success = _numpy(result.success)
    report = {
        "robot": solver.model.name,
        "arm": args.arm,
        "dof": solver.dof,
        "backend": solver.backend,
        "device": solver.device,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "fk_ms": fk_seconds * 1e3,
        "fk_poses_per_second": args.batch_size / fk_seconds,
        "jacobian_ms": jacobian_seconds * 1e3,
        "jacobians_per_second": args.batch_size / jacobian_seconds,
        "ik_targets": args.ik_targets,
        "ik_restarts": args.restarts,
        "ik_ms": ik_seconds * 1e3,
        "ik_success_rate": float(np.mean(success)),
        "ik_residual_p95": float(np.percentile(residual, 95)),
        "ik_residual_max": float(np.max(residual)),
    }
    print(json.dumps(report, indent=2))


def _synchronize(solver) -> None:
    if solver.backend == "torch" and solver.device.startswith("cuda"):
        import torch

        torch.cuda.synchronize(solver.device)


if __name__ == "__main__":
    main()
