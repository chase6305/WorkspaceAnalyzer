"""Reproducible FK/Jacobian/IK benchmark for one Marvin M6 arm."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from workspace_analyzer import create_solver
from workspace_analyzer.presets import default_robot_urdf, require_robot_urdf

DEFAULT_URDF = default_robot_urdf()


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
    parser.add_argument("--rescue-restarts", type=int, default=0)
    parser.add_argument("--rescue-rounds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()
    args.urdf = require_robot_urdf(args.urdf, parser)
    if args.batch_size < 1 or args.ik_targets < 1:
        parser.error("--batch-size and --ik-targets must be positive")
    if args.repeats < 1 or args.warmup < 0:
        parser.error("--repeats must be positive and --warmup must be non-negative")

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

    fk_seconds, _ = _measure(solver, lambda: solver.forward(q), args)
    jacobian_seconds, _ = _measure(solver, lambda: solver.jacobian(q), args)

    ik_q = rng.uniform(lower, upper, (args.ik_targets, solver.dof))
    targets = solver.forward(ik_q)
    ik_seconds, result = _measure(
        solver,
        lambda: solver.inverse(
            targets,
            restarts=args.restarts,
            rescue_restarts=args.rescue_restarts,
            rescue_rounds=args.rescue_rounds,
        ),
        args,
    )
    residual = _numpy(result.residual)
    success = _numpy(result.success)
    report = {
        "robot": solver.model.name,
        "arm": args.arm,
        "dof": solver.dof,
        "backend": solver.backend,
        "device": solver.device,
        "dtype": args.dtype,
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "seed": args.seed,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "timing_statistic": "median",
        "batch_size": args.batch_size,
        "fk_ms": fk_seconds * 1e3,
        "fk_poses_per_second": args.batch_size / fk_seconds,
        "jacobian_ms": jacobian_seconds * 1e3,
        "jacobians_per_second": args.batch_size / jacobian_seconds,
        "ik_targets": args.ik_targets,
        "ik_restarts": args.restarts,
        "ik_rescue_restarts": args.rescue_restarts,
        "ik_rescue_rounds": args.rescue_rounds,
        "ik_ms": ik_seconds * 1e3,
        "ik_success_rate": float(np.mean(success)),
        "ik_residual_p95": float(np.percentile(residual, 95)),
        "ik_residual_max": float(np.max(residual)),
    }
    if solver.backend == "torch":
        import torch

        report["environment"]["torch"] = torch.__version__
        if solver.device.startswith("cuda"):
            report["environment"]["gpu"] = torch.cuda.get_device_name(solver.device)
    print(json.dumps(report, indent=2))


def _measure(solver, operation, args):
    for _ in range(args.warmup):
        operation()
    durations = []
    for _ in range(args.repeats):
        _synchronize(solver)
        started = time.perf_counter()
        result = operation()
        _synchronize(solver)
        durations.append(time.perf_counter() - started)
    return float(np.median(durations)), result


def _synchronize(solver) -> None:
    if solver.backend == "torch" and solver.device.startswith("cuda"):
        import torch

        torch.cuda.synchronize(solver.device)


if __name__ == "__main__":
    main()
