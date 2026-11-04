"""Reproducible FK/Jacobian/IK benchmark for a selected robot chain."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from workspace_analyzer import KinematicsSolver, RobotModel, SolverConfig
from workspace_analyzer._arrays import as_numpy as _numpy
from workspace_analyzer._atomic import atomic_write_text
from workspace_analyzer._output_paths import validate_output_paths
from workspace_analyzer.presets import (
    default_robot_urdf,
    require_robot_urdf,
    select_arm_links,
)

DEFAULT_URDF = default_robot_urdf()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    parser.add_argument("--base-link")
    parser.add_argument("--tip-link")
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
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    try:
        return _run(args, parser)
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        parser.error(str(exc))


def _run(args, parser):
    args.urdf = require_robot_urdf(args.urdf, parser)
    validate_output_paths(report=args.report, inputs=(args.urdf,))
    if args.batch_size < 1 or args.ik_targets < 1:
        parser.error("--batch-size and --ik-targets must be positive")
    if args.repeats < 1 or args.warmup < 0:
        parser.error("--repeats must be positive and --warmup must be non-negative")

    model = RobotModel.from_urdf(args.urdf)
    base, tip = select_arm_links(model, args.arm, args.base_link, args.tip_link)
    solver = KinematicsSolver(
        model,
        base,
        tip,
        SolverConfig(
            backend=args.backend,
            device=args.device,
            dtype=args.dtype,
            random_seed=args.seed,
        ),
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
        "base_link": solver.base_link,
        "tip_link": solver.tip_link,
        "joint_names": list(solver.joint_names),
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
        "ik_seed": solver.config.random_seed,
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
    text = json.dumps(report, indent=2, allow_nan=False)
    print(text)
    if args.report is not None:
        atomic_write_text(args.report, text + "\n")
    return report


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
