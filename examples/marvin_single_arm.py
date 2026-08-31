"""Marvin M6 single-arm reachability analysis with optional Viser display."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from workspace_analyzer import (
    CartesianConfig,
    SamplingConfig,
    SamplingStrategy,
    WorkspaceAnalyzer,
    WorkspaceConfig,
    create_solver,
)

DEFAULT_URDF = Path(
    "/home/ubuntu/workspace/chase/HumanoidAssets/Marvin_M6_S_CCS_696_V4.0/robot.urdf"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    parser.add_argument("--backend", choices=("auto", "numpy", "torch"), default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument(
        "--strategy", choices=[x.value for x in SamplingStrategy], default="sobol"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/marvin_left_arm.npz")
    )
    parser.add_argument("--viser", action="store_true")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--mode", choices=("joint", "cartesian"), default="joint")
    parser.add_argument("--ik-restarts", type=int, default=4)
    parser.add_argument(
        "--full-pose",
        action="store_true",
        help="require the FK reference orientation in Cartesian mode",
    )
    parser.add_argument(
        "--reference-joints",
        type=float,
        nargs="+",
        help="reference joint vector; FK defines the Cartesian orientation",
    )
    args = parser.parse_args()

    tip = f"{args.arm}_ee"
    solver = create_solver(
        str(args.urdf),
        base_link="base_link",
        tip_link=tip,
        backend=args.backend,
        device=args.device,
        dtype=args.dtype,
    )
    config = WorkspaceConfig(
        sampling=SamplingConfig(
            strategy=SamplingStrategy(args.strategy),
            num_samples=args.samples,
            batch_size=args.batch_size,
        )
    )
    analyzer = WorkspaceAnalyzer(solver, config)
    reference_q = (
        solver.joint_limits.mean(axis=1)
        if args.reference_joints is None
        else np.asarray(args.reference_joints, dtype=float)
    )
    if reference_q.shape != (solver.dof,):
        parser.error(f"--reference-joints requires exactly {solver.dof} values")
    if np.any(reference_q < solver.joint_limits[:, 0]) or np.any(
        reference_q > solver.joint_limits[:, 1]
    ):
        parser.error("--reference-joints contains a value outside the URDF limits")
    started = time.perf_counter()
    if args.mode == "cartesian":
        # Estimate a useful XYZ box cheaply from FK, then test targets directly with IK.
        estimate_config = WorkspaceConfig(
            sampling=SamplingConfig(
                strategy=SamplingStrategy.SOBOL,
                num_samples=min(16_384, max(4096, args.samples)),
                batch_size=args.batch_size,
            ),
            compute_jacobians=False,
        )
        estimate = WorkspaceAnalyzer(solver, estimate_config).analyze()
        margin = 0.02
        bounds = np.stack(
            (estimate.points.min(0) - margin, estimate.points.max(0) + margin), axis=1
        )
        reference_pose = solver.forward(reference_q)
        if hasattr(reference_pose, "detach"):
            reference_pose = reference_pose.detach().cpu().numpy()
        else:
            reference_pose = np.asarray(reference_pose)
        cartesian_config = CartesianConfig(
            bounds=bounds,
            sampling=config.sampling,
            position_only=not args.full_pose,
            restarts=args.ik_restarts,
            reference_pose=reference_pose,
            reference_joints=reference_q,
        )
        result = analyzer.analyze_cartesian(cartesian_config)
    else:
        result = analyzer.analyze()
    elapsed = time.perf_counter() - started
    result.save(args.output)

    print(f"URDF: {args.urdf.resolve()}")
    print(
        f"chain: base_link -> {tip} ({solver.dof} DoF: {', '.join(solver.joint_names)})"
    )
    print(f"backend: {solver.backend}:{solver.device} ({args.dtype})")
    print(
        f"samples: {len(result.points):,} in {elapsed:.3f}s "
        f"({len(result.points) / elapsed:,.0f} samples/s)"
    )
    print(f"XYZ min: {np.array2string(result.points.min(0), precision=4)}")
    print(f"XYZ max: {np.array2string(result.points.max(0), precision=4)}")
    if result.reachable is not None:
        print(
            f"reachable: {np.count_nonzero(result.reachable):,}/"
            f"{len(result.reachable):,} ({np.mean(result.reachable):.1%})"
        )
        print(f"reference joints: {np.array2string(reference_q, precision=4)}")
        print(f"reference pose:\n{np.array2string(reference_pose, precision=5)}")
    print(f"saved: {args.output.resolve()}")

    if args.viser:
        from workspace_analyzer.visualization import ViserWorkspace

        viewer = ViserWorkspace(
            solver,
            port=args.port,
            label=f"marvin_{args.arm}_arm",
            initial_q=reference_q,
        )
        viewer.add_workspace(result)
        if args.mode == "cartesian":
            viewer.configure_cartesian_recompute(analyzer, cartesian_config)
        viewer.wait()


if __name__ == "__main__":
    main()
