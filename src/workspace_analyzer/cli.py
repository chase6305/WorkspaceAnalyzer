from __future__ import annotations

import argparse

from .analyzer import WorkspaceAnalyzer, WorkspaceConfig
from .kinematics import create_solver
from .sampling import SamplingConfig, SamplingStrategy


def main():
    parser = argparse.ArgumentParser(description="Analyze a URDF robot workspace")
    parser.add_argument("urdf")
    parser.add_argument("--base-link")
    parser.add_argument("--tip-link")
    parser.add_argument("--backend", choices=("auto", "numpy", "torch"), default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument(
        "--strategy", choices=[x.value for x in SamplingStrategy], default="random"
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--output")
    parser.add_argument("--viser", action="store_true")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    solver = create_solver(
        args.urdf,
        base_link=args.base_link,
        tip_link=args.tip_link,
        backend=args.backend,
        device=args.device,
    )
    cfg = WorkspaceConfig(
        SamplingConfig(SamplingStrategy(args.strategy), args.samples, args.batch_size)
    )
    result = WorkspaceAnalyzer(solver, cfg).analyze()
    if args.output:
        result.save(args.output)
    backend_label = f"{solver.backend}:{solver.device}"
    print(f"{len(result.points)} poses | {solver.dof} DoF | {backend_label}")
    print(f"bounds: {result.points.min(0)} .. {result.points.max(0)}")
    if args.viser:
        from .visualization import ViserWorkspace

        view = ViserWorkspace(solver, port=args.port)
        view.add_workspace(result)
        view.wait()


if __name__ == "__main__":
    main()
