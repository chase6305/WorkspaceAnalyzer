"""Select a plane and export its reachability report, samples and SVG plot.

Uses the shared CLI parser: defaults to XY/base, centered at zero, position only.
With --viser, select settings in the browser and click Compute plane reachability.
"""

from pathlib import Path

from workspace_analyzer import KinematicsSolver, RobotModel, SolverConfig
from workspace_analyzer.cli import build_parser, entrypoint, run_arguments
from workspace_analyzer.presets import (
    default_reference_joints,
    require_robot_urdf,
    select_arm_links,
)


def main(argv=None):
    parser = build_parser(demo=True)
    args = parser.parse_args(argv)
    if args.mode != "plane":
        parser.error("this demo requires --mode plane")
    output = Path(args.output)
    if output.suffix.lower() not in {".json", ".npz"}:
        parser.error("--output must end in .json or .npz; both files are written")
    args.urdf = require_robot_urdf(args.urdf, parser)
    try:
        model = RobotModel.from_urdf(args.urdf)
        args.base_link, args.tip_link = select_arm_links(
            model, args.arm, args.base_link, args.tip_link
        )
        if args.reference_joints is None:
            args.reference_joints = default_reference_joints(
                KinematicsSolver(
                    model, args.base_link, args.tip_link, SolverConfig(backend="numpy")
                )
            ).tolist()
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    args.output = str(output.with_suffix(".npz"))
    args.report = args.report or output.with_suffix(".json")
    args.plot = args.plot or output.with_suffix(".svg")
    return run_arguments(parser, args)


if __name__ == "__main__":
    entrypoint(main)
