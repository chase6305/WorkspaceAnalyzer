"""Run a reproducible set of single-arm reachability experiments.

This convenience demo runs joint-sampled workspace, position-only Cartesian IK,
and fixed-orientation Cartesian IK with the same URDF and reference pose.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--urdf", type=Path, required=True)
    p.add_argument("--arm", choices=("left", "right"), default="left")
    p.add_argument("--samples", type=int, default=10000)
    p.add_argument("--viser", action="store_true")
    p.add_argument(
        "--output-dir", type=Path, default=Path("outputs/reachability_suite")
    )
    args = p.parse_args()
    common = [
        sys.executable,
        str(Path(__file__).with_name("marvin_single_arm.py")),
        "--urdf",
        str(args.urdf),
        "--arm",
        args.arm,
        "--samples",
        str(args.samples),
    ]
    jobs = [
        ("joint_workspace", []),
        ("position_ik", ["--mode", "cartesian"]),
        ("fixed_pose_ik", ["--mode", "cartesian", "--full-pose"]),
    ]
    for index, (name, extra) in enumerate(jobs):
        cmd = common + extra + ["--output", str(args.output_dir / f"{name}.npz")]
        if args.viser:
            cmd += ["--viser", "--port", str(8080 + index)]
        print("RUN", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
