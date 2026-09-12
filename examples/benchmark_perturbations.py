"""Benchmark perturbation reports on repeated analytic measurements.

Timing excludes input construction and IK. Peak traced allocation excludes the
already-built study and result; it is not total process memory.
"""

import argparse
import hashlib
import json
import tracemalloc
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

from workspace_analyzer import (
    PosePerturbations,
    ReachabilityConfig,
    WorkspaceAnalyzer,
    create_solver,
)


def benchmark(references=20000, repeats=5):
    if references < 1 or repeats < 1:
        raise ValueError("references and repeats must be positive")
    pose = np.eye(4)[None]
    pose[0, 0, 3] = 0.95
    small = PosePerturbations(pose, translation_m=0.1, rotation_rad=0.1)
    base = WorkspaceAnalyzer(
        create_solver(
            Path(__file__).resolve().parents[1] / "tests/fixtures/cartesian_stage.urdf",
            backend="numpy",
        )
    ).analyze_targets(
        small.targets,
        ReachabilityConfig(
            position_only=False,
            restarts=1,
            rescue_restarts=0,
            minimum_joint_limit_margin=0.1,
        ),
    )
    study = PosePerturbations(
        np.repeat(pose, references, axis=0), translation_m=0.1, rotation_rad=0.1
    )
    rows = np.arange(len(study.targets)) % len(small.targets)
    result = replace(
        base,
        points=study.targets[:, :3, 3].copy(),
        target_poses=study.targets.copy(),
        joint_positions=base.joint_positions[rows],
        manipulability=base.manipulability[rows],
        reachable=base.reachable[rows],
        residual=base.residual[rows],
        metrics={k: v[rows] for k, v in base.metrics.items()},
    )
    times = []
    for _ in range(repeats):
        start = perf_counter()
        report = study.summarize(result)
        times.append(perf_counter() - start)
    tracemalloc.start()
    try:
        measured = study.summarize(result)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert measured == report
    record = dict(
        references=references,
        variants=len(study.variant_names),
        median_s=float(np.median(times)),
        peak_traced_bytes=peak,
        report_digest=hashlib.sha256(
            json.dumps(report, sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
    )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=int, default=20000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.references < 1 or args.repeats < 1:
        parser.error("references and repeats must be positive")
    encoded = json.dumps(benchmark(args.references, args.repeats), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
