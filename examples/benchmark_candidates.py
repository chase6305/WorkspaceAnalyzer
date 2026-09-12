"""Benchmark offline candidate processing using repeated analytic measurements.

Run from a source checkout with PYTHONPATH=src. Timing excludes IK and study
construction; repeated targets measure processing cost, not robot coverage.
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
    IKStabilityResult,
    IKTrial,
    ReachabilityConfig,
    analyze_ik_stability,
    create_solver,
)
from workspace_analyzer.selection import _validated_candidates


def benchmark(*, targets=30000, trials=12, repeats=5, measure_memory=False):
    if targets < 1 or trials < 2 or repeats < 1:
        raise ValueError("targets/repeats must be positive and trials >= 2")
    solver = create_solver(
        Path(__file__).resolve().parents[1] / "tests/fixtures/redundant_stage.urdf",
        backend="numpy",
    )
    base = analyze_ik_stability(
        solver,
        [[0, 0, 0], [0.4, 0, 0], [3, 0, 0]],
        [IKTrial("a", seed=[0, 0, 0, 0]), IKTrial("b", seed=[0.3, -0.3, 0, 0])],
        ReachabilityConfig(
            restarts=1, rescue_restarts=0, minimum_joint_limit_margin=0.5
        ),
    )
    rows = np.arange(targets) % 3
    results = []
    for i in range(trials):
        source = base.results[i % 2]
        results.append(
            replace(
                source,
                points=source.points[rows],
                joint_positions=source.joint_positions[rows],
                manipulability=source.manipulability[rows],
                reachable=source.reachable[rows],
                residual=source.residual[rows],
                metrics={key: value[rows] for key, value in source.metrics.items()},
            )
        )
    study = IKStabilityResult(
        tuple(f"trial_{i}" for i in range(trials)), tuple(results)
    )
    report = {"targets": len(rows), "trials": trials}
    for name, callback in [
        ("validation", lambda: _validated_candidates(study)),
        ("selection", study.select_solutions),
        ("diversity", study.summarize_diversity),
        ("study_report", study.to_dict),
    ]:
        durations = []
        for _ in range(repeats):
            start = perf_counter()
            value = callback()
            durations.append(perf_counter() - start)
        if name == "selection":
            digest = hashlib.sha256()
            for array in [
                value.points,
                value.joint_positions,
                value.reachable,
                value.residual,
                value.manipulability,
                *value.metrics.values(),
            ]:
                digest.update(array.tobytes())
            digest.update(json.dumps(value.metadata, sort_keys=True).encode())
            report["selection_digest"] = digest.hexdigest()
        if name in {"diversity", "study_report"}:
            report[name + "_digest"] = hashlib.sha256(
                json.dumps(value, sort_keys=True).encode()
            ).hexdigest()
        report[name + "_median_s"] = float(np.median(durations))
        if measure_memory:
            # Measure a separate invocation so tracing does not affect timings.
            # This is traced allocation, not total process RSS or input storage.
            tracemalloc.start()
            try:
                measured = callback()
                report[name + "_peak_traced_bytes"] = tracemalloc.get_traced_memory()[1]
                del measured
            finally:
                tracemalloc.stop()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=int, default=30000)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--measure-memory", action="store_true")
    args = parser.parse_args()
    if args.targets < 1 or args.trials < 2 or args.repeats < 1:
        parser.error("targets/repeats must be positive and trials >= 2")
    report = benchmark(
        targets=args.targets,
        trials=args.trials,
        repeats=args.repeats,
        measure_memory=args.measure_memory,
    )
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
