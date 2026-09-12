"""Compare compressed and uncompressed NPZ I/O on deterministic synthetic data.

Timings include atomic writes and validated loads, but exclude input construction
and result comparison. Repeated local reads may benefit from the OS page cache.
"""

import argparse
import json
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np

from workspace_analyzer import AnalysisResult


def benchmark(samples=50000, repeats=3):
    if samples < 1 or repeats < 1:
        raise ValueError("samples and repeats must be positive")
    rng = np.random.default_rng(42)
    result = AnalysisResult(
        points=rng.normal(size=(samples, 3)),
        joint_positions=rng.normal(size=(samples, 7)),
        manipulability=rng.random(samples),
        metadata={"scope": "synthetic I/O benchmark", "seed": 42},
        reachable=rng.random(samples) > 0.2,
        residual=rng.random(samples),
        metrics={
            name: rng.random(samples) for name in ("isotropy", "joint_limit_margin")
        },
    )
    report = {"samples": samples, "repeats": repeats}
    with tempfile.TemporaryDirectory(prefix="workspace-result-io-") as directory:
        for compressed in (True, False):
            path = Path(directory) / "result.npz"
            writes, reads = [], []
            for _ in range(repeats):
                start = perf_counter()
                result.save(path, compressed=compressed)
                writes.append(perf_counter() - start)
                start = perf_counter()
                loaded = AnalysisResult.load(path)
                reads.append(perf_counter() - start)
                for name in (
                    "points",
                    "joint_positions",
                    "manipulability",
                    "reachable",
                    "residual",
                ):
                    np.testing.assert_array_equal(
                        getattr(result, name), getattr(loaded, name)
                    )
                for name in result.metrics:
                    np.testing.assert_array_equal(
                        result.metrics[name], loaded.metrics[name]
                    )
                assert loaded.metadata == result.metadata
            report["compressed" if compressed else "uncompressed"] = {
                "bytes": path.stat().st_size,
                "write_median_s": float(np.median(writes)),
                "read_median_s": float(np.median(reads)),
                "roundtrip_equal": True,
            }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.samples < 1 or args.repeats < 1:
        parser.error("samples and repeats must be positive")
    encoded = json.dumps(benchmark(args.samples, args.repeats), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
