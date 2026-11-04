"""Compare separate and fused DLS posture projections on fixed batched inputs.

This measures the update kernel only, excluding FK, Jacobian construction and
IK convergence. Run with OPENBLAS_NUM_THREADS=1 for a controlled NumPy baseline.
"""

import argparse
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np

from workspace_analyzer._atomic import atomic_write_text
from workspace_analyzer.kinematics import _biased_dls, _dls


def benchmark(*, batch_size=4096, repeats=9, backend="numpy", dtype="float64"):
    if batch_size < 1 or repeats < 1:
        raise ValueError("batch_size and repeats must be positive")
    rng = np.random.default_rng(26)
    report = {
        "scope": "DLS update kernel; excludes FK, Jacobian construction and full IK",
        "backend": backend,
        "dtype": dtype,
        "batch_size": batch_size,
        "repeats": repeats,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cases": [],
    }
    if backend == "torch":
        import torch

        torch.set_num_threads(1)
        report.update(torch=torch.__version__, torch_threads=1, device="cpu")

    for rows in (3, 6):
        arrays = [
            rng.normal(size=shape).astype(dtype)
            for shape in [
                (batch_size, rows, 7),
                (batch_size, rows),
                (batch_size, 7),
                (batch_size, 7),
            ]
        ]
        if backend == "torch":
            arrays = [torch.as_tensor(value) for value in arrays]
        jacobian, error, center, posture = arrays
        center, posture = center * 0.02, posture * 0.03
        for preferences in (1, 2):

            def separate():
                value = _dls(jacobian, error, 1e-3, backend)
                for bias in (posture,) if preferences == 1 else (center, posture):
                    value = (
                        value
                        + bias
                        - _dls(
                            jacobian,
                            (jacobian @ bias[..., None])[..., 0],
                            1e-3,
                            backend,
                        )
                    )
                return value

            def fused():
                bias = posture if preferences == 1 else center + posture
                return _biased_dls(jacobian, error, 1e-3, backend, bias)

            old, new = separate(), fused()
            if backend == "torch":
                old, new = old.numpy(), new.numpy()
            tolerance = 3e-4 if dtype == "float32" else 1e-10
            np.testing.assert_allclose(old, new, rtol=tolerance, atol=tolerance)
            timings = {"separate": [], "fused": []}
            callbacks = [("separate", separate), ("fused", fused)]
            for iteration in range(repeats):
                # Alternate order to reduce systematic warm-cache/order bias.
                for name, callback in callbacks[:: 1 if iteration % 2 == 0 else -1]:
                    start = perf_counter()
                    callback()
                    timings[name].append(perf_counter() - start)
            medians = {key: float(np.median(value)) for key, value in timings.items()}
            report["cases"].append(
                {
                    "task_rows": rows,
                    "posture_preferences": preferences,
                    "separate_median_s": medians["separate"],
                    "fused_median_s": medians["fused"],
                    "speedup": medians["separate"] / medians["fused"],
                    "max_absolute_difference": float(np.max(np.abs(old - new))),
                }
            )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--backend", choices=("numpy", "torch"), default="numpy")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = benchmark(
            batch_size=args.batch_size,
            repeats=args.repeats,
            backend=args.backend,
            dtype=args.dtype,
        )
        encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            atomic_write_text(args.output, encoded)
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        parser.error(str(exc))
    print(encoded, end="")


if __name__ == "__main__":
    main()
