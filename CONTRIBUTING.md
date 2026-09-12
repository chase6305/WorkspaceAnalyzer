# Contributing

Thank you for improving WorkspaceAnalyzer.

## Development setup

```bash
git clone <repository-url>
cd WorkspaceAnalyzer
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[all,dev]'
```

## Checks

Run the complete local gate before opening a pull request:

```bash
make check
```

Tests under `tests/test_marvin_integration.py` use an external Marvin asset and skip automatically when it is unavailable. New unit tests must not depend on files outside this repository.

Prioritize examples backed by real robot URDFs: Marvin workspace, plane scans,
trajectories, dexterity, and W1/Marvin comparisons. Synthetic fixtures and internal
benchmarks support algorithm regression; avoid expanding them into additional
user-facing demos. See [the example guide](examples/README.md).

Run `make test-robot` for Marvin left/right Jacobians, the 90° elbow reference,
mesh loading, and all four Cartesian trajectory shapes. Unlike the general suite,
this target fails when Marvin assets are missing. Set `HUMANOID_ASSETS` when the
asset checkout is not adjacent to this repository. These are URDF-based numerical
checks, not physical hardware or collision-clearance validation.

`make check` runs lint/format checks, pytest, the analytic robot workflow, and
package builds. Use `make test`, `make test-workflow`, `make lint`, or `make build`
for individual steps. Pytest writes `outputs/test-results.xml`; the robot workflow
writes JSON/Markdown reports and NPZ diagnostics under `outputs/test-workflow`.
Workflow acceptance failures return a nonzero exit status while retaining results.
See [the testing workflow](docs/testing-workflow.md) for robot-specific runs.

CI keeps a NumPy-only runtime matrix on Python 3.10/3.12 and a separate Python 3.12
job with Torch CPU, SciPy, Viser, and trimesh. The latter verifies imports and local
socket support before running tests. Both jobs retain JUnit and robot reports.
To prepare the optional CPU profile locally, use the CPU distribution described
in the [official PyTorch installation guide](https://pytorch.org/get-started/locally/):

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev,sampling,viser]'
make check
```

Skipped optional-runtime tests in a minimal environment are expected; the optional
CI job is responsible for actually running them. CUDA checks require an environment
with a suitable device and are not covered by the CPU job.

## Design expectations

- Keep the core importable with NumPy alone; optional features must fail with actionable installation messages.
- Keep NumPy and Torch behavior consistent, including input shapes and result semantics.
- Preserve batch dimensions and avoid per-sample Python loops in compute-heavy paths.
- Check IK `success` and `residual` explicitly in examples.
- Keep rendering outside kinematics and analysis code.
- Add numerical tests for changes to FK, Jacobians, rotation errors, or IK.

## Commit and pull-request scope

Prefer focused changes with a clear test. Do not commit generated workspaces, virtual environments, build outputs, robot assets with unclear licensing, or benchmark claims without the command and hardware context needed to reproduce them.
