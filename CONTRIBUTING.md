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

Run the checks that match your change before opening a pull request:

```bash
pytest -q
ruff check src tests examples
ruff format --check src tests examples
python -m build
```

Tests under `tests/test_marvin_integration.py` use an external Marvin asset and skip automatically when it is unavailable. New unit tests must not depend on files outside this repository.

## Design expectations

- Keep the core importable with NumPy alone; optional features must fail with actionable installation messages.
- Keep NumPy and Torch behavior consistent, including input shapes and result semantics.
- Preserve batch dimensions and avoid per-sample Python loops in compute-heavy paths.
- Check IK `success` and `residual` explicitly in examples.
- Keep rendering outside kinematics and analysis code.
- Add numerical tests for changes to FK, Jacobians, rotation errors, or IK.

## Commit and pull-request scope

Prefer focused changes with a clear test. Do not commit generated workspaces, virtual environments, build outputs, robot assets with unclear licensing, or benchmark claims without the command and hardware context needed to reproduce them.
