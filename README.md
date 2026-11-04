# WorkspaceAnalyzer

[English](README.md) | [简体中文](README.zh-CN.md)

Analyze a robot's workspace directly from URDF. Sample joint configurations,
check Cartesian targets with IK, and inspect results in a browser with Viser.
The core needs only **Python 3.10+ and NumPy**; no simulator or ROS is required.

## Choose an analysis

| Question | Mode | Method |
| --- | --- | --- |
| Where can the TCP go? | `joint` | Joint samples → FK → workspace cloud |
| Which points in a box can it reach? | `cartesian` | XYZ samples → IK → reachability |
| Can it reach my task points or poses? | `targets` | Supplied targets → IK + configuration quality |
| Which parts of a work surface are reachable? | `plane` | Plane samples → IK + quality map |

All positions use metres; joint angles and RPY use radians. IK failure means
**no solution was found within the chosen budget**. Results do not include
collision checks, dynamics, or physical robot validation.

## Install

From this checkout:

```bash
python -m pip install -e .
```

Add only the capabilities you need:

| Install | Adds |
| --- | --- |
| `python -m pip install -e '.[viser]'` | Browser visualization and URDF mesh loading |
| `python -m pip install -e '.[torch]'` | PyTorch CPU/CUDA backend |
| `python -m pip install -e '.[sampling]'` | Sobol, Halton, and Latin-hypercube sampling |
| `python -m pip install -e '.[all,dev]'` | All runtimes and development tools |

CUDA requires a compatible PyTorch installation. Start with `--backend numpy`
for predictable CPU execution; `auto` selects Torch when installed.

## Run on a real robot

Marvin M6 and Dexforce W1 demos use an external `HumanoidAssets` checkout.
Place it next to this repository or set its location:

```bash
export HUMANOID_ASSETS=/path/to/HumanoidAssets
python -m pip install -e '.[viser]'
python examples/marvin_single_arm.py --backend numpy --samples 20000 --viser
```

Open the URL printed by the process, normally `http://localhost:8080`.
Use `--arm right` for the other arm. A plane scan starts with browser controls;
choose the surface and click **Compute plane reachability**:

```bash
python examples/plane_reachability.py --backend numpy --resolution 21 --viser
```

For your own model, use the main CLI. Specify the chain for a robot with multiple
arms or end effectors:

```bash
workspace-analyzer /path/to/robot.urdf \
  --base-link base_link --tip-link tool0 \
  --backend numpy --samples 20000 --output outputs/workspace.npz --viser
```

See the [real robot examples](examples/README.md) for asset paths, trajectories,
robot comparisons, and benchmarks. To check the installation without external
assets, run this small regression fixture from the repository root:

```bash
workspace-analyzer tests/fixtures/cartesian_stage.urdf \
  --backend numpy --samples 1000 --output outputs/smoke.npz
```

## Use Python

```python
import workspace_analyzer as wa

solver = wa.create_solver("robot.urdf", backend="numpy")
q = solver.joint_limits.mean(axis=1)
pose = solver.forward(q)          # (4, 4)
jacobian = solver.jacobian(q)     # (6, DoF)
ik = solver.inverse(pose, seed=q)
# Check success before using ik.positions.
assert bool(ik.success)

config = wa.WorkspaceConfig(
    sampling=wa.SamplingConfig(num_samples=10_000)
)
analyzer = wa.WorkspaceAnalyzer(solver, config)
result = analyzer.analyze()
result.save("workspace.npz")
print(result.points.min(0), result.points.max(0))
```

Batched joint inputs `(N, DoF)` produce poses `(N, 4, 4)` and Jacobians
`(N, 6, DoF)`. NumPy returns arrays; the Torch solver preserves tensors and
autograd. Analysis results are NumPy arrays for storage and reporting.
See the [Python guide](docs/python-api.md) for Cartesian analysis, task quality,
trajectories, caching, and cancellation.

## Save and inspect results

Every CLI mode can write NPZ measurements and a JSON report. For example,
scan an XY plane at Z = 0.2 m without opening a browser:

```bash
workspace-analyzer /path/to/robot.urdf --backend numpy --mode plane \
  --plane xy --constant 0.2 --center 0.4 0 0.2 --span 0.6 --resolution 21 \
  --output outputs/plane.npz --report outputs/plane.json --plot outputs/plane.svg

workspace-analyzer inspect outputs/plane.npz
```

`inspect` reads saved measurements without loading a robot or running IK.
For target and plane analyses it can also change quality gates:

```bash
workspace-analyzer inspect outputs/plane.npz \
  --min-isotropy 0.05 --min-joint-limit-margin 0.1 \
  --output outputs/reassessed.npz
```

Reachability records IK convergence. Quality gates evaluate the returned
configuration's dexterity and joint-limit margin. Changing gates reuses measured
values; it does not search for a better IK branch.

Use `workspace-analyzer --help` for all options. The
[plane guide](docs/plane-reachability.md) covers frames and orientation;
[reachability testing](docs/reachability-testing.md) covers target files and weights.

## Architecture and development

The main flow is **URDF model → batched solver → analysis → saved results / viewer**.
NumPy is the only required runtime dependency. Optional runtimes are loaded when
used; kinematics and analysis do not depend on rendering.

Read the [architecture guide](docs/architecture.md) for module responsibilities
and dependency boundaries. Advanced validation has separate guides for
[perturbations](docs/robustness-testing.md), [boundaries](docs/boundary-testing.md),
[IK stability](docs/stability-testing.md), and [candidate selection](docs/candidate-selection.md).

```bash
python -m pip install -e '.[all,dev]'
make check
```

`make check` runs lint, tests, the analytic workflow, and package builds.
`make test-robot` requires real Marvin assets. See
[Contributing](CONTRIBUTING.md) and the [testing workflow](docs/testing-workflow.md).

Apache License 2.0 — [LICENSE](LICENSE).
