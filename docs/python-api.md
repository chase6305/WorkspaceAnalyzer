# Python API guide

[Home](../README.md) · [中文首页](../README.zh-CN.md) · [Architecture](architecture.md)

Examples below use one solver and its selected serial chain. Replace `robot.urdf`
and, for a branched robot, supply `base_link` and `tip_link`.
Positions are in metres; revolute joint angles and RPY are in radians.

## Solver and batch shapes

```python
import numpy as np
from workspace_analyzer import create_solver

solver = create_solver("robot.urdf", backend="numpy", dtype="float64")
q_reference = solver.joint_limits.mean(axis=1)
reference_pose = solver.forward(q_reference)

q_batch = np.stack([q_reference, q_reference])
poses, jacobians = solver.forward_with_jacobian(q_batch)
ik = solver.inverse(poses, seed=q_reference, restarts=4)
print(ik.success, ik.residual)
```

| Operation | Single input → output | Batched input → output |
| --- | --- | --- |
| `forward` | `(DoF,)` → `(4, 4)` | `(N, DoF)` → `(N, 4, 4)` |
| `jacobian` | `(DoF,)` → `(6, DoF)` | `(N, DoF)` → `(N, 6, DoF)` |
| `inverse` | `(4, 4)` → positions `(DoF,)` | `(N, 4, 4)` → positions `(N, DoF)` |

A one-row batch stays a batch. Torch execution returns tensors on the selected
device and preserves autograd; use `.detach().cpu().numpy()` for reporting.
`forward_with_jacobian` shares the chain traversal between FK and the Jacobian.

Without explicit links, the URDF must have one root and the solver selects the
leaf with the most active joints. Fixed transforms stay in the chain;
revolute, continuous, and prismatic joints are supported. Mimic coupling and
floating joints are rejected. Use a separate solver for each arm or end effector.

IK uses bounded damped least squares. `position_only=True` ignores target rotation.
`restarts` runs parallel initial guesses; `rescue_restarts` and `rescue_rounds`
retry failed targets. Always inspect `success` and `residual`: a failed numerical
solve does not prove that a target is geometrically unreachable. Returned residuals
correspond to the returned joints, including when the iteration budget is exhausted.
For float32 on NumPy or Torch CPU/CUDA, the small DLS system is solved in float64
to retain damping near singularities; returned arrays and tensors stay float32.

## Joint workspace

```python
from workspace_analyzer import SamplingConfig, WorkspaceAnalyzer, WorkspaceConfig

sampling = SamplingConfig(num_samples=10_000, batch_size=1024, seed=42)
analyzer = WorkspaceAnalyzer(solver, WorkspaceConfig(sampling=sampling))
workspace = analyzer.analyze()
```

Joint samples follow URDF limits and map to XYZ with FK. By default, each sample
also has translational manipulability, minimum singular value, isotropy, and
joint-limit margin. Set `compute_jacobians=False` in `WorkspaceConfig` when only
positions are needed.

`random`, `uniform`, and `gaussian` sampling need only NumPy. `sobol`, `halton`,
and `lhs` need the sampling extra. A uniform grid returns the requested prefix
of the grid; a truncated grid can cover axes unevenly.

## Cartesian volume, plane, or line

```python
from workspace_analyzer import CartesianConfig

cartesian_config = CartesianConfig(
    bounds=np.array([[-0.7, 0.7], [-0.3, 0.9], [0.4, 1.8]]),
    sampling=sampling,
    position_only=True,
    restarts=4,
    reference_joints=q_reference,
    reference_pose=reference_pose,
)
reachability = analyzer.analyze_cartesian(cartesian_config)
print(reachability.metadata["success_rate"])
```

Equal endpoints fix an axis. For example, `[0.2, 0.2]` on Z samples a plane;
fix two axes for a line. Fixed axes do not consume sampling dimensions.
Use `position_only=False` to constrain every target to `reference_pose`'s rotation.
Sampling uses `SamplingConfig.seed`. Cartesian IK uses `SolverConfig.random_seed`,
supplied as `random_seed` to `create_solver`; targets and planes instead use
`ReachabilityConfig.random_seed`. Sampling and IK seeds can be varied independently.

For named frames, tilted planes, regular grids, quality measurements, and SVG
exports, use the plane API:

```python
from workspace_analyzer import PlaneConfig, ReachabilityConfig, analyze_plane

plane = analyze_plane(
    solver,
    PlaneConfig(plane="xy", frame="base", center=(0.4, 0, 0.2), span=0.6, resolution=21),
    ReachabilityConfig(position_only=True, restarts=4),
    seed=q_reference,
)
```

See the [plane guide](plane-reachability.md) for coordinate-frame conventions.

## Supplied targets and quality

```python
from workspace_analyzer import ReachabilityConfig

targets = np.array([[0.4, 0, 0.2], [0.5, 0, 0.2]])
assessment = analyzer.analyze_targets(
    targets,
    ReachabilityConfig(minimum_isotropy=0.05, minimum_joint_limit_margin=0.1),
    seed=q_reference,
)
print(assessment.reachable)                 # IK convergence
print(assessment.metrics["quality_pass"])   # IK success and quality gates
print(assessment.metadata["assessment"])
```

Targets may be XYZ rows or `(N, 4, 4)` poses. Use `position_only=False` for pose
constraints. Optional `base_from_targets` transforms targets into the solver base;
optional `weights` controls task-point coverage statistics. These point weights
are separate from `dexterity_weights`, which scales Jacobian task directions.

`minimum_singular_value`, `minimum_isotropy`, `minimum_joint_limit_margin`, and
`require_full_rank` evaluate the **returned configuration**. Rank follows
`dexterity_task`: `position` and `rotation` require rank 3; `pose` requires rank 6.
IK constraints are independent: `position_only=False` still defaults to position
dexterity. Set `dexterity_task="pose"` for six-dimensional quality gates. Compact
SVD alone does not prove control over every task direction on robots with fewer joints.

Change gates without running IK again:

```python
reassessed = assessment.reassess_quality(minimum_joint_limit_margin=0.2)
```

The Python reassessment call replaces all gates; unspecified gates are disabled.
The `inspect` CLI preserves saved gates unless `--reset-quality-gates` is supplied.
Neither operation searches for another IK branch. To choose among already measured
configurations, see [candidate selection](candidate-selection.md).

## Dexterity and trajectories

```python
quality = solver.dexterity(q_batch, task="position", weights=[0.5, 0.5, 1.0])
print(quality.minimum_singular_value, quality.joint_limit_margin)

joint_span = np.ptp(solver.joint_limits, axis=1)
q_path = q_reference + np.linspace(0, 0.05, 40)[:, None] * joint_span
trajectory = solver.solve_trajectory(
    solver.forward(q_path),
    seed=q_reference,
    position_only=False,
    dt=0.02,
    failure_restarts=8,
)
print(trajectory.summary())
```

The geometric Jacobian maps joint velocity to TCP linear/angular velocity:
`[v, omega] = J(q) @ q_dot`. Position IK uses the first three rows. Singular values
describe local dexterity; joint-limit margin is a separate configuration measure.
Use `task="position"` or `"rotation"` to keep linear and angular units separate.
Non-negative directional weights must include a positive value. Rank-deficient
configurations have infinite condition numbers; exclude them from finite losses.

Trajectory IK uses the previous successful configuration as its seed and reports
joint jumps, velocity, acceleration, dexterity, and velocity-limit violations.
Supply `dt` or explicit `timestamps` for time-based metrics. Continuous joints
are unwrapped; failed frames invalidate adjacent continuity segments. Closed-loop
refinement projects joints back onto target poses. For interactive trajectory
examples, see [the example guide](../examples/README.md).

## Results, caching, and cancellation

```python
from threading import Event
from workspace_analyzer import AnalysisResult, ResultCache

workspace.save("workspace.npz")
restored = AnalysisResult.load("workspace.npz")
cache = ResultCache(".cache/workspace-analyzer")
cancel = Event()
workspace = analyzer.analyze(cache=cache, cancel_event=cancel, progress_callback=print)
```

`AnalysisResult` stores points, joint configurations, metadata, and optional
reachability, residuals, quality metrics, and full target poses. NPZ archives use
versioned JSON plus numeric arrays and load with `allow_pickle=False`. Saves are
atomic. Use `compressed=False` for faster, larger local files; `ResultCache` accepts
the same option.

Cache keys include the parsed model, chain, solver settings, and analysis inputs.
Changing a source URDF after constructing a solver does not alter that solver;
reconstruct it to load the new model. Quality-only changes can reuse target
measurements. Cache encoding does not affect the key.

Analysis, IK, and trajectories accept `cancel_event`; set it from another thread
to raise `AnalysisCancelled` at the next batch or iteration boundary. Progress
callbacks receive fractions from 0 to 1. A single NumPy/Torch operation finishes
before cancellation is checked. IK batch memory grows with `batch_size * restarts`;
reduce either value when memory is limited.

## Visualization and further validation

```python
from workspace_analyzer.visualization import ViserWorkspace

with ViserWorkspace(solver, port=8080, initial_q=q_reference) as viewer:
    viewer.add_workspace(workspace, color_metric="isotropy")
    viewer.add_trajectory(trajectory)
    viewer.wait()
```

Viser supports meshes and primitives, live joints, TCP frames, workspace colors,
and trajectory playback. Large clouds are downsampled for display only. It runs
a web service and does not need a desktop display server.

Further guides cover [target assessment](reachability-testing.md),
[orientation coverage](reachability-testing.md), [perturbations](robustness-testing.md),
[translation boundaries](boundary-testing.md), [IK stability](stability-testing.md),
[candidate diversity](candidate-diversity.md), and [test acceptance](testing-workflow.md).
Use float64 for numerical validation; benchmark float32 on the target hardware
before choosing it for large CUDA workloads.
