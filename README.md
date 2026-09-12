# WorkspaceAnalyzer

**Examples focus on real robot models.** Start with the [Marvin / W1 example guide](examples/README.md)
for workspace, plane reachability, trajectory, and comparison demos. Synthetic models primarily support algorithm regression tests.

[English](README.md) | [简体中文](README.zh-CN.md)

![WorkspaceAnalyzer architecture](docs/assets/workspace-analyzer-overview.png)

WorkspaceAnalyzer is a simulator-independent Python toolkit for robot workspace analysis. It builds a general serial-chain solver directly from URDF, provides batched FK, geometric Jacobian, and numerical IK on NumPy or PyTorch, and renders robot geometry and reachability results with Viser.

The project separates robot models, compute backends, analysis, and visualization. Its core package depends only on NumPy and does not require EmbodiChain, Isaac Lab, ROS, or a specific robot wrapper.

## Features

- Automatic URDF root, tip, active-joint, fixed-transform, axis, and limit handling.
- One batched API for NumPy CPU and PyTorch CPU/CUDA.
- Forward kinematics, analytic geometric Jacobians, and bounded damped-least-squares IK.
- Parallel multi-start IK with per-target best-solution selection.
- Joint-space analysis: sample joints and map them to Cartesian space with FK.
- Cartesian-space analysis: sample XYZ targets and classify them with batched IK.
- Assess supplied task points/poses with weighted coverage, dexterity gates,
  task-rank checks, and JSON reports.
- Random, grid, Gaussian, Halton, Sobol, and Latin-hypercube sampling.
- Translational manipulability metrics.
- Viser support for URDF meshes and primitives, live joint controls, skeletons, TCP frames, and workspace clouds.
- Minimal dependencies with optional extras for Torch, Viser, and SciPy.

## Requirements

- Python 3.10 or newer
- NumPy 1.24 or newer
- Optional: PyTorch 2.1+, SciPy 1.10+, Viser, and trimesh

CUDA availability is controlled by the installed PyTorch build. This project intentionally does not pin a CUDA wheel; install the PyTorch build appropriate for the target driver before installing the Torch extra.

## Installation

```bash
# NumPy-only core
python -m pip install -e .

# Choose only the capabilities you need
python -m pip install -e '.[torch]'
python -m pip install -e '.[viser]'
python -m pip install -e '.[sampling]'

# Complete runtime and development environment
python -m pip install -e '.[all,dev]'
```

## Quick start

### Construct a solver

```python
from workspace_analyzer import create_solver

solver = create_solver(
    "robot.urdf",
    base_link="base_link",  # optional for a single-root URDF
    tip_link="tool0",       # optional; defaults to the longest active chain
    backend="torch",        # auto, numpy, or torch
    device="cuda",          # auto, cpu, cuda, or cuda:1
    dtype="float32",
)

poses = solver.forward(q_batch)                 # (N, 4, 4)
jacobians = solver.jacobian(q_batch)            # (N, 6, DoF)
ik = solver.inverse(target_poses, seed=q_seed)
robust_ik = solver.inverse(target_poses, restarts=4)

if not bool(ik.success.all()):
    print("Some IK targets did not converge", ik.residual)
```

Use `position_only=True` for position reachability or robots with fewer than six task-space degrees of freedom. Multi-start IK combines all seeds into one batch and selects the lowest-residual result for each target. Offline Cartesian analysis can additionally retry only failed targets with fresh deterministic seeds through `rescue_restarts` and `rescue_rounds`. For real-time tracking, using the previous joint state as `seed` is usually faster than rescue rounds.

Rescue attempts run iteratively without accumulating recursive solver frames.
Without autograd, completed attempts release their intermediate arrays.
`IKResult.iterations` counts iteration-loop
passes across the initial solve and all executed rescue attempts; it is not multiplied
by the number of parallel seeds.

### Joint-space workspace analysis

```python
from workspace_analyzer import WorkspaceAnalyzer, WorkspaceConfig

result = WorkspaceAnalyzer(solver, WorkspaceConfig()).analyze()
result.save("workspace.npz")
```

This mode follows `joint samples -> FK -> Cartesian points`. When enabled, the analyzer also computes a translational manipulability score for each point.

### Cartesian-space reachability analysis

Equal bounds fix an axis: `bounds=[[-0.5, 0.5], [-0.5, 0.5], [0.2, 0.2]]`
samples a plane at Z = 0.2 m. Fix two axes for a line, or all three for repeated
evaluation of one point. All samplers operate only on varying axes, so a uniform
plane grid does not waste samples on repeated fixed-axis levels. The same bounds
work in the CLI: `--bounds -0.5 0.5 -0.5 0.5 0.2 0.2`.

```python
import numpy as np
from workspace_analyzer import CartesianConfig, SamplingConfig, WorkspaceAnalyzer

q_reference = solver.joint_limits.mean(axis=1)
config = CartesianConfig(
    bounds=np.array([
        [-0.7, 0.7],   # x min/max
        [-0.3, 0.9],   # y min/max
        [0.4, 1.8],    # z min/max
    ]),
    sampling=SamplingConfig(num_samples=20_000, batch_size=1024),
    position_only=True,
    restarts=4,
    reference_joints=q_reference,
    reference_pose=solver.forward(q_reference),
)
result = WorkspaceAnalyzer(solver).analyze_cartesian(config)
print(result.metadata["success_rate"])
result.save("cartesian_reachability.npz")
restored = type(result).load("cartesian_reachability.npz")
```

This mode follows `XYZ targets -> IK -> reachable/unreachable classification`. Results contain all query points, best joint solutions, Boolean reachability flags, and IK residuals.

### Test supplied targets and configuration quality

```python
from workspace_analyzer import ReachabilityConfig

assessment = WorkspaceAnalyzer(solver).analyze_targets(
    targets,  # (N, 3), or (N, 4, 4) with position_only=False
    ReachabilityConfig(minimum_isotropy=0.05, minimum_joint_limit_margin=0.1),
)
print(assessment.metadata["assessment"])
print(assessment.reachable)
print(assessment.metrics["quality_pass"])
```

This evaluates the selected IK configuration and separates IK failure from quality
rejection. Metrics include task rank, condition number, and separate translation/
rotation errors. Failed targets skip dexterity computation. `require_full_rank=True`
requires rank 3 for position/rotation tasks or rank 6 for pose tasks; compact SVD
alone does not establish control over every task direction. No collision checks
or additional search for a higher-quality IK branch are performed.

The CLI supports headerless XYZ CSV, NPY targets, and NPZ bundles with `targets`
plus optional `weights` and `base_from_targets`:

```bash
workspace-analyzer robot.urdf --backend numpy --mode targets \
  --targets targets.npz --batch-size 1024 \
  --min-isotropy 0.05 --min-joint-limit-margin 0.1 \
  --output assessment.npz --report assessment.json
```

Use `--full-pose` for pose targets. Full target poses are retained in result files.
See the [testing guide](docs/reachability-testing.md) for units, weighting, cache
behavior, repeatability, and a runnable analytic robot example.

Use `assessment.reassess_quality(minimum_joint_limit_margin=0.2)` to replace all
quality gates using saved measurements, without repeating IK. Unspecified gates
are disabled. Cached runs also reuse measurements when only gates or task weights
change. For full-pose results, `assessment.orientation_coverage(position_ids)`
reports per-position IK and quality coverage of the supplied orientation samples.
The [complete testing workflow](docs/testing-workflow.md) adds known-target and
out-of-range checks, local orientation tests, quality sweeps, and saved reports.

Add `--robustness-test` to `examples/reachability_workflow.py` for paired XYZ
translation and three-axis rotation perturbations. `PosePerturbations` also exposes
the generated targets and offline summaries in Python. Reports identify per-axis
IK/quality losses and gains, worst solved quality, and tasks passing every variant.
See the [perturbation guide](docs/robustness-testing.md) and
[further validation options](docs/validation-options.md) for acceptance criteria.

`--boundary-test` adds fixed-orientation translation brackets, adaptive bisection,
and final failed-endpoint rechecks. The `refine_translation_boundary` API accepts
arbitrary segment directions and retains unresolved states, measured endpoints,
and query traces. See [boundary testing](docs/boundary-testing.md) for resolution,
budgets, and the distinction between numerical brackets and geometric boundaries.

`--stability-test` compares the same targets across iteration budgets and restart
seeds. Select pose, perturbation, or boundary targets with `--stability-targets`.
The `IKTrial` / `analyze_ik_stability` API also accepts explicit initial joint
configurations and supports saved studies and offline quality reassessment.
See [IK stability testing](docs/stability-testing.md) for disagreement thresholds,
per-target gains/losses, and quality ranges across the returned configurations.

`study.select_solutions(objective="joint_limit_margin")` selects measured candidates
offline, prioritizing quality gates and retaining each target's source trial.
The workflow enables this with `--select-ik-solutions joint_limit_margin` alongside
`--stability-test`. See [candidate selection](docs/candidate-selection.md) for ranking,
acceptance, and the distinction between reassessing and reselecting configurations.

Use `make_ik_seed_trials` or `--stability-initial-guesses K` to add reproducible
joint initial guesses. `study.summarize_diversity()` / `--candidate-diversity`
reports new and repeated configurations using joint-specific tolerances and
continuous-angle wrapping. Grouping preserves all candidates for quality selection.
See [candidate diversity](docs/candidate-diversity.md) for units and counting rules.

### Dexterity and joint continuity

Reachability is binary; dexterity describes the quality of a reachable configuration.
The batched API reports Jacobian singular values, Yoshikawa manipulability, minimum
singular value, isotropy, condition number, and normalized distance from joint limits:

```python
quality = solver.dexterity(q_batch, task="position")
vertical_priority = solver.dexterity(
    q_batch, task="position", weights=[0.5, 0.5, 1.0]
)
safe = (quality.minimum_singular_value > 0.05) & (
    quality.joint_limit_margin > 0.15
)
```

`task="position"` and `task="rotation"` avoid mixing linear and angular units. Use
`task="pose"` only when the unweighted 6D Jacobian is meaningful for the application.
Use non-negative `weights` to emphasize task-space directions; all-zero weights are
rejected because they do not define a task.
Singular values use the compact SVD (`min(task rows, DoF)` values). Condition numbers
use a relative numerical rank threshold, so uniformly rescaling all task weights
preserves conditioning within numerical precision.
Torch guards divisions in inactive metric branches, so zero Jacobians and masked
infinite condition numbers do not introduce division-generated NaNs in backward.
Condition numbers remain infinite for numerically rank-deficient configurations;
exclude them when forming a finite loss. See the
[gradient validation record](docs/optimization-2026-09-08-round16.md).
Joint-space workspace results store `minimum_singular_value`, `isotropy`, and
`joint_limit_margin` arrays in `result.metrics`; Viser's `color_metric` argument can
color a cloud by any of them.

Continuity must be evaluated on an ordered path, not an unordered workspace cloud:

```python
trajectory = solver.solve_trajectory(
    target_poses,
    seed=q_start,
    position_only=False,
    failure_restarts=8,
    dt=0.01,
)
print(trajectory.max_joint_jump)
print(trajectory.velocity_violation.any())
print(trajectory.summary())
```

The trajectory solver warm-starts each frame from the previous successful solution,
uses multi-start only after a failure, unwraps continuous joints, and reports joint
deltas, velocity, acceleration, and URDF velocity-limit violations. Segments adjacent
to a failed IK frame are marked `NaN` and must not be treated as executable motion.
Continuous joints wrap during IK instead of stopping at ±pi; trajectory results
unwrap successful frames so failed solutions cannot introduce spurious revolutions.
Per-frame minimum singular value and joint-limit margin expose singular or
limit-constrained portions of an otherwise continuous path. Pass `timestamps=[...]`
instead of `dt` for nonuniform trajectory timing.
Pass `batch_size=1024` (the default) to bound closed-loop reprojection and dexterity
batches on long trajectories. Failed frames skip dexterity computation and retain
`NaN` diagnostics. A successful multi-start repair is reused when checking joint jumps.

Run both analyses on the Marvin arm and optionally inspect any metric in Viser:

```bash
PYTHONPATH=src python examples/marvin_dexterity.py \
  --arm left --samples 10000 --trajectory-frames 200 \
  --color-metric isotropy --viser
```

### Dexforce W1 versus Marvin M6

Run a matched single-arm comparison and open both robots in one Viser session:

```bash
python3 examples/compare_w1_marvin.py \
  --arm left --samples 20000 --shared-targets 1000 \
  --trajectory-frames 200 --circle-preset horizontal --viser --autoplay \
  --report outputs/w1_vs_marvin.json
```

The solver keeps the complete seven-axis chain from `<arm>_arm_base` to the
end-effector, while comparison coordinates use the child link of active joint 2 as a
structural `link2` base. This preserves full-pose IK capability while making workspace,
target, and arm-length definitions independent of brand-specific link names. Reports
include joint ranges, AABB and convex-hull
volume, common-grid occupied volume, overlap, matched dexterity percentiles, identical
Cartesian-target reachability, and a shared circle-trajectory continuity test. Use
`--base-link base_link` only when intentionally comparing complete embodied chains;
W1 then includes additional movable torso/lower-body joints and is no longer a pure
seven-axis arm comparison.

Use `--ik-profile fast` for interactive checks, `balanced` (default) for normal reports,
and `rigorous` for final audits. The profiles allow at most 10, 36, and 104 seeds per
shared target respectively, with iteration limits of 120, 200, and 400. Every report
records the selected profile and its exact solver settings.

The rigorous report uses paired normalized joint samples, records convex-hull volume
at 25/50/75/100% of the requested samples, and reports Wilson 95% intervals for shared
target reachability. The same shared positions are tested with position-only IK and
full-pose IK using the circle's fixed world orientation. Convex hull is an outer
envelope; `common_voxel_grid.occupied_volume` is a more conservative sampled occupancy
estimate, and neither replaces collision-aware task reachability.
Scale-normalized metrics use the URDF chain-length upper bound by default. If validated
manufacturer dimensions use a different convention, supply `--w1-arm-length` and
`--marvin-arm-length`; the report records whether each length came from the URDF or a
nominal CLI override. Volume and position manipulability are divided by length cubed,
while position-Jacobian singular values are divided by length.

Use `--align-arm-length` to run a virtual equal-length comparison. Link2-relative
Cartesian coordinates are scaled by `common_length / robot_length`; shared targets are
mapped back by the inverse scale before IK. The default common length is the shorter
arm, or set it with `--aligned-arm-length`. Generic aliases `--robot-a-urdf`,
`--robot-b-urdf`, `--robot-a-arm-length`, and `--robot-b-arm-length` allow the same
pipeline to compare arbitrary URDF arms without relying on product names.

```bash
python3 examples/compare_w1_marvin.py \
  --robot-a-urdf robot_a.urdf --robot-b-urdf robot_b.urdf \
  --reference-link-index 2 --align-arm-length \
  --samples 20000 --shared-targets 1000 --report outputs/aligned_arms.json
```

Choose `horizontal`, `vertical_xz`, `vertical_yz`, or `chest_front` with
`--circle-preset`. Their
default centers are link2-relative world offsets, and `--circle-center-offset X Y Z`
overrides the center for controlled experiments. `--circle-radius` controls the common
radius. Thus both robots trace circles in genuinely parallel world planes rather than
in unrelated local arm frames.
`chest_front` places a world-YZ frontal circle at a shoulder-relative center
`[+0.40, -0.10, -0.10]` m for the left arm (`+X` forward, `-Y` inward, `-Z` down).
It defaults to radius `0.08 m` and fixed world RPY `[0, +pi/2, 0]`, which is
jointly reachable by both evaluated robots.

The comparison uses fixed-world-orientation IK by default: the TCP keeps one constant
world rotation while moving around the circle. The default rotation matches the common
zero-pose TCP orientation (`roll=-pi/2` for the left arm and `+pi/2` for the right arm).
Override it with `--target-rpy R P Y`, or use `--orientation-mode position-only` when
measuring positional reachability without a tooling-orientation constraint. Viser
draws sampled RGB coordinate frames along the circle; the **Trajectory coordinate
frames** checkbox controls their visibility. Workspace clouds start hidden to keep the
robots and path readable. Enable **Workspace** when inspecting coverage. The larger
world-origin and structural-base frames both use parallel world axes; the link2 frame
is translated only and must not be interpreted as the URDF joint's local axes.
During playback, the larger target frame highlights the requested TCP pose, the robot
TCP frame shows FK actual pose, and the GUI reports per-frame position/orientation
error. A short error segment connects actual and target TCP positions.
The captured `reference_tool` pose is stored in solver-base coordinates for IK, but is
rendered after applying `world_from_solver_base`; its panel lists both coordinate
representations. The side-by-side scene offset is presentation-only and is not part of
robot kinematics.

Trajectory IK uses previous-frame warm starts, a small null-space bias toward the
first successful posture, and multi-start repair when an adjacent joint change exceeds
`--jump-repair-threshold`. Tune the closed-loop posture bias with
`--trajectory-posture-gain` (default `0.005`). Reports distinguish raw and normalized
maximum jump and include joint-path length, loop-closure error, peak/RMS velocity,
peak acceleration, singular-value margin, joint-limit margin, and velocity violations.
Global joint centering remains opt-in through `--joint-centering-gain`; excessive gain
can reduce convergence on tightly constrained full-pose targets.
Circle trajectories enable `--loop-closure` by default. The solver distributes the
first/last joint drift across the whole path, uses each interpolated posture as an IK
seed, and reprojects every frame onto the original Cartesian pose. This closes the
joint path without replacing the Cartesian circle by unconstrained joint interpolation.
Reprojection solves those seeds in a batch. Nearly equal endpoint targets retain
separate joint solutions and residuals; only identical targets can reuse the first pose.
Use `--no-loop-closure` only for diagnostic comparison.

Although analysis is restricted to one arm, Viser loads every visual from each full
`robot.urdf`. Non-analyzed joints remain at zero while the selected
arm is driven by the IK trajectory. Each robot has Play/Stop, frame, FPS, and loop
controls. `--autoplay` starts the safe shared circle immediately. The web UI includes
per-robot trajectory metrics and a comparison table with the main conclusions.

## Command-line interface

The package CLI supports both analysis modes. Joint-space analysis:

```bash
workspace-analyzer robot.urdf \
  --base-link base_link --tip-link tool0 \
  --backend torch --device cuda \
  --strategy sobol --samples 100000 --batch-size 8192 \
  --output workspace.npz --viser --port 8080
```

Cartesian analysis with explicit bounds and a persistent content-addressed cache:

```bash
workspace-analyzer robot.urdf \
  --mode cartesian --base-link base_link --tip-link tool0 \
  --bounds -0.7 0.7 -0.3 0.9 0.4 1.8 \
  --full-pose --reference-joints 0 0 0 0 0 0 0 \
  --ik-restarts 4 --ik-rescue-restarts 16 --ik-rescue-rounds 3 \
  --samples 20000 --cache-dir .cache/workspace-analyzer \
  --output cartesian.npz --viser
```

The cache key fingerprints the parsed URDF model used by the solver, selected chain, solver/backend settings, and complete analysis configuration. Editing or removing the source file does not change the identity of an already loaded model; construct a new solver to use updated kinematics. Cache schema changes automatically trigger recomputation. Result files use a versioned JSON-plus-array NPZ format, load with `allow_pickle=False`, and are written atomically. A browser-based Viser client can be opened at the URL printed by the process; no desktop display server is required.

Both analyzer modes accept `cancel_event` and `progress_callback`. IK and trajectory
solving also accept `cancel_event`; set the event from another thread to raise
`AnalysisCancelled` at the next batch or IK iteration boundary, including rescue
attempts and closed-loop refinement. An individual NumPy/Torch operation runs to
completion before cancellation is checked. Closing a viewer cancels its background
work and prevents late results from updating the scene.

## Viser visualization

The viewer supports:

- URDF mesh, box, cylinder, and sphere visuals.
- Visual origins, RPY transforms, mesh scaling, and GLB materials.
- Live per-joint sliders and a reset action.
- Independent visibility for robot geometry, skeleton, joints, TCP, and workspace.
- Manipulability coloring for FK workspaces.
- Green/red reachable/unreachable coloring for Cartesian analysis.
- Deterministic display-only downsampling above 250,000 points.

```python
from workspace_analyzer.visualization import ViserWorkspace

viewer = ViserWorkspace(solver, port=8080)
viewer.add_workspace(result, color_metric="isotropy")
viewer.wait()
```

The trajectory overlay shows requested TCP samples in green/red for IK success/failure;
valid path segments are colored by minimum singular value. Add one before `wait()` with
`viewer.add_trajectory(trajectory)`.

The Display panel provides independent robot, skeleton, TCP, workspace, trajectory
target, and trajectory path visibility. It also exposes workspace point size,
reachable/unreachable opacity, trajectory target size, trajectory line width, and a
live workspace-color metric selector with percentile legend. Multiple viewers can
share one Viser server through `server=first_viewer.server`, with independent
`workspace_root`, `gui_label`, and `scene_offset` values.

### Trajectory gallery

The gallery uses offsets in the FK reference-TCP frame and supports both position-only
and fixed-orientation full-pose IK:

```bash
# Straight line
python3 examples/trajectory_gallery.py --shape line --viser

# Circle with fixed reference orientation
python3 examples/trajectory_gallery.py --shape circle --full-pose --viser

# Figure eight
python3 examples/trajectory_gallery.py --shape figure8 --viser

# 3D helix on Dexforce W1
python3 examples/trajectory_gallery.py \
  --urdf /home/ubuntu/workspace/chase/HumanoidAssets/Dexforce_W1_V3/robot.urdf \
  --shape helix --viser
```

Use `--scale`, `--depth`, `--frames`, and `--dt` to expose reachability, singularity,
joint-jump, and velocity-limit failures deliberately.

## Marvin M6 single-arm example

The examples default to this external asset:

```text
/home/ubuntu/workspace/chase/HumanoidAssets/Marvin_M6_S_CCS_696_V4.0/robot.urdf
```

Override it with `--urdf` on another machine. The selected chain includes fixed torso transforms while sampling only the seven joints of one arm.

Joint-space analysis:

```bash
PYTHONPATH=src python examples/marvin_single_arm.py \
  --mode joint --arm left --backend torch --device auto \
  --samples 100000 --batch-size 8192 --viser
```

Cartesian position reachability:

```bash
PYTHONPATH=src python examples/marvin_single_arm.py \
  --mode cartesian --arm left --backend torch --device auto \
  --samples 20000 --batch-size 1024 --ik-restarts 4 --viser
```

Add `--full-pose` to require the FK reference orientation as well as position. Without `--reference-joints`, Marvin J4 starts at a 90° elbow bend (−90° in this URDF's joint convention); other joints use their joint-limit centers. Explicit reference values are in radians. Use `--arm right` to select the right arm.

The viewer initially frames the robot and shows its workspace. **Reset to initial pose** restores the startup joint values, including an explicitly supplied reference pose. Expand **Display** to adjust visibility or click **Frame robot and workspace** to restore the camera framing. The comparison demo keeps workspace clouds hidden initially so the trajectories remain readable.

For full-pose analysis, provide a reproducible reference configuration:

```bash
PYTHONPATH=src python examples/marvin_single_arm.py \
  --mode cartesian --arm left --full-pose --viser \
  --reference-joints 0.0 0.2 -0.4 0.0 0.3 0.0 0.0
```

The reference joints are passed through FK to obtain `R_ref`; every Cartesian target is built as `T_target = [R_ref, p_sample]`. The same joint vector is the first IK seed, while additional restarts use deterministic random seeds. In Viser, move the joint sliders and click **Capture current FK pose**, then **Recompute Cartesian reachability** to repeat this workflow interactively. Display sliders control workspace point size and reachable/unreachable opacity independently.

Reproducible benchmark:

```bash
PYTHONPATH=src python examples/benchmark_marvin.py \
  --backend numpy --batch-size 4096 --ik-targets 256 --restarts 4
```

The benchmark reports the median of five timed runs after one warmup by default;
adjust these with `--repeats` and `--warmup`. Reports include the random seed and
runtime versions, and sample exactly `--ik-targets` configurations independently of
the FK batch size. CUDA timing includes synchronization before and after each run.

Duration-based randomized regression:

```bash
PYTHONPATH=src python examples/stress_marvin.py \
  --duration 3600 --backend numpy --report outputs/stress.json
```

The stress runner continuously checks FK rotation orthogonality, analytic Jacobians against finite differences, and random full-pose IK success.

On the reference CPU environment, a one-hour NumPy/float64 stress run checked
24,720,384 FK poses and 386,256 random reachable full-pose IK targets. IK success
was 99.9702%, maximum successful residual was below `1e-5`, maximum rotation
orthogonality error was `1.45e-15`, and maximum position-Jacobian finite-difference
error was `3.78e-9`. These numbers are regression evidence, not a hardware-neutral
performance guarantee; rerun the command above on the deployment machine.

For Cartesian IK, memory scales roughly with `batch_size * restarts`. The Marvin
example defaults to a balanced batch size of 2048. Use a larger batch on CUDA or
high-memory hosts, and reduce it when memory is constrained.

## Automatic solver construction

1. Parse URDF links, joints, origins, axes, types, and limits.
2. Require a unique root when `base_link` is omitted.
3. Select the leaf with the most active joints when `tip_link` is omitted.
4. Preserve fixed joints in the transform chain.
5. Treat revolute, continuous, and prismatic joints as solver variables.
6. Select Torch when `backend="auto"` and Torch is importable; choose CUDA only when it is available.

For branched, dual-arm, or multi-end-effector robots, construct one solver per `(base_link, tip_link)` pair. Solver instances do not share mutable kinematic state.

## Accuracy and performance notes

- FK and geometric Jacobians are analytic and batch-vectorized.
- IK shares one FK/Jacobian traversal per iteration, and NumPy preserves the selected
  `float32` or `float64` compute dtype. Returned residuals describe the returned
  joint positions, including when the iteration budget is exhausted.
- Converged IK candidates leave the active computation batch while all remaining
  starts continue, preserving per-target best-residual and nearest-seed selection.
- A single `(4, 4)` IK target returns a `(DoF,)` solution; an `(N, 4, 4)` batch
  always returns `(N, DoF)`, including a one-target batch.
- Analysis writes batches into preallocated result arrays. Uniform grid sampling
  generates only the requested prefix, preserving grid order with memory proportional
  to `num_samples * dimensions`; a truncated grid may not cover every axis evenly.
- IK is a bounded numerical DLS solver, not a closed-form solver.
- Always inspect `success` and `residual` before consuming an IK result.
- Singularities, restrictive joint limits, and distant seeds can require multiple starts.
- `float64` is recommended for validation; `float32` is generally preferable for large CUDA batches.
- Viser belongs outside a hard real-time control path.

The included Marvin integration test checks the analytic position Jacobian against finite differences. Hardware-specific throughput should be measured with `examples/benchmark_marvin.py` rather than inferred from results on another machine.

For sensitivity analysis around the nominal circle, add `--robustness-test`. It evaluates
six ±2 cm center offsets, ±10% radius, and (for fixed-world IK) six ±5° RPY offsets.
The JSON report stores each case and per-robot worst success rate, joint jump, singular
value, and joint-limit margin. Tune these with `--robustness-position-delta`,
`--robustness-radius-fraction`, and `--robustness-angle-deg`.

```bash
python3 examples/compare_w1_marvin.py --arm left --align-arm-length \
  --circle-preset chest_front --robustness-test --viser --autoplay \
  --report outputs/w1_vs_marvin_robustness.json
```

For a complete reachability sweep (joint workspace, position-only IK, and fixed-pose IK)
use the bundled suite:

```bash
python3 examples/reachability_suite.py --urdf /path/to/robot.urdf --arm left --viser
```

For reachability on a fixed-height (or fixed-coordinate) plane, use the plane
scan example:

```bash
python3 examples/plane_reachability.py \
  --urdf /path/to/robot.urdf --arm left \
  --frame world --plane xy --constant 0.8 --span 0.8 --resolution 41 \
  --position-only --output outputs/plane.json
```

`--frame world` uses the URDF root coordinate system with upstream joints at zero;
`base` uses the arm base. The default, `reference`, uses the second active link's
zero-joint frame (the first for a one-joint chain), fixed throughout the scan.
`--plane` selects `xy`, `xz`, or `yz`; `--constant` overrides the normal component
of `--center`, in metres. Both the center and `--rpy` angles (radians) use the
selected frame. The JSON `grid` records reachability in `ij` order: rows follow
the first named plane axis and columns the second. Omit `--position-only` to
enforce the fixed TCP orientation as well.

`--batch-size` (default 1024) limits targets per IK call; `--seed` (default 42)
controls restart sampling. Repeat with the same batch size, seed, backend, and
solver settings to reproduce a scan; changing the batch size can change restart
candidates and reachability classifications. The report records these settings,
the effective plane center, and the transform into solver-base coordinates.
The viewer uses those same target coordinates and the actual IK solutions.

The [feature assessment](docs/feature-roadmap-2026-09-05.md) distinguishes completed
target assessment and orientation aggregation from proposed comparison APIs,
collision checks, global orientation maps, and streaming analysis.

## Development

```bash
python -m pip install -e '.[all,dev]'
make check
```

Marvin integration tests are skipped automatically if its external URDF is unavailable. See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow.

`make check` runs lint, pytest, the analytic robot workflow, and package builds.
JUnit results and robot reports are written under `outputs/`. Use
`make test-workflow` to run the robot checks alone; failed acceptance checks retain
diagnostics and return a nonzero exit status. CI includes NumPy-only and optional
Torch CPU/SciPy/Viser profiles, with report artifacts for both.

## Project layout

```text
src/workspace_analyzer/
  model.py           URDF model and chain selection
  kinematics.py      NumPy/Torch FK, Jacobian, and IK
  sampling.py        sampling strategies
  metrics.py         dexterity and joint-limit quality metrics
  trajectory.py      ordered-path IK and continuity diagnostics
  analyzer.py        joint and Cartesian analysis workflows
  reachability.py    supplied-target assessment and offline quality gates
  coverage.py        grouped orientation coverage summaries
  robustness.py      paired pose perturbations and robustness summaries
  boundary.py        adaptive translation brackets and endpoint verification
  stability.py       paired IK budget, seed, and configuration-quality studies
  selection.py       quality-first selection of measured IK candidates
  diversity.py       reproducible joint seeds and observed configuration groups
  visualization.py   Viser renderer and URDF visual loader
  cli.py             command-line entry point
examples/             Marvin workspace, dexterity, trajectory, and benchmarks
tests/                unit and optional asset integration tests
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
