# Changelog

All notable changes to WorkspaceAnalyzer are documented here. The project follows semantic versioning once its public API reaches stability.

## 0.1.0 - Unreleased

### Fixed

- Share configurable Marvin/W1 asset discovery across robot demos, replacing
  machine-specific paths with HUMANOID_ASSETS or a sibling checkout. Add a
  real-asset test target covering both Marvin arms and four trajectory shapes.

- Use a 90° elbow bend (URDF J4 = −90°) for Marvin demo reference poses and
  restore startup joint values on reset. Frame the initial robot and workspace,
  show workspace clouds by default, and collapse display controls initially.

- Aggregate perturbation minima with masked reductions and batch JSON conversion,
  avoiding full floating-point replacement arrays and scalar finiteness loops.
- Reuse per-reference success counts for paired rates, gains, and losses without
  copying all retained perturbations; preserve missing-value and zero-variant behavior.

- Add cooperative cancellation to orientation coverage and pose perturbation
  summaries, including pre-input and final checks without mutating source results.

- Reject stale quality flags in orientation coverage, perturbation, and stability
  reports by checking current measurements against complete saved quality gates.
- Reuse the shared consistency check in candidate analysis without rebuilding
  per-trial statistical reports or modifying source results.

- Validate stability report structure, version, trial names, and initialization
  metadata before reading measurement archives.
- Normalize damaged initial-seed archives and verify seed dimensions against actual
  trial measurements, including public provenance edited before saving or selection.

- Normalize corrupt ZIP/DEFLATE streams and missing required result members into
  descriptive load errors, allowing damaged cache entries to be recomputed.
- Reject NPY files passed as result archives and invalid cache-key types explicitly;
  preserve filesystem errors such as missing files for direct callers.

- Preflight every stability trial seed in solver precision before the first solve,
  cache access, or progress callback, preserving original seed provenance.
- Propagate cancellation through final stability compatibility checks and recheck
  before returning the completed study.

- Skip non-representative trials and completed matches during IK diversity grouping;
  preserve greedy representative order, all measurements, and quality contributions.
- Avoid redundant float64 copies when comparing already-indexed candidate arrays.

- Recheck finite inputs after dtype conversion; reject overflowing targets, seeds,
  and transforms before solver or cache access.
- Validate task-direction weights in solver precision before IK/Jacobian work;
  reject overflow and all-positive weights rounding to zero, and validate float64
  task-weight conversion during offline quality reassessment.

- Require finite measurements for enabled quality thresholds; positive infinity
  no longer passes minimum singular-value, isotropy, or joint-margin gates.
  Preserve raw measurements and IK flags across fresh runs and offline reassessment.

- Guard Torch dexterity denominators before branch selection so zero Jacobians
  and masked rank-deficient condition numbers do not poison gradients with NaNs.
  Forward values and numerical rank thresholds are unchanged.

- Revalidate public result arrays before atomic saves so invalid edits cannot replace
  a usable archive; publish cache keys only after a successful write.
- Add optional uncompressed NPZ output and cache storage with automatic loading of
  either encoding; expose the cache policy through `--cache-uncompressed`.

- Reduce stability metrics incrementally and serialize missing extrema in batches,
  avoiding trial-sized float stacks while preserving complete report contents.
- Support cancellation during stability report generation and candidate compatibility
  checks; reject mutated residual arrays with invalid shapes or numeric types.

- Validate candidate quality flags without rebuilding per-trial statistical reports;
  share gate evaluation with quality reassessment and reject malformed measurements.
- Reuse selected row indices across all result fields and skip unused trials.

- Sample Cartesian planes, lines, and fixed points directly using equal bounds;
  generate grids and random sequences only in varying dimensions.
- Avoid overflow when mapping samples into extreme finite intervals.

- Reject Boolean, string, and nonfinite quality thresholds before assessment.
- Preserve fractional Cartesian targets with integer/list reference poses and
  snapshot configuration arrays so caller mutations cannot change active runs.
- Correct plane-frame transforms for IK and visualization; retain actual IK
  configurations and residuals, report the effective plane center, and reject
  nonfinite scan inputs before loading the robot.
- Report zero 3-D hull volume for degenerate point clouds, unavailable estimates
  for other Qhull failures, and actual sample counts for small convergence scans.
- Replace recursive IK rescue with iterative attempts and include rescue work in
  `IKResult.iterations`; preserve Torch gradients when merging rescue solutions.
- Reuse successful multi-start trajectory repairs instead of repeating the same solve.
- Use a relative singular-value threshold for scale-invariant condition numbers.
- Normalize finite URDF axes without overflow or underflow at extreme magnitudes.
- Wrap continuous joints in IK and use shortest-angle posture bias; exclude them
  from joint-limit centering and unwrap successful trajectory frames only.
- Preserve final-pose residuals for approximately closed trajectories.
- Cancel IK iterations, rescue attempts, and trajectory refinement cooperatively.
- Serialize viewer worker startup and shutdown; prevent duplicate recomputation,
  updates after closure, and playback restart after closing the viewer.
- Preserve singleton IK batches and support Cartesian analysis with one-target tails.
- Recompute IK diagnostics after the final allowed update before restart selection.
- Stabilize quaternion errors near zero and half turns, including mixed-sign axes
  and float32 poses already at their targets.
- Preserve NumPy float32 compute dtype throughout FK, Jacobians, and IK.
- Return a successful console exit status after analysis and concise input errors.
- Reject nonfinite joint limits and invalid iteration, sample, and restart counts.
- Fingerprint the loaded model for caches; reject malformed result metadata and
  arrays that would require pickle. Cache keys now use schema 3.
- Honor cancellation after the final Cartesian batch before publishing results.

### Changed

- Cache target measurements independently of quality gates and task weights;
  reclassify cached results without repeating IK or Jacobian evaluation.
- Avoid revalidating Boolean result flags with an additional full-array membership pass.
- Reuse FK/Jacobian traversal for successful target quality checks; skip dexterity
  for failed targets and hash target arrays without expanding them into lists.
- Reuse sampling arrays for clipping and limit scaling, avoiding full-size
  temporaries without retaining an unused Sobol power-of-two tail.
- Batch plane-scan IK (default 1024 targets), recording restart seeds and batch size.
- Bound trajectory reprojection and quality batches with `batch_size` (default 1024),
  skip metrics for failed frames, and check cancellation between metric batches.
- Reject invalid task weights before computing Jacobians.
- Retire converged IK candidates from subsequent FK/Jacobian/DLS batches while
  retaining all candidates for final restart selection.
- Preallocate trajectory outputs and batch closed-loop Cartesian reprojection.
- Support joint-workspace progress callbacks and cancellation alongside Cartesian analysis.
- Share FK/Jacobian traversal in IK; avoid unnecessary link-transform copies and
  per-sample construction of axis skew matrices.
- Preallocate workspace outputs and compute joint-limit margins in batches.
- Generate only requested uniform-grid points without materializing the full grid.
- Benchmark warmups and repeated median timings, synchronize CUDA at both timing
  boundaries, record runtime context, and sample the requested IK target count.
- Resolve the reachability suite's child script independently of the current directory.

### Added

- Reproducible explicit joint-seed trials with bounded uniform sampling, an
  independent RNG stream, immutable arrays, and a stable generated prefix.
- Offline configuration-diversity reports with angular/linear tolerances,
  continuous-joint wrapping, per-trial novelty, and quality-aware group counts;
  original candidates remain available for quality selection.
- Workflow initial-guess sweeps and saved diversity reports, joint-type metadata
  for offline studies, and analytic periodicity/near-threshold regressions.
- Offline selection of measured IK candidates, prioritizing quality gates before
  joint margin, isotropy, or minimum singular value; per-target trial provenance,
  selection-time audits, independent result arrays, and recomputed statistics.
- Optional workflow candidate selection and quality acceptance, with analytic
  two-branch/singularity regressions on NumPy and Torch CPU in both precisions.
- Candidate compatibility checks for joint ordering, coordinate provenance, and
  complete non-budget settings, including mixed cached and fresh trial metadata.
- Fixed-target IK stability studies with named budget/restart/initial-configuration
  trials, paired gains/losses, disagreement indices, and observed quality ranges.
- Saved trial measurements and explicit initial seeds, model/settings alignment
  checks, immutable input snapshots, cached replay, and offline quality reassessment.
- Workflow stability tests for pose, perturbation, or boundary targets, configurable
  disagreement acceptance, and Make/CI integration.
- Batched translation-boundary refinement with warm starts, spatial resolution
  and round budgets, final failed-endpoint rechecks, precision-limit detection,
  cached replay, cooperative cancellation, and persisted query traces.
- Boundary workflow acceptance and Markdown diagnostics, backed by analytic
  Cartesian-stage and planar-arm tests on NumPy and Torch CPU.
- Deterministic `PosePerturbations` with explicit base/tool axes, aligned paired
  IK/quality summaries, worst solved metrics, and offline reassessment support.
- Optional workflow perturbation testing and robust-task acceptance thresholds;
  analytic boundary, frame invariance, zero-offset, and CPU backend regressions.
- Observed/required values and per-axis paired diagnostics in Markdown reports;
  perturbation workflows included in Make and optional-runtime CI checks.
- Offline quality reassessment and per-position raw/weighted orientation coverage,
  including explicit zero-weight group handling and JSON-safe summaries.
- Repeatable robot testing workflow with FK-generated and out-of-range targets,
  local orientation samples, quality sweeps, configurable acceptance criteria,
  diagnostic artifacts, and failure exit codes.
- `make check` and `make test-workflow`, JUnit/report retention in CI, and a
  separate Torch CPU/SciPy/Viser runtime test job.
- Supplied-target assessment with `ReachabilityConfig` and
  `WorkspaceAnalyzer.analyze_targets`: points/poses, coordinate transforms, task
  weights, dexterity gates, full-task rank checks, cancellation, and result caching.
- Target-mode CLI input from CSV/NPY/NPZ and JSON assessment reports, separating
  IK failures from selected-configuration quality rejection.
- Optional `AnalysisResult.target_poses` persistence and Viser quality summaries.
- Analytic Cartesian-stage and planar-arm assessment regressions for reachability,
  singularities, limits, task rank, frame transforms, and NumPy/Torch CPU precision.
- Explicit reference, arm-base, and URDF-world frame selection for plane scans.
- General serial-chain construction from URDF.
- Batched NumPy and PyTorch CPU/CUDA FK, analytic Jacobian, and bounded DLS IK.
- Joint-space and Cartesian-space workspace analysis.
- Deterministic multi-start and failed-target-only rescue IK.
- Viser URDF visuals, workspace coloring, live controls, captured FK references, and cancellable recomputation.
- Safe versioned NPZ results and content-addressed result caching.
- Batched dexterity metrics and per-workspace metric arrays.
- Sequential warm-start trajectory IK with continuity and velocity diagnostics.
- Marvin M6 single-arm demos, benchmark, and duration-based stress runner.
- Marvin dexterity and ordered-trajectory continuity example.
- Matched Dexforce W1 versus Marvin M6 analysis with shared-session Viser rendering.
- Cartesian line, circle, figure-eight, and helix trajectory gallery.
- Shared Viser servers, side-by-side scene offsets, trajectory controls, and metric legends.
- Full-robot URDF rendering during single-arm analysis, animated trajectory playback,
  and in-view comparison conclusions.
- English and Simplified Chinese documentation.
