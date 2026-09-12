import json
import threading
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from workspace_analyzer import (
    AnalysisCancelled,
    IKStabilityResult,
    IKTrial,
    KinematicsSolver,
    ReachabilityConfig,
    ResultCache,
    analyze_ik_stability,
    create_solver,
)

URDF = Path(__file__).parent / "fixtures/cartesian_stage.urdf"


def _solver(backend="numpy", dtype="float64", path=URDF):
    if backend == "torch":
        pytest.importorskip("torch")
    return create_solver(
        path, backend=backend, device="cpu", dtype=dtype, max_iterations=40
    )


def _config(**kwargs):
    return ReachabilityConfig(restarts=1, rescue_restarts=0, batch_size=2, **kwargs)


def _trials():
    return [
        IKTrial("short", max_iterations=1),
        IKTrial("complete"),
        IKTrial("shifted", max_iterations=1, seed=[0.4, 0, 0]),
    ]


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_analytic_study_preserves_gains_losses_and_quality_differences(backend, dtype):
    solver = _solver(backend, dtype)
    original = solver.config
    study = analyze_ik_stability(
        solver,
        [[0, 0, 0], [0.4, 0, 0], [2, 0, 0]],
        _trials(),
        _config(minimum_joint_limit_margin=0.9),
    )
    assert solver.config is original
    report = study.to_dict()
    assert report["per_target"]["ik_success_counts"] == [2, 2, 0]
    assert report["per_target"]["ik_disagreement_indices"] == [0, 1]
    assert report["summary"]["ik_disagreement_rate"] == pytest.approx(2 / 3)
    assert report["summary"]["ik_all_success_count"] == 0
    assert report["summary"]["ik_no_success_count"] == 1
    assert report["per_target"]["quality_success_counts"] == [2, 0, 0]
    comparisons = report["baseline_comparisons"]
    assert comparisons[0]["ik_gained_indices"] == [1]
    assert comparisons[1]["ik_gained_indices"] == [1]
    assert comparisons[1]["ik_lost_indices"] == [0]
    assert comparisons[1]["quality_lost_indices"] == [0]
    np.testing.assert_allclose(
        report["per_target"]["joint_limit_margin_minimum"][:2], [1, 0.6], atol=1e-5
    )
    assert report["per_target"]["joint_limit_margin_minimum"][2] is None
    assert report["trials"][2]["initialization"]["kind"] == "provided"
    assert all(
        result.joint_positions.dtype == np.dtype(dtype) for result in study.results
    )
    json.dumps(report, allow_nan=False)


def test_cached_trials_do_not_reload_urdf_or_run_ik_and_gates_reassess_offline(
    tmp_path, monkeypatch
):
    path = tmp_path / "robot.urdf"
    path.write_text(URDF.read_text())
    solver = _solver(path=path)
    path.unlink()
    config = _config(minimum_joint_limit_margin=0.9)
    cache = ResultCache(tmp_path / "cache")
    study = analyze_ik_stability(
        solver, [[0, 0, 0], [0.4, 0, 0]], _trials(), config, cache=cache
    )
    original = study.to_dict()

    def unexpected(*args, **kwargs):
        pytest.fail("cached trials or offline summaries must not run IK/FK")

    for name in ("inverse", "forward", "forward_with_jacobian"):
        monkeypatch.setattr(KinematicsSolver, name, unexpected)
    repeated = analyze_ik_stability(
        solver, [[0, 0, 0], [0.4, 0, 0]], _trials(), config, cache=cache
    )
    assert repeated.to_dict()["per_target"] == original["per_target"]
    assert all(result.metadata["cache_hit"] for result in repeated.results)
    relaxed = study.reassess_quality()
    assert (
        relaxed.to_dict()["per_target"]["quality_success_counts"]
        == original["per_target"]["ik_success_counts"]
    )
    assert study.to_dict() == original
    # Names are display labels and must not invalidate identical measurements.
    renamed = [replace(trial, name=f"new {i}") for i, trial in enumerate(_trials())]
    again = analyze_ik_stability(
        solver, [[0, 0, 0], [0.4, 0, 0]], renamed, _config(), cache=cache
    )
    assert again.to_dict()["summary"] == relaxed.to_dict()["summary"]


def test_trial_budget_and_random_seed_each_invalidate_measurement_cache(tmp_path):
    solver = _solver()
    cache = ResultCache(tmp_path)
    trials = [
        IKTrial("base"),
        IKTrial("different seed", random_seed=99),
        IKTrial("different budget", max_iterations=60),
        IKTrial("different restarts", restarts=2),
    ]
    study = analyze_ik_stability(solver, [[0.2, 0, 0]], trials, _config(), cache=cache)
    assert len({result.metadata["cache_key"] for result in study.results}) == 4
    assert [
        result.metadata["solver_settings"]["max_iterations"] for result in study.results
    ] == [40, 40, 60, 40]
    assert [result.metadata["random_seed"] for result in study.results] == [
        42,
        99,
        42,
        42,
    ]


def test_study_snapshots_targets_weights_frame_and_trial_initial_guess():
    solver = _solver()
    targets = np.array([[0.0, 0, 0], [0.2, 0, 0]])
    weights = np.array([1.0, 2.0])
    transform = np.eye(4)
    transform[0, 3] = 0.1
    seed = np.zeros(3)
    first = IKTrial("first", seed=seed)
    seed[:] = 100
    with pytest.raises(ValueError):
        first.seed[0] = 100
    progress = []

    def callback(value):
        progress.append(value)
        targets[:] = 100
        weights[:] = 0
        transform[:] = 0
        solver.config = replace(solver.config, tolerance=0.5, max_iterations=1)

    study = analyze_ik_stability(
        solver,
        targets,
        [first, IKTrial("second")],
        _config(),
        weights=weights,
        base_from_targets=transform,
        progress_callback=callback,
    )
    for result in study.results:
        np.testing.assert_allclose(result.points, [[0.1, 0, 0], [0.3, 0, 0]])
        np.testing.assert_array_equal(result.metrics["task_weight"], [1, 2])
        assert result.metadata["solver_settings"]["tolerance"] == 1e-5
        assert result.metadata["solver_settings"]["max_iterations"] == 40
    assert progress == [0.5, 1.0]


def test_saved_study_roundtrip_preserves_seeds_and_recomputes_stale_summaries(tmp_path):
    study = analyze_ik_stability(
        _solver(),
        [[0, 0, 0], [0.4, 0, 0]],
        [IKTrial("../../outside"), IKTrial("seeded", seed=[[0.1, 0, 0]])],
        _config(),
    )
    report = study.to_dict()
    study.save(tmp_path)
    restored = IKStabilityResult.load(tmp_path)
    assert restored.to_dict() == report
    np.testing.assert_array_equal(restored.initial_seeds[1], [[0.1, 0, 0]])
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "initial_seeds.npz",
        "report.json",
        "trial_000.npz",
        "trial_001.npz",
    ]
    report["summary"]["ik_disagreement_count"] = 1000
    (tmp_path / "report.json").write_text(json.dumps(report))
    assert (
        IKStabilityResult.load(tmp_path).to_dict()["summary"]["ik_disagreement_count"]
        == 0
    )


@pytest.mark.parametrize("when", ["before", "during", "final"])
def test_study_cancellation(when):
    event = threading.Event()
    if when == "before":
        event.set()

    def callback(value):
        if when == "during" or value == 1:
            event.set()

    with pytest.raises(AnalysisCancelled):
        analyze_ik_stability(
            _solver(),
            [[0, 0, 0]],
            _trials(),
            _config(),
            cancel_event=event,
            progress_callback=callback,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": ""},
        {"name": 5},
        {"max_iterations": 0},
        {"max_iterations": True},
        {"random_seed": -1},
        {"restarts": 1.5},
        {"rescue_restarts": -1},
        {"seed": []},
        {"seed": [np.nan]},
        {"seed": "bad"},
    ],
)
def test_invalid_trial_settings(kwargs):
    with pytest.raises(ValueError):
        IKTrial(**{"name": "trial", **kwargs})


@pytest.mark.parametrize(
    "trials",
    [
        [],
        [IKTrial("single")],
        [IKTrial("same"), IKTrial("same")],
        [IKTrial("ok"), "bad"],
        [IKTrial("ok"), IKTrial("bad seed", seed=[0, 0])],
    ],
)
def test_invalid_trial_set_fails_before_solving(trials, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid later trials must be rejected before the first solve")

    monkeypatch.setattr(KinematicsSolver, "inverse", unexpected)
    with pytest.raises(ValueError):
        analyze_ik_stability(_solver(), [[0, 0, 0]], trials, _config())


@pytest.mark.parametrize(
    "kind",
    ["targets", "weights", "threshold", "tolerance", "mode", "quality", "residual"],
)
def test_reports_reject_incomparable_or_invalid_measurements(kind):
    study = analyze_ik_stability(
        _solver(), [[0, 0, 0], [2, 0, 0]], [IKTrial("a"), IKTrial("b")], _config()
    )
    result = study.results[1]
    if kind == "targets":
        result.points[0, 0] = 1
    elif kind == "weights":
        result.metrics["task_weight"][0] = 2
    elif kind == "threshold":
        result.metadata["quality_thresholds"]["minimum_isotropy"] = 0.5
    elif kind == "tolerance":
        result.metadata["solver_settings"]["tolerance"] = 0.5
    elif kind == "mode":
        result.metadata["position_only"] = False
    elif kind == "quality":
        result.metrics["quality_pass"][-1] = True
    else:
        result.residual = None
    with pytest.raises(ValueError):
        study.to_dict()


@pytest.mark.parametrize("dtype", ["float32", "float64", "mixed"])
def test_incremental_report_statistics_match_scalar_reference(dtype):
    study = analyze_ik_stability(
        _solver(), [[0, 0, 0], [0.4, 0, 0], [2, 0, 0]], _trials(), _config()
    )
    measurements = [
        [np.nan, 0.4, np.inf],
        [0.2, np.nan, -np.inf],
        [-0.1, 0.8, np.nan],
    ]
    for index, result in enumerate(study.results):
        precision = (
            ("float32" if index % 2 else "float64") if dtype == "mixed" else dtype
        )
        result.residual = np.asarray(measurements[index], dtype=precision)
        for metric in ("isotropy", "joint_limit_margin"):
            result.metrics[metric] = np.asarray(measurements[index], dtype=precision)
    before = [result.residual.copy() for result in study.results]
    report = study.to_dict()
    for metric in ("residual", "isotropy", "joint_limit_margin"):
        for row in range(3):
            finite = []
            for result in study.results:
                value = (
                    result.residual if metric == "residual" else result.metrics[metric]
                )[row]
                if np.isfinite(value) and (
                    metric == "residual" or result.reachable[row]
                ):
                    finite.append(float(value))
            assert report["per_target"][f"{metric}_finite_counts"][row] == len(finite)
            assert report["per_target"][f"{metric}_minimum"][row] == (
                min(finite) if finite else None
            )
            assert report["per_target"][f"{metric}_maximum"][row] == (
                max(finite) if finite else None
            )
    for result, original in zip(study.results, before):
        np.testing.assert_array_equal(result.residual, original)
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "operation", ["to_dict", "select_solutions", "summarize_diversity"]
)
def test_offline_operations_cancel_within_compatibility_validation(
    operation, monkeypatch
):
    import workspace_analyzer.stability as module

    study = analyze_ik_stability(_solver(), [[0, 0, 0]], _trials(), _config())
    event = threading.Event()
    original = module._assessment_arrays
    calls = []

    def cancelling(result):
        calls.append(1)
        event.set()
        return original(result)

    monkeypatch.setattr(module, "_assessment_arrays", cancelling)
    with pytest.raises(AnalysisCancelled):
        getattr(study, operation)(cancel_event=event)
    assert len(calls) == 1


def test_report_cancellation_before_during_reduction_and_final():
    study = analyze_ik_stability(_solver(), [[0, 0, 0]], _trials(), _config())

    class CountingEvent:
        def __init__(self, stop=None):
            self.calls = 0
            self.stop = stop

        def is_set(self):
            self.calls += 1
            return self.calls == self.stop

    event = CountingEvent()
    original = study.to_dict(cancel_event=event)
    for stop in (1, event.calls // 2, event.calls):
        with pytest.raises(AnalysisCancelled):
            study.to_dict(cancel_event=CountingEvent(stop))
    assert study.to_dict() == original


@pytest.mark.parametrize(
    "residual",
    [np.zeros((3, 1)), np.zeros(2), np.ones(3, dtype=complex), np.ones(3, dtype=bool)],
)
def test_report_rejects_mutated_residual_schema(residual):
    study = analyze_ik_stability(
        _solver(), [[0, 0, 0], [0.4, 0, 0], [2, 0, 0]], _trials(), _config()
    )
    study.results[0].residual = residual
    with pytest.raises(ValueError, match="residuals"):
        study.to_dict()


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("shape", [(3,), (1, 3), (2, 3)])
def test_later_seed_overflow_fails_before_first_solver_or_cache_access(
    backend, shape, monkeypatch
):
    solver = _solver(backend, "float32")
    seed = np.full(shape, 1e100, dtype=np.float64)
    trials = [IKTrial("valid"), IKTrial("later invalid", seed=seed)]

    def unexpected(*args, **kwargs):
        pytest.fail("all trial seeds must be checked before the first evaluation")

    class NoCacheAccess:
        get = unexpected
        put = unexpected

    monkeypatch.setattr(KinematicsSolver, "inverse", unexpected)
    with np.errstate(over="raise", invalid="raise"):
        with pytest.raises(ValueError, match="trial seed.*float32"):
            analyze_ik_stability(
                solver,
                [[0, 0, 0], [0.2, 0, 0]],
                trials,
                _config(),
                cache=NoCacheAccess(),
                progress_callback=unexpected,
            )
    np.testing.assert_array_equal(trials[1].seed, seed)
    assert not trials[1].seed.flags.writeable


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_valid_mixed_precision_seed_keeps_original_provenance(backend, tmp_path):
    seed = np.array([0.123456789123, 0, 0], dtype=np.float64)
    trials = [IKTrial("default"), IKTrial("precise input", seed=seed)]
    study = analyze_ik_stability(
        _solver(backend, "float32"), [[0, 0, 0]], trials, _config()
    )
    assert all(result.reachable.all() for result in study.results)
    assert study.initial_seeds[1].dtype == np.float64
    np.testing.assert_array_equal(study.initial_seeds[1], seed)
    study.save(tmp_path)
    restored = IKStabilityResult.load(tmp_path)
    np.testing.assert_array_equal(restored.initial_seeds[1], seed)
    assert restored.to_dict() == study.to_dict()


def test_study_cancellation_in_final_compatibility_check(monkeypatch):
    import workspace_analyzer.stability as module

    event = threading.Event()
    original = module._assessment_arrays
    checks = []

    def cancel(result):
        checks.append(1)
        event.set()
        return original(result)

    monkeypatch.setattr(module, "_assessment_arrays", cancel)
    with pytest.raises(AnalysisCancelled):
        analyze_ik_stability(
            _solver(), [[0, 0, 0]], _trials(), _config(), cancel_event=event
        )
    assert len(checks) == 1


@pytest.mark.parametrize(
    "report",
    [
        None,
        [],
        1,
        {"study_version": True, "trials": [{}, {}]},
        {"study_version": 1, "trials": []},
        {"study_version": 1, "trials": [None, None]},
        {"study_version": 1, "trials": [{"name": "a", "initialization": None}] * 2},
        {
            "study_version": 1,
            "trials": [{"name": "a", "initialization": {"kind": "unknown"}}] * 2,
        },
    ],
)
def test_load_rejects_invalid_report_before_reading_arrays(
    tmp_path, report, monkeypatch
):
    from workspace_analyzer import AnalysisResult

    (tmp_path / "report.json").write_text(json.dumps(report))

    def unexpected(*args, **kwargs):
        pytest.fail("invalid reports must fail before reading trial arrays")

    monkeypatch.setattr(AnalysisResult, "load", unexpected)
    with pytest.raises(ValueError):
        IKStabilityResult.load(tmp_path)


@pytest.mark.parametrize(
    "kind", ["missing_member", "npy", "truncated", "wrong_dof", "wrong_count"]
)
def test_seed_archive_errors_are_clear_and_match_actual_measurements(tmp_path, kind):
    study = analyze_ik_stability(
        _solver(),
        [[0, 0, 0], [0.2, 0, 0]],
        [IKTrial("default"), IKTrial("provided", seed=[0, 0, 0])],
        _config(),
    )
    study.save(tmp_path)
    path = tmp_path / "initial_seeds.npz"
    if kind == "missing_member":
        np.savez(path)
    elif kind == "npy":
        with path.open("wb") as stream:
            np.save(stream, np.zeros(3))
    elif kind == "truncated":
        path.write_bytes(path.read_bytes()[:40])
    else:
        seed = np.zeros(2) if kind == "wrong_dof" else np.zeros((3, 3))
        np.savez(path, trial_001=seed)
        report_path = tmp_path / "report.json"
        report = json.loads(report_path.read_text())
        # A matching manifest is insufficient: compare against actual DoF/N too.
        report["trials"][1]["initialization"]["shape"] = list(seed.shape)
        report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="seed"):
        IKStabilityResult.load(tmp_path)


@pytest.mark.parametrize(
    "seed",
    [
        np.array([np.nan, 0, 0]),
        np.zeros(2),
        np.zeros((3, 3)),
        np.ones(3, dtype=complex),
    ],
)
def test_mutated_initial_seed_provenance_is_rejected_before_save(tmp_path, seed):
    study = analyze_ik_stability(
        _solver(), [[0, 0, 0]], [IKTrial("a"), IKTrial("b")], _config()
    )
    study.initial_seeds = (None, seed)
    for operation in (study.to_dict, study.select_solutions, study.summarize_diversity):
        with pytest.raises(ValueError, match="initial seed"):
            operation()
    with pytest.raises(ValueError, match="initial seed"):
        study.save(tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()
