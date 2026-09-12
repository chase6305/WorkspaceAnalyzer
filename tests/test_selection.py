import json
import threading
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from workspace_analyzer import (
    AnalysisCancelled,
    AnalysisResult,
    IKStabilityResult,
    IKTrial,
    KinematicsSolver,
    ReachabilityConfig,
    analyze_ik_stability,
    create_solver,
    select_ik_solutions,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _branches(backend="numpy", dtype="float64"):
    if backend == "torch":
        pytest.importorskip("torch")
    solver = create_solver(
        FIXTURES / "two_branch.urdf",
        backend=backend,
        device="cpu",
        dtype=dtype,
        max_iterations=1,
    )
    # Analytic elbow-up/down roots of x=cos(a)+cos(a+b), y=sin(a)+sin(a+b).
    near_limit = [np.pi / 2, -np.pi / 2]
    roomy = [0, np.pi / 2]
    study = analyze_ik_stability(
        solver,
        [[1, 1, 0], [3, 3, 0], [2, 0, 0]],
        [
            IKTrial("near elbow limit", seed=[near_limit, near_limit, [0, 0]]),
            IKTrial("roomy elbow", seed=[roomy, roomy, [0, 0]]),
        ],
        ReachabilityConfig(
            restarts=1,
            rescue_restarts=0,
            minimum_joint_limit_margin=0.2,
            minimum_isotropy=0.05,
        ),
    )
    return solver, study


def _stage_study():
    return analyze_ik_stability(
        create_solver(FIXTURES / "cartesian_stage.urdf", max_iterations=40),
        [[0, 0, 0], [0.4, 0, 0], [2, 0, 0]],
        [
            IKTrial("short", max_iterations=1),
            IKTrial("complete"),
            IKTrial("shifted", max_iterations=1, seed=[0.4, 0, 0]),
        ],
        ReachabilityConfig(restarts=1, rescue_restarts=0),
        weights=[0, 3, 1],
    )


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize(
    "objective", ["joint_limit_margin", "isotropy", "minimum_singular_value"]
)
def test_analytic_branches_choose_quality_accepted_root_and_keep_singular_failure(
    backend,
    dtype,
    objective,
):
    solver, study = _branches(backend, dtype)
    result = study.select_solutions(objective=objective)
    assert result.reachable.tolist() == [True, False, True]
    assert result.metrics["quality_pass"].tolist() == [True, False, False]
    assert result.metrics["selected_trial_index"][0] == 1
    np.testing.assert_allclose(result.joint_positions[0], [0, np.pi / 2], atol=1e-6)
    # Independent analytic FK, not a comparison with the implementation's FK.
    a, b = result.joint_positions[0]
    np.testing.assert_allclose(
        [np.cos(a) + np.cos(a + b), np.sin(a) + np.sin(a + b)], [1, 1], atol=1e-6
    )
    assert result.metrics["joint_limit_margin"][0] == pytest.approx(
        (2.4 - np.pi / 2) / 2, abs=1e-6
    )
    assert result.metrics["isotropy"][2] == pytest.approx(0, abs=1e-6)
    assert result.residual[result.reachable].max() < solver.config.tolerance
    assert result.joint_positions.dtype == np.dtype(dtype)
    audit = result.metadata["candidate_selection"]
    assert audit["selection_time_summary"]["quality_gained_indices"] == [0]
    assert audit["selection_time_summary"]["ik_lost_count"] == 0
    assert "max_iterations" not in result.metadata["solver_settings"]
    assert "random_seed" not in result.metadata
    assert "cache_key" not in result.metadata
    for row, trial in enumerate(result.metrics["selected_trial_index"]):
        source = study.results[trial]
        np.testing.assert_array_equal(
            result.joint_positions[row], source.joint_positions[row]
        )
        assert result.residual[row] == source.residual[row]
        for metric, values in source.metrics.items():
            np.testing.assert_array_equal(result.metrics[metric][row], values[row])


def test_selection_retains_candidate_union_and_recomputes_weighted_statistics():
    study = _stage_study()
    result = select_ik_solutions(study)
    ik, quality = study._flags()
    np.testing.assert_array_equal(result.reachable, ik.any(axis=0))
    np.testing.assert_array_equal(result.metrics["quality_pass"], quality.any(axis=0))
    assert result.metadata["assessment"]["weighted_ik_success_rate"] == 0.75
    summary = result.metadata["candidate_selection"]["selection_time_summary"]
    assert summary["ik_gained_indices"] == [1]
    assert summary["ik_lost_indices"] == summary["quality_lost_indices"] == []


def test_quality_gates_outrank_requested_metric_and_report_the_tradeoff():
    # Controlled recorded metrics: a higher-margin root misses isotropy gate.
    study = _stage_study().reassess_quality()
    for i, result in enumerate(study.results):
        result.metrics["joint_limit_margin"][0] = [0.9, 0.6, 0.7][i]
        result.metrics["isotropy"][0] = [0.01, 0.3, 0.4][i]
    study = study.reassess_quality(minimum_isotropy=0.2)
    selected = study.select_solutions()
    assert selected.metrics["selected_trial_index"][0] == 1
    assert selected.metrics["quality_pass"][0]
    audit = selected.metadata["candidate_selection"]["selection_time_summary"]
    assert audit["objective_decreased_indices"] == [0]
    assert audit["quality_gained_indices"] == [0, 1]
    # Gate replacement must reselect from the study to reconsider other roots.
    relaxed = study.reassess_quality().select_solutions()
    assert relaxed.metrics["selected_trial_index"][0] == 0


@pytest.mark.parametrize("missing", [np.nan, np.inf, -np.inf])
def test_nonfinite_objective_never_outranks_measured_success(missing):
    study = _stage_study()
    study.results[0].metrics["joint_limit_margin"][0] = missing
    selected = study.select_solutions()
    assert selected.metrics["selected_trial_index"][0] == 1
    assert selected.reachable[0]


def test_failures_cannot_win_on_quality_metrics_and_ties_use_residual_then_order():
    study = _stage_study()
    # All roots fail for the last target; only residual is meaningful there.
    for index, result in enumerate(study.results):
        result.metrics["joint_limit_margin"][2] = [999, 0, 50][index]
        result.residual[2] = [np.nan, 0.5, 0.5][index]
    # Two identical measured successes choose the first trial on an exact tie.
    selected = study.select_solutions()
    assert selected.metrics["selected_trial_index"][[0, 2]].tolist() == [0, 1]
    assert not selected.reachable[2]
    assert not selected.metrics["quality_pass"][2]


def test_offline_selection_persistence_owns_arrays_and_preserves_gate_audit(
    tmp_path, monkeypatch
):
    _, study = _branches()
    original = study.to_dict()

    def unexpected(*args, **kwargs):
        pytest.fail("offline selection must not run kinematics")

    for method in ("inverse", "forward", "forward_with_jacobian"):
        monkeypatch.setattr(KinematicsSolver, method, unexpected)
    study.save(tmp_path)
    selected = IKStabilityResult.load(tmp_path).select_solutions()
    selected.save(tmp_path / "selected.npz")
    loaded = AnalysisResult.load(tmp_path / "selected.npz")
    assert loaded.metadata == json.loads(json.dumps(selected.metadata))
    np.testing.assert_array_equal(
        loaded.metrics["selected_trial_index"], selected.metrics["selected_trial_index"]
    )
    reassessed = selected.reassess_quality(minimum_joint_limit_margin=0.9)
    assert not reassessed.metrics["quality_pass"].any()
    assert (
        reassessed.metadata["candidate_selection"]
        == selected.metadata["candidate_selection"]
    )
    for source in study.results:
        for name in (
            "points",
            "joint_positions",
            "manipulability",
            "reachable",
            "residual",
        ):
            assert not np.shares_memory(getattr(source, name), getattr(selected, name))
        for name in source.metrics:
            assert not np.shares_memory(source.metrics[name], selected.metrics[name])
    selected.metadata["candidate_selection"]["trials"][0]["metadata"]["robot"] = (
        "changed"
    )
    selected.joint_positions[:] = 99
    assert study.to_dict() == original
    json.dumps(loaded.metadata, allow_nan=False)


@pytest.mark.parametrize(
    "mutation",
    [
        "flags",
        "joints",
        "residual",
        "residual_over_tolerance",
        "metric_schema",
        "joint_names",
        "solver_setting",
        "thresholds",
        "joint_shape",
    ],
)
def test_invalid_or_incompatible_candidates_are_rejected(mutation):
    study = _stage_study()
    result = study.results[1]
    if mutation == "flags":
        result.metrics["quality_pass"][0] = False
    elif mutation == "joints":
        result.joint_positions[0, 0] = np.nan
    elif mutation == "residual":
        result.residual[0] = np.inf
    elif mutation == "residual_over_tolerance":
        result.residual[0] = 2 * result.metadata["solver_settings"]["tolerance"]
    elif mutation == "metric_schema":
        result.metrics["extra"] = np.zeros(3)
    elif mutation == "joint_names":
        result.metadata["joint_names"] = ["other"] * 3
    elif mutation == "solver_setting":
        result.metadata["solver_settings"]["unexpected"] = 1
    elif mutation == "thresholds":
        for candidate in study.results:
            candidate.metadata["quality_thresholds"] = {}
    else:
        result.joint_positions = result.joint_positions[:, :2]
    with pytest.raises(ValueError):
        study.select_solutions()


def test_invalid_objective_and_cancellation_before_during_and_final(monkeypatch):
    import workspace_analyzer.selection as module

    study = _stage_study()
    with pytest.raises(ValueError, match="objective"):
        study.select_solutions(objective="unknown")
    with pytest.raises(ValueError, match="IKStabilityResult"):
        select_ik_solutions(study.results)
    event = threading.Event()
    event.set()
    with pytest.raises(AnalysisCancelled):
        study.select_solutions(cancel_event=event)
    event.clear()
    original = module.check_cancelled
    calls = []

    def counted(event):
        calls.append(1)
        original(event)

    monkeypatch.setattr(module, "check_cancelled", counted)
    study.select_solutions()
    for cancel_at in (2, len(calls)):
        count = 0

        def cancelling(event):
            nonlocal count
            count += 1
            if count == cancel_at:
                event.set()
            original(event)

        event.clear()
        monkeypatch.setattr(module, "check_cancelled", cancelling)
        with pytest.raises(AnalysisCancelled):
            study.select_solutions(cancel_event=event)


def test_target_pose_selection_keeps_original_order_and_independent_arrays():
    solver = create_solver(FIXTURES / "cartesian_stage.urdf")
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    poses[1, 0, 3] = 0.2
    study = analyze_ik_stability(
        solver,
        poses,
        [IKTrial("a"), IKTrial("b")],
        ReachabilityConfig(position_only=False, restarts=1, rescue_restarts=0),
    )
    result = study.select_solutions()
    np.testing.assert_array_equal(result.target_poses, poses)
    assert not np.shares_memory(result.target_poses, study.results[0].target_poses)
    np.testing.assert_array_equal(
        result.orientation_coverage([0, 1]).ik_coverage, [1, 1]
    )


def test_missing_objective_in_all_successes_still_preserves_ik_union():
    study = _stage_study()
    for result in study.results:
        result.metrics["joint_limit_margin"][:] = np.nan
    selected = study.select_solutions()
    assert selected.reachable.tolist() == [True, True, False]
    assert (
        selected.metadata["candidate_selection"]["selection_time_summary"][
            "objective_comparable_count"
        ]
        == 0
    )


def test_selection_rejects_mixed_coordinate_provenance():
    study = deepcopy(_stage_study())
    study.results[1].metadata["base_from_targets"][0][3] += 1
    with pytest.raises(ValueError, match="base_from_targets"):
        study.select_solutions()


@pytest.mark.parametrize(
    "objective", ["joint_limit_margin", "isotropy", "minimum_singular_value"]
)
def test_batched_selection_matches_independent_scalar_ranking(objective):
    from dataclasses import replace

    rng = np.random.default_rng(721)
    source = _stage_study().results[0]
    count, trials = 61, 7
    rows = np.arange(count) % len(source.points)
    results = []
    for trial in range(trials):
        ik = rng.random(count) > 0.35
        metrics = {key: value[rows].copy() for key, value in source.metrics.items()}
        for metric in ("joint_limit_margin", "isotropy", "minimum_singular_value"):
            metrics[metric] = rng.choice([np.nan, 0.1, 0.5, 0.9], count)
        metrics["task_rank"] = rng.choice([2, 3], count)
        metrics["quality_pass"] = (
            ik
            & (metrics["joint_limit_margin"] >= 0.2)
            & (metrics["isotropy"] >= 0.2)
            & (metrics["minimum_singular_value"] >= 0.2)
            & (metrics["task_rank"] == 3)
        )
        metrics["external_measurement"] = rng.normal(size=count)
        residual = np.where(
            ik, rng.choice([0, 1e-7], count), rng.choice([np.nan, 1, 2], count)
        )
        metadata = deepcopy(source.metadata)
        metadata["quality_thresholds"] = dict(
            minimum_joint_limit_margin=0.2,
            minimum_isotropy=0.2,
            minimum_singular_value=0.2,
            require_full_rank=True,
        )
        results.append(
            replace(
                source,
                points=source.points[rows],
                joint_positions=np.full((count, 3), trial / trials),
                manipulability=source.manipulability[rows],
                reachable=ik,
                residual=residual,
                metrics=metrics,
                metadata=metadata,
            )
        )
    study = IKStabilityResult(tuple(map(str, range(trials))), tuple(results))
    selected = study.select_solutions(objective=objective)
    expected = []
    for row in range(count):

        def rank(index):
            candidate = results[index]
            success = bool(candidate.reachable[row])
            accepted = bool(candidate.metrics["quality_pass"][row])
            score = candidate.metrics[objective][row]
            if not success or not np.isfinite(score):
                score = -np.inf
            error = candidate.residual[row]
            if not np.isfinite(error):
                error = np.inf
            return (int(success) + int(accepted), score, -error, -index)

        expected.append(max(range(trials), key=rank))
    np.testing.assert_array_equal(selected.metrics["selected_trial_index"], expected)
    for row, index in enumerate(expected):
        for name in ("joint_positions", "residual", "reachable", "manipulability"):
            np.testing.assert_array_equal(
                getattr(selected, name)[row], getattr(results[index], name)[row]
            )
        for name, values in results[index].metrics.items():
            np.testing.assert_array_equal(selected.metrics[name][row], values[row])


def test_candidate_validation_avoids_reports_but_rejects_stale_gates(monkeypatch):
    import workspace_analyzer.reachability as module

    study = _stage_study()
    calls = []
    original = module._assessment_summary

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_assessment_summary", counted)
    study.select_solutions()
    assert len(calls) == 1  # Only the final selected result needs a report.
    study.summarize_diversity()
    assert len(calls) == 1
    study.results[0].metrics["quality_pass"][0] = False
    for operation in (study.select_solutions, study.summarize_diversity):
        with pytest.raises(ValueError, match="quality flags"):
            operation()
    assert len(calls) == 1


@pytest.mark.parametrize(
    "mutation",
    [
        "residual_shape",
        "extra_shape",
        "extra_complex",
        "points_shape",
        "zero_weights",
        "negative_weights",
    ],
)
def test_lightweight_validation_rejects_malformed_measurements(mutation):
    study = _stage_study()
    for result in study.results:
        if mutation == "residual_shape":
            result.residual = result.residual[:, None]
        elif mutation == "extra_shape":
            result.metrics["external"] = np.ones((3, 1))
        elif mutation == "extra_complex":
            result.metrics["external"] = np.ones(3, dtype=complex)
        elif mutation == "points_shape":
            result.points = result.points[:, :2]
        elif mutation == "zero_weights":
            result.metrics["task_weight"][:] = 0
        else:
            result.metrics["task_weight"][:] = -1
    for operation in (study.select_solutions, study.summarize_diversity):
        with pytest.raises(ValueError):
            operation()
