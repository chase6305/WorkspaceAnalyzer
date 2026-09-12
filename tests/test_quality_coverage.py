import json
import threading
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from workspace_analyzer import (
    AnalysisCancelled,
    AnalysisResult,
    ReachabilityConfig,
    ResultCache,
    WorkspaceAnalyzer,
    create_solver,
)

URDF = Path(__file__).parent / "fixtures/cartesian_stage.urdf"


def _setup():
    solver = create_solver(URDF, backend="numpy", max_iterations=30)
    config = ReachabilityConfig(restarts=1, rescue_restarts=0, batch_size=3)
    return solver, WorkspaceAnalyzer(solver), config


def test_reassessment_matches_new_run_without_changing_source(monkeypatch, tmp_path):
    solver, analyzer, config = _setup()
    targets = [[0, 0, 0], [0.95, 0, 0], [2, 0, 0]]
    result = analyzer.analyze_targets(targets, config)
    expected = analyzer.analyze_targets(
        targets, replace(config, minimum_joint_limit_margin=0.1), weights=[1, 2, 4]
    )
    result.save(tmp_path / "saved")
    result = AnalysisResult.load(tmp_path / "saved.npz")
    original_metadata = deepcopy(result.metadata)
    original_pass = result.metrics["quality_pass"].copy()

    def unexpected(*args, **kwargs):
        pytest.fail("reassessment must not evaluate kinematics")

    for name in ("inverse", "forward", "forward_with_jacobian", "dexterity"):
        monkeypatch.setattr(solver, name, unexpected)
    actual = result.reassess_quality(minimum_joint_limit_margin=0.1, weights=[1, 2, 4])
    assert actual.metadata["assessment"] == expected.metadata["assessment"]
    np.testing.assert_array_equal(actual.metrics["quality_pass"], [True, False, False])
    np.testing.assert_array_equal(result.metrics["quality_pass"], original_pass)
    assert result.metadata == original_metadata
    assert np.shares_memory(result.joint_positions, actual.joint_positions)
    actual.metadata["solver_settings"]["tolerance"] = 99
    assert result.metadata == original_metadata
    reset = actual.reassess_quality()
    np.testing.assert_array_equal(reset.metrics["quality_pass"], reset.reachable)
    np.testing.assert_array_equal(reset.metrics["task_weight"], [1, 2, 4])


def test_cache_reuses_measurements_when_only_gates_or_task_weights_change(
    tmp_path, monkeypatch
):
    solver, analyzer, config = _setup()
    targets = [[0, 0, 0], [0.95, 0, 0], [2, 0, 0]]
    cache = ResultCache(tmp_path)
    original = analyzer.analyze_targets(targets, config, cache=cache)
    expected = analyzer.analyze_targets(
        targets, replace(config, minimum_joint_limit_margin=0.1), weights=[2, 1, 1]
    )

    def unexpected(*args, **kwargs):
        pytest.fail("cache reclassification should skip IK/FK/Jacobian")

    for name in ("inverse", "forward", "forward_with_jacobian"):
        monkeypatch.setattr(solver, name, unexpected)
    progress = []
    hit = analyzer.analyze_targets(
        targets,
        replace(config, minimum_joint_limit_margin=0.1),
        cache=cache,
        weights=[2, 1, 1],
        progress_callback=progress.append,
    )
    assert hit.metadata["cache_hit"] is True
    assert hit.metadata["cache_key"] == original.metadata["cache_key"]
    assert hit.metadata["assessment"] == expected.metadata["assessment"]
    assert progress == [1.0]
    assert len(list(tmp_path.rglob("*.npz"))) == 1
    again = analyzer.analyze_targets(targets, config, cache=cache)
    assert again.metadata["assessment"] == original.metadata["assessment"]


def test_incomplete_measurement_cache_is_recomputed(tmp_path):
    _, analyzer, config = _setup()
    cache = ResultCache(tmp_path)
    original = analyzer.analyze_targets([[0, 0, 0]], config, cache=cache)
    original.metrics.pop("condition_number")
    cache.put(original.metadata["cache_key"], original)
    repaired = analyzer.analyze_targets([[0, 0, 0]], config, cache=cache)
    assert repaired.metadata["cache_hit"] is False
    assert repaired.metrics["condition_number"][0] == 1


def test_reassessment_refreshes_statistics_and_discards_attached_coverage():
    _, analyzer, config = _setup()
    result = analyzer.analyze_targets([[0, 0, 0]], config)
    result.metrics["isotropy"][0] = 0.1
    result.metadata["orientation_coverage"] = {"stale": True}
    updated = result.reassess_quality(minimum_isotropy=0.2)
    assert not updated.metrics["quality_pass"][0]
    assert (
        updated.metadata["assessment"]["quality_statistics"]["isotropy"]["median"]
        == 0.1
    )
    assert "orientation_coverage" not in updated.metadata
    assert "orientation_coverage" in result.metadata


def test_reassessment_honors_pre_cancellation():
    event = threading.Event()
    event.set()
    from workspace_analyzer import reassess_quality

    with pytest.raises(AnalysisCancelled):
        reassess_quality(None, cancel_event=event)


@pytest.mark.parametrize("value", [True, [0.1], "0.1", float("nan")])
def test_quality_thresholds_require_real_scalars(value):
    with pytest.raises(ValueError):
        ReachabilityConfig(minimum_isotropy=value)


@pytest.mark.parametrize("weights", [[0], [-1], [float("nan")], [1, 2]])
def test_reassessment_validates_replacement_weights(weights):
    _, analyzer, config = _setup()
    result = analyzer.analyze_targets([[0, 0, 0]], config)
    with pytest.raises(ValueError, match="weights"):
        result.reassess_quality(weights=weights)


def _pose_result():
    _, analyzer, config = _setup()
    # Deliberately interleave groups and start with a lexically later ID.
    targets = np.broadcast_to(np.eye(4), (5, 4, 4)).copy()
    targets[[0, 2, 4], 0, 3] = 0.95
    targets[2, :3, :3] = np.diag([1, -1, -1])
    targets[3, :3, :3] = np.diag([-1, 1, -1])
    result = analyzer.analyze_targets(
        targets,
        replace(config, position_only=False, minimum_joint_limit_margin=0.1),
        weights=[1, 0, 3, 0, 2],
    )
    return result, np.array(["z", "a", "z", "a", "z"])


def test_orientation_coverage_preserves_group_order_and_counts():
    result, ids = _pose_result()
    coverage = result.orientation_coverage(ids)
    np.testing.assert_array_equal(coverage.position_ids, ["z", "a"])
    np.testing.assert_array_equal(coverage.sample_counts, [3, 2])
    np.testing.assert_array_equal(coverage.ik_success_counts, [2, 1])
    np.testing.assert_array_equal(coverage.quality_pass_counts, [0, 1])
    np.testing.assert_allclose(coverage.ik_coverage, [2 / 3, 0.5])
    np.testing.assert_allclose(coverage.quality_coverage, [0, 0.5])
    assert coverage.weighted_ik_coverage[0] == 0.5
    assert np.isnan(coverage.weighted_ik_coverage[1])
    report = coverage.to_dict()
    assert report["weighted_ik_coverage"] == [0.5, None]
    assert report["summary"]["positions_with_any_quality"] == 1
    assert report["summary"]["positions_with_all_ik"] == 0
    json.dumps(report, allow_nan=False)
    # Relaxation only changes quality coverage, preserving the tested poses.
    relaxed = result.reassess_quality().orientation_coverage(ids)
    np.testing.assert_allclose(relaxed.quality_coverage, relaxed.ik_coverage)


def test_group_weight_scaling_handles_extreme_relative_masses():
    result, ids = _pose_result()
    result = result.reassess_quality(weights=[1e308, 1e-308, 1e308, 1e-308, 1e308])
    coverage = result.orientation_coverage(ids)
    np.testing.assert_allclose(coverage.weighted_ik_coverage, [2 / 3, 0.5])


def test_group_position_tolerance_and_integer_ids():
    result, _ = _pose_result()
    result.points[4, 0] += 1e-9
    result.target_poses[4, 0, 3] += 1e-9
    ids = [7, -1, 7, -1, 7]
    np.testing.assert_array_equal(
        result.orientation_coverage(ids).position_ids, [7, -1]
    )
    with pytest.raises(ValueError, match="share a position"):
        result.orientation_coverage(ids, position_tolerance=1e-12)


@pytest.mark.parametrize(
    "kind",
    [
        "position_only",
        "missing_poses",
        "bad_ids",
        "wrong_group",
        "negative_tolerance",
        "bad_quality",
        "bad_pose_points",
    ],
)
def test_orientation_coverage_rejects_invalid_groups_and_nonpose_results(kind):
    result, ids = _pose_result()
    kwargs = {}
    if kind == "position_only":
        result.metadata["position_only"] = True
    elif kind == "missing_poses":
        result.target_poses = None
    elif kind == "bad_ids":
        ids = [0.1] * 5
    elif kind == "wrong_group":
        ids = [1] * 5
    elif kind == "negative_tolerance":
        kwargs["position_tolerance"] = -1
    elif kind == "bad_quality":
        result.metrics["quality_pass"][2] = True
    else:
        result.target_poses[0, 0, 3] += 1
    with pytest.raises(ValueError):
        result.orientation_coverage(ids, **kwargs)


QUALITY_GATES = [
    ("isotropy", "minimum_isotropy"),
    ("joint_limit_margin", "minimum_joint_limit_margin"),
    ("minimum_singular_value", "minimum_singular_value"),
]


@pytest.mark.parametrize("metric,gate", QUALITY_GATES)
def test_enabled_gates_require_finite_measurements_and_keep_ik_outcomes(metric, gate):
    _, analyzer, config = _setup()
    targets = [[0, 0, 0]] * 5 + [[2, 0, 0]]
    source = analyzer.analyze_targets(targets, config, weights=[1, 2, 3, 4, 5, 6])
    source.metrics[metric][:] = [np.inf, np.nan, -np.inf, 0.2, 0.19, 0.9]
    updated = source.reassess_quality(**{gate: 0.2})
    np.testing.assert_array_equal(updated.reachable, [True] * 5 + [False])
    np.testing.assert_array_equal(
        updated.metrics["quality_pass"], [False, False, False, True, False, False]
    )
    summary = updated.metadata["assessment"]
    assert summary["threshold_failures"] == {gate: 4}
    assert summary["quality_pass_count"] == 1
    assert summary["weighted_quality_pass_rate"] == pytest.approx(4 / 21)
    assert summary["quality_statistics"][metric]["infinite_count"] == 2
    assert summary["quality_statistics"][metric]["finite_count"] == 2
    json.dumps(summary, allow_nan=False)
    np.testing.assert_array_equal(updated.metrics[metric], source.metrics[metric])
    np.testing.assert_array_equal(source.metrics["quality_pass"], source.reachable)
    # With no gate on this metric, it does not alter geometric IK classification.
    np.testing.assert_array_equal(
        updated.reassess_quality().metrics["quality_pass"], source.reachable
    )


@pytest.mark.parametrize("metric,gate", QUALITY_GATES)
def test_fresh_assessment_rejects_overflowed_quality_measurement(
    monkeypatch, metric, gate
):
    import workspace_analyzer.reachability as module

    _, analyzer, config = _setup()
    original = module._dexterity_from_jacobian

    def overflowed(*args, **kwargs):
        result = original(*args, **kwargs)
        getattr(result, metric)[0] = np.inf
        return result

    monkeypatch.setattr(module, "_dexterity_from_jacobian", overflowed)
    result = analyzer.analyze_targets(
        [[0, 0, 0], [0.4, 0, 0]], replace(config, **{gate: 0.2})
    )
    np.testing.assert_array_equal(result.reachable, [True, True])
    np.testing.assert_array_equal(result.metrics["quality_pass"], [False, True])
    assert np.isinf(result.metrics[metric][0])
    assert result.metadata["assessment"]["threshold_failures"] == {gate: 1}


@pytest.mark.parametrize("metric,gate", QUALITY_GATES)
def test_cached_infinite_measurements_are_reclassified_without_ik(
    tmp_path, monkeypatch, metric, gate
):
    solver, analyzer, config = _setup()
    cache = ResultCache(tmp_path)
    targets = [[0, 0, 0]]
    result = analyzer.analyze_targets(targets, config, cache=cache)
    result.metrics[metric][0] = np.inf
    cache.put(result.metadata["cache_key"], result)

    def unexpected(*args, **kwargs):
        pytest.fail("quality reclassification must reuse cached measurements")

    for name in ("inverse", "forward", "forward_with_jacobian"):
        monkeypatch.setattr(solver, name, unexpected)
    replay = analyzer.analyze_targets(
        targets, replace(config, **{gate: 0.2}), cache=cache
    )
    assert replay.metadata["cache_hit"]
    assert replay.reachable[0] and not replay.metrics["quality_pass"][0]
    assert np.isinf(replay.metrics[metric][0])


@pytest.mark.parametrize("metric,gate", QUALITY_GATES)
def test_candidates_reject_legacy_infinite_acceptance_and_select_finite_alternative(
    metric, gate
):
    from workspace_analyzer import IKTrial, analyze_ik_stability

    solver, _, config = _setup()
    study = analyze_ik_stability(
        solver,
        [[0, 0, 0]],
        [IKTrial("overflowed"), IKTrial("finite")],
        replace(config, **{gate: 0.2}),
    )
    study.results[0].metrics[metric][0] = np.inf
    for operation in (study.select_solutions, study.summarize_diversity):
        with pytest.raises(ValueError, match="quality flags"):
            operation()
    updated = study.reassess_quality(**{gate: 0.2})
    selected = updated.select_solutions(objective=metric)
    assert selected.metrics["selected_trial_index"].tolist() == [1]
    assert selected.metrics["quality_pass"].tolist() == [True]
    assert updated.summarize_diversity()["summary"]["quality_candidate_count"] == 1


@pytest.mark.parametrize("value", ["1e400", "1e-400"])
def test_reassessment_rejects_weights_lost_in_float64_conversion(value):
    if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
        pytest.skip("extended floating precision is unavailable")
    _, analyzer, config = _setup()
    result = analyzer.analyze_targets([[0, 0, 0]], config)
    original = result.metadata.copy()
    weights = np.array([value], dtype=np.longdouble)
    with np.errstate(over="raise", invalid="raise"):
        with pytest.raises(ValueError, match="weights.*float64"):
            result.reassess_quality(weights=weights)
    assert result.metadata == original
    np.testing.assert_array_equal(result.metrics["task_weight"], [1])


@pytest.mark.parametrize("consumer", ["coverage", "perturbations", "stability"])
@pytest.mark.parametrize("mutation", ["measurement", "flags", "thresholds"])
def test_derived_reports_reject_stale_quality_until_explicit_reassessment(
    consumer, mutation, monkeypatch
):
    import workspace_analyzer.reachability as module
    from workspace_analyzer import IKTrial, PosePerturbations, analyze_ik_stability

    solver, analyzer, config = _setup()
    config = replace(config, position_only=False, minimum_isotropy=0.5)
    poses = np.eye(4)[None]
    perturbations = PosePerturbations(poses, translation_m=0, rotation_rad=0)
    if consumer == "stability":
        study = analyze_ik_stability(
            solver, poses, [IKTrial("a"), IKTrial("b")], config
        )
        results = study.results
    else:
        results = (analyzer.analyze_targets(perturbations.targets, config),)
    for result in results:
        if mutation == "measurement":
            result.metrics["isotropy"][0] = np.inf
        elif mutation == "flags":
            result.metrics["quality_pass"][0] = False
        else:
            result.metrics["isotropy"][0] = 0.75
            result.metadata["quality_thresholds"]["minimum_isotropy"] = 0.9

    def summarize():
        if consumer == "coverage":
            return results[0].orientation_coverage([0]).to_dict()
        if consumer == "perturbations":
            return perturbations.summarize(results[0])
        return study.to_dict()

    def unexpected(*args, **kwargs):
        pytest.fail("checking saved flags must not rebuild statistical reports")

    with monkeypatch.context() as patch:
        patch.setattr(module, "_assessment_summary", unexpected)
        with pytest.raises(ValueError, match="quality flags"):
            summarize()
    # Reassessment is explicit and preserves source measurements/flags.
    updated = tuple(
        result.reassess_quality(**result.metadata["quality_thresholds"])
        for result in results
    )
    if consumer == "stability":
        from workspace_analyzer import IKStabilityResult

        report = IKStabilityResult(study.names, updated).to_dict()
    elif consumer == "coverage":
        report = updated[0].orientation_coverage([0]).to_dict()
    else:
        report = perturbations.summarize(updated[0])
    json.dumps(report, allow_nan=False)
    assert updated[0].metrics["quality_pass"][0] == (mutation == "flags")
    with pytest.raises(ValueError, match="quality flags"):
        summarize()
