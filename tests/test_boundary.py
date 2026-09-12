import json
import threading
from pathlib import Path

import numpy as np
import pytest

from workspace_analyzer import (
    AnalysisCancelled,
    AnalysisResult,
    BoundaryConfig,
    ReachabilityConfig,
    ResultCache,
    create_solver,
    refine_translation_boundary,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _solver(backend="numpy", dtype="float64", fixture="cartesian_stage.urdf"):
    if backend == "torch":
        pytest.importorskip("torch")
    return create_solver(
        FIXTURES / fixture, backend=backend, dtype=dtype, max_iterations=80
    )


def _poses(points, dtype="float64"):
    poses = np.broadcast_to(np.eye(4, dtype=dtype), (len(points), 4, 4)).copy()
    poses[:, :3, 3] = points
    return poses


def _settings(**kwargs):
    return ReachabilityConfig(restarts=1, rescue_restarts=0, batch_size=3, **kwargs)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_analytic_stage_brackets_have_verified_witnesses_and_bounded_cost(
    backend, dtype, tmp_path
):
    solver = _solver(backend, dtype)
    poses = _poses([[0, 0, 0], [0.1, 0.3, 0]], dtype)
    ends = [[2, 0, 0], [-2, 0.3, 0]]
    progress = []
    result = refine_translation_boundary(
        solver,
        poses,
        ends,
        BoundaryConfig(tolerance_m=5e-4),
        reachability=_settings(position_only=False),
        progress_callback=progress.append,
    )
    np.testing.assert_array_equal(result.statuses, ["refined", "refined"])
    m = result.measurements
    assert m.reachable[result.lower_indices].all()
    assert not m.reachable[result.upper_indices].any()
    report = result.to_dict()
    for row, boundary in zip(report["segments"], [1, -1]):
        assert row["width_m"] <= 5e-4
        # Numerical IK tolerance permits tiny geometric violations of the cube.
        lo, hi = sorted([row["lower"]["point"][0], row["upper"]["point"][0]])
        assert lo - solver.config.tolerance <= boundary <= hi + solver.config.tolerance
        assert row["rounds"] <= 13
    assert report["summary"]["target_evaluations"] <= 2 * (3 + 13)
    assert np.all(np.diff(progress) >= 0) and progress[-1] == 1
    assert m.points.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(
        result.stages[result.upper_indices], ["end_recheck"] * 2
    )
    json.dumps(report, allow_nan=False)
    result.save(tmp_path)
    loaded = AnalysisResult.load(tmp_path / "measurements.npz")
    np.testing.assert_array_equal(loaded.joint_positions, m.joint_positions)
    with np.load(tmp_path / "trace.npz", allow_pickle=False) as trace:
        np.testing.assert_array_equal(trace["segment_ids"], result.segment_ids)
    assert json.loads((tmp_path / "report.json").read_text()) == report


def test_two_link_radial_boundary_against_analytic_outer_radius():
    solver = _solver(fixture="two_link.urdf")
    # Seed with a bent arm to avoid a zero radial derivative at the straight pose.
    result = refine_translation_boundary(
        solver,
        _poses([[1.5, 0, 0]]),
        [2.5, 0, 0],
        BoundaryConfig(tolerance_m=1e-3),
        reachability=_settings(position_only=True),
        seed=[-0.7, 1.4],
    )
    row = result.to_dict()["segments"][0]
    assert row["status"] == "refined"
    assert abs(row["lower"]["point"][0] - 2) < 1e-3
    assert abs(row["upper"]["point"][0] - 2) < 1e-3


def test_unbracketed_segments_do_not_spend_refinement_budget():
    solver = _solver()
    result = refine_translation_boundary(
        solver,
        _poses([[0, 0, 0], [2, 0, 0], [2, 0, 0]]),
        [[0.5, 0, 0], [0, 0, 0], [3, 0, 0]],
        reachability=_settings(),
    )
    np.testing.assert_array_equal(
        result.statuses, ["end_succeeded", "start_failed", "start_failed"]
    )
    assert result.to_dict()["summary"]["initial_brackets"] == 0
    assert len(result.measurements.points) == 6
    assert not result.rounds.any()


def test_small_budget_reports_unresolved_interval_without_false_precision():
    result = refine_translation_boundary(
        _solver(),
        np.eye(4),
        [2, 0, 0],
        BoundaryConfig(max_rounds=1),
        reachability=_settings(),
    )
    row = result.to_dict()["segments"][0]
    assert row["status"] == "budget_exhausted"
    assert row["width_m"] > 0.9
    assert row["rounds"] == 1
    assert row["upper"]["ik_success"] is False


def test_initial_bracket_within_tolerance_needs_only_endpoint_checks():
    result = refine_translation_boundary(
        _solver(),
        _poses([[0.999, 0, 0]]),
        [1.001, 0, 0],
        BoundaryConfig(tolerance_m=0.01, max_rounds=0),
        reachability=_settings(),
    )
    assert result.statuses.tolist() == ["refined"]
    assert result.rounds.tolist() == [0]
    assert result.to_dict()["summary"]["target_evaluations"] == 3


def test_float32_precision_limit_terminates_without_repeating_same_target():
    result = refine_translation_boundary(
        _solver(dtype="float32"),
        np.eye(4),
        [2, 0, 0],
        BoundaryConfig(tolerance_m=1e-12, max_rounds=100),
        reachability=_settings(),
    )
    row = result.to_dict()["segments"][0]
    assert row["status"] == "precision_limit"
    assert 0 < row["width_m"] < 1e-6
    assert row["rounds"] < 30


def test_final_endpoint_recovery_invalidates_boundary(monkeypatch):
    # Emulate an IK search that recovers its provisional failure with a nearer seed.
    import workspace_analyzer.boundary as module

    original = module.analyze_targets
    calls = 0

    def classify(*args, **kwargs):
        nonlocal calls
        calls += 1
        measured = original(*args, **kwargs)
        if calls == 2:
            measured.reachable[:] = False
            measured.metrics["quality_pass"][:] = False
        return measured

    monkeypatch.setattr(module, "analyze_targets", classify)
    result = refine_translation_boundary(
        _solver(),
        np.eye(4),
        [0.5, 0, 0],
        BoundaryConfig(max_rounds=1),
        reachability=_settings(),
    )
    assert result.statuses.tolist() == ["end_recovered"]
    row = result.to_dict()["segments"][0]
    assert row["upper"]["ik_success"]
    assert result.to_dict()["summary"]["refined"] == 0


def test_boundary_cache_replay_skips_ik_and_matches_trace(tmp_path, monkeypatch):
    solver = _solver()
    kwargs = dict(
        config=BoundaryConfig(tolerance_m=1e-3),
        reachability=_settings(),
        cache=ResultCache(tmp_path),
    )
    result = refine_translation_boundary(solver, np.eye(4), [2, 0, 0], **kwargs)

    def unexpected(*args, **kwargs):
        pytest.fail("identical adaptive replay should use cached measurements")

    monkeypatch.setattr(solver, "inverse", unexpected)
    repeated = refine_translation_boundary(solver, np.eye(4), [2, 0, 0], **kwargs)
    assert repeated.to_dict()["segments"] == result.to_dict()["segments"]
    np.testing.assert_array_equal(repeated.parameters, result.parameters)
    assert all(
        batch["cache_hit"]
        for batch in repeated.measurements.metadata["measurement_batches"]
    )


@pytest.mark.parametrize("when", ["before", "during", "final"])
def test_boundary_cancellation_does_not_return_a_result(when):
    event = threading.Event()
    if when == "before":
        event.set()

    def progress(value):
        if when == "during" or value == 1:
            event.set()

    with pytest.raises(AnalysisCancelled):
        refine_translation_boundary(
            _solver(),
            np.eye(4),
            [2, 0, 0],
            reachability=_settings(),
            cancel_event=event,
            progress_callback=progress,
        )


@pytest.mark.parametrize(
    "settings",
    [
        {"tolerance_m": 0},
        {"tolerance_m": np.nan},
        {"tolerance_m": True},
        {"tolerance_m": "0.01"},
        {"max_rounds": -1},
        {"max_rounds": 2.5},
        {"max_rounds": True},
    ],
)
def test_invalid_boundary_settings(settings):
    with pytest.raises(ValueError):
        BoundaryConfig(**settings)


@pytest.mark.parametrize(
    "poses, ends",
    [
        (np.eye(4), [0, 0, 0]),
        (np.eye(4), [np.nan, 0, 0]),
        (np.eye(4), [[1, 0, 0], [2, 0, 0]]),
        ([], []),
        (np.zeros((4, 4)), [1, 0, 0]),
    ],
)
def test_invalid_segment_inputs(poses, ends):
    with pytest.raises(ValueError):
        refine_translation_boundary(_solver(), poses, ends)
