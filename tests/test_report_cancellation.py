"""Cooperative cancellation of offline coverage and perturbation aggregation."""

import threading
from pathlib import Path

import numpy as np
import pytest

from workspace_analyzer import (
    AnalysisCancelled,
    PosePerturbations,
    ReachabilityConfig,
    WorkspaceAnalyzer,
    create_solver,
    summarize_orientation_coverage,
)


def _inputs():
    solver = create_solver(
        Path(__file__).parent / "fixtures/cartesian_stage.urdf", backend="numpy"
    )
    poses = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
    poses[1, 0, 3] = 0.95
    perturbations = PosePerturbations(poses, translation_m=0.1, rotation_rad=0)
    result = WorkspaceAnalyzer(solver).analyze_targets(
        perturbations.targets,
        ReachabilityConfig(position_only=False, restarts=1, rescue_restarts=0),
    )
    return perturbations, result


@pytest.mark.parametrize(
    "operation", ["coverage", "coverage_function", "perturbations"]
)
def test_pre_cancelled_reports_do_not_access_inputs(operation):
    event = threading.Event()
    event.set()

    def call():
        if operation == "perturbations":
            generator = PosePerturbations(np.eye(4)[None])
            return generator.summarize(object(), cancel_event=event)
        if operation == "coverage_function":
            return summarize_orientation_coverage(
                object(), object(), cancel_event=event
            )
        from workspace_analyzer import AnalysisResult

        return AnalysisResult.orientation_coverage(
            object(), object(), cancel_event=event
        )

    with pytest.raises(AnalysisCancelled):
        call()


@pytest.mark.parametrize("operation", ["coverage", "perturbations"])
@pytest.mark.parametrize("when", ["middle", "final"])
def test_reports_cancel_without_partial_result_or_source_mutation(operation, when):
    generator, source = _inputs()
    # Each pose is an independent coverage group because translations vary.
    ids = np.arange(len(source.points))

    class Event:
        def __init__(self, stop=None):
            self.stop, self.calls = stop, 0

        def is_set(self):
            self.calls += 1
            return self.calls == self.stop

    def report(event=None):
        if operation == "coverage":
            return source.orientation_coverage(ids, cancel_event=event).to_dict()
        return generator.summarize(source, cancel_event=event)

    counted = Event()
    expected = report(counted)
    points = source.points.copy()
    quality = source.metrics["quality_pass"].copy()
    stop = counted.calls // 2 if when == "middle" else counted.calls
    with pytest.raises(AnalysisCancelled):
        report(Event(stop))
    np.testing.assert_array_equal(source.points, points)
    np.testing.assert_array_equal(source.metrics["quality_pass"], quality)
    assert report() == expected


def test_coverage_observes_cancellation_after_grouping(monkeypatch):
    import workspace_analyzer.coverage as module

    _, result = _inputs()
    event = threading.Event()
    original = module.np.unique

    def grouping(*args, **kwargs):
        values = original(*args, **kwargs)
        event.set()
        return values

    monkeypatch.setattr(module.np, "unique", grouping)
    with pytest.raises(AnalysisCancelled):
        result.orientation_coverage(np.arange(len(result.points)), cancel_event=event)
