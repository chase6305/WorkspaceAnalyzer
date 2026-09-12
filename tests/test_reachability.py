import json
import threading
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from workspace_analyzer import (
    AnalysisCancelled,
    AnalysisResult,
    CartesianConfig,
    ReachabilityConfig,
    ResultCache,
    SamplingConfig,
    WorkspaceAnalyzer,
    analysis_cache_key,
    create_solver,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _solver(backend="numpy", dtype="float64", fixture="cartesian_stage.urdf"):
    if backend == "torch":
        pytest.importorskip("torch")
    return create_solver(
        FIXTURES / fixture,
        backend=backend,
        device="cpu",
        dtype=dtype,
        max_iterations=40,
    )


def _config(**kwargs):
    return ReachabilityConfig(restarts=1, rescue_restarts=0, batch_size=3, **kwargs)


def _numpy(value):
    return (
        value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    )


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_analytic_stage_separates_ik_failure_and_quality_rejection(backend, dtype):
    solver = _solver(backend, dtype)
    targets = np.array([[0, 0, 0], [0.2, -0.3, 0.4], [0.95, 0, 0], [1.4, 0, 0]])
    progress = []
    result = WorkspaceAnalyzer(solver).analyze_targets(
        targets,
        _config(
            minimum_joint_limit_margin=0.1, minimum_isotropy=0.5, require_full_rank=True
        ),
        weights=[1, 2, 4, 8],
        progress_callback=progress.append,
    )
    np.testing.assert_array_equal(result.reachable, [True, True, True, False])
    np.testing.assert_array_equal(
        result.metrics["quality_pass"], [True, True, False, False]
    )
    np.testing.assert_array_equal(result.metrics["task_rank"], [3, 3, 3, -1])
    np.testing.assert_allclose(result.manipulability[:3], 1)
    np.testing.assert_allclose(result.metrics["minimum_singular_value"][:3], 1)
    np.testing.assert_allclose(result.metrics["isotropy"][:3], 1)
    np.testing.assert_allclose(result.metrics["condition_number"][:3], 1)
    np.testing.assert_allclose(
        result.metrics["joint_limit_margin"][:3], [1, 0.6, 0.05], atol=1e-5
    )
    assert np.isnan(result.manipulability[-1])
    assert np.isnan(result.metrics["rotation_error"]).all()
    assert progress == [0.75, 1.0]
    assert result.joint_positions.dtype == np.dtype(dtype)
    reached = _numpy(solver.forward(result.joint_positions))[:, :3, 3]
    np.testing.assert_allclose(
        result.metrics["position_error"],
        np.linalg.norm(reached - result.points, axis=1),
        atol=1e-7,
    )
    report = result.metadata["assessment"]
    assert report["ik_success_count"] == 3
    assert report["ik_failure_count"] == report["quality_rejected_count"] == 1
    assert report["weighted_ik_success_rate"] == pytest.approx(7 / 15)
    assert report["weighted_quality_pass_rate"] == pytest.approx(3 / 15)
    assert report["threshold_failures"]["minimum_joint_limit_margin"] == 1
    assert report["quality_statistics"]["isotropy"]["count"] == 3
    assert result.metadata["collision_checked"] is False
    json.dumps(result.metadata, allow_nan=False)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_full_pose_reports_rotation_error_and_task_rank(backend, tmp_path):
    solver = _solver(backend)
    targets = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
    targets[1, :3, :3] = np.diag([1, -1, -1])
    result = WorkspaceAnalyzer(solver).analyze_targets(
        targets,
        _config(position_only=False, dexterity_task="pose", require_full_rank=True),
    )
    np.testing.assert_array_equal(result.reachable, [True, False])
    np.testing.assert_array_equal(result.metrics["quality_pass"], [False, False])
    np.testing.assert_array_equal(result.metrics["task_rank"], [3, -1])
    assert result.metadata["assessment"]["task_dimension"] == 6
    np.testing.assert_allclose(result.metrics["rotation_error"], [0, np.pi])
    np.testing.assert_allclose(result.metrics["position_error"], 0)
    result.save(tmp_path / "poses")
    restored = AnalysisResult.load(tmp_path / "poses.npz")
    np.testing.assert_array_equal(restored.target_poses, targets)
    np.testing.assert_array_equal(restored.metrics["quality_pass"], [False, False])


def test_planar_arm_distinguishes_singular_pose_from_unreachable_target():
    solver = _solver(fixture="two_link.urdf")
    seeds = np.array([[0, 0], [0.2, 0.7], [0, 0]])
    targets = solver.forward(seeds)[:, :3, 3]
    targets[-1] = [3, 0, 0]
    result = WorkspaceAnalyzer(solver).analyze_targets(
        targets, _config(minimum_isotropy=0.1), seed=seeds
    )
    np.testing.assert_array_equal(result.reachable, [True, True, False])
    np.testing.assert_array_equal(result.metrics["quality_pass"], [False, True, False])
    np.testing.assert_array_equal(result.metrics["task_rank"], [1, 2, -1])
    assert np.isinf(result.metrics["condition_number"][0])
    assert result.metadata["assessment"]["full_task_rank_count"] == 0
    stats = result.metadata["assessment"]["quality_statistics"]["condition_number"]
    assert stats["infinite_count"] == stats["finite_count"] == 1
    json.dumps(result.metadata, allow_nan=False)


@pytest.mark.parametrize("pose_input", [False, True])
def test_coordinate_transform_and_single_target_preserve_target_data(pose_input):
    solver = _solver()
    base_from_input = np.eye(4)
    base_from_input[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    base_from_input[:3, 3] = [0.1, 0.2, 0.3]
    base_target = np.eye(4)
    base_target[:3, 3] = [0.4, -0.2, 0.1]
    input_pose = np.linalg.inv(base_from_input) @ base_target
    targets = input_pose if pose_input else input_pose[:3, 3]
    original = targets.copy()
    result = WorkspaceAnalyzer(solver).analyze_targets(
        targets,
        _config(position_only=not pose_input),
        base_from_targets=base_from_input,
    )
    assert result.reachable.shape == (1,)
    assert result.reachable.all()
    np.testing.assert_allclose(result.points[0], base_target[:3, 3])
    np.testing.assert_array_equal(targets, original)
    if pose_input:
        np.testing.assert_allclose(result.target_poses[0], base_target)


def test_failed_targets_skip_jacobians_and_successful_targets_reuse_fk(monkeypatch):
    solver = _solver()
    calls = []
    original = solver.forward_with_jacobian

    def record(q):
        calls.append(len(q))
        return original(q)

    def unexpected(*args, **kwargs):
        pytest.fail("assessment must reuse the FK/Jacobian traversal")

    monkeypatch.setattr(solver, "forward_with_jacobian", record)
    monkeypatch.setattr(solver, "_jacobian_batch", unexpected)
    result = WorkspaceAnalyzer(solver).analyze_targets(
        [[0, 0, 0], [2, 0, 0], [0.1, 0.1, 0.1], [3, 0, 0]], _config()
    )
    assert calls == [2]
    np.testing.assert_array_equal(result.reachable, [True, False, True, False])


def test_all_failed_assessment_has_empty_quality_statistics():
    result = WorkspaceAnalyzer(_solver()).analyze_targets([[2, 0, 0]], _config())
    stats = result.metadata["assessment"]["quality_statistics"]["isotropy"]
    assert stats["count"] == 0 and stats["median"] is None
    assert np.isnan(result.manipulability).all()
    assert result.metadata["assessment"]["quality_pass_rate"] == 0


def test_direction_weights_and_large_task_weights_are_distinct():
    config = _config(dexterity_weights=[1, 0.5, 0.25], minimum_singular_value=0.3)
    result = WorkspaceAnalyzer(_solver()).analyze_targets(
        [[0, 0, 0], [2, 0, 0]], config, weights=[1e308, 1e308]
    )
    assert result.metrics["minimum_singular_value"][0] == pytest.approx(0.25)
    assert not result.metrics["quality_pass"].any()
    assert result.metadata["assessment"]["weighted_ik_success_rate"] == 0.5
    json.dumps(result.metadata, allow_nan=False)


def test_cached_measurements_track_targets_transform_and_seed(tmp_path):
    analyzer = WorkspaceAnalyzer(_solver())
    cache = ResultCache(tmp_path)
    targets = np.array([[0, 0, 0], [0.2, 0, 0]])
    config = _config()
    first = analyzer.analyze_targets(targets, config, cache=cache)
    hit = analyzer.analyze_targets(targets, config, cache=cache)
    assert hit.metadata["cache_hit"] is True
    keys = {first.metadata["cache_key"]}
    transform = np.eye(4)
    transform[0, 3] = 0.1
    for changed_targets, changed_config, options in [
        (targets + 0.1, config, {}),
        (targets, config, {"base_from_targets": transform}),
        (targets, config, {"seed": [0, 0.1, 0]}),
        (targets, replace(config, dexterity_weights=[1.0, 2.0, 1.0]), {}),
    ]:
        result = analyzer.analyze_targets(
            changed_targets, changed_config, cache=cache, **options
        )
        assert result.metadata["cache_hit"] is False
        keys.add(result.metadata["cache_key"])
    assert len(keys) == 5


@pytest.mark.parametrize("cached", [False, True])
def test_final_cancellation_does_not_publish_result(cached, tmp_path):
    analyzer = WorkspaceAnalyzer(_solver())
    cache = ResultCache(tmp_path)
    targets = [[0, 0, 0]]
    if cached:
        analyzer.analyze_targets(targets, _config(), cache=cache)
    cancel = threading.Event()
    with pytest.raises(AnalysisCancelled):
        analyzer.analyze_targets(
            targets,
            _config(),
            cache=cache,
            cancel_event=cancel,
            progress_callback=lambda _: cancel.set(),
        )
    assert len(list(tmp_path.rglob("*.npz"))) == int(cached)


def test_pre_cancelled_assessment_does_not_read_targets():
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(AnalysisCancelled):
        WorkspaceAnalyzer(_solver()).analyze_targets(object(), cancel_event=cancel)


@pytest.mark.parametrize(
    "targets,options,message",
    [
        ([], {}, "shape"),
        (np.empty((0, 3)), {}, "at least one"),
        ([[0, 0]], {}, "shape"),
        ([[0, np.nan, 0]], {}, "finite"),
        ([[0, 0, 0]], {"weights": [0]}, "weights"),
        ([[0, 0, 0]], {"weights": [-1]}, "weights"),
        ([[0, 0, 0]], {"weights": [1, 2]}, "weights"),
        ([[0, 0, 0]], {"seed": [0]}, "seed"),
        ([[0, 0, 0]], {"base_from_targets": np.zeros((3, 3))}, "shape"),
        ([[0, 0, 0]], {"config": _config(position_only=False)}, "full-pose"),
        (np.zeros((1, 4, 4)), {}, "homogeneous"),
        (np.diag([1, 1, -1, 1])[None], {}, "proper"),
    ],
)
def test_invalid_targets_and_options_are_rejected(targets, options, message):
    with pytest.raises(ValueError, match=message):
        WorkspaceAnalyzer(_solver()).analyze_targets(targets, **options)


@pytest.mark.parametrize(
    "options",
    [
        {"batch_size": 0},
        {"restarts": True},
        {"rescue_rounds": -1},
        {"random_seed": 1.5},
        {"position_only": 1},
        {"require_full_rank": "yes"},
        {"minimum_isotropy": 1.1},
        {"minimum_joint_limit_margin": -0.1},
        {"minimum_singular_value": np.inf},
        {"dexterity_task": "unknown"},
        {"dexterity_weights": [1, 1]},
        {"dexterity_weights": [0, 0, 0]},
    ],
)
def test_invalid_assessment_settings_are_rejected(options):
    with pytest.raises(ValueError):
        ReachabilityConfig(**options)


def test_cartesian_integer_reference_pose_keeps_fractional_targets():
    solver = _solver()
    config = CartesianConfig(
        bounds=[[0.2, 0.3], [0.3, 0.4], [0.4, 0.5]],
        reference_pose=np.eye(4, dtype=int),
        sampling=SamplingConfig(num_samples=4, batch_size=3),
        restarts=1,
        rescue_restarts=0,
    )
    result = WorkspaceAnalyzer(solver).analyze_cartesian(config)
    assert result.reachable.all()
    np.testing.assert_allclose(
        solver.forward(result.joint_positions)[:, :3, 3], result.points, atol=1e-5
    )


def test_config_snapshots_cannot_be_changed_through_caller_arrays():
    solver = _solver()
    bounds, pose, seed, weights = (
        np.tile([-1.0, 1.0], (3, 1)),
        np.eye(4),
        np.zeros(3),
        np.ones(3),
    )
    config = CartesianConfig(bounds=bounds, reference_pose=pose, reference_joints=seed)
    assessment = _config(dexterity_weights=weights)
    original_key = analysis_cache_key(solver, "cartesian", config)
    bounds[:] = 5
    pose[:] = 0
    seed[:] = 1
    weights[:] = 0
    assert analysis_cache_key(solver, "cartesian", config) == original_key
    np.testing.assert_array_equal(assessment.dexterity_weights, np.ones(3))
    for value in (
        config.bounds,
        config.reference_pose,
        config.reference_joints,
        assessment.dexterity_weights,
    ):
        assert not value.flags.writeable


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize(
    "kind",
    ["targets", "seed", "base_from_targets", "weight_overflow", "weight_underflow"],
)
def test_precision_invalid_inputs_fail_before_solver_or_cache(
    backend, kind, monkeypatch
):
    solver = _solver(backend, "float32")
    config = ReachabilityConfig(restarts=1, rescue_restarts=0)
    targets = [[0, 0, 0]]
    kwargs = {}
    expected = kind
    if kind == "targets":
        targets = [[1e100, 0, 0]]
    elif kind == "seed":
        kwargs["seed"] = [1e100, 0, 0]
    elif kind == "base_from_targets":
        transform = np.eye(4)
        transform[0, 3] = 1e100
        kwargs["base_from_targets"] = transform
    else:
        expected = "weights"
        config = replace(
            config,
            dexterity_weights=[1e100 if kind == "weight_overflow" else 1e-100] * 3,
        )

    def unexpected(*args, **kwargs):
        pytest.fail(
            "invalid converted inputs must fail before computation or cache I/O"
        )

    class NoReads:
        get = unexpected

    for name in ("inverse", "forward_with_jacobian"):
        monkeypatch.setattr(solver, name, unexpected)
    with np.errstate(over="raise", invalid="raise"):
        with pytest.raises(ValueError, match=expected):
            WorkspaceAnalyzer(solver).analyze_targets(
                targets, config, cache=NoReads(), **kwargs
            )


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_float32_representable_small_weights_preserve_task_conditioning(backend):
    solver = _solver(backend, "float32")
    result = WorkspaceAnalyzer(solver).analyze_targets(
        [[0, 0, 0]],
        ReachabilityConfig(
            restarts=1,
            rescue_restarts=0,
            dexterity_weights=[1e-12] * 3,
        ),
    )
    assert result.reachable[0]
    assert result.metrics["isotropy"][0] == pytest.approx(1)
    assert result.metrics["minimum_singular_value"][0] == pytest.approx(
        1e-12, rel=1e-5, abs=0
    )


def test_real_array_conversion_retains_independent_contiguous_storage():
    from workspace_analyzer.reachability import _real_array

    source = np.arange(30, dtype=np.float64).reshape(5, 6)[:, ::2]
    converted = _real_array(source, "targets", dtype="float32")
    assert converted.dtype == np.float32
    assert converted.flags.c_contiguous
    assert not np.shares_memory(source, converted)
    np.testing.assert_array_equal(converted, source)
