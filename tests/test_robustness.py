import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from workspace_analyzer import (
    AnalysisResult,
    PosePerturbations,
    ReachabilityConfig,
    ResultCache,
    WorkspaceAnalyzer,
    create_solver,
)

URDF = Path(__file__).parent / "fixtures/cartesian_stage.urdf"


def _poses(xs, dtype="float64"):
    poses = np.broadcast_to(np.eye(4, dtype=dtype), (len(xs), 4, 4)).copy()
    poses[:, 0, 3] = xs
    return poses


def _setup(backend="numpy", dtype="float64"):
    if backend == "torch":
        pytest.importorskip("torch")
    solver = create_solver(URDF, backend=backend, dtype=dtype, max_iterations=40)
    config = ReachabilityConfig(
        position_only=False,
        restarts=1,
        rescue_restarts=0,
        batch_size=4,
        minimum_joint_limit_margin=0.1,
    )
    return solver, WorkspaceAnalyzer(solver), config


def test_offsets_use_declared_frames_and_rotate_about_tcp():
    pose = np.eye(4)
    pose[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    pose[:3, 3] = [2, 3, 4]
    base = PosePerturbations(
        pose, translation_m=0.2, rotation_rad=np.pi / 2, rotation_frame="base"
    )
    tool = PosePerturbations(
        pose,
        translation_m=0.2,
        rotation_rad=np.pi / 2,
        translation_frame="tool",
        rotation_frame="tool",
    )
    np.testing.assert_allclose(base.targets[1, :3, 3], [2.2, 3, 4])
    np.testing.assert_allclose(tool.targets[1, :3, 3], [2, 3.2, 4])
    rx = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    np.testing.assert_allclose(base.targets[7, :3, :3], rx @ pose[:3, :3], atol=1e-15)
    np.testing.assert_allclose(tool.targets[7, :3, :3], pose[:3, :3] @ rx, atol=1e-15)
    np.testing.assert_array_equal(base.targets[7:, :3, 3], np.tile([2, 3, 4], (6, 1)))
    assert base.variant_names[1:3] == ("translation_x+", "translation_x-")
    rotations = tool.targets[:, :3, :3]
    np.testing.assert_allclose(
        rotations @ rotations.swapaxes(1, 2),
        np.broadcast_to(np.eye(3), rotations.shape),
        atol=1e-15,
    )
    np.testing.assert_allclose(np.linalg.det(rotations), 1)
    pose[:] = 0
    assert tool.reference_poses[0, 0, 3] == 2
    with pytest.raises(ValueError):
        tool.targets[0, 0, 3] = 100


def test_tool_frame_study_commutes_with_rigid_base_transform():
    poses = _poses([0.2, 0.5])
    poses[1, :3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    transform = np.eye(4)
    angle = 0.37
    transform[:3, :3] = [
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1],
    ]
    transform[:3, 3] = [0.1, -0.4, 0.9]
    original = PosePerturbations(poses, translation_frame="tool", rotation_frame="tool")
    transformed = PosePerturbations(
        transform @ poses, translation_frame="tool", rotation_frame="tool"
    )
    np.testing.assert_allclose(
        transformed.targets, transform @ original.targets, atol=1e-15
    )


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_analytic_boundary_oracle_and_paired_quality_counts(backend, dtype, tmp_path):
    _, analyzer, config = _setup(backend, dtype)
    study = PosePerturbations(
        _poses([0.95, 1.05, 0], dtype), translation_m=0.1, rotation_rad=0
    )
    result = analyzer.analyze_targets(study.targets, config)
    result.save(tmp_path / "measurements")
    result = AnalysisResult.load(tmp_path / "measurements.npz")
    expected = np.all(np.abs(study.targets[:, :3, 3]) <= 1, axis=1)
    np.testing.assert_array_equal(result.reachable, expected)
    report = study.summarize(result)
    assert report["summary"]["variants_per_reference"] == 7
    assert report["summary"]["perturbed_samples"] == 18
    ik = report["summary"]["ik"]
    assert ik["reference_success_count"] == 2
    assert ik["perturbed_success_count"] == 12
    assert ik["lost_pairs"] == ik["gained_pairs"] == 1
    assert ik["retained_success_rate"] == pytest.approx(11 / 12)
    assert ik["all_variants_pass_rate"] == pytest.approx(1 / 3)
    quality = report["summary"]["quality"]
    assert quality["reference_success_count"] == 1
    assert quality["perturbed_success_count"] == 7
    assert quality["lost_pairs"] == 0 and quality["gained_pairs"] == 1
    assert quality["retained_success_rate"] == 1
    np.testing.assert_allclose(
        report["per_reference"]["worst_solved_joint_limit_margin"],
        [0.05, 0.05, 0.9],
        atol=1e-5,
    )
    assert report["by_variant"][1]["ik_lost_from_reference"] == 1
    assert report["by_variant"][2]["ik_gained_from_reference"] == 1
    assert study.targets.dtype == np.dtype(dtype)
    json.dumps(report, allow_nan=False)


def test_stage_rejects_rotation_perturbations():
    _, analyzer, config = _setup()
    study = PosePerturbations(_poses([0]))
    result = analyzer.analyze_targets(study.targets, config)
    np.testing.assert_array_equal(result.reachable, [True] * 7 + [False] * 6)
    report = study.summarize(result)
    assert report["summary"]["ik"]["perturbed_success_rate"] == 0.5
    assert report["summary"]["ik"]["all_variants_pass_rate"] == 0


def test_zero_offsets_reduce_to_reference_and_do_not_claim_robustness():
    _, analyzer, config = _setup()
    poses = _poses([0, 0.2])
    study = PosePerturbations(poses, translation_m=0, rotation_rad=0)
    np.testing.assert_array_equal(study.targets, poses)
    assert study.variant_names == ("reference",)
    result = analyzer.analyze_targets(study.targets, config)
    baseline = analyzer.analyze_targets(poses, config)
    np.testing.assert_array_equal(result.joint_positions, baseline.joint_positions)
    report = study.summarize(result)
    assert report["summary"]["perturbed_samples"] == 0
    for metric in ("ik", "quality"):
        assert report["summary"][metric]["all_variants_pass_rate"] is None
        assert report["summary"][metric]["retained_success_rate"] is None
        assert report["per_reference"][f"perturbed_{metric}_rate"] == [None, None]


def test_no_success_has_null_conditional_rates_and_metrics():
    _, analyzer, config = _setup()
    study = PosePerturbations(_poses([3]), rotation_rad=0)
    report = study.summarize(analyzer.analyze_targets(study.targets, config))
    assert report["summary"]["ik"]["reference_success_count"] == 0
    assert report["summary"]["ik"]["retained_success_rate"] is None
    assert report["summary"]["ik"]["all_variants_pass_rate"] == 0
    assert report["per_reference"]["worst_solved_isotropy"] == [None]
    json.dumps(report, allow_nan=False)


def test_repeated_study_uses_cache_and_quality_reassessment_is_offline(
    tmp_path, monkeypatch
):
    solver, analyzer, config = _setup()
    study = PosePerturbations(_poses([0.85]), translation_m=0.1, rotation_rad=0)
    cache = ResultCache(tmp_path)
    original = analyzer.analyze_targets(study.targets, config, cache=cache)

    def unexpected(*args, **kwargs):
        pytest.fail("cached or offline perturbation reporting must not run IK/FK")

    for name in ("inverse", "forward", "forward_with_jacobian"):
        monkeypatch.setattr(solver, name, unexpected)
    repeated = analyzer.analyze_targets(study.targets, config, cache=cache)
    assert study.summarize(repeated) == study.summarize(original)
    relaxed = study.summarize(original.reassess_quality())
    assert relaxed["summary"]["quality"] == relaxed["summary"]["ik"]
    assert study.summarize(original)["summary"]["quality"]["lost_pairs"] == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"translation_m": -1},
        {"translation_m": True},
        {"translation_m": "0.1"},
        {"rotation_rad": np.nan},
        {"rotation_rad": np.pi},
        {"rotation_rad": np.inf},
        {"translation_frame": "world"},
        {"rotation_frame": "tcp"},
    ],
)
def test_invalid_perturbation_settings(kwargs):
    with pytest.raises(ValueError):
        PosePerturbations(np.eye(4), **kwargs)


@pytest.mark.parametrize("kind", ["empty", "point", "nan", "rotation", "homogeneous"])
def test_invalid_reference_poses(kind):
    poses = _poses([0])
    if kind == "empty":
        poses = poses[:0]
    elif kind == "point":
        poses = [0, 0, 0]
    elif kind == "nan":
        poses[0, 0, 3] = np.nan
    elif kind == "rotation":
        poses[0, 0, 0] = 2
    else:
        poses[0, 3, 3] = 0
    with pytest.raises(ValueError):
        PosePerturbations(poses)


@pytest.mark.parametrize(
    "kind", ["position_only", "order", "points", "quality", "shape"]
)
def test_summary_rejects_misaligned_or_invalid_measurements(kind):
    _, analyzer, config = _setup()
    study = PosePerturbations(_poses([0]))
    result = analyzer.analyze_targets(study.targets, config)
    if kind == "position_only":
        result = analyzer.analyze_targets(
            study.targets, replace(config, position_only=True)
        )
    elif kind == "order":
        result.target_poses[[1, 2]] = result.target_poses[[2, 1]]
    elif kind == "points":
        result.points[0, 0] += 1
    elif kind == "quality":
        result.metrics["quality_pass"][-1] = True
    else:
        result = analyzer.analyze_targets(study.targets[:1], config)
    with pytest.raises(ValueError):
        study.summarize(result)


@pytest.mark.parametrize(
    "case", ["all_true", "all_false", "mixed", "reference_only", "no_baseline"]
)
def test_paired_summary_matches_independent_pair_enumeration(case):
    from workspace_analyzer.robustness import _paired_summary

    rows = np.random.default_rng(731).random((7, 5)) > 0.4
    if case == "all_true":
        rows[:] = True
    elif case == "all_false":
        rows[:] = False
    elif case == "reference_only":
        rows = rows[:, :1]
    elif case == "no_baseline":
        rows[:, 0] = False
    baseline = [bool(row[0]) for row in rows]
    pairs = [(bool(row[0]), bool(value)) for row in rows for value in row[1:]]
    retained = [value for before, value in pairs if before]
    all_count = sum(all(row) for row in rows)
    expected = {
        "reference_success_count": sum(baseline),
        "perturbed_success_count": sum(value for _, value in pairs),
        "perturbed_success_rate": sum(value for _, value in pairs) / len(pairs)
        if pairs
        else None,
        "all_variants_pass_count": all_count,
        "all_variants_pass_rate": all_count / len(rows) if pairs else None,
        "retained_success_rate": sum(retained) / len(retained) if retained else None,
        "lost_pairs": sum(before and not after for before, after in pairs),
        "gained_pairs": sum(not before and after for before, after in pairs),
    }
    assert _paired_summary(rows) == expected


@pytest.mark.parametrize("dtype", ["float32", "float64", "int64"])
def test_masked_worst_quality_matches_scalar_minimum_without_mutating_data(dtype):
    _, analyzer, config = _setup()
    study = PosePerturbations(
        _poses([0.95, 1.05, 0]), translation_m=0.1, rotation_rad=0
    )
    result = analyzer.analyze_targets(study.targets, config).reassess_quality()
    variants = len(study.variant_names)
    result.reachable[-variants:] = False
    result.metrics["quality_pass"][-variants:] = False
    metrics = ("minimum_singular_value", "isotropy", "joint_limit_margin")
    original = {}
    for offset, name in enumerate(metrics):
        values = (np.arange(len(result.points)) % 5 + offset).astype(dtype)
        if dtype != "int64":
            values[:3] = [np.nan, np.inf, -np.inf]
        result.metrics[name] = values
        original[name] = values.copy()
    report = study.summarize(result)
    for name in metrics:
        expected = []
        for row in range(3):
            values = [
                float(result.metrics[name][i])
                for i in range(row * variants, (row + 1) * variants)
                if result.reachable[i] and np.isfinite(result.metrics[name][i])
            ]
            expected.append(min(values) if values else None)
        assert report["per_reference"][f"worst_solved_{name}"] == expected
        assert expected[-1] is None
        np.testing.assert_array_equal(result.metrics[name], original[name])
    json.dumps(report, allow_nan=False)
