import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from workspace_analyzer import AnalysisResult, KinematicsSolver

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "reachability_workflow", ROOT / "examples/reachability_workflow.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _args(output, backend="numpy"):
    return [
        "--urdf",
        str(ROOT / "tests/fixtures/cartesian_stage.urdf"),
        "--output-dir",
        str(output),
        "--samples",
        "3",
        "--batch-size",
        "4",
        "--backend",
        backend,
        "--max-iterations",
        "30",
        "--restarts",
        "1",
        "--rescue-restarts",
        "0",
    ]


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("robustness", [False, True])
@pytest.mark.parametrize("boundary", [False, True])
def test_workflow_produces_verified_artifacts_and_reuses_cache(
    backend, robustness, boundary, tmp_path, monkeypatch
):
    if backend == "torch":
        pytest.importorskip("torch")
    args = _args(tmp_path, backend) + (["--robustness-test"] if robustness else [])
    if boundary:
        args += ["--boundary-test"]
    assert MODULE.main(args) == 0
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["passed"]
    assert len(report["checks"]) == 5 + int(robustness) + 2 * int(boundary)
    if boundary:
        summary = report["translation_boundary"]["summary"]
        assert summary["initial_brackets"] == summary["refined"] == 3
        assert (tmp_path / "boundary/measurements.npz").is_file()
        assert (tmp_path / "boundary/trace.npz").is_file()
    else:
        assert report["translation_boundary"] is None
    if robustness:
        summary = report["perturbation_robustness"]["summary"]
        assert summary["references"] == 3
        assert summary["variants_per_reference"] == 13
        assert summary["ik"]["perturbed_success_rate"] == 0.5
        assert summary["ik"]["all_variants_pass_rate"] == 0
        assert (tmp_path / "perturbation_targets.npz").is_file()
        assert (tmp_path / "perturbations.npz").is_file()
    else:
        assert report["perturbation_robustness"] is None
    for name in ("position", "pose"):
        a = report["cases"][name]["assessment"]
        assert a["samples"] == 9 and a["ik_success_count"] == 3
        assert report["checks"][f"{name}_far_targets"]["rejected"] == 6
        assert (tmp_path / f"{name}.npz").is_file()
    np.testing.assert_allclose(
        report["orientation_coverage"]["ik_coverage"], [1 / 3] * 3
    )
    assert "PASS" in (tmp_path / "report.md").read_text()
    assert (tmp_path / "targets.npz").is_file()
    result = AnalysisResult.load(tmp_path / "local_orientations.npz")
    counts = [
        step["assessment"]["quality_pass_count"] for step in report["quality_sweep"]
    ]
    assert counts == sorted(counts, reverse=True)
    original = MODULE.create_solver

    def without_solves(*args, **kwargs):
        solver = original(*args, **kwargs)

        def unexpected(*a, **k):
            pytest.fail("repeated workflow must reuse all IK measurements")

        monkeypatch.setattr(solver, "inverse", unexpected)
        return solver

    monkeypatch.setattr(MODULE, "create_solver", without_solves)
    assert MODULE.main(args) == 0
    repeated = json.loads((tmp_path / "report.json").read_text())
    assert all(case["cache_hit"] for case in repeated["cases"].values())
    np.testing.assert_array_equal(
        AnalysisResult.load(tmp_path / "local_orientations.npz").reachable,
        result.reachable,
    )
    assert repeated["perturbation_robustness"] == report["perturbation_robustness"]
    if boundary:
        assert (
            repeated["translation_boundary"]["segments"]
            == report["translation_boundary"]["segments"]
        )
        assert all(
            batch["cache_hit"]
            for batch in repeated["translation_boundary"]["metadata"][
                "measurement_batches"
            ]
        )


@pytest.mark.parametrize(
    "extra, status",
    [
        (["--boundary-max-rounds", "0"], "budget_exhausted"),
        (["--boundary-distance-m", "0.01"], "end_succeeded"),
    ],
)
def test_boundary_acceptance_failures_keep_unresolved_brackets(extra, status, tmp_path):
    assert MODULE.main([*_args(tmp_path), "--boundary-test", *extra]) == 1
    report = json.loads((tmp_path / "report.json").read_text())
    assert not report["checks"]["boundary_brackets_refined"]["passed"]
    assert all(
        row["status"] == status for row in report["translation_boundary"]["segments"]
    )
    assert (tmp_path / "boundary/measurements.npz").is_file()
    assert status in (tmp_path / "report.md").read_text()


@pytest.mark.parametrize("source", ["pose", "perturbations", "boundary"])
def test_stability_uses_selected_target_rows_and_cached_trials(
    source, tmp_path, monkeypatch
):
    args = [
        *_args(tmp_path),
        "--stability-test",
        "--stability-targets",
        source,
        "--stability-iterations",
        "30",
        "60",
        "--stability-seeds",
        "77",
    ]
    if source == "boundary":
        args += ["--boundary-test"]
    elif source == "perturbations":
        args += ["--robustness-test"]
    assert MODULE.main(args) == 0
    report = json.loads((tmp_path / "report.json").read_text())
    study = report["ik_stability"]
    assert study["summary"]["trials"] == 2
    assert (
        study["summary"]["targets"]
        == {"pose": 9, "perturbations": 39, "boundary": 6}[source]
    )
    origin = study["target_source"]
    measurement = AnalysisResult.load(tmp_path / origin["result_file"])
    first = AnalysisResult.load(tmp_path / "stability/trial_000.npz")
    np.testing.assert_array_equal(
        first.target_poses, measurement.target_poses[origin["row_indices"]]
    )
    assert (tmp_path / "stability/initial_seeds.npz").is_file()
    assert "IK stability" in (tmp_path / "report.md").read_text()

    def unexpected(*args, **kwargs):
        pytest.fail("repeated workflow must cache cloned-solver trials too")

    monkeypatch.setattr(KinematicsSolver, "inverse", unexpected)
    assert MODULE.main(args) == 0
    repeated = json.loads((tmp_path / "report.json").read_text())["ik_stability"]
    assert repeated["per_target"] == study["per_target"]
    assert all(trial["metadata"]["cache_hit"] for trial in repeated["trials"])


def test_stability_disagreement_threshold_fails_and_preserves_counterexamples(tmp_path):
    args = [
        *_args(tmp_path),
        "--stability-test",
        "--stability-iterations",
        "1",
        "30",
        "--stability-seeds",
        "77",
        "--max-ik-disagreement",
        "0",
    ]
    assert MODULE.main(args) == 1
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["checks"]["pose_known_targets"]["passed"]
    check = report["checks"]["ik_stability_disagreement"]
    assert not check["passed"]
    assert check["observed_disagreement_rate"] == pytest.approx(3 / 9)
    assert report["ik_stability"]["per_target"]["ik_disagreement_indices"] == [0, 1, 2]
    assert (tmp_path / "stability/trial_001.npz").is_file()


@pytest.mark.parametrize("source", ["pose", "perturbations", "boundary"])
def test_candidate_selection_workflow_preserves_rows_and_cache(
    source, tmp_path, monkeypatch
):
    from workspace_analyzer import IKStabilityResult

    args = [
        *_args(tmp_path),
        "--stability-test",
        "--stability-targets",
        source,
        "--select-ik-solutions",
        "joint_limit_margin",
    ]
    if source == "boundary":
        args += ["--boundary-test"]
    elif source == "perturbations":
        args += ["--robustness-test"]
    assert MODULE.main(args) == 0
    report = json.loads((tmp_path / "report.json").read_text())
    selected = AnalysisResult.load(tmp_path / "stability/selected.npz")
    study = IKStabilityResult.load(tmp_path / "stability")
    ik, quality = study._flags()
    np.testing.assert_array_equal(selected.reachable, ik.any(axis=0))
    np.testing.assert_array_equal(selected.metrics["quality_pass"], quality.any(axis=0))
    np.testing.assert_array_equal(selected.target_poses, study.results[0].target_poses)
    selection = report["ik_stability"]["selection"]
    assert selection["file"] == "selected.npz"
    assert selection["metadata"] == selected.metadata
    assert selected.metadata["candidate_selection"]["source_study_directory"] == "."
    assert "Observed candidate selection" in (tmp_path / "report.md").read_text()

    def unexpected(*args, **kwargs):
        pytest.fail("selection must reuse measured candidates")

    monkeypatch.setattr(KinematicsSolver, "inverse", unexpected)
    assert MODULE.main(args) == 0
    replay = AnalysisResult.load(tmp_path / "stability/selected.npz")
    np.testing.assert_array_equal(replay.joint_positions, selected.joint_positions)
    assert all(
        row["metadata"]["cache_hit"]
        for row in replay.metadata["candidate_selection"]["trials"]
    )


@pytest.mark.parametrize(
    "minimum, expected_exit", [("0.3333333333333333", 0), ("0.34", 1)]
)
def test_selected_quality_acceptance_counts_all_targets_and_keeps_diagnostics(
    minimum, expected_exit, tmp_path
):
    args = [
        *_args(tmp_path),
        "--stability-test",
        "--stability-iterations",
        "1",
        "30",
        "--stability-seeds",
        "77",
        "--select-ik-solutions",
        "isotropy",
        "--min-selected-quality-rate",
        minimum,
    ]
    assert MODULE.main(args) == expected_exit
    report = json.loads((tmp_path / "report.json").read_text())
    check = report["checks"]["selected_candidate_quality"]
    assert check["observed_quality_rate"] == pytest.approx(3 / 9)
    assert check["passed"] == (expected_exit == 0)
    selected = AnalysisResult.load(tmp_path / "stability/selected.npz")
    audit = selected.metadata["candidate_selection"]["selection_time_summary"]
    assert audit["ik_gained_indices"] == [0, 1, 2]
    assert audit["quality_gained_indices"] == [0, 1, 2]
    assert audit["ik_lost_count"] == audit["quality_lost_count"] == 0


def test_selection_does_not_override_failed_stability_acceptance(tmp_path):
    assert (
        MODULE.main(
            [
                *_args(tmp_path),
                "--stability-test",
                "--stability-iterations",
                "1",
                "30",
                "--stability-seeds",
                "77",
                "--max-ik-disagreement",
                "0",
                "--select-ik-solutions",
                "joint_limit_margin",
            ]
        )
        == 1
    )
    report = json.loads((tmp_path / "report.json").read_text())
    assert not report["checks"]["ik_stability_disagreement"]["passed"]
    assert (
        report["ik_stability"]["selection"]["metadata"]["assessment"][
            "ik_success_count"
        ]
        == 3
    )


@pytest.mark.parametrize("source", ["pose", "perturbations", "boundary"])
def test_workflow_generated_guesses_diversity_and_selection_are_replayable(
    source, tmp_path, monkeypatch
):
    from workspace_analyzer import IKStabilityResult

    args = [
        *_args(tmp_path),
        "--stability-test",
        "--stability-targets",
        source,
        "--stability-iterations",
        "30",
        "--stability-seeds",
        "77",
        "--stability-initial-guesses",
        "2",
        "--candidate-diversity",
        "--select-ik-solutions",
        "joint_limit_margin",
    ]
    if source == "boundary":
        args += ["--boundary-test"]
    elif source == "perturbations":
        args += ["--robustness-test"]
    assert MODULE.main(args) == 0
    report = json.loads((tmp_path / "report.json").read_text())
    study = IKStabilityResult.load(tmp_path / "stability")
    assert len(study.names) == 3
    assert study.initial_seeds[0] is None
    assert all(
        seed is not None and seed.shape == (3,) for seed in study.initial_seeds[1:]
    )
    diversity = json.loads((tmp_path / "stability/diversity.json").read_text())
    assert (
        diversity == report["ik_stability"]["diversity"] == study.summarize_diversity()
    )
    # The three-axis stage has exactly one joint configuration per solved target.
    assert diversity["summary"]["targets_with_multiple_configurations"] == 0
    assert diversity["summary"]["duplicate_candidate_count"] > 0
    assert "Observed configuration diversity" in (tmp_path / "report.md").read_text()

    def unexpected(*args, **kwargs):
        pytest.fail("repeated generated trials must reuse cached measurements")

    monkeypatch.setattr(KinematicsSolver, "inverse", unexpected)
    assert MODULE.main(args) == 0
    repeated = json.loads((tmp_path / "stability/diversity.json").read_text())
    assert repeated == diversity


def test_initial_guesses_are_paired_across_budget_and_restart_seed_conditions(tmp_path):
    from workspace_analyzer import IKStabilityResult

    assert (
        MODULE.main(
            [
                *_args(tmp_path),
                "--stability-test",
                "--stability-initial-guesses",
                "2",
            ]
        )
        == 0
    )
    study = IKStabilityResult.load(tmp_path / "stability")
    assert len(study.names) == 12  # 2 budgets × 2 restart seeds × (default + 2 guesses)
    assert study.initial_seeds[:4] == (None,) * 4
    for start in (4, 6, 8, 10):
        np.testing.assert_array_equal(
            study.initial_seeds[start], study.initial_seeds[4]
        )
        np.testing.assert_array_equal(
            study.initial_seeds[start + 1], study.initial_seeds[5]
        )
    assert len({r.metadata["cache_key"] for r in study.results}) == 12


@pytest.mark.parametrize("rotation, expected_exit", [("0", 0), ("5", 1)])
def test_workflow_enforces_robust_rate_and_retains_pair_diagnostics(
    rotation, expected_exit, tmp_path
):
    args = [
        *_args(tmp_path),
        "--robustness-test",
        "--min-robust-rate",
        "1",
        "--perturb-rotation-deg",
        rotation,
    ]
    assert MODULE.main(args) == expected_exit
    report = json.loads((tmp_path / "report.json").read_text())
    check = report["checks"]["perturbation_robust_rate"]
    assert check["passed"] == (expected_exit == 0)
    assert report["checks"]["perturbation_reference_targets"]["passed"]
    assert (tmp_path / "perturbations.npz").is_file()
    markdown = (tmp_path / "report.md").read_text()
    assert "IK lost / gained" in markdown
    assert "100.00%" in markdown


def test_workflow_failure_returns_nonzero_and_preserves_diagnostics(tmp_path):
    args = _args(tmp_path)
    args[args.index("--max-iterations") + 1] = "1"
    assert MODULE.main(args) == 1
    report = json.loads((tmp_path / "report.json").read_text())
    assert not report["passed"]
    assert not report["checks"]["position_known_targets"]["passed"]
    assert report["checks"]["position_far_targets"]["passed"]
    assert (tmp_path / "pose.npz").is_file()
    assert (tmp_path / "local_orientations.npz").is_file()


def test_workflow_can_enforce_quality_acceptance(tmp_path):
    assert (
        MODULE.main(
            [
                *_args(tmp_path),
                "--min-quality-rate",
                "1",
                "--min-joint-limit-margin",
                ".99",
            ]
        )
        == 1
    )
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["checks"]["position_known_targets"]["passed"]
    assert not report["checks"]["position_known_quality"]["passed"]


@pytest.mark.parametrize(
    "args",
    [
        ["--samples", "0"],
        ["--joint-fraction", "nan"],
        ["--orientation-offset-deg", "180"],
        ["--min-known-success", "1.1"],
        ["--min-quality-rate", "nan"],
        ["--min-robust-rate", "0.9"],
        ["--robustness-test", "--min-robust-rate", "nan"],
        ["--robustness-test", "--perturb-translation-m", "-1"],
        ["--robustness-test", "--perturb-rotation-deg", "180"],
        ["--boundary-test", "--boundary-distance-m", "0"],
        ["--boundary-test", "--boundary-tolerance-m", "nan"],
        ["--boundary-test", "--boundary-max-rounds", "-1"],
        ["--max-ik-disagreement", "0.1"],
        ["--stability-initial-guesses", "-1"],
        ["--stability-initial-guesses", "2"],
        ["--candidate-diversity"],
        [
            "--stability-test",
            "--candidate-diversity",
            "--candidate-angular-tolerance-rad",
            "nan",
        ],
        [
            "--stability-test",
            "--candidate-diversity",
            "--candidate-linear-tolerance-m",
            "0",
        ],
        ["--select-ik-solutions", "joint_limit_margin"],
        ["--min-selected-quality-rate", "0.5"],
        [
            "--stability-test",
            "--select-ik-solutions",
            "isotropy",
            "--min-selected-quality-rate",
            "nan",
        ],
        [
            "--stability-test",
            "--select-ik-solutions",
            "isotropy",
            "--min-selected-quality-rate",
            "1.1",
        ],
        ["--stability-test", "--max-ik-disagreement", "nan"],
        ["--stability-test", "--stability-targets", "boundary"],
        ["--stability-test", "--stability-targets", "perturbations"],
        ["--stability-test", "--stability-seeds", "-1"],
        ["--stability-test", "--stability-iterations", "0"],
        [
            "--stability-test",
            "--stability-iterations",
            "30",
            "30",
            "--stability-seeds",
            "77",
        ],
        [
            "--robustness-test",
            "--perturb-translation-m",
            "0",
            "--perturb-rotation-deg",
            "0",
            "--min-robust-rate",
            "1",
        ],
    ],
)
def test_workflow_validates_settings_before_writing(args, tmp_path):
    with pytest.raises(SystemExit) as error:
        MODULE.main([*_args(tmp_path), *args])
    assert error.value.code == 2
    assert not (tmp_path / "report.json").exists()
