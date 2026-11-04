"""Acceptance preserves diagnostics and reports pass/fail through exit codes."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from workspace_analyzer import (
    AcceptanceConfig,
    AnalysisResult,
    PlaneConfig,
    ReachabilityConfig,
    analyze_plane,
    apply_acceptance,
    create_solver,
    reassess_acceptance,
)
from workspace_analyzer.cli import main

URDF = Path(__file__).parent / "fixtures/cartesian_stage.urdf"


@pytest.mark.parametrize("minimum,passed", [(0.36, True), (0.37, False)])
def test_plane_acceptance_exit_and_saved_reports(minimum, passed, tmp_path, capsys):
    args = [
        str(URDF),
        "--plane",
        "xy",
        "--span",
        "3",
        "--resolution",
        "5",
        "--backend",
        "numpy",
        "--min-success-rate",
        str(minimum),
        "--output",
        str(tmp_path / "result.npz"),
        "--report",
        str(tmp_path / "report.json"),
        "--plot",
        str(tmp_path / "result.svg"),
    ]
    if passed:
        main(args)
    else:
        with pytest.raises(SystemExit) as error:
            main(args)
        assert error.value.code == 1
    result = AnalysisResult.load(tmp_path / "result.npz")
    acceptance = result.metadata["acceptance"]
    assert acceptance["passed"] == passed
    assert acceptance["observed"]["success_rate"] == 0.36
    assert (
        json.loads((tmp_path / "report.json").read_text())["acceptance"] == acceptance
    )
    assert (tmp_path / "result.svg").is_file()
    assert (
        "acceptance: PASS" if passed else "acceptance: FAIL"
    ) in capsys.readouterr().out


def test_reassessment_updates_saved_acceptance_without_modifying_original():
    solver = create_solver(URDF, backend="numpy")
    result = analyze_plane(
        solver,
        PlaneConfig(resolution=3, span=2),
        ReachabilityConfig(restarts=1, rescue_restarts=0),
    )
    result = apply_acceptance(result, AcceptanceConfig(min_quality_pass_rate=0.9))
    assert result.metadata["acceptance"]["passed"]
    stricter = result.reassess_quality(minimum_joint_limit_margin=0.1)
    assert not stricter.metadata["acceptance"]["passed"]
    assert (
        stricter.metadata["plane_scan"]["acceptance"] == stricter.metadata["acceptance"]
    )
    assert result.metadata["acceptance"]["passed"]
    np.testing.assert_array_equal(result.reachable, stricter.reachable)


def test_inspect_acceptance_reports_failure_and_writes_reassessed_copy(
    tmp_path, capsys
):
    solver = create_solver(URDF, backend="numpy")
    source = analyze_plane(
        solver,
        PlaneConfig(resolution=3, span=2),
        ReachabilityConfig(restarts=1, rescue_restarts=0),
    )
    source.save(tmp_path / "source.npz")
    with pytest.raises(SystemExit) as error:
        main(
            [
                "inspect",
                str(tmp_path / "source.npz"),
                "--min-joint-limit-margin",
                "0.1",
                "--min-quality-pass-rate",
                "0.9",
                "--output",
                str(tmp_path / "assessed.npz"),
            ]
        )
    assert error.value.code == 1
    report = json.loads(capsys.readouterr().out)
    assert not report["metadata"]["acceptance"]["passed"]
    assert not AnalysisResult.load(tmp_path / "assessed.npz").metadata["acceptance"][
        "passed"
    ]
    assert "acceptance" not in AnalysisResult.load(tmp_path / "source.npz").metadata


@pytest.mark.parametrize(
    "flags",
    [
        ["--min-success-rate", "nan"],
        ["--min-success-rate", "1.1"],
        ["--min-success-rate", "0.5", "--viser"],
        ["--mode", "joint", "--min-success-rate", "0.5"],
    ],
)
def test_invalid_acceptance_rejected_before_analysis(flags, capsys):
    with pytest.raises(SystemExit) as error:
        main([str(URDF), "--mode", "plane", *flags])
    assert error.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_empty_results_do_not_pass_zero_acceptance_threshold():
    result = AnalysisResult(
        np.empty((0, 3)), np.empty((0, 3)), None, {}, reachable=np.empty(0, dtype=bool)
    )
    outcome = apply_acceptance(result, AcceptanceConfig(min_success_rate=0))
    assert not outcome.metadata["acceptance"]["passed"]
    assert outcome.metadata["acceptance"]["failures"] == ["no targets were evaluated"]


@pytest.mark.parametrize("value", [True, np.bool_(False), "0.5", [0.5], np.nan, np.inf])
def test_acceptance_requires_a_finite_real_fraction(value):
    with pytest.raises(ValueError, match="finite and between"):
        AcceptanceConfig(min_success_rate=value)


def test_acceptance_normalizes_numpy_scalars_for_json():
    from dataclasses import asdict

    config = AcceptanceConfig(np.float32(0.5), np.int64(1))
    assert json.loads(json.dumps(asdict(config))) == {
        "min_success_rate": 0.5,
        "min_quality_pass_rate": 1.0,
    }


@pytest.fixture
def accepted_plane(tmp_path):
    result = analyze_plane(
        create_solver(URDF, backend="numpy"),
        PlaneConfig(resolution=5, span=3),
        ReachabilityConfig(restarts=1, rescue_restarts=0),
    )
    result = apply_acceptance(
        result, AcceptanceConfig(min_success_rate=0.9, min_quality_pass_rate=0.2)
    )
    path = tmp_path / "accepted.npz"
    result.save(path)
    return path, result


def test_inspect_preserves_unspecified_acceptance_requirements(accepted_plane, capsys):
    path, original = accepted_plane
    with pytest.raises(SystemExit) as error:
        main(["inspect", str(path), "--min-quality-pass-rate", ".1"])
    assert error.value.code == 1
    verdict = json.loads(capsys.readouterr().out)["metadata"]["acceptance"]
    assert verdict["requirements"] == {
        "min_success_rate": 0.9,
        "min_quality_pass_rate": 0.1,
    }
    assert not verdict["passed"]
    assert (
        original.metadata["acceptance"]["requirements"]["min_quality_pass_rate"] == 0.2
    )


@pytest.mark.parametrize("replacement", [False, True])
def test_inspect_explicitly_resets_acceptance_without_clearing_quality_gates(
    accepted_plane, tmp_path, capsys, replacement
):
    path, _ = accepted_plane
    output = tmp_path / "reset.npz"
    args = [
        "inspect",
        str(path),
        "--reset-acceptance",
        "--min-joint-limit-margin",
        ".1",
        "--output",
        str(output),
    ]
    if replacement:
        args += ["--min-quality-pass-rate", ".1"]
    result = main(args)
    assert result.metadata["quality_thresholds"]["minimum_joint_limit_margin"] == 0.1
    for metadata in (result.metadata, result.metadata["plane_scan"]):
        if replacement:
            assert metadata["acceptance"]["requirements"] == {
                "min_success_rate": None,
                "min_quality_pass_rate": 0.1,
            }
            assert metadata["acceptance"]["passed"]
        else:
            assert "acceptance" not in metadata
    assert "acceptance" in AnalysisResult.load(path).metadata
    assert AnalysisResult.load(output).metadata == result.metadata


@pytest.mark.parametrize("minimum", [0.2, 0.9])
def test_inspect_recomputes_stale_verdict_without_kinematics(
    accepted_plane, tmp_path, capsys, monkeypatch, minimum
):
    from workspace_analyzer import KinematicsSolver

    path, result = accepted_plane
    result = apply_acceptance(result, AcceptanceConfig(min_success_rate=minimum))
    passed = result.metadata["acceptance"]["passed"]
    result.metadata["acceptance"]["passed"] = not passed
    result.metadata["acceptance"]["observed"]["success_rate"] = 1.0
    result.save(path)
    before = path.read_bytes()
    for method in ("forward", "inverse", "jacobian", "forward_with_jacobian"):
        monkeypatch.setattr(
            KinematicsSolver,
            method,
            lambda *a, **k: pytest.fail("offline verdict ran kinematics"),
        )
    args = ["inspect", str(path), "--output", str(tmp_path / "fresh.npz")]
    if passed:
        main(args)
    else:
        with pytest.raises(SystemExit) as error:
            main(args)
        assert error.value.code == 1
    verdict = json.loads(capsys.readouterr().out)["metadata"]["acceptance"]
    assert verdict["passed"] == passed
    assert verdict["observed"]["success_rate"] == 0.36
    assert path.read_bytes() == before
    fresh = AnalysisResult.load(tmp_path / "fresh.npz")
    assert fresh.metadata["plane_scan"]["acceptance"] == verdict


@pytest.mark.parametrize(
    "record",
    [
        None,
        [],
        {},
        {"requirements": {}},
        {"requirements": {"wrong_key": 0.2}},
        {"requirements": {"min_success_rate": "bad"}},
    ],
)
def test_invalid_saved_acceptance_has_a_clear_error_and_explicit_reset(
    accepted_plane, record, capsys
):
    path, result = accepted_plane
    result.metadata["acceptance"] = record
    result.save(path)
    with pytest.raises(SystemExit) as error:
        main(["inspect", str(path)])
    assert error.value.code == 2
    assert "saved acceptance" in capsys.readouterr().err
    repaired = main(
        ["inspect", str(path), "--reset-acceptance", "--min-isotropy", ".1"]
    )
    assert "acceptance" not in repaired.metadata
    assert "acceptance" not in repaired.metadata["plane_scan"]


def test_acceptance_api_preserves_source_and_allows_zero_override(accepted_plane):
    from copy import deepcopy

    _, original = accepted_plane
    before = deepcopy(original.metadata)
    updated = reassess_acceptance(original, AcceptanceConfig(min_success_rate=0))
    assert updated.metadata["acceptance"]["requirements"] == {
        "min_success_rate": 0.0,
        "min_quality_pass_rate": 0.2,
    }
    assert updated.metadata["acceptance"]["passed"]
    cleared = reassess_acceptance(original, reset=True)
    assert "acceptance" not in cleared.metadata
    assert "acceptance" not in cleared.metadata["plane_scan"]
    assert original.metadata == before
    assert cleared.points is original.points
    for name, values in original.metrics.items():
        assert updated.metrics[name] is values


def test_reset_quality_gates_preserves_run_acceptance(accepted_plane, capsys):
    path, original = accepted_plane
    with pytest.raises(SystemExit) as error:
        main(["inspect", str(path), "--reset-quality-gates"])
    assert error.value.code == 1
    metadata = json.loads(capsys.readouterr().out)["metadata"]
    assert (
        metadata["acceptance"]["requirements"]
        == original.metadata["acceptance"]["requirements"]
    )


@pytest.mark.parametrize("mutation", ["measurement", "flags", "thresholds"])
def test_quality_acceptance_rejects_stale_target_flags_until_reassessed(
    mutation, tmp_path
):
    result = analyze_plane(
        create_solver(URDF, backend="numpy"),
        PlaneConfig(resolution=3, span=1),
        ReachabilityConfig(restarts=1, rescue_restarts=0, minimum_isotropy=0.5),
    )
    config = AcceptanceConfig(min_quality_pass_rate=1)
    result = apply_acceptance(result, config)
    if mutation == "measurement":
        result.metrics["isotropy"][0] = 0
    elif mutation == "flags":
        result.metrics["quality_pass"][0] = False
    else:
        result.metadata["quality_thresholds"]["minimum_singular_value"] = 2
    with pytest.raises(ValueError, match="quality flags disagree"):
        apply_acceptance(result, config)
    source = tmp_path / "stale.npz"
    result.save(source)
    original = source.read_bytes()
    run = subprocess.run(
        [sys.executable, "-m", "workspace_analyzer", "inspect", str(source)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert run.returncode == 2
    assert "quality flags disagree" in run.stderr
    assert "Traceback" not in run.stderr
    assert source.read_bytes() == original
    updated = result.reassess_quality(**result.metadata["quality_thresholds"])
    assert updated.metadata["acceptance"]["passed"] == (mutation == "flags")


def test_generic_quality_acceptance_keeps_boolean_flag_interface(monkeypatch):
    import workspace_analyzer.reachability as reachability

    monkeypatch.setattr(
        reachability,
        "_validated_quality_flags",
        lambda *args: pytest.fail("generic flags must not require target measurements"),
    )
    result = AnalysisResult(
        np.zeros((3, 3)),
        np.zeros((3, 2)),
        None,
        {"mode": "custom"},
        reachable=np.array([True, True, False]),
        metrics={"quality_pass": np.array([True, False, False])},
    )
    updated = apply_acceptance(result, AcceptanceConfig(min_quality_pass_rate=1 / 3))
    assert updated.metadata["acceptance"]["passed"]
    assert updated.points is result.points
    assert updated.metrics["quality_pass"] is result.metrics["quality_pass"]
    assert result.metadata == {"mode": "custom"}


@pytest.mark.parametrize("plane_scan", [None, "damaged", 3, []])
@pytest.mark.parametrize("reset", [False, True])
def test_invalid_plane_metadata_has_clean_acceptance_error(
    plane_scan, reset, accepted_plane, tmp_path
):
    source, result = accepted_plane
    result.metadata["plane_scan"] = plane_scan
    result.save(source)
    original = source.read_bytes()
    output = tmp_path / "previous.npz"
    output.write_bytes(b"previous completed result")
    options = ["--reset-acceptance"] if reset else ["--min-success-rate", "0.5"]
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "workspace_analyzer",
            "inspect",
            str(source),
            "--output",
            str(output),
            *options,
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert run.returncode == 2
    assert "plane_scan metadata must be a dictionary" in run.stderr
    assert "Traceback" not in run.stderr
    assert source.read_bytes() == original
    assert output.read_bytes() == b"previous completed result"
