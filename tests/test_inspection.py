"""Offline inspection uses saved measurements and preserves existing quality gates."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from workspace_analyzer import (
    AnalysisResult,
    KinematicsSolver,
    PlaneConfig,
    ReachabilityConfig,
    analyze_plane,
    create_solver,
)
from workspace_analyzer.cli import main

URDF = Path(__file__).parent / "fixtures/cartesian_stage.urdf"


@pytest.fixture
def measured(tmp_path):
    solver = create_solver(URDF, backend="numpy")
    result = analyze_plane(
        solver,
        PlaneConfig(resolution=5, span=2),
        ReachabilityConfig(
            restarts=1, rescue_restarts=0, minimum_joint_limit_margin=0.3
        ),
    )
    path = tmp_path / "measured.npz"
    result.save(path)
    return path, result


def test_inspect_reassesses_without_kinematics_and_preserves_other_gates(
    measured, tmp_path, monkeypatch, capsys
):
    path, original = measured
    before = path.read_bytes()
    for method in ("forward", "jacobian", "forward_with_jacobian", "inverse"):
        monkeypatch.setattr(
            KinematicsSolver,
            method,
            lambda *a, **k: pytest.fail("inspection ran kinematics"),
        )
    result = main(
        [
            "inspect",
            str(path),
            "--min-isotropy",
            "0.5",
            "--output",
            str(tmp_path / "new.npz"),
            "--report",
            str(tmp_path / "summary.json"),
            "--plot",
            str(tmp_path / "new.svg"),
        ]
    )
    assert result.metadata["quality_thresholds"]["minimum_joint_limit_margin"] == 0.3
    assert result.metadata["quality_thresholds"]["minimum_isotropy"] == 0.5
    np.testing.assert_array_equal(result.reachable, original.reachable)
    np.testing.assert_array_equal(result.joint_positions, original.joint_positions)
    np.testing.assert_array_equal(
        result.metrics["quality_pass"], original.metrics["quality_pass"]
    )
    assert path.read_bytes() == before
    summary = json.loads(capsys.readouterr().out)
    assert summary == json.loads((tmp_path / "summary.json").read_text())
    assert summary["samples"] == 25
    assert AnalysisResult.load(tmp_path / "new.npz").metadata == result.metadata
    assert (tmp_path / "new.svg").read_text().count("<circle ") == 25


def test_inspect_reset_gates_is_explicit(measured, capsys):
    path, original = measured
    unchanged = main(["inspect", str(path)])
    np.testing.assert_array_equal(
        unchanged.metrics["quality_pass"], original.metrics["quality_pass"]
    )
    cleared = main(["inspect", str(path), "--reset-quality-gates"])
    assert cleared.metrics["quality_pass"].all()
    assert not original.metrics["quality_pass"].all()


@pytest.mark.parametrize(
    "options,message",
    [
        (["--min-isotropy", "-0.1"], "minimum_isotropy"),
        (["--plot", "result.png"], ".svg"),
    ],
)
def test_inspect_rejects_invalid_options(measured, options, message, capsys):
    with pytest.raises(SystemExit) as error:
        main(["inspect", str(measured[0]), *options])
    assert error.value.code == 2
    assert message in capsys.readouterr().err


def test_inspect_cannot_replace_input(measured, capsys):
    path, _ = measured
    original = path.read_bytes()
    with pytest.raises(SystemExit) as error:
        main(["inspect", str(path), "--output", str(path)])
    assert error.value.code == 2
    assert path.read_bytes() == original
    assert "distinct paths" in capsys.readouterr().err


def test_inspect_entrypoint_outputs_valid_json(measured):
    completed = subprocess.run(
        [sys.executable, "-m", "workspace_analyzer", "inspect", str(measured[0])],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["reachable"] == 25


def test_inspect_empty_result_uses_null_bounds_and_rate(tmp_path, capsys):
    path = tmp_path / "empty.npz"
    AnalysisResult(np.empty((0, 3)), np.empty((0, 2)), None, {}).save(path)
    main(["inspect", str(path)])
    data = json.loads(capsys.readouterr().out)
    assert (
        data["samples"] == 0 and data["bounds"] is None and data["success_rate"] is None
    )


def test_inspect_summary_encoding_failure_keeps_other_diagnostics(
    measured, tmp_path, capsys
):
    source, result = measured
    result.metadata["external_measurement"] = float("nan")
    result.save(source)
    output, report, plot = (
        tmp_path / "diagnostics.npz",
        tmp_path / "summary.json",
        tmp_path / "samples.svg",
    )
    report.write_text("previous completed report")
    with pytest.raises(SystemExit) as error:
        main(
            [
                "inspect",
                str(source),
                "--output",
                str(output),
                "--report",
                str(report),
                "--plot",
                str(plot),
            ]
        )
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert "summary" in captured.err and "saved:" in captured.err
    assert str(output) in captured.err and str(plot) in captured.err
    assert captured.out == ""
    assert report.read_text() == "previous completed report"
    np.testing.assert_array_equal(AnalysisResult.load(output).points, result.points)
    assert plot.read_text().count("<circle ") == len(result.points)


def test_inspect_symlink_loop_has_clean_error(measured, tmp_path, capsys):
    source, _ = measured
    output = tmp_path / "loop.npz"
    output.symlink_to(output.name)
    with pytest.raises(SystemExit) as error:
        main(["inspect", str(source), "--output", str(output)])
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert "artifact path" in message and "loop.npz" in message
    assert "Traceback" not in message


@pytest.mark.parametrize("failed", ["npz", "json", "svg"])
def test_inspect_attempts_other_exports_after_one_failure(
    measured, tmp_path, failed, capsys
):
    source, original = measured
    paths = {kind: tmp_path / f"export.{kind}" for kind in ("npz", "json", "svg")}
    # Destination itself is valid at preflight; its parent fails during writing.
    blocker = tmp_path / "blocked-parent"
    blocker.write_text("keep this file")
    paths[failed] = blocker / f"export.{failed}"
    with pytest.raises(SystemExit) as error:
        main(
            [
                "inspect",
                str(source),
                "--output",
                str(paths["npz"]),
                "--report",
                str(paths["json"]),
                "--plot",
                str(paths["svg"]),
            ]
        )
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert "export failed" in captured.err and "saved:" in captured.err
    assert json.loads(captured.out)["samples"] == len(original.points)
    for kind, path in paths.items():
        assert not path.exists() if kind == failed else path.is_file()
    assert blocker.read_text() == "keep this file"
    np.testing.assert_array_equal(AnalysisResult.load(source).points, original.points)
