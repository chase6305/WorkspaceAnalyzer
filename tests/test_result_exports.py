"""Every CLI mode retains independent diagnostic exports after a write failure."""

import json
from pathlib import Path

import numpy as np
import pytest

from workspace_analyzer import AnalysisResult
from workspace_analyzer.cli import main

URDF = Path(__file__).parent / "fixtures/cartesian_stage.urdf"


def arguments(mode, tmp_path):
    args = [
        str(URDF),
        "--backend",
        "numpy",
        "--mode",
        mode,
        "--ik-restarts",
        "1",
        "--ik-rescue-restarts",
        "0",
    ]
    if mode == "targets":
        path = tmp_path / "targets.npy"
        np.save(path, [[0, 0, 0], [2, 0, 0]])
        return args + ["--targets", str(path)]
    if mode == "plane":
        return args + ["--plane", "xy", "--span", "3", "--resolution", "5"]
    args += ["--samples", "16"]
    if mode == "cartesian":
        args += ["--bounds", "-1.5", "1.5", "-1.5", "1.5", "0", "0"]
    return args


@pytest.mark.parametrize("mode", ["joint", "cartesian"])
def test_workspace_modes_export_summary_matching_inspect(mode, tmp_path, capsys):
    output, report = tmp_path / "result.npz", tmp_path / "report.json"
    result = main(
        arguments(mode, tmp_path) + ["--output", str(output), "--report", str(report)]
    )
    capsys.readouterr()
    main(["inspect", str(output)])
    assert json.loads(report.read_text()) == json.loads(capsys.readouterr().out)
    summary = json.loads(report.read_text())
    assert summary["samples"] == len(result.points) == 16
    assert summary["dof"] == 3
    if mode == "joint":
        assert summary["reachable"] is None and summary["success_rate"] is None
    else:
        assert summary["reachable"] == int(result.reachable.sum())


@pytest.mark.parametrize("mode", ["joint", "cartesian", "targets", "plane"])
@pytest.mark.parametrize("failed", ["npz", "json"])
def test_each_mode_attempts_all_exports_on_write_failure(
    mode, failed, tmp_path, capsys
):
    paths = {kind: tmp_path / f"result.{kind}" for kind in ("npz", "json")}
    blocker = tmp_path / "blocked-parent"
    blocker.write_text("keep")
    paths[failed] = blocker / f"result.{failed}"
    with pytest.raises(SystemExit) as error:
        main(
            arguments(mode, tmp_path)
            + ["--output", str(paths["npz"]), "--report", str(paths["json"])]
        )
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert "calculation completed" in message and "Saved:" in message
    assert str(paths[failed]) in message
    assert blocker.read_text() == "keep"
    if failed == "json":
        assert len(AnalysisResult.load(paths["npz"]).points) > 0
    else:
        assert json.loads(paths["json"].read_text())


def test_cartesian_failed_acceptance_still_saves_report_and_measurements(
    tmp_path, capsys
):
    output, report = tmp_path / "result.npz", tmp_path / "report.json"
    with pytest.raises(SystemExit) as error:
        main(
            arguments("cartesian", tmp_path)
            + [
                "--min-success-rate",
                "1",
                "--output",
                str(output),
                "--report",
                str(report),
            ]
        )
    assert error.value.code == 1
    result = AnalysisResult.load(output)
    data = json.loads(report.read_text())
    assert 0 < data["success_rate"] < 1
    assert data["metadata"]["acceptance"]["passed"] is False
    assert result.metadata["acceptance"] == data["metadata"]["acceptance"]
