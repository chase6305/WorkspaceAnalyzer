"""Exercise demo entry points with real solver execution and saved artifacts."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from workspace_analyzer import AnalysisResult

ROOT = Path(__file__).parents[1]
URDF = ROOT / "tests/fixtures/cartesian_stage.urdf"


def test_suite_runs_all_experiments_on_generic_robot(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples/reachability_suite.py"),
            "--urdf",
            str(URDF),
            "--backend",
            "numpy",
            "--samples",
            "9",
            "--batch-size",
            "4",
            "--bounds",
            "-0.5",
            "0.5",
            "-0.5",
            "0.5",
            "0",
            "0",
            "--ik-restarts",
            "1",
            "--ik-rescue-restarts",
            "0",
            "--output-dir",
            str(tmp_path),
        ],
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["exports_complete"]
    assert all(case["saved"] for case in report["cases"].values())
    assert report["base_link"] == "base"
    assert set(report["cases"]) == {"joint_workspace", "position_ik", "fixed_pose_ik"}
    for name in ("position_ik", "fixed_pose_ik"):
        result = AnalysisResult.load(tmp_path / f"{name}.npz")
        assert result.reachable.all()
        np.testing.assert_array_equal(result.points[:, 2], 0)
        assert report["cases"][name]["success_rate"] == 1


def test_single_arm_demo_accepts_explicit_chain_and_plane_bounds(tmp_path):
    output = tmp_path / "single.npz"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples/marvin_single_arm.py"),
            "--urdf",
            str(URDF),
            "--base-link",
            "base",
            "--tip-link",
            "tcp",
            "--backend",
            "numpy",
            "--dtype",
            "float64",
            "--samples",
            "9",
            "--batch-size",
            "4",
            "--mode",
            "cartesian",
            "--bounds",
            "-0.5",
            "0.5",
            "-0.5",
            "0.5",
            "0.2",
            "0.2",
            "--ik-restarts",
            "1",
            "--ik-rescue-restarts",
            "0",
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr
    result = AnalysisResult.load(output)
    assert result.reachable.all()
    np.testing.assert_array_equal(result.points[:, 2], 0.2)
    assert "base -> tcp" in completed.stdout


def test_suite_report_reproduces_samples_and_solutions(tmp_path):
    import hashlib

    original_dir = tmp_path / "first"
    command = [sys.executable, str(ROOT / "examples/reachability_suite.py")]
    completed = subprocess.run(
        command
        + [
            "--urdf",
            str(URDF),
            "--backend",
            "numpy",
            "--samples",
            "19",
            "--strategy",
            "uniform",
            "--seed",
            "17",
            "--batch-size",
            "7",
            "--ik-max-iterations",
            "20",
            "--ik-restarts",
            "1",
            "--ik-rescue-restarts",
            "0",
            "--output-dir",
            str(original_dir),
        ],
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads((original_dir / "report.json").read_text())
    assert report["urdf_sha256"] == hashlib.sha256(URDF.read_bytes()).hexdigest()
    assert report["sampling"] == {
        "strategy": "uniform",
        "num_samples": 19,
        "batch_size": 7,
        "seed": 17,
    }
    assert report["solver"]["max_iterations"] == 20
    assert report["bounds_source"] == "joint_samples"
    replay_dir = tmp_path / "replay"
    replay = subprocess.run(
        command + report["reproduce_args"] + ["--output-dir", str(replay_dir)],
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert replay.returncode == 0, replay.stderr
    for name, case in report["cases"].items():
        first = AnalysisResult.load(original_dir / case["artifact"])
        second = AnalysisResult.load(replay_dir / case["artifact"])
        np.testing.assert_array_equal(first.points, second.points)
        np.testing.assert_array_equal(first.joint_positions, second.joint_positions)
        np.testing.assert_array_equal(first.reachable, second.reachable)
        if name != "joint_workspace":
            assert case["reachable"] == int(first.reachable.sum())
    position = AnalysisResult.load(original_dir / "position_ik.npz")
    pose = AnalysisResult.load(original_dir / "fixed_pose_ik.npz")
    np.testing.assert_array_equal(position.points, pose.points)


def _small_suite_args(output):
    return [
        "--urdf",
        str(URDF),
        "--backend",
        "numpy",
        "--samples",
        "5",
        "--ik-restarts",
        "1",
        "--ik-rescue-restarts",
        "0",
        "--output-dir",
        str(output),
    ]


@pytest.mark.parametrize(
    "failed_case", ["joint_workspace", "position_ik", "fixed_pose_ik"]
)
def test_suite_preserves_other_results_and_report_on_save_failure(
    failed_case, tmp_path, monkeypatch, capsys
):
    from examples import reachability_suite as demo

    original_save = AnalysisResult.save
    attempted = []

    def save(result, path, **kwargs):
        attempted.append(Path(path).stem)
        if Path(path).stem == failed_case:
            raise OSError("simulated disk failure")
        return original_save(result, path, **kwargs)

    monkeypatch.setattr(AnalysisResult, "save", save)
    with pytest.raises(SystemExit) as error:
        demo.main(_small_suite_args(tmp_path))
    assert error.value.code == 2
    assert attempted == ["joint_workspace", "position_ik", "fixed_pose_ik"]
    report = json.loads((tmp_path / "report.json").read_text())
    assert not report["exports_complete"]
    for name, case in report["cases"].items():
        assert case["saved"] == (name != failed_case)
        assert (tmp_path / case["artifact"]).is_file() == case["saved"]
        if name == failed_case:
            assert "simulated disk failure" in case["export_error"]
        else:
            assert len(AnalysisResult.load(tmp_path / case["artifact"]).points) == 5
    diagnostic = capsys.readouterr()
    assert "saved:" in diagnostic.err and "report.json" in diagnostic.err
    assert "All three experiments saved" not in diagnostic.out


@pytest.mark.parametrize("destination", ["joint_workspace.npz", "report.json"])
def test_suite_rejects_output_aliasing_input_before_computation(
    destination, tmp_path, monkeypatch, capsys
):
    from examples import reachability_suite as demo

    robot = tmp_path / "robot.urdf"
    robot.write_bytes(URDF.read_bytes())
    before = robot.read_bytes()
    (tmp_path / destination).symlink_to(robot)
    monkeypatch.setattr(
        demo.WorkspaceAnalyzer,
        "analyze",
        lambda *a, **k: pytest.fail("invalid paths reached analysis"),
    )
    args = _small_suite_args(tmp_path)
    args[1] = str(robot)
    with pytest.raises(SystemExit) as error:
        demo.main(args)
    assert error.value.code == 2
    assert "distinct paths" in capsys.readouterr().err
    assert robot.read_bytes() == before


def test_suite_report_failure_keeps_all_npz_results(tmp_path, monkeypatch, capsys):
    from examples import reachability_suite as demo

    def fail_report(*args, **kwargs):
        raise OSError("report write failed")

    monkeypatch.setattr(demo, "atomic_write_text", fail_report)
    with pytest.raises(SystemExit) as error:
        demo.main(_small_suite_args(tmp_path))
    assert error.value.code == 2
    for name in ("joint_workspace", "position_ik", "fixed_pose_ik"):
        assert len(AnalysisResult.load(tmp_path / f"{name}.npz").points) == 5
    diagnostic = capsys.readouterr().err
    assert "saved:" in diagnostic
    assert "report write failed" in diagnostic


@pytest.mark.parametrize("invalid_destination", ["directory", "alias"])
def test_suite_rejects_invalid_artifact_destinations_before_analysis(
    invalid_destination, tmp_path, monkeypatch, capsys
):
    from examples import reachability_suite as demo

    first = tmp_path / "joint_workspace.npz"
    if invalid_destination == "directory":
        first.mkdir()
    else:
        first.write_bytes(b"previous result")
        (tmp_path / "position_ik.npz").hardlink_to(first)
    monkeypatch.setattr(
        demo.WorkspaceAnalyzer,
        "analyze",
        lambda *a, **k: pytest.fail("invalid paths reached analysis"),
    )
    with pytest.raises(SystemExit) as error:
        demo.main(_small_suite_args(tmp_path))
    assert error.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_suite_marks_previous_artifact_as_unsaved_when_replacement_fails(
    tmp_path, monkeypatch, capsys
):
    from examples import reachability_suite as demo

    previous = tmp_path / "position_ik.npz"
    previous.write_bytes(b"previous run")
    original_save = AnalysisResult.save

    def save(result, path, **kwargs):
        if Path(path) == previous:
            raise OSError("cannot replace previous result")
        return original_save(result, path, **kwargs)

    monkeypatch.setattr(AnalysisResult, "save", save)
    with pytest.raises(SystemExit) as error:
        demo.main(_small_suite_args(tmp_path))
    assert error.value.code == 2
    report = json.loads((tmp_path / "report.json").read_text())
    assert not report["cases"]["position_ik"]["saved"]
    assert "cannot replace" in report["cases"]["position_ik"]["export_error"]
    assert previous.read_bytes() == b"previous run"
