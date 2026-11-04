"""Run benchmark/stress CLIs on a generic, analytically solvable chain."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
URDF = ROOT / "tests/fixtures/cartesian_stage.urdf"


@pytest.mark.parametrize("entry", ["benchmark_marvin.py", "stress_marvin.py"])
def test_benchmark_entries_use_generic_chain_and_requested_counts(entry, tmp_path):
    report = tmp_path / "report.json"
    options = (
        ["--repeats", "1", "--warmup", "0"]
        if entry.startswith("benchmark")
        else ["--duration", "0.000000001"]
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples" / entry),
            "--urdf",
            str(URDF),
            "--backend",
            "numpy",
            "--batch-size",
            "2",
            "--ik-targets",
            "7",
            "--seed",
            "17",
            "--restarts",
            "1",
            "--rescue-restarts",
            "0",
            "--report",
            str(report),
            *options,
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr
    data = json.loads(report.read_text())
    assert data["base_link"] == "base" and data["tip_link"] == "tcp"
    assert data["dof"] == 3
    assert data["seed"] == 17
    assert data["ik_targets"] == 7
    assert data["ik_success_rate"] == 1
    if entry.startswith("stress"):
        assert data["batches"] == 1
        assert data["max_success_fk_position_error"] <= 1e-5
        assert data["max_success_fk_rotation_error_rad"] <= 1e-5
        assert data["max_success_fk_residual"] <= 1e-5
        assert data["max_joint_limit_violation"] == 0
        assert data["jacobian_columns_checked"] == [0]
    else:
        assert data["ik_seed"] == 17


@pytest.mark.parametrize(
    "entry,flags",
    [
        ("stress_marvin.py", ["--duration", "nan"]),
        ("stress_marvin.py", ["--duration", "inf"]),
        ("stress_marvin.py", ["--duration", "0"]),
        ("stress_marvin.py", ["--min-success-rate", "1.1"]),
        ("stress_marvin.py", ["--seed", "-1"]),
        ("benchmark_marvin.py", ["--seed", "-1"]),
        ("benchmark_marvin.py", ["--repeats", "0"]),
        ("benchmark_marvin.py", ["--base-link", "missing"]),
    ],
)
def test_benchmark_entries_reject_invalid_arguments_cleanly(entry, flags):
    completed = subprocess.run(
        [sys.executable, str(ROOT / "examples" / entry), "--urdf", str(URDF), *flags],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 2
    assert "error:" in completed.stderr
    assert "Traceback" not in completed.stderr


@pytest.mark.parametrize("entry", ["benchmark_marvin.py", "stress_marvin.py"])
def test_benchmark_report_cannot_replace_robot_input(entry, tmp_path):
    robot = tmp_path / "robot.urdf"
    original = URDF.read_bytes()
    robot.write_bytes(original)
    report = tmp_path / "report.json"
    report.symlink_to(robot)
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples" / entry),
            "--urdf",
            str(robot),
            "--report",
            str(report),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 2
    assert "distinct paths" in completed.stderr
    assert robot.read_bytes() == original


def test_stress_independently_rejects_incorrect_successful_orientation(
    tmp_path, monkeypatch
):
    import numpy as np

    from examples import stress_marvin
    from workspace_analyzer import IKResult

    robot = tmp_path / "rotation.urdf"
    robot.write_text(
        '<robot name="wrist"><link name="base"/><link name="tcp"/>'
        '<joint name="wrist" type="revolute"><parent link="base"/>'
        '<child link="tcp"/><axis xyz="0 0 1"/>'
        '<limit lower="-3" upper="3"/></joint></robot>'
    )

    def incorrect_inverse(self, targets, **kwargs):
        count = len(targets)
        return IKResult(
            np.zeros((count, self.dof)), np.ones(count, dtype=bool), np.zeros(count), 1
        )

    monkeypatch.setattr(stress_marvin.KinematicsSolver, "inverse", incorrect_inverse)
    report = tmp_path / "report.json"
    with pytest.raises(SystemExit, match="numerical accuracy threshold"):
        stress_marvin.main(
            [
                "--urdf",
                str(robot),
                "--duration",
                "1e-9",
                "--batch-size",
                "2",
                "--ik-targets",
                "3",
                "--report",
                str(report),
            ]
        )
    data = json.loads(report.read_text())
    assert data["max_success_fk_position_error"] == 0
    assert data["max_success_residual"] == 0
    assert data["max_success_fk_rotation_error_rad"] > 0.1
    assert data["max_success_fk_residual"] > 0.1
