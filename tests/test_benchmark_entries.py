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
