"""Command seeds must control IK restarts as well as target sampling."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from workspace_analyzer import (
    AnalysisResult,
    CartesianConfig,
    SamplingConfig,
    WorkspaceAnalyzer,
    create_solver,
)

ROOT = Path(__file__).parents[1]
ROBOT = ROOT / "tests/fixtures/two_link.urdf"


@pytest.mark.parametrize("entry", ["cli", "single", "suite"])
def test_cartesian_entry_seed_changes_ik_at_identical_targets(entry, tmp_path):
    results = []
    for index, seed in enumerate((1, 2, 1)):
        output = tmp_path / str(index)
        common = [
            "--backend",
            "numpy",
            "--dtype",
            "float64",
            "--samples",
            "1",
            "--seed",
            str(seed),
            "--bounds",
            "1",
            "1",
            "0",
            "0",
            "0",
            "0",
            "--ik-restarts",
            "2",
            "--ik-rescue-restarts",
            "0",
        ]
        if entry == "cli":
            command = [
                sys.executable,
                "-m",
                "workspace_analyzer",
                str(ROBOT),
                "--mode",
                "cartesian",
                "--output",
                str(output),
                *common,
            ]
            artifact = output.with_suffix(".npz")
        elif entry == "single":
            command = [
                sys.executable,
                str(ROOT / "examples/marvin_single_arm.py"),
                "--urdf",
                str(ROBOT),
                "--mode",
                "cartesian",
                "--output",
                str(output),
                *common,
            ]
            artifact = output.with_suffix(".npz")
        else:
            command = [
                sys.executable,
                str(ROOT / "examples/reachability_suite.py"),
                "--urdf",
                str(ROBOT),
                "--output-dir",
                str(output),
                *common,
            ]
            artifact = output / "position_ik.npz"
        completed = subprocess.run(command, capture_output=True, text=True, timeout=20)
        assert completed.returncode == 0, completed.stderr
        result = AnalysisResult.load(artifact)
        assert result.reachable.all()
        assert result.metadata["sampling_seed"] == seed
        assert result.metadata["ik_random_seed"] == seed
        assert result.metadata["solver_settings"]["random_seed"] == seed
        solver = create_solver(ROBOT, backend="numpy")
        np.testing.assert_allclose(
            solver.forward(result.joint_positions)[:, :3, 3], result.points, atol=1e-5
        )
        results.append(result)
    for result in results:
        np.testing.assert_array_equal(result.points, [[1, 0, 0]])
    assert np.sign(results[0].joint_positions[0, 1]) != np.sign(
        results[1].joint_positions[0, 1]
    )
    np.testing.assert_array_equal(
        results[0].joint_positions, results[2].joint_positions
    )


def test_cartesian_api_records_independent_solver_and_sampling_seeds():
    solver = create_solver(ROBOT, backend="numpy", random_seed=1, max_iterations=123)
    config = CartesianConfig(
        bounds=[[1, 1], [0, 0], [0, 0]],
        sampling=SamplingConfig(num_samples=1, seed=17),
        restarts=2,
        rescue_restarts=0,
    )
    result = WorkspaceAnalyzer(solver).analyze_cartesian(config)
    assert result.metadata["sampling_seed"] == 17
    assert result.metadata["ik_random_seed"] == 1
    assert result.metadata["solver_settings"]["max_iterations"] == 123
    assert result.reachable.all()
    assert result.joint_positions[0, 1] > 0


def test_single_arm_bounds_estimation_uses_requested_seed(tmp_path, capsys):
    from examples.marvin_single_arm import main

    robot = ROOT / "tests/fixtures/cartesian_stage.urdf"
    output = tmp_path / "single.npz"
    main(
        [
            "--urdf",
            str(robot),
            "--backend",
            "numpy",
            "--dtype",
            "float64",
            "--mode",
            "cartesian",
            "--samples",
            "1",
            "--seed",
            "17",
            "--ik-restarts",
            "1",
            "--ik-rescue-restarts",
            "0",
            "--output",
            str(output),
        ]
    )
    result = AnalysisResult.load(output)
    q = np.random.default_rng(17).uniform(-1, 1, (4096, 3))
    expected = np.stack((q.min(0) - 0.02, q.max(0) + 0.02), axis=1)
    np.testing.assert_allclose(result.metadata["bounds"], expected, atol=1e-12)
    assert result.metadata["bounds_estimation"]["sampling_seed"] == 17
    assert result.metadata["bounds_estimation"]["samples"] == 4096
