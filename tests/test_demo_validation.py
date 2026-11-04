"""Examples reject invalid inputs before solving or overwriting robot assets."""

import importlib
from pathlib import Path

import pytest

URDF = Path(__file__).parent / "fixtures/cartesian_stage.urdf"


@pytest.mark.parametrize(
    "module,flags",
    [
        ("trajectory_gallery", ["--scale", "nan"]),
        ("trajectory_gallery", ["--depth", "inf"]),
        ("trajectory_gallery", ["--dt", "nan"]),
        ("marvin_dexterity", ["--seed", "-1"]),
        ("marvin_dexterity", ["--dt", "inf"]),
        ("compare_w1_marvin", ["--circle-radius", "nan"]),
        ("compare_w1_marvin", ["--trajectory-posture-gain", "inf"]),
        ("compare_w1_marvin", ["--circle-center-offset", "0", "nan", "0"]),
        ("compare_w1_marvin", ["--target-rpy", "0", "0", "inf"]),
        ("compare_w1_marvin", ["--seed", "-1"]),
        ("benchmark_ik_bias", ["--batch-size", "0"]),
        ("benchmark_ik_bias", ["--repeats", "0"]),
    ],
)
def test_demo_invalid_arguments_have_clear_errors(module, flags, monkeypatch, capsys):
    entry = importlib.import_module(f"examples.{module}")
    if hasattr(entry, "create_solver"):
        monkeypatch.setattr(
            entry,
            "create_solver",
            lambda *a, **k: pytest.fail("invalid arguments reached solver"),
        )
    robot_args = (
        []
        if module == "benchmark_ik_bias"
        else (
            ["--w1-urdf", str(URDF), "--marvin-urdf", str(URDF)]
            if module == "compare_w1_marvin"
            else ["--urdf", str(URDF)]
        )
    )
    with pytest.raises(SystemExit) as error:
        entry.main([*robot_args, *flags])
    assert error.value.code == 2
    assert "error:" in capsys.readouterr().err


@pytest.mark.parametrize(
    "module",
    [
        "trajectory_gallery",
        "marvin_dexterity",
        "compare_w1_marvin",
        "marvin_single_arm",
    ],
)
def test_demo_export_cannot_replace_robot(module, tmp_path, capsys):
    entry = importlib.import_module(f"examples.{module}")
    robot = tmp_path / "robot.urdf"
    original = URDF.read_bytes()
    robot.write_bytes(original)
    single = module == "marvin_single_arm"
    destination = tmp_path / ("report.npz" if single else "report.json")
    destination.symlink_to(robot)
    robot_args = (
        ["--w1-urdf", str(robot), "--marvin-urdf", str(robot)]
        if module == "compare_w1_marvin"
        else ["--urdf", str(robot)]
    )
    with pytest.raises(SystemExit) as error:
        entry.main(
            [*robot_args, "--output" if single else "--report", str(destination)]
        )
    assert error.value.code == 2
    assert "distinct paths" in capsys.readouterr().err
    assert robot.read_bytes() == original


def test_dexterity_rejects_colliding_output_and_report(tmp_path, capsys):
    from examples.marvin_dexterity import main

    with pytest.raises(SystemExit) as error:
        main(
            [
                "--urdf",
                str(URDF),
                "--output",
                str(tmp_path / "same"),
                "--report",
                str(tmp_path / "same.npz"),
            ]
        )
    assert error.value.code == 2
    assert "distinct paths" in capsys.readouterr().err
    assert not (tmp_path / "same.npz").exists()
