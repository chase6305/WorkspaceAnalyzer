import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from workspace_analyzer import create_solver

EXAMPLE = Path(__file__).parents[1] / "examples" / "plane_reachability.py"
SPEC = importlib.util.spec_from_file_location("plane_reachability", EXAMPLE)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

RZ = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
RX = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
RY = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])


def _transform(rotation, translation):
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


@pytest.fixture
def mounted_solver(tmp_path):
    urdf = tmp_path / "mounted.urdf"
    urdf.write_text("""<robot name="mounted">
      <link name="world"/><link name="left_arm_base"/>
      <link name="joint1"/><link name="joint2"/><link name="left_ee"/>
      <joint name="mount" type="fixed">
        <parent link="world"/><child link="left_arm_base"/>
        <origin xyz="0.2 -0.3 0.5" rpy="0 0 1.5707963267948966"/>
      </joint>
      <joint name="first" type="revolute">
        <parent link="left_arm_base"/><child link="joint1"/>
        <origin xyz="0.1 0.2 0.3" rpy="1.5707963267948966 0 0"/>
        <axis xyz="0 0 1"/><limit lower="-3" upper="3"/>
      </joint>
      <joint name="second" type="revolute">
        <parent link="joint1"/><child link="joint2"/>
        <origin xyz="0.4 -0.1 0.2" rpy="0 1.5707963267948966 0"/>
        <axis xyz="0 0 1"/><limit lower="-3" upper="3"/>
      </joint>
      <joint name="tip" type="fixed">
        <parent link="joint2"/><child link="left_ee"/>
        <origin xyz="0.3 0 0"/>
      </joint>
    </robot>""")
    return create_solver(
        urdf, base_link="left_arm_base", tip_link="left_ee", backend="numpy"
    )


@pytest.mark.parametrize("frame", ["reference", "base", "world"])
@pytest.mark.parametrize(
    "plane,axes", [("xy", (0, 1, 2)), ("xz", (0, 2, 1)), ("yz", (1, 2, 0))]
)
def test_scan_uses_same_frame_for_targets_report_and_viewer(
    frame, plane, axes, mounted_solver, monkeypatch, tmp_path
):
    import workspace_analyzer.visualization as visualization

    calls = []
    displayed = []

    def inverse(targets, **kwargs):
        start = sum(len(target) for target, _ in calls)
        ids = np.arange(start, start + len(targets))
        calls.append((targets.copy(), kwargs))
        return SimpleNamespace(
            positions=np.column_stack((ids + 0.1, ids + 0.2)),
            success=ids % 2 == 0,
            residual=ids * 0.01,
        )

    monkeypatch.setattr(mounted_solver, "inverse", inverse)
    monkeypatch.setattr(MODULE, "create_solver", lambda *a, **k: mounted_solver)
    monkeypatch.setattr(
        visualization,
        "ViserWorkspace",
        lambda *a, **k: SimpleNamespace(
            add_workspace=displayed.append, wait=lambda: None
        ),
    )
    output = tmp_path / "plane.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(EXAMPLE),
            "--frame",
            frame,
            "--plane",
            plane,
            "--center",
            "0.7",
            "0.8",
            "0.9",
            "--constant",
            "0.6",
            "--span",
            "0.4",
            "--resolution",
            "3",
            "--batch-size",
            "4",
            "--seed",
            "11",
            "--rpy",
            "0",
            "0",
            str(np.pi / 2),
            "--viser",
            "--output",
            str(output),
        ],
    )
    MODULE.main()

    world_from_base = _transform(RZ, [0.2, -0.3, 0.5])
    base_from_reference = _transform(RX, [0.1, 0.2, 0.3]) @ _transform(
        RY, [0.4, -0.1, 0.2]
    )
    world_from_frame = {
        "world": np.eye(4),
        "base": world_from_base,
        "reference": world_from_base @ base_from_reference,
    }[frame]
    expected_points = np.tile([0.7, 0.8, 0.9], (9, 1))
    expected_points[:, axes[:2]] += np.array(
        [(a, b) for a in (-0.2, 0, 0.2) for b in (-0.2, 0, 0.2)]
    )
    expected_points[:, axes[2]] = 0.6
    targets = np.concatenate([target for target, _ in calls])
    np.testing.assert_allclose(
        targets[:, :3, 3] @ RZ.T + world_from_base[:3, 3],
        expected_points @ world_from_frame[:3, :3].T + world_from_frame[:3, 3],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        RZ @ targets[:, :3, :3],
        np.broadcast_to(world_from_frame[:3, :3] @ RZ, (9, 3, 3)),
        atol=1e-12,
    )
    assert [len(target) for target, _ in calls] == [4, 4, 1]
    assert [kwargs["random_seed"] for _, kwargs in calls] == [11, 15, 19]
    assert all(not kwargs["position_only"] for _, kwargs in calls)
    result = displayed[0]
    np.testing.assert_allclose(result.points, targets[:, :3, 3])
    np.testing.assert_allclose(result.joint_positions[:, 0], np.arange(9) + 0.1)
    np.testing.assert_allclose(result.residual, np.arange(9) * 0.01)
    assert result.manipulability is None
    assert result.metadata["coordinate_frame"] == mounted_solver.base_link
    report = json.loads(output.read_text())
    assert (
        report["coordinate_frame"]
        == {"reference": "joint2", "base": "left_arm_base", "world": "world"}[frame]
    )
    assert report["plane_center"][axes[2]] == 0.6
    assert report["reachable"] == 5
    np.testing.assert_array_equal(np.array(report["grid"]).ravel(), result.reachable)
    np.testing.assert_allclose(
        world_from_base @ np.array(report["base_from_plane_frame"]),
        world_from_frame,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "args",
    [
        ["--span", "nan"],
        ["--span", "inf"],
        ["--span", "0"],
        ["--resolution", "2"],
        ["--batch-size", "0"],
        ["--seed", "-1"],
        ["--constant", "inf"],
        ["--center", "nan", "0", "0"],
        ["--rpy", "0", "inf", "0"],
    ],
)
def test_invalid_scan_arguments_fail_before_loading_robot(args, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [str(EXAMPLE), *args])
    with pytest.raises(SystemExit) as error:
        MODULE.main()
    assert error.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_reference_frame_supports_single_joint(mounted_solver):
    solver = create_solver(
        mounted_solver.model.source,
        base_link="left_arm_base",
        tip_link="joint1",
        backend="numpy",
    )
    name, base_from_frame = MODULE._plane_frame(solver, "reference")
    assert name == "joint1"
    np.testing.assert_allclose(
        base_from_frame, _transform(RX, [0.1, 0.2, 0.3]), atol=1e-12
    )


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_scan_preserves_solved_configurations(mounted_solver, backend):
    if backend == "torch":
        pytest.importorskip("torch")
    solver = create_solver(
        mounted_solver.model.source,
        base_link="left_arm_base",
        tip_link="left_ee",
        backend=backend,
        device="cpu",
    )
    xyz = MODULE._numpy(solver.forward([[0, 0], [0.2, -0.3], [-0.2, 0.3]]))[:, :3, 3]
    result = MODULE._scan_plane(
        solver,
        xyz,
        np.eye(4),
        [0, 0, 0],
        position_only=True,
        batch_size=2,
        seed=42,
    )
    assert result.reachable.all()
    reached = MODULE._numpy(solver.forward(result.joint_positions))[:, :3, 3]
    np.testing.assert_allclose(reached, result.points, atol=solver.config.tolerance)
    np.testing.assert_allclose(
        result.residual, np.linalg.norm(reached - xyz, axis=1), atol=1e-12
    )
