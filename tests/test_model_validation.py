from pathlib import Path

import numpy as np
import pytest

from workspace_analyzer import RobotModel


def _write_urdf(tmp_path: Path, body: str) -> Path:
    target = tmp_path / "invalid.urdf"
    target.write_text(f'<robot name="test">{body}</robot>', encoding="utf-8")
    return target


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ('<link name="a"/><link name="a"/>', "duplicate link"),
        ('<link/><link name="b"/>', "missing a name"),
        (
            '<link name="a"/><link name="b"/>'
            '<joint name="j" type="floating"><parent link="a"/>'
            '<child link="b"/></joint>',
            "unsupported type",
        ),
        (
            '<link name="a"/><link name="b"/>'
            '<joint name="j" type="revolute"><parent link="a"/>'
            '<child link="b"/><limit lower="0" upper="1"/>'
            '<mimic joint="driver"/></joint>',
            "mimic coupling",
        ),
        (
            '<link name="a"/><link name="b"/>'
            '<joint name="j" type="revolute"><parent link="a"/>'
            '<child link="b"/><limit lower="0" upper="1" velocity="-1"/></joint>',
            "positive finite velocity",
        ),
        (
            '<link name="a"/><link name="b"/>'
            '<joint name="j" type="fixed"><parent link="missing"/>'
            '<child link="b"/></joint>',
            "unknown parent or child",
        ),
        (
            '<link name="a"/><link name="b"/>'
            '<joint name="j1" type="fixed"><parent link="a"/>'
            '<child link="b"/></joint>'
            '<joint name="j2" type="fixed"><parent link="b"/>'
            '<child link="a"/></joint>',
            "cycle",
        ),
    ],
)
def test_invalid_urdf_is_rejected(tmp_path, body, message):
    with pytest.raises(ValueError, match=message):
        RobotModel.from_urdf(_write_urdf(tmp_path, body))


def test_malformed_urdf_xml_reports_the_input_path(tmp_path):
    source = _write_urdf(tmp_path, '<link name="base">')
    with pytest.raises(ValueError, match="invalid URDF XML") as error:
        RobotModel.from_urdf(source)
    assert str(source) in str(error.value)


@pytest.mark.parametrize("reverse_joints", [False, True])
@pytest.mark.parametrize(
    "branches,expected",
    [
        (
            (("long", ("revolute", "fixed", "fixed")), ("active", ("revolute",) * 2)),
            "active_1",
        ),
        (
            (("short", ("revolute",) * 2), ("deep", ("revolute", "fixed", "revolute"))),
            "deep_2",
        ),
        ((("a", ("revolute",)), ("z", ("revolute",))), "z_0"),
    ],
)
def test_default_tip_prioritizes_active_joints_then_depth_then_name(
    tmp_path, branches, expected, reverse_joints
):
    links = ['<link name="root"/>']
    joints = []
    for branch_name, kinds in branches:
        parent = "root"
        for index, kind in enumerate(kinds):
            child = f"{branch_name}_{index}"
            links.append(f'<link name="{child}"/>')
            joints.append(
                f'<joint name="{child}" type="{kind}"><parent link="{parent}"/>'
                f'<child link="{child}"/><limit lower="-1" upper="1"/></joint>'
            )
            parent = child
    if reverse_joints:
        joints.reverse()
    model = RobotModel.from_urdf(_write_urdf(tmp_path, "".join(links + joints)))
    assert model.chain()[-1].child == expected
    assert model.chain(expected) == ()


def test_default_tip_stays_in_the_requested_tree_of_a_forest(tmp_path):
    model = RobotModel.from_urdf(
        _write_urdf(
            tmp_path,
            '<link name="base"/><link name="tool"/><link name="other"/>'
            '<link name="other_mid"/><link name="other_tool"/>'
            '<link name="isolated"/>'
            '<joint name="arm" type="revolute"><parent link="base"/>'
            '<child link="tool"/><limit lower="-1" upper="1"/></joint>'
            '<joint name="other_a" type="revolute"><parent link="other"/>'
            '<child link="other_mid"/><limit lower="-1" upper="1"/></joint>'
            '<joint name="other_b" type="revolute"><parent link="other_mid"/>'
            '<child link="other_tool"/><limit lower="-1" upper="1"/></joint>',
        )
    )
    with pytest.raises(ValueError, match="forest"):
        model.chain()
    assert model.chain("base")[-1].child == "tool"
    assert model.chain("other")[-1].child == "other_tool"
    assert model.chain("isolated") == ()


def test_unknown_chain_links_are_rejected():
    fixture = Path(__file__).parent / "fixtures/two_link.urdf"
    model = RobotModel.from_urdf(fixture)
    with pytest.raises(ValueError, match="unknown base"):
        model.chain("missing", "tcp")
    with pytest.raises(ValueError, match="unknown tip"):
        model.chain("base", "missing")


@pytest.mark.parametrize("lower,upper", [("nan", "1"), ("0", "inf"), ("-inf", "1")])
def test_nonfinite_joint_limits_are_rejected(tmp_path, lower, upper):
    body = (
        '<link name="a"/><link name="b"/>'
        '<joint name="j" type="revolute"><parent link="a"/>'
        f'<child link="b"/><limit lower="{lower}" upper="{upper}"/></joint>'
    )
    with pytest.raises(ValueError, match="finite limits"):
        RobotModel.from_urdf(_write_urdf(tmp_path, body))


def test_unknown_base_is_reported_before_automatic_tip_selection():
    fixture = Path(__file__).parent / "fixtures/two_link.urdf"
    with pytest.raises(ValueError, match="unknown base"):
        RobotModel.from_urdf(fixture).chain("missing")


@pytest.mark.parametrize("scale", [1e-300, 1e300])
def test_finite_axes_normalize_without_overflow_or_underflow(tmp_path, scale):
    from workspace_analyzer import KinematicsSolver, SolverConfig

    body = (
        '<link name="a"/><link name="b"/>'
        '<joint name="j" type="revolute"><parent link="a"/><child link="b"/>'
        f'<axis xyz="{scale} {-scale} {scale}"/>'
        '<limit lower="-3" upper="3"/></joint>'
    )
    with np.errstate(over="raise", invalid="raise", under="raise"):
        model = RobotModel.from_urdf(_write_urdf(tmp_path, body))
    axis = np.array([1, -1, 1]) / np.sqrt(3)
    np.testing.assert_allclose(model.joints[0].axis, axis)
    solver = KinematicsSolver(model, config=SolverConfig(backend="numpy"))
    rotation = solver.forward([0.5])[:3, :3]
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(rotation @ axis, axis, atol=1e-12)
    assert np.trace(rotation) == pytest.approx(1 + 2 * np.cos(0.5))


@pytest.mark.parametrize(
    "fixture",
    ["two_link.urdf", "mixed_chain.urdf", "cartesian_stage.urdf", "continuous.urdf"],
)
def test_zero_tree_transforms_match_chain_fk(fixture):
    from workspace_analyzer import KinematicsSolver, SolverConfig

    model = RobotModel.from_urdf(Path(__file__).parent / "fixtures" / fixture)
    solver = KinematicsSolver(model, config=SolverConfig(backend="numpy"))
    neutral = model.zero_transforms()
    assert set(neutral) == set(model.links)
    for link, transform in solver.forward(np.zeros(solver.dof), all_links=True).items():
        np.testing.assert_allclose(neutral[link], transform, atol=1e-12)
    neutral[solver.tip_link][:] = 0
    assert np.linalg.det(model.zero_transforms()[solver.tip_link]) == pytest.approx(1)


def test_zero_tree_transforms_include_branches_and_disconnected_roots(tmp_path):
    model = RobotModel.from_urdf(
        _write_urdf(
            tmp_path,
            '<link name="root"/><link name="arm"/><link name="tool"/>'
            '<link name="sensor"/><link name="detached"/>'
            '<joint name="tool" type="fixed"><parent link="arm"/>'
            '<child link="tool"/><origin xyz="0 2 0"/></joint>'
            '<joint name="sensor" type="fixed"><parent link="root"/>'
            '<child link="sensor"/><origin xyz="0 0 3"/></joint>'
            '<joint name="arm" type="prismatic"><parent link="root"/>'
            '<child link="arm"/><origin xyz="1 0 0"/>'
            '<limit lower="1" upper="2"/></joint>',
        )
    )
    poses = model.zero_transforms()
    assert set(poses) == set(model.links)
    np.testing.assert_allclose(poses["tool"][:3, 3], [1, 2, 0])
    np.testing.assert_allclose(poses["sensor"][:3, 3], [0, 0, 3])
    np.testing.assert_array_equal(poses["detached"], np.eye(4))
