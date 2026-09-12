import numpy as np
import pytest

from examples.trajectory_gallery import _canonical_trajectory
from workspace_analyzer import create_solver
from workspace_analyzer.presets import default_reference_joints, default_robot_urdf
from workspace_analyzer.visualization import _resolve_mesh, _visual_mesh

MARVIN_URDF = default_robot_urdf()

pytestmark = pytest.mark.skipif(
    not MARVIN_URDF.is_file(), reason="Marvin asset repository is unavailable"
)


@pytest.mark.parametrize("arm", ["left", "right"])
def test_marvin_arm_construction_and_jacobian_accuracy(arm):
    solver = create_solver(
        str(MARVIN_URDF),
        base_link="base_link",
        tip_link=f"{arm}_ee",
        backend="numpy",
        dtype="float64",
    )
    assert solver.dof == 7
    assert all(f"_{arm[0].upper()}_" in name for name in solver.joint_names)
    q = default_reference_joints(solver)[None]
    assert q[0, 3] == pytest.approx(-np.pi / 2)
    jacobian = solver.jacobian(q)[0, :3]
    epsilon = 1e-7
    for joint in range(solver.dof):
        plus, minus = q.copy(), q.copy()
        plus[0, joint] += epsilon
        minus[0, joint] -= epsilon
        finite_difference = (
            solver.forward(plus)[0, :3, 3] - solver.forward(minus)[0, :3, 3]
        ) / (2 * epsilon)
        np.testing.assert_allclose(jacobian[:, joint], finite_difference, atol=1e-8)


def test_marvin_visual_meshes_are_resolvable():
    trimesh = pytest.importorskip("trimesh")
    import xml.etree.ElementTree as ET

    root = ET.parse(MARVIN_URDF).getroot()
    left_links = {
        "base_link",
        "shoulder_pitch_l_j1_link",
        "shoulder_roll_l_j2_link",
        "elbow_pitch_l_j3_link",
        "elbow_yaw_l_j4_link",
        "wrist_pitch_l_j5_link",
        "wrist_yaw_l_j6_link",
        "wrist_roll_l_j7_link",
    }
    visuals = [
        visual
        for link in root.findall("link")
        if link.get("name") in left_links
        for visual in link.findall("visual")
    ]
    assert len(visuals) == 8
    for visual in visuals:
        filename = visual.find("geometry/mesh").get("filename")
        assert _resolve_mesh(filename, MARVIN_URDF.parent).is_file()
        mesh = _visual_mesh(visual, MARVIN_URDF.parent, trimesh)
        assert len(mesh.vertices) > 0


@pytest.mark.parametrize("arm", ["left", "right"])
@pytest.mark.parametrize("shape", ["line", "circle", "figure8", "helix"])
def test_marvin_demo_trajectory(arm, shape):
    solver = create_solver(
        str(MARVIN_URDF),
        base_link=f"{arm}_arm_base",
        tip_link=f"{arm}_ee",
        backend="numpy",
        dtype="float64",
        max_iterations=200,
    )
    reference = default_reference_joints(solver)
    targets = solver.forward(reference)[None] @ _canonical_trajectory(
        shape,
        frames=120,
        scale=0.15,
        depth=0.15,
    )
    result = solver.solve_trajectory(
        targets,
        seed=reference,
        position_only=True,
        failure_restarts=16,
        dt=0.02,
    )
    assert result.success.all(), result.summary()
    actual = solver.forward(result.positions)
    np.testing.assert_allclose(actual[:, :3, 3], targets[:, :3, 3], atol=1e-4)
    assert np.all(result.positions >= solver.joint_limits[:, 0] - 1e-8)
    assert np.all(result.positions <= solver.joint_limits[:, 1] + 1e-8)
    assert result.max_joint_jump < 0.1, result.summary()
    assert result.summary()["velocity_violations"] == 0, result.summary()
