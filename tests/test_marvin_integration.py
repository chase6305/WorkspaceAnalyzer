from pathlib import Path

import numpy as np
import pytest

from workspace_analyzer import create_solver
from workspace_analyzer.visualization import _resolve_mesh, _visual_mesh

MARVIN_URDF = Path(
    "/home/ubuntu/workspace/chase/HumanoidAssets/Marvin_M6_S_CCS_696_V4.0/robot.urdf"
)

pytestmark = pytest.mark.skipif(
    not MARVIN_URDF.is_file(), reason="Marvin asset repository is unavailable"
)


def test_marvin_left_arm_construction_and_jacobian_accuracy():
    solver = create_solver(
        str(MARVIN_URDF),
        base_link="base_link",
        tip_link="left_ee",
        backend="numpy",
        dtype="float64",
    )
    assert solver.dof == 7
    assert all("_L_" in name for name in solver.joint_names)
    q = solver.joint_limits.mean(axis=1)[None]
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
