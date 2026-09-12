from types import SimpleNamespace

import numpy as np
import pytest

from workspace_analyzer.presets import default_reference_joints


@pytest.mark.parametrize("name", ["ELBOW_YAW_L_J4", "ELBOW_YAW_R_J4"])
def test_marvin_reference_bends_named_elbow_90_degrees(name):
    solver = SimpleNamespace(
        joint_names=("torso", name, "wrist"),
        joint_limits=np.deg2rad([[-10, 10], [-145, 60], [-30, 30]]),
    )
    original = solver.joint_limits.copy()
    np.testing.assert_allclose(default_reference_joints(solver), [0, -np.pi / 2, 0])
    np.testing.assert_array_equal(solver.joint_limits, original)


def test_other_robots_keep_joint_centers():
    solver = SimpleNamespace(
        joint_names=("j1", "j4"), joint_limits=np.array([[0.0, 2.0], [-2.0, 3.0]])
    )
    np.testing.assert_array_equal(default_reference_joints(solver), [1, 0.5])


def test_incompatible_marvin_limits_are_not_silently_clipped():
    solver = SimpleNamespace(
        joint_names=("ELBOW_YAW_L_J4",), joint_limits=np.array([[0.0, 1.0]])
    )
    with pytest.raises(ValueError, match="limits"):
        default_reference_joints(solver)
