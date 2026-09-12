import importlib.util
from pathlib import Path

import numpy as np
import pytest

EXAMPLE = Path(__file__).parents[1] / "examples" / "compare_w1_marvin.py"
SPEC = importlib.util.spec_from_file_location("compare_w1_marvin", EXAMPLE)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
_local_targets = MODULE._local_targets
_shared_circle = MODULE._shared_circle
_wilson_interval = MODULE._wilson_interval
_ik_profile = MODULE._ik_profile
_circle_perturbation_cases = MODULE._circle_perturbation_cases


def test_circle_presets_use_parallel_world_planes():
    expected_constant_axis = {
        "horizontal": 2,
        "vertical_xz": 1,
        "vertical_yz": 0,
        "chest_front": 0,
    }
    for preset, constant_axis in expected_constant_axis.items():
        poses, center = _shared_circle(65, 0.12, preset)
        offsets = poses[:, :3, 3] - center
        np.testing.assert_allclose(offsets[:, constant_axis], 0.0, atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(offsets, axis=1), 0.12)


def test_right_arm_presets_mirror_shoulder_relative_lateral_offset():
    _, left = _shared_circle(17, 0.08, "chest_front", arm="left")
    _, right = _shared_circle(17, 0.08, "chest_front", arm="right")
    np.testing.assert_allclose(right, left * [1.0, -1.0, 1.0])


def test_wilson_interval_contains_observed_rate_and_shrinks_with_sample_count():
    low_small, high_small = _wilson_interval(57, 100)
    low_large, high_large = _wilson_interval(570, 1000)
    assert low_small < 0.57 < high_small
    assert low_large < 0.57 < high_large
    assert high_large - low_large < high_small - low_small


def test_ik_profiles_increase_assurance_monotonically():
    fast = _ik_profile("fast")
    balanced = _ik_profile("balanced")
    rigorous = _ik_profile("rigorous")
    assert fast["maximum_seeds_per_target"] < balanced["maximum_seeds_per_target"]
    assert balanced["maximum_seeds_per_target"] < rigorous["maximum_seeds_per_target"]
    assert fast["max_iterations"] < balanced["max_iterations"]
    assert balanced["max_iterations"] < rigorous["max_iterations"]


def test_circle_perturbation_cases_are_deterministic():
    import argparse

    args = argparse.Namespace(
        robustness_position_delta=0.02,
        robustness_angle_deg=5.0,
        robustness_radius_fraction=0.1,
        orientation_mode="fixed-world",
    )
    cases = _circle_perturbation_cases(np.array([1.0, 2.0, 3.0]), 0.1, (0, 0, 0), args)
    assert len(cases) == 14
    assert cases[0][0] == "center_x_minus"
    assert np.isclose(cases[1][1][0], 1.02)
    assert np.isclose(cases[6][2], 0.09)


def test_circle_center_override_maps_from_reference_world_to_solver_base():
    center = np.array([0.1, 0.45, -0.05])
    poses, returned_center = _shared_circle(
        17, 0.08, "horizontal", center_override=center
    )
    world_from_base = np.eye(4)
    world_from_base[:3, 3] = [0.2, -0.1, 1.0]
    reference_world = np.eye(4)
    reference_world[:3, 3] = [-0.05, 0.25, 1.3]
    analysis = type(
        "Analysis",
        (),
        {
            "metadata": {
                "world_from_solver_base": world_from_base.tolist(),
                "reference_anchor_world": reference_world.tolist(),
                "coordinate_scale": 1.0,
            }
        },
    )()

    local = _local_targets(poses, analysis)
    recovered_world = (
        local[:, :3, 3] @ world_from_base[:3, :3].T + world_from_base[:3, 3]
    )
    np.testing.assert_allclose(returned_center, center)
    np.testing.assert_allclose(
        recovered_world, poses[:, :3, 3] + reference_world[:3, 3]
    )
    recovered_rotation = world_from_base[:3, :3] @ local[:, :3, :3]
    np.testing.assert_allclose(recovered_rotation, poses[:, :3, :3], atol=1e-12)


@pytest.mark.parametrize(
    "points",
    [
        np.empty((0, 3)),
        np.ones((1, 3)),
        np.eye(3),
        np.arange(18).reshape(6, 3),
        np.array([[x, y, x + y] for x in range(3) for y in range(3)]),
    ],
)
def test_degenerate_clouds_have_zero_hull_volume_and_valid_counts(points):
    pytest.importorskip("scipy")
    assert MODULE._convex_hull_volume(points) == 0.0
    convergence = MODULE._convex_hull_convergence(points)
    assert max(convergence["sample_counts"]) == len(points)
    assert convergence["volumes"] == [0.0] * 4
    assert convergence["relative_change_75_to_100"] is None


def test_tetrahedron_hull_volume():
    pytest.importorskip("scipy")
    points = np.vstack((np.zeros(3), np.eye(3)))
    assert MODULE._convex_hull_volume(points) == pytest.approx(1 / 6)


def test_full_rank_hull_precision_failure_is_unavailable(monkeypatch):
    spatial = pytest.importorskip("scipy.spatial")

    def fail(*args, **kwargs):
        raise spatial.QhullError("precision error")

    monkeypatch.setattr(spatial, "ConvexHull", fail)
    assert MODULE._convex_hull_volume(np.vstack((np.zeros(3), np.eye(3)))) is None
