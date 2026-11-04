"""Reference postures for demos and visualization, independent of IK defaults."""

import os
from pathlib import Path

import numpy as np


def default_robot_urdf(robot="marvin"):
    """Locate demo assets via HUMANOID_ASSETS or a sibling asset checkout."""
    directories = {
        "marvin": "Marvin_M6_S_CCS_696_V4.0",
        "w1": "Dexforce_W1_V3",
    }
    root = Path(
        os.environ.get(
            "HUMANOID_ASSETS", Path(__file__).resolve().parents[3] / "HumanoidAssets"
        )
    ).expanduser()
    return root / directories[robot] / "robot.urdf"


def require_robot_urdf(path, parser):
    """Report missing external assets as an actionable CLI usage error."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        parser.error(
            f"Robot URDF not found: {path}. Pass an explicit URDF path "
            "or set HUMANOID_ASSETS to your HumanoidAssets checkout. "
            "Robot assets are not bundled with workspace-analyzer."
        )
    return path


def default_reference_joints(solver):
    """Use joint centers, with Marvin's named elbow J4 joints bent by 90 degrees.

    Only URDF robot names ``marvin`` or ``marvin_*`` select this preset.
    Marvin's URDF uses negative angles for elbow flexion (-145 to +60 degrees).
    Match joint names rather than indices so arm-only and torso chains agree.
    """
    limits = np.asarray(solver.joint_limits)
    positions = limits.mean(axis=1)
    model_name = solver.model.name.casefold()
    if model_name != "marvin" and not model_name.startswith("marvin_"):
        return positions
    for index, name in enumerate(solver.joint_names):
        if name in {"ELBOW_YAW_L_J4", "ELBOW_YAW_R_J4"}:
            angle = -np.pi / 2
            if not limits[index, 0] <= angle <= limits[index, 1]:
                raise ValueError(
                    f"{name} cannot use the -90 degree reference within its limits"
                )
            positions[index] = angle
    return positions


def select_arm_links(model, arm="left", base_link=None, tip_link=None):
    """Use a recognized arm-only chain, or leave generic chain inference intact."""
    if base_link is None and tip_link is None:
        base, tip = f"{arm}_arm_base", f"{arm}_ee"
        if base in model.links and tip in model.links:
            return base, tip
    return base_link, tip_link
