"""Compare Dexforce W1 and Marvin M6 single-arm workspace quality."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from workspace_analyzer import AnalysisResult, create_solver
from workspace_analyzer.presets import (
    default_reference_joints,
    default_robot_urdf,
    require_robot_urdf,
)

W1_URDF = default_robot_urdf("w1")
MARVIN_URDF = default_robot_urdf()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--w1-urdf", "--robot-a-urdf", dest="w1_urdf", type=Path, default=W1_URDF
    )
    parser.add_argument(
        "--marvin-urdf",
        "--robot-b-urdf",
        dest="marvin_urdf",
        type=Path,
        default=MARVIN_URDF,
    )
    parser.add_argument(
        "--w1-arm-length",
        "--robot-a-arm-length",
        dest="w1_arm_length",
        type=float,
        help="optional nominal shoulder-to-TCP length used for scale normalization",
    )
    parser.add_argument(
        "--marvin-arm-length",
        "--robot-b-arm-length",
        dest="marvin_arm_length",
        type=float,
        help="optional nominal shoulder-to-TCP length used for scale normalization",
    )
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    parser.add_argument(
        "--base-link",
        help="comparison base; defaults to <arm>_arm_base for a fair arm-only study",
    )
    parser.add_argument("--backend", choices=("auto", "numpy", "torch"), default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument(
        "--ik-profile",
        choices=("fast", "balanced", "rigorous"),
        default="balanced",
        help="trade IK assurance for runtime",
    )
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--shared-targets", type=int, default=1_000)
    parser.add_argument("--trajectory-frames", type=int, default=200)
    parser.add_argument(
        "--align-arm-length",
        action="store_true",
        help="scale link2-relative Cartesian coordinates to one common arm length",
    )
    parser.add_argument(
        "--aligned-arm-length",
        type=float,
        help="common virtual length in metres; defaults to the shorter arm",
    )
    parser.add_argument(
        "--reference-link-index",
        type=int,
        default=2,
        help="use the child of this 1-based active joint as the comparison base",
    )
    parser.add_argument("--circle-radius", type=float, default=None)
    parser.add_argument(
        "--orientation-mode",
        choices=("position-only", "fixed-world"),
        default="fixed-world",
        help="constrain one common world-frame TCP orientation along the circle",
    )
    parser.add_argument(
        "--target-rpy",
        type=float,
        nargs=3,
        default=None,
        metavar=("ROLL", "PITCH", "YAW"),
        help="fixed world-frame TCP orientation in radians",
    )
    parser.add_argument(
        "--circle-preset",
        choices=("horizontal", "vertical_xz", "vertical_yz", "chest_front"),
        default="horizontal",
    )
    parser.add_argument(
        "--circle-center-offset",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="override the link2-relative center in parallel world axes",
    )
    parser.add_argument("--voxel-resolution", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--joint-centering-gain",
        type=float,
        default=0.0,
        help="null-space bias away from joint limits during IK",
    )
    parser.add_argument(
        "--trajectory-posture-gain",
        type=float,
        default=0.005,
        help="null-space bias toward the first trajectory solution",
    )
    parser.add_argument(
        "--jump-repair-threshold",
        type=float,
        default=0.25,
        help="re-run multi-start IK above this adjacent joint jump in radians",
    )
    parser.add_argument(
        "--loop-closure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="distribute periodic joint drift and reproject every circle frame",
    )
    parser.add_argument(
        "--robustness-test",
        action="store_true",
        help="sweep circle center, radius, and fixed-orientation perturbations",
    )
    parser.add_argument("--robustness-position-delta", type=float, default=0.02)
    parser.add_argument("--robustness-angle-deg", type=float, default=5.0)
    parser.add_argument("--robustness-radius-fraction", type=float, default=0.1)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--viser", action="store_true")
    parser.add_argument("--autoplay", action="store_true")
    parser.add_argument("--no-visuals", action="store_true")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    args.w1_urdf = require_robot_urdf(args.w1_urdf, parser)
    args.marvin_urdf = require_robot_urdf(args.marvin_urdf, parser)
    if args.target_rpy is None:
        args.target_rpy = (
            (0.0, np.pi / 2, 0.0)
            if args.circle_preset == "chest_front"
            else (-np.pi / 2 if args.arm == "left" else np.pi / 2, 0.0, 0.0)
        )
    if args.circle_radius is None:
        args.circle_radius = 0.08 if args.circle_preset == "chest_front" else 0.12
    if min(args.samples, args.shared_targets, args.trajectory_frames) <= 0:
        parser.error("sample counts must be positive")
    if args.reference_link_index < 1:
        parser.error("reference-link-index must be positive")
    if args.voxel_resolution < 4:
        parser.error("voxel-resolution must be at least 4")
    if args.circle_radius <= 0:
        parser.error("circle-radius must be positive")
    if args.joint_centering_gain < 0:
        parser.error("joint-centering-gain must be non-negative")
    if args.trajectory_posture_gain < 0 or args.jump_repair_threshold <= 0:
        parser.error("trajectory posture gain/repair threshold are invalid")
    if any(
        value is not None and value <= 0
        for value in (args.w1_arm_length, args.marvin_arm_length)
    ):
        parser.error("arm-length overrides must be positive")
    if args.aligned_arm_length is not None and args.aligned_arm_length <= 0:
        parser.error("aligned-arm-length must be positive")
    if (
        min(
            args.robustness_position_delta,
            args.robustness_angle_deg,
            args.robustness_radius_fraction,
        )
        <= 0
    ):
        parser.error("robustness perturbations must be positive")

    solvers = {
        "dexforce_w1": _solver(args.w1_urdf, args),
        "marvin_m6": _solver(args.marvin_urdf, args),
    }
    ik_settings = _ik_profile(args.ik_profile)
    arm_lengths = {
        "dexforce_w1": args.w1_arm_length
        or _effective_arm_length(solvers["dexforce_w1"], args.reference_link_index),
        "marvin_m6": args.marvin_arm_length
        or _effective_arm_length(solvers["marvin_m6"], args.reference_link_index),
    }
    aligned_length = None
    if args.align_arm_length:
        aligned_length = args.aligned_arm_length or min(arm_lengths.values())
    rng = np.random.default_rng(args.seed)
    dofs = {solver.dof for solver in solvers.values()}
    unit_q = rng.random((args.samples, next(iter(dofs)))) if len(dofs) == 1 else None
    analyses = {
        name: _analyze(
            solver,
            unit_q
            if unit_q is not None
            else np.random.default_rng(args.seed).random((args.samples, solver.dof)),
            args.reference_link_index,
            arm_lengths[name],
            aligned_length,
        )
        for name, solver in solvers.items()
    }
    canonical = {name: _canonical_points(result) for name, result in analyses.items()}
    occupancy = _occupancy_comparison(canonical, args.voxel_resolution)
    shared_xyz = _shared_targets(canonical, args.shared_targets, rng)
    shared_reachability = {
        name: _reachability(solver, shared_xyz, analyses[name], ik_settings=ik_settings)
        for name, solver in solvers.items()
    }
    canonical_trajectory, circle_center = _shared_circle(
        args.trajectory_frames,
        args.circle_radius,
        args.circle_preset,
        args.circle_center_offset,
        args.target_rpy,
        args.arm,
    )
    shared_fixed_orientation = {
        name: _reachability(
            solver,
            shared_xyz,
            analyses[name],
            position_only=False,
            target_rotation=canonical_trajectory[0, :3, :3],
            ik_settings=ik_settings,
        )
        for name, solver in solvers.items()
    }
    trajectories = {
        name: solver.solve_trajectory(
            _local_targets(canonical_trajectory, analyses[name]),
            position_only=args.orientation_mode == "position-only",
            failure_restarts=ik_settings["trajectory_restarts"],
            dt=0.02,
            posture_gain=args.trajectory_posture_gain,
            jump_repair_threshold=args.jump_repair_threshold,
            enforce_loop_closure=args.loop_closure,
        )
        for name, solver in solvers.items()
    }
    robustness = (
        _circle_robustness(solvers, analyses, circle_center, args, ik_settings)
        if args.robustness_test
        else None
    )
    report = {
        "arm": args.arm,
        "methodology": {
            "paired_joint_samples": unit_q is not None,
            "random_seed": args.seed,
            "workspace_samples_per_robot": args.samples,
            "shared_targets": args.shared_targets,
            "reachability_interval": "Wilson score 95%",
            "convex_hull_caveat": (
                "outer envelope; may include unreachable holes and is not "
                "occupied volume"
            ),
            "arm_length_alignment": args.align_arm_length,
            "aligned_arm_length": aligned_length,
            "reference_active_link_index": args.reference_link_index,
            "ik_profile": args.ik_profile,
            "ik_settings": ik_settings,
            "not_included": [
                "self_collision",
                "environment_collision",
                "payload_and_joint_torque",
                "stiffness_and_deflection",
                "controller_tracking_error",
                "thermal_limits",
            ],
        },
        "backend": {
            name: f"{solver.backend}:{solver.device}"
            for name, solver in solvers.items()
        },
        "robots": {
            name: _robot_report(
                solvers[name],
                analyses[name],
                canonical[name],
                (
                    args.w1_arm_length
                    if name == "dexforce_w1"
                    else args.marvin_arm_length
                ),
                args.reference_link_index,
            )
            for name in solvers
        },
        "common_voxel_grid": occupancy,
        "shared_position_targets": shared_reachability,
        "shared_fixed_orientation_targets": shared_fixed_orientation,
        "metric_definitions": _metric_definitions(),
        "shared_circle_trajectory": {
            "frame": "parallel world axes translated to each structural link2 base",
            "preset": args.circle_preset,
            "center_offset": circle_center.tolist(),
            "radius": args.circle_radius,
            "orientation_mode": args.orientation_mode,
            "target_rpy_world": list(args.target_rpy),
            "trajectory_posture_gain": args.trajectory_posture_gain,
            "jump_repair_threshold": args.jump_repair_threshold,
            "loop_closure_enforced": args.loop_closure,
            "results": {
                name: trajectory.summary() for name, trajectory in trajectories.items()
            },
        },
    }
    if robustness is not None:
        report["circle_perturbation_robustness"] = robustness
    report["relative_marvin_over_w1"] = _relative_summary(report)
    text = json.dumps(report, indent=2)
    print(text)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    if args.viser:
        _visualize(solvers, analyses, trajectories, report, args)


def _solver(path: Path, args):
    settings = _ik_profile(args.ik_profile)
    return create_solver(
        str(path),
        base_link=args.base_link or f"{args.arm}_arm_base",
        tip_link=f"{args.arm}_ee",
        backend=args.backend,
        device=args.device,
        dtype=args.dtype,
        max_iterations=settings["max_iterations"],
        joint_centering_gain=args.joint_centering_gain,
    )


def _analyze(
    solver,
    unit_q,
    reference_link_index=2,
    normalization_arm_length=None,
    aligned_arm_length=None,
) -> AnalysisResult:
    from workspace_analyzer.visualization import _zero_tree_transforms

    lower, upper = solver.joint_limits.T
    q = lower + unit_q * (upper - lower)
    poses = _numpy(solver.forward(q))
    quality = solver.dexterity(q)
    world_from_base = _zero_tree_transforms(solver.model)[solver.base_link]
    reference_link = _active_reference_link(solver, reference_link_index)
    reference_world = _zero_tree_transforms(solver.model)[reference_link]
    normalization_arm_length = normalization_arm_length or _effective_arm_length(
        solver, reference_link_index
    )
    coordinate_scale = (
        1.0
        if aligned_arm_length is None
        else aligned_arm_length / normalization_arm_length
    )
    return AnalysisResult(
        points=poses[:, :3, 3],
        joint_positions=q,
        manipulability=_numpy(quality.manipulability),
        metrics={
            "minimum_singular_value": _numpy(quality.minimum_singular_value),
            "isotropy": _numpy(quality.isotropy),
            "joint_limit_margin": _numpy(quality.joint_limit_margin),
        },
        metadata={
            "robot": solver.model.name,
            "samples": len(q),
            "mode": "comparison",
            "comparison_frame": "parallel_world_axes_at_structural_link2",
            "world_from_solver_base": world_from_base.tolist(),
            "reference_link": reference_link,
            "reference_active_link_index": reference_link_index,
            "reference_anchor_world": reference_world.tolist(),
            "normalization_arm_length": normalization_arm_length,
            "aligned_arm_length": aligned_arm_length,
            "coordinate_scale": coordinate_scale,
        },
    )


def _robot_report(
    solver,
    result,
    canonical_points,
    arm_length_override=None,
    reference_link_index=2,
):
    xyz_min, xyz_max = canonical_points.min(0), canonical_points.max(0)
    metrics = {"manipulability": result.manipulability, **result.metrics}
    urdf_arm_length = _effective_arm_length(solver, reference_link_index)
    arm_length = (
        urdf_arm_length if arm_length_override is None else float(arm_length_override)
    )
    coordinate_length = result.metadata.get("aligned_arm_length") or arm_length
    hull_volume = _convex_hull_volume(canonical_points)
    return {
        "name": solver.model.name,
        "dof": solver.dof,
        "joint_names": solver.joint_names,
        "joint_ranges": (
            solver.joint_limits[:, 1] - solver.joint_limits[:, 0]
        ).tolist(),
        "urdf_effective_arm_length": urdf_arm_length,
        "normalization_arm_length": arm_length,
        "normalization_arm_length_source": (
            "urdf_chain" if arm_length_override is None else "cli_nominal_override"
        ),
        "reference_link": result.metadata["reference_link"],
        "cartesian_coordinate_scale": result.metadata["coordinate_scale"],
        "maximum_sampled_radius": float(
            np.max(np.linalg.norm(canonical_points, axis=1))
        ),
        "normalized_maximum_sampled_radius": float(
            np.max(np.linalg.norm(canonical_points, axis=1)) / coordinate_length
        ),
        "reference_base_frame_xyz_min": xyz_min.tolist(),
        "reference_base_frame_xyz_max": xyz_max.tolist(),
        "reference_base_frame_xyz_extent": (xyz_max - xyz_min).tolist(),
        "reference_base_frame_aabb_volume": float(np.prod(xyz_max - xyz_min)),
        "reference_base_frame_convex_hull_volume": hull_volume,
        "normalized_aabb_volume": float(
            np.prod(xyz_max - xyz_min) / coordinate_length**3
        ),
        "normalized_convex_hull_volume": (
            None if hull_volume is None else float(hull_volume / coordinate_length**3)
        ),
        "convex_hull_convergence": _convex_hull_convergence(canonical_points),
        "metrics": {name: _percentiles(values) for name, values in metrics.items()},
        "normalized_metrics": {
            "manipulability_over_length_cubed": _percentiles(
                result.manipulability / arm_length**3
            ),
            "minimum_singular_value_over_length": _percentiles(
                result.metrics["minimum_singular_value"] / arm_length
            ),
            "isotropy": _percentiles(result.metrics["isotropy"]),
        },
    }


def _effective_arm_length(solver, reference_link_index=2):
    """Return chain length from a structural active-link reference to TCP."""
    active_count = 0
    found_reference = False
    length = 0.0
    for joint in solver.chain:
        if found_reference:
            length += float(np.linalg.norm(joint.origin[:3, 3]))
        if joint.active:
            active_count += 1
            if active_count == reference_link_index:
                found_reference = True
    if not found_reference or length <= 0:
        raise ValueError("cannot determine a positive effective arm length")
    return length


def _active_reference_link(solver, reference_link_index):
    active_count = 0
    for joint in solver.chain:
        if joint.active:
            active_count += 1
            if active_count == reference_link_index:
                return joint.child
    raise ValueError(
        f"reference-link-index {reference_link_index} exceeds solver DoF {solver.dof}"
    )


def _occupancy_comparison(point_sets, resolution):
    all_points = np.concatenate(list(point_sets.values()))
    lower, upper = all_points.min(0), all_points.max(0)
    cell = np.maximum((upper - lower) / resolution, 1e-12)
    occupied = {}
    for name, points in point_sets.items():
        index = np.floor((points - lower) / cell).astype(int)
        index = np.clip(index, 0, resolution - 1)
        occupied[name] = set(np.ravel_multi_index(index.T, (resolution,) * 3))
    first, second = occupied.values()
    voxel_volume = float(np.prod(cell))
    union = first | second
    return {
        "resolution": resolution,
        "voxel_volume": voxel_volume,
        "occupied_volume": {
            name: len(values) * voxel_volume for name, values in occupied.items()
        },
        "intersection_volume": len(first & second) * voxel_volume,
        "overlap_jaccard": len(first & second) / max(len(union), 1),
    }


def _shared_targets(point_sets, count, rng):
    lower = np.maximum.reduce([points.min(0) for points in point_sets.values()])
    upper = np.minimum.reduce([points.max(0) for points in point_sets.values()])
    if np.any(lower >= upper):
        raise RuntimeError("robot workspace AABBs do not overlap")
    return rng.uniform(lower, upper, (count, 3))


def _reachability(
    solver,
    xyz,
    analysis,
    *,
    position_only=True,
    target_rotation=None,
    ik_settings=None,
):
    ik_settings = _ik_profile("balanced") if ik_settings is None else ik_settings
    targets = np.broadcast_to(np.eye(4), (len(xyz), 4, 4)).copy()
    if target_rotation is not None:
        targets[:, :3, :3] = target_rotation
    targets[:, :3, 3] = xyz
    targets = _local_targets(targets, analysis)
    result = solver.inverse(
        targets,
        position_only=position_only,
        restarts=ik_settings["restarts"],
        rescue_restarts=ik_settings["rescue_restarts"],
        rescue_rounds=ik_settings["rescue_rounds"],
    )
    success, residual = _numpy(result.success).astype(bool), _numpy(result.residual)
    failed = residual[~success]
    reachable = int(np.count_nonzero(success))
    return {
        "samples": len(xyz),
        "task": "position_only" if position_only else "fixed_orientation",
        "reachable": reachable,
        "success_rate": float(np.mean(success)),
        "success_rate_wilson_95": list(_wilson_interval(reachable, len(xyz))),
        "failure_residual_median": None
        if not len(failed)
        else float(np.median(failed)),
    }


def _ik_profile(name):
    profiles = {
        "fast": {
            "max_iterations": 120,
            "restarts": 2,
            "rescue_restarts": 8,
            "rescue_rounds": 1,
            "trajectory_restarts": 4,
            "maximum_seeds_per_target": 10,
        },
        "balanced": {
            "max_iterations": 200,
            "restarts": 4,
            "rescue_restarts": 16,
            "rescue_rounds": 2,
            "trajectory_restarts": 8,
            "maximum_seeds_per_target": 36,
        },
        "rigorous": {
            "max_iterations": 400,
            "restarts": 8,
            "rescue_restarts": 32,
            "rescue_rounds": 3,
            "trajectory_restarts": 16,
            "maximum_seeds_per_target": 104,
        },
    }
    try:
        return profiles[name].copy()
    except KeyError as exc:
        raise ValueError(f"unknown IK profile {name!r}") from exc


def _shared_circle(
    frames,
    radius,
    preset,
    center_override=None,
    target_rpy=(0, 0, 0),
    arm="left",
):
    presets = {
        # Centers and plane axes use the same world-axis convention for both robots.
        "horizontal": (np.array([0.0, 0.48, 0.0]), (0, 1)),
        "vertical_xz": (np.array([0.0, 0.55, 0.0]), (0, 2)),
        "vertical_yz": (np.array([0.0, 0.52, 0.10]), (1, 2)),
        # Frontal plane: +X forward, -Y inward for the left shoulder, -Z down.
        "chest_front": (np.array([0.40, -0.10, -0.10]), (1, 2)),
    }
    center, axes = presets[preset]
    center = center.copy()
    if arm == "right":
        center[1] *= -1.0
    if center_override is not None:
        center = np.asarray(center_override, dtype=float)
    phase = np.linspace(0.0, 2.0 * np.pi, frames)
    poses = np.broadcast_to(np.eye(4), (frames, 4, 4)).copy()
    poses[:, :3, :3] = _rpy_rotation(target_rpy)
    poses[:, :3, 3] = center
    poses[:, axes[0], 3] += radius * np.cos(phase)
    poses[:, axes[1], 3] += radius * np.sin(phase)
    return poses, center


def _circle_perturbation_cases(center, radius, target_rpy, args):
    """Build deterministic sensitivity cases around the nominal circle."""
    center = np.asarray(center, dtype=float)
    delta = float(args.robustness_position_delta)
    cases = []
    for axis, label in enumerate(("x", "y", "z")):
        for sign, suffix in ((-1.0, "minus"), (1.0, "plus")):
            offset = center.copy()
            offset[axis] += sign * delta
            cases.append((f"center_{label}_{suffix}", offset, radius, target_rpy))
    dr = radius * float(args.robustness_radius_fraction)
    cases.extend(
        [
            ("radius_minus", center, max(radius - dr, radius * 0.01), target_rpy),
            ("radius_plus", center, radius + dr, target_rpy),
        ]
    )
    if args.orientation_mode != "position-only":
        angle = np.deg2rad(float(args.robustness_angle_deg))
        for axis, label in enumerate(("roll", "pitch", "yaw")):
            for sign, suffix in ((-1.0, "minus"), (1.0, "plus")):
                rpy = np.asarray(target_rpy, dtype=float).copy()
                rpy[axis] += sign * angle
                cases.append((f"orientation_{label}_{suffix}", center, radius, rpy))
    return cases


def _circle_robustness(solvers, analyses, center, args, ik_settings):
    cases = _circle_perturbation_cases(
        center, args.circle_radius, args.target_rpy, args
    )
    results = {name: [] for name in solvers}
    for case_name, case_center, case_radius, case_rpy in cases:
        canonical, _ = _shared_circle(
            args.trajectory_frames,
            case_radius,
            args.circle_preset,
            case_center,
            case_rpy,
            args.arm,
        )
        for name, solver in solvers.items():
            trajectory = solver.solve_trajectory(
                _local_targets(canonical, analyses[name]),
                position_only=args.orientation_mode == "position-only",
                failure_restarts=ik_settings["trajectory_restarts"],
                dt=0.02,
                posture_gain=args.trajectory_posture_gain,
                jump_repair_threshold=args.jump_repair_threshold,
                enforce_loop_closure=args.loop_closure,
            )
            results[name].append({"case": case_name, **trajectory.summary()})
    aggregate = {}
    for name, entries in results.items():

        def vals(key):
            return [float(e[key]) for e in entries if e.get(key) is not None]

        success = [float(e.get("success_rate", 0.0)) for e in entries]
        aggregate[name] = {
            "cases": len(entries),
            "cases_with_100_percent_success": sum(v >= 1.0 for v in success),
            "worst_success_rate": min(success) if success else None,
            "maximum_joint_jump_across_cases": max(
                vals("max_joint_jump"), default=None
            ),
            "minimum_singular_value_across_cases": min(
                vals("minimum_singular_value"), default=None
            ),
            "minimum_joint_limit_margin_across_cases": min(
                vals("minimum_joint_limit_margin"), default=None
            ),
            "case_results": entries,
        }
    return {
        "perturbations": {
            "position_delta_m": args.robustness_position_delta,
            "angle_delta_deg": args.robustness_angle_deg,
            "radius_fraction": args.robustness_radius_fraction,
            "orientation_cases_included": args.orientation_mode != "position-only",
        },
        "robots": aggregate,
    }


def _rpy_rotation(rpy):
    roll, pitch, yaw = np.asarray(rpy, dtype=float)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _first_active_joint_frame(solver):
    transform = np.eye(4)
    for joint in solver.chain:
        transform = transform @ joint.origin
        if joint.active:
            return transform
    raise RuntimeError("solver chain has no active joint")


def _canonical_points(result):
    world_from_base = np.asarray(result.metadata["world_from_solver_base"])
    reference_world = np.asarray(result.metadata["reference_anchor_world"])
    world_points = result.points @ world_from_base[:3, :3].T + world_from_base[:3, 3]
    return (world_points - reference_world[:3, 3]) * result.metadata.get(
        "coordinate_scale", 1.0
    )


def _local_targets(canonical_targets, analysis):
    world_from_base = np.asarray(analysis.metadata["world_from_solver_base"])
    base_from_world = np.linalg.inv(world_from_base)
    reference_world = np.asarray(analysis.metadata["reference_anchor_world"])
    coordinate_scale = analysis.metadata.get("coordinate_scale", 1.0)
    local = np.asarray(canonical_targets).copy()
    local[:, :3, :3] = (
        base_from_world[:3, :3][None, :, :] @ canonical_targets[:, :3, :3]
    )
    world_points = (
        canonical_targets[:, :3, 3] / coordinate_scale + reference_world[:3, 3]
    )
    local[:, :3, 3] = world_points @ base_from_world[:3, :3].T + base_from_world[:3, 3]
    return local


def _visualize(solvers, analyses, trajectories, report, args):
    from workspace_analyzer.visualization import ViserWorkspace

    w1 = ViserWorkspace(
        solvers["dexforce_w1"],
        port=args.port,
        label="dexforce_w1",
        workspace_root="/workspaces/dexforce_w1",
        gui_label="Dexforce W1",
        scene_offset=(-1.1, 0.0, 0.0),
        initial_q=np.zeros(solvers["dexforce_w1"].dof),
        load_robot_visuals=not args.no_visuals,
        load_full_robot=True,
        reference_active_link_index=args.reference_link_index,
    )
    marvin = ViserWorkspace(
        solvers["marvin_m6"],
        server=w1.server,
        label="marvin_m6",
        workspace_root="/workspaces/marvin_m6",
        gui_label="Marvin M6",
        scene_offset=(1.1, 0.0, 0.0),
        initial_q=default_reference_joints(solvers["marvin_m6"]),
        load_robot_visuals=not args.no_visuals,
        load_full_robot=True,
        reference_active_link_index=args.reference_link_index,
    )
    for name, viewer in (("dexforce_w1", w1), ("marvin_m6", marvin)):
        viewer.add_workspace(analyses[name], color_metric="isotropy", visible=False)
        viewer.add_trajectory(trajectories[name])
    w1.add_dashboard(_comparison_markdown(report))
    w1.frame_view((marvin,))
    if args.autoplay:
        w1.play_trajectory()
        marvin.play_trajectory()
    try:
        w1.wait()
    finally:
        marvin.close()


def _comparison_markdown(report):
    w1 = report["robots"]["dexforce_w1"]
    marvin = report["robots"]["marvin_m6"]
    relative = report["relative_marvin_over_w1"]
    reach = report["shared_position_targets"]
    fixed_reach = report["shared_fixed_orientation_targets"]
    trajectory = report["shared_circle_trajectory"]["results"]
    circle = report["shared_circle_trajectory"]
    occupancy = report["common_voxel_grid"]["occupied_volume"]
    aligned = report["methodology"]["arm_length_alignment"]
    w1_manip = w1["metrics"]["manipulability"]["median"]
    marvin_manip = marvin["metrics"]["manipulability"]["median"]
    w1_normalized_manip = w1["normalized_metrics"]["manipulability_over_length_cubed"][
        "median"
    ]
    marvin_normalized_manip = marvin["normalized_metrics"][
        "manipulability_over_length_cubed"
    ]["median"]
    scale_label = "link2 equal-length virtual" if aligned else "original physical"
    interpretation = (
        "After link2 length alignment, W1 has slightly greater workspace envelope, "
        "normalized manipulability, isotropy, and fixed-orientation reachability; "
        "Marvin's principal advantage is longer physical reach."
        if aligned
        else "At original scale, Marvin emphasizes absolute workspace and aggregate "
        "manipulability; W1 has more directionally balanced median dexterity."
    )
    return (
        "## W1 versus Marvin M6 conclusions\n"
        "| Metric | Dexforce W1 | Marvin M6 |\n"
        "|---|---:|---:|\n"
        "| Convex-hull volume | "
        f"{_md_number(w1['reference_base_frame_convex_hull_volume'], ' m³')} | "
        f"{_md_number(marvin['reference_base_frame_convex_hull_volume'], ' m³')} |\n"
        f"| URDF effective arm length | {w1['urdf_effective_arm_length']:.3f} m | "
        f"{marvin['urdf_effective_arm_length']:.3f} m |\n"
        f"| Normalization arm length | {w1['normalization_arm_length']:.3f} m | "
        f"{marvin['normalization_arm_length']:.3f} m |\n"
        "| Length-normalized convex hull | "
        f"{_md_number(w1['normalized_convex_hull_volume'])} | "
        f"{_md_number(marvin['normalized_convex_hull_volume'])} |\n"
        f"| Raw manipulability median | {w1_manip:.4f} "
        f"| {marvin_manip:.4f} |\n"
        "| Length-normalized manipulability median | "
        f"{w1_normalized_manip:.4f} | {marvin_normalized_manip:.4f} |\n"
        f"| Sampled voxel occupancy | {occupancy['dexforce_w1']:.3f} m³ | "
        f"{occupancy['marvin_m6']:.3f} m³ |\n"
        f"| Isotropy median | {w1['metrics']['isotropy']['median']:.4f} "
        f"| {marvin['metrics']['isotropy']['median']:.4f} |\n"
        f"| Shared position reachability (95% CI) | {_rate_ci(reach['dexforce_w1'])} "
        f"| {_rate_ci(reach['marvin_m6'])} |\n"
        f"| Shared fixed-orientation reachability (95% CI) | "
        f"{_rate_ci(fixed_reach['dexforce_w1'])} | "
        f"{_rate_ci(fixed_reach['marvin_m6'])} |\n"
        f"| Circle max joint jump | {trajectory['dexforce_w1']['max_joint_jump']:.4f} "
        f"rad | {trajectory['marvin_m6']['max_joint_jump']:.4f} rad |\n"
        f"| Normalized max jump | "
        f"{trajectory['dexforce_w1']['max_normalized_joint_jump']:.5f} | "
        f"{trajectory['marvin_m6']['max_normalized_joint_jump']:.5f} |\n"
        f"| Normalized loop-closure error | "
        f"{_md_number(trajectory['dexforce_w1']['normalized_closure_joint_error'])} | "
        f"{_md_number(trajectory['marvin_m6']['normalized_closure_joint_error'])} |\n"
        f"| Minimum singular value on circle | "
        f"{_md_number(trajectory['dexforce_w1']['minimum_singular_value'])} | "
        f"{_md_number(trajectory['marvin_m6']['minimum_singular_value'])} |\n"
        f"| Minimum joint-limit margin | "
        f"{_md_number(trajectory['dexforce_w1']['minimum_joint_limit_margin'])} | "
        f"{_md_number(trajectory['marvin_m6']['minimum_joint_limit_margin'])} |\n"
        f"| Peak joint velocity | "
        f"{_md_number(trajectory['dexforce_w1']['max_joint_velocity'], ' rad/s')} | "
        f"{_md_number(trajectory['marvin_m6']['max_joint_velocity'], ' rad/s')} |\n\n"
        "- Marvin convex-hull workspace ratio: "
        f"**{_md_number(relative['convex_hull_volume_ratio'], '×')}**\n"
        "- Marvin manipulability median ratio: "
        f"**{_md_number(relative['manipulability_median_ratio'], '×')}**\n"
        "- Marvin/W1 length-normalized hull ratio: "
        f"**{_md_number(relative['normalized_convex_hull_volume_ratio'], '×')}**\n"
        "- Marvin/W1 length-normalized manipulability ratio: "
        f"**{_md_number(relative['normalized_manipulability_median_ratio'], '×')}**\n"
        "- Marvin/W1 isotropy median ratio: "
        f"**{_md_number(relative['isotropy_median_ratio'], '×')}**\n"
        f"- Shared random targets: **position-only IK**; circle: "
        f"**{circle['orientation_mode']} IK**, `{circle['preset']}` world plane.\n"
        "- Convex-hull 75%-to-100% sample change: W1 "
        "**"
        f"{_md_percent(w1['convex_hull_convergence']['relative_change_75_to_100'])}"
        "**, "
        "Marvin "
        "**"
        f"{_md_percent(marvin['convex_hull_convergence']['relative_change_75_to_100'])}"
        "**.\n"
        f"- Comparison scale: **{scale_label}**.\n"
        f"- Interpretation: {interpretation}\n\n"
        "### How to read the metrics\n"
        "- **Convex-hull volume:** outer workspace envelope; it may include "
        "unreachable holes.\n"
        "- **Manipulability:** aggregate local Cartesian velocity capacity; "
        "compare only under identical units and Jacobian definition.\n"
        "- **Isotropy:** minimum/maximum Jacobian singular-value ratio; higher "
        "means more directionally balanced.\n"
        "- **Shared reachability:** success on identical link2-relative "
        "position targets; it does not impose tool orientation.\n"
        "- **Joint jump:** adjacent-frame change and therefore dependent on "
        "trajectory sampling and timing.\n"
        "- **Singular value / limit margin:** lower values indicate proximity "
        "to singularity or a joint boundary.\n"
        "- No single row is an overall robot score; collision, payload, torque, "
        "stiffness, and task-specific poses are not included."
    )


def _metric_definitions():
    return {
        "convex_hull_volume": {
            "meaning": "outer envelope of sampled TCP positions",
            "unit": "m^3",
            "higher_is": (
                "larger positional workspace, not guaranteed filled reachability"
            ),
        },
        "normalization_arm_length": {
            "meaning": ("URDF chain length or an explicitly supplied nominal length"),
            "unit": "m",
            "caveat": "geometric upper-bound chain length, not guaranteed radial reach",
        },
        "normalized_convex_hull_volume": {
            "meaning": "convex hull volume divided by effective arm length cubed",
            "higher_is": (
                "larger workspace envelope after removing uniform length scale"
            ),
        },
        "manipulability": {
            "meaning": "sqrt(det(J_position J_position^T))",
            "higher_is": "greater aggregate local Cartesian motion capacity",
        },
        "isotropy": {
            "meaning": "minimum over maximum position-Jacobian singular value",
            "range": [0.0, 1.0],
            "higher_is": "more directionally balanced local motion",
        },
        "shared_position_reachability": {
            "meaning": ("position-only IK success on identical link2-relative targets"),
            "unit": "fraction",
        },
        "shared_fixed_orientation_reachability": {
            "meaning": (
                "full-pose IK success on identical positions and one common "
                "world rotation"
            ),
            "unit": "fraction",
        },
        "max_joint_jump": {
            "meaning": "maximum absolute single-joint change between adjacent frames",
            "unit": "rad",
            "depends_on": ["trajectory geometry", "frame count", "IK branch"],
        },
        "normalized_closure_joint_error": {
            "meaning": "joint-space first/last error normalized by each joint range",
            "lower_is": "better closed-loop posture consistency",
        },
    }


def _md_number(value, suffix=""):
    return "n/a" if value is None else f"{value:.3f}{suffix}"


def _md_percent(value):
    return "n/a" if value is None else f"{value:.2%}"


def _rate_ci(result):
    low, high = result["success_rate_wilson_95"]
    return f"{result['success_rate']:.1%} ({low:.1%}–{high:.1%})"


def _percentiles(values):
    p05, median, p95 = np.percentile(values, (5, 50, 95))
    return {"p05": float(p05), "median": float(median), "p95": float(p95)}


def _convex_hull_volume(points):
    try:
        from scipy.spatial import ConvexHull, QhullError
    except ImportError:
        return None
    if len(points) < 4:
        return 0.0
    try:
        return float(ConvexHull(points).volume)
    except QhullError:
        # Planar and linear workspaces have zero 3-D volume. Other numerical
        # failures are unavailable estimates, rather than empty workspaces.
        if np.linalg.matrix_rank(points - points[0]) < 3:
            return 0.0
        return None


def _convex_hull_convergence(points):
    fractions = (0.25, 0.5, 0.75, 1.0)
    counts = [
        min(len(points), max(4, int(round(len(points) * value)))) for value in fractions
    ]
    volumes = [_convex_hull_volume(points[:count]) for count in counts]
    final = volumes[-1]
    return {
        "sample_counts": counts,
        "volumes": volumes,
        "relative_change_75_to_100": (
            None
            if final in (None, 0) or volumes[-2] is None
            else float(abs(final - volumes[-2]) / final)
        ),
    }


def _wilson_interval(successes, samples, z=1.959963984540054):
    if samples <= 0:
        return (None, None)
    rate = successes / samples
    denominator = 1.0 + z * z / samples
    center = (rate + z * z / (2.0 * samples)) / denominator
    radius = z * np.sqrt(rate * (1.0 - rate) / samples + z * z / (4 * samples**2))
    radius /= denominator
    return (float(max(0.0, center - radius)), float(min(1.0, center + radius)))


def _relative_summary(report):
    w1 = report["robots"]["dexforce_w1"]
    marvin = report["robots"]["marvin_m6"]

    def ratio(key):
        return marvin[key] / w1[key]

    return {
        "aabb_volume_ratio": ratio("reference_base_frame_aabb_volume"),
        "convex_hull_volume_ratio": (
            None
            if w1["reference_base_frame_convex_hull_volume"] is None
            else ratio("reference_base_frame_convex_hull_volume")
        ),
        "urdf_effective_arm_length_ratio": ratio("urdf_effective_arm_length"),
        "normalization_arm_length_ratio": ratio("normalization_arm_length"),
        "normalized_convex_hull_volume_ratio": ratio("normalized_convex_hull_volume"),
        "manipulability_median_ratio": (
            marvin["metrics"]["manipulability"]["median"]
            / w1["metrics"]["manipulability"]["median"]
        ),
        "normalized_manipulability_median_ratio": (
            marvin["normalized_metrics"]["manipulability_over_length_cubed"]["median"]
            / w1["normalized_metrics"]["manipulability_over_length_cubed"]["median"]
        ),
        "minimum_singular_value_median_ratio": (
            marvin["metrics"]["minimum_singular_value"]["median"]
            / w1["metrics"]["minimum_singular_value"]["median"]
        ),
        "isotropy_median_ratio": (
            marvin["metrics"]["isotropy"]["median"]
            / w1["metrics"]["isotropy"]["median"]
        ),
        "shared_target_success_rate_difference": (
            report["shared_position_targets"]["marvin_m6"]["success_rate"]
            - report["shared_position_targets"]["dexforce_w1"]["success_rate"]
        ),
    }


def _numpy(value):
    return (
        value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    )


if __name__ == "__main__":
    main()
