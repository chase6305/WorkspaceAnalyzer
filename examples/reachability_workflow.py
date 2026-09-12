"""Run repeatable reachability, local orientation, and quality-gate checks."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from workspace_analyzer import (
    BoundaryConfig,
    IKDiversityConfig,
    IKTrial,
    PosePerturbations,
    ReachabilityConfig,
    ResultCache,
    WorkspaceAnalyzer,
    analyze_ik_stability,
    create_solver,
    make_ik_seed_trials,
    refine_translation_boundary,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--base-link")
    parser.add_argument("--tip-link")
    parser.add_argument(
        "--backend", choices=("auto", "numpy", "torch"), default="numpy"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--max-iterations", type=int, default=150)
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--rescue-restarts", type=int, default=8)
    parser.add_argument("--rescue-rounds", type=int, default=2)
    parser.add_argument("--joint-fraction", type=float, default=0.12)
    parser.add_argument("--orientation-offset-deg", type=float, default=15.0)
    parser.add_argument("--min-known-success", type=float, default=1.0)
    parser.add_argument(
        "--min-quality-rate",
        type=float,
        default=None,
        help="optional minimum quality acceptance among known FK targets",
    )
    parser.add_argument("--min-isotropy", type=float, default=0.05)
    parser.add_argument("--min-joint-limit-margin", type=float, default=0.1)
    parser.add_argument("--robustness-test", action="store_true")
    parser.add_argument("--perturb-translation-m", type=float, default=0.01)
    parser.add_argument("--perturb-rotation-deg", type=float, default=5.0)
    parser.add_argument(
        "--min-robust-rate",
        type=float,
        default=None,
        help="optional fraction of references with IK success for every perturbation",
    )
    parser.add_argument("--boundary-test", action="store_true")
    parser.add_argument("--boundary-distance-m", type=float, default=2.0)
    parser.add_argument("--boundary-tolerance-m", type=float, default=1e-4)
    parser.add_argument("--boundary-max-rounds", type=int, default=16)
    parser.add_argument("--stability-test", action="store_true")
    parser.add_argument(
        "--stability-targets",
        choices=("pose", "perturbations", "boundary"),
        default="pose",
    )
    parser.add_argument("--stability-seeds", type=int, nargs="+")
    parser.add_argument("--stability-iterations", type=int, nargs="+")
    parser.add_argument("--stability-initial-guesses", type=int, default=0)
    parser.add_argument("--candidate-diversity", action="store_true")
    parser.add_argument("--candidate-angular-tolerance-rad", type=float, default=1e-3)
    parser.add_argument("--candidate-linear-tolerance-m", type=float, default=1e-4)
    parser.add_argument("--max-ik-disagreement", type=float, default=None)
    parser.add_argument(
        "--select-ik-solutions",
        choices=("joint_limit_margin", "isotropy", "minimum_singular_value"),
        help="select quality-first candidates from the stability trials",
    )
    parser.add_argument(
        "--min-selected-quality-rate",
        type=float,
        default=None,
        help="optional quality acceptance fraction over ALL selected study targets",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/reachability_workflow")
    )
    args = parser.parse_args(argv)
    try:
        report = run_workflow(args)
    except (ValueError, OSError, ImportError) as error:
        parser.error(str(error))
    for name, case in report["cases"].items():
        summary = case["assessment"]
        print(
            f"{name}: IK {summary['ik_success_count']}/{summary['samples']}, "
            f"quality {summary['quality_pass_count']}/{summary['samples']}"
        )
    print(
        f"{'PASS' if report['passed'] else 'FAIL'}: {args.output_dir / 'report.json'}"
    )
    return 0 if report["passed"] else 1


def run_workflow(args) -> dict:
    if args.samples < 1:
        raise ValueError("samples must be positive")
    if not np.isfinite(args.joint_fraction) or not 0 < args.joint_fraction <= 0.5:
        raise ValueError("joint-fraction must lie in (0, 0.5]")
    if (
        not np.isfinite(args.orientation_offset_deg)
        or not 0 < args.orientation_offset_deg < 180
    ):
        raise ValueError("orientation-offset-deg must lie in (0, 180)")
    if not np.isfinite(args.min_known_success) or not 0 <= args.min_known_success <= 1:
        raise ValueError("min-known-success must lie in [0, 1]")
    if args.min_quality_rate is not None and (
        not np.isfinite(args.min_quality_rate) or not 0 <= args.min_quality_rate <= 1
    ):
        raise ValueError("min-quality-rate must lie in [0, 1]")
    if args.min_robust_rate is not None:
        if not args.robustness_test:
            raise ValueError("min-robust-rate requires --robustness-test")
        if not np.isfinite(args.min_robust_rate) or not 0 <= args.min_robust_rate <= 1:
            raise ValueError("min-robust-rate must lie in [0, 1]")
    boundary_config = None
    if args.boundary_test:
        boundary_config = BoundaryConfig(
            tolerance_m=args.boundary_tolerance_m, max_rounds=args.boundary_max_rounds
        )
        if not np.isfinite(args.boundary_distance_m) or args.boundary_distance_m <= 0:
            raise ValueError("boundary-distance-m must be finite and positive")
    if args.stability_initial_guesses < 0:
        raise ValueError("stability-initial-guesses must be non-negative")
    if args.stability_initial_guesses and not args.stability_test:
        raise ValueError("stability-initial-guesses requires --stability-test")
    if args.candidate_diversity and not args.stability_test:
        raise ValueError("candidate-diversity requires --stability-test")
    diversity_config = IKDiversityConfig(
        angular_tolerance_rad=args.candidate_angular_tolerance_rad,
        linear_tolerance_m=args.candidate_linear_tolerance_m,
    )
    trials = None
    if args.stability_test:
        if args.stability_targets == "boundary" and not args.boundary_test:
            raise ValueError("boundary stability targets require --boundary-test")
        if args.stability_targets == "perturbations" and not args.robustness_test:
            raise ValueError("perturbation stability targets require --robustness-test")
        trial_seeds = list(
            dict.fromkeys(args.stability_seeds or [args.seed, args.seed + 1])
        )
        budgets = list(
            dict.fromkeys(
                args.stability_iterations
                or [args.max_iterations, 2 * args.max_iterations]
            )
        )
        trials = [
            IKTrial(
                f"iterations_{budget}_seed_{seed}",
                max_iterations=budget,
                random_seed=seed,
            )
            for budget in budgets
            for seed in trial_seeds
        ]
        if len(trials) < 2 and not args.stability_initial_guesses:
            raise ValueError(
                "stability requires at least two distinct budget/seed combinations"
            )
    if args.max_ik_disagreement is not None:
        if not args.stability_test:
            raise ValueError("max-ik-disagreement requires --stability-test")
        if (
            not np.isfinite(args.max_ik_disagreement)
            or not 0 <= args.max_ik_disagreement <= 1
        ):
            raise ValueError("max-ik-disagreement must lie in [0, 1]")
    if args.select_ik_solutions is not None and not args.stability_test:
        raise ValueError("select-ik-solutions requires --stability-test")
    if args.min_selected_quality_rate is not None:
        if args.select_ik_solutions is None:
            raise ValueError("min-selected-quality-rate requires --select-ik-solutions")
        if (
            not np.isfinite(args.min_selected_quality_rate)
            or not 0 <= args.min_selected_quality_rate <= 1
        ):
            raise ValueError("min-selected-quality-rate must lie in [0, 1]")
    settings = ReachabilityConfig(
        batch_size=args.batch_size,
        random_seed=args.seed,
        restarts=args.restarts,
        rescue_restarts=args.rescue_restarts,
        rescue_rounds=args.rescue_rounds,
        minimum_isotropy=args.min_isotropy,
        minimum_joint_limit_margin=args.min_joint_limit_margin,
    )
    solver = create_solver(
        args.urdf,
        base_link=args.base_link,
        tip_link=args.tip_link,
        backend=args.backend,
        device=args.device,
        dtype=args.dtype,
        max_iterations=args.max_iterations,
        random_seed=args.seed,
    )
    if args.stability_initial_guesses:
        initial_guesses = make_ik_seed_trials(
            solver,
            args.stability_initial_guesses,
            random_seed=args.seed,
        )
        trials += [
            replace(trial, name=f"{trial.name}_{guess.name}", seed=guess.seed)
            for trial in trials
            for guess in initial_guesses
        ]
    rng = np.random.default_rng(args.seed)
    limits = solver.joint_limits
    known_q = limits.mean(1) + rng.uniform(
        -args.joint_fraction, args.joint_fraction, (args.samples, solver.dof)
    ) * (limits[:, 1] - limits[:, 0])
    known = _numpy(solver.forward(known_q))
    perturbations = (
        PosePerturbations(
            known,
            translation_m=args.perturb_translation_m,
            rotation_rad=np.deg2rad(args.perturb_rotation_deg),
        )
        if args.robustness_test
        else None
    )
    if args.min_robust_rate is not None and len(perturbations.variant_names) == 1:
        raise ValueError("min-robust-rate requires at least one nonzero perturbation")
    far, radius = _far_targets(solver)
    position_targets = np.concatenate((known[:, :3, 3], far[:, :3, 3]))
    pose_targets = np.concatenate((known, far))
    local_targets = _orientation_targets(known, args.orientation_offset_deg)
    position_ids = np.repeat(np.arange(args.samples), 3)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "targets.npz",
        targets=local_targets,
        position_ids=position_ids,
        known_joint_positions=known_q,
        known_poses=known,
        far_poses=far,
    )
    analyzer = WorkspaceAnalyzer(solver)
    cache = ResultCache(output / "cache")
    cases = {}
    outcomes = {}
    specifications = [
        ("position", position_targets, settings),
        ("pose", pose_targets, replace(settings, position_only=False)),
        ("local_orientations", local_targets, replace(settings, position_only=False)),
    ]
    if perturbations is not None:
        np.savez_compressed(
            output / "perturbation_targets.npz",
            targets=perturbations.targets,
            reference_poses=perturbations.reference_poses,
            variant_names=np.asarray(perturbations.variant_names),
        )
        specifications.append(
            (
                "perturbations",
                perturbations.targets,
                replace(settings, position_only=False),
            )
        )
    for name, targets, config in specifications:
        result = analyzer.analyze_targets(targets, config, cache=cache)
        result.save(output / f"{name}.npz")
        outcomes[name] = result
        cases[name] = result.metadata
    coverage = outcomes["local_orientations"].orientation_coverage(position_ids)
    checks = {}
    for name in ("position", "pose"):
        flags = outcomes[name].reachable
        checks[f"{name}_known_targets"] = {
            "observed_success_rate": float(flags[: args.samples].mean()),
            "required_success_rate": args.min_known_success,
            "passed": bool(flags[: args.samples].mean() >= args.min_known_success),
        }
        checks[f"{name}_far_targets"] = {
            "rejected": int(np.count_nonzero(~flags[args.samples :])),
            "expected_rejections": len(far),
            "passed": bool(not flags[args.samples :].any()),
        }
        if args.min_quality_rate is not None:
            quality_rate = float(
                outcomes[name].metrics["quality_pass"][: args.samples].mean()
            )
            checks[f"{name}_known_quality"] = {
                "observed_quality_rate": quality_rate,
                "required_quality_rate": args.min_quality_rate,
                "passed": quality_rate >= args.min_quality_rate,
            }
    base_orientations = outcomes["local_orientations"].reachable.reshape(
        args.samples, 3
    )[:, 0]
    checks["local_orientation_reference_targets"] = {
        "observed_success_rate": float(base_orientations.mean()),
        "required_success_rate": args.min_known_success,
        "passed": bool(base_orientations.mean() >= args.min_known_success),
    }
    robustness = None
    if perturbations is not None:
        robustness = perturbations.summarize(outcomes["perturbations"])
        ik = robustness["summary"]["ik"]
        reference_rate = ik["reference_success_count"] / args.samples
        checks["perturbation_reference_targets"] = {
            "observed_success_rate": reference_rate,
            "required_success_rate": args.min_known_success,
            "passed": reference_rate >= args.min_known_success,
        }
        if args.min_robust_rate is not None:
            rate = ik["all_variants_pass_rate"]
            checks["perturbation_robust_rate"] = {
                "observed_success_rate": rate,
                "required_success_rate": args.min_robust_rate,
                "passed": rate >= args.min_robust_rate,
            }
    boundary = None
    if boundary_config is not None:
        end_positions = known[:, :3, 3].copy()
        end_positions[:, 0] += args.boundary_distance_m
        refined = refine_translation_boundary(
            solver,
            known,
            end_positions,
            boundary_config,
            reachability=replace(settings, position_only=False),
            seed=outcomes["pose"].joint_positions[: args.samples],
            cache=cache,
        )
        refined.save(output / "boundary")
        boundary = refined.to_dict()
        summary = boundary["summary"]
        # An endpoint that succeeds does not supply a bracket to refine.
        # Report missing brackets explicitly rather than claiming a boundary.
        checks["boundary_brackets_found"] = {
            "observed_success_rate": summary["initial_brackets"] / args.samples,
            "required_success_rate": 1.0,
            "passed": summary["initial_brackets"] == args.samples,
        }
        checks["boundary_brackets_refined"] = {
            "observed_success_rate": summary["refined"] / args.samples,
            "required_success_rate": 1.0,
            "passed": summary["refined"] == args.samples,
        }
    stability = None
    if trials is not None:
        if args.stability_targets == "boundary":
            source_indices = np.column_stack(
                (refined.lower_indices, refined.upper_indices)
            ).ravel()
            study_targets = refined.measurements.target_poses[source_indices]
            source_file = "boundary/measurements.npz"
        else:
            study_targets = (
                pose_targets
                if args.stability_targets == "pose"
                else perturbations.targets
            )
            source_indices = np.arange(len(study_targets))
            source_file = (
                "pose.npz" if args.stability_targets == "pose" else "perturbations.npz"
            )
        study = analyze_ik_stability(
            solver,
            study_targets,
            trials,
            replace(settings, position_only=False),
            cache=cache,
        )
        study.save(output / "stability")
        stability = study.to_dict()
        stability["directory"] = "stability"
        stability["target_source"] = {
            "case": args.stability_targets,
            "result_file": source_file,
            "row_indices": source_indices.tolist(),
        }
        stability["diversity"] = None
        if args.candidate_diversity:
            diversity = study.summarize_diversity(diversity_config)
            (output / "stability/diversity.json").write_text(
                json.dumps(diversity, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            stability["diversity"] = diversity
        stability["selection"] = None
        if args.select_ik_solutions is not None:
            selected = study.select_solutions(objective=args.select_ik_solutions)
            selected.metadata["candidate_selection"]["source_study_directory"] = "."
            selected.save(output / "stability/selected.npz")
            stability["selection"] = {
                "file": "selected.npz",
                "metadata": selected.metadata,
            }
            if args.min_selected_quality_rate is not None:
                rate = selected.metadata["assessment"]["quality_pass_rate"]
                checks["selected_candidate_quality"] = {
                    "observed_quality_rate": rate,
                    "required_quality_rate": args.min_selected_quality_rate,
                    "passed": rate >= args.min_selected_quality_rate,
                }
        if args.max_ik_disagreement is not None:
            rate = stability["summary"]["ik_disagreement_rate"]
            checks["ik_stability_disagreement"] = {
                "observed_disagreement_rate": rate,
                "maximum_disagreement_rate": args.max_ik_disagreement,
                "passed": rate <= args.max_ik_disagreement,
            }
    sweep = []
    for margin in sorted({0.0, args.min_joint_limit_margin, 0.2}):
        reassessed = outcomes["local_orientations"].reassess_quality(
            minimum_isotropy=args.min_isotropy,
            minimum_joint_limit_margin=margin,
        )
        sweep.append(
            {
                "minimum_joint_limit_margin": margin,
                "assessment": reassessed.metadata["assessment"],
                "coverage": reassessed.orientation_coverage(position_ids).summary(),
            }
        )
    report = {
        "workflow_version": 6,
        "configuration": {
            name: str(value) if isinstance(value, Path) else value
            for name, value in vars(args).items()
        },
        "robot": solver.model.name,
        "base_link": solver.base_link,
        "tip_link": solver.tip_link,
        "passed": all(check["passed"] for check in checks.values()),
        "checks": checks,
        "cases": cases,
        "far_target_radius_m": radius,
        "orientation_offsets_deg": [
            0.0,
            args.orientation_offset_deg,
            -args.orientation_offset_deg,
        ],
        "orientation_coverage": coverage.to_dict(),
        "quality_sweep": sweep,
        "perturbation_robustness": robustness,
        "translation_boundary": boundary,
        "ik_stability": stability,
        "scope": (
            "FK-generated targets, local tool-X orientations, and optional "
            "independent axis perturbations, translation brackets, "
            "IK stability trials, configuration diversity, and candidate selection; "
            "no collision checks"
        ),
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output / "report.md").write_text(_markdown(report), encoding="utf-8")
    return report


def _far_targets(solver):
    # Triangle inequality bounds the base-to-TCP distance of a serial chain.
    # Joint rotations preserve length; bounded prismatic translations add at
    # most the largest absolute joint limit to each joint's origin translation.
    bound = sum(
        float(np.linalg.norm(joint.origin[:3, 3]))
        + (
            max(abs(joint.limit.lower), abs(joint.limit.upper))
            if joint.kind == "prismatic"
            else 0.0
        )
        for joint in solver.chain
    )
    radius = 2 * bound + 1.0
    targets = np.broadcast_to(np.eye(4, dtype=solver.config.dtype), (6, 4, 4)).copy()
    targets[:, :3, 3] = np.concatenate((np.eye(3), -np.eye(3))) * radius
    return targets, radius


def _orientation_targets(known, offset_deg):
    offsets = np.broadcast_to(np.eye(3, dtype=known.dtype), (3, 3, 3)).copy()
    for index, angle in ((1, np.deg2rad(offset_deg)), (2, -np.deg2rad(offset_deg))):
        c, s = np.cos(angle), np.sin(angle)
        offsets[index] = [[1, 0, 0], [0, c, -s], [0, s, c]]
    targets = np.repeat(known, 3, axis=0)
    targets[:, :3, :3] = (known[:, None, :3, :3] @ offsets[None]).reshape(-1, 3, 3)
    return targets


def _markdown(report):
    lines = [
        "# Reachability workflow",
        "",
        f"Status: **{'PASS' if report['passed'] else 'FAIL'}**",
        "",
        f"Robot: {report['robot']} ({report['base_link']} → {report['tip_link']})",
        "",
        "| Case | Pose/point samples | IK successes | Quality accepted |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, case in report["cases"].items():
        a = case["assessment"]
        lines.append(
            f"| {name} | {a['samples']} | {a['ik_success_count']} "
            f"| {a['quality_pass_count']} |"
        )
    lines.extend(
        [
            "",
            "## Checks",
            "",
            "| Check | Observed | Required | Status |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for name, check in report["checks"].items():
        if "observed_disagreement_rate" in check:
            observed = f"{check['observed_disagreement_rate']:.2%}"
            required = f"<= {check['maximum_disagreement_rate']:.2%}"
        elif "rejected" in check:
            observed, required = (
                str(check["rejected"]),
                str(check["expected_rejections"]),
            )
        else:
            key = "quality_rate" if "observed_quality_rate" in check else "success_rate"
            observed = f"{check[f'observed_{key}']:.2%}"
            required = f"{check[f'required_{key}']:.2%}"
        lines.append(
            f"| {name} | {observed} | {required} | "
            f"{'PASS' if check['passed'] else 'FAIL'} |"
        )
    robustness = report["perturbation_robustness"]
    if robustness is not None:
        summary = robustness["summary"]
        lines.extend(
            [
                "",
                "## Paired perturbations",
                "",
                f"References: {summary['references']}; variants per reference: "
                f"{summary['variants_per_reference']} (including reference).",
                "",
                "| Variant | IK successes | IK lost / gained | "
                "Quality accepted | Quality lost / gained |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in robustness["by_variant"]:
            lines.append(
                f"| {row['name']} | {row['ik_success_count']} | "
                f"{row['ik_lost_from_reference']} / "
                f"{row['ik_gained_from_reference']} | {row['quality_success_count']} | "
                f"{row['quality_lost_from_reference']} / "
                f"{row['quality_gained_from_reference']} |"
            )
        lines.extend(
            [
                "",
                "Lost/gained counts compare each variant with its own reference.",
                "Single-axis samples do not represent "
                "an error probability distribution.",
            ]
        )
    boundary = report["translation_boundary"]
    if boundary is not None:
        summary = boundary["summary"]
        lines.extend(
            [
                "",
                "## Translation boundary",
                "",
                f"Observed brackets: {summary['initial_brackets']}; "
                f"refined: {summary['refined']}; target evaluations: "
                f"{summary['target_evaluations']}.",
                "",
                "| Segment | Status | Rounds | Lower offset (m) | "
                "Upper offset (m) | Width (m) |",
                "| ---: | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in boundary["segments"]:
            lines.append(
                f"| {row['segment_id']} | {row['status']} | {row['rounds']} | "
                f"{row['lower']['offset_m']:.6f} | {row['upper']['offset_m']:.6f} | "
                f"{row['width_m']:.6f} |"
            )
        lines.extend(
            [
                "",
                "Intervals describe observed IK classifications, "
                "not proven geometric boundaries.",
            ]
        )
    stability = report["ik_stability"]
    if stability is not None:
        summary = stability["summary"]
        lines.extend(
            [
                "",
                "## IK stability",
                "",
                f"Target set: {stability['target_source']['case']}; "
                f"{summary['targets']} targets, {summary['trials']} trials.",
                f"IK disagreement: {summary['ik_disagreement_count']} targets; "
                f"quality disagreement: "
                f"{summary['quality_disagreement_count']} targets.",
                "",
                "| Trial | Iterations | Random seed | "
                "IK successes | Quality accepted |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for trial in stability["trials"]:
            m = trial["metadata"]
            lines.append(
                f"| {trial['name']} | {m['solver_settings']['max_iterations']} | "
                f"{m['random_seed']} | {m['assessment']['ik_success_count']} | "
                f"{m['assessment']['quality_pass_count']} |"
            )
        lines.extend(
            [
                "",
                "Agreement across trials does not establish geometric "
                "reachability or a physical success probability.",
            ]
        )
        diversity = stability.get("diversity")
        if diversity is not None:
            summary = diversity["summary"]
            lines.extend(
                [
                    "",
                    "## Observed configuration diversity",
                    "",
                    f"IK candidates: {summary['ik_candidate_count']}; "
                    "configurations at the specified tolerances: "
                    f"{summary['configuration_count']}; "
                    f"duplicate candidates: {summary['duplicate_candidate_count']}.",
                    "",
                    "| Trial | New configurations | Duplicates | New quality groups |",
                    "| --- | ---: | ---: | ---: |",
                ]
            )
            for row in diversity["by_trial"]:
                lines.append(
                    f"| {row['name']} | {row['new_configuration_count']} | "
                    f"{row['duplicate_candidate_count']} | "
                    f"{row['new_quality_configuration_count']} |"
                )
            lines.extend(
                [
                    "",
                    "Groups depend on trial order and joint tolerances. "
                    "They do not enumerate IK branches or discard measured candidates.",
                ]
            )
        selection = stability.get("selection")
        if selection is not None:
            metadata = selection["metadata"]
            audit = metadata["candidate_selection"]
            summary = audit["selection_time_summary"]
            assessment = metadata["assessment"]
            lines.extend(
                [
                    "",
                    "## Observed candidate selection",
                    "",
                    f"Objective: {audit['objective']}; quality gates take priority.",
                    f"IK successes: {assessment['ik_success_count']}"
                    f"/{assessment['samples']}; quality accepted: "
                    f"{assessment['quality_pass_count']}/{assessment['samples']}.",
                    f"Compared with the first trial: IK gained/lost "
                    f"{summary['ik_gained_count']}/{summary['ik_lost_count']}; "
                    "quality gained/lost "
                    f"{summary['quality_gained_count']}/{summary['quality_lost_count']}.",
                    "",
                    "Each selected row retains its source trial. "
                    "Independent target choices "
                    "do not establish a continuous or collision-free path.",
                ]
            )
    lines.extend(
        [
            "",
            "Coverage measures the supplied local orientation samples.",
            "Quality gates evaluate selected IK solutions. "
            "Collision checking is not included.",
            "A failed known-target check means the configured numerical search "
            "did not meet its acceptance rate.",
            "",
        ]
    )
    return "\n".join(lines)


def _numpy(value):
    return (
        value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    )


if __name__ == "__main__":
    raise SystemExit(main())
