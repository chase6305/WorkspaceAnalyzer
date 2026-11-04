"""Inspect and reassess saved measurements without loading a robot or running IK."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ._output_paths import validate_output_paths
from .acceptance import AcceptanceConfig, add_acceptance_arguments, reassess_acceptance
from .analyzer import AnalysisResult
from .result_export import result_summary, save_result_artifacts


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, prog="workspace-analyzer inspect"
    )
    parser.add_argument("result", type=Path, help="saved AnalysisResult .npz")
    parser.add_argument(
        "--report", type=Path, help="write summary and full metadata as JSON"
    )
    parser.add_argument("--plot", type=Path, help="write all plane samples as SVG")
    parser.add_argument(
        "--output", type=Path, help="save the inspected/reassessed result as NPZ"
    )
    parser.add_argument("--min-singular-value", type=float)
    parser.add_argument("--min-isotropy", type=float)
    parser.add_argument("--min-joint-limit-margin", type=float)
    rank = parser.add_mutually_exclusive_group()
    rank.add_argument("--require-full-rank", dest="full_rank", action="store_true")
    rank.add_argument("--allow-rank-deficient", dest="full_rank", action="store_false")
    parser.set_defaults(full_rank=None)
    parser.add_argument(
        "--reset-quality-gates",
        action="store_true",
        help="clear saved gates before applying overrides; otherwise preserve them",
    )
    add_acceptance_arguments(parser)
    parser.add_argument(
        "--reset-acceptance",
        action="store_true",
        help="clear saved run acceptance before applying overrides; keep quality gates",
    )
    return parser


def inspect_result(result, *, overrides=None, reset_quality_gates=False):
    """Keep original gates unless explicitly overridden or reset; never run IK."""
    if not overrides and not reset_quality_gates:
        return result
    gates = (
        {}
        if reset_quality_gates
        else dict(result.metadata.get("quality_thresholds", {}))
    )
    gates.update(overrides or {})
    return result.reassess_quality(**gates)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        acceptance = AcceptanceConfig(args.min_success_rate, args.min_quality_pass_rate)
        args.output, args.report, args.plot = validate_output_paths(
            output=args.output,
            report=args.report,
            plot=args.plot,
            inputs=(args.result,),
        )
        if args.plot is not None and args.plot.suffix.lower() != ".svg":
            raise ValueError("--plot must use an .svg path")
        result = AnalysisResult.load(args.result)
        if args.plot is not None and "plane_scan" not in result.metadata:
            raise ValueError("--plot requires a saved plane analysis")
        overrides = {
            name: value
            for name, value in (
                ("minimum_singular_value", args.min_singular_value),
                ("minimum_isotropy", args.min_isotropy),
                ("minimum_joint_limit_margin", args.min_joint_limit_margin),
                ("require_full_rank", args.full_rank),
            )
            if value is not None
        }
        if args.reset_acceptance:
            result = reassess_acceptance(result, reset=True)
        result = inspect_result(
            result, overrides=overrides, reset_quality_gates=args.reset_quality_gates
        )
        # Quality reassessment already refreshes saved acceptance requirements.
        if acceptance.enabled or not (overrides or args.reset_quality_gates):
            result = reassess_acceptance(result, acceptance)
        summary = result_summary(result)
        text = json.dumps(summary, indent=2, allow_nan=False) + "\n"
        paths = {
            kind: path
            for kind, path in (
                ("npz", args.output),
                ("json", args.report),
                ("svg", args.plot),
            )
            if path is not None
        }
        saved, errors = save_result_artifacts(result, paths)
        print(text, end="")
        if errors:
            details = "; ".join(
                f"{paths[kind]}: {error}" for kind, error in errors.items()
            )
            raise ValueError(
                "export failed: "
                + details
                + (
                    "; saved: " + ", ".join(str(path) for path in saved.values())
                    if saved
                    else ""
                )
            )
        if result.metadata.get("acceptance", {}).get("passed") is False:
            raise SystemExit(1)
        return result
    except (ValueError, OSError, TypeError, KeyError) as exc:
        parser.error(str(exc))
