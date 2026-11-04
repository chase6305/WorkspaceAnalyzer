"""Shared summaries and independent, atomic exports of measured results."""

import json

from ._atomic import atomic_write_text
from .plane_plot import save_plane_svg


def result_summary(result):
    """Summarize measured rows, retaining complete analysis metadata."""
    count = len(result.points)
    return {
        "samples": count,
        "dof": result.joint_positions.shape[1],
        "bounds": None
        if count == 0
        else [result.points.min(0).tolist(), result.points.max(0).tolist()],
        "reachable": None if result.reachable is None else int(result.reachable.sum()),
        "success_rate": None
        if result.reachable is None or not count
        else float(result.reachable.mean()),
        "metadata": result.metadata,
    }


def save_result_artifacts(result, paths, *, report_format="summary"):
    """Attempt all requested npz/json/svg exports; return saved paths and errors.

    Callers validate destinations first. Files are replaced atomically and
    independently; failure of one writer does not discard other diagnostics.
    Plane and targets report formats preserve the existing CLI JSON schemas.
    """
    if report_format not in {"summary", "metadata", "plane"}:
        raise ValueError("unknown report format")

    def save_report(path):
        if report_format == "summary":
            data = result_summary(result)
        elif report_format == "plane":
            data = result.metadata["plane_scan"]
        else:
            data = result.metadata
        atomic_write_text(path, json.dumps(data, indent=2, allow_nan=False) + "\n")

    writers = {
        "npz": result.save,
        "json": save_report,
        "svg": lambda path: save_plane_svg(result, path),
    }
    saved, errors = {}, {}
    for kind, path in paths.items():
        try:
            writers[kind](path)
        except Exception as exc:
            errors[kind] = str(exc)
        else:
            saved[kind] = path
    return saved, errors
