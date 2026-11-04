"""Resolve export destinations before a task runs or opens a viewer."""

from pathlib import Path


def validate_output_paths(*, output=None, report=None, plot=None, inputs=()):
    # AnalysisResult.save replaces the suffix, rather than appending to it.
    destinations = validate_artifact_paths(
        {
            "output": None if output is None else Path(output).with_suffix(".npz"),
            "report": report,
            "plot": plot,
        },
        inputs=inputs,
    )
    return destinations["output"], destinations["report"], destinations["plot"]


def validate_artifact_paths(paths, *, inputs=()):
    """Validate any named set of artifacts without changing their extensions."""
    destinations = {
        name: None if path is None else Path(path) for name, path in paths.items()
    }
    occupied = [("input", Path(path)) for path in inputs if path is not None]
    try:
        for name, path in destinations.items():
            if path is None:
                continue
            if path.is_dir():
                raise ValueError(f"{name} destination is a directory: {path}")
            resolved = path.resolve()
            for other_name, other in occupied:
                if resolved == other.resolve() or (
                    path.exists() and other.exists() and path.samefile(other)
                ):
                    raise ValueError(
                        f"{name} and {other_name} must use distinct paths: {path}"
                    )
            occupied.append((name, path))
    except RuntimeError as exc:
        raise ValueError(f"cannot resolve artifact path: {exc}") from exc
    return destinations
