"""Dependency-free SVG plot of measured plane targets, in the sampling frame."""

from html import escape

import numpy as np

from ._atomic import atomic_write_text
from ._validation import require_integer


def _preview_indices(reachable, quality, max_points):
    """Retain each outcome category when limiting a browser preview."""
    count = len(reachable)
    if max_points is None:
        return np.arange(count)
    require_integer(max_points, "max_points", minimum=3)
    if count <= max_points:
        return np.arange(count)
    groups = [
        np.flatnonzero(mask)
        for mask in (
            reachable & quality,
            reachable & ~quality,
            ~reachable,
        )
        if np.any(mask)
    ]
    # Reserve one point per present category; distribute the rest by population.
    remaining = max_points - len(groups)
    capacity = np.array([len(group) - 1 for group in groups])
    shares = remaining * capacity / capacity.sum()
    allocation = np.floor(shares).astype(int)
    extra = remaining - allocation.sum()
    allocation[np.argsort(-(shares - allocation), kind="stable")[:extra]] += 1
    return np.sort(
        np.concatenate(
            [
                group[np.linspace(0, len(group) - 1, int(amount) + 1, dtype=int)]
                for group, amount in zip(groups, allocation, strict=True)
            ]
        )
    )


def plane_svg(result, *, max_points=None):
    """Plot all measured targets, or a labeled, category-preserving preview."""
    report = result.metadata["plane_scan"]
    axes = ["xyz".index(report[name]) for name in ("row_axis", "column_axis")]
    transform = np.asarray(report["base_from_plane_frame"])
    reachable = np.asarray(result.reachable, dtype=bool)
    quality = np.asarray(
        (result.metrics or {}).get("quality_pass", reachable), dtype=bool
    )
    indices = _preview_indices(reachable, quality, max_points)
    local = (result.points[indices] - transform[:3, 3]) @ transform[:3, :3]
    truncated = len(indices) < len(result.points)
    height = 594 if truncated else 570
    bounds = np.asarray(report["sampling_bounds_m"])[axes]
    plot_x, plot_y, size = 64, 56, 420
    uv = (local[:, axes] - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    header = f"{report['plane'].upper()} · {report['coordinate_frame']}"
    if any(report.get("frame_rpy_rad", ())):
        header += " · rotated frame"
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 560 {height}" role="img" '
        'aria-label="Measured plane reachability">',
        f'<rect width="560" height="{height}" fill="white"/>',
        '<g font-family="sans-serif" font-size="13" fill="#203047">',
        f'<text x="64" y="24" font-size="19">{escape(header)}</text>',
        f'<text x="64" y="44">IK solved: {int(result.reachable.sum())}/'
        f"{len(result.points)}</text>",
        f'<rect x="{plot_x}" y="{plot_y}" width="{size}" height="{size}" '
        'fill="#f3f5f7" stroke="#aab4c2"/>',
    ]
    radius = max(1.2, min(5.5, size / max(np.sqrt(len(local)), 2) * 0.18))
    for displayed, (index, (u, v)) in enumerate(zip(indices, uv, strict=True)):
        solved = bool(result.reachable[index])
        accepted = solved and bool(quality[index])
        color = "#16846a" if accepted else "#d99b18" if solved else "#ce4653"
        state = (
            "quality accepted"
            if accepted
            else "below quality gates"
            if solved
            else "IK not solved"
        )
        title = (
            f"Target {index}: {local[displayed, axes[0]]:.6g}, "
            f"{local[displayed, axes[1]]:.6g} m; {state}; "
            f"residual {result.residual[index]:.4g}"
        )
        lines.append(
            f'<circle data-target-index="{index}" cx="{plot_x + u * size:.3f}" '
            f'cy="{plot_y + (1 - v) * size:.3f}" '
            f'r="{radius:.3f}" '
            f'fill="{color}"><title>{escape(title)}</title></circle>'
        )
    for fraction in (0, 0.25, 0.5, 0.75, 1):
        x = bounds[0, 0] + fraction * (bounds[0, 1] - bounds[0, 0])
        y = bounds[1, 0] + fraction * (bounds[1, 1] - bounds[1, 0])
        lines.append(
            f'<text x="{plot_x + fraction * size:.1f}" y="494" text-anchor="middle">'
            f"{x:.3g}</text>"
        )
        lines.append(
            f'<text x="55" y="{plot_y + (1 - fraction) * size + 4:.1f}" '
            f'text-anchor="end">{y:.3g}</text>'
        )
    lines.extend(
        [
            f'<text x="274" y="516" text-anchor="middle">'
            f"{report['row_axis'].upper()} (m)</text>",
            '<text x="17" y="266" text-anchor="middle" '
            'transform="rotate(-90 17 266)">'
            f"{report['column_axis'].upper()} (m)</text>",
            '<text x="64" y="538" fill="#16846a">● Quality accepted</text>',
            '<text x="270" y="538" fill="#d99b18">● Below quality gates</text>',
            '<text x="64" y="558" fill="#ce4653">● IK not solved</text>',
            '<text x="270" y="558">Sampled points only</text>',
        ]
    )
    if truncated:
        lines.append(
            f'<text x="64" y="582">Preview: {len(indices):,}/{len(result.points):,} '
            "targets; download SVG for all.</text>"
        )
    lines.append("</g></svg>")
    return "\n".join(lines)


def save_plane_svg(result, path):
    atomic_write_text(path, plane_svg(result))
