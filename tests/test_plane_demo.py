import json
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

from workspace_analyzer import (
    AnalysisCancelled,
    AnalysisResult,
    PlaneConfig,
    ReachabilityConfig,
    ResultCache,
    analyze_plane,
    create_solver,
)
from workspace_analyzer.plane import plane_frame

EXAMPLE = Path(__file__).parents[1] / "examples" / "plane_reachability.py"

RZ = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
RX = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
RY = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])


def _transform(rotation, translation):
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


@pytest.fixture
def mounted_solver(tmp_path):
    urdf = tmp_path / "mounted.urdf"
    urdf.write_text("""<robot name="mounted">
      <link name="world"/><link name="left_arm_base"/>
      <link name="joint1"/><link name="joint2"/><link name="left_ee"/>
      <joint name="mount" type="fixed">
        <parent link="world"/><child link="left_arm_base"/>
        <origin xyz="0.2 -0.3 0.5" rpy="0 0 1.5707963267948966"/>
      </joint>
      <joint name="first" type="revolute">
        <parent link="left_arm_base"/><child link="joint1"/>
        <origin xyz="0.1 0.2 0.3" rpy="1.5707963267948966 0 0"/>
        <axis xyz="0 0 1"/><limit lower="-3" upper="3"/>
      </joint>
      <joint name="second" type="revolute">
        <parent link="joint1"/><child link="joint2"/>
        <origin xyz="0.4 -0.1 0.2" rpy="0 1.5707963267948966 0"/>
        <axis xyz="0 0 1"/><limit lower="-3" upper="3"/>
      </joint>
      <joint name="tip" type="fixed">
        <parent link="joint2"/><child link="left_ee"/>
        <origin xyz="0.3 0 0"/>
      </joint>
    </robot>""")
    return create_solver(
        urdf, base_link="left_arm_base", tip_link="left_ee", backend="numpy"
    )


@pytest.mark.parametrize("frame", ["reference", "base", "world", "joint1"])
@pytest.mark.parametrize(
    "plane,axes", [("xy", (0, 1, 2)), ("xz", (0, 2, 1)), ("yz", (1, 2, 0))]
)
def test_real_scan_preserves_frame_grid_pose_and_report(
    frame, plane, axes, mounted_solver
):
    config = PlaneConfig(
        plane=plane,
        frame=frame,
        center=(0.7, 0.8, 0.9),
        constant=0.6,
        span=0.4,
        resolution=3,
        rpy=(0, 0, np.pi / 2),
    )
    result = analyze_plane(
        mounted_solver,
        config,
        ReachabilityConfig(
            position_only=False, batch_size=4, restarts=1, rescue_restarts=0
        ),
    )
    world_from_base = _transform(RZ, [0.2, -0.3, 0.5])
    first = _transform(RX, [0.1, 0.2, 0.3])
    second = _transform(RY, [0.4, -0.1, 0.2])
    world_from_frame = {
        "world": np.eye(4),
        "base": world_from_base,
        "reference": world_from_base @ first @ second,
        "joint1": world_from_base @ first,
    }[frame]
    expected_points = np.tile([0.7, 0.8, 0.9], (9, 1))
    expected_points[:, axes[:2]] += [
        (a, b) for a in (-0.2, 0, 0.2) for b in (-0.2, 0, 0.2)
    ]
    expected_points[:, axes[2]] = 0.6
    np.testing.assert_allclose(
        result.points @ RZ.T + world_from_base[:3, 3],
        expected_points @ world_from_frame[:3, :3].T + world_from_frame[:3, 3],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        RZ @ result.target_poses[:, :3, :3],
        np.broadcast_to(world_from_frame[:3, :3] @ RZ, (9, 3, 3)),
        atol=1e-12,
    )
    report = result.metadata["plane_scan"]
    np.testing.assert_allclose(
        world_from_base @ np.array(report["base_from_plane_frame"]),
        world_from_frame,
        atol=1e-12,
    )
    np.testing.assert_array_equal(np.array(report["grid"]).ravel(), result.reachable)
    assert report["row_axis"] == "xyz"[axes[0]]
    assert report["column_axis"] == "xyz"[axes[1]]
    assert result.metadata["coordinate_frame"] == mounted_solver.base_link


@pytest.mark.parametrize("entry", ["module", "demo", "console"])
@pytest.mark.parametrize("plane", ["xy", "xz", "yz"])
def test_actual_entry_points_export_analytic_plane(entry, plane, tmp_path):
    urdf = Path(__file__).parent / "fixtures/cartesian_stage.urdf"
    output = tmp_path / f"{entry}-{plane}"
    if entry == "demo":
        command = [
            sys.executable,
            str(EXAMPLE),
            "--urdf",
            str(urdf),
            "--position-only",
            "--output",
            str(output.with_suffix(".json")),
        ]
    else:
        command = (
            [sys.executable, "-m", "workspace_analyzer"]
            if entry == "module"
            else [
                sys.executable,
                "-c",
                "import sys; from workspace_analyzer.cli import entrypoint; "
                "sys.exit(entrypoint())",
            ]
        )
        command += [
            str(urdf),
            "--mode",
            "plane",
            "--output",
            str(output.with_suffix(".npz")),
            "--report",
            str(output.with_suffix(".json")),
        ]
    completed = subprocess.run(
        command
        + [
            "--backend",
            "numpy",
            "--frame",
            "base",
            "--plane",
            plane,
            "--center",
            "0",
            "0",
            "0",
            "--constant",
            "0.4",
            "--span",
            "3",
            "--resolution",
            "5",
            "--batch-size",
            "7",
            "--ik-restarts",
            "1",
            "--ik-rescue-restarts",
            "0",
        ],
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr
    result = AnalysisResult.load(output.with_suffix(".npz"))
    report = json.loads(output.with_suffix(".json").read_text())
    expected = np.all(np.abs(result.points) <= 1, axis=1)
    assert expected.sum() == 9
    np.testing.assert_array_equal(result.reachable, expected)
    np.testing.assert_array_equal(np.array(report["grid"]).ravel(), expected)
    assert report["success_rate"] == pytest.approx(9 / 25)
    assert report["resolution"] == 5
    reached = create_solver(urdf, backend="numpy").forward(
        result.joint_positions[expected]
    )
    np.testing.assert_allclose(reached[:, :3, 3], result.points[expected], atol=1e-5)


@pytest.mark.parametrize(
    "options",
    [
        ["--resolution", "1"],
        ["--span", "0"],
        ["--span", "nan"],
        ["--constant", "inf"],
        ["--center", "nan", "0", "0"],
        ["--rpy", "0", "inf", "0"],
        ["--frame", "missing-link"],
        ["--batch-size", "0"],
        ["--seed", "-1"],
        ["--samples", "10", "--resolution", "3"],
        ["--samples", "0"],
        ["--plot", "wrong.png"],
        ["--strategy", "sobol"],
        ["--base-link", "missing-link"],
    ],
)
def test_plane_cli_rejects_invalid_or_ignored_arguments(options):
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "workspace_analyzer",
            str(Path(__file__).parent / "fixtures/cartesian_stage.urdf"),
            "--backend",
            "numpy",
            "--mode",
            "plane",
            *options,
        ],
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert completed.returncode == 2
    assert "error:" in completed.stderr
    assert "Traceback" not in completed.stderr


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_plane_cache_and_orientation_constraints(backend, tmp_path, monkeypatch):
    if backend == "torch":
        pytest.importorskip("torch")
    solver = create_solver(
        Path(__file__).parent / "fixtures/cartesian_stage.urdf",
        backend=backend,
        device="cpu",
        max_iterations=10,
    )
    plane = PlaneConfig(resolution=3, span=0.4, rpy=(0.2, 0, 0))
    cache = ResultCache(tmp_path)
    settings = ReachabilityConfig(batch_size=4, restarts=1, rescue_restarts=0)
    first = analyze_plane(solver, plane, settings, cache=cache)
    assert first.reachable.all()
    constrained = analyze_plane(
        solver,
        plane,
        ReachabilityConfig(position_only=False, restarts=1, rescue_restarts=0),
    )
    assert not constrained.reachable.any()
    monkeypatch.setattr(
        solver, "inverse", lambda *a, **k: pytest.fail("cache replay called IK")
    )
    replay = analyze_plane(solver, plane, settings, cache=cache)
    assert replay.metadata["cache_hit"]
    np.testing.assert_array_equal(replay.reachable, first.reachable)
    stricter = replay.reassess_quality(minimum_joint_limit_margin=0.9)
    assert stricter.metadata["assessment"]["quality_pass_count"] < len(first.points)


def test_pre_cancelled_plane_does_not_generate_targets(monkeypatch, mounted_solver):
    cancel = threading.Event()
    cancel.set()
    monkeypatch.setattr(
        PlaneConfig, "points", lambda _: pytest.fail("generated targets after cancel")
    )
    with pytest.raises(AnalysisCancelled):
        analyze_plane(mounted_solver, PlaneConfig(), cancel_event=cancel)


def test_reference_frame_supports_single_joint(mounted_solver):
    solver = create_solver(
        mounted_solver.model.source,
        base_link="left_arm_base",
        tip_link="joint1",
        backend="numpy",
    )
    name, transform = plane_frame(solver, "reference")
    assert name == "joint1"
    np.testing.assert_allclose(transform, _transform(RX, [0.1, 0.2, 0.3]), atol=1e-12)


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize(
    "robot,lower,upper,reference",
    [("marvin_elbow", -2.5, 1.1, -np.pi / 2), ("other_elbow", 0, 2.5, 1.25)],
)
def test_demo_keeps_reference_posture_and_allows_override(
    tmp_path, explicit, robot, lower, upper, reference
):
    urdf = tmp_path / "elbow.urdf"
    urdf.write_text(f"""<robot name="{robot}"><link name="base"/><link name="tcp"/>
      <joint name="ELBOW_YAW_L_J4" type="revolute">
        <parent link="base"/><child link="tcp"/><axis xyz="0 0 1"/>
        <limit lower="{lower}" upper="{upper}"/>
      </joint></robot>""")
    output = tmp_path / "elbow.json"
    command = [
        sys.executable,
        str(EXAMPLE),
        "--urdf",
        str(urdf),
        "--backend",
        "numpy",
        "--position-only",
        "--resolution",
        "2",
        "--ik-max-iterations",
        "1",
        "--ik-restarts",
        "1",
        "--ik-rescue-restarts",
        "0",
        "--output",
        str(output),
    ]
    if explicit:
        command += ["--reference-joints", "0.2"]
    completed = subprocess.run(command, text=True, capture_output=True, timeout=10)
    assert completed.returncode == 0, completed.stderr
    report = json.loads(output.read_text())
    assert report["reference_joints"] == pytest.approx([0.2 if explicit else reference])


@pytest.mark.parametrize(
    "strategy", ["random", "uniform", "halton", "sobol", "lhs", "gaussian"]
)
@pytest.mark.parametrize("plane", ["xy", "xz", "yz"])
def test_sampled_plane_count_bounds_reproducibility_and_real_ik(
    strategy, plane, mounted_solver
):
    from workspace_analyzer import SamplingConfig

    if strategy in {"halton", "sobol", "lhs"}:
        pytest.importorskip("scipy")
    config = PlaneConfig(
        plane=plane,
        frame="world",
        center=(0.2, -0.1, 0.3),
        span=0.8,
        sampling=SamplingConfig(strategy=strategy, num_samples=23, seed=17),
    )
    points = config.points()
    assert points.shape == (23, 3)
    np.testing.assert_array_equal(points, config.points())
    assert np.all(points >= config.bounds[:, 0])
    assert np.all(points <= config.bounds[:, 1])
    result = analyze_plane(
        mounted_solver, config, ReachabilityConfig(restarts=1, rescue_restarts=0)
    )
    report = result.metadata["plane_scan"]
    transform = np.array(report["base_from_plane_frame"])
    np.testing.assert_allclose(
        (result.points - transform[:3, 3]) @ transform[:3, :3], points, atol=1e-12
    )
    assert report["samples"] == config.num_samples == 23
    assert report["sampling_strategy"] == strategy
    assert report["sampling_seed"] == 17
    assert report["grid"] is None
    assert report["resolution"] is None


@pytest.mark.parametrize("entry", ["module", "demo"])
@pytest.mark.parametrize("sampled", [False, True])
def test_plane_defaults_keep_center_and_position_task(entry, sampled, tmp_path):
    urdf = Path(__file__).parent / "fixtures/cartesian_stage.urdf"
    output = tmp_path / "plane"
    command = [sys.executable]
    if entry == "demo":
        command += [
            str(EXAMPLE),
            "--urdf",
            str(urdf),
            "--output",
            str(output.with_suffix(".json")),
        ]
    else:
        # Selecting a plane infers mode without --mode plane.
        command += [
            "-m",
            "workspace_analyzer",
            str(urdf),
            "--output",
            str(output.with_suffix(".npz")),
            "--report",
            str(output.with_suffix(".json")),
            "--plot",
            str(output.with_suffix(".svg")),
        ]
    command += [
        "--plane",
        "xy",
        "--center",
        "0",
        "0",
        "0",
        "--span",
        "0.4",
        "--backend",
        "numpy",
    ]
    command += (
        ["--samples", "19", "--strategy", "random"]
        if sampled
        else ["--resolution", "3"]
    )
    completed = subprocess.run(command, text=True, capture_output=True, timeout=20)
    assert completed.returncode == 0, completed.stderr
    result = AnalysisResult.load(output.with_suffix(".npz"))
    report = json.loads(output.with_suffix(".json").read_text())
    assert report["plane_center"] == [0, 0, 0]
    assert report["frame"] == "base"
    assert report["task"] == "position_only"
    assert report["reachable"] == report["samples"] == (19 if sampled else 9)
    np.testing.assert_array_equal(result.points[:, 2], 0)
    assert output.with_suffix(".svg").is_file()


def test_plane_svg_matches_local_samples_and_quality_categories(mounted_solver):
    import xml.etree.ElementTree as ET

    from workspace_analyzer import SamplingConfig
    from workspace_analyzer.plane_plot import plane_svg

    result = analyze_plane(
        mounted_solver,
        PlaneConfig(frame="world", plane="yz", sampling=SamplingConfig(num_samples=3)),
        ReachabilityConfig(restarts=1, rescue_restarts=0),
    )
    # Isolate plotting categories from IK; numerical behavior is tested above.
    result.reachable[:] = [True, True, False]
    result.metrics["quality_pass"][:] = [True, False, False]
    result.metadata["plane_scan"]["coordinate_frame"] = 'frame <&"'
    root = ET.fromstring(plane_svg(result))
    circles = root.findall(".//{http://www.w3.org/2000/svg}circle")
    assert len(circles) == 3
    assert [circle.get("fill") for circle in circles] == [
        "#16846a",
        "#d99b18",
        "#ce4653",
    ]
    report = result.metadata["plane_scan"]
    t = np.asarray(report["base_from_plane_frame"])
    points = (result.points - t[:3, 3]) @ t[:3, :3]
    bounds = np.asarray(report["sampling_bounds_m"])
    for i, circle in enumerate(circles):
        assert float(circle.get("cx")) == pytest.approx(
            64 + (points[i, 1] - bounds[1, 0]) / report["span_m"] * 420, abs=0.001
        )
        assert float(circle.get("cy")) == pytest.approx(
            56 + (1 - (points[i, 2] - bounds[2, 0]) / report["span_m"]) * 420, abs=0.001
        )


@pytest.mark.parametrize("extension", ["json", "svg"])
@pytest.mark.parametrize("failure", ["write", "replace"])
def test_plane_text_export_failure_preserves_previous_artifact(
    extension, failure, tmp_path, monkeypatch
):
    import workspace_analyzer._atomic as atomic
    from workspace_analyzer.plane import save_plane_report
    from workspace_analyzer.plane_plot import save_plane_svg

    solver = create_solver(
        Path(__file__).parent / "fixtures/cartesian_stage.urdf", backend="numpy"
    )
    result = analyze_plane(
        solver,
        PlaneConfig(resolution=3),
        ReachabilityConfig(restarts=1, rescue_restarts=0),
    )
    path = tmp_path / f"plane.{extension}"
    writer = save_plane_report if extension == "json" else save_plane_svg
    writer(result, path)
    original = path.read_bytes()
    if failure == "replace":

        def fail_replace(*args):
            raise OSError("destination unavailable")

        monkeypatch.setattr(atomic.os, "replace", fail_replace)
    else:
        temporary_file = atomic.tempfile.NamedTemporaryFile

        class InterruptedWrite:
            def __init__(self, *args, **kwargs):
                self.stream = temporary_file(*args, **kwargs)
                self.name = self.stream.name

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.stream.close()

            def write(self, content):
                self.stream.write(content[:10])
                raise OSError("disk full during write")

        monkeypatch.setattr(atomic.tempfile, "NamedTemporaryFile", InterruptedWrite)
    with pytest.raises(OSError):
        writer(result, path)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("limit", [3, 5, 30, 100, None])
def test_plane_plot_preview_preserves_rare_outcomes_and_full_download(limit):
    import xml.etree.ElementTree as ET

    from workspace_analyzer import SamplingConfig
    from workspace_analyzer.plane_plot import plane_svg

    solver = create_solver(
        Path(__file__).parent / "fixtures/cartesian_stage.urdf", backend="numpy"
    )
    result = analyze_plane(
        solver,
        PlaneConfig(sampling=SamplingConfig(num_samples=91)),
        ReachabilityConfig(restarts=1, rescue_restarts=0),
    )
    # Rendering must retain even single-point outcome categories.
    result.reachable[:] = False
    result.reachable[[3, 74]] = True
    result.metrics["quality_pass"][:] = False
    result.metrics["quality_pass"][3] = True
    text = plane_svg(result, max_points=limit)
    nodes = ET.fromstring(text).findall(".//{http://www.w3.org/2000/svg}circle")
    assert len(nodes) == (91 if limit is None else min(limit, 91))
    indices = [int(node.get("data-target-index")) for node in nodes]
    assert len(set(indices)) == len(indices)
    assert 3 in indices and 74 in indices
    assert set(node.get("fill") for node in nodes) == {"#16846a", "#d99b18", "#ce4653"}
    assert text == plane_svg(result, max_points=limit)
    assert "IK solved: 2/91" in text
    assert ("Preview:" in text) == (limit is not None and limit < 91)
    assert plane_svg(result).count("<circle ") == 91


def test_plane_plot_preview_validates_limit():
    from workspace_analyzer.plane_plot import _preview_indices

    for bad in [True, 0, 2, 3.5]:
        with pytest.raises(ValueError, match="max_points"):
            _preview_indices(np.ones(10, dtype=bool), np.ones(10, dtype=bool), bad)


@pytest.mark.parametrize("frame", ["base", "world", "reference", "joint1"])
@pytest.mark.parametrize("plane", ["xy", "xz", "yz"])
def test_rotated_sampling_frame_composes_after_named_frame(
    mounted_solver, frame, plane
):
    config = PlaneConfig(
        plane=plane,
        frame=frame,
        center=(0.2, 0.3, 0.4),
        span=0.2,
        resolution=2,
        frame_rpy=(0, 0, np.pi / 2),
        rpy=(0, np.pi / 2, 0),
    )
    name, base_from_original = plane_frame(mounted_solver, frame)
    result = analyze_plane(
        mounted_solver,
        config,
        ReachabilityConfig(position_only=False, restarts=1, rescue_restarts=0),
    )
    expected_transform = base_from_original @ _transform(RZ, [0, 0, 0])
    np.testing.assert_allclose(
        result.points,
        config.points() @ expected_transform[:3, :3].T + expected_transform[:3, 3],
        atol=1e-12,
    )
    expected_rotation = expected_transform[:3, :3] @ RY
    np.testing.assert_allclose(
        result.target_poses[:, :3, :3],
        np.broadcast_to(expected_rotation, (4, 3, 3)),
        atol=1e-12,
    )
    report = result.metadata["plane_scan"]
    assert report["frame_rpy_rad"] == [0, 0, np.pi / 2]
    assert report["coordinate_frame"] == name
    np.testing.assert_allclose(
        report["base_from_plane_frame"], expected_transform, atol=1e-12
    )


@pytest.mark.parametrize(
    "full_pose,cancel_rotation,expected",
    [(False, False, 15), (True, False, 0), (True, True, 15)],
)
def test_tilted_plane_matches_analytic_cube_and_relative_tcp_orientation(
    full_pose, cancel_rotation, expected
):
    solver = create_solver(
        Path(__file__).parent / "fixtures/cartesian_stage.urdf", backend="numpy"
    )
    result = analyze_plane(
        solver,
        PlaneConfig(
            frame_rpy=(np.pi / 4, 0, 0),
            rpy=(-np.pi / 4 if cancel_rotation else 0, 0, 0),
            resolution=5,
            span=2.4,
        ),
        ReachabilityConfig(position_only=not full_pose, restarts=1, rescue_restarts=0),
    )
    assert result.reachable.sum() == expected
    np.testing.assert_allclose(result.points[:, 1], result.points[:, 2], atol=1e-12)
    if not full_pose or cancel_rotation:
        np.testing.assert_array_equal(
            result.reachable, np.all(np.abs(result.points) <= 1, axis=1)
        )


@pytest.mark.parametrize("entry", ["module", "demo"])
def test_tilted_plane_cli_preserves_geometry_and_export(entry, tmp_path):
    command = (
        [sys.executable, "-m", "workspace_analyzer"]
        if entry == "module"
        else [sys.executable, str(EXAMPLE), "--urdf"]
    )
    command += [
        str(Path(__file__).parent / "fixtures/cartesian_stage.urdf"),
        "--plane",
        "xy",
        "--frame-rpy",
        str(np.pi / 4),
        "0",
        "0",
        "--span",
        "2.4",
        "--resolution",
        "5",
        "--backend",
        "numpy",
        "--output",
        str(tmp_path / "result.npz"),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=20)
    assert completed.returncode == 0, completed.stderr
    result = AnalysisResult.load(tmp_path / "result.npz")
    assert result.reachable.sum() == 15
    assert result.metadata["plane_scan"]["frame_rpy_rad"] == [np.pi / 4, 0, 0]
