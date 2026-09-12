import json
from pathlib import Path

import numpy as np
import pytest

from workspace_analyzer import AnalysisResult


def test_result_round_trip_without_pickle(tmp_path: Path):
    result = AnalysisResult(
        points=np.array([[1.0, 2.0, 3.0]]),
        joint_positions=np.array([[0.1, 0.2]]),
        manipulability=None,
        metadata={"mode": "cartesian", "joint_names": ("a", "b")},
        reachable=np.array([True]),
        residual=np.array([1e-7]),
        metrics={"isotropy": np.array([0.8])},
    )
    result.save(tmp_path / "result")
    target = tmp_path / "result.npz"
    assert target.is_file()
    with np.load(target, allow_pickle=False) as archive:
        assert "metadata_json" in archive.files
        assert "metadata" not in archive.files
        assert json.loads(str(archive["metadata_json"]))["mode"] == "cartesian"
    loaded = AnalysisResult.load(target)
    np.testing.assert_array_equal(loaded.points, result.points)
    np.testing.assert_array_equal(loaded.reachable, result.reachable)
    assert loaded.manipulability is None
    np.testing.assert_array_equal(loaded.metrics["isotropy"], [0.8])
    assert loaded.metadata["joint_names"] == ["a", "b"]


def test_result_rejects_unknown_format(tmp_path: Path):
    target = tmp_path / "future.npz"
    np.savez(
        target,
        format_version=np.asarray(999),
        points=np.empty((0, 3)),
        joint_positions=np.empty((0, 0)),
        metadata_json=np.asarray("{}"),
    )
    with pytest.raises(ValueError, match="unsupported result format"):
        AnalysisResult.load(target)


def test_result_rejects_unsafe_metric_name(tmp_path: Path):
    result = AnalysisResult(
        points=np.empty((0, 3)),
        joint_positions=np.empty((0, 0)),
        manipulability=None,
        metadata={},
        metrics={"bad/name": np.empty(0)},
    )
    with pytest.raises(ValueError, match="metric name"):
        result.save(tmp_path / "bad.npz")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"points": np.empty((2, 2)), "joint_positions": np.empty((2, 1))},
        {"points": np.empty((2, 3)), "joint_positions": np.empty((1, 1))},
        {
            "points": np.empty((2, 3)),
            "joint_positions": np.empty((2, 1)),
            "reachable": np.empty(1),
        },
        {
            "points": np.empty((2, 3)),
            "joint_positions": np.empty((2, 1)),
            "metrics": {"isotropy": np.empty(1)},
        },
    ],
)
def test_result_rejects_inconsistent_shapes(kwargs):
    with pytest.raises(ValueError, match="shape"):
        AnalysisResult(manipulability=None, metadata={}, **kwargs)


def test_result_normalizes_list_inputs_for_indexing_and_round_trip(tmp_path):
    result = AnalysisResult(
        points=[[1, 2, 3]],
        joint_positions=[[0.1, 0.2]],
        manipulability=[0.5],
        metadata={},
        reachable=[1],
        residual=[0.0],
        metrics={"isotropy": [0.8]},
    )
    np.testing.assert_array_equal(result.points[np.array([0])], [[1, 2, 3]])
    assert result.reachable.dtype == np.bool_
    result.save(tmp_path / "lists.npz")
    restored = AnalysisResult.load(tmp_path / "lists.npz")
    np.testing.assert_array_equal(restored.metrics["isotropy"], [0.8])


def test_result_rejects_arrays_that_require_pickle():
    with pytest.raises(ValueError, match="numeric"):
        AnalysisResult(
            points=np.array([[1, 2, object()]], dtype=object),
            joint_positions=[[0.1]],
            manipulability=None,
            metadata={},
        )


@pytest.mark.parametrize("version", [1.5, [1], "1"])
def test_result_rejects_malformed_version(tmp_path, version):
    path = tmp_path / "invalid.npz"
    np.savez(path, format_version=np.asarray(version))
    with pytest.raises(ValueError, match="integer scalar"):
        AnalysisResult.load(path)


def test_result_rejects_mismatched_target_pose_count():
    with pytest.raises(ValueError, match="target_poses.*shape"):
        AnalysisResult(
            points=np.zeros((1, 3)),
            joint_positions=np.zeros((1, 2)),
            manipulability=None,
            metadata={},
            target_poses=np.zeros((2, 4, 4)),
        )


@pytest.mark.parametrize("compressed", [False, True])
def test_storage_modes_preserve_arrays_and_use_requested_compression(
    tmp_path, compressed
):
    from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

    result = AnalysisResult(
        points=np.zeros((2, 3), dtype=np.float32),
        joint_positions=np.ones((2, 2)),
        manipulability=[np.nan, np.inf],
        metadata={"name": "test"},
        reachable=[True, False],
        residual=[0, np.inf],
        metrics={"isotropy": [0.1, 0.2]},
    )
    path = tmp_path / "result.npz"
    result.save(path, compressed=compressed)
    with ZipFile(path) as archive:
        assert {entry.compress_type for entry in archive.infolist()} == {
            ZIP_DEFLATED if compressed else ZIP_STORED
        }
    loaded = AnalysisResult.load(path)
    for name in (
        "points",
        "joint_positions",
        "manipulability",
        "reachable",
        "residual",
    ):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(result, name))
        assert getattr(loaded, name).dtype == getattr(result, name).dtype
    np.testing.assert_array_equal(
        loaded.metrics["isotropy"], result.metrics["isotropy"]
    )


@pytest.mark.parametrize(
    "mutation", ["object", "shape", "metric_name", "metadata", "reachable"]
)
def test_mutated_result_cannot_overwrite_valid_archive(tmp_path, mutation):
    result = AnalysisResult(np.zeros((2, 3)), np.zeros((2, 1)), None, {})
    path = tmp_path / "result.npz"
    result.save(path)
    original = path.read_bytes()
    if mutation == "object":
        result.points = result.points.astype(object)
    elif mutation == "shape":
        result.joint_positions = np.zeros((3, 1))
    elif mutation == "metric_name":
        result.metrics = {12: np.zeros(2)}
    elif mutation == "metadata":
        result.metadata = []
    else:
        result.reachable = np.array([0, 2])
    with pytest.raises(ValueError):
        result.save(path)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]
    assert AnalysisResult.load(path).points.shape == (2, 3)


@pytest.mark.parametrize("compressed", [False, True])
def test_failed_writer_preserves_previous_archive_and_cleans_temporary_file(
    tmp_path, monkeypatch, compressed
):
    result = AnalysisResult(np.zeros((1, 3)), np.zeros((1, 1)), None, {})
    path = tmp_path / "result.npz"
    result.save(path)
    original = path.read_bytes()

    def fail(path, **payload):
        Path(path).write_bytes(b"partial archive")
        raise OSError("simulated disk failure")

    monkeypatch.setattr(np, "savez_compressed" if compressed else "savez", fail)
    with pytest.raises(OSError, match="disk failure"):
        result.save(path, compressed=compressed)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("compressed", [0, 1, "false", None])
def test_save_rejects_ambiguous_compression_flag(tmp_path, compressed):
    result = AnalysisResult(np.zeros((1, 3)), np.zeros((1, 1)), None, {})
    with pytest.raises(ValueError, match="compressed"):
        result.save(tmp_path / "result.npz", compressed=compressed)
    assert not list(tmp_path.iterdir())


def test_npy_file_is_rejected_with_a_format_error(tmp_path):
    path = tmp_path / "array.npy"
    np.save(path, np.zeros((3, 3)))
    with pytest.raises(ValueError, match="NPZ archive"):
        AnalysisResult.load(path)


@pytest.mark.parametrize(
    "missing", ["format_version", "points", "joint_positions", "metadata_json"]
)
def test_missing_required_members_have_clear_archive_errors(tmp_path, missing):
    payload = dict(
        format_version=np.asarray(1),
        points=np.zeros((1, 3)),
        joint_positions=np.zeros((1, 2)),
        metadata_json=np.asarray("{}"),
    )
    payload.pop(missing)
    path = tmp_path / "incomplete.npz"
    np.savez(path, **payload)
    with pytest.raises(ValueError, match=f"invalid result archive.*{missing}"):
        AnalysisResult.load(path)


def test_truncated_zip_has_clear_error_and_missing_file_keeps_os_error(tmp_path):
    path = tmp_path / "broken.npz"
    result = AnalysisResult(np.zeros((1, 3)), np.zeros((1, 1)), None, {})
    result.save(path)
    path.write_bytes(path.read_bytes()[:40])
    with pytest.raises(ValueError, match="invalid result archive"):
        AnalysisResult.load(path)
    with pytest.raises(FileNotFoundError):
        AnalysisResult.load(tmp_path / "absent.npz")
