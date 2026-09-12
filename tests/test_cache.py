from pathlib import Path

import numpy as np
import pytest

from workspace_analyzer import (
    ResultCache,
    RobotModel,
    SamplingConfig,
    SolverConfig,
    WorkspaceAnalyzer,
    WorkspaceConfig,
    analysis_cache_key,
)
from workspace_analyzer.kinematics import KinematicsSolver

URDF = Path(__file__).parent / "fixtures/two_link.urdf"


def _analyzer(samples=12):
    solver = KinematicsSolver(
        RobotModel.from_urdf(URDF), config=SolverConfig(backend="numpy")
    )
    config = WorkspaceConfig(SamplingConfig(num_samples=samples, seed=3))
    return WorkspaceAnalyzer(solver, config)


def test_analysis_cache_miss_then_hit(tmp_path):
    analyzer = _analyzer()
    cache = ResultCache(tmp_path / "cache")
    first = analyzer.analyze(cache=cache)
    assert first.metadata["cache_hit"] is False
    second = analyzer.analyze(cache=cache)
    assert second.metadata["cache_hit"] is True
    assert second.metadata["cache_key"] == analysis_cache_key(
        analyzer.solver, "joint", analyzer.config
    )


def test_cache_key_changes_with_configuration_and_rejects_bad_keys(tmp_path):
    first, second = _analyzer(12), _analyzer(13)
    first_key = analysis_cache_key(first.solver, "joint", first.config)
    second_key = analysis_cache_key(second.solver, "joint", second.config)
    assert first_key != second_key
    cache = ResultCache(tmp_path)
    try:
        cache.path_for("not-a-digest")
    except ValueError as error:
        assert "SHA-256" in str(error)
    else:
        raise AssertionError("invalid cache key was accepted")


def test_corrupt_cache_entry_is_treated_as_a_miss(tmp_path):
    analyzer = _analyzer()
    cache = ResultCache(tmp_path)
    key = analysis_cache_key(analyzer.solver, "joint", analyzer.config)
    path = cache.path_for(key)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not an npz")
    assert cache.get(key) is None


def test_cache_key_tracks_loaded_model_instead_of_changed_source(tmp_path):
    source = tmp_path / "robot.urdf"
    original = URDF.read_text()
    source.write_text(original)
    config = WorkspaceConfig(SamplingConfig(num_samples=2))
    solver_config = SolverConfig(backend="numpy")
    old_solver = KinematicsSolver(RobotModel.from_urdf(source), config=solver_config)
    original_key = analysis_cache_key(old_solver, "joint", config)
    source.write_text(original.replace('xyz="1 0 0"', 'xyz="2 0 0"'))
    new_solver = KinematicsSolver(RobotModel.from_urdf(source), config=solver_config)
    changed_key = analysis_cache_key(new_solver, "joint", config)
    assert changed_key != original_key
    assert analysis_cache_key(old_solver, "joint", config) == original_key
    source.unlink()
    assert analysis_cache_key(old_solver, "joint", config) == original_key
    assert analysis_cache_key(new_solver, "joint", config) == changed_key


@pytest.mark.parametrize("metadata", ["null", "[]", '"text"'])
def test_cache_with_invalid_metadata_is_a_miss(tmp_path, metadata):
    analyzer = _analyzer()
    cache = ResultCache(tmp_path)
    key = analysis_cache_key(analyzer.solver, "joint", analyzer.config)
    path = cache.path_for(key)
    path.parent.mkdir(parents=True)
    np.savez(
        path,
        format_version=np.asarray(1),
        metadata_json=np.asarray(metadata),
        points=np.zeros((1, 3)),
        joint_positions=np.zeros((1, 2)),
    )
    assert cache.get(key) is None


@pytest.mark.parametrize("compressed", [False, True])
def test_cache_encodings_share_keys_and_can_be_read_by_either_policy(
    tmp_path, compressed
):
    from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

    analyzer = _analyzer()
    cache = ResultCache(tmp_path, compressed=compressed)
    first = analyzer.analyze(cache=cache)
    key = first.metadata["cache_key"]
    with ZipFile(cache.path_for(key)) as archive:
        assert {entry.compress_type for entry in archive.infolist()} == {
            ZIP_DEFLATED if compressed else ZIP_STORED
        }
    replay = analyzer.analyze(cache=ResultCache(tmp_path, compressed=not compressed))
    assert replay.metadata["cache_hit"]
    assert replay.metadata["cache_key"] == key
    np.testing.assert_array_equal(replay.points, first.points)


def test_cache_failure_does_not_change_callers_key_or_existing_entry(
    tmp_path, monkeypatch
):
    analyzer = _analyzer()
    cache = ResultCache(tmp_path)
    result = analyzer.analyze(cache=cache)
    original_key = result.metadata["cache_key"]
    destination = "a" * 64
    cache.put(destination, result)
    result.metadata["cache_key"] = original_key
    original = cache.path_for(destination).read_bytes()

    def fail(*args, **kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(np, "savez_compressed", fail)
    with pytest.raises(OSError, match="disk failure"):
        cache.put(destination, result)
    assert result.metadata["cache_key"] == original_key
    assert cache.path_for(destination).read_bytes() == original
    assert cache.get(destination).metadata["cache_key"] == destination


@pytest.mark.parametrize("compressed", [0, 1, "false", None])
def test_cache_rejects_ambiguous_compression_flag(tmp_path, compressed):
    with pytest.raises(ValueError, match="compressed"):
        ResultCache(tmp_path / "new", compressed=compressed)
    assert not (tmp_path / "new").exists()


def test_corrupt_deflate_stream_is_recomputed_then_replayed(tmp_path):
    import struct
    from zipfile import ZipFile

    analyzer = _analyzer()
    cache = ResultCache(tmp_path)
    first = analyzer.analyze(cache=cache)
    key = first.metadata["cache_key"]
    path = cache.path_for(key)
    with ZipFile(path) as archive:
        member = archive.infolist()[0]
    data = bytearray(path.read_bytes())
    name_length, extra_length = struct.unpack_from(
        "<HH", data, member.header_offset + 26
    )
    # Invalid DEFLATE block type; keep the ZIP directory intact so decompression
    # reaches the zlib error path rather than merely rejecting a truncated ZIP.
    data[member.header_offset + 30 + name_length + extra_length] = 7
    path.write_bytes(data)
    from workspace_analyzer import AnalysisResult

    with pytest.raises(ValueError, match="invalid result archive"):
        AnalysisResult.load(path)
    assert cache.get(key) is None
    repaired = analyzer.analyze(cache=cache)
    assert not repaired.metadata["cache_hit"]
    np.testing.assert_array_equal(repaired.points, first.points)
    assert analyzer.analyze(cache=cache).metadata["cache_hit"]


@pytest.mark.parametrize(
    "key", [None, 42, b"a" * 64, ["a"] * 64, np.array(["a"] * 64), "A" * 64]
)
def test_invalid_cache_key_types_have_consistent_errors(tmp_path, key):
    cache = ResultCache(tmp_path)
    for method in (cache.path_for, cache.get):
        with pytest.raises(ValueError, match="SHA-256"):
            method(key)
