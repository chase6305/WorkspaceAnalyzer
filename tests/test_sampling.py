import numpy as np
import pytest

from workspace_analyzer import SamplingConfig, SamplingStrategy
from workspace_analyzer.sampling import sample


@pytest.mark.parametrize("strategy", list(SamplingStrategy))
def test_sampling_strategies_are_bounded_and_deterministic(strategy):
    if strategy in {
        SamplingStrategy.HALTON,
        SamplingStrategy.SOBOL,
        SamplingStrategy.LATIN_HYPERCUBE,
    }:
        pytest.importorskip("scipy")
    limits = np.array([[-2.0, 1.0], [3.0, 5.0]])
    original_limits = limits.copy()
    config = SamplingConfig(strategy, num_samples=17, batch_size=8, seed=9)
    first, second = sample(limits, config), sample(limits, config)
    assert first.shape == (17, 2)
    np.testing.assert_allclose(first, second)
    assert np.all(first >= limits[:, 0])
    assert np.all(first <= limits[:, 1])
    np.testing.assert_array_equal(limits, original_limits)


@pytest.mark.parametrize(
    ("limits", "message"),
    [
        (np.array([1.0, 2.0]), "shape"),
        (np.array([[1.0, 1.0]]), "lower < upper"),
        (np.array([[0.0, np.inf]]), "finite"),
    ],
)
def test_sampling_rejects_invalid_limits(limits, message):
    with pytest.raises(ValueError, match=message):
        sample(limits, SamplingConfig(num_samples=2))


def test_sampling_config_rejects_invalid_sizes():
    with pytest.raises(ValueError, match="num_samples"):
        SamplingConfig(num_samples=0)
    with pytest.raises(ValueError, match="batch_size"):
        SamplingConfig(batch_size=0)


@pytest.mark.parametrize("dimensions,count", [(1, 3), (2, 17), (3, 29), (7, 130)])
def test_uniform_sampling_preserves_meshgrid_order(dimensions, count):
    side = max(2, int(np.ceil(count ** (1 / dimensions))))
    grid = np.meshgrid(*([np.linspace(0, 1, side)] * dimensions), indexing="ij")
    expected = np.stack(grid, axis=-1).reshape(-1, dimensions)[:count]
    actual = sample(
        np.tile([0.0, 1.0], (dimensions, 1)),
        SamplingConfig(strategy="uniform", num_samples=count),
    )
    np.testing.assert_array_equal(actual, expected)


def test_high_dimensional_grid_only_allocates_requested_points():
    # A full two-level grid here would contain 2**64 configurations.
    actual = sample(
        np.tile([0.0, 1.0], (64, 1)),
        SamplingConfig(strategy="uniform", num_samples=3),
    )
    assert actual.shape == (3, 64)
    np.testing.assert_array_equal(actual[:, :-2], 0.0)
    np.testing.assert_array_equal(actual[:, -2:], [[0, 0], [0, 1], [1, 0]])


@pytest.mark.parametrize("name", ["num_samples", "batch_size", "seed"])
@pytest.mark.parametrize("value", [-1, 1.5, True, float("nan")])
def test_sampling_config_rejects_invalid_integer_values(name, value):
    with pytest.raises(ValueError, match=name):
        SamplingConfig(**{name: value})


def test_sampling_rejects_empty_dimensions():
    with pytest.raises(ValueError, match="dimensions"):
        sample(np.empty((0, 2)), SamplingConfig())


def test_sobol_truncation_does_not_retain_unrequested_samples():
    pytest.importorskip("scipy")
    points = sample(
        np.tile([0.0, 1.0], (7, 1)),
        SamplingConfig(strategy="sobol", num_samples=17),
    )
    storage = points
    while isinstance(storage.base, np.ndarray):
        storage = storage.base
    assert storage.nbytes == points.nbytes


@pytest.mark.parametrize("strategy", list(SamplingStrategy))
@pytest.mark.parametrize("active", [[], [1], [0, 2]])
def test_fixed_axes_preserve_reduced_dimension_sampling(strategy, active):
    if strategy in {
        SamplingStrategy.HALTON,
        SamplingStrategy.SOBOL,
        SamplingStrategy.LATIN_HYPERCUBE,
    }:
        pytest.importorskip("scipy")
    limits = np.array([[0.25, 0.25], [-0.5, -0.5], [0.75, 0.75]])
    limits[active, 1] += 1
    config = SamplingConfig(strategy, num_samples=9, seed=17)
    actual = sample(limits, config, allow_fixed=True)
    fixed = np.ones(3, dtype=bool)
    fixed[active] = False
    np.testing.assert_array_equal(
        actual[:, fixed], np.broadcast_to(limits[fixed, 0], (9, fixed.sum()))
    )
    if active:
        np.testing.assert_array_equal(actual[:, active], sample(limits[active], config))
        if strategy == SamplingStrategy.UNIFORM:
            assert len(np.unique(actual, axis=0)) == 9
    assert actual.flags.writeable


@pytest.mark.parametrize("strategy", list(SamplingStrategy))
def test_extreme_finite_intervals_do_not_overflow(strategy):
    if strategy in {
        SamplingStrategy.HALTON,
        SamplingStrategy.SOBOL,
        SamplingStrategy.LATIN_HYPERCUBE,
    }:
        pytest.importorskip("scipy")
    largest = np.finfo(float).max
    limits = np.array([[-largest, largest], [2.0, 3.0]])
    with np.errstate(over="raise", invalid="raise"):
        actual = sample(limits, SamplingConfig(strategy, num_samples=16))
    assert np.isfinite(actual).all()
    assert np.all(actual >= limits[:, 0])
    assert np.all(actual <= limits[:, 1])
    if strategy == SamplingStrategy.UNIFORM:
        np.testing.assert_array_equal(actual[[0, -1]], limits.T)


def test_fixed_axes_still_reject_reversed_intervals():
    with pytest.raises(ValueError, match="lower <= upper"):
        sample([[0, 0], [1, -1]], SamplingConfig(), allow_fixed=True)
