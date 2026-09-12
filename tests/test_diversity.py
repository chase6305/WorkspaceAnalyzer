import json
import threading
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from workspace_analyzer import (
    AnalysisCancelled,
    IKDiversityConfig,
    IKStabilityResult,
    IKTrial,
    KinematicsSolver,
    ReachabilityConfig,
    ResultCache,
    analyze_ik_stability,
    create_solver,
    make_ik_seed_trials,
    summarize_ik_diversity,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _solver(fixture="two_branch.urdf", backend="numpy", dtype="float64"):
    if backend == "torch":
        pytest.importorskip("torch")
    return create_solver(
        FIXTURES / fixture,
        backend=backend,
        device="cpu",
        dtype=dtype,
        max_iterations=100,
    )


def _redundant_study(values, **gates):
    return analyze_ik_stability(
        _solver("redundant_stage.urdf"),
        [[0, 0, 0]],
        [
            IKTrial(f"trial_{i}", seed=[value, -value, 0, 0])
            for i, value in enumerate(values)
        ],
        ReachabilityConfig(restarts=1, rescue_restarts=0, **gates),
    )


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_generated_seeds_are_bounded_reproducible_and_preserve_prefix(backend, dtype):
    solver = _solver(backend=backend, dtype=dtype)
    before = np.random.get_state()
    guesses = make_ik_seed_trials(solver, 4, random_seed=77, joint_fraction=0.1)
    repeated = make_ik_seed_trials(solver, 7, random_seed=77, joint_fraction=0.1)
    limits = solver.joint_limits
    for first, again in zip(guesses, repeated):
        assert first.name == again.name
        np.testing.assert_array_equal(first.seed, again.seed)
        assert first.seed.dtype == np.dtype(dtype)
        assert not first.seed.flags.writeable
        assert not np.shares_memory(first.seed, again.seed)
        assert first.max_iterations is first.random_seed is None
        fraction = (first.seed - limits[:, 0]) / (limits[:, 1] - limits[:, 0])
        assert np.all((fraction >= 0.4) & (fraction <= 0.6))
    assert not np.array_equal(
        guesses[0].seed, make_ik_seed_trials(solver, 1, random_seed=78)[0].seed
    )
    assert len({tuple(trial.seed) for trial in guesses}) == 4
    after = np.random.get_state()
    for left, right in zip(before, after):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_generated_trials_find_quality_branch_and_distinguish_repeated_roots(
    backend, dtype
):
    solver = _solver(backend=backend, dtype=dtype)
    trials = [
        IKTrial("baseline", seed=[np.pi / 2, -np.pi / 2]),
        *make_ik_seed_trials(solver, 4, random_seed=77),
    ]
    study = analyze_ik_stability(
        solver,
        [[1, 1, 0], [5, 5, 0]],
        trials,
        ReachabilityConfig(
            restarts=1,
            rescue_restarts=0,
            minimum_isotropy=0.05,
            minimum_joint_limit_margin=0.2,
        ),
    )
    report = study.summarize_diversity()
    assert report["per_target"]["configuration_counts"] == [2, 0]
    assert report["per_target"]["quality_configuration_counts"] == [1, 0]
    summary = report["summary"]
    assert summary["ik_candidate_count"] == 5
    assert summary["configuration_count"] == 2
    assert summary["duplicate_candidate_count"] == 3
    assert summary["targets_without_ik"] == 1
    assert all(row[1] == -1 for row in report["representative_trial_indices"])
    selected = study.select_solutions()
    assert selected.metrics["quality_pass"].tolist() == [True, False]
    a, b = selected.joint_positions[0]
    np.testing.assert_allclose(
        [np.cos(a) + np.cos(a + b), np.sin(a) + np.sin(a + b)], [1, 1], atol=1e-5
    )
    assert selected.metrics["joint_limit_margin"][0] > 0.4
    np.testing.assert_array_equal(
        selected.reachable, np.any([r.reachable for r in study.results], axis=0)
    )
    json.dumps(report, allow_nan=False)


def test_near_duplicate_can_pass_a_gate_even_when_its_representative_fails():
    study = _redundant_study([0.500025, 0.499975], minimum_joint_limit_margin=0.5)
    report = study.summarize_diversity()
    assert report["representative_trial_indices"] == [[0], [0]]
    assert report["per_target"]["configuration_counts"] == [1]
    assert report["per_target"]["quality_configuration_counts"] == [1]
    assert report["by_trial"][1]["new_configuration_count"] == 0
    assert report["by_trial"][1]["new_quality_configuration_count"] == 1
    assert study.select_solutions().metrics["selected_trial_index"].tolist() == [1]
    # Grouping does not discard a close candidate that satisfies a quality gate.
    assert len(study.results) == 2
    strict = study.reassess_quality(
        minimum_joint_limit_margin=0.6
    ).summarize_diversity()
    assert (
        strict["representative_trial_indices"] == report["representative_trial_indices"]
    )
    assert strict["per_target"]["quality_configuration_counts"] == [0]


def test_greedy_groups_use_representative_distance_not_transitive_connectivity():
    study = _redundant_study([0, 0.000075, 0.00015])
    report = study.summarize_diversity()
    assert report["representative_trial_indices"] == [[0], [0], [2]]
    # The middle configuration is within tolerance of both endpoints.
    reordered = IKStabilityResult(
        tuple(study.names[i] for i in [1, 0, 2]),
        tuple(study.results[i] for i in [1, 0, 2]),
    ).summarize_diversity()
    assert reordered["per_target"]["configuration_counts"] == [1]
    assert report["per_target"]["configuration_counts"] == [2]


def test_prismatic_grouping_uses_metres_and_all_joints_must_match():
    study = _redundant_study([0, 0.0002])
    report = study.summarize_diversity(IKDiversityConfig(angular_tolerance_rad=100))
    assert report["per_target"]["configuration_counts"] == [2]
    coarse = study.summarize_diversity(IKDiversityConfig(linear_tolerance_m=0.0003))
    assert coarse["per_target"]["configuration_counts"] == [1]
    assert coarse["joint_tolerances"] == [0.0003] * 4


@pytest.mark.parametrize("kind, expected", [("continuous", 1), ("revolute", 3)])
def test_only_continuous_joints_wrap_equivalent_turns(kind, expected, tmp_path):
    text = (FIXTURES / "continuous.urdf").read_text()
    if kind == "revolute":
        text = text.replace('type="continuous"', 'type="revolute"').replace(
            '<limit velocity="1"/>', '<limit lower="-12.6" upper="12.6" velocity="1"/>'
        )
    path = tmp_path / "robot.urdf"
    path.write_text(text)
    solver = create_solver(path, backend="numpy", max_iterations=1)
    study = analyze_ik_stability(
        solver,
        [-1, 0, 0],
        [
            IKTrial(str(i), seed=[angle])
            for i, angle in enumerate([np.pi, -np.pi, 3 * np.pi])
        ],
        ReachabilityConfig(restarts=1, rescue_restarts=0),
    )
    if kind == "continuous":
        # An external trajectory can store continuous angles without normalization.
        for result, angle in zip(study.results, [np.pi, -np.pi, 3 * np.pi]):
            result.joint_positions[0, 0] = angle
            np.testing.assert_allclose(
                solver.forward([angle])[:3, 3], [-1, 0, 0], atol=1e-14
            )
    report = study.summarize_diversity(IKDiversityConfig(linear_tolerance_m=100))
    assert report["per_target"]["configuration_counts"] == [expected]
    assert report["joint_tolerances"] == [1e-3]


def test_cached_and_saved_diversity_is_offline_and_keeps_original_measurements(
    tmp_path, monkeypatch
):
    solver = _solver()
    trials = make_ik_seed_trials(solver, 3, random_seed=77)
    settings = ReachabilityConfig(restarts=1, rescue_restarts=0)
    cache = ResultCache(tmp_path / "cache")
    first = analyze_ik_stability(solver, [[1, 1, 0]], trials, settings, cache=cache)
    original = first.to_dict()
    report = first.summarize_diversity()

    def unexpected(*args, **kwargs):
        pytest.fail("cached studies and diversity must not run kinematics")

    for method in ("inverse", "forward", "forward_with_jacobian"):
        monkeypatch.setattr(KinematicsSolver, method, unexpected)
    again = analyze_ik_stability(solver, [[1, 1, 0]], trials, settings, cache=cache)
    assert all(result.metadata["cache_hit"] for result in again.results)
    assert again.summarize_diversity() == report
    first.save(tmp_path / "study")
    saved = IKStabilityResult.load(tmp_path / "study")
    assert saved.summarize_diversity() == report
    assert saved.results[0].metadata["joint_kinds"] == ["revolute", "revolute"]
    assert first.to_dict() == original


def test_legacy_studies_require_explicit_joint_types_and_reject_conflicting_types():
    study = _redundant_study([0, 0.01])
    with pytest.raises(ValueError, match="disagree"):
        study.summarize_diversity(joint_kinds=["continuous"] * 4)
    for result in study.results:
        result.metadata.pop("joint_kinds")
    with pytest.raises(ValueError, match="joint_kinds"):
        study.summarize_diversity()
    assert (
        study.summarize_diversity(joint_kinds=["prismatic"] * 4)["summary"][
            "configuration_count"
        ]
        == 2
    )
    for kinds in (
        ["fixed"] * 4,
        ["prismatic"],
        "prismatic",
        0,
        np.array("prismatic"),
        [["prismatic"]] * 4,
    ):
        with pytest.raises(ValueError, match="joint_kinds"):
            study.summarize_diversity(joint_kinds=kinds)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"count": 0},
        {"count": -1},
        {"count": True},
        {"count": 1.5},
        {"random_seed": -1},
        {"random_seed": True},
        {"joint_fraction": 0},
        {"joint_fraction": 0.6},
        {"joint_fraction": np.nan},
        {"joint_fraction": True},
    ],
)
def test_seed_generation_rejects_invalid_inputs(kwargs):
    arguments = {"count": 2, **kwargs}
    with pytest.raises(ValueError):
        make_ik_seed_trials(_solver(), **arguments)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"angular_tolerance_rad": 0},
        {"angular_tolerance_rad": np.nan},
        {"angular_tolerance_rad": True},
        {"linear_tolerance_m": -1},
        {"linear_tolerance_m": np.inf},
        {"linear_tolerance_m": "1"},
    ],
)
def test_diversity_tolerances_reject_invalid_units_or_values(kwargs):
    with pytest.raises(ValueError):
        IKDiversityConfig(**kwargs)


def test_diversity_rejects_stale_quality_and_nonfinite_successful_joints():
    study = _redundant_study([0, 0.01])
    study.results[1].metrics["quality_pass"][0] = False
    with pytest.raises(ValueError, match="quality"):
        study.summarize_diversity()
    study.results[1].metrics["quality_pass"][0] = True
    study.results[1].joint_positions[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite joints"):
        study.summarize_diversity()


def test_diversity_cancellation_before_during_and_final(monkeypatch):
    import workspace_analyzer.diversity as module

    study = _redundant_study([0, 0.01, 0.02])
    original = module.check_cancelled
    calls = []

    def counted(event):
        calls.append(1)
        original(event)

    monkeypatch.setattr(module, "check_cancelled", counted)
    study.summarize_diversity()
    for cancel_at in (1, 3, len(calls)):
        event = threading.Event()
        count = 0

        def cancelling(event):
            nonlocal count
            count += 1
            if count == cancel_at:
                event.set()
            original(event)

        monkeypatch.setattr(module, "check_cancelled", cancelling)
        with pytest.raises(AnalysisCancelled):
            study.summarize_diversity(cancel_event=event)


def test_report_rebuild_after_editing_summary_does_not_mutate_study():
    study = _redundant_study([0, 0.01])
    report = summarize_ik_diversity(study)
    report["representative_trial_indices"][0][0] = 99
    assert study.summarize_diversity()["representative_trial_indices"] == [[0], [1]]
    # Extending the same ordered pool cannot lose an existing representative.
    extended = IKStabilityResult(
        (*study.names, "again"), (*study.results, replace(study.results[0]))
    )
    assert extended.summarize_diversity()["per_target"]["configuration_counts"] == [2]


@pytest.mark.parametrize("random_seed", [17, 53, 91])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_optimized_grouping_matches_scalar_oracle_with_sparse_representatives(
    random_seed, dtype
):
    from copy import deepcopy

    rng = np.random.default_rng(random_seed)
    prototype = _redundant_study([0, 0.1]).results[0]
    trials, targets = 16, 13
    kinds = ["continuous", "revolute", "prismatic", "prismatic"]
    tolerance = [1e-3, 1e-3, 1e-4, 1e-4]
    results = []
    for index in range(trials):
        # Synthetic recorded candidates isolate grouping from IK search behavior.
        joints = np.column_stack(
            [
                rng.choice([0.0, 0.0005, 0.01], targets)
                + rng.integers(-2, 3, targets) * (2 * np.pi),
                rng.choice([0.0, 0.0005, 0.01], targets),
                rng.choice([0.0, 0.00005, 0.002], targets),
                rng.choice([0.0, 0.00005, 0.002], targets),
            ]
        ).astype(dtype)
        success = rng.random(targets) > 0.3
        if index == 0:
            success[:] = False  # No representatives exist after the first trial.
        if index > 1 and index % 3 == 0:
            joints = results[1].joint_positions.copy()
            success = results[1].reachable.copy()
        metrics = {
            name: np.repeat(value, targets) for name, value in prototype.metrics.items()
        }
        metrics["isotropy"] = rng.choice([0.1, 0.8], targets)
        metrics["quality_pass"] = success & (metrics["isotropy"] >= 0.5)
        metadata = deepcopy(prototype.metadata)
        metadata["joint_kinds"] = kinds.copy()
        metadata["quality_thresholds"]["minimum_isotropy"] = 0.5
        results.append(
            replace(
                prototype,
                points=np.repeat(prototype.points, targets, axis=0),
                joint_positions=joints,
                residual=np.where(success, 0.0, 1.0),
                reachable=success,
                manipulability=np.repeat(prototype.manipulability, targets),
                metrics=metrics,
                metadata=metadata,
            )
        )
    study = IKStabilityResult(
        tuple(f"trial_{i}" for i in range(trials)), tuple(results)
    )
    report = study.summarize_diversity()
    labels = np.full((trials, targets), -1, dtype=int)
    representatives = [[] for _ in range(targets)]
    quality_groups = [set() for _ in range(targets)]
    for index, result in enumerate(results):
        new_groups = new_quality = 0
        for row in range(targets):
            if not result.reachable[row]:
                continue
            chosen = index
            for prior in representatives[row]:
                matches = True
                for joint, tol in enumerate(tolerance):
                    delta = float(result.joint_positions[row, joint]) - float(
                        results[prior].joint_positions[row, joint]
                    )
                    if kinds[joint] == "continuous":
                        delta = (delta + np.pi) % (2 * np.pi) - np.pi
                    if abs(delta) > tol:
                        matches = False
                        break
                if matches:
                    chosen = prior
                    break
            if chosen == index:
                representatives[row].append(index)
                new_groups += 1
            labels[index, row] = chosen
            if (
                result.metrics["quality_pass"][row]
                and chosen not in quality_groups[row]
            ):
                quality_groups[row].add(chosen)
                new_quality += 1
        trial_report = report["by_trial"][index]
        assert trial_report["new_configuration_count"] == new_groups
        assert trial_report["new_quality_configuration_count"] == new_quality
        assert (
            trial_report["duplicate_candidate_count"]
            == int(result.reachable.sum()) - new_groups
        )
        assert trial_report["cumulative_configuration_count"] == sum(
            map(len, representatives)
        )
    np.testing.assert_array_equal(report["representative_trial_indices"], labels)
    assert report["per_target"]["configuration_counts"] == list(
        map(len, representatives)
    )
    assert report["per_target"]["quality_configuration_counts"] == list(
        map(len, quality_groups)
    )
    json.dumps(report, allow_nan=False)
