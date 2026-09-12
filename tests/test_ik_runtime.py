import threading
from pathlib import Path

import numpy as np
import pytest

from workspace_analyzer import AnalysisCancelled, create_solver

FIXTURES = Path(__file__).parent / "fixtures"


def _solver(backend, **kwargs):
    if backend == "torch":
        pytest.importorskip("torch")
    return create_solver(
        str(FIXTURES / "two_link.urdf"), backend=backend, device="cpu", **kwargs
    )


def _numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else value


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_solved_candidates_retire_without_changing_other_solutions(
    backend, monkeypatch
):
    import workspace_analyzer.kinematics as kinematics

    solver = _solver(backend, max_iterations=15)
    seed = np.array([[0.2, 0.7], [0.2, 0.7], [0.2, 0.7]])
    targets = _numpy(solver.forward(seed)).copy()
    targets[1] = _numpy(solver.forward([0.21, 0.69]))
    targets[2, :3, 3] = [10.0, 0.0, 0.0]
    name = f"_geometric_jacobian_{backend}"
    original = getattr(kinematics, name)
    batch_sizes = []

    def record(solver, q, **kwargs):
        batch_sizes.append(len(q))
        return original(solver, q, **kwargs)

    monkeypatch.setattr(kinematics, name, record)
    result = solver.inverse(targets, seed=seed, position_only=True)
    assert batch_sizes[0] == 3
    assert batch_sizes[1] == 2
    assert batch_sizes[-1] == 1
    np.testing.assert_array_equal(_numpy(result.success), [True, True, False])
    np.testing.assert_array_equal(_numpy(result.positions)[0], seed[0])
    for index in (1, 2):
        individual = solver.inverse(
            targets[index], seed=seed[index], position_only=True
        )
        np.testing.assert_allclose(
            _numpy(result.positions)[index], _numpy(individual.positions), atol=1e-12
        )
        assert float(result.residual[index]) == pytest.approx(
            float(individual.residual)
        )


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("during_rescue", [False, True])
def test_inverse_cancels_between_iterations_and_during_rescue(
    backend, during_rescue, monkeypatch
):
    import workspace_analyzer.kinematics as kinematics

    solver = _solver(backend, max_iterations=2 if during_rescue else 20)
    cancel = threading.Event()
    name = f"_geometric_jacobian_{backend}"
    original = getattr(kinematics, name)
    sizes = []

    def cancel_during_compute(solver, q, **kwargs):
        sizes.append(len(q))
        result = original(solver, q, **kwargs)
        if not during_rescue or len(q) == 3:
            cancel.set()
        return result

    monkeypatch.setattr(kinematics, name, cancel_during_compute)
    target = np.eye(4)
    target[:3, 3] = [10, 0, 0]
    with pytest.raises(AnalysisCancelled):
        solver.inverse(
            target,
            position_only=True,
            cancel_event=cancel,
            rescue_restarts=3,
            rescue_rounds=2,
        )
    assert sizes == ([1, 1, 3] if during_rescue else [1])


def test_pre_cancelled_inverse_and_trajectory_skip_computation(monkeypatch):
    solver = _solver("numpy")
    cancel = threading.Event()
    cancel.set()

    def unexpected(*args, **kwargs):
        pytest.fail("pre-cancelled work should not convert or solve targets")

    monkeypatch.setattr(solver, "_array", unexpected)
    with pytest.raises(AnalysisCancelled):
        solver.inverse(np.eye(4), cancel_event=cancel)
    with pytest.raises(AnalysisCancelled):
        solver.solve_trajectory(np.eye(4)[None], cancel_event=cancel)


def test_torch_retirement_preserves_autograd():
    torch = pytest.importorskip("torch")
    solver = _solver("torch", max_iterations=3)
    seed = torch.tensor(
        [[0.2, 0.4], [0.1, 0.3]], dtype=torch.float64, requires_grad=True
    )
    target = solver.forward([[0.2, 0.4], [0.3, 0.5]])
    result = solver.inverse(target, seed=seed, position_only=True)
    result.positions.sum().backward()
    assert torch.isfinite(seed.grad).all()
    torch.testing.assert_close(seed.grad[0], torch.ones(2, dtype=torch.float64))


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("name", ["seed", "posture_reference"])
def test_scalar_configuration_arguments_have_shape_errors(backend, name):
    solver = _solver(backend)
    with pytest.raises(ValueError, match="shape"):
        solver.inverse(np.eye(4), **{name: 0.2})


def test_joint_analysis_reports_progress_and_honors_final_cancellation(tmp_path):
    from workspace_analyzer import (
        ResultCache,
        SamplingConfig,
        WorkspaceAnalyzer,
        WorkspaceConfig,
    )

    solver = _solver("numpy")
    analyzer = WorkspaceAnalyzer(
        solver, WorkspaceConfig(SamplingConfig(num_samples=7, batch_size=3))
    )
    progress = []
    analyzer.analyze(progress_callback=progress.append)
    np.testing.assert_allclose(progress, [3 / 7, 6 / 7, 1.0])
    cancel = threading.Event()

    def cancel_on_final(value):
        if value == 1.0:
            cancel.set()

    with pytest.raises(AnalysisCancelled):
        analyzer.analyze(
            cache=ResultCache(tmp_path),
            cancel_event=cancel,
            progress_callback=cancel_on_final,
        )
    assert not list(tmp_path.rglob("*.npz"))


def test_many_rescue_rounds_do_not_exhaust_the_call_stack():
    solver = _solver("numpy", max_iterations=1)
    target = np.eye(4)
    target[:3, 3] = [10, 0, 0]
    result = solver.inverse(
        target, position_only=True, rescue_restarts=1, rescue_rounds=1100
    )
    assert result.iterations == 1101
    assert not result.success
    np.testing.assert_allclose(
        result.residual,
        np.linalg.norm(solver.forward(result.positions)[:3, 3] - target[:3, 3]),
    )


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_rescue_retains_best_seeded_attempt_and_counts_all_iterations(backend):
    solver = _solver(backend, max_iterations=1, tolerance=1e-12)
    targets = solver.forward([[0.3, 0.7], [-0.4, 0.5]])
    attempts = [solver.inverse(targets, seed=[0, 0], position_only=True)]
    for offset in range(1, 4):
        attempts.append(
            solver.inverse(
                targets,
                seed=[0, 0],
                position_only=True,
                restarts=3,
                random_seed=solver.config.random_seed + offset,
            )
        )
    assert not any(bool(item.success.any()) for item in attempts)
    result = solver.inverse(
        targets,
        seed=[0, 0],
        position_only=True,
        rescue_restarts=3,
        rescue_rounds=3,
    )
    residuals = np.stack([_numpy(item.residual) for item in attempts])
    positions = np.stack([_numpy(item.positions) for item in attempts])
    best = np.argmin(residuals, axis=0)
    columns = np.arange(2)
    np.testing.assert_allclose(_numpy(result.residual), residuals[best, columns])
    np.testing.assert_allclose(_numpy(result.positions), positions[best, columns])
    assert result.iterations == sum(item.iterations for item in attempts)


def test_successful_ik_does_not_run_rescue_attempts():
    solver = _solver("numpy")
    q = [[0.3, 0.7]]
    result = solver.inverse(
        solver.forward(q), seed=q, rescue_restarts=4, rescue_rounds=50
    )
    assert result.iterations == 1
    assert result.success.all()


def test_torch_rescue_merge_preserves_autograd():
    torch = pytest.importorskip("torch")
    solver = _solver("torch", max_iterations=2)
    seed = torch.tensor(
        [[0.1, 0.2], [0.2, 0.3]], dtype=torch.float64, requires_grad=True
    )
    targets = solver.forward([[0.6, 0.9], [-0.7, 0.4]])
    result = solver.inverse(
        targets,
        seed=seed,
        position_only=True,
        rescue_restarts=3,
        rescue_rounds=3,
    )
    result.positions.sum().backward()
    assert torch.isfinite(seed.grad).all()
