"""Runtime contracts and independent physical/numerical checks."""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from rage_mill.baseline import ASSETS, ClassicalPolicy
from rage_mill.env import CartPendulumEnv, task_gates
from rage_mill.numpy_policy import NumpyEncoder, NumpyJEPAPolicy
from rage_mill.pole_control import PoleEnv, derivative, initial_state


def test_environment_encoder_and_baseline_import_without_loading_torch():
    script = """
import sys
from rage_mill.env import CartPendulumEnv
from rage_mill.baseline import ClassicalPolicy
from rage_mill.numpy_policy import NumpyJEPAPolicy
env = CartPendulumEnv(n=2, substeps=20)
env.reset_seed(123)
assert env.observation().shape == (10,)
assert env.step(.3).shape == (6,)
assert ClassicalPolicy().gains.shape == (900,22)
assert NumpyJEPAPolicy().encode_state([0.]*22).shape == (64,)
assert 'torch' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)


@pytest.mark.skipif(shutil.which("cc") is None, reason="optional compiler unavailable")
@pytest.mark.parametrize("n", [2, 6, 10])
def test_compiled_acceleration_matches_independent_numpy(n):
    from rage_mill.pole_cpu import acceleration
    rng = np.random.default_rng(2410000 + n)
    for _ in range(32):
        state = np.r_[rng.uniform(-.08, .08), rng.uniform(-np.pi, np.pi, n),
                      rng.uniform(-.02, .02), rng.uniform(-4., 4., n)]
        action = rng.uniform(-1.2, 1.2)
        np.testing.assert_allclose(acceleration(state, action, n), derivative(state, action, n)[n + 1:],
                                   rtol=2e-11, atol=2e-11)


@pytest.mark.skipif(shutil.which("cc") is None, reason="optional compiler unavailable")
@pytest.mark.parametrize("delay", [0, 1])
@pytest.mark.parametrize("substeps", [1, 20, 12800])
def test_compiled_ticks_match_numpy_and_actual_queue(delay, substeps):
    state = initial_state(10, "swingup", 2410010)
    fast = CartPendulumEnv(engine="euler", backend="c", delay=delay, substeps=substeps, initial_state=state)
    reference = PoleEnv(10, "euler", delay=delay, substeps=substeps, initial_state=state)
    for command in [.4, -.3, 1.2, -1.3]:
        np.testing.assert_allclose(fast.step(command), reference.step(command), rtol=2e-11, atol=2e-11)
        assert fast.pending == reference.pending
        assert fast.last_applied == reference.last_applied
        assert fast.ticks == reference.ticks


@pytest.mark.skipif(shutil.which("cc") is None, reason="optional compiler unavailable")
def test_compiled_acceleration_matches_independent_native_mujoco():
    pytest.importorskip("mujoco")
    from rage_mill.pole_cpu import acceleration
    state = np.r_[.03, np.linspace(-.12, .12, 10), .04, np.linspace(-.5, .5, 10)]
    env = CartPendulumEnv(engine="mujoco", delay=0, initial_state=state)
    env.data.ctrl[0] = .3
    env.mj.mj_forward(env.model, env.data)
    reference = np.r_[env.data.qacc[0], np.cumsum(env.data.qacc[1:])]
    np.testing.assert_allclose(acceleration(state, .3, 10), reference, rtol=3e-5, atol=8e-6)


def test_task_gates_check_all_251_endpoints_and_force_queue():
    states = np.zeros((1001, 22))
    commands = np.zeros(1000)
    applied = np.zeros(1000)
    assert task_gates(states, commands, applied)["pass"]
    states[750, 1] = .23
    assert not task_gates(states, commands, applied)["final_251_endpoints"]
    states[750, 1] = 0.
    commands[40] = .3
    assert not task_gates(states, commands, applied)["one_tick_queue"]
    applied[41] = .3
    assert task_gates(states, commands, applied)["pass"]
    states[20, 0] = 2.4
    assert not task_gates(states, commands, applied)["rail"]
    assert not task_gates(states[:-1], commands[:-1], applied[:-1])["pass"]


def test_inverse_recovers_state_and_rejects_impossible_latent_without_fallback():
    policy = NumpyJEPAPolicy()
    state = initial_state(10, "swingup", 2410200)
    latent = policy.encode_state(state, .2)
    decoded = policy.estimate_latent(latent, .2)
    np.testing.assert_allclose(decoded, state, atol=1e-11, rtol=0)
    assert policy.last_residual <= 1e-12
    assert policy.last_refinements < 20
    with pytest.raises(RuntimeError, match="JEPA inverse"):
        policy.estimate_latent(latent + 1000., .2)
    with pytest.raises(ValueError):
        policy.encode_state(np.full(22, np.nan))
    with pytest.raises(ValueError):
        policy.estimate_latent(np.zeros(63))


def test_encoder_rejects_corrupt_normalization(tmp_path):
    path = tmp_path / "bad.npz"
    with np.load(ASSETS / "npole10-target-encoder.npz", allow_pickle=False) as original:
        arrays = dict(original)
    arrays["std"] = np.zeros(33)
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="normalization"):
        NumpyEncoder(path)
    arrays["std"] = np.ones(33)
    arrays["mean"] = np.full(33, np.nan)
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="normalization"):
        NumpyEncoder(path)


def test_optional_gym_contract_and_seed_reproducibility():
    pytest.importorskip("gymnasium")
    from gymnasium.utils.env_checker import check_env
    from rage_mill.gym_env import CartPendulumGym
    env = CartPendulumGym(n=2, substeps=20)
    check_env(env, skip_render_check=True)
    first, _ = env.reset(seed=123)
    second, _ = env.reset(seed=123)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (10,)
    _, _, _, _, _ = env.step(np.array([.3]))
    assert env.plant.last_applied == 0. and env.plant.pending == .3
    with pytest.raises(ValueError):
        env.step(np.array([np.nan]))
    env.reset(seed=123, options={"state": np.r_[2.399, np.pi, np.pi, 2., 0., 0.]})
    _, _, terminated, _, info = env.step(np.array([0.]))
    assert terminated and not info["task_gates"]["pass"]
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.array([0.]))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="optional jepa extra unavailable")
def test_exported_encoder_and_feedback_match_original(tmp_path):
    pytest.importorskip("casadi")
    import torch
    from rage_mill.jepa_state_control import JEPAStatePolePolicy
    from rage_mill.pole_control import LQRBalance
    from rage_mill.swing_trajectory import TVLQRTracker
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    checkpoint = root / "artifacts/npole10/jepa/model.pt"
    classical = ClassicalPolicy()
    plan = classical.trajectory
    # Exercise the unchanged historical .act methods with published gains.
    # Avoid their constructors: the offline longdouble solve can fail on
    # 80-bit x86 even though the exported float64 coefficients run correctly.
    original_balance = LQRBalance.__new__(LQRBalance)
    original_balance.n, original_balance.delay = 10, 1
    original_balance.k = classical.balance_k.copy()
    original_teacher = TVLQRTracker.__new__(TVLQRTracker)
    original_teacher.trajectory, original_teacher.delay = plan, 1
    original_teacher.gains = classical.gains.copy()
    original_teacher.balance, original_teacher.tick = original_balance, 0
    np.testing.assert_array_equal(classical.gains, original_teacher.gains)
    np.testing.assert_array_equal(classical.balance_k, original_teacher.balance.k)
    original = JEPAStatePolePolicy(checkpoint, 10, teacher_factory=ClassicalPolicy,
                                   reference_plan=plan, iterations=20, residual_tolerance=1e-12)
    optimized = NumpyJEPAPolicy()
    for tick in [0, 1, 40, 200, 450, 750, 899, 900, 999]:
        state = plan.states[min(tick, 900)].copy()
        state += np.random.default_rng(2410300 + tick).uniform(-1e-3, 1e-3, 22)
        last, pending = .1, -.2
        classical.tick = original_teacher.tick = tick
        assert classical.act(state, last, pending) == original_teacher.act(state, last, pending)
        np.testing.assert_allclose(optimized.encode_state(state, last), original.encode_state(state, last),
                                   rtol=0, atol=5e-15)
        original.tick = optimized.tick = tick
        original.teacher.tick = optimized.teacher.tick = tick
        a, b = original.act(state, last, pending), optimized.act(state, last, pending)
        assert abs(a-b) < 1e-6
        np.testing.assert_allclose(original.last_estimate, optimized.last_estimate, rtol=0, atol=1e-11)
