import math

import pytest

torch = pytest.importorskip("torch")

from rage_mill.npole import (
    Chain,
    Plant1,
    accel1,
    alive1,
    pack_obs1,
    pd_action1,
    reset1,
    step_euler1,
    step_rk41,
    mass_matrix,
    _params,
)


def test_pack_obs_keys():
    s = torch.tensor([[0.1, 0.0, 0.2, 0.8]])
    o = pack_obs1(s, torch.tensor([0.5]))
    assert o.shape == (1, 6)
    assert abs(o[0, 2].item() - math.sin(0.2)) < 1e-6


def test_upright_zero_accel():
    s = torch.zeros(1, 4)
    d = accel1(s, torch.zeros(1), Plant1())
    assert torch.allclose(d, torch.zeros(1, 4), atol=1e-6)


def test_pd_holds_on_both_integrators():
    p = Plant1(delay=0)
    for step in (step_euler1, step_rk41):
        s = reset1(64, torch.device("cpu"), seed=3)
        ok = torch.ones(64, dtype=torch.bool)
        for _ in range(200):
            s = step(s, pd_action1(s), p)
            ok = ok & alive1(s)
        assert float(ok.float().mean()) >= 0.8, step.__name__


def test_random_policy_fails():
    p = Plant1()
    s = reset1(64, torch.device("cpu"), seed=3)
    g = torch.Generator().manual_seed(0)
    ok = torch.ones(64, dtype=torch.bool)
    for _ in range(200):
        a = torch.rand(64, generator=g) * 2 - 1
        s = step_euler1(s, a, p)
        ok = ok & alive1(s)
    assert float(ok.float().mean()) < 0.3


def test_n2_mass_spd():
    ch = Chain(n=2)
    q = torch.zeros(8, 3)
    q[:, 1] = 0.05
    m, L, I = _params(ch, q.device)
    M = mass_matrix(q, ch, m, L, I)
    eigs = torch.linalg.eigvalsh(M)
    assert torch.all(eigs > 0)


def test_chain_coriolis_matches_point_mass_centrifugal_forces():
    from rage_mill.npole import coriolis

    # Absolute-angle double pendulum, derived directly from each tip's
    # Cartesian acceleration. This exercises the nonzero-velocity terms
    # that an upright linearization cannot check.
    ch = Chain(n=2)
    q = torch.tensor([[0.2, 0.3, -0.4]], dtype=torch.float64)
    velocity = torch.tensor([[0.7, 1.7, -1.2]], dtype=torch.float64)
    mass = torch.full((2,), 0.1, dtype=torch.float64)
    length = torch.full((2,), 0.5, dtype=torch.float64)
    actual = coriolis(q, velocity, ch, mass, length, None)
    expected = torch.tensor([[
        -0.1 * math.sin(0.3) * 1.7**2 - 0.05 * math.sin(-0.4) * 1.2**2,
        0.025 * math.sin(0.7) * 1.2**2,
        -0.025 * math.sin(0.7) * 1.7**2,
    ]], dtype=torch.float64)
    # The mill intentionally approximates derivatives with finite differences.
    assert torch.allclose(actual, expected, atol=2e-5, rtol=1e-4)


def test_n2_lqr_holds_euler():
    from rage_mill.npole import lqr_k, lqr_action, rollout_n, step_euler_n

    ch = Chain(n=2)
    device = torch.device("cpu")
    K = lqr_k(ch, device)

    def pol(s, last):
        return lqr_action(s, K)

    r = rollout_n(step_euler_n, pol, 32, ch, device, seed=3, delay=0)
    assert r["success"] >= 0.8


def test_mlp_forward_zero_is_zero():
    from rage_mill.npole import mlp_forward, mlp_n_params, obs_dim

    nin = obs_dim(3)
    P, B = 4, 12
    theta = torch.zeros(P, mlp_n_params(nin))
    obs = torch.zeros(B, nin)
    idx = torch.arange(B) // 3
    u = mlp_forward(obs, theta, idx, nin)
    assert u.shape == (B,)
    assert torch.allclose(u, torch.zeros(B), atol=1e-6)
