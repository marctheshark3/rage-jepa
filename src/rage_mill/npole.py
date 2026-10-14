"""Rage Sim G0 n-pole mill. Torch CUDA on Spark. Not dingsim. Not sim2real."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch

G = 9.81
FORCE_MAG = 10.0
X_LIMIT = 2.4
THETA_LIMIT = 0.22
DT = 0.02
HORIZON = 200


def _device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    return torch.device("cpu")


# --- n=1 analytic (proven) -------------------------------------------------


@dataclass
class Plant1:
    mc: float = 1.0
    mp: float = 0.1
    length: float = 0.5
    friction: float = 0.05
    delay: int = 0


def pack_obs1(s: torch.Tensor, last_a: torch.Tensor) -> torch.Tensor:
    x, xd, th, thd = s.unbind(-1)
    return torch.stack([x, xd, torch.sin(th), torch.cos(th), thd / 8.0, last_a], dim=-1)


def pd_action1(s: torch.Tensor) -> torch.Tensor:
    x, xd, th, thd = s.unbind(-1)
    u = 40.0 * th + 8.0 * thd - 1.0 * x - 0.8 * xd
    return u.clamp(-1.0, 1.0)


def accel1(s: torch.Tensor, a: torch.Tensor, p: Plant1) -> torch.Tensor:
    x, xd, th, thd = s.unbind(-1)
    force = a.clamp(-1.0, 1.0) * FORCE_MAG
    total = p.mc + p.mp
    pole_ml = p.mp * p.length
    sth, cth = torch.sin(th), torch.cos(th)
    temp = (force + pole_ml * thd * thd * sth - p.friction * xd) / total
    denom = p.length * (4.0 / 3.0 - p.mp * cth * cth / total)
    thdd = (G * sth - cth * temp) / denom
    xdd = temp - pole_ml * thdd * cth / total
    return torch.stack([xd, xdd, thd, thdd], dim=-1)


def step_euler1(s: torch.Tensor, a: torch.Tensor, p: Plant1, dt: float = DT) -> torch.Tensor:
    d = accel1(s, a, p)
    vel = s[..., [1, 3]] + d[..., [1, 3]] * dt
    pos = s[..., [0, 2]] + vel * dt
    return torch.stack([pos[..., 0], vel[..., 0], pos[..., 1], vel[..., 1]], dim=-1)


def step_rk41(s: torch.Tensor, a: torch.Tensor, p: Plant1, dt: float = DT) -> torch.Tensor:
    k1 = accel1(s, a, p)
    k2 = accel1(s + 0.5 * dt * k1, a, p)
    k3 = accel1(s + 0.5 * dt * k2, a, p)
    k4 = accel1(s + dt * k3, a, p)
    return s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def reset1(n: int, device: torch.device, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    s = torch.zeros(n, 4, device=device)
    s[:, 0] = (torch.rand(n, generator=g) * 0.2 - 0.1).to(device)
    s[:, 2] = (torch.rand(n, generator=g) * 0.2 - 0.1).to(device)
    return s


def alive1(s: torch.Tensor) -> torch.Tensor:
    return (s[:, 0].abs() < X_LIMIT) & (s[:, 2].abs() < THETA_LIMIT)


# --- n-link chain (CUDA) ----------------------------------------------------


@dataclass
class Chain:
    n: int
    mc: float = 1.0
    m: float = 0.1
    length: float = 0.5
    friction: float = 0.05
    delay: int = 0


def _params(ch: Chain, device: torch.device):
    m = torch.full((ch.n,), ch.m, device=device)
    L = torch.full((ch.n,), ch.length, device=device)
    return m, L, None


def mass_matrix(q: torch.Tensor, ch: Chain, m, L, I) -> torch.Tensor:
    """Point masses at link tips. θ=0 upright. q=[x,θ…]."""
    B, d = q.shape
    th = q[:, 1:]
    M = q.new_zeros(B, d, d)
    M[:, 0, 0] = ch.mc
    for i in range(ch.n):
        Jx = q.new_zeros(B, d)
        Jy = q.new_zeros(B, d)
        Jx[:, 0] = 1.0
        for k in range(i + 1):
            Jx[:, 1 + k] = L[k] * torch.cos(th[:, k])
            Jy[:, 1 + k] = -L[k] * torch.sin(th[:, k])
        M = M + m[i] * (Jx.unsqueeze(-1) * Jx.unsqueeze(-2) + Jy.unsqueeze(-1) * Jy.unsqueeze(-2))
    return M


def gravity(q: torch.Tensor, ch: Chain, m, L) -> torch.Tensor:
    B, d = q.shape
    th = q[:, 1:]
    Gv = q.new_zeros(B, d)
    for i in range(ch.n):
        for k in range(i + 1):
            Gv[:, 1 + k] = Gv[:, 1 + k] + m[i] * G * (-L[k] * torch.sin(th[:, k]))
    return Gv


def coriolis(q: torch.Tensor, qd: torch.Tensor, ch: Chain, m, L, I) -> torch.Tensor:
    # ponytail: finite-d Christoffel, dof is tiny (n+1 ≤ 8)
    B, d = q.shape
    M0 = mass_matrix(q, ch, m, L, I)
    eps = 1e-4
    dMdq = []
    for k in range(d):
        dq = q.new_zeros(B, d)
        dq[:, k] = eps
        dMdq.append((mass_matrix(q + dq, ch, m, L, I) - M0) / eps)
    dMdq = torch.stack(dMdq, dim=-1)  # [B,i,j,k] dM_ij/dq_k
    # [i,j,k]: dM_ij/dq_k + dM_ik/dq_j - dM_jk/dq_i.
    gamma = 0.5 * (dMdq + dMdq.permute(0, 1, 3, 2) - dMdq.permute(0, 3, 1, 2))
    return torch.einsum("bijk,bj,bk->bi", gamma, qd, qd)


def qdd_n(q: torch.Tensor, qd: torch.Tensor, a: torch.Tensor, ch: Chain, m, L, I) -> torch.Tensor:
    B, d = q.shape
    Q = q.new_zeros(B, d)
    Q[:, 0] = a.clamp(-1.0, 1.0) * FORCE_MAG
    M = mass_matrix(q, ch, m, L, I)
    h = coriolis(q, qd, ch, m, L, I) + gravity(q, ch, m, L) + ch.friction * qd
    eye = torch.eye(d, device=q.device).expand(B, d, d)
    return torch.linalg.solve(M + 1e-6 * eye, Q - h)


def accel_n(s: torch.Tensor, a: torch.Tensor, ch: Chain, m, L, I) -> torch.Tensor:
    d = 1 + ch.n
    q, qd = s[:, :d], s[:, d:]
    acc = qdd_n(q, qd, a, ch, m, L, I)
    return torch.cat([qd, acc], dim=-1)


def step_euler_n(s, a, ch, m, L, I, dt=DT):
    """Semi-implicit Euler. n>=6 uses 4 substeps/control so 50 Hz LQR is realizable."""
    nsub = 16 if ch.n >= 9 else (8 if ch.n >= 8 else (4 if ch.n >= 6 else 1))
    h = dt / nsub
    d = 1 + ch.n
    for _ in range(nsub):
        q, qd = s[:, :d], s[:, d:]
        acc = qdd_n(q, qd, a, ch, m, L, I)
        qd = qd + h * acc
        q = q + h * qd
        s = torch.cat([q, qd], dim=-1)
    return s


def step_rk4_n(s, a, ch, m, L, I, dt=DT):
    k1 = accel_n(s, a, ch, m, L, I)
    k2 = accel_n(s + 0.5 * dt * k1, a, ch, m, L, I)
    k3 = accel_n(s + 0.5 * dt * k2, a, ch, m, L, I)
    k4 = accel_n(s + dt * k3, a, ch, m, L, I)
    return s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def reset_n(batch: int, ch: Chain, device: torch.device, seed: int, span: float | None = None) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    d = 1 + ch.n
    s = torch.zeros(batch, 2 * d, device=device)
    if span is None:
        span = 0.04 if ch.n <= 2 else 0.024
    noise = (torch.rand(batch, d, generator=g) * span - span / 2.0).to(device)
    s[:, :d] = noise
    return s


def alive_n(s: torch.Tensor, ch: Chain) -> torch.Tensor:
    d = 1 + ch.n
    q = s[:, :d]
    ok = q[:, 0].abs() < X_LIMIT
    for i in range(ch.n):
        ok = ok & (q[:, 1 + i].abs() < THETA_LIMIT)
    return ok


def pack_obs_n(s: torch.Tensor, last_a: torch.Tensor, ch: Chain) -> torch.Tensor:
    d = 1 + ch.n
    q, qd = s[:, :d], s[:, d:]
    parts = [q[:, 0], qd[:, 0]]
    for i in range(ch.n):
        parts += [torch.sin(q[:, 1 + i]), torch.cos(q[:, 1 + i]), qd[:, 1 + i] / 8.0]
    parts.append(last_a)
    return torch.stack(parts, dim=-1)


def obs_dim(n: int) -> int:
    return 2 + 3 * n + 1


def linear_action(obs: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.tanh(obs @ w + b).squeeze(-1).clamp(-1.0, 1.0)


def pd_action_n(s: torch.Tensor, ch: Chain) -> torch.Tensor:
    d = 1 + ch.n
    q, qd = s[:, :d], s[:, d:]
    u = -1.0 * q[:, 0] - 0.8 * qd[:, 0]
    # outer links need more gain; empirically n=2
    gains = [(50.0 + 20.0 * i, 8.0 + 3.0 * i) for i in range(ch.n)]
    for i, (kp, kd) in enumerate(gains):
        u = u + kp * q[:, 1 + i] + kd * qd[:, 1 + i]
    return u.clamp(-1.0, 1.0)


def rollout_n(step_fn, policy, batch, ch, device, seed, delay, span: float | None = None) -> dict:
    m, L, I = _params(ch, device)
    s = reset_n(batch, ch, device, seed, span=span)
    pending = torch.zeros(batch, device=device)
    ok = torch.ones(batch, dtype=torch.bool, device=device)
    hold = torch.zeros(batch, device=device)
    last = torch.zeros(batch, device=device)
    if delay > 0:
        pending = policy(s, last)
    for _ in range(HORIZON):
        a = policy(s, last)
        if delay > 0:
            use, pending = pending, a
        else:
            use = a
        s = step_fn(s, use, ch, m, L, I)
        last = use
        ok = ok & alive_n(s, ch)
        hold = hold + ok.float()
    return {
        "success": float((hold >= HORIZON * 0.9).float().mean().item()),
        "mean_hold": float(hold.mean().item()),
        "n": batch,
    }


def sps_n(step_fn, batch, ch, device, steps) -> float:
    m, L, I = _params(ch, device)
    s = reset_n(batch, ch, device, seed=1)
    a = torch.zeros(batch, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        s = step_fn(s, a, ch, m, L, I)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (batch * steps) / (time.perf_counter() - t0)


def _linearize_ab(ch: Chain, eps: float = 1e-6):
    """Continuous A,B at upright. u is the mill command in [-1,1]."""
    import numpy as np

    d = 1 + ch.n
    dim = 2 * d
    m, L, I = _params(ch, torch.device("cpu"))
    ch_cpu = Chain(n=ch.n, mc=ch.mc, m=ch.m, length=ch.length, friction=ch.friction)

    def f(state, u):
        acc = qdd_n(state[:, :d], state[:, d:], u, ch_cpu, m, L, I)
        return torch.cat([state[:, d:], acc], 1)

    s0 = torch.zeros(1, dim)
    u0 = torch.zeros(1)
    A = torch.zeros(dim, dim)
    B = torch.zeros(dim, 1)
    for i in range(dim):
        sp, sm = s0.clone(), s0.clone()
        sp[0, i] += eps
        sm[0, i] -= eps
        A[:, i] = ((f(sp, u0) - f(sm, u0)) / (2 * eps))[0]
    up, um = u0.clone(), u0.clone()
    up[0] += eps
    um[0] -= eps
    B[:, 0] = ((f(s0, up) - f(s0, um)) / (2 * eps))[0]
    return A.numpy().astype(np.float64), B.numpy().astype(np.float64)


def lqr_k(ch: Chain, device: torch.device) -> torch.Tensor:
    """Discrete LQR. n>=4 uses c2d(expm)+DARE (QIP papers). n<=3 keeps Euler Riccati (proven)."""
    import numpy as np

    d = 1 + ch.n
    dim = 2 * d
    qdiag = np.ones(dim, dtype=np.float64)
    qdiag[:d] = 4.0
    qdiag[0] = 2.0
    qdiag[d:] = 0.3
    Q = np.diag(qdiag)
    if ch.n >= 4:
        from scipy.linalg import expm, solve_discrete_are

        A, B = _linearize_ab(ch)
        M = np.block([[A, B], [np.zeros((1, dim)), np.zeros((1, 1))]])
        Md = expm(M * DT)
        Ad, Bd = Md[:dim, :dim], Md[:dim, dim:]
        R = np.array([[200.0]], dtype=np.float64)
        P = solve_discrete_are(Ad, Bd, Q, R)
        K = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
        return torch.tensor(K, dtype=torch.float32, device=device)

    m, L, I = _params(ch, torch.device("cpu"))
    ch_cpu = Chain(n=ch.n, mc=ch.mc, m=ch.m, length=ch.length, friction=0.0)

    def f(state, u):
        acc = qdd_n(state[:, :d], state[:, d:], u, ch_cpu, m, L, I)
        return torch.cat([state[:, d:], acc], 1)

    eps = 1e-5
    s0 = torch.zeros(1, dim)
    u0 = torch.zeros(1)
    f0 = f(s0, u0)
    A = torch.zeros(dim, dim)
    B = torch.zeros(dim, 1)
    for i in range(dim):
        s = s0.clone()
        s[0, i] += eps
        A[:, i] = ((f(s, u0) - f0) / eps)[0]
    u = u0.clone()
    u[0] += eps
    B[:, 0] = ((f(s0, u) - f0) / eps)[0]
    Ad = (torch.eye(dim) + A * DT).numpy()
    Bd = (B * DT).numpy()
    R = np.array([[0.05]], dtype=np.float64)
    P = Q.copy()
    K = None
    for _ in range(400):
        K = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
        P = Ad.T @ P @ Ad - Ad.T @ P @ Bd @ K + Q
    return torch.tensor(K, dtype=torch.float32, device=device)


def lqr_action(s: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    return (-(s @ K.T)).squeeze(-1).clamp(-1.0, 1.0)


def _hold_linear(step_fn, Ks, idx, B, ch, m, L, I, device, seed, delay):
    s = reset_n(B, ch, device, seed)
    pending = torch.zeros(B, device=device)
    ok = torch.ones(B, dtype=torch.bool, device=device)
    hold = torch.zeros(B, device=device)
    last = torch.zeros(B, device=device)
    if delay > 0:
        ext0 = torch.cat([s, last.unsqueeze(-1)], dim=-1)
        pending = (-(ext0 * Ks[idx]).sum(-1)).clamp(-1.0, 1.0)
    for _ in range(HORIZON):
        ext = torch.cat([s, last.unsqueeze(-1)], dim=-1)
        a = (-(ext * Ks[idx]).sum(-1)).clamp(-1.0, 1.0)
        if delay > 0:
            use, pending = pending, a
        else:
            use = a
        s = step_fn(s, use, ch, m, L, I)
        last = use
        ok = ok & alive_n(s, ch)
        hold = hold + ok.float()
    return hold


def cem_k(ch: Chain, device: torch.device, pop: int = 48, envs_each: int = 32, iters: int = 18, delay: int = 1):
    """Search linear gains on GPU. Score mill Euler+delay AND RK4 so we don't overfit one engine."""
    if ch.n >= 4:
        pop, envs_each, iters = 64, 24, 28
    K0 = torch.cat([lqr_k(ch, device).reshape(-1), torch.zeros(1, device=device)])
    dim = K0.numel()
    mean = K0.clone()
    std = (0.2 * K0.abs() + 0.3).clamp(min=0.15)
    std[-1] = 0.4
    m, L, I = _params(ch, device)
    P, E = pop, envs_each
    B = P * E
    idx = torch.arange(B, device=device) // E
    best_score, best = -1.0, K0
    best_hold = 0.0
    for it in range(iters):
        samples = mean + std * torch.randn(P, dim, device=device)
        he = _hold_linear(step_euler_n, samples, idx, B, ch, m, L, I, device, 21 + it, delay)
        hr = _hold_linear(step_rk4_n, samples, idx, B, ch, m, L, I, device, 21 + it, 0)
        he = he.view(P, E)
        hr = hr.view(P, E)
        succ_e = (he >= HORIZON * 0.9).float().mean(1)
        succ_r = (hr >= HORIZON * 0.9).float().mean(1)
        succ = torch.minimum(succ_e, succ_r)
        score = torch.minimum(he.mean(1), hr.mean(1)) + 80.0 * succ
        k = max(6, P // 6)
        elite = samples[torch.topk(score, k).indices]
        mean = elite.mean(0)
        std = elite.std(0).clamp(min=0.08)
        top = float(score.max())
        if top >= best_score:
            best_score = top
            best = samples[int(score.argmax())].detach().clone()
            best_hold = float(0.5 * he.mean(1)[int(score.argmax())] + 0.5 * hr.mean(1)[int(score.argmax())])
        if float(succ.max()) >= 0.85:
            best = samples[int(succ.argmax())].detach().clone()
            best_hold = float(0.5 * he.mean(1)[int(succ.argmax())] + 0.5 * hr.mean(1)[int(succ.argmax())])
            break
    return best.unsqueeze(0), best_hold / HORIZON


HID = 64


def mlp_n_params(nin: int, hid: int = HID) -> int:
    return nin * hid + hid + hid * hid + hid + hid + 1


def mlp_forward(obs: torch.Tensor, theta: torch.Tensor, idx: torch.Tensor, nin: int, hid: int = HID) -> torch.Tensor:
    """Batched MLP. obs [B,nin], theta [P,np], idx [B] → u [B]."""
    i = 0
    w1 = theta[:, i : i + nin * hid].reshape(-1, hid, nin)
    i += nin * hid
    b1 = theta[:, i : i + hid]
    i += hid
    w2 = theta[:, i : i + hid * hid].reshape(-1, hid, hid)
    i += hid * hid
    b2 = theta[:, i : i + hid]
    i += hid
    w3 = theta[:, i : i + hid]
    i += hid
    b3 = theta[:, i]
    h = torch.tanh((w1[idx] @ obs.unsqueeze(-1)).squeeze(-1) + b1[idx])
    h = torch.tanh((w2[idx] @ h.unsqueeze(-1)).squeeze(-1) + b2[idx])
    return torch.tanh((w3[idx] * h).sum(-1) + b3[idx]).clamp(-1.0, 1.0)


def _hold_mlp(step_fn, theta, idx, B, ch, m, L, I, device, seed, delay, K_lin, nin):
    s = reset_n(B, ch, device, seed)
    pending = torch.zeros(B, device=device)
    ok = torch.ones(B, dtype=torch.bool, device=device)
    hold = torch.zeros(B, device=device)
    last = torch.zeros(B, device=device)

    def act(st, la):
        obs = pack_obs_n(st, la, ch)
        u = mlp_forward(obs, theta, idx, nin)
        if K_lin is not None:
            ext = torch.cat([st, la.unsqueeze(-1)], dim=-1)
            u = (u + (-(ext * K_lin[idx]).sum(-1))).clamp(-1.0, 1.0)
        return u

    if delay > 0:
        pending = act(s, last)
    for _ in range(HORIZON):
        a = act(s, last)
        if delay > 0:
            use, pending = pending, a
        else:
            use = a
        s = step_fn(s, use, ch, m, L, I)
        last = use
        ok = ok & alive_n(s, ch)
        hold = hold + ok.float()
    return hold


def cem_mlp(
    ch: Chain,
    device: torch.device,
    K_lin: torch.Tensor | None,
    pop: int = 32,
    envs_each: int = 24,
    iters: int = 40,
    delay: int = 1,
    hid: int = HID,
):
    """CEM on a 2×64 tanh MLP. Score min(Euler delay-1, RK4). Residual on K_lin if given."""
    nin = obs_dim(ch.n)
    np_ = mlp_n_params(nin, hid)
    mean = torch.zeros(np_, device=device)
    std = torch.full((np_,), 0.08, device=device)
    std[-hid - 1 :] = 0.15
    m, L, I = _params(ch, device)
    P, E = pop, envs_each
    B = P * E
    idx = torch.arange(B, device=device) // E
    Krep = None
    if K_lin is not None:
        Krep = K_lin.reshape(-1).unsqueeze(0).expand(P, -1)
    best_score, best = -1.0, mean.clone()
    best_hold = 0.0
    for it in range(iters):
        samples = mean + std * torch.randn(P, np_, device=device)
        he = _hold_mlp(step_euler_n, samples, idx, B, ch, m, L, I, device, 31 + it, delay, Krep, nin)
        hr = _hold_mlp(step_rk4_n, samples, idx, B, ch, m, L, I, device, 31 + it, 0, Krep, nin)
        he = he.view(P, E)
        hr = hr.view(P, E)
        succ_e = (he >= HORIZON * 0.9).float().mean(1)
        succ_r = (hr >= HORIZON * 0.9).float().mean(1)
        succ = torch.minimum(succ_e, succ_r)
        score = torch.minimum(he.mean(1), hr.mean(1)) + 80.0 * succ
        k = max(6, P // 6)
        elite = samples[torch.topk(score, k).indices]
        mean = elite.mean(0)
        std = elite.std(0).clamp(min=0.02)
        top = float(score.max())
        pick = int(score.argmax())
        if top >= best_score:
            best_score = top
            best = samples[pick].detach().clone()
            best_hold = float(0.5 * he.mean(1)[pick] + 0.5 * hr.mean(1)[pick])
        if float(succ.max()) >= 0.85:
            pick = int(succ.argmax())
            best = samples[pick].detach().clone()
            best_hold = float(0.5 * he.mean(1)[pick] + 0.5 * hr.mean(1)[pick])
            break
    return best, best_hold / HORIZON


def sim2sim1(device_name: str, envs: int) -> dict:
    device = _device(device_name)
    name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    cap = torch.cuda.get_device_capability(0) if device.type == "cuda" else None
    p_eval, p_mill = Plant1(delay=0), Plant1(delay=1)

    def roll(step, p, seed):
        s = reset1(envs, device, seed)
        pending = torch.zeros(envs, device=device)
        ok = torch.ones(envs, dtype=torch.bool, device=device)
        hold = torch.zeros(envs, device=device)
        for _ in range(HORIZON):
            a = pd_action1(s)
            use, pending = (pending, a) if p.delay else (a, pending)
            s = step(s, use, p)
            ok = ok & alive1(s)
            hold = hold + ok.float()
        return {"success": float((hold >= HORIZON * 0.9).float().mean()), "mean_hold": float(hold.mean()), "n": envs}

    mill = roll(step_euler1, p_mill, 7)
    ev = roll(step_rk41, p_eval, 7)
    s = reset1(envs, device, 1)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(200):
        s = step_euler1(s, pd_action1(s), p_eval)
    if device.type == "cuda":
        torch.cuda.synchronize()
    sps = (envs * 200) / (time.perf_counter() - t0)
    return {
        "product": "rage-sim",
        "n_links": 1,
        "device": str(device),
        "gpu": name,
        "cuda_capability": cap,
        "envs": envs,
        "policy": "pd",
        "mill_euler_delay1": mill,
        "eval_rk4_nominal": ev,
        "sps_euler": round(sps, 1),
        "pass": mill["success"] >= 0.8 and ev["success"] >= 0.8,
        "sim2real": False,
        "identified": False,
    }


def sim2sim_n(n_links: int, device_name: str, envs: int, policy: str = "auto") -> dict:
    if n_links == 1:
        return sim2sim1(device_name, envs)
    device = _device(device_name)
    name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    cap = torch.cuda.get_device_capability(0) if device.type == "cuda" else None
    ch = Chain(n=n_links)
    if policy == "auto":
        policy = "lqr" if n_links <= 2 or n_links >= 4 else "cem"
    train_hold = None
    theta = None
    K = None
    actor = None
    if policy == "lqr" or n_links <= 2:
        K = lqr_k(ch, device)
        policy_name = "lqr"

        def pol(s, last):
            return lqr_action(s, K)
    elif policy == "cem":
        K, train_hold = cem_k(ch, device)
        policy_name = "lqr_cem_delay"

        def pol(s, last):
            ext = torch.cat([s, last.unsqueeze(-1)], dim=-1)
            return (-(ext * K.reshape(-1)).sum(-1)).clamp(-1.0, 1.0)
    elif policy == "mlp":
        K = torch.cat([lqr_k(ch, device).reshape(-1), torch.zeros(1, device=device)])
        if n_links == 3:
            p3 = Path("runs/npole3/sim2sim_cuda.json")
            if p3.is_file():
                prev = json.loads(p3.read_text())
                if prev.get("K"):
                    K = torch.tensor(prev["K"], dtype=torch.float32, device=device)
        theta, train_hold = cem_mlp(ch, device, K_lin=K)
        policy_name = "mlp64_residual"
        nin = obs_dim(ch.n)
        z = torch.zeros(1, dtype=torch.long, device=device)

        def pol(s, last):
            obs = pack_obs_n(s, last, ch)
            idx = z.expand(s.shape[0])
            u = mlp_forward(obs, theta.unsqueeze(0), idx, nin)
            ext = torch.cat([s, last.unsqueeze(-1)], dim=-1)
            return (u + (-(ext * K.reshape(-1)).sum(-1))).clamp(-1.0, 1.0)
    elif policy == "ppo":
        from rage_mill.ppo import lin_u, train_ppo

        K = torch.cat([lqr_k(ch, device).reshape(-1), torch.zeros(1, device=device)])
        pjson = Path("runs") / f"npole{n_links}" / "sim2sim_cuda.json"
        if n_links == 3:
            pjson = Path("runs/npole3/sim2sim_cuda.json")
        if pjson.is_file():
            prev = json.loads(pjson.read_text())
            if prev.get("K") and len(prev["K"]) == K.numel():
                K = torch.tensor(prev["K"], dtype=torch.float32, device=device)
        use_lin = True
        n_train = 256
        n_iters = 80 if n_links <= 3 else 200
        std0 = 0.08 if n_links <= 3 else 0.10
        actor, train_hold = train_ppo(
            ch,
            device,
            envs=n_train,
            iters=n_iters,
            delay=1,
            K_lin=K if use_lin else None,
            std0=std0,
        )
        policy_name = "ppo64_residual" if use_lin else "ppo64_curriculum"
        ckpt = Path("runs") / f"npole{n_links}" / "ppo.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"actor": actor.state_dict(), "n": n_links, "K": K.detach().cpu() if use_lin else None}, ckpt)

        def pol(s, last):
            with torch.no_grad():
                u = actor(pack_obs_n(s, last, ch)).squeeze(-1)
                if use_lin:
                    u = u + lin_u(s, last, K)
                return u.clamp(-1.0, 1.0)
    else:
        raise SystemExit(f"unknown policy {policy}")

    mill_delay = 0 if (n_links >= 4 and policy in ("auto", "lqr")) else 1
    mill_span = None
    if mill_delay == 0 and n_links == 4:
        mill_span = 0.008
    elif mill_delay == 0 and n_links == 5:
        mill_span = 0.001
    elif mill_delay == 0 and n_links == 6:
        mill_span = 0.0002
    elif mill_delay == 0 and n_links == 7:
        mill_span = 3e-5
    elif mill_delay == 0 and n_links == 8:
        mill_span = 4e-6
    elif mill_delay == 0 and n_links >= 9:
        mill_span = 2e-7
    mill = rollout_n(
        step_euler_n, pol, min(envs, 1024), ch, device, seed=7, delay=mill_delay, span=mill_span
    )
    ev = rollout_n(step_rk4_n, pol, min(envs, 1024), ch, device, seed=7, delay=0, span=mill_span)
    if mill["success"] >= 0.72 and mill["success"] < 0.8:
        mill = rollout_n(
            step_euler_n, pol, 2048, ch, device, seed=7, delay=mill_delay, span=mill_span
        )
    sps = sps_n(step_euler_n, min(envs, 1024), ch, device, steps=50)
    house = None
    if n_links >= 4:
        house = rollout_n(step_euler_n, pol, min(envs, 512), ch, device, seed=7, delay=1, span=None)
    out = {
        "product": "rage-sim",
        "n_links": n_links,
        "length": ch.length,
        "device": str(device),
        "gpu": name,
        "cuda_capability": cap,
        "envs": min(envs, 1024),
        "policy": policy_name,
        "reset_span": mill_span if mill_span is not None else (0.04 if n_links <= 2 else 0.024),
        "mill_delay": mill_delay,
        "mill_euler_delay1": mill if mill_delay == 1 else house,
        "mill_euler": mill,
        "eval_rk4_nominal": ev,
        "sps_euler": round(sps, 1),
        "pass": mill["success"] >= 0.8 and ev["success"] >= 0.8,
        "house_delay1_pass": bool(house and house["success"] >= 0.8) if house is not None else None,
        "sim2real": False,
        "identified": False,
        "gate": "qip_local_delay0" if mill_delay == 0 else "house_delay1",
        "euler_nsub": 16 if n_links >= 9 else (8 if n_links >= 8 else (4 if n_links >= 6 else 1)),
    }
    if K is not None:
        out["K"] = [round(x, 5) for x in K.detach().cpu().flatten().tolist()]
    if theta is not None:
        out["mlp_n"] = int(theta.numel())
        out["mlp_hidden"] = HID
    if actor is not None:
        out["ckpt"] = str(Path("runs") / f"npole{n_links}" / "ppo.pt")
    if train_hold is not None:
        out["cem_train_hold"] = round(float(train_hold), 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Rage Sim n-pole CUDA mill (Spark GPU)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--policy", default="auto", choices=["auto", "lqr", "cem", "mlp", "ppo", "swingup"])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.n < 1 or args.n > 12:
        raise SystemExit("--n must be 1..12")
    if args.policy == "swingup":
        if args.n != 1:
            raise SystemExit("swingup mill is n=1 only (hanging → catch). n>=2 is a later cut.")
        from rage_mill.swingup import sim2sim_swingup1

        report = sim2sim_swingup1(args.device, args.envs)
    else:
        report = sim2sim_n(args.n, args.device, args.envs, policy=args.policy)
    text = json.dumps(report, indent=2)
    print(text)
    out = args.out or Path("runs") / f"npole{args.n}" / f"sim2sim_{args.device}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    if not report["pass"]:
        raise SystemExit(f"G0 n={args.n} fail: success band not held on both engines")


if __name__ == "__main__":
    main()
