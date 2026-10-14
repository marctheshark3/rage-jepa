"""Frozen closed-loop pole policies and honest CPU cross-engine evaluation.

All engines expose [x, absolute theta..., x velocity, absolute omega...].
The one-pole plant is the existing uniform rod; longer chains use the mill's
point masses. Actions are normalized cart forces, held for one 50 Hz tick.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from typing import Callable

import numpy as np

CONTROL_DT = 0.02
RAIL_LIMIT = 2.4
ANGLE_LIMIT = 0.22
OMEGA_LIMIT = 1.0
ENGINES = ("euler", "rk4", "mujoco")
MAX_POLES = 10


def _pole_count(n: int) -> None:
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or not 1 <= n <= MAX_POLES:
        raise ValueError(f"n must be an integer in 1..{MAX_POLES}")


def wrap_angles(theta: np.ndarray) -> np.ndarray:
    return (np.asarray(theta) + np.pi) % (2 * np.pi) - np.pi


def initial_state(n: int, task: str, seed: int) -> np.ndarray:
    """Identical seeded physical initial state for every engine and policy."""
    _pole_count(n)
    if task not in ("balance", "swingup"):
        raise ValueError("task must be balance or swingup")
    rng = np.random.default_rng(seed)
    state = np.zeros(2 * (n + 1), dtype=np.float64)
    state[0] = rng.uniform(-0.08, 0.08)
    span = 0.04 if task == "swingup" else 0.02
    state[1:n + 1] = rng.uniform(-span, span, n)
    if task == "swingup":
        state[1:n + 1] += np.pi
    state[n + 1:] = rng.uniform(-0.02, 0.02, n + 1)
    return state


def derivative(state: np.ndarray, action: float, n: int) -> np.ndarray:
    """Exact analytic form of Plant1 / Chain mechanics in absolute coordinates.

    Chain's finite difference Christoffel terms reduce to the sine terms here.
    Its 1e-6 mass regularization is retained. No contacts or rail collisions.
    """
    state = np.asarray(state, dtype=np.float64)
    d = n + 1
    q, vel = state[:d], state[d:]
    force = np.clip(action, -1.0, 1.0) * 10.0
    if n == 1:
        theta, omega, xd = q[1], vel[1], vel[0]
        sine, cosine = np.sin(theta), np.cos(theta)
        temporary = (force + 0.05 * omega**2 * sine - 0.05 * xd) / 1.1
        thdd = (9.81 * sine - cosine * temporary) / (0.5 * (4 / 3 - 0.1 * cosine**2 / 1.1))
        xdd = temporary - 0.05 * thdd * cosine / 1.1
        return np.r_[vel, xdd, thdd]
    theta, omega = q[1:], vel[1:]
    remaining = 0.1 * (n - np.arange(n))
    links = remaining[np.maximum.outer(np.arange(n), np.arange(n))]
    difference = theta[:, None] - theta[None, :]
    mass = np.zeros((d, d), dtype=np.float64)
    mass[0, 0] = 1 + 0.1 * n
    mass[0, 1:] = mass[1:, 0] = 0.5 * remaining * np.cos(theta)
    mass[1:, 1:] = 0.25 * links * np.cos(difference)
    centrifugal = np.r_[-np.sum(0.5 * remaining * np.sin(theta) * omega**2),
                        (0.25 * links * np.sin(difference)) @ (omega**2)]
    gravity = np.r_[0.0, -9.81 * 0.5 * remaining * np.sin(theta)]
    applied = np.r_[force, np.zeros(n)]
    acceleration = np.linalg.solve(mass + 1e-6 * np.eye(d), applied - centrifugal - gravity - 0.05 * vel)
    return np.r_[vel, acceleration]


def analytic_step(state: np.ndarray, action: float, n: int, engine: str, dt: float) -> np.ndarray:
    if engine == "euler":
        d = n + 1
        velocity = state[d:] + dt * derivative(state, action, n)[d:]
        return np.r_[state[:d] + dt * velocity, velocity]
    if engine != "rk4":
        raise ValueError("analytic engine must be euler or rk4")
    k1 = derivative(state, action, n)
    k2 = derivative(state + 0.5 * dt * k1, action, n)
    k3 = derivative(state + 0.5 * dt * k2, action, n)
    k4 = derivative(state + dt * k3, action, n)
    return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def apply_mujoco_friction(data, n: int) -> None:
    """Apply absolute-coordinate viscous forces in relative hinge coordinates."""
    data.qfrc_applied[:] = 0
    data.qfrc_applied[0] = -0.05 * data.qvel[0]
    if n > 1:
        absolute_velocity = np.cumsum(data.qvel[1:])
        data.qfrc_applied[1:] = np.cumsum((-0.05 * absolute_velocity)[::-1])[::-1]


def _native_damping_xml(n: int) -> str:
    """Same viscous law, evaluated inside each MuJoCo integrator stage.

    Fixed tendon i sums relative hinges h1..hi, so its velocity is absolute
    omega_i and its damping force transforms back into relative coordinates.
    Plant1 has cart damping only, with no pole hinge damping.
    """
    import xml.etree.ElementTree as ET
    from .jepa_data import _npole_xml

    tree = ET.fromstring(_npole_xml(n))
    tree.find(".//joint[@name='slide']").set("damping", "0.05")
    if n > 1:
        tendons = ET.SubElement(tree, "tendon")
        for i in range(1, n + 1):
            tendon = ET.SubElement(tendons, "fixed", name=f"absolute_angle_{i}", damping="0.05")
            for j in range(1, i + 1):
                ET.SubElement(tendon, "joint", joint=f"h{j}", coef="1")
    return ET.tostring(tree, encoding="unicode")


class PoleEnv:
    """CPU plant with a real optional one-tick actuator queue.

    ``last_applied`` is the force used for the most recent transition;
    ``pending`` is the already-queued force for the next transition. The queue
    starts at zero. Controllers with delay must use pending, not last_applied.
    """

    def __init__(self, n: int, engine: str = "euler", delay: int = 1,
                 substeps: int = 4, initial_state: np.ndarray | None = None,
                 mujoco_native_damping: bool = False):
        _pole_count(n)
        if engine not in ENGINES:
            raise ValueError("engine must be euler, rk4 or mujoco")
        if delay not in (0, 1) or isinstance(delay, bool):
            raise ValueError("delay must be 0 or 1")
        if not isinstance(substeps, int) or isinstance(substeps, bool) or substeps < 1:
            raise ValueError("substeps must be a positive integer")
        if not isinstance(mujoco_native_damping, bool):
            raise ValueError("mujoco_native_damping must be boolean")
        self.n, self.engine, self.delay, self.substeps = n, engine, delay, substeps
        self.mujoco_native_damping = engine == "mujoco" and mujoco_native_damping
        self.numerical_damping_mode = ("native_joint_and_fixed_tendon_absolute_viscous"
                                       if self.mujoco_native_damping else
                                       "substep_held_external_absolute_viscous" if engine == "mujoco"
                                       else "analytic_absolute_viscous")
        self.last_applied = self.pending = 0.0
        self.ticks = 0
        self.model = self.data = self.mj = None
        self.xml_hash = None
        if engine == "mujoco":
            from .jepa_data import _mujoco, _npole_xml
            self.mj = _mujoco()
            xml = _native_damping_xml(n) if self.mujoco_native_damping else _npole_xml(n)
            self.xml_hash = hashlib.sha256(xml.encode()).hexdigest()
            self.model = self.mj.MjModel.from_xml_string(xml)
            self.model.opt.timestep = CONTROL_DT / substeps
            self.data = self.mj.MjData(self.model)
        self.reset(np.zeros(2 * (n + 1)) if initial_state is None else initial_state)

    @property
    def state(self) -> np.ndarray:
        if self.engine == "mujoco":
            return np.r_[self.data.qpos[0], np.cumsum(self.data.qpos[1:]),
                         self.data.qvel[0], np.cumsum(self.data.qvel[1:])]
        return self._state.copy()

    def reset(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (2 * (self.n + 1),) or not np.isfinite(state).all():
            raise ValueError("initial state must be a finite vector of length 2*(n+1)")
        self._state = state.copy()
        self.last_applied = self.pending = 0.0
        self.ticks = 0
        if self.engine == "mujoco":
            self.mj.mj_resetData(self.model, self.data)
            d = self.n + 1
            self.data.qpos[0], self.data.qvel[0] = state[0], state[d]
            self.data.qpos[1:] = np.diff(np.r_[0., state[1:d]])
            self.data.qvel[1:] = np.diff(np.r_[0., state[d + 1:]])
            self.mj.mj_forward(self.model, self.data)
        return self.state

    def step(self, action: float) -> np.ndarray:
        value = np.asarray(action)
        if value.size != 1 or not np.isfinite(value).all():
            raise ValueError("action must be one finite normalized cart force")
        action = float(np.clip(value.reshape(-1)[0], -1.0, 1.0))
        if self.delay:
            applied, self.pending = self.pending, action
        else:
            applied = action
            self.pending = 0.0
        if self.engine == "mujoco":
            from .jepa_data import _checked_mj_step
            self.data.ctrl[0] = applied
            for _ in range(self.substeps):
                if self.mujoco_native_damping:
                    self.data.qfrc_applied[:] = 0.
                else:
                    apply_mujoco_friction(self.data, self.n)
                _checked_mj_step(self.mj, self.model, self.data)
        else:
            for _ in range(self.substeps):
                self._state = analytic_step(self._state, applied, self.n, self.engine, CONTROL_DT / self.substeps)
        self.last_applied = applied
        self.ticks += 1
        state = self.state
        if not np.isfinite(state).all():
            raise RuntimeError("plant produced nonfinite state")
        return state

    def observation(self) -> np.ndarray:
        state = self.state
        d = self.n + 1
        parts = [state[0], state[d]]
        for i in range(self.n):
            parts += [np.sin(state[i + 1]), np.cos(state[i + 1]), state[d + i + 1] / 8]
        return np.r_[parts, self.last_applied, self.pending]


def _extended_riccati(a: np.ndarray, b: np.ndarray, q: np.ndarray,
                      r: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """Scalar-input DARE with extended precision for ill-conditioned long chains.

    Only the offline cost recursion uses longdouble. The plant, observations,
    queued commands and executed gains remain float64. A converged recursion
    must also satisfy its equation and stabilize the executed float64 system.
    """
    work_a, work_b, work_q = (np.asarray(value, dtype=np.longdouble) for value in (a, b, q))
    cost = np.longdouble(r)
    p = work_q.copy()
    tolerance = np.longdouble("1e-15")
    for iteration in range(30000):
        denominator = cost + (work_b.T @ p @ work_b)[0, 0]
        if not np.isfinite(denominator) or denominator <= 0:
            raise RuntimeError("extended Riccati recursion lost its positive scalar control cost")
        k = (work_b.T @ p @ work_a) / denominator
        following = work_q + work_a.T @ p @ work_a - work_a.T @ p @ work_b @ k
        following = np.longdouble(.5) * (following + following.T)
        scale = max(np.longdouble(1.), np.max(np.abs(p)))
        change = np.max(np.abs(following - p)) / scale
        p = following
        if np.isfinite(p).all() and change <= tolerance:
            break
    else:
        raise RuntimeError("extended Riccati iteration did not converge on this arithmetic dtype")
    k = (work_b.T @ p @ work_a) / (cost + (work_b.T @ p @ work_b)[0, 0])
    residual = work_q + work_a.T @ p @ work_a - work_a.T @ p @ work_b @ k - p
    relative_residual = np.max(np.abs(residual)) / max(np.longdouble(1.), np.max(np.abs(p)))
    executed_gain = np.asarray(k, dtype=np.float64)
    radius = float(np.max(np.abs(np.linalg.eigvals(np.asarray(a, dtype=np.float64)
                                                 - np.asarray(b, dtype=np.float64) @ executed_gain))))
    if not np.isfinite(relative_residual) or relative_residual > 10 * tolerance or not np.isfinite(radius) or radius >= 1:
        raise RuntimeError("extended Riccati solution failed its residual or executed-gain stability check")
    diagnostics = {"arithmetic_dtype": str(np.dtype(np.longdouble)),
                   "arithmetic_epsilon": float(np.finfo(np.longdouble).eps),
                   "executed_gain_dtype": "float64", "converged": True,
                   "iterations": iteration + 1, "relative_cost_change": float(change),
                   "relative_dare_residual": float(relative_residual),
                   "closed_loop_spectral_radius_float64": radius}
    return p, k, diagnostics


@lru_cache(maxsize=128)
def _lqr_solution(n: int, delay: int, r: float, q_weights: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict | None]:
    import torch
    dim = 2 * (n + 1)
    zero = np.zeros(dim)
    epsilon = 1e-6
    a = np.column_stack([(derivative(zero + epsilon * np.eye(dim)[i], 0., n)
                          - derivative(zero - epsilon * np.eye(dim)[i], 0., n)) / (2 * epsilon)
                         for i in range(dim)])
    b = ((derivative(zero, epsilon, n) - derivative(zero, -epsilon, n)) / (2 * epsilon))[:, None]
    # Exact 50 Hz zero-order hold of the continuous 200 Hz plant linearization.
    augmented = np.zeros((dim + 1, dim + 1))
    augmented[:dim, :dim], augmented[:dim, dim:] = a, b
    discrete = torch.matrix_exp(torch.as_tensor(augmented * CONTROL_DT, dtype=torch.float64)).numpy()
    ad, bd = discrete[:dim, :dim], discrete[:dim, dim:]
    qdiag = np.asarray(q_weights, dtype=np.float64)
    if delay:
        ac = np.block([[ad, bd], [np.zeros((1, dim + 1))]])
        bc = np.r_[np.zeros((dim, 1)), [[1.]]]
        q = np.diag(np.r_[qdiag, 0.])
    else:
        ac, bc, q = ad, bd, np.diag(qdiag)
    if n >= 9:
        _, gain, diagnostics = _extended_riccati(ac, bc, q, r)
        return np.asarray(gain, dtype=np.float64).reshape(-1), ac, bc, diagnostics
    cost = np.array([[r]])
    p = q.copy()
    for _ in range(30000):
        k = np.linalg.solve(cost + bc.T @ p @ bc, bc.T @ p @ ac)
        following = q + ac.T @ p @ ac - ac.T @ p @ bc @ k
        following = 0.5 * (following + following.T)
        if np.max(np.abs(following - p)) <= 1e-10 * max(1., np.max(np.abs(p))):
            p = following
            break
        p = following
    else:
        raise RuntimeError("discrete Riccati iteration did not converge")
    k = np.linalg.solve(cost + bc.T @ p @ bc, bc.T @ p @ ac).reshape(-1)
    if np.max(np.abs(np.linalg.eigvals(ac - bc @ k[None, :]))) >= 1:
        raise RuntimeError("LQR linearization is not stable")
    return k, ac, bc, None


@dataclass
class LQRBalance:
    """Frozen DARE feedback using the actual pending command for delay one."""
    n: int
    delay: int = 1
    r: float = 1.0
    q_weights: tuple[float, ...] | None = None

    def __post_init__(self):
        _pole_count(self.n)
        if self.delay not in (0, 1) or not np.isfinite(self.r) or self.r <= 0:
            raise ValueError("invalid LQR configuration")
        if self.q_weights is None:
            self.q_weights = tuple([20.] + [50.] * self.n + [2.] * (self.n + 1))
        else:
            self.q_weights = tuple(self.q_weights)
        if len(self.q_weights) != 2 * (self.n + 1) or not np.isfinite(self.q_weights).all() or min(self.q_weights) <= 0:
            raise ValueError("q_weights must contain one positive finite cost per physical state")
        solution = _lqr_solution(self.n, self.delay, float(self.r), self.q_weights)
        self.k, self.ad, self.bd = (part.copy() for part in solution[:3])
        self.riccati_diagnostics = None if solution[3] is None else solution[3].copy()

    def act(self, state: np.ndarray, last_applied: float = 0., pending: float = 0.) -> float:
        state = np.asarray(state, dtype=np.float64).copy()
        if state.shape != (2 * (self.n + 1),) or not np.isfinite(state).all() or not np.isfinite(pending):
            raise ValueError("policy input must be finite and match its pole count")
        state[1:self.n + 1] = wrap_angles(state[1:self.n + 1])
        extended = np.r_[state, pending] if self.delay else state
        return float(np.clip(-self.k @ extended, -1., 1.))

    def to_dict(self) -> dict:
        result = {"policy": "delay_aware_dare_lqr", "n": self.n, "delay": self.delay,
                "r": self.r, "q_diagonal": list(self.q_weights),
                "K": self.k.tolist(), "state_layout": "x, absolute_theta..., x_velocity, absolute_omega..., pending_if_delay1"}
        if self.riccati_diagnostics is not None:
            result["riccati"] = self.riccati_diagnostics.copy()
        return result


class EnergySwingup1:
    """Existing energy pump and sticky catch, with a delay-aware balance gain."""
    def __init__(self, delay: int = 1):
        self.delay = delay
        self.balance = LQRBalance(1, delay=delay)
        self.latched = False

    def act(self, state: np.ndarray, last_applied: float = 0., pending: float = 0.) -> float:
        x, theta, xd, omega = np.asarray(state, dtype=np.float64)
        theta_wrapped = float(wrap_angles(theta))
        self.latched |= abs(theta_wrapped) < 0.40 and abs(omega) < 3.0 and abs(x) < 1.7
        if self.latched:
            return self.balance.act(state, last_applied, pending)
        energy = 0.5 * 0.1 * 0.5**2 * omega**2 + 0.1 * 9.81 * 0.5 * (np.cos(theta) - 1.)
        pump = np.clip(0.25 * energy * omega * np.cos(theta) - 0.40 * x - 0.28 * xd, -0.65, 0.65)
        kick = 0.5 * np.sign(x + 0.12) if abs(theta_wrapped) > 2.05 and abs(omega) < 0.4 else 0.
        action = np.clip(pump + kick, -1., 1.)
        if abs(x) > 1.55:
            action = np.clip(-1.6 * x - 1.2 * xd, -1., 1.)
        return float(action)

    def to_dict(self) -> dict:
        return {"policy": "energy_swing_sticky_delay_aware_lqr", "n": 1, "delay": self.delay,
                "catch_rad": 0.4, "catch_omega_rad_s": 3.0, "energy_gain": 0.25,
                "balance": self.balance.to_dict()}


def state_good(state: np.ndarray, n: int) -> bool:
    d = n + 1
    return bool(np.isfinite(state).all() and abs(state[0]) < RAIL_LIMIT
                and np.all(np.abs(wrap_angles(state[1:d])) < ANGLE_LIMIT)
                and np.all(np.abs(state[d + 1:]) < OMEGA_LIMIT))


def evaluate_policy(policy_factory: Callable, n: int, engine: str, task: str = "balance",
                    episodes: int = 32, seeds: list[int] | None = None, seed: int = 1000,
                    delay: int = 1, horizon: int | None = None, tail_steps: int | None = None,
                    trajectories: bool = False, initial_states: np.ndarray | None = None,
                    substeps: int = 4, mujoco_native_damping: bool = False) -> dict:
    """Score final continuous hold and rail integrity throughout each episode.

    Balance requires all 500 ticks; swing-up needs a 250-tick continuous final
    hold within 1000 ticks. Numeric warnings/resets and any rail escape fail.
    No early successful prefix or best-ever hold can satisfy this gate.
    """
    if task not in ("balance", "swingup"):
        raise ValueError("task must be balance or swingup")
    horizon = (500 if task == "balance" else 1000) if horizon is None else horizon
    tail_steps = (horizon if task == "balance" else 250) if tail_steps is None else tail_steps
    if not isinstance(horizon, int) or horizon < 1 or not isinstance(tail_steps, int) or not 1 <= tail_steps <= horizon:
        raise ValueError("horizon and tail_steps must be positive integers with tail_steps <= horizon")
    if seeds is None:
        if not isinstance(episodes, int) or episodes < 1:
            raise ValueError("episodes must be a positive integer")
        seeds = list(range(seed, seed + episodes))
    else:
        seeds = list(seeds)
        if not seeds or any(isinstance(s, bool) or not isinstance(s, (int, np.integer)) for s in seeds):
            raise ValueError("seeds must be a nonempty list of integers")
        episodes = len(seeds)
    if initial_states is not None:
        initial_states = np.asarray(initial_states, dtype=np.float64)
        if initial_states.shape != (episodes, 2 * (n + 1)) or not np.isfinite(initial_states).all():
            raise ValueError("initial_states must have one finite row per episode")
    records, replays = [], []
    environment = PoleEnv(n, engine, delay, substeps, mujoco_native_damping=mujoco_native_damping)
    policy_description = None
    for ep, episode_seed in enumerate(seeds):
        policy = policy_factory()
        if policy_description is None and hasattr(policy, "to_dict"):
            policy_description = policy.to_dict()
        start = initial_state(n, task, episode_seed) if initial_states is None else initial_states[ep]
        state = environment.reset(start)
        rail_ok, numeric_ok = abs(state[0]) < RAIL_LIMIT, True
        hold = peak = 0
        max_rail, max_angle = abs(state[0]), float(np.max(np.abs(wrap_angles(state[1:n + 1]))))
        states, commands, applied, pending = [state.copy()], [], [], []
        error = None
        for tick in range(horizon):
            try:
                action = policy.act(state, environment.last_applied, environment.pending)
                state = environment.step(action)
            except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                numeric_ok, error = False, str(exc)
                hold = 0
                break
            rail_ok = rail_ok and abs(state[0]) < RAIL_LIMIT
            good = rail_ok and state_good(state, n)
            hold = hold + 1 if good else 0
            peak = max(peak, hold)
            max_rail = max(max_rail, abs(state[0]))
            max_angle = max(max_angle, float(np.max(np.abs(wrap_angles(state[1:n + 1])))))
            if trajectories:
                states.append(state.copy())
                commands.append(float(np.clip(action, -1., 1.)))
                applied.append(environment.last_applied)
                pending.append(environment.pending)
        success = bool(numeric_ok and rail_ok and hold >= tail_steps)
        record = {"seed": int(episode_seed), "success": success, "rail_ok": bool(rail_ok),
                  "numeric_ok": numeric_ok, "final_hold_ticks": hold, "peak_hold_ticks": peak,
                  "max_abs_cart_x_m": float(max_rail), "max_abs_angle_rad": float(max_angle),
                  "steps_completed": environment.ticks, "initial_state": start.tolist(),
                  "final_state": state.tolist() if np.isfinite(state).all() else None}
        if error is not None:
            record["error"] = error
        records.append(record)
        if trajectories:
            replays.append({"seed": int(episode_seed), "states": np.asarray(states),
                            "commands": np.asarray(commands), "applied_actions": np.asarray(applied),
                            "pending_actions": np.asarray(pending)})
    rate = float(np.mean([row["success"] for row in records]))
    result = {"schema": "rage_mill.pole_control.evaluation.v1", "n": n, "engine": engine,
              "physics_device": "cpu", "physics_hz": 50 * substeps, "control_hz": 50,
              "delay_steps": delay, "task": task, "episodes": episodes, "seeds": [int(s) for s in seeds],
              "horizon_ticks": horizon, "required_final_hold_ticks": tail_steps,
              "angle_limit_rad": ANGLE_LIMIT, "angular_velocity_limit_rad_s": OMEGA_LIMIT,
              "rail_limit_m": RAIL_LIMIT, "success_rate": rate, "pass": rate >= 0.8,
              "mean_final_hold_ticks": float(np.mean([r["final_hold_ticks"] for r in records])),
              "rail_integrity_rate": float(np.mean([r["rail_ok"] for r in records])),
              "numeric_integrity_rate": float(np.mean([r["numeric_ok"] for r in records])),
              "policy": policy_description, "episode_metrics": records,
              "identified": False, "sim2real": False}
    result["numerical_damping_mode"] = environment.numerical_damping_mode
    result["mujoco_native_damping"] = environment.mujoco_native_damping
    if engine == "mujoco":
        result.update({"mujoco_version": environment.mj.__version__, "mjcf_sha256": environment.xml_hash})
    if trajectories:
        result["trajectories"] = replays
    return result
