"""CPU trajectory optimization and frozen feedback for hanging cart chains.

The optimizer uses the house point-tip mechanics, a 10 N cart actuator and
50 Hz commands. It plans in full physical state, independently of JEPA.
CasADi is an optional dependency; importing a saved plan does not require it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import time

import numpy as np


@lru_cache(maxsize=9)
def _continuous_functions(n: int):
    import casadi as ca

    d = n + 1
    state, action = ca.SX.sym("state", 2 * d), ca.SX.sym("action")
    q, velocity = state[:d], state[d:]
    theta, omega = q[1:], velocity[1:]
    mass = ca.SX.zeros(d, d)
    mass[0, 0] = 1.0 + 0.1 * n
    centrifugal, gravity = ca.SX.zeros(d), ca.SX.zeros(d)
    for i in range(n):
        remaining = 0.1 * (n - i)
        mass[0, i + 1] = mass[i + 1, 0] = 0.5 * remaining * ca.cos(theta[i])
        centrifugal[0] -= 0.5 * remaining * ca.sin(theta[i]) * omega[i] ** 2
        gravity[i + 1] = -9.81 * 0.5 * remaining * ca.sin(theta[i])
        for j in range(n):
            coupling = 0.025 * (n - max(i, j))
            difference = theta[i] - theta[j]
            mass[i + 1, j + 1] = coupling * ca.cos(difference)
            centrifugal[i + 1] += coupling * ca.sin(difference) * omega[j] ** 2
    applied = ca.vertcat(10.0 * action, ca.SX.zeros(n))
    acceleration = ca.solve(mass + 1e-6 * ca.SX.eye(d), applied - centrifugal - gravity - 0.05 * velocity)
    continuous = ca.Function(f"pole_derivative_{n}", [state, action], [ca.vertcat(velocity, acceleration)])
    prescribed_acceleration = ca.SX.sym("acceleration", d)
    residual = ca.Function(f"pole_residual_{n}", [state, action, prescribed_acceleration],
                           [ca.mtimes(mass + 1e-6 * ca.SX.eye(d), prescribed_acceleration)
                            + centrifugal + gravity + 0.05 * velocity - applied])
    return continuous, residual


@lru_cache(maxsize=24)
def symbolic_dynamics(n: int, dt: float = 0.02, substeps: int = 4, engine: str = "rk4"):
    """Return mechanics, an integrator transition and exact local Jacobians."""
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n not in range(2, 11) or not np.isfinite(dt) or dt <= 0 or not isinstance(substeps, int) or isinstance(substeps, bool) or substeps < 1 or engine not in ("euler", "rk4"):
        raise ValueError("n must be an integer in 2..10 and integration settings must be positive")
    import casadi as ca

    continuous = _continuous_functions(n)[0]
    state, action = ca.SX.sym("state", 2 * (n + 1)), ca.SX.sym("action")
    following = state
    h = dt / substeps
    for _ in range(substeps):
        k1 = continuous(following, action)
        if engine == "euler":
            d = n + 1
            velocity = following[d:] + h * k1[d:]
            following = ca.vertcat(following[:d] + h * velocity, velocity)
        else:
            k2 = continuous(following + h * k1 / 2, action)
            k3 = continuous(following + h * k2 / 2, action)
            k4 = continuous(following + h * k3, action)
            following = following + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    transition = ca.Function(f"pole_transition_{n}_{engine}", [state, action], [following])
    jacobian = ca.Function(f"pole_jacobian_{n}_{engine}", [state, action],
                           [ca.jacobian(following, state), ca.jacobian(following, action)])
    return continuous, transition, jacobian


def _collocation_initialization(n, duration, dt, seed, target, rail, action_limit,
                               max_seconds, max_iterations, warm_start=None):
    """Sparse implicit mechanics supply an initial guess for exact shooting."""
    import casadi as ca

    knots, d = round(duration / dt), n + 1
    dim = 2 * d
    x = ca.MX.sym("collocation_states", dim, knots + 1)
    u = ca.MX.sym("collocation_actions", 1, knots + 1)
    acceleration = ca.MX.sym("collocation_acceleration", d, knots + 1)
    mechanics = _continuous_functions(n)[1].map(knots + 1)(x, u, acceleration)
    constraints = ca.vertcat(ca.vec(mechanics),
        ca.vec(x[:d, 1:] - x[:d, :-1] - dt / 2 * (x[d:, 1:] + x[d:, :-1])),
        ca.vec(x[d:, 1:] - x[d:, :-1] - dt / 2 * (acceleration[:, 1:] + acceleration[:, :-1])),
        x[:, 0] - ca.vertcat(0., *([np.pi] * n), *([0.] * d)), x[:, -1] - target)
    objective = dt * ca.sumsqr(u) + 0.05 * ca.sumsqr(u[:, 1:] - u[:, :-1])
    variables = ca.vertcat(ca.vec(x), ca.vec(u), ca.vec(acceleration))
    lower = np.r_[np.tile(np.r_[-rail, np.full(n, -4 * np.pi), np.full(d, -40.)], knots + 1),
                  np.full(knots + 1, -action_limit), np.full(d * (knots + 1), -500.)]
    upper = np.r_[np.tile(np.r_[rail, np.full(n, 4 * np.pi), np.full(d, 40.)], knots + 1),
                  np.full(knots + 1, action_limit), np.full(d * (knots + 1), 500.)]
    initial = np.r_[0., np.full(n, np.pi), np.zeros(d)]
    phase = np.linspace(0., 1., knots + 1)
    guess = initial[:, None] * (1 - phase) + target[:, None] * phase
    guess[0] = np.random.default_rng(seed).uniform(-0.6, 0.6) * np.sin(2 * np.pi * phase)
    guess[d:] = np.gradient(guess[:d], dt, axis=1)
    guess[d:, [0, -1]] = 0.
    controls = np.zeros(knots + 1)
    if warm_start is not None:
        if warm_start.n != n:
            raise ValueError("warm start must have the same pole count")
        previous_phase = np.linspace(0., 1., len(warm_start.states))
        guess = np.stack([np.interp(phase, previous_phase, column) for column in warm_start.states.T])
        guess[d:] *= warm_start.dt * len(warm_start.actions) / duration
        controls = np.interp(phase, previous_phase, np.r_[warm_start.actions, 0.])
        controls = np.clip(controls, -action_limit, action_limit)
    continuous = _continuous_functions(n)[0]
    acceleration_guess = np.asarray(continuous.map(knots + 1)(guess, controls[None]))[d:]
    solver = ca.nlpsol(f"collocation_{n}_{seed}", "ipopt", {"x": variables, "f": objective, "g": constraints},
                      {"ipopt.print_level": 0, "print_time": False, "ipopt.sb": "yes",
                       "ipopt.max_iter": max_iterations, "ipopt.max_cpu_time": max_seconds,
                       "ipopt.tol": 1e-6, "ipopt.acceptable_tol": 1e-5})
    result = solver(x0=np.r_[guess.T.reshape(-1), controls, acceleration_guess.T.reshape(-1)],
                    lbx=lower, ubx=upper, lbg=0., ubg=0.)
    if not solver.stats()["success"]:
        raise RuntimeError(f"collocation initial guess failed: {solver.stats()['return_status']}")
    vector = np.array(result["x"]).reshape(-1)
    states = vector[:dim * (knots + 1)].reshape(knots + 1, dim)
    commands = vector[dim * (knots + 1):(dim + 1) * (knots + 1)]
    return SwingTrajectory(n, states, (commands[:-1] + commands[1:]) / 2, dt,
                           {"initial_guess": "implicit_trapezoidal_collocation"})


@dataclass
class SwingTrajectory:
    n: int
    states: np.ndarray
    actions: np.ndarray
    dt: float = 0.02
    metadata: dict = field(default_factory=dict)
    integration_engine: str = "rk4"
    integration_substeps: int = 4
    _gain_cache: dict = field(default_factory=dict, init=False, repr=False)
    _gain_diagnostics_cache: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        self.states = np.array(self.states, dtype=np.float64, copy=True)
        self.actions = np.array(self.actions, dtype=np.float64, copy=True)
        if isinstance(self.n, bool) or not isinstance(self.n, (int, np.integer)) or self.n not in range(2, 11) or self.actions.ndim != 1 or self.states.shape != (len(self.actions) + 1, 2 * (self.n + 1)):
            raise ValueError("trajectory dimensions do not match its pole count")
        if not np.isfinite(self.states).all() or not np.isfinite(self.actions).all() or np.any(np.abs(self.actions) > 1 + 1e-8):
            raise ValueError("trajectory must be finite with bounded normalized commands")
        if not np.isfinite(self.dt) or self.dt <= 0 or self.integration_engine not in ("euler", "rk4") or not isinstance(self.integration_substeps, int) or isinstance(self.integration_substeps, bool) or self.integration_substeps < 1:
            raise ValueError("trajectory integrator must be euler or rk4 with positive substeps")
        self.states.setflags(write=False)
        self.actions.setflags(write=False)

    def transition(self, state: np.ndarray, action: float) -> np.ndarray:
        return np.array(symbolic_dynamics(self.n, self.dt, self.integration_substeps, self.integration_engine)[1](state, action)).reshape(-1)

    def linearize(self, state: np.ndarray, action: float) -> tuple[np.ndarray, np.ndarray]:
        a, b = symbolic_dynamics(self.n, self.dt, self.integration_substeps, self.integration_engine)[2](state, action)
        return np.array(a), np.array(b)

    def save(self, path) -> None:
        import json
        np.savez_compressed(path, n=self.n, states=self.states, actions=self.actions,
                            dt=self.dt, metadata=json.dumps(self.metadata),
                            integration_engine=self.integration_engine, integration_substeps=self.integration_substeps)

    @classmethod
    def load(cls, path) -> "SwingTrajectory":
        import json
        with np.load(path, allow_pickle=False) as archive:
            return cls(int(archive["n"]), archive["states"], archive["actions"],
                       float(archive["dt"]), json.loads(str(archive["metadata"])),
                       str(archive["integration_engine"]) if "integration_engine" in archive else "rk4",
                       int(archive["integration_substeps"]) if "integration_substeps" in archive else 4)


def solve_swing_trajectory(n: int, duration: float = 6.0, dt: float = 0.02,
                           seed: int = 0, force: float = 10.0, rail: float = 2.0,
                           max_seconds: float = 80.0, max_iterations: int = 2000,
                           terminal_angles: np.ndarray | None = None,
                           action_limit: float = 1.0,
                           warm_start: SwingTrajectory | None = None,
                           collocation_initialization: bool = False,
                           engine: str = "rk4", substeps: int = 4,
                           verbose: bool = False) -> SwingTrajectory:
    """Solve a bounded full-state direct multiple-shooting swing-up on CPU.

    Every knot obeys the selected control-rate integration dynamics. Endpoint velocities and
    all absolute pole angles are constrained, so partial upright poses cannot
    satisfy this optimization. A returned plan still needs feedback validation
    on independently seeded simulation engines.
    """
    import casadi as ca

    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n not in range(2, 11) or force != 10.0 or not 0 < rail < 2.4:
        raise ValueError("use integer 2..10 poles, the house 10 N actuator, and rail margin below 2.4 m")
    if not np.isfinite(action_limit) or not 0 < action_limit <= 1:
        raise ValueError("planning action limit must be in (0, 1]")
    if not np.isfinite(duration) or duration <= 0 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("duration and dt must be positive")
    knots = round(duration / dt)
    if knots < 2 or not np.isclose(knots * dt, duration):
        raise ValueError("duration must contain an integer number of control ticks")
    d, dim = n + 1, 2 * (n + 1)
    _, transition, _ = symbolic_dynamics(n, dt, substeps, engine)
    x, u = ca.MX.sym("states", dim, knots + 1), ca.MX.sym("actions", 1, knots)
    initial = np.r_[0., np.full(n, np.pi), np.zeros(d)]
    target = np.zeros(dim)
    if terminal_angles is not None:
        angles = np.asarray(terminal_angles, dtype=np.float64)
        if angles.shape != (n,) or not np.isfinite(angles).all() or not np.allclose(np.sin(angles), 0., atol=1e-8) or not np.allclose(np.cos(angles), 1., atol=1e-8):
            raise ValueError("terminal angles must be upright multiples of 2*pi")
        target[1:d] = angles
    defects = x[:, 1:] - transition.map(knots)(x[:, :-1], u)
    constraints = ca.vertcat(ca.vec(defects), x[:, 0] - initial, x[:, -1] - target)
    objective = dt * (ca.sumsqr(u) + 0.01 * ca.sumsqr(x[0, :])) + 0.05 * ca.sumsqr(u[:, 1:] - u[:, :-1])
    variables = ca.vertcat(ca.vec(x), ca.vec(u))
    qlow, qhigh = np.r_[-rail, np.full(n, -4 * np.pi)], np.r_[rail, np.full(n, 4 * np.pi)]
    lower = np.r_[np.tile(np.r_[qlow, np.full(d, -40.)], knots + 1), np.full(knots, -action_limit)]
    upper = np.r_[np.tile(np.r_[qhigh, np.full(d, 40.)], knots + 1), np.full(knots, action_limit)]
    phase = np.linspace(0., 1., knots + 1)
    guess = initial[:, None] * (1 - phase) + target[:, None] * phase
    # A smooth cart excursion breaks the otherwise completely straight guess.
    rng = np.random.default_rng(seed)
    guess[0] = (0.3 if seed == 0 else rng.uniform(-0.6, 0.6)) * np.sin(2 * np.pi * phase) * np.sin(np.pi * phase)
    guess[d:] = np.gradient(guess[:d], dt, axis=1)
    guess[d:, [0, -1]] = 0.
    initial_guess = np.r_[guess.T.reshape(-1), np.zeros(knots)]
    if collocation_initialization:
        warm_start = _collocation_initialization(n, duration, dt, seed, target, rail,
                                                 action_limit, max_seconds, max_iterations, warm_start)
    if warm_start is not None:
        if warm_start.n != n:
            raise ValueError("warm start must have the same pole count")
        old_phase = np.linspace(0., 1., len(warm_start.states))
        guess = np.stack([np.interp(phase, old_phase, column) for column in warm_start.states.T])
        guess[d:] *= warm_start.dt * len(warm_start.actions) / duration
        action_guess = np.interp(phase[:-1], old_phase[:-1], warm_start.actions)
        initial_guess = np.r_[guess.T.reshape(-1), np.clip(action_guess, -action_limit, action_limit)]
    options = {"ipopt.print_level": 5 if verbose else 0, "print_time": verbose,
               "ipopt.max_iter": max_iterations, "ipopt.max_cpu_time": max_seconds,
               "ipopt.tol": 1e-7, "ipopt.acceptable_tol": 1e-5,
               "ipopt.sb": "yes", "expand": False}
    started = time.perf_counter()
    solver = ca.nlpsol(f"swing_{n}_{seed}", "ipopt", {"x": variables, "f": objective, "g": constraints}, options)
    result = solver(x0=initial_guess, lbx=lower, ubx=upper, lbg=0., ubg=0.)
    stats = solver.stats()
    vector = np.array(result["x"]).reshape(-1)
    states = vector[:dim * (knots + 1)].reshape(knots + 1, dim)
    actions = vector[dim * (knots + 1):]
    defect_values = states[1:] - np.array(transition.map(knots)(states[:-1].T, actions[None])).T
    maximum_defect = float(np.max(np.abs(defect_values)))
    endpoint_error = float(max(np.max(np.abs(states[0] - initial)), np.max(np.abs(states[-1] - target))))
    if not stats["success"] or maximum_defect > 1e-5 or endpoint_error > 1e-5:
        raise RuntimeError(f"swing trajectory failed: {stats['return_status']}; dynamics defect {maximum_defect:.3g}, endpoint error {endpoint_error:.3g}")
    metadata = {"method": "full_state_direct_multiple_shooting", "optimizer": "casadi_ipopt_cpu",
                "casadi_version": ca.__version__, "solver_status": stats["return_status"],
                "iterations": int(stats["iter_count"]), "solve_seconds": time.perf_counter() - started,
                "seed": seed, "duration_s": duration, "control_hz": 1 / dt, "physics_hz": substeps / dt,
                "planning_engine": engine, "integration_substeps": substeps,
                "force_limit_N": force, "planning_rail_limit_m": rail,
                "planning_force_limit_N": force * action_limit,
                "max_dynamics_defect": maximum_defect, "max_endpoint_error": endpoint_error,
                "peak_cart_m": float(np.max(np.abs(states[:, 0]))), "peak_force_N": float(force * np.max(np.abs(actions))),
                "terminal_angles_rad": target[1:d].tolist(), "identified": False,
                "state_layout": "x, absolute_theta..., x_velocity, absolute_omega..."}
    return SwingTrajectory(n, states, actions, dt, metadata, engine, substeps)


class TVLQRTracker:
    """Frozen trajectory and Riccati gains with a known-command delay predictor."""

    def __init__(self, trajectory: SwingTrajectory, delay: int = 1, r: float = 0.2,
                 balance_r: float = 1.0):
        from .pole_control import LQRBalance

        if delay not in (0, 1) or not np.isfinite(r) or r <= 0 or not np.isfinite(balance_r) or balance_r <= 0:
            raise ValueError("delay must be 0 or 1 and r must be positive")
        self.trajectory, self.delay, self.r, self.tick = trajectory, delay, r, 0
        self.balance_r = balance_r
        self.balance = LQRBalance(trajectory.n, delay=delay, r=balance_r)
        cache_key = (r, balance_r)
        if cache_key not in trajectory._gain_cache:
            gains = self._gains()
            gains.setflags(write=False)
            trajectory._gain_cache[cache_key] = gains
        self.gains = trajectory._gain_cache[cache_key]
        self.riccati_diagnostics = trajectory._gain_diagnostics_cache.get(cache_key)

    def _gains(self) -> np.ndarray:
        n, dim = self.trajectory.n, self.trajectory.states.shape[1]
        if n >= 9:
            return self._extended_gains()
        qcatch = np.diag(np.r_[20., np.full(n, 50.), 2., np.full(n, 2.)])
        a, b = self.trajectory.linearize(np.zeros(dim), 0.)
        p = qcatch.copy()
        for _ in range(30000):
            k = np.linalg.solve([[self.balance_r]] + b.T @ p @ b, b.T @ p @ a)
            following = qcatch + a.T @ p @ a - a.T @ p @ b @ k
            following = 0.5 * (following + following.T)
            if np.max(np.abs(following - p)) < 1e-10 * max(1., np.max(np.abs(p))):
                p = following
                break
            p = following
        else:
            raise RuntimeError("terminal discrete Riccati iteration did not converge")
        qtrack = np.diag(np.r_[10., np.full(n, 200.), 4., np.full(n, 4.)])
        gains = np.empty((len(self.trajectory.actions), dim))
        for i in range(len(gains) - 1, -1, -1):
            a, b = self.trajectory.linearize(self.trajectory.states[i], self.trajectory.actions[i])
            k = np.linalg.solve([[self.r]] + b.T @ p @ b, b.T @ p @ a)
            gains[i] = k.reshape(-1)
            p = qtrack + a.T @ p @ a - a.T @ p @ b @ k
            p = 0.5 * (p + p.T)
        return gains

    def _extended_gains(self) -> np.ndarray:
        from .pole_control import _extended_riccati

        n, dim = self.trajectory.n, self.trajectory.states.shape[1]
        qcatch = np.diag(np.r_[20., np.full(n, 50.), 2., np.full(n, 2.)])
        a, b = self.trajectory.linearize(np.zeros(dim), 0.)
        p, _, terminal = _extended_riccati(a, b, qcatch, self.balance_r)
        qtrack = np.diag(np.r_[10., np.full(n, 200.), 4., np.full(n, 4.)]).astype(np.longdouble)
        gains = np.empty((len(self.trajectory.actions), dim), dtype=np.float64)
        for i in range(len(gains) - 1, -1, -1):
            a, b = self.trajectory.linearize(self.trajectory.states[i], self.trajectory.actions[i])
            a, b = a.astype(np.longdouble), b.astype(np.longdouble)
            denominator = np.longdouble(self.r) + (b.T @ p @ b)[0, 0]
            if not np.isfinite(denominator) or denominator <= 0:
                raise RuntimeError("extended trajectory Riccati recursion lost its positive scalar control cost")
            k = (b.T @ p @ a) / denominator
            gains[i] = np.asarray(k, dtype=np.float64).reshape(-1)
            p = qtrack + a.T @ p @ a - a.T @ p @ b @ k
            p = np.longdouble(.5) * (p + p.T)
            if not np.isfinite(p).all() or not np.isfinite(gains[i]).all():
                raise RuntimeError("extended trajectory Riccati recursion produced nonfinite costs or gains")
        self.trajectory._gain_diagnostics_cache[(self.r, self.balance_r)] = {
            "terminal": terminal, "backward_arithmetic_dtype": str(np.dtype(np.longdouble)),
            "executed_gain_dtype": "float64", "backward_ticks": len(gains),
            "finite_backward_costs_and_gains": True,
            "tracking_stability_claim": "requires actual closed-loop development and final trials"}
        return gains

    def reset(self) -> None:
        self.tick = 0

    def act(self, state: np.ndarray, last_applied: float = 0., pending: float = 0.) -> float:
        from .pole_control import wrap_angles

        state = np.asarray(state, dtype=np.float64)
        if state.shape != (2 * (self.trajectory.n + 1),) or not np.isfinite(state).all() or not np.isfinite(pending):
            raise ValueError("tracker state and pending command must be finite with matching dimensions")
        i = self.tick
        self.tick += 1
        if i >= len(self.trajectory.actions):
            return self.balance.act(state, last_applied, pending)
        predicted = self.trajectory.transition(state, pending) if self.delay else state
        error = predicted - self.trajectory.states[i]
        error[1:self.trajectory.n + 1] = wrap_angles(error[1:self.trajectory.n + 1])
        command = self.trajectory.actions[i] - self.gains[i] @ error
        return float(np.clip(command, -1., 1.))

    def to_dict(self) -> dict:
        result = {"policy": "optimized_swing_time_varying_lqr_then_balance", "n": self.trajectory.n,
                "delay": self.delay, "delay_compensation": "one_tick_known_pending_prediction",
                "plan": self.trajectory.metadata, "tracking_r": self.r,
                "balance": self.balance.to_dict(), "learned": False}
        if self.riccati_diagnostics is not None:
            result["riccati"] = self.riccati_diagnostics
        return result
