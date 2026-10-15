"""Environment-only entry point. NumPy is the only required dependency."""
from __future__ import annotations

import numpy as np

from .pole_control import CONTROL_DT, PoleEnv, _pole_count, initial_state, wrap_angles


class CartPendulumEnv(PoleEnv):
    """Original plant/queue, with an explicitly selected optional C Euler loop.

    Ten-pole defaults retain the qualified integration rates. ``backend='c'``
    requires a C compiler on Linux; it never silently changes the backend.
    MuJoCo is optional and always uses native absolute-coordinate damping here.
    ``state`` has 22 values for ten poles; ``observation()`` has 34, including
    last-applied and pending commands. No model is loaded by this environment.
    """

    def __init__(self, n: int = 10, engine: str = "euler", delay: int = 1,
                 substeps: int | None = None, initial_state=None, backend: str = "numpy"):
        _pole_count(n)
        if backend not in ("numpy", "c"):
            raise ValueError("backend must be numpy or c")
        if backend == "c" and (engine != "euler" or not 2 <= n <= 10):
            raise ValueError("C backend requires Euler and 2..10 point-mass poles")
        if substeps is None:
            substeps = 12800 if n == 10 and engine == "euler" else 20 if engine == "mujoco" else 4
        super().__init__(n, engine, delay, substeps, initial_state, mujoco_native_damping=True)
        self.backend = backend
        if backend == "c":
            from .pole_cpu import backend_provenance
            self.backend_provenance = backend_provenance(n)
        else:
            self.backend_provenance = {"backend": "numpy" if engine != "mujoco" else "mujoco",
                                       "xml_sha256": self.xml_hash}

    def reset_seed(self, seed: int, task: str = "swingup") -> np.ndarray:
        return self.reset(initial_state(self.n, task, seed))

    def step(self, action: float) -> np.ndarray:
        if self.backend != "c":
            return super().step(action)
        from .pole_cpu import euler_tick
        value = np.asarray(action)
        if value.size != 1 or not np.isfinite(value).all():
            raise ValueError("action must be one finite normalized cart force")
        command = float(np.clip(value.reshape(-1)[0], -1., 1.))
        applied = self.pending if self.delay else command
        # Commit the queue and state only after a successful integration.
        following = euler_tick(self._state, applied, self.n, self.substeps,
                               CONTROL_DT / self.substeps)
        self._state = following
        self.pending = command if self.delay else 0.
        self.last_applied = applied
        self.ticks += 1
        return self.state


def task_gates(states, commands, applied_actions, n: int = 10, horizon: int = 1000) -> dict:
    """Audit complete 20-second trials, including all 251 final-window endpoints."""
    states = np.asarray(states)
    commands, applied = np.asarray(commands), np.asarray(applied_actions)
    complete = states.shape == (horizon + 1, 2 * (n + 1)) and commands.shape == applied.shape == (horizon,)
    finite = bool(np.isfinite(states).all() and np.isfinite(commands).all() and np.isfinite(applied).all())
    rail = bool(states.ndim == 2 and len(states) and np.max(np.abs(states[:, 0])) < 2.4)
    force = bool(commands.size and np.max(np.abs(commands)) <= 1.)
    queue = bool(applied.size and applied[0] == 0. and np.array_equal(applied[1:], commands[:-1]))
    angles = wrap_angles(states[-251:, 1:n + 1]) if complete else np.array([np.inf])
    speeds = states[-251:, n + 2:] if complete else np.array([np.inf])
    hold = bool(complete and np.max(np.abs(angles)) < .22 and np.max(np.abs(speeds)) < 1.)
    result = {"complete": complete, "finite": finite, "rail": rail, "force": force,
              "one_tick_queue": queue, "final_251_endpoints": hold}
    result["pass"] = all(result.values())
    return result
