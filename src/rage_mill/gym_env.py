"""Optional Gymnasium adapter; import requires ``rage-jepa[gym]``."""
from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .env import CartPendulumEnv, task_gates
from .pole_control import initial_state, wrap_angles


class CartPendulumGym(gym.Env):
    """Fully observed force-control task with both actuator queue values.

    Dense reward = -mean(wrapped_angle**2) - .01*x**2 - .001*command**2.
    This convenience reward is distinct from the binary published task gates.
    Rail crossing terminates; 1000 ticks truncates. ``info['task_gates']`` audits
    the full episode at either boundary. No rendering or learned agent is implied.
    """
    metadata = {"render_modes": []}

    def __init__(self, n: int = 10, engine: str = "euler", backend: str = "numpy",
                 substeps: int | None = None):
        self.plant = CartPendulumEnv(n=n, engine=engine, backend=backend, substeps=substeps)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(4 + 3 * n,), dtype=np.float64)
        self.action_space = spaces.Box(-1., 1., shape=(1,), dtype=np.float64)
        self._done = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        state_seed = int(self.np_random.integers(0, 2**32))
        state = initial_state(self.plant.n, "swingup", state_seed)
        if options and "state" in options:
            state = options["state"]
        self.plant.reset(state)
        self._states, self._commands, self._applied = [self.plant.state], [], []
        self._done = False
        return self.plant.observation(), {}

    def step(self, action):
        if self._done:
            raise RuntimeError("reset is required before stepping a new episode")
        value = np.asarray(action)
        if value.shape != (1,) or not np.isfinite(value).all() or np.any(np.abs(value) > 1):
            raise ValueError("Gym action must be a finite vector in [-1,1] with shape (1,)")
        command = float(value[0])
        state = self.plant.step(command)
        self._states.append(state)
        self._commands.append(command)
        self._applied.append(self.plant.last_applied)
        n = self.plant.n
        reward = -float(np.mean(wrap_angles(state[1:n + 1])**2) + .01 * state[0]**2 + .001 * command**2)
        terminated = bool(abs(state[0]) >= 2.4)
        truncated = self.plant.ticks >= 1000
        self._done = terminated or truncated
        info = {}
        if self._done:
            info["task_gates"] = task_gates(self._states, self._commands, self._applied, n=n)
        return self.plant.observation(), reward, terminated, truncated, info
