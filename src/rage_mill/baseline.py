"""Frozen ten-pole classical feedback. No Torch import or online gain solve."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .pole_control import wrap_angles
from .swing_trajectory import SwingTrajectory

ASSETS = Path(__file__).with_name("assets")


class ClassicalPolicy:
    """The existing R=20 tracker / R=1000 balance, using exported executed gains.

    The fixed four-substep RK4 delay predictor still uses CasADi, available in
    the ``simulation`` extra. Offline optimization and Riccati recursion are
    unnecessary at runtime. This is a classical baseline, not a learned policy.
    """

    def __init__(self, path: str | Path = ASSETS / "npole10-classical.npz"):
        with np.load(path, allow_pickle=False) as data:
            self.trajectory = SwingTrajectory(10, data["states"], data["actions"],
                                              dt=.02, integration_engine="rk4", integration_substeps=4)
            self.gains = np.array(data["gains"], dtype=np.float64)
            self.balance_k = np.array(data["balance_k"], dtype=np.float64)
        if (self.gains.shape != (900, 22) or self.balance_k.shape != (23,)
                or not np.isfinite(self.gains).all() or not np.isfinite(self.balance_k).all()
                or self.trajectory.states.shape != (901, 22)):
            raise ValueError("expected finite frozen ten-pole gains and 18-second reference")
        self.gains.setflags(write=False)
        self.balance_k.setflags(write=False)
        self.tick = 0
        self.delay = 1

    def reset(self) -> None:
        self.tick = 0

    def act(self, state, last_applied: float = 0., pending: float = 0.) -> float:
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (22,) or not np.isfinite(state).all() or not np.isfinite(pending):
            raise ValueError("expected finite ten-pole state and pending command")
        i = self.tick
        if i >= len(self.trajectory.actions):
            error = state.copy()
            error[1:11] = wrap_angles(error[1:11])
            command = -(self.balance_k @ np.r_[error, pending]).item()
        else:
            predicted = self.trajectory.transition(state, pending)
            error = predicted - self.trajectory.states[i]
            error[1:11] = wrap_angles(error[1:11])
            command = self.trajectory.actions[i] - self.gains[i] @ error
        self.tick += 1
        return float(np.clip(command, -1., 1.))

    def to_dict(self) -> dict:
        return {"policy": "frozen_classical_tvlqr_then_balance", "n": 10, "delay": 1,
                "tracking_r": 20., "balance_r": 1000., "learned": False,
                "gain_file": "npole10-classical.npz", "runtime_torch": False}
