"""Inference-only frozen JEPA encoder and strict local inversion, using NumPy."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .baseline import ASSETS, ClassicalPolicy
from .pole_control import wrap_angles


class NumpyEncoder:
    """Exact saved float32 target tensors promoted to float64; no quantization."""

    def __init__(self, path: str | Path = ASSETS / "npole10-target-encoder.npz"):
        with np.load(path, allow_pickle=False) as data:
            self.layers = [(np.array(data[f"weight_{i}"], dtype=np.float64),
                            np.array(data[f"bias_{i}"], dtype=np.float64)) for i in range(3)]
            self.mean = np.array(data["mean"], dtype=np.float64)
            self.std = np.array(data["std"], dtype=np.float64)
        shapes = [(128, 33), (128, 128), (64, 128)]
        if (self.mean.shape != (33,) or self.std.shape != (33,)
                or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all()
                or np.any(self.std <= 0)
                or any(w.shape != shape or b.shape != (shape[0],)
                       or not np.isfinite(w).all() or not np.isfinite(b).all()
                       for (w, b), shape in zip(self.layers, shapes))):
            raise ValueError("expected finite ten-pole encoder tensors and positive normalization")

    def __call__(self, normalized) -> np.ndarray:
        result = np.asarray(normalized, dtype=np.float64)
        if result.ndim != 2 or result.shape[1] != 33 or not np.isfinite(result).all():
            raise ValueError("encoder input must be finite [batch,33] normalized observations")
        for i, (weight, bias) in enumerate(self.layers):
            result = result @ weight.T + bias
            if i < 2:
                exponent = np.exp(-np.abs(result))
                sigmoid = np.where(result >= 0, 1. / (1. + exponent), exponent / (1. + exponent))
                result = result * sigmoid
        if not np.isfinite(result).all():
            raise RuntimeError("nonfinite encoder output")
        return result

    def encode_states(self, states, last_applied: float = 0.) -> np.ndarray:
        states = np.asarray(states, dtype=np.float64)
        if (states.ndim != 2 or states.shape[1] != 22 or not np.isfinite(states).all()
                or not np.isfinite(last_applied)):
            raise ValueError("expected finite [batch,22] states and last-applied command")
        obs = np.empty((len(states), 33), dtype=np.float64)
        obs[:, 0], obs[:, 1] = states[:, 0], states[:, 11]
        links = obs[:, 2:-1].reshape(len(states), 10, 3)
        links[:, :, 0], links[:, :, 1], links[:, :, 2] = np.sin(states[:, 1:11]), np.cos(states[:, 1:11]), states[:, 12:] / 8.
        obs[:, -1] = last_applied
        return self((obs - self.mean) / self.std)


class NumpyJEPAPolicy:
    """Early-exit inverse with the original 1e-12 residual gate and no fallback.

    Feedback receives only the recovered state. The trained dynamics predictor
    remains unused. This implementation is deliberately specific to ten poles.
    """

    def __init__(self, encoder_path: str | Path = ASSETS / "npole10-target-encoder.npz",
                 controller_path: str | Path = ASSETS / "npole10-classical.npz"):
        self.encoder = NumpyEncoder(encoder_path)
        self.teacher = ClassicalPolicy(controller_path)
        self.reference_plan = self.teacher.trajectory
        self.dim, self.n, self.delay = 22, 10, 1
        self.iterations, self.jacobian_epsilon, self.residual_tolerance = 20, 1e-5, 1e-12
        self.reset()

    def reset(self) -> None:
        self.tick = 0
        self.last_latent = self.last_estimate = self.last_residual = self.last_condition_number = None
        self.last_refinements = 0
        self.teacher.reset()

    def encode_state(self, state, last_applied: float = 0.) -> np.ndarray:
        state = self._state(state)
        return self.encoder.encode_states(state[None], last_applied)[0]

    @staticmethod
    def _state(state):
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (22,) or not np.isfinite(state).all():
            raise ValueError("expected finite ten-pole physical state/reference")
        return state.copy()

    def estimate_latent(self, latent, last_applied: float = 0., reference=None) -> np.ndarray:
        target = np.asarray(latent, dtype=np.float64)
        if target.shape != (64,) or not np.isfinite(target).all():
            raise ValueError("expected finite 64-dimensional latent")
        if reference is None:
            index = min(max(0, self.tick - 1), len(self.reference_plan.states) - 1)
            reference = self.reference_plan.states[index].copy()
        else:
            reference = self._state(reference)
        estimate = reference.copy()
        offsets = self.jacobian_epsilon * np.eye(22)
        self.last_refinements = 0
        for _ in range(2):
            candidates = np.r_[estimate[None] + offsets, estimate[None] - offsets]
            encoded = self.encoder.encode_states(candidates, last_applied)
            jacobian = ((encoded[:22] - encoded[22:]) / (2 * self.jacobian_epsilon)).T
            singular = np.linalg.svd(jacobian, compute_uv=False)
            if not np.isfinite(singular).all() or singular[-1] <= 1e-12:
                raise RuntimeError("JEPA inverse observation Jacobian is rank deficient")
            self.last_condition_number = float(singular[0] / singular[-1])
            inverse = np.linalg.pinv(jacobian, rcond=1e-12)
            for _ in range(self.iterations):
                residual = target - self.encoder.encode_states(estimate[None], last_applied)[0]
                if np.linalg.norm(residual) <= 1e-14:
                    break
                estimate += inverse @ residual
                estimate[1:11] = reference[1:11] + wrap_angles(estimate[1:11] - reference[1:11])
                if not np.isfinite(estimate).all():
                    raise RuntimeError("JEPA inverse state solver produced a nonfinite estimate")
                self.last_refinements += 1
            self.last_residual = float(np.linalg.norm(self.encoder.encode_states(estimate[None], last_applied)[0] - target))
            if self.last_residual <= self.residual_tolerance:
                break
        if not np.isfinite(self.last_residual) or self.last_residual > self.residual_tolerance:
            raise RuntimeError(f"JEPA inverse failed; latent residual {self.last_residual:.3g}")
        return estimate

    def act(self, state, last_applied: float = 0., pending: float = 0.) -> float:
        if not np.isfinite(pending):
            raise ValueError("pending command must be finite")
        latent = self.encode_state(state, last_applied)
        estimated = self.estimate_latent(latent, last_applied)
        command = self.teacher.act(estimated, last_applied, pending)
        self.last_latent, self.last_estimate = latent.copy(), estimated.copy()
        self.tick += 1
        return command

    def to_dict(self) -> dict:
        return {"policy": "numpy_frozen_jepa_inverse_classical", "n": 10, "delay": 1,
                "runtime_torch": False, "raw_state_feedback_bypass": False,
                "inverse_iterations_max_per_phase": 20, "inverse_phases_max": 2,
                "early_exit_residual": 1e-14, "final_residual_gate": 1e-12,
                "learned_predictor_selects_actions": False, "feedback": self.teacher.to_dict()}
