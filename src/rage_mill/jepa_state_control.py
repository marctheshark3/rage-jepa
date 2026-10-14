"""JEPA state estimation followed by explicitly classical pole feedback.

The current physical observation is used only to compute a frozen target
embedding. Feedback consumes a state decoded from that embedding. This hybrid
is different from the separately trained JEPA latent action head.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from .jepa_train import load_checkpoint
from .pole_control import EnergySwingup1, LQRBalance, analytic_step, wrap_angles


def _physical_observation(states: torch.Tensor, last_applied: float, n: int) -> torch.Tensor:
    """The original JEPA observation contract; no raw state reaches feedback."""
    d = n + 1
    parts = [states[:, 0], states[:, d]]
    for i in range(n):
        parts.extend((torch.sin(states[:, i + 1]), torch.cos(states[:, i + 1]), states[:, d + i + 1] / 8.))
    parts.append(torch.full_like(states[:, 0], float(last_applied)))
    return torch.stack(parts, dim=1)


class JEPAStatePolePolicy:
    """Frozen JEPA inverse observation model plus an explicit feedback teacher.

    ``teacher_factory`` creates feedback that receives decoded state only.
    A nominal ``reference_plan`` supplies an inverse initialization and angle
    branch; no future observation is consumed. Without a plan, the previous
    estimated state is propagated using the known last applied command.
    Inversion failure raises an error; it never falls back to raw state.
    """

    def __init__(self, base_checkpoint: Path | str, n: int, teacher_factory=None,
                 reference_plan=None, delay: int = 1, iterations: int = 5,
                 initial_reference: np.ndarray | None = None,
                 jacobian_epsilon: float = 1e-5, residual_tolerance: float = 1e-8):
        if (isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer))
                or not 1 <= n <= 10 or delay not in (0, 1) or isinstance(delay, bool)):
            raise ValueError("n must be 1..10 and delay must be 0 or 1")
        if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not np.isfinite(jacobian_epsilon) or jacobian_epsilon <= 0 or not np.isfinite(residual_tolerance) or residual_tolerance <= 0:
            raise ValueError("inverse solver tolerances must be positive and finite")
        self.n, self.delay, self.iterations = n, delay, iterations
        self.env_id, self.dim = f"npole{n}", 2 * (n + 1)
        self.jacobian_epsilon, self.residual_tolerance = jacobian_epsilon, residual_tolerance
        self.model, norms, _contents = load_checkpoint(base_checkpoint, "cpu")
        if self.env_id not in self.model.env_dims or self.env_id not in norms or self.model.env_dims[self.env_id] != (3 + 3 * n, 1):
            raise ValueError("JEPA checkpoint does not contain the requested pole observation contract")
        self.model.double().eval().requires_grad_(False)
        self.encoder = self.model.target_encoders[self.model._key(self.env_id)]
        self.mean = torch.as_tensor(norms[self.env_id].mean, dtype=torch.float64)
        self.std = torch.as_tensor(norms[self.env_id].std, dtype=torch.float64)
        if not bool(torch.isfinite(self.mean).all() and torch.isfinite(self.std).all() and torch.all(self.std > 0)):
            raise ValueError("JEPA normalization must be finite with positive standard deviations")
        self.base_hash = hashlib.sha256(Path(base_checkpoint).read_bytes()).hexdigest()
        self.reference_plan = reference_plan
        if reference_plan is not None and reference_plan.n != n:
            raise ValueError("reference trajectory must match the pole count")
        if teacher_factory is None:
            if reference_plan is not None:
                from .swing_trajectory import TVLQRTracker
                teacher_factory = lambda: TVLQRTracker(reference_plan, delay=delay)
            elif n == 1:
                teacher_factory = lambda: EnergySwingup1(delay=delay)
            else:
                teacher_factory = lambda: LQRBalance(n, delay=delay)
        self.teacher = teacher_factory()
        if not hasattr(self.teacher, "act"):
            raise ValueError("teacher_factory must produce a feedback policy with act")
        if initial_reference is None:
            initial_reference = np.zeros(self.dim)
            if n == 1 and reference_plan is None:
                initial_reference[1] = np.pi
            elif reference_plan is not None:
                initial_reference = reference_plan.states[0]
        self.initial_reference = self._validate_state(initial_reference)
        self.tick = 0
        self.last_estimate = None
        self.last_latent = None
        self.last_residual = None
        self.last_condition_number = None
        self.total_inference_seconds = 0.

    def _validate_state(self, state: np.ndarray) -> np.ndarray:
        value = np.asarray(state, dtype=np.float64)
        if value.shape != (self.dim,) or not np.isfinite(value).all():
            raise ValueError("physical state/reference must be a finite vector matching the pole count")
        return value.copy()

    @torch.no_grad()
    def _encode_candidates(self, states: np.ndarray, last_applied: float) -> np.ndarray:
        if not np.isfinite(last_applied):
            raise ValueError("last applied command must be finite")
        tensor = torch.as_tensor(states, dtype=torch.float64)
        observation = _physical_observation(tensor, last_applied, self.n)
        latent = self.encoder((observation - self.mean) / self.std)
        result = latent.numpy()
        if not np.isfinite(result).all():
            raise RuntimeError("frozen JEPA encoder produced a nonfinite latent")
        return result

    def encode_state(self, state: np.ndarray, last_applied: float = 0.) -> np.ndarray:
        """Encode current observable state without exposing coordinates to feedback."""
        return self._encode_candidates(self._validate_state(state)[None], last_applied)[0]

    def _jacobian(self, reference: np.ndarray, last_applied: float) -> np.ndarray:
        offsets = self.jacobian_epsilon * np.eye(self.dim)
        candidates = np.r_[reference[None] + offsets, reference[None] - offsets]
        encoded = self._encode_candidates(candidates, last_applied)
        return ((encoded[:self.dim] - encoded[self.dim:]) / (2 * self.jacobian_epsilon)).T

    def _reference(self, last_applied: float) -> np.ndarray:
        if self.reference_plan is not None:
            # Delay-one begins with a zero queue, so observation t corresponds
            # to nominal knot t-1. The reference is fixed, not a future sample.
            index = max(0, self.tick - self.delay)
            return self.reference_plan.states[min(index, len(self.reference_plan.states) - 1)].copy()
        if self.last_estimate is None:
            return self.initial_reference.copy()
        predicted = self.last_estimate.copy()
        for _ in range(4):
            predicted = analytic_step(predicted, last_applied, self.n, "rk4", 0.005)
        return predicted

    def estimate_latent(self, latent: np.ndarray, last_applied: float = 0.,
                        reference: np.ndarray | None = None) -> np.ndarray:
        """Recover physical state using only a latent, known command, and prior."""
        latent = np.asarray(latent, dtype=np.float64)
        if latent.shape != (self.model.config.latent_dim,) or not np.isfinite(latent).all():
            raise ValueError("latent must be a finite vector with the frozen encoder dimension")
        reference = self._reference(last_applied) if reference is None else self._validate_state(reference)
        estimate = reference.copy()
        jacobian = self._jacobian(reference, last_applied)
        singular = np.linalg.svd(jacobian, compute_uv=False)
        if singular[-1] <= 1e-12 or not np.isfinite(singular).all():
            raise RuntimeError("JEPA inverse observation Jacobian is rank deficient")
        self.last_condition_number = float(singular[0] / singular[-1])
        inverse = np.linalg.pinv(jacobian, rcond=1e-12)
        for _ in range(self.iterations):
            encoded = self._encode_candidates(estimate[None], last_applied)[0]
            correction = inverse @ (latent - encoded)
            estimate += correction
            # Select the periodic angle branch nearest the nominal/prior.
            estimate[1:self.n + 1] = reference[1:self.n + 1] + wrap_angles(estimate[1:self.n + 1] - reference[1:self.n + 1])
            if not np.isfinite(estimate).all():
                raise RuntimeError("JEPA inverse state solver produced a nonfinite estimate")
        residual = self._encode_candidates(estimate[None], last_applied)[0] - latent
        self.last_residual = float(np.linalg.norm(residual))
        if self.last_residual > self.residual_tolerance:
            # Recompute at the estimate once if the nominal linearization was
            # too far away. The same fixed number of Newton refinements is used.
            inverse = np.linalg.pinv(self._jacobian(estimate, last_applied), rcond=1e-12)
            for _ in range(self.iterations):
                correction = inverse @ (latent - self._encode_candidates(estimate[None], last_applied)[0])
                estimate += correction
                estimate[1:self.n + 1] = reference[1:self.n + 1] + wrap_angles(estimate[1:self.n + 1] - reference[1:self.n + 1])
            self.last_residual = float(np.linalg.norm(self._encode_candidates(estimate[None], last_applied)[0] - latent))
        if not np.isfinite(estimate).all() or not np.isfinite(self.last_residual) or self.last_residual > self.residual_tolerance:
            raise RuntimeError(f"JEPA inverse state solver failed; latent residual {self.last_residual:.3g}")
        return estimate

    def act(self, state: np.ndarray, last_applied: float = 0., pending: float = 0.) -> float:
        if not np.isfinite(pending):
            raise ValueError("pending command must be finite")
        started = time.perf_counter()
        latent = self.encode_state(state, last_applied)
        estimated = self.estimate_latent(latent, last_applied)
        # Deliberately pass only the recovered state to classical feedback.
        command = self.teacher.act(estimated, last_applied, pending)
        self.last_latent, self.last_estimate = latent.copy(), estimated.copy()
        self.tick += 1
        self.total_inference_seconds += time.perf_counter() - started
        return float(command)

    def reset(self) -> None:
        self.tick = 0
        self.last_estimate = self.last_latent = self.last_residual = self.last_condition_number = None
        self.total_inference_seconds = 0.
        if hasattr(self.teacher, "reset"):
            self.teacher.reset()
        elif hasattr(self.teacher, "latched"):
            self.teacher.latched = False

    def to_dict(self) -> dict:
        return {"policy": "jepa_encoder_state_estimator_with_analytic_feedback", "n": self.n,
                "delay": self.delay, "JEPA_used_for_action": True, "jepa_used_for_action": True,
                "raw_state_feedback_bypass": False, "controller_family": "hybrid",
                "encoder": "frozen_target_float64", "base_checkpoint_sha256": self.base_hash,
                "inverse": {"method": "local_encoder_jacobian_newton", "iterations": self.iterations,
                            "jacobian_epsilon": self.jacobian_epsilon,
                            "latent_residual_tolerance": self.residual_tolerance,
                            "jacobian_update": "current_reference_and_actual_known_last_command_each_tick",
                            "initialization": "fixed_nominal_reference" if self.reference_plan is not None else "previous_estimate_known_force_prediction"},
                "future_observation_input": False,
                "feedback": self.teacher.to_dict() if hasattr(self.teacher, "to_dict") else type(self.teacher).__name__}
