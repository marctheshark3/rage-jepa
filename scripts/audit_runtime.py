#!/usr/bin/env python3
"""Check runtime exports against all original actions and independent mechanics."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
import torch
from rage_mill.baseline import ASSETS, ClassicalPolicy
from rage_mill.jepa_state_control import JEPAStatePolePolicy
from rage_mill.numpy_policy import NumpyJEPAPolicy
from rage_mill.pole_control import analytic_step, derivative
from rage_mill.pole_cpu import acceleration, euler_tick

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    start = time.perf_counter()
    optimized = NumpyJEPAPolicy()
    original = JEPAStatePolePolicy(ROOT / "artifacts/npole10/jepa/model.pt", 10,
        teacher_factory=ClassicalPolicy, reference_plan=optimized.reference_plan,
        iterations=20, residual_tolerance=1e-12)
    maxima = {"exported_command_error": 0., "encoder_error": 0., "recovered_state_error": 0.,
              "acceleration_error": 0., "full_rate_tick_error": 0.}
    actions_checked = encoder_checks = tick_checks = 0
    for engine in ("euler", "mujoco"):
        with np.load(ROOT / f"artifacts/npole10/final/npole10-{engine}-replays.npz", allow_pickle=False) as data:
            for episode in range(32):
                prefix = f"episode_{episode}_"
                states, commands = data[prefix + "states"], data[prefix + "commands"]
                inferred, applied = data[prefix + "inferred_states"], data[prefix + "applied_actions"]
                teacher = ClassicalPolicy()
                for tick in range(1000):
                    pending = 0. if tick == 0 else commands[tick - 1]
                    last = 0. if tick == 0 else applied[tick - 1]
                    command = teacher.act(inferred[tick], last, pending)
                    maxima["exported_command_error"] = max(maxima["exported_command_error"], abs(command - commands[tick]))
                    actions_checked += 1
                for tick in (37, 487, 903):
                    last = applied[tick - 1]
                    state = states[tick]
                    latent = optimized.encode_state(state, last)
                    reference = optimized.reference_plan.states[min(max(0, tick - 1), 900)]
                    estimate = optimized.estimate_latent(latent, last, reference)
                    maxima["encoder_error"] = max(maxima["encoder_error"], float(np.max(np.abs(latent - original.encode_state(state, last)))))
                    maxima["recovered_state_error"] = max(maxima["recovered_state_error"], float(np.max(np.abs(estimate - state))))
                    encoder_checks += 1
                # Full 12,800-substep ticks sampled across hanging, swing and catch.
                if episode < 4:
                    for tick in (0, 237, 501, 899):
                        state, action = states[tick].copy(), applied[tick]
                        reference = state.copy()
                        for _ in range(12800):
                            reference = analytic_step(reference, action, 10, "euler", .02 / 12800)
                        actual = euler_tick(state, action, 10, 12800, .02 / 12800)
                        np.testing.assert_allclose(actual, reference, rtol=2e-11, atol=2e-11)
                        maxima["full_rate_tick_error"] = max(maxima["full_rate_tick_error"], float(np.max(np.abs(actual - reference))))
                        tick_checks += 1
    rng = np.random.default_rng(2410400)
    for _ in range(128):
        state = np.r_[rng.uniform(-.08, .08), rng.uniform(-np.pi, np.pi, 10),
                      rng.uniform(-.02, .02), rng.uniform(-4., 4., 10)]
        action = rng.uniform(-1.2, 1.2)
        actual, reference = acceleration(state, action, 10), derivative(state, action, 10)[11:]
        np.testing.assert_allclose(actual, reference, rtol=2e-11, atol=2e-11)
        maxima["acceleration_error"] = max(maxima["acceleration_error"], float(np.max(np.abs(actual - reference))))
    if maxima["exported_command_error"] != 0. or maxima["encoder_error"] > 5e-15 or maxima["recovered_state_error"] > 1e-11:
        raise ValueError(f"runtime export equivalence failed: {maxima}")
    result = {"pass": True, "original_commands_checked": actions_checked,
              "encoder_and_inverse_states_checked": encoder_checks, "full_rate_tick_checks": tick_checks,
              "random_acceleration_checks": 128, "physics_rtol": 2e-11, "physics_atol": 2e-11,
              "bitwise_physics_equivalence_claim": False, "maxima": maxima,
              "seconds": time.perf_counter() - start,
              "runtime_assets": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in ASSETS.glob("*.npz")}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
