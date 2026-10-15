#!/usr/bin/env python3
"""Serial CPU latency on identical replayed inputs; no training or GPU physics."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import numpy as np

from rage_mill.comparison import CONDITIONS, make_policy, source_hashes, statistics
from rage_mill.env import CartPendulumEnv

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    replay = ROOT / "artifacts/npole10/final/npole10-mujoco-replays.npz"
    with np.load(replay, allow_pickle=False) as data:
        states = data["episode_0_states"][:-1]
        commands = data["episode_0_commands"]
        applied = data["episode_0_applied_actions"]
    started = time.perf_counter()
    load = list(os.getloadavg())
    measured_sources = source_hashes()
    policies, setup = {}, {}
    checkpoint = ROOT / "artifacts/npole10/jepa/model.pt"
    for name in CONDITIONS:
        begin = time.perf_counter()
        policies[name] = make_policy(name, checkpoint)
        setup[name] = time.perf_counter() - begin
        policies[name].act(states[0], 0., 0.)
        policies[name].reset()
    measurements = {name: [] for name in CONDITIONS}
    command_outputs = {}
    # Same states, actual last force, pending queue, and control tick. Rotate
    # conditions across three repetitions to reduce cache/order confounding.
    for repeat in range(3):
        order = CONDITIONS[repeat:] + CONDITIONS[:repeat]
        for name in order:
            policy = policies[name]
            policy.reset()
            outputs = []
            for tick, state in enumerate(states):
                last = applied[tick - 1] if tick else 0.
                pending = commands[tick - 1] if tick else 0.
                begin = time.perf_counter_ns()
                output = policy.act(state, last, pending)
                measurements[name].append((time.perf_counter_ns() - begin) / 1e6)
                outputs.append(output)
            command_outputs[name] = np.asarray(outputs)
    physics = {name: [] for name in ("numpy-euler", "c-euler", "mujoco")}
    max_physics_error = 0.
    plants = {"numpy-euler": CartPendulumEnv(delay=0),
              "c-euler": CartPendulumEnv(delay=0, backend="c"),
              "mujoco": CartPendulumEnv(engine="mujoco", delay=0)}
    indices = np.linspace(0, 999, 12, dtype=int)
    for repeat in range(3):
        for tick in indices:
            outputs = {}
            for name, env in plants.items():
                env.reset(states[tick])
                begin = time.perf_counter_ns()
                outputs[name] = env.step(applied[tick])
                physics[name].append((time.perf_counter_ns() - begin) / 1e6)
            np.testing.assert_allclose(outputs["numpy-euler"], outputs["c-euler"], atol=2e-11, rtol=2e-11)
            max_physics_error = max(max_physics_error, float(np.max(np.abs(outputs["numpy-euler"] - outputs["c-euler"]))))
    if source_hashes() != measured_sources:
        raise RuntimeError("runtime changed during measurement")
    result = {"pass": True, "physics_device": "cpu", "inference_device": "cpu", "machine": platform.machine(),
              "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
              "torch_threads": 1, "thread_environment": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
              "load_average_start": load, "load_average_end": list(os.getloadavg()),
              "input_replay_sha256": hashlib.sha256(replay.read_bytes()).hexdigest(),
              "input_episode_seed": 1000000, "repetitions": 3, "ticks_per_repetition": 1000,
              "controller_ms": {name: statistics(values) for name, values in measurements.items()},
              "physics_tick_ms": {name: statistics(values) for name, values in physics.items()},
              "setup_seconds_warm_imports": setup, "max_compiled_vs_numpy_tick_error": max_physics_error,
              "max_numpy_vs_original_command_error": float(np.max(np.abs(command_outputs["numpy-jepa"] - command_outputs["original-jepa"]))),
              "source_sha256": measured_sources, "wall_seconds": time.perf_counter() - started,
              "limits": "Serial replay latency on a shared Spark ARM CPU; warm imports, caches and predictor. No GPU throughput or hardware real-time guarantee."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "source_sha256"}, indent=2))


if __name__ == "__main__":
    main()
