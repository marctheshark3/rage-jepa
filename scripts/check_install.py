#!/usr/bin/env python3
"""Exercise a wheel installed with simulation/gym extras and no Torch."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np

from rage_mill.baseline import ClassicalPolicy
from rage_mill.env import CartPendulumEnv, task_gates
from rage_mill.gym_env import CartPendulumGym
from rage_mill.numpy_policy import NumpyJEPAPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if importlib.util.find_spec("torch") is not None:
        raise ValueError("run this check in a fresh inference-only environment without Torch")
    results = []
    for policy_type in (ClassicalPolicy, NumpyJEPAPolicy):
        env = CartPendulumEnv(engine="mujoco")
        env.reset_seed(2420000)
        policy = policy_type()
        states, commands, applied = [env.state], [], []
        started = time.perf_counter()
        for _ in range(1000):
            command = policy.act(states[-1], env.last_applied, env.pending)
            states.append(env.step(command))
            commands.append(command)
            applied.append(env.last_applied)
        gates = task_gates(states, commands, applied)
        if not gates["pass"]:
            raise ValueError(f"installed-wheel closed loop failed: {policy_type.__name__}: {gates}")
        results.append({"policy": policy_type.__name__, "seed": 2420000, "engine": "mujoco",
                        "gates": gates, "episode_seconds": time.perf_counter() - started})
    from gymnasium.utils.env_checker import check_env
    check_env(CartPendulumGym(n=2, substeps=20), skip_render_check=True)
    import rage_mill
    if not Path(rage_mill.__file__).is_relative_to(Path(sys.prefix)):
        raise ValueError("this check must load the installed wheel, not an editable checkout")
    if "torch" in sys.modules:
        raise ValueError("unexpected Torch import")
    prefix = Path(sys.prefix)
    result = {"pass": True, "installation": "fresh noneditable wheel plus simulation/gym extras",
              "torch_installed": False, "torch_imported": False, "gym_checker": "pass",
              "closed_loop_trials": results,
              "venv_apparent_bytes": sum(p.stat().st_size for p in prefix.rglob("*") if p.is_file()),
              "distributions": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
