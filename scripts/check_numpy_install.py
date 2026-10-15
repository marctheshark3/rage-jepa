#!/usr/bin/env python3
"""Prove a fresh base wheel needs only NumPy, with packaged runtime arrays."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys
from rage_mill.env import CartPendulumEnv
from rage_mill.numpy_policy import NumpyJEPAPolicy
import rage_mill


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--wheel", type=Path)
    args = parser.parse_args()
    for name in ("torch", "mujoco", "casadi", "gymnasium"):
        if importlib.util.find_spec(name) is not None:
            raise ValueError(f"base-install check found optional dependency: {name}")
    if not Path(rage_mill.__file__).is_relative_to(Path(sys.prefix)):
        raise ValueError("expected a noneditable wheel installation")
    env = CartPendulumEnv(n=2, substeps=20)
    env.reset_seed(2420000)
    env.step(.4)
    assert env.last_applied == 0. and env.pending == .4
    assert env.observation().shape == (10,)
    assert NumpyJEPAPolicy().encode_state([0.]*22).shape == (64,)
    assert "torch" not in sys.modules
    result = {"pass": True, "installation": "fresh noneditable base wheel",
              "dependencies": sorted(d.metadata["Name"] for d in importlib.metadata.distributions()),
              "torch_installed": False, "torch_imported": False, "mujoco_installed": False,
              "casadi_installed": False, "gym_installed": False, "numpy_environment_step": True,
              "packaged_encoder_load": True}
    if args.wheel:
        result.update(wheel_sha256=hashlib.sha256(args.wheel.read_bytes()).hexdigest(), wheel_bytes=args.wheel.stat().st_size)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
