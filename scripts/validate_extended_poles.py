"""Evaluate frozen JEPA/classical-feedback pole controllers from four to seven.

All physics and inference run on CPU. Configuration is recorded before any
seed is initialized; final trials use 32 previously unused initializations.
Training, development and final control results have separate directories.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from rage_mill.jepa_state_control import JEPAStatePolePolicy
from rage_mill.pole_control import LQRBalance, evaluate_policy, wrap_angles
from rage_mill.swing_trajectory import SwingTrajectory, TVLQRTracker


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class LoggedPolicy(JEPAStatePolePolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.estimates, self.residuals = [], []

    def act(self, *args, **kwargs):
        result = super().act(*args, **kwargs)
        self.estimates.append(self.last_estimate.copy())
        self.residuals.append(self.last_residual)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, choices=range(4, 11), required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/jepa-extended/model.pt"))
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--task", choices=("swingup", "balance"), default="swingup")
    parser.add_argument("--engines", nargs="+", choices=("euler", "rk4", "mujoco"), default=("euler", "mujoco"))
    parser.add_argument("--physics-substeps", type=int, default=20)
    parser.add_argument("--mujoco-physics-substeps", type=int,
                        help="Optional independent MuJoCo integration rate; control stays at 50 Hz")
    parser.add_argument("--mujoco-native-damping", action="store_true",
                        help="Use MuJoCo's passive tendons for the same absolute-angle viscous law")
    parser.add_argument("--tracking-r", type=float, default=.2)
    parser.add_argument("--balance-r", type=float, default=1000.)
    parser.add_argument("--inverse-iterations", type=int, default=10)
    parser.add_argument("--inverse-tolerance", type=float, default=1e-12)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--tail-steps", type=int)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=540000)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--final-shard-start", type=int,
                        help="First seed of the fixed final subset for this pole count")
    parser.add_argument("--final-shard-size", type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.task == "swingup" and args.plan is None:
        parser.error("hanging swing-up needs a frozen nominal trajectory")
    if args.physics_substeps < 1 or args.episodes < 1 or args.seed < 0:
        parser.error("substeps/episodes must be positive and seed nonnegative")
    if args.mujoco_physics_substeps is not None and args.mujoco_physics_substeps < 1:
        parser.error("MuJoCo substeps must be positive")
    if (args.final_shard_start is None) != (args.final_shard_size is None):
        parser.error("both final shard start and size are required")
    if args.final and args.final_shard_start is not None:
        parser.error("choose complete final evaluation or a fixed final shard")
    if args.final:
        args.episodes, args.seed = 32, (600000 if args.n <= 7 else args.n * 100000)
    elif args.final_shard_start is not None:
        final_start = 600000 if args.n <= 7 else args.n * 100000
        if not (final_start <= args.final_shard_start < final_start + 32 and args.final_shard_size > 0
                and args.final_shard_start + args.final_shard_size <= final_start + 32):
            parser.error("final shard must be a nonempty subset of the fixed 32 seeds for this body")
        args.episodes, args.seed = args.final_shard_size, args.final_shard_start
    partition = "final" if args.final else "final_shard" if args.final_shard_start is not None else "development"
    torch.set_num_threads(1)
    horizon = args.horizon if args.horizon is not None else (1000 if args.task == "swingup" else 500)
    tail = args.tail_steps if args.tail_steps is not None else (250 if args.task == "swingup" else horizon)
    plan = SwingTrajectory.load(args.plan) if args.plan is not None else None
    if plan is not None and plan.n != args.n:
        parser.error("nominal plan pole count differs")
    feedback = (lambda: TVLQRTracker(plan, delay=1, r=args.tracking_r, balance_r=args.balance_r)) if plan else (
        lambda: LQRBalance(args.n, delay=1, r=args.balance_r))
    policies = []

    def factory():
        policy = LoggedPolicy(args.checkpoint, args.n, teacher_factory=feedback,
                              reference_plan=plan, delay=1, iterations=args.inverse_iterations,
                              residual_tolerance=args.inverse_tolerance)
        policies.append(policy)
        return policy

    description = factory().to_dict()
    policies.clear()
    source_paths = [Path("src/rage_mill") / name for name in (
        "jepa.py", "jepa_train.py", "jepa_state_control.py", "pole_control.py", "swing_trajectory.py", "jepa_data.py")]
    configuration = {"n": args.n, "partition": partition,
                     "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": digest(args.checkpoint),
                     "plan": str(args.plan.resolve()) if args.plan else None,
                     "plan_sha256": digest(args.plan) if args.plan else None,
                     "controller": description, "task": args.task,
                     "horizon_ticks": horizon, "required_final_hold_ticks": tail,
                     "control_hz": 50, "physics_hz": args.physics_substeps * 50,
                     "engine_physics_hz": {engine: 50 * (args.mujoco_physics_substeps
                          if engine == "mujoco" and args.mujoco_physics_substeps is not None else args.physics_substeps)
                          for engine in args.engines},
                     "mujoco_native_damping": args.mujoco_native_damping,
                     "delay_steps": 1, "seeds": list(range(args.seed, args.seed + args.episodes)),
                     "training_during_evaluation": False,
                     "source_hashes": {str(p): digest(p) for p in source_paths}}
    args.out.mkdir(parents=True, exist_ok=True)
    frozen = args.out / f"npole{args.n}-frozen.json"
    if frozen.exists():
        if json.loads(frozen.read_text())["configuration"] != configuration:
            raise ValueError("different frozen configuration exists; use a separate development output directory")
    else:
        frozen.write_text(json.dumps({"frozen_at": datetime.now(timezone.utc).isoformat(),
                                      "configuration": configuration}, indent=2, allow_nan=False) + "\n")
    for name, wanted in configuration["source_hashes"].items():
        path = Path(name)
        archived = args.out / "sources" / f"npole{args.n}" / path
        if digest(path) != wanted:
            raise ValueError("source changed while freezing the experiment")
        archived.parent.mkdir(parents=True, exist_ok=True)
        if archived.exists() and digest(archived) != wanted:
            raise ValueError("refusing to replace a different frozen source archive")
        if not archived.exists():
            shutil.copy2(path, archived)
    for engine in args.engines:
        policies.clear()
        started = time.perf_counter()
        substeps = (args.mujoco_physics_substeps if engine == "mujoco"
                    and args.mujoco_physics_substeps is not None else args.physics_substeps)
        report = evaluate_policy(factory, args.n, engine, task=args.task,
                                 episodes=args.episodes, seed=args.seed, delay=1,
                                 horizon=horizon, tail_steps=tail, substeps=substeps,
                                 mujoco_native_damping=args.mujoco_native_damping,
                                 trajectories=True)
        traces = report.pop("trajectories")
        arrays, errors, residuals = {}, [], []
        for i, (trace, policy, metric) in enumerate(zip(traces, policies, report["episode_metrics"])):
            for key in ("states", "commands", "applied_actions", "pending_actions"):
                arrays[f"episode_{i}_{key}"] = trace[key]
            count = len(trace["commands"])
            inferred = np.asarray(policy.estimates[:count]).reshape(-1, 2 * (args.n + 1))
            arrays[f"episode_{i}_inferred_states"] = inferred
            arrays[f"episode_{i}_inverse_residuals"] = np.asarray(policy.residuals[:count])
            if count:
                error = inferred - trace["states"][:-1]
                error[:, 1:args.n + 1] = wrap_angles(error[:, 1:args.n + 1])
                errors.append(float(np.abs(error).max()))
                residuals.extend(policy.residuals[:count])
                metric["max_decode_error"] = errors[-1]
        replay = args.out / f"npole{args.n}-{engine}-replays.npz"
        np.savez_compressed(replay, **arrays)
        report.update(controller=description, partition=configuration["partition"],
                      configuration_frozen_before_seed_initialization=True,
                      frozen_configuration_sha256=digest(frozen), checkpoint_sha256=digest(args.checkpoint),
                      plan_sha256=configuration["plan_sha256"], replay_sha256=digest(replay),
                      max_decode_error=max(errors, default=None), max_inverse_residual=max(residuals, default=None),
                      evaluation_seconds=time.perf_counter() - started,
                      success_count=sum(m["success"] for m in report["episode_metrics"]),
                      minimum_final_hold_ticks=min(m["final_hold_ticks"] for m in report["episode_metrics"]))
        (args.out / f"npole{args.n}-{engine}.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"n": args.n, "engine": engine, "success": report["success_count"],
                          "episodes": args.episodes, "pass": report["pass"],
                          "minimum_final_hold_ticks": report["minimum_final_hold_ticks"],
                          "max_decode_error": report["max_decode_error"]}), flush=True)


if __name__ == "__main__":
    main()
