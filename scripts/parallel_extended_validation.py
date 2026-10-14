"""Freeze and shard CPU-only extended-pole validation without policy changes.

Workers run validate_extended_poles.py in isolated directories. Every episode,
including failures, is retained and rekeyed by its global seed index. Final
validation covers all32 fixed seeds:600000..600031 for4..7, or n*100000 through n*100000+31 for8..10.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = tuple("src/rage_mill/" + name for name in (
    "jepa.py", "jepa_train.py", "jepa_state_control.py", "pole_control.py", "swing_trajectory.py", "jepa_data.py"))
FROZEN_FILES = SOURCE_FILES + ("scripts/validate_extended_poles.py", "scripts/parallel_extended_validation.py")
EPISODE_KEY = re.compile(r"^episode_(\d+)_(.+)$")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split_seeds(seeds: list[int], workers: int) -> list[list[int]]:
    """Deterministic contiguous shards, including every seed exactly once."""
    if not seeds or len(set(seeds)) != len(seeds) or workers < 1:
        raise ValueError("seeds must be nonempty/unique and workers positive")
    workers = min(workers, len(seeds))
    size, extra = divmod(len(seeds), workers)
    shards, position = [], 0
    for i in range(workers):
        count = size + (i < extra)
        shards.append(seeds[position:position + count])
        position += count
    return shards


def validate_artifacts(configuration):
    for field in ("checkpoint", "plan"):
        if configuration[field] is not None and digest(configuration[field]) != configuration[field + "_sha256"]:
            raise ValueError(f"frozen {field} changed")
    for source, expected in configuration["source_hashes"].items():
        if digest(REPO_ROOT / source) != expected:
            raise ValueError(f"frozen source changed: {source}")


def freeze(args, shards):
    # Import policy code only for its pre-evaluation description, never to run
    # physics or make a decision about which episodes should be evaluated.
    import torch
    from rage_mill.jepa_state_control import JEPAStatePolePolicy
    from rage_mill.pole_control import LQRBalance
    from rage_mill.swing_trajectory import SwingTrajectory, TVLQRTracker

    torch.set_num_threads(1)
    plan = SwingTrajectory.load(args.plan) if args.plan else None
    if plan is not None and plan.n != args.n:
        raise ValueError("nominal plan pole count differs")
    feedback = (lambda: TVLQRTracker(plan, delay=1, r=args.tracking_r, balance_r=args.balance_r)) if plan else (
        lambda: LQRBalance(args.n, delay=1, r=args.balance_r))
    policy = JEPAStatePolePolicy(args.checkpoint, args.n, teacher_factory=feedback, reference_plan=plan,
                                delay=1, iterations=args.inverse_iterations,
                                residual_tolerance=args.inverse_tolerance)
    horizon = args.horizon if args.horizon is not None else (1000 if args.task == "swingup" else 500)
    tail = args.tail_steps if args.tail_steps is not None else (250 if args.task == "swingup" else horizon)
    configuration = {"n": args.n, "partition": "final" if args.final else "development",
                     "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": digest(args.checkpoint),
                     "plan": str(args.plan.resolve()) if args.plan else None,
                     "plan_sha256": digest(args.plan) if args.plan else None,
                     "controller": policy.to_dict(), "task": args.task,
                     "horizon_ticks": horizon, "required_final_hold_ticks": tail,
                     "control_hz": 50, "physics_hz": 50 * args.physics_substeps,
                     "engine_physics_hz": {engine: 50 * (args.mujoco_physics_substeps
                           if engine == "mujoco" and args.mujoco_physics_substeps is not None else args.physics_substeps)
                           for engine in args.engines},
                     "mujoco_native_damping": args.mujoco_native_damping, "delay_steps": 1,
                     "seeds": [seed for shard in shards for seed in shard],
                     "training_during_evaluation": False,
                     "source_hashes": {source: digest(REPO_ROOT / source) for source in FROZEN_FILES},
                     "workers": len(shards), "shards": shards,
                     "worker_runtime": {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
                                        "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}}
    args.out.mkdir(parents=True, exist_ok=True)
    frozen = args.out / f"npole{args.n}-GLOBALfrozen.json"
    if frozen.exists():
        if json.loads(frozen.read_text())["configuration"] != configuration:
            raise ValueError("different global frozen configuration exists; use a new --out directory")
    else:
        frozen.write_text(json.dumps({"frozen_at": datetime.now(timezone.utc).isoformat(),
                                      "configuration": configuration}, indent=2, allow_nan=False) + "\n")
    # The standard alias has identical bytes and supports the existing audit.
    alias = args.out / f"npole{args.n}-frozen.json"
    if alias.exists() and digest(alias) != digest(frozen):
        raise ValueError("different standard frozen receipt already exists")
    if not alias.exists():
        shutil.copy2(frozen, alias)
    for source, expected in configuration["source_hashes"].items():
        target = args.out / "sources" / f"npole{args.n}" / source
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and digest(target) != expected:
            raise ValueError(f"different archived global source: {target}")
        if not target.exists():
            shutil.copy2(REPO_ROOT / source, target)
    validate_artifacts(configuration)
    return configuration, frozen


def child_command(args, seeds, directory):
    command = [sys.executable, str(REPO_ROOT / "scripts/validate_extended_poles.py"),
               "--n", str(args.n), "--checkpoint", str(args.checkpoint.resolve()),
               "--task", args.task, "--engines", *args.engines,
               "--physics-substeps", str(args.physics_substeps),
               "--tracking-r", str(args.tracking_r), "--balance-r", str(args.balance_r),
               "--inverse-iterations", str(args.inverse_iterations),
               "--inverse-tolerance", str(args.inverse_tolerance), "--out", str(directory.resolve())]
    if args.plan:
        command += ["--plan", str(args.plan.resolve())]
    if args.mujoco_physics_substeps is not None:
        command += ["--mujoco-physics-substeps", str(args.mujoco_physics_substeps)]
    if args.mujoco_native_damping:
        command.append("--mujoco-native-damping")
    for name, value in (("horizon", args.horizon), ("tail-steps", args.tail_steps)):
        if value is not None:
            command += ["--" + name, str(value)]
    if args.final:
        # The child accepts absolute final seeds, not a zero-based offset, and
        # --final is deliberately omitted because it would select all 32 again.
        command += ["--final-shard-start", str(seeds[0]), "--final-shard-size", str(len(seeds))]
    else:
        command += ["--episodes", str(len(seeds)), "--seed", str(seeds[0])]
    return command


def run_worker(args, index, seeds, configuration):
    validate_artifacts(configuration)
    directory = args.out / "shards" / f"npole{args.n}" / f"worker_{index:03d}"
    directory.mkdir(parents=True, exist_ok=True)
    command = child_command(args, seeds, directory)
    env = os.environ.copy()
    env.update(configuration["worker_runtime"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    log = directory / f"worker-{stamp}.log"
    started = time.perf_counter()
    with log.open("w") as output:
        output.write(json.dumps({"command": command, "seeds": seeds, "runtime": configuration["worker_runtime"]}) + "\n")
        output.flush()
        result = subprocess.run(command, cwd=REPO_ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
    record = {"worker": index, "seeds": seeds, "directory": str(directory.resolve()),
              "log": str(log.resolve()), "returncode": result.returncode,
              "wall_seconds": time.perf_counter() - started, "command": command}
    (directory / f"worker-{stamp}-result.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({key: record[key] for key in ("worker", "seeds", "returncode", "wall_seconds")}), flush=True)
    return record


def combine_engine(n, engine, seeds, shard_directories, configuration, global_hash, out, elapsed):
    """Losslessly combine receipts/arrays; failures are ordinary retained rows."""
    global_indices = {seed: i for i, seed in enumerate(seeds)}
    metrics_by_seed, arrays, summaries, first = {}, {}, [], None
    maximum_decode, maximum_residual, worker_seconds = [], [], 0.
    invariant_fields = ("n", "engine", "physics_device", "physics_hz", "control_hz", "delay_steps", "task",
                        "horizon_ticks", "required_final_hold_ticks", "angle_limit_rad", "angular_velocity_limit_rad_s",
                        "rail_limit_m", "policy", "controller", "checkpoint_sha256", "plan_sha256",
                        "mujoco_native_damping", "numerical_damping_mode", "mjcf_sha256", "mujoco_version")
    for directory in shard_directories:
        directory = Path(directory)
        receipt_path = directory / f"npole{n}-{engine}.json"
        replay_path = directory / f"npole{n}-{engine}-replays.npz"
        frozen_path = directory / f"npole{n}-frozen.json"
        report = json.loads(receipt_path.read_text())
        child_configuration = json.loads(frozen_path.read_text())["configuration"]
        if report["frozen_configuration_sha256"] != digest(frozen_path) or report["replay_sha256"] != digest(replay_path):
            raise ValueError(f"child artifact hash mismatch: {directory}")
        if report["seeds"] != child_configuration["seeds"] or len(report["episode_metrics"]) != report["episodes"]:
            raise ValueError("child seed/episode counts do not agree")
        if report["controller"] != configuration["controller"] or report["checkpoint_sha256"] != configuration["checkpoint_sha256"] or report["plan_sha256"] != configuration["plan_sha256"]:
            raise ValueError("child controller or frozen model/plan differs from the global freeze")
        if child_configuration["source_hashes"] != {source: configuration["source_hashes"][source] for source in SOURCE_FILES}:
            raise ValueError("child source snapshot differs from the global freeze")
        for source, expected in child_configuration["source_hashes"].items():
            if digest(directory / "sources" / f"npole{n}" / source) != expected:
                raise ValueError("child archived source does not match its frozen hash")
        if report["physics_hz"] != configuration["engine_physics_hz"][engine]:
            raise ValueError("child engine integration rate differs from the global freeze")
        if first is None:
            first = report
        else:
            for field in invariant_fields:
                if report.get(field) != first.get(field):
                    raise ValueError(f"incompatible child receipt field: {field}")
        local_seeds = report["seeds"]
        for seed, metric in zip(local_seeds, report["episode_metrics"]):
            if seed not in global_indices or seed in metrics_by_seed or metric["seed"] != seed:
                raise ValueError("unknown, repeated or misaligned child seed")
            metrics_by_seed[seed] = metric
        with np.load(replay_path, allow_pickle=False) as archive:
            for key in archive.files:
                match = EPISODE_KEY.match(key)
                if match is None or int(match[1]) >= len(local_seeds):
                    raise ValueError(f"unknown child replay key: {key}")
                global_index = global_indices[local_seeds[int(match[1])]]
                new_key = f"episode_{global_index}_{match[2]}"
                if new_key in arrays:
                    raise ValueError("duplicate global replay key")
                arrays[new_key] = archive[key].copy()
        for field, values in (("max_decode_error", maximum_decode), ("max_inverse_residual", maximum_residual)):
            if report.get(field) is not None:
                values.append(report[field])
        worker_seconds += report["evaluation_seconds"]
        summaries.append({"receipt": str(receipt_path.resolve()), "receipt_sha256": digest(receipt_path),
                          "replay": str(replay_path.resolve()), "replay_sha256": digest(replay_path),
                          "child_frozen_configuration_sha256": digest(frozen_path),
                          "partition": report["partition"], "seeds": local_seeds,
                          "success_count": report["success_count"]})
    if set(metrics_by_seed) != set(seeds):
        raise ValueError("some global seeds are missing from the child receipts")
    metrics = [metrics_by_seed[seed] for seed in seeds]
    count = sum(bool(metric["success"]) for metric in metrics)
    report = dict(first)
    report.update(episodes=len(seeds), seeds=seeds, episode_metrics=metrics,
                  success_count=count, success_rate=count / len(seeds), **{"pass": count / len(seeds) >= 0.8},
                  mean_final_hold_ticks=float(np.mean([metric["final_hold_ticks"] for metric in metrics])),
                  minimum_final_hold_ticks=min(metric["final_hold_ticks"] for metric in metrics),
                  rail_integrity_rate=float(np.mean([bool(metric["rail_ok"]) for metric in metrics])),
                  numeric_integrity_rate=float(np.mean([bool(metric["numeric_ok"]) for metric in metrics])),
                  max_decode_error=max(maximum_decode, default=None), max_inverse_residual=max(maximum_residual, default=None),
                  partition=configuration["partition"], frozen_configuration_sha256=global_hash,
                  global_frozen_configuration_sha256=global_hash,
                  configuration_frozen_before_seed_initialization=True, training_during_evaluation=False,
                  aggregation="all child episodes preserved and ordered by the global frozen seed list",
                  shards=summaries, worker_evaluation_seconds_sum=worker_seconds, evaluation_seconds=elapsed)
    replay = out / f"npole{n}-{engine}-replays.npz"
    np.savez_compressed(replay, **arrays)
    report["replay_sha256"] = digest(replay)
    (out / f"npole{n}-{engine}.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"n": n, "engine": engine, "success_count": count, "episodes": len(seeds),
                      "minimum_final_hold_ticks": report["minimum_final_hold_ticks"], "pass": report["pass"]}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, choices=range(4, 11), required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/jepa-extended/model.pt"))
    parser.add_argument("--task", choices=("swingup", "balance"), default="swingup")
    parser.add_argument("--engines", nargs="+", choices=("euler", "rk4", "mujoco"), default=["euler", "mujoco"])
    parser.add_argument("--physics-substeps", type=int, default=20)
    parser.add_argument("--mujoco-physics-substeps", type=int)
    parser.add_argument("--mujoco-native-damping", action="store_true")
    parser.add_argument("--tracking-r", type=float, default=0.2)
    parser.add_argument("--balance-r", type=float, default=1000.)
    parser.add_argument("--inverse-iterations", type=int, default=10)
    parser.add_argument("--inverse-tolerance", type=float, default=1e-12)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--tail-steps", type=int)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=540000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.task == "swingup" and args.plan is None:
        parser.error("swing-up needs a frozen plan")
    if args.workers < 1 or args.episodes < 1 or args.seed < 0 or args.physics_substeps < 1 or (args.mujoco_physics_substeps is not None and args.mujoco_physics_substeps < 1):
        parser.error("counts/substeps must be positive and seed nonnegative")
    if len(set(args.engines)) != len(args.engines):
        parser.error("engines must be unique")
    if args.final:
        args.episodes, args.seed = 32, (600000 if args.n <= 7 else args.n * 100000)
    seeds = list(range(args.seed, args.seed + args.episodes))
    shards = split_seeds(seeds, args.workers)
    configuration, frozen = freeze(args, shards)
    print(json.dumps({"global_frozen_sha256": digest(frozen), "partition": configuration["partition"],
                      "seeds": seeds, "workers": len(shards), "shards": shards}), flush=True)
    started, records = time.perf_counter(), []
    with ThreadPoolExecutor(max_workers=len(shards)) as executor:
        futures = [executor.submit(run_worker, args, i, shard, configuration) for i, shard in enumerate(shards)]
        for future in as_completed(futures):
            records.append(future.result())
    records.sort(key=lambda row: row["worker"])
    (args.out / f"npole{args.n}-workers.json").write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
    validate_artifacts(configuration)
    failed = [record for record in records if record["returncode"] != 0]
    if failed:
        raise SystemExit(f"{len(failed)} child processes failed; all logs/outputs retained, no complete aggregate produced")
    passed = True
    for engine in args.engines:
        report = combine_engine(args.n, engine, seeds, [row["directory"] for row in records],
                                configuration, digest(frozen), args.out, time.perf_counter() - started)
        passed &= report["pass"]
    if not passed:
        raise SystemExit("complete aggregate includes control failures; inspect preserved per-episode receipts")


if __name__ == "__main__":
    main()
