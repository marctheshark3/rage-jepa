"""Reproducible matched classical / original JEPA / NumPy JEPA evaluation."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import platform
import time

import numpy as np

from .baseline import ASSETS, ClassicalPolicy
from .env import CartPendulumEnv, task_gates
from .pole_control import initial_state

CONDITIONS = ("classical", "original-jepa", "numpy-jepa")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_policy(condition, checkpoint=None):
    if condition == "classical":
        return ClassicalPolicy()
    if condition == "numpy-jepa":
        from .numpy_policy import NumpyJEPAPolicy
        return NumpyJEPAPolicy()
    if condition == "original-jepa":
        import torch
        torch.set_num_threads(1)
        from .jepa_state_control import JEPAStatePolePolicy
        plan = ClassicalPolicy().trajectory
        return JEPAStatePolePolicy(checkpoint, 10, teacher_factory=ClassicalPolicy,
                                   reference_plan=plan, iterations=20, residual_tolerance=1e-12)
    raise ValueError(f"unknown condition: {condition}")


def portable_provenance(provenance):
    result = provenance.copy()
    if "shared_library_path" in result:
        result["shared_library_path"] = "<user-cache>/rage-jepa/pole-cpu/" + "/".join(Path(result["shared_library_path"]).parts[-2:])
    return result


def run_job(job):
    engine, seed, conditions, checkpoint, output = job
    results = []
    # Rotate all conditions for each shared seed; no policy-specific reset draw.
    offset = seed % len(conditions)
    order = conditions[offset:] + conditions[:offset]
    for order_index, condition in enumerate(order):
        setup = time.perf_counter()
        env = CartPendulumEnv(engine=engine, backend="c" if engine == "euler" else "numpy")
        start = initial_state(10, "swingup", seed)
        env.reset(start)
        policy = make_policy(condition, checkpoint)
        setup_seconds = time.perf_counter() - setup
        states, commands, applied = [start.copy()], [], []
        estimates, residuals, inference_ns, physics_ns, refinements = [], [], [], [], []
        error = None
        warmup_seconds = None
        begin = time.perf_counter()
        try:
            warmup = time.perf_counter()
            policy.act(start, 0., 0.)
            policy.reset()
            warmup_seconds = time.perf_counter() - warmup
            begin = time.perf_counter()
            for tick in range(1000):
                t = time.perf_counter_ns()
                command = policy.act(states[-1], env.last_applied, env.pending)
                inference_ns.append(time.perf_counter_ns() - t)
                if condition != "classical":
                    estimates.append(policy.last_estimate.copy())
                    residuals.append(policy.last_residual)
                    refinements.append(getattr(policy, "last_refinements", -1))
                t = time.perf_counter_ns()
                state = env.step(command)
                physics_ns.append(time.perf_counter_ns() - t)
                commands.append(command)
                applied.append(env.last_applied)
                states.append(state)
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc), "completed_ticks": len(commands)}
        episode_seconds = time.perf_counter() - begin
        arrays = {"states": np.asarray(states, dtype=np.float64).reshape(-1, 22),
                  "commands": np.asarray(commands, dtype=np.float64),
                  "applied_actions": np.asarray(applied, dtype=np.float64),
                  "inferred_states": np.asarray(estimates, dtype=np.float64).reshape(-1, 22),
                  "inverse_residuals": np.asarray(residuals, dtype=np.float64),
                  "inference_ns": np.asarray(inference_ns, dtype=np.int64),
                  "physics_ns": np.asarray(physics_ns, dtype=np.int64),
                  "refinements": np.asarray(refinements, dtype=np.int32)}
        gates = task_gates(arrays["states"], arrays["commands"], arrays["applied_actions"])
        if condition != "classical":
            gates["strict_inverse"] = bool(len(residuals) == 1000 and np.isfinite(residuals).all()
                                             and max(residuals) <= 1e-12)
            gates["recovered_state"] = bool(len(estimates) == 1000 and len(states) == 1001
                and np.max(np.abs(arrays["inferred_states"] - arrays["states"][:-1])) < 1e-10)
        gates["pass"] = error is None and all(gates.values())
        stem = f"{engine}-{condition}-{seed}"
        archive = Path(output) / f"{stem}.npz"
        np.savez_compressed(archive, **arrays)
        result = {"engine": engine, "condition": condition, "seed": seed,
                  "condition_order": order_index, "gates": gates, "error": error,
                  "setup_seconds": setup_seconds, "warmup_seconds": warmup_seconds,
                  "episode_seconds": episode_seconds, "completed_ticks": len(commands),
                  "initial_state_sha256": hashlib.sha256(start.tobytes()).hexdigest(),
                  "archive": archive.name, "archive_sha256": sha(archive),
                  "policy": policy.to_dict(), "backend": portable_provenance(env.backend_provenance)}
        (Path(output) / f"{stem}.json").write_text(json.dumps(result, indent=2) + "\n")
        results.append(result)
    return results


def source_hashes():
    package = Path(__file__).parent
    paths = sorted(p for p in package.rglob("*") if p.suffix in (".py", ".c", ".npz", ".json") and "__pycache__" not in p.parts)
    return {str(p.relative_to(package)): sha(p) for p in paths}


def statistics(values):
    if not len(values):
        return {"count": 0, "median": None, "p95": None, "mean": None, "max": None}
    return {"median": float(np.median(values)), "p95": float(np.percentile(values, 95)),
            "mean": float(np.mean(values)), "max": float(np.max(values))}


def summarize(output, definition, results, wall_seconds):
    summary = {"definition_sha256": sha(output / "definition.json"), "wall_seconds": wall_seconds,
               "trials": len(results), "groups": [],
               "timing_limit": "Shared-host concurrent cohort timings; use benchmark_runtime.py for serial latency comparisons."}
    for engine in definition["engines"]:
        for condition in definition["conditions"]:
            selected = [r for r in results if r["engine"] == engine and r["condition"] == condition]
            inference, physics = [], []
            for r in selected:
                with np.load(output / r["archive"], allow_pickle=False) as data:
                    inference.extend(data["inference_ns"] / 1e6)
                    physics.extend(data["physics_ns"] / 1e6)
            successes = sum(r["gates"]["pass"] for r in selected)
            summary["groups"].append({"engine": engine, "condition": condition, "successes": successes,
                "trials": len(selected), "passes_80_percent_gate": successes / len(selected) >= .8,
                "failed_seeds": [r["seed"] for r in selected if not r["gates"]["pass"]],
                "inference_ms": statistics(inference), "physics_tick_ms": statistics(physics),
                "episode_seconds": statistics([r["episode_seconds"] for r in selected]),
                "setup_seconds": statistics([r["setup_seconds"] for r in selected])})
    (output / "episodes.json").write_text(json.dumps(results, indent=2) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="new output directory; never overwrite a cohort")
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--seed-start", type=int, default=3100000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--engines", nargs="+", choices=("euler", "mujoco"), default=["euler", "mujoco"])
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/npole10/jepa/model.pt"))
    args = parser.parse_args()
    if args.count < 1 or args.workers < 1 or args.seed_start < 0:
        parser.error("count/workers must be positive and seed-start nonnegative")
    if len(set(args.conditions)) != len(args.conditions) or len(set(args.engines)) != len(args.engines):
        parser.error("duplicate engines or conditions")
    if "original-jepa" in args.conditions and not args.checkpoint.is_file():
        parser.error("original-jepa requires the saved checkpoint (and jepa extra)")
    # Fail early on dependency/backend availability, before freezing seeds.
    from .swing_trajectory import symbolic_dynamics
    symbolic_dynamics(10)
    if "mujoco" in args.engines:
        import mujoco
    if "original-jepa" in args.conditions:
        import torch
        torch.set_num_threads(1)
    backend = None
    if "euler" in args.engines:
        from .pole_cpu import backend_provenance
        backend = portable_provenance(backend_provenance(10))
    args.out.mkdir(parents=True, exist_ok=False)
    definition = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "conditions": args.conditions, "engines": args.engines,
        "seeds": list(range(args.seed_start, args.seed_start + args.count)),
        "workers": args.workers, "n": 10, "horizon": 1000, "control_hz": 50,
        "euler_substeps": 12800, "mujoco_substeps": 20, "delay_ticks": 1, "force_limit_N": 10,
        "native_mujoco_absolute_damping": True, "rail_limit_m": 2.4,
        "final_window_endpoints": 251, "angle_limit_rad": .22, "omega_limit_rad_s": 1.,
        "minimum_success_fraction": .8, "inverse_residual_tolerance": 1e-12,
        "reset": "seeded hanging: cart +/-0.08m, angles pi+/-0.04rad, all velocities +/-0.02",
        "common_feedback": "exact exported R20/R1000 float64 gains and original CasADi RK4 delay predictor",
        "checkpoint_sha256": sha(args.checkpoint) if "original-jepa" in args.conditions else None,
        "source_sha256": source_hashes(), "euler_backend": backend,
        "machine": platform.machine(), "python": platform.python_version(),
        "versions": {name: importlib.metadata.version(name) for name in
                     ["numpy", "casadi"] + (["mujoco"] if "mujoco" in args.engines else [])
                     + (["torch"] if "original-jepa" in args.conditions else [])},
        "thread_environment": {key: os.environ.get(key) for key in
                               ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "torch_threads": 1, "blas_threads_required": 1, "physics_device": "cpu"}
    (args.out / "definition.json").write_text(json.dumps(definition, indent=2) + "\n")
    jobs = [(engine, seed, args.conditions, str(args.checkpoint.resolve()), str(args.out.resolve()))
            for seed in definition["seeds"] for engine in args.engines]
    started = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for batch in pool.map(run_job, jobs):
            results.extend(batch)
            print(json.dumps({"completed": len(results), "of": len(jobs) * len(args.conditions),
                              "seed": batch[0]["seed"], "engine": batch[0]["engine"],
                              "passes": sum(r["gates"]["pass"] for r in batch)}), flush=True)
    if source_hashes() != definition["source_sha256"]:
        raise RuntimeError("runtime source or assets changed during cohort; receipts are not qualified")
    results.sort(key=lambda r: (r["engine"], r["seed"], r["condition"]))
    summary = summarize(args.out, definition, results, time.perf_counter() - started)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
