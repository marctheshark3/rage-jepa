#!/usr/bin/env python3
"""Measure full-WM CPU inference in a fresh NumPy-only installed environment."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time


def rss_bytes(counter="VmRSS"):
    # This probe targets the Linux ARM/x86 validation hosts.
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(counter + ":"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("Linux resident-memory counter unavailable")


def peak_bytes():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    for name in ("torch", "mujoco", "casadi", "gymnasium"):
        if importlib.util.find_spec(name) is not None:
            raise ValueError(f"expected NumPy-only inference environment; found {name}")
    before_numpy = rss_bytes()
    peak_before_numpy = {"kernel_vm_hwm_bytes": rss_bytes("VmHWM"), "rusage_maxrss_bytes": peak_bytes()}
    started = time.perf_counter()
    import numpy as np
    from rage_mill.world_model import ASSETS, NumpyWorldModel
    import rage_mill.world_model as module
    if not Path(module.__file__).is_relative_to(Path(sys.prefix)):
        raise ValueError("measure the noneditable installed wheel")
    import_seconds = time.perf_counter() - started
    after_numpy = rss_bytes()
    started = time.perf_counter()
    model = NumpyWorldModel()
    load_seconds = time.perf_counter() - started
    after_model = rss_bytes()
    with np.load(ASSETS / "npole10-world-model-example.npz", allow_pickle=False) as example:
        observations, recorded_actions = example["observations"], example["actions"]
    digest_before = {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in model.weights.items()}
    cases = []
    for batch, horizon, repetitions in ((1, 1, 2000), (1, 8, 1000), (1, 32, 500), (32, 8, 200)):
        obs = np.repeat(observations, batch, axis=0)
        actions = np.tile(recorded_actions, (batch, (horizon + 7) // 8, 1))[:, :horizon]
        for _ in range(50):
            model.rollout("npole10", obs, actions)
        timings = []
        for _ in range(repetitions):
            begin = time.perf_counter_ns()
            prediction = model.rollout("npole10", obs, actions)
            timings.append((time.perf_counter_ns() - begin) / 1e6)
        assert prediction.shape == (batch, horizon, 64) and np.isfinite(prediction).all()
        cases.append({"batch": batch, "horizon": horizon, "repetitions": repetitions,
                      "kernel_peak_rss_bytes": rss_bytes("VmHWM"), "rusage_maxrss_bytes": peak_bytes(),
                      "latency_ms": {"median": float(np.median(timings)), "p95": float(np.percentile(timings, 95)),
                                     "mean": float(np.mean(timings)), "max": float(np.max(timings))}})
    if digest_before != {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in model.weights.items()}:
        raise ValueError("inference changed frozen weights")
    if "torch" in sys.modules:
        raise ValueError("unexpected Torch import")
    prefix = Path(sys.prefix)
    files = [p for p in prefix.rglob("*") if p.is_file()]
    result = {"pass": True, "full_world_model": True, "learned_predictor_executed": True,
              "training": False, "torch_installed": False, "simulator_installed": False,
              "machine": platform.machine(), "python": platform.python_version(), "numpy": np.__version__,
              "thread_environment": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
              "dtype": str(model.dtype), "parameter_count": model.parameter_count,
              "parameter_bytes": model.parameter_bytes, "normalization_bytes": model.normalization_bytes,
              "export_file_bytes": (ASSETS / "npole10-world-model.npz").stat().st_size,
              "installed_payload_bytes_excluding_python_symlinks": sum(p.stat().st_size for p in files if not p.is_symlink()),
              "venv_apparent_bytes_following_python_symlinks": sum(p.stat().st_size for p in files),
              "memory": {"python_process_before_numpy_bytes": before_numpy,
                         "after_numpy_and_runtime_import_bytes": after_numpy, "after_model_load_bytes": after_model,
                         "after_inference_bytes": rss_bytes(), "kernel_peak_process_rss_bytes": rss_bytes("VmHWM"),
                         "rusage_peak_bytes": peak_bytes(), "peak_before_numpy": peak_before_numpy},
              "import_seconds": import_seconds, "model_load_seconds": load_seconds, "cases": cases,
              "model_sha256": model.artifact_sha256, "weights_unchanged": True,
              "installed_distributions": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
              "limits": "Shared-host CPU latency with warm caches; Linux VmRSS/VmHWM include Python/NumPy. getrusage high-water is also recorded and may retain pre-exec launcher history. Installed payload excludes shared Python executable symlinks and includes all runtime assets."}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
