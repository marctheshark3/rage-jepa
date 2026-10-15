"""Explicit compiled CPU Euler acceleration for the released point-mass chains.

This optional path changes execution speed, not the point-mass equations or
the control clock. It never silently falls back to NumPy. Compiler and binary
provenance are part of each environment's evaluation receipt.
"""
from __future__ import annotations

import ctypes
from functools import lru_cache
import hashlib
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile

import numpy as np

COMPILE_FLAGS = ("-std=c11", "-O3", "-fno-fast-math", "-ffp-contract=off",
                 "-fPIC", "-shared")
SOURCE = Path(__file__).with_name("assets") / "pole_cpu.c"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def _library():
    import fcntl

    compiler = shutil.which("cc")
    if compiler is None:
        raise RuntimeError("compiled_cpu requires an available C compiler; no fallback is permitted")
    compiler_path = str(Path(compiler).resolve())
    version = subprocess.run([compiler_path, "--version"], check=True, text=True,
                             capture_output=True).stdout.strip()
    source_sha = _sha(SOURCE)
    compiler_sha = _sha(Path(compiler_path))
    key = hashlib.sha256(repr((source_sha, compiler_sha, version, COMPILE_FLAGS,
                              platform.machine())).encode()).hexdigest()
    cache = Path.home() / ".cache" / "rage-jepa" / "pole-cpu" / key
    cache.mkdir(parents=True, exist_ok=True)
    binary = cache / "pole_cpu.so"
    with (cache / "compile.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not binary.exists():
            with tempfile.TemporaryDirectory(dir=cache) as tmp:
                output = Path(tmp) / "pole_cpu.so"
                completed = subprocess.run([compiler_path, *COMPILE_FLAGS, str(SOURCE),
                                            "-o", str(output), "-lm"], text=True,
                                           capture_output=True)
                if completed.returncode:
                    raise RuntimeError(f"compiled_cpu compiler failed: {completed.stderr.strip()}")
                output.replace(binary)
    library = ctypes.CDLL(str(binary))
    pointer = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
    library.rage_pole_acceleration.argtypes = [pointer, ctypes.c_double, ctypes.c_int, pointer]
    library.rage_pole_acceleration.restype = ctypes.c_int
    library.rage_pole_euler_tick.argtypes = [pointer, ctypes.c_double, ctypes.c_int,
                                           ctypes.c_int, ctypes.c_double]
    library.rage_pole_euler_tick.restype = ctypes.c_int
    provenance = {"backend": "compiled_cpu", "arithmetic_dtype": "float64",
                  "method": "analytic_point_mass_partial_pivot_lu_velocity_first_euler",
                  "source_sha256": source_sha, "compiler_path": compiler_path,
                  "compiler_sha256": compiler_sha, "compiler_version": version,
                  "compiler_flags": list(COMPILE_FLAGS), "linker_flags": ["-lm"],
                  "machine": platform.machine(), "shared_library_path": str(binary),
                  "shared_library_sha256": _sha(binary), "silent_fallback": False}
    return library, provenance


def provenance() -> dict:
    import copy
    return copy.deepcopy(_library()[1])


def backend_provenance(n: int, backend: str = "c") -> dict:
    from .pole_control import MAX_POLES
    if backend != "c" or isinstance(n, bool) or not isinstance(n, (int, np.integer)) or not 2 <= n <= MAX_POLES:
        raise ValueError(f"compiled CPU provenance requires backend='c' and integer n in 2..{MAX_POLES}")
    result = provenance()
    result.update({"analytic_cpu_backend": "c", "pole_count": int(n)})
    return result


def _input(state, action, n):
    from .pole_control import MAX_POLES
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or not 2 <= n <= MAX_POLES:
        raise ValueError(f"compiled_cpu supports integer counts 2..{MAX_POLES}")
    state = np.array(state, dtype=np.float64, copy=True, order="C")
    value = np.asarray(action)
    if state.shape != (2 * (n + 1),) or not np.isfinite(state).all() or value.size != 1 or not np.isfinite(value).all():
        raise ValueError("compiled_cpu requires a finite physical state and one finite action")
    return state, float(np.clip(value.reshape(-1)[0], -1., 1.))


def _checked(status):
    if status:
        messages = {1: "invalid input", 2: "singular mass solve", 3: "nonfinite physics"}
        raise RuntimeError(f"compiled_cpu failed: {messages.get(status, 'unknown kernel status')}")


def acceleration(state, action, n) -> np.ndarray:
    state, action = _input(state, action, n)
    output = np.empty(n + 1, dtype=np.float64)
    _checked(_library()[0].rage_pole_acceleration(state, action, n, output))
    return output


def euler_tick(state, action, n, substeps, dt) -> np.ndarray:
    state, action = _input(state, action, n)
    if isinstance(substeps, bool) or not isinstance(substeps, int) or not 1 <= substeps <= 2**31 - 1 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("compiled_cpu requires positive finite integration settings")
    _checked(_library()[0].rage_pole_euler_tick(state, action, n, substeps, dt))
    return state
