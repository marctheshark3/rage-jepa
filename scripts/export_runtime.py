#!/usr/bin/env python3
"""Export portable runtime arrays from the frozen public checkpoint/reference.

Requires the jepa and simulation extras. This does not train or optimize.
"""
from pathlib import Path
import hashlib
import json
import platform
import time
import numpy as np
import torch
from rage_mill.jepa_train import load_checkpoint
from rage_mill.swing_trajectory import SwingTrajectory, TVLQRTracker

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    checkpoint = ROOT / "artifacts/npole10/jepa/model.pt"
    plan_path = ROOT / "artifacts/npole10/verified/npole10-rk44-18s-feasible.npz"
    out = ROOT / "src/rage_mill/assets"
    out.mkdir(exist_ok=True)
    started = time.perf_counter()
    model, norms, _ = load_checkpoint(checkpoint, "cpu")
    encoder = model.target_encoders[model._key("npole10")]
    arrays = {"mean": norms["npole10"].mean, "std": norms["npole10"].std}
    for i, layer in enumerate(m for m in encoder.modules() if isinstance(m, torch.nn.Linear)):
        arrays[f"weight_{i}"] = layer.weight.detach().numpy()
        arrays[f"bias_{i}"] = layer.bias.detach().numpy()
    encoder_path = out / "npole10-target-encoder.npz"
    np.savez_compressed(encoder_path, **arrays)
    plan = SwingTrajectory.load(plan_path)
    policy = TVLQRTracker(plan, delay=1, r=20., balance_r=1000.)
    controller_path = out / "npole10-classical.npz"
    np.savez_compressed(controller_path, states=plan.states, actions=plan.actions,
                        gains=policy.gains, balance_k=policy.balance.k)
    import casadi
    receipt = {"method": "export_existing_target_tensors_and_executed_classical_gains",
               "training_or_optimization": False, "checkpoint_sha256": sha(checkpoint),
               "reference_sha256": sha(plan_path), "tensor_dtype_saved": "float32",
               "runtime_dtype": "float64", "quantization": False,
               "torch_version": torch.__version__, "numpy_version": np.__version__,
               "casadi_version": casadi.__version__, "platform": platform.machine(),
               "export_seconds": time.perf_counter() - started,
               "tracking_r": 20., "balance_r": 1000., "delay": 1,
               "riccati": policy.to_dict().get("riccati"),
               "balance_riccati": policy.balance.to_dict().get("riccati"),
               "outputs": {p.name: {"sha256": sha(p), "bytes": p.stat().st_size}
                           for p in (encoder_path, controller_path)}}
    (out / "npole10-runtime.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
