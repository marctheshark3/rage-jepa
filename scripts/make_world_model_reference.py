#!/usr/bin/env python3
"""Freeze independent Torch outputs for full world-model inference parity."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from rage_mill.jepa_data import Trajectories, load_trajectories, sample_windows
from rage_mill.jepa_train import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence/inference-v0.3"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ROOT / "artifacts/npole10/jepa/model.pt"
    model, norms, checkpoint = load_checkpoint(checkpoint_path, "cpu")
    training = ROOT / "artifacts/npole10/jepa/data/npole10_train.npz"
    evaluation = ROOT / "artifacts/npole10/jepa/data/npole10_evaluation.npz"
    data = load_trajectories(training)
    indices = checkpoint["provenance"]["npole10"]["validation_episodes"]
    heldout = Trajectories(data.observations[indices], data.actions[indices], data.metadata)
    datasets = {"heldout-euler": heldout, "independent-mujoco": load_trajectories(evaluation)}
    definition = {"source_checkpoint_sha256": sha(checkpoint_path), "seed": 4830000,
        "scope": "Full frozen forward model: online/target encoders, action projection, environment embedding, learned residual dynamics",
        "acceptance_tolerances": {"float32": {"atol": 1e-5, "rtol": 1e-5},
                                  "float64": {"atol": 1e-11, "rtol": 1e-11}},
        "horizons": [1, 8, 32], "main_batch": 128, "training": False,
        "variants": ["recorded", "zero", "shuffled"],
        "source_datasets": {"heldout-euler": sha(training), "independent-mujoco": sha(evaluation)},
        "euler_validation_episode_indices": indices,
        "reference_code_sha256": sha(ROOT / "src/rage_mill/jepa.py"),
        "runtime_code_sha256": sha(ROOT / "src/rage_mill/world_model.py"),
        "torch_version": torch.__version__, "numpy_version": np.__version__}
    # Declare inputs/gates before evaluating the candidate.
    (OUT / "definition.json").write_text(json.dumps(definition, indent=2) + "\n")
    arrays = {f"tensor__{name}": tensor.detach().cpu().numpy().copy()
              for name, tensor in model.state_dict().items()}
    cases = []
    for engine_index, (engine, data) in enumerate(datasets.items()):
        for batch, horizon in ((128, 8), (8, 32), (1, 1)):
            windows, actions = sample_windows(data, horizon, batch,
                np.random.default_rng(definition["seed"] + engine_index * 1000 + horizon))
            variants = ["recorded", "zero", "shuffled"] if batch == 128 else ["recorded"]
            for variant in variants:
                candidate_actions = actions.copy()
                if variant == "zero":
                    candidate_actions[:] = 0.
                elif variant == "shuffled":
                    candidate_actions = candidate_actions[np.random.default_rng(4831000).permutation(batch)]
                for dtype, torch_dtype in (("float32", torch.float32), ("float64", torch.float64)):
                    model.to(dtype=torch_dtype)
                    raw = windows[:, 0].astype(dtype)
                    normalized = (raw - norms["npole10"].mean.astype(dtype)) / norms["npole10"].std.astype(dtype)
                    act = candidate_actions.astype(dtype)
                    obs_tensor, act_tensor = torch.from_numpy(normalized), torch.from_numpy(act)
                    with torch.inference_mode():
                        online = model.encode("npole10", obs_tensor).numpy()
                        target = model.encode("npole10", obs_tensor, target=True).numpy()
                        predictions = model.rollout("npole10", obs_tensor, act_tensor).numpy()
                    name = f"case_{len(cases):03d}"
                    for key, value in {"observations": raw, "normalized": normalized, "actions": act,
                                       "online": online, "target": target, "predictions": predictions}.items():
                        arrays[f"{name}_{key}"] = value
                    cases.append({"id": name, "dataset": engine, "batch": batch, "horizon": horizon,
                                  "variant": variant, "dtype": dtype})
    reference = OUT / "torch-reference.npz"
    np.savez_compressed(reference, **arrays)
    metrics = json.loads((ROOT / "artifacts/npole10/jepa/metrics.json").read_text())
    description = {"definition_sha256": sha(OUT / "definition.json"), "reference_sha256": sha(reference),
                   "tensor_count": 21, "parameter_count": 107904, "cases": cases,
                   "original_predictive_pass": metrics["all_predictive_pass"],
                   "original_physical_forecast_pass": metrics["all_physical_forecast_pass"],
                   "claim": "Numerical equivalence and inference execution; forecast quality remains the original model's."}
    (OUT / "reference.json").write_text(json.dumps(description, indent=2) + "\n")
    print(json.dumps({"cases": len(cases), "reference_bytes": reference.stat().st_size,
                      "learned_predictor_reference_executed": True, "original_forecast_failures_retained": True}, indent=2))


if __name__ == "__main__":
    main()
