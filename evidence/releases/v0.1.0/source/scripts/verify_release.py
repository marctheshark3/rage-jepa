#!/usr/bin/env python3
"""Verify released hashes and independently audit all saved final trials."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from rage_mill.jepa import JEPAConfig, MultiEnvJEPA
from rage_mill.jepa_data import load_trajectories
from rage_mill.jepa_train import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]

def require(condition, message):
    if not condition:
        raise ValueError(message)

def main():
    manifest = json.loads((ROOT / "release-manifest.json").read_text())
    for entry in manifest["files"]:
        path = (ROOT / entry["path"]).resolve()
        require(path.is_relative_to(ROOT) and path.is_file(), "Invalid manifest path")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], f"Hash mismatch: {entry['path']}")
    base = ROOT / "artifacts/npole10"
    model, normalizations, metadata = load_checkpoint(base / "jepa/model.pt")
    architecture = MultiEnvJEPA(metadata["env_dims"], JEPAConfig(**metadata["model_config"]))
    count = sum(p.numel() for p in architecture.parameters() if p.requires_grad)
    require(count == 78784 and model.env_dims == {"npole10": (33, 1)}, "Model contract mismatch")
    data = load_trajectories(base / "jepa/data/npole10_train.npz")
    require(data.episodes == 48 and data.steps == 128, "Training dataset shape mismatch")
    obs = torch.as_tensor(normalizations["npole10"].apply(data.observations[:, 0]))
    latent = model.encode("npole10", obs, target=True)
    require(latent.shape == (48, 64) and torch.isfinite(latent).all(), "Encoder failed")
    metrics = json.loads((base / "jepa/metrics.json").read_text())
    require(not metrics["all_predictive_pass"] and not metrics["all_physical_forecast_pass"], "Forecast failures were lost")
    for engine, rate in (("euler", 640000), ("mujoco", 1000)):
        report = json.loads((base / f"final/npole10-{engine}.json").read_text())
        require(report["seeds"] == list(range(1000000, 1000032)), "Final seed coverage mismatch")
        require(report["physics_hz"] == rate and report["control_hz"] == 50, "Rate mismatch")
        with np.load(base / f"final/npole10-{engine}-replays.npz", allow_pickle=False) as archive:
            require(len(archive.files) == 32 * 6, "Final arrays missing or added")
            for episode in range(32):
                prefix = f"episode_{episode}_"
                states, commands = archive[prefix + "states"], archive[prefix + "commands"]
                applied = archive[prefix + "applied_actions"]
                decoded, residuals = archive[prefix + "inferred_states"], archive[prefix + "inverse_residuals"]
                require(states.shape == (1001, 22) and commands.shape == (1000,), "Incomplete trial")
                require(np.isfinite(states).all() and np.isfinite(commands).all(), "Nonfinite physics")
                require(np.max(np.abs(states[:, 0])) < 2.4 and np.max(np.abs(commands)) <= 1., "Rail/force violation")
                require(applied[0] == 0 and np.array_equal(applied[1:], commands[:-1]), "One-tick queue violation")
                angles = (states[-251:, 1:11] + np.pi) % (2 * np.pi) - np.pi
                require(np.max(np.abs(angles)) < .22 and np.max(np.abs(states[-251:, 12:])) < 1., "Final hold failed")
                require(np.max(residuals) <= 1e-12 and np.isfinite(decoded).all(), "Strict inversion failed")
                require(np.max(np.abs(decoded - states[:-1])) < 1e-10, "Recovered state mismatch")
                initial_angles = (states[0, 1:11] - np.pi + np.pi) % (2*np.pi) - np.pi
                require(np.max(np.abs(initial_angles)) <= .04 and abs(states[0, 0]) <= .08
                        and np.max(np.abs(states[0, 11:])) <= .02, "Hanging reset mismatch")
    print(json.dumps({"files_verified": len(manifest["files"]), "trainable_parameters": count,
                      "final_trials_audited": 64, "final_endpoints_per_trial": 251,
                      "forecast_failures_retained": True, "physics_rerun": False}, indent=2))

if __name__ == "__main__":
    main()
