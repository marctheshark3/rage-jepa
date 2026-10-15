#!/usr/bin/env python3
"""Verify released hashes and independently audit all saved final trials."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from rage_mill.jepa_data import load_trajectories

ROOT = Path(__file__).resolve().parents[1]

def require(condition, message):
    if not condition:
        raise ValueError(message)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-model", action="store_true", help="also load the original Torch checkpoint (jepa extra)")
    args = parser.parse_args()
    maintenance_relocations = json.loads((ROOT / "evidence/releases/v0.3.0/relocations.json").read_text())["paths"]
    require(set(maintenance_relocations) == {"tests/test_jepa_train.py", "scripts/verify_release.py", "scripts/build_manifest.py"}, "Unexpected validation relocations")
    legacy_test = {"tests/test_jepa_train.py": maintenance_relocations["tests/test_jepa_train.py"]}
    manifest = json.loads((ROOT / "release-manifest.json").read_text())
    relocations = json.loads((ROOT / "evidence/releases/v0.1.0/relocations.json").read_text())["paths"]
    require(set(relocations) == {"README.md", "pyproject.toml", "uv.lock", ".github/workflows/ci.yml", "scripts/verify_release.py"}, "Unexpected historical relocations")
    for entry in manifest["files"]:
        path = (ROOT / relocations.get(entry["path"], legacy_test.get(entry["path"], entry["path"]))).resolve()
        require(path.is_relative_to(ROOT) and path.is_file(), "Invalid manifest path")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], f"Hash mismatch: {entry['path']}")
    runtime_manifest = ROOT / "release-v0.2-manifest.json"
    require(runtime_manifest.is_file(), "Missing v0.2 release manifest")
    runtime_relocations = json.loads((ROOT / "evidence/releases/v0.2.0/relocations.json").read_text())["paths"]
    require(set(runtime_relocations) == {"README.md", "pyproject.toml", "uv.lock", ".github/workflows/ci.yml",
                                         "scripts/verify_release.py", "scripts/build_manifest.py"}, "Unexpected v0.2 relocations")
    current_count = 0
    if runtime_manifest.exists():
        for entry in json.loads(runtime_manifest.read_text())["files"]:
            path = (ROOT / runtime_relocations.get(entry["path"], legacy_test.get(entry["path"], entry["path"]))).resolve()
            require(path.is_relative_to(ROOT) and path.is_file(), "Invalid runtime manifest path")
            require(hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], f"Runtime hash mismatch: {entry['path']}")
            current_count += 1
    inference_count = 0
    inference_manifest = ROOT / "release-v0.3-manifest.json"
    require(inference_manifest.is_file(), "Missing v0.3 release manifest")
    for entry in json.loads(inference_manifest.read_text())["files"]:
        path = (ROOT / maintenance_relocations.get(entry["path"], entry["path"])).resolve()
        require(path.is_relative_to(ROOT) and path.is_file(), "Invalid inference manifest path")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], f"Inference hash mismatch: {entry['path']}")
        inference_count += 1
    maintenance_count = 0
    maintenance_manifest = ROOT / "validation-manifest.json"
    require(maintenance_manifest.is_file(), "Missing current validation manifest")
    for entry in json.loads(maintenance_manifest.read_text())["files"]:
        path = (ROOT / entry["path"]).resolve()
        require(path.is_relative_to(ROOT) and path.is_file(), "Invalid validation manifest path")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], f"Validation hash mismatch: {entry['path']}")
        maintenance_count += 1
    base = ROOT / "artifacts/npole10"
    data = load_trajectories(base / "jepa/data/npole10_train.npz")
    require(data.episodes == 48 and data.steps == 128, "Training dataset shape mismatch")
    count = None
    if args.with_model:
        import torch
        from rage_mill.jepa import JEPAConfig, MultiEnvJEPA
        from rage_mill.jepa_train import load_checkpoint
        model, normalizations, metadata = load_checkpoint(base / "jepa/model.pt")
        architecture = MultiEnvJEPA(metadata["env_dims"], JEPAConfig(**metadata["model_config"]))
        count = sum(p.numel() for p in architecture.parameters() if p.requires_grad)
        require(count == 78784 and model.env_dims == {"npole10": (33, 1)}, "Model contract mismatch")
        obs = torch.as_tensor(normalizations["npole10"].apply(data.observations[:, 0]))
        latent = model.encode("npole10", obs, target=True)
        require(latent.shape == (48, 64) and torch.isfinite(latent).all(), "Encoder failed")
    from rage_mill.baseline import ASSETS
    from rage_mill.numpy_policy import NumpyEncoder
    runtime = json.loads((ASSETS / "npole10-runtime.json").read_text())
    for name, receipt in runtime["outputs"].items():
        require(hashlib.sha256((ASSETS / name).read_bytes()).hexdigest() == receipt["sha256"], "Runtime asset mismatch")
    require(NumpyEncoder().encode_states(np.zeros((1, 22))).shape == (1, 64), "NumPy encoder failed")
    from rage_mill.world_model import NumpyWorldModel
    full = NumpyWorldModel()
    require(full.parameter_count == 107904 and len(full.weights) == 21, "Full WM tensors missing")
    require(full.rollout("npole10", np.zeros((1, 33)), np.zeros((1, 8, 1))).shape == (1, 8, 64), "Full WM inference failed")
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
    print(json.dumps({"historical_files_verified": len(manifest["files"]), "runtime_files_verified": current_count,
                      "inference_files_verified": inference_count, "full_world_model_parameters": full.parameter_count,
                      "current_validation_files_verified": maintenance_count,
                      "historical_metadata_files_archived": len(relocations),
                      "torch_checkpoint_loaded": args.with_model, "trainable_parameters": count,
                      "final_trials_audited": 64, "final_endpoints_per_trial": 251,
                      "forecast_failures_retained": True, "physics_rerun": False}, indent=2))

if __name__ == "__main__":
    main()
