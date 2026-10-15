#!/usr/bin/env python3
"""Export every trained JEPA tensor for complete Torch-free forward inference."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from rage_mill.jepa_data import load_trajectories
from rage_mill.jepa_train import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "src/rage_mill/assets"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    source = ROOT / "artifacts/npole10/jepa/model.pt"
    model, norms, checkpoint = load_checkpoint(source, "cpu")
    tensors = {name: value.detach().cpu().numpy().copy() for name, value in model.state_dict().items()}
    assert all(a.dtype == np.float32 and np.isfinite(a).all() for a in tensors.values())
    metadata = {"format_version": 1, "model_config": checkpoint["model_config"],
                "environments": [{"id": env_id, "observation_dim": dims[0], "action_dim": dims[1]}
                                 for env_id, dims in model.env_dims.items()],
                "source_checkpoint": "artifacts/npole10/jepa/model.pt", "source_checkpoint_sha256": sha(source),
                "parameter_count": sum(a.size for a in tensors.values()),
                "trainable_parameters": 78784, "frozen_target_parameters": 29120,
                "training_or_quantization": False, "saved_dtype": "float32"}
    arrays = tensors.copy()
    for i, env_id in enumerate(model.env_dims):
        arrays[f"normalization.body_{i}.mean"] = norms[env_id].mean
        arrays[f"normalization.body_{i}.std"] = norms[env_id].std
    arrays["metadata"] = json.dumps(metadata, sort_keys=True)
    output = ASSETS / "npole10-world-model.npz"
    np.savez_compressed(output, **arrays)
    evaluation = ROOT / "artifacts/npole10/jepa/data/npole10_evaluation.npz"
    data = load_trajectories(evaluation)
    example = ASSETS / "npole10-world-model-example.npz"
    np.savez_compressed(example, observations=data.observations[:1, 0], actions=data.actions[:1, :8])
    receipt = {**metadata, "tensor_count": len(tensors), "parameter_bytes_float32": sum(a.nbytes for a in tensors.values()),
               "source_evaluation_sha256": sha(evaluation), "example_episode": 0, "example_tick": 0,
               "outputs": {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in (output, example)},
               "torch_version": torch.__version__, "numpy_version": np.__version__}
    (ASSETS / "npole10-world-model.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
