"""Complete frozen JEPA forward inference with NumPy as its only dependency.

Online encoder, EMA target encoder, action projection, environment embedding,
and residual autoregressive dynamics are all loaded from the trained model.
This is a latent predictor; exporting it does not improve its forecast quality
or qualify a learned controller. Training and Torch are deliberately absent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np

ASSETS = Path(__file__).with_name("assets")


def _silu(value):
    # Algebraically identical SiLU with a stable sigmoid for both signs.
    exponent = np.exp(-np.abs(value))
    sigmoid = np.where(value >= 0, 1. / (1. + exponent), exponent / (1. + exponent))
    return value * sigmoid


class NumpyWorldModel:
    """Inference counterpart of MultiEnvJEPA, with no optimizer or autograd.

    Observations are raw by default and normalized with saved training-only
    statistics. Set ``normalized=True`` for the original model's normalized
    input contract. Actions are passed through unchanged, without clipping.
    Float32 is the original training dtype; float64 is also supported.
    """

    def __init__(self, path: str | Path = ASSETS / "npole10-world-model.npz", *, dtype=np.float32):
        self.dtype = np.dtype(dtype)
        if self.dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("inference dtype must be float32 or float64")
        path = Path(path)
        self.artifact_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            if metadata.get("format_version") != 1:
                raise ValueError("unsupported full world-model export format")
            self.latent_dim = self._dimension(metadata["model_config"]["latent_dim"])
            self.hidden_dim = self._dimension(metadata["model_config"]["hidden_dim"])
            environments = metadata["environments"]
            if not isinstance(environments, list) or not environments:
                raise ValueError("world model must declare at least one environment")
            self.env_dims, self._indices, self._keys = {}, {}, {}
            expected = {"env_embeddings.weight": (len(environments), self.latent_dim)}
            normalization_shapes = {}
            for i, entry in enumerate(environments):
                env_id = entry["id"]
                if not isinstance(env_id, str) or not env_id or env_id in self.env_dims:
                    raise ValueError("environment IDs must be unique nonempty strings")
                obs, act = self._dimension(entry["observation_dim"]), self._dimension(entry["action_dim"])
                self.env_dims[env_id] = (obs, act)
                self._indices[env_id], self._keys[env_id] = i, f"body_{i}"
                for group in ("encoders", "target_encoders"):
                    expected.update(self._mlp_shapes(f"{group}.body_{i}", obs))
                expected[f"action_projections.body_{i}.0.weight"] = (self.latent_dim, act)
                expected[f"action_projections.body_{i}.0.bias"] = (self.latent_dim,)
                normalization_shapes[f"normalization.body_{i}.mean"] = (obs,)
                normalization_shapes[f"normalization.body_{i}.std"] = (obs,)
            expected.update(self._mlp_shapes("predictor", self.latent_dim * 3))
            if set(data.files) != set(expected) | set(normalization_shapes) | {"metadata"}:
                raise ValueError("missing or unexpected full world-model arrays")
            weights, normalizations = {}, {}
            for name, shape in {**expected, **normalization_shapes}.items():
                array = data[name]
                if array.dtype != np.float32 or array.shape != shape or not np.isfinite(array).all():
                    raise ValueError(f"invalid frozen tensor: {name}")
                value = np.array(array, dtype=self.dtype, order="C", copy=True)
                if name.endswith(".std") and np.any(value <= 0):
                    raise ValueError("normalization standard deviations must be positive")
                value.setflags(write=False)
                (weights if name in expected else normalizations)[name] = value
        self.weights = MappingProxyType(weights)
        self._normalizations = normalizations
        self.metadata = metadata
        self.parameter_count = sum(a.size for a in weights.values())
        if metadata.get("parameter_count") != self.parameter_count:
            raise ValueError("world-model parameter count does not match its export")
        self.parameter_bytes = sum(a.nbytes for a in weights.values())
        self.normalization_bytes = sum(a.nbytes for a in normalizations.values())

    @staticmethod
    def _dimension(value):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("model dimensions must be positive integers")
        return value

    def _mlp_shapes(self, prefix, nin):
        return {f"{prefix}.{i}.{name}": shape
                for i, incoming, outgoing in ((0, nin, self.hidden_dim),
                                               (2, self.hidden_dim, self.hidden_dim),
                                               (4, self.hidden_dim, self.latent_dim))
                for name, shape in (("weight", (outgoing, incoming)), ("bias", (outgoing,)))}

    def _key(self, env_id):
        if env_id not in self.env_dims:
            raise ValueError(f"unknown environment {env_id!r}")
        return self._keys[env_id]

    def _array(self, name, value, ndim, last):
        result = np.asarray(value, dtype=self.dtype)
        if result.ndim != ndim or any(d == 0 for d in result.shape) or result.shape[-1] != last or not np.isfinite(result).all():
            raise ValueError(f"{name} must be a finite nonempty {ndim}D array ending in {last}")
        return result

    @staticmethod
    def _finite(result):
        if not np.isfinite(result).all():
            raise RuntimeError("world model produced nonfinite predictions")
        return result

    def _mlp(self, prefix, value):
        for i in (0, 2, 4):
            value = value @ self.weights[f"{prefix}.{i}.weight"].T + self.weights[f"{prefix}.{i}.bias"]
            if i != 4:
                value = _silu(value)
        return self._finite(value)

    def normalize(self, env_id, observations):
        key = self._key(env_id)
        obs = self._array("observations", observations, 2, self.env_dims[env_id][0])
        return self._finite((obs - self._normalizations[f"normalization.{key}.mean"])
                            / self._normalizations[f"normalization.{key}.std"])

    def encode(self, env_id, observations, *, target=False, normalized=False):
        """Encode [batch, observation_dim] to [batch, latent_dim]."""
        key = self._key(env_id)
        obs = (self._array("observations", observations, 2, self.env_dims[env_id][0])
               if normalized else self.normalize(env_id, observations))
        return self._mlp(f"{'target_encoders' if target else 'encoders'}.{key}", obs)

    def project_actions(self, env_id, actions):
        """Apply the saved learned action adapter to [batch, horizon, action_dim]."""
        key = self._key(env_id)
        actions = self._array("actions", actions, 3, self.env_dims[env_id][1])
        value = actions @ self.weights[f"action_projections.{key}.0.weight"].T
        value = value + self.weights[f"action_projections.{key}.0.bias"]
        return self._finite(_silu(value))

    def rollout_latent(self, env_id, latent, actions):
        """Run the learned residual predictor autoregressively, without future observations."""
        self._key(env_id)
        latent = self._array("latent", latent, 2, self.latent_dim)
        projected = self.project_actions(env_id, actions)
        if len(projected) != len(latent):
            raise ValueError("latent and actions must have the same batch size")
        embedding = np.broadcast_to(self.weights["env_embeddings.weight"][self._indices[env_id]], latent.shape)
        predictions = []
        for i in range(projected.shape[1]):
            latent = latent + self._mlp("predictor", np.concatenate((latent, projected[:, i], embedding), axis=-1))
            predictions.append(latent)
        return self._finite(np.stack(predictions, axis=1))

    def rollout(self, env_id, observations, actions, *, normalized=False):
        """Predict [batch, horizon, latent_dim] from current observations and actions."""
        initial = self.encode(env_id, observations, normalized=normalized)
        return self.rollout_latent(env_id, initial, actions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ASSETS / "npole10-world-model.npz")
    parser.add_argument("--inputs", type=Path, default=ASSETS / "npole10-world-model-example.npz")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--out", type=Path, help="optional NPZ containing the complete latent predictions")
    args = parser.parse_args()
    model = NumpyWorldModel(args.model, dtype=args.dtype)
    with np.load(args.inputs, allow_pickle=False) as data:
        observations, actions = data["observations"], data["actions"]
    predictions = model.rollout("npole10", observations, actions)
    if args.out:
        np.savez_compressed(args.out, predicted_latents=predictions)
    print(json.dumps({"full_world_model": True, "learned_dynamics_predictor_executed": True,
                      "training": False, "parameter_count": model.parameter_count,
                      "dtype": str(model.dtype), "prediction_shape": list(predictions.shape),
                      "all_finite": bool(np.isfinite(predictions).all()),
                      "model_sha256": model.artifact_sha256}, indent=2))


if __name__ == "__main__":
    main()
