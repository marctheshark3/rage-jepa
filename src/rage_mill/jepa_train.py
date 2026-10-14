"""Train a shared state/action JEPA from random weights; collect physics on CPU.

Whole episodes are held out before normalization. A frozen final checkpoint is
evaluated on fresh trajectories from a second engine, with explicit baselines.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import platform
from pathlib import Path
import re
import time

import numpy as np
import torch

from rage_mill.jepa import JEPAConfig, MultiEnvJEPA
from rage_mill.jepa_data import (
    ALL_ENVIRONMENTS,
    ENVIRONMENTS,
    PREFERRED_ENGINES,
    Trajectories,
    collect_trajectories,
    load_trajectories,
    sample_windows,
    save_trajectories,
)


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 200
    updates_per_epoch: int = 4
    batch_size: int = 128
    horizon: int = 8
    learning_rate: float = 1e-3
    seed: int = 17
    eval_samples: int = 512

    def __post_init__(self):
        for key in ("epochs", "updates_per_epoch", "batch_size", "horizon", "eval_samples"):
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.batch_size < 2 or self.eval_samples < 2:
            raise ValueError("batch_size and eval_samples must be at least two")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")


@dataclass(frozen=True)
class Normalization:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, data: Trajectories) -> Normalization:
        obs = data.observations.astype(np.float64)
        return cls(obs.mean(axis=(0, 1)).astype(np.float32),
                   np.maximum(obs.std(axis=(0, 1)), 1e-3).astype(np.float32))

    def apply(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray((obs - self.mean) / self.std, dtype=np.float32)

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


def split_episodes(data: Trajectories, seed: int, train_fraction: float = .75):
    data.validate()
    count = len(data.observations)
    if count < 4:
        raise ValueError("At least four episodes are needed for an episode-heldout split")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between zero and one")
    order = np.random.default_rng(seed).permutation(count)
    cut = max(2, min(count - 2, int(count * train_fraction)))
    parts = []
    for label, indices in (("train", order[:cut]), ("validation", order[cut:])):
        metadata = {**data.metadata, "split": label, "episode_indices": indices.tolist()}
        timestamps = None if data.timestamps is None else data.timestamps[indices]
        parts.append(Trajectories(data.observations[indices], data.actions[indices], metadata, timestamps))
    return tuple(parts)


def _tensor(array: np.ndarray, device: torch.device):
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def _probe_fit(model, env_id, data, normalization, device, seed, count=2048):
    """Fit an evaluation-only linear readout; it never enters the JEPA loss."""
    rng = np.random.default_rng(seed)
    flat = data.observations.reshape(-1, data.observations.shape[-1])
    index = rng.choice(len(flat), size=min(count, len(flat)), replace=False)
    obs = normalization.apply(flat[index])
    with torch.no_grad():
        z = model.encode(env_id, _tensor(obs, device), target=True).cpu().numpy()
    design = np.column_stack((z, np.ones(len(z)))).astype(np.float64)
    penalty = np.eye(design.shape[1]) * 1e-2
    penalty[-1, -1] = 0
    return np.linalg.solve(design.T @ design + penalty, design.T @ obs)


def _r2(prediction: np.ndarray, target: np.ndarray):
    residual = np.sum((prediction - target) ** 2, axis=0)
    variance = np.sum((target - target.mean(axis=0)) ** 2, axis=0)
    active = variance > 1e-6
    scores = np.zeros_like(variance)
    np.divide(residual, variance, out=scores, where=active)
    scores = 1 - scores
    return float(scores[active].mean()) if active.any() else None


@torch.no_grad()
def evaluate(model, env_id, data, normalization, device, config, *, probe=None):
    model.eval()
    observations, actions = sample_windows(
        data, config.horizon, config.eval_samples, np.random.default_rng(config.seed + 1001))
    normalized = normalization.apply(observations)
    obs, act = _tensor(normalized, device), _tensor(actions, device)
    predicted = model.rollout(env_id, obs[:, 0], act)
    target = model.encode(env_id, obs[:, 1:].reshape(-1, obs.shape[-1]), target=True)
    target = target.reshape_as(predicted)
    # Persistence must use the target encoder too: EMA lag is not dynamics.
    current = model.encode(env_id, obs[:, 0], target=True).unsqueeze(1).expand_as(target)
    permutation = np.random.default_rng(config.seed + 2001).permutation(len(actions))
    shuffled = model.rollout(env_id, obs[:, 0], act[torch.as_tensor(permutation, device=device)])
    zero = model.rollout(env_id, obs[:, 0], torch.zeros_like(act))
    mse = ((predicted - target) ** 2).mean(dim=(0, 2)).cpu().numpy()
    persistence = ((current - target) ** 2).mean(dim=(0, 2)).cpu().numpy()
    shuffled_mse = float(((shuffled - target) ** 2).mean())
    zero_mse = float(((zero - target) ** 2).mean())
    centered = target.reshape(-1, target.shape[-1])
    std = centered.std(dim=0, unbiased=False)
    variance = float(std.square().mean())
    eigenvalues = torch.linalg.eigvalsh(torch.atleast_2d(torch.cov(centered.T))).clamp_min(0)
    weights = eigenvalues / eigenvalues.sum().clamp_min(1e-12)
    effective_rank = float(torch.exp(-(weights * weights.clamp_min(1e-12).log()).sum()))
    mean_mse = float(mse.mean())
    persistence_mse = float(persistence.mean())
    mean_std = float(std.mean())
    noncollapsed = mean_std >= .05 and effective_rank >= 2
    metrics = {
        "episodes": len(data.observations),
        "source_kind": data.metadata["source_kind"],
        "engine": data.metadata["engine"],
        "horizon_steps": config.horizon,
        "horizon_seconds": config.horizon / float(data.metadata["control_hz"]),
        "samples": config.eval_samples,
        "latent_mse": mean_mse,
        "normalized_latent_mse": mean_mse / max(variance, 1e-12),
        "persistence_mse": persistence_mse,
        "persistence_ratio": mean_mse / max(persistence_mse, 1e-12),
        "shuffled_action_mse": shuffled_mse,
        "zero_action_mse": zero_mse,
        "shuffled_action_ratio": shuffled_mse / max(mean_mse, 1e-12),
        "mean_latent_std": mean_std,
        "min_latent_std": float(std.min()),
        "effective_rank": effective_rank,
        "noncollapsed": noncollapsed,
        "per_horizon_mse": mse.tolist(),
        "per_horizon_persistence_mse": persistence.tolist(),
        "predictive_pass": noncollapsed and mean_mse < .95 * persistence_mse
                           and shuffled_mse > 1.05 * mean_mse,
    }
    if probe is not None:
        target_flat = target.cpu().numpy().reshape(-1, target.shape[-1])
        pred_flat = predicted.cpu().numpy().reshape(-1, predicted.shape[-1])
        truth = normalized[:, 1:].reshape(-1, obs.shape[-1])
        decoded = np.column_stack((pred_flat, np.ones(len(pred_flat)))) @ probe
        state_decoded = np.column_stack((target_flat, np.ones(len(target_flat)))) @ probe
        obs_persistence = np.repeat(normalized[:, :1], config.horizon, axis=1).reshape(truth.shape)
        metrics.update({
            "state_probe_r2": _r2(state_decoded, truth),
            "prediction_probe_r2": _r2(decoded, truth),
            "observation_persistence_r2": _r2(obs_persistence, truth),
            "prediction_probe_mse": float(np.mean((decoded - truth) ** 2)),
            "observation_persistence_mse": float(np.mean((obs_persistence - truth) ** 2)),
        })
        keys = data.metadata.get("observation_keys") or [f"feature_{i}" for i in range(truth.shape[1])]
        metrics["per_feature_probe_r2"] = {
            key: {"state": _r2(state_decoded[:, i:i+1], truth[:, i:i+1]),
                  "prediction": _r2(decoded[:, i:i+1], truth[:, i:i+1]),
                  "persistence": _r2(obs_persistence[:, i:i+1], truth[:, i:i+1])}
            for i, key in enumerate(keys)
        }
        physical = [i for i, key in enumerate(keys)
                    if not any(token in key.lower() for token in ("last", "pwm", "bias", "command"))
                    and np.var(truth[:, i]) > 1e-6]
        if physical and "observation_keys" in data.metadata:
            phys_pred = float(np.mean((decoded[:, physical] - truth[:, physical]) ** 2))
            phys_base = float(np.mean((obs_persistence[:, physical] - truth[:, physical]) ** 2))
            phys_state_r2 = _r2(state_decoded[:, physical], truth[:, physical])
            metrics.update({
                "physical_observation_keys": [keys[i] for i in physical],
                "physical_state_probe_r2": phys_state_r2,
                "physical_prediction_probe_r2": _r2(decoded[:, physical], truth[:, physical]),
                "physical_prediction_probe_mse": phys_pred,
                "physical_observation_persistence_mse": phys_base,
                "physical_forecast_pass": phys_state_r2 is not None and phys_state_r2 >= .5
                                          and phys_pred < .95 * phys_base,
            })
        else:
            metrics["physical_forecast_pass"] = None
    return metrics


def save_checkpoint(path, model, normalizations, config, provenance):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": 1,
        "env_dims": model.env_dims,
        "model_config": asdict(model.config),
        "model_state": model.state_dict(),
        "normalizations": {key: norm.to_dict() for key, norm in normalizations.items()},
        "train_config": asdict(config),
        "provenance": provenance,
    }, path)


def load_checkpoint(path, device="cpu"):
    contents = torch.load(path, map_location=device, weights_only=True)
    if contents.get("format_version") != 1:
        raise ValueError("Unsupported JEPA checkpoint format")
    model = MultiEnvJEPA(contents["env_dims"], JEPAConfig(**contents["model_config"])).to(device)
    model.load_state_dict(contents["model_state"])
    model.eval().requires_grad_(False)
    normalizations = {
        key: Normalization(np.asarray(values["mean"], dtype=np.float32),
                           np.asarray(values["std"], dtype=np.float32))
        for key, values in contents["normalizations"].items()
    }
    return model, normalizations, contents


def _check_pair(env_id, data, evaluation):
    for trajectories in (data, evaluation):
        if trajectories.metadata["source_kind"] == "hardware" and not \
           trajectories.metadata.get("sampling_matches_declared_hz", False):
            raise ValueError(f"{env_id}: hardware timestamps do not support fixed-rate trajectory windows")
    for key in ("env_id", "body", "q_semantics", "control_hz", "observation_keys", "actuator_model"):
        if key == "env_id" and (data.metadata.get(key) != env_id or evaluation.metadata.get(key) != env_id):
            raise ValueError(f"{env_id}: dataset environment id differs from its model adapter")
        if data.metadata.get(key) != evaluation.metadata.get(key):
            raise ValueError(f"{env_id}: train/evaluation {key} disagree")
    if data.observation_dim != evaluation.observation_dim or data.action_dim != evaluation.action_dim:
        raise ValueError(f"{env_id}: train/evaluation dimensions disagree")
    def episode_hashes(trajectories):
        return {hashlib.sha256(obs.tobytes() + act.tobytes()).hexdigest()
                for obs, act in zip(trajectories.observations, trajectories.actions)}
    if episode_hashes(data) & episode_hashes(evaluation):
        raise ValueError(f"{env_id}: training and external evaluation contain identical episodes")


def _artifact_id(env_id):
    slug = re.sub(r"[^A-Za-z0-9_-]", "_", env_id)[:64] or "environment"
    return slug if slug == env_id else slug + "_" + hashlib.sha256(env_id.encode()).hexdigest()[:12]


def train(datasets, evaluation_datasets, out, config=TrainConfig(), model_config=JEPAConfig(),
          device="cpu"):
    """Fit one model across all supplied bodies, save weights and measured receipts."""
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if set(datasets) != set(evaluation_datasets):
        raise ValueError("Training and evaluation must contain the same environment ids")
    if not datasets:
        raise ValueError("At least one environment is required")
    torch.manual_seed(config.seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    training, validation, normalizations, provenance = {}, {}, {}, {}
    dimensions = {}
    for index, (env_id, data) in enumerate(datasets.items()):
        data.validate()
        eval_data = evaluation_datasets[env_id]
        eval_data.validate()
        _check_pair(env_id, data, eval_data)
        if config.horizon > min(data.actions.shape[1], eval_data.actions.shape[1]):
            raise ValueError(f"{env_id}: horizon exceeds episode length")
        training[env_id], validation[env_id] = split_episodes(data, config.seed + index)
        normalizations[env_id] = Normalization.fit(training[env_id])
        dimensions[env_id] = (data.observations.shape[-1], data.actions.shape[-1])
        artifact_id = _artifact_id(env_id)
        save_trajectories(data, out / "data" / f"{artifact_id}_train.npz")
        save_trajectories(eval_data, out / "data" / f"{artifact_id}_evaluation.npz")
        provenance[env_id] = {"training": data.metadata, "evaluation": eval_data.metadata,
                              "train_episodes": training[env_id].metadata["episode_indices"],
                              "validation_episodes": validation[env_id].metadata["episode_indices"]}
    model = MultiEnvJEPA(dimensions, model_config).to(device)
    initial = {}
    for env_id in datasets:
        random_probe = _probe_fit(model, env_id, training[env_id], normalizations[env_id], device, config.seed)
        initial[env_id] = evaluate(model, env_id, evaluation_datasets[env_id],
                                  normalizations[env_id], device, config, probe=random_probe)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=config.learning_rate, weight_decay=1e-4)
    rng = np.random.default_rng(config.seed)
    history = []
    for epoch in range(config.epochs):
        model.train()
        totals = {env_id: [] for env_id in datasets}
        for _ in range(config.updates_per_epoch):
            for env_id in datasets:
                obs, act = sample_windows(training[env_id], config.horizon, config.batch_size, rng)
                optimizer.zero_grad(set_to_none=True)
                losses = model.loss(env_id, _tensor(normalizations[env_id].apply(obs), device),
                                    _tensor(act, device))
                if not torch.isfinite(losses["total"]):
                    raise RuntimeError(f"{env_id}: training diverged at epoch {epoch + 1}")
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                model.update_target()
                totals[env_id].append({key: float(value.detach()) for key, value in losses.items()})
        record = {"epoch": epoch + 1, "environments": {
            env_id: {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
            for env_id, rows in totals.items()}}
        history.append(record)
        if epoch == 0 or (epoch + 1) % max(1, config.epochs // 10) == 0:
            print(json.dumps(record), flush=True)
    # Select no weights on the heldout second engine: evaluate the final fixed model.
    model.eval()
    final = {}
    for env_id in datasets:
        probe = _probe_fit(model, env_id, training[env_id], normalizations[env_id], device, config.seed)
        simulated_pair = datasets[env_id].metadata["source_kind"] == "simulation" and \
                         evaluation_datasets[env_id].metadata["source_kind"] == "simulation"
        independent_engine = simulated_pair and datasets[env_id].metadata["engine"] != \
                             evaluation_datasets[env_id].metadata["engine"]
        final[env_id] = {
            "evaluation_kind": "independent_engine" if independent_engine else
                               ("recorded_hardware" if evaluation_datasets[env_id].metadata["source_kind"] == "hardware"
                                else "same_engine_holdout"),
            "initial_evaluation": initial[env_id],
            "validation": evaluate(model, env_id, validation[env_id], normalizations[env_id],
                                   device, config, probe=probe),
            "evaluation": evaluate(model, env_id, evaluation_datasets[env_id], normalizations[env_id],
                                      device, config, probe=probe),
        }
    checkpoint = out / "model.pt"
    save_checkpoint(checkpoint, model, normalizations, config, provenance)
    report = {
        "format_version": 1,
        "model": "from-scratch state/action JEPA with shared predictor",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "initialization": "random; no pretrained weights",
        "device": str(device), "machine": platform.machine(),
        "torch_version": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "compute_capability": list(torch.cuda.get_device_capability(device)) if device.type == "cuda" else None,
        "physics_device": "cpu",
        "parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "model_config": asdict(model_config), "train_config": asdict(config),
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "identified": False,
        "hardware_deployment": False,
        "evaluation_policy": "Final fixed model; no model selection on evaluation trajectories. Probes fit training episodes only.",
        "criterion": "Within-body mean latent std >=0.05 and effective rank >=2; MSE <0.95x persistence; shuffled actions MSE >1.05x correct actions.",
        "environments": final, "provenance": provenance,
        "all_predictive_pass": all(row["validation"]["predictive_pass"] and
                                  row["evaluation"]["predictive_pass"] for row in final.values()),
        "all_physical_forecast_pass": all(row["validation"]["physical_forecast_pass"] is True and
                                         row["evaluation"]["physical_forecast_pass"] is True
                                         for row in final.values()),
    }
    (out / "history.json").write_text(json.dumps(history, indent=2, allow_nan=False) + "\n")
    (out / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"metrics": str(out / "metrics.json"), "checkpoint": str(checkpoint),
                      "all_predictive_pass": report["all_predictive_pass"]}), flush=True)
    return model, normalizations, report


def _dataset_arguments(items):
    result = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError("Datasets must use ENV_ID=PATH")
        env_id, path = item.split("=", 1)
        if not env_id or not path or env_id in result:
            raise ValueError(f"Invalid or duplicate dataset argument: {item}")
        result[env_id] = Path(path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", nargs="+", default=list(ENVIRONMENTS))
    parser.add_argument("--out", type=Path, default=Path("runs/jepa"))
    parser.add_argument("--dataset", action="append", metavar="ENV=PATH")
    parser.add_argument("--eval-dataset", action="append", metavar="ENV=PATH")
    parser.add_argument("--episodes", type=int, default=48)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--updates-per-epoch", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--eval-samples", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    try:
        config = TrainConfig(epochs=args.epochs, updates_per_epoch=args.updates_per_epoch,
                             batch_size=args.batch_size, horizon=args.horizon,
                             learning_rate=args.learning_rate, seed=args.seed, eval_samples=args.eval_samples)
        model_config = JEPAConfig(latent_dim=args.latent_dim, hidden_dim=args.hidden_dim)
        train_paths = _dataset_arguments(args.dataset)
        eval_paths = _dataset_arguments(args.eval_dataset)
        if len(set(args.envs)) != len(args.envs):
            raise ValueError("Environment ids must be unique")
        if (set(train_paths) | set(eval_paths)) - set(args.envs):
            raise ValueError("Dataset environment ids must be selected with --envs")
        datasets, evaluation = {}, {}
        for index, env_id in enumerate(args.envs):
            if env_id not in ALL_ENVIRONMENTS and (env_id not in train_paths or env_id not in eval_paths):
                raise ValueError(f"{env_id}: custom environments require both dataset paths")
            engines = PREFERRED_ENGINES.get(env_id)
            print(f"Preparing {env_id}", flush=True)
            datasets[env_id] = load_trajectories(train_paths[env_id]) if env_id in train_paths else \
                collect_trajectories(env_id, episodes=args.episodes, steps=args.steps,
                                     seed=args.seed + index, engine=engines[0])
            evaluation[env_id] = load_trajectories(eval_paths[env_id]) if env_id in eval_paths else \
                collect_trajectories(env_id, episodes=max(8, args.episodes // 4), steps=args.steps,
                                     seed=args.seed + 10000 + index, engine=engines[1])
            for data in (datasets[env_id], evaluation[env_id]):
                if data.metadata.get("env_id") != env_id:
                    raise ValueError(f"Dataset body id differs from {env_id}")
        _, _, report = train(datasets, evaluation, args.out, config, model_config, args.device)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        parser.exit(2, f"rage-jepa: {exc}\n")
    if args.require_pass and not report["all_predictive_pass"]:
        parser.exit(1, "Some environments failed the stated predictive criterion; inspect metrics.json\n")


if __name__ == "__main__":
    main()
