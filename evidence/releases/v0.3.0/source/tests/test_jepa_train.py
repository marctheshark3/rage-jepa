import hashlib
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn

from rage_mill.jepa import JEPAConfig, MultiEnvJEPA
from rage_mill.jepa_data import Trajectories, load_trajectories
from rage_mill.jepa_train import (
    Normalization,
    TrainConfig,
    evaluate,
    load_checkpoint,
    save_checkpoint,
    split_episodes,
    train,
)


def synthetic_trajectories(
    env_id="fixture", *, episodes=8, steps=6, obs_dim=3, action_dim=1,
    seed=1, engine="synthetic_fixture_train", hardware=False,
):
    """Small explicit fixtures; these do not identify any physical body."""
    rng = np.random.default_rng(seed)
    actions = rng.uniform(-0.8, 0.8, (episodes, steps, action_dim)).astype(np.float32)
    observations = np.zeros((episodes, steps + 1, obs_dim), dtype=np.float32)
    observations[:, 0] = rng.normal(0, 0.5, (episodes, obs_dim))
    projection = rng.normal(0, 0.1, (action_dim, obs_dim)).astype(np.float32)
    for t in range(steps):
        observations[:, t + 1] = 0.95 * observations[:, t] + actions[:, t] @ projection
    metadata = {
        "env_id": env_id,
        "body": env_id,
        "source_kind": "hardware" if hardware else "simulation",
        "engine": engine,
        "control_hz": 50.0,
        "q_semantics": "fixture_state_vector",
        "actuator_model": "normalized_synthetic_fixture_action",
        "source_hashes": {"test_fixture": hashlib.sha256(b"explicit synthetic test fixture").hexdigest()},
        "observation_keys": [f"state_{i}" for i in range(obs_dim)],
        "identified": False,
    }
    timestamps = np.broadcast_to(np.arange(steps + 1) * 0.02, (episodes, steps + 1)).copy() if hardware else None
    return Trajectories(observations, actions, metadata, timestamps).validate()


def tiny_config(**overrides):
    return TrainConfig(**{
        "epochs": 1, "updates_per_epoch": 1, "batch_size": 4,
        "horizon": 2, "eval_samples": 8, "seed": 7, **overrides,
    })


def test_whole_episode_split_is_disjoint_deterministic_and_preserves_identity():
    data = synthetic_trajectories(episodes=10)
    data.observations[:, :, 0] = np.arange(data.episodes)[:, None]
    first, heldout = split_episodes(data, seed=17)
    repeated, repeated_heldout = split_episodes(data, seed=17)
    train_ids = first.metadata["episode_indices"]
    heldout_ids = heldout.metadata["episode_indices"]
    assert not set(train_ids) & set(heldout_ids)
    assert sorted(train_ids + heldout_ids) == list(range(data.episodes))
    assert np.array_equal(first.observations, repeated.observations)
    assert np.array_equal(heldout.observations, repeated_heldout.observations)
    assert np.array_equal(first.observations[:, 0, 0], train_ids)
    assert np.array_equal(heldout.observations[:, 0, 0], heldout_ids)
    assert first.metadata["split"] == "train"
    assert heldout.metadata["split"] == "validation"
    assert "split" not in data.metadata


def test_split_requires_four_episodes_and_keeps_two_in_each_partition():
    with pytest.raises(ValueError, match="four episodes"):
        split_episodes(synthetic_trajectories(episodes=3), seed=0)
    training, heldout = split_episodes(synthetic_trajectories(episodes=4), seed=0)
    assert training.episodes == heldout.episodes == 2
    for fraction in (0, 1, -0.1, float("nan")):
        with pytest.raises(ValueError):
            split_episodes(synthetic_trajectories(), seed=0, train_fraction=fraction)


def test_hardware_split_preserves_measured_timestamps():
    data = synthetic_trajectories(hardware=True)
    data.timestamps += np.arange(data.episodes)[:, None] * 10.0
    training, heldout = split_episodes(data, seed=17)
    for part in (training, heldout):
        assert part.timestamps is not None
        assert np.array_equal(part.timestamps, data.timestamps[part.metadata["episode_indices"]])
        assert part.validate().metadata["sampling_matches_declared_hz"] is True
        assert part.metadata["identified"] is False


def test_normalization_uses_training_episodes_and_handles_constant_features():
    data = synthetic_trajectories(obs_dim=3)
    data.observations[:, :, -1] = 5.0
    training, heldout = split_episodes(data, seed=17)
    heldout.observations[:, :, 0] += 1000
    norm = Normalization.fit(training)
    assert np.allclose(norm.mean, training.observations.mean(axis=(0, 1)), atol=1e-6)
    assert norm.std[-1] == pytest.approx(1e-3)
    normalized = norm.apply(training.observations)
    assert normalized.dtype == np.float32
    assert np.allclose(normalized.mean(axis=(0, 1)), 0, atol=1e-5)
    assert np.allclose(normalized[:, :, -1], 0)
    assert norm.apply(heldout.observations)[:, :, 0].mean() > 100
    assert set(norm.to_dict()) == {"mean", "std"}
    json.dumps(norm.to_dict(), allow_nan=False)


def _assert_safe_checkpoint_value(value):
    assert not isinstance(value, (np.ndarray, np.generic))
    if isinstance(value, torch.Tensor):
        return
    if isinstance(value, dict):
        for key, member in value.items():
            assert isinstance(key, str)
            _assert_safe_checkpoint_value(member)
    elif isinstance(value, (tuple, list)):
        for member in value:
            _assert_safe_checkpoint_value(member)
    else:
        assert value is None or type(value) in (bool, int, float, str)


def test_checkpoint_is_weights_only_safe_frozen_and_preserves_all_body_predictions(tmp_path):
    torch.manual_seed(19)
    dimensions = {"zeta.body": (4, 2), "alpha": (3, 1)}
    model = MultiEnvJEPA(dimensions, JEPAConfig(latent_dim=4, hidden_dim=8, ema_decay=0.75))
    datasets = {env_id: synthetic_trajectories(env_id, obs_dim=obs_dim, action_dim=action_dim)
                for env_id, (obs_dim, action_dim) in dimensions.items()}
    norms = {env_id: Normalization.fit(data) for env_id, data in datasets.items()}
    # Save genuinely different online/target states, not just duplicate initialization.
    with torch.no_grad():
        for parameter in model.encoders.parameters():
            parameter.add_(0.05)
    model.update_target()
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, norms, tiny_config(), {key: value.metadata for key, value in datasets.items()})
    raw = torch.load(path, weights_only=True, map_location="cpu")
    _assert_safe_checkpoint_value(raw)
    restored, restored_norms, contents = load_checkpoint(path)
    assert list(restored.env_dims) == list(dimensions)
    assert list(contents["env_dims"]) == list(dimensions)
    assert not restored.training
    assert all(not parameter.requires_grad for parameter in restored.parameters())
    assert all(not module.training for module in restored.target_encoders.modules())
    for env_id, (obs_dim, action_dim) in dimensions.items():
        obs = torch.randn(2, obs_dim)
        actions = torch.randn(2, 3, action_dim)
        assert torch.equal(model.rollout(env_id, obs, actions), restored.rollout(env_id, obs, actions))
        assert torch.equal(model.encode(env_id, obs, target=True), restored.encode(env_id, obs, target=True))
        assert np.array_equal(norms[env_id].mean, restored_norms[env_id].mean)
        assert np.array_equal(norms[env_id].std, restored_norms[env_id].std)


def test_persistence_baseline_uses_frozen_target_space():
    class OffsetOnlineEncoder(nn.Module):
        def encode(self, _env_id, observations, target=False):
            return observations if target else observations + 3.0

        def rollout(self, env_id, observations, actions):
            return self.encode(env_id, observations)[:, None].expand(-1, actions.shape[1], -1)

    data = synthetic_trajectories(obs_dim=2)
    data.observations[:] = [1.0, 0.5]
    normalization = Normalization(np.zeros(2, np.float32), np.ones(2, np.float32))
    metrics = evaluate(OffsetOnlineEncoder(), "fixture", data, normalization, torch.device("cpu"), tiny_config())
    assert metrics["persistence_mse"] == pytest.approx(0.0)
    assert metrics["per_horizon_persistence_mse"] == pytest.approx([0.0, 0.0])
    assert metrics["latent_mse"] == pytest.approx(9.0)
    assert metrics["noncollapsed"] is False


def test_evaluate_handles_one_latent_dimension():
    data = synthetic_trajectories()
    model = MultiEnvJEPA({"fixture": (3, 1)}, JEPAConfig(latent_dim=1, hidden_dim=8))
    metrics = evaluate(model, "fixture", data, Normalization.fit(data), torch.device("cpu"), tiny_config())
    assert metrics["effective_rank"] == pytest.approx(1.0)
    assert metrics["noncollapsed"] is False
    json.dumps(metrics, allow_nan=False)


def test_physical_forecast_metric_excludes_previous_pwm_and_constant_bias():
    class CommandsOnlyWorldModel(nn.Module):
        def encode(self, _env_id, observations, target=False):
            return observations

        def rollout(self, _env_id, observations, actions):
            # Correct commanded PWM, but persistence for the actual position.
            position = observations[:, None, :1].expand(-1, actions.shape[1], -1)
            return torch.cat((position, actions, torch.ones_like(actions)), dim=-1)

    data = synthetic_trajectories(obs_dim=3)
    data.observations[:, :, 0] = np.arange(data.steps + 1)[None, :] * 0.1
    data.observations[:, 0, 1] = 0
    data.observations[:, 1:, 1] = data.actions[:, :, 0]
    data.observations[:, :, 2] = 1
    data.metadata["observation_keys"] = ["position_m", "last_pwm", "bias"]
    norm = Normalization(np.zeros(3, np.float32), np.ones(3, np.float32))
    identity_probe = np.vstack((np.eye(3), np.zeros(3)))
    metrics = evaluate(CommandsOnlyWorldModel(), "fixture", data, norm, torch.device("cpu"),
                       tiny_config(), probe=identity_probe)
    assert metrics["physical_observation_keys"] == ["position_m"]
    assert metrics["physical_state_probe_r2"] == pytest.approx(1)
    assert metrics["per_feature_probe_r2"]["last_pwm"]["prediction"] == pytest.approx(1)
    assert metrics["physical_prediction_probe_mse"] == pytest.approx(metrics["physical_observation_persistence_mse"])
    assert metrics["physical_forecast_pass"] is False


def test_tiny_shared_training_writes_reproducible_artifacts_without_live_physics(tmp_path):
    dimensions = {"zeta.body": (4, 2), "alpha": (3, 1)}
    datasets = {key: synthetic_trajectories(key, obs_dim=obs, action_dim=act, seed=1)
                for key, (obs, act) in dimensions.items()}
    evaluation = {key: synthetic_trajectories(key, obs_dim=obs, action_dim=act, seed=101,
                                             engine="synthetic_fixture_eval")
                  for key, (obs, act) in dimensions.items()}
    config = tiny_config()
    model, normalizations, report = train(
        datasets, evaluation, tmp_path, config,
        JEPAConfig(latent_dim=4, hidden_dim=8), device="cpu",
    )
    history = json.loads((tmp_path / "history.json").read_text())
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert len(history) == 1
    assert history[0]["epoch"] == 1
    assert set(history[0]["environments"]) == set(dimensions)
    assert set(metrics["environments"]) == set(dimensions)
    assert metrics["identified"] is False
    assert metrics["hardware_deployment"] is False
    assert metrics["initialization"] == "random; no pretrained weights"
    assert metrics["checkpoint_sha256"] == hashlib.sha256((tmp_path / "model.pt").read_bytes()).hexdigest()
    assert metrics["all_predictive_pass"] is report["all_predictive_pass"]
    restored, saved_normalizations, contents = load_checkpoint(tmp_path / "model.pt")
    _assert_safe_checkpoint_value(contents)
    saved_training = {}
    for path in (tmp_path / "data").glob("*_train.npz"):
        recording = load_trajectories(path)
        saved_training[recording.metadata["env_id"]] = recording
    assert set(saved_training) == set(dimensions)
    for index, env_id in enumerate(dimensions):
        train_part, _ = split_episodes(datasets[env_id], config.seed + index)
        assert np.allclose(normalizations[env_id].mean, Normalization.fit(train_part).mean)
        assert np.array_equal(saved_normalizations[env_id].mean, normalizations[env_id].mean)
        provenance = metrics["provenance"][env_id]
        assert not set(provenance["train_episodes"]) & set(provenance["validation_episodes"])
        for label in ("training", "evaluation"):
            assert provenance[label]["source_kind"] == "simulation"
            assert provenance[label]["identified"] is False
        source = saved_training[env_id]
        assert np.array_equal(source.observations, datasets[env_id].observations)
        obs = torch.as_tensor(normalizations[env_id].apply(datasets[env_id].observations[:2, 0]))
        acts = torch.as_tensor(datasets[env_id].actions[:2, :2])
        assert torch.equal(model.rollout(env_id, obs, acts), restored.rollout(env_id, obs, acts))


@pytest.mark.parametrize("field, replacement", [
    ("env_id", "different_body"),
    ("body", "different_body"),
    ("q_semantics", "encoder_angles_instead_of_pwm"),
    ("control_hz", 25.0),
    ("actuator_model", "different_motor_family"),
    ("observation_keys", ["state_2", "state_1", "state_0"]),
])
def test_training_rejects_incompatible_body_semantics(tmp_path, field, replacement):
    training = synthetic_trajectories()
    evaluation = synthetic_trajectories(seed=101, engine="synthetic_fixture_eval")
    evaluation.metadata[field] = replacement
    with pytest.raises(ValueError):
        train({"fixture": training}, {"fixture": evaluation}, tmp_path,
              tiny_config(), JEPAConfig(latent_dim=3, hidden_dim=8))


def test_training_rejects_declared_hardware_rate_that_disagrees_with_timestamps(tmp_path):
    training = synthetic_trajectories(hardware=True)
    training.timestamps *= 2  # Actual 25 Hz, declared 50 Hz.
    evaluation = synthetic_trajectories(seed=101, engine="synthetic_fixture_eval", hardware=True)
    with pytest.raises(ValueError):
        train({"fixture": training}, {"fixture": evaluation}, tmp_path,
              tiny_config(), JEPAConfig(latent_dim=3, hidden_dim=8))


def test_training_rejects_one_copied_episode_in_external_evaluation(tmp_path):
    training = synthetic_trajectories()
    evaluation = synthetic_trajectories(seed=101, engine="synthetic_fixture_eval")
    evaluation.observations[0] = training.observations[3]
    evaluation.actions[0] = training.actions[3]
    with pytest.raises(ValueError, match="identical episodes"):
        train({"fixture": training}, {"fixture": evaluation}, tmp_path,
              tiny_config(), JEPAConfig(latent_dim=3, hidden_dim=8))


def test_external_environment_id_cannot_escape_dataset_artifact_directory(tmp_path):
    env_id = "../outside"
    training = synthetic_trajectories(env_id, episodes=4, steps=3)
    evaluation = synthetic_trajectories(env_id, episodes=4, steps=3, seed=101,
                                        engine="synthetic_fixture_eval")
    out = tmp_path / "run"
    train({env_id: training}, {env_id: evaluation}, out,
          tiny_config(), JEPAConfig(latent_dim=3, hidden_dim=8))
    archives = list(out.rglob("*.npz"))
    assert len(archives) == 2
    assert all(path.parent == out / "data" for path in archives)
    assert all(load_trajectories(path).metadata["env_id"] == env_id for path in archives)


def test_unsupported_checkpoint_version_is_rejected(tmp_path):
    path = tmp_path / "unsupported.pt"
    torch.save({"format_version": 999}, path)
    with pytest.raises(ValueError, match="checkpoint format"):
        load_checkpoint(path)


@pytest.mark.parametrize("kwargs", [
    {"epochs": 0}, {"horizon": True}, {"batch_size": 1}, {"eval_samples": 1},
    {"learning_rate": 0.0}, {"learning_rate": float("nan")}, {"seed": -1}, {"seed": False},
])
def test_invalid_training_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        tiny_config(**kwargs)
