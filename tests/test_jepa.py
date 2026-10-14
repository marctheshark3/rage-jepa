from copy import deepcopy

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from rage_mill.jepa import JEPAConfig, MultiEnvJEPA


@pytest.fixture
def model():
    torch.manual_seed(7)
    return MultiEnvJEPA({"cartpole": (6, 1), "fixture.arm": (11, 3)}, JEPAConfig(latent_dim=8, hidden_dim=16))


def test_one_predictor_supports_different_body_dimensions(model):
    for env_id, (obs_dim, action_dim) in model.env_dims.items():
        observations = torch.randn(3, obs_dim)
        actions = torch.randn(3, 4, action_dim)
        assert model.encode(env_id, observations).shape == (3, 8)
        assert model.rollout(env_id, observations, actions).shape == (3, 4, 8)
        losses = model.loss(env_id, torch.randn(3, 5, obs_dim), actions)
        assert set(losses) == {"total", "prediction", "variance", "covariance", "latent_std"}
        assert all(loss.shape == () and torch.isfinite(loss) for loss in losses.values())


def test_targets_receive_no_gradient_and_all_online_components_do(model):
    model.train()
    observations = torch.randn(4, 5, 6, requires_grad=True)
    actions = torch.randn(4, 4, 1)
    losses = model.loss("cartpole", observations, actions)
    losses["total"].backward()
    assert model.target_encoders.training is False
    assert all(not p.requires_grad and p.grad is None for p in model.target_encoders.parameters())
    for component in (model.encoders["body_0"], model.action_projections["body_0"], model.predictor, model.env_embeddings):
        assert sum(p.grad.abs().sum().item() for p in component.parameters() if p.grad is not None) > 0
    assert observations.grad[:, 1:].abs().sum() > 0  # future online variance/covariance regularization
    assert losses["latent_std"].requires_grad is False
    assert model.encode("cartpole", observations[:, 0], target=True).requires_grad is False
    model.eval().train()
    assert all(not module.training for module in model.target_encoders.modules())


def test_ema_parameters_and_buffers_follow_online_state():
    model = MultiEnvJEPA({"body": (3, 1)}, JEPAConfig(latent_dim=4, hidden_dim=8, ema_decay=0.75))
    online = model.encoders["body_0"]
    target = model.target_encoders["body_0"]
    online.register_buffer("running_mean", torch.tensor([4.0, 8.0]))
    target.register_buffer("running_mean", torch.tensor([0.0, 4.0]))
    online.register_buffer("count", torch.tensor(9))
    target.register_buffer("count", torch.tensor(2))
    old = {name: parameter.clone() for name, parameter in target.named_parameters()}
    with torch.no_grad():
        for parameter in online.parameters():
            parameter.add_(2.0)
    model.update_target()
    for name, parameter in target.named_parameters():
        expected = old[name] * 0.75 + online.get_parameter(name) * 0.25
        assert torch.allclose(parameter, expected)
    assert torch.equal(target.running_mean, torch.tensor([1.0, 5.0]))
    assert target.count.item() == 9


def test_constant_embeddings_incurs_collapse_penalty(model):
    with torch.no_grad():
        for parameter in model.encoders["body_0"].parameters():
            parameter.zero_()
    losses = model.loss("cartpole", torch.randn(4, 4, 6), torch.zeros(4, 3, 1))
    assert losses["variance"].item() == pytest.approx(0.99, abs=1e-6)
    assert losses["covariance"].item() == pytest.approx(0.0)
    assert losses["latent_std"].item() == pytest.approx(0.0)
    assert losses["total"].item() >= losses["variance"].item()


def test_covariance_penalizes_redundant_features():
    model = MultiEnvJEPA({"body": (2, 1)}, JEPAConfig(latent_dim=2, hidden_dim=8))
    model.encoders["body_0"] = nn.Identity()
    model.target_encoders["body_0"] = nn.Identity()
    values = torch.tensor([-1.0, 1.0, -1.0, 1.0]).reshape(2, 2, 1)
    observations = values.expand(-1, -1, 2).contiguous()
    losses = model.loss("body", observations, torch.zeros(2, 1, 1))
    assert losses["variance"].item() == pytest.approx(0.0)
    assert losses["covariance"].item() > 1.0


def test_actions_change_autoregressive_predictions(model):
    observations = torch.randn(3, 6)
    zeros = torch.zeros(3, 3, 1)
    ones = torch.ones(3, 3, 1)
    assert not torch.allclose(model.rollout("cartpole", observations, zeros), model.rollout("cartpole", observations, ones))
    actions = zeros.clone()
    actions[:, -1] = 1
    assert torch.allclose(model.rollout("cartpole", observations, zeros)[:, :2], model.rollout("cartpole", observations, actions)[:, :2])


def test_planner_is_bounded_reproducible_and_keeps_global_rng(model):
    observations = torch.randn(2, 11)
    goals = torch.randn(2, 11)
    state = torch.random.get_rng_state().clone()
    kwargs = dict(horizon=3, candidates=16, iterations=2, seed=11)
    first = model.plan("fixture.arm", observations, goals, **kwargs)
    assert first.shape == (2, 3)
    assert first.min() >= -1 and first.max() <= 1
    assert first.requires_grad is False
    assert torch.equal(first, model.plan("fixture.arm", observations, goals, **kwargs))
    assert torch.equal(state, torch.random.get_rng_state())


def test_planner_selects_actions_that_reach_latent_goal():
    class ActionDelta(nn.Module):
        def forward(self, concatenated):
            return concatenated[:, 1:2]

    model = MultiEnvJEPA({"point": (1, 1)}, JEPAConfig(latent_dim=1, hidden_dim=8))
    model.encoders["body_0"] = nn.Identity()
    model.target_encoders["body_0"] = nn.Identity()
    model.action_projections["body_0"] = nn.Identity()
    model.predictor = ActionDelta()
    actions = model.plan("point", torch.zeros(2, 1), torch.tensor([[0.8], [-0.6]]), horizon=1, candidates=64, iterations=4, seed=9)
    assert torch.allclose(actions, torch.tensor([[0.8], [-0.6]]), atol=0.05)


@pytest.mark.parametrize("kwargs", [
    {"latent_dim": 0}, {"hidden_dim": 1.5}, {"latent_dim": True},
    {"ema_decay": -0.1}, {"ema_decay": 1.0}, {"ema_decay": float("nan")},
    {"variance_weight": -1}, {"covariance_weight": float("inf")},
])
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        JEPAConfig(**kwargs)


@pytest.mark.parametrize("dimensions", [{}, {"body": (0, 1)}, {"": (1, 1)}, {"body": (1,)}, {"body": (True, 1)}])
def test_invalid_body_dimensions_are_rejected(dimensions):
    with pytest.raises(ValueError):
        MultiEnvJEPA(dimensions)


def test_shape_dtype_and_nonfinite_validation(model):
    with pytest.raises(ValueError, match="unknown environment"):
        model.encode("unknown", torch.zeros(2, 6))
    with pytest.raises(ValueError, match="dimension 6"):
        model.encode("cartpole", torch.zeros(2, 5))
    with pytest.raises(TypeError, match="floating"):
        model.encode("cartpole", torch.zeros(2, 6, dtype=torch.int64))
    with pytest.raises(ValueError, match="dtype"):
        model.encode("cartpole", torch.zeros(2, 6, dtype=torch.float64))
    with pytest.raises(ValueError, match="nonfinite"):
        model.encode("cartpole", torch.full((2, 6), float("nan")))
    with pytest.raises(ValueError, match="actions must have shape"):
        model.rollout("cartpole", torch.zeros(2, 6), torch.zeros(3, 4, 1))
    with pytest.raises(ValueError, match="horizon \\+ 1"):
        model.loss("cartpole", torch.zeros(2, 4, 6), torch.zeros(2, 4, 1))
    with pytest.raises(ValueError, match="nonempty"):
        model.rollout("cartpole", torch.zeros(2, 6), torch.zeros(2, 0, 1))


def test_model_can_follow_explicit_float64_dtype(model):
    model.double()
    output = model.rollout("cartpole", torch.zeros(2, 6, dtype=torch.float64), torch.zeros(2, 2, 1, dtype=torch.float64))
    assert output.dtype == torch.float64


@pytest.mark.parametrize("kwargs", [
    {"horizon": 0}, {"candidates": 0}, {"iterations": False},
    {"elite_fraction": 0}, {"elite_fraction": 1.1}, {"action_penalty": -1}, {"seed": -1},
])
def test_invalid_planner_options_are_rejected(model, kwargs):
    with pytest.raises(ValueError):
        model.plan("cartpole", torch.zeros(2, 6), torch.zeros(2, 6), **kwargs)


def test_random_initialization_and_checkpoint_roundtrip(model):
    another = MultiEnvJEPA(model.env_dims, model.config)
    assert any(not torch.equal(a, b) for a, b in zip(model.predictor.parameters(), another.predictor.parameters()))
    another.load_state_dict(deepcopy(model.state_dict()))
    observations, actions = torch.randn(2, 6), torch.randn(2, 3, 1)
    assert torch.equal(model.rollout("cartpole", observations, actions), another.rollout("cartpole", observations, actions))
    assert all(not parameter.requires_grad for parameter in another.target_encoders.parameters())
