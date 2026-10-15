"""Full inference checked against the independent original Torch implementation."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from rage_mill.world_model import ASSETS, NumpyWorldModel


def test_complete_parameter_inventory_and_no_training_imports():
    model = NumpyWorldModel()
    assert model.parameter_count == 107904 and len(model.weights) == 21
    assert model.parameter_bytes == 431616
    assert sum(v.size for k, v in model.weights.items() if k.startswith("target_encoders.")) == 29120
    assert sum(v.size for k, v in model.weights.items() if k.startswith("predictor.")) == 49472
    script = """
import sys, numpy as np
from rage_mill.world_model import NumpyWorldModel
model = NumpyWorldModel()
y = model.rollout('npole10', np.zeros((1,33)), np.zeros((1,8,1)))
assert y.shape == (1,8,64)
assert not any(x in sys.modules for x in ('torch','mujoco','casadi','gymnasium','rage_mill.jepa_train'))
"""
    subprocess.run([sys.executable, "-c", script], check=True)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("batch,horizon", [(1, 1), (4, 8), (8, 32)])
def test_full_forward_matches_original_torch(dtype, batch, horizon):
    torch = pytest.importorskip("torch")
    from rage_mill.jepa_train import load_checkpoint
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    original, norms, _ = load_checkpoint(root / "artifacts/npole10/jepa/model.pt", "cpu")
    original.to(dtype=torch.float32 if dtype == np.float32 else torch.float64)
    model = NumpyWorldModel(dtype=dtype)
    assert set(model.weights) == set(original.state_dict())
    for key, tensor in original.state_dict().items():
        np.testing.assert_array_equal(model.weights[key], tensor.numpy())
    rng = np.random.default_rng(4840000 + horizon)
    observations = rng.normal(size=(batch, 33)).astype(dtype)
    actions = rng.uniform(-1., 1., size=(batch, horizon, 1)).astype(dtype)
    atol = rtol = 1e-5 if dtype == np.float32 else 1e-11
    obs, act = torch.from_numpy(observations), torch.from_numpy(actions)
    with torch.inference_mode():
        online = original.encode("npole10", obs).numpy()
        target = original.encode("npole10", obs, target=True).numpy()
        projected = original.action_projections[original._key("npole10")](act).numpy()
        forecast = original.rollout("npole10", obs, act).numpy()
    np.testing.assert_allclose(model.encode("npole10", observations, normalized=True), online, atol=atol, rtol=rtol)
    np.testing.assert_allclose(model.encode("npole10", observations, target=True, normalized=True), target, atol=atol, rtol=rtol)
    np.testing.assert_allclose(model.project_actions("npole10", actions), projected, atol=atol, rtol=rtol)
    np.testing.assert_allclose(model.rollout("npole10", observations, actions, normalized=True), forecast, atol=atol, rtol=rtol)
    np.testing.assert_allclose(model.rollout_latent("npole10", online, actions), forecast, atol=atol, rtol=rtol)
    raw = observations.astype(dtype)
    normalized = (raw - norms["npole10"].mean.astype(dtype)) / norms["npole10"].std.astype(dtype)
    np.testing.assert_array_equal(model.normalize("npole10", raw), normalized)


def test_actions_and_trained_predictor_affect_forecasts_without_changing_weights(tmp_path):
    model = NumpyWorldModel()
    with np.load(ASSETS / "npole10-world-model-example.npz", allow_pickle=False) as data:
        obs, actions = data["observations"], data["actions"]
    before = {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in model.weights.items()}
    normal, zero = model.rollout("npole10", obs, actions), model.rollout("npole10", obs, np.zeros_like(actions))
    assert np.max(np.abs(normal - zero)) > 1e-6
    assert before == {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in model.weights.items()}
    with pytest.raises(ValueError):
        model.weights["predictor.4.bias"][0] = 1.
    with np.load(ASSETS / "npole10-world-model.npz", allow_pickle=False) as data:
        arrays = dict(data)
    arrays["predictor.4.bias"] = arrays["predictor.4.bias"] + np.float32(.1)
    changed = tmp_path / "changed.npz"
    np.savez_compressed(changed, **arrays)
    other = NumpyWorldModel(changed)
    assert np.max(np.abs(other.rollout("npole10", obs, actions) - normal)) > .01


@pytest.mark.parametrize("observations,actions", [
    (np.zeros((1, 32)), np.zeros((1, 8, 1))),
    (np.full((1, 33), np.nan), np.zeros((1, 8, 1))),
    (np.zeros((1, 33)), np.zeros((1, 0, 1))),
    (np.zeros((1, 33)), np.zeros((2, 8, 1))),
    (np.zeros((1, 33)), np.full((1, 8, 1), np.inf)),
])
def test_invalid_forward_inputs_are_rejected(observations, actions):
    with pytest.raises(ValueError):
        NumpyWorldModel().rollout("npole10", observations, actions)


@pytest.mark.parametrize("corruption", ["missing_predictor", "nonfinite", "bad_normalization", "bad_format"])
def test_corrupt_or_partial_exports_are_rejected(corruption, tmp_path):
    with np.load(ASSETS / "npole10-world-model.npz", allow_pickle=False) as data:
        arrays = dict(data)
    if corruption == "missing_predictor":
        del arrays["predictor.4.weight"]
    elif corruption == "nonfinite":
        arrays["predictor.4.weight"][0, 0] = np.nan
    elif corruption == "bad_normalization":
        arrays["normalization.body_0.std"][0] = 0.
    else:
        metadata = json.loads(str(arrays["metadata"]))
        metadata["format_version"] = 0
        arrays["metadata"] = json.dumps(metadata)
    path = tmp_path / "bad.npz"
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError):
        NumpyWorldModel(path)


def test_dtype_and_environment_contract():
    with pytest.raises(ValueError, match="dtype"):
        NumpyWorldModel(dtype=np.float16)
    with pytest.raises(ValueError, match="environment"):
        NumpyWorldModel().encode("unknown", np.zeros((1, 33)))


def test_cli_executes_complete_saved_predictor(tmp_path):
    path = tmp_path / "predictions.npz"
    result = subprocess.run([sys.executable, "-m", "rage_mill.world_model", "--out", str(path)],
                            check=True, capture_output=True, text=True)
    report = json.loads(result.stdout)
    assert report["learned_dynamics_predictor_executed"] and not report["training"]
    with np.load(path, allow_pickle=False) as data:
        assert data["predicted_latents"].shape == (1, 8, 64)
