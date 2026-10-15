# Rage JEPA

A ten-link cart-pendulum environment, a ready classical baseline, and a small
**78,784-parameter state-vector JEPA** with its trained checkpoint, datasets
and complete evaluation recordings. Code, weights, authored simulation data
and documentation are MIT licensed.

**Full world-model inference now runs with NumPy only:** all 21 trained
tensors, both encoders, the action adapter, environment embedding and learned
dynamics predictor are included. An isolated install matches over a million
Torch reference outputs. See the [inference proof and measurements](docs/INFERENCE.md).

**This project was extracted from our private Rage Mill research repository
on October 14, 2026.** Its first public commit is a curated release of an
existing experiment, rather than the start of the research. The trained model
and results are included. Public Git history does not contain the earlier
private development history. See [provenance](docs/PROVENANCE.md).

Classical trajectory tracking and balance feedback select the forces. The
successful JEPA path encodes an already observed physical state, reconstructs
it through a local inverse, and supplies that state to classical feedback.
The learned dynamics predictor fails its forecast gates and does not plan
these actions. The environment and classical baseline work without JEPA or
Torch. See [the matched comparison and performance measurements](docs/BASELINES.md).

**New matched result:** all three controllers pass **32/32 Euler and 32/32
MuJoCo** on fresh shared starts (192 trials). The optimized JEPA controller is
about **4.5× faster** than the original in serial CPU replay measurements;
classical feedback alone is about **18× faster** and passes the same task.

## Run the environment without Torch

For just the runtime, install the small wheel from
[v0.3.0](https://github.com/marctheshark3/rage-jepa/releases/tag/v0.3.0);
it excludes the full recording archives. A source checkout also includes the
scientific evidence:

```bash
git clone https://github.com/marctheshark3/rage-jepa.git
cd rage-jepa
uv sync                         # NumPy only
uv run python - <<'PY'
from rage_mill.env import CartPendulumEnv
env = CartPendulumEnv(n=2, substeps=20)
state = env.reset_seed(123)
state = env.step(0.2)            # normalized cart force; queued for next tick
print(state, env.observation())
PY
```

For ten poles, the qualified rates are 12,800 Euler substeps per control tick
or 20 native MuJoCo substeps. Select `backend="c"` explicitly to accelerate
the Euler loop; this requires a C compiler on Linux, preserves the equations
and integration rate, and never silently falls back. NumPy remains the
reference backend. These are different floating-point implementations;
bitwise-identical physics is not claimed.

```python
env = CartPendulumEnv(n=10, engine="euler", backend="c")
env.reset_seed(123)
env.step(0.2)
```

Physical state is `[x, absolute_angles[10], x_velocity, absolute_omegas[10]]`
(22 values). `observation()` returns cart position/velocity, each link's
sin/cos and angular speed divided by eight, then last-applied and pending
commands (34 values). The JEPA encoder uses the first 33 values; the feedback
controller also receives the pending force for delay compensation.

## Run the complete world model for inference

```bash
python -m pip install "https://github.com/marctheshark3/rage-jepa/releases/download/v0.3.0/rage_jepa-0.3.0-py3-none-any.whl"
rage-jepa-world-model
```

This executes the online encoder, learned action projection, environment
embedding and autoregressive residual predictor on a packaged recorded
example, returning eight future 64-value latents. It loads all 107,904
parameters, including the target encoder, without Torch, CUDA or a simulator.
There is no retraining or quantization. The full-model NPZ export is 406 KB;
the complete runtime wheel is about 0.9 MB.

```python
from rage_mill.world_model import NumpyWorldModel
model = NumpyWorldModel()        # Original float32; float64 also supported.
# observations: [batch,33] raw state-vector observations
# actions: [batch,horizon,1] normalized cart-force commands
predicted_latents = model.rollout("npole10", observations, actions)
```

The [proof](docs/INFERENCE.md) compares both encoders and full learned
predictions against frozen Torch outputs, including 1/8/32-step forecasts,
recorded/shuffled/zero actions, and float32/float64. Numerical equivalence
proves inference execution; the original predictor's failed accuracy gates
remain unchanged. The qualified control result below uses classical feedback.

## Run the ready classical or JEPA baseline

```bash
uv sync --extra simulation --extra gym
uv run --extra simulation --extra gym python - <<'PY'
from rage_mill.env import CartPendulumEnv
from rage_mill.baseline import ClassicalPolicy
env = CartPendulumEnv(engine="mujoco")
env.reset_seed(123)
policy = ClassicalPolicy()
for _ in range(1000):
    command = policy.act(env.state, env.last_applied, env.pending)
    env.step(command)
print(env.state)
PY
```

Replace `ClassicalPolicy` with
`from rage_mill.numpy_policy import NumpyJEPAPolicy` and use
`policy = NumpyJEPAPolicy()` for the optimized frozen encoder/inverse path.
Neither needs Torch. Both use the same exported classical gains and the
original CasADi four-substep RK4 predictor for the known pending force.
There is no online training, trajectory optimization or Riccati solve.

For Gymnasium:

```python
from rage_mill.gym_env import CartPendulumGym
env = CartPendulumGym(engine="mujoco")
observation, info = env.reset(seed=123)
observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

Its dense reward is documented in the adapter. It is a convenience for new
agents, separate from the binary task-success gates; no RL policy is included.

## Check the evidence and reproduce the comparison

```bash
uv sync --extra dev
uv run --extra dev python scripts/verify_release.py
uv run --extra dev python scripts/audit_comparison.py evidence/runtime-v0.2/cohort
uv run --extra dev pytest

# Full original model + simulator checks (Torch is optional):
uv sync --all-extras
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
uv run --all-extras python scripts/verify_release.py --with-model
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
uv run --all-extras pytest

# New matched closed-loop trials; choose a new output directory.
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
uv run --all-extras rage-jepa-compare --count 32 --seed-start 3200000 \
  --workers 4 --out runs/my-comparison
uv run python scripts/audit_comparison.py runs/my-comparison
```

All three conditions use the same seeds, hanging resets, reference, feedback
gains, force limits, real one-tick queue and 50 Hz controller. Euler retains
640,000 integration steps per simulated second; independent native MuJoCo
uses 1,000. All physics and inference run on CPU. Setup, controller, physics
and episode timings are recorded separately. The saved v0.2 cohort uses
previously unused seeds 3,100,000–3,100,031. Reusing those seeds is replication,
rather than a new held-out evaluation.

Each 20-second episode must keep the cart strictly inside ±2.4 m and all ten
links within 0.22 radians of upright and below 1 rad/s throughout the final
five seconds, checking all 251 bounding state endpoints. Commands are limited
to ±10 N and have a queued 20 ms delay initialized to zero. Complete replays,
failures and independent audits are retained. The cohort gate remains 80%.
Simulation timing does not establish a hardware real-time guarantee.

## Load or retrain the original model

The full trained checkpoint is
[`artifacts/npole10/jepa/model.pt`](artifacts/npole10/jepa/model.pt) (442,606 bytes).
The inference-only target encoder is packaged separately (109,785 bytes).

```bash
uv sync --extra jepa
uv run --extra jepa python - <<'PY'
from rage_mill.jepa_train import load_checkpoint
model, normalizations, metadata = load_checkpoint("artifacts/npole10/jepa/model.pt")
# Normalize using normalizations["npole10"] before encoding.
# model.encode("npole10", observations, target=True) returns 64 latent values.
PY
CUDA_VISIBLE_DEVICES='' uv run --extra jepa python scripts/train_model.py \
  --device cpu --out runs/retrained
```

The released data contains 48 Euler and 12 independent MuJoCo episodes.
Training uses the original episode split, 200 epochs, four updates per epoch,
batch size 128 and eight-step windows. The original neural training ran on
DGX Spark's NVIDIA GB10; physics and control ran on CPU. Spark did not make
the network larger. New training produces a new experiment and does not
inherit the original checkpoint's control result.

## Research and release history

- [Baseline comparison, speed and next experiments](docs/BASELINES.md).
- [Private origin, exports and immutable history](docs/PROVENANCE.md).
- [Original model card](MODEL_CARD.md): architecture, original results and limits.
- [Climbing toward a comparable record](docs/ROADMAP.md): twelve is qualified;
  thirteen remains in planning. Global record priority is unverified.
- [NeuralYogi article](https://neuralyogi.com/essays/a-small-model-and-ten-pendulums/).

The original v0.1 release passed 32/32 Euler and 32/32 MuJoCo trials and remains
available at [v0.1.0](https://github.com/marctheshark3/rage-jepa/tree/v0.1.0).
Its weights, datasets, reference, source and all 64 recordings retain their
original public bytes. Five evolving release/documentation files are archived
with their original hashes. `release-manifest.json` still describes that
historical snapshot; `release-v0.2-manifest.json` describes the baseline release.
Its six evolving metadata/tool files are archived under `evidence/releases/v0.2.0/`.
`release-v0.3-manifest.json` checks the current full-inference release. All
original scientific source, weights and recordings keep their public bytes.
No robot or private laboratory assets ship. Prediction advantage, image-based
control, hardware transfer and world-record priority remain unestablished.
