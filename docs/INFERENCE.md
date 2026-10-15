# Full world-model inference without the training stack

Yes: the **complete trained world model** runs with NumPy as its only external
dependency. This export includes the online encoder, frozen target encoder,
learned action projection, environment embedding and learned residual dynamics
predictor. It is a complete forward-inference implementation, distinct from
the encoder/inverse controller introduced in v0.2.

The original checkpoint remains unchanged. All **21 tensors and 107,904
parameters** are exported exactly: 78,784 trained parameters plus 29,120 frozen
target parameters. No retraining, pruning or quantization occurs. Training
losses, EMA updates, optimizers and gradients are unnecessary for inference.
The model outputs future latent vectors, rather than decoded physical states.
No new learned controller or CEM planner is supplied by this NumPy API.

## Install and execute the full predictor

```bash
uv venv /tmp/jepa-inference
uv pip install --python /tmp/jepa-inference/bin/python \
  'https://github.com/marctheshark3/rage-jepa/releases/download/v0.3.0/rage_jepa-0.3.0-py3-none-any.whl'
/tmp/jepa-inference/bin/rage-jepa-world-model
```

Expected summary: `full_world_model: true`,
`learned_dynamics_predictor_executed: true`, `training: false`, 107,904
parameters, and prediction shape `[1,8,64]`. The bundled example is eight
recorded commands and the initial observation from the first released
MuJoCo evaluation episode. Passing `--out predictions.npz` saves all latents.

For your own observations and commands:

```python
from rage_mill.world_model import NumpyWorldModel
wm = NumpyWorldModel()  # float32 by default; dtype="float64" is also supported
prediction = wm.rollout("npole10", observations, actions)
# observations: raw [batch,33]; actions: [batch,horizon,1]
# prediction: [batch,horizon,64]
target = wm.encode("npole10", observations, target=True)
```

Saved training-only normalization is applied automatically. Use
`normalized=True` to supply already normalized observations, matching the
original Torch model's interface. Commands are used as supplied, without
clipping. Inputs must be finite and nonempty with matching batch dimensions.
Inference arrays use the requested dtype and the saved weights remain read-only.

## Independent numerical proof

[`scripts/make_world_model_reference.py`](../scripts/make_world_model_reference.py)
uses the original, unchanged Torch implementation and original public
checkpoint to freeze reference outputs. Inputs come from held-out Euler
episodes and independently recorded MuJoCo episodes. The definition and
tolerances are saved before evaluating the candidate. Reference coverage:

- Both online and EMA target encoders, with every trained tensor checked exactly.
- Complete action-conditioned autoregressive predictions with the learned dynamics.
- Horizons of 1, 8 and 32 steps; batch sizes 1, 8 and 128.
- Recorded, zero and shuffled action sequences.
- Original float32 inference and optional float64 inference.

[`scripts/prove_world_model_inference.py`](../scripts/prove_world_model_inference.py)
runs in a fresh, noneditable wheel installation containing **only `numpy`
and `rage-jepa`**. Torch, MuJoCo, CasADi and Gymnasium must be absent. An
import hook rejects training/simulation modules and the old encoder-only
controller, preventing a fallback to those implementations. Every model
weight is compared with its original Torch tensor and checked unchanged
after inference.

All **20 cases** pass, comparing **1,053,440 output values**:

| Runtime dtype | Maximum encoder error | Maximum full-rollout error | Declared atol / rtol |
| --- | --- | --- | --- |
| float32 | 1.44e−6 | 2.39e−6 | 1e−5 / 1e−5 |
| float64 | 1.78e−15 | 4.67e−15 | 1e−11 / 1e−11 |

These are numerical equivalents, rather than bitwise-identical arithmetic
across BLAS libraries. All saved parameter values are identical. The complete
[proof receipt](../evidence/inference-v0.3/isolated-proof.json),
[frozen definition](../evidence/inference-v0.3/definition.json) and
[Torch reference](../evidence/inference-v0.3/reference.json) are published.

The downloadable `rage-jepa-inference-proof-v0.3.0.zip` in the
[release](https://github.com/marctheshark3/rage-jepa/releases/tag/v0.3.0)
contains the scripts and reference data, without requiring the full repository's
recording archives. From a source checkout or extracted proof bundle:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/tmp/jepa-inference/bin/python -I scripts/prove_world_model_inference.py \
  --reference evidence/inference-v0.3

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/tmp/jepa-inference/bin/python -I scripts/measure_world_model_inference.py
```

`-I` isolates Python from user-site packages and `PYTHONPATH`. The proof also
checks that the model comes from the installed wheel, rather than an editable
source checkout. GitHub CI repeats the isolated parity proof on x86; the
originating measurements use Spark's ARM CPU. Neither path needs a GPU.

## Measured footprint and CPU cost

| Item | Observed size / latency |
| --- | --- |
| Complete model NPZ export | 406,040 bytes |
| Model parameters loaded in float32 | 431,616 bytes |
| NumPy + package installed file payload | About 58 MB, excluding shared Python executable symlinks |
| Resident process RAM after model load | About 39 MB, including Python and NumPy |
| Peak resident RAM during the measured inference process | About 41 MB |
| Batch 1, one-step forecast | About 0.037 ms median |
| Batch 1, eight-step forecast | About 0.13 ms median |
| Batch 1, 32-step forecast | About 0.45 ms median |
| Batch 32, eight-step forecast | About 0.68 ms median for the whole batch |

The [measurement receipt](../evidence/inference-v0.3/measurement.json) records
exact values, Python/NumPy versions, model hash, shapes, warmups, repetitions,
dtype and single-thread settings. These are warm-cache shared-host CPU
measurements, including normalization and online encoding. They do not
include physics, control selection, CEM search or a hardware deadline claim.

RAM uses Linux `/proc/self/status` VmRSS/VmHWM for the current interpreter's
resident-memory history, including Python and NumPy. The receipt also records
`getrusage` and the pre-import baseline; the measured interpreter was launched
from a small standard-library-only parent. Disk payload counts regular files
once and excludes shared Python executable symlinks;
following those three links instead gives about 127 MB apparent size. The
earlier v0.2 installation figures used the latter counting convention.

The wheel also includes the existing classical reference/gains and target-only
encoder assets. Those need not be loaded for full world-model inference.
MuJoCo/CasADi/Gymnasium are optional extras for simulation. Torch is an optional
extra for training and the historical implementation. The original 5.7 GB
local training/simulation installation is not required for this forward path.

## What this proves

This proves complete trained-model inference, numerical fidelity and a small
deployment footprint. The original predictor still fails its predictive and
physical forecasting gates. Matching that model's outputs does not make its
forecasts more accurate, establish a learned control advantage or increase
the qualified pendulum count. The passing control suite continues to use
classical feedback as described in [the baseline comparison](BASELINES.md).

The project remains a [public extraction from private Rage Mill](PROVENANCE.md).
Version 0.3 adds this inference path without replacing the original weights,
v0.1 results or v0.2 comparison. Historical manifests remain intact, with
explicit archived copies of evolving metadata/tool files.
