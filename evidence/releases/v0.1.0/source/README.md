# Rage JEPA

A **78,784-parameter state-vector JEPA**, trained from random weights, with its
ten-link pendulum checkpoint, simulation datasets, frozen hybrid controller
and all **64 final closed-loop recordings**. Code, weights, authored data and
documentation are MIT licensed.

One cart swings ten passive serial pendulums from hanging to upright. The
frozen controller passes **32/32 fresh initializations on CPU Euler and 32/32
on independent MuJoCo** under the stated protocol.

The successful path is:

```text
physical observations → frozen JEPA target encoder → local inverse
→ recovered state → classical trajectory tracking / balance → queued cart force
```

**Classical feedback selects the forces.** The learned dynamics predictor
fails its prediction gates on this ten-pole cohort and does not plan these
actions. The observations already contain positions, angles and velocities.
This result establishes neither a JEPA advantage over classical control nor
control from images. Hardware transfer and real-time execution are unverified.

## Start with the saved evidence

```bash
git clone https://github.com/marctheshark3/rage-jepa.git
cd rage-jepa
uv sync --extra dev
CUDA_VISIBLE_DEVICES='' uv run python scripts/verify_release.py
CUDA_VISIBLE_DEVICES='' uv run pytest
```

The verifier checks every released file against the manifest, loads the
checkpoint with `weights_only=True`, checks the model dimensions and tests
the task gates and actual one-tick actuator queue in every final recording.
It audits saved evidence; it does not rerun physics.

Weights are at `artifacts/npole10/jepa/model.pt` (about 443 KB). Load them:

```python
from rage_mill.jepa_train import load_checkpoint
model, normalizations, metadata = load_checkpoint("artifacts/npole10/jepa/model.pt")
# Normalize observations using normalizations["npole10"] before encoding.
# model.encode("npole10", observations, target=True) gives 64 latent values.
```

## Retrain the model

```bash
CUDA_VISIBLE_DEVICES='' uv run python scripts/train_model.py --device cpu --out runs/retrained
# On a compatible CUDA machine, use --device cuda.
```

This uses the released 48 Euler episodes and 12 independent MuJoCo episodes,
the original episode split seed, 200 epochs, four updates per epoch, batch
size 128 and eight-step (160 ms) prediction windows. Training-only statistics
normalize the observations. New training is a new experiment: it does not
replace the qualified checkpoint or inherit its control result. The original
neural training ran on NVIDIA GB10 (DGX Spark, sm_121); physics was CPU.

## Reproduce the hybrid controller

```bash
uv sync --extra dev --extra simulation
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
uv run python scripts/parallel_extended_validation.py \
  --n 10 --checkpoint artifacts/npole10/jepa/model.pt \
  --plan artifacts/npole10/verified/npole10-rk44-18s-feasible.npz \
  --physics-substeps 12800 --mujoco-physics-substeps 20 --mujoco-native-damping \
  --tracking-r 20 --balance-r 1000 --inverse-iterations 20 --inverse-tolerance 1e-12 \
  --workers 32 --final --out runs/reproduction/pole10
```

This is expensive CPU work. Euler uses **640,000 integration steps per
simulated second**, MuJoCo 1,000; commands stay **50 Hz**, with a queued **20 ms
delay** and **±10 N** limit. These are integration rates, not a real-time speed
claim. Offline Riccati recursion uses `numpy.longdouble`; its precision varies
by platform. Released source and numeric payloads trace the qualified run,
but a different machine/library stack need not reproduce identical trajectories.
This command is a replication with published seeds, not a new held-out cohort.

Each 20-second episode must stay inside the ±2.4 m rail and keep all ten links
within 0.22 radians of upright and below 1 rad/s throughout the final five
seconds. All 251 bounding state endpoints are checked. Resets start near
hanging, and final seeds 1,000,000–1,000,031 were untouched until the original
configuration was frozen. The minimum observed final hold is 8.32 simulated
seconds (417 scored ticks / 8.34 seconds).

## Read the experiment

- [Model card](MODEL_CARD.md): architecture, provenance, results and limits.
- [Climbing toward a comparable record](docs/ROADMAP.md): twelve is qualified;
  thirteen remains in planning, with clear gates for subsequent counts.
- [Rage Industries article](https://neuralyogi.com/essays/a-small-model-and-ten-pendulums/).
- `artifacts/npole10/`: weights, datasets, 200-epoch history, reference,
  full final recordings, independent audits and retained development receipts.
- `evidence/twelve-poles/`: later-cohort summary evidence, separate from this
  ten-pole model/control release.

`release-manifest.json` distinguishes original archive hashes from public file
hashes. Public metadata uses portable aliases in place of machine paths.
Every model tensor, normalization and numeric array is preserved; checkpoint
and archive container bytes change when metadata is rewritten. Historical
hashes in receipts identify the original files. Follow the manifest to find
the corresponding public files rather than comparing historical hashes to
rewritten containers. Captured model/controller source is used; data adapters
are restricted to poles. No private laboratory history or robot assets ship.
