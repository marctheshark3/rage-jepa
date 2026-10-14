# Ten-pole state-vector JEPA

**Release:** 0.1.0 · **Author:** Marc / Rage Industries · **License:** MIT for
code, model weights, authored simulation data and documentation.

## What this model does

An independent PyTorch joint embedding predictive architecture trained from
random weights. It learns embeddings from state-vector/action windows. The
ten-pole control experiment uses its frozen target encoder as an invertible
local observation representation; classical feedback controls the plant.

This is not an I-JEPA or V-JEPA pretrained checkpoint. Those papers informed
the design; no upstream weights are used. The implementation additionally
uses variance/covariance penalties inspired by VICReg.

## Architecture and training

| Item | Recorded ten-pole configuration |
| --- | --- |
| Observation | 33 values: cart x/velocity, per-link sin/cos/omega÷8, last applied force |
| Action | One normalized cart command in [−1, 1] |
| Physical state | 22 values: 11 positions and 11 velocities |
| Latent | 64 values |
| Hidden layers | Two 128-unit SiLU layers in each MLP |
| Trainable parameters | 78,784; frozen EMA target parameters excluded |
| Prediction | Residual, autoregressive, action conditioned |
| Target encoder | EMA decay 0.99, stop-gradient targets |
| Regularization | Variance weight 1.0; covariance weight 0.01 |
| Data | 48 Euler and 12 independent MuJoCo episodes, 128 actions per episode |
| Training split | 36 whole Euler episodes; 12 held-out Euler episodes |
| Training | 200 epochs × 4 updates; batch 128; horizon 8; AdamW 0.001 |
| Seed | 2026101310 |
| Device | NVIDIA GB10, sm_121, aarch64; PyTorch 2.14.1+cu130 |
| Measured neural training | 7.0683 seconds; peak PyTorch allocation 72.8555 MiB |
| Training physics | CPU; actual 1,000 Hz collection; 50 Hz observations |

The single-body adapter uses the same multi-environment architecture, with
one environment embedding/action projection. A 64-value latent expands a
33-value observation; this experiment is not evidence of dimensional
compression. Data are synthetic simulation trajectories with no personal
records, robot teaching data or external training corpus.

Training and model selection do not use the independent evaluation episodes.
Training collection is undelayed; the separate closed-loop protocol adds the
actual one-tick command queue. Keep prediction and control tests distinct.

## Results that must stay separate

The ten-pole predictor fails both its physical-forecast persistence comparison
and its latent action-sensitivity gates on held-out Euler and independent
MuJoCo. Its public metrics retain the failure flags. Useful local state
reconstruction does not establish useful forecasting or action planning.

The target encoder's tested Jacobians have full rank for all 22 physical state
coordinates. Along 31 sampled verified-reference states, reconstruction error
is below 5e−14. This is a local path check, not global invertibility. Runtime
inversion uses the nominal reference as its initial guess, at most 20
iterations and 1e−12 latent residual tolerance. Inversion failure fails the
episode; there is no raw-state fallback.

Classical time-varying LQR tracks the verified swing reference and hands off
to balance feedback. Tracking R=20, balance R=1000; known pending force enters
the delay compensation. The independently verified reference retains its
optimizer timeout status; feasibility was checked separately and objective
optimality is unverified.

With the fixed checkpoint/reference/controller, all 32 fresh hanging starts
pass on each physics engine. Full arrays and queue/rail/final-hold audits are
included. These 64 starts are evidence under this exact setup, not a guarantee
for arbitrary disturbances, body parameters or other starts.

## Why a small model can serve this task

The encoder receives a compact, explicit state vector. It need not learn
vision, infer hidden mass/friction, or discover a swing-up policy. The dynamics
and classical controller carry much of the problem structure. Local rank and
reconstruction checks support preserving state near the tested reference.
They do not establish that JEPA training is necessary: a raw-state controller
and a random encoder with identical inversion are important ablations that
remain unvalidated as matched final cohorts.

## Provenance and limitations

Original qualified checkpoint SHA256:
`ae387e7eebbf28097bb62cc7c8e664a27a0c1c23a4255ef56cea033e96bc5c6a`.
Public provenance aliases change the checkpoint container hash while
preserving all model tensors and normalization values. The manifest records
both identities and the exact source commit and captured-source digests.

Only simulation is validated. No robot was actuated or identified. Logical
50 Hz ticks do not establish a measured hardware log. RTX 3090 validation
remains open; the recorded neural training machine was Spark GB10. A global
record, cross-body transfer, predictor superiority and real-time operation
are unverified.

## References

- [I-JEPA](https://arxiv.org/abs/2301.08243), Assran et al. (2023): predict target
  representations rather than reconstructing the raw signal.
- [V-JEPA 2](https://arxiv.org/abs/2506.09985), Bardes et al. (2025): video world
  models with action-conditioned prediction/planning.
- [VICReg](https://arxiv.org/abs/2105.04906), Bardes, Ponce and LeCun (2022):
  explicit variance and covariance regularization against collapse.
