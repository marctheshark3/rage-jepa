# A small model, a faster loop, and a classical baseline

The default recommendation for this fully observed ten-pole task is the
**classical baseline**. It completes the same task while avoiding the learned
encoder and its local inverse. The optimized JEPA path remains useful for
reproducing and studying the representation experiment. This comparison does
not establish a benefit from a learned world model.

This release follows [Joseph Suarez's feedback](https://x.com/jsuarez/status/2110823673087025232)
about trying the environment without the world model and his experience of
world models being slow and bloated. That is a useful test here. The original
training stack was unnecessarily required for environment users, and the
inverse did redundant work after it had converged. Spark training did not
cause either design choice.

## Matched closed-loop result

| Controller | CPU Euler | Independent native MuJoCo | Torch at runtime |
| --- | --- | --- | --- |
| Direct-state classical R20/R1000 | 32/32 | 32/32 | No |
| Original frozen JEPA encoder/inverse + same feedback | 32/32 | 32/32 | Yes |
| NumPy frozen JEPA encoder/early-exit inverse + same feedback | 32/32 | 32/32 | No |

All **192 complete recordings** are in
[`evidence/runtime-v0.2/cohort/`](../evidence/runtime-v0.2/cohort/).
The [definition](../evidence/runtime-v0.2/cohort/definition.json) was written
before running seeds 3,100,000–3,100,031. No controller, reference, tolerance
or integration rate was retuned on those results. Every group passes the
unchanged ≥80% gate; no failures occurred in this cohort. An
[independent audit](../evidence/runtime-v0.2/cohort/independent-audit.json)
checks every recording, reset, force/queue bound and all 251 final-window
endpoints. The 32 starts are a bounded evaluation of this body and reset
distribution, rather than general robustness evidence.

The same 18-second reference, exported float64 gains, original CasADi RK4
delay predictor, hanging initial states, ±10 N command limit, queued 20 ms
delay, 50 Hz command clock, ±2.4 m rail and continuous final-five-second hold
apply to all three conditions. Euler uses 12,800 velocity-first substeps per
tick through an explicit C CPU backend. MuJoCo uses 20 native substeps with
the original absolute-coordinate damping law. The C backend is a new numeric
implementation; the original NumPy-qualified v0.1 result remains separate.

The original JEPA estimator source is unchanged. For the matched comparison,
its classical feedback uses the same exported gains as the other conditions,
avoiding a platform-dependent offline Riccati solve. On the originating
stack, that feedback reproduced **all 64,000 original saved commands exactly**
when given their saved inferred states and pending forces.

## What is faster

Serial replay profiling uses identical states, tick indices, last-applied
commands and pending forces from the original MuJoCo episode, three rotated
repetitions of 1,000 ticks, CPU float64, one Torch/BLAS thread, and warmed
imports/caches. These timings exclude physics and setup:

| Controller | Median per tick | p95 per tick | Speed relative to original |
| --- | --- | --- | --- |
| Original JEPA + classical feedback | 2.094 ms | 2.203 ms | 1× |
| NumPy JEPA + early-exit inverse + same feedback | 0.464 ms | 0.544 ms | 4.5× |
| Direct-state classical feedback | 0.115 ms | 0.125 ms | 18.2× |

The [serial benchmark receipt](../evidence/runtime-v0.2/serial-benchmark.json)
records hardware/library versions, input/source hashes and shared-host load.
The largest replayed command difference between original and optimized JEPA
was 3.93e−8 in normalized force. Closed-loop behavior was tested separately
in the 192-trial cohort.

The physics cost is a different measurement: 36 full ticks per backend on
the same 12 sampled physical states, with resets outside the timing window:

| Physics backend | Substeps per 20 ms tick | Median wall time per tick |
| --- | --- | --- |
| Original NumPy Euler | 12,800 | 344.760 ms |
| Explicit C Euler, same equations/update/rate | 12,800 | 9.802 ms |
| Independent native MuJoCo | 20 | 0.398 ms |

The C implementation is about **35× faster per Euler tick** on this host.
It solves the same point-mass mass matrix and uses the same velocity-first
Euler update, with fast math and FMA contraction disabled. It uses a different
linear solve and operation order, so bitwise-equivalent physics is not claimed.
Paired full-rate ticks in the serial benchmark differed by at most 2.67e−15.
MuJoCo uses a different integrator and rate; its lower cost is not an
equal-algorithm speedup claim.

The [runtime-equivalence audit](../evidence/runtime-v0.2/runtime-equivalence.json)
also checks 128 random accelerations, 32 full-rate ticks sampled from the
original recordings, and 192 encoder/inverse states. The physics acceptance
tolerance was rtol=atol=2e−11. Maximum acceleration error was 2.12e−13,
full-tick error 3.56e−15, encoder error 1.78e−15 and state reconstruction
error 5.77e−14. Compiler/source/binary hashes and flags are frozen in the cohort.

The parallel final cohort took **313.6 seconds wall time with four workers**.
Its individual episode/controller/physics timings are also retained, but
concurrent shared-host timings should not replace serial latency comparisons.
Setup is reported separately: exporting the existing encoder and constructing
the frozen classical gains took 7.05 seconds once; warm-import controller
loads in the serial benchmark took about 1.4 ms classical, 2.4 ms NumPy JEPA
and 11.2 ms original JEPA. Fresh imports and symbolic predictor construction
have additional cost. These are simulation measurements on Spark's ARM CPU,
not GPU throughput results or hardware deadline guarantees.

## Where the bloat was

The **trained model is already in the repository** at
`artifacts/npole10/jepa/model.pt`: 442,606 bytes, with 78,784 trainable
parameters and 29,120 additional frozen target-encoder parameters. The
runtime target encoder export is 109,785 bytes, and the reference/gain export
is 311,830 bytes. Neither was retrained or quantized.

The base wheel requires **NumPy only**. Torch is the optional `jepa` extra;
MuJoCo/CasADi are `simulation`, and Gymnasium is `gym`. The wheel excludes
the full scientific recordings. A fresh wheel installation with simulation
and Gymnasium occupied about **354 MB apparent size** on the validation host,
including dependencies, and completed a full episode with each no-Torch
controller. The original local training/simulation environment occupied
about 5.68 GB, mostly Torch/CUDA/Triton dependencies. This is a packaging
comparison of specific installations, not the size of the neural weights or
a universal download-size claim.

Fresh base and inference install receipts are retained alongside the cohort.
`scripts/check_numpy_install.py` proves the base wheel has no Torch, CasADi,
MuJoCo or Gymnasium. `scripts/check_install.py` checks a noneditable wheel,
both 1,000-tick MuJoCo controllers and the Gymnasium API without Torch.

## Why the small model works here, and what it does not show

The observation is a 33-value state vector with angles, velocities and cart
state, rather than images or a long video history. A small MLP maps it to
64 latent values. Its encoder has a locally full-rank 22-state Jacobian on
the checked path, enabling accurate local inversion. Classical trajectory
feedback supplies the actual control solution. Successful inversion is a
much narrower result than learning useful predictive dynamics.

The original inverse always performed 20 refinements before checking its
result, potentially repeating another 20. The optimized version stops when
the residual reaches 1e−14, retains the final 1e−12 acceptance gate and rank
checks, and raises on failure. It never substitutes raw state for failed
inversion. NumPy executes the same frozen target tensors in float64.

The learned dynamics predictor still fails its physical forecasting and
action-sensitivity gates. It does not select the successful actions. A
larger network or a different training GPU would not by itself fix that
missing evidence. The fully observed classical baseline already succeeds.

## Next experiments and increasing the pole count

1. Use the classical policy as the incumbent for this observed-state task.
   Keep the NumPy JEPA path for representation research and the original
   implementation for historical reproduction.
2. Give future JEPA work a falsifiable job: physical multi-step prediction
   that beats persistence and known-dynamics baselines, action-shuffle
   sensitivity, or a closed-loop advantage under predeclared observation
   loss/noise or unknown dynamics. Freeze data splits and success gates before
   evaluating; report cost alongside accuracy and retain failures.
3. For increasing pole count, first solve the reference trajectory under the
   original equations, then check numerical convergence, delay compensation
   and gain conditioning. Compare direct-state and learned-estimator paths
   under the same body and budget before attributing progress to learning.
4. Follow the [sequential roadmap](ROADMAP.md): twelve is qualified in the
   later private cohort; thirteen still needs an original-equation feasible
   reference and fresh closed-loop qualification. This ten-pole acceleration
   does not establish a faster twelve-pole run or a thirteen-pole pass.
5. Freeze each next controller and protocol before 32 reserved starts on
   each independent engine. Publish complete traces and a mechanical/task
   comparison to external work before claiming a record. Priority remains
   unverified.

Reproduce the serial profile with:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
uv run --all-extras python scripts/benchmark_runtime.py --out runs/serial-benchmark.json
```

The project is [a public extraction from private Rage Mill](PROVENANCE.md).
Earlier private history and later private-count experiments are not implied
to be part of this public Git history or this ten-pole qualification.
