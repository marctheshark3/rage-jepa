# Origin and provenance

Rage JEPA was extracted from **our private Rage Mill research repository** on
October 14, 2026. This public repository begins with a curated scientific
snapshot of work already completed privately. Its first commit is not the
date the model was designed or trained, and the public Git log is not the
complete research history. The `rage_mill` Python namespace reflects that origin.

The public package includes the actual trained ten-pole model, normalization,
datasets, reference trajectory, controller implementation, final recordings
and retained development evidence. It does not depend on access to the private
repository. A clean clone cannot contain the private repository's ignored
`runs/` directory; the released material lives under `artifacts/` and `evidence/`.

## Historical extraction

The original [release manifest](../release-manifest.json) records source commit
`f2c27f48f96901c34d199f7060b0653638b37a3f` from the captured private experiment
source. This identifier is provenance, not an accessible public Git commit.
The curated export removed laboratory-only data adapters, machine paths and
non-pole assets. It did not change the model tensors or numeric datasets.

Metadata rewriting changed archive/container hashes. The original private
checkpoint hash is
`ae387e7eebbf28097bb62cc7c8e664a27a0c1c23a4255ef56cea033e96bc5c6a`;
the portable public checkpoint hash is
`c4b56d3dbe26134516fd23863911861c70d6fc42e842a3502f0ac28e9901b084`.
The tensors and normalizations are identical. Historical receipts refer to
original files; use the manifest's explicit mapping to their public equivalents.

The complete initial public snapshot is tag `v0.1.0`, commit
`3cfb3ecc84628c2134b93c17b4e6625b02536ee8`. The original public model,
datasets, scientific source, reference and 64 final recordings remain unchanged
in v0.2. Its manifest remains unchanged too. README, dependency metadata,
lockfile, CI workflow and verifier necessarily evolve; their original bytes
are preserved under [`evidence/releases/v0.1.0/source/`](../evidence/releases/v0.1.0/source/).
The verifier resolves only those five explicit historical relocations and
still checks every original hash. Checking out `v0.1.0` restores the original
layout without relocations.

## Runtime exports and new results

Version 0.2 adds an environment-only API, optional Gymnasium adapter, direct
classical baseline, inference-only NumPy JEPA path, optional compiled CPU Euler
backend and a matched comparison. These are additive implementations. They do
not overwrite the original passing result or make the failed predictor pass.

`scripts/export_runtime.py` exports the frozen target encoder tensors and
executed R20/R1000 feedback gains. Its receipt, input hashes, library versions,
Riccati diagnostics and output hashes are packaged at
`src/rage_mill/assets/npole10-runtime.json`. This is export and offline gain
construction from the existing reference, without retraining or reoptimization.
The executed gains are fixed float64; reproducing the offline longdouble solve
on another architecture need not produce identical gains.

The optional C CPU mechanics were also extracted from our authored private
Rage Mill code and restricted here to the public 2–10 pole point-mass bodies.
The compiler, source, binary, flags and architecture are recorded in each run.
The equations and velocity-first Euler update are unchanged, but a different
linear solve and operation order mean floating-point trajectories need not be
bitwise identical. Paired numeric checks and fresh closed-loop tests are
required; speed alone does not establish equivalence.

New results live under `evidence/runtime-v0.2/`, with seeds frozen before
evaluation, source and asset hashes, all recordings, failures and an independent
audit. They are separate from the original v0.1 final cohort. Twelve-pole
summary evidence describes later private research; this release does not
include every artifact needed to reproduce that separate experiment.

Code, model weights, authored simulation data and documentation are MIT
licensed. Private robot assets, credentials and unrelated laboratory projects
are excluded. No noncommercial robot meshes are included.
