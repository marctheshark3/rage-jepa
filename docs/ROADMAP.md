# Climbing the pendulum count toward a comparable record

The ten-link model is this release's reproducible starting point. The later
lab cohort qualifies **twelve passive serial links**: 32/32 final hanging starts
on CPU Euler and 32/32 on native MuJoCo. Eleven passed 31/32 Euler and 32/32
MuJoCo; that failed catch remains in the lab evidence. Twelve's predictor
fails its forecast gates too. Summary audits and metrics are included under
`evidence/twelve-poles/`; twelve's controller, checkpoint and raw recordings
are outside the scope of this ten-pole release.

Twelve's Euler integrator required 163.84 million steps per simulated second.
Its final parallel batch took about 136 minutes for twenty simulated seconds
per trial. MuJoCo remained at 1,000 Hz. Neither machine was actuating hardware;
neither figure establishes wall-clock real-time operation.

## Next qualified count: thirteen

Thirteen has a trained encoder and local inversion checks, but no qualified
swing reference or controller trials. Its latest continuation candidate is
feasible for modified optimization equations; its defect under the original
equations is about 0.00886 and still fails the ≤1e−7 reference requirement.
An artificial-equation warm start is progress in planning, not a control pass.

1. Resume the retained feasible continuation iterate. Gradually remove its
   artificial defect while retaining failed attempts and genuine solver status.
2. Check the candidate against the original dynamics, actuator/rail bounds,
   endpoints and sampled encoder inversion. A timed-out solver can supply a
   feasible reference; never claim optimality from a timeout.
3. Run bounded development trials on both engines. Inspect capture failures,
   encoder conditioning, Riccati precision and integration-step convergence.
   More network parameters are only one possible intervention, not the default.
4. Freeze model, plan, gains, physics settings and source identities before
   opening 32 reserved final starts per engine. Require ≥26/32 on each engine,
   with every passing trial satisfying the continuous final-five-second hold.
5. Keep every failure and full trace. Advance to fourteen only after thirteen
   qualifies under the unchanged task gates.

The steep Euler cost is now a research bottleneck. Independently test more
efficient integration and conditioning as separate experiments; define any
new comparable protocol before final trials, retain the historical protocol
and report its changed conditions. Faster integration is not evidence of a
larger qualified count by itself.

## A record needs a definition and independent reproduction

Use a dated comparison table: simulation versus hardware; serial versus
parallel links; one cart actuator versus actuated joints; hanging swing-up
versus a tiny upright start; masses/lengths, damping, cart/force limits, actual
delay, command rate, hold duration, number of fresh trials and wall-clock cost.
Link primary source code or papers and mark unknown conditions as unknown.

For context, [Wolfram's 2011 n-link control demonstration](https://blog.wolfram.com/2011/03/01/stabilized-n-link-pendulum/)
and [the quadruple-pendulum control report](https://arxiv.org/abs/2203.01409)
show why pendulum count alone is insufficient: upright stabilization and
hanging swing-up test different behavior. Published clips also need a
complete mechanical/control specification to become comparable evidence.

Invite another group to reproduce the released cohort and the next qualified
count. Publish frozen artifacts and failed traces, then establish priority
within an agreed category. The current evidence supports a twelve-link
simulation milestone; global record priority remains unverified.

## Improve the learning claim separately

Hold mechanics, reference, gains, seeds and inversion tolerance fixed while
comparing the trained encoder, a random encoder and raw-state feedback on new
reserved cohorts. For a future learned planner, require physical forecasts to
beat persistence and action shuffling to make predictions measurably worse,
then evaluate its actual closed-loop actions. A small invertible encoder and
a useful learned dynamics model are separate research outcomes.
