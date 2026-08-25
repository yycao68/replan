# Verification code for the predictive-realizability paper draft

Implements and runs a subset of the benchmark design in
`../predictive_realizability_paper_draft.md`, Sec. VIII, on a 3-DOF planar
reduced-order arm (the FR3-realizable analogue used throughout that section).
Run `python3 run_all.py` from this directory.

## What this is

- `models/planar3r.xml` -- a MuJoCo model of a 3-link planar revolute arm moving
  in a vertical plane (so gravity produces a real, configuration-dependent torque
  demand), with a programmatically-settable end-effector payload and Franka-FR3-
  like per-joint torque limits (87/87/12 Nm).
- `dynamics.py` -- thin wrapper around MuJoCo's own forward/inverse dynamics
  (`mj_inverse`, `mj_fullM`). No hand-derived symbolic dynamics anywhere, except
  the external-end-effector-force term: `mj_inverse` was found, empirically, to
  simply not respond to `xfrc_applied` at all (verified via a linear-algebra
  check against a ctrl=0 forward-dynamics rollout, which sidesteps actuator
  clipping and confirms the true generalized force is `J(q)^T @ F`), so that term
  is added by hand using the module's own (finite-difference-verified) Jacobian.
  `tests/test_dynamics.py` has a dedicated equilibrium test for this.
- `certificate.py` -- the m_tau / m_phys predictive realizability certificate
  (paper Sec. IV), with an explicit, fixed uncertainty margin `delta_tau`.
- `local_planner.py` -- the Level 0-4 hierarchy (paper Sec. V-B): Level 1 (retime)
  and Level 3 (reroute) are decided once, at planning time, over the *whole*
  candidate route; Level 2 (reshape, a convex QP over the horizon's acceleration
  profile) and Level 4 (brake) are monitored online, every control cycle. See the
  module docstring for why that split exists -- an earlier version of this code
  applied retiming as an instantaneous per-cycle correction and it silently did
  nothing, because the position reference kept marching forward at the original
  pace while only qdot/qddot were locally scaled.
- `executor.py` / `baselines.py` -- B1 (no handling), B2 (reactive, current-
  instant-only clipping/throttling), B3 (the full predictive architecture, or an
  ablation of it via `PlannerConfig` flags) all share one computed-torque tracking
  controller and one MuJoCo plant, differing only in what reference they feed it
  -- the fairness point Sec. VIII-A insists on for B2 vs. B3.
- `metrics.py` -- task success, saturation events, tracking error, and the
  conservatism check (Sec. VIII-I): whether a certificate trigger corresponded to
  an actual ground-truth torque-limit violation of the *un-adapted* nominal
  trajectory, or was a false positive.

## What's implemented and verified

`tests/test_dynamics.py` and `tests/test_planner.py` are correctness checks, not
just smoke tests: mass-matrix symmetry/PD, a closed-form static-gravity-torque
check, a finite-difference Jacobian check, and -- importantly -- a negative case
confirming that a pure static (gravity-only) torque deficit is *not* fixable by
retiming, only by reshaping/rerouting/braking, which is what the paper's own
Level-1 discussion implies should happen.

All six of `experiments/exp1_baseline.py` through `exp6_severity_sweep.py`, plus
`experiments/ablation_batch.py` (A1-A5, Sec. VIII-H), are implemented and
runnable, with results below from an actual run (not fabricated -- rerun
`run_all.py` to reproduce). Getting Exp 3/4 correct required extending the force
interface from a constant `(2,)` value to a callable `ee_force_fn(t, q) ->
force`, since Exp 3 needs a purely time-based schedule (ramp onset) while Exp 4
needs a *position*-based one (a contact force that depends on where the
end-effector actually is, which depends on which reference is currently
active); `local_planner.py`'s `plan_route` vs. `online_step` split also governs
whether B3's certificate gets to see that force schedule in full at planning
time (`force_known_at_plan_time=True`, Exp 4: a known upcoming contact
transition) or only within the bounded online prediction horizon as it comes
into view (Exp 3: an unanticipated disturbance -- this is what gives "detection
lead time" a real, horizon-bounded meaning instead of oracle knowledge). The
ablation batch needed one more addition: `PlannerConfig.allow_level4`, since A4
("Levels 0-2 only; Level 3/4 disabled") couldn't previously be expressed --
Level 4's brake fallback was unconditional.

`experiments/timing_benchmark.py` (Sec. VIII-J) measures wall-clock per-cycle
computation time directly, wrapping each baseline's policy call with
`time.perf_counter()` -- B3's ONE-TIME route-planning cost (Level 1/3, a whole-
route search) is timed separately from its PER-CYCLE online cost (Level 0/2/4),
since only the latter is actually subject to the 20ms (50 Hz) real-time budget.

## Real results from the current implementation

- **Exp 1 (no-regression check):** B1/B2/B3 are bit-for-bit identical on a benign
  trajectory (0.3 mm final error, no saturation) -- no regression from adding the
  predictive layer when it isn't needed.
- **Exp 2 (payload sweep, same geometry+timing):** B1/B2/B3 all first saturate at
  2.0 kg payload. An earlier draft of this README claimed B3 uniquely completes
  the task at 3.0 kg where B1/B2 both fail -- **that was wrong**, caught by
  rerunning at finer payload resolution: by 3.0 kg *all three*, including B3,
  fail the task (117-190 mm final error). The real, verified crossover is at
  2.4 kg, and even there the honest picture is narrower than first claimed: B1
  fails (34.8 mm error) but **B2 also succeeds** (not just B3) -- B2's simple
  reactive throttle is just as capable as B3's predictive retiming in this
  specific scenario, so this sweep does not by itself demonstrate B3 beating
  B2. A further surprise turned up while checking this: at 2.6-3.0 kg, **B3 is
  actually worse than B1** (e.g. 189.6 mm vs. 117.2 mm error at 3.0 kg) --
  partial Level-1 retiming that can't fully restore margin apparently leaves
  the arm in a worse tracking state than doing nothing, an unflattering but
  real finding, reported rather than smoothed over. Conservatism over the full
  sweep: 0/13 triggers were false positives (every trigger corresponded to a
  real violation of the unmodified nominal trajectory).
- **Exp 3 (interaction force, detection lead time):** a downward push (ramp onset
  at t=0.35s, held at -55N) is applied to a moderate-payload trajectory. Ground-
  truth failure of the unmodified nominal trajectory occurs at t=0.90s. B2 (no
  lookahead) first reacts at t=0.90s -- exactly at failure, T_warning=0.00s, as
  the paper's own framing predicts. **B3 first detects at t=0.48s, T_warning=0.42s.**
  All three still fail the task at this force magnitude (peak torque ratio 1.00
  for all) -- an honest result: the primary claim here is the lead time itself,
  not that early detection guarantees recovery at any severity. (A parameter
  sweep confirms B3 *does* fully avoid failure at lower force magnitudes, e.g.
  30N, but so do B1/B2 there -- not a differentiator -- and at intermediate
  magnitudes, 37-40N, B3 does not reliably show fewer saturation events than
  B1/B2 either; that non-monotonicity is not yet understood and is flagged as an
  open item rather than smoothed over.)
- **Exp 4 (contact-stiffness transition, known in advance):** a scripted contact
  force turns on once the end-effector descends past a virtual plane; unlike
  Exp 3, B3 is given this contact model at planning time. B2 (reactive) first
  responds only once already in contact, t=0.70s; **B3 responds at t=0.42s**,
  before the transition, and finishes with **9 saturation events vs. 27-29 for
  B1/B2** (a ~3x reduction), though again none of the three reach full task
  success at this contact stiffness -- the same honest caveat as Exp 3.
- **Exp 5 (flagship):** P_A (outstretched pose under a 4.5 kg payload) saturates
  and fails under B1 and B2 (peak torque ratio 1.00, 36-37 saturation events).
  B3's certificate predicts the deficit before execution, reroutes to P_B, and
  completes with **zero saturation events**. This is the paper's central claim
  (Theorem 3) exercised end to end.
- **Exp 6 (severity sweep):** with payload as the severity knob, the hierarchy
  visibly moves through Level 0 -> 1 -> 3 -> 4 as severity increases, with task
  success finally failing at the most extreme severities even with Level 3/4
  engaged (an honest negative case -- adaptation is not unconditional). One
  finding worth flagging: **Level 2 (reshape) was only ever selected when Level 1
  was ablated** -- in the un-ablated hierarchy, Level 1 is checked first and
  handles every deficit in the tested range that Level 2 could also have fixed,
  so Level 2's marginal contribution over Level 1 is not yet demonstrated by
  these experiments. That would need a scenario engineered so retiming is
  provably insufficient but a *non-uniform* acceleration change is not -- not yet
  constructed here.
- **Ablation batch A1-A5:** run on two scenarios chosen because they're known to
  need different levels of the hierarchy. On Exp 2's 2.4 kg crossover
  ("retime-suffices"): **A3 (predict, don't act) is bit-for-bit identical to A1
  (no prediction at all)** -- same level trace, same 10 saturation events, same
  failure -- a clean, exact confirmation that prediction alone has zero value
  unless something acts on it. A2 and A4/A5 all succeed here (consistent with
  Exp 2's finding above that this scenario doesn't separate reactive from
  predictive handling). On Exp 5's flagship P_A/P_B scenario
  ("reroute-required"): **A4 (predict + retime/reshape, no reroute or brake)
  fails** -- it does trigger Level 2 reshaping (`levels=['0','2']`) but ends up
  *worse* than doing nothing (41 saturation events vs. A1's 36) -- while **A5
  (the full architecture) succeeds cleanly via Level 3, zero saturation
  events**. This is exactly the isolation the paper's own Sec. VIII-H text asks
  for: rerouting's marginal contribution, shown on the one scenario nothing
  short of it can fix.
- **Real-time timing benchmark (Sec. VIII-J):** on Exp 1's nominal trajectory,
  B3's online per-cycle step easily fits the 20ms/50Hz budget (mean 0.27ms, max
  under 0.5ms across repetitions) -- unsurprising, since Level 0 (do nothing)
  is all it ever needs there. On **Exp 5's stress case (P_A), it does not**.
  This finding needed a second pass to get right: a single rollout's max is
  not trustworthy on a shared dev machine, since wall-clock latency is noisy
  (one early run showed max=17.2ms, i.e. apparently fitting, until repeated
  runs revealed that was the lucky outlier, not the typical case). The
  benchmark now pools 5 independent repetitions: mean (~12ms) and p95
  (~17.5ms) are stable and reproducible across repeats, both nominally under
  budget on average, but **the per-repetition max is consistently over
  budget** -- typically 40-90ms (2-4.5x budget) in 4 of 5 repeats, with an
  occasional repeat closer to the boundary (~21-22ms) but never actually
  under it in repeated testing. Disabling Level 2 drops the mean to ~0.4ms,
  isolating the cause: the Level-2 QP (cvxpy/OSQP, which reconstructs the
  optimization problem from scratch every single call rather than reusing a
  compiled/parametrized form) accounts for **~97% of the online-step cost**
  whenever it's actually being solved every cycle. This is a genuine, reported-
  as-found limitation, not smoothed over: none of the simulated results above
  are invalidated by it (this is offline simulation -- wall-clock Python cost
  doesn't change what the physics/control-logic computed), but it does mean
  this specific implementation, as written, would not meet a real 50Hz
  hardware control loop under stress-case conditions without either a
  parametrized/warm-started QP formulation (cvxpy supports this via
  `Parameter`, not yet used here) or a lower online replan rate for Level 2
  specifically.

## Known simplifications (stated once, applies throughout)

Reduced-order 3-DOF planar arm, not a full FR3 model. Level-1/3 decided once at
planning time rather than continuously re-optimized (see `local_planner.py`
docstring). `delta_tau` (Theorem 1's uncertainty bound) is a fixed 5% of
`tau_max`, not estimated online. No collision/obstacle feasibility set is
modeled (`F_obs` from the paper's Sec. III is out of scope here; only `F_dyn` is
tested). Theorem 4 (recursive feasibility) is not attempted, consistent with the
paper draft's own "deferred" framing.
