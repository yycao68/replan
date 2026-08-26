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

  Level 4 had a second, related bug, found while investigating an "unexplained
  non-monotonicity" this README used to flag in Exp 3/4 (see below): its brake
  profile was starting from the NOMINAL trajectory's position at the current,
  still-advancing wall-clock time rather than from where the robot had actually
  stopped, so it never really held position -- it kept crawling forward to chase
  the original plan. Fixed by passing the true `q_actual`/`qdot_actual` into
  `online_step` for the brake profile specifically (not for Level 0/2's
  returned reference -- an earlier, broader attempt at this routed the true
  state into those too and broke tracking entirely, since Level 0's whole job
  is to keep pulling toward the *nominal* reference, not "wherever the robot
  currently is"). Fixing the brake itself then exposed a THIRD, previously
  unreachable bug: once Level 4 genuinely holds position for a while, the
  certificate can later swing back to "feasible" (checked against the nominal
  trajectory's own, un-overridden samples) and return a raw nominal reference
  that is now far from the robot's real position -- an instantaneous reference
  jump that visibly destabilized the MuJoCo simulation (NaN/Inf qacc warnings,
  torque values around 1e13). `baselines.policy_b3` now makes Level 4 *sticky*:
  once triggered, the policy holds the robot at its actual position for the
  rest of the rollout rather than going back to querying the planner, matching
  the paper's own framing of Level 4 as a "terminal safe-set policy" -- there is
  no automatic resume, that would need a fresh route decision (Level 3) or a
  higher-level replan, and this codebase does not invoke either automatically.
- `executor.py` / `baselines.py` -- B1 (no handling), B2 (reactive, current-
  instant-only clipping/throttling), B3 (the full predictive architecture, or an
  ablation of it via `PlannerConfig` flags) all share one computed-torque tracking
  controller and one MuJoCo plant, differing only in what reference they feed it
  -- the fairness point Sec. VIII-A insists on for B2 vs. B3.
- `metrics.py` -- task success, saturation samples, tracking error, and the
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
  the paper's own framing predicts. **B3 first detects at t=0.48s, T_warning=0.42s**
  (unchanged by the Level-4 fix below -- detection timing is independent of
  what happens afterward). All three still fail the task at this force
  magnitude, but B3's *character* of failure changed once Level 4 was fixed to
  actually hold position (see "What this is" above): saturation samples dropped
  from 39 to **19**, and tracking error against its own (now sensible, static)
  reference collapsed from 671.78 mrad to **1.49 mrad** -- it isn't chasing a
  moving target anymore, it's genuinely stopped. This README used to flag an
  "unexplained non-monotonicity" here (B3 sometimes showing *more* saturation
  events than B1/B2 at intermediate force magnitudes, 37-40N) as an open item.
  It's now understood, via the same investigation that found the Level-4 bugs
  above: at fmag=37N, B3 correctly holds position from t=0.68s once the force
  is still ramping, but the force keeps rising after that (ramp completes at
  t=1.25s and holds at full magnitude) -- and the STATIC holding torque at
  whatever pose B3 happened to stop in can itself exceed the actuator limit
  once the force reaches its full value, so the robot starts sliding under the
  disturbance anyway, still commanding "hold" the entire time. B1/B2 never
  stop moving and may pass through momentarily more favorable configurations
  or clear the high-demand region before the force reaches full magnitude.
  This is a genuine, physically real property of Level 4 as designed -- a
  static hold provides no guaranteed robustness against a disturbance that
  keeps growing past what CAN be held statically -- not a bug, and not
  obviously fixable without either a smarter Level 4 that re-picks its holding
  pose if the current one stops being tenable, or coupling it to Level 3
  (reroute to a genuinely better configuration instead of freezing wherever
  the trigger first fired). A parameter sweep still confirms B3 fully avoids
  failure at low force magnitudes (<=25N, matching B1/B2 there -- not a
  differentiator) and, past ~30N, B3 fails the task via a permanent hold
  (Level 4 has no resume path) even at magnitudes B1/B2 still complete despite
  transient saturation -- a real, honest safety-vs-completion trade-off, not
  a strict win for either side.
- **Exp 4 (contact-stiffness transition, known in advance):** a scripted contact
  force turns on once the end-effector descends past a virtual plane; unlike
  Exp 3, B3 is given this contact model at planning time. B2 (reactive) first
  responds only once already in contact, t=0.70s; **B3 responds at t=0.42s**,
  before the transition, and -- after the Level-4 fix -- finishes with **zero
  saturation samples** (previously reported as 9, before Level 4 was fixed to
  actually hold rather than crawl) **vs. 27 for B1 and 8 for B2**. B2's number
  dropped from a previously-reported 29 once its one-step throttle became a
  genuine torque-feasible QP projection instead of a heuristic uniform scale
  (see "What this is" above) -- a real, more capable B2, and still clearly
  worse than B3's 0. None of the three reach full task success at this contact
  stiffness (B3 again holds permanently once braked), but B3's own outcome is
  now unambiguously clean: it detects early, stops, and never saturates again.
- **Exp 5 (flagship):** P_A and P_B are two ROUTES between the SAME start and
  SAME goal (via `trajectory.ViaPointTrajectory`'s q0 -> via -> qf), not two
  different final configurations. (An earlier version of this experiment had
  P_A and P_B end at two different goals, with the success metric scored
  against "whichever goal was actually reached" -- that let B3 pass by simply
  abandoning the task for an easier target, which is not the paper's claim,
  and inflated `replans` to roughly one per control cycle instead of counting
  actual replanning events. Both are fixed now; see `baselines.policy_b3` and
  `trajectory.ViaPointTrajectory`.) P_A passes through a near-singular,
  outstretched via-point under a 4.5 kg payload; B1 and B2 attempt it and
  never reach the shared goal (peak torque ratio 1.00, final position error
  0.94 m and 0.82 m respectively, 45/54 saturation samples -- B2's error grew
  from a previously-reported 0.10 m once its throttle became a genuine
  torque-feasible QP projection instead of a heuristic scale that, on this
  scenario, happened to permit more motion than it actually certified as
  safe; see "What this is" above). B3's certificate
  predicts the deficit before execution (persists even at 8x retiming --
  the bounded-retiming-exhausted case Theorem 3's sufficient condition
  `m*_1(p) < m_safe` is built to detect, not a proof of the idealized
  `T_dyn(p) = empty` itself), reroutes to P_B, and reaches the
  IDENTICAL goal with **0.001 m final error and zero saturation samples**,
  `replans=1`. This is the paper's central claim (Theorem 3) exercised end to
  end, on a genuine same-goal route comparison.
- **Exp 6 (severity sweep):** reuses Exp 5's exact same-goal P_A/P_B geometry
  (same fix as above), swept by payload. The hierarchy visibly moves through
  Level 0 -> 1 -> 1 -> 3 -> 3 -> 4 as severity increases (payloads
  0/1.5/2.5/3.5/5.0/8.0 kg), succeeding at every level up to and including
  Level 3, with task success finally failing only at the most extreme severity
  (8.0 kg) even with Level 4 engaged (an honest negative case -- adaptation is
  not unconditional). `replans` is 0 at Level 0 and 1 at every other severity,
  correctly counting the single route-level (or brake-engagement) event, not
  control cycles. One finding worth flagging: **Level 2 (reshape) was never
  selected in the un-ablated hierarchy at any tested severity** -- Level 1 is
  checked first and handles every deficit in the tested range that Level 2
  could also have fixed, or Level 3 is needed outright, so Level 2's marginal
  contribution over Level 1 is not yet demonstrated by these experiments. That
  would need a scenario engineered so retiming is provably insufficient but a
  *non-uniform* acceleration change is not -- not yet constructed here.
- **Ablation batch A1-A5:** run on two scenarios chosen because they're known to
  need different levels of the hierarchy. On Exp 2's 2.4 kg crossover
  ("retime-suffices"): **A3 (predict, don't act) is bit-for-bit identical to A1
  (no prediction at all)** -- same level trace, same 10 saturation samples, same
  failure -- a clean, exact confirmation that prediction alone has zero value
  unless something acts on it. A2 and A4/A5 all succeed here (consistent with
  Exp 2's finding above that this scenario doesn't separate reactive from
  predictive handling). On Exp 5's flagship P_A/P_B scenario
  ("reroute-required", same same-goal fix as Exp 5/6 above): **A4 (predict +
  retime/reshape, no reroute or brake) fails** -- it does trigger Level 2
  reshaping (`levels=['0','2']`) but ends up with a final position error of
  1.35 m, actually worse than A1's 0.94 m (53 vs. 45 saturation samples --
  since the Level-2 objective change below, reshaping-without-rerouting no
  longer merely fails to help here, it fails more than doing nothing) --
  while **A5 (the full architecture) succeeds cleanly via Level 3**,
  reaching the identical goal with 0.001 m error and zero saturation samples.
  This is exactly the isolation the paper's own Sec. VIII-H text asks for:
  rerouting's marginal contribution, shown on the one scenario nothing short
  of it can fix.
- **Level-2 (reshape) objective now penalizes deviation from the nominal
  trajectory, not raw acceleration magnitude** (found during a code-vs-paper
  review): the QP previously minimized `sum(qddot^2)`, which has no reason to
  prefer a reshaped motion close to the one the planner originally intended --
  it just prefers small accelerations, which happens to correlate with margin
  but is not the same thing as "closest physically realizable trajectory."
  The objective is now `w_acc*sum((qddot-qddot_nom)^2) +
  w_pos*sum((q-q_nom)^2) + w_vel*sum((qdot-qdot_nom)^2)`
  (`PlannerConfig.reshape_w_acc/w_pos/w_vel`, defaults 1.0/0.1/0.1). This is a
  genuine behavior change, not just a relabeling: staying close to the
  nominal acceleration is not always compatible with restoring margin, and on
  the `test_reshape_qp_solves_online_and_respects_bounds` scenario the
  reshape QP that used to succeed (`m_after=5.62`) now finds a nominal-close
  candidate the nonlinear certificate rejects (`m_after=0.95 < m_safe=2.0`),
  so the planner correctly falls through to Level 4 instead -- an honest
  consequence of the fix, not a regression the test hides (the test's own
  pass/fail assertion is conditional on `level==2` and still passes; it is no
  longer exercising a successful Level-2 outcome, which is itself worth
  flagging as a coverage gap rather than silently accepted). The same change
  is why the ablation-batch A4 numbers above (0.98 m/47 -> 1.35 m/53) and
  Exp 3's B3 numbers (20/1.60 mrad -> 19/1.49 mrad, `replans` 2 -> 1) moved
  from what was previously reported here -- both updated in this README, and
  in the paper draft, to match a fresh rerun after the fix.
- **Real-time timing benchmark (Sec. VIII-J):** on Exp 1's nominal trajectory,
  B3's online per-cycle step easily fits the 20ms/50Hz budget (mean 0.27ms, max
  under 0.5ms across repetitions) -- unsurprising, since Level 0 (do nothing)
  is all it ever needs there. On **Exp 5's stress case (P_A)**, the picture
  changed with the Level-4 fix above, and the change itself is informative.
  Before the fix, B3 kept failing to hold position and re-triggering the
  expensive Level-2 QP essentially every cycle, which is what the earlier
  version of this benchmark measured: a *sustained* load, mean ~12ms, p95
  ~17.5ms, and a per-repetition max consistently over budget (40-90ms in most
  repeats). After the fix, B3 hits Level 4 almost immediately and then holds
  (sticky, no further QP calls) for the rest of the rollout, so the QP is now
  solved just **once** per rollout instead of dozens of times: pooled over 5
  repeats, mean drops to ~0.35ms and **p95 is ~0.002ms** (the overwhelming
  majority of cycles are now the trivial sticky-hold branch), but **max is
  still ~17-18ms** -- that one QP solve, wherever it lands, still costs what a
  QP solve costs. This is a better-supported, more honest way to say the same
  underlying thing the pre-fix number was pointing at: a single Level-2 QP
  solve (cvxpy/OSQP, which reconstructs the optimization problem from scratch
  every call rather than reusing a compiled/parametrized form) costs close to
  the entire 20ms budget on its own, whether it happens once or fifty times.
  The earlier repeated-QP measurements also showed real tail risk up to
  40-90ms for a *single* solve under some conditions, so a lucky one-shot
  ~17-18ms in this particular scenario should not be read as "safely under
  budget" so much as "not the sustained problem it looked like before, but the
  underlying per-solve cost is still a real risk on any single cycle it's
  invoked." None of the simulated results elsewhere in this README are
  invalidated by any of this (offline simulation -- wall-clock Python cost
  doesn't change what the physics/control-logic computed), but it does mean
  this specific implementation, as written, would not safely meet a real 50Hz
  hardware control loop on whichever cycle Level 2 actually engages, without
  either a parametrized/warm-started QP formulation (cvxpy supports this via
  `Parameter`, not yet used here) or accepting that cycle as a bounded,
  planned-for overrun.

  A further, real cost increase came from the Level-2 dynamics-consistency fix
  (below): `_try_reshape`'s QP grew from optimizing only the acceleration
  profile (`N_JOINTS` variables/step) to jointly optimizing acceleration
  *and* the double-integrator state trajectory it implies (`3*N_JOINTS`
  variables/step, plus `2*(horizon_steps-1)` integration-equality
  constraints) -- a materially larger problem for OSQP to solve from scratch
  each call. Measured directly on this same stress scenario: the one-shot
  solve's cost roughly doubled, mean ~4.3ms, p95 ~27ms, **max ~54ms** (was
  ~18-20ms before the fix). This is the honest price of the QP now
  certifying a trajectory it can actually produce instead of a cheaper one
  that mixed the nominal route's positions with different accelerations (see
  `_try_reshape`'s docstring) -- a correctness fix, not a performance one,
  and it makes the pre-existing over-budget finding above worse, not better.

## Known simplifications (stated once, applies throughout)

Reduced-order 3-DOF planar arm, not a full FR3 model. Level-1/3 decided once at
planning time rather than continuously re-optimized (see `local_planner.py`
docstring). `delta_tau` (Theorem 1's uncertainty bound) is a fixed 5% of
`tau_max`, not estimated online. No collision/obstacle feasibility set is
modeled (`F_obs` from the paper's Sec. III is out of scope here; only `F_dyn` is
tested). Theorem 4 (recursive feasibility) is not attempted, consistent with the
paper draft's own "deferred" framing. Level 4 (brake) is sticky and terminal:
once engaged, `baselines.policy_b3` holds the robot at its stopped position for
the rest of the rollout with no automatic resume (see "What this is" above) --
this is a genuine, deliberate architectural limitation, not an oversight, but it
means every result reported here where B3 reaches Level 4 is reporting a safety
outcome (did it stop without violating limits), not a task-completion outcome
(it never finishes the original plan once stopped).
