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

All seven of `experiments/exp1_baseline.py` through
`exp7_environment_conditioned_reroute.py`, plus `experiments/ablation_batch.py`
(A1-A5, Sec. VIII-H), are implemented and runnable, with results below from an
actual run (not fabricated -- rerun
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

`experiments/exp7_environment_conditioned_reroute.py`, added in a follow-up
review pass, closes a gap the reviewer identified directly: Exp 5's flagship
reroute is triggered by a configuration/payload deficit (an outstretched
via-point under a heavy payload), and Exp 4 shows adaptation to a known
contact transition but only ever gives the planner one route -- neither shows
rerouting *because of the environment specifically*. Exp 7 reuses Exp 4's
scripted contact-stiffness model (a virtual plane, penetration-proportional
force via `xfrc_applied`) but gives the planner two routes to the same goal,
as in Exp 5: P_A's via-point dips the end-effector below the contact plane,
P_B's stays clear of it. See "Real results" below for the numbers; the
distinguishing feature by design is that VIA_A's static margin (no force) is
healthy under the light payload used here -- unlike Exp 5, the deficit exists
only once the known environmental force is accounted for, and (like Exp 5's
static deficit) is retiming-proof for the analogous reason: the spring force
is a function of position only, evaluated at the via-point's own zero-
velocity/zero-acceleration boundary, so slowing down does not reduce it.

`local_planner.LocalPlanner._search_reshape_whole_route`, added in a further
follow-up pass, closes the gap the paper's own Sec. V-C flagged as an open
item: `plan_route` previously tried only retiming (Level 1) as route-level
regeneration before falling back to reroute (Level 3) -- reshaping (Level 2)
was tried online only, per control cycle, never at the route-planning
decision itself. Reshape and retime are both genuine regeneration of the
SAME route; only rerouting changes route entirely, so `plan_route`'s order
is now retime, then reshape, then reroute, symmetrically on both the primary
and (if reached) the alternate route. `_try_reshape` gained an optional
terminal-state constraint (`terminal_q`/`terminal_qdot`) so a whole-route
reshape is pinned to reach the SAME goal at rest, not free to drift to a
different final configuration -- the goal-changing failure mode this
codebase already fixed once for Level 3 (see Exp 5's docstring), reopened
for Level 2 if left unconstrained. The reshaped route is wrapped in a new
`trajectory.SampledTrajectory` (piecewise-quadratic reconstruction of the
QP's own double-integrator propagation, so `sample(t)` is an exact, not
approximate, reconstruction -- unit-tested directly in `trajectory.py`) so
it drops into `online_step`/`rollout`/a further retime search like any other
trajectory. Three things were checked empirically before trusting this,
not assumed: (1) whole-route reshape genuinely CANNOT resolve Exp 5's
flagship or Exp 7's environment-conditioned deficit either (confirmed
against SCS as an independent solver, not just OSQP's own status flag --
see below), so Level 3 remains genuinely necessary in both of this
benchmark's existing reroute-required scenarios, not merely because reshape
was never tried; (2) a scenario DOES exist where reshape succeeds where
retiming cannot (`tests/test_planner.py`'s
`test_route_level_reshape_restores_feasibility_when_retiming_cannot`) --
interestingly, not by geometrically routing AROUND the force field (the
reshaped path's minimum end-effector height is, if anything, lower than the
nominal path's), but by reallocating velocity/timing WITHIN the same total
route duration, a genuinely different mechanism from retiming's uniform
global time-scaling; (3) this whole-route QP is large enough that OSQP does
not reliably converge on it (see the real-time cost finding below), so
`_try_reshape` now falls back to SCS on OSQP's failure, confirmed necessary
by direct testing, not adopted speculatively.

### Non-monotonic retiming margin: a real bug in `_search_retime_whole_route`, found and fixed

Raised as a theoretical concern in a review of the paper draft, not found by
code inspection first: per joint, torque under uniform time-dilation by
`lambda` decomposes as `tau(lambda) = A/lambda**2 + B`, where `A` is the
inertial/Coriolis contribution (evaluated at the same path fraction, so `A`
itself is independent of `lambda`) and `B = g(q) + J^T F_ext(q)` is the
velocity-independent (gravity + position-only external force) contribution,
unchanged by retiming. `m_phys(lambda)` is monotonically non-decreasing in
`lambda` -- what the old bisection search assumed without checking -- only
when `sign(A_i) == sign(B_i)` for the binding joint at every `lambda` tested.
When some joint's inertial/Coriolis torque OPPOSES its gravity/external-force
torque (e.g. decelerating a downswing partially unloads gravity-holding
torque), `A/lambda**2 + B` can cross zero at a finite `lambda*` and grow
again beyond it, so `m_phys(lambda)` is genuinely non-monotonic.

This was checked directly against the code, not just algebraically: a random
search over 3000 (q0, qf, T, payload) draws on this platform found 23 cases
(~0.8%) of the exact failure mode that matters -- `_search_retime_whole_route`
checks only `lambda_max`, and if that alone fails, returned `None`
("retiming exhausted, proceed to reroute") even when a dense scan shows an
INTERIOR `lambda`, sometimes far from `lambda_max`, actually restores the
margin:

```
trial=26   m0(lambda=1)=-1.51   old code: None (declares retiming exhausted)
           dense scan: max m_phys=2.86 (>= m_safe=2.0) at lambda=1.55
           m_phys(lambda_max=4.0)=1.66 (< m_safe) -- the only point old code checked
trial=146  m0(lambda=1)=-82.58  old code: None
           dense scan: max m_phys=2.06 (>= m_safe=2.0) at lambda=3.50
           m_phys(lambda_max=4.0)=1.80 (< m_safe)
```

Fixed: `_search_retime_whole_route` keeps the fast bisection path when
`lambda_max` alone clears `m_safe` (unaffected, same cost as before), but no
longer gives up immediately when it doesn't -- it dense-scans 41 points
across `[1, lambda_max]`, and if any clear `m_safe`, refines the bracket
around the first crossing with a local bisection (valid there even though
the function isn't globally monotonic) rather than assuming the interval is
hopeless. Verified directly: both trials above now correctly return a
rescuing `lambda` (1.216 and 3.452 respectively, each landing exactly at
`m_phys = m_safe = 2.00`).

**This does not change any number already reported in this paper.** The two
scenarios whose Theorem-3 conclusion is actually cited (Exp 5's flagship,
Exp 7's environment-conditioned reroute) were checked with a dense scan
before and after the fix: the flagship's `m_phys(lambda)` is genuinely
monotonic there (sup is at `lambda_max`, as assumed), and Exp 7's is
technically non-monotonic but the interior deviation (~0.5) is negligible
against how far below `m_safe` it stays (~-20) -- neither conclusion
flips. The bug is real and reproducible, but it was a soundness gap in the
implementation, not a corrupted result. It does, however, make retiming
itself more expensive on scenarios that hit the new fallback: on the
timing-benchmark stress case (which was already known to fail lambda_max's
check), retime-only rose from ~3ms to ~63ms -- still two orders of
magnitude cheaper than reshape's 300-900ms, so the qualitative real-time
story is unchanged, but the specific retime-only baseline number moved and
is updated above and in the timing benchmark's own output.

### Closed-form retiming candidates: implemented, found a real gap between the theory and the actual dynamics, fixed it, and it turned out not to be a speedup

The dense-grid fallback above was always meant to be a stopgap;
`monotonicity_lemma_draft.md` Sec. 4(b) sketches an exact alternative: since
`tau_i(lambda) = A_i/lambda**2 + B_i`, wherever `sign(A_i) != sign(B_i)` the
interior extremum of `|tau_i(lambda)|` has a closed form (`lambda*_i =
sqrt(A_i/-B_i)`), so the true supremum can in principle be checked at a
finite candidate set instead of a grid. Implementing this directly (a first
pass, `A/lambda**2 + B` exactly as derived "by the manipulator equation")
and testing it against the actual code surfaced a real, previously
undocumented gap: `models/planar3r.xml` gives every joint real viscous
damping (`dof_damping=0.3`), which the idealized manipulator equation the
Lemma is derived from does not include. Verified directly -- with
`dof_damping` zeroed, the 2-term model reproduces `mj_inverse`'s actual
torque to numerical precision at every `lambda` tested; with damping
present (the model this benchmark actually uses), the 2-term model is off
by several percent and growing with `lambda`, enough to occasionally flip
a margin decision near `m_safe`.

Fixed by extending the decomposition to a THIRD term: under retiming, a
linear damping torque `-c*qdot` scales as `1/lambda` (not `1/lambda**2`),
so the exact model is `tau_i(lambda) = A_i x**2 + D_i x + B_i` with
`x = 1/lambda` -- a genuine quadratic in `x`, not an assumption about
MuJoCo's specific passive-force implementation: `A_i`, `D_i`, `B_i` are
recovered by fitting three `required_torque` evaluations at the same `q`
with scaled velocity/acceleration (`x=0`, `x=0.5`, `x=1`) to the known
functional form, so it is robust to whatever MuJoCo's internal numerics
actually are. This closes the residual to ~0.02 Nm (consistent with the
much smaller, genuinely non-quadratic Coulomb friction term,
`dof_frictionloss=0.05`, which no closed polynomial form captures
exactly). The candidate set itself generalizes correspondingly: each
joint/step now contributes up to two zero-crossings (quadratic formula)
plus the parabola's own vertex, mapped back to `lambda`, instead of one
`lambda*_i`.

**Honest result: this did not deliver a wall-clock speedup on this
benchmark, and the reason is structural, not a modeling-accuracy issue.**
Re-running the same 3000-scenario random search with the fixed model:

```
lambda_max alone already feasible:        1916/3000 (63.9%) -- no search needed
needs the search (lambda_max infeasible): 1084/3000 (36.1%)
  closed-form candidates sufficed:          21/1084  (1.9%)
  dense-grid fallback still needed:       1063/1084 (98.1%)
  of those, the fine (401-pt) grid found a
  feasible lambda the closed-form set missed:  2 cases
```

This 1.9% hit rate is essentially unchanged from the naive (pre-damping-fix)
2-term model's own 1.9% -- the damping fix genuinely closed a real physics
gap (confirmed above), but it did not meaningfully improve how often the
closed-form set alone suffices, because the actual bottleneck is a
different, structural limitation documented in `local_planner.py`'s own
docstring: the supremum of `m_phys(lambda) = min` over many per-(joint,step)
curves can occur at a pairwise CROSSING point between two different curves,
not at either curve's own extremum, and no per-curve closed form -- however
accurate -- captures that. With a 3-DOF arm and dozens of route steps, a
whole route contributes dozens of candidate curves, so this is the binding
constraint, not curve-level accuracy. Timing confirms this directly: the
closed-form path's own candidate-evaluation cost (computing `A,B,D` for
every step, then evaluating the real whole-route margin at every candidate)
is comparable to or larger than the 41-point dense grid's cost, so even
when it succeeds it is not meaningfully cheaper --

```
lambda_max-alone-feasible search: mean=21.9ms, p95=32.6ms, max=75.4ms
closed-form-sufficed search:      mean=96.2ms, p95=165.3ms, max=181.7ms
dense-grid-fallback search:       mean=107.5ms, p95=185.5ms, max=320.5ms
```

-- and on the paper's own timing-benchmark stress case (Exp 5, P_A), the
measured retime-only cost actually ROSE, from ~63ms (dense-grid-only
version above) to ~98ms (this version), since every search now pays the
closed-form computation's overhead before potentially still needing the
dense grid too. The dense-grid safety net was kept for exactly this reason
(it is not a leftover -- it is load-bearing) and is confirmed necessary in
practice, not just in theory: both cases above where the closed-form set
missed a feasible lambda were correctly rescued by the real implementation
(`_search_retime_whole_route`, which includes the safety net), returning
`lambda=1.219` and `lambda=1.526` respectively.

**What this is worth keeping for, honestly:** the closed-form path is
provably exact for whatever it directly checks (never a false positive,
same as the dense grid), is grounded in the actual simulated dynamics
rather than an idealized frictionless model (a real correctness fix in its
own right, independent of the speed question), and costs nothing extra
when `lambda_max` alone already clears `m_safe` (the majority, 63.9%, of
cases) since it is only invoked after that check fails. It is not,
however, the "recommended path" speedup `monotonicity_lemma_draft.md`
projected for a narrower, single-joint or few-step setting -- on this
benchmark's whole-route candidate counts, it is roughly cost-neutral to
mildly more expensive than the dense grid it was meant to shortcut. See
the paper's `predictive_realizability_paper_draft.md` Sec. VI and Draft
Status item 14, and `monotonicity_lemma_draft.md`'s own status note, for
where this is disclosed.

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
  failure at low force magnitudes (<=24N, matching B1/B2 there -- not a
  differentiator) and, from 25N on, B3 fails the task via a permanent hold
  (Level 4 has no resume path) even at magnitudes B1/B2 still complete despite
  transient saturation -- a real, honest safety-vs-completion trade-off, not
  a strict win for either side. (Re-verified directly during a full-codebase
  review: the earlier-reported "<=25N safe / past ~30N fails" thresholds were
  off by one sample point and overstated the safe margin by several N; the
  boundary is confirmed independent of the Level-2 route-level-reshape work
  above -- identical with Level 2 disabled entirely, so this predates that
  change rather than being caused by it.)
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
- **Exp 7 (environment-conditioned reroute):** P_A and P_B share q0/qf as in
  Exp 5, but here VIA_A's static margin under the (light, 1.0 kg) payload
  alone is healthy (**9.87 Nm**) -- the deficit exists only once the known
  contact-plane force is included (**-20.22 Nm**), and VIA_B's margin never
  needs the force accounted for since its path never enters the field
  (**8.63 Nm**, no force). The certificate's own bounded retiming search
  confirms this is retiming-proof, exactly as designed. B1 and B2 both attempt
  P_A, saturate through the contact region (peak torque ratio 1.00, 40 and 44
  saturation samples), and fail to reach the goal (**0.234 m** and **0.366 m**
  final error -- B2 slightly worse than B1 here, an unflattering but real
  finding reported rather than smoothed over, plausible since B2's one-step
  reactive projection has no lookahead into the field it's already
  entering). B3, given the same contact model at planning time
  (`force_known_at_plan_time=True`), predicts P_A's margin before ever
  entering the field, reroutes to P_B, and reaches the identical goal with
  **0.000 m error and zero saturation samples**, `replans=1`. This is the
  isolation the flagship result (Exp 5) does not by itself provide:
  rerouting driven by the environment specifically, not by configuration or
  payload.
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

  **Route-level reshape (below) adds a large one-time route-planning cost,
  paid regardless of whether it succeeds.** `plan_route` now tries reshape
  over the WHOLE route -- not just the online horizon -- before falling back
  to reroute, and this is a much bigger QP (n scales with route duration/dt,
  not the fixed online horizon). On the timing benchmark's stress case (Exp
  5's near-singular goal, no alt route, so the route decision runs retime
  then reshape then gives up): route planning went from **~3ms (retime
  only) to ~300ms** with reshape enabled, confirmed by a controlled
  before/after comparison, not inferred from noise. The cause: OSQP does not
  reliably converge on a QP this size within a practical iteration budget
  (hits `user_limit` even at `max_iter=50000` in direct testing), so
  `_try_reshape` falls back to SCS, which does converge cleanly but is
  itself not fast. (The retime-only baseline itself later moved to ~63ms
  on this same scenario, still far below reshape's cost, once
  `_search_retime_whole_route`'s own dense-grid fallback below was added --
  see "Non-monotonic retiming margin" further down -- since this scenario's
  lambda_max check alone fails and now correctly triggers a wider search
  instead of giving up immediately.) This cost is paid whether reshape
  ultimately succeeds or fails -- on Exp 5's flagship and Exp 7's
  environment-conditioned scenario (both of which still correctly require
  Level 3, confirmed against an independent solver, not just OSQP's status
  flag) it is pure overhead before the reroute decision; on scenarios where
  it succeeds (see the new
  `test_route_level_reshape_restores_feasibility_when_retiming_cannot`) it
  is the cost of the improvement. It remains a ONE-TIME, pre-execution cost
  (not compared against the 20ms per-cycle budget), but at FR3-scale route
  durations this could become the dominant term in "how long before the
  robot starts moving," which is worth disclosing plainly rather than
  leaving implicit in the per-cycle numbers above.

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
