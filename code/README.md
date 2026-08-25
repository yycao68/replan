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
  (`mj_inverse`, `mj_fullM`). No hand-derived symbolic dynamics anywhere.
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

`experiments/exp1_baseline.py`, `exp2_payload_sweep.py`,
`exp5_flagship_reroute.py`, `exp6_severity_sweep.py` are fully implemented and
runnable, with results below from an actual run (not fabricated -- rerun
`run_all.py` to reproduce). **`exp3_interaction_force.py` and
`exp4_contact_stiffness_step.py` are NOT implemented yet** -- the `xfrc_applied`
mechanism they'd need (external end-effector force) is already wired into
`dynamics.py` and `executor.py`'s `ee_force_schedule` argument, but neither
experiment script has been written or run. The ablation sweep (A1-A5, Sec.
VIII-H) is likewise just `PlannerConfig` flag combinations away but has not been
scripted as a batch. Real-time per-cycle timing (Sec. VIII-J) has not been
measured.

## Real results from the current implementation

- **Exp 1 (no-regression check):** B1/B2/B3 are bit-for-bit identical on a benign
  trajectory (0.3 mm final error, no saturation) -- no regression from adding the
  predictive layer when it isn't needed.
- **Exp 2 (payload sweep, same geometry+timing):** B1/B2 first saturate at 2.0 kg
  payload and fail the task outright by 3.0 kg. B3 also shows its first
  saturation event at 2.0 kg but, via Level-1 retiming, still **completes the
  task successfully at 3.0 kg where B1/B2 both fail** -- a genuine, task-level
  win, not just a torque-margin number. Conservatism over this sweep: 0/9
  triggers were false positives (every trigger corresponded to a real violation
  of the unmodified nominal trajectory).
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

## Known simplifications (stated once, applies throughout)

Reduced-order 3-DOF planar arm, not a full FR3 model. Level-1/3 decided once at
planning time rather than continuously re-optimized (see `local_planner.py`
docstring). `delta_tau` (Theorem 1's uncertainty bound) is a fixed 5% of
`tau_max`, not estimated online. No collision/obstacle feasibility set is
modeled (`F_obs` from the paper's Sec. III is out of scope here; only `F_dyn` is
tested). Theorem 4 (recursive feasibility) is not attempted, consistent with the
paper draft's own "deferred" framing.
