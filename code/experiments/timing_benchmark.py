"""Real-time performance (Sec. VIII-J): local-planner per-cycle computation time,
measured directly (not assumed) for B2 and B3, under both Exp 1's nominal
trajectory and Exp 5's near-singular-configuration stress case (P_A), reporting
mean/p95/max against the local planner's target cycle rate. Whether the added
certificate evaluation fits the real-time budget at the stress case specifically
-- not only the nominal case -- is treated as an open empirical question this
benchmark answers, not a design assumption (per the paper's own framing).

B1 is excluded: by definition (Sec. VIII-A) it performs no per-cycle computation
of its own -- it always samples the nominal trajectory directly -- so there is
nothing to time.

For B3, two distinct costs are reported separately, since only one of them is
actually subject to the real-time budget:
  - "route planning" (Level 1/3): a ONE-TIME cost paid once per rollout, when
    plan_route searches the whole candidate route(s) (Sec. V-B/V-C's design:
    this is a planning-time decision, not a per-cycle one -- see
    local_planner.py's module docstring). Not compared against the cycle budget.
  - "online step" (Level 0/2/4): the PER-CYCLE cost paid every control tick,
    which must fit within dt=1/50Hz=20ms for the architecture to be real-time
    feasible. This is the number Sec. VIII-J is actually asking about.

Wall-clock latency on a shared, non-realtime dev machine is noisy: rerunning the
IDENTICAL deterministic stress-case scenario back-to-back was observed, while
building this benchmark, to show a per-cycle max anywhere from ~17ms to ~57ms
(OS scheduling/thermal/GC jitter, not algorithmic nondeterminism -- the sequence
of Level 0/2/4 decisions is exactly the same every time). A single rollout's max
is therefore not trustworthy on its own, so every scenario below is repeated
several times and both the pooled statistics and the per-repetition maxima are
reported, so the spread itself is visible rather than hidden behind one number.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from dynamics import Arm
from certificate import Certificate
from trajectory import JointTrajectory
from local_planner import PlannerConfig
from baselines import policy_b2, policy_b3
from executor import rollout

DT = 0.02
BUDGET_MS = DT * 1000  # 20 ms at 50 Hz


def _time_policy(policy_fn):
    """Wrap a policy(t, q, qdot) -> (...) callable, recording each call's
    wall-clock cost (time.perf_counter, seconds) into the returned list."""
    times = []
    def wrapped(t, q, qdot):
        t0 = time.perf_counter()
        result = policy_fn(t, q, qdot)
        times.append(time.perf_counter() - t0)
        return result
    return wrapped, times


def _stats_ms(times_s):
    arr = np.array(times_s) * 1000.0
    return {
        "n": len(arr),
        "mean_ms": float(arr.mean()),
        "p95_ms": float(np.percentile(arr, 95)),
        "max_ms": float(arr.max()),
    }


def _timed_repeats(make_arm_and_policy, q0, qdot0, duration, dt, repeats):
    """Run `repeats` independent rollouts of `make_arm_and_policy() -> (arm,
    policy)`, returning (pooled_times_s, per_rep_max_ms)."""
    pooled, per_rep_max = [], []
    for _ in range(repeats):
        arm, pol = make_arm_and_policy()
        timed_pol, times = _time_policy(pol)
        rollout(arm, timed_pol, q0, qdot0, duration=duration, dt=dt)
        pooled.extend(times)
        per_rep_max.append(max(times) * 1000.0)
    return pooled, per_rep_max


def _report(label, stats, per_rep_max_ms):
    fits = "FITS" if stats["max_ms"] < BUDGET_MS else "EXCEEDS"
    spread = ", ".join(f"{m:.2f}" for m in per_rep_max_ms)
    print(f"{label}: n={stats['n']} (pooled over {len(per_rep_max_ms)} repeats), "
          f"mean={stats['mean_ms']:.3f}ms, p95={stats['p95_ms']:.3f}ms, "
          f"max={stats['max_ms']:.3f}ms -> {fits} the {BUDGET_MS:.1f}ms (50 Hz) "
          f"cycle budget (max uses {100*stats['max_ms']/BUDGET_MS:.1f}% of budget)")
    print(f"  per-repetition max (ms): [{spread}]")


def _run_scenario(scenario_name, Q0, QF, T, payload, alt_traj_fn=None, repeats=5):
    print(f"\n--- Scenario: {scenario_name} (payload={payload} kg, T={T}s, "
          f"{repeats} repeats) ---")
    duration = T + 0.3

    # B2: reactive, current-state-only -- one required_torque call per cycle.
    def make_b2():
        arm = Arm.create(); arm.set_payload_mass(payload)
        traj = JointTrajectory(Q0, QF, T=T)
        return arm, policy_b2(traj, arm)
    times2, per_rep2 = _timed_repeats(make_b2, Q0, np.zeros(3), duration, DT, repeats)
    _report("B2 online step ", _stats_ms(times2), per_rep2)

    # B3: full predictive architecture. Route planning (one-time, Level 1/3) is
    # timed on a single instance; the online per-cycle step (Level 0/2/4) is
    # pooled over repeats like B2, since that's the number subject to noise.
    arm3 = Arm.create(); arm3.set_payload_mass(payload)
    traj3 = JointTrajectory(Q0, QF, T=T)
    cert3 = Certificate(arm=arm3, m_safe=2.0)
    alt_traj = alt_traj_fn() if alt_traj_fn else None
    t0 = time.perf_counter()
    policy_b3(traj3, arm3, cert3, PlannerConfig(), alt_traj=alt_traj)
    route_planning_s = time.perf_counter() - t0
    print(f"B3 route planning (one-time, Level 1/3): {route_planning_s*1000:.3f}ms "
          f"(not compared against the per-cycle budget)")

    def make_b3():
        arm = Arm.create(); arm.set_payload_mass(payload)
        traj = JointTrajectory(Q0, QF, T=T)
        cert = Certificate(arm=arm, m_safe=2.0)
        alt = alt_traj_fn() if alt_traj_fn else None
        return arm, policy_b3(traj, arm, cert, PlannerConfig(), alt_traj=alt)
    times3, per_rep3 = _timed_repeats(make_b3, Q0, np.zeros(3), duration, DT, repeats)
    stats_with = _stats_ms(times3)
    _report("B3 online step ", stats_with, per_rep3)

    # Diagnostic: if the online step is expensive, is it the Level-2 QP
    # (cvxpy/OSQP problem construction+solve, rebuilt from scratch every call)?
    # Rerun (once -- this is a diagnostic, not the headline number) with
    # Level 2 disabled to isolate its contribution. Gated on MAX, not mean:
    # since Level 4 is sticky (baselines.py), a scenario that hits Level 4
    # typically pays the expensive QP on only one cycle per rollout, which
    # dilutes the mean below any reasonable fixed threshold (observed
    # directly: mean fluctuated either side of a mean>1.0 gate run to run,
    # depending on how many of the `repeats` rollouts happened to need the
    # QP at all, making the diagnostic silently vanish on some runs even
    # though the underlying per-solve cost was unchanged) -- max instead
    # reliably reflects "did an expensive call happen at all."
    if stats_with["max_ms"] > 1.0:
        def make_b3_no_l2():
            arm = Arm.create(); arm.set_payload_mass(payload)
            traj = JointTrajectory(Q0, QF, T=T)
            cert = Certificate(arm=arm, m_safe=2.0)
            alt = alt_traj_fn() if alt_traj_fn else None
            return arm, policy_b3(traj, arm, cert, PlannerConfig(allow_level2=False), alt_traj=alt)
        times3b, _ = _timed_repeats(make_b3_no_l2, Q0, np.zeros(3), duration, DT, repeats=1)
        stats_without = _stats_ms(times3b)
        print(f"  diagnostic: with Level-2 QP mean={stats_with['mean_ms']:.3f}ms vs. "
              f"without mean={stats_without['mean_ms']:.3f}ms -- the Level-2 QP "
              f"(cvxpy/OSQP, rebuilt from scratch every cycle) accounts for "
              f"~{100*(1 - stats_without['mean_ms']/stats_with['mean_ms']):.0f}% "
              f"of the online-step cost here.")


def run():
    # Exp 1's nominal trajectory (benign case).
    _run_scenario(
        "nominal (Exp 1)",
        Q0=np.array([0.0, -0.6, -0.2]), QF=np.array([0.9, -0.9, 0.5]),
        T=1.2, payload=0.5, repeats=3,
    )

    # Exp 5's near-singular-configuration stress case (P_A alone, no alt route
    # -- isolates the per-cycle cost under the hardest trajectory, not the
    # one-time reroute decision, which Exp 5 itself already covers).
    _run_scenario(
        "stress (Exp 5, P_A)",
        Q0=np.array([0.2, -1.0, -0.6]), QF=np.array([1.1, -0.15, 0.1]),
        T=0.7, payload=4.5, repeats=5,
    )


if __name__ == "__main__":
    run()
