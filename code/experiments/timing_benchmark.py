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


def _report(label, stats):
    fits = "FITS" if stats["max_ms"] < BUDGET_MS else "EXCEEDS"
    print(f"{label}: n={stats['n']}, mean={stats['mean_ms']:.3f}ms, "
          f"p95={stats['p95_ms']:.3f}ms, max={stats['max_ms']:.3f}ms "
          f"-> {fits} the {BUDGET_MS:.1f}ms (50 Hz) cycle budget "
          f"(max uses {100*stats['max_ms']/BUDGET_MS:.1f}% of budget)")


def _run_scenario(scenario_name, Q0, QF, T, payload, alt_traj_fn=None):
    print(f"\n--- Scenario: {scenario_name} (payload={payload} kg, T={T}s) ---")

    # B2: reactive, current-state-only -- one required_torque call per cycle.
    arm2 = Arm.create(); arm2.set_payload_mass(payload)
    traj2 = JointTrajectory(Q0, QF, T=T)
    pol2 = policy_b2(traj2, arm2)
    timed_pol2, times2 = _time_policy(pol2)
    rollout(arm2, timed_pol2, Q0, np.zeros(3), duration=T + 0.3, dt=DT)
    _report("B2 online step ", _stats_ms(times2))

    # B3: full predictive architecture. Time route planning (one-time,
    # Level 1/3) and the online per-cycle step (Level 0/2/4) separately.
    arm3 = Arm.create(); arm3.set_payload_mass(payload)
    traj3 = JointTrajectory(Q0, QF, T=T)
    cert3 = Certificate(arm=arm3, m_safe=2.0)
    alt_traj = alt_traj_fn() if alt_traj_fn else None
    cfg = PlannerConfig()

    t0 = time.perf_counter()
    pol3 = policy_b3(traj3, arm3, cert3, cfg, alt_traj=alt_traj)
    route_planning_s = time.perf_counter() - t0
    print(f"B3 route planning (one-time, Level 1/3): {route_planning_s*1000:.3f}ms "
          f"(not compared against the per-cycle budget)")

    timed_pol3, times3 = _time_policy(pol3)
    rollout(arm3, timed_pol3, Q0, np.zeros(3), duration=T + 0.3, dt=DT)
    _report("B3 online step ", _stats_ms(times3))

    # Diagnostic: if the online step is expensive, is it the Level-2 QP
    # (cvxpy/OSQP problem construction+solve, rebuilt from scratch every call)?
    # Rerun with Level 2 disabled to isolate its contribution.
    stats_with = _stats_ms(times3)
    if stats_with["mean_ms"] > 1.0:
        arm3b = Arm.create(); arm3b.set_payload_mass(payload)
        traj3b = JointTrajectory(Q0, QF, T=T)
        cert3b = Certificate(arm=arm3b, m_safe=2.0)
        cfg_no_l2 = PlannerConfig(allow_level2=False)
        pol3b = policy_b3(traj3b, arm3b, cert3b, cfg_no_l2, alt_traj=alt_traj)
        timed_pol3b, times3b = _time_policy(pol3b)
        rollout(arm3b, timed_pol3b, Q0, np.zeros(3), duration=T + 0.3, dt=DT)
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
        T=1.2, payload=0.5,
    )

    # Exp 5's near-singular-configuration stress case (P_A alone, no alt route
    # -- isolates the per-cycle cost under the hardest trajectory, not the
    # one-time reroute decision, which Exp 5 itself already covers).
    _run_scenario(
        "stress (Exp 5, P_A)",
        Q0=np.array([0.2, -1.0, -0.6]), QF=np.array([1.1, -0.15, 0.1]),
        T=0.7, payload=4.5,
    )


if __name__ == "__main__":
    run()
