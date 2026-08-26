"""Ablation batch A1-A5 (Sec. VIII-H), all sharing LocalPlanner's single code
path via PlannerConfig flags -- A2 is the one exception, since 'current-state
saturation only' isn't a LocalPlanner configuration at all, it's the separate
reactive mechanism in baselines.policy_b2 (Sec. VIII-A's B2), so A2 is run as
policy_b2 directly rather than through the planner.

  A1 -- no predictive feedback at all                  (== B1, policy_b1)
  A2 -- current-state saturation only, no prediction    (== B2, policy_b2)
  A3 -- predicts, but the response is never acted on    (predict=True, act=False)
  A4 -- predicts + retimes/reshapes, but never reroutes
        or brakes                                       (allow_level3=False, allow_level4=False)
  A5 -- the full architecture                            (== B3, all levels on)

Run across two scenarios chosen because they're known (from Exp2 and Exp5) to
actually need different levels of the hierarchy to succeed, so the ablation
differences are real rather than vacuous:
  - "retime-suffices": Exp2's payload=2.4 point, where Level 1 alone is enough
    (see scenario_retime_suffices' docstring below for why not 3.0 -- an
    earlier draft used that value and it was wrong).
  - "reroute-required": Exp5's flagship P_A/P_B scenario, where only Level 3
    (rerouting) succeeds -- this is the scenario the paper's own Sec. VIII-H
    text singles out as where A4 (no rerouting) should show degraded
    performance relative to A5.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from dynamics import Arm
from certificate import Certificate
from trajectory import JointTrajectory, ViaPointTrajectory
from local_planner import LocalPlanner, PlannerConfig
from baselines import policy_b1, policy_b2, policy_b3
from executor import rollout
import metrics as M

ABLATION_CFGS = {
    "A1": None,  # handled specially: policy_b1
    "A2": None,  # handled specially: policy_b2
    "A3": PlannerConfig(predict=True, act=False),
    "A4": PlannerConfig(predict=True, act=True, allow_level1=True, allow_level2=True,
                         allow_level3=False, allow_level4=False),
    "A5": PlannerConfig(predict=True, act=True, allow_level1=True, allow_level2=True,
                         allow_level3=True, allow_level4=True),
}


def run_one(name, arm, traj, cert, goal_pos, alt_traj=None, duration=None):
    if name == "A1":
        pol = policy_b1(traj)
    elif name == "A2":
        pol = policy_b2(traj, arm)
    else:
        pol = policy_b3(traj, arm, cert, ABLATION_CFGS[name], alt_traj=alt_traj)
    rr = rollout(arm, pol, traj.q0, np.zeros(3), duration=duration, dt=0.02)
    m = M.compute(rr, goal_pos)
    levels = sorted(set(str(l) for l in rr.levels))
    return m, levels


def scenario_retime_suffices():
    """Exp2's payload=2.4 point -- the actual task-success crossover in that
    sweep (not 3.0kg, an earlier draft of this script/README got that wrong;
    by 3.0kg every baseline, including the full architecture, already fails).
    B1 (no prediction) fails here; B2 (reactive) and B3/A5 (predictive+act,
    via Level-1 retiming) both succeed -- this scenario does NOT differentiate
    reactive from predictive handling, which is itself worth knowing rather
    than papering over."""
    Q0 = np.array([0.0, -0.5, -0.3]); QF = np.array([1.1, -1.0, 0.6]); T = 0.55
    PAYLOAD = 2.4
    print(f"\n--- Scenario: retime-suffices (payload={PAYLOAD} kg, T={T}s) ---")
    for name in ABLATION_CFGS:
        arm = Arm.create(); arm.set_payload_mass(PAYLOAD)
        cert = Certificate(arm=arm, m_safe=2.0)
        traj = JointTrajectory(Q0, QF, T=T)
        goal = arm.ee_position(QF)
        m, levels = run_one(name, arm, traj, cert, goal, duration=T + 0.3)
        print(f"{name}: success={m.task_success}, levels={levels}, "
              f"sat_events={m.saturation_events}, peak_tau_ratio={m.peak_torque_ratio:.2f}")


def scenario_reroute_required():
    """Exp5's flagship: P_A (via an outstretched via-point, heavy payload) is
    only rescuable by rerouting to P_B -- SAME start, SAME goal, different
    via-point (see exp5_flagship_reroute's docstring for why this must be a
    same-goal route change, not two different final configurations, and
    trajectory.ViaPointTrajectory for the representation that makes that
    possible). A4 (no reroute) should fail here even though A5 succeeds."""
    Q0 = np.array([0.2, -1.0, -0.6])
    QG = np.array([0.75, -1.15, -0.55])         # shared goal for every ablation
    VIA_A = np.array([1.1, -0.15, 0.1])         # P_A's via-point: outstretched, risky
    VIA_B = 0.5 * (Q0 + QG)                     # P_B's via-point: safe midpoint
    T1_A, T2_A = 0.5, 0.4
    T1_B, T2_B = 0.7, 0.6
    PAYLOAD = 4.5
    print(f"\n--- Scenario: reroute-required (flagship P_A/P_B, payload={PAYLOAD} kg) ---")
    for name in ABLATION_CFGS:
        arm = Arm.create(); arm.set_payload_mass(PAYLOAD)
        cert = Certificate(arm=arm, m_safe=2.0)
        traj_A = ViaPointTrajectory(Q0, VIA_A, QG, T1=T1_A, T2=T2_A)
        traj_B = ViaPointTrajectory(Q0, VIA_B, QG, T1=T1_B, T2=T2_B)
        assert np.allclose(traj_A.qf, traj_B.qf), "P_A and P_B must share the same goal"
        alt = traj_B if name in ("A4", "A5") else None  # A1/A2/A3 have no route
                                                          # switching in their
                                                          # own definitions
        goal_pos = arm.ee_position(QG)

        if name == "A1":
            pol = policy_b1(traj_A)
            duration = traj_A.T + 0.3
        elif name == "A2":
            pol = policy_b2(traj_A, arm)
            duration = traj_A.T + 0.3
        else:
            pol = policy_b3(traj_A, arm, cert, ABLATION_CFGS[name], alt_traj=alt)
            # Size duration to what plan_route ACTUALLY selects (peeked at
            # directly) rather than a flat worst-case retiming budget -- see
            # exp5_flagship_reroute's comment at the same call for why a flat
            # cfg.lam_max budget is wrong (it surfaces an unrelated long-
            # duration tracking instability on routes that were never retimed).
            route = LocalPlanner(arm, cert, ABLATION_CFGS[name]).plan_route(
                traj_A, alt_traj=alt)
            duration = route.traj.T + 0.3
        rr = rollout(arm, pol, Q0, np.zeros(3), duration=duration, dt=0.02)
        m = M.compute(rr, goal_pos)
        levels = sorted(set(str(l) for l in rr.levels))
        print(f"{name}: success={m.task_success} (final_pos_error={m.final_pos_error_m:.3f} m), "
              f"levels={levels}, sat_events={m.saturation_events}, "
              f"peak_tau_ratio={m.peak_torque_ratio:.2f}")


def run():
    scenario_retime_suffices()
    scenario_reroute_required()


if __name__ == "__main__":
    run()
