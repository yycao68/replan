"""Experiment 5 (Sec. VIII-F, flagship): two candidate ROUTES to the SAME start
and the SAME goal. P_A passes through a nearly fully-outstretched via-point
(large gravity lever arm under payload -- a persistent, speed-independent
torque deficit, the planar-arm analogue of the paper's 'near-singular
configuration under payload' framing) before continuing on to the goal; P_B
reaches the identical goal via a tucked-elbow via-point that keeps the static
torque demand low. A conventional planner (B1/B2) would attempt P_A on
distance/path-length grounds alone and only discover the deficit during
execution; B3's certificate should predict that Level 1/2 cannot fix P_A (the
deficit is static, not dynamic -- Theorem 3's "T_dyn(p) = empty" case) and
reroute to P_B before execution begins -- while still reaching the SAME task
goal, not a different, easier one.

(This corrects an earlier version of this experiment, where P_A and P_B ended
at two DIFFERENT final configurations: that made B3's "reroute" indistinguishable
from simply abandoning the task for an easier one, which is not the paper's
claim. ViaPointTrajectory (trajectory.py) exists specifically so P_A and P_B
can share q0 and qf while differing in the path between them.)"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from dynamics import Arm, TAU_MAX
from certificate import Certificate
from trajectory import ViaPointTrajectory
from local_planner import LocalPlanner, PlannerConfig
from baselines import policy_b1, policy_b2, policy_b3
from executor import rollout
import metrics as M

Q0 = np.array([0.2, -1.0, -0.6])            # shared start, tucked
QG = np.array([0.75, -1.15, -0.55])         # shared goal, tucked (both routes end here)
VIA_A = np.array([1.1, -0.15, 0.1])         # P_A's via-point: outstretched, risky
VIA_B = 0.5 * (Q0 + QG)                     # P_B's via-point: straight-line midpoint, safe
T1_A, T2_A = 0.5, 0.4                       # P_A: fast, passes through the risky region
T1_B, T2_B = 0.7, 0.6                       # P_B: slower, stays clear of it
PAYLOAD = 4.5


def static_margin(arm, q):
    tau = arm.required_torque(q, np.zeros(3), np.zeros(3))
    return (TAU_MAX - np.abs(tau)).min()


def run():
    arm_probe = Arm.create()
    arm_probe.set_payload_mass(PAYLOAD)
    print(f"Static margin at Q0:            {static_margin(arm_probe, Q0):.2f} Nm")
    print(f"Static margin at QG (shared goal): {static_margin(arm_probe, QG):.2f} Nm")
    print(f"Static margin at VIA_A (P_A, outstretched): {static_margin(arm_probe, VIA_A):.2f} Nm")
    print(f"Static margin at VIA_B (P_B, safe):          {static_margin(arm_probe, VIA_B):.2f} Nm")

    traj_A = ViaPointTrajectory(Q0, VIA_A, QG, T1=T1_A, T2=T2_A)
    traj_B = ViaPointTrajectory(Q0, VIA_B, QG, T1=T1_B, T2=T2_B)
    assert np.allclose(traj_A.qf, traj_B.qf), "P_A and P_B must share the same goal"

    goal_pos = Arm.create().ee_position(QG)   # the ONE goal every policy is judged against

    results = {}
    for name in ["B1", "B2", "B3"]:
        arm = Arm.create()
        arm.set_payload_mass(PAYLOAD)
        cert = Certificate(arm=arm, m_safe=2.0)

        if name == "B1":
            pol = policy_b1(traj_A)
            duration = traj_A.T + 0.3
        elif name == "B2":
            pol = policy_b2(traj_A, arm)
            duration = traj_A.T + 0.3
        else:
            cfg = PlannerConfig(allow_level1=True, allow_level2=True, allow_level3=True)
            pol = policy_b3(traj_A, arm, cert, cfg, alt_traj=traj_B)
            # Level 1 can retime the selected route -- policy_b3 makes that
            # decision internally (deterministically, from arm/cert/cfg/traj,
            # the same inputs available here) but doesn't expose it, so peek
            # at it directly rather than guessing a worst-case duration: a
            # flat cfg.lam_max budget was tried first and found to be too
            # generous when retiming ISN'T selected -- it ran the (already-
            # finished, holding-position) rollout for several extra seconds
            # and surfaced an unrelated long-duration tracking instability
            # having nothing to do with this experiment. Sizing duration to
            # what was ACTUALLY selected avoids both problems.
            route = LocalPlanner(arm, cert, cfg).plan_route(traj_A, alt_traj=traj_B)
            duration = route.traj.T + 0.3
        rr = rollout(arm, pol, Q0, np.zeros(3), duration=duration, dt=0.02)
        m = M.compute(rr, goal_pos)
        results[name] = m
        levels = sorted(set(str(l) for l in rr.levels))
        print(f"{name}: success={m.task_success} (final_pos_error={m.final_pos_error_m:.3f} m), "
              f"levels={levels}, sat_events={m.saturation_events}, "
              f"peak_tau_ratio={m.peak_torque_ratio:.2f}, replans={m.replans}")
    return results


if __name__ == "__main__":
    run()
