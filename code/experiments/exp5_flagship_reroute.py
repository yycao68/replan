"""Experiment 5 (Sec. VIII-F, flagship): P_A reaches for a target via a nearly
fully-outstretched pose (large gravity lever arm under payload -- a persistent,
speed-independent torque deficit, the planar-arm analogue of the paper's
'near-singular configuration under payload' framing); P_B reaches a nearby target
via a tucked-elbow pose that keeps the static torque demand low. A conventional
planner (B1/B2) would attempt P_A on distance/path-length grounds alone and only
discover the deficit during execution; B3's certificate should predict that
Level 1/2 cannot fix P_A (the deficit is static, not dynamic -- Theorem 3's
"T_dyn(p) = empty" case) and reroute to P_B before execution begins."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from dynamics import Arm, TAU_MAX
from certificate import Certificate
from trajectory import JointTrajectory
from local_planner import PlannerConfig
from baselines import policy_b1, policy_b2, policy_b3
from executor import rollout
import metrics as M

Q0 = np.array([0.2, -1.0, -0.6])          # tucked start
QF_A = np.array([1.1, -0.15, 0.1])        # outstretched: P_A ("shorter", risky)
QF_B = np.array([0.75, -1.15, -0.55])     # stays tucked: P_B ("longer", safe)
T_A = 0.7
T_B = 0.9
PAYLOAD = 4.5


def static_margin(arm, q):
    tau = arm.required_torque(q, np.zeros(3), np.zeros(3))
    return (TAU_MAX - np.abs(tau)).min()


def run():
    arm_probe = Arm.create()
    arm_probe.set_payload_mass(PAYLOAD)
    print(f"Static margin at Q0: {static_margin(arm_probe, Q0):.2f} Nm")
    print(f"Static margin at QF_A (outstretched): {static_margin(arm_probe, QF_A):.2f} Nm")
    print(f"Static margin at QF_B (tucked):        {static_margin(arm_probe, QF_B):.2f} Nm")

    traj_A = JointTrajectory(Q0, QF_A, T=T_A)
    traj_B = JointTrajectory(Q0, QF_B, T=T_B)

    results = {}
    for name in ["B1", "B2", "B3"]:
        arm = Arm.create()
        arm.set_payload_mass(PAYLOAD)
        cert = Certificate(arm=arm, m_safe=2.0)

        if name == "B1":
            pol = policy_b1(traj_A)
            goal = QF_A
        elif name == "B2":
            pol = policy_b2(traj_A, arm)
            goal = QF_A
        else:
            cfg = PlannerConfig(allow_level1=True, allow_level2=True, allow_level3=True)
            pol = policy_b3(traj_A, arm, cert, cfg, alt_traj=traj_B)
            goal = QF_A  # metrics below check against whichever goal was actually used

        rr = rollout(arm, pol, Q0, np.zeros(3), duration=max(T_A, T_B) + 0.3, dt=0.02)
        goal_pos_A = arm.ee_position(QF_A)
        goal_pos_B = arm.ee_position(QF_B)
        err_to_A = np.linalg.norm(rr.ee_positions[-1] - goal_pos_A)
        err_to_B = np.linalg.norm(rr.ee_positions[-1] - goal_pos_B)
        reached_goal = goal_pos_A if err_to_A < err_to_B else goal_pos_B
        m = M.compute(rr, reached_goal)
        results[name] = m
        levels = sorted(set(str(l) for l in rr.levels))
        print(f"{name}: success={m.task_success} (reached {'A' if err_to_A<err_to_B else 'B'}), "
              f"levels={levels}, sat_events={m.saturation_events}, "
              f"peak_tau_ratio={m.peak_torque_ratio:.2f}, replans={m.replans}")
    return results


if __name__ == "__main__":
    run()
