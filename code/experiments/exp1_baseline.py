"""Experiment 1 (Sec. VIII-B): baseline manipulation trajectory, no payload change,
no disturbance. Sanity check that B3 does not regress ordinary-case performance
relative to B1/B2."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from dynamics import Arm
from certificate import Certificate
from trajectory import JointTrajectory
from local_planner import PlannerConfig
from baselines import policy_b1, policy_b2, policy_b3
from executor import rollout
import metrics as M

Q0 = np.array([0.0, -0.6, -0.2])
QF = np.array([0.9, -0.9, 0.5])
T = 1.2
PAYLOAD = 0.5


def run():
    results = {}
    for name in ["B1", "B2", "B3"]:
        arm = Arm.create()
        arm.set_payload_mass(PAYLOAD)
        traj = JointTrajectory(Q0, QF, T=T)
        cert = Certificate(arm=arm, m_safe=2.0)

        if name == "B1":
            pol = policy_b1(traj)
        elif name == "B2":
            pol = policy_b2(traj, arm)
        else:
            pol = policy_b3(traj, arm, cert, PlannerConfig())

        rr = rollout(arm, pol, Q0, np.zeros(3), duration=T + 0.3, dt=0.02)
        goal_pos = arm.ee_position(QF)
        m = M.compute(rr, goal_pos)
        results[name] = m
        print(f"{name}: success={m.task_success}, final_err={m.final_pos_error_m*1000:.2f}mm, "
              f"peak_tau_ratio={m.peak_torque_ratio:.2f}, sat_events={m.saturation_events}, "
              f"track_rms={m.tracking_error_rms_m*1000:.2f}mm, replans={m.replans}")
    return results


if __name__ == "__main__":
    run()
