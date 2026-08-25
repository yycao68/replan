"""Experiment 6 (Sec. VIII-G): a single scenario family swept by payload severity,
checking which hierarchy level (0/1/2/3/4) actually resolves each severity, so the
whole Level 0-4 hierarchy is exercised, not just its endpoints (Exp1, Exp5)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from dynamics import Arm
from certificate import Certificate
from trajectory import JointTrajectory
from local_planner import PlannerConfig
from baselines import policy_b3
from executor import rollout
import metrics as M

Q0 = np.array([0.1, -0.6, -0.3])
QF = np.array([1.0, -0.9, 0.5])
QF_SAFE = np.array([0.3, -1.1, -0.4])   # fallback target for Level-3 reroute case
T = 0.6
SEVERITIES = [0.0, 1.5, 2.5, 3.5, 5.0, 8.0]


def run():
    print(f"{'payload':>8} | {'level(s) used':>18} | {'success':>8} {'sat_events':>10} {'replans':>8}")
    for payload in SEVERITIES:
        arm = Arm.create()
        arm.set_payload_mass(payload)
        cert = Certificate(arm=arm, m_safe=2.0)
        traj = JointTrajectory(Q0, QF, T=T)
        alt = JointTrajectory(Q0, QF_SAFE, T=1.2)
        cfg = PlannerConfig(allow_level1=True, allow_level2=True, allow_level3=True)
        pol = policy_b3(traj, arm, cert, cfg, alt_traj=alt)
        rr = rollout(arm, pol, Q0, np.zeros(3), duration=max(T, 1.2) + 0.3, dt=0.02)

        goal_A = arm.ee_position(QF)
        goal_B = arm.ee_position(QF_SAFE)
        err_A = np.linalg.norm(rr.ee_positions[-1] - goal_A)
        err_B = np.linalg.norm(rr.ee_positions[-1] - goal_B)
        m = M.compute(rr, goal_A if err_A < err_B else goal_B)
        levels = sorted(set(str(l) for l in rr.levels))
        print(f"{payload:8.1f} | {str(levels):>18} | {str(m.task_success):>8} "
              f"{m.saturation_events:10d} {m.replans:8d}")


if __name__ == "__main__":
    run()
