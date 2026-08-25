"""Experiment 2 (Sec. VIII-C): payload sweep on an IDENTICAL geometric trajectory
and time law. Isolates the paper's central claim -- identical geometry does not
imply identical physical feasibility -- by reporting the payload level at which
each baseline first violates a torque limit (ground truth) vs. the payload level
at which B3's certificate first triggers adaptation."""
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

Q0 = np.array([0.0, -0.5, -0.3])
QF = np.array([1.1, -1.0, 0.6])
T = 0.55  # fast enough that heavy payloads actually saturate
PAYLOADS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]


def run():
    print(f"{'payload':>8} | {'B1 sat?':>8} {'B1 succ':>8} | {'B2 sat?':>8} {'B2 succ':>8} | "
          f"{'B3 lvl':>7} {'B3 sat?':>8} {'B3 succ':>8} {'m_phys0':>9}")
    rows = []
    first_violation = {"B1": None, "B2": None}
    first_b3_trigger = None
    n_triggers = 0
    n_conservative = 0
    for payload in PAYLOADS:
        row = {"payload": payload}
        levels_seen = set()
        m0_min = None
        for name in ["B1", "B2", "B3"]:
            arm = Arm.create()
            arm.set_payload_mass(payload)
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
            row[name] = m
            saturated = m.saturation_events > 0
            if saturated and first_violation.get(name) is None and name in ("B1", "B2"):
                first_violation[name] = payload
            if name == "B3":
                levels_seen = set(str(l) for l in rr.levels)
                triggered = any(l not in ("0",) for l in levels_seen)
                if triggered and first_b3_trigger is None:
                    first_b3_trigger = payload
                m0_min = min(v for v in rr.m_phys_trace if v is not None) if any(
                    v is not None for v in rr.m_phys_trace) else float("nan")
                if triggered:
                    n_triggers += 1
                    # Conservatism check (Sec. VIII-I): would the ORIGINAL, un-
                    # adapted nominal trajectory actually have violated a torque
                    # limit? If not, this trigger was a false positive.
                    n_full = int(np.ceil(traj.T / 0.02)) + 1
                    Qn, Qdn, Qddn = traj.sample_horizon(0.0, 0.02, n_full)
                    if not cert.ground_truth_violation(Qn, Qdn, Qddn):
                        n_conservative += 1
        b1, b2, b3 = row["B1"], row["B2"], row["B3"]
        levels_str = ",".join(sorted(levels_seen))
        print(f"{payload:8.1f} | {str(b1.saturation_events>0):>8} {str(b1.task_success):>8} | "
              f"{str(b2.saturation_events>0):>8} {str(b2.task_success):>8} | "
              f"{levels_str:>7} {str(b3.saturation_events>0):>8} {str(b3.task_success):>8} "
              f"{m0_min:9.2f}")
        rows.append(row)

    print(f"\nFirst payload level with an observed torque-limit violation: "
          f"B1={first_violation['B1']}, B2={first_violation['B2']}")
    print(f"First payload level at which B3's certificate triggers Level>=1 adaptation: "
          f"{first_b3_trigger}")
    if first_violation["B1"] is not None and first_b3_trigger is not None:
        lead = first_violation["B1"] - first_b3_trigger
        print(f"Certificate lead over B1 (payload units, kg): {lead:.1f}")
    conservatism = M.conservatism(n_conservative, n_triggers)
    print(f"Conservatism (Sec. VIII-I): {n_conservative}/{n_triggers} triggers fired despite the "
          f"nominal trajectory not actually violating a limit -> {conservatism:.2f}")
    return rows


if __name__ == "__main__":
    run()
