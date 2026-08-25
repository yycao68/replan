"""Experiment 4 (Sec. VIII-E): an FR3-realizable substitute for the original
terrain scenario. At a KNOWN position along an otherwise fixed geometric
trajectory, the end-effector transitions from free space into contact with a
stiff virtual surface (a scripted penetration-proportional spring force via
xfrc_applied -- NOT MuJoCo's native contact solver; see the module docstring in
dynamics.py). Unlike Exp 3, this transition is treated as KNOWN in advance --
part of the 'predicted environment E' the paper's certificate is conditioned on
-- so B3 is allowed to see the whole contact model at planning time
(force_known_at_plan_time=True) and should anticipate/adapt (Level 1/2) before
the transition occurs, rather than discovering it only once in contact."""
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

Q0 = np.array([0.1, -0.7, -0.2])
QF = np.array([0.75, -0.35, 0.15])  # path descends toward the contact plane
T = 1.1
PAYLOAD = 1.0

Z_CONTACT = 0.72          # world z of the virtual contact plane
K_CONTACT = 900.0         # N/m, contact stiffness


def contact_force(t, q):
    """Position-based (not time-based) contact model: an upward restoring force
    once the end-effector penetrates below the contact plane."""
    ee_z = _arm_probe.ee_position(q)[1]
    penetration = Z_CONTACT - ee_z
    if penetration <= 0:
        return None
    return np.array([0.0, K_CONTACT * penetration])


_arm_probe = Arm.create()  # only used for the kinematics query inside contact_force


def run():
    traj0 = JointTrajectory(Q0, QF, T=T)
    z0 = _arm_probe.ee_position(Q0)[1]
    zf = _arm_probe.ee_position(QF)[1]
    print(f"EE height: start z={z0:.3f}, end z={zf:.3f}, contact plane z={Z_CONTACT:.3f}")

    results = {}
    for name in ["B1", "B2", "B3"]:
        arm = Arm.create()
        arm.set_payload_mass(PAYLOAD)
        traj = JointTrajectory(Q0, QF, T=T)
        cert = Certificate(arm=arm, m_safe=2.0)

        if name == "B1":
            pol = policy_b1(traj)
        elif name == "B2":
            pol = policy_b2(traj, arm, ee_force_schedule=contact_force)
        else:
            cfg = PlannerConfig(allow_level1=True, allow_level2=True, allow_level3=False)
            pol = policy_b3(traj, arm, cert, cfg, ee_force_schedule=contact_force,
                             force_known_at_plan_time=True)

        rr = rollout(arm, pol, Q0, np.zeros(3), duration=T + 0.3, dt=0.02,
                     ee_force_schedule=contact_force)
        goal_pos = arm.ee_position(QF)
        m = M.compute(rr, goal_pos)
        results[name] = m

        first_online_trigger = None
        for t, lv in zip(rr.t, rr.levels):
            if lv not in (0, None):
                first_online_trigger = t
                break
        pre_planned = name == "B3"  # B3's route-level decision, if any, happens at t=0

        print(f"{name}: success={m.task_success}, first_online_trigger={first_online_trigger}, "
              f"sat_events={m.saturation_events}, peak_tau_ratio={m.peak_torque_ratio:.2f}, "
              f"replans={m.replans}")

    return results


if __name__ == "__main__":
    run()
