"""Experiment 3 (Sec. VIII-D): a scripted external end-effector wrench (ramp onset
then held) is applied during execution of an otherwise-benign trajectory. This is
an UNANTICIPATED disturbance: B3 does not get to see the whole force schedule at
planning time (unlike Exp 4) -- it only detects the ramp once it enters the
certificate's bounded online prediction horizon (Sec. IV / online_step). The
primary metric is the detection lead time

    T_warning = T_failure - T_detection

where T_failure is when the UNMODIFIED nominal trajectory would actually violate a
torque limit (ground truth, ignoring whatever any baseline does about it) and
T_detection is when each baseline's own mechanism first reacts. By construction,
B1 has no detection mechanism; B2 only reacts once currently-measured torque
already exceeds the limit (i.e. at/after T_failure itself); B3 should detect
before T_failure by up to one online-horizon length (Sec. VIII-D)."""
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

Q0 = np.array([0.0, -0.55, -0.25])
QF = np.array([0.8, -0.8, 0.35])
T = 1.4
PAYLOAD = 1.0

T_ONSET = 0.35     # s, when the push begins
RAMP_DURATION = 0.9  # s, time to ramp up to full magnitude
F_MAX = np.array([0.0, -55.0])  # N, a downward push at the end-effector


def force_schedule(t, q):
    if t < T_ONSET:
        return None
    frac = min((t - T_ONSET) / RAMP_DURATION, 1.0)
    return F_MAX * frac


def ground_truth_failure_time(arm, cert, traj, dt=0.02):
    """First time along the UNMODIFIED nominal trajectory (with the scripted force
    applied) at which a torque limit is actually violated."""
    n = int(np.ceil(traj.T / dt)) + 1
    for j in range(n):
        t = j * dt
        q, qdot, qddot = traj.sample(t)
        f = force_schedule(t, q)
        if cert.ground_truth_violation(q[None, :], qdot[None, :], qddot[None, :], f):
            return t
    return None


def run():
    arm0 = Arm.create(); arm0.set_payload_mass(PAYLOAD)
    cert0 = Certificate(arm=arm0, m_safe=2.0)
    traj0 = JointTrajectory(Q0, QF, T=T)
    t_failure = ground_truth_failure_time(arm0, cert0, traj0)
    print(f"Ground-truth failure time (unmodified nominal trajectory): "
          f"{t_failure if t_failure is not None else 'never'} s")

    results = {}
    detection_times = {}
    for name in ["B1", "B2", "B3"]:
        arm = Arm.create()
        arm.set_payload_mass(PAYLOAD)
        traj = JointTrajectory(Q0, QF, T=T)
        cert = Certificate(arm=arm, m_safe=2.0)

        if name == "B1":
            pol = policy_b1(traj)
        elif name == "B2":
            pol = policy_b2(traj, arm, ee_force_schedule=force_schedule)
        else:
            cfg = PlannerConfig(allow_level1=True, allow_level2=True, allow_level3=False)
            pol = policy_b3(traj, arm, cert, cfg, ee_force_schedule=force_schedule,
                             force_known_at_plan_time=False)

        rr = rollout(arm, pol, Q0, np.zeros(3), duration=T + 0.3, dt=0.02,
                     ee_force_schedule=force_schedule)
        goal_pos = arm.ee_position(QF)
        m = M.compute(rr, goal_pos)
        results[name] = m

        t_detect = None
        for t, lv in zip(rr.t, rr.levels):
            if lv not in (0, None):
                t_detect = t
                break
        detection_times[name] = t_detect

        print(f"{name}: success={m.task_success}, t_detect={t_detect}, "
              f"sat_samples={m.saturation_samples}, peak_tau_ratio={m.peak_torque_ratio:.2f}, "
              f"track_rms={m.tracking_error_rms_rad*1000:.2f}mrad, replans={m.replans}")

    if t_failure is not None:
        for name in ["B2", "B3"]:
            if detection_times[name] is not None:
                warning = t_failure - detection_times[name]
                print(f"T_warning[{name}] = T_failure({t_failure:.2f}) - "
                      f"T_detection({detection_times[name]:.2f}) = {warning:.2f} s")
            else:
                print(f"T_warning[{name}]: no detection occurred before end of rollout")
    return results, detection_times, t_failure


if __name__ == "__main__":
    run()
