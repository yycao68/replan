"""Experiment 7: environment-conditioned rerouting.

Exp 5's flagship result reroutes because of a persistent STATIC/CONFIGURATION
deficit (an outstretched via-point under a heavy payload -- the torque demand
comes from the arm's own kinematics/inertia, not from anything in the
environment). This experiment isolates the complementary case the paper's own
Exp 4 stops short of: a route that enters a known ENVIRONMENTAL feature -- the
same scripted contact-stiffness region as Exp 4 (Sec. VIII-E), a virtual plane
at Z_CONTACT applying a penetration-proportional spring force via xfrc_applied,
NOT MuJoCo's native contact solver; see dynamics.py's module docstring -- while
an alternate route to the SAME goal avoids that region entirely.

Exp 4 shows the certificate can adapt (Level 1/2) to a known contact
transition on a SINGLE fixed route. It does not show rerouting away from an
environmental feature, because it only ever gives the planner one route. This
experiment gives it two, sharing q0 and qf (ViaPointTrajectory, as in Exp 5):
P_A's via-point dips the end-effector below the contact plane (deep enough
that the quasi-static spring force alone, evaluated at the via-point's own
zero-velocity/zero-acceleration boundary condition, exceeds the actuator
margin -- a persistent, RETIMING-PROOF deficit for the same reason Exp 5's
static-gravity deficit is: the spring force is a function of position only,
so slowing down cannot reduce it, exactly the "T_dyn(p) persists at every
retiming factor" case Theorem 3's sufficient condition m*_1(p) < m_safe is
built to detect); P_B's via-point stays above the plane throughout, so the
force never activates along it and its margin is whatever the (light) payload
alone requires. Both q1-only paths (q2=q3=0 throughout, for both routes) are
monotonic in the sole varying joint, so the via-point IS each segment's
extremum -- there is no need to sample between via-points to find the deepest
penetration or confirm P_B never dips.

B3 is given the contact-force model at PLANNING time (force_known_at_plan_time
=True, as in Exp 4) so its route-level decision (Level 1/3) can evaluate each
candidate route's margin under the known force profile, not just kinematics/
payload -- this is what lets it reroute BEFORE ever entering the field, not
merely react once already in contact."""
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

Q0 = np.array([0.15, 0.0, 0.0])     # shared start, ee well above the contact plane
QG = np.array([0.25, 0.0, 0.0])     # shared goal, also well above it
VIA_A = np.array([0.90, 0.0, 0.0])  # P_A's via-point: dips ee below Z_CONTACT
VIA_B = np.array([0.20, 0.0, 0.0])  # P_B's via-point: stays clear of it
T1, T2 = 0.6, 0.5
PAYLOAD = 1.0            # light -- the deficit comes from the environment, not payload
Z_CONTACT = 0.55         # world z of the virtual contact plane (Exp 4's mechanism)
K_CONTACT = 700.0        # N/m

_arm_probe = Arm.create()


def contact_force(t, q):
    """Same position-based model as Exp 4: an upward restoring force once the
    end-effector penetrates below the contact plane, independent of route."""
    ee_z = _arm_probe.ee_position(q)[1]
    penetration = Z_CONTACT - ee_z
    if penetration <= 0:
        return None
    return np.array([0.0, K_CONTACT * penetration])


def _static_margin(arm, q, force=None):
    tau = arm.required_torque(q, np.zeros(3), np.zeros(3), force)
    return (TAU_MAX - np.abs(tau)).min()


def run():
    probe = Arm.create()
    probe.set_payload_mass(PAYLOAD)
    for name, q in [("Q0", Q0), ("QG", QG), ("VIA_A (P_A)", VIA_A), ("VIA_B (P_B)", VIA_B)]:
        ee = _arm_probe.ee_position(q)
        print(f"{name:14s} ee=({ee[0]:.3f}, {ee[1]:.3f})  "
              f"{'BELOW' if ee[1] < Z_CONTACT else 'above'} contact plane (z={Z_CONTACT})")

    # This is the isolation Exp 5 does not give: VIA_A's deficit is entirely
    # environmental. Without the contact force, both via-points have a healthy
    # static margin under this (light) payload -- unlike Exp 5, where VIA_A is
    # deficient on kinematics/payload alone, force or no force.
    print(f"static margin at VIA_A, NO force (payload only): "
          f"{_static_margin(probe, VIA_A):.2f} Nm")
    print(f"static margin at VIA_A, WITH the known contact force: "
          f"{_static_margin(probe, VIA_A, contact_force(0.0, VIA_A)):.2f} Nm")
    print(f"static margin at VIA_B, NO force: {_static_margin(probe, VIA_B):.2f} Nm")

    traj_A = ViaPointTrajectory(Q0, VIA_A, QG, T1=T1, T2=T2)
    traj_B = ViaPointTrajectory(Q0, VIA_B, QG, T1=T1, T2=T2)
    assert np.allclose(traj_A.qf, traj_B.qf), "P_A and P_B must share the same goal"

    # The deficit at VIA_A is purely a function of position (spring force) at a
    # zero-velocity/zero-acceleration boundary point, so it is retiming-proof
    # by construction -- confirm the certificate's own bounded retiming search
    # agrees, rather than just asserting it.
    cert_probe = Certificate(arm=probe, m_safe=2.0)
    planner_probe = LocalPlanner(probe, cert_probe, PlannerConfig(allow_level1=True))
    lam = planner_probe._search_retime_whole_route(traj_A, contact_force)
    print(f"Level-1 retime search on P_A (with known force): "
          f"{'found a fix' if lam is not None else 'no lambda in range restores margin'}")

    goal_pos = Arm.create().ee_position(QG)

    results = {}
    for name in ["B1", "B2", "B3"]:
        arm = Arm.create()
        arm.set_payload_mass(PAYLOAD)
        cert = Certificate(arm=arm, m_safe=2.0)

        if name == "B1":
            pol = policy_b1(traj_A)
            duration = traj_A.T + 0.3
        elif name == "B2":
            pol = policy_b2(traj_A, arm, ee_force_schedule=contact_force)
            duration = traj_A.T + 0.3
        else:
            cfg = PlannerConfig(allow_level1=True, allow_level2=True, allow_level3=True)
            pol = policy_b3(traj_A, arm, cert, cfg, alt_traj=traj_B,
                             ee_force_schedule=contact_force, force_known_at_plan_time=True)
            # Same reasoning as exp5_flagship_reroute: peek at the route-level
            # decision to size duration off what was actually selected, rather
            # than a worst-case budget that overruns an already-finished
            # holding-position rollout.
            route = LocalPlanner(arm, cert, cfg).plan_route(
                traj_A, alt_traj=traj_B, ee_force_fn=contact_force)
            duration = route.traj.T + 0.3

        rr = rollout(arm, pol, Q0, np.zeros(3), duration=duration, dt=0.02,
                     ee_force_schedule=contact_force)
        m = M.compute(rr, goal_pos)
        results[name] = m
        levels = sorted(set(str(l) for l in rr.levels))
        print(f"{name}: success={m.task_success} (final_pos_error={m.final_pos_error_m:.3f} m), "
              f"levels={levels}, sat_samples={m.saturation_samples}, "
              f"peak_tau_ratio={m.peak_torque_ratio:.2f}, replans={m.replans}")
    return results


if __name__ == "__main__":
    run()
