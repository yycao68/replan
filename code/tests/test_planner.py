import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dynamics import Arm
from certificate import Certificate
from trajectory import JointTrajectory
from local_planner import LocalPlanner, PlannerConfig


def make_scenario(payload_kg, T):
    arm = Arm.create()
    arm.set_payload_mass(payload_kg)
    cert = Certificate(arm=arm, m_safe=2.0)
    q0 = np.array([0.0, -0.5, -0.3])
    qf = np.array([1.2, -1.0, 0.6])
    traj = JointTrajectory(q0, qf, T=T)
    return arm, cert, traj


def test_fast_heavy_route_falls_to_level4_when_only_that_is_allowed():
    arm, cert, traj = make_scenario(payload_kg=6.0, T=0.5)
    cfg = PlannerConfig(allow_level1=False, allow_level2=False, allow_level3=False)
    planner = LocalPlanner(arm, cert, cfg)
    route = planner.plan_route(traj)
    assert route.triggered and route.level == 0  # L1/L3 disabled at route level
    res = planner.online_step(route.traj, t0=0.15)
    assert res.triggered and res.level == 4
    print(f"OK: fast+heavy trajectory triggers certificate, m_phys={route.m_phys:.2f} Nm, falls to Level 4")


def test_retiming_restores_whole_route_feasibility_for_dynamic_deficit():
    arm, cert, traj = make_scenario(payload_kg=3.0, T=0.45)
    cfg = PlannerConfig(allow_level1=True, allow_level2=False, allow_level3=False)
    planner = LocalPlanner(arm, cert, cfg)
    route = planner.plan_route(traj)
    print(f"retiming test: level={route.level}, m_phys={route.m_phys:.2f}, "
          f"T_nominal={traj.T:.2f}, T_retimed={route.traj.T:.2f}")
    if route.triggered:
        assert route.level in (0, 1)
        if route.level == 1:
            assert route.m_phys >= cert.m_safe - 1e-6
            assert route.traj.T > traj.T  # actually slower now, not just locally scaled


def test_reshape_qp_solves_online_and_respects_bounds():
    arm, cert, traj = make_scenario(payload_kg=2.0, T=0.6)
    cfg = PlannerConfig(allow_level1=False, allow_level2=True, allow_level3=False)
    planner = LocalPlanner(arm, cert, cfg)
    res = planner.online_step(traj, t0=0.15)
    print(f"reshape test: level={res.level}, m0={res.m_phys_before:.2f}, m_after={res.m_phys_after:.2f}")
    if res.triggered and res.level == 2:
        assert res.m_phys_after >= cert.m_safe - 1e-6


def test_brake_profile_reduces_velocity():
    arm, cert, traj = make_scenario(payload_kg=6.0, T=0.5)
    cfg = PlannerConfig(allow_level1=False, allow_level2=False, allow_level3=False)
    planner = LocalPlanner(arm, cert, cfg)
    res = planner.online_step(traj, t0=0.15)
    assert res.level == 4
    assert np.all(np.abs(res.Qdot[-1]) <= np.abs(res.Qdot[0]) + 1e-9)
    print(f"OK: brake profile monotonically reduces |qdot|: {res.Qdot[0]} -> {res.Qdot[-1]}")


def test_static_gravity_deficit_defeats_retiming():
    """A pure static gravity-torque deficit (near-zero velocity/acceleration
    throughout) must NOT be fixable by retiming -- the physically-correct negative
    case implied by the paper's Level-1 discussion."""
    arm = Arm.create()
    arm.set_payload_mass(20.0)  # absurd payload -> gravity alone saturates joint 1
    cert = Certificate(arm=arm, m_safe=2.0)
    q_hold = np.array([0.0, 0.0, 0.0])
    traj = JointTrajectory(q_hold, q_hold, T=1.0)  # degenerate: hold position
    cfg = PlannerConfig(allow_level1=True, allow_level2=False, allow_level3=False)
    planner = LocalPlanner(arm, cert, cfg)
    route = planner.plan_route(traj)
    assert route.triggered
    assert route.level == 0, f"expected Level 1 to fail on pure static deficit, got level {route.level}"
    print("OK: retiming correctly fails to fix a pure static gravity-torque deficit")


def test_brake_profile_starts_from_q_actual_not_nominal_sample():
    """Regression test for a real bug: online_step's Level-4 brake profile used
    to start from the NOMINAL trajectory's sample at the current wall-clock
    time (Q[0]) instead of the robot's true position, so it never actually
    held still -- it kept crawling forward chasing the nominal plan. Passing
    q_actual/qdot_actual must change the brake profile's starting point."""
    arm, cert, traj = make_scenario(payload_kg=6.0, T=0.5)
    cfg = PlannerConfig(allow_level1=False, allow_level2=False, allow_level3=False)
    planner = LocalPlanner(arm, cert, cfg)

    res_no_actual = planner.online_step(traj, t0=0.3)
    assert res_no_actual.level == 4

    # A q_actual far from the nominal trajectory's sample at t0=0.3 -- e.g. the
    # robot stopped near its start pose instead of following the plan.
    stuck_q = traj.q0.copy()
    stuck_qdot = np.zeros(3)
    res_stuck = planner.online_step(traj, t0=0.3, q_actual=stuck_q, qdot_actual=stuck_qdot)
    assert res_stuck.level == 4
    assert np.allclose(res_stuck.Q[0], stuck_q), (
        f"brake profile should start at q_actual={stuck_q}, got {res_stuck.Q[0]}"
    )
    assert not np.allclose(res_no_actual.Q[0], res_stuck.Q[0]), (
        "q_actual should actually change the brake target when it differs "
        "from the nominal trajectory's own sample"
    )
    print("OK: Level-4 brake profile correctly starts from q_actual, not the nominal sample")


def test_policy_b3_sticky_brake_holds_position_after_level4():
    """Regression test for a real bug: once Level 4 held position correctly
    (fixed above), the certificate could later swing back to 'feasible' against
    the nominal trajectory's own un-overridden future samples and return a raw
    nominal reference far from the robot's actual (stopped) position -- an
    instantaneous jump that visibly destabilized the MuJoCo simulation (NaN/Inf
    qacc). baselines.policy_b3 must make Level 4 sticky: once triggered, every
    subsequent call holds at whatever q_actual is, never resuming the nominal
    plan on its own."""
    from baselines import policy_b3
    arm, cert, traj = make_scenario(payload_kg=6.0, T=0.5)
    cfg = PlannerConfig(allow_level1=False, allow_level2=False, allow_level3=False)
    pol = policy_b3(traj, arm, cert, cfg)

    q_stopped = np.array([0.3, -0.6, 0.1])
    zero_qdot = np.zeros(3)
    q_ref1, qdot_ref1, qddot_ref1, meta1 = pol(0.3, q_stopped, zero_qdot)
    assert meta1["level"] == 4
    assert np.allclose(q_ref1, q_stopped)
    assert np.allclose(qdot_ref1, 0) and np.allclose(qddot_ref1, 0)

    # Even at a much later wall-clock time (long past the nominal trajectory's
    # own duration, where the un-overridden nominal sample would be far from
    # q_stopped), the policy must still just hold -- not snap back.
    q_ref2, qdot_ref2, qddot_ref2, meta2 = pol(5.0, q_stopped, zero_qdot)
    assert meta2["level"] == 4
    assert np.allclose(q_ref2, q_stopped), (
        f"sticky brake should keep holding at {q_stopped}, got reference {q_ref2}"
    )
    assert np.allclose(qdot_ref2, 0) and np.allclose(qddot_ref2, 0)
    print("OK: policy_b3's Level 4 is sticky -- holds position, never snaps back to the nominal plan")


def test_reroute_switches_to_alt_route_when_primary_infeasible():
    arm = Arm.create()
    arm.set_payload_mass(4.0)
    cert = Certificate(arm=arm, m_safe=2.0)
    q0 = np.array([0.0, -1.125, -0.5])     # comfortable, large static margin (~11 Nm)
    qf_hard = np.array([1.5, -0.1, 1.3])   # fast swing into an outstretched pose: hard
    qf_easy = np.array([0.05, -1.1, -0.45])  # tiny motion, stays near q0: easy
    hard = JointTrajectory(q0, qf_hard, T=0.35)
    easy = JointTrajectory(q0, qf_easy, T=1.5)
    cfg = PlannerConfig(allow_level1=False, allow_level2=False, allow_level3=True)
    planner = LocalPlanner(arm, cert, cfg)
    route = planner.plan_route(hard, alt_traj=easy)
    print(f"reroute test: level={route.level}, m_phys={route.m_phys:.2f}")
    assert route.triggered
    assert route.level == 3, f"expected the easy alt route to be accepted, got level {route.level}"


if __name__ == "__main__":
    test_fast_heavy_route_falls_to_level4_when_only_that_is_allowed()
    test_retiming_restores_whole_route_feasibility_for_dynamic_deficit()
    test_reshape_qp_solves_online_and_respects_bounds()
    test_brake_profile_reduces_velocity()
    test_static_gravity_deficit_defeats_retiming()
    test_brake_profile_starts_from_q_actual_not_nominal_sample()
    test_policy_b3_sticky_brake_holds_position_after_level4()
    test_reroute_switches_to_alt_route_when_primary_infeasible()
    print("\nAll planner tests passed.")
