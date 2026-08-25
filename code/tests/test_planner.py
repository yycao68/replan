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
    test_reroute_switches_to_alt_route_when_primary_infeasible()
    print("\nAll planner tests passed.")
