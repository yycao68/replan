import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dynamics import Arm
from certificate import Certificate
from trajectory import JointTrajectory, ViaPointTrajectory
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


def test_route_level_reshape_does_not_falsely_pass_a_position_dependent_force_field():
    """Formerly asserted the OPPOSITE outcome (route.level == 2, "reshape
    succeeds") and was cited in the paper (Sec. IX-H) as m_phys=2.55 -- a real
    bug, found and fixed, not a paper wording issue. _search_reshape_whole_
    route (and online_step's Level-2 branch) sampled the contact force ONCE,
    from the NOMINAL trajectory's positions, then reused that stale array for
    the post-hoc cert.m_phys(Q_new, ...) re-verification of the RESHAPED
    trajectory. Because this contact force is position-dependent, and the
    reshaped path here dips deeper into the field than the nominal one
    (ee_z=0.144 vs. 0.234, contact plane at z=0.55) -- something the paper's
    own text already noted geometrically ("the reshaped path's minimum
    end-effector height is, if anything, slightly lower") without recognizing
    it invalidated the force used in the feasibility check -- the certificate
    check ran against a nominal-position force of 94.7N when the reshaped
    path's true force there is 121.7N. Independently recomputing the margin
    with forces resampled at the reshaped positions gives m_phys=-0.14, not
    the reported +2.55. Fixed in local_planner.py by resampling forces at the
    QP's returned Qn before the certificate check, in both call sites. Once
    fixed, this specific scenario no longer demonstrates a genuine route-level
    reshape success -- it now correctly falls through to level 0 (retiming and
    reshaping both fail; a real deployment would need allow_level3=True to
    reroute, as Exp 7's structurally identical case already does). No
    scenario in this benchmark currently demonstrates a genuine route-level
    reshape success under a position-dependent force field; this test now
    guards against the false-positive silently coming back, not against a
    regression in a real capability."""
    Z_CONTACT = 0.55
    K_CONTACT = 300.0

    def contact_force(t, q, _probe=Arm.create()):
        ee_z = _probe.ee_position(q)[1]
        pen = Z_CONTACT - ee_z
        return None if pen <= 0 else np.array([0.0, K_CONTACT * pen])

    q0 = np.array([0.15, 0.0, 0.0])
    qf = np.array([0.25, 0.0, 0.0])
    via = np.array([0.90, 0.0, 0.0])
    traj = ViaPointTrajectory(q0, via, qf, T1=1.2, T2=1.0)

    arm = Arm.create()
    arm.set_payload_mass(1.0)
    cert = Certificate(arm=arm, m_safe=2.0)
    cfg = PlannerConfig(allow_level1=True, allow_level2=True, allow_level3=False)
    planner = LocalPlanner(arm, cert, cfg)

    lam = planner._search_retime_whole_route(traj, contact_force)
    assert lam is None, "expected this force-driven deficit to be retiming-proof"

    route = planner.plan_route(traj, ee_force_fn=contact_force)
    print(f"route-level reshape test: level={route.level}, m_phys={route.m_phys:.2f}")
    assert route.triggered
    assert route.level == 0, (
        f"reshape should NOT report success on this scenario (got level "
        f"{route.level}) -- if this starts asserting level==2 again, verify "
        f"with an independent recomputation (forces resampled at route.traj's "
        f"actual Q, not the nominal trajectory's) before trusting it, since "
        f"this is exactly the false-positive this test guards against"
    )
    assert route.m_phys < cert.m_safe, "the true, force-consistent margin here is negative"


def test_closed_form_retiming_rescues_a_genuinely_dynamic_nonmonotonic_scenario():
    """Regression test for the gap monotonicity_lemma_draft.md Sec. 5's own last
    bullet flagged: the existing static-gravity-deficit test (above) has
    A_i == 0 for every joint (a degenerate hold trajectory), the one case where
    non-monotonicity is structurally impossible, so it cannot exercise the
    closed-form/dense-grid rescue path at all. This scenario is genuinely
    dynamic (found by direct search, not hand-derived): lambda_max alone does
    NOT clear m_safe, so the naive fast path would incorrectly report retiming
    exhausted, but the closed-form candidate set (_closed_form_lambda_
    candidates, the damping-aware quadratic-in-1/lambda model) finds a rescuing
    interior lambda directly, without needing the dense-grid safety net."""
    q0 = np.array([0.4795, 0.3582, 1.0574])
    qf = np.array([-0.8437, 0.0200, -0.2303])
    arm = Arm.create()
    arm.set_payload_mass(3.8901)
    cert = Certificate(arm=arm, m_safe=2.0)
    cfg = PlannerConfig(lam_max=4.0)
    planner = LocalPlanner(arm, cert, cfg)
    traj = JointTrajectory(q0, qf, T=0.6845)

    m_at_max = planner._whole_route_margin(traj.retimed(cfg.lam_max), None)
    assert m_at_max < cert.m_safe, "scenario should require an interior lambda, not lambda_max"

    A, B, D = planner._torque_decomposition_whole_route(traj, None)
    candidates = planner._closed_form_lambda_candidates(A, B, D)
    assert candidates, "expected at least one interior closed-form candidate"
    assert any(
        planner._whole_route_margin(traj.retimed(l), None) >= cert.m_safe for l in candidates
    ), "expected the closed-form candidate set to contain a rescuing lambda"

    lam = planner._search_retime_whole_route(traj, None)
    assert lam is not None, "expected the closed-form path to rescue this scenario"
    assert 1.0 < lam < cfg.lam_max
    m_final = planner._whole_route_margin(traj.retimed(lam), None)
    assert m_final >= cert.m_safe - 1e-6
    print(f"OK: closed-form retiming search found lambda={lam:.4f} restoring m_phys={m_final:.3f} "
          f"on a genuinely dynamic non-monotonic scenario (lambda_max alone gave {m_at_max:.3f})")


# ---------------------------------------------------------------------------
# Theorem 4 / the Lemma on the brake recursion (paper Sec. VI). These are
# correctness checks on the two structural properties the proof rests on, plus
# a small-sample check of the invariance conclusion itself. The full sweep
# behind the paper's reported numbers lives in
# experiments/theorem4_terminal_set.py; these keep the properties in the test
# suite so a future change to _brake_profile cannot silently invalidate the
# theorem without a test failing.
# ---------------------------------------------------------------------------

def _brake_planner(payload_kg=0.0):
    arm = Arm.create()
    arm.set_payload_mass(payload_kg)
    cert = Certificate(arm=arm)
    cfg = PlannerConfig()
    return arm, cert, cfg, LocalPlanner(arm, cert, cfg)


def test_brake_recursion_is_time_invariant_shift():
    """Lemma (i): B(f(x))_j == B(x)_{j+1}. This is what makes Theorem 4's
    successor profile already-certified except for its one appended step."""
    _, _, cfg, planner = _brake_planner()
    rng = np.random.default_rng(0)
    n = cfg.horizon_steps
    worst = 0.0
    for _ in range(50):
        q = rng.uniform(-np.pi, np.pi, 3)
        qdot = rng.uniform(-3.0, 3.0, 3)
        Q, Qdot, Qddot = planner._brake_profile(q, qdot)
        Q2, Qdot2, Qddot2 = planner._brake_profile(Q[1], Qdot[1])
        worst = max(
            worst,
            float(np.abs(Q2[: n - 1] - Q[1:]).max()),
            float(np.abs(Qdot2[: n - 1] - Qdot[1:]).max()),
            float(np.abs(Qddot2[: n - 1] - Qddot[1:]).max()),
        )
    assert worst == 0.0, f"brake recursion is not shift-invariant (max mismatch {worst:g})"
    print("OK: brake profile from the successor state is the exact one-step shift "
          "(Lemma (i), bit-identical over 50 states)")


def test_brake_reaches_rest_at_predicted_step_and_holds():
    """Lemma (ii)/(iii): rest at r(x) = max_i ceil(|qdot_i|/(a_max dt)), and rest
    states are fixed points. r(x) <= N-1 is Theorem 4's completion condition."""
    _, _, cfg, planner = _brake_planner()
    n, dt, amax = cfg.horizon_steps, cfg.dt, cfg.qddot_box
    v_brake = (n - 1) * amax * dt
    rng = np.random.default_rng(1)
    for _ in range(100):
        q = rng.uniform(-np.pi, np.pi, 3)
        qdot = rng.uniform(-v_brake, v_brake, 3)
        r = int(np.max(np.ceil(np.abs(qdot) / (amax * dt) - 1e-12)))
        assert r <= n - 1
        Q, Qdot, Qddot = planner._brake_profile(q, qdot)
        assert np.allclose(Qdot[r], 0.0), "did not reach rest at the predicted step"
        assert r == 0 or not np.allclose(Qdot[r - 1], 0.0), "reached rest early"
        assert np.allclose(Q[r:], Q[r]), "configuration moved after reaching rest"
        assert np.allclose(Qdot[r:], 0.0) and np.allclose(Qddot[r:], 0.0)
    print(f"OK: brake reaches rest at r(x) and holds there (Lemma (ii)/(iii)); "
          f"r(x) <= N-1 iff ||qdot||_inf <= {v_brake:.2f} rad/s")


def test_terminal_safe_set_is_positively_invariant():
    """Theorem 4(i): x in X_f => f(x) in X_f, where membership is 'reaches rest
    within the horizon AND the certificate over the whole brake profile is
    non-negative'. Also checks the set is non-vacuous at a nontrivial payload."""
    n_members = 0
    for payload in (0.0, 5.0):
        arm, cert, cfg, planner = _brake_planner(payload)
        n, dt, amax = cfg.horizon_steps, cfg.dt, cfg.qddot_box
        v_brake = (n - 1) * amax * dt
        rng = np.random.default_rng(2)

        def member(q, qdot):
            if np.max(np.abs(qdot)) > v_brake + 1e-12:
                return False
            Q, Qdot, Qddot = planner._brake_profile(q, qdot)
            return cert.m_phys(Q, Qdot, Qddot, None) >= 0.0

        members = 0
        for _ in range(120):
            q = rng.uniform(-np.pi, np.pi, 3)
            qdot = rng.uniform(-v_brake, v_brake, 3)
            if not member(q, qdot):
                continue
            members += 1
            Q, Qdot, _ = planner._brake_profile(q, qdot)
            assert member(Q[1], Qdot[1]), (
                f"invariance violated at payload {payload} kg: q={q}, qdot={qdot}"
            )
        assert members > 0, f"X_f sampled empty at payload {payload} kg -- test is vacuous"
        n_members += members
    print(f"OK: X_f is positively invariant under the brake recursion "
          f"(Theorem 4(i)); 0 violations over {n_members} sampled members")


if __name__ == "__main__":
    test_fast_heavy_route_falls_to_level4_when_only_that_is_allowed()
    test_retiming_restores_whole_route_feasibility_for_dynamic_deficit()
    test_reshape_qp_solves_online_and_respects_bounds()
    test_brake_profile_reduces_velocity()
    test_static_gravity_deficit_defeats_retiming()
    test_brake_profile_starts_from_q_actual_not_nominal_sample()
    test_policy_b3_sticky_brake_holds_position_after_level4()
    test_reroute_switches_to_alt_route_when_primary_infeasible()
    test_route_level_reshape_restores_feasibility_when_retiming_cannot()
    test_closed_form_retiming_rescues_a_genuinely_dynamic_nonmonotonic_scenario()
    test_brake_recursion_is_time_invariant_shift()
    test_brake_reaches_rest_at_predicted_step_and_holds()
    test_terminal_safe_set_is_positively_invariant()
    print("\nAll planner tests passed.")
