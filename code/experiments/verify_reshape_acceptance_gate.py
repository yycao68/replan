"""Follow-up to reshape_linearization_gap.py's counterexample finding.

That script treated "_try_reshape returns a solution" (the QP's OWN frozen-
linearization belief that its constraints hold) as equivalent to "the planner
accepts this as a Level-2 success." It is not: every real call site
(local_planner.online_step and _search_reshape_whole_route) computes

    m2 = cert.m_phys(Q_new, Qdot_new, Qddot_new, forces)

before ever reporting success, and cert.m_phys calls arm.required_torque(...)
-- MuJoCo's FULL NONLINEAR inverse dynamics -- evaluated AT THE ACTUAL
(Q_new, Qdot_new, Qddot_new) the QP returned, not at the frozen nominal point
the QP's own internal linear model used to search. So m2 is already a true,
independent post-hoc re-verification, and both call sites reject the
solution (return None / fall through to best_effort, which the paper
explicitly discloses as a failure case) whenever m2 < m_safe.

This script checks that claim two ways:

  1. Reconstructs the EXACT scenario the paper's Sec. IX-H cites
     (test_planner.py::test_route_level_reshape_restores_feasibility_when_
     retiming_cannot) and independently recomputes the true per-(step,joint)
     torque from the actual returned route.traj, confirming it matches
     route.m_phys and stays within TAU_MAX with margin -- i.e. the paper's
     one cited "reshape succeeds" claim is genuinely true-dynamics-feasible,
     not just QP-feasible.

  2. Reruns reshape_linearization_gap.py's own "realistic" sweep, but gates
     acceptance the way the real planner does (m2 = cert.m_phys(...) >=
     m_safe) instead of "QP returned non-None". Reports: of the QP-optimal
     solves, how many the real gate would actually accept, and -- the
     substantive check -- whether ANY accepted (gate-passed) solution is
     still truly torque-infeasible. This should be tautologically impossible
     given cert.m_phys's own definition; this checks there is no numerical
     slip in practice.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from dynamics import Arm, TAU_MAX, TAU_MIN, N_JOINTS
from certificate import Certificate
from trajectory import JointTrajectory, ViaPointTrajectory
from local_planner import LocalPlanner, PlannerConfig


def part1_sec_ix_h_scenario():
    print("=" * 70)
    print("PART 1: the paper's own cited Sec. IX-H scenario (m_phys=2.55 claim)")
    print("=" * 70)

    Z_CONTACT = 0.55
    K_CONTACT = 300.0
    _probe = Arm.create()

    def contact_force(t, q):
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
    print(f"retiming search result: {lam} (expected None -- retiming-proof)")

    route = planner.plan_route(traj, ee_force_fn=contact_force)
    print(f"route.level = {route.level} (expected 2)")
    print(f"route.m_phys (as reported/certified) = {route.m_phys:.4f}")

    # Independently recompute the true per-(step,joint) margin from the
    # ACTUAL returned trajectory, using arm.required_torque directly -- the
    # same computation cert.m_phys does internally, just done here by hand
    # to double check there is no discrepancy.
    Qn, Qdotn, Qddotn = route.traj.Q, route.traj.Qdot, route.traj.Qddot
    n = Qn.shape[0]
    ts = cfg.dt * np.arange(n)
    forces = np.array([
        [0.0, 0.0] if (f := contact_force(t, q)) is None else f
        for t, q in zip(ts, Qn)
    ])

    worst_margin = np.inf
    worst_true_violation = -np.inf
    for j in range(n):
        tau_true = arm.required_torque(Qn[j], Qdotn[j], Qddotn[j], forces[j])
        margin = TAU_MAX - np.abs(tau_true) - cert.delta_tau
        worst_margin = min(worst_margin, float(margin.min()))
        true_violation = float(np.max(np.abs(tau_true) - TAU_MAX))
        worst_true_violation = max(worst_true_violation, true_violation)

    print(f"independently recomputed worst-case margin = {worst_margin:.4f} "
          f"(should match route.m_phys = {route.m_phys:.4f})")
    print(f"worst TRUE torque violation over TAU_MAX across the whole route = "
          f"{worst_true_violation:.4f} Nm (<=0 means genuinely feasible everywhere)")
    print()
    if worst_true_violation <= 0 and abs(worst_margin - route.m_phys) < 1e-6:
        print("CONFIRMED: the paper's cited m_phys=2.55 success is a genuine "
              "true-nonlinear-dynamics feasibility result, computed at the "
              "ACTUAL optimized trajectory -- not the QP's frozen-linearization "
              "belief. No soundness gap in this specific claim.")
    else:
        print("DISCREPANCY FOUND -- does not match expectation, needs investigation.")
    print()


def part2_real_acceptance_gate(regime_name="realistic", n_trials=3000):
    print("=" * 70)
    print(f"PART 2: real acceptance-gate rerun, regime={regime_name}, "
          f"n_trials={n_trials}")
    print("=" * 70)

    REGIMES = {
        "adversarial": dict(q_lo=-np.pi, q_hi=np.pi, perturb=None, t_lo=0.4, t_hi=1.5),
        "realistic": dict(q_lo=-1.5, q_hi=1.5, perturb=0.6, t_lo=1.0, t_hi=2.5),
    }
    FORCE_LO, FORCE_HI = -40.0, 40.0
    PAYLOADS = [0.0, 2.0, 5.0, 8.0, 12.0]
    CFG = PlannerConfig()
    regime = REGIMES[regime_name]
    rng = np.random.default_rng(0)

    n_nominal_violates = 0
    n_qp_optimal = 0          # _try_reshape returned non-None (its OWN belief)
    n_gate_accepted = 0       # cert.m_phys(Q_new,...) >= m_safe (REAL planner behavior)
    n_gate_rejected_but_qp_optimal = 0
    n_true_counterexamples_among_accepted = 0  # should stay exactly 0

    for trial in range(n_trials):
        q0 = rng.uniform(regime["q_lo"], regime["q_hi"], N_JOINTS)
        if regime["perturb"] is None:
            qf = rng.uniform(regime["q_lo"], regime["q_hi"], N_JOINTS)
        else:
            qf = q0 + rng.uniform(-regime["perturb"], regime["perturb"], N_JOINTS)
        T = rng.uniform(regime["t_lo"], regime["t_hi"])
        payload = rng.choice(PAYLOADS)
        force = rng.uniform(FORCE_LO, FORCE_HI, 2)

        arm = Arm.create()
        arm.set_payload_mass(payload)
        cert = Certificate(arm=arm)
        pl = LocalPlanner(arm, cert, CFG)

        n = CFG.horizon_steps
        traj = JointTrajectory(q0, qf, T)
        Q, Qdot, Qddot = traj.sample_horizon(0.0, CFG.dt, n)
        forces = np.tile(force, (n, 1))

        m = cert.horizon_margins(Q, Qdot, Qddot, forces)
        if m.min() >= 0:
            continue
        n_nominal_violates += 1

        result = pl._try_reshape(Q, Qdot, Qddot, forces)
        if result is None:
            continue
        Q_new, Qdot_new, Qddot_new = result
        if np.any(np.isnan(Q_new)) or np.any(np.isnan(Qddot_new)):
            continue
        n_qp_optimal += 1

        # THE REAL GATE, exactly as online_step / _search_reshape_whole_route
        # compute it.
        m2 = cert.m_phys(Q_new, Qdot_new, Qddot_new, forces)
        if m2 < cert.m_safe:
            n_gate_rejected_but_qp_optimal += 1
            continue
        n_gate_accepted += 1

        # Now check: among gate-accepted solutions, is there EVER a true
        # torque violation? (Should be impossible by cert.m_phys's own
        # definition -- this checks for numerical slip only.)
        for j in range(n):
            tau_true = arm.required_torque(Q_new[j], Qdot_new[j], Qddot_new[j], force)
            if np.any(np.abs(tau_true) > TAU_MAX):
                n_true_counterexamples_among_accepted += 1
                break

    print(f"nominal-violates: {n_nominal_violates}")
    print(f"QP-optimal (the frozen-linearization belief succeeded): {n_qp_optimal}")
    print(f"  of which REAL gate (cert.m_phys >= m_safe) ACCEPTS: {n_gate_accepted}")
    print(f"  of which REAL gate REJECTS (would fall to reroute/brake/best_effort): "
          f"{n_gate_rejected_but_qp_optimal}")
    print(f"TRUE torque counterexamples among gate-ACCEPTED solutions: "
          f"{n_true_counterexamples_among_accepted} (must be 0 for the certificate "
          f"to be sound)")
    print()
    if n_qp_optimal > 0:
        reject_rate = n_gate_rejected_but_qp_optimal / n_qp_optimal
        print(f"So {reject_rate:.1%} of the QP's own 'optimal' solves are actually "
              f"caught and discarded by the real post-hoc gate -- this is a "
              f"CONSERVATISM/completeness cost (the QP's linearized search wastes "
              f"effort proposing candidates the true-dynamics gate then rejects), "
              f"not a soundness gap: nothing rejected by the gate is ever reported "
              f"as a Level-2 success.")


if __name__ == "__main__":
    part1_sec_ix_h_scenario()
    part2_real_acceptance_gate("realistic", n_trials=3000)
    print()
    part2_real_acceptance_gate("adversarial", n_trials=3000)
