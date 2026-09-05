"""Draft Status item 2 (README.md): does "reshape QP infeasible" imply "no
trajectory in the true nonlinear reshape class A_2(p) restores the
certificate"?

The reshape QP (`local_planner._try_reshape`) linearizes torque at each step
as tau_j = M(Q_nominal[j]) @ qddot_vars[j] + h(Q_nominal[j], Qdot_nominal[j]),
fixed at the NOMINAL trajectory's state -- not the optimization variables
(Q_new, Qdot_new) the QP itself solves for and returns. This script checks,
empirically, what that approximation actually costs on this benchmark's
dynamics, rather than leaving the question unaddressed:

  1. SAFETY DIRECTION (the one that matters for the certificate's soundness):
     when the QP reports a solution FEASIBLE, is that solution ALSO feasible
     under the TRUE (nonlinear) dynamics evaluated at the actual optimized
     state (Q_new, Qdot_new, Qddot_new)? A single counterexample here would
     mean the certificate can pass a truly torque-infeasible reshape -- this
     is checked over every (joint, step) of every QP-feasible solution found,
     not sampled.

  2. Linearization error magnitude: |tau_linearized - tau_true| per (joint,
     step), to characterize how large the approximation gap actually is,
     not just whether it ever flips a feasible/infeasible verdict.

  3. CONSERVATISM DIRECTION (the direction item 2 names as the open gap):
     among scenarios where the QP reports INFEASIBLE, could a true nonlinear
     reshape still have succeeded? Not directly checkable (the true nonlinear
     reshape problem is nonconvex), but relating (2)'s error magnitude to
     delta_tau's own safety margin gives an indirect, honestly-labeled
     plausibility read, not a proof.

This does not turn item 2 into a closed proof; it replaces "unaddressed" with
a measured empirical characterization, the same standard this project applies
elsewhere when a general first-principles bound isn't available (e.g. the
retiming Lemma's 23/3000 sign-condition violation rate, Theorem 4's 0/6708
invariance-violation sweep).

Run: python3 experiments/reshape_linearization_gap.py   (or via run_all.py)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from dynamics import Arm, TAU_MAX, TAU_MIN, N_JOINTS
from certificate import Certificate
from trajectory import JointTrajectory
from local_planner import LocalPlanner, PlannerConfig

CFG = PlannerConfig()
N_TRIALS = 3000
PAYLOADS = [0.0, 2.0, 5.0, 8.0, 12.0]

# Two regimes, to give a fair two-sided picture rather than one number that
# could either overstate or understate the practical risk:
#   "adversarial": q0/qf drawn independently across the whole joint range with
#     short durations -- a broad, general characterization (same spirit as the
#     retiming Lemma's own 3000-scenario sweep), but far more extreme than how
#     this codebase's own experiments actually use reshape.
#   "realistic": qf a small perturbation from q0 (magnitude matching
#     exp5_flagship_reroute.py's own Q0->QG displacement, max component ~0.55
#     rad) with longer, more generous durations -- the regime this paper's
#     actual scenarios (flagship, Experiment 7) operate in.
REGIMES = {
    "adversarial": dict(q_lo=-np.pi, q_hi=np.pi, perturb=None, t_lo=0.4, t_hi=1.5),
    "realistic": dict(q_lo=-1.5, q_hi=1.5, perturb=0.6, t_lo=1.0, t_hi=2.5),
}
FORCE_LO, FORCE_HI = -40.0, 40.0  # N, per axis, end-effector (x, z)


def _make(payload):
    arm = Arm.create()
    arm.set_payload_mass(payload)
    cert = Certificate(arm=arm)
    return arm, cert, LocalPlanner(arm, cert, CFG)


def _random_scenario(rng, regime):
    q0 = rng.uniform(regime["q_lo"], regime["q_hi"], N_JOINTS)
    if regime["perturb"] is None:
        qf = rng.uniform(regime["q_lo"], regime["q_hi"], N_JOINTS)
    else:
        qf = q0 + rng.uniform(-regime["perturb"], regime["perturb"], N_JOINTS)
    T = rng.uniform(regime["t_lo"], regime["t_hi"])
    payload = rng.choice(PAYLOADS)
    force = rng.uniform(FORCE_LO, FORCE_HI, 2)
    return q0, qf, T, payload, force


def run(regime_name="adversarial"):
    regime = REGIMES[regime_name]
    print(f"=== regime: {regime_name} {regime} ===")
    rng = np.random.default_rng(0)
    n_generated = 0
    n_nominal_violates = 0
    n_qp_feasible = 0
    n_qp_infeasible = 0
    n_qp_error = 0

    max_true_violation = 0.0  # > 0 would mean a real counterexample found
    worst_lin_error = 0.0
    lin_errors = []  # per-(joint,step) |tau_lin - tau_true| across all feasible solves
    counterexamples = []
    q_deviations = []  # max |Q_new - Q_nominal| per feasible solve, to explain WHY
    qdot_deviations = []  # max |Qdot_new - Qdot_nominal| per feasible solve
    delta_tau_ref = None

    for trial in range(N_TRIALS):
        q0, qf, T, payload, force = _random_scenario(rng, regime)
        arm, cert, pl = _make(payload)
        n = CFG.horizon_steps
        traj = JointTrajectory(q0, qf, T)
        Q, Qdot, Qddot = traj.sample_horizon(0.0, CFG.dt, n)
        forces = np.tile(force, (n, 1))

        m = cert.horizon_margins(Q, Qdot, Qddot, forces)
        n_generated += 1
        if m.min() >= 0:
            continue  # nominal already feasible; reshape wouldn't be invoked
        n_nominal_violates += 1

        result = pl._try_reshape(Q, Qdot, Qddot, forces)
        if result is None:
            n_qp_infeasible += 1
            continue
        Q_new, Qdot_new, Qddot_new = result
        if np.any(np.isnan(Q_new)) or np.any(np.isnan(Qddot_new)):
            n_qp_error += 1
            continue
        n_qp_feasible += 1
        if delta_tau_ref is None:
            delta_tau_ref = cert.delta_tau.copy()
        q_deviations.append(float(np.max(np.abs(Q_new - Q))))
        qdot_deviations.append(float(np.max(np.abs(Qdot_new - Qdot))))

        # Re-derive exactly what the QP's own linear model believed at each
        # step (M, h evaluated at the NOMINAL point, matching _try_reshape's
        # own construction) and compare to the TRUE torque at the ACTUAL
        # optimized state.
        for j in range(n):
            M_nom = arm.mass_matrix(Q[j])
            h_nom = arm.required_torque(Q[j], Qdot[j], np.zeros(N_JOINTS), force)
            tau_lin = M_nom @ Qddot_new[j] + h_nom
            tau_true = arm.required_torque(Q_new[j], Qdot_new[j], Qddot_new[j], force)

            err = np.abs(tau_lin - tau_true)
            lin_errors.append(err)
            worst_lin_error = max(worst_lin_error, float(np.max(err)))

            true_violation = np.max(np.abs(tau_true) - TAU_MAX)
            if true_violation > max_true_violation:
                max_true_violation = float(true_violation)
            if true_violation > 0:
                counterexamples.append({
                    "trial": trial, "step": j, "true_violation_Nm": float(true_violation),
                    "tau_true": tau_true.tolist(), "tau_lin": tau_lin.tolist(),
                })

    lin_errors = np.array(lin_errors)  # (n_solves*n_steps, N_JOINTS)

    print(f"Scenarios generated: {n_generated}")
    print(f"Nominal trajectory violates certificate: {n_nominal_violates}/{n_generated}")
    print(f"  reshape QP feasible:   {n_qp_feasible}/{n_nominal_violates}")
    print(f"  reshape QP infeasible: {n_qp_infeasible}/{n_nominal_violates}")
    print(f"  reshape QP solver error/NaN: {n_qp_error}/{n_nominal_violates}")
    print()
    print("(1) SAFETY DIRECTION: is every QP-feasible solution also true-dynamics-feasible?")
    print(f"    Counterexamples found: {len(counterexamples)}/{n_qp_feasible}")
    print(f"    Max TRUE torque violation across all feasible solves: {max_true_violation:.6f} Nm")
    print(f"    (delta_tau, the certificate's own safety margin, is {delta_tau_ref.tolist() if delta_tau_ref is not None else None} Nm)")
    if counterexamples:
        print(f"    First counterexample: {counterexamples[0]}")
    if q_deviations:
        qd = np.array(q_deviations)
        print(f"    max|Q_new - Q_nominal| across feasible solves: mean={qd.mean():.3f} rad, "
              f"p95={np.percentile(qd, 95):.3f} rad, max={qd.max():.3f} rad")
        print("    (large deviations mean the reshape QP moved far from the point where M/h "
              "were evaluated -- exactly where the code's own 'Q_new stays close to nominal' "
              "assumption, stated in _try_reshape's docstring, would break down)")
    if qdot_deviations:
        qvd = np.array(qdot_deviations)
        print(f"    max|Qdot_new - Qdot_nominal| across feasible solves: mean={qvd.mean():.3f} rad/s, "
              f"p95={np.percentile(qvd, 95):.3f} rad/s, max={qvd.max():.3f} rad/s")
    print()
    print("(2) Linearization error magnitude |tau_lin - tau_true|, per joint, over "
          f"{lin_errors.shape[0]} (solve, step) evaluations:")
    for i in range(N_JOINTS):
        col = lin_errors[:, i]
        print(f"    joint {i}: mean={col.mean():.4f}  p95={np.percentile(col, 95):.4f}  "
              f"max={col.max():.4f} Nm  (tau_max={TAU_MAX[i]} Nm)")
    print()
    print("(3) CONSERVATISM DIRECTION (indirect plausibility read, not a proof):")
    print(f"    worst-case linearization error {worst_lin_error:.4f} Nm vs. "
          f"delta_tau {delta_tau_ref.max():.4f} Nm (max over joints)")
    if worst_lin_error < delta_tau_ref.min():
        print("    -> worst-case error stays under even the SMALLEST per-joint delta_tau: "
              "an infeasible linearized QP is unlikely to be masking a truly-feasible "
              "nonlinear solution by more than the certificate's own uncertainty budget "
              "already absorbs, on the scenarios sampled here.")
    else:
        print("    -> worst-case error EXCEEDS at least one joint's delta_tau: the "
              "linearization gap is not dominated by the certificate's own uncertainty "
              "margin on the scenarios sampled here -- a real, unresolved source of "
              "potential conservatism (or optimism), not ruled out by this sweep.")

    return {
        "n_generated": n_generated,
        "n_nominal_violates": n_nominal_violates,
        "n_qp_feasible": n_qp_feasible,
        "n_qp_infeasible": n_qp_infeasible,
        "n_qp_error": n_qp_error,
        "counterexamples": counterexamples,
        "max_true_violation_Nm": max_true_violation,
        "worst_lin_error_Nm": worst_lin_error,
        "lin_error_mean_per_joint": lin_errors.mean(axis=0).tolist() if len(lin_errors) else None,
        "lin_error_p95_per_joint": np.percentile(lin_errors, 95, axis=0).tolist() if len(lin_errors) else None,
        "lin_error_max_per_joint": lin_errors.max(axis=0).tolist() if len(lin_errors) else None,
    }


if __name__ == "__main__":
    run("adversarial")
    print()
    run("realistic")
