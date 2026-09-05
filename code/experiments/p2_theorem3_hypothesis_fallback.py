"""P2 Theorem 3: does the safety fallback survive the world model being wrong?

BELONGS TO THE SECOND PAPER (`../../world_model_realizability_core.md`), not to
the predictive-realizability draft this directory otherwise verifies. It is
deliberately NOT wired into `run_all.py`, so that `run_all.py` keeps reproducing
exactly the numbers P1's Sec. IX cites and nothing else. It reuses P1's platform
(planar3r, the Sec. IV certificate, and the Level-4 brake recursion whose
terminal-set machinery is in `theorem4_terminal_set.py`) because the physics
question is the same one at a different level of the architecture.

THE CLAIM UNDER TEST (P2, Theorem 3). A terminal safe set computed under the
world model's own environment hypothesis H_E gives no guarantee once H_E is
false, because the falsification event that invalidates the plan can invalidate
the fallback at the same instant. A terminal set computed under a
hypothesis-free physical bound E_wc does give one. Quotably: you cannot use the
world model to plan your recovery from the world model being wrong.

WHAT WOULD MAKE THIS EXPERIMENT WORTHLESS, and how each is avoided:

  (a) Rigging by construction. "A set computed under a worse environment
      survives a worse environment" is a tautology. So the headline number here
      is NOT the wc-set's survival rate -- that is a consistency check, and it
      is labelled as one. The headline is the H_E-set's FAILURE rate, plus the
      conservatism cost of avoiding it, plus a falsification sweep that runs
      PAST the edge of E_wc so that the wc construction is seen to fail too.
      A guarantee that never fails anywhere is a guarantee that was assumed.

  (b) Confounding with P1's own reference-vs-executed gap. P1's Proposition 6
      already established that the executed sticky-hold law saturates from
      states P1's Theorem 4 admits, with no falsification involved at all
      (0/36/57/85/94 of 150 rollouts at 0/2/5/8/12 kg). Mixing that in would
      make falsification look responsible for failures it did not cause. This
      module therefore measures at the REFERENCE level throughout -- the object
      P2's Theorem 3 is stated about -- and the executed-level gap is left where
      P1 measured it.

  (c) A hand-picked scenario. Part A samples states rather than choosing them,
      Part B sweeps the falsification magnitude rather than fixing it, and the
      one hand-built trajectory (Part C) is stated with its parameters so the
      reader can see it was tuned to be certified-then-violated, which is the
      point rather than a concealment.

ENVIRONMENT MODEL. P1's Exp 4/7 contact plane: a virtual surface at world height
z, exerting an upward penetration-proportional force K*max(0, z - ee_z) on the
end-effector. An environment realization is the pair (z, K); an environment SET
is a box over both. This is the smallest model that supports a falsifiable
hypothesis: the world model asserts where the floor is and how stiff it is, and
reality can put it somewhere else.

Run: python3 experiments/p2_theorem3_hypothesis_fallback.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass

import numpy as np

from dynamics import Arm, TAU_MAX, N_JOINTS
from certificate import Certificate
from trajectory import ViaPointTrajectory
from local_planner import LocalPlanner, PlannerConfig

CFG = PlannerConfig()
N, DT, AMAX = CFG.horizon_steps, CFG.dt, CFG.qddot_box
V_BRAKE = (N - 1) * AMAX * DT      # brake-completion bound, P1 Theorem 4's Lemma (ii)
M_SAFE = 2.0

# --- the hypothesis the world model asserts, and the physical bound -------
Z_ASSERTED, K_ASSERTED = 0.55, 700.0
H_E = None      # set in _envs(), a narrow band about the assertion
E_WC = None     # a wide band from platform/physics limits, NOT from the world model


@dataclass(frozen=True)
class EnvBox:
    """A set of contact-plane realizations {(z, K) : z in [z_lo,z_hi], K in [k_lo,k_hi]}.

    A singleton (lo == hi) is an admissible degenerate case and is what a
    point-predicting world model emits -- P2 Def. 1's remark."""
    z_lo: float
    z_hi: float
    k_lo: float
    k_hi: float
    name: str = ""

    @staticmethod
    def point(z, k, name=""):
        return EnvBox(z, z, k, k, name)

    def contains(self, z, k):
        return (self.z_lo - 1e-12 <= z <= self.z_hi + 1e-12
                and self.k_lo - 1e-12 <= k <= self.k_hi + 1e-12)


def _envs(z_band=0.01, k_frac=0.10, wc_z=(0.45, 0.70), wc_k=(0.0, 900.0)):
    h = EnvBox(Z_ASSERTED - z_band, Z_ASSERTED + z_band,
               K_ASSERTED * (1 - k_frac), K_ASSERTED * (1 + k_frac), "H_E")
    wc = EnvBox(wc_z[0], wc_z[1], wc_k[0], wc_k[1], "E_wc")
    return h, wc


# ----------------------------------------------------------------------
# Worst case over an environment box, computed EXACTLY rather than sampled.
#
# The contact force is F = [0, s] with the scalar s = K * max(0, z - ee_z(q)),
# and required torque enters as tau = tau_id(q,qdot,qddot) - J_z(q)^T s, i.e.
# AFFINELY in s (verified against dynamics.required_torque in _self_check).
# s is monotone increasing in both z and K, so over the box it ranges over the
# interval [s_lo, s_hi] attained at the box's corners, and |tau_i(s)| -- a
# modulus of an affine function -- attains its supremum on that interval at an
# endpoint. Two inverse-dynamics evaluations therefore give the EXACT per-joint
# supremum, so this implementation realizes P2 Theorem 1(ii)'s tightness rather
# than approximating it: the worst case reported is achieved by an environment
# genuinely admissible under the box.
# ----------------------------------------------------------------------
def force_scalar_range(arm, q, box: EnvBox):
    ee_z = arm.ee_position(q)[1]
    return (box.k_lo * max(0.0, box.z_lo - ee_z),
            box.k_hi * max(0.0, box.z_hi - ee_z))


def force_scalar(arm, q, z, k):
    return k * max(0.0, z - arm.ee_position(q)[1])


def beta_at(arm, cert, q, box: EnvBox, z_nom, k_nom):
    """P2 Def. 5's detection threshold at one state: the largest torque discrepancy
    the hypothesis itself can explain, plus the robot-model error bound. A residual
    above this cannot be accounted for by any environment the hypothesis admits."""
    s_lo, s_hi = force_scalar_range(arm, q, box)
    s_nom = force_scalar(arm, q, z_nom, k_nom)
    jz = np.abs(arm.jacobian(q)[1, :])
    swing = max(abs(s_hi - s_nom), abs(s_nom - s_lo))
    return float(np.max(jz * swing) + np.max(cert.delta_tau))


def worst_case_margins(arm, cert, Q, Qdot, Qddot, box: EnvBox):
    """(n_steps, n_joints) array of tau_max - sup_{e in box}|tau_req,i| - delta_tau."""
    n_steps = Q.shape[0]
    m = np.zeros((n_steps, N_JOINTS))
    for j in range(n_steps):
        tau_id = arm.required_torque(Q[j], Qdot[j], Qddot[j], None)
        jz = arm.jacobian(Q[j])[1, :]
        s_lo, s_hi = force_scalar_range(arm, Q[j], box)
        worst = np.maximum(np.abs(tau_id - jz * s_lo), np.abs(tau_id - jz * s_hi))
        m[j, :] = TAU_MAX - worst - cert.delta_tau
    return m


def rho(arm, cert, Q, Qdot, Qddot, box: EnvBox):
    """P2 Def. 2: the realizability margin of a trajectory under an environment set."""
    return float(worst_case_margins(arm, cert, Q, Qdot, Qddot, box).min())


def _r_of(qdot):
    return int(np.max(np.ceil(np.abs(qdot) / (AMAX * DT) - 1e-12)))


def in_terminal_set(planner, arm, cert, q, qdot, box: EnvBox):
    """Membership in X_f^box: the brake reaches rest inside the horizon (P1
    Theorem 4's completion condition) and the whole brake profile, including its
    terminal hold, is actuator-feasible for EVERY environment in the box."""
    if _r_of(qdot) > N - 1:
        return False
    Q, Qdot, Qddot = planner._brake_profile(q, qdot)
    return rho(arm, cert, Q, Qdot, Qddot, box) >= 0.0


def brake_safe_under(planner, arm, cert, q, qdot, truth: EnvBox):
    """Ground truth: does the brake profile from (q,qdot) actually stay within
    actuator limits when the environment turns out to be `truth`?"""
    if _r_of(qdot) > N - 1:
        return False
    Q, Qdot, Qddot = planner._brake_profile(q, qdot)
    return rho(arm, cert, Q, Qdot, Qddot, truth) >= 0.0


# ----------------------------------------------------------------------
def _make(payload):
    arm = Arm.create()
    arm.set_payload_mass(payload)
    arm.model.opt.timestep = DT
    cert = Certificate(arm=arm, m_safe=M_SAFE)
    return arm, cert, LocalPlanner(arm, cert, CFG)


def _sample_contact_states(arm, rng, n, z_ref, band=(-0.15, 0.10)):
    """States whose end-effector is near the contact plane -- the only region
    where the hypothesis is load-bearing. Sampling the whole configuration space
    would dilute every rate reported here with states that never touch anything."""
    out = []
    tries = 0
    while len(out) < n and tries < 400 * n:
        tries += 1
        q = rng.uniform(-1.2, 1.2, N_JOINTS)
        ee_z = arm.ee_position(q)[1]
        if not (z_ref + band[0] <= ee_z <= z_ref + band[1]):
            continue
        out.append((q, rng.uniform(-V_BRAKE, V_BRAKE, N_JOINTS)))
    return out


def _self_check():
    """The affine-in-s claim the exact worst case rests on."""
    arm, cert, _ = _make(1.0)
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        q = rng.uniform(-1.2, 1.2, N_JOINTS)
        qd = rng.uniform(-2, 2, N_JOINTS)
        qdd = rng.uniform(-8, 8, N_JOINTS)
        s = rng.uniform(0, 300)
        direct = arm.required_torque(q, qd, qdd, np.array([0.0, s]))
        affine = arm.required_torque(q, qd, qdd, None) - arm.jacobian(q)[1, :] * s
        worst = max(worst, float(np.abs(direct - affine).max()))
    print(f"   self-check: torque is affine in the contact-force scalar to "
          f"{worst:.3e} N*m over 200 random (q,qdot,qddot,s)"
          f"  -> the box worst case below is exact, not sampled")


# ----------------------------------------------------------------------
def part_a(payloads=(1.0, 3.0, 5.0), n=400):
    """Conservatism cost: how much smaller is the hypothesis-free terminal set?"""
    print("\nA. Set sizes -- what hypothesis-free recovery costs, before asking what it buys")
    h, wc = _envs()
    print(f"   H_E  : z in [{h.z_lo:.3f},{h.z_hi:.3f}], K in [{h.k_lo:.0f},{h.k_hi:.0f}]"
          f"   (a calibrated world model)")
    print(f"   E_wc : z in [{wc.z_lo:.3f},{wc.z_hi:.3f}], K in [{wc.k_lo:.0f},{wc.k_hi:.0f}]"
          f"   (physical bound, not from the world model)")
    print(f"   {'payload':>8} | {'X_f^H_E':>9} {'X_f^wc':>9} {'ratio':>7}")
    out = {}
    for payload in payloads:
        arm, cert, pl = _make(payload)
        rng = np.random.default_rng(11)
        states = _sample_contact_states(arm, rng, n, Z_ASSERTED)
        n_h = sum(in_terminal_set(pl, arm, cert, q, qd, h) for q, qd in states)
        n_w = sum(in_terminal_set(pl, arm, cert, q, qd, wc) for q, qd in states)
        ratio = (n_w / n_h) if n_h else float("nan")
        print(f"   {payload:8.1f} | {100*n_h/len(states):8.1f}% {100*n_w/len(states):8.1f}% "
              f"{ratio:7.2f}")
        out[payload] = (n_h, n_w, len(states))
    print("   (states sampled with the end-effector within [-0.15,+0.10] m of the plane)")
    return out


def part_b(payload=1.0, n=400, deltas=(0.0, 0.02, 0.05, 0.10, 0.15, 0.20)):
    """The headline. For each falsification magnitude, how often is a state the
    H_E-certified fallback admits actually unsafe -- and what does E_wc do?"""
    print("\nB. Certified-but-unsafe rate as the falsification grows")
    h, wc = _envs()
    arm, cert, pl = _make(payload)
    rng = np.random.default_rng(11)
    states = _sample_contact_states(arm, rng, n, Z_ASSERTED)
    in_h = [(q, qd) for q, qd in states if in_terminal_set(pl, arm, cert, q, qd, h)]
    in_w = [(q, qd) for q, qd in states if in_terminal_set(pl, arm, cert, q, qd, wc)]
    print(f"   payload {payload} kg; {len(in_h)} states in X_f^H_E, {len(in_w)} in X_f^wc,"
          f" of {len(states)} sampled")
    print(f"   {'dz (m)':>7} {'z_true':>7} {'in E_wc?':>9} | "
          f"{'X_f^H_E unsafe':>15} | {'X_f^wc unsafe':>14}")
    out = {}
    for dz in deltas:
        z_true = Z_ASSERTED + dz
        truth = EnvBox.point(z_true, K_ASSERTED, "truth")
        covered = wc.contains(z_true, K_ASSERTED)
        bad_h = sum(not brake_safe_under(pl, arm, cert, q, qd, truth) for q, qd in in_h)
        bad_w = sum(not brake_safe_under(pl, arm, cert, q, qd, truth) for q, qd in in_w)
        fh = f"{bad_h}/{len(in_h)} ({100*bad_h/max(len(in_h),1):.0f}%)"
        fw = f"{bad_w}/{len(in_w)} ({100*bad_w/max(len(in_w),1):.0f}%)"
        print(f"   {dz:7.2f} {z_true:7.3f} {str(covered):>9} | {fh:>15} | {fw:>14}")
        out[dz] = (bad_h, len(in_h), bad_w, len(in_w), covered)
    print("   in E_wc = the true environment is still inside the hypothesis-free bound.")
    print("   Where it is, the X_f^wc column is a CONSISTENCY CHECK (0 is what the")
    print("   construction guarantees), and only the X_f^H_E column is a finding. The")
    print("   last rows, where truth escapes E_wc, are the control: the wc construction")
    print("   is not magic, it is exactly as good as the bound's coverage.")
    return out


def _plan(via_q1, T1=0.6, T2=0.5):
    return ViaPointTrajectory(np.array([0.15, 0.0, 0.0]),
                              np.array([via_q1, 0.0, 0.0]),
                              np.array([0.25, 0.0, 0.0]), T1=T1, T2=T2)


def _samples(traj):
    ts = np.arange(0.0, traj.T, DT)
    Q = np.array([traj.sample(t)[0] for t in ts])
    Qd = np.array([traj.sample(t)[1] for t in ts])
    Qdd = np.array([traj.sample(t)[2] for t in ts])
    return ts, Q, Qd, Qdd


def part_c1(payload=1.0, via_q1=0.46, dz=0.07):
    """The rock, stated as a scenario: a plan comfortably certified under the
    hypothesis, and a surface 7 cm higher than asserted."""
    print("\nC1. The rock -- a certified plan that the truth violates")
    h, _ = _envs()
    arm, cert, _ = _make(payload)
    truth = EnvBox.point(Z_ASSERTED + dz, K_ASSERTED, "truth")
    traj = _plan(via_q1)
    _, Q, Qd, Qdd = _samples(traj)
    r_h, r_t = rho(arm, cert, Q, Qd, Qdd, h), rho(arm, cert, Q, Qd, Qdd, truth)
    print(f"   plan via q1={via_q1:.2f}, min ee_z = {min(arm.ee_position(q)[1] for q in Q):.4f} m,"
          f" payload {payload} kg")
    print(f"   asserted plane z={Z_ASSERTED:.2f}: plan margin {r_h:+.2f} N*m"
          f"  -> certified (m_safe={M_SAFE})" if r_h >= M_SAFE else f"   NOT certified ({r_h:+.2f})")
    print(f"   true     plane z={truth.z_lo:.2f}: plan margin {r_t:+.2f} N*m"
          f"  -> {'VIOLATED' if r_t < 0 else 'still feasible'}")
    return traj, truth


def part_c2(payload=1.0, via_q1=0.46, dz=0.07,
            bands=(0.100, 0.050, 0.020, 0.010, 0.005, 0.002)):
    """P2 Theorems 1(iii) and 2 together: tightening the hypothesis returns margin
    and lowers the detection threshold, and warning time becomes GUARANTEED once
    rho > beta. A single hypothesis width would show neither."""
    print("\nC2. Hypothesis width vs warning time (Theorem 1(iii) + Theorem 2)")
    arm, cert, _ = _make(payload)
    truth = EnvBox.point(Z_ASSERTED + dz, K_ASSERTED, "truth")
    traj = _plan(via_q1)
    ts, Q, Qd, Qdd = _samples(traj)

    tau_h = np.array([arm.required_torque(Q[j], Qd[j], Qdd[j],
                      np.array([0.0, force_scalar(arm, Q[j], Z_ASSERTED, K_ASSERTED)]))
                      for j in range(len(ts))])
    tau_t = np.array([arm.required_torque(Q[j], Qd[j], Qdd[j],
                      np.array([0.0, force_scalar(arm, Q[j], truth.z_lo, truth.k_lo)]))
                      for j in range(len(ts))])
    dev = (np.abs(tau_t) - np.abs(tau_h)).max(axis=1)
    onset = np.flatnonzero(dev > 1e-9)
    viol = np.flatnonzero([np.any(np.abs(tau_t[j]) > TAU_MAX) for j in range(len(ts))])
    if onset.size == 0 or viol.size == 0:
        print("   scenario produced no onset or no violation -- retune")
        return None
    i0, iv = int(onset[0]), int(viol[0])
    L = float(np.max(np.diff(dev[i0:iv + 1]) / DT))
    print(f"   onset t0={ts[i0]:.3f}s, plan saturates t_v={ts[iv]:.3f}s,"
          f" measured deviation rate L={L:.1f} N*m/s")
    print(f"   {'z band':>8} {'beta':>7} {'rho':>7} {'rho>beta':>9} "
          f"{'t_d':>7} {'meas T_w':>9} {'pred (rho-b)/L':>15}")
    out = {}
    for band in bands:
        h = EnvBox(Z_ASSERTED - band, Z_ASSERTED + band,
                   K_ASSERTED * 0.9, K_ASSERTED * 1.1, "H_E")
        beta_t = np.array([beta_at(arm, cert, Q[j], h, Z_ASSERTED, K_ASSERTED)
                           for j in range(len(ts))])
        r = rho(arm, cert, Q[i0:], Qd[i0:], Qdd[i0:], h)
        fired = np.flatnonzero(dev > beta_t)
        fired = fired[fired >= i0]
        b_at_det = float(beta_t[fired[0]]) if fired.size else float(beta_t[i0])
        td = ts[fired[0]] if fired.size else float("nan")
        tw = (ts[iv] - td) if fired.size else float("nan")
        pred = (r - b_at_det) / L
        print(f"   {band:8.3f} {b_at_det:7.2f} {r:7.2f} {str(r > b_at_det):>9} "
              f"{td:7.3f} {tw:9.3f} {pred:15.3f}")
        out[band] = (b_at_det, r, td, tw, pred)
    print("   rho is the H_E margin banked over the plan from onset onward.")
    print("   Read the last two columns together: the prediction is a GUARANTEED LOWER")
    print("   BOUND, so pred <= meas is the theorem holding, and a negative pred means")
    print("   no guarantee is available at that hypothesis width -- not a refutation.")
    return out


def part_c3(payload=1.0, dz=0.07, vias=np.arange(0.20, 0.52, 0.02)):
    """P2 Theorem 3 proper. Condition (b) says the planner must MAINTAIN
    x in X_f^wc along the plan. The previous version of this experiment did not
    enforce it, so the wc set merely refused the state and demonstrated nothing.
    Here two planners are compared: one enforcing only the H_E certificate, one
    also enforcing (b). The falsification is identical for both."""
    print("\nC3. Theorem 3: two planners, one falsification")
    h, wc = _envs()
    arm, cert, pl = _make(payload)
    truth = EnvBox.point(Z_ASSERTED + dz, K_ASSERTED, "truth")

    best_h, best_b = None, None
    for v in vias:
        traj = _plan(float(v))
        ts, Q, Qd, Qdd = _samples(traj)
        if rho(arm, cert, Q, Qd, Qdd, h) < M_SAFE:
            continue
        best_h = float(v)                                    # deepest H_E-certified dip
        if all(in_terminal_set(pl, arm, cert, Q[j], Qd[j], wc) for j in range(len(ts))):
            best_b = float(v)                                # deepest that also keeps (b)
    if best_h is None:
        print("   no via depth is certified under H_E -- retune")
        return None
    print(f"   deepest dip certified under H_E alone        : q1 = {best_h:.2f}")
    print(f"   deepest dip that also maintains x in X_f^wc  : "
          f"{('q1 = %.2f' % best_b) if best_b is not None else 'NONE in the swept range'}")

    ee_b = min(arm.ee_position(_plan(best_b).sample(t)[0])[1]
               for t in np.arange(0, _plan(best_b).T, DT)) if best_b is not None else float("nan")
    print(f"   (the cond.-(b) plan's lowest end-effector height is {ee_b:.3f} m -- above the")
    print(f"    asserted plane entirely: hypothesis-free recoverability, on this platform and")
    print(f"    with this E_wc, means not approaching the uncertain surface at all)")

    # Sweep the falsification so the interesting case appears: one large enough to
    # violate BOTH plans. Only there does the fallback get to do its job, rather
    # than the conservative planner simply never getting into trouble.
    print(f"\n   {'dz':>5} {'in E_wc':>8} | {'H_E-only: plan':>15} {'fallback':>22}"
          f" | {'cond.(b): plan':>15} {'fallback':>22}")
    for dzs in (0.05, 0.07, 0.10, 0.15, 0.20):
        tr = EnvBox.point(Z_ASSERTED + dzs, K_ASSERTED, "truth")
        row = []
        for v in (best_h, best_b):
            if v is None:
                row += ["--", "--"]
                continue
            ts, Q, Qd, Qdd = _samples(_plan(v))
            r_t = rho(arm, cert, Q, Qd, Qdd, tr)
            bad = sum(not brake_safe_under(pl, arm, cert, Q[j], Qd[j], tr)
                      for j in range(len(ts)))
            row += [f"{r_t:+.2f}" + (" VIOL" if r_t < 0 else "     "),
                    "safe from all" if bad == 0 else f"UNSAFE {bad}/{len(ts)}"]
        print(f"   {dzs:5.2f} {str(wc.contains(Z_ASSERTED + dzs, K_ASSERTED)):>8} | "
              f"{row[0]:>15} {row[1]:>22} | {row[2]:>15} {row[3]:>22}")
    print("   Read this honestly, because it is not the storybook version of Theorem 3.")
    print("   CONFIRMED, strongly: for the H_E-only planner, plan and fallback fail")
    print("   TOGETHER at every falsification magnitude -- one hypothesis failure, both")
    print("   layers gone, which is Theorem 3's converse.")
    print("   NOT observed: a row where the cond.-(b) plan is violated and its fallback")
    print("   then rescues it. Inside E_wc's coverage the cond.-(b) plan is never violated")
    print("   at all, so its fallback is never exercised; outside coverage (dz=0.20) both")
    print("   fail. So this scenario demonstrates the converse but leaves the forward")
    print("   direction untested -- condition (b) turns out to be strong enough here that")
    print("   it subsumes plan safety. Part D measures why.")
    return dict(best_h=best_h, best_b=best_b)


def part_d(payload=1.0, dzs=(0.05, 0.10, 0.15)):
    """How much authority does a stop-in-place fallback actually have against this
    falsification class? Theorem 3 guarantees the fallback is FEASIBLE; it says
    nothing about whether stopping helps. For a position-dependent contact force
    the deficit is quasi-static -- P1's Exp 7 makes the same point about retiming --
    so braking can only prevent the robot going DEEPER, not relieve the force
    already acting. That distinction decides whether Theorem 3 buys anything real,
    and it is measurable rather than arguable."""
    print("\nD. Fallback authority: when continuing is infeasible, does stopping help?")
    h, _ = _envs()
    arm, cert, pl = _make(payload)
    rng = np.random.default_rng(11)
    states = _sample_contact_states(arm, rng, 400, Z_ASSERTED)
    print(f"   {'dz':>5} | {'continue infeasible':>20} | {'of those, brake feasible':>25}")
    for dz in dzs:
        truth = EnvBox.point(Z_ASSERTED + dz, K_ASSERTED, "truth")
        n_bad = n_rescued = 0
        for q, qd in states:
            if _r_of(qd) > N - 1:
                continue
            # "continue" = hold the current velocity for one horizon, coasting on
            cont_Q = np.array([q + qd * (j * DT) for j in range(N)])
            cont_Qd = np.tile(qd, (N, 1))
            cont_Qdd = np.zeros((N, N_JOINTS))
            if rho(arm, cert, cont_Q, cont_Qd, cont_Qdd, truth) >= 0:
                continue
            n_bad += 1
            if brake_safe_under(pl, arm, cert, q, qd, truth):
                n_rescued += 1
        pct = 100 * n_rescued / max(n_bad, 1)
        print(f"   {dz:5.2f} | {n_bad:20d} | {n_rescued:12d}  ({pct:5.1f}%)")
    print("   A high percentage means the fallback has real authority against this")
    print("   falsification class; a low one means Theorem 3 guarantees a stop that does")
    print("   not save the robot, and hypothesis-free recoverability degenerates into")
    print("   'never approach the uncertain surface' -- which is exactly what C3's")
    print("   cond.-(b) planner was forced into.")
    return None


def run():
    print("P2 Theorem 3 -- hypothesis-free recovery on P1's contact-plane platform")
    _self_check()
    part_a()
    part_b()
    part_c1()
    part_c2()
    part_c3()
    part_d()


if __name__ == "__main__":
    run()
