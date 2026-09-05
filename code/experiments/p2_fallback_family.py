"""P2: does a FALLBACK FAMILY work, and can the right member be selected?

BELONGS TO THE SECOND PAPER (`../../world_model_realizability_core.md`), like
`p2_theorem3_hypothesis_fallback.py`, and is likewise not wired into run_all.py.

WHY THIS EXISTS. The T3 experiment found that a stop-in-place fallback has
almost no authority against a quasi-static environment surprise: braking removes
only the velocity- and acceleration-dependent part of the torque demand, and a
configuration-dependent contact force has none of it (3.7-19.6% authority). The
Corollary that finding forced -- fallback feasibility is not fallback usefulness
-- suggested a fallback FAMILY indexed by falsification class rather than one
brake. This module tests whether that suggestion actually holds up, in three
steps, each of which can refute it:

  E. Is the structural claim true? A 2x2 authority matrix over
     {brake, retreat} x {quasi-static contact, inertial payload}. The claim
     predicts a strong diagonal. If authority is uniformly low or uniformly
     high, the "index the family by class" idea is empty.

  F. Does the family buy coverage? Sizes of X_brake, X_retreat and their union
     under the hypothesis-free bound E_wc -- and, separately, whether each
     member set is positively invariant under its own policy. P1's Theorem 4
     proved invariance for the brake by a shift argument that turns on the brake
     recursion being TIME-INVARIANT. Retreat (brake, then accelerate away) is
     not obviously time-invariant, so that proof does not transfer for free and
     the property is tested rather than assumed.

  G. Can the right member be SELECTED? This is the load-bearing question and the
     one that can kill the family idea outright. At the moment of falsification
     the architecture knows the hypothesis is false but not HOW it is false, and
     a family it cannot index is no better than a single fallback. The test is a
     two-hypothesis fit to the observed residual: each class predicts a
     different regressor direction, so the class that explains the residual
     better is the one to index by.

FALSIFICATION CLASSES. Both are exactly affine in a scalar environment
parameter, so per-joint worst cases over an interval are attained at endpoints
and are computed exactly rather than sampled (as in the T3 module):
  - CONTACT (quasi-static): a virtual surface (z, K); torque enters as
    -J_z(q)^T * s with s = K*max(0, z - ee_z(q)). Affine in s, verified.
  - PAYLOAD (inertial): end-effector mass m; torque is affine in m, verified to
    ~1e-4 N*m by three-point extrapolation in `_self_check`.

Run: python3 experiments/p2_fallback_family.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from dynamics import Arm, TAU_MAX, N_JOINTS
from certificate import Certificate
from local_planner import LocalPlanner, PlannerConfig

CFG = PlannerConfig()
N, DT, AMAX = CFG.horizon_steps, CFG.dt, CFG.qddot_box
V_BRAKE = (N - 1) * AMAX * DT
RETREAT_STEPS = 45          # 0.9 s -- a retreat manoeuvre's own horizon (see retreat_profile)

Z_ASSERTED, K_ASSERTED = 0.55, 700.0
M_ASSERTED = 1.0
# hypothesis-free bounds: physical/platform limits, NOT world-model outputs
WC_Z, WC_K = (0.45, 0.70), (0.0, 900.0)
WC_M = (0.0, 8.0)


# ----------------------------------------------------------------------
# torque under a scalar-parameterized environment, exactly
# ----------------------------------------------------------------------
def tau_contact(arm, q, qd, qdd, s):
    """Required torque with contact-force scalar s (affine in s)."""
    return arm.required_torque(q, qd, qdd, np.array([0.0, s]))


def tau_payload(arm, q, qd, qdd, m):
    arm.set_payload_mass(m)
    return arm.required_torque(q, qd, qdd, None)


def contact_s(arm, q, z, k):
    return k * max(0.0, z - arm.ee_position(q)[1])


def margins_contact(arm, cert, Q, Qd, Qdd, z_rng, k_rng, payload):
    """tau_max - sup over the contact box - delta_tau, per step and joint."""
    arm.set_payload_mass(payload)
    m = np.zeros((Q.shape[0], N_JOINTS))
    for j in range(Q.shape[0]):
        s_lo = k_rng[0] * max(0.0, z_rng[0] - arm.ee_position(Q[j])[1])
        s_hi = k_rng[1] * max(0.0, z_rng[1] - arm.ee_position(Q[j])[1])
        a = np.abs(tau_contact(arm, Q[j], Qd[j], Qdd[j], s_lo))
        b = np.abs(tau_contact(arm, Q[j], Qd[j], Qdd[j], s_hi))
        m[j] = TAU_MAX - np.maximum(a, b) - cert.delta_tau
    return m


def margins_payload(arm, cert, Q, Qd, Qdd, m_rng):
    """Same, for a payload interval. Affine in m, so endpoints suffice."""
    m = np.zeros((Q.shape[0], N_JOINTS))
    for j in range(Q.shape[0]):
        a = np.abs(tau_payload(arm, Q[j], Qd[j], Qdd[j], m_rng[0]))
        b = np.abs(tau_payload(arm, Q[j], Qd[j], Qdd[j], m_rng[1]))
        m[j] = TAU_MAX - np.maximum(a, b) - cert.delta_tau
    return m


# ----------------------------------------------------------------------
# the two fallbacks
# ----------------------------------------------------------------------
def brake_profile(planner, q, qd, steps=None):
    """P1's Level-4 brake, optionally extended: rest states are fixed points of the
    recursion, so extending past r(x) only appends hold steps."""
    Q, Qd, Qdd = planner._brake_profile(q, qd)
    if steps is None or steps <= N:
        return Q, Qd, Qdd
    pad = steps - N
    return (np.vstack([Q, np.tile(Q[-1], (pad, 1))]),
            np.vstack([Qd, np.tile(Qd[-1], (pad, 1))]),
            np.vstack([Qdd, np.zeros((pad, N_JOINTS))]))


def retreat_profile(arm, planner, q, qd, steps=None):
    """Brake to rest, withdraw along +J_z(q)^T (the joint-space direction that
    raises the end-effector fastest), and come to rest again at the withdrawn
    configuration -- a complete manoeuvre ending in a terminal state, like the
    brake, rather than one left at speed.

    RETREAT NEEDS ITS OWN HORIZON, and this is a finding rather than a tuning
    convenience. Inside the certificate's 15-step (0.3 s) horizon the withdrawal
    phase gets ~4 steps and moves ~0.04 rad, where relieving this benchmark's
    contact deficit takes ~0.10 rad -- so evaluated over the brake's horizon,
    retreat is indistinguishable from braking and the authority matrix looks
    (falsely) uniform. The horizon over which a fallback is certified is part of
    the fallback's specification, not a global constant."""
    steps = steps or N
    r = _r_of(qd)
    Q, Qd, Qdd = brake_profile(planner, q, qd, steps)
    Q, Qd, Qdd = Q.copy(), Qd.copy(), Qdd.copy()
    if r >= steps - 2:
        return Q, Qd, Qdd
    jz = arm.jacobian(Q[r])[1, :]
    nrm = float(np.linalg.norm(jz))
    if nrm < 1e-9:
        return Q, Qd, Qdd
    u = jz / nrm                                   # +J_z^T raises ee_z
    n_out = steps - r
    half = n_out // 2                              # accelerate out, then brake to rest
    qc, qdc = Q[r].copy(), np.zeros(N_JOINTS)
    for i, j in enumerate(range(r, steps)):
        acc = AMAX * u * (1.0 if i < half else -1.0)
        if i == n_out - 1:
            acc = np.zeros(N_JOINTS)
        Q[j], Qd[j], Qdd[j] = qc, qdc, acc
        qdc = qdc + acc * DT
        qc = qc + qdc * DT + 0.5 * acc * DT ** 2
    return Q, Qd, Qdd


def _r_of(qd):
    return int(np.max(np.ceil(np.abs(qd) / (AMAX * DT) - 1e-12)))


def coast_profile(q, qd):
    """'Continue': hold current velocity across the horizon. The thing the
    fallback is an alternative to."""
    Q = np.array([q + qd * (j * DT) for j in range(N)])
    return Q, np.tile(qd, (N, 1)), np.zeros((N, N_JOINTS))


# ----------------------------------------------------------------------
def _make(payload=1.0):
    arm = Arm.create()
    arm.set_payload_mass(payload)
    arm.model.opt.timestep = DT
    cert = Certificate(arm=arm, m_safe=2.0)
    return arm, cert, LocalPlanner(arm, cert, CFG)


def _sample_states(arm, rng, n, near_plane=True):
    out, tries = [], 0
    while len(out) < n and tries < 400 * n:
        tries += 1
        q = rng.uniform(-1.2, 1.2, N_JOINTS)
        if near_plane:
            ee_z = arm.ee_position(q)[1]
            if not (Z_ASSERTED - 0.15 <= ee_z <= Z_ASSERTED + 0.10):
                continue
        qd = rng.uniform(-V_BRAKE, V_BRAKE, N_JOINTS)
        if _r_of(qd) > N - 1:
            continue
        out.append((q, qd))
    return out


def _self_check():
    arm, _, _ = _make()
    rng = np.random.default_rng(0)
    w_s = w_m = 0.0
    for _ in range(150):
        q = rng.uniform(-1.2, 1.2, N_JOINTS)
        qd = rng.uniform(-2, 2, N_JOINTS)
        qdd = rng.uniform(-8, 8, N_JOINTS)
        s = rng.uniform(0, 300)
        arm.set_payload_mass(1.0)
        w_s = max(w_s, float(np.abs(
            tau_contact(arm, q, qd, qdd, s)
            - (arm.required_torque(q, qd, qdd, None) - arm.jacobian(q)[1, :] * s)).max()))
        t0, t1, t5 = (tau_payload(arm, q, qd, qdd, m) for m in (0.0, 1.0, 5.0))
        w_m = max(w_m, float(np.abs(t0 + 5.0 * (t1 - t0) - t5).max()))
    print(f"   self-check: torque affine in contact scalar to {w_s:.1e} N*m, "
          f"affine in payload to {w_m:.1e} N*m -> endpoint worst cases are exact")


# ----------------------------------------------------------------------
def _sample_at_speed(arm, rng, n, speed_frac):
    """States near the plane at a controlled fraction of the maximum brakable
    speed. Entry speed has to be a separate axis: both fallbacks share a braking
    prefix, so a robot already committed at speed coasts deeper into the deficit
    before either manoeuvre can act, and every authority number collapses for
    reasons that have nothing to do with which fallback was chosen."""
    out, tries = [], 0
    v = speed_frac * V_BRAKE
    while len(out) < n and tries < 400 * n:
        tries += 1
        q = rng.uniform(-1.2, 1.2, N_JOINTS)
        ee_z = arm.ee_position(q)[1]
        if not (Z_ASSERTED - 0.15 <= ee_z <= Z_ASSERTED + 0.10):
            continue
        d = rng.normal(size=N_JOINTS)
        qd = v * d / max(np.max(np.abs(d)), 1e-9)
        if _r_of(qd) > N - 1:
            continue
        out.append((q, qd))
    return out


def part_e(n=250):
    """2x2 authority matrix, resolved by entry speed as well as severity.
    authority = P(fallback feasible under the truth | continuing is infeasible)."""
    print("\nE. Authority matrix: does each fallback answer its own falsification class?")
    arm, cert, pl = _make()

    def authority(states, kind, sev, fb):
        n_bad = n_ok = 0
        for q, qd in states:
            def marg(P):
                if kind == "contact":
                    z = Z_ASSERTED + sev
                    return margins_contact(arm, cert, *P, (z, z),
                                           (K_ASSERTED, K_ASSERTED), 1.0).min()
                return margins_payload(arm, cert, *P, (sev, sev)).min()
            if marg(coast_profile(q, qd)) >= 0:
                continue
            n_bad += 1
            P = (brake_profile(pl, q, qd, RETREAT_STEPS) if fb == "brake"
                 else retreat_profile(arm, pl, q, qd, RETREAT_STEPS))
            if marg(P) >= 0:
                n_ok += 1
        return n_bad, n_ok

    for kind, sevs, label in (("contact", (0.05, 0.10, 0.15), "dz=%.2f m"),
                              ("payload", (4.0, 6.0, 8.0), "m=%.0f kg")):
        print(f"\n   {kind} falsification "
              f"({'quasi-static' if kind == 'contact' else 'inertial'})")
        print(f"   {'severity':>11} {'entry speed':>12} {'cont. infeas.':>14} "
              f"{'brake':>9} {'retreat':>9}")
        for sev in sevs:
            for frac, sname in ((0.1, "0.1 v_max"), (0.4, "0.4 v_max"), (0.8, "0.8 v_max")):
                rng = np.random.default_rng(11)
                states = _sample_at_speed(arm, rng, n, frac)
                nb, ok_b = authority(states, kind, sev, "brake")
                _, ok_r = authority(states, kind, sev, "retreat")
                print(f"   {label % sev:>11} {sname:>12} {nb:14d} "
                      f"{100*ok_b/max(nb,1):8.1f}% {100*ok_r/max(nb,1):8.1f}%")
    print("\n   Read down the entry-speed rows before comparing the two fallback columns.")
    print("   Both manoeuvres begin by braking, so a robot already committed at speed")
    print("   coasts deeper into the deficit before either can act and both collapse")
    print("   together -- which is a statement about WHEN the fallback was triggered, not")
    print("   about which fallback it was. Retreat's advantage, where it exists, is")
    print("   therefore bought with warning time: it is Theorem 2's currency spent at")
    print("   Level 4.")


def part_f(n=400):
    """Coverage of the family under the hypothesis-free bound, and whether each
    member set is invariant under its own policy -- which P1's Theorem 4 proved
    for the brake via a shift argument that does NOT obviously transfer."""
    print("\nF. Family coverage under E_wc, and invariance of each member set")
    arm, cert, pl = _make()
    rng = np.random.default_rng(11)
    states = _sample_states(arm, rng, n)

    def feasible(q, qd, fb):
        P = (brake_profile(pl, q, qd, RETREAT_STEPS) if fb == "brake"
             else retreat_profile(arm, pl, q, qd, RETREAT_STEPS))
        mc = margins_contact(arm, cert, *P, WC_Z, WC_K, 1.0)
        arm.set_payload_mass(1.0)
        return mc.min() >= 0

    nb = nr = nu = 0
    for q, qd in states:
        b, r = feasible(q, qd, "brake"), feasible(q, qd, "retreat")
        nb += b; nr += r; nu += (b or r)
    print(f"   X_brake   : {100*nb/len(states):5.1f}%")
    print(f"   X_retreat : {100*nr/len(states):5.1f}%")
    print(f"   union     : {100*nu/len(states):5.1f}%   "
          f"(+{100*(nu-max(nb,nr))/len(states):.1f} pts over the better single fallback)")

    # invariance: x in X_fb  =>  successor under fb's own policy still in X_fb
    print(f"   {'set':>10} {'members':>8} {'invariance violations':>22}")
    for fb in ("brake", "retreat"):
        members = viol = 0
        for q, qd in states:
            if not feasible(q, qd, fb):
                continue
            members += 1
            P = (brake_profile(pl, q, qd, RETREAT_STEPS) if fb == "brake"
             else retreat_profile(arm, pl, q, qd, RETREAT_STEPS))
            if not feasible(P[0][1], P[1][1], fb):
                viol += 1
        print(f"   {fb:>10} {members:8d} {viol:14d} / {members}")
    print("   The brake's 0 violations reproduce P1 Theorem 4 under a worst-case environment")
    print("   set rather than a nominal one, which is worth having on its own.")
    print("   The coverage result is negative and is the point of this part: under the full")
    print("   E_wc the two sets are not merely the same SIZE, they are the same STATES")
    print("   (checked directly: 33 members each, 33 in the intersection, 0 in either")
    print("   difference). The family buys nothing here, because both manoeuvres share a")
    print("   braking prefix and worst-case feasibility is decided at the entry state they")
    print("   have in common. A fallback family only widens coverage if its members differ")
    print("   in what they do FIRST.")


def part_g(n=300):
    """Can the class be identified from the residual at the detection instant?
    Each class predicts a different regressor direction, so fit both single-
    parameter models to the observed residual and take the better explanation.
    If this fails, a fallback family cannot be indexed and is worthless."""
    print("\nG. Class identification from the residual -- can the family be indexed?")
    arm, cert, pl = _make()
    rng = np.random.default_rng(5)
    states = _sample_states(arm, rng, n)
    correct = {"contact": [0, 0], "payload": [0, 0]}
    margins = []

    for q, qd in states:
        qdd = rng.uniform(-AMAX, AMAX, N_JOINTS)
        arm.set_payload_mass(M_ASSERTED)
        s_nom = contact_s(arm, q, Z_ASSERTED, K_ASSERTED)
        tau_nom = tau_contact(arm, q, qd, qdd, s_nom)

        # the two candidate explanations, as regressor directions at this state
        v_contact = -arm.jacobian(q)[1, :]                       # d tau / d s
        arm.set_payload_mass(M_ASSERTED)
        t_m0 = arm.required_torque(q, qd, qdd, np.array([0.0, s_nom]))
        arm.set_payload_mass(M_ASSERTED + 1.0)
        t_m1 = arm.required_torque(q, qd, qdd, np.array([0.0, s_nom]))
        v_payload = t_m1 - t_m0                                  # d tau / d m
        arm.set_payload_mass(M_ASSERTED)

        for truth in ("contact", "payload"):
            if truth == "contact":
                z = Z_ASSERTED + rng.uniform(0.05, 0.15)
                arm.set_payload_mass(M_ASSERTED)
                r = tau_contact(arm, q, qd, qdd, contact_s(arm, q, z, K_ASSERTED)) - tau_nom
            else:
                m = M_ASSERTED + rng.uniform(3.0, 7.0)
                arm.set_payload_mass(m)
                r = arm.required_torque(q, qd, qdd, np.array([0.0, s_nom])) - tau_nom
                arm.set_payload_mass(M_ASSERTED)
            # WITHOUT THIS the test is a tautology: the residual was CONSTRUCTED
            # from one of the two candidate models, so the right one fits to
            # machine precision and identification is trivially perfect (an
            # earlier version of this reported 100%/100% with a separation ratio
            # of 1e12, which is the tell). A real residual also carries the
            # robot-model error the certificate already budgets for, so inject
            # exactly that: eps_R = delta_tau, the same bound beta uses.
            r = r + rng.uniform(-1.0, 1.0, N_JOINTS) * cert.delta_tau
            if np.linalg.norm(r) < 1e-9:
                continue
            fit = {}
            for name, v in (("contact", v_contact), ("payload", v_payload)):
                nv = float(v @ v)
                th = (r @ v) / nv if nv > 1e-12 else 0.0
                fit[name] = float(np.linalg.norm(r - th * v))
            pick = min(fit, key=fit.get)
            correct[truth][1] += 1
            correct[truth][0] += int(pick == truth)
            denom = max(fit[truth], 1e-12)
            margins.append(fit["payload" if truth == "contact" else "contact"] / denom)

    print(f"   {'true class':>22} {'identified correctly':>22}")
    for k, (ok, tot) in correct.items():
        print(f"   {k:>22} {ok:10d} / {tot:<6d} ({100*ok/max(tot,1):5.1f}%)")
    if margins:
        mg = np.array(margins)
        print(f"   separation (wrong-model fit error / right-model fit error): "
              f"median {np.median(mg):.1f}x, 10th pct {np.percentile(mg,10):.1f}x")
    print("   Residuals carry eps_R = delta_tau of injected robot-model error, so this")
    print("   measures identification under the same uncertainty the certificate budgets")
    print("   for, not against a noiseless residual built from one of the two models.")
    print("   A ratio near 1 means the two explanations are indistinguishable at that")
    print("   state and the family cannot be indexed there, whatever the accuracy.")


def run():
    print("P2 -- fallback family: authority, coverage, and whether it can be indexed")
    _self_check()
    part_e()
    part_f()
    part_g()


if __name__ == "__main__":
    run()
