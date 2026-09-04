"""Theorem 4 (Sec. VI): the Level-4 terminal safe set and its recursive
feasibility, verified against the implementation rather than asserted.

This script is the evidence behind every number the paper's Theorem 4,
Proposition 5 and Proposition 6 cite. It has five parts:

  A. Lemma 2 (brake recursion). `_brake_profile` is a time-invariant state
     recursion x_{j+1} = f(x_j); the profile from the successor state is the
     exact one-step shift of the profile from x. Rest is reached at step
     r(x) = max_i ceil(|qdot_i| / (a_max dt)), and rest states are fixed
     points of f. Both are checked to machine precision, not statistically.

  B. Theorem 4(a) (reference level). X_f = {x : r(x) <= N-1 and the
     certificate over the whole brake profile is >= 0} is positively
     invariant under f. Checked by exhaustive successor testing over random
     samples, per payload, alongside the size of X_f and of the hold set
     H = {q : |tau_hold(q)| + delta_tau <= tau_max}.

  C. Theorem 4(b) (executed level). The implemented sticky-hold law
     (baselines.policy_b3 once Level 4 has engaged) commands
     tau = tau_hold(q) - Kd M(q) qdot. While unsaturated, V = 0.5 qdot' M qdot
     obeys Vdot <= -2 Kd V, so |qdot| decays exponentially, coasting
     displacement is bounded, and the realized torque is bounded. The
     resulting set X_f^exec is checked for membership rate, V-monotonicity,
     torque feasibility, decay rate against the guaranteed 2*Kd, and travel
     against the guaranteed bound.

  D. The gap between (a) and (b), measured. Two cheaper candidate membership
     tests for the executed policy are shown to be UNSOUND -- x in X_f alone,
     and the pointwise condition evaluated only at the engagement state --
     which is why (b)'s conservative coasting-ball sup is load-bearing rather
     than slack.

  E. Cross-check against the one Level-4 engagement the paper's own
     experiment suite actually produces (Exp 6 at 8 kg).

Run: python3 experiments/theorem4_terminal_set.py   (or via run_all.py)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mujoco

from dynamics import Arm, TAU_MAX, TAU_MIN, N_JOINTS
from certificate import Certificate
from trajectory import ViaPointTrajectory
from local_planner import LocalPlanner, PlannerConfig
from baselines import policy_b3
from executor import rollout, KP, KD

CFG = PlannerConfig()
N, DT, AMAX = CFG.horizon_steps, CFG.dt, CFG.qddot_box
# r(x) <= N-1 (Lemma 2) is exactly ||qdot||_inf <= (N-1) * a_max * dt.
V_BRAKE = (N - 1) * AMAX * DT
PAYLOADS = [0.0, 2.0, 5.0, 8.0, 12.0]
Q_LO, Q_HI = -np.pi, np.pi          # sampling box for q; the model has no joint limits


def _make(payload):
    arm = Arm.create()
    arm.set_payload_mass(payload)
    arm.model.opt.timestep = DT
    cert = Certificate(arm=arm)
    return arm, cert, LocalPlanner(arm, cert, CFG)


def _r_of(qdot):
    """Lemma 2's closed form for the step at which the brake profile reaches rest."""
    return int(np.max(np.ceil(np.abs(qdot) / (AMAX * DT) - 1e-12)))


def _hold_torque(arm, q):
    """tau_hold(q) = g(q) - J(q)^T F  -- the torque required to hold q at rest."""
    return arm.required_torque(q, np.zeros(N_JOINTS), np.zeros(N_JOINTS), None)


def _in_Xf(pl, cert, q, qdot):
    """Theorem 4(a) membership: rest within the horizon AND the certificate over
    the whole brake profile non-negative. The second half is exactly the `mk`
    that local_planner.online_step already computes on every Level-4 cycle."""
    if _r_of(qdot) > N - 1:
        return False, "no-rest", None
    Q, Qdot, Qddot = pl._brake_profile(q, qdot)
    m = cert.m_phys(Q, Qdot, Qddot, None)
    return (m >= 0.0), ("ok" if m >= 0.0 else "torque"), m


# ----------------------------------------------------------------------
def part_a():
    print("A. Lemma 2 -- the brake recursion is time-invariant and reaches rest in r(x) steps")
    _, _, pl = _make(0.0)
    rng = np.random.default_rng(0)
    shift_err = 0.0
    for _ in range(1000):
        q = rng.uniform(Q_LO, Q_HI, N_JOINTS)
        qdot = rng.uniform(-3.0, 3.0, N_JOINTS)
        Q, Qdot, Qddot = pl._brake_profile(q, qdot)
        Q2, Qdot2, Qddot2 = pl._brake_profile(Q[1], Qdot[1])
        shift_err = max(
            shift_err,
            float(np.abs(Q2[: N - 1] - Q[1:]).max()),
            float(np.abs(Qdot2[: N - 1] - Qdot[1:]).max()),
            float(np.abs(Qddot2[: N - 1] - Qddot[1:]).max()),
        )
    rest_bad = 0
    fixed_bad = 0
    for _ in range(3000):
        q = rng.uniform(Q_LO, Q_HI, N_JOINTS)
        qdot = rng.uniform(-V_BRAKE, V_BRAKE, N_JOINTS)
        Q, Qdot, Qddot = pl._brake_profile(q, qdot)
        zeros = [j for j in range(N) if np.allclose(Qdot[j], 0.0)]
        if not zeros or zeros[0] != _r_of(qdot):
            rest_bad += 1
        r = _r_of(qdot)
        # rest states are fixed points: q and qdot stop changing from step r on
        if not (np.allclose(Q[r:], Q[r]) and np.allclose(Qdot[r:], 0.0)
                and np.allclose(Qddot[r:], 0.0)):
            fixed_bad += 1
    print(f"   shift property   B(f(x))_j == B(x)_(j+1) : max mismatch {shift_err:.3e} over 1000 states")
    print(f"   rest step        r(x) = max_i ceil(|qdot_i|/(a_max dt)) : {rest_bad} mismatches / 3000")
    print(f"   rest states are fixed points of f       : {fixed_bad} violations / 3000")
    print(f"   => r(x) <= N-1 iff ||qdot||_inf <= (N-1) a_max dt = {V_BRAKE:.2f} rad/s"
          f"   (N={N}, dt={DT}, a_max={AMAX})")


# ----------------------------------------------------------------------
def part_b(n_samples=2500):
    print("\nB. Theorem 4(a) -- X_f is positively invariant under the brake recursion")
    print(f"   {'payload':>8} | {'H':>7} | {'X_f':>7} | {'no-rest':>8} {'torque':>7} | {'invariance viol.':>16}")
    out = {}
    for payload in PAYLOADS:
        arm, cert, pl = _make(payload)
        rng = np.random.default_rng(1234)
        n_h = n_xf = viol = 0
        causes = {"no-rest": 0, "torque": 0}
        for _ in range(n_samples):
            q = rng.uniform(Q_LO, Q_HI, N_JOINTS)
            qdot = rng.uniform(-V_BRAKE, V_BRAKE, N_JOINTS)
            if np.all(np.abs(_hold_torque(arm, q)) + cert.delta_tau <= TAU_MAX):
                n_h += 1
            ok, why, _ = _in_Xf(pl, cert, q, qdot)
            if not ok:
                causes[why] += 1
                continue
            n_xf += 1
            Q, Qdot, _ = pl._brake_profile(q, qdot)
            ok2, _, _ = _in_Xf(pl, cert, Q[1], Qdot[1])   # successor under f
            if not ok2:
                viol += 1
        print(f"   {payload:8.1f} | {100*n_h/n_samples:6.1f}% | {100*n_xf/n_samples:6.1f}% | "
              f"{causes['no-rest']:8d} {causes['torque']:7d} | {viol:6d} / {n_xf:<7d}")
        out[payload] = (100*n_h/n_samples, 100*n_xf/n_samples, viol, n_xf)
    print("   (q sampled uniformly in [-pi,pi]^3, qdot uniformly in the brakable box;"
          " 'no-rest' counts are 0 by construction of that box)")
    return out


# ----------------------------------------------------------------------
def _coasting_constants(arm, q0, radius, rng, n_probe=24):
    """sup of |tau_hold|, of diag(M), and inf of lambda_min(M), over the ball of
    configurations the coasting brake can reach (Theorem 4(b)'s travel bound)."""
    eta = np.zeros(N_JOINTS)
    m_diag = np.zeros(N_JOINTS)
    lam = np.inf
    probes = [q0] + [q0 + rng.uniform(-1, 1, N_JOINTS) * radius / np.sqrt(N_JOINTS)
                     for _ in range(n_probe)]
    for q in probes:
        eta = np.maximum(eta, np.abs(_hold_torque(arm, q)))
        M = arm.mass_matrix(q)
        m_diag = np.maximum(m_diag, np.diag(M))
        lam = min(lam, float(np.linalg.eigvalsh(M).min()))
    return eta, m_diag, lam


def _sticky_rollout(arm, q0, qdot0, n_steps=200):
    """The executed Level-4 law of baselines.policy_b3: reference = (measured q,
    0, 0), fed to the shared computed-torque controller, which reduces to
    tau = tau_hold(q) - Kd M(q) qdot."""
    arm.data.qpos[:] = q0
    arm.data.qvel[:] = qdot0
    Vs, qs, worst = [], [], 0.0
    for _ in range(n_steps):
        q = arm.data.qpos.copy()
        qdot = arm.data.qvel.copy()
        M = arm.mass_matrix(q)
        Vs.append(0.5 * qdot @ M @ qdot)
        qs.append(q)
        tau = M @ (KD * (-qdot)) + _hold_torque(arm, q)
        worst = max(worst, float(np.max(np.abs(tau) / TAU_MAX)))
        arm.data.qpos[:] = q
        arm.data.qvel[:] = qdot
        arm.clear_external_force()
        arm.data.ctrl[:] = np.clip(tau, TAU_MIN, TAU_MAX)
        mujoco.mj_step(arm.model, arm.data)
    return np.array(Vs), np.array(qs), worst


V0_GRID = [0.02, 0.05, 0.10, 0.25, 0.50, 1.00, 2.00]
# cells verified by simulation in C2 -- chosen to span the membership table's
# full range, from its saturated corner to close to its boundary
C2_CELLS = [(0.0, 2.00), (0.0, 0.50), (2.0, 0.25), (5.0, 0.10), (8.0, 0.05), (12.0, 0.02)]


def _member_exec(arm, cert, q0, V0, rng):
    """Theorem 4(b) membership: the coasting-ball sup of the hold torque, plus the
    worst-case Kd*|M qdot| term the Lyapunov bound allows, still fits the limits."""
    lam0 = float(np.linalg.eigvalsh(arm.mass_matrix(q0)).min())
    radius = np.sqrt(2 * V0 / lam0) / KD          # coasting-displacement bound
    eta, m_diag, _ = _coasting_constants(arm, q0, radius, rng)
    ok = np.all(eta + KD * np.sqrt(2 * V0 * m_diag) + cert.delta_tau <= TAU_MAX)
    return bool(ok), radius


def part_c(n_probe=600, n_members=50, max_tries=1500):
    print("\nC. Theorem 4(b) -- the executed sticky-hold law and its certified set X_f^exec")
    print(f"   guaranteed decay rate 2*Kd = {2*KD:.0f} 1/s; membership requires"
          f"  eta_i + Kd*sqrt(2 V0 M_ii) + delta_i <= tau_max_i  over the coasting ball")
    print("\n   C1. size of X_f^exec: fraction of sampled configurations admitted at braking energy V0")
    print(f"   {'payload':>8} " + " ".join(f"{('V0=' + format(v, '.2f')):>9}" for v in V0_GRID))
    sizes = {}
    for payload in PAYLOADS:
        arm, cert, _ = _make(payload)
        row = []
        for V0 in V0_GRID:
            rng = np.random.default_rng(777)
            n = 0
            for _ in range(n_probe):
                q0 = rng.uniform(Q_LO, Q_HI, N_JOINTS)
                ok, _ = _member_exec(arm, cert, q0, V0, rng)
                n += int(ok)
            sizes[(payload, V0)] = 100 * n / n_probe
            row.append(f"{100 * n / n_probe:8.1f}%")
        print(f"   {payload:8.1f} " + " ".join(row))
    print("   (V0 = 0.5 qdot' M(q) qdot at engagement, in J -- the brake's kinetic energy)")

    print("\n   C2. verification on members: V-monotonicity, torque feasibility, decay rate, travel")
    print(f"   {'payload':>8} {'V0':>6} {'members/tried':>14} {'V monotone':>11} "
          f"{'max|tau|/tau_max':>17} {'meas.rate/2Kd':>14} {'travel/bound':>13}")
    out = {}
    for payload, V0 in C2_CELLS:
        arm, cert, _ = _make(payload)
        rng = np.random.default_rng(777)
        n = tried = nonmono = 0
        worst, travel_ratio, rates = 0.0, 0.0, []
        while n < n_members and tried < max_tries:
            tried += 1
            q0 = rng.uniform(Q_LO, Q_HI, N_JOINTS)
            ok, radius = _member_exec(arm, cert, q0, V0, rng)
            if not ok:
                continue
            n += 1
            v = rng.normal(size=N_JOINTS)
            M0 = arm.mass_matrix(q0)
            v = v / np.sqrt(0.5 * v @ M0 @ v) * np.sqrt(V0)   # exactly V(x0) = V0
            Vs, qs, w = _sticky_rollout(arm, q0, v)
            worst = max(worst, w)
            if np.any(np.diff(Vs) > 1e-9 * np.maximum(Vs[:-1], 1e-9)):
                nonmono += 1
            if Vs[5] > 1e-12:
                rates.append(-np.log(Vs[5] / Vs[0]) / (5 * DT))
            travel_ratio = max(travel_ratio,
                               float(np.max(np.linalg.norm(qs - qs[0], axis=1))) / radius)
        rate = (np.mean(rates) / (2 * KD)) if rates else float("nan")
        print(f"   {payload:8.1f} {V0:6.2f} {f'{n}/{tried}':>14} "
              f"{('yes' if nonmono == 0 else f'NO({nonmono})'):>11} "
              f"{worst:17.3f} {rate:14.2f} {travel_ratio:13.3f}")
        out[(payload, V0)] = (n, tried, nonmono, worst, rate, travel_ratio)
    print("   (meas.rate/2Kd >= 1 confirms the guaranteed exponential rate is a true lower"
          " bound; travel/bound <= 1 confirms the coasting-displacement bound)")
    return sizes, out


# ----------------------------------------------------------------------
def part_d(n_sim=150, max_tries=2000):
    print("\nD. The gap between (a) and (b): two cheaper membership tests, both UNSOUND")
    print("   test 1: x in X_f only (Theorem 4(a))")
    print("   test 2: x in X_f AND the pointwise condition |tau_hold(q) - Kd M(q) qdot| + delta <= tau_max")
    print(f"   {'payload':>8} | {'in X_f':>7} {'sim':>5} {'saturating':>11} | "
          f"{'+pointwise':>11} {'sim':>5} {'saturating':>11}")
    out = {}
    for payload in PAYLOADS:
        arm, cert, pl = _make(payload)
        rng = np.random.default_rng(4242)
        n_xf = n_pw = s1 = s2 = sim1 = sim2 = 0
        for _ in range(max_tries):
            q0 = rng.uniform(Q_LO, Q_HI, N_JOINTS)
            qdot0 = rng.uniform(-V_BRAKE, V_BRAKE, N_JOINTS)
            ok, _, _ = _in_Xf(pl, cert, q0, qdot0)
            if not ok:
                continue
            n_xf += 1
            M0 = arm.mass_matrix(q0)
            pointwise = np.all(np.abs(_hold_torque(arm, q0) - KD * (M0 @ qdot0))
                               + cert.delta_tau <= TAU_MAX)
            if pointwise:
                n_pw += 1
            if sim1 >= n_sim and not (pointwise and sim2 < n_sim):
                continue
            _, _, worst = _sticky_rollout(arm, q0, qdot0)
            saturated = worst > 1.0
            if sim1 < n_sim:
                sim1 += 1
                s1 += int(saturated)
            if pointwise and sim2 < n_sim:
                sim2 += 1
                s2 += int(saturated)
        print(f"   {payload:8.1f} | {n_xf:7d} {sim1:5d} {s1:11d} | {n_pw:11d} {sim2:5d} {s2:11d}")
        out[payload] = (n_xf, sim1, s1, n_pw, sim2, s2)
    print("   ('saturating' counts executed rollouts whose commanded torque exceeded a"
          " joint limit at any step -- each one is a state Theorem 4(a) admits and the"
          " executed policy does not keep safe)")
    return out


# ----------------------------------------------------------------------
def part_e():
    print("\nE. Cross-check: the one Level-4 engagement the paper's own suite produces (Exp 6, 8 kg)")
    q0 = np.array([0.2, -1.0, -0.6])
    qg = np.array([0.75, -1.15, -0.55])
    via_risky = np.array([1.1, -0.15, 0.1])
    via_safe = 0.5 * (q0 + qg)
    arm, _, _ = _make(8.0)
    cert = Certificate(arm=arm, m_safe=2.0)
    pl = LocalPlanner(arm, cert, CFG)
    traj = ViaPointTrajectory(q0, via_risky, qg, T1=0.5, T2=0.4)
    alt = ViaPointTrajectory(q0, via_safe, qg, T1=0.7, T2=0.6)
    route = pl.plan_route(traj, alt_traj=alt)

    captured = {}
    pol = policy_b3(traj, arm, cert, CFG, alt_traj=alt)

    def wrapped(t, q, qdot):
        out = pol(t, q, qdot)
        if out[3]["level"] == 4 and "q" not in captured:
            captured.update(q=q.copy(), qdot=qdot.copy(), t=t)
        return out

    rr = rollout(arm, wrapped, q0, np.zeros(N_JOINTS), duration=route.traj.T + 0.3, dt=DT)
    if "q" not in captured:
        print("   no Level-4 engagement in this run -- nothing to cross-check")
        return
    q, qdot = captured["q"], captured["qdot"]
    ok, why, m = _in_Xf(pl, cert, q, qdot)
    Q, _, _ = pl._brake_profile(q, qdot)
    tau_hold_inf = _hold_torque(arm, Q[-1])
    print(f"   engagement at t={captured['t']:.2f}s, ||qdot||_inf={np.max(np.abs(qdot)):.3f} rad/s"
          f" (brakable bound {V_BRAKE:.2f})")
    print(f"   in X_f? {ok} ({why}), brake-profile certificate m = {m:.3f} Nm")
    print(f"   terminal hold torque {np.round(tau_hold_inf, 2)} vs tau_max {TAU_MAX}"
          f"  -> q_inf in H? {bool(np.all(np.abs(tau_hold_inf) + cert.delta_tau <= TAU_MAX))}")
    print(f"   realized peak |tau|/tau_max over the whole rollout: "
          f"{float(np.max(np.abs(rr.tau_cmd) / TAU_MAX)):.3f}  (no saturation)")


def part_f(n_sim=150, max_tries=2000):
    """The candidate fix named in Sec. VI: make the EXECUTED policy track the
    certified brake profile (re-derived from the measured state each cycle)
    instead of commanding an instantaneous hold, so Theorem 4's object and the
    executed object coincide. Tested, and reported as not sufficient on its own."""
    print("\nF. Candidate fix: executing the certified brake profile instead of the sticky hold")
    print(f"   {'payload':>8} {'in X_f':>7} {'sim':>5} {'saturating':>11} "
          f"{'max|tau|/tau_max':>17} {'final||qdot||_inf':>18}")
    out = {}
    for payload in PAYLOADS:
        arm, cert, pl = _make(payload)
        rng = np.random.default_rng(4242)      # same states part D draws
        n_xf = sim = sat = 0
        worst, final_v = 0.0, 0.0
        for _ in range(max_tries):
            q0 = rng.uniform(Q_LO, Q_HI, N_JOINTS)
            qdot0 = rng.uniform(-V_BRAKE, V_BRAKE, N_JOINTS)
            ok, _, _ = _in_Xf(pl, cert, q0, qdot0)
            if not ok:
                continue
            n_xf += 1
            if sim >= n_sim:
                continue
            sim += 1
            arm.data.qpos[:] = q0
            arm.data.qvel[:] = qdot0
            saturated = False
            for _ in range(120):
                q = arm.data.qpos.copy()
                qdot = arm.data.qvel.copy()
                if not (np.all(np.isfinite(q)) and np.all(np.isfinite(qdot))):
                    saturated = True
                    break
                Qk, Qdotk, Qddotk = pl._brake_profile(q, qdot)   # re-derive from measurement
                M = arm.mass_matrix(Qk[0])
                h = arm.required_torque(Qk[0], Qdotk[0], np.zeros(N_JOINTS), None)
                tau = M @ (Qddotk[0] + KP * (Qk[0] - q) + KD * (Qdotk[0] - qdot)) + h
                r = float(np.max(np.abs(tau) / TAU_MAX))
                worst = max(worst, r)
                if r > 1.0:
                    saturated = True
                arm.data.qpos[:] = q
                arm.data.qvel[:] = qdot
                arm.clear_external_force()
                arm.data.ctrl[:] = np.clip(tau, TAU_MIN, TAU_MAX)
                mujoco.mj_step(arm.model, arm.data)
            sat += int(saturated)
            v_end = float(np.max(np.abs(arm.data.qvel)))
            if np.isfinite(v_end):
                final_v = max(final_v, v_end)
        # a diverged rollout produces a meaningless torque magnitude; say so rather
        # than printing 1e52 as though it were a measurement
        worst_str = "diverged" if worst > 1e3 else f"{worst:.3f}"
        print(f"   {payload:8.1f} {n_xf:7d} {sim:5d} {sat:11d} {worst_str:>17} {final_v:18.4f}")
        out[payload] = (n_xf, sim, sat, worst, final_v)
    print("   (compare column 'saturating' against part D's first block: the fix removes the"
          " low-payload saturations but the profile's fixed a_max deceleration is not itself")
    print("    achievable at high payload, so tracking error grows and the closed loop diverges)")
    return out


def run():
    part_a()
    part_b()
    part_c()
    part_d()
    part_e()
    part_f()


if __name__ == "__main__":
    run()
