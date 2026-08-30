"""The Level 0-4 hierarchical response of paper Sec. V-B, plus the ablation flags
of Sec. VIII-H (A1-A5), all implemented as configuration on a single planner class
so every baseline/ablation shares one code path (the fairness point Sec. VIII-A
insists on for B2 vs B3).

Design note (simplification, documented honestly): route-level decisions --
Level 1 (retime), Level 2 (reshape, ALSO tried once at route-planning time, in
addition to online -- see below), and Level 3 (reroute) -- are resolved once,
as a *planning-time* decision over the full candidate route(s), producing a
(possibly new) active trajectory. Level 4 (brake) is the one response that
only ever makes sense online, every control cycle, against whichever route is
currently active. This split exists because a retimed reference is a full
re-parameterization of the route's time law: correcting it only for a single
instantaneous horizon window (and not persisting the slower schedule into
subsequent cycles) does not actually slow the executed motion down -- an
earlier version of this code made exactly that mistake, silently reference-
tracking a "slowed" qdot/qddot against a q_ref that kept marching forward at
the original pace.

`plan_route`'s ordering is retime, then reshape, then reroute: both retiming
and reshaping are genuine trajectory REGENERATION over the SAME route (same
q0/qf), so both are tried, in that order, before ever escalating to a
DIFFERENT route. This was previously only true of retiming -- reshaping was
tried online only, so a route-level deficit retiming couldn't fix went
straight to reroute even when reshaping the same route might have sufficed
(the gap the paper's own Sec. V-C flags: "closing that expressiveness gap ...
is identified as a concrete next step, not yet implemented"). Route-level
reshape is a MATERIALLY larger QP than the online horizon's (n scales with
route duration / dt, not the fixed online horizon_steps) and OSQP does not
reliably converge on it within a practical iteration budget -- confirmed
empirically, not assumed: a solver fallback to SCS is needed when OSQP hits
its iteration limit (see _try_reshape), and even then this route-level check
costs on the order of hundreds of ms to ~1s in the tested scenarios,
regardless of whether it ultimately succeeds -- a real, disclosed one-time
cost (see README/paper's real-time sections), not a per-cycle one. Reshape
and brake, run purely online, do not have the re-parameterization problem
above (they don't change the time law), so they still run per-cycle as
before.

What Level 3 does and does not do (also flagged in the paper's own Draft
Status, §V-C "open item"): `plan_route` SELECTS between `traj` and an
already-supplied `alt_traj` by their certificates; it does not search for or
generate candidate routes. There is no route generator anywhere in this
codebase. Every caller is responsible for constructing whatever candidates
it wants evaluated (e.g. `trajectory.ViaPointTrajectory` for two routes that
share a start and goal). Read "Level 3 rerouting" in code comments/output as
"certificate-guided selection among caller-supplied candidates," not as a
general replanner.

The Level 2 QP is solved with cvxpy/OSQP over a linear-in-qddot torque model
(tau = M(q) qddot + h(q,qdot), both M and h evaluated at the nominal q,qdot for
the horizon -- the 'fixed structure, state-dependent vector terms' design
principle in the paper's Sec. VI 'Computational structure' paragraph).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import cvxpy as cp

from dynamics import Arm, TAU_MAX, TAU_MIN, N_JOINTS
from certificate import Certificate
from trajectory import JointTrajectory, SampledTrajectory


@dataclass
class PlannerConfig:
    dt: float = 0.02
    horizon_steps: int = 15
    lam_max: float = 4.0
    qddot_box: float = 8.0  # rad/s^2, kinematic bound used by Level-2 QP
    # Level-2 reshape QP objective weights (see _try_reshape): the cost
    # penalizes deviation from the NOMINAL q/qdot/qddot, not raw magnitude, so
    # the QP picks the closest physically realizable trajectory rather than
    # whatever minimizes acceleration. w_acc dominates because the deficit
    # the QP exists to fix is a torque violation, which only qddot controls
    # directly; w_pos/w_vel are smaller secondary terms that discourage
    # compounding position/velocity drift once qddot has some freedom to
    # choose among several feasible corrections.
    reshape_w_acc: float = 1.0
    reshape_w_pos: float = 0.1
    reshape_w_vel: float = 0.1
    predict: bool = True
    act: bool = True
    allow_level1: bool = True
    allow_level2: bool = True
    allow_level3: bool = True
    allow_level4: bool = True


@dataclass
class RouteDecision:
    traj: JointTrajectory
    level: int  # 0 (nominal kept) | 1 (retimed) | 2 (reshaped) | 3 (rerouted)
    m_phys: float
    triggered: bool


@dataclass
class PlanResult:
    level: int
    Q: np.ndarray
    Qdot: np.ndarray
    Qddot: np.ndarray
    m_phys_before: float
    m_phys_after: float
    triggered: bool  # True if sigma_4 (margin below m_safe) fired this cycle


class LocalPlanner:
    def __init__(self, arm: Arm, cert: Certificate, cfg: PlannerConfig):
        self.arm = arm
        self.cert = cert
        self.cfg = cfg

    # ---- planning-time: Level 0 / 1 / 3 -------------------------------
    def plan_route(
        self,
        traj: JointTrajectory,
        alt_traj: Optional[JointTrajectory] = None,
        ee_force_fn=None,
    ) -> RouteDecision:
        """ee_force_fn(t, q) -> force or None. Passing a non-None ee_force_fn here
        means Level 1/2/3 are decided WITH KNOWLEDGE of the whole predicted force/
        contact profile over the route -- appropriate when that profile is part of
        the 'predicted environment E' the paper's Sec. IV certificate is
        conditioned on (Exp 4: a known upcoming contact transition), not when it
        represents a disturbance that should only become visible within a bounded
        online prediction horizon (Exp 3: see online_step instead).

        Ordering: retime, then reshape, then reroute -- both forms of route-
        level REGENERATION (retiming and reshaping) are tried, on the current
        route and then on the alternate, before ever falling back to
        rerouting. This closes the paper's own flagged open item (Sec. V-C):
        the pre-execution adaptation class previously included only retiming,
        so a deficit retiming can't fix (a function of position, not speed --
        _search_retime_whole_route) went straight to reroute even when
        reshaping the SAME route might have sufficed. Checked empirically on
        both scenarios that currently demonstrate Level 3 in this benchmark
        (Exp 5's flagship, Exp 7's environment-conditioned reroute): whole-
        route reshape genuinely cannot restore either one's margin (confirmed
        against an independent solver, not just OSQP's own status flag -- see
        _try_reshape), so Level 3 remains genuinely necessary there, not an
        artifact of never having tried reshape first."""
        cfg = self.cfg
        m0 = self._whole_route_margin(traj, ee_force_fn)

        if not cfg.predict:
            return RouteDecision(traj, 0, np.nan, triggered=False)
        if m0 >= self.cert.m_safe:
            return RouteDecision(traj, 0, m0, triggered=False)
        if not cfg.act:
            return RouteDecision(traj, 0, m0, triggered=True)

        if cfg.allow_level1:
            lam = self._search_retime_whole_route(traj, ee_force_fn)
            if lam is not None:
                retimed = traj.retimed(lam)
                m1 = self._whole_route_margin(retimed, ee_force_fn)
                return RouteDecision(retimed, 1, m1, triggered=True)

        if cfg.allow_level2:
            reshaped = self._search_reshape_whole_route(traj, ee_force_fn)
            if reshaped is not None:
                new_traj, m2 = reshaped
                return RouteDecision(new_traj, 2, m2, triggered=True)

        if cfg.allow_level3 and alt_traj is not None:
            mb = self._whole_route_margin(alt_traj, ee_force_fn)
            if mb >= self.cert.m_safe:
                return RouteDecision(alt_traj, 3, mb, triggered=True)
            if cfg.allow_level1:
                lam_b = self._search_retime_whole_route(alt_traj, ee_force_fn)
                if lam_b is not None:
                    retimed_b = alt_traj.retimed(lam_b)
                    mb1 = self._whole_route_margin(retimed_b, ee_force_fn)
                    return RouteDecision(retimed_b, 3, mb1, triggered=True)
            if cfg.allow_level2:
                reshaped_b = self._search_reshape_whole_route(alt_traj, ee_force_fn)
                if reshaped_b is not None:
                    new_traj_b, mb2 = reshaped_b
                    return RouteDecision(new_traj_b, 3, mb2, triggered=True)

        # Nothing at route level restores feasibility; online Level 2/4 must cope.
        return RouteDecision(traj, 0, m0, triggered=True)

    @staticmethod
    def _sample_forces(ts: np.ndarray, Q: np.ndarray, ee_force_fn):
        if ee_force_fn is None:
            return None
        out = []
        for t, q in zip(ts, Q):
            f = ee_force_fn(t, q)
            out.append([0.0, 0.0] if f is None else f)
        return np.array(out)

    def _whole_route_margin(self, traj: JointTrajectory, ee_force_fn) -> float:
        n = int(np.ceil(traj.T / self.cfg.dt)) + 1
        Q, Qdot, Qddot = traj.sample_horizon(0.0, self.cfg.dt, n)
        ts = self.cfg.dt * np.arange(n)
        forces = self._sample_forces(ts, Q, ee_force_fn)
        return self.cert.m_phys(Q, Qdot, Qddot, forces)

    def _torque_decomposition_whole_route(self, traj: JointTrajectory, ee_force_fn):
        """Quadratic-in-x decomposition, x = 1/lambda: tau_i(lambda) = A_i x^2 +
        D_i x + B_i. This EXTENDS monotonicity_lemma_draft.md Sec. 1-2's
        tau_i(lambda) = A_i/lambda^2 + B_i (derived 'by the manipulator
        equation', i.e. assuming no joint damping/friction) with a linear-in-x
        term D_i, found necessary and confirmed directly against this
        benchmark's actual dynamics: models/planar3r.xml gives every joint real
        viscous damping (dof_damping=0.3), and a linear damping torque -c*qdot
        scales as 1/lambda (=x) under retiming, not 1/lambda^2 -- something the
        idealized manipulator equation's A/lambda^2+B form cannot represent at
        all. Verified empirically: with dof_damping zeroed, the idealized
        2-term model reproduces mj_inverse's actual torque to numerical
        precision at every lambda tested; with damping present (the model this
        benchmark actually uses), the 2-term model is off by several percent,
        growing with lambda -- enough to occasionally flip a margin decision
        near m_safe. This 3-term fit closes that gap to ~0.02 Nm residual,
        consistent with the small, non-quadratic Coulomb friction term
        (dof_frictionloss=0.05) that no closed polynomial form captures
        exactly, but too small to matter operationally the way the damping gap
        did.

        B_i, A_i, D_i are recovered from THREE required_torque evaluations at
        the SAME configuration q with SCALED velocity/acceleration (a real,
        closed-form fit to the known quadratic-in-x functional form, not an
        assumption about the specific passive-force law MuJoCo implements
        internally -- robust to whatever exact numerics mj_inverse uses):
          B_i = tau_i(q, 0, 0)                                    (x=0 anchor)
          u_i = tau_i(q, qdot, qddot) - B_i = A_i + D_i            (x=1 anchor)
          v_i = tau_i(q, qdot/2, qddot/4) - B_i = A_i/4 + D_i/2    (x=0.5 anchor)
        Solving: A_i = 2*u_i - 4*v_i, D_i = 4*v_i - u_i.

        Sampled at the BASE (lambda=1) trajectory's own dt-spaced wall-clock
        grid -- the same discretization _whole_route_margin already uses to
        evaluate any concrete lambda -- so this is a dt-resolution-limited
        proxy for the continuous-path-fraction closed form, exactly as
        resolution-limited as the dense-lambda-grid fallback below already is
        in lambda. Returns (A, B, D), each (n_steps, N_JOINTS)."""
        cfg = self.cfg
        n = int(np.ceil(traj.T / cfg.dt)) + 1
        Q, Qdot, Qddot = traj.sample_horizon(0.0, cfg.dt, n)
        ts = cfg.dt * np.arange(n)
        forces = self._sample_forces(ts, Q, ee_force_fn)
        A = np.zeros((n, N_JOINTS))
        B = np.zeros((n, N_JOINTS))
        D = np.zeros((n, N_JOINTS))
        zeros = np.zeros(N_JOINTS)
        for j in range(n):
            fj = None if forces is None else forces[j]
            b = self.arm.required_torque(Q[j], zeros, zeros, fj)
            u = self.arm.required_torque(Q[j], Qdot[j], Qddot[j], fj) - b
            v = self.arm.required_torque(Q[j], Qdot[j] / 2, Qddot[j] / 4, fj) - b
            B[j] = b
            A[j] = 2 * u - 4 * v
            D[j] = 4 * v - u
        return A, B, D

    def _closed_form_lambda_candidates(self, A: np.ndarray, B: np.ndarray, D: np.ndarray):
        """Closed-form candidates for interior extrema of |tau_i(lambda)| under
        the quadratic-in-x model tau_i(x) = A_i x^2 + D_i x + B_i (x=1/lambda):
        this generalizes monotonicity_lemma_draft.md Sec. 4(b)'s single
        zero-crossing lambda*_i to the two candidate TYPES a quadratic (rather
        than a pure 1/lambda^2 term) admits --
          (1) zero-crossings of tau_i(x) (up to two, quadratic formula; one,
              linear, if A_i==0): local MAXIMA of margin, generalizing Sec.
              4(b)'s single lambda*_i to account for D_i;
          (2) the parabola's own vertex x_v = -D_i/(2A_i): where tau_i(x)
              itself is extremal, which is also where |tau_i(x)| is extremal
              whenever tau_i doesn't cross zero nearby -- a candidate type Sec.
              4(b)'s pure-1/lambda^2 form did not have (a 1/lambda^2 term alone
              is monotonic in x, so had no vertex to speak of).
        Every candidate is directly, exactly computable (no grid); each is
        still just a per-(joint,step) local extremum, so (as documented in
        _search_retime_whole_route) the AGGREGATE m_phys(lambda) = min over
        (joint,step) may still have its true supremum at a pairwise crossing
        this set does not include -- the dense-grid fallback remains the
        safety net for that gap, unchanged."""
        cfg = self.cfg
        out = set()
        A_flat, B_flat, D_flat = A.ravel(), B.ravel(), D.ravel()
        for a, b, d in zip(A_flat, B_flat, D_flat):
            if a == 0:
                if d != 0:
                    x_roots = [-b / d]
                else:
                    x_roots = []
            else:
                disc = d * d - 4 * a * b
                if disc < 0:
                    x_roots = []
                else:
                    sq = np.sqrt(disc)
                    x_roots = [(-d + sq) / (2 * a), (-d - sq) / (2 * a)]
                x_roots.append(-d / (2 * a))  # vertex of the parabola
            for x in x_roots:
                if x <= 0:
                    continue  # lambda = 1/x must be positive and finite
                lam = 1.0 / x
                if 1.0 < lam < cfg.lam_max:
                    out.add(float(lam))
        return sorted(out)

    def _bisect_feasible(self, traj: JointTrajectory, ee_force_fn, lo: float, hi: float):
        """Standard bisection for the smallest feasible lambda in [lo, hi],
        assuming (as the caller must ensure) hi is feasible and lo is not --
        valid locally on this bracket regardless of whether m_phys(lambda) is
        monotonic globally."""
        for _ in range(20):
            mid = 0.5 * (lo + hi)
            if self._whole_route_margin(traj.retimed(mid), ee_force_fn) >= self.cert.m_safe:
                hi = mid
            else:
                lo = mid
        return hi

    def _search_retime_whole_route(self, traj: JointTrajectory, ee_force_fn):
        """This is the m*_1(p) computation from the paper's Theorem 3 (Sufficient,
        checkable condition for rerouting necessity). The paper's Lemma (Sec.
        VI, per-joint sign condition for retiming monotonicity) is derived for
        the idealized frictionless manipulator equation and is NOT used here as
        an operational fast-path gate: checked directly against this
        benchmark's actual scenarios, that 2-term sign condition holds almost
        nowhere (0/3000 in a broad random sweep, and even the flagship scenario
        of Sec. IX-B violates it at 65/138 (joint, step) pairs) -- not because
        the Lemma is wrong about the idealized case, but because this
        benchmark's actual dynamics (models/planar3r.xml) include real joint
        damping the idealized 2-term model cannot represent at all (see
        _torque_decomposition_whole_route). So instead: check lambda_max
        directly (cheap, exact, no assumption); if it already clears m_safe,
        bisect for the smallest feasible lambda (valid regardless of global
        monotonicity, since bisection here only ever narrows within an interval
        whose upper end is independently confirmed feasible at every step, so
        it cannot return an infeasible lambda even if the true feasible set is
        not an interval). If lambda_max alone fails, use the closed-form
        candidate set (_closed_form_lambda_candidates, now the damping-aware
        quadratic-in-1/lambda model) before paying for a dense scan.

        The closed-form candidate set is exact for what it directly checks (an
        is-this-lambda-feasible query is always a real, direct evaluation of
        the whole-route certificate, never an estimate), but it is NOT proven
        exhaustive for the AGGREGATE m_phys(lambda) = min over (joint, step) of
        several such curves: the supremum of a min of several unimodal curves
        can in general occur at a pairwise CROSSING point between two different
        curves, not at either curve's own extremum -- a standard lower-envelope
        phenomenon, not something this benchmark's 3-DOF dynamics are known to
        be structurally immune to. So a missed feasible lambda is possible in
        principle (a false negative -- reporting retiming exhausted when an
        untested interior lambda would in fact have cleared m_safe), never a
        false positive. The dense-grid scan is therefore kept as a safety net:
        it only runs if the closed-form set finds nothing, so overall recall
        cannot regress relative to the dense-grid-only version this replaces.
        See code/README.md for the empirical comparison across random
        scenarios (hit rate of the closed-form set alone, and how often the
        dense-grid safety net was actually needed, before and after adding the
        damping-aware D term)."""
        cfg = self.cfg
        m_at_max = self._whole_route_margin(traj.retimed(cfg.lam_max), ee_force_fn)
        if m_at_max >= self.cert.m_safe:
            return self._bisect_feasible(traj, ee_force_fn, 1.0, cfg.lam_max)

        # lambda_max alone fails: try the closed-form candidate set before
        # paying for a dense scan.
        A, B, D = self._torque_decomposition_whole_route(traj, ee_force_fn)
        candidates = sorted(set([1.0, cfg.lam_max] + self._closed_form_lambda_candidates(A, B, D)))
        margins = [self._whole_route_margin(traj.retimed(l), ee_force_fn) for l in candidates]
        feasible_idx = [i for i, m in enumerate(margins) if m >= self.cert.m_safe]
        if feasible_idx:
            i = min(feasible_idx)
            hi = candidates[i]
            lo = candidates[i - 1] if i > 0 else 1.0
            return self._bisect_feasible(traj, ee_force_fn, lo, hi)

        # Closed-form set found nothing: fall back to the dense scan as a
        # safety net against the crossing-point gap described above, rather
        # than concluding retiming is exhausted on an unproven candidate set.
        grid = np.linspace(1.0, cfg.lam_max, 41)
        margins = np.array([self._whole_route_margin(traj.retimed(l), ee_force_fn) for l in grid])
        feasible = margins >= self.cert.m_safe
        if not np.any(feasible):
            return None  # even the densely-sampled interval cannot fix it
        j = int(np.argmax(feasible))  # first grid point that clears m_safe
        lo, hi = grid[max(j - 1, 0)], grid[j]
        return self._bisect_feasible(traj, ee_force_fn, lo, hi)

    def _search_reshape_whole_route(self, traj: JointTrajectory, ee_force_fn):
        """Level 2 tried at ROUTE-PLANNING time, not just online: reshape the
        WHOLE route (not the bounded online horizon), pinned by
        _try_reshape's terminal_q/terminal_qdot to reach the SAME goal traj
        already does, at rest. This is the 'genuine trajectory regeneration
        before rerouting' step Theorem 3's adaptation class did not
        previously include (paper Sec. V-C's own flagged open item): retiming
        cannot help a deficit that is a function of position, not speed (see
        _search_retime_whole_route), but reshaping CAN, since it is free to
        choose a different Q path entirely, not just a different time law
        along the SAME path -- so this is checked only after retiming has
        already failed (plan_route's ordering), not instead of it.

        Returns (SampledTrajectory, margin) if a reshaped route clears
        m_safe, else None. This is a materially larger QP than the online
        horizon's (n scales with the WHOLE route duration, not
        cfg.horizon_steps), but it is a one-time planning cost like the
        retime search above, not a per-cycle one."""
        cfg = self.cfg
        n = int(np.ceil(traj.T / cfg.dt)) + 1
        Q, Qdot, Qddot = traj.sample_horizon(0.0, cfg.dt, n)
        ts = cfg.dt * np.arange(n)
        forces = self._sample_forces(ts, Q, ee_force_fn)
        reshaped = self._try_reshape(Q, Qdot, Qddot, forces,
                                      terminal_q=traj.qf, terminal_qdot=np.zeros(N_JOINTS))
        if reshaped is None:
            return None
        Qn, Qdotn, Qddotn = reshaped
        m2 = self.cert.m_phys(Qn, Qdotn, Qddotn, forces)
        if m2 < self.cert.m_safe:
            return None
        return SampledTrajectory(ts, Qn, Qdotn, Qddotn), m2

    # ---- online, per control cycle: Level 0 / 2 / 4 --------------------
    def online_step(
        self,
        traj: JointTrajectory,
        t0: float,
        ee_force_fn=None,
        q_actual: Optional[np.ndarray] = None,
        qdot_actual: Optional[np.ndarray] = None,
    ) -> PlanResult:
        """ee_force_fn(t, q) -> force or None, looked ahead only over this call's
        bounded horizon (cfg.horizon_steps * cfg.dt) -- this is what gives Exp 3's
        'detection lead time' a real, horizon-bounded meaning rather than oracle
        full-schedule knowledge.

        q_actual/qdot_actual, if given, are used for exactly ONE purpose: as
        the starting point for Level 4's brake profile. They deliberately do
        NOT touch the feasibility check (m0, still computed from the nominal
        Q/Qdot) or what gets RETURNED as the commanded reference for Level 0/2
        -- an earlier, broader version of this fix routed them into both of
        those too, on the reasoning that a receding-horizon controller should
        measure the true current state before predicting forward. That is
        standard practice in general, but here it interacted badly with a
        second, pre-existing gap: once the robot has genuinely diverged from
        the nominal trajectory (which is the whole point of having braked),
        the nominal trajectory's LATER samples (Q[1:], still un-overridden)
        can be far from where continuing the plan from q_actual would
        actually go, so the certificate can swing back to 'feasible' and
        Level 0 gets returned with a raw nominal reference that is now far
        from the robot's actual position -- an instantaneous reference jump
        that made the closed loop go unstable (observed directly: MuJoCo
        NaN/Inf qacc warnings and torque values around 1e13 in Exp 2's high-
        payload rows). Properly resolving that would mean replanning a fresh
        trajectory from the robot's actual state once it has diverged, which
        this codebase does not do online (Level 1/3 are planning-time-only
        decisions, Sec. V-C) -- out of scope for this fix. Scoping q_actual/
        qdot_actual down to just the brake-profile start point avoids the
        interaction entirely while still fixing the original, narrower bug:
        without it, every subsequent cycle re-derived its brake target from
        the NOMINAL trajectory's hypothetical position at the current
        (still-advancing) wall-clock time, rather than from where the robot
        had actually stopped -- so the 'brake' reference kept crawling
        forward to chase the original plan instead of holding position,
        defeating the entire point of Level 4 and inflating saturation events
        for as long as the certificate kept re-triggering."""
        cfg = self.cfg
        Q, Qdot, Qddot = traj.sample_horizon(t0, cfg.dt, cfg.horizon_steps)
        ts = t0 + cfg.dt * np.arange(cfg.horizon_steps)
        forces = self._sample_forces(ts, Q, ee_force_fn)

        if not cfg.predict:
            return PlanResult(0, Q, Qdot, Qddot, np.nan, np.nan, triggered=False)

        m0 = self.cert.m_phys(Q, Qdot, Qddot, forces)
        if m0 >= self.cert.m_safe:
            return PlanResult(0, Q, Qdot, Qddot, m0, m0, triggered=False)
        if not cfg.act:
            return PlanResult(0, Q, Qdot, Qddot, m0, m0, triggered=True)

        best_effort = None
        if cfg.allow_level2:
            reshaped = self._try_reshape(Q, Qdot, Qddot, forces)
            if reshaped is not None:
                Qn, Qdotn, Qddot2 = reshaped
                m2 = self.cert.m_phys(Qn, Qdotn, Qddot2, forces)
                if m2 >= self.cert.m_safe:
                    return PlanResult(2, Qn, Qdotn, Qddot2, m0, m2, triggered=True)
                best_effort = (Qn, Qdotn, Qddot2, m2)  # didn't fully restore margin,
                                            # but is still the least-bad profile found
                                            # -- used only if Level 4 is also disabled
                                            # (ablation A4), below.

        if not cfg.allow_level4:
            # A4 (Sec. VIII-H): prediction + adaptation, no rerouting AND no
            # braking fallback. If Level 2 found no usable solution either,
            # there is nothing left to do but continue with the nominal
            # reference and accept whatever happens -- the point of this
            # ablation is to show that failing without a safety net.
            if best_effort is not None:
                Qn, Qdotn, Qddot2, m2 = best_effort
                return PlanResult(2, Qn, Qdotn, Qddot2, m0, m2, triggered=True)
            return PlanResult(0, Q, Qdot, Qddot, m0, m0, triggered=True)

        brake_q0 = q_actual if q_actual is not None else Q[0]
        brake_qdot0 = qdot_actual if qdot_actual is not None else Qdot[0]
        Qk, Qdotk, Qddotk = self._brake_profile(brake_q0, brake_qdot0)
        ts_k = t0 + cfg.dt * np.arange(cfg.horizon_steps)
        forces_k = self._sample_forces(ts_k, Qk, ee_force_fn)
        mk = self.cert.m_phys(Qk, Qdotk, Qddotk, forces_k)
        return PlanResult(4, Qk, Qdotk, Qddotk, m0, mk, triggered=True)

    # ------------------------------------------------------------------
    def _try_reshape(self, Q, Qdot, Qddot, forces, terminal_q=None, terminal_qdot=None):
        """Level 2: convex QP over the horizon's acceleration profile.

        terminal_q/terminal_qdot, if given, pin q_vars[-1]/qdot_vars[-1] to
        those values -- used when this is called over a WHOLE route (route-
        planning time, see _search_reshape_whole_route) rather than the
        bounded online horizon, so the reshaped route is constrained to reach
        the SAME goal the original route does, not merely whatever state
        minimizes the cost. Without this, a whole-route reshape could
        silently drift to a different final configuration -- exactly the
        goal-changing failure mode this codebase's ViaPointTrajectory/exp5
        fix already closed once for Level 3, reopened here for Level 2 if
        left unconstrained.

        Q_new/Qdot_new are explicit double-integrator STATE variables,
        anchored at the true current state (Q[0]/Qdot[0]) and propagated
        forward by discrete integration of the optimized qddot, so the
        returned (Q_new, Qdot_new, Qddot) triple is a trajectory that
        recursively applying Qddot from Q[0]/Qdot[0] actually produces --
        not the nominal trajectory's own Q/Qdot paired with a DIFFERENT
        acceleration profile (an inconsistent pairing that does not
        correspond to any trajectory the system would actually follow,
        found during a code-vs-paper review: the certificate evaluated over
        that pairing did not certify what recursively applying the new
        accelerations would actually realize).

        Torque is still linearized (M, h) around the NOMINAL Q/Qdot at each
        step, not Q_new/Qdot_new -- keeping M(q) fixed per step is what
        keeps this a QP rather than a nonconvex joint dynamics+torque
        optimization (the 'fixed structure, state-dependent vector terms'
        design principle, paper Sec. VI). This is re-solved fresh every
        control cycle like any receding-horizon controller, and Q_new stays
        close to the nominal Q over one short horizon window, so evaluating
        M/h at the nominal point is a reasonable first-order approximation,
        not an exact one -- the same approximation the unmodified code
        already made, just no longer paired with a state trajectory that
        can't actually result from the returned accelerations.

        The objective penalizes deviation from the nominal Q/Qdot/Qddot
        (weights: PlannerConfig.reshape_w_pos/reshape_w_vel/reshape_w_acc),
        not raw acceleration magnitude -- a bare minimize-||qddot|| objective
        has no reason to prefer a reshaped trajectory that stays close to the
        one the planner originally intended, and could pick an equally
        feasible but arbitrarily different motion (found during a code-vs-
        paper review). This is a 'closest physically realizable trajectory'
        objective, not a minimum-effort one.
        """
        cfg = self.cfg
        n = Q.shape[0]
        dt = cfg.dt
        qddot_vars = [cp.Variable(N_JOINTS) for _ in range(n)]
        q_vars = [cp.Variable(N_JOINTS) for _ in range(n)]
        qdot_vars = [cp.Variable(N_JOINTS) for _ in range(n)]
        constraints = [q_vars[0] == Q[0], qdot_vars[0] == Qdot[0]]
        if terminal_q is not None:
            constraints.append(q_vars[-1] == terminal_q)
        if terminal_qdot is not None:
            constraints.append(qdot_vars[-1] == terminal_qdot)
        cost = 0
        for j in range(n):
            M = self.arm.mass_matrix(Q[j])
            fj = None if forces is None else forces[j]
            h = self.arm.required_torque(Q[j], Qdot[j], np.zeros(N_JOINTS), fj)
            tau_j = M @ qddot_vars[j] + h
            constraints += [
                tau_j <= TAU_MAX - self.cert.delta_tau,
                tau_j >= TAU_MIN + self.cert.delta_tau,
                cp.abs(qddot_vars[j]) <= cfg.qddot_box,
            ]
            if j + 1 < n:
                constraints += [
                    q_vars[j + 1] == q_vars[j] + dt * qdot_vars[j]
                                     + 0.5 * dt**2 * qddot_vars[j],
                    qdot_vars[j + 1] == qdot_vars[j] + dt * qddot_vars[j],
                ]
            cost += cfg.reshape_w_acc * cp.sum_squares(qddot_vars[j] - Qddot[j])
            cost += cfg.reshape_w_pos * cp.sum_squares(q_vars[j] - Q[j])
            cost += cfg.reshape_w_vel * cp.sum_squares(qdot_vars[j] - Qdot[j])
        prob = cp.Problem(cp.Minimize(cost), constraints)
        try:
            prob.solve(solver=cp.OSQP, verbose=False, max_iter=20000)
        except cp.error.SolverError:
            pass
        if prob.status not in ("optimal", "optimal_inaccurate"):
            # OSQP hits its iteration budget (status "user_limit") rather than
            # converging on the larger whole-route problem this is also used
            # for (n ~ route duration / dt, tens of steps, vs. the online
            # horizon's ~15) -- found empirically while testing route-level
            # reshape on the flagship scenario: OSQP alone reported
            # user_limit even at max_iter=50000, while SCS solved the
            # IDENTICAL problem to a clean "optimal" in well under a second,
            # confirming this is a solver/scaling limitation, not genuine
            # infeasibility. SCS is only tried on OSQP's failure path, so the
            # common (small, fast-converging online horizon) case is
            # unaffected; this is a one-time route-planning cost, not a
            # per-cycle one, when it does trigger.
            try:
                prob.solve(solver=cp.SCS, verbose=False)
            except cp.error.SolverError:
                return None
        if prob.status not in ("optimal", "optimal_inaccurate"):
            return None
        Q_new = np.array([v.value for v in q_vars])
        Qdot_new = np.array([v.value for v in qdot_vars])
        Qddot_new = np.array([v.value for v in qddot_vars])
        return Q_new, Qdot_new, Qddot_new

    # ------------------------------------------------------------------
    def _brake_profile(self, q0, qdot0):
        """Level 4: decelerate at the maximum allowed |qddot| until qdot reaches
        zero (or the horizon ends), then hold position."""
        cfg = self.cfg
        Q, Qdot, Qddot = [q0.copy()], [qdot0.copy()], []
        q, qdot = q0.copy(), qdot0.copy()
        for _ in range(cfg.horizon_steps):
            qddot = -np.clip(qdot / cfg.dt, -cfg.qddot_box, cfg.qddot_box)
            qdot_new = qdot + qddot * cfg.dt
            qdot_new = np.where(np.sign(qdot_new) != np.sign(qdot), 0.0, qdot_new)
            q_new = q + qdot * cfg.dt + 0.5 * qddot * cfg.dt**2
            Qddot.append(qddot)
            q, qdot = q_new, qdot_new
            Q.append(q.copy()); Qdot.append(qdot.copy())
        Qddot.append(np.zeros(N_JOINTS))
        return (np.array(Q[:cfg.horizon_steps]), np.array(Qdot[:cfg.horizon_steps]),
                np.array(Qddot[:cfg.horizon_steps]))
