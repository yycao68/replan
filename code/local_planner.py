"""The Level 0-4 hierarchical response of paper Sec. V-B, plus the ablation flags
of Sec. VIII-H (A1-A5), all implemented as configuration on a single planner class
so every baseline/ablation shares one code path (the fairness point Sec. VIII-A
insists on for B2 vs B3).

Design note (simplification, documented honestly): route-level decisions --
Level 1 (retime) and Level 3 (reroute) -- are resolved once, as a *planning-time*
decision over the full candidate route(s), producing a (possibly new) active
trajectory. Instant-level corrections -- Level 2 (reshape) and Level 4 (brake) --
are resolved online, every control cycle, against whichever route is currently
active. This split exists because a retimed reference is a full re-parameterization
of the route's time law: correcting it only for a single instantaneous horizon
window (and not persisting the slower schedule into subsequent cycles) does not
actually slow the executed motion down -- an earlier version of this code made
exactly that mistake, silently reference-tracking a "slowed" qdot/qddot against a
q_ref that kept marching forward at the original pace. Reshape and brake do not
have this problem (they don't change the time law), so they run online per-cycle.

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
from trajectory import JointTrajectory


@dataclass
class PlannerConfig:
    dt: float = 0.02
    horizon_steps: int = 15
    lam_max: float = 4.0
    qddot_box: float = 8.0  # rad/s^2, kinematic bound used by Level-2 QP
    predict: bool = True
    act: bool = True
    allow_level1: bool = True
    allow_level2: bool = True
    allow_level3: bool = True
    allow_level4: bool = True


@dataclass
class RouteDecision:
    traj: JointTrajectory
    level: int  # 0 (nominal kept) | 1 (retimed) | 3 (rerouted)
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
        means Level 1/3 are decided WITH KNOWLEDGE of the whole predicted force/
        contact profile over the route -- appropriate when that profile is part of
        the 'predicted environment E' the paper's Sec. IV certificate is
        conditioned on (Exp 4: a known upcoming contact transition), not when it
        represents a disturbance that should only become visible within a bounded
        online prediction horizon (Exp 3: see online_step instead)."""
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

    def _search_retime_whole_route(self, traj: JointTrajectory, ee_force_fn):
        cfg = self.cfg
        if self._whole_route_margin(traj.retimed(cfg.lam_max), ee_force_fn) < self.cert.m_safe:
            return None  # even maximal slowdown cannot fix it (e.g. a pure static/
                          # gravity torque deficit) -- Level 1 correctly fails
        lo, hi = 1.0, cfg.lam_max
        for _ in range(20):
            mid = 0.5 * (lo + hi)
            if self._whole_route_margin(traj.retimed(mid), ee_force_fn) >= self.cert.m_safe:
                hi = mid
            else:
                lo = mid
        return hi

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
            Qddot2 = self._try_reshape(Q, Qdot, forces)
            if Qddot2 is not None:
                m2 = self.cert.m_phys(Q, Qdot, Qddot2, forces)
                if m2 >= self.cert.m_safe:
                    return PlanResult(2, Q, Qdot, Qddot2, m0, m2, triggered=True)
                best_effort = (Qddot2, m2)  # didn't fully restore margin, but is
                                            # still the least-bad qddot profile
                                            # found -- used only if Level 4 is
                                            # also disabled (ablation A4), below.

        if not cfg.allow_level4:
            # A4 (Sec. VIII-H): prediction + adaptation, no rerouting AND no
            # braking fallback. If Level 2 found no usable solution either,
            # there is nothing left to do but continue with the nominal
            # reference and accept whatever happens -- the point of this
            # ablation is to show that failing without a safety net.
            if best_effort is not None:
                Qddot2, m2 = best_effort
                return PlanResult(2, Q, Qdot, Qddot2, m0, m2, triggered=True)
            return PlanResult(0, Q, Qdot, Qddot, m0, m0, triggered=True)

        brake_q0 = q_actual if q_actual is not None else Q[0]
        brake_qdot0 = qdot_actual if qdot_actual is not None else Qdot[0]
        Qk, Qdotk, Qddotk = self._brake_profile(brake_q0, brake_qdot0)
        ts_k = t0 + cfg.dt * np.arange(cfg.horizon_steps)
        forces_k = self._sample_forces(ts_k, Qk, ee_force_fn)
        mk = self.cert.m_phys(Qk, Qdotk, Qddotk, forces_k)
        return PlanResult(4, Qk, Qdotk, Qddotk, m0, mk, triggered=True)

    # ------------------------------------------------------------------
    def _try_reshape(self, Q, Qdot, forces):
        """Level 2: convex QP over the horizon's acceleration profile, torque
        constraints linearized (exactly, since torque is affine in qddot at fixed
        q,qdot) around the nominal q,qdot for each step."""
        cfg = self.cfg
        n = Q.shape[0]
        qddot_vars = [cp.Variable(N_JOINTS) for _ in range(n)]
        constraints = []
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
            cost += cp.sum_squares(qddot_vars[j])
        prob = cp.Problem(cp.Minimize(cost), constraints)
        try:
            prob.solve(solver=cp.OSQP, verbose=False)
        except cp.error.SolverError:
            return None
        if prob.status not in ("optimal", "optimal_inaccurate"):
            return None
        return np.array([v.value for v in qddot_vars])

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
