"""Quintic (minimum-jerk) joint-space trajectory generation.

Standard closed-form minimum-jerk time scaling (Craig, "Introduction to Robotics",
or any standard robotics text): zero velocity and acceleration at both endpoints.
Used as the surrogate for "the global planner's output route," consistent with the
paper's framing (Sec. VII) that this module does not touch or replace OMPL/TOTG --
it stands in for their output on the reduced-order benchmark.
"""
from __future__ import annotations

import numpy as np


def _s(tau: np.ndarray) -> np.ndarray:
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


def _sdot(tau: np.ndarray) -> np.ndarray:
    return 30 * tau**2 - 60 * tau**3 + 30 * tau**4


def _sddot(tau: np.ndarray) -> np.ndarray:
    return 60 * tau - 180 * tau**2 + 120 * tau**3


class JointTrajectory:
    """A single quintic joint-space segment from q0 to qf over duration T,
    optionally time-dilated by a scalar factor lam >= 1 (Level-1 retiming)."""

    def __init__(self, q0: np.ndarray, qf: np.ndarray, T: float, lam: float = 1.0):
        self.q0 = np.asarray(q0, dtype=float)
        self.qf = np.asarray(qf, dtype=float)
        self.T = float(T) * float(lam)
        self.lam = float(lam)

    def sample(self, t: float):
        tau = np.clip(t / self.T, 0.0, 1.0)
        dq = self.qf - self.q0
        q = self.q0 + dq * _s(np.array(tau))
        qdot = dq * _sdot(np.array(tau)) / self.T
        qddot = dq * _sddot(np.array(tau)) / self.T**2
        return q, qdot, qddot

    def sample_horizon(self, t0: float, dt: float, n_steps: int):
        """Returns (Q, Qdot, Qddot), each (n_steps, n_joints)."""
        ts = t0 + dt * np.arange(n_steps)
        Q, Qdot, Qddot = [], [], []
        for t in ts:
            q, qdot, qddot = self.sample(t)
            Q.append(q); Qdot.append(qdot); Qddot.append(qddot)
        return np.array(Q), np.array(Qdot), np.array(Qddot)

    def retimed(self, lam: float) -> "JointTrajectory":
        return JointTrajectory(self.q0, self.qf, self.T / self.lam, lam=lam)


class ViaPointTrajectory:
    """Two consecutive quintic segments, q0 -> q_via -> qf, each individually
    zero-velocity/zero-acceleration at its own endpoints -- a stop-and-go
    route, not a blended spline, so the via point is trivially continuous
    without needing spline-continuity machinery.

    Exists so two candidate routes can share the SAME start q0 and SAME goal
    qf while differing in the path taken between them (e.g. one passes
    through a near-singular/outstretched via-point, the other stays tucked).
    A plain JointTrajectory cannot represent this: two different q0->qf
    segments to two different qf's are two different TASKS, not two routes
    to the same task -- see local_planner's paper-Sec.-V-B framing of Level 3
    as a route-level change, and the (now-fixed) exp5/exp6 bug that compared
    reaching different goals under the name "rerouting."

    Implements the same protocol as JointTrajectory (q0, qf, T, sample,
    sample_horizon, retimed) so it drops into LocalPlanner/executor/policy_b3
    unchanged.
    """

    def __init__(self, q0: np.ndarray, q_via: np.ndarray, qf: np.ndarray,
                 T1: float, T2: float, lam: float = 1.0):
        self.q0 = np.asarray(q0, dtype=float)
        self.q_via = np.asarray(q_via, dtype=float)
        self.qf = np.asarray(qf, dtype=float)
        self._T1 = float(T1) * float(lam)
        self._T2 = float(T2) * float(lam)
        self.T = self._T1 + self._T2
        self.lam = float(lam)

    def sample(self, t: float):
        if t <= self._T1:
            origin, dq, T, tau = self.q0, self.q_via - self.q0, self._T1, \
                np.clip(t / self._T1, 0.0, 1.0)
        else:
            origin, dq, T, tau = self.q_via, self.qf - self.q_via, self._T2, \
                np.clip((t - self._T1) / self._T2, 0.0, 1.0)
        tau_arr = np.array(tau)
        q = origin + dq * _s(tau_arr)
        qdot = dq * _sdot(tau_arr) / T
        qddot = dq * _sddot(tau_arr) / T**2
        return q, qdot, qddot

    def sample_horizon(self, t0: float, dt: float, n_steps: int):
        """Returns (Q, Qdot, Qddot), each (n_steps, n_joints)."""
        ts = t0 + dt * np.arange(n_steps)
        Q, Qdot, Qddot = [], [], []
        for t in ts:
            q, qdot, qddot = self.sample(t)
            Q.append(q); Qdot.append(qdot); Qddot.append(qddot)
        return np.array(Q), np.array(Qdot), np.array(Qddot)

    def retimed(self, lam: float) -> "ViaPointTrajectory":
        return ViaPointTrajectory(
            self.q0, self.q_via, self.qf,
            self._T1 / self.lam, self._T2 / self.lam, lam=lam,
        )


class SampledTrajectory:
    """Wraps a discretely-optimized (Q, Qdot, Qddot) array -- e.g. the output
    of a whole-route Level-2 reshape QP (local_planner._search_reshape_whole_
    route) -- in the same protocol as JointTrajectory/ViaPointTrajectory
    (sample, sample_horizon, retimed, T, q0, qf), so a reshaped route can be
    used anywhere a planned route is (online_step, rollout, a further retime
    search) without every caller needing to know it isn't a closed-form
    quintic.

    The QP that produces (Q, Qdot, Qddot) already treats qddot as piecewise-
    CONSTANT over each dt-spaced step and propagates q/qdot by exact double-
    integrator integration (q[j+1] = q[j] + dt*qdot[j] + 0.5*dt^2*qddot[j],
    qdot[j+1] = qdot[j] + dt*qddot[j] -- see _try_reshape). sample(t)
    reproduces that SAME piecewise-quadratic/linear form within each step,
    so it is an exact reconstruction of what the optimized qddot sequence
    produces at any t, not an independent (and potentially inconsistent)
    interpolation scheme."""

    def __init__(self, ts: np.ndarray, Q: np.ndarray, Qdot: np.ndarray,
                 Qddot: np.ndarray, lam: float = 1.0):
        self.ts = np.asarray(ts, dtype=float)
        self.Q = np.asarray(Q, dtype=float)
        self.Qdot = np.asarray(Qdot, dtype=float)
        self.Qddot = np.asarray(Qddot, dtype=float)
        self.T = float(self.ts[-1])
        self.q0 = self.Q[0].copy()
        self.qf = self.Q[-1].copy()
        self.lam = float(lam)

    def sample(self, t: float):
        if t > self.T:
            # Strictly past the end (e.g. rollout's post-trajectory settle
            # padding): hold position, not whatever Qdot[-1] happens to be.
            return self.Q[-1].copy(), np.zeros_like(self.Qdot[-1]), np.zeros_like(self.Qddot[-1])
        t = float(np.clip(t, 0.0, self.T))
        if t == self.T:
            # Exactly the end: report the trajectory's own terminal state,
            # which is zero only if the caller pinned terminal_qdot=0 when
            # building it (_search_reshape_whole_route always does).
            return self.Q[-1].copy(), self.Qdot[-1].copy(), self.Qddot[-1].copy()
        j = int(np.searchsorted(self.ts, t, side="right") - 1)
        j = max(0, min(j, len(self.ts) - 2))
        tau = t - self.ts[j]
        q = self.Q[j] + self.Qdot[j] * tau + 0.5 * self.Qddot[j] * tau**2
        qdot = self.Qdot[j] + self.Qddot[j] * tau
        qddot = self.Qddot[j]
        return q, qdot, qddot

    def sample_horizon(self, t0: float, dt: float, n_steps: int):
        """Returns (Q, Qdot, Qddot), each (n_steps, n_joints)."""
        ts = t0 + dt * np.arange(n_steps)
        Q, Qdot, Qddot = [], [], []
        for t in ts:
            q, qdot, qddot = self.sample(t)
            Q.append(q); Qdot.append(qdot); Qddot.append(qddot)
        return np.array(Q), np.array(Qdot), np.array(Qddot)

    def retimed(self, lam: float) -> "SampledTrajectory":
        """Standard time-dilation transform (q(t) -> q(t/lam), so velocity
        scales by 1/lam and acceleration by 1/lam^2) applied relative to this
        object's OWN current timing -- consistent with JointTrajectory/
        ViaPointTrajectory's retimed(), which is also relative to their own
        current self.lam, not cumulative across repeated calls."""
        rel = lam / self.lam
        return SampledTrajectory(
            self.ts * rel, self.Q, self.Qdot / rel, self.Qddot / rel**2, lam=lam,
        )


def unit_tests():
    tau = np.array([0.0, 0.5, 1.0])
    assert np.allclose(_s(tau), [0.0, 0.5, 1.0])
    assert np.allclose(_sdot(tau), [0.0, 1.875, 0.0])
    assert np.allclose(_sddot(tau), [0.0, 0.0, 0.0])
    traj = JointTrajectory(np.zeros(3), np.ones(3), T=2.0)
    q0, qd0, qdd0 = traj.sample(0.0)
    qf, qdf, qddf = traj.sample(2.0)
    assert np.allclose(q0, 0) and np.allclose(qd0, 0) and np.allclose(qdd0, 0)
    assert np.allclose(qf, 1) and np.allclose(qdf, 0) and np.allclose(qddf, 0)
    print("OK: quintic trajectory boundary conditions correct")

    vtraj = ViaPointTrajectory(np.zeros(3), np.array([2.0, -1.0, 0.5]), np.ones(3),
                               T1=1.0, T2=1.5)
    assert np.isclose(vtraj.T, 2.5)
    q0, qd0, qdd0 = vtraj.sample(0.0)
    q_via_start, _, _ = vtraj.sample(1.0)
    q_via_end, qd_via, qdd_via = vtraj.sample(1.0 + 1e-9)
    qf, qdf, qddf = vtraj.sample(2.5)
    assert np.allclose(q0, 0) and np.allclose(qd0, 0) and np.allclose(qdd0, 0)
    assert np.allclose(q_via_start, [2.0, -1.0, 0.5])
    assert np.allclose(q_via_end, [2.0, -1.0, 0.5])   # continuous across the via point
    assert np.allclose(qd_via, 0, atol=1e-6) and np.allclose(qdd_via, 0, atol=1e-6)
    assert np.allclose(qf, 1) and np.allclose(qdf, 0) and np.allclose(qddf, 0)
    vtraj2 = vtraj.retimed(2.0)
    assert np.isclose(vtraj2.T, 5.0)
    q0b, qd0b, qdd0b = vtraj2.sample(0.0)
    qfb, qdfb, qddfb = vtraj2.sample(5.0)
    assert np.allclose(q0b, 0) and np.allclose(qfb, 1) and np.allclose(qdfb, 0)
    print("OK: via-point trajectory boundary conditions and continuity correct")

    # SampledTrajectory: build a synthetic piecewise-constant-qddot sequence
    # by hand (bypassing the QP) and check sample() reconstructs it exactly.
    rng = np.random.default_rng(0)
    dt = 0.1
    n = 6
    Q = np.zeros((n, 3)); Qdot = np.zeros((n, 3)); Qddot = rng.normal(size=(n, 3))
    Q[0] = np.array([1.0, -0.5, 0.2]); Qdot[0] = np.array([0.3, -0.1, 0.0])
    for j in range(n - 1):
        Q[j + 1] = Q[j] + dt * Qdot[j] + 0.5 * dt**2 * Qddot[j]
        Qdot[j + 1] = Qdot[j] + dt * Qddot[j]
    ts = dt * np.arange(n)
    straj = SampledTrajectory(ts, Q, Qdot, Qddot)
    assert np.isclose(straj.T, ts[-1])
    for j in range(n):
        q, qd, qdd = straj.sample(ts[j])
        assert np.allclose(q, Q[j], atol=1e-9), (q, Q[j])
        assert np.allclose(qd, Qdot[j], atol=1e-9)
    # midpoint of step 2 must match the SAME quadratic form _try_reshape's
    # own integration constraint uses, not an independent interpolation.
    j, tau = 2, dt / 2
    q_mid, qd_mid, qdd_mid = straj.sample(ts[j] + tau)
    q_expect = Q[j] + Qdot[j] * tau + 0.5 * Qddot[j] * tau**2
    qd_expect = Qdot[j] + Qddot[j] * tau
    assert np.allclose(q_mid, q_expect, atol=1e-9)
    assert np.allclose(qd_mid, qd_expect, atol=1e-9)
    assert np.allclose(qdd_mid, Qddot[j], atol=1e-9)
    q_end, qd_end, qdd_end = straj.sample(straj.T + 1.0)  # past the end: hold
    assert np.allclose(q_end, Q[-1]) and np.allclose(qd_end, 0) and np.allclose(qdd_end, 0)
    straj2 = straj.retimed(2.0)
    assert np.isclose(straj2.T, straj.T * 2.0)
    q0c, qd0c, _ = straj2.sample(0.0)
    qfc, qdfc, _ = straj2.sample(straj2.T)
    assert np.allclose(q0c, Q[0]) and np.allclose(qfc, Q[-1])
    assert np.allclose(qd0c, Qdot[0] / 2.0) and np.allclose(qdfc, Qdot[-1] / 2.0)
    # sample_horizon must agree with sample() point-by-point.
    Qh, Qdh, Qddh = straj.sample_horizon(0.05, 0.03, 7)
    for i, t in enumerate(0.05 + 0.03 * np.arange(7)):
        q, qd, qdd = straj.sample(t)
        assert np.allclose(Qh[i], q) and np.allclose(Qdh[i], qd) and np.allclose(Qddh[i], qdd)
    print("OK: SampledTrajectory exactly reconstructs its own piecewise integration and retimes correctly")


if __name__ == "__main__":
    unit_tests()
