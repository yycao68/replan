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


if __name__ == "__main__":
    unit_tests()
