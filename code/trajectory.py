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


if __name__ == "__main__":
    unit_tests()
