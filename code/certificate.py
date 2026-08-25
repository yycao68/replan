"""Predictive physical-realizability certificate (paper Sec. IV).

m_tau[j,i]      = tau_max[i] - |predicted tau[j,i]|                  (Theorem 1 object)
m_tau_robust    = m_tau - delta_tau[i]  (uncertainty-tightened margin)
m_phys          = min over horizon steps j and joints i of m_tau_robust

delta_tau is an explicit, fixed additive uncertainty bound representing model
mismatch and disturbance-prediction error (Sec. IV / Theorem 1's assumption),
not something estimated online in this reduced-order benchmark -- consistent with
the paper's own honest scoping (Theorem 1 is exactly as strong as this bound).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dynamics import Arm, TAU_MAX


@dataclass
class Certificate:
    arm: Arm
    delta_tau: np.ndarray = field(default_factory=lambda: 0.05 * TAU_MAX)  # 5% of tau_max
    m_safe: float = 2.0  # Nm, minimum margin the planner will tolerate (Level-0 threshold)

    @staticmethod
    def _force_at(ee_force_xz, j):
        """ee_force_xz may be None, a constant (2,) force, or a per-step
        (n_steps,2) array -- normalize to whatever required_torque expects at
        step j."""
        if ee_force_xz is None:
            return None
        arr = np.asarray(ee_force_xz)
        return arr if arr.ndim == 1 else arr[j]

    def horizon_margins(
        self, Q: np.ndarray, Qdot: np.ndarray, Qddot: np.ndarray,
        ee_force_xz: np.ndarray | None = None,
    ) -> np.ndarray:
        """Returns m_tau_robust, shape (n_steps, n_joints)."""
        n_steps = Q.shape[0]
        m = np.zeros((n_steps, TAU_MAX.shape[0]))
        for j in range(n_steps):
            tau = self.arm.required_torque(Q[j], Qdot[j], Qddot[j], self._force_at(ee_force_xz, j))
            m[j, :] = TAU_MAX - np.abs(tau) - self.delta_tau
        return m

    def m_phys(
        self, Q: np.ndarray, Qdot: np.ndarray, Qddot: np.ndarray,
        ee_force_xz: np.ndarray | None = None,
    ) -> float:
        return float(self.horizon_margins(Q, Qdot, Qddot, ee_force_xz).min())

    def is_realizable(
        self, Q: np.ndarray, Qdot: np.ndarray, Qddot: np.ndarray,
        ee_force_xz: np.ndarray | None = None,
    ) -> bool:
        return self.m_phys(Q, Qdot, Qddot, ee_force_xz) >= self.m_safe

    def ground_truth_violation(
        self, Q: np.ndarray, Qdot: np.ndarray, Qddot: np.ndarray,
        ee_force_xz: np.ndarray | None = None,
    ) -> bool:
        """The TRUE feasibility check with no uncertainty margin (delta_tau=0) --
        used only for the conservatism metric (Sec. VIII-I), never by the planner
        itself. Returns True if the nominal trajectory actually violates a torque
        limit (i.e. the certificate's caution was NOT a false positive)."""
        n_steps = Q.shape[0]
        for j in range(n_steps):
            tau = self.arm.required_torque(Q[j], Qdot[j], Qddot[j], self._force_at(ee_force_xz, j))
            if np.any(np.abs(tau) > TAU_MAX):
                return True
        return False
