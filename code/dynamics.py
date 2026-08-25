"""MuJoCo-backed dynamics wrapper for the reduced-order verification arm.

All mass-matrix, Coriolis/centrifugal, and gravity terms are computed by MuJoCo's
own forward/inverse dynamics (mj_inverse, mj_fullM) -- this module does not
hand-derive or hard-code any symbolic dynamics equations. That is a deliberate
choice: rigid-body dynamics for even a 3-DOF chain is easy to get subtly wrong by
hand (sign errors in Coriolis terms are a classic failure mode), and the paper's
certificate claims depend on the predicted torque being correct.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import mujoco

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "planar3r.xml")
N_JOINTS = 3
TAU_MAX = np.array([87.0, 87.0, 12.0])  # matches ctrlrange in the MJCF, Nm
TAU_MIN = -TAU_MAX


@dataclass
class Arm:
    """Thin, explicit wrapper around a MuJoCo model/data pair for the 3R arm."""

    model: mujoco.MjModel
    data: mujoco.MjData
    payload_body_id: int

    @classmethod
    def create(cls) -> "Arm":
        model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        data = mujoco.MjData(model)
        payload_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "payload")
        return cls(model=model, data=data, payload_body_id=payload_id)

    # ---- configuration ----------------------------------------------------
    def set_payload_mass(self, mass_kg: float) -> None:
        """Set the end-effector payload mass. mass_kg=0 is a numerically-safe
        floor of 1e-6 kg (MuJoCo requires strictly positive body mass)."""
        self.model.body_mass[self.payload_body_id] = max(mass_kg, 1e-6)

    def set_state(self, q: np.ndarray, qdot: np.ndarray) -> None:
        self.data.qpos[:] = q
        self.data.qvel[:] = qdot
        self.data.qacc[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def clear_external_force(self) -> None:
        self.data.xfrc_applied[:, :] = 0.0

    def apply_ee_force(self, force_xz: np.ndarray) -> None:
        """Apply an external force at the end-effector body's origin.
        force_xz: [Fx, Fz] in the world (sagittal) plane, N."""
        self.clear_external_force()
        ee_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "payload")
        self.data.xfrc_applied[ee_body, 0] = force_xz[0]
        self.data.xfrc_applied[ee_body, 2] = force_xz[1]

    # ---- kinematics ---------------------------------------------------
    def ee_position(self, q: np.ndarray) -> np.ndarray:
        self.data.qpos[:] = q
        mujoco.mj_kinematics(self.model, self.data)
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee")
        return self.data.site_xpos[site_id, [0, 2]].copy()  # (x, z) in sagittal plane

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """2xN task-space (x,z) Jacobian at configuration q."""
        self.data.qpos[:] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee")
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id)
        return jacp[[0, 2], :]

    # ---- dynamics -------------------------------------------------------
    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        self.data.qpos[:] = q
        mujoco.mj_forward(self.model, self.data)
        M = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, self.data, M)
        return M

    def required_torque(
        self,
        q: np.ndarray,
        qdot: np.ndarray,
        qddot: np.ndarray,
        ee_force_xz: np.ndarray | None = None,
    ) -> np.ndarray:
        """Inverse dynamics: the joint torque required to realize (q, qdot, qddot)
        at this instant, including gravity, Coriolis/centrifugal terms, and any
        applied external end-effector force. This is the mj_inverse-based
        implementation of tau in Sec. IV of the paper -- 'the actuator command
        required to realize the requested motion at x under E'."""
        self.data.qpos[:] = q
        self.data.qvel[:] = qdot
        self.data.qacc[:] = qddot
        if ee_force_xz is not None:
            self.apply_ee_force(ee_force_xz)
        else:
            self.clear_external_force()
        mujoco.mj_inverse(self.model, self.data)
        return self.data.qfrc_inverse.copy()
