"""Closed-loop execution: one shared computed-torque tracking controller, driven
by a per-baseline `policy` that supplies the reference (q_ref, qdot_ref, qddot_ref)
at each control step. Holding the tracking controller identical across B1/B2/B3
is what makes the B2-vs-B3 comparison fair (Sec. VIII-A): both share the same
low-level executor and the same global route family; they differ only in whether
the reference-generation layer is reactive or predictive.

tau_cmd = M(q_ref) qddot_ref + h(q_ref, qdot_ref) + Kp(q_ref-q) + Kd(qdot_ref-qdot)
tau_applied = clip(tau_cmd, TAU_MIN, TAU_MAX)   <-- real hardware saturation

The simulated plant is stepped forward with MuJoCo (mj_step) using tau_applied as
the actuator command, so any saturation-induced tracking error is a genuine
consequence of forward simulation, not scripted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import mujoco

from dynamics import Arm, TAU_MAX, TAU_MIN, N_JOINTS

# Acceleration-space computed-torque gains (rad/s^2 per rad, rad/s^2 per rad/s):
# tau = M(q_ref) @ (qddot_ref + KP*(q_ref-q) + KD*(qdot_ref-qdot)) + h(q_ref,qdot_ref)
# Unlike raw torque-unit gains, this makes the idealized closed-loop error dynamics
# e_ddot + KD*e_dot + KP*e = 0 independent of the (here, fairly small and coupled)
# link inertias, which is what makes the loop stable at a 50 Hz control rate.
KP = 100.0  # -> omega_n = 10 rad/s
KD = 20.0   # -> zeta = 1.0 (critically damped)


@dataclass
class RolloutResult:
    t: np.ndarray
    q: np.ndarray
    qdot: np.ndarray
    q_ref: np.ndarray
    tau_cmd: np.ndarray
    tau_applied: np.ndarray
    levels: list  # planner level per step (baseline-specific meaning), or None
    m_phys_trace: list
    replans: int
    ee_positions: np.ndarray


def rollout(
    arm: Arm,
    policy: Callable,
    q0: np.ndarray,
    qdot0: np.ndarray,
    duration: float,
    dt: float,
    ee_force_schedule: Optional[Callable[[float], Optional[np.ndarray]]] = None,
) -> RolloutResult:
    n_steps = int(round(duration / dt))
    arm.data.qpos[:] = q0
    arm.data.qvel[:] = qdot0
    arm.model.opt.timestep = dt

    ts, qs, qdots, q_refs, tau_cmds, tau_apps, levels, margins = (
        [], [], [], [], [], [], [], []
    )
    replans = 0

    for k in range(n_steps):
        t = k * dt
        q_actual = arm.data.qpos.copy()
        qdot_actual = arm.data.qvel.copy()

        q_ref, qdot_ref, qddot_ref, meta = policy(t, q_actual, qdot_actual)
        level = meta.get("level")
        if meta.get("replanned"):
            replans += 1
        margins.append(meta.get("m_phys"))

        ee_force = ee_force_schedule(t) if ee_force_schedule else None
        h = arm.required_torque(q_ref, qdot_ref, np.zeros(N_JOINTS), ee_force)
        M = arm.mass_matrix(q_ref)
        accel_correction = KP * (q_ref - q_actual) + KD * (qdot_ref - qdot_actual)
        tau_cmd = M @ (qddot_ref + accel_correction) + h
        tau_applied = np.clip(tau_cmd, TAU_MIN, TAU_MAX)

        # arm.required_torque/mass_matrix are stateless-looking but mutate the
        # shared arm.data.qpos/qvel as a side effect (they call mj_inverse/
        # mj_forward at q_ref, not q_actual). Restore the true simulated state
        # before integrating, or mj_step silently steps from the wrong point.
        arm.data.qpos[:] = q_actual
        arm.data.qvel[:] = qdot_actual
        if ee_force is not None:
            arm.apply_ee_force(ee_force)
        else:
            arm.clear_external_force()
        arm.data.ctrl[:] = tau_applied
        mujoco.mj_step(arm.model, arm.data)

        ts.append(t); qs.append(q_actual); qdots.append(qdot_actual)
        q_refs.append(q_ref); tau_cmds.append(tau_cmd); tau_apps.append(tau_applied)
        levels.append(level)

    q_arr = np.array(qs)
    ee_pos = np.array([arm.ee_position(q) for q in q_arr])

    return RolloutResult(
        t=np.array(ts), q=q_arr, qdot=np.array(qdots), q_ref=np.array(q_refs),
        tau_cmd=np.array(tau_cmds), tau_applied=np.array(tau_apps), levels=levels,
        m_phys_trace=margins, replans=replans, ee_positions=ee_pos,
    )
