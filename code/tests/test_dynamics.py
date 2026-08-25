"""Sanity checks on the MuJoCo-backed dynamics wrapper.

These are not unit tests of hand-derived formulas (there are none here); they are
physical-consistency checks on what MuJoCo returns, so a model/wiring bug (wrong
axis, wrong sign convention, mis-set payload body) would be caught before any
paper-facing number depends on it.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynamics import Arm, N_JOINTS, TAU_MAX


def test_mass_matrix_symmetric_positive_definite():
    arm = Arm.create()
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.uniform(-np.pi, np.pi, N_JOINTS)
        M = arm.mass_matrix(q)
        assert np.allclose(M, M.T, atol=1e-8), "mass matrix not symmetric"
        eigvals = np.linalg.eigvalsh(M)
        assert np.all(eigvals > 0), f"mass matrix not PD, eigvals={eigvals}"
    print("OK: mass matrix symmetric & positive-definite over 20 random configs")


def test_static_gravity_torque_horizontal_arm():
    """With the arm fully horizontal (q = [0,0,0], all links along +x) and zero
    velocity/acceleration, the required torque at each joint must equal the
    gravity moment of everything distal to that joint -- a closed-form check
    independent of MuJoCo's internals."""
    arm = Arm.create()
    arm.set_payload_mass(0.0)
    q = np.zeros(N_JOINTS)
    qdot = np.zeros(N_JOINTS)
    qddot = np.zeros(N_JOINTS)
    tau = arm.required_torque(q, qdot, qddot)

    g = 9.81
    m1, m2, m3 = 1.5, 1.2, 0.8
    l1, l2, l3 = 0.30, 0.30, 0.25
    # link i's COM is at its geometric midpoint (uniform capsule), at distance
    # l_i/2 from its proximal joint, l1 (+ l2/2 etc.) from more proximal joints.
    tau3_expected = m3 * g * (l3 / 2)
    tau2_expected = tau3_expected + (m2 * g * (l2 / 2) + m3 * g * l2)
    tau1_expected = tau2_expected + (m1 * g * (l1 / 2) + (m2 + m3) * g * l1)

    # MuJoCo's hinge convention (axis (0,1,0), right-hand rule) rotates +x toward
    # -z for positive q, i.e. the opposite sign from the naive r x F convention
    # used above -- confirmed by the near-exact magnitude match below.
    expected = -np.array([tau1_expected, tau2_expected, tau3_expected])
    assert np.allclose(tau, expected, atol=0.05), (
        f"gravity torque mismatch: got {tau}, expected {expected}"
    )
    print(f"OK: static horizontal gravity torque matches closed form: {tau} Nm")


def test_payload_increases_required_torque_monotonically():
    arm = Arm.create()
    q = np.array([0.0, -0.3, -0.4])
    qdot = np.zeros(N_JOINTS)
    qddot = np.zeros(N_JOINTS)
    prev_tau1 = None
    for m in [0.0, 1.0, 2.0, 3.0, 4.0]:
        arm.set_payload_mass(m)
        tau = arm.required_torque(q, qdot, qddot)
        if prev_tau1 is not None:
            assert abs(tau[0]) >= prev_tau1 - 1e-9, "torque should grow with payload"
        prev_tau1 = abs(tau[0])
    print("OK: joint-1 required torque grows monotonically with payload mass")


def test_external_force_equilibrium_via_forward_dynamics():
    """required_torque's external-force handling (tau -= J^T@F, computed by hand
    since mj_inverse itself was found not to respond to xfrc_applied at all --
    see dynamics.py's docstring) must produce a torque that, when applied as
    ctrl alongside the SAME external force in a genuine forward-dynamics
    rollout, yields exactly zero acceleration. Uses a small force so the needed
    compensation stays within the actuators' declared ctrlrange -- otherwise
    MuJoCo's own actuator clipping would (correctly) confound the check."""
    import mujoco
    arm = Arm.create()
    arm.set_payload_mass(1.0)
    rng = np.random.default_rng(2)
    for _ in range(10):
        q = rng.uniform(-1.0, 1.0, N_JOINTS)
        F = rng.uniform(-5, 5, 2)
        tau = arm.required_torque(q, np.zeros(N_JOINTS), np.zeros(N_JOINTS), F)
        assert np.all(np.abs(tau) < TAU_MAX), "test force too large, exceeds ctrlrange"

        d, m = arm.data, arm.model
        d.qpos[:] = q; d.qvel[:] = 0
        d.xfrc_applied[:, :] = 0
        d.xfrc_applied[arm.payload_body_id, 0] = F[0]
        d.xfrc_applied[arm.payload_body_id, 2] = F[1]
        d.ctrl[:] = tau
        mujoco.mj_forward(m, d)
        assert np.allclose(d.qacc, 0, atol=1e-6), f"qacc={d.qacc}, expected ~0"
    print("OK: external-force compensation achieves exact equilibrium over 10 random cases")


def test_jacobian_matches_finite_difference():
    arm = Arm.create()
    rng = np.random.default_rng(1)
    for _ in range(10):
        q = rng.uniform(-1.5, 1.5, N_JOINTS)
        J = arm.jacobian(q)
        eps = 1e-6
        J_fd = np.zeros((2, N_JOINTS))
        for i in range(N_JOINTS):
            dq = np.zeros(N_JOINTS)
            dq[i] = eps
            p_plus = arm.ee_position(q + dq)
            p_minus = arm.ee_position(q - dq)
            J_fd[:, i] = (p_plus - p_minus) / (2 * eps)
        assert np.allclose(J, J_fd, atol=1e-3), f"Jacobian mismatch:\n{J}\nvs FD\n{J_fd}"
    print("OK: analytic Jacobian matches finite-difference over 10 random configs")


if __name__ == "__main__":
    test_mass_matrix_symmetric_positive_definite()
    test_static_gravity_torque_horizontal_arm()
    test_payload_increases_required_torque_monotonically()
    test_external_force_equilibrium_via_forward_dynamics()
    test_jacobian_matches_finite_difference()
    print("\nAll dynamics sanity checks passed.")
