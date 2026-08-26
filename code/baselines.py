"""Policy factories for B1 (no handling), B2 (reactive, current-state only), and
B3 (proposed predictive architecture) -- Sec. VIII-A. Each returns a callable
policy(t, q_actual, qdot_actual) -> (q_ref, qdot_ref, qddot_ref, meta) for use with
executor.rollout, so all three share the exact same tracking controller and plant.
"""
from __future__ import annotations

from typing import Optional

import cvxpy as cp
import numpy as np

from dynamics import Arm, TAU_MAX, TAU_MIN, N_JOINTS
from certificate import Certificate
from trajectory import JointTrajectory
from local_planner import LocalPlanner, PlannerConfig


def policy_b1(traj: JointTrajectory):
    """No predictive feedback of any kind: always the original nominal reference."""
    def policy(t, q_actual, qdot_actual):
        q, qdot, qddot = traj.sample(t)
        return q, qdot, qddot, {"level": 0, "m_phys": None}
    return policy


def _torque_feasible_qddot(arm: Arm, q, qdot, qddot_nominal, f):
    """One-step QP: the closest (in acceleration L2 norm) qddot to the nominal
    request that keeps tau = M(q) qddot + h(q,qdot,F) within the actuator
    envelope, at the FIXED (q, qdot, F) this is called with -- i.e. an exact
    torque-feasible projection at that point, not an approximation. tau is
    affine in qddot at fixed q/qdot, so this is a genuine convex QP, unlike
    naively scaling qddot (and qdot) by 1/ratio: h(q,qdot,F) (gravity/
    Coriolis/external-force terms) does not scale proportionally with
    qddot, so a scaled command's ACTUAL torque is not guaranteed to respect
    the limit the scale factor was computed from. Returns None if the QP is
    infeasible or fails to solve (e.g. h(q,qdot,F) alone already exceeds the
    envelope -- no qddot can fix a deficit qddot doesn't control)."""
    M = arm.mass_matrix(q)
    h = arm.required_torque(q, qdot, np.zeros(N_JOINTS), f)
    qddot_var = cp.Variable(N_JOINTS)
    tau = M @ qddot_var + h
    constraints = [tau <= TAU_MAX, tau >= TAU_MIN]
    cost = cp.sum_squares(qddot_var - qddot_nominal)
    prob = cp.Problem(cp.Minimize(cost), constraints)
    try:
        prob.solve(solver=cp.OSQP, verbose=False)
    except cp.error.SolverError:
        return None
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return None
    return qddot_var.value


def policy_b2(traj: JointTrajectory, arm: Arm, ee_force_schedule=None):
    """Reactive, current-state-only handling: if the feedforward torque needed
    for the CURRENT instant alone (no lookahead) exceeds the actuator limit,
    project this step's qddot onto the torque-feasible set at the nominal
    (q, qdot) via a one-step QP (see _torque_feasible_qddot) -- the closest
    admissible acceleration, not a heuristic uniform throttle. No prediction,
    no route awareness -- purely local and reactive, matching Sec. VIII-A's
    B2 definition. ee_force_schedule(t, q) -> force, if given, is sampled
    only at the CURRENT actual state -- B2 reacts to a force already
    present, never to one that hasn't started yet.

    (An earlier version of this policy scaled qddot AND qdot by a single
    ratio = tau_max / tau0 factor. That is not a valid projection: torque is
    tau = M(q) qddot + h(q,qdot,F), and h -- gravity, Coriolis, external
    force -- does not scale with qddot, so the scaled command's actual
    torque was not guaranteed to respect the limit the scale was computed
    from. Found during a code-vs-paper review.)"""
    def policy(t, q_actual, qdot_actual):
        q, qdot, qddot = traj.sample(t)
        f = ee_force_schedule(t, q_actual) if ee_force_schedule else None
        tau0 = arm.required_torque(q, qdot, qddot, f)
        ratio = np.max(np.abs(tau0) / TAU_MAX)
        if ratio > 1.0:
            projected = _torque_feasible_qddot(arm, q, qdot, qddot, f)
            if projected is not None:
                return q, qdot, projected, {"level": "reactive-throttle", "m_phys": None}
            # No qddot can restore feasibility here (e.g. gravity/force alone
            # already saturates the envelope at this q,qdot) -- fall through
            # to the nominal reference; tau_applied's hardware clip in
            # executor.rollout is still the final, unconditional safety net.
            return q, qdot, qddot, {"level": "reactive-throttle-infeasible", "m_phys": None}
        return q, qdot, qddot, {"level": 0, "m_phys": None}
    return policy


def policy_b3(
    traj: JointTrajectory,
    arm: Arm,
    cert: Certificate,
    cfg: Optional[PlannerConfig] = None,
    alt_traj: Optional[JointTrajectory] = None,
    ee_force_schedule=None,
    force_known_at_plan_time: bool = False,
):
    """Full predictive architecture (or an ablation of it, via cfg flags).

    Route-level Level 1/3 decisions are made once, up front, over the full
    candidate route(s) (see local_planner.LocalPlanner.plan_route's docstring for
    why this is a deliberate simplification rather than continuous re-retiming).
    Level 2/4 are then monitored online, every control cycle, against whichever
    route that up-front decision selected.

    ee_force_schedule(t, q) -> force or None. force_known_at_plan_time controls
    whether the route-level decision (Level 1/3) gets to see this schedule in
    full (Exp 4: a known upcoming contact transition is part of the 'predicted
    environment') or only the bounded online horizon sees it as it comes into
    view (Exp 3: an unanticipated disturbance, detected only within the online
    prediction horizon -- see online_step's docstring).

    Level 4 is STICKY: once online_step returns level 4, this policy holds the
    robot at its actual position for the rest of the rollout rather than going
    back to querying online_step (which would keep sampling the nominal
    trajectory at the current, still-advancing wall-clock time -- once the
    robot has been sitting still for a while, that nominal sample can be far
    from the robot's real position, and returning it as a reference again
    produces an instantaneous reference jump that destabilizes the closed
    loop; this was found directly, as MuJoCo NaN/Inf qacc warnings, while
    fixing Level 4's brake-profile starting point to use the true current
    state instead of the nominal trajectory's -- see online_step's
    docstring). This matches the paper's own Sec. V-B framing of Level 4 as a
    'terminal safe-set policy': resuming the original plan after a genuine
    stop would need a fresh route decision (Level 3) or a higher-level
    replan, neither of which is invoked automatically here -- that is an
    honest, stated limitation (README), not something this fix works around."""
    cfg = cfg or PlannerConfig()
    planner = LocalPlanner(arm, cert, cfg)
    route_force_fn = ee_force_schedule if force_known_at_plan_time else None
    route = planner.plan_route(traj, alt_traj=alt_traj, ee_force_fn=route_force_fn)
    state = {
        "active_traj": route.traj, "route_level": route.level, "braked": False,
        # replanned must count EVENTS, not cycles: route_level is decided once
        # at plan time and never changes again, so testing it every cycle (the
        # old code) would count ~one "replan" per control tick for the entire
        # remainder of the rollout whenever Level 1/3 ever fired. Report it
        # exactly once, on the first policy() call.
        "route_event_reported": False,
        "prev_online_level": 0,   # for Level 2's rising-edge event detection below
    }

    def policy(t, q_actual, qdot_actual):
        if state["braked"]:
            zeros = np.zeros_like(q_actual)
            # Braking is a continuing STATE once engaged, not a new event each
            # cycle -- its one event was already reported on the engagement
            # cycle below.
            meta = {"level": 4, "m_phys": None, "m_phys_after": None, "replanned": False}
            return q_actual, zeros, zeros, meta

        res = planner.online_step(state["active_traj"], t, ee_force_fn=ee_force_schedule,
                                   q_actual=q_actual, qdot_actual=qdot_actual)

        replanned_now = False
        if not state["route_event_reported"] and state["route_level"] != 0:
            replanned_now = True
            state["route_event_reported"] = True
        if res.level == 4:
            state["braked"] = True
            replanned_now = True          # the brake ENGAGEMENT is the single event
        elif res.level == 2 and state["prev_online_level"] != 2:
            replanned_now = True          # rising edge: reshape correction just (re)started
        state["prev_online_level"] = res.level

        level = res.level if res.level != 0 else state["route_level"]
        meta = {
            "level": level, "m_phys": res.m_phys_before,
            "m_phys_after": res.m_phys_after, "replanned": replanned_now,
        }
        return res.Q[0], res.Qdot[0], res.Qddot[0], meta

    return policy
