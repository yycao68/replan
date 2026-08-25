"""Policy factories for B1 (no handling), B2 (reactive, current-state only), and
B3 (proposed predictive architecture) -- Sec. VIII-A. Each returns a callable
policy(t, q_actual, qdot_actual) -> (q_ref, qdot_ref, qddot_ref, meta) for use with
executor.rollout, so all three share the exact same tracking controller and plant.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from dynamics import Arm, TAU_MAX
from certificate import Certificate
from trajectory import JointTrajectory
from local_planner import LocalPlanner, PlannerConfig


def policy_b1(traj: JointTrajectory):
    """No predictive feedback of any kind: always the original nominal reference."""
    def policy(t, q_actual, qdot_actual):
        q, qdot, qddot = traj.sample(t)
        return q, qdot, qddot, {"level": 0, "m_phys": None}
    return policy


def policy_b2(traj: JointTrajectory, arm: Arm, ee_force_schedule=None):
    """Reactive, current-state-only handling: if the feedforward torque needed for
    the CURRENT instant alone (no lookahead) exceeds the actuator limit, uniformly
    throttle this step's qddot/qdot-error demand so the requested torque is closer
    to the limit. No prediction, no route awareness -- purely local and reactive,
    matching Sec. VIII-A's B2 definition. ee_force_schedule(t, q) -> force, if
    given, is sampled only at the CURRENT actual state -- B2 reacts to a force
    already present, never to one that hasn't started yet."""
    def policy(t, q_actual, qdot_actual):
        q, qdot, qddot = traj.sample(t)
        f = ee_force_schedule(t, q_actual) if ee_force_schedule else None
        tau0 = arm.required_torque(q, qdot, qddot, f)
        ratio = np.max(np.abs(tau0) / TAU_MAX)
        if ratio > 1.0:
            scale = 1.0 / ratio
            qddot = qddot * scale
            qdot = qdot_actual + (qdot - qdot_actual) * scale
            return q, qdot, qddot, {"level": "reactive-throttle", "m_phys": None}
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
    prediction horizon -- see online_step's docstring)."""
    cfg = cfg or PlannerConfig()
    planner = LocalPlanner(arm, cert, cfg)
    route_force_fn = ee_force_schedule if force_known_at_plan_time else None
    route = planner.plan_route(traj, alt_traj=alt_traj, ee_force_fn=route_force_fn)
    state = {"active_traj": route.traj, "route_level": route.level}

    def policy(t, q_actual, qdot_actual):
        res = planner.online_step(state["active_traj"], t, ee_force_fn=ee_force_schedule)
        level = res.level if res.level != 0 else state["route_level"]
        replanned = res.level != 0 or state["route_level"] != 0
        meta = {
            "level": level, "m_phys": res.m_phys_before,
            "m_phys_after": res.m_phys_after, "replanned": replanned,
        }
        return res.Q[0], res.Qdot[0], res.Qddot[0], meta

    return policy
