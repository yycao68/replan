"""Experiment 6 (Sec. VIII-G): a single scenario family swept by payload severity,
checking which hierarchy level (0/1/2/3/4) actually resolves each severity, so the
whole Level 0-4 hierarchy is exercised, not just its endpoints (Exp1, Exp5).

Both the nominal and the Level-3 fallback route share the SAME start Q0 and
SAME goal QG (via ViaPointTrajectory), differing only in the via-point taken
to get there -- see exp5_flagship_reroute's docstring for why a shared goal,
not two different final configurations, is the paper's actual Level-3 claim."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from dynamics import Arm
from certificate import Certificate
from trajectory import ViaPointTrajectory
from local_planner import LocalPlanner, PlannerConfig
from baselines import policy_b3
from executor import rollout
import metrics as M

Q0 = np.array([0.2, -1.0, -0.6])          # same scenario geometry as exp5's flagship
QG = np.array([0.75, -1.15, -0.55])       # (shared goal for both routes)
VIA_RISKY = np.array([1.1, -0.15, 0.1])   # nominal route's via-point: outstretched
VIA_SAFE = 0.5 * (Q0 + QG)                # fallback route's via-point: safe midpoint
T1_RISKY, T2_RISKY = 0.5, 0.4
T1_SAFE, T2_SAFE = 0.7, 0.6
SEVERITIES = [0.0, 1.5, 2.5, 3.5, 5.0, 8.0]


def run():
    print(f"{'payload':>8} | {'level(s) used':>18} | {'success':>8} {'sat_events':>10} {'replans':>8}")
    for payload in SEVERITIES:
        arm = Arm.create()
        arm.set_payload_mass(payload)
        cert = Certificate(arm=arm, m_safe=2.0)
        traj = ViaPointTrajectory(Q0, VIA_RISKY, QG, T1=T1_RISKY, T2=T2_RISKY)
        alt = ViaPointTrajectory(Q0, VIA_SAFE, QG, T1=T1_SAFE, T2=T2_SAFE)
        assert np.allclose(traj.qf, alt.qf), "nominal and fallback routes must share the same goal"
        cfg = PlannerConfig(allow_level1=True, allow_level2=True, allow_level3=True)
        pol = policy_b3(traj, arm, cert, cfg, alt_traj=alt)
        # Size duration to what plan_route ACTUALLY selects (peeked at
        # directly, same inputs policy_b3 uses internally), not a flat
        # cfg.lam_max worst case -- see exp5_flagship_reroute's comment at
        # the same call: an overly generous flat budget ran already-finished
        # (holding-position) rollouts for several extra seconds and surfaced
        # an unrelated long-duration tracking instability.
        route = LocalPlanner(arm, cert, cfg).plan_route(traj, alt_traj=alt)
        duration = route.traj.T + 0.3
        rr = rollout(arm, pol, Q0, np.zeros(3), duration=duration, dt=0.02)

        goal_pos = arm.ee_position(QG)
        m = M.compute(rr, goal_pos)
        levels = sorted(set(str(l) for l in rr.levels))
        print(f"{payload:8.1f} | {str(levels):>18} | {str(m.task_success):>8} "
              f"{m.saturation_events:10d} {m.replans:8d}")


if __name__ == "__main__":
    run()
