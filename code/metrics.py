"""Metrics aggregation (Sec. VIII-I), computed from an executor.RolloutResult."""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from dynamics import TAU_MAX


@dataclass
class RunMetrics:
    task_success: bool
    final_pos_error_m: float
    peak_torque_ratio: float          # max(|tau_applied|/tau_max) actually reached
    saturation_events: int            # steps where clipping altered the command
    saturation_fraction: float
    tracking_error_rms_m: float
    tracking_error_peak_m: float
    min_actuator_margin_nm: float     # ground-truth min margin actually realized
    replans: int
    level_counts: dict


def compute(rollout, goal_ee_pos: np.ndarray, pos_tol: float = 0.03) -> RunMetrics:
    final_err = float(np.linalg.norm(rollout.ee_positions[-1] - goal_ee_pos))
    success = final_err <= pos_tol

    ratio = np.abs(rollout.tau_applied) / TAU_MAX
    peak_ratio = float(ratio.max())

    sat_mask = np.any(np.abs(rollout.tau_applied - rollout.tau_cmd) > 1e-6, axis=1)
    sat_events = int(sat_mask.sum())
    sat_fraction = float(sat_mask.mean())

    track_err = np.linalg.norm(rollout.q - rollout.q_ref, axis=1)
    rms = float(np.sqrt(np.mean(track_err**2)))
    peak = float(track_err.max())

    min_margin = float((TAU_MAX - np.abs(rollout.tau_cmd)).min())

    level_counts: dict = {}
    for lv in rollout.levels:
        key = str(lv)
        level_counts[key] = level_counts.get(key, 0) + 1

    return RunMetrics(
        task_success=success,
        final_pos_error_m=final_err,
        peak_torque_ratio=peak_ratio,
        saturation_events=sat_events,
        saturation_fraction=sat_fraction,
        tracking_error_rms_m=rms,
        tracking_error_peak_m=peak,
        min_actuator_margin_nm=min_margin,
        replans=rollout.replans,
        level_counts=level_counts,
    )


def conservatism(triggered_but_unnecessary: int, total_triggers: int) -> float:
    """Fraction of certificate triggers (Level >=2) that fired despite the
    ground-truth nominal trajectory never actually violating a torque limit."""
    if total_triggers == 0:
        return 0.0
    return triggered_but_unnecessary / total_triggers
