"""Generates the three figures embedded in predictive_realizability_paper_draft.md
(§IX-B/§IX-G, §VIII-C, §VI), from a fresh rerun of the same scenarios the
paper's own numbers come from -- never from cached/hand-copied data, per this
project's standing verification discipline.

Fig 1 plots the certificate margin ALONG THE ROUTE (not a live rollout trace):
B3's route-level decision (Level 1/3, local_planner.plan_route) is made once,
before execution begins, by evaluating the whole candidate route's margin
statically -- so the mechanism the paper describes ("the deficit is static,
persists at every retiming factor") is directly visible as a margin-vs-route-
time curve for each candidate route, not something that shows up in a live
closed-loop trace of whichever route B3 already picked.

Sec. IX-H's second finding (reshape "succeeds") is now known false (README.md
item 19) -- this script deliberately does NOT plot a reshape trajectory as a
success case. Fig 1's right panel uses Experiment 7 (Sec. IX-G) instead, the
other environment-conditioned reroute scenario, which is unaffected by that bug
(rechecked directly in item 19) and demonstrates the same margin-along-route
reroute mechanism.

Run: python3 experiments/generate_paper_figures.py   (writes into ../../figures/)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dynamics import Arm
from certificate import Certificate
from trajectory import ViaPointTrajectory, JointTrajectory
from local_planner import PlannerConfig
from baselines import policy_b1, policy_b2, policy_b3
from executor import rollout
import metrics as M
import exp5_flagship_reroute as exp5
import exp7_environment_conditioned_reroute as exp7
import theorem4_terminal_set as t4

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "figures")
os.makedirs(FIG_DIR, exist_ok=True)
DT = 0.02

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150,
})


def _margin_along_route(traj, payload, m_safe, ee_force_fn=None):
    arm = Arm.create()
    arm.set_payload_mass(payload)
    cert = Certificate(arm=arm, m_safe=m_safe)
    n = int(np.ceil(traj.T / DT)) + 1
    Q, Qdot, Qddot = traj.sample_horizon(0.0, DT, n)
    ts = DT * np.arange(n)
    if ee_force_fn is None:
        forces = None
    else:
        forces = np.array([
            [0.0, 0.0] if (f := ee_force_fn(t, q)) is None else f
            for t, q in zip(ts, Q)
        ])
    margins = cert.horizon_margins(Q, Qdot, Qddot, forces).min(axis=1)
    return ts, margins, cert.m_safe


def fig1_margin_along_route():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=False)

    # --- Left: flagship (Sec. IX-B) -- static payload/gravity deficit, no
    # environment force. ---
    traj_A = ViaPointTrajectory(exp5.Q0, exp5.VIA_A, exp5.QG, T1=exp5.T1_A, T2=exp5.T2_A)
    traj_B = ViaPointTrajectory(exp5.Q0, exp5.VIA_B, exp5.QG, T1=exp5.T1_B, T2=exp5.T2_B)
    ts_a, m_a, m_safe = _margin_along_route(traj_A, exp5.PAYLOAD, 2.0)
    ts_b, m_b, _ = _margin_along_route(traj_B, exp5.PAYLOAD, 2.0)
    ax = axes[0]
    ax.plot(ts_a, m_a, color="#b3492b", lw=1.8, label=r"$P_A$ (outstretched via-point)")
    ax.plot(ts_b, m_b, color="#2b6ea8", lw=1.8, label=r"$P_B$ (B3's chosen route)")
    ax.axhline(m_safe, color="0.25", lw=1.1, ls="--", label=r"$m_{\mathrm{safe}}$")
    ax.axhline(0.0, color="0.6", lw=0.8, ls=":")
    ax.set_title("Flagship (§IX-B): static payload deficit")
    ax.set_xlabel("route time (s)")
    ax.set_ylabel(r"certificate margin $m_{\tau,\mathrm{robust}}$ (Nm)")
    ax.legend(loc="lower left", framealpha=0.9)

    # --- Right: Experiment 7 (Sec. IX-G) -- environment-conditioned (contact
    # force) deficit; also demonstrates the margin-along-route mechanism, and
    # is unaffected by the reshape-certificate bug fixed in README item 19. ---
    traj_A7 = ViaPointTrajectory(exp7.Q0, exp7.VIA_A, exp7.QG, T1=exp7.T1, T2=exp7.T2)
    traj_B7 = ViaPointTrajectory(exp7.Q0, exp7.VIA_B, exp7.QG, T1=exp7.T1, T2=exp7.T2)
    ts_a7, m_a7, _ = _margin_along_route(traj_A7, exp7.PAYLOAD, 2.0, exp7.contact_force)
    ts_b7, m_b7, _ = _margin_along_route(traj_B7, exp7.PAYLOAD, 2.0, exp7.contact_force)
    ax = axes[1]
    ax.plot(ts_a7, m_a7, color="#b3492b", lw=1.8, label=r"$P_A$ (enters contact field)")
    ax.plot(ts_b7, m_b7, color="#2b6ea8", lw=1.8, label=r"$P_B$ (B3's chosen route)")
    ax.axhline(2.0, color="0.25", lw=1.1, ls="--", label=r"$m_{\mathrm{safe}}$")
    ax.axhline(0.0, color="0.6", lw=0.8, ls=":")
    ax.set_title("Experiment 7 (§IX-G): known contact-force deficit")
    ax.set_xlabel("route time (s)")
    ax.legend(loc="lower left", framealpha=0.9)

    fig.suptitle(r"Predicted certificate margin along each candidate route, evaluated at plan time"
                 "\n(dashed line: $m_{\\mathrm{safe}}$; B3 rejects any route that dips below it before ever executing)",
                 fontsize=8.5, y=1.06)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig1_margin_along_route.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  (P_A min margin: flagship={m_a.min():.2f} Nm, exp7={m_a7.min():.2f} Nm; "
          f"P_B min margin: flagship={m_b.min():.2f} Nm, exp7={m_b7.min():.2f} Nm)")


def fig2_payload_sweep():
    payloads = list(exp2_module().PAYLOADS)
    sat = {"B1": [], "B2": [], "B3": []}
    err = {"B1": [], "B2": [], "B3": []}
    for payload in payloads:
        for name, pol_fn in [
            ("B1", lambda arm, traj, cert: policy_b1(traj)),
            ("B2", lambda arm, traj, cert: policy_b2(traj, arm)),
            ("B3", lambda arm, traj, cert: policy_b3(traj, arm, cert, PlannerConfig())),
        ]:
            arm = Arm.create()
            arm.set_payload_mass(payload)
            traj = JointTrajectory(exp2_module().Q0, exp2_module().QF, T=exp2_module().T)
            cert = Certificate(arm=arm, m_safe=2.0)
            pol = pol_fn(arm, traj, cert)
            rr = rollout(arm, pol, exp2_module().Q0, np.zeros(3), duration=exp2_module().T + 0.3, dt=DT)
            m = M.compute(rr, arm.ee_position(exp2_module().QF))
            sat[name].append(m.saturation_samples)
            err[name].append(m.final_pos_error_m * 1000.0)  # mm

    fig, axes = plt.subplots(2, 1, figsize=(5.2, 5.4), sharex=True)
    colors = {"B1": "#b3492b", "B2": "#c98a1f", "B3": "#2b6ea8"}
    labels = {"B1": "B1 (no handling)", "B2": "B2 (reactive throttle)", "B3": "B3 (predictive, this paper)"}
    for name in ["B1", "B2", "B3"]:
        axes[0].plot(payloads, sat[name], "o-", color=colors[name], ms=3.5, lw=1.6, label=labels[name])
        axes[1].plot(payloads, err[name], "o-", color=colors[name], ms=3.5, lw=1.6, label=labels[name])
    axes[0].set_ylabel("saturation samples")
    axes[0].set_title("Experiment 2 (§VIII-C): identical geometry, varying payload")
    axes[0].legend(loc="upper left", framealpha=0.9)
    axes[1].set_ylabel("final position error (mm)")
    axes[1].set_xlabel("payload (kg)")
    axes[1].axvspan(2.6, 3.0, color="0.85", zorder=0,
                     label="partial-retiming worse-than-nothing band (§X, item 13)")
    axes[1].legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig2_payload_sweep.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


_exp2_mod = None


def exp2_module():
    global _exp2_mod
    if _exp2_mod is None:
        import exp2_payload_sweep as m
        _exp2_mod = m
    return _exp2_mod


def fig3_theorem4_membership():
    out_b = t4.part_b(n_samples=2500)
    payloads = sorted(out_b.keys())
    h_pct = [out_b[p][0] for p in payloads]
    xf_pct = [out_b[p][1] for p in payloads]

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    width = 0.35
    x = np.arange(len(payloads))
    ax.bar(x - width/2, h_pct, width, color="#6b9e78", label=r"in $\mathcal{H}$ (hold set)")
    ax.bar(x + width/2, xf_pct, width, color="#2b6ea8", label=r"in $\mathcal{X}_f$ (Theorem 4(a))")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p:g}" for p in payloads])
    ax.set_xlabel("payload (kg)")
    ax.set_ylabel("membership rate over sampled states (%)")
    ax.set_title(r"Theorem 4(a): $\mathcal{X}_f$/$\mathcal{H}$ membership vs. payload"
                 f"\n({2500} sampled states per payload)")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(0, 105)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig3_theorem4_membership.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  (H%={h_pct}, X_f%={xf_pct})")


if __name__ == "__main__":
    fig1_margin_along_route()
    fig2_payload_sweep()
    fig3_theorem4_membership()
