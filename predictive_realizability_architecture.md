# Predictive Physical Realizability: A Unified Planning–Execution Feedback Architecture

**Status:** Internal planning document. Not a paper draft. Synthesizes (a) the original DI-QP/HAE/certificate architecture discussion, (b) the OMPL/MoveIt positioning correction, (c) the full "Research Plan: Predictive Physical-Realizability-Aware Motion Planning" draft, and (d) a literature prior-art search — with every open critique from review kept visible as a flagged item rather than silently resolved.

**Working title:** Predictive Physical Realizability Feedback for Safe Motion Planning and Replanning
**Alternative titles:**
- From Geometric Feasibility to Physical Realizability: Predictive Feedback for Motion Planning
- Predictive Actuation-Feasibility Feedback for Reactive Robot Motion Planning
- Physical-Realizability-Aware Hybrid Motion Planning with Predictive Saturation Certificates

**One-sentence thesis:** We propose a predictive physical-realizability feedback architecture that enables a motion planner to anticipate actuator-authority loss caused by robot state, environment, contact, payload, interaction, and disturbance, and to respond through local adaptation, rerouting, or safe fallback before physical infeasibility becomes execution failure.

---

## 1. Motivation

Motion planners — sampling-based (OMPL), optimization-based, or the constant-$A_d$ DI-QP planner in this line of work — certify **geometric and kinematic feasibility**. None of the widely deployed stacks certify **dynamic/physical realizability**: whether the robot's actuators, given configuration, contact, payload, and disturbance, can actually execute the trajectory as time progresses. For a robot with

$$\tau = M(q)u + C(q,\dot q)\dot q + g(q) + J_c^T F_c + \tau_{ext},$$

future acceleration/command authority depends on configuration, velocity, contact configuration, payload, interaction forces, external disturbances, terrain, uncertainty, and actuator limits. Therefore

$$\text{kinematically feasible} \;\not\Rightarrow\; \text{physically realizable}.$$

**Key research question:** Can future loss of physical realizability be predicted during execution and fed back to motion planning early enough that the robot can adapt, reroute, or safely brake before loss of control authority causes task failure, collision, or falling?

**MoveIt 2 as the real-world anchor (not a strawman).** MoveIt 2's default pipeline is exactly the "path-first" architecture the DI-QP paper benchmarks against: OMPL for geometric search, TOTG as the default time-parameterization stage; Ruckig can subsequently provide jerk-limited smoothing. A confirmed, verifiable pathology is [moveit/moveit2#2600](https://github.com/moveit/moveit2/issues/2600): TOTG reconstructs ~200 rad/s² against a declared 0.2 rad/s² limit at a path inflection point. MoveIt 2's own **Hybrid Planning** architecture explicitly separates a slower global planner from a faster recurrent local planner and supports event-driven, sensor-reactive replanning — a real, currently-empty slot this work would fill, not a contrived application.

> ⚠️ **Unverified claim — must check before citing.** Two additional MoveIt 2 issues (TOTG/TOPP-RA on a Panda reaching 2.75× declared acceleration limit; an acceleration-scaling-factor bug), both dated June 2026, were referenced in an earlier pass but could **not** be located or confirmed by a follow-up search. `moveit/moveit2` did open several bug-labeled issues between late June and mid-August 2026 (#3778, #3780, #3781, #3785, #3799, #3802, #3814) that have not been individually read. **Open and read each candidate issue number before citing.**

**Positioning OMPL correctly.** OMPL answers $\exists$ collision-free path $q_{start}\to q_{goal}$? This work asks the strictly downstream question $\exists$ a physically realizable trajectory along this path, given configuration, contact, payload, terrain, and disturbance? **OMPL is never a competitor** — it is the standard global/geometric planner; the contribution is the predictive physical-realizability feedback layer inserted at the local-planner slot of MoveIt's own Hybrid Planning architecture, downstream of OMPL's search.

The proposed work does not claim conventional pipelines are useless — it asks whether their feasibility representation is incomplete because actuator authority is state- and environment-dependent.

---

## 2. Core Research Problem

> Existing motion planners certify geometric/kinematic feasibility. Robot execution is additionally bounded by a **dynamic realizability boundary** jointly determined by configuration, contact, payload, disturbance, and environment. How should this future physical-realizability information be fed back to the planner so it adjusts motion *before*, rather than after, execution capability is lost?

Working name: **Predictive Physical Realizability.** This is framed as one architecture-level planning–execution interface problem, never as "planner + escape mechanism + saturation controller + software integration" stitched together — that framing is the single biggest desk-reject risk for a T-RO submission.

---

## 3. Architecture — Three Layers, One Feedback Loop

```
        xgoal
          │
 ┌────────▼─────────┐        σ1∨σ2∨σ3 (geometric)         σ4 (physical)
 │   Layer 1: DI-QP   │───────────────────────┐   ┌──────────────────────┐
 │  nominal planner   │                        ▼   ▼                      │
 │  P: x_k → u^nom     │              ┌───────────────────┐               │
 └────────┬───────────┘              │  Layer 2: HAE       │               │
          │                          │  geometric recovery │               │
          │  x(k), u^nom             │  (change WHERE)     │               │
          │                          └──────────┬──────────┘               │
          │                                     │ qesc (cost-only, h)      │
          ▼                                     ▼                          │
 ┌────────────────────────────────────────────────────────────┐           │
 │              Primary convex QP  min ½UᵀHU + hᵀU              │◄──────────┘
 │              s.t. C_kin U ≤ d_kin(x(k), margin(k))            │  Layer 3: Predictive
 └────────────────────────────────────────────────────────────┘  Saturation
                                                                    (change HOW)
```

**Expanded system view** (from the full Research Plan draft — shows environment inputs and the decision supervisor explicitly, complementary to the QP-level diagram above):

```
                  Environment / World
              ┌──────────┼──────────┐
           terrain    obstacles   contacts
              └──────────┼──────────┘
                         ▼
                  Global Planner (OMPL)
                         │  global route
                         ▼
                  DI-QP Local Planner
                         │  nominal trajectory
                         ▼
          Predictive Physical Realizability
          ┌──────────────┼──────────────┐
     torque margin   contact margin  directional authority
          └──────────────┼──────────────┘
                         ▼
                  Decision / Supervisor
                 ┌───────┼────────┐
              execute  adapt    reroute
                         │ retime/reshape
                         ▼
                    Controller → Robot ── feedback
```

**Layer 1 — Motion planning (`mp_main` / DI-QP).** $\mathcal{P}: x_k \to u^{nom}_{0:N}$. Constant-$A_d$ double-integrator backbone; $\Phi, \Gamma, H$ precomputed offline. **Explicitly a local planner** — `mp_main` disclaims global search. A global planner (OMPL) sits upstream and seeds the corridor/waypoints Layer 1 executes and locally adapts at 100 Hz.

**Layer 2 — Geometric failure recovery (HAE).** $\sigma_1\vee\sigma_2\vee\sigma_3$ (QP infeasibility / stagnation / linearization drift) trigger escalating L1→L2→L3 escape, acting only through $h$, leaving $\mathcal{F}_{kin}$ invariant (HAE Theorem 1). *The route doesn't work — change WHERE.*

**Layer 3 — Physical realizability (saturation certificate).** A predicted margin $m_{phys}(k{+}i)$ maps torque-domain saturation risk into the DI-QP's bound $u_{i,max}(x(k),\text{margin}(k))$ — same mechanism obstacle rows already use; $C_{kin}, \Phi, \Gamma, H$ stay fixed. *The route is fine, but at current speed/contact/payload the robot cannot execute it — change HOW first, change WHERE only if that fails.*

**The σ4 gap.** HAE's L1/L2/L3 correct traps by moving the escape *target* through $h$ — they cannot help when $\mathcal{F}_{kin}(x(k))$ itself is empty because margin has collapsed. No retargeting fixes this; only braking/recursive-feasibility fallback or relaxed certificate conservatism does.

---

## 4. Failure Taxonomy and Hierarchical Response

$$\sigma_{geo} = \sigma_1\vee\sigma_2\vee\sigma_3 \;\Rightarrow\; \textbf{CHANGE WHERE}$$
$$\sigma_4 = \text{predicted physical-realizability loss} \;\Rightarrow\; \textbf{CHANGE HOW first, CHANGE WHERE only if adaptation cannot restore feasibility}$$

$\sigma_4$ must **not** be treated as just another HAE escape trigger — the response mechanism is fundamentally different (retiming/reshaping vs. retargeting).

**Hierarchical response (Levels 0–4):**
- **Level 0 — Normal execution.** If $m_{phys} \ge m_{safe}$, execute nominal trajectory.
- **Level 1 — Retiming.** Route remains realizable but nominal timing over-demands: $q(s) \to q(t_{new})$ with reduced velocity/acceleration. *(Not novel — see §12.3.)*
- **Level 2 — Trajectory reshaping.** Modify acceleration profile / CoM motion / contact timing / foot trajectory / local target to restore $m_{phys} > 0$.
- **Level 3 — Rerouting/replanning.** If $\mathcal{T}_{dyn}(p) = \emptyset$ for route $p$, no timing modification recovers it; $p \to p'$ via global replanning or HAE.
- **Level 4 — Safe braking/fallback.** If no feasible continuation exists within the reaction horizon, invoke braking/terminal safe set. This prevents the architecture from pretending every failure is solvable by replanning.

> ⚠️ **Open gap (from review): Level 3's "using global replanning or HAE" is underspecified.** HAE's escape-target selection (L1 tangential, L2 DS modulation, L3 RRT) is optimized for topological/geometric clearance — it has no notion of torque margin. There is no reason a HAE-selected target automatically satisfies lower dynamic demand. Either (a) prove that HAE-found targets are more realizable than the original route under stated conditions, or (b) specify that $\sigma_4$-triggered rerouting uses a *different* mechanism (e.g. re-invoking the global planner with a torque-cost term, or constraining HAE's target search by $\mathcal{F}_{dyn}$). This step must exist explicitly between Theorem 3 (§6) and the Level 3 response — it is currently assumed, not shown.

---

## 5. Predictive Physical-Realizability Certificate — Signal Definitions

For nominal predicted $\mathbf{x}_{0:N}, \mathbf{u}_{0:N}$:

$$m_{\tau,i}(k+j) = \tau_{i,max} - |\tau_i(k+j)|, \qquad m_{\tau,i}^{robust}(k+j) = \tau_{i,max} - |\hat\tau_i| - \Delta\tau_i \;\text{(uncertainty-tightened)}$$

$$m_{phys}(k) = \min_{j=0,\ldots,N} \min_i m_{\tau,i}^{robust}(k+j) \quad\text{(aggregate horizon certificate)}$$

Complementary diagnostic signals: directional authority $\alpha_{dir}(k+j)$ (remaining control/acceleration authority in the demanded direction); time-to-loss $T_{loss} = \min\{j\Delta t : m_{phys}(k+j)\le 0\}$; cumulative-risk $J_{phys} = \sum_j \phi(m_{phys}(k+j))$.

> ⚠️ **Scope recommendation (from review).** Five signals is more than a 3–4-contribution paper should carry through formal proofs. **Only $m_{phys}$ should enter the theorem statements** (§6). $\alpha_{dir}$, $T_{loss}$, $J_{phys}$ should be reported as experimental diagnostics — they add interpretability in figures/tables without inflating the required proof burden.

**Environment-conditioned realizability.** $m_{phys}$ can be conditioned on environment/contact information $E_{k:k+N}$ (terrain height, slope, holes/steps, friction, expected contact, payload, disturbance): $(E_{k:k+N}, x_k, u^{nom}_{k:k+N}) \to \tau_{k:k+N} \to m_{phys}$, giving $\text{collision-free} \not\Rightarrow \text{physically realizable}$. Illustrative (thought experiment, not yet built — see §8 resourcing risk): flat ground $m_{ankle}=0.42$; predicted step into a hole $m_{ankle}=0.05$; continuing nominal trajectory drives $m_{ankle}<0$; planner adjusts footstep/timing/CoM before margin goes negative. This generalization is what elevates the contribution from "torque-aware planning" to "environment-conditioned physical realizability."

---

## 6. Theoretical Results

**From HAE (established, reused as background).** Theorem 1 (Constraint Invariance): any escape target enters only through $h_{esc}$; $\mathcal{F}_{kin}(x(k))$ is invariant to it. Theorems 2–4: finite-sample L1 bound; L2 no-new-equilibria + Lyapunov convergence (A2–A3b); L3 asymptotic probabilistic completeness (A4–A5); no-Zeno dwell-time guarantee.

**New, for the realizability layer.** Define $\mathcal{F}_{kin}(x)$, $\mathcal{F}_{obs}(x,E)$, $\mathcal{F}_{dyn}(x,E)$, and $\mathcal{F}_{safe} = \mathcal{F}_{kin}\cap\mathcal{F}_{obs}\cap\mathcal{F}_{dyn}$.

**Theorem 1 (Realizability Certificate).** Under bounded model uncertainty and bounded external/contact-force prediction error, if $m_{phys}(k+j)\ge 0\ \forall j\in[0,N]$, the predicted nominal trajectory satisfies actuator constraints over the horizon. State all assumptions explicitly; do not claim unconditional safety.

**Theorem 2 (Realizability-Preserving Adaptation).** Suppose the nominal route admits at least one trajectory with $\mathcal{F}_{kin}\cap\mathcal{F}_{obs}\cap\mathcal{F}_{dyn}\neq\emptyset$ *(explicit precondition — fixes the gap flagged in an earlier review pass, where the correction step silently assumed a nonempty intersection)*. If the local adaptation problem includes the predictive dynamic-feasibility constraint and is feasible, the resulting trajectory remains geometrically feasible while restoring actuator realizability, without necessarily changing the global route.

**Theorem 3 (Rerouting Necessity).** For route $p$, define $\mathcal{T}_{dyn}(p) = \{\text{dynamically realizable trajectories along } p\}$. If $\mathcal{T}_{dyn}(p) = \emptyset$, no retiming-only or local acceleration modification recovers physical realizability along $p$ — a route-level change is necessary. This is the mathematical justification for adapt → reroute, and is exactly the theorem the $\sigma_4$/$\mathcal{F}_{kin}$-collapse gap (§3, §4) needed.

**Theorem 4 (Recursive Feasibility / Safe Fallback) — attempt only if assumptions hold.** Terminal safe set $\mathcal{X}_f$ such that every $x_k\in\mathcal{X}_f$ admits a bounded safe braking policy satisfying actuator/collision constraints; prove $x_k\in\mathcal{X}_f \Rightarrow x_{k+1}\in\mathcal{X}_f$. **If the proof becomes too restrictive, omit from the primary paper and state as future work** — matches the previously agreed "reduction logic" review standard (prove the minimum sufficient for the thesis, not the maximal possible result).

**Computational structure.** $H,\Phi,\Gamma,C_{kin}$ remain fixed; state/environment-dependent information enters through $h(x,E)$, $d(x,E)$, margin vectors. **Adaptive numerical values $\neq$ adaptive solver structure** — the central computational philosophy carried over from DI-QP/HAE.

---

## 7. Relationship to Existing Building Blocks

| Piece | Role | Status |
|---|---|---|
| `mp_main` (DI-QP) | Layer 1 implementation backbone | Targeting ICRA 2027 |
| HAE | Layer 2 geometric recovery mechanism | Under review at IJSS |
| Saturation certificate (T1–T3) | Layer 3 physical feasibility mechanism | **Not yet submitted / not yet written** |
| MoveIt 2 Hybrid Planning | Real-world software validation target | External, not authored |

**Framing discipline:** these are *mechanisms serving one problem*, never four parallel contributions. §2's one-sentence problem statement goes on page 1.

---

## 8. Benchmark Experiment Plan

**Baselines.**
- **B1** — MoveIt standard: OMPL → TOTG/Ruckig.
- **B2** — MoveIt Hybrid Planning: OMPL (global) + a conventional reactive local planner.
- **B3** — OMPL (global) + DI-QP + HAE + predictive realizability (local) — **this work.**

> ⚠️ **Fairness correction.** B3 must retain OMPL upstream — DI-QP has no global search capability. B2 vs. B3 (same global planner, different local planner) is the fair contrast and the literal instantiation of MoveIt's own global/local slot.

**Named experiments** (from the full Research Plan; platform fit corrected against §8 resourcing note below):

1. **Baseline trajectory** — B1/B2/B3/B4(this work) compared on planning time, execution time, peak torque, min torque margin, acceleration violation, tracking error, collision clearance. *(FR3/Panda — reachable now.)*
2. **Payload variation** — identical geometric trajectory, varying payload; show the method detects authority loss before saturation. *(FR3/Panda — reachable now.)*
3. **External interaction force** — apply $F_{int}(t)$ during execution; compare no-prediction / reactive-saturation / predictive-feedback; primary metric $T_{warning} = T_{failure} - T_{detection}$. *(FR3/Panda — reachable now.)*
4. **Uneven terrain** — flat/slope/depression/hole/step; demonstrate terrain → contact prediction → torque-authority loss → adapt/reroute.
   > ⚠️ **Platform mismatch (from review).** This experiment as written reads as legged/mobile-platform, but §8's resourcing note below shows no legged/humanoid pipeline exists yet, while Phase I / Stage 1 are explicitly FR3/Panda-only. **Resolve before drafting:** either substitute an FR3-realizable analogue (e.g. end-effector payload/contact-stiffness step change simulating a "terrain discontinuity") for the first submission, or explicitly scope this experiment to Phase II/III and drop it from the required set.
5. **Geometrically-feasible-but-dynamically-infeasible route ($P_A$ vs. $P_B$)** — $P_A$ shorter/high dynamic demand, $P_B$ longer/realizable; conventional planner picks $P_A$; proposed method detects $\mathcal{T}_{dyn}(P_A)=\emptyset$ and reroutes to $P_B$. **Flagship experiment — and unlike #4, this is FR3-realizable now**: construct $P_A$ near a high-inertia/near-singular configuration direction under payload, $P_B$ as the routed-around alternative. No terrain rig required.
6. **Adaptation vs. rerouting continuum** — Case A (small loss → retiming succeeds), B (moderate → reshaping succeeds), C (severe → only rerouting succeeds), D (no safe route → braking succeeds). Validates the Level 0–4 hierarchy directly.

**Ablations.** A1 — no predictive feedback (standard planner/controller). A2 — current-state saturation only, no prediction. A3 — prediction without planner feedback (detects, doesn't replan). A4 — prediction + adaptation, no rerouting. A5 — full system (prediction + adaptation + rerouting + fallback).

**Platforms.**
- **Phase I / Level A — Manipulator (FR3/Panda, MoveIt 2, ROS 2, sim ± real hardware).** Mature MoveIt integration, well-defined torque limits, payload/force experiments manageable. Reachable with existing code/hardware access.
- **Phase II — Mobile manipulator/legged** — uneven terrain, steps, holes, changing contact, payload, pushes.
- **Phase III — Humanoid** — flat/uneven walking, steps, lateral disturbance, payload, footstep adaptation. Valuable but explicitly **not a prerequisite** for the first proof-of-concept.

> ⚠️ **Resourcing risk (from review, unresolved).** No legged/humanoid simulation or hardware pipeline has surfaced in this line of work — everything to date (DI-QP, HAE, pHRI) is FR3 fixed-base. Phase II/III need a floating-base, contact-scheduled dynamics model architecturally distinct from the double-integrator backbone underlying everything else here. **Get an honest build-cost estimate before treating Phase II/III as required** — plausibly 6–12 months, not "one more scenario."

**Comparison metrics (full set):** task success rate, collision rate, fall/failure rate, minimum actuator margin, number of saturation samples, tracking error, minimum obstacle clearance, replanning count, replanning latency, local-planner computation time, global-planner invocation rate, **conservatism**.

> The **conservatism** metric matters most and is easy to omit by accident: an overly conservative certificate can trivially prevent failure by stopping the robot everywhere, which would make every other safety metric look perfect while the system is useless. It must be reported alongside every safety claim, not as an afterthought.

---

## 9. Contributions (target: 3–4, no more)

1. **New problem formulation** — predictive physical realizability as a feedback variable for motion planning, distinguishing geometric/kinematic feasibility from future actuator realizability.
2. **Predictive certificate** — a state-, environment-, contact-, payload-, and disturbance-conditioned certificate predicting future loss of physical authority ($m_{phys}$; §5).
3. **Hierarchical planning response** — adapt → reroute → safe fallback (Levels 0–4, §4), with theorem-backed conditions (§6) for when local adaptation is insufficient and route-level replanning is necessary.
4. **Real-time implementation and validation** — fast local-planning architecture with MoveIt 2 Hybrid Planning as the software integration platform, experiments spanning manipulator and (resourcing permitting) legged systems.

---

## 10. What Should NOT Be Claimed

Avoid: "first torque-aware motion planner"; "first actuator-aware robot planner"; "MoveIt cannot handle dynamic constraints"; "our method guarantees safety under arbitrary disturbances"; "predictive saturation guarantees no falling"; "no previous work has considered torque limits."

**Additions from the literature search (§12):**
- Do not claim novelty for "predict a margin, feed it back before violation" as a general idea — Reference/Explicit Reference Governors have done this since the 1990s (§12.2).
- Do not claim novelty for Level 1 retiming-under-predicted-saturation — this is a mature sub-field (§12.3).
- Do not claim novelty for terrain/contact-conditioned actuation limits in legged planning without citing and differentiating from Orsolino et al. and Acosta & Posa, IEEE T-RO 2025 (§12.4).

**Claim instead:** *This paper introduces a predictive feedback interface that converts future physical-realizability information from the execution layer into motion-planning adaptation and route-level replanning.* The novelty is the planning–realization feedback loop and its unification across geometric and physical failure modes — not the existence of torque limits, margins, or saturation constraints individually.

---

## 11. Research Questions

**RQ1** — Can future actuator infeasibility be predicted sufficiently early to enable meaningful planner intervention? **RQ2** — Does predictive feedback reduce failures vs. reactive saturation handling? **RQ3** — When is retiming/reshaping sufficient, and when is rerouting necessary? **RQ4** — Can the certificate incorporate environment/contact/payload/disturbance information without rebuilding the local planner? **RQ5** — Can the architecture operate within real-time local-planner requirements? **RQ6** — Does the approach generalize across manipulators and legged/humanoid systems?

---

## 12. Related Work — Prior Art Search Findings (August 2026)

No single paper matches "Predictive Physical Realizability" as a named, unified concept — but substantial adjacent prior art exists. Every item below must be read in full and explicitly differentiated before drafting.

### 12.1 Closest architectural match: Path Feasibility Governor (PathFG)
Shu Zhang, James Y. Z. Liu, Dominic Liao-McPherson, *"Integrating Planning and Predictive Control Using the Path Feasibility Governor,"* arXiv:2507.09134 (2025). Architecture: path planner → PathFG → nonlinear MPC, guaranteeing constraint satisfaction, stability, and **recursive feasibility**; modular, replanning-compatible.
- **Overlap:** near-identical high-level architecture to §3/§6.
- **Not (yet) covered, per this search:** environment/contact/payload conditioning, explicit σ-taxonomy with escalating response, actuator/torque realizability specifically.
- **Action:** cite and differentiate on page 1 of related work.

### 12.2 The margin concept itself: Reference / Explicit Reference Governors (RG/ERG)
Garone, Di Cairano, Kolmanovsky (2017 survey); Bemporad, Casavola, Mosca (1990s origin). ERG's **Dynamic Safety Margin (DSM)** — a Lyapunov-derived scalar quantifying distance to constraint violation — is conceptually close to $m_{phys}$. "Predict a margin, feed it back" is a mature, decades-old idea; novelty must rest on the actuator/torque-specific formulation, environment conditioning, and hierarchical response, not the margin-feedback idea itself.

### 12.3 Online retiming under predicted torque saturation — mature sub-field
Disturbance-observer-based path tracking under torque saturation (SERD model, ~2001); "path velocity controller" retiming; recent control-aware trajectory planning (2026, UR5). **Level 1 is not novel** — present as baseline mechanism only.

### 12.4 Legged/humanoid actuation-aware terrain planning — high overlap, one hit in the target venue
- Orsolino, Focchi et al., *"Feasible Region: an Actuation-Aware Extension of the Support Region"* (HyQ) — terrain → joint-torque-limit projection → CoM/foothold planning, close to §5's environment-conditioned story for legged robots.
- *"On the Hardware Feasibility of Nonlinear Trajectory Optimization for Legged Locomotion based on a Simplified Dynamics"* (2019) — torque limits projected into task space for non-flat terrain.
- **B. Acosta and M. Posa, "Perceptive Mixed-Integer Footstep Control for Underactuated Bipedal Walking on Rough Terrain," IEEE Transactions on Robotics, vol. 41, 2025.** In the target venue itself — **must be read before any T-RO submission touching legged terrain**.

### 12.5 Net assessment
No single paper covers the full combination: unified σ-taxonomy across geometric and physical failure, hierarchical adapt→reroute→fallback, environment/contact/payload conditioning, and cross-platform generality under one predictive certificate. But the surrounding territory is more populated than earlier drafts assumed, and each individual mechanism is independently well-established. **Novelty claims must narrow accordingly** (§10).

### 12.6 On naming: "interaction-aware" was considered and rejected
"Interaction-aware" is dominantly used in the literature for multi-agent/social trajectory prediction (crowd navigation, multi-vehicle traffic, decentralized multi-robot planning) — predicting *other agents'* motion, not the robot's own actuator authority. It also collides with the authors' own prior pHRI paper's use of "interaction" (contact/external forces). Within this Research Plan itself, "interaction forces" is one of nine listed factors (§1.1, §5), not the organizing concept — the terrain/payload generalization (§5) is the actual center of gravity. **"Predictive Physical Realizability" remains the correct primary name**; "actuation-aware" (with precedent in Orsolino et al., §12.4) is an acceptable secondary/subtitle qualifier if a more descriptive adjective is wanted.

---

## 13. Proposed Paper Structure

**I. Introduction** — (1) planning usually assumes execution feasibility; (2) real robots have state/environment-dependent authority; (3) future saturation caused by terrain/contact/payload/disturbance/interaction; (4) existing global/local architectures provide an interface; (5) introduce predictive realizability feedback; (6) state contributions.
**II. Related Work** — A. motion planning & time parameterization; B. dynamic/torque-aware planning; C. reactive/hybrid planning; D. reference governors & safety filters; E. actuator saturation & predictive feasibility; F. position of this framework.
**III. Problem Formulation** — dynamics, nominal planner, environment prediction, actuator limits, uncertainty, $\mathcal{F}_{safe}=\mathcal{F}_{kin}\cap\mathcal{F}_{obs}\cap\mathcal{F}_{dyn}$.
**IV. Predictive Physical-Realizability Certificate** — torque prediction, uncertainty tightening, directional authority, horizon margin, time-to-loss.
**V. Feedback Planning Architecture** — normal execution; local adaptation; infeasibility detection; rerouting (integrate HAE here); safe fallback.
**VI. Theoretical Analysis** — certificate correctness; adaptation feasibility; route-level infeasibility; optional recursive feasibility; computational complexity.
**VII. MoveIt 2 / Real-Time Implementation** — Hybrid Planning integration; global planner; local DI-QP; realization module; ROS 2 communication; timing.
**VIII. Experiments** — baseline; payload; interaction force; disturbance; terrain/contact; dynamically infeasible route; adaptation vs. rerouting; ablation; real-time performance.
**IX. Discussion** — conservatism; model/environment uncertainty; sensing latency; computational scaling; certificate limitations; relationship to learned/world-model planners.
**X. Conclusion** — motion planning should not only ask whether a trajectory can be geometrically generated, but continuously ask whether the robot remains physically capable of realizing it.

---

## 14. Publication Sequencing & Portfolio Dependencies

Current state:
- **HAE** — under review at IJSS. Do not modify during review.
- **`mp_main` (DI-QP)** — about to submit to ICRA 2027; 8-page budget full; do not append certificate material now.
- **Saturation certificate paper** — not yet started. Fully free; should be next.
- **T-RO synthesis paper (this document)** — not yet started, and *should not* start in earnest yet.

**Recommended order:** (1) submit `mp_main` to ICRA 2027 as-is → (2) write the saturation certificate paper (target SCL/L-CSS), which is where Theorem 1/2/3 above can be proven first with no external dependency → (3) wait for HAE (IJSS) and `mp_main` (ICRA) decisions → (4) only once both have outcomes — ideally acceptances, converting them into citable, publicly verifiable prior work — assemble the T-RO synthesis paper.

**Why this matters for T-RO specifically:** leaning on your own under-review/unsubmitted work for core motivation and baselines is a known desk-reject/major-revision trigger, elevated given the prior T-RO editorial reject on the pHRI paper.

---

## 15. Immediate Implementation Plan

**Stage 1 — Simulation proof (MoveIt 2 + FR3/Panda):** OMPL/TOTG baseline; proposed local planner; payload; external force; torque prediction; adaptive retiming.
**Stage 2 — Physical rerouting:** two routes of different dynamic demand; demonstrate short/geometrically-valid → physically-infeasible → reroute (= Experiment 5, §8).
**Stage 3 — Environment-aware test:** terrain/contact prediction → future contact → authority margin → replanning. *(Subject to the §8 platform-mismatch flag — confirm FR3-realizable analogue or defer to Phase II.)*
**Stage 4 — Real robot:** validate on FR3/Panda if available.
**Stage 5 — Legged/humanoid:** uneven terrain, steps, disturbances, falling prevention. *(Subject to the §8 resourcing-risk flag.)*

---

## 16. Long-Term Research Direction

This framework can eventually interface high-level AI/world models with model-based physical control:

$$\text{VLM / World Model} \to \text{Global Motion Planner} \to \text{Predictive Physical Realizability} \to \text{Whole-Body / Interaction Controller} \to \text{Robot + Environment}$$

The high-level system determines *what* the robot should do; the planner determines *how* it should move; the predictive realization layer determines *whether* the robot can physically execute that behavior under current and predicted environment. Closing the loop: World → Plan → Predict Physical Realizability → Execute → Replan. Proposed as a long-term foundation for a model-based Physical AI architecture — not a near-term deliverable.

---

## 17. Target Venue & Positioning

**Primary target:** IEEE T-RO (regular paper), contingent on §14 sequencing and the open items in §18.
**Positioning line:** *"We address a fundamental mismatch between motion-planning feasibility and physical execution feasibility"* — stated on page 1, before DI-QP, HAE, or MoveIt are named.
**Secondary targets** (if the theoretical contribution proves insufficiently mature): RA-L + ICRA/IROS; an ICRA/IROS systems-oriented paper; a control-oriented venue if the theorem set becomes the dominant contribution. MoveIt integration can also be released independently as an open-source ROS 2 package regardless of paper outcome.
**Fallback:** if Phase II/III (legged) cannot be resourced in a reasonable timeframe, a manipulator/MoveIt-only scope is a legitimate scaled-down RA-L or strong ICRA/IROS target instead of T-RO — keep as explicit fallback rather than treating T-RO as all-or-nothing.

---

## 18. Open Risks / Verification Checklist

- [ ] Verify the two specific MoveIt 2 GitHub issues (2.75× overshoot; scaling-factor bug) by opening candidate issue numbers directly. Do not cite until confirmed.
- [ ] Close the Level-3/HAE gap (§4): prove HAE-found targets restore $\mathcal{F}_{dyn}$ under stated conditions, or specify a distinct $\sigma_4$-rerouting mechanism.
- [ ] Confirm only $m_{phys}$ carries into theorem statements; $\alpha_{dir}/T_{loss}/J_{phys}$ stay diagnostic-only (§5).
- [ ] Resolve Experiment 4 / Stage 3 platform mismatch: FR3-realizable analogue, or explicit deferral to Phase II (§8, §15).
- [ ] Decide whether Theorem 4 (recursive feasibility) is in-scope or explicitly deferred.
- [ ] Get a real build-cost estimate for Phase II/III (legged/humanoid) before treating it as required for T-RO viability.
- [ ] Read Path Feasibility Governor (arXiv:2507.09134) in full; write explicit differentiation before drafting the introduction (§12.1).
- [ ] Read Acosta & Posa, IEEE T-RO 2025, before any submission touching legged/terrain scenarios (§12.4).
- [ ] Re-tighten the contribution list (§9) to explicitly exclude territory covered by §12.1–12.4; add the §10 "what not to claim" items driven by the literature search.
- [ ] Do not begin substantive writing until `mp_main` and HAE have decisions (§14).
- [ ] When writing begins: one problem statement (§2) on page 1; all mechanisms framed as serving one problem, never as a parallel contribution list.
