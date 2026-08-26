# Predictive Physical Realizability: A Unified Planning–Execution Feedback Architecture for Motion Planning

**Draft status:** Internal working draft, not submission-ready. Assembled from the source planning document and the companion research plan. Every open item flagged in those documents is preserved below as an explicit, visible flag rather than silently resolved — see the *Draft Status and Open Items* section at the end before treating any claim here as final. Per the source document's own publication-sequencing note, substantive external submission should wait until `mp_main` (ICRA 2027) and HAE (IJSS) have decisions.

---

## Abstract

Motion planners — sampling-based, optimization-based, or fixed-structure time-domain planners — certify geometric and kinematic feasibility: that a path is collision-free and respects declared velocity/acceleration bounds. None of the widely deployed planning stacks certify *dynamic physical realizability*: whether the robot's actuators, given its current configuration, contact state, payload, and disturbance environment, can actually execute the planned motion as time progresses. Because actuator authority is state- and environment-dependent, a trajectory can be kinematically feasible and still drive one or more joints toward saturation in the near future. This paper introduces **predictive physical realizability**, a feedback signal that predicts future loss of actuator authority over a receding horizon and feeds it back into motion planning before the loss becomes an execution failure. We define a horizon-wide realizability certificate conditioned on predicted state, contact, payload, and disturbance information; a hierarchical planner response — retime, reshape, reroute, or safely brake — governed by whether local adaptation can restore feasibility along the current route; and conditions under which a route-level change is provably necessary. The architecture preserves the fixed-structure computational philosophy of the underlying local planner: state- and environment-dependent information enters through linear/vector terms rather than rebuilding the optimization structure online. We position the contribution against reference-governor theory, actuation-aware legged planning, and MoveIt 2's Hybrid Planning architecture, and propose a benchmark suite — reachable today on a fixed-base manipulator, with legged/humanoid extensions explicitly scoped as future work pending a resourcing decision — designed to isolate what predictive feedback adds over reactive saturation handling.

---

## I. Introduction

Robot motion planning is conventionally organized as a pipeline: a global or sampling-based planner searches for a geometrically valid path, a time-parameterization stage converts that path into a timed trajectory subject to declared velocity/acceleration bounds, and an execution layer tracks the result. This separation is effective when the robot's dynamic capability is state-independent, but it is not: the torque required to realize a given acceleration depends on configuration, velocity, active contacts, payload, interaction forces, and external disturbance, through

$$
\tau = M(q)\,u + C(q,\dot q)\,\dot q + g(q) + J_c^\top F_c + \tau_{\mathrm{ext}}.
$$

A trajectory that is geometrically and kinematically valid at planning time can therefore still drive an actuator toward saturation once contact, payload, or disturbance conditions change — a fact that current pipelines discover only when it happens, not before. We write this as

$$
\text{kinematically feasible} \;\not\Rightarrow\; \text{physically realizable}.
$$

**MoveIt 2 as the concrete anchor for this gap.** We do not treat this as a hypothetical failure mode. MoveIt 2's default pipeline — OMPL for geometric search, Time-Optimal Trajectory Generation (TOTG) as the default time-parameterization stage, with optional Ruckig jerk-limited smoothing — is exactly the path-first architecture this paper is positioned against. A confirmed, independently verifiable pathology is documented in [moveit/moveit2#2600](https://github.com/moveit/moveit2/issues/2600) [1]: TOTG reconstructs an acceleration of roughly 200 rad/s² at a path inflection point against a declared limit of 0.2 rad/s². We cite this one issue because it is directly verifiable; two additional reports referenced in an earlier internal draft of this work could not be re-located on a follow-up search and are *not* cited here (see *Draft Status and Open Items*). Independently of any specific bug, MoveIt 2's own **Hybrid Planning** architecture already separates a slower global planner from a faster, recurrent local planner and explicitly supports event-driven, sensor-reactive replanning — a real, currently-unoccupied slot for exactly the kind of feedback this paper proposes, not a contrived integration target.

**Positioning relative to OMPL.** OMPL answers the question *does a collision-free path from start to goal exist?* This paper asks the strictly downstream question: *does a physically realizable trajectory exist along this path, given the robot's configuration, contact state, payload, and predicted environment?* OMPL is not a competitor at any point in this paper; it is the standard upstream global planner, and the contribution sits at the local-planner slot of MoveIt's own Hybrid Planning architecture, consuming OMPL's output rather than replacing it.

We do not claim that conventional planning pipelines are useless. We claim that their feasibility representation is incomplete, because actuator authority is a function of state and environment that the geometric/kinematic feasibility check does not see.

**Key research question.** *Can future loss of physical realizability be predicted during execution and fed back to motion planning early enough that the robot can adapt, reroute, or safely brake before loss of control authority causes task failure, collision, or falling?*

### A. Contributions

1. **A problem formulation** that distinguishes geometric/kinematic feasibility from future actuator realizability, and defines the combined safe set $\mathcal{F}_{\mathrm{safe}} = \mathcal{F}_{\mathrm{kin}} \cap \mathcal{F}_{\mathrm{obs}} \cap \mathcal{F}_{\mathrm{dyn}}$ that existing pipelines do not jointly certify.
2. **A predictive realizability certificate** $m_{\mathrm{phys}}$, conditioned on predicted state, contact, payload, and disturbance information, that bounds actuator margin over a receding horizon without requiring the local planner's optimization structure to be rebuilt online.
3. **A hierarchical planning response** — execute, retime, reshape, reroute, or safely fall back — together with a checkable, sufficient condition (Theorem 3) for when the architecture's own implemented local-adaptation mechanism has been exhausted and route-level replanning is warranted, distinct from the (true but definitional) idealized statement that no modification of an infeasible route can ever be feasible.
4. **A real-time architecture and benchmark design** targeting MoveIt 2's Hybrid Planning interface, with an experiment suite that isolates the contribution of prediction, adaptation, and rerouting individually via ablation.

We deliberately do *not* claim to be the first work to consider actuator limits in motion planning, to have invented margin-based feedback, or to guarantee safety under arbitrary disturbance — see §XI and *What This Paper Does Not Claim* below. The novelty we claim is narrower: a predictive feedback interface that converts future physical-realizability information from the execution layer into planner-level adaptation and route-level replanning, unified across geometric and physical failure modes under one hierarchical response.

---

## II. Related Work

**A. Motion planning and time parameterization.** Sampling-based planners such as OMPL certify geometric feasibility; time-parameterization methods (TOTG, TOPP-RA, Ruckig) subsequently fit a time law respecting declared kinematic bounds. These methods do not represent configuration-, contact-, or payload-dependent actuator authority as a planning-time signal; the pathology in moveit/moveit2#2600 is a direct, currently open, symptom of that gap.

**B. Reactive and hybrid motion planning.** MoveIt 2's Hybrid Planning architecture separates a slower global planner from a faster recurrent local planner and explicitly supports event-driven, sensor-reactive local adaptation, including to changing surface conditions. This paper's contribution is not the hybrid architecture itself, but a new feedback signal and local planning constraint that occupies that architecture's local-planner slot.

**C. Reference governors and safety filters.** Reference Governors and Explicit Reference Governors (Garone, Di Cairano, and Kolmanovsky, *Automatica* survey, 2017 [2]; origins in Bemporad, Casavola, and Mosca, 1997 [3]) predict a scalar margin to constraint violation — the Dynamic Safety Margin — and modulate a reference command to keep the closed loop inside a safe invariant set before violation occurs. This is conceptually the closest prior architecture to the margin-feedback idea at the center of this paper, and we do not claim novelty for "predict a margin and feed it back before a constraint is violated" as a general principle: that principle is three decades old. What is not addressed by the reference-governor literature, to our knowledge, is a margin specifically defined over actuator/torque authority, conditioned jointly on predicted contact, payload, and terrain information, coupled to a planner response that includes route-level replanning rather than only reference modulation within a fixed route.

A closely related recent architecture is the **Path Feasibility Governor** (Zhang, Liu, and Liao-McPherson, arXiv:2507.09134, 2025 [4]), which places a governor between a path planner and a nonlinear MPC controller and proves constraint satisfaction, stability, and recursive feasibility under a modular, replanning-compatible design. The high-level architecture is close to the one proposed here (§III–§V). We differ in that the present work (i) conditions the realizability certificate explicitly on predicted environment, contact, and payload information rather than the nominal path alone, (ii) defines an explicit taxonomy of geometric versus physical failure with a hierarchical, mechanism-differentiated response (§V), and (iii) targets actuator/torque realizability specifically rather than general MPC constraint satisfaction. We flag this as the piece of prior art requiring the most careful textual differentiation before any external submission (see *Draft Status and Open Items*).

**D. Actuation-aware and dynamics-aware planning.** Prior work has incorporated torque limits, feasible wrench sets, and dynamic feasibility into trajectory optimization and legged locomotion planning. Retiming a nominal trajectory under predicted torque saturation is itself a mature sub-field, with roots the source planning document places in disturbance-observer-based path tracking under torque saturation in the early 2000s; we treat retiming (our Level 1, §V) as an established baseline mechanism, not a contribution, and flag this specific historical claim as not yet backed by a specific citation (see *Draft Status and Open Items*). For legged systems specifically, Orsolino, Focchi, and colleagues' actuation-aware extension of the support region [5] projects joint-torque limits through terrain-dependent contact configurations into a feasible base-motion region, closely related to the environment-conditioning story in §IV for legged platforms; a related task-space projection of torque limits for trajectory optimization on non-flat terrain is referenced in the source planning document without a specific citation and is flagged the same way. Acosta and Posa's perceptive mixed-integer footstep control for underactuated bipedal walking on rough terrain (*IEEE Transactions on Robotics*, 2025) [6] is, to our knowledge, the closest legged/terrain-aware actuation work published in this paper's own target venue, and must be read and differentiated in full before any submission that touches legged or terrain scenarios (see *Draft Status and Open Items* — this has not yet been done as of this draft).

**E. Position of this work.** No single piece of prior art we have located combines (i) a unified failure taxonomy spanning geometric and physical failure modes, (ii) a hierarchical adapt-then-reroute-then-fallback response with an explicit necessity condition for route-level change, (iii) a certificate jointly conditioned on predicted contact, payload, terrain, and disturbance, and (iv) demonstrated generality across a fixed-base manipulator and, pending resourcing, legged platforms. Each individual mechanism, however, is independently well established, and our contribution claims are scoped accordingly (§I-A, and see *What This Paper Does Not Claim* below).

---

## III. Problem Formulation

Let $x_k = (q_k, \dot q_k)$ be the robot state at discrete time $k$, and let $E_{k:k+N}$ denote predicted environment information over a horizon of length $N$ — terrain height and slope, expected contact locations and timing, friction, payload, and anticipated external disturbance or interaction force. A nominal motion planner $\mathcal{P}$ produces a candidate trajectory $u^{\mathrm{nom}}_{0:N-1}$ (and the induced state sequence $x_{0:N}$) that is certified against two feasibility sets already representable by existing pipelines:

$$
\mathcal{F}_{\mathrm{kin}}(x) \quad \text{(kinematic feasibility: velocity, acceleration, joint-limit bounds)},
$$
$$
\mathcal{F}_{\mathrm{obs}}(x, E) \quad \text{(collision/obstacle feasibility, given predicted environment } E\text{)}.
$$

We introduce a third set, not represented by conventional pipelines: $\mathcal{F}_{\mathrm{dyn}}(x, E)$, the set of states $x$ at which the actuator command required to realize the requested motion under $E$ lies within actuator limits.

The joint safe set the planner should certify is

$$
\mathcal{F}_{\mathrm{safe}} = \mathcal{F}_{\mathrm{kin}} \cap \mathcal{F}_{\mathrm{obs}} \cap \mathcal{F}_{\mathrm{dyn}}.
$$

Existing planning pipelines certify $\mathcal{F}_{\mathrm{kin}} \cap \mathcal{F}_{\mathrm{obs}}$ at planning time and discover violations of $\mathcal{F}_{\mathrm{dyn}}$ only at execution time, typically through actuator saturation, tracking-error growth, or, on a floating-base or legged platform, loss of balance. The central problem this paper addresses is: *under what conditions can $\mathcal{F}_{\mathrm{dyn}}$-membership be predicted ahead of execution, and how should the planner respond when prediction indicates future loss of membership?*

We assume bounded model uncertainty (bounded error between the dynamics model used for prediction and the true plant) and bounded external/contact-force prediction error over the horizon; both assumptions are stated explicitly wherever they are used and are not claimed to hold unconditionally (see the assumption-scoped statement of Theorem 1, §VI).

---

## IV. Predictive Physical-Realizability Certificate

For the nominal predicted trajectory $\mathbf{x}_{0:N}, \mathbf{u}_{0:N}$, define the per-joint, per-horizon-step torque margin

$$
m_{\tau,i}(k+j) = \tau_{i,\max} - |\tau_i(k+j)|,
$$

and its uncertainty-tightened form, using a predicted torque estimate $\hat\tau_i$ and an explicit uncertainty bound $\Delta\tau_i$ that accounts for model mismatch and contact/disturbance prediction error,

$$
m_{\tau,i}^{\mathrm{robust}}(k+j) = \tau_{i,\max} - |\hat\tau_i(k+j)| - \Delta\tau_i(k+j).
$$

The aggregate horizon certificate — the single scalar we carry into the theoretical results of §VI — is

$$
m_{\mathrm{phys}}(k) = \min_{j=0,\dots,N} \; \min_i \; m_{\tau,i}^{\mathrm{robust}}(k+j).
$$

Three complementary signals are reported as experimental diagnostics rather than carried into the formal results, to keep the required proof burden proportional to a 3–4-contribution paper: directional authority $\alpha_{\mathrm{dir}}(k+j)$ (remaining control/acceleration authority in the direction the current task demands, useful for distinguishing "some margin remains, but not in the direction needed" from an aggregate scalar); time-to-loss $T_{\mathrm{loss}} = \min\{\,j\Delta t : m_{\mathrm{phys}}(k+j) \le 0\,\}$; and cumulative risk $J_{\mathrm{phys}} = \sum_j \phi(m_{\mathrm{phys}}(k+j))$ for a chosen penalty function $\phi$.

**Environment conditioning.** The certificate is a function of predicted environment and contact information as well as state:

$$
\big(E_{k:k+N},\, x_k,\, u^{\mathrm{nom}}_{k:k+N}\big) \;\longrightarrow\; \tau_{k:k+N} \;\longrightarrow\; m_{\mathrm{phys}}.
$$

This is the step that distinguishes the contribution from a torque-limit checker: a trajectory can be entirely collision-free and still fail $\mathcal{F}_{\mathrm{dyn}}$ because the *predicted* terrain, contact, or payload state along the route demands more torque than is available, i.e.

$$
\text{collision-free} \;\not\Rightarrow\; \text{physically realizable}.
$$

A representative (illustrative, not yet run — see *Draft Status and Open Items*) instance: on flat ground a legged platform's ankle-torque margin is $m_{\mathrm{ankle}} = 0.42$; the same nominal footstep trajectory into a predicted shallow hole yields $m_{\mathrm{ankle}} = 0.05$; continuing the unmodified nominal trajectory would drive $m_{\mathrm{ankle}} < 0$ at the moment of contact. The certificate exposes this before the step is taken, giving the planner the opportunity to adjust footstep placement, timing, or center-of-mass trajectory rather than discovering the deficit at contact.

---

## V. Feedback Planning Architecture

### A. Failure taxonomy

We retain the geometric/topological failure signal from prior work in this line ($\sigma_1 \vee \sigma_2 \vee \sigma_3$: QP infeasibility, planner stagnation, and prediction/Jacobian linearization drift, respectively) and add a physical-realizability signal:

$$
\sigma_{\mathrm{geo}} = \sigma_1 \vee \sigma_2 \vee \sigma_3, \qquad \sigma_4 = \big[\, m_{\mathrm{phys}}(k+j) < m_{\mathrm{safe}} \text{ for some } j \in [0,N] \,\big].
$$

The two signals demand structurally different responses, and $\sigma_4$ must not be collapsed into the existing geometric-escape mechanism as though it were a fourth instance of the same trigger:

$$
\begin{aligned}
&\sigma_{\mathrm{geo}} \Rightarrow \textbf{change WHERE}, \\
&\sigma_4 \Rightarrow \textbf{change HOW first, change WHERE only if adaptation cannot restore feasibility.}
\end{aligned}
$$

The reason this distinction matters architecturally: a geometric escape mechanism retargets the route ($h_{\mathrm{esc}}$ in the underlying QP) while leaving the kinematic feasible set unchanged. This is the correct response to a topological trap, but it does nothing for $\sigma_4$: if $m_{\mathrm{phys}}$ has collapsed because the *current* route demands more torque than is available, retargeting within the same route's local neighborhood does not necessarily restore torque margin, and only retiming, reshaping, or a genuine route-level change can.

### B. Hierarchical response

- **Level 0 — Normal execution.** If $m_{\mathrm{phys}}(k) \ge m_{\mathrm{safe}}$, execute the nominal trajectory unmodified.
- **Level 1 — Retiming.** The route remains dynamically realizable, but the nominal timing over-demands the actuators: reparameterize $q(s) \to q(t_{\mathrm{new}})$ with reduced velocity/acceleration along the same geometric path. This mechanism is a mature baseline (§II-D), not a contribution of this paper.
- **Level 2 — Trajectory reshaping.** Modify the acceleration profile, center-of-mass motion, contact timing, or local target while remaining on the same route, to restore $m_{\mathrm{phys}} > 0$ without changing the route.
- **Level 3 — Rerouting/replanning.** If no retiming along the current route restores realizability — formally, if the retiming-adaptation margin $m^\star_1(p) < m_{\mathrm{safe}}$ for the current route $p$ (Theorem 3, §VI, a sufficient, checkable condition for the idealized $\mathcal{T}_{\mathrm{dyn}}(p) = \emptyset$) — the route itself must change: $p \to p'$.
- **Level 4 — Safe braking/fallback.** If no feasible continuation exists within the available reaction horizon, invoke a braking or terminal-safe-set policy. This level exists so the architecture does not implicitly assume every failure is solvable by further replanning.

### C. The Level 3 mechanism (open item)

Theorem 3 (§VI) answers *when* the architecture should stop attempting local adaptation and invoke Level 3, via the checkable condition $m^\star_1(p) < m_{\mathrm{safe}}$. It does not answer *what the rerouter should then do*, and that remains a genuinely open design question rather than something to assume away. Simply invoking "global replanning or the existing geometric-escape mechanism" is not yet justified: the existing geometric-escape mechanism selects an escape target to restore topological/geometric clearance; it has no explicit notion of torque margin, and there is no a priori reason a geometrically-motivated escape target also satisfies lower dynamic demand. Two directions close this gap, neither implemented as of this draft: (a) show that geometric-escape-selected targets are more dynamically realizable than the original route under stated conditions, or (b) give $\sigma_4$-triggered rerouting an explicit objective distinct from the geometric-escape mechanism's — for example, selecting among candidate routes $P$ by $\min_P J_{\mathrm{geometric}}(P)$ subject to $m_{\mathrm{phys}}(P) \ge m_{\mathrm{safe}}$ (a physical-feasibility *constraint* on route selection, not merely a preference), or constraining the geometric-escape target search to $\mathcal{F}_{\mathrm{dyn}}$-feasible candidates directly. Either would also let Theorem 3's adaptation class be extended from $\mathcal{A}_1(p)$ (retiming only) to include reshaping and, eventually, route selection itself, closing more of the gap to the idealized $\mathcal{T}_{\mathrm{dyn}}(p) = \emptyset$ condition discussed in Theorem 3's proof. This step is currently assumed, not shown, and must be closed before the Level 0–4 hierarchy's Level 3 mechanism (as opposed to its triggering condition) can be considered scientifically justified rather than a placeholder.

---

## VI. Theoretical Analysis

Throughout, we use $\xi_k$ generically for a tracking-error or deviation state where relevant to a specific result, and reuse the sets $\mathcal{F}_{\mathrm{kin}}$, $\mathcal{F}_{\mathrm{obs}}$, $\mathcal{F}_{\mathrm{dyn}}$, $\mathcal{F}_{\mathrm{safe}}$ from §III.

**Theorem 1 (Realizability certificate).** *Under bounded model uncertainty and bounded external/contact-force prediction error over the horizon, if $m_{\mathrm{phys}}(k+j) \ge 0$ for all $j \in [0, N]$, then the predicted nominal trajectory satisfies the actuator constraints over the horizon.*

*Proof strategy.* This is definitional once the assumptions are made explicit: $m_{\mathrm{phys}}(k+j) \ge 0$ means, by construction of $m_{\tau,i}^{\mathrm{robust}}$, that for every joint $i$ and every horizon step $j$, $\tau_{i,\max} - |\hat\tau_i(k+j)| - \Delta\tau_i(k+j) \ge 0$. If the true torque $\tau_i(k+j)$ satisfies $|\tau_i(k+j) - \hat\tau_i(k+j)| \le \Delta\tau_i(k+j)$ — the bounded-uncertainty assumption, restated as an explicit per-step guarantee rather than an unconditional claim — then $|\tau_i(k+j)| \le |\hat\tau_i(k+j)| + \Delta\tau_i(k+j) \le \tau_{i,\max}$ follows directly, for every joint and every horizon step, which is the statement that the actuator constraints hold over the horizon. The theorem is exactly as strong as the uncertainty bound $\Delta\tau_i$ used to construct it, and it certifies the constraint-inactive predicted trajectory *as predicted* — it does not, by itself, say anything about closed-loop behavior once feedback, replanning, or an inaccurate $\Delta\tau_i$ are in play; that is addressed separately by the assumptions attached to Theorems 2 and 3 and is not claimed here.

**Theorem 2 (Realizability-preserving adaptation).** *Suppose the nominal route admits at least one trajectory with $\mathcal{F}_{\mathrm{kin}} \cap \mathcal{F}_{\mathrm{obs}} \cap \mathcal{F}_{\mathrm{dyn}} \ne \emptyset$ (an explicit precondition). If the local adaptation problem (Level 1/2, §V-B) includes the predictive dynamic-feasibility constraint and is feasible, the resulting trajectory remains geometrically feasible while restoring actuator realizability, without necessarily changing the global route.*

*Proof strategy.* By the stated precondition, a nonempty target set exists within the current route's neighborhood. The Level 1/2 adaptation problem is posed as a constrained optimization over retiming/reshaping parameters subject to $\mathcal{F}_{\mathrm{kin}} \cap \mathcal{F}_{\mathrm{obs}} \cap \mathcal{F}_{\mathrm{dyn}}$ jointly, rather than $\mathcal{F}_{\mathrm{kin}} \cap \mathcal{F}_{\mathrm{obs}}$ alone as in a conventional retiming stage; if this constrained problem returns a feasible solution, that solution is a certificate of membership in the joint set by construction of the problem, and membership in $\mathcal{F}_{\mathrm{kin}} \cap \mathcal{F}_{\mathrm{obs}}$ in particular means the result is unchanged in its geometric route. The result is essentially an existence statement conditioned on solver feasibility, not a claim that the adaptation problem is always feasible when the precondition holds — the precondition guarantees *some* feasible trajectory exists along the route, not that the specific retiming/reshaping parameterization used by Level 1/2 is expressive enough to reach it. Closing that expressiveness gap for a specific parameterization is implementation-dependent and is addressed, for the retiming parameterization specifically, by Theorem 3 below.

**Theorem 3 (Sufficient, checkable condition for rerouting necessity).** *Let $p$ be a route and $\xi_0$ the nominal trajectory the local planner produces along it. For a retiming factor $\lambda$ in the allowed range $\Lambda = [1, \lambda_{\max}]$, let $\xi_\lambda$ denote $\xi_0$ time-reparameterized by $\lambda$ (Level 1, §V-B), and define the* retiming-adaptation class *$\mathcal{A}_1(p) = \{\xi_\lambda : \lambda \in \Lambda\}$ — the set of trajectories the architecture's pre-execution route decision actually searches before invoking Level 3 — and the* retiming-adaptation margin *$m^\star_1(p) = \sup_{\lambda \in \Lambda} m_{\mathrm{phys}}(\xi_\lambda)$, the aggregate horizon certificate (§IV) evaluated over the whole route under $\xi_\lambda$. Under the assumption that $\lambda \mapsto m_{\mathrm{phys}}(\xi_\lambda)$ is non-decreasing on $\Lambda$ (retiming monotonicity — stated as an explicit hypothesis, in the manner of Theorem 1's uncertainty bound, not derived from first principles for the general nonlinear case; see proof), if $m^\star_1(p) < m_{\mathrm{safe}}$, then no trajectory in $\mathcal{A}_1(p)$ restores the certificate, and Level 3 rerouting is architecturally sound: the specific, implemented pre-execution adaptation mechanism has been genuinely exhausted, not merely assumed exhausted.*

*This is a sufficient, not necessary, condition for the idealized statement that $\mathcal{T}_{\mathrm{dyn}}(p) = \emptyset$ (the set of* all *dynamically realizable trajectories along $p$, of which the observation "$\mathcal{T}_{\mathrm{dyn}}(p) = \emptyset \Rightarrow$ no modification staying on $p$ is dynamically realizable" is immediate from the definition, and is not by itself a substantial claim — an earlier version of this theorem stated exactly that idealized observation and presented it as the main result, which is the "almost tautological" framing a T-RO reviewer would reasonably object to). Since $\mathcal{A}_1(p) \subseteq \mathcal{T}_{\mathrm{dyn}}(p)$ whenever any element of $\mathcal{A}_1(p)$ happens to be dynamically feasible, $m^\star_1(p) < m_{\mathrm{safe}}$ does not imply $\mathcal{T}_{\mathrm{dyn}}(p) = \emptyset$: a dynamically realizable trajectory along $p$ may exist outside $\mathcal{A}_1(p)$, reachable only through a richer adaptation mechanism the current pre-execution decision does not attempt — most directly, Level 2 reshaping, which is available online (§V-B) but not yet folded into this route-level check. Closing that gap by defining a joint retiming-and-reshaping class $\mathcal{A}(p) \supseteq \mathcal{A}_1(p)$ is identified as a concrete next step, not yet implemented (§V-C, *Draft Status and Open Items*).*

*Proof.* Under the monotonicity assumption, $\sup_{\lambda \in \Lambda} m_{\mathrm{phys}}(\xi_\lambda) = m_{\mathrm{phys}}(\xi_{\lambda_{\max}})$, so $m^\star_1(p) < m_{\mathrm{safe}}$ is equivalent to $m_{\mathrm{phys}}(\xi_{\lambda_{\max}}) < m_{\mathrm{safe}}$, checkable with a single evaluation at the boundary of $\Lambda$ (in the implementation, a bisection search over $\Lambda$ that only needs to rule out $\lambda_{\max}$ to certify infeasibility of the whole class). Monotonicity itself is not proven for the fully general nonlinear case: under uniform time-dilation by $\lambda$, velocity and acceleration along the path scale as $1/\lambda$ and $1/\lambda^2$, so the torque contributions from the mass-matrix and Coriolis/centrifugal terms shrink in magnitude as $\lambda$ grows, while the gravity and any velocity-independent external-force contribution are unchanged by $\lambda$ — under this structure, slowing down cannot increase the torque required, but the argument does not rule out adversarial Coriolis cross-terms in principle, and the assumption is stated rather than derived for exactly that reason. Given the assumption, $m_{\mathrm{phys}}(\xi_\lambda) \le m_{\mathrm{phys}}(\xi_{\lambda_{\max}}) < m_{\mathrm{safe}}$ for every $\lambda \in \Lambda$, i.e. no element of $\mathcal{A}_1(p)$ restores the certificate, which is exactly the condition under which the implemented pre-execution decision correctly proceeds to Level 3. $\blacksquare$

This is empirically consistent with the one negative case already tested for it (§IX): a pure static (gravity-only) torque deficit, where the velocity/acceleration-driven contribution is zero throughout and retiming provably cannot help regardless of $\lambda$, correctly causes $m^\star_1(p) < m_{\mathrm{safe}}$ and a fall-through past Level 1 — not a proof that monotonicity holds in general, but a case where the assumption's own justification is exact rather than approximate. A related, separate subtlety worth stating explicitly: retiming $\xi_0$ to $\xi_\lambda$ changes which predicted environment sample $E(k+j)$ a given configuration is paired with whenever $E$ is genuinely a function of wall-clock time, $E = E(t)$ (an independently time-driven disturbance, unaffected by how fast the robot moves through it) rather than a function of configuration or path progress, $E = E(q)$ (e.g. a terrain/contact feature encountered at a specific point along the route, however long that takes to reach). For $E(q)$-type environments, retiming leaves the configuration-to-environment pairing unchanged and $m^\star_1(p)$ is evaluated consistently; for $E(t)$-type environments, slowing down changes which $E(k+j)$ a given point on the route encounters, so $m^\star_1(p)$ implicitly assumes the SAME predicted $E(t)$ schedule applies regardless of $\lambda$ — correct only if the environment prediction is itself re-sampled at the retimed schedule, which this architecture does (the implementation's force/environment interface takes $(t, q)$ jointly and is re-sampled at every $\lambda$ tested, so this is handled correctly in the current implementation) but is not yet stated as a distinct case in the theorem itself.

**Theorem 4 (Recursive feasibility / safe fallback) — deferred.** A terminal safe set $\mathcal{X}_f$ such that every $x_k \in \mathcal{X}_f$ admits a bounded safe braking policy satisfying actuator and collision constraints, with the recursive-feasibility property $x_k \in \mathcal{X}_f \Rightarrow x_{k+1} \in \mathcal{X}_f$, would strengthen Level 4 (§V-B) from a heuristic fallback into a formally guaranteed one. We have not constructed such a set or proved this property for the present architecture, and we do not claim it here. Per the source planning document's own guidance, this result is attempted only if the required assumptions can be established without becoming so restrictive as to be vacuous for the platforms in §VIII; if that turns out not to be achievable, Theorem 4 should be omitted from any submission-ready version of this paper and Level 4 presented as an engineered, empirically-validated fallback rather than a formally certified one. This is stated as an open item, not resolved.

**Computational structure.** A property we carry over unchanged from the underlying fixed-structure local planner: the prediction matrices and QP Hessian ($H, \Phi, \Gamma, C_{\mathrm{kin}}$) remain fixed online; state- and environment-dependent information enters only through vector terms ($h(x,E)$, $d(x,E)$, margin vectors), the same mechanism already used for obstacle rows. Adaptive numerical values are not the same as an adaptive solver structure, and preserving this distinction is what keeps the architecture real-time-capable as more predictive signals are added (§VII).

---

## VII. MoveIt 2 / Real-Time Implementation

The proposed integration preserves MoveIt 2's global planner (OMPL) unmodified and replaces only the local-planner component of MoveIt's Hybrid Planning interface, which is explicitly designed to accept custom global/local planner plugins and event-driven planning logic:

```
MoveIt Global Planner (OMPL)
        |
        v
  Global trajectory
        |
        v
Predictive-Realizability-Aware Local Planner
        |
        +-- fixed-structure local planner (nominal trajectory)
        +-- geometric-escape mechanism (sigma_1..sigma_3)
        +-- predictive physical-realizability certificate (sigma_4, Sec. IV)
        |
        v
   Controller --> Robot --> state/contact/force feedback
```

The local-planner slot runs the fixed-structure planner at its native rate, evaluates $m_{\mathrm{phys}}$ over the receding horizon each cycle using the current state estimate and the latest available environment/contact prediction, and triggers the Level 0–4 response of §V based on $\sigma_{\mathrm{geo}}$ and $\sigma_4$. Because the certificate and the geometric-escape mechanism both enter the underlying QP only through vector terms rather than restructuring it, the additional real-time cost of evaluating $m_{\mathrm{phys}}$ each cycle is a torque-prediction pass over the horizon (an inverse-dynamics evaluation per step) plus a margin computation, not a change in the QP's sparsity structure or size. §VIII-J reports the measured cost of this addition rather than asserting it is negligible.

---

## VIII. Experimental Evaluation — Proposed Benchmark Design

No experiments at the FR3 scale specified below have been run as of this draft; this section remains a design specification, not a results section, and every number below is a planned parameter, not a measured outcome. §IX reports a preliminary reduced-order check of the same design (certificate, hierarchy, ablations, real-time protocol) on a smaller, faster-to-iterate platform, and should not be read as satisfying the FR3-scale study specified here. All experiments in this section are scoped to what is reachable today (fixed-base manipulator, simulation with optional real-hardware validation); legged/humanoid extensions are described separately in §VIII-K and explicitly deferred pending the resourcing decision noted in *Draft Status and Open Items*.

### A. Platform and baselines

**Platform.** Franka FR3 (or Panda), MoveIt 2, ROS 2, MuJoCo (or Isaac Sim) as the primary simulator, with optional real-hardware validation on the FR3 if available. This platform is chosen because it has mature MoveIt integration, well-characterized per-joint torque limits, and manageable payload/interaction-force experiment design — matching Phase I of the source planning document's platform staging.

**Baselines and the system under test.**
- **B1 — MoveIt standard.** OMPL (global) $\to$ TOTG (time parameterization) $\to$ execution. No predictive feedback of any kind.
- **B2 — MoveIt Hybrid Planning, reactive local planner.** OMPL (global) + a conventional reactive local planner that responds to *current*-state saturation (e.g., clips or re-scales the command when the current torque exceeds a limit) but does not predict.
- **B3 — Proposed architecture.** OMPL (global) + fixed-structure local planner + geometric-escape mechanism + predictive physical-realizability certificate (this work).

B2 vs. B3 is the primary, fair comparison: both share the same global planner and the same local-planner slot in MoveIt's architecture, differing only in whether the local planner is reactive or predictive. B1 is retained as the unmodified-default reference point.

### B. Experiment 1 — Baseline manipulation trajectory

A representative reach-and-place trajectory (pick-and-place style, no payload change, static obstacles) is executed under B1/B2/B3. Metrics: planning time, execution time, peak per-joint torque, minimum torque margin achieved during execution, count of declared-acceleration-limit violations, tracking error (RMS and peak), and minimum obstacle clearance. This experiment establishes that the proposed architecture does not regress ordinary-case performance relative to B1/B2 before any of the harder scenarios below are introduced.

### C. Experiment 2 — Payload variation (authority loss without geometry change)

The identical geometric trajectory from Experiment 1 is re-executed at a sequence of payload masses spanning the manipulator's rated capacity (e.g., 0%, 40%, 70%, 90%, 100%, 110% of rated payload — the 110% point deliberately probes just past the nominal rating). Because the geometric path and time law are held fixed, this experiment isolates the central claim that identical geometry does not imply identical physical feasibility. Primary metric: at which payload level does each baseline first exhibit a torque-limit violation or tracking-error growth, versus at which payload level B3's certificate first reports $m_{\mathrm{phys}} < m_{\mathrm{safe}}$ and triggers Level 1/2 adaptation — the gap between these two payload levels (if any) is the quantity of interest, not a single pass/fail outcome.

### D. Experiment 3 — External interaction force (detection lead time)

During execution of the Experiment 1 trajectory, an external wrench $F_{\mathrm{int}}(t)$ is applied at the end-effector or a specified link (e.g., a scripted push profile with a ramp onset, magnitude swept across a range informed by the robot's rated payload/force capacity). Compared: B1 (no handling), B2 (reactive, current-state saturation handling only), B3 (proposed). Primary metric:

$$
T_{\mathrm{warning}} = T_{\mathrm{failure}} - T_{\mathrm{detection}},
$$

the lead time between when the system first has actionable information that a failure is coming and when the failure would actually occur under an unmodified nominal trajectory. $T_{\mathrm{warning}} = 0$ or undefined for B1 by construction (no detection mechanism); the comparison of interest is B2 versus B3, since B2 detects only once torque is already at the limit while B3 predicts the crossing before it occurs.

### E. Experiment 4 — Contact-stiffness/payload-step discontinuity (FR3-realizable terrain analogue)

The source planning document's original terrain experiment (flat/slope/hole/step) presumes a legged or mobile platform that is not part of the reachable Phase I scope (see §VIII-K and *Draft Status and Open Items*). We substitute an FR3-realizable analogue that preserves the experiment's actual point — that a trajectory can be collision-free while the *predicted* environment along the route changes what torque is required — without requiring a terrain rig: a step change in end-effector contact stiffness or an abrupt, scripted payload change (e.g., a simulated hand-off or a contact transition from free space to a stiff surface) at a known point along an otherwise fixed geometric trajectory. The environment-prediction signal $E_{k:k+N}$ is the known upcoming contact/stiffness transition; the comparison is whether B3's certificate anticipates the transition and adapts (Level 1/2) before the transition occurs, versus B1/B2 discovering the transition's effect only once already in contact. This experiment demonstrates the environment-conditioning contribution (§IV) without deferring it entirely to Phase II.

### F. Experiment 5 — Flagship: geometrically feasible, dynamically infeasible route

Construct two candidate routes between the same start and goal configuration: $P_A$, shorter in path length but passing through a region of high dynamic demand (e.g., a near-singular wrist configuration under payload, where large joint velocities are required for a given end-effector speed); and $P_B$, longer but remaining in a well-conditioned region throughout, under the same payload. A conventional planner selects $P_A$ on path-length grounds alone. This experiment is FR3-realizable without any terrain rig — the "environment" here is the manipulator's own configuration-dependent conditioning under a fixed payload, which is already fully characterized by the FR3's kinematics and dynamics model. Expected comparison: B1/B2 select and attempt $P_A$, discovering the dynamic infeasibility during execution (torque saturation, tracking failure, or a stall); B3's certificate predicts $m^\star_1(P_A) < m_{\mathrm{safe}}$ (retiming cannot restore margin) before committing to $P_A$ and triggers Level 3 rerouting to $P_B$. This is the experiment most directly validating Theorem 3, and is proposed as the paper's flagship result. Note (raised in review): a route through a near-singular configuration risks reading as a kinematic-conditioning problem rather than a physical-realizability one specifically; a version of this experiment built instead around payload- and external-force-driven torque demand at a well-conditioned but gravity-disadvantaged configuration would make the physical (not merely kinematic) nature of the deficit unambiguous, and is worth adopting before the FR3-scale study is run — not yet changed in this draft.

### G. Experiment 6 — Adaptation-versus-rerouting continuum

A single scenario family is parameterized by severity (e.g., a swept payload or interaction-force magnitude) to produce four regimes, directly validating the Level 0–4 hierarchy of §V-B rather than only its endpoints:
- **Case A (small margin loss):** Level 1 retiming alone is sufficient to restore $m_{\mathrm{phys}} \ge m_{\mathrm{safe}}$.
- **Case B (moderate loss):** retiming alone is insufficient; Level 2 reshaping is required and sufficient.
- **Case C (severe loss):** no retiming or reshaping along the current route suffices; Level 3 rerouting succeeds.
- **Case D (no safe route within the reaction horizon):** Level 4 braking/fallback is invoked.
Metric: at each swept severity level, which hierarchy level actually resolves the scenario, compared against which level the certificate *predicts* will be sufficient — the interesting failure mode to look for is the certificate under- or over-predicting the required response level, not merely whether the scenario is eventually resolved.

### H. Ablations

- **A1 — No predictive feedback.** Equivalent to B1.
- **A2 — Current-state saturation only, no prediction.** Equivalent to B2.
- **A3 — Prediction without planner feedback.** The certificate is computed and logged but not connected to the planner response (detects, does not adapt). Isolates whether prediction alone (as a monitoring signal, e.g. for an operator alert) has value distinct from acting on it.
- **A4 — Prediction + adaptation, no rerouting.** Levels 0–2 only; Level 3/4 disabled. Isolates the marginal value of rerouting specifically, expected to show degraded performance on Experiment 5/6-Case-C specifically.
- **A5 — Full proposed system.** Levels 0–4 (equivalent to B3).

### I. Metrics (full set)

Task success rate; collision rate; minimum actuator margin achieved; count of saturation events; tracking error (RMS, peak); minimum obstacle clearance; replanning count; replanning latency; local-planner per-cycle computation time; global-planner invocation rate; and **conservatism** — the fraction of executions in which the system triggers Level 2/3/4 response despite the nominal trajectory being, in fact, realizable (a false-positive rate on $\sigma_4$). Conservatism is reported alongside every safety-improvement claim in this paper, not as a secondary metric: an overly conservative certificate can trivially eliminate every failure mode above by triggering fallback constantly, which would make every other metric look good while the system is useless for its task. Any headline claim of reduced failure rate must be reported together with the corresponding conservatism figure.

### J. Real-time performance

Local-planner per-cycle computation time is measured directly (not assumed) for B2 and B3 under Experiment 1's nominal trajectory and under Experiment 5's near-singular-configuration stress case, reporting mean, p95, and maximum, against the local planner's target cycle rate. Whether the added certificate evaluation fits the real-time budget at the stress case specifically — not only the nominal case — is treated as an open empirical question the experiment must answer, not a design assumption.

### K. Deferred: legged/humanoid extensions (Phase II/III)

Uneven-terrain, footstep-adaptation, and humanoid-balance experiments matching the original terrain scenario in the source planning document remain valuable future extensions of this benchmark suite but are explicitly **not** part of the Phase I reachable scope, pending the resourcing assessment in *Draft Status and Open Items*: no legged/humanoid simulation or hardware pipeline currently exists in this line of work, and building a floating-base, contact-scheduled dynamics model is architecturally distinct from the fixed-base manipulator work above.

---

## IX. Preliminary Reduced-Order Verification

Since the benchmark design of §VIII was written, the mechanisms it specifies — the certificate, the Level 0–4 hierarchy, the ablation set, and the real-time protocol — have been implemented and run, not on the FR3-scale platform §VIII-A specifies, but on a smaller reduced-order system built to check whether the design behaves as intended before committing to that larger study. This section reports those results; §VIII's own FR3-scale benchmark, the $\mathcal{F}_{\mathrm{obs}}$/collision dimension, the MoveIt 2 integration of §VII, and any legged/humanoid extension (§VIII-K) remain unattempted.

### A. Scope and platform

A 3-degree-of-freedom planar revolute arm, moving in a vertical plane so gravity contributes a real, configuration-dependent torque demand, with per-joint torque limits (87/87/12 Nm) chosen to be FR3-representative in scale though not in kinematic structure. Dynamics are computed by a physics engine (MuJoCo) rather than hand-derived, and were independently checked before any certificate result was trusted: the mass matrix is symmetric and positive-definite over random configurations, the static torque at a known configuration matches a closed-form gravity calculation, and the task-space Jacobian matches a finite-difference check — a rigid-body model that is subtly wrong in an axis convention or a sign is a classic, easy-to-miss failure mode, and the certificate's claims are only as good as the torque prediction underneath them. The same three-baseline structure as §VIII-A is used (B1 no handling, B2 reactive current-instant throttling, B3 the proposed architecture), sharing one computed-torque tracking controller so that the B2-vs-B3 comparison stays fair.

### B. The certificate and hierarchy behave as designed

On a benign trajectory where the certificate never triggers, B1/B2/B3 are bit-for-bit identical (0.3mm final error, no saturation) — adding the predictive layer costs nothing when it is not needed. On the flagship scenario (§VIII-F's design: two routes to nearby targets under a fixed heavy payload, one through a persistently high-static-torque "outstretched" pose, one that stays in a low-demand "tucked" region), B1 and B2 attempt the shorter route and never reach the shared goal (peak torque ratio 1.00, final position error 0.94m and 0.82m, 45 and 54 saturation events respectively); B3's certificate predicts before execution that $m^\star_1(P_A) < m_{\mathrm{safe}}$ — retiming alone cannot restore margin along $P_A$, persisting even at 8x the maximum retiming factor tested — reroutes to the safe alternative, and reaches the identical goal with 0.001m final error and zero saturation events. This is Theorem 3 exercised end to end on real (if reduced-order) dynamics: the pre-execution decision that triggers Level 3 here is exactly the bisection search over $\lambda \in \Lambda$ the theorem formalizes, not a placeholder.

The A1–A5 ablations (§VIII-H) separate out why. On a scenario where retiming alone suffices, A3 (predict without acting) is bit-for-bit identical to A1 (no prediction at all) — same saturation count, same failure — a clean confirmation that a certificate which only detects and never feeds back into the planner has zero effect on outcomes; whatever value prediction-as-monitoring might have (the diagnostic signals of §IV, not evaluated here) is a separate claim from the one this architecture makes. On the flagship scenario, A4 (predict, retime, and reshape, but Level 3/4 disabled) fails, and fails *no better than doing nothing* (47 vs. 45 saturation events, final position error 0.98m vs. A1's 0.94m, neither reaching the shared goal), while A5 (the full hierarchy) succeeds cleanly, reaching the identical goal with 0.001m error and zero saturation events — isolating that rerouting specifically, not adaptation in general, is load-bearing here, exactly the separation §VIII-H's ablation design was built to produce.

### C. Detection lead time and conservatism

Under an unanticipated external disturbance (Exp. 3, §VIII-D), B2's reactive check fires only once torque is already at the limit ($T_{\mathrm{warning}} = 0$ by construction); B3's certificate fires 0.42s earlier, using only a 0.3s bounded online prediction horizon, not oracle knowledge of the disturbance schedule. When the same class of disturbance is instead known in advance as part of the predicted environment (Exp. 4, §VIII-E: a contact-stiffness transition), B3 adapts before the transition occurs and finishes with zero saturation events against 27 (B1) and 8 (B2) for the two baselines. Conservatism (§VIII-I): across a payload sweep spanning no-adaptation-needed to severe deficits (Exp. 2, §VIII-C), the certificate triggered adaptation at 13 of the tested payload levels, and zero of those 13 triggers were false positives. This is a single-scenario-family result, not a general conservatism bound, but it is the direction the architecture needs to succeed in, and is reported precisely because an overly conservative certificate could otherwise make every other number above look good while triggering fallback constantly.

### D. Negative and boundary results

Two findings argue against overstating the architecture's benefit and are kept here rather than dropped. First, **partial adaptation is not always better than none**: in the payload sweep, once payload exceeds the point where Level-1 retiming can fully restore margin (2.6–3.0kg in this platform), the partially-retimed trajectory leaves the arm in a *worse* final tracking state than the unmodified baseline (e.g. 189.6mm vs. 117.2mm final error at 3.0kg) — a certificate-triggered correction that only partially succeeds is not guaranteed to dominate doing nothing. Second, **Level 4 is a safety outcome, not a task-completion outcome**, and this is a real trade-off rather than an incidental implementation detail: consistent with Level 4's "terminal safe-set policy" framing (§V-B) and Theorem 4's deferred status, once triggered it holds the robot at its stopped configuration rather than resuming automatically — confirmed necessary during implementation, where an earlier version that let the certificate re-evaluate the original plan after a stop produced an unstable closed loop once the robot had genuinely diverged from it. The consequence: past roughly 30N in Exp. 3, B3 fails the task via a permanent hold at disturbance magnitudes B1/B2 still complete despite transient saturation — a genuine safety-vs-completion trade-off, not a strict win. A further, physically real limitation surfaced in the same experiment: once a growing disturbance reaches its full magnitude, the *static* holding torque at whatever pose Level 4 stopped in can itself exceed the actuator limit — a fixed hold provides no guaranteed robustness against a disturbance that outgrows what can be held statically, and closing this would need either a Level 4 that re-picks its holding pose or a coupling to Level 3 rather than freezing at the first trigger, left as future work. Third, at the payload sweep's actual task-success crossover (2.4kg — corrected after rerunning at finer resolution than an earlier pass, which had mistakenly reported 3.0kg), B2's reactive throttle succeeds exactly as well as B3's predictive retiming; this specific scenario does not demonstrate an advantage for prediction over reaction, which instead shows up in Exp. 3–5, not uniformly.

### E. Real-time cost of the certificate

On the benign trajectory, B3's online per-cycle cost is negligible (mean 0.27ms against the 20ms/50Hz budget used throughout §VIII). Under the flagship stress case, the online step that matters is a single Level-2 QP solve triggered before the certificate falls back to (sticky) braking: pooled over repeated measurements, this one-shot solve costs roughly 27–54ms — 135–270% of the budget, i.e. it EXCEEDS the budget outright rather than merely approaching it — and disabling Level 2 isolates that the QP (solved via a general-purpose convex solver that reconstructs the optimization problem from scratch on every call rather than reusing a compiled, parametrized form) accounts for approximately 100% of the online-step cost whenever it fires. (This cost roughly doubled after a code-vs-paper review found the QP was not dynamically consistent -- see §V-B's Level 2 description -- and fixing it added state-trajectory variables and integration constraints to the same problem; the increase is the honest price of the QP now certifying a trajectory it can actually produce, not a regression to be hidden.) This is reported as a genuine, unresolved implementation limitation: a single Level-2 solve costs more than the entire real-time budget on its own, and closing this gap would need a parametrized/warm-started QP formulation before any real-time deployment against §VII's MoveIt integration.

### F. What this does and does not establish

This is a 3-DOF planar reduced-order platform built to check that the certificate, hierarchy, and Theorem 3's rerouting claim behave as designed on a system simple enough to verify independently at each step — not the FR3-scale evaluation §VIII specifies. §VIII's own benchmark design, ablation set, and real-time protocol were followed as written; what remains unattempted is the FR3-scale platform itself, the obstacle/collision feasibility set $\mathcal{F}_{\mathrm{obs}}$ (out of scope for this reduced-order study, which addresses $\mathcal{F}_{\mathrm{dyn}}$ only), the MoveIt 2 integration of §VII, and any legged/humanoid extension. Complete code, every scenario parameter behind the numbers above, and a documented account of three implementation bugs found and fixed during this verification effort — kept visible rather than silently corrected, in keeping with this draft's own standard for what stays open — are available in the accompanying repository.

---

## X. Discussion

**Conservatism versus responsiveness.** Every mechanism in §V trades safety margin against unnecessary intervention; §VIII-I's conservatism metric is the paper's answer to the concern that any predictive-safety architecture can trivially "solve" every failure mode by refusing to act. §IX-C's reduced-order sweep reported zero false-positive triggers, a promising direction but, being a single-scenario-family, reduced-order result, not yet the general conservatism bound this concern needs answered at FR3 scale.

**Model and environment-prediction uncertainty.** Theorem 1's guarantee is exactly as strong as the uncertainty bound $\Delta\tau_i$ used to construct the robust margin; if the environment prediction $E_{k:k+N}$ is itself wrong (e.g., an unanticipated contact), the certificate degrades gracefully to the accuracy of that prediction rather than failing catastrophically, but this claim is not yet backed by an experiment specifically stressing environment-prediction error and should not be asserted more strongly than that.

**Sensing latency and computational scaling.** §IX-E's reduced-order measurement found the online certificate check itself negligible but a single Level-2 QP solve costing close to the entire real-time budget on its own — a genuine, unresolved cost that should be treated as representative of the concern this paragraph raises, not fully addressed by it: the FR3-scale measurement called for in §VIII-J remains to be done, and a parametrized/warm-started QP formulation is the most likely fix if the same bottleneck reappears at scale.

**Relationship to learned/world-model planners.** §XII sketches a longer-term architecture in which a learned high-level planner (a world model or vision-language-action policy) determines *what* the robot should do, this architecture's local planner determines *how* it should move, and the predictive-realizability layer determines *whether* the robot can physically execute that behavior under current and predicted conditions. This is presented as a long-term direction, not a near-term deliverable of the present paper, and no learned-planner integration has been attempted.

---

## XI. What This Paper Does Not Claim

To keep the contribution claims defensible, this paper explicitly does not claim: to be the first torque-aware or actuator-aware motion planner; that MoveIt cannot handle dynamic constraints; that the proposed method guarantees safety under arbitrary disturbance; that predictive-saturation feedback guarantees a legged platform will not fall; or that no prior work has considered torque limits in planning. Reference and Explicit Reference Governors have predicted a margin and fed it back before constraint violation since the 1990s (§II-C); retiming under predicted torque saturation is an established sub-field (§II-D); and terrain/contact-conditioned actuation limits in legged planning have prior published treatment that this paper must cite and differentiate from, not merely acknowledge in passing (§II-D). The claim this paper makes is narrower: a predictive feedback interface that converts future physical-realizability information from the execution layer into motion-planning adaptation and, when necessary, route-level replanning, unified across geometric and physical failure modes under one hierarchical response — not the existence of torque limits, margins, or saturation constraints considered individually.

---

## XII. Long-Term Direction (not a deliverable of this paper)

$$
\begin{gathered}
\text{World model / VLM} \to \text{Global motion planner} \to \text{Predictive physical realizability} \\
{} \to \text{Whole-body / interaction controller} \to \text{Robot + environment},
\end{gathered}
$$

with the loop closed by re-planning on the realizability layer's own feedback. The high-level system decides *what* the robot should do; the planner decides *how* it should move; the realizability layer decides *whether* the robot can execute that behavior under current and predicted conditions. This is proposed as a long-term organizing idea for a model-based Physical AI stack, not a contribution defended by any result in this draft.

---

## XIII. Conclusion

Motion planning pipelines certify that a trajectory can be geometrically generated and kinematically executed; they do not, in general, continuously ask whether the robot will remain physically capable of realizing that trajectory as configuration, contact, payload, and disturbance evolve. This paper proposes predictive physical realizability as the missing feedback signal — a receding-horizon certificate on actuator margin, conditioned on predicted environment and contact information, coupled to a hierarchical planner response that adapts before rerouting and reroutes before falling back, with an explicit theorem-backed condition for when adaptation is provably insufficient. The architecture is designed to occupy an existing, currently-empty slot in MoveIt 2's own Hybrid Planning interface rather than propose a new integration target. What remains is empirical: whether the certificate's prediction lead time is large enough to matter in practice, and whether the conservatism cost of acting on it is small enough to be worth paying — both questions §VIII's benchmark design is built to answer, and both given a first, reduced-order answer in §IX (a 0.42s detection lead on an unanticipated disturbance; zero false-positive triggers across a 13-point sweep) that is encouraging but explicitly not a substitute for the FR3-scale study §VIII specifies. §IX also surfaced the trade-offs a reduced-order platform makes visible early and cheaply: partial adaptation is not always better than none, a terminal safety response trades task completion for safety rather than guaranteeing both, and a single QP solve in the current implementation costs close to the entire real-time budget on its own. None of these close the gap to a submission-ready result; they are the reason the honest next step is the FR3-scale study, not a claim that this draft has already taken it.

---

## References

[1] M. Prats, "TOTG not respecting acceleration limits," moveit/moveit2 GitHub Issue #2600, reported Dec. 2023, closed Jan. 2024. [Online]. Available: https://github.com/moveit/moveit2/issues/2600

[2] E. Garone, S. Di Cairano, and I. Kolmanovsky, "Reference and command governors for systems with constraints: A survey of their theory and applications," *Automatica*, vol. 75, pp. 306–328, 2017.

[3] A. Bemporad, A. Casavola, and E. Mosca, "Nonlinear control of constrained linear systems via predictive reference management," *IEEE Transactions on Automatic Control*, vol. 42, no. 3, pp. 340–349, 1997.

[4] S. Zhang, J. Y. Z. Liu, and D. Liao-McPherson, "Integrating Planning and Predictive Control Using the Path Feasibility Governor," arXiv:2507.09134, Jul. 2025. Submitted to *IEEE Transactions on Automatic Control*.

[5] R. Orsolino, M. Focchi, S. Caron, G. Raiola, V. Barasuol, and C. Semini, "Feasible Region: an Actuation-Aware Extension of the Support Region," *IEEE Transactions on Robotics*, vol. 36, no. 4, pp. 1239–1255, Aug. 2020. arXiv:1903.07999.

[6] B. Acosta and M. Posa, "Perceptive Mixed-Integer Footstep Control for Underactuated Bipedal Walking on Rough Terrain," *IEEE Transactions on Robotics*, vol. 41, pp. 4518–4537, 2025. arXiv:2501.19391.

All six entries above were independently verified against their primary source (GitHub issue pulled directly via `gh issue view`; papers checked via arXiv/publisher pages) during this draft's review process, not copied from the source planning documents without checking — see *Draft Status and Open Items*, items 1 and 4, for what that verification did and did not confirm. Two further claims in §II-D (disturbance-observer-based retiming "in the early 2000s"; a task-space torque-limit projection for non-flat-terrain trajectory optimization) are carried over from the source planning document without a specific citation and are not included here — see *Draft Status and Open Items*, item 8.

---

## Draft Status and Open Items

This section is deliberately not folded into the Discussion, so it cannot be silently dropped in a later editing pass. None of the following should be treated as resolved:

1. **Unverified citations.** Only [moveit/moveit2#2600](https://github.com/moveit/moveit2/issues/2600) ([1] in *References*) is cited as a confirmed pathology (§I). Two additional MoveIt 2 issues referenced in an earlier internal pass (a reported 2.75× acceleration-limit overshoot on a Panda; an acceleration-scaling-factor bug) could not be re-located on a follow-up search and are not cited anywhere in this draft. If they are to be used, each candidate issue number must be opened and read directly before citing. A later review pass pulled #2600 directly and confirmed the cited numbers exactly (200 rad/s² against a 0.2 rad/s² limit) — but the issue's own reporter, in a follow-up comment on the same thread, attributes the failure to an unsupported edge case (a full 180° joint turn) and a default `path_tolerance=0.0` in the raw TOTG class that MoveIt's standard `RobotTrajectory` wrapper already avoids by setting a different default. §I currently presents this as a clean, general pathology; that framing is now known to be stronger than the source supports and should be softened or replaced with a more clear-cut example before submission. Not yet fixed in the main text.
2. **The Level 3 / geometric-escape mechanism gap (§V-C) is partially closed, not fully.** A review pass identified that the original Theorem 3 was "almost tautological" (its own proof strategy admitted "this is immediate from the definition..."), a genuine T-RO-level desk-reject risk, not a stylistic nitpick. Theorem 3 has been replaced with a version that defines a concrete, implemented adaptation class $\mathcal{A}_1(p)$ (retiming only, matching the code's actual pre-execution route decision exactly) and states $m^\star_1(p) < m_{\mathrm{safe}}$ as a sufficient, checkable condition for rerouting, explicitly relating it to (rather than conflating it with) the idealized, uncomputable $\mathcal{T}_{\mathrm{dyn}}(p) = \emptyset$ statement. This closes the "*when* should the architecture stop adapting and reroute" question with a real, non-trivial, code-verified result. It does **not** close the separate "*what should the rerouter then do*" question §V-C already flagged — either proving geometric-escape-selected targets restore $\mathcal{F}_{\mathrm{dyn}}$, or giving rerouting an explicit physical-feasibility objective/constraint (a sketch is now in §V-C) — nor does it fold Level 2 reshaping into the pre-execution adaptation class, which would make $\mathcal{A}_1(p)$ a proper subset of a richer, still-sound $\mathcal{A}(p)$ and further shrink the gap to the idealized condition. Both remain open, tracked separately from the (now resolved) tautology concern.
3. **Theorem 4 (recursive feasibility) is explicitly deferred**, not proved, not disproved, and not attempted in this draft. A decision to attempt it, or to drop Level 4 to an empirically-validated (not formally certified) fallback, has not been made.
4. **Prior art requiring full reading before any submission:** the Path Feasibility Governor paper (arXiv:2507.09134, [4]) has been summarized from its abstract-level description in the source planning document, not read in full; Acosta and Posa, *IEEE T-RO* 2025 ([6]), has not been read at all and must be read before any submission touching legged or terrain scenarios, since it is published in this paper's own target venue. A later review pass confirmed both papers' titles, authors, venues, and years exactly as cited in §II-C/D and in the *References* section (Zhang, Liu, and Liao-McPherson, *IEEE Trans. Automatic Control*, submitted July 2025; Acosta and Posa, *IEEE T-RO* 41, 2025, pp. 4518–4537, arXiv:2501.19391) — the bibliographic entries are not fabricated — but neither has been read in full, and the differentiation argument in §II-C in particular still rests on the abstract-level summary only.
5. **Legged/humanoid resourcing is unresolved.** No legged/humanoid simulation or hardware pipeline exists in this line of work as of this draft. Phase II/III should not be treated as required scope for a first submission until a real build-cost estimate (plausibly 6–12 months per the source planning document, not confirmed here) has been obtained. §VIII-K reflects this by scoping all designed experiments to the fixed-base manipulator.
6. **Publication sequencing.** Per the source planning document (§14), this paper should not proceed to substantive external submission until `mp_main` (ICRA 2027) and HAE (IJSS) have received decisions, and the recommended next concrete step is a separate, narrower saturation-certificate paper (targeting SCL/L-CSS) that proves Theorems 1–3 above independent of this paper's broader scope. This draft exists so that structure, positioning, and benchmark design are ready when that sequencing allows work to continue — it is not itself the next thing to do.
7. **§VIII's FR3-scale experiments have not been run; a reduced-order proxy has (§IX), and its results are folded in.** Every number, scenario parameter, and expected-outcome description in §VIII itself remains a design choice, not a measured result — that has not changed. What has changed since this item was first written: the same certificate, Level 0–4 hierarchy, ablation set, and real-time protocol were implemented and run on a 3-DOF planar reduced-order platform, and §IX reports those results, including negative and boundary findings (partial adaptation sometimes worse than none; Level 4 trading task completion for safety rather than guaranteeing both; a single Level-2 QP solve costing more than the entire real-time budget) that were kept visible rather than smoothed over. This draft contains zero fabricated data in either §VIII or §IX; §IX's own text states plainly where its reduced-order platform stops short of what §VIII specifies.
8. **Two claims in §II-D still lack a specific citation.** "Disturbance-observer-based path tracking under torque saturation in the early 2000s" (the claimed historical root of retiming as a mature sub-field) and a "task-space projection of torque limits for trajectory optimization on non-flat terrain" (a related-work claim distinct from the now-cited Orsolino/Focchi *Feasible Region* work, [5]) were both carried over from the source planning document without an author or title. Rather than invent a plausible-sounding citation for either, §II-D now states plainly that these are uncited; a specific paper must be identified and read for each before submission, or the claims should be removed or rephrased as general disciplinary background not requiring a citation.
9. **Two real bugs in §IX's implementation, found by an external code-vs-paper review and independently verified before fixing, are now fixed and the affected §IX numbers above updated to match.** (a) Exp5/6's B3 "reroute" was changing the GOAL, not the route: $P_A$ and $P_B$ ended at two different final configurations, and the success metric scored "whichever goal was actually reached" — B3 could pass by abandoning the task for an easier target, which does not support this paper's actual Theorem 3 claim (a physically unrealizable route rejected in favor of a realizable route to the SAME goal). §VIII-F's design text already specified "two candidate routes between the same start and goal configuration" — the paper's design was correct; the implementation did not match it. Fixed with a two-segment via-point trajectory representation so $P_A$ and $P_B$ genuinely share $q_0$ and $q_g$; the same bug, independently found in the ablation-batch code during this fix (the review did not name it), is fixed there too. (b) The replanning-count metric was re-testing a flag set once at plan time on every control cycle, so it counted roughly one "replan" per cycle for the rest of a rollout whenever Level 1/3 ever fired, rather than counting actual replanning events; fixed to count the route decision once, brake engagement once, and Level-2 corrections on rising edges only. A third, smaller issue was found while verifying these fixes with real reruns (not assumed): the reduced-order experiment scripts sized rollout duration off the nominal trajectory length, but Level 1 can retime a route up to $\lambda_{\max}$ longer, so a retimed rollout was sometimes truncated before finishing; fixed by sizing duration to what the route decision actually selects. Separately, an independent code-vs-paper review of §V-B's Level 2 (reshaping) found it optimized the acceleration profile against the nominal trajectory's own $(Q,\dot Q)$ held fixed, with no constraint tying the optimized accelerations back to a state trajectory they could actually produce — so the certificate evaluated over the result did not certify a trajectory the system could actually be commanded to follow. Fixed by adding explicit double-integrator state variables and integration constraints to the same QP, anchored at the true current state; this is a materially larger optimization problem, and its real cost is reflected in the roughly doubled Level-2 QP timing figures above. B2's one-step reactive throttle (a uniform scaling of $\ddot q$ and $\dot q$ by $\tau_{\max}/\tau_0$) was also found not to guarantee its own claimed torque bound, since the gravity/Coriolis/external-force term in $\tau = M(q)\ddot q + h(q,\dot q,F)$ does not scale with $\ddot q$; replaced with an exact one-step QP projection onto the torque-feasible set at the same $(q,\dot q)$, which is what §VIII-J's and §IX's updated B2 numbers above now reflect.
