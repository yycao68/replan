# Predictive Physical Realizability: A Certified Interface Between World Models and Physical Robots

**Formal core, drafted first and deliberately without an Introduction.** This document fixes the four parts that decide whether the paper has research value: the problem formulation, the definition of predictive physical realizability, the conditional certificate, and three theorems. If these hold up, the rest of the paper is writing. If they do not, no amount of framing rescues it. Everything here is stated so that it can be attacked directly.

**Relationship to the companion paper.** `predictive_realizability_paper_draft.md` (hereafter **P1**) proposes a receding-horizon actuator-margin certificate, a Level 0–4 planner response, and Theorems 1–5 on a fixed-base manipulator. Its §XII already sketches the chain *world model → planner → realizability → controller → robot* and explicitly labels it "a long-term organizing idea… not a contribution defended by any result in this draft." **This paper is the promotion of that sketch to a defended result.** That gives useful continuity but sets a hard bar: the formal content below must go beyond P1's certificate, or this is P1 with a new introduction. §6 states the delta explicitly and §7 states what would kill the paper.

---

## 1. Problem Formulation

### 1.1 The chain, and where it is broken

Four stages, four different modalities, currently connected by nothing that checks the transition between the third and the fourth:

$$
\underbrace{\mathcal{W}}_{\text{world model}} \xrightarrow{\ \text{what \emph{may} happen}\ }
\underbrace{\mathcal{P}}_{\text{planner}} \xrightarrow{\ \text{what \emph{should} happen}\ }
\underbrace{\mathcal{R}}_{\text{realizability}} \xrightarrow{\ \text{what \emph{can} happen}\ }
\underbrace{\Sigma}_{\text{plant}} \xrightarrow{\ \text{what \emph{does} happen}\ }
$$

with the loop closed by returning the plant's observations to $\mathcal{W}$ and the realizability verdict to $\mathcal{P}$.

The claim this paper is organized around is not that stage $\mathcal{R}$ is missing — feasibility checking exists in many forms. It is that **stage $\mathcal{R}$ is currently defined against the wrong object**. Existing dynamic-feasibility machinery evaluates a trajectory against *a* model of the environment, treating that model as given. When the environment model is a learned generative world model, that treatment is unsound in a specific and consequential way: the world model's output is a *hypothesis about the future*, not a measurement, and it can be wrong in ways that invalidate the plan and the safety fallback simultaneously.

### 1.2 Plant and the robot/environment partition

Let $x = (q,\dot q) \in \mathcal{X} \subseteq \mathbb{R}^{2n}$ be the robot state, $u \in \mathbb{R}^n$ a commanded generalized acceleration, and $\tau \in \mathbb{R}^n$ the actuator torque. The plant is

$$
M(q)\,\ddot q + C(q,\dot q)\,\dot q + g(q) + D(\dot q) \;=\; \tau \;+\; \textstyle\sum_c J_c(q)^\top \lambda_c \;+\; w .
$$

Partition everything that appears here into two parts with **different epistemic status**, because the whole paper turns on that difference:

- **Robot-side model** $\mathcal{M}_R = \big(M, C, g, D, \{J_c\}, \mathcal{T}\big)$, with $\mathcal{T} = \{\tau : |\tau_i| \le \tau_{i,\max}\}$. Identified offline from the platform, verifiable by system identification, and *independent of where the robot is*. We assume a known bound $\varepsilon_R \in \mathbb{R}^n_{\ge 0}$ on the residual robot-model error.
- **Environment realization** $e \in \mathfrak{E}$, collecting everything the robot does not carry with it: terrain geometry, contact schedule and locations, friction coefficients, payload, interaction and disturbance wrenches. $e$ determines $\lambda_c$ and $w$.

Define the **required-torque map**

$$
\tau_{\mathrm{req}} : \mathcal{X} \times \mathbb{R}^n \times \mathfrak{E} \to \mathbb{R}^n,
\qquad
\tau_{\mathrm{req}}(x,u,e) = \text{torque required to realize } u \text{ at } x \text{ under } e .
$$

*Standing assumption (A0).* $\mathfrak{E}$ is a compact metric space and $\tau_{\mathrm{req}}$ is continuous in $e$ for each $(x,u)$. This is the minimum needed for the suprema below to be attained; it is satisfied by the usual finite-dimensional parameterizations (wrenches, friction cones, payload masses, terrain heights over a bounded grid). It is *not* satisfied by contact-mode switching treated as a discrete jump, which is handled by taking $\mathfrak{E}$ to be a finite union of compact continuous pieces — stated here rather than glossed, since contact-mode discreteness is exactly where this kind of formulation usually breaks.

### 1.3 Two failure modes that must not be conflated

Let $\xi = (x_{k+j}, u_{k+j})_{j=0}^{N}$ be a planned trajectory over a horizon of $N$ steps and $e^\star$ the environment realization that actually occurs.

- **(F1) In-hypothesis infeasibility.** The environment behaves as the world model said, and the plan still demands more torque than the robot has. This is decidable *before execution*, given the environment model.
- **(F2) Hypothesis falsification.** The environment does *not* behave as the world model said. This is **not** decidable before execution by any amount of planning-time computation, because the information required to detect it does not exist yet. It is detectable only at runtime, from the mismatch between predicted and observed dynamics.

Dynamic-feasibility-aware planning, trajectory optimization under torque limits, and time-parameterization under actuator bounds all address (F1). This paper's position is that **(F2) is the failure mode that governs whether a world-model-driven robot is safe, and that it requires a different formal object than (F1)** — not a tighter feasibility check, but a certificate that carries its own assumptions explicitly enough to be falsified at runtime, plus a recovery path that does not depend on the assumption that just failed.

That sentence is the paper's whole claim. §7 discusses what has to be true of the literature for it to survive.

---

## 2. Definition of Predictive Physical Realizability

**Definition 1 (Environment hypothesis).** An *environment hypothesis* issued at time $k$ over horizon $N$ is a sequence of nonempty compact sets

$$
\mathcal{H}_E^k \;=\; \big(\mathfrak{E}_{k+j}\big)_{j=0}^{N}, \qquad \mathfrak{E}_{k+j} \subseteq \mathfrak{E},
$$

emitted by the world model $\mathcal{W}$ from the observation history. Semantically, $\mathcal{H}_E^k$ is the **assertion** $e^\star_{k+j} \in \mathfrak{E}_{k+j}$ for all $j$. It is an assertion, not a fact, and the rest of this paper is organized around never forgetting that.

*Remark (where the sets come from).* A generative world model emits samples or a distribution, not sets. $\mathfrak{E}_{k+j}$ is obtained from that output by any construction with a stated coverage property — a quantile region over sampled rollouts, a calibrated conformal region, or a bounded-support latent decoded through a known map. The paper is agnostic to which; what it requires is that the construction come with an explicit, checkable coverage claim, because that claim is exactly what the runtime monitor of §3.2 tests. A world model that emits only a point prediction is admissible with $\mathfrak{E}_{k+j}$ a singleton, and then every guarantee below degenerates to a nominal one — which is the correct and honest degeneration, not a failure of the framework.

**Definition 2 (Realizability margin).** For a planned trajectory $\xi$, robot model $\mathcal{M}_R$ with error bound $\varepsilon_R$, and hypothesis $\mathcal{H}_E$,

$$
\boxed{\;
\rho\big(\xi \mid \mathcal{M}_R, \mathcal{H}_E\big)
\;=\;
\min_{0 \le j \le N}\ \min_{1 \le i \le n}
\Big[\;
\tau_{i,\max}
\;-\;
\sup_{e \in \mathfrak{E}_{k+j}} \big| \tau_{\mathrm{req},i}(x_{k+j}, u_{k+j}, e) \big|
\;-\;
\varepsilon_{R,i}
\;\Big].\;}
$$

By (A0) the supremum is attained, so $\rho$ is well defined and the worst case is realized by an admissible environment.

**Definition 3 (Predictive physical realizability).** $\xi$ is *predictively physically realizable under $\mathcal{H}_E$*, written $\xi \in \mathrm{PPR}(\mathcal{H}_E)$, iff $\rho(\xi \mid \mathcal{M}_R, \mathcal{H}_E) \ge 0$.

### 2.1 What this is not

The paper lives or dies on these three distinctions being real, so they are stated as separable claims rather than asserted in passing:

| | quantified over | temporal scope | set is chosen by | can the set be *wrong*? |
|---|---|---|---|---|
| **Dynamic feasibility** | one nominal environment | the present instant | the modeller | not a question the formalism asks |
| **Robust / tube MPC** | a disturbance set | the horizon | the designer, offline, fixed | no — assumed valid by construction |
| **Reference governor** | a known constraint set | the horizon | the designer, offline, fixed | no |
| **PPR (Def. 3)** | a hypothesis set | the horizon | **a learned model, online, per cycle** | **yes, and detecting that is part of the architecture** |

The load-bearing column is the last one. Robust MPC and reference governors both quantify over sets and both look ahead; neither has a notion of its own set being falsified, because in both the set is a designer-chosen invariant of the problem. When the set is emitted by a learned model each cycle, "is the set right?" becomes a first-class runtime question, and answering it is what §3.2 and Theorem 2 are for.

*This table is a claim about the literature, not a verified survey.* It must be checked against contract-based design, runtime assurance, conformal-prediction-based safe planning, and residual-based fault detection before submission; §7 says why each is a genuine threat and what would have to be true for the claim to stand.

---

## 3. The Conditional Certificate

### 3.1 Definition and the entailment form

**Definition 4 (Conditional certificate).** Let $\mathcal{C}_R(\xi)$ denote the robot-side property "executing $\xi$ keeps every actuator within its limit over the horizon." A *conditional certificate* is the entailment

$$
\boxed{\;\mathcal{H}_E \;\models\; \mathcal{C}_R(\xi)\;}
$$

together with a witness — here, $\rho(\xi\mid\mathcal{M}_R,\mathcal{H}_E) \ge 0$ — establishing it.

The certificate is **unconditional in the robot model and conditional in the environment**. It never asserts that the world model is right. It asserts precisely what follows *if* it is, and it makes the "if" a named, monitorable object rather than an implicit modelling convenience. This is the structural difference from a feasibility check: a feasibility check returns a Boolean; a conditional certificate returns a Boolean *together with the assumption that Boolean is contingent on*, in a form that can be tested against reality while the robot is moving.

*Honest antecedent.* This is an assume–guarantee contract in the sense of contract-based design for cyber-physical systems. The paper should say so plainly and claim novelty not for the contract form but for (i) the antecedent being **machine-generated per cycle by a learned model** rather than designer-fixed, and (ii) the consequences that follow from that, which are Theorems 2 and 3. Claiming novelty for the entailment form itself would be wrong and a reviewer would say so.

### 3.2 Falsification monitor

The certificate's antecedent is testable because a violated environment hypothesis leaves a signature in the dynamics residual.

**Definition 5 (Residual and detection threshold).** With $\hat e_k$ the hypothesis' nominal (e.g. central) realization, define the residual

$$
r_k \;=\; \tau_{\mathrm{applied},k} \;-\; \tau_{\mathrm{req}}\big(x_k, \hat u_k, \hat e_k\big),
$$

evaluated at the *measured* state and the *applied* torque. Under $\mathcal{H}_E$ the residual is bounded by

$$
\beta \;=\; \sup_{e \in \mathfrak{E}_k} \big\| \tau_{\mathrm{req}}(x_k,\hat u_k, e) - \tau_{\mathrm{req}}(x_k, \hat u_k, \hat e_k) \big\|_\infty \;+\; \|\varepsilon_R\|_\infty ,
$$

which is computable from the hypothesis itself. The **falsification monitor** fires at

$$
t_d \;=\; \inf\{\, t \;:\; \|r(t)\|_\infty > \beta \,\}.
$$

**Definition 6 (Contract state).** At each cycle the system occupies exactly one of

$$
\textsf{VALID}\ (\rho \ge \rho_{\min},\ \text{monitor silent}), \qquad
\textsf{MARGINAL}\ (0 \le \rho < \rho_{\min}), \qquad
\textsf{FALSIFIED}\ (\text{monitor fired}),
$$

and the planner's response is indexed by this state, not by the margin alone.

*Why this is architecturally new relative to P1.* P1 already has two trigger classes and is emphatic that they must not be collapsed: $\sigma_{\mathrm{geo}} = \sigma_1 \vee \sigma_2 \vee \sigma_3$ (geometric/topological) and $\sigma_4$ (margin), in its §V-A. The novelty here is not a second trigger but a second *kind* of trigger: both of P1's are statements about the **plan** — it is geometrically stuck, or it is physically infeasible — and neither can fire on the environment model being wrong while the plan still looks fine. Here there are **two logically independent triggers**: margin exhaustion (a statement about the plan) and hypothesis falsification (a statement about the world model). A falsification event is not a margin event and must not be handled as one: the margin can be large and healthy at the exact moment the hypothesis it was computed under becomes false. Conflating them is the specific error this paper argues against, and Theorem 3 shows it has teeth.

---

## 4. Three Theorems

### Theorem 1 (Conditional soundness, tightness, and attribution of conservatism)

*Let $\xi$ be a planned trajectory and $\mathcal{H}_E = (\mathfrak{E}_{k+j})_j$ an environment hypothesis. Then:*

*(i) **Soundness.** If $\rho(\xi\mid\mathcal{M}_R,\mathcal{H}_E) \ge 0$, and the realized environment satisfies $e^\star_{k+j} \in \mathfrak{E}_{k+j}$ for all $j$, and the robot-model error respects $\varepsilon_R$, then $|\tau_i(k+j)| \le \tau_{i,\max}$ for every joint $i$ and every $j \in [0,N]$. That is, $\mathcal{H}_E \models \mathcal{C}_R(\xi)$.*

*(ii) **Tightness.** If $\rho(\xi\mid\mathcal{M}_R,\mathcal{H}_E) < 0$, then there exist an environment realization admissible under $\mathcal{H}_E$ and a robot-model error admissible under $\varepsilon_R$ for which some actuator saturates. The test is therefore **exact** relative to $(\mathcal{H}_E, \varepsilon_R)$: it introduces no conservatism of its own.*

*(iii) **Attribution.** $\rho$ is monotone in the hypothesis: if $\mathfrak{E}'_{k+j} \subseteq \mathfrak{E}_{k+j}$ for all $j$, then $\rho' \ge \rho$. Consequently all conservatism in the certificate is carried by the world model's set width, and none by the certificate's construction.*

**Proof.** (i) By Definition 2, $\rho \ge 0$ gives, for every $i,j$, $\sup_{e \in \mathfrak{E}_{k+j}}|\tau_{\mathrm{req},i}(x_{k+j},u_{k+j},e)| \le \tau_{i,\max} - \varepsilon_{R,i}$. Since $e^\star_{k+j} \in \mathfrak{E}_{k+j}$, the realized required torque is bounded by that supremum, and adding the model error, itself bounded by $\varepsilon_{R,i}$, keeps the total within $\tau_{i,\max}$.

(ii) $\rho < 0$ means some $(i,j)$ has $\sup_{e \in \mathfrak{E}_{k+j}}|\tau_{\mathrm{req},i}| > \tau_{i,\max} - \varepsilon_{R,i}$. By (A0) the supremum is attained at some $e^\dagger \in \mathfrak{E}_{k+j}$, which is admissible under $\mathcal{H}_E$; choosing the model error at its bound in the sign of $\tau_{\mathrm{req},i}$ produces $|\tau_i| > \tau_{i,\max}$.

(iii) The infimum of a fixed function over a subset is no smaller than over the superset; $\rho$ is a minimum of such infima. $\blacksquare$

**On non-triviality.** Part (i) alone is definitional — it is the same triangle-inequality argument as P1's Theorem 1, and presenting it as a result would invite exactly the "almost tautological" objection P1's own Theorem 3 was rewritten to escape. The content is (ii) and (iii) together: they say the certificate is an *exact* transformation of the world model's uncertainty into actuator margin, so that the engineering question "how do I get margin back?" has one answer — calibrate the world model tighter — and not two. That is a falsifiable claim about the system, and §5 states how to test it.

---

### Theorem 2 (Detection precedes violation iff margin exceeds the detection threshold)

This is the theorem the paper is really about. It converts the qualitative claim "predicting is better than reacting" into a quantity with a design rule attached.

**Setup.** Suppose the true environment departs from the hypothesis beginning at time $t_0$, and let $\delta(t)$ denote the resulting excess in required torque at the binding joint, with $\delta(t_0) = 0$. Assume two-sided rate bounds over the window of interest:

$$
\ell\,(t - t_0) \;\le\; |\delta(t)| \;\le\; L\,(t - t_0), \qquad 0 < \ell \le L < \infty ,
$$

and write $\kappa = L/\ell \ge 1$ for the **rate-uncertainty ratio**. Let $\rho$ be the realizability margin banked at $t_0$ and $\beta$ the detection threshold of Definition 5. Let $t_d$ be the monitor's firing time and $t_v$ the first time the unmodified plan saturates an actuator.

**Theorem.** *Under the above,*

$$
t_0 + \frac{\beta}{L} \;\le\; t_d \;\le\; t_0 + \frac{\beta}{\ell},
\qquad
t_0 + \frac{\rho}{L} \;\le\; t_v \;\le\; t_0 + \frac{\rho}{\ell},
$$

*so the guaranteed warning time $T_w = t_v - t_d$ satisfies*

$$
\boxed{\; T_w \;\ge\; \frac{\rho}{L} - \frac{\beta}{\ell} \;}
$$

*and is guaranteed positive if and only if*

$$
\boxed{\;\rho \;>\; \kappa\,\beta\;.}
$$

*In the exactly-known-rate case $\ell = L$, the bound is tight and $T_w = (\rho - \beta)/L$.*

**Proof.** The monitor fires when $\|r\| > \beta$; under an exact robot model the residual is $\delta$, so firing occurs at the first $t$ with $|\delta(t)| > \beta$. From $|\delta| \le L(t-t_0)$, $|\delta|$ cannot exceed $\beta$ before $t_0 + \beta/L$; from $|\delta| \ge \ell(t-t_0)$, it must exceed $\beta$ by $t_0 + \beta/\ell$. This gives the bracket for $t_d$; the bracket for $t_v$ is identical with $\rho$ in place of $\beta$, since by Definition 2 the plan saturates exactly when the excess exceeds the banked margin. Subtracting the worst case of each bracket gives $T_w \ge \rho/L - \beta/\ell$, which is positive iff $\rho/L > \beta/\ell$, i.e. $\rho > (L/\ell)\beta$. With $\ell = L$ both brackets collapse to points and $T_w = (\rho-\beta)/L$ exactly. $\blacksquare$

**Corollary (design rule for $\rho_{\min}$).** *Let $T_{\mathrm{req}}$ be the reaction time the architecture needs between detection and effective response (planner latency plus actuator rise time). Then enforcing*

$$
\boxed{\;\rho_{\min} \;\ge\; \kappa\,\beta \;+\; L\,T_{\mathrm{req}}\;}
$$

*guarantees the monitor fires at least $T_{\mathrm{req}}$ before the plan would saturate.*

**Why this matters, stated plainly.** The safety threshold $\rho_{\min}$ has, in every architecture of this kind including P1, been a tuned constant. Here it is *derived*: it is the amount of actuator margin you must decline to spend in order to purchase a fixed quantity of warning time, given how tightly your world model is calibrated ($\beta$), how fast the environment can surprise you ($L$), how well you know that rate ($\kappa$), and how long your stack takes to react ($T_{\mathrm{req}}$). **Margin is not a safety buffer; margin is warning time.** That reframing is, in my judgement, the single most defensible original contribution available to this paper.

**Assumptions that are load-bearing, and where each fails.**

1. *Continuous onset.* $|\delta(t)| \le L(t-t_0)$ excludes impact. A rigid collision delivers its excess torque in one control period and no residual-based monitor can precede it. The theorem is therefore a statement about *compliant or progressive* environment surprises — a foot entering a hole, a wheel meeting a slope, a payload slipping, friction degrading — and the paper must say that impacts are out of scope rather than let the reader assume coverage. This is not a weak restriction in practice: the motivating "unexpected rock" is progressive precisely because compliance and finite contact stiffness make it so.
2. *The residual is dominated by $\delta$.* If tracking error or the low-level controller's own corrective action contributes to $r$ at the scale of $\beta$, the monitor's false-alarm rate rises and the bracket for $t_d$ loses its lower end. **P1 has already observed exactly this failure empirically**: its FR3-scale Experiment 4 found the certificate reporting a healthy margin throughout a run in which the real commanded effort was saturating on every joint, because a closed-loop controller-induced saturation is invisible to a certificate defined on the reference trajectory. That observation is the direct empirical motivation for stating this assumption rather than assuming it away, and separating $\delta$ from controller-induced residual is a concrete open problem this paper should name, not hide.
3. *$L$ is known or upper-boundable.* $L$ is a property of the environment class, not the robot, so it comes from the same world model whose hypothesis is under test — a circularity the paper must confront. The honest resolution is that $L$ is a *physical* bound (bounded contact stiffness, bounded relative approach velocity) rather than a learned one, and should be derived from platform and task limits, not from $\mathcal{W}$.

---

### Theorem 3 (Hypothesis-free recovery, and why fallbacks must not share assumptions with plans)

**Motivation, stated as the failure the theorem rules out.** After falsification, the world model has been shown wrong. Any recovery that is certified *under the world model's hypothesis* is therefore certified under an assumption known to be false at the moment it is needed. If the plan and the fallback share the hypothesis, one hypothesis failure destroys both, simultaneously, and the architecture has no recovery at all despite appearing to have one.

**Setup.** Let $\mathfrak{E}_{\mathrm{wc}} \subseteq \mathfrak{E}$ be a **hypothesis-free** worst-case environment set: a bound derived from physics and platform ratings (friction in $[0,\mu_{\max}]$, contact wrench bounded by what the structure can transmit, payload at most rated), *not* from $\mathcal{W}$, and satisfying $\mathfrak{E}_{k+j} \subseteq \mathfrak{E}_{\mathrm{wc}}$ for every hypothesis the world model may emit. Let $\pi_{\mathrm{stop}}$ be a safe-stop policy and define the **hypothesis-free terminal set**

$$
\begin{aligned}
\mathcal{X}_f^{\mathrm{wc}} \;=\; \Big\{\, x \;:\;\; & \pi_{\mathrm{stop}} \text{ from } x \text{ keeps } \tau \in \mathcal{T} \text{ over the stopping horizon,} \\[2pt]
& \text{and terminates at a statically holdable configuration,} \\[2pt]
& \text{both for \emph{every} } e \in \mathfrak{E}_{\mathrm{wc}} \,\Big\}.
\end{aligned}
$$

**Theorem.** *Suppose the planner enforces at every cycle both*

  *(a) $\rho(\xi \mid \mathcal{M}_R, \mathcal{H}_E) \ge \rho_{\min}$ with $\rho_{\min} \ge \kappa\beta + L\,T_{\mathrm{req}}$ (Theorem 2's corollary), and*

  *(b) $x_{k+j} \in \mathcal{X}_f^{\mathrm{wc}}$ for all $j \in [0,N]$ — the recursive-feasibility constraint evaluated under $\mathfrak{E}_{\mathrm{wc}}$, not under $\mathcal{H}_E$.*

*Then for any falsification onset at rate at most $L$: the monitor fires at $t_d \le t_v - T_{\mathrm{req}}$, the state at $t_d$ lies in $\mathcal{X}_f^{\mathrm{wc}}$, and $\pi_{\mathrm{stop}}$ executes to a statically-holdable stop within actuator limits — **regardless of whether $e^\star \in \mathcal{H}_E$**. The safety guarantee is therefore unconditional in the environment, while the task guarantee remains conditional on it.*

*Conversely, if (b) is replaced by $x_{k+j} \in \mathcal{X}_f^{\mathcal{H}_E}$ — the terminal set computed under the world model's own hypothesis — then no such guarantee holds, and there exist falsification events that invalidate the plan and the fallback simultaneously.*

**Proof.** *Forward.* By (a) and Theorem 2's corollary, $t_d \le t_v - T_{\mathrm{req}}$, so the monitor fires with at least $T_{\mathrm{req}}$ to spare before the plan would saturate; by definition of $T_{\mathrm{req}}$ the switch to $\pi_{\mathrm{stop}}$ is effective before $t_v$. By (b), the state at $t_d$ lies in $\mathcal{X}_f^{\mathrm{wc}}$, whose defining property quantifies over every $e \in \mathfrak{E}_{\mathrm{wc}}$. Since $\mathfrak{E}_{\mathrm{wc}}$ contains every physically admissible realization — including those outside $\mathcal{H}_E$, which is what falsification means — the stopping guarantee holds for $e^\star$ whether or not $e^\star \in \mathcal{H}_E$.

*Converse (by construction).* Take $\mathcal{H}_E$ asserting flat, high-friction terrain, and let $\xi$ be a plan whose margin under that hypothesis is comfortable. Compute the fallback set under the same hypothesis: $\mathcal{X}_f^{\mathcal{H}_E}$ then contains states whose stopping manoeuvre relies on ground reaction available only on flat, high-friction terrain. Let $e^\star$ place an unmodelled obstacle in the stance path. The excess torque invalidates $\xi$; the *same* excess invalidates the stopping manoeuvre, because the stopping manoeuvre was certified against the same false antecedent. The monitor fires correctly, the architecture switches correctly, and the fallback is nonetheless infeasible. $\blacksquare$

**The quotable form.** *You cannot use the world model to plan your recovery from the world model being wrong.* Safety fallbacks must be certified under a hypothesis-free bound, and must therefore be **assumption-disjoint** from the plans they back up. This is the formal content of the rock counterexample, and it is a design constraint on any architecture that puts a learned world model in the loop — not a property of this particular certificate.

**Cost, stated rather than buried.** $\mathcal{X}_f^{\mathrm{wc}}$ is strictly smaller than $\mathcal{X}_f^{\mathcal{H}_E}$, and constraint (b) is correspondingly expensive: the robot must decline motions it could safely perform if the world model were trusted. This is a real, quantifiable conservatism cost and the paper must measure it rather than assert it is acceptable. P1's Proposition 6 gives a concrete precedent for the shape of that measurement — its own terminal set shrinks from 100% to 7.3% of sampled states as payload grows from 0 to 12 kg — which is evidence that such sets are neither vacuous nor free. The natural relaxation is a graded family $\mathfrak{E}_{\mathrm{wc}}(\alpha)$ trading coverage against conservatism, with $\alpha$ an explicit risk parameter; whether that relaxation preserves the converse's force is open.

**A second requirement, found by running this and absent from the theorem as stated above.** Theorem 3 guarantees the fallback stays *feasible*; it does not guarantee that stopping *helps*. On the benchmark the two come apart badly — a stop-in-place fallback retains feasibility in under $20\%$ of the states where continuing has become infeasible, falling to $3.7\%$ as the falsification grows. §5 states the resulting Corollary (fallback authority); §5.1 then reports that the obvious remedy — a richer family of fallbacks indexed by falsification class — was tested and largely fails, so against quasi-static falsifications condition (b) is not merely one way to obtain recoverability but the only one. Read Theorem 3 with both attached.

---

## 5. Predictions, and what the first experiment found

`code/experiments/p2_theorem3_hypothesis_fallback.py` runs the three theorems' predictions on P1's contact-plane platform: the world model asserts a surface at $z=0.55$ with stiffness $700$ N/m, and reality puts it up to $20$ cm higher — the rock. Environment sets are boxes over $(z,K)$; because required torque is exactly affine in the contact-force scalar (verified to $0.000$ N·m over 200 random states), the per-joint supremum over a box is attained at an endpoint and is computed exactly rather than sampled.

**One bookkeeping caveat, which affects two numbers below.** The implementation subtracts P1's $\Delta\tau$ where the theory calls for $\varepsilon_R$. These are not the same bound: P1's $\Delta\tau$ covers "model mismatch *and* contact/disturbance prediction error", whereas $\varepsilon_R$ here is robot-model error alone, because environment uncertainty is carried by $\mathcal{H}_E$. Using $\Delta\tau$ as $\varepsilon_R$ therefore double-counts the environment's share, and the experiments are correspondingly *more* conservative than Definition 2 requires — sound, but not tight, so this implementation does **not** exhibit Theorem 1(ii)'s tightness. The conflation is kept rather than silently corrected, because P1 never decomposed $\Delta\tau$ and any split would be invented here; instead both experiments sweep the $\varepsilon_R$ scale so the cost is measured.

**T1(iii) — confirmed.** Tightening the hypothesis returns margin monotonically: narrowing the asserted plane's band from $\pm100$ mm to $\pm2$ mm moves the plan's realizability margin from $-7.49$ to $+6.90$ N·m, and the detection threshold $\beta$ from $57.42$ to $6.27$ N·m.

**T2 — confirmed as a lower bound, with a sharp practical limit.** Across that same sweep the predicted warning time $(\rho-\beta)/L$ was below the measured $t_v - t_d$ in every row where detection occurred, which is the theorem holding. But the sufficient condition $\rho > \beta$ was met only at the very tightest hypothesis ($\pm2$ mm), where the guaranteed warning is $0.003$ s against a measured $0.120$ s. Taken at face value this says a guaranteed warning requires a millimetre-calibrated world model. The $\varepsilon_R$ sweep shows how much of that is the over-count above rather than physics: the band at which $\rho > \beta$ first holds loosens from $\pm2$ mm at $\varepsilon_R = \Delta\tau$ to $\pm5$ mm at half and $\pm10$ mm as $\varepsilon_R \to 0$, i.e. **a factor of five** — so the honest claim is a *centimetre*-calibrated hypothesis under Definition 2's own bookkeeping, tightening to millimetres only if the robot model is as uncertain as P1's bundled bound assumes. Either way the structural point stands: the guarantee is not free, and the theory says exactly what calibration must be bought to obtain it.

**T3's converse — confirmed, strongly.** For a planner enforcing only the $\mathcal{H}_E$ certificate, the plan and the $\mathcal{H}_E$-certified fallback fail *together* at every falsification magnitude tested: at $\delta z = 0.05$–$0.15$ m the plan margin goes to $-2.10 \ldots -17.46$ N·m and the fallback is simultaneously infeasible from $16$–$33$ of the plan's $55$ states. One hypothesis failure, both layers gone. Sampled over states rather than along a plan, the same effect appears as a certified-but-unsafe rate rising $0\% \to 14\% \to 45\% \to 73\%$ as the rock grows.

**The cost is large and is not hidden.** The hypothesis-free terminal set is $0.13$–$0.26$ the size of the $\mathcal{H}_E$ one across payloads. Enforcing condition (b) forces the deepest admissible dip from $q_1 = 0.50$ to $q_1 = 0.30$, which on this geometry means the end-effector never goes below $0.649$ m — it does not approach the uncertain surface at all.

**T3's forward direction — NOT demonstrated, and the reason is a genuine gap in the theorem.** No tested falsification produced the storybook case: condition-(b) plan violated, fallback rescues it. Inside $\mathfrak{E}_{\mathrm{wc}}$'s coverage the (b)-planner's plan is never violated, so its fallback is never exercised; outside coverage both fail. Measuring why (part D) gives the real finding:

> Among states where continuing is infeasible under the truth, the brake is still feasible in only $19.6\%$ / $10.1\%$ / $3.7\%$ of cases at $\delta z = 0.05/0.10/0.15$ m.

A stop-in-place fallback has **almost no authority** against this falsification class, and less as the falsification grows. The reason is structural: braking removes only the velocity- and acceleration-dependent part of the torque demand, and a configuration-dependent contact force has none of it. This is the same fact P1's retiming Lemma turns on — a position-only force is retiming-proof — reappearing one level down, at Level 4 instead of Level 1.

**Consequently Theorem 3 needs the following addition, which the experiment produced and the drafted theory did not anticipate.**

**Corollary (fallback authority is a separate requirement from fallback feasibility).** *Theorem 3 guarantees that $\pi_{\mathrm{stop}}$ remains* feasible *when $\mathcal{H}_E$ is falsified. It does not guarantee that stopping* helps*. Define the* authority *of a fallback against a falsification class as the fraction of states at which the continuation is infeasible under the realized environment while the fallback remains feasible. A fallback whose authority against a class is near zero satisfies Theorem 3 vacuously: the architecture is certified to execute a manoeuvre that does not rescue it, and condition (b) degenerates into avoiding the uncertain region entirely.*

*A stop-in-place fallback relieves only the $\dot q$- and $\ddot q$-dependent components of demand. Its authority against a quasi-static falsification — one whose added demand is a function of configuration alone — is therefore structurally small, independent of how the terminal set is computed.*

The design consequence appeared to be that a world-model-driven architecture needs a fallback *family* indexed by falsification class, not one brake. **That follow-up was run, and it mostly does not work** (`code/experiments/p2_fallback_family.py`). The result is negative and is more useful than the corollary that prompted it.

### 5.1 The fallback family: tested, and largely refuted

A second falsification class was added — payload, which is *inertial* where the contact surface is *quasi-static*, and likewise exactly affine in its scalar parameter — together with a second fallback, **retreat** (brake to rest, withdraw along $+J_z(q)^\top$, come to rest again at the withdrawn configuration). Three things came out, in increasing order of consequence.

**The diagonal exists in one direction only.** Against an inertial falsification, braking is clearly right and retreat is clearly wrong — authority $43$–$57\%$ versus $0$–$2\%$. Against a quasi-static one, *neither* works: authority tops out near $30\%$ and is usually under $20\%$, and retreat's advantage over braking is within a few points everywhere. So class-indexing has real content ("do not retreat against an inertial deficit") but does not supply what the corollary needed, which was a fallback that answers the quasi-static case.

**The family buys no coverage at all.** Under the hypothesis-free bound the two terminal sets are not merely the same size but *the same states* — 33 members each, 33 in the intersection, nothing in either difference. Both manoeuvres begin by braking, and worst-case feasibility is decided at the entry state they share. A fallback family widens coverage only if its members differ in what they do **first**.

**And indexing the family is hard, though less hopeless than it first appeared.** Identifying the falsification class from the residual, by fitting the two classes' regressor directions and taking the better explanation, succeeds $72.3\%$ of the time at $\varepsilon_R = \Delta\tau$, with a median separation ratio of $1.3\times$ and a 10th percentile of $0.8\times$. But that figure inherits the over-count described above, and the sweep shows it is largely an artifact of it: accuracy rises to $79.0\%$, $87.0\%$ and $92.3\%$ as $\varepsilon_R$ falls to $0.5$, $0.25$ and $0.1 \times \Delta\tau$, with separation rising $1.3 \to 8.7\times$. So class identification is workable given a well-identified robot model, and the honest statement is that it degrades sharply with robot-model uncertainty rather than that it fails. (Without that noise the test reports $100\%$ and a separation of $10^{12}$, which is the tautology of scoring a residual built from one of the two candidate models; the noisy number is the real one.)

**What this changes.** Authority against a quasi-static falsification is not a matter of choosing a better manoeuvre. Both manoeuvres must brake first, so a robot that detects late has already coasted deeper into a position-dependent deficit, and no reactive fallback recovers it — the authority numbers fall monotonically with entry speed for exactly this reason. The remedy is therefore not a richer fallback set but the planning-time constraint Theorem 3 already imposes:

> Against quasi-static falsifications, hypothesis-free recoverability is a **planning-time** property and cannot be recovered at runtime. The only effective response is not to enter the region — which is precisely what condition (b) forced in the two-planner comparison reported above (part C3 of `p2_theorem3_hypothesis_fallback.py`), where the compliant planner kept the end-effector above $0.649$ m and never approached the surface at all.

This is a sharper and more defensible claim than the fallback-family suggestion it replaces, and it is the one the paper should make. It also re-links Theorem 3 to Theorem 2: where a fallback *does* have authority, the authority is bought with detection lead time, so $\rho_{\min}$ is doing double duty — it purchases warning (Theorem 2) and it purchases the ability to act on the warning (here). Both are the same currency, spent at different levels of the hierarchy.

---

## 6. Delta against P1, stated so a reviewer can check it

| | **P1** | **This paper** |
|---|---|---|
| Environment enters as | a point prediction plus a fixed scalar bound $\Delta\tau$ | a per-cycle, set-valued, model-emitted hypothesis $\mathcal{H}_E$ |
| Certificate form | Boolean margin $m_{\mathrm{phys}} \ge m_{\mathrm{safe}}$ | entailment $\mathcal{H}_E \models \mathcal{C}_R(\xi)$ with a monitorable antecedent |
| Triggers | two, both about the plan ($\sigma_{\mathrm{geo}}$, $\sigma_4$) | adds one about the world model (falsification), independent of both |
| $\rho_{\min}$ / $m_{\mathrm{safe}}$ | a tuned constant | derived from $\beta, \kappa, L, T_{\mathrm{req}}$ (T2 corollary) |
| Terminal safe set | computed under the nominal environment | computed under a hypothesis-free bound, and shown to *require* it (T3) |
| Platform | fixed-base manipulator | same platform so far (§5); an environment-rich mobile/legged study is future work, not a claim of this draft |

The overlap is real and must be disclosed, not minimized: P1's Theorem 1 is the ancestor of T1(i), and P1's Theorems 4–5 are the ancestor of T3's terminal-set machinery. What is new is the set-valued conditional antecedent, the falsification monitor as an independent trigger, the derivation of the safety threshold from warning time, and the assumption-disjointness requirement on fallbacks. If a reviewer judges that list insufficient, the paper is salami-slicing and should be merged into P1 instead.

**One sequencing conflict to resolve before committing.** P1's status tracker — moved out of the paper into `README.md`, where the item numbering is preserved — records a different recommended next step at item 6: "a separate, narrower saturation-certificate paper proving Theorems 1–3 independent of this paper's broader scope." That paper and this one are not the same paper, and they compete for the same slot — the narrower one is safer and duller; this one is riskier and, if the literature gap survives §7, considerably more valuable. Item 6 also states P1 should not go to substantive external submission until `mp_main` (ICRA 2027) and HAE (IJSS) have decisions, which constrains when either can go out. That decision should be made explicitly rather than by drift.

---

## 7. What would kill this paper

Three objections, in descending order of danger. Each is stated as a condition that must be checked against the literature — which I could not do in this session, as `arxiv.org` is blocked by the network policy here.

1. **"This is assume–guarantee contracts applied to robotics."** The most dangerous objection, because it is partly true. Contract-based design (Benveniste et al. and successors) has the entailment form, the assume/guarantee split, and compositional reasoning. **Survival condition:** the novelty must be located precisely at the antecedent being *machine-generated per cycle and falsifiable at runtime*, and at the two results that only make sense in that setting — the warning-time identity (T2) and assumption-disjointness of fallbacks (T3). Neither is a statement contract theory makes, because contract theory's assumptions are static. If a paper is found that treats a *learned, time-varying, runtime-falsifiable* antecedent in this way, the contribution collapses to T2's design rule alone.

2. **"Theorem 2 is standard fault-detection detectability analysis."** Residual-based FDI has a well-developed theory of detection delay versus threshold. **Survival condition:** the novelty is not the detection-delay bound but its *coupling to the actuator margin* — that the same physical quantity $\rho$ appears in both the feasibility test and the detection-lead bound, making $\rho_{\min}$ derivable. FDI sets thresholds to trade false alarms against delay; it does not, as far as I know, tie the threshold to a planning-time margin the planner is free to spend. That specific coupling is the claim, and it needs a targeted search of the FDI and reachability-monitoring literature.

3. **"The hypothesis sets are conformal prediction, and safe planning with conformal predictors exists."** Constructing $\mathfrak{E}_{k+j}$ from a generative model with a coverage guarantee is, in the natural implementation, conformal prediction; there is recent work on conformal-prediction-based safe planning with learned predictors. **Survival condition:** this paper should *adopt* that machinery for Definition 1 rather than compete with it, and claim novelty downstream — in what happens when coverage fails, which is exactly what conformal guarantees do not cover (they bound the *rate* of miscoverage, not the *consequences* of a particular miscoverage event). T3 is a statement about the consequences of a miscoverage event, which is a complementary question. If that framing holds, this becomes a strength rather than a threat: the paper inherits a calibrated $\mathcal{H}_E$ and supplies the recovery theory.

**My overall judgement, updated after the first experiment.** T2 and T3 are, as far as I can tell without a literature search, genuine and non-obvious, and they are coupled: T2 says margin buys warning time and derives the threshold; T3 says the warning is only useful if the recovery does not share the failed assumption. Together they are a thesis, not a position. T1 is scaffolding and should be presented as such rather than inflated.

The experiment changed the picture in two specific ways, both worth more than the confirmations. First, T2's sufficient condition $\rho > \kappa\beta$ is *hard to meet*: on this platform it needs a millimetre-calibrated hypothesis, so the paper's honest claim is that the theory tells you what calibration you must buy, not that the guarantee comes free. Second, and more consequentially, T3 as drafted guarantees the wrong thing — feasibility of the fallback rather than its usefulness — and the fallback-authority corollary (§5) is now a required part of the result rather than an extension of it. The follow-up that corollary suggested was then run and mostly refuted (§5.1), which is what turned a soft recommendation ('use a fallback family') into a hard claim ('against quasi-static falsifications there is no runtime remedy, only a planning-time one'). That the decisive experiment forced one change to the theorem and its own follow-up forced a second is the strongest available evidence that the four parts were worth fixing before writing prose.

The literature check remains the gating task and should still be done before any prose is written, because objection 1 could still redirect the whole framing.
