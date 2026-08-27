# Retiming Monotonicity: Decomposition, a Checkable Sufficient Condition, and a Fallback

**Status:** The Lemma (§3) and its empirical grounding (§5) have now been folded directly into `predictive_realizability_paper_draft.md`'s §VI, immediately after Theorem 3's proof, using the sign-condition/dense-scan-fallback version (§4(a)) — not the closed-form exact fallback (§4(b)) below, which remains unimplemented. This file remains the source material and is still where the closed-form fallback is being developed for direct reuse in the narrower saturation-certificate paper (targeting SCL/L-CSS) that the main draft's Draft Status item 6 identifies as the next concrete step. Notation matches the main draft's Theorem 3 (§VI). Raised in review, verified against the actual reduced-order implementation (`code/local_planner.py`) before being trusted — see the empirical section at the end.

---

## 1. Setup

Let $p$ be a route and $\xi_0$ the nominal trajectory the local planner produces along it, parameterized by $s \in [0, 1]$ (path progress). For a retiming factor $\lambda \in \Lambda = [1, \lambda_{\max}]$, let $\xi_\lambda$ denote $\xi_0$ time-reparameterized by $\lambda$: if $\xi_0$ traces configuration $q_0(t)$, then $\xi_\lambda$ traces $q_\lambda(t) \equiv q_0(t/\lambda)$, so that

$$
\dot q_\lambda(t) = \frac{1}{\lambda}\dot q_0(t/\lambda), \qquad \ddot q_\lambda(t) = \frac{1}{\lambda^2}\ddot q_0(t/\lambda).
$$

At a fixed path fraction $s = t/\lambda \cdot (1/T_0) \cdot T_0$ (i.e., comparing $\xi_\lambda$ at time $t$ to $\xi_0$ at time $t/\lambda$, the same point along the path), the configuration $q$ is identical regardless of $\lambda$ — only its time derivatives scale.

The joint torque required to realize $\xi_\lambda$ at a given path fraction is, by the manipulator equation with a possibly position-dependent external force $F_{\mathrm{ext}}(q)$ (contact/interaction force known as a function of configuration, not of time — the case Assumption/Theorem 3's own environment-conditioning discussion, main draft §VI, already treats separately from time-driven disturbance),

$$
\tau_i(\lambda) = \underbrace{\frac{1}{\lambda^2}\Big[M_i(q)\,\ddot q_0(s) + C_i(q, \dot q_0(s))\,\dot q_0(s)/\lambda \cdot \lambda \Big]}_{\text{inertial + Coriolis, scale as } 1/\lambda^2} \;+\; \underbrace{g_i(q) + J_i(q)^\top F_{\mathrm{ext}}(q)}_{\text{independent of } \lambda}.
$$

Since the Coriolis term is itself quadratic in velocity ($C(q,\dot q)\dot q$ is bilinear in $\dot q$, so it also scales as $1/\lambda^2$ under $\dot q \to \dot q/\lambda$), both the inertia and Coriolis contributions share the same $1/\lambda^2$ scaling. Writing

$$
A_i \equiv M_i(q)\,\ddot q_0(s) + C_i(q, \dot q_0(s))\,\dot q_0(s) \quad \text{(evaluated once, at } \lambda=1\text{'s velocity/acceleration at this path fraction)},
$$
$$
B_i \equiv g_i(q) + J_i(q)^\top F_{\mathrm{ext}}(q),
$$

the per-joint torque at retiming factor $\lambda$ is exactly

$$
\boxed{\tau_i(\lambda) = \frac{A_i}{\lambda^2} + B_i.} \tag{1}
$$

$A_i$ does not depend on $\lambda$ (it is evaluated at the same path fraction regardless of how fast the path is traversed); only the $1/\lambda^2$ scaling multiplying it does. $B_i$ is entirely independent of $\lambda$.

## 2. The margin as a function of $\lambda$

The per-joint, per-step certificate margin (main draft §IV) is $m_{\tau,i}(\lambda) = \tau_{i,\max} - |\tau_i(\lambda)| - \Delta\tau_i$, and the aggregate is $m_{\mathrm{phys}}(\lambda) = \min_i m_{\tau,i}(\lambda)$ (minimized also over horizon/route steps, suppressed here for a fixed step). From (1),

$$
m_{\tau,i}(\lambda) = \tau_{i,\max} - \Delta\tau_i - \left| \frac{A_i}{\lambda^2} + B_i \right|.
$$

**Case $\mathrm{sign}(A_i) = \mathrm{sign}(B_i)$ (or $A_i = 0$).** Then $|A_i/\lambda^2 + B_i| = |A_i|/\lambda^2 + |B_i|$, which is strictly decreasing in $\lambda$ (for $A_i \neq 0$) or constant (for $A_i = 0$). Hence $m_{\tau,i}(\lambda)$ is monotonically non-decreasing on $\Lambda$.

**Case $\mathrm{sign}(A_i) \neq \mathrm{sign}(B_i)$ and $A_i \neq 0$.** Write $A_i = -\alpha$, $B_i = \beta$ with $\alpha, \beta$ same sign (WLOG both positive). Then $\tau_i(\lambda) = \beta - \alpha/\lambda^2$, which is zero at

$$
\lambda^\star = \sqrt{\alpha/\beta},
$$

and $|\tau_i(\lambda)|$ **decreases** from $|\tau_i(1)| = |\beta - \alpha|$ toward $0$ as $\lambda \to \lambda^\star$ (if $\lambda^\star > 1$), then **increases** from $0$ back toward $|\beta|$ as $\lambda \to \infty$. Correspondingly $m_{\tau,i}(\lambda)$ **increases** to a maximum at $\lambda^\star$ (if $\lambda^\star \in \Lambda$) and then **decreases**. This is exactly the non-monotonic shape: the supremum of $m_{\tau,i}(\lambda)$ over $\Lambda$ can be at an *interior* point, not at $\lambda_{\max}$.

## 3. Lemma (sufficient condition for monotonicity)

**Lemma (per-joint sign condition).** *If, for every joint $i$ and every horizon/route step evaluated, either $A_i = 0$ or $\mathrm{sign}(A_i) = \mathrm{sign}(B_i)$, then $\lambda \mapsto m_{\mathrm{phys}}(\lambda)$ is monotonically non-decreasing on $\Lambda = [1, \lambda_{\max}]$, and Theorem 3's bisection search (checking only $\lambda_{\max}$, then bisecting) correctly computes $m^\star_1(p) = m_{\mathrm{phys}}(\lambda_{\max})$.*

*Proof.* Immediate from §2, Case 1, applied to every $(i, \text{step})$ pair; the minimum of a finite collection of non-decreasing functions is non-decreasing, so $m_{\mathrm{phys}}(\lambda) = \min_{i,\text{step}} m_{\tau,i}(\lambda)$ is itself non-decreasing on $\Lambda$. $\blacksquare$

**Physical reading of the condition.** $A_i$ and $B_i$ share a sign exactly when the inertial/Coriolis torque and the gravity/external-force torque are *pulling the same direction* at that joint, at that instant — e.g., accelerating upward against gravity (both terms resist the same motion), or a purely static configuration ($A_i = 0$ trivially). The condition **fails** when inertial/Coriolis torque *partially unloads* the gravity/external-force torque — the physically common case of decelerating while gravity is already doing part of the required work, e.g. the second half of a downward swing, or a joint whose Coriolis coupling from another joint's motion happens to counteract gravity at that configuration. This is not an adversarial or contrived regime; §5 below finds it in essentially every scenario with genuine multi-joint coupling under a non-trivial payload.

**Checkability.** $A_i$ and $B_i$ are both already computed as intermediate quantities when evaluating $\tau_i(1)$ via inverse dynamics (main draft, `dynamics.py`'s `required_torque`): $B_i$ is the torque at $\dot q = \ddot q = 0$ (a call with zero velocity/acceleration, at the same $q$), and $A_i = \tau_i(1) - B_i$. So the sufficient condition can be checked with one extra inverse-dynamics evaluation per already-evaluated step, at negligible marginal cost, and used as a certificate that the fast (single-point, bisection) search is valid *before* trusting its answer — rather than trusting it unconditionally, as the current implementation does.

## 4. Fallback when the sufficient condition fails

When the Lemma's condition is violated at some (joint, step), monotonicity is not certified and $\lambda_{\max}$ alone cannot be trusted as the arg-sup. Two options, in increasing order of rigor:

**(a) Dense grid + local refinement (implemented, `code/local_planner.py::_search_retime_whole_route`).** Sample $m_{\mathrm{phys}}(\lambda)$ on a grid $\{1 = \lambda_0 < \lambda_1 < \cdots < \lambda_K = \lambda_{\max}\}$; if any grid point clears $m_{\mathrm{safe}}$, bisect locally around the first (smallest-$\lambda$) crossing bracket to refine. This is sound whenever $K$ is large enough that $m_{\mathrm{phys}}$ does not have more oscillations than the grid can resolve — a resolution assumption, not a proof, but a checkable and tunable one (unlike silently trusting $\lambda_{\max}$). Cost: $O(K)$ certificate evaluations instead of $O(1)$, paid once per route decision (a planning-time, not per-cycle, cost).

**(b) Exact interior extremum (closed form per joint, not yet implemented).** From §2, the interior extremum of $|\tau_i(\lambda)|$ (a zero-crossing of $\tau_i$) occurs at $\lambda^\star_i = \sqrt{A_i/(-B_i)}$ when $\mathrm{sign}(A_i) \neq \mathrm{sign}(B_i)$ and $A_i/(-B_i) > 0$; each joint's own candidate extremum is known in closed form, so the true supremum of $m_{\tau,i}(\lambda)$ over $\Lambda$ can be evaluated at the finite candidate set $\{1, \lambda_{\max}\} \cup \{\lambda^\star_i : i \in \text{joints}, \lambda^\star_i \in \Lambda\}$ exactly, without a grid. This is the natural closed-form successor to (a) for the spin-off paper: it upgrades the fallback from "resolution-limited search" to "exact for this decomposition," at the cost of stating and using (1) explicitly rather than treating $m_{\mathrm{phys}}(\lambda)$ as a black box. **This is the recommended path for the SCL/L-CSS submission** — it turns Theorem 3's bisection into a provably correct two-phase procedure (check the Lemma's sign condition; if it holds, bisect as before; if not, evaluate the closed-form candidate set exactly) rather than a heuristic dense scan.

## 5. Empirical verification (this platform, not yet in the spin-off paper's own experiments)

Checked directly against `code/local_planner.py` and `code/dynamics.py` (the reduced-order 3-DOF planar arm), not assumed from the algebra alone:

- A random search over 3000 draws of $(q_0, q_f, T, \text{payload})$ found the Lemma's sign condition violated (non-monotonic $m_{\mathrm{phys}}(\lambda)$) in **21/300** of an initial coarse pass and, in a targeted follow-up, **23/3000 (~0.8%)** of cases produced the failure mode that matters operationally: checking $\lambda_{\max}$ alone reports $m_{\mathrm{phys}} < m_{\mathrm{safe}}$ (retiming "exhausted") while the true supremum, at an interior $\lambda$, clears $m_{\mathrm{safe}}$. Two representative cases:

  | trial | $T$ (s) | payload (kg) | $m_{\mathrm{phys}}(1)$ | $m_{\mathrm{phys}}(\lambda_{\max}=4)$ | true $\sup_\lambda m_{\mathrm{phys}}(\lambda)$ | at $\lambda$ |
  |---|---|---|---|---|---|---|
  | 26  | 0.67 | 3.86 | $-1.51$  | $1.66$ (below $m_{\mathrm{safe}}=2.0$) | $2.86$ | $1.55$ |
  | 146 | 0.70 | 3.83 | $-82.58$ | $1.80$ (below $m_{\mathrm{safe}}=2.0$) | $2.06$ | $3.50$ |

- The two scenarios whose Theorem-3 conclusion is actually cited in the main draft's results (§IX) were checked separately and are **not** affected: the flagship scenario's $m_{\mathrm{phys}}(\lambda)$ is genuinely monotonic (Lemma's condition holds throughout, verified by dense scan), and the environment-conditioned scenario (Experiment 7) is technically non-monotonic but the interior deviation ($\sim 0.5$) is negligible against how far below $m_{\mathrm{safe}}$ it stays throughout ($\sim -20$) — the qualitative conclusion ("Level 3 is genuinely necessary") is unchanged in both.
- The existing regression test for Theorem 3's monotonicity assumption (`tests/test_planner.py::test_static_gravity_deficit_defeats_retiming`) uses a degenerate hold trajectory ($q_0 = q_f$, so $\dot q \equiv \ddot q \equiv 0$ throughout, i.e. $A_i \equiv 0$ for every joint). This is exactly the case where the Lemma's condition holds *trivially* ($A_i = 0$) and $\tau(\lambda) = B$ is constant — the one regime in which non-monotonicity is structurally impossible. The existing test therefore cannot, by construction, expose this failure mode; a new test targeting a genuinely dynamic (non-degenerate $A_i$) scenario with an opposing-sign joint is needed for real coverage and is a natural next addition alongside adopting option (b) above.

## 6. What remains to make this submission-ready

- Extend §2–3 from a single joint/step to the aggregate $m_{\mathrm{phys}} = \min_{i,\text{step}}$ formally (routine — minimum of non-decreasing functions is non-decreasing — but should be stated, not left implicit as it is in §3 above).
- Decide between fallback (a) [implemented, resolution-limited] and (b) [closed-form, exact] as the paper's actual claim; (b) is strictly stronger and is recommended.
- If (b) is adopted, implement and test it in `code/local_planner.py` alongside the current dense-grid fallback (a), and re-verify the two reported scenarios (flagship, Experiment 7) and the 3000-case empirical sweep above against the closed-form candidate-set method, not just the grid.
- Extend the $A_i$/$B_i$ decomposition's derivation from a single time-invariant external force $F_{\mathrm{ext}}(q)$ to the more general position-and-time case, or state explicitly (as the main draft's Theorem 3 discussion already does for the wall-clock-driven $E(t)$ case) that this lemma is scoped to $E(q)$-type environments.
