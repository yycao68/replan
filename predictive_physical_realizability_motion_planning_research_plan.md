# Research Plan: Predictive Physical-Realizability-Aware Motion Planning

## Working title

**Predictive Physical Realizability Feedback for Safe Motion Planning and Replanning**

Alternative titles:

- **From Geometric Feasibility to Physical Realizability: Predictive Feedback for Motion Planning**
- **Predictive Actuation-Feasibility Feedback for Reactive Robot Motion Planning**
- **Physical-Realizability-Aware Hybrid Motion Planning with Predictive Saturation Certificates**

---

# 1. Motivation

## 1.1 The central problem

Most robot motion-planning pipelines establish that a trajectory is geometrically and kinematically feasible, but execution feasibility is not generally represented as a first-class planning signal.

A typical pipeline is approximately

\[
\text{global/path planning}
\rightarrow
\text{time parameterization}
\rightarrow
\text{trajectory execution}.
\]

MoveIt 2 explicitly describes itself as primarily a kinematic motion-planning framework. Its standard trajectory-processing pipeline adds velocity and acceleration through time parameterization, with TOTG currently documented as the default time-parameterization method; Ruckig can subsequently be used for jerk-limited smoothing. [MoveIt Time Parameterization](https://moveit.picknik.ai/main/doc/examples/time_parameterization/time_parameterization_tutorial.html)

The important missing quantity is not merely a declared joint limit, but the **future physical realizability of the planned motion**.

For a robot with

\[
\tau =
M(q)u+C(q,\dot q)\dot q+g(q)
+J_c^TF_c+\tau_{\mathrm{ext}},
\]

the acceleration/command authority available at a future state depends on:

- configuration \(q\),
- velocity \(\dot q\),
- contact configuration,
- payload,
- interaction forces,
- external disturbances,
- terrain/environment,
- uncertainty,
- actuator limits.

Therefore,

\[
\text{kinematically feasible}
\not\Rightarrow
\text{physically realizable}.
\]

A trajectory may be collision-free and satisfy nominal velocity/acceleration limits while still driving one or more actuators toward saturation in the future.

The key research question is:

> **Can future loss of physical realizability be predicted during execution and fed back to motion planning early enough that the robot can adapt, reroute, or safely brake before the loss of control authority causes task failure, collision, or falling?**

---

# 2. Core hypothesis

The central hypothesis is:

\[
\boxed{
\text{Predictive physical-realizability information can serve as a planning feedback signal.}
}
\]

Instead of treating actuator saturation as a low-level execution problem,

\[
u_{\mathrm{nom}}
\rightarrow
\tau
\rightarrow
\mathrm{clip},
\]

the proposed architecture evaluates the planned future motion before the failure occurs:

\[
u^{\mathrm{nom}}_{k:k+N}
\rightarrow
\text{predictive realization model}
\rightarrow
\text{physical margin}
\rightarrow
\text{planner feedback}.
\]

The planner can then:

1. execute normally if sufficient authority exists;
2. retime or slow the trajectory if the route remains physically realizable;
3. reshape the trajectory if acceleration/contact demands can be reduced;
4. reroute/replan if the current route is intrinsically physically infeasible;
5. brake/fallback to a safe invariant region if no safe continuation remains.

This creates a **bidirectional planning-execution loop** rather than a one-way planner-to-controller pipeline.

---

# 3. Relationship to the existing three research components

The proposed work should not be presented as simply merging three independent papers. The three components become layers of one new problem.

## 3.1 DI-QP motion-planning backbone

The existing fixed-structure DI-QP planner provides a fast time-domain local trajectory generator.

Its role is:

\[
\boxed{\text{Generate a candidate motion}}
\]

The important implementation property is that prediction matrices and QP structure can remain fixed while state-dependent quantities enter through vector terms.

---

## 3.2 HAE: geometric/topological failure recovery

The Hybrid Asynchronous Escape mechanism detects geometric/planning failures such as:

- QP infeasibility,
- stagnation,
- prediction/Jacobian drift,

and invokes hierarchical escape/replanning.

Its role is:

\[
\boxed{\text{CHANGE WHERE}}
\]

That is, if the current route is geometrically or topologically problematic, the system should seek another route.

---

## 3.3 Predictive saturation / physical-realizability layer

The predictive saturation manager estimates future actuator feasibility and directional authority.

Its role is:

\[
\boxed{\text{CHANGE HOW, or CHANGE WHERE if necessary}}
\]

The critical extension is that the manager is not restricted to current saturation. It predicts future loss of realizability under:

- contact,
- payload,
- interaction forces,
- external disturbances,
- terrain changes,
- uneven surfaces,
- steps/holes,
- changing contact conditions.

The new paper therefore generalizes the existing saturation concept into:

\[
\boxed{\text{Predictive Physical Realizability}}
\]

with saturation as a primary measurable failure mechanism.

---

# 4. Related work and positioning

## 4.1 Conventional motion planning and time parameterization

MoveIt 2 currently describes its motion planning as primarily kinematic and uses post-processing to add velocity and acceleration information. TOTG is documented as the default time-parameterization method, while Ruckig provides additional jerk-limited smoothing. [MoveIt Time Parameterization](https://moveit.picknik.ai/main/doc/examples/time_parameterization/time_parameterization_tutorial.html)

This establishes a strong real-world baseline for the proposed study:

\[
\text{path}
\rightarrow
\text{time parameterization}
\rightarrow
\text{execution}.
\]

The proposed work does **not** claim that such pipelines are useless. Rather, it asks whether their feasibility representation is incomplete because actuator authority is state- and environment-dependent.

---

## 4.2 MoveIt 2 Hybrid Planning

MoveIt 2 already provides a Hybrid Planning architecture with recurrent global and local planners. The global planner generates a global solution while the local planner continuously adapts it using current robot state, world information, and local constraints. The architecture explicitly supports reactive replanning and adaptive motion. [MoveIt Hybrid Planning](https://moveit.picknik.ai/main/doc/concepts/hybrid_planning/hybrid_planning.html)

This is an important enabling architecture for the proposed work.

The proposed contribution is not "inventing hybrid planning." Instead:

> **The proposed work introduces predictive physical realizability as a new feedback signal and local planning constraint within an existing global/local planning architecture.**

This is particularly relevant because MoveIt describes adaptive motion examples involving local adaptation to changing conditions such as uneven surfaces. [MoveIt Hybrid Planning](https://moveit.picknik.ai/main/doc/concepts/hybrid_planning/hybrid_planning.html)

---

## 4.3 Actuation-aware / dynamics-aware planning

Prior work has incorporated torque limits, dynamic feasibility, feasible wrench sets, and actuator constraints into motion planning.

The proposed work should therefore avoid the weak claim:

> "No one has ever considered actuator limits in motion planning."

Instead, the novelty should be framed around the **interface and prediction mechanism**:

1. physical realizability is evaluated over the future execution horizon;
2. the information is generated by the execution/realization layer;
3. the signal is state-, environment-, contact-, payload-, and disturbance-dependent;
4. it is fed back to a running planner;
5. the response is hierarchical: adaptation, rerouting, and safe fallback.

The research question is therefore not simply torque-constrained planning, but:

\[
\boxed{
\text{predictive execution feasibility}
\rightarrow
\text{planner-level feedback}
}
\]

---

# 5. Proposed architecture

## 5.1 Overall architecture

```text
                  Environment / World
              ┌──────────┼──────────┐
              │          │          │
           terrain    obstacles   contacts
              │          │          │
              └──────────┼──────────┘
                         ▼
                  Global Planner
                         │
                    global route
                         ▼
                  DI-QP Local Planner
                         │
                 nominal trajectory
                         ▼
          Predictive Physical Realizability
                         │
          ┌──────────────┼──────────────┐
          │              │              │
     torque margin   contact margin  directional
                                      authority
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                  Decision / Supervisor
                 ┌───────┼────────┐
                 │       │        │
              execute  adapt    reroute
                 │       │        │
                 │   retime /    │
                 │   reshape     │
                 │       │        │
                 └───────┼────────┘
                         ▼
                    Controller
                         │
                         ▼
                       Robot
                         │
                         └────────── feedback
```

---

# 6. Failure taxonomy

The HAE signals can be retained as the geometric/planning failure class:

\[
\sigma_{\mathrm{geo}}
=
\sigma_1\vee\sigma_2\vee\sigma_3
\]

where the existing signals represent planning/QP infeasibility, stagnation, and prediction/linearization degradation.

Add:

\[
\boxed{\sigma_4 =
\text{predicted physical-realizability loss}}
\]

However, \(\sigma_4\) should **not** simply be treated as another HAE escape trigger.

The response mechanisms are fundamentally different.

### Geometric failure

\[
\sigma_{\mathrm{geo}}
\Rightarrow
\boxed{\text{CHANGE WHERE}}
\]

### Dynamic/physical failure

\[
\sigma_4
\Rightarrow
\boxed{\text{CHANGE HOW first}}
\]

and only when adaptation cannot restore feasibility:

\[
\sigma_4
\Rightarrow
\boxed{\text{CHANGE WHERE}}
\]

This distinction is central to the proposed theory.

---

# 7. Predictive physical-realizability certificate

For a nominal predicted trajectory

\[
\mathbf{x}_{0:N},
\qquad
\mathbf{u}_{0:N},
\]

define a future actuator-feasibility margin.

A generic formulation is:

\[
m_{\tau,i}(k+j)
=
\tau_{i,\max}
-
|\tau_i(k+j)|
\]

with uncertainty tightening:

\[
m_{\tau,i}^{\mathrm{robust}}
=
\tau_{i,\max}
-
|\hat\tau_i|
-
\Delta\tau_i.
\]

The aggregate horizon certificate can be defined as

\[
m_{\mathrm{phys}}
=
\min_{j=0,\ldots,N}
\min_i
m_{\tau,i}^{\mathrm{robust}}(k+j).
\]

A directional authority metric can complement the scalar torque margin:

\[
\alpha_{\mathrm{dir}}(k+j)
=
\text{remaining control/acceleration authority in the demanded direction}.
\]

Potential additional signals include:

\[
T_{\mathrm{loss}}
=
\min\{j\Delta t:m_{\mathrm{phys}}(k+j)\le 0\},
\]

and

\[
J_{\mathrm{phys}}
=
\sum_{j=0}^{N}
\phi(m_{\mathrm{phys}}(k+j)).
\]

These provide richer feedback than a binary saturated/not-saturated signal.

---

# 8. Environment-conditioned realizability

The proposed model should explicitly allow environmental information to influence future actuator feasibility.

Let

\[
E_{k:k+N}
\]

represent predicted environmental/contact information, including:

- terrain height,
- slope,
- holes/steps,
- friction,
- expected contact locations,
- obstacles,
- moving objects,
- payload changes,
- human interaction,
- external disturbances.

Then:

\[
(E_{k:k+N},x_k,u^{nom}_{k:k+N})
\rightarrow
\tau_{k:k+N}
\rightarrow
m_{\mathrm{phys}}.
\]

This produces an important conceptual distinction:

\[
\boxed{
\text{collision-free}
\not\Rightarrow
\text{physically realizable}
}
\]

For example, a humanoid may have a collision-free footstep trajectory that becomes infeasible because the predicted terrain/contact condition requires excessive ankle or hip torque.

The planner should therefore evaluate:

\[
\boxed{
\exists\;\text{physically realizable trajectory along this route?}
}
\]

not merely:

\[
\boxed{
\exists\;\text{collision-free path?}
}
\]

---

# 9. Hierarchical response strategy

The planner should not immediately reroute whenever a margin decreases.

## Level 0 — Normal execution

If

\[
m_{\mathrm{phys}}\ge m_{\mathrm{safe}},
\]

execute the nominal trajectory.

---

## Level 1 — Retiming

If the route remains physically realizable but the nominal timing causes excessive demand:

\[
q(s)
\rightarrow
q(t_{\mathrm{new}})
\]

with reduced velocity/acceleration.

---

## Level 2 — Trajectory reshaping

Modify:

- acceleration profile,
- CoM motion,
- contact timing,
- foot trajectory,
- local target,
- velocity profile.

The objective is to restore:

\[
m_{\mathrm{phys}}>0.
\]

---

## Level 3 — Rerouting/replanning

If

\[
\mathcal T_{\mathrm{dyn}}(p)=\emptyset
\]

for the current route \(p\), then no timing modification can make that route physically realizable.

Therefore:

\[
p\rightarrow p'
\]

using global replanning or HAE.

This is **physical-infeasibility-induced rerouting**.

---

## Level 4 — Safe braking/fallback

If no feasible continuation is available within the available reaction horizon:

\[
\boxed{\text{braking / terminal safe set}}
\]

should be invoked.

This prevents the architecture from pretending that every failure can be solved by replanning.

---

# 10. Main theoretical framework

The theory should focus on a new object:

\[
\boxed{
\mathcal R_{\mathrm{pred}}(x,E)
}
\]

the predicted set of realizable future commands/trajectories.

Define:

\[
\mathcal F_{\mathrm{kin}}(x)
\]

for kinematic feasibility,

\[
\mathcal F_{\mathrm{obs}}(x,E)
\]

for collision/environment feasibility,

and

\[
\mathcal F_{\mathrm{dyn}}(x,E)
\]

for actuator/dynamic realizability.

Then the desired planning set is:

\[
\boxed{
\mathcal F_{\mathrm{safe}}
=
\mathcal F_{\mathrm{kin}}
\cap
\mathcal F_{\mathrm{obs}}
\cap
\mathcal F_{\mathrm{dyn}}.
}
\]

The central theoretical question becomes:

> Under what assumptions does the predictive certificate correctly identify whether the nominal trajectory belongs to \(\mathcal F_{\mathrm{dyn}}\), and under what conditions does feedback replanning recover a trajectory in \(\mathcal F_{\mathrm{safe}}\)?

---

# 11. Candidate Theorems

## Theorem 1 — Predictive physical-realizability certificate

Under bounded model uncertainty and bounded external/contact-force prediction error, if

\[
m_{\mathrm{phys}}(k+j)\ge0
\quad
\forall j=0,\ldots,N,
\]

then the predicted nominal trajectory satisfies the actuator constraints over the prediction horizon.

This theorem should explicitly state all assumptions rather than claim unconditional safety.

---

## Theorem 2 — Realizability-preserving adaptation

Suppose the nominal route admits at least one trajectory satisfying

\[
\mathcal F_{\mathrm{kin}}
\cap
\mathcal F_{\mathrm{obs}}
\cap
\mathcal F_{\mathrm{dyn}}
\neq\emptyset.
\]

If the local adaptation problem includes the predictive dynamic-feasibility constraints and is feasible, then the resulting trajectory remains geometrically feasible while restoring actuator realizability.

The important point is that adaptation changes the execution profile without necessarily changing the global route.

---

## Theorem 3 — Rerouting necessity

For a candidate route \(p\), define

\[
\mathcal T_{\mathrm{dyn}}(p)
=
\{\text{dynamically realizable trajectories along }p\}.
\]

If

\[
\mathcal T_{\mathrm{dyn}}(p)=\emptyset,
\]

then no retiming-only or local acceleration modification can recover physical realizability along that route. Therefore a route-level change is necessary.

This gives a mathematical justification for the transition:

\[
\boxed{
\text{adapt}
\rightarrow
\text{reroute}.
}
\]

---

## Theorem 4 — Recursive feasibility / safe fallback

This should be attempted only if the required assumptions can be established.

Define a terminal safe set

\[
\mathcal X_f
\]

such that for every state

\[
x_k\in\mathcal X_f
\]

there exists a bounded safe braking/fallback policy satisfying actuator and collision constraints.

Then prove:

\[
x_k\in\mathcal X_f
\Rightarrow
x_{k+1}\in\mathcal X_f.
\]

This would establish a stronger recursive-feasibility result.

If the proof becomes too restrictive, omit it from the primary paper rather than forcing an artificial theorem.

---

# 12. Computational structure

A key architectural property should be preservation of the fast local-planner structure.

Where possible:

\[
H,\Phi,\Gamma,C_{\mathrm{kin}}
\]

remain fixed or structurally unchanged.

State/environment-dependent information should preferably enter through:

\[
h(x,E),
\qquad
d(x,E),
\qquad
\text{margin vectors},
\]

rather than rebuilding the entire optimization structure at every iteration.

This maintains the central computational philosophy of the DI-QP/HAE work:

\[
\boxed{
\text{adaptive numerical values}
\neq
\text{adaptive solver structure}.
}
\]

---

# 13. MoveIt 2 integration

MoveIt 2 provides an unusually appropriate validation platform because its Hybrid Planning architecture explicitly separates a slower global planner from a faster recurrent local planner and supports event-driven replanning. [MoveIt Hybrid Planning](https://moveit.picknik.ai/main/doc/concepts/hybrid_planning/hybrid_planning.html)

The proposed implementation can therefore be:

```text
MoveIt Global Planner
        │
        ▼
Global trajectory
        │
        ▼
Predictive Realizability-Aware Local Planner
        │
        ├── DI-QP
        ├── HAE
        └── predictive physical certificate
        │
        ▼
Controller
        │
        ▼
Robot
        │
        └── state/contact/force feedback
```

The integration should preserve MoveIt's global planner and replace/customize the local planning component rather than modifying the entire MoveIt architecture.

MoveIt explicitly allows custom global/local planner plugins and event-driven planning logic. [MoveIt Hybrid Planning tutorial](https://moveit.picknik.ai/main/doc/examples/hybrid_planning/hybrid_planning.html)

---

# 14. Benchmark platforms

## Phase I — Manipulator

Recommended first platform:

- Franka FR3 or Panda
- MoveIt 2
- ROS 2
- Gazebo / Isaac Sim / equivalent simulator
- optional real hardware

Advantages:

- mature MoveIt integration,
- well-defined joint torque limits,
- easy payload experiments,
- external-force experiments,
- collision scenarios,
- manageable computation.

---

## Phase II — Mobile manipulator / legged robot

Test:

- uneven terrain,
- steps,
- holes,
- changing contact,
- payload,
- external pushes.

The central experiment should demonstrate that geometric planning alone can produce a nominally valid trajectory whose future actuator margin collapses.

---

## Phase III — Humanoid

If hardware becomes available, evaluate:

- flat walking,
- uneven terrain,
- upward/downward steps,
- lateral disturbance,
- payload,
- contact changes,
- predicted ankle/hip authority loss,
- rerouting or footstep adaptation.

Humanoid validation is valuable but should not be a prerequisite for the first proof-of-concept.

---

# 15. Benchmark experiments

## Experiment 1 — Baseline trajectory

Compare:

1. MoveIt OMPL + TOTG;
2. MoveIt OMPL + Ruckig;
3. MoveIt Hybrid Planning baseline;
4. proposed predictive-realizability-aware local planner.

Metrics:

- planning time,
- execution time,
- peak torque,
- minimum torque margin,
- acceleration violation,
- tracking error,
- collision clearance.

---

## Experiment 2 — Payload variation

Use identical geometric trajectories while varying payload.

Expected result:

\[
\text{same path}
\quad\Rightarrow\quad
\text{different physical feasibility}.
\]

Show that the proposed method detects the loss of authority before saturation.

---

## Experiment 3 — External interaction force

Apply:

\[
F_{\mathrm{int}}(t)
\]

during execution.

Compare:

- no prediction,
- reactive saturation handling,
- proposed predictive feedback.

Primary metric:

\[
T_{\mathrm{warning}}
=
T_{\mathrm{failure}}
-
T_{\mathrm{detection}}.
\]

---

## Experiment 4 — Uneven terrain

Provide future terrain information.

Examples:

- flat ground,
- slope,
- shallow depression,
- deep hole,
- step-up,
- step-down.

Demonstrate:

\[
\text{terrain}
\rightarrow
\text{contact prediction}
\rightarrow
\text{torque-authority loss}
\rightarrow
\text{adapt/reroute}.
\]

This experiment is particularly important because it demonstrates that the method is not merely a torque-limit checker.

---

## Experiment 5 — Route that is geometrically feasible but dynamically infeasible

Construct two routes:

\[
P_A:\quad
\text{shorter but high dynamic demand}
\]

\[
P_B:\quad
\text{longer but physically realizable}.
\]

A conventional planner selects \(P_A\).

The proposed method detects:

\[
\mathcal T_{\mathrm{dyn}}(P_A)=\emptyset
\]

and triggers rerouting:

\[
P_A\rightarrow P_B.
\]

This should become one of the flagship experiments.

---

## Experiment 6 — Adaptation versus rerouting

Create a continuum of difficulty.

### Case A

Small margin loss:

\[
m_{\mathrm{phys}}\downarrow
\]

Retiming succeeds.

### Case B

Moderate margin loss:

Trajectory reshaping succeeds.

### Case C

Severe loss:

Only rerouting succeeds.

### Case D

No safe route:

Braking/fallback succeeds.

This directly validates the proposed hierarchy:

\[
\boxed{
\text{execute}
\rightarrow
\text{adapt}
\rightarrow
\text{reroute}
\rightarrow
\text{brake}.
}
\]

---

# 16. Ablation studies

At minimum:

### A1 — No predictive physical feedback

Standard planner/controller.

### A2 — Current-state saturation only

No future prediction.

### A3 — Prediction without planner feedback

Detects future saturation but does not replan.

### A4 — Prediction + adaptation

No rerouting.

### A5 — Full proposed system

Prediction + adaptation + rerouting + fallback.

This isolates the contribution of each layer.

---

# 17. Important comparison metrics

The paper should not focus only on torque.

Use:

\[
\boxed{
\begin{array}{l}
\text{task success rate}\\
\text{collision rate}\\
\text{fall/failure rate}\\
\text{minimum actuator margin}\\
\text{number of saturation events}\\
\text{tracking error}\\
\text{minimum obstacle clearance}\\
\text{replanning count}\\
\text{replanning latency}\\
\text{local-planner computation time}\\
\text{global-planner invocation rate}\\
\text{conservatism}
\end{array}}
\]

The last metric is important because an overly conservative feasibility certificate can trivially prevent failure by stopping the robot everywhere.

---

# 18. Key research questions

The paper should explicitly answer:

### RQ1
Can future actuator infeasibility be predicted sufficiently early to enable meaningful planner intervention?

### RQ2
Does predictive physical-realizability feedback reduce failures compared with reactive saturation handling?

### RQ3
When is retiming/reshaping sufficient, and when is rerouting necessary?

### RQ4
Can the proposed certificate incorporate environment/contact/payload/disturbance information without rebuilding the local planner?

### RQ5
Can the architecture operate within the real-time requirements of a local planner?

### RQ6
Does the approach generalize across manipulators and legged/humanoid systems?

---

# 19. Expected contributions

The final paper should ideally claim only 3–4 contributions.

### Contribution 1 — New problem formulation

A formal formulation of **predictive physical realizability as a feedback variable for motion planning**, distinguishing geometric/kinematic feasibility from future actuator realizability.

### Contribution 2 — Predictive certificate

A state-, environment-, contact-, payload-, and disturbance-conditioned certificate that predicts future loss of physical authority.

### Contribution 3 — Hierarchical planning response

A principled hierarchy:

\[
\text{adapt}
\rightarrow
\text{reroute}
\rightarrow
\text{safe fallback},
\]

with conditions determining when local adaptation is insufficient and route-level replanning is necessary.

### Contribution 4 — Real-time implementation and validation

Implementation in a fast local-planning architecture, with MoveIt 2 Hybrid Planning as a realistic software integration platform and experiments spanning manipulator and/or legged systems.

---

# 20. What should NOT be claimed

To reduce editorial-rejection risk, avoid claims such as:

- "first torque-aware motion planner";
- "first actuator-aware robot planner";
- "MoveIt cannot handle dynamic constraints";
- "our method guarantees safety under arbitrary disturbances";
- "predictive saturation guarantees no falling";
- "no previous work has considered torque limits."

Instead, claim the narrower and stronger contribution:

> **The paper introduces a predictive feedback interface that converts future physical-realizability information from the execution layer into motion-planning adaptation and route-level replanning.**

The novelty is the **planning–realization feedback loop**, not the existence of torque limits or saturation constraints.

---

# 21. Target paper positioning

## Primary target

**IEEE Transactions on Robotics (T-RO)**

The paper should be presented as a fundamental robotics problem rather than a MoveIt integration paper.

The central narrative should be:

\[
\boxed{
\text{Geometric feasibility}
\neq
\text{physical realizability}
}
\]

and

\[
\boxed{
\text{Predict physical failure}
\rightarrow
\text{feed realization information back to planning}
\rightarrow
\text{avoid failure before it occurs}.
}
\]

---

## Secondary targets

If the theoretical contribution is not sufficiently mature:

- RA-L + ICRA/IROS;
- ICRA/IROS systems-oriented paper;
- Control-oriented venue if the theorem becomes the dominant contribution.

MoveIt integration can also be released independently as an open-source ROS 2 package.

---

# 22. Proposed paper structure

## I. Introduction

1. Motion planning usually assumes execution feasibility.
2. Real robots have state/environment-dependent physical authority.
3. Future saturation can be caused by terrain, contact, payload, disturbance, and interaction.
4. Existing global/local planning architectures provide an interface for reactive adaptation.
5. Introduce predictive physical realizability feedback.
6. State contributions.

---

## II. Related Work

### A. Motion planning and time parameterization
### B. Dynamic/torque-aware planning
### C. Reactive and hybrid motion planning
### D. Reference governors and safety filters
### E. Actuator saturation and predictive feasibility
### F. Position of the proposed framework

---

## III. Problem Formulation

Define:

- robot dynamics,
- nominal planner,
- environment prediction,
- actuator limits,
- uncertainty,
- physical-realizability set.

Define:

\[
\mathcal F_{\mathrm{safe}}
=
\mathcal F_{\mathrm{kin}}
\cap
\mathcal F_{\mathrm{obs}}
\cap
\mathcal F_{\mathrm{dyn}}.
\]

---

## IV. Predictive Physical-Realizability Certificate

Derive:

- torque prediction,
- uncertainty tightening,
- directional authority,
- horizon margin,
- time-to-loss-of-authority.

---

## V. Feedback Planning Architecture

### A. Normal execution
### B. Local adaptation
### C. Physical-infeasibility detection
### D. Rerouting
### E. Safe fallback

Integrate HAE here as the geometric escape/rerouting mechanism.

---

## VI. Theoretical Analysis

- certificate correctness;
- adaptation feasibility;
- route-level infeasibility;
- optional recursive feasibility theorem;
- computational complexity.

---

## VII. MoveIt 2 / Real-Time Implementation

- Hybrid Planning integration;
- global planner;
- local DI-QP;
- predictive realization module;
- ROS 2 communication;
- timing measurements.

MoveIt 2's official architecture explicitly supports replaceable global/local planner plugins and event-driven planning logic. [MoveIt Hybrid Planning](https://moveit.picknik.ai/main/doc/concepts/hybrid_planning/hybrid_planning.html)

---

## VIII. Experiments

1. baseline manipulator;
2. payload;
3. interaction force;
4. disturbance;
5. uneven terrain/contact;
6. dynamically infeasible route;
7. adaptation versus rerouting;
8. ablation;
9. real-time performance.

---

## IX. Discussion

Discuss:

- conservatism;
- model uncertainty;
- environment prediction uncertainty;
- sensing latency;
- computational scaling;
- limitations of the certificate;
- relationship to learned/world-model planners.

---

## X. Conclusion

The final message:

> Motion planning should not only ask whether a trajectory can be geometrically generated; it should continuously ask whether the robot will remain physically capable of realizing it.

---

# 23. Long-term research direction

This framework can eventually provide a concrete interface between high-level AI/world models and model-based physical control.

A possible hierarchy is:

\[
\text{VLM / World Model}
\]

\[
\downarrow
\]

\[
\text{Global Motion Planner}
\]

\[
\downarrow
\]

\[
\text{Predictive Physical Realizability}
\]

\[
\downarrow
\]

\[
\text{Whole-Body / Interaction Controller}
\]

\[
\downarrow
\]

\[
\text{Robot + Environment}.
\]

The high-level system determines **what the robot should do**.

The planner determines **how it should move**.

The predictive realization layer determines **whether the robot can physically execute that behavior under the current and predicted environment**.

The resulting feedback closes the loop:

\[
\boxed{
\text{World}
\rightarrow
\text{Plan}
\rightarrow
\text{Predict Physical Realizability}
\rightarrow
\text{Execute}
\rightarrow
\text{Replan}.
}
\]

This is the proposed long-term foundation for a model-based Physical AI architecture.

---

# 24. Immediate implementation plan

### Stage 1 — Simulation proof

MoveIt 2 + FR3/Panda:

- OMPL/TOTG baseline;
- proposed local planner;
- payload;
- external force;
- torque prediction;
- adaptive retiming.

### Stage 2 — Physical rerouting

Construct two routes with different dynamic demands and demonstrate:

\[
\text{short/geometrically valid}
\rightarrow
\text{physically infeasible}
\rightarrow
\text{reroute}.
\]

### Stage 3 — Environment-aware test

Add terrain/contact prediction:

\[
\text{terrain}
\rightarrow
\text{future contact}
\rightarrow
\text{authority margin}
\rightarrow
\text{replanning}.
\]

### Stage 4 — Real robot

Validate on FR3/Panda if available.

### Stage 5 — Legged/humanoid

Extend to uneven terrain, steps, disturbances, and falling prevention.

---

# 25. One-sentence paper thesis

> **We propose a predictive physical-realizability feedback architecture that enables a motion planner to anticipate actuator-authority loss caused by robot state, environment, contact, payload, interaction, and disturbance, and to respond through local adaptation, rerouting, or safe fallback before physical infeasibility becomes execution failure.**
