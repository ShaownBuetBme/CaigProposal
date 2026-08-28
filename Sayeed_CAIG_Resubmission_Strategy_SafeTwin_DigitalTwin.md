# SafeTwin
### AI-Calibrated Digital Twins and Trustworthy Learning for Mixed-Autonomy Transportation

**SafeTwin Resubmission Strategy | August 2026**

A revised resubmission strategy giving Sayeed Chowdhury substantive co-leadership in digital-twin/simulation methodology and trustworthy AI

**Core strategic move:** Make the digital twin itself a research instrument: combine physics/GIS simulation with ML-based calibration, discrepancy correction, learned surrogates, uncertainty modeling, and scenario generation; then use that adaptive twin to develop trustworthy multi-agent control and prevent local failures from becoming network cascades.

Prepared for: Prof. Mohammad Hasan and collaborators
Proposed co-PI contribution: Sayeed Shafayet Chowdhury, PhD | Indiana University Indianapolis, Luddy School of Informatics, Computing, and Engineering
Updated August 13, 2026

> **Basis:** uploaded NSF CAIG proposal + current NSF program pages checked August 13, 2026. The uploaded PDF does not contain the actual NSF reviewer/panel comments; fit diagnoses are therefore source-based inferences rather than fabricated reviewer quotations.

---

## Executive verdict

The strongest revision is **NOT** to force SNNs, neuromorphic computing, or event cameras into this proposal. Those are optional future extensions, not requirements for a compelling co-PI role. The revised project should instead exploit Sayeed's broader machine-learning expertise in robustness, uncertainty, safety, efficient training, multimodal modeling, and simulation-based evaluation.

> **One-sentence thesis:** How can an AI-calibrated digital twin learn where its simulation is reliable, efficiently generate rare and shifted operating conditions, and support autonomous agents that remain safe under uncertainty while preventing local errors from triggering network-scale cascades?

### Why this is strategically stronger

- It upgrades the digital twin from a static platform-development task into a scientific contribution: an adaptive, uncertainty-aware, ML-calibrated twin that can quantify its own mismatch, emulate expensive rollouts, and generate targeted stress scenarios.
- It creates genuine AI novelty: uncertainty-aware state estimation, risk-sensitive/constraint-aware multi-agent learning, selective autonomy, and cascade-aware control.
- It uses the original team assets rather than discarding them: TransModeler/GIS, INDOT data, mixed-autonomy traffic, winter conditions, and network-complexity analysis remain central.
- It gives Sayeed two visible intellectual roles: co-lead the AI/simulation layer of the digital twin (T1) and lead trustworthy multi-agent learning (T2), with joint ownership of cascade-aware control (T3). These are software-first and executable without new vehicles or specialized hardware.
- It aligns cleanly with NSF Future CoRe CPS-FR; the same research engine can be reframed for CPS-CIR or ISP if the team wants more use-inspired/infrastructure emphasis.

### Recommended primary target

NSF Future CoRe - Cyber-Physical System Foundations (CPS-FR), with Robust Intelligence (RI) as a possible secondary program identifier/co-review if the AI methods are developed deeply enough. CPS-FR explicitly welcomes AI-enabled autonomy, safe learning, adaptive systems, planning/optimization under uncertainty, assurance, resource management, and robustness to failures. The adaptive digital twin is both a CPS modeling contribution and the controlled-but-realistic validation environment for the safe-learning contributions. [N2-N4]

---

## 1. What should be preserved from the original proposal

The original CAIG proposal already contains a valuable systems backbone. It proposes a geospatially faithful Central Indiana transportation digital twin, calibrated from real traffic and weather data, with mixed human/CAV agents and a network-complexity layer. [P1]

| Original component | Why it is worth keeping | How it changes in SafeTwin |
|---|---|---|
| GIS/TransModeler digital twin | Rare, realistic test environment with lane-level infrastructure and agency relevance | Keep the GIS/traffic core, but add an AI-calibrated twin layer: learned discrepancy correction, uncertainty, surrogates, active scenario design, and held-out validation. |
| INDOT/ATLAS data and calibration | Gives empirical grounding and stakeholder credibility | Becomes a joint Banerjee-Sayeed research task: Bayesian/GP calibration, residual ML correction, uncertainty decomposition, and data-assimilation updates. |
| Mixed human / partially automated / CAV traffic | Scientifically important nonstationarity and interaction | Becomes a source of partial observability, behavior shift, and strategic uncertainty. |
| Winter weather and incidents | Useful stressors under which policies may fail | Reframed as distribution shift / operating-condition perturbations rather than a geoscience claim. |
| MARL + game-theoretic coordination | Natural control baseline and multi-agent test setting | Baseline methods are retained, but the contribution becomes new safety/uncertainty/cascade-aware learning. |
| Percolation / cascades / resilience | Potentially distinctive systems-level analysis | Moved from post-hoc descriptive metrics into the training objective and safety constraints. |

The original proposal itself states that some planned RL methods are mature and differentiates them mainly through context (CAV + weather + a digital-twin backend). That is exactly the part to fix: the new version must claim and test a new learning problem, not simply a new simulation setting. [P1]

---

## 2. New scientific spine: trustworthy learning before the cascade

The proposal should be organized around a local-to-global failure chain:

> uncertain observation → incorrect local belief/action → interaction with nearby human/CAV agents → queue/route perturbation → spillback and network cascade.

This creates four research questions that are both ambitious and experimentally tractable:

1. **Adaptive digital twin:** How can field data be used to calibrate a traffic twin, learn systematic simulator discrepancy, quantify where the twin is uncertain, and update the model as conditions change?
2. **Efficient simulation and scenario discovery:** Can learned multi-fidelity surrogates and active/adversarial scenario generation reduce the cost of large simulation sweeps while preserving the rare events and critical transitions that matter for safety?
3. **Safe decision-making under partial observability:** How can multi-agent policies reason over missing, delayed, noisy, or shifted observations while controlling safety tail risk?
4. **Local-to-network assurance:** Can a learned predictor estimate the downstream cascade risk of local actions early enough to constrain or redirect policies before the network crosses a critical regime?

**Key conceptual novelty:** The proposal no longer treats the simulator as unquestioned ground truth. It asks how to build and validate an adaptive AI-calibrated twin, where the twin is uncertain, how to use it efficiently for rare-event discovery, and how learning failures propagate through — and can be stopped within — the cyber-physical network.

*Figure 1. The cascade-risk model is part of the controller loop, so complexity analysis changes decisions rather than merely describing simulation output.*

---

## 3. Revised research thrusts and ownership

| Thrust | Scientific objective | Core methods | Primary ownership |
|---|---|---|---|
| **T1. Adaptive AI-Calibrated Digital Twin + Simulation Science** | Build, validate, and accelerate a mixed-autonomy twin that explicitly models simulator discrepancy and uncertainty and can generate targeted rare/shifted scenarios. | Bayesian/GP calibration; residual ML correction; ensemble uncertainty; multi-fidelity surrogate emulator; active/adversarial scenario generation; held-out corridor/day validation. | Banerjee + Sayeed co-lead; Hasan |
| **T2. Trustworthy AI Under Partial Observability** | Develop learning policies that know when state estimates are uncertain and control safety tail risk while interacting with heterogeneous drivers. | Belief-state encoder; risk-sensitive/constrained MARL; uncertainty calibration; selective action/fallback safety layer. | Sayeed lead; Hasan |
| **T3. Cascade-Aware Network Control** | Learn when local errors/actions are likely to induce network-level congestion/safety cascades, then feed that risk into policy optimization. | Temporal graph risk predictor; cascade-size/CVaR critic; percolation and topology features; intervention-aware reward/constraints. | Hasan + Sayeed co-lead; Banerjee |

### Why Sayeed is scientifically necessary rather than additive

- T1 gives Sayeed genuine digital-twin ownership without pretending to be the transportation/GIS specialist: Banerjee builds the geospatial traffic model; Sayeed develops the ML calibration, discrepancy, surrogate, uncertainty, and scenario-generation layer around it.
- T2 is an independent algorithmic thrust with its own hypotheses, methods, baselines, and deliverables.
- T3 requires robust-learning expertise to convert network cascade metrics into a predictive control signal; this is the bridge between Hasan's network science and the learning controller.
- The T1/T2/T3 pipeline is software-first and executable with simulator access, traffic data, standard ML libraries, and ordinary GPU/CPU compute - no specialized sensing hardware is required.
- The output is legible to AI reviewers (calibration, surrogates, uncertainty, robustness, generalization) and CPS/infrastructure reviewers (validated digital twins, safe operation, and resilience).

---

## 4. Sayeed's concrete digital-twin and simulation contribution

Yes — Sayeed can be deeply involved in the digital twin without taking over the transportation-engineering tasks. The defensible role is to own the AI-enabled twin layer that sits between field data and the TransModeler simulation core. This is methodological work in machine learning, probabilistic modeling, simulation, and validation.

| Digital-twin task | What Sayeed can lead | Executable experiment / deliverable |
|---|---|---|
| 1. Data-driven calibration | Estimate uncertain simulator parameters from INDOT observations using GP/Bayesian optimization or neural amortized calibration; report posterior/ensemble uncertainty rather than one best parameter vector. | Train on selected days/corridors and evaluate speed-flow-queue fit on held-out days/corridors; compare default, classical calibration, and uncertainty-aware calibration. |
| 2. Simulator-discrepancy learning | Learn systematic residual error between TransModeler and observed traffic as a function of traffic state, weather, geometry, and time; use the residual model to correct predictions and expose regions where the twin is unreliable. | Fit XGBoost/MLP/GNN residual models on calibration data; evaluate whether corrected twin reduces held-out error and whether uncertainty predicts high-error regimes. |
| 3. Multi-fidelity surrogate twin | Train a fast learned emulator of expensive corridor/network rollouts for screening, RL curriculum construction, and sensitivity analysis, while periodically anchoring to full TransModeler runs. | Predict link speeds, queues, travel times, or cascade summaries from scenario inputs; measure fidelity, uncertainty, and wall-clock acceleration versus full simulation. |
| 4. Active scenario generation | Use uncertainty, rare-event labels, or adversarial search to choose the next simulations that are most informative instead of uniformly sweeping a huge parameter grid. | Compare random/LHS sampling with uncertainty- or risk-guided acquisition on number of full simulator runs needed to discover failures or estimate tail risk. |
| 5. Failure-injection and OOD simulation framework | Create reusable perturbation modules for V2X dropout/latency, stale state, sensor corruption, incidents, demand shifts, driver heterogeneity, and weather shifts. | Publish a scenario/failure library and benchmark policies under matched nominal, shifted, and rare-event test sets. |
| 6. Twin validity / sim-to-real assessment | Separate calibration fit from validation and explicitly quantify reality gap, uncertainty, sensitivity, and transfer instead of claiming realism at unobserved CAV penetration. | Hold out corridors/time windows; perform sensitivity and posterior predictive checks; report where conclusions are robust to twin uncertainty. |

**Recommended ownership boundary:** Banerjee should remain the lead authority for GIS, roadway topology, TransModeler configuration, traffic-engineering assumptions, and agency data semantics. Sayeed should co-lead T1 by owning the machine-learning and simulation-science layer: calibration, discrepancy correction, uncertainty, surrogate modeling, active scenario generation, and validation analytics. That division is credible and makes Sayeed a digital-twin co-developer rather than merely an AI user.

---

## 5. Concrete AI contributions that are novel but executable

Beyond the twin layer in T1, the control side should avoid promising a vague "new AI framework." Commit to tightly scoped learning contributions that can be implemented incrementally on top of MAPPO/PPO and standard sequence/graph models.

| Method | What is new | Implementation path | Why feasible |
|---|---|---|---|
| **A. Uncertainty-Aware Belief MARL** | Learn a latent belief state from asynchronous/missing observations and explicitly carry predictive uncertainty into the policy/critic. | Sequence encoder (GRU/Transformer) + observation masks/time-since-last-update; deep ensembles or distributional heads; MAPPO baseline. | All data are simulator state/traffic streams. No sensor hardware is required. |
| **B. Tail-Risk / Constraint-Aware Policy Optimization** | Optimize not only expected reward but safety and network tail risk, e.g., CVaR of collision/near-miss/cascade delay, with hard or Lagrangian constraints. | Add safety/cascade critics and constrained policy update to PPO/MAPPO; compare with ordinary weighted rewards. | Builds on mature RL code while creating a clear methodological hypothesis and ablation structure. |
| **C. Selective Autonomy Safety Gate** | When belief uncertainty or predicted cascade risk exceeds a calibrated threshold, the agent abstains from aggressive actions and switches to a conservative controller. | Calibrated risk score + conformal/coverage-based thresholding; fallback IDM/CACC or safe-speed controller. | Simple enough to implement early; yields strong safety-coverage and utility-coverage curves. |
| **D. Cascade-Risk Graph Critic** | Predict future cascade size/critical transition from local and network state, then regularize or constrain local policies using predicted downstream harm. | Temporal GNN over road links + current traffic/incident/policy features; supervision from simulation rollouts. | Directly leverages Hasan's network expertise and existing simulated trajectory logs. |

**Do not overclaim theory:** The proposal can promise formal definitions and empirical bounds/guarantees where justified, but it should not promise full provable safety for a city-scale MARL system. A credible goal is calibrated uncertainty + risk constraints + extensive stress testing + limited analytical results on simplified network models.

---

## 6. Experimental program: reviewer-ready and realistically executable

Each experiment should answer one hypothesis. The first two experiments make Sayeed visibly responsible for digital-twin/simulation methodology; the remaining experiments connect that twin to safe control and cascade resilience. All are designed to run with simulator/API access and ordinary ML compute.

| Exp. | Question / manipulation | Baselines | Primary metrics |
|---|---|---|---|
| **E1** | Twin calibration + discrepancy: train on selected Indianapolis days/corridors; hold out days/corridors; learn residual error after base calibration. | Default TransModeler; classical calibration; GP/Bayesian calibration; + residual ML correction. | speed/flow/queue error; posterior predictive fit; residual error; uncertainty-error correlation. |
| **E2** | Surrogate / active simulation: learn a fast emulator of corridor/network summaries and use uncertainty/risk to choose new full simulations. | Full TransModeler only; random/LHS scenario sampling; GP/MLP/GNN surrogate. | surrogate MAE/calibration; wall-clock cost; number of full runs to discover failures / estimate tail metrics. |
| **E3** | Observation dropout/noise: randomly mask or corrupt local state and V2X messages at 0-50%. | MAPPO with privileged/full state; recurrent MAPPO; proposed belief MARL. | collision/near-miss; TTC; reward; calibration; graceful-degradation slope. |
| **E4** | Latency and communication failure: 0-1000 ms delay, packet loss, stale messages, RSU outage. | No-communication; naive last-value; standard MARL. | safety tail risk; throughput; bytes/messages; recovery time. |
| **E5** | Distribution shift: train in clear/moderate demand; test heavy snow, incidents, special-event demand, altered driver aggressiveness. | Domain randomization; ordinary MARL; robust/adversarial training baseline. | OOD performance drop; CVaR; calibration error; worst-case delay. |
| **E6** | Rare-event stress testing: risk-guided/adversarially sample incident location, duration, lane closure, sudden braking and merge conflicts. | Random scenario sampling; standard curriculum. | failure discovery rate per simulator-hour; collision/near-miss; cascade size distribution. |
| **E7** | Selective autonomy: vary uncertainty/risk threshold to trade utility for safety. | Always-autonomous; always-conservative fallback. | risk-coverage curve; throughput-coverage; interventions per vehicle-mile. |
| **E8** | Local-to-global cascade prediction: predict cascade size 5-15 min ahead from partial network state. | Centrality-only model; XGBoost/MLP; static GNN. | AUROC/AUPRC for cascade onset; MAE for size; calibration; lead time. |
| **E9** | Cascade-aware control: add predicted cascade risk to policy constraints and compare. | Ordinary MAPPO; hand-tuned congestion reward; complexity metric only. | cascade probability/size; CVaR delay; travel time; safety. |
| **E10** | Generalization: train/calibrate in Indianapolis core; test adjacent counties / unseen layouts and shifted twin parameters. | Same model without adaptation; fine-tuned model. | zero-shot/few-shot degradation; twin error; safety; throughput; calibration. |

---

## 7. Evaluation: replace "average reward" with a safety + generalization scorecard

| Dimension | Metrics to report | Reviewer-facing interpretation |
|---|---|---|
| Safety | collision rate, near-miss rate, TTC distribution, hard-brake rate, CVaR of safety cost | Does the method control the tail, not only the mean? |
| Traffic efficiency | travel time, throughput, queue length, vehicle-hours of delay | Does safety come at an unacceptable operational cost? |
| Network resilience | cascade probability, cascade size, recovery time, critical-threshold shift | Does local learning reduce system-level fragility? |
| Uncertainty quality | NLL/Brier/ECE where applicable; risk calibration; conformal coverage | Does the model know when it is likely to fail? |
| Robustness / OOD | relative degradation under weather, demand, driver, sensor, and communication shift | Does behavior degrade gracefully outside training conditions? |
| Communication / compute | messages or bytes per agent-step, inference latency, GPU training cost | Is the controller operationally plausible and scalable? |
| Equity / heterogeneity | worst-corridor or subgroup delay where policy analysis is retained | Do improvements hide localized burdens? |

### Statistical design

- Use at least 5-10 independent random seeds for learned policies and bootstrap confidence intervals over scenario-level outcomes.
- Predefine primary endpoints (e.g., CVaR delay, near-miss rate, cascade probability) so the proposal does not read like a fishing expedition over dozens of metrics.
- Use factorial scenario design for CAV penetration × demand × weather × communication quality, but reserve a smaller set of stress-test scenarios for final blinded/held-out evaluation.
- Report Pareto fronts (safety vs throughput; intervention rate vs risk) rather than a single weighted reward whenever objectives compete.

---

## 8. Feasibility: what Sayeed can actually execute

The redesign is intentionally software-first. Sayeed can contribute to the twin by working on calibration, learned emulation, scenario generation, and validation analytics around the existing TransModeler/GIS core; he does not need to become a traffic-simulator developer from scratch or require instrumented vehicles, event cameras, neuromorphic hardware, or proprietary AV perception stacks.

| Resource | Minimum viable plan | Fallback / de-risking |
|---|---|---|
| Simulator | Use the team's TransModeler model and API for corridor/network experiments. | Prototype algorithms first in SUMO or a small custom Gym environment, then transfer to TransModeler. |
| Data | INDOT counts/trajectories available through the team; ATLAS/GIS; historical weather used in original proposal. | Public trajectory data can be used only as auxiliary behavioral validation if access to a particular INDOT stream is delayed. |
| Compute | 1-2 modern GPUs plus CPU simulation workers; distributed rollouts only when scaling to city scenarios. | Start with corridors/intersections; train centralized critics on sampled sub-networks; scale after algorithm stabilizes. |
| Software | PyTorch + probabilistic/GP tools + RL/graph libraries + simulator bridge; modular APIs for calibration, surrogate rollout, scenario generation, and policy evaluation. | Keep the AI twin layer simulator-agnostic via Gym/PettingZoo-style interfaces so methods can be prototyped independently of one proprietary backend. |
| Validation | Hold out time periods, corridors, and scenario families; compare to field traffic where real data exist. | Do not claim "realism" at unobserved CAV penetrations; quantify simulator discrepancy and sensitivity instead. |

**Scope discipline:** City-scale simulation can be computationally expensive. Train and debug on 1-3 representative corridors/intersections, validate on multiple seeds and shifts, then use the full regional twin primarily for transfer/generalization and network-cascade studies.

---

## 9. Preliminary studies to generate before submission

A small amount of preliminary evidence would dramatically change reviewer confidence. The goal is not to solve the project before submission; it is to prove the interface works and expose the exact failure mode the grant will solve.

| Week | Deliverable | Figure/result to place in proposal |
|---|---|---|
| 1 | Verify one TransModeler corridor and export simulator input/output traces; reproduce default simulation baseline. | Observed vs simulated speed/flow/queue distributions. |
| 2 | Fit a simple GP/Bayesian calibration and residual ML correction on a subset of days; evaluate held-out days. | Before/after reality-gap plot + uncertainty vs held-out error. |
| 3 | Train a lightweight surrogate for corridor summary outcomes and quantify fidelity/cost. | Surrogate prediction plot + measured simulator wall-clock savings. |
| 4 | Create simulator-to-RL interface and reproduce rule-based/MAPPO baseline; inject dropout/latency/noise. | Baseline performance and first brittleness curve. |
| 5 | Add recurrent belief/uncertainty model and selective safety gate. | Robustness + safety-coverage/throughput-coverage curves. |
| 6 | Generate incident/shifted scenarios using random and risk-guided sampling; label cascade size/onset. | Failure discovery efficiency and example cascade visualization. |
| 7 | Train simple cascade predictor (XGBoost/MLP or first GNN baseline). | Cascade prediction accuracy vs prediction horizon. |
| 8 | Add cascade-risk penalty/constraint to policy and run small ablation. | Evidence that risk-aware training shifts the tail without destroying throughput. |

### The single most valuable preliminary result

The strongest preliminary package is a two-part story: first show that ML calibration/discrepancy modeling improves held-out twin fidelity and quantifies where the simulator is uncertain; then show that a standard MARL policy that looks strong under nominal twin state fails under realistic perturbations while an uncertainty-aware/risk-gated variant degrades more gracefully. Together these establish that Sayeed contributes to both the twin and the controller.

---

## 10. Team structure that makes the co-PI roles non-overlapping

| Investigator | Distinct intellectual ownership | Major deliverables |
|---|---|---|
| Prof. Mohammad Hasan | Network learning, graph/network theory, complexity and cascade mechanisms; system-level prediction/control. | Cascade definitions and predictors; topology-aware risk; network-theoretic analysis; joint T3 algorithms. |
| Prof. Aniruddha Banerjee | GIS/TransModeler core, roadway/infrastructure representation, traffic-engineering assumptions, agency data semantics, scenario realism. | Geospatial traffic twin core; TransModeler configuration; field-data pipeline; transportation-valid corridor/network scenarios. |
| Prof. Sayeed Chowdhury | AI-enabled digital-twin/simulation methodology plus trustworthy ML: calibration, discrepancy learning, surrogate emulation, active scenario generation, uncertainty/OOD evaluation, safe multi-agent learning. | Co-led T1 AI twin layer; calibration/discrepancy models; surrogate/scenario engine; T2 algorithms; joint cascade-aware controller; open simulation/AI software. |

### Recommended wording for the management plan

> Co-PI Chowdhury will co-lead Thrust I, lead Thrust II, and co-lead Thrust III. In Thrust I, he will lead the AI-enabled twin layer: data-driven calibration, simulator-discrepancy learning, uncertainty quantification, learned surrogate emulation, active/rare-event scenario generation, and validation analytics, working with Banerjee's GIS/TransModeler core. In Thrust II, he will develop uncertainty-aware belief representations, risk-sensitive multi-agent learning, selective autonomy mechanisms, and the safety/OOD evaluation framework. In Thrust III, he will work with Hasan to integrate learned cascade-risk models into policy optimization.

This role is large enough to justify co-PI status because it co-owns the digital-twin methodology, owns a complete safe-learning research question, and co-owns cascade-aware control; it is not a support role for merely running simulations.

---

## 11. Where to submit: program-fit ranking as of August 13, 2026

| Program | Fit after redesign | What must lead the narrative | Timing / caution |
|---|---|---|---|
| NSF Future CoRe CPS-FR | EXCELLENT - first choice | Foundational principles for adaptive AI-enabled digital twins plus safe learning/control of networked cyber-physical autonomy under uncertainty; broadly applicable methods validated in transportation. | Target date Sep. 10, 2026; proposals accepted anytime. Discuss fit with CPS program officer well before submission. [N2,N3] |
| NSF Future CoRe CPS-CIR | VERY STRONG if stakeholder/community co-design is emphasized | Transportation/community resilience; INDOT/local stakeholder needs shape research questions and assessment. | Same Future CoRe target dates; requires meaningful community/stakeholder engagement, not just letters. [N2,N3] |
| NSF Infrastructure Systems and People (ISP) | VERY STRONG infrastructure-first fallback | Resilience of transportation infrastructure under innovation, extreme conditions, and human/AI interaction; AI is the system method. | Accepted anytime; NSF encourages a 1-page summary to program officers for multidisciplinary proposals. [N5] |
| NSF Future CoRe Robust Intelligence (RI) | STRONG only if AI novelty is deep | New generalizable AI/ML methods for uncertainty, risk, multi-agent intelligence, and complex realistic contexts. | Do not pitch as "apply MARL to traffic"; RI says simple domain application of existing techniques belongs elsewhere. [N3,N4] |
| NSF SaTC 2.0 RES | CONDITIONAL pivot | Trust/resilience under malicious V2X or sensor attacks, false data, adversarial coordination, or cyber-physical compromise. | Target Sep. 28, 2026; only use if security/adversarial threat model becomes central, not decorative. [N6] |
| NSF CAIG | NOT RECOMMENDED without a true Earth-system science pivot | Must advance a genuine atmosphere/Earth/ocean/polar geoscience question and AI together. GIS + winter-weather covariates are not enough. | Current CAIG scope explicitly centers Earth-system understanding and AI/geoscience partnership. [N1] |

---

## 12. Recommended first submission: CPS-FR framing

### Working title

CPS-FR: Adaptive AI-Calibrated Digital Twins and Cascade-Safe Learning for Mixed-Autonomy Transportation Networks

The title foregrounds the computing/CPS contribution and keeps transportation as the validation domain.

### Three NSF-style project objectives

1. Develop an adaptive AI-calibrated digital-twin methodology that learns simulator discrepancy and uncertainty, supports multi-fidelity surrogate emulation, and actively generates informative/rare operating scenarios.
2. Develop uncertainty-aware, risk-sensitive, and selectively autonomous multi-agent policies that remain effective under missing information, distribution shift, and communication failures.
3. Develop local-to-global cascade-risk models and integrate predicted downstream network harm into multi-agent decision-making to prevent system-level instability.

### Central hypotheses

- **H1:** Learning simulator discrepancy and predictive uncertainty will improve held-out twin fidelity and identify operating regimes where simulation-derived conclusions are unreliable.
- **H2:** Multi-fidelity surrogate + risk/uncertainty-guided scenario selection will discover critical and rare behaviors with fewer expensive full-simulator runs than uniform sampling.
- **H3:** Explicit belief uncertainty and tail-risk constraints will improve OOD safety relative to privileged-state or point-estimate MARL with quantifiable efficiency tradeoffs.
- **H4:** A graph-based cascade-risk critic will identify local states/actions that precede network-wide spillback and enable policies that reduce cascade probability and severity.
- **H5:** Policies trained across calibrated twin uncertainty distributions will transfer better across unseen corridors, demand regimes, and weather conditions than policies trained on nominal scenarios.

### Broad applicability language

Although transportation is the main validation domain, the methods target a broader CPS class: systems in which a digital twin must be calibrated from imperfect field data, expensive simulation must be accelerated without losing critical rare behavior, and local autonomous learners act on incomplete evidence over shared physical resources where errors can become system-level cascades. This generality is important for CPS-FR reviewers.

---

## 13. Draft revised NSF Project Summary — concept version

### Overview

Learning-enabled autonomous systems increasingly share physical infrastructure with humans and with one another. Their decisions depend on observations and communication that can be delayed, missing, noisy, or shifted from training conditions. In transportation networks, a local decision error can propagate through merging, rerouting, queue spillback, and coordination dynamics until it becomes a network-scale safety or congestion cascade. Existing mixed-autonomy simulation studies largely evaluate policies under clean simulator state and average-performance objectives, while network resilience analyses are commonly performed after the fact. This project will develop SafeTwin, an adaptive AI-calibrated digital-twin and trustworthy-learning framework for Central Indiana transportation. The twin will learn simulator discrepancy and uncertainty from field data, use multi-fidelity surrogate models and active scenario generation to make large-scale stress testing computationally tractable, and connect uncertainty-aware multi-agent decision-making to predictive network-cascade risk.

### Intellectual Merit

The project will contribute new methods in three tightly coupled thrusts. First, it will create an adaptive digital-twin methodology that combines field-data calibration, learned simulator-discrepancy correction, uncertainty quantification, multi-fidelity surrogate emulation, and risk-guided scenario generation rather than treating the simulator as deterministic ground truth. Second, it will develop uncertainty-aware belief representations, risk-sensitive multi-agent reinforcement learning, and selective autonomy mechanisms that can abstain or fall back when evidence is insufficient or predicted risk is high. Third, it will develop temporal graph models that predict the downstream cascade risk of local network states and actions and incorporate this risk directly into policy optimization. The resulting algorithms will be evaluated under systematic missing-data, latency, communication-failure, rare-event, and distribution-shift experiments, with metrics spanning safety tail risk, calibration, throughput, network cascade size, and generalization. The research will establish principles for preventing local learning failures from becoming global cyber-physical failures.

### Broader Impacts

The project will provide an open AI-enabled digital-twin toolkit for calibration, discrepancy/uncertainty estimation, surrogate simulation and scenario generation; benchmark scenarios; a simulator-agnostic interface for trustworthy mixed-autonomy evaluation; calibrated failure-injection tools; and network-resilience analysis software. The Indiana digital twin and partnership with transportation stakeholders will allow results to inform planning for mixed autonomy in mid-sized and rural regions that are underrepresented in current AV pilots. The project will train graduate and undergraduate researchers at the intersection of AI, cyber-physical systems, network science, and transportation resilience and will produce educational modules on trustworthy AI for infrastructure systems.

---

## 14. What to remove or rewrite from the old proposal

| Old framing | Why it weakens the revised proposal | Replacement |
|---|---|---|
| "GIS makes this a geoscience object" | Program-fit defense distracts from the actual science and is unnecessary outside CAIG. | Describe GIS as the physical-context representation and validation substrate. |
| "RL methods are mature; novelty is scope/context" | Invites the reviewer to conclude the AI is application engineering. | State specific new learning problems, methods, hypotheses and ablations. |
| Many independent algorithms: route guidance + speed harmonization + GLOSA + game theory + MARL | Creates breadth without a single methodological spine. | Use 1-2 canonical control tasks (merge/corridor + routing) as testbeds for the same safe-learning methods. |
| Complexity/percolation only after simulation | Can read as descriptive analysis detached from AI. | Train a cascade-risk critic and feed risk into policy constraints/reward. |
| Evaluation by average/cumulative reward | Insufficient for safety-critical CPS. | Tail risk + calibration + OOD + cascade + efficiency + compute/communication. |
| Claims of realistic behavior at CAV penetrations without field observations | Reality-gap extrapolation is not direct validation. | Make twin validity a research task: held-out calibration/validation, learned discrepancy, predictive uncertainty, sensitivity, and transfer. |
| Very broad list of outputs/papers/courses/dashboard | Makes feasibility and scientific priority harder to see. | Prioritize 3 algorithmic outputs + benchmark + decision-support translation. |

---

## 15. Two-minute pitch to Prof. Hasan

> **Suggested pitch:** I think the traffic twin and cascade ideas are strong, but we should stop trying to make GIS itself the scientific novelty. I also do not think we need to force my SNN or event-camera work into this project. A cleaner role for me is to lead the trustworthy-AI side: what happens when the CAV agents do not have perfect simulator state — their sensing or V2X is missing, delayed, noisy, or out-of-distribution? I can build uncertainty-aware belief models, risk-sensitive MARL, and a selective safety mechanism. Then your network-science piece becomes much more powerful: instead of only measuring cascades after simulation, we learn a cascade-risk critic and use it to constrain the agents before local failures become system-wide failures. Banerjee's TransModeler/INDOT backbone gives us a realistic testbed. That gives each of us a clean thrust and, in my view, makes the project a much more natural CPS-FR proposal than CAIG.

### What you are offering Hasan concretely

- A rewritten T1 AI-digital-twin subplan: calibration, discrepancy learning, surrogate emulation, scenario generation, and validation, co-led with Banerjee.
- A rewritten T2 with specific trustworthy-learning algorithms and a complete evaluation plan.
- A preliminary twin-fidelity + surrogate benchmark and a robustness benchmark within roughly 4-8 weeks, assuming simulator/API and data access.
- Joint ownership of a new T3 mechanism: predictive cascade-risk learning integrated into control.
- A CPS-FR/RI-facing narrative and program-summary rewrite rather than a cosmetic CAIG revision.
- A feasible student work plan that can produce AI/CPS papers even before the full regional twin is complete.

---

## 16. Submission action plan

| When | Action | Decision criterion |
|---|---|---|
| Immediately | Send a 1-page concept to CPS program team and request fit feedback. If considering ISP, send its program officers a separate infrastructure-first 1-pager. | Confirm whether CPS-FR or CPS-CIR should be primary and whether RI co-review is appropriate. |
| Week 1 | Freeze one scientific spine and assign T1/T2/T3 ownership, with Sayeed explicitly co-leading the AI/twin layer of T1. Drop extra algorithms not serving the spine. | Every task must map to one hypothesis and one co-PI owner. |
| Weeks 1-3 | Build one corridor-scale twin benchmark: export simulator traces, fit baseline calibration/discrepancy model, and establish the RL interface. | Twin baseline must support held-out fidelity checks and stable policy rollouts. |
| Weeks 2-5 | Measure held-out twin discrepancy and surrogate fidelity; then run missing/noisy observation, V2X latency/dropout, weather/demand-shift stress tests. | Obtain at least one twin-fidelity/uncertainty plot and one clear policy brittleness plot. |
| Weeks 3-6 | Implement belief/uncertainty model and selective safety gate; start cascade labeling/prediction. | Show measurable robustness or safety benefit with interpretable tradeoff. |
| Weeks 5-8 | Rewrite project summary, intellectual merit, evaluation, and management around the new hypotheses. | A reviewer should be able to state the three contributions in one sentence each. |
| Before submission | Ask one AI/CPS colleague and one transportation/infrastructure colleague to independently review the first 3 pages. | Both should agree on the primary scientific novelty and program fit. |

**Bottom line:** Sayeed should be a digital-twin co-developer, not merely a consumer of the simulator. The strongest role is to co-lead the AI-enabled twin layer (calibration, discrepancy learning, uncertainty, surrogates, scenario generation, validation), lead trustworthy learning under uncertainty, and co-own cascade-aware control. This is a substantial, executable ML/simulation contribution and strengthens the CPS-FR/ISP/RI case without forcing SNNs or event-camera technology.

---

## Sources and program notes

- **[P1]** Uploaded proposal: "CAIG: GeoAI Traffic Twin to Study the Impact of Sub-Autonomous, Autonomous, and Connected Vehicles in Midwest Corridors," 26 pages. Key passages include the three original thrusts, the statement that the RL methods are mature, the proposed evaluation, and project-management/team roles.
- **[N1]** NSF 25-530, Collaborations in Artificial Intelligence and Geosciences (CAIG): `https://www.nsf.gov/funding/opportunities/caig-collaborations-artificial-intelligence-geosciences/nsf25-530/solicitation`
- **[N2]** NSF Cyber-Physical System Foundations and Connected Communities (CPS) program page: `https://www.nsf.gov/funding/opportunities/cps-cyber-physical-system-foundations-connected-communities`
- **[N3]** NSF 25-543, CISE Future Computing Research (Future CoRe): `https://www.nsf.gov/funding/opportunities/future-core-computer-information-science-engineering-future-computing/nsf25-543/solicitation`
- **[N4]** NSF Robust Intelligence (RI): `https://www.nsf.gov/funding/opportunities/ri-robust-intelligence`
- **[N5]** NSF Infrastructure Systems and People (ISP): `https://www.nsf.gov/funding/opportunities/isp-infrastructure-systems-people`
- **[N6]** NSF Security, Privacy, and Trust in Cyberspace (SaTC 2.0): `https://www.nsf.gov/funding/opportunities/satc-20-security-privacy-trust-cyberspace`

Program dates and descriptions were checked against NSF webpages on August 13, 2026. Because NSF opportunities can change, the team should re-check the live program pages and contact cognizant program officers before final submission.
