# From CAIG Decline to a Fundable Program
### Diagnosis, technical upgrades, and a six-month resubmission roadmap for the Central Indiana mixed-autonomy traffic digital twin

**Memo — Strategy & Technical Plan · August 2026**

Prepared by Sayeed Chowdhury, Ph.D. · Incoming Assistant Professor, Luddy School of Informatics, Computing, and Engineering, IU Indianapolis
For discussion with Prof. Mohammad Al Hasan and Prof. Aniruddha Banerjee · Based on the CAIG proposal "GeoAI Traffic Twin" and the program landscape as verified on August 13, 2026

---

## Executive summary

**Why the proposal was declined, in one sentence.** CAIG (NSF 25-530) funds AI that advances scientific understanding of the Earth system — the research portfolio of NSF's Geosciences Directorate — and a transportation digital twin, however geospatially sophisticated, answers transportation-engineering questions rather than Earth-system questions. In NSF's taxonomy, GIS is a data infrastructure, not a geoscience. The proposal's own central fit argument ("the usage of GIS as the core integrative layer makes this CAIG project genuinely co-equal in AI and Geosciences") is precisely the claim a GEO-led panel is trained to reject, which is fully consistent with a "not a good fit" decline.

**The strategic good news.** Nothing about the decline impugns the underlying assets. The INDOT partnership, the ATLAS 11-county lane-level network, two decades of TransModeler depth, and the complexity-science framing are genuinely differentiating — they were simply aimed at the wrong directorate. Repositioned as a learning-enabled cyber-physical system under environmental stress, the same project sits squarely inside NSF's new CISE Future CoRe solicitation (NSF 25-543, which absorbed the former CPS and Smart & Connected Communities programs; up to $1M over up to 4 years; proposals accepted anytime with target dates of September 10, 2026 and February 4, 2027), and inside FHWA's FY2026 Exploratory Advanced Research interest in data science for highway transportation.

**What I propose to contribute.** I would join as co-PI and own a Learning-Accelerated Simulation and Safe Multi-Agent Learning thrust, structured as five work packages: (WP1) graph world-model surrogates of the microsimulator; (WP2) simulation-based-inference calibration with uncertainty quantification; (WP3) safety-shielded cooperative multi-agent RL; (WP4) a learned emulator of percolation and cascade metrics; and (WP5) generative stress-scenario synthesis. These directly repair the four vulnerabilities most likely to sink a CISE resubmission — compute feasibility, validation, uncertainty, and safety — while adding the publishable AI novelty the current draft under-claims. Everything in WP1–WP5 runs on the standard deep-learning/RL stack I use daily; no exotic hardware, no dependence on my prior neuromorphic tooling.

**Recommended path.** Target the Future CoRe CPS program at the February 4, 2027 panel cycle (September 10 is available but leaves no room for pilot results); run a 10-week pilot this fall so the resubmission ships with preliminary evidence; and in parallel open two USDOT doors — the FHWA EAR FY2026 data-science solicitation and an INDOT-led SMART Stage 1 application.

### Five verified facts that shape the strategy (details and links in §6 and Appendix B)

1. CAIG's scope is "scientific understanding of the Earth system," with mandatory Geosciences Advancement / AI Advancement / Partnership subsections; its last posted deadline was Feb 4, 2026 and no next cycle is currently announced.
2. NSF consolidated the CISE core, CPS, and S&CC solicitations into Future CoRe (NSF 25-543): one project class up to $1M / 4 years, accepted anytime, target dates the 2nd Thursday of September and 1st Thursday of February.
3. Future CoRe caps each individual at two proposals as PI/co-PI/Senior Personnel per rolling 12 months across all its programs.
4. FHWA's EAR program posted an FY2026 solicitation focused on data-science applications in highway transportation (verify current dates on SAM.gov).
5. IU Indianapolis earned Carnegie R1 status in Feb 2025 — which, notably, makes us ineligible for CRII (it excludes R1-affiliated PIs), so the team resubmission is also my own fastest funding path.

---

## 1. What the proposal proposes (faithful one-paragraph-per-thrust summary)

The proposal builds a geospatially faithful digital twin of Central Indiana's road network — Marion County plus Hendricks, Hamilton, Johnson, Boone, and Morgan; I-65/I-70, arterials, collectors, and rural connectors — inside the TransModeler microsimulation platform, fusing ATLAS street centerlines, USGS LiDAR-derived grades and sight distances, INDOT real-time and historical traffic data, and ten years of snow/precipitation records.

- **Thrust I** constructs and calibrates the twin (Gaussian-process surrogate calibration against a 0%-CAV baseline) and implements V2V/V2I-enabled route guidance, freeway speed harmonization, and traffic-light optimal speed advisory.
- **Thrust II** populates the twin with three vehicle classes — calibrated human drivers (IDM/MOBIL/gap acceptance), rule-based SAE L2–3 vehicles, and SAE L4–5 CAVs controlled by multi-agent PPO plus Stackelberg / Nash-bargaining modules at merges and intersections — to test whether cooperative multi-objective learning beats selfish optimization on congestion and safety.
- **Thrust III** applies complexity science — phase transitions in network order parameters, percolation thresholds of the CAV communication graph, cascade dynamics, and Kuramoto-style synchronization — to locate critical penetration levels and vulnerable corridors, ending in a predictive framework and agency-facing tools (CAV Policy Evaluator dashboard, resilience maps) delivered with INDOT and IGIO.

Two PhD students, two new courses, and annual practitioner workshops round out the plan. This is a strong engineering core with real institutional assets; the sections below are about packaging, rigor, and venue, not about the vision.

---

## 2. Why CAIG said "not a good fit" — and what that implies

CAIG is run out of the Geosciences Directorate (with CISE and MPS as partners) and its synopsis is explicit: the program "seeks to advance the development and adoption of innovative AI methods to increase scientific understanding of the Earth system." Every proposal must argue three required subsections — Geosciences Advancement, AI Advancement, and Partnership — and the geoscience half must be a discovery contribution to atmospheric, ocean, Earth, or geospace science, evaluated by geoscientists. Read against that bar, three things doomed us regardless of technical quality:

- **The "Geo" in our GeoAI is geospatial, not geoscience.** Lane-level GIS, LiDAR grades, and land-use layers are data engineering for a transportation model. No Earth-system question is answered anywhere in the proposal; weather enters only as a forcing input to traffic, never as an object of scientific study.
- **No geoscientist among senior personnel.** The team is ML + GIS/transportation; the evaluators (IGIO) are GIS practitioners. A CAIG panel looks for an atmospheric/hydrologic/earth scientist with co-equal intellectual ownership and finds none.
- **The fit argument invited the rejection.** Asserting that GIS-as-integrative-layer makes the project "co-equal in AI and Geosciences" reads to a GEO panel as a category error, and likely triaged the proposal early — consistent with a decline on fit rather than a science critique.

**Implications.**
(a) Do not resubmit to CAIG in its current form.
(b) A CAIG-compliant version is a different project: e.g., AI for mesoscale winter precipitation and road-surface state estimation (RWIS, HRRR, radar assimilation) with the traffic twin demoted to an impact layer — viable only with an atmospheric-science co-PI, and moot for now since no next CAIG deadline is posted.
(c) The project's natural reviewers live in CISE (cyber-physical systems, multi-agent learning) and at USDOT/FHWA — so that is where we go.

---

## 3. Ten weaknesses a CISE panel would still flag — each with its fix

Fixing the venue is necessary but not sufficient. Below is the red-team read I would give any colleague, ordered by how likely each issue is to draw a fatal reviewer comment at CPS/Robust-Intelligence panels. Each fix points into the work packages of §5 or the cross-cutting plan of §5.6.

**W1. Compute feasibility of MARL at metro scale is unaddressed.**
Thrust II promises MAPPO training over "tens of millions of steps" with thousands of agents inside a commercial microsimulator, but names no compute resources and no acceleration strategy. Reviewers will estimate wall-clock and stop reading. *Fix:* WP1 world-model surrogate (target: two orders of magnitude faster policy training) plus a named facilities plan — IU's Big Red 200 GPUs, the Jetstream2 NSF cloud housed at IU for elastic CPU simulation farms, and a NAIRR Pilot allocation request.

**W2. The validation plan cannot support the headline claims.**
The twin is calibrated only at 0% CAV penetration, then the "reality gap" is "extrapolated" as penetration grows — an extrapolation with no ground truth and no external anchor. *Fix:* a three-tier validation ladder (§5.6): held-out-day replication at 0% penetration; reproduction of established mixed-autonomy results (e.g., the Stern et al. ring-road field experiment and the Flow benchmark findings the draft already cites) as go/no-go gates; and perception/communication degradation priors drawn from public winter AV datasets rather than assumed.

**W3. Calibration is point-estimate and dated; uncertainty never reaches the conclusions.**
GP-based surrogate calibration yields point estimates for dozens of context-dependent parameters, with identifiability unexamined and no uncertainty propagated into policy conclusions — risky when the deliverable is agency decision support. *Fix:* WP2 simulation-based inference (neural posterior estimation) with posterior-predictive checks, plus conformal prediction intervals on every dashboard-facing quantity.

**W4. Learned controllers have no engineered safety.**
DRL agents optimizing reward mixtures can and will crash; "safety" appears only inside reward terms and metrics. For deployment-adjacent claims, panels now expect constraints, not incentives. *Fix:* WP3 safety shields — friction-aware braking-envelope action filters (control-barrier-function style) that make hard violations impossible by construction, with shielded vs. unshielded ablations.

**W5. Open-science promises contradict the proprietary platform.**
The proposal pledges an open TransModeler–RL interface, open data, and open toolkits — but nothing is reproducible without a commercial license, and reviewers will say so. *Fix:* a dual-platform strategy (§5.6): develop at full fidelity in TransModeler, and release a validated open SUMO mirror of a benchmark subnetwork plus all learned models, making every paper's results independently checkable.

**W6. Hypotheses are not falsifiable as written; metrics are generic.**
Thrust II's central claims ("MARL can enable safe coexistence," "cooperation will outperform selfishness") carry no effect sizes, thresholds, or pre-registered metrics, and Thrust II's evaluation is "cumulative reward." *Fix:* the hypothesis table in §4 (H1–H5) with quantitative success criteria and ablation plans wired to each work package.

**W7. Thrust integration is asserted, not mechanized.**
The thrusts exchange "outputs" in prose, but no interfaces, data contracts, or feedback loops are specified — a classic "three projects stapled together" comment. *Fix:* the explicit dependency map in §5.6 (surrogate feeds control and complexity; the emulator guides scenario search; everything lands in one evaluated dashboard).

**W8. Novelty is positioned on well-trodden applications.**
Route guidance, GLOSA, and speed harmonization each have mature literatures (which the draft candidly reviews); a CISE panel discounts "we integrate known pieces" unless a crisp methods claim leads. *Fix:* lead with the methods — metro-scale world-model MARL, SBI-calibrated twins, learned complexity emulation, guided rare-event generation — and demote the three ITS applications to evaluation scenarios.

**W9. Scope vs. staffing reads over-committed.**
Two PhD students are asked to deliver a calibrated metro twin, novel MARL, complexity theory, open software, a dashboard, courses, and workshops. *Fix:* a third co-PI (me) with one additional student, and moving deployment-flavored deliverables (dashboard hardening, corridor pilots) onto USDOT vehicles (§6).

**W10. Polish and compliance erosion.**
Typos ("steet lights," "Prenetration," "reseach," "collaboratino"), inconsistent terminology (SAC vs. CAV; Transmodeler vs. TransModeler), a duplicated "Phase 4" in Thrust II, a truncated reference ([23]), a misplaced citation ([64] on query substitution cited for surrogate modeling), and a Yann LeCun name-drop — individually small, collectively they cost benefit-of-the-doubt on everything else. *Fix:* the copy-edit punch list in Appendix A plus a terminology freeze ("CAV" throughout, defined once).

---

## 4. The repositioned project

**Working title.** FROST-Twin: Foundations for Resilient Operations of Snow-affected Transportation — Learning-Accelerated Digital Twins for Safe Mixed-Autonomy Mobility. (Alternate, if the team wants a wink at the ML crowd: COLDSTART.)

**The one-sentence pitch:** mixed-autonomy traffic in winter weather is a learning-enabled cyber-physical system whose safety, efficiency, and resilience phase behavior we will make predictable — with calibrated uncertainty — at the scale of a real metropolitan network. Same twin, same partners, same complexity science; the scientific claims are now about learning-enabled CPS under environmental stress, which is exactly the CPS-FR remit inside Future CoRe ("intersection of computation, physical systems, and human interaction").

### 4.1 Thrust architecture (old → new)

| Thrust | Lead / co-lead | Keeps from the original | Adds (work packages, §5) |
|---|---|---|---|
| A. Twin & Inference | Banerjee / Chowdhury, Hasan | ATLAS + LiDAR + INDOT fusion; TransModeler build; V2V/V2I stack; GP calibration retained as baseline | WP2 simulation-based-inference calibration with full posteriors; conformal-UQ layer; open SUMO mirror of a benchmark subnetwork |
| B. Accelerated Safe Control | Chowdhury / Hasan | Three vehicle classes; MAPPO backbone; Stackelberg / Nash-bargaining modules at merges and intersections | WP1 graph world-model surrogate for fast training; WP3 safety shields with hard constraints; anchored validation ladder |
| C. Predictive Complexity & Resilience | Hasan / Chowdhury, Banerjee | Phase transitions, percolation thresholds, cascade metrics, synchronization analysis; agency-facing maps and dashboard | WP4 learned complexity emulator (minutes, not weeks, per what-if); WP5 generative stress-testing that hunts cascade-onset modes |

### 4.2 Pre-registered hypotheses with quantitative success criteria

This table is the single highest-leverage addition to the resubmission: it converts aspirations into falsifiable, panel-checkable claims and gives every work package a finish line.

| # | Hypothesis | Primary metric & threshold | WP |
|---|---|---|---|
| H1 | A graph world model trained on at most ~20k simulator rollouts supports policy learning with at least 100× wall-clock speedup and at most 10% return degradation when policies transfer back to the full simulator. | Wall-clock ratio; surrogate-to-simulator return gap; 15/30/60-min rollout error (speed/density MAPE) | WP1 |
| H2 | Neural posterior estimation yields calibrated parameter posteriors and beats GP point calibration on unseen days, including snow regimes. | Posterior-predictive coverage within ±5% of nominal at 80/95%; at least 20% lower held-out-day RMSE vs. GP baseline | WP2 |
| H3 | Shielded cooperative MARL coexists safely with calibrated human drivers and beats both human-only and selfish-MARL baselines in heavy snow at moderate penetration. | Hard-safety violations = 0 by construction; near-miss rate (TTC under 2 s per 1k veh-km) reduced; network delay reduced by at least 15% vs. human-only at 20% penetration | WP3 |
| H4 | Percolation thresholds and cascade-size statistics are predictable from topology, demand, weather, and penetration features — and transfer to counties never seen in training. | Threshold relative error at most 10%; CRPS on cascade sizes; zero-shot county-transfer degradation at most 15% | WP4 |
| H5 | Guided generative scenario search discovers qualitatively distinct cascade-onset failure modes far faster than space-filling sampling at matched plausibility. | At least 5× distinct failure modes per 1,000 simulations vs. Latin-hypercube baseline; plausibility score vs. historical extremes | WP5 |

---

## 5. My proposed contribution: five work packages I can execute

Design constraints I imposed on myself: every package (i) runs on the standard PyTorch / graph-learning / RL stack, (ii) needs only two things from the existing plan — programmatic access to the twin through the team's planned GISDK/Gym interface, and batch simulation capacity — and (iii) produces a standalone publishable result even if sister packages slip. Nothing depends on spiking networks, event cameras, or any speciality of my dissertation tooling; this is mainstream modern ML aimed at the proposal's pressure points. Milestones assume a 4-year award.

### WP1 — TwinBoost: graph world models as fast surrogates of the microsimulator

**Motivation.** Direct MARL inside TransModeler at metro scale is the proposal's feasibility cliff (W1). Model-based RL solved the analogous problem elsewhere: learn the environment, train in the learned model, verify in the real one.

**Approach.** A lane-graph message-passing network with a recurrent/attention temporal head predicts link speeds, densities, and queue states at 30–60 s resolution, conditioned on signal states, boundary demand, weather covariates, and CAV penetration; optional physics-informed losses (flow conservation, cell-transmission consistency) regularize long rollouts. Training data: 5–20k stratified one-hour TransModeler rollouts batch-generated on HPC. The surrogate then serves as (a) the meso-level training environment for WP3 and (b) the ensemble engine behind WP4/WP5 sweeps.

**Baselines / metrics.** Per-link LSTM, DCRNN / Graph-WaveNet-class forecasters, and direct-simulator RL wall-clock. Metrics: multi-step rollout MAPE, queue-onset timing F1, interval calibration, and — decisively — the policy-transfer gap of H1.

**Milestones / risks.** M6 pilot on the I-465 NE quadrant; M12 metro scale; M24 fully in the WP3 loop. Risks: compounding rollout error (mitigate: scheduled sampling, shorter-horizon composition); distribution shift at high penetration (mitigate: DAgger-style active resampling from the full simulator).

### WP2 — SBI-Twin: simulation-based inference for a calibrated, uncertainty-aware twin

**Motivation.** Agencies will act on this twin; point-calibrated parameters with no identifiability analysis (W3) are a liability. Simulation-based inference gives full posteriors over behavioral and environmental parameters at the cost of the same simulation budget the team already plans to spend.

**Approach.** Neural posterior estimation (sequential NPE via the open-source sbi toolkit) over grouped parameter vectors — car-following, lane-change aggressiveness, weather capacity multipliers, V2X latency/failure — conditioned on summary statistics of INDOT detector and probe data, stratified by weather regime; posterior-predictive checks on held-out days; a conformal-prediction wrapper turns every dashboard number into a calibrated interval. Priors on AV sensing range and detection degradation in snow are set from public winter AV datasets (CADC/Waterloo, Ithaca365/Cornell, Boreas/Toronto, WADS/Michigan Tech) instead of being assumed.

**Baselines / metrics.** The team's GP/Bayesian-optimization calibration (retained, and credited, as the baseline) and ABC rejection. Metrics: held-out-day RMSE on speeds/volumes/travel times; 80/95% interval coverage; identifiability diagnostics per parameter group (H2).

**Milestones / risks.** M9 clear-weather posterior; M15 snow regimes; M20 conformal layer live in the dashboard. Risk: high-dimensional posteriors (mitigate: sequential rounds + parameter grouping). Data governance: all inference runs on aggregate summaries, so raw INDOT feeds never leave their environment.

### WP3 — ShieldedFlow: safety-constrained cooperative MARL for mixed autonomy

**Motivation.** Keep the team's MAPPO + game-theory design — it is sound — but make safety structural (W4) and training tractable (W1).

**Approach.** MAPPO backbone with density-dependent cooperative reward mixing; an analytic safety shield filters every action through friction-aware stopping-distance and time-headway envelopes (weather-dependent friction coefficient µ), in the spirit of control-barrier-function filters, so hard violations are impossible by construction. Stackelberg merge and Nash-bargaining intersection modules from the original Thrust II are layered at critical zones. Policies train primarily in the WP1 surrogate and are fine-tuned and evaluated in TransModeler.

**Validation anchors.** Before any new claims: reproduce the single-AV ring-road wave-dampening result (Stern et al. field experiment) and Flow-benchmark bottleneck findings as go/no-go gates — cheap, credible, and it converts W2 from a weakness into a selling point.

**Baselines / metrics.** Human-only calibrated baseline, selfish MARL, unshielded cooperative MARL, rule-based CACC. Metrics: violations (must be zero), near-miss rate, delay, throughput, jerk/comfort, fairness (Gini of user delays), and robustness under 10–50% V2X packet loss and latency jitter (H3). Milestones: M12 anchors; M18 corridor scale; M30 full simulation matrix.

### WP4 — PercNet: a learned emulator of network complexity metrics

**Motivation.** Thrust III as written is post-hoc analysis of expensive simulation campaigns. Learning the map from (topology, demand, weather, penetration) to (percolation threshold, phase-transition location, cascade-size distribution) turns it into resilience nowcasting — the difference between a research finding and a planning tool.

**Approach.** A GNN / graph-transformer trained on WP1-accelerated ensembles predicts critical CAV penetration, the percolation curve, and cascade-size distribution parameters; attribution analysis (e.g., Shapley values, which Thrust II already planned) exposes which topological and demand features drive fragility, in planner language.

**Baselines / metrics.** Direct simulation (gold standard), classical structural predictors (betweenness, spectral gap). Metrics: threshold relative error, CRPS on cascade sizes, and — the headline — zero-shot transfer to held-out counties (H4). Milestones: M18 first emulator; M27 dashboard integration; M33 INDOT tabletop exercise using it live.

### WP5 — StormForge: generative stress-testing of the twin

**Motivation.** Random scenario sampling wastes the simulation budget on benign days. The proposal already cites TeraSim-style adversarial scenario generation; this package brings that idea to the network scale, tied to winter weather.

**Approach.** A conditional generative model (diffusion or normalizing flow) over scenario tuples — spatiotemporal storm fields consistent with NCEI climatology, incident placements/durations fit to INDOT logs, demand surges — with rare-event guidance: samples are tilted toward high predicted cascade size from WP4 under an explicit plausibility constraint (likelihood bound), giving adversarial-but-plausible winters.

**Baselines / metrics.** Latin-hypercube and historical-replay sampling at matched compute. Metrics: distinct cascade-onset modes discovered per 1,000 simulations (target: at least 5× baseline), plausibility score, coverage of historical extreme days (H5). Milestones: M21 unconditional generator; M27 guided search; M33 red-team report delivered to INDOT.

### 5.6 Cross-cutting plan (fixes W2, W5, W7 in one page of the resubmission)

- **Compute.** Big Red 200 (IU's GPU-equipped HPC) for training; Jetstream2 — the NSF cloud resource housed at IU, allocated through ACCESS — for elastic CPU simulation farms running containerized TransModeler workers; a NAIRR Pilot allocation request as supplement. Named machines, estimated GPU-hours, and a scaling argument go in the facilities section; this alone neutralizes W1's reviewer math.
- **Open science without license contradictions.** Develop at fidelity in TransModeler; release (i) a validated open SUMO mirror of one benchmark subnetwork, (ii) the trained world model, policies, and emulator, and (iii) "IndyWinterFlow," an anonymized detector-summary benchmark (subject to INDOT approval). Every paper becomes independently reproducible; the TransModeler–Gym interface still ships for licensed users.
- **Integration as data contracts.** One diagram in the resubmission: WP2 posteriors parameterize the twin; the twin generates WP1 training rollouts; WP1 accelerates WP3 policy learning and WP4/WP5 ensembles; WP4 guides WP5; all five report into a single evaluated dashboard reviewed annually by INDOT/IGIO. Each arrow is an API, not a sentence.
- **Validation ladder.** Tier 1: held-out-day replication at 0% penetration (their plan, kept). Tier 2: reproduction of published mixed-autonomy phenomena as gates. Tier 3: sensitivity and robustness analyses with WP2 uncertainty propagated end-to-end — conclusions reported as intervals, never points.

---

## 6. Where to submit: verified targets for the next six months

Deadlines below were checked against posted solicitations on August 13, 2026; items marked "verify" have live listings whose exact dates should be confirmed on the linked source (Appendix B) before we commit calendars. Note the Future CoRe participation cap: at most two proposals per person per rolling 12 months across all Future CoRe programs — so we choose our two shots deliberately.

| Target | Vehicle / window | Size | Fit & what must change |
|---|---|---|---|
| 1. NSF Future CoRe — CPS Foundations & Connected Communities (CPS-FR) — primary | NSF 25-543; accepted anytime; target dates Sep 10, 2026 and Feb 4, 2027 (recommended) | Up to $1M / up to 4 yrs | Learning-enabled CPS under environmental stress: sensing (V2X/RSU) + learned control + human drivers in closed loop. Reframe per §4; add hypothesis table, WP1–WP5, compute plan. Email cognizant CPS program directors a 1-page concept first. |
| 2. Same solicitation — Robust Intelligence (RI) — alternate routing | NSF 25-543; same windows | Up to $1M / up to 4 yrs | If the pitch leads with multi-agent learning + world models + UQ science and treats transportation as the testbed, RI is the cleaner panel. Decide after PD feedback; counts against the same 2-proposal cap. |
| 3. FHWA Exploratory Advanced Research (EAR), FY2026 | FY2026 solicitation on data-science applications in highway transportation — live listing; verify dates on SAM.gov | Contracts / cooperative agreements | Near-perfect domain fit for the twin + dashboard + INDOT partnership; deliverables re-angled to FHWA operations priorities (winter operations, incident management). The INDOT letter is gold here. |
| 4. USDOT SMART, Stage 1 (Planning & Prototyping) | Applicant must be a public-sector entity — INDOT or the City of Indianapolis leads, IU as research partner; FY22–26 authorization (final year) — verify current NOFO | Planning-grant scale | The deployment-flavored slice (CAV Policy Evaluator, corridor pilots) belongs here, not in the NSF budget. Open the conversation with INDOT now; six weeks of agency lead time is typical. |
| 5. NSF CIVIC Innovation Challenge (next cycle) — watch | Latest was NSF 24-534 (two-stage; Track A: climate/resilience); new cycle timing unannounced | $75k Stage 1, up to $1M Stage 2 | Winter-storm mobility resilience with a civic partner as true co-lead. Keep warm with the MPO; move only if a solicitation posts. |
| 6. CAIG resubmission — long shot, parked | NSF 25-530; last deadline Feb 4, 2026; no next cycle posted | 2–3 senior personnel, up to 3 yrs | Viable only as a redesigned Earth-system project (AI for winter road-weather / mesoscale precipitation, twin as impact layer) with an atmospheric-science co-PI. Revisit only if the program reopens. |

**Also on the radar, for completeness.**
(a) CRII is off the table for me: the solicitation excludes PIs at Carnegie R1 institutions, and IU Indianapolis was designated R1 in February 2025 — worth knowing before anyone suggests it.
(b) My NSF CAREER window opens July 2027 (outside this six-month plan), and the WP1/WP2 pilot doubles as its preliminary-results section — another reason to run the pilot now.
(c) NSF announced State and Regional AI Infrastructure Hubs (NSF 26-513; consortium-scale, $4–12M, November 4 deadline) on August 4, 2026 — not a vehicle for this project, but Luddy leadership should know Indiana could anchor a consortium, and our twin is exactly the kind of flagship use case such a hub cites.

---

## 7. Six-month roadmap (August 2026 → February 2027)

| When | Actions | Owner |
|---|---|---|
| Aug 2026 (now) | Team alignment meeting; pick CPS-FR vs. RI framing; send 1-page concept to cognizant NSF program directors; request refreshed INDOT/IGIO letters; I draft §4–§5 as proposal text; submit ACCESS request for Jetstream2/Big Red 200 allocations. | All / SC |
| Sep 2026 | Skip the Sep 10 target date unless PD feedback is unusually strong; instead launch the 10-week pilot: WP1 mini world model on the I-465 NE quadrant + WP2 clear-weather SBI on ~30 INDOT days; extend the INDOT data-use agreement to cover aggregate benchmark release; INDOT conversation on SMART Stage 1 lead. | SC (pilot), MAH/AB (agency) |
| Oct 2026 | Pilot results v1 (speedup + calibration-coverage figures); internal red-team with two external readers; prepare FHWA EAR response per the posted BAA instructions. | SC / all |
| Nov 2026 | Full 15-page draft: repositioned narrative, hypothesis table, WP1–WP5, pilot figures, compute + open-science plans; compliance and reference audit (Appendix A punch list). | All |
| Dec 2026 | Revision round; letters finalized; facilities, data-management, and mentoring documents; budget shaped to the $1M cap (§8). | All |
| Jan 2027 | Freeze and submit via Research.gov (Future CoRe accepts anytime) so the proposal is queued for the February panel cycle; EAR/SMART tracks proceed on their own clocks. | MAH (submitting PI) |
| Feb 4, 2027 | Future CoRe target date — proposal in review, carrying pilot evidence no competing team is likely to have. | — |

---

## 8. Proposed team structure, budget shape, and my role

- **Prof. Hasan (PI):** overall lead; complexity science (Thrust C lead); co-lead on multi-agent learning; submitting institution history (prior NSF award IIS-1909916 also keeps a future PFI-RP translation path open for the dashboard).
- **Prof. Banerjee (co-PI):** twin construction and GIS validation (Thrust A lead); TransModeler/GISDK interface; agency relationships (ATLAS, INDOT, IGIO); patent-backed network-reduction expertise feeding WP4 features.
- **Chowdhury (co-PI):** Thrust B lead and WP1–WP5 execution; one PhD student in deep learning / RL; modern training infrastructure (experiment tracking, containerized simulation workers, reproducibility discipline). I am asking for co-PI status, joint ownership of the methods papers, API access to the calibrated twin, and shared supervision credit — and I am offering to start the pilot before the ink is dry.

**Budget shapes under the $1M Future CoRe cap (indicative, not priced).**
- *Shape A (lean):* two PhD students (Thrusts A and B), one summer month per investigator per year, workshops trimmed to one; Thrust C staffed by the Thrust B student in years 3–4 once WP1 amortizes simulation cost.
- *Shape B (preferred):* three PhD students (one per thrust), half a summer month per investigator, with the dashboard hardening, practitioner workshops, and corridor-pilot line items moved onto the FHWA EAR / SMART tracks where they are a better fit anyway.

Either shape is honest about effort in a way the two-student original was not (W9).

---

## 9. Closing note

The CAIG outcome was a venue error, not a verdict on the science — and venue errors are the cheapest kind to fix. What the project needs to clear a CISE panel is exactly what I do: make the learning tractable (WP1), the calibration honest (WP2), the control safe (WP3), the complexity predictive (WP4), and the stress-testing sharp (WP5), all stated as falsifiable hypotheses with numbers attached. If the team will have me as co-PI, I will have the I-465 pilot running within two weeks of our first meeting, and pilot figures in hand before the November draft. I would welcome the chance to walk through this document together. — S.C.

---

## Appendix A. Copy-edit and compliance punch list for the existing draft

Concrete items found on a close read; fifteen minutes of fixes that buy disproportionate reviewer goodwill.

- "steet lights" → "street lights" (§1.2 Thrust I and §2); "Prenetration" → "Penetration" (Thrust I heading).
- "reseach" → "research" (Thrust III overview and §1.2); "collaboratino letter" → "collaboration letter" (§5); "deleop" → "develop" (§2.2); "Transportatino" → "Transportation" (§2.1); "followoing" → "following" (§4.2).
- Grammar in the Project Summary: "The project will uses novel economics…" and "Primary innovation is an CAV-ready digital twin"; §1.4 "The project is lead by" → "led by"; §10 "Co-PI Banerjee do not have" → "does not have."
- Terminology freeze: the Overview uses "SAC vehicles," the rest "CAV" — define "CAV" once and use it throughout; standardize "TransModeler" capitalization (currently mixed with "Transmodeler").
- Thrust II research plan announces four phases, then narrates "third phase," "final phase," and a second "Phase 4" — renumber and reconcile.
- Reference [23] (Goldman Sachs) is truncated mid-sentence ("forecast to comprise 10 URL…"); repair or replace.
- Reference [64] (Jones et al., "Generating query substitutions," WWW 2006) is cited as support for surrogate-model training — almost certainly a bibliography paste error; replace with an actual surrogate-modeling citation.
- Remove the "Yann LeCun's vision of physical AI" sentence (§1.3); make the physical-grounding argument on its own merits — name-drops read poorly in panel.
- Evaluator name appears as both "Dr. Steven Aldritch" and "Dr. Aldrich" (§1.4) — pick one spelling.
- Figure 1 reads as a composited raster; replace with an original vector diagram (panels increasingly comment on AI-generated-looking figures), and fix its caption ("interacting with AVs").

---

## Appendix B. Sources and starting points (verified Aug 13, 2026)

- NSF 25-543 — Future Computing Research (Future CoRe): target dates, $1M/4-yr project class, 2-proposal cap
  `nsf.gov/funding/opportunities/future-core-computer-information-science-engineering-future-computing/nsf25-543/solicitation`
- CPS Foundations & Connected Communities program page (CPS-FR / CPS-CIR tracks)
  `nsf.gov/funding/opportunities/cps-cyber-physical-system-foundations-connected-communities`
- NSF 25-530 — CAIG solicitation (scope: "scientific understanding of the Earth system"; last deadline Feb 4, 2026)
  `nsf.gov/funding/opportunities/caig-collaborations-artificial-intelligence-geosciences/nsf25-530/solicitation`
- FHWA Exploratory Advanced Research program (FY2026 data-science solicitation — confirm dates on SAM.gov)
  `highways.dot.gov/research/research-programs/exploratory-advanced-research`
- USDOT SMART grants program (Stage 1 planning grants; public-sector applicants)
  `transportation.gov/grants/SMART`
- NSF CIVIC Innovation Challenge (latest solicitation NSF 24-534; watch for next cycle)
  `nsf.gov/funding/opportunities/civic-civic-innovation-challenge`
- Jetstream2 (NSF cloud housed at IU; allocations via ACCESS) and IU Big Red 200 — facilities for the compute plan
  `jetstream-cloud.org` · `access-ci.org`
- NAIRR Pilot — national AI compute allocations
  `nairrpilot.org`
- sbi toolkit for simulation-based inference (WP2)
  `github.com/sbi-dev/sbi`
- Flow project — mixed-autonomy RL benchmarks used as WP3 validation anchors
  `flow-project.github.io`
- Winter AV datasets for WP2 priors: CADC (Waterloo), Ithaca365 (Cornell), Boreas (Toronto), WADS (Michigan Tech)
  `cadcd.uwaterloo.ca` · `ithaca365.cs.cornell.edu` · `boreas.utias.utoronto.ca`

Program deadlines and structures changed substantially during the 2025–26 NSF reorganization; every date above was re-verified against the posted solicitation text on August 13, 2026, but confirm on the official page before committing internal deadlines.
