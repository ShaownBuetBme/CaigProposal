# Pilot Experiment Plan — SafeTwin / FROST-Twin Preliminary Evidence

**Purpose.** Both resubmission memos (`CAIG_Proposal_Improvement_and_Resubmission_Strategy.md`, `Sayeed_CAIG_Resubmission_Strategy_SafeTwin_DigitalTwin.md`) agree on one point: the CPS-FR resubmission is far stronger with **preliminary pilot results in hand** before the November draft. Neither memo's target compute (TransModeler license, INDOT live feeds, Big Red 200 / Jetstream2 allocations) is available outside the university, so this plan defines a **self-contained, open-data proxy pilot** that can run end-to-end in a single Kaggle notebook (free CPU/GPU tier) and still produce figures/numbers directly reusable in the proposal's preliminary-results section.

**Design principle.** Every experiment below is a *small-scale, literature-anchored stand-in* for one of the memos' work packages (WP1–WP5 / T1–T3, E1–E10), using a lightweight custom ring-road IDM traffic simulator in place of TransModeler. The ring-road setup is not an arbitrary substitute — it is the exact validation anchor both memos cite (**W2/WP3**: "reproduce the single-AV ring-road wave-dampening result (Stern et al. field experiment)... as go/no-go gates"). So this pilot doubles as that validation gate.

**Feasibility constraints self-imposed** (mirrors §5 design constraints in the FROST-Twin memo):
- Runs on standard PyTorch / scikit-learn / NumPy stack only — no TransModeler, no SUMO binary, no INDOT data, no GPU strictly required (Kaggle GPU used opportunistically for the RL/GNN stages).
- Total wall-clock target: **under 45 minutes** on a Kaggle CPU/GPU session, so it is cheap to re-run and iterate.
- Every experiment produces a standalone plot + metric table, usable as a proposal figure even if later experiments are cut for time.

---

## 0. Shared substrate: ring-road IDM microsimulator

A single-lane circular road with `N` vehicles governed by the Intelligent Driver Model (IDM) for car-following, optionally with one or more "CAV" agents whose acceleration is set by a learned controller instead of IDM. This reproduces the classic Sugiyama/Stern et al. finding: below a critical density the flow is stable; above it, spontaneous stop-and-go waves emerge — the same phase-transition phenomenon Thrust III / T3 of both memos wants to predict and control at metro scale.

Parameters exposed for calibration/UQ experiments: desired time headway `T`, max acceleration `a`, comfortable deceleration `b`, jam spacing `s0`, desired speed `v0`. Weather/friction stress is injected as a multiplicative penalty on `a`/`b` (a crude analogue of the memos' weather-dependent friction coefficient µ).

---

## 1. Experiment E0 — Validation anchor: reproduce the ring-road phase transition

**Maps to:** FROST-Twin W2 fix / WP3 "validation anchors"; SafeTwin E5 (distribution shift precursor).

**Question.** Does the ring-road simulator reproduce the known critical-density onset of stop-and-go waves (go/no-go gate before any new claims are trusted)?

**Method.** Sweep vehicle count `N` from sparse to dense on a fixed ring length; run each density for multiple seeds; measure the standard deviation of vehicle velocities over the last portion of each run as the wave-amplitude order parameter.

**Metrics.** Velocity-std vs. density curve; critical density at which std departs from ~0; qualitative match to the literature's phase transition shape.

**Output artifact.** `fig_e0_phase_transition.png` + `e0_results.csv`.

---

## 2. Experiment E1 (WP1 proxy) — Learned surrogate ("TwinBoost" mini)

**Maps to:** FROST-Twin H1 / WP1; SafeTwin E2 (surrogate half).

**Question.** Can a small sequence model trained on a modest number of simulated rollouts predict multi-step ring-road speed/position evolution fast enough and accurately enough to stand in for the full simulator?

**Method.** Generate 300–1000 short rollouts (varying density and IDM parameters) as training data for a GRU/MLP surrogate that takes the current per-vehicle state window and predicts the next-`k`-step speed trajectory. Compare against a per-vehicle LSTM baseline and against direct-simulator wall-clock.

**Metrics (mirrors H1's thresholds at pilot scale).**
- Rollout MAPE at 3 short horizons.
- Wall-clock speedup ratio (surrogate inference vs. running the IDM simulator for the same horizon).
- Return/behavior-gap: downstream density-std order parameter computed from surrogate rollouts vs. true simulator rollouts.

**Output artifact.** `fig_e1_surrogate_accuracy.png`, `fig_e1_speedup.png`, `e1_results.csv`.

---

## 3. Experiment E2 (WP2 proxy) — Uncertainty-aware calibration ("SBI-Twin mini")

**Maps to:** FROST-Twin H2 / WP2; SafeTwin E1.

**Question.** Does an amortized, uncertainty-aware calibration approach recover IDM parameters with well-calibrated posteriors, and does it beat a point-estimate (grid-search / least-squares) baseline on held-out rollouts, including a synthetic "snow" (low-friction) regime?

**Method.** Simulate rollouts under randomly sampled ground-truth IDM parameter vectors (clear and low-friction regimes) → summary statistics (mean speed, flow, std) as pseudo-detector observations. Train a small ensemble/MC-dropout regressor to map summary stats → parameter posterior (mean + variance), i.e., a lightweight neural-posterior-estimation stand-in for the `sbi` toolkit. Compare against grid-search point calibration.

**Metrics (mirrors H2's thresholds at pilot scale).**
- Held-out RMSE of recovered parameters, ensemble/NPE-mini vs. grid-search baseline.
- Posterior-predictive coverage at 80/95% nominal levels.
- Split results by regime (clear vs. low-friction) to show snow-regime behavior explicitly.

**Output artifact.** `fig_e2_calibration_coverage.png`, `fig_e2_reality_gap.png`, `e2_results.csv`.

---

## 4. Experiment E3 (WP3 proxy) — Belief-state robustness under observation dropout

**Maps to:** FROST-Twin H3 / WP3 (safety-shield spirit, simplified); SafeTwin T2 / E3.

**Question.** Does a small recurrent "belief" controller that tracks time-since-last-observation degrade more gracefully under missing/noisy state than a privileged-state controller, when one vehicle is CAV-controlled?

**Method.** Train a lightweight one-agent policy (behavior-cloned or reward-shaped heuristic learner — kept small enough to train in minutes on CPU) to control the single CAV's acceleration to dampen the ring-road wave, first with full state, then with 0–50% randomly dropped observations. Compare (a) privileged/full-state controller, (b) naive last-value-hold controller, (c) GRU belief-encoder controller with masks/time-since-update, each evaluated at increasing dropout.

**Metrics (mirrors H3 in miniature).**
- Wave-damping performance (velocity-std reduction vs. no-CAV baseline) at each dropout level.
- "Graceful degradation slope": rate of performance loss per +10% dropout, belief model vs. baselines.
- Hard-safety proxy: minimum time-headway violations (count), each condition — analogue of the WP3 "must be zero" shield criterion, reported as a rate rather than a formal guarantee (per the SafeTwin memo's "do not overclaim theory" caution).

**Output artifact.** `fig_e3_degradation_curves.png`, `e3_results.csv`.

---

## 5. Experiment E4 (WP4 proxy) — Early cascade/wave-onset prediction

**Maps to:** FROST-Twin H4 / WP4 ("PercNet mini"); SafeTwin E8.

**Question.** Can a simple model predict, from an early partial window of a rollout, whether a stop-and-go wave (cascade proxy) will emerge later in that run — and does it transfer to densities/parameter regimes not seen in training?

**Method.** Label each simulated rollout as wave/no-wave using the E0 order parameter computed on the full run; train a classifier (gradient-boosted trees / small MLP) on only the first 20–30% of each rollout's summary features (density, early speed variance, parameter regime) to predict the eventual label. Hold out an entire density band and an entire parameter regime as a simulated "unseen county" transfer test.

**Metrics (mirrors H4 in miniature).**
- AUROC/AUPRC for wave-onset prediction from partial state.
- Threshold (critical density) relative error vs. the true E0 curve.
- Zero-shot degradation on the held-out density band / parameter regime.

**Output artifact.** `fig_e4_onset_auroc.png`, `fig_e4_transfer.png`, `e4_results.csv`.

---

## 6. Experiment E5 (WP5 proxy) — Guided vs. random scenario search

**Maps to:** FROST-Twin H5 / WP5 ("StormForge mini"); SafeTwin E6.

**Question.** Does a simple uncertainty/risk-guided sampler discover wave-onset ("failure") scenarios more efficiently than random/Latin-hypercube sampling of the same parameter space, at matched simulation budget?

**Method.** Define a small parameter space (density, IDM aggressiveness, friction penalty). Compare (a) Latin-hypercube sampling and (b) a simple guided sampler that uses the E4 classifier's predicted failure probability to bias the next batch of simulations toward the decision boundary, at matched total simulation budget.

**Metrics (mirrors H5 in miniature).**
- Number of distinct failure (wave-onset) scenarios discovered per fixed number of full simulator runs, guided vs. random.
- Coverage of the true critical-density boundary (from E0) by discovered failure points.

**Output artifact.** `fig_e5_discovery_efficiency.png`, `e5_results.csv`.

---

## 7. Deliverables and mapping back to the proposal

| Pilot experiment | Proposal hypothesis it de-risks | Figure(s) for proposal |
|---|---|---|
| E0 | Validation anchor for W2/WP3 | Phase-transition curve |
| E1 | H1 (WP1 TwinBoost) | Surrogate accuracy + speedup |
| E2 | H2 (WP2 SBI-Twin) | Calibration coverage + reality-gap |
| E3 | H3 (WP3 ShieldedFlow, simplified) | Degradation-under-dropout curves |
| E4 | H4 (WP4 PercNet) | Onset AUROC + zero-shot transfer |
| E5 | H5 (WP5 StormForge) | Discovery-efficiency curve |

**What this pilot does NOT claim.** It does not use TransModeler, real INDOT/ATLAS data, or a true multi-agent RL stack at scale, and the "cascade" here is a single-lane wave rather than a network-level spillback. Per the SafeTwin memo's explicit caution ("do not overclaim theory... do not claim realism at unobserved CAV penetrations"), this pilot is presented as a **methodology-validation proxy at toy scale**, establishing that each WP's pipeline runs end-to-end and behaves in the qualitatively expected direction — exactly the role "preliminary studies" play in §9 of the SafeTwin memo — not as a substitute for the full metro-scale study.

**Execution.** All five experiments (plus E0) are implemented in one Kaggle notebook (`safetwin_pilot_experiments.ipynb`), run top-to-bottom on a single Kaggle kernel session, with each experiment's numeric results and figures saved to `/kaggle/working/` for download and inclusion in `experiment_report.md`.
