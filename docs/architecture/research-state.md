# H2AI Research State

Critical self-assessment of what the system claims, where the mathematics is genuinely
defensible, where it is an honest engineering heuristic, and where empirical validation
is still missing.

This document is not a roadmap and does not track historical work.  For formulas see
[`math.md`](math.md).  For implementation details see [`architecture.md`](architecture.md).

---

## 1. What Is Implemented and Validated

### USL ensemble sizing

The Universal Scalability Law (USL) formula `X(N) = N/(1 + α(N−1) + β·N(N−1))` has
decades of empirical support in distributed systems.  Its application here is principled:
we use it to compute N_max, the point at which adding agents hurts rather than helps.
The bounded `β_eff = max(β₀ × (1 − CG), 1e−6)` form ensures the bound never diverges.

**Validation status**: the formula is mathematically correct.  The default constants
`α = 0.12` and `β₀ = 0.039` are engineering priors for AI-agent workloads, not
empirically fitted to any specific deployment.  Operators with oracle data can refine
them via `baseline_accuracy_proxy` and the `EnsembleCalibration::from_empirical()` path.

### Condorcet Jury Theorem (CJT) majority vote

The CJT guarantee `Q(N,p,ρ) = p + (Q_ind(N,p) − p) × (1 − ρ)` is a well-established
result in social choice theory.  The even-N tie-breaking, the `n_it_optimal` formula,
and the `RHO_UPPER_CLAMP = 0.99` guard are all correctly implemented in `sizing.rs`.

**Validation status**: the CJT assumptions (each agent independently draws from the
same competence distribution) are idealised.  Real LLM agents share training data,
making true independence unlikely.  The ρ-proxy `ρ_mean = 1 − CG_mean` and the empirical
ρ EMA tracker are honest workarounds for unmeasured correlation.  The 30-observation
CLT threshold for switching from proxy to EMA is a sound statistical heuristic.

### EigenCalibration (N_eff)

The participation-ratio formula `N_eff = (Σλ)²/Σλ²` applied to pairwise agreement
matrices gives a principled measure of effective pool diversity.  It is the standard
Herfindahl–Hirschman index applied to eigenvalues.

**Validation status**: the math is sound.  The choice of Hamming distance between
constraint satisfaction profiles as the agreement kernel is a reasonable engineering
approximation for boolean constraint evaluation.

### MAPE-K retry loop

The Monitor-Analyze-Plan-Execute-Knowledge loop is a standard autonomic computing
pattern.  The implementation (up to `max_autonomic_retries = 2` waves, with failure
mode detection and retry-context injection) is well-understood.

**Validation status**: implemented and exercised by the integration test suite.

### Oracle calibration (ECE, calibration basis)

The ECE formula, the Angelopoulos–Bates P90 index, the Bootstrap → Conformal basis
transition at n=30, and the FIFO window management are all standard calibration
methodology with clear references (Angelopoulos & Bates 2021; Lehmann & Romano 2005;
DiCiccio & Efron 1996).

**Validation status**: correctly implemented.  The 200-observation FIFO window and the
0.15 ECE alert threshold are engineering constants that may need adjustment for specific
deployment sizes and latency requirements.

### Grounding chain (fabrication detection)

`GroundingChecker` wraps either `LlmGroundingJudge` (when `grounding.enabled = true`)
or `HeuristicGroundingJudge` (fallback).  The LLM path extracts architectural nouns
from the spec to distinguish legitimate implementation sub-components from genuine
fabrications, reducing false positives without sacrificing recall.
(`CompositeGroundingJudge` is available to combine multiple judges but is not used in
the current engine wiring.)

**Validation status**: implemented in the SRANI → Universal Grounding refactoring.
Recall and precision on real tasks are not yet systematically measured.

---

## 2. Honest Heuristics

### CG-to-competence proxy

`p_mean = 0.5 + CG/2` and `ρ_mean = 1 − CG` are convenient proxies that give reasonable
priors when no oracle data is available.  They are not derived from first principles —
they assume a linear mapping between constraint-profile agreement and individual agent
accuracy that may not hold in practice.

### Thompson Sampling bandit for N selection

The three-phase bandit (exploration → ε-greedy → Thompson Sampling) is a standard
reinforcement-learning strategy for selecting ensemble size.  The warm Gaussian prior
centred on N_max_USL is a sensible initialisation.  Whether the bandit converges to the
true optimum for a given task distribution depends on the stationarity of that distribution.

### β₀ = 0.039 for AI agents

This constant was chosen as a calibration prior for "AI-agent workloads" without rigorous
measurement against any specific model family or task set.  It should be empirically
refitted using the oracle path once enough Tier 1 tasks have accumulated
(`auto_baseline_eval_min_tasks = 50`).

### τ-alignment and Talagrand adaptation

The exponential alignment kernel `exp(−3×|a−b|)` and the Talagrand KL τ-spread rule
are engineering heuristics for maintaining temperature diversity.  The learning rate
`η = 0.1` and floor `τ_min = 0.5` are not backed by a convergence proof; they are
reasonable bounds that keep the system from degenerating.

---

## 3. Features Implemented but Disabled by Default

The following subsystems are fully implemented and available but require explicit
opt-in because they involve additional LLM calls, increased latency, or carry
risk of false positives that need tuning per deployment:

| Subsystem | Config flag | Cost driver |
|-----------|-------------|-------------|
| Thinking loop | `thinking_loop.enabled` | Up to 5 × brainstorm LLM calls pre-task |
| Reasoning memory | `reasoning_memory.enabled` | Background induction cycles |
| Complexity routing | `complexity_routing.enabled` | One probe LLM call per task |
| Tiered early exit | `tiered_exit.enabled` | Exits early — trades quality for latency |
| Convergence gate | `convergence_gate.enabled` | Additional wave comparison logic |
| Token budget enforcement | `cost_guard.enabled` | Hard abort at budget limit |
| Knowledge gap researcher | `gap_i1.enabled` | Web search + distillation per cold check |
| Constraint coherence probe | `gap_k1.enabled` | 5 × LlmJudge calls per check at calibration |
| Plan-awareness probe | `awareness_probe.enabled` | Batched judge call per wave |
| DPPM-MetaRefine | `dppm.enabled` | Parallel cluster solvers in synthesis |
| Oracle gate | `oracle_gate.enabled` | Post-merge oracle round-trip |
| Leader election | `leader_enabled` | Per-wave leader diagnosis re-prompt |
| Auto drift recalibration | `auto_recalibrate_on_drift` | Full `POST /calibrate` on changepoint |
| NATS dispatch | `nats_dispatch_enabled` | External TaoAgent round-trip per slot |
| Sequential grafting | `sequential_grafting_enabled` | Up to `max_rounds` graft × verify cycles |

---

## 4. Open Research Questions

### Empirical β₀ fitting

The coherency cost constant `β₀` is a prior.  A systematic methodology for fitting it
from oracle data across a corpus of representative tasks does not yet exist in the
codebase.  Until then, N_max recommendations carry the uncertainty of this prior.

### Correlation structure of LLM ensembles

The CJT gain formula requires knowing ρ.  The empirical ρ-EMA tracker (`RhoEmaState`)
provides a running estimate, but 30 observations are required for steady state.  No
systematic study of how ρ varies across model families, prompt structures, or task types
has been conducted.

### Thinking loop coverage threshold

The `coverage_threshold = 0.75` and `convergence_threshold = 0.90` for the thinking loop
were chosen by engineering judgment.  Whether they lead to better downstream verification
pass rates than fixed-iteration runs has not been measured.

### Oracle family separation

Three oracle families (Syntactic, Semantic, Human) are defined and mapped via
`OracleDomain::family()`, but the oracle gate infrastructure is opt-in and disabled by
default.  The benefit of separating calibration windows per family versus a single
window has not been measured empirically.

### Constraint coherence and instability detection

The `gap_k1` coherence probe and instability detection (`instability_threshold = 0.10`
on Jaccard similarity) are implemented but disabled.  Whether automated spec repair
improves downstream task pass rates without introducing regressions is an open question.

### Integration wave convergence

The `IntegrationWaveConfig` plateau detector (`plateau_threshold = 0.02`) fires a
Branch-Solve-Merge integration wave when sequential repair has stalled.  Whether this
converges more reliably than additional unconstrained retries has not been measured.

### Bandit stationarity assumption

The Thompson Sampling bandit assumes a stationary reward distribution over N.  In
practice task difficulty and constraint corpus composition evolve; the soft-reset
mechanism (`bandit_soft_reset_decay = 0.3`) addresses adapter model changes but does
not account for task distribution drift.
