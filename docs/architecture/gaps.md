# H2AI Gaps — Research and Engineering Agenda

Every item here is a falsifiable question with a concrete engineering path.  Each section
names the problem, the current state (what's implemented), and what would need to change
to close the gap.

For the research-state assessment of these items see [`research-state.md`](research-state.md).

---

## Empirical β₀ Calibration

**Problem**: The USL coherency cost `β₀ = 0.039` is a prior for AI-agent workloads, not
a measured value for any specific model family or task distribution.  N_max recommendations
carry the uncertainty of this prior; over- or under-sizing the ensemble wastes tokens or
degrades reliability.

**Current state**: `baseline_accuracy_proxy = 0.0` (disabled).  When set to a non-zero
value, it overrides the CG proxy for p_mean.  The `EnsembleCalibration::from_empirical()`
path is implemented and called when oracle data is available.

**What closes it**: A systematic methodology for fitting β₀ from a corpus of oracle-labelled
tasks.  `auto_baseline_eval = false` and `auto_baseline_eval_min_tasks = 50` gate the
automatic switch to the empirical basis.  Enabling `auto_baseline_eval` after accumulating
50 Tier 1 oracle tasks gives an empirical p_mean; the remaining question is fitting β₀
rather than only p_mean.

---

## Empirical ρ Measurement

**Problem**: The correlation proxy `ρ_mean = 1 − CG_mean` is a mathematical convenience,
not a measured value.  True error correlation between LLM agents sharing training data is
likely higher than this proxy suggests, which would lower the effective CJT gain.

**Current state**: `RhoEmaState` tracks per-adapter-pair Pearson score products with an
EMA (`α = 0.10`, initial prior 0.45).  Steady state requires 30 observations; until then
the proxy is used.

**What closes it**: Accumulate 30+ tasks in a deployment and verify that `rho_mean()` from
`RhoEmaState` gives stable, deployment-consistent estimates.  Compare CJT gain predictions
against oracle pass rates to check whether the measured ρ improves calibration accuracy.

---

## Thinking Loop Tuning

**Problem**: The thinking loop (`thinking_loop.enabled = false` by default) adds up to 5
pre-task brainstorm LLM calls.  Whether this improves downstream verification pass rates
enough to justify the latency and token cost has not been measured.

**Current state**: Fully implemented.  Config: `coverage_threshold = 0.75`,
`convergence_threshold = 0.90`, `max_archetypes = 4`.  The `ThinkingLoopCompletedEvent`
carries `coverage_score` and `iterations_run` for measurement.

**What closes it**: A/B comparison of identical task sets with and without the thinking loop,
measuring final `VerificationScoredEvent.score` distributions and token cost.

---

## Constraint Coherence Probe Coverage

**Problem**: Incoherent or contradictory constraints cause the MAPE-K loop to retry without
making progress — every proposal violates at least one constraint regardless of quality.
The constraint coherence probe (`gap_k1.enabled = false`) detects this but is disabled.

**Current state**: `GapK1Config` is implemented with pre-flight coherence probing
(`probe_runs = 5` LlmJudge calls per check), instability detection
(`instability_threshold = 0.10`), and automated spec repair (`auto_repair_enabled = false`).
The probe cache TTL is 24 h.

**What closes it**: Enable `gap_k1.enabled = true` on a staging deployment, monitor
`CalibrationChangepoint` and instability events, and verify that automated repair
(`auto_repair_enabled`) does not introduce constraint regressions before enabling it.

---

## Knowledge Gap Researcher Coverage

**Problem**: Some constraint checks require domain knowledge the LLM does not have (cold
checks).  These fail consistently across all retries, burning token budget without progress.
The researcher loop (`gap_i1.enabled = false`) can retrieve external knowledge but is disabled.

**Current state**: `GapI1Config` is implemented.  The researcher fires only on checks with
pass rate ≤ `cold_check_threshold = 0.0` (currently disabled by threshold).  Web search
integration (`[web_search]`) and distillation are wired through `GapResearchChain`.

**What closes it**: Enable `gap_i1.enabled = true` with `cold_check_threshold` set to a
meaningful threshold (e.g. 0.3), configure `[web_search]`, and measure the improvement in
cold-check pass rate vs researcher call cost.

---

## Tiered Early Exit

**Problem**: Some tasks reach a high-quality partial-pass solution after wave 1 that would
satisfy operators, but the MAPE-K loop continues retrying until `max_autonomic_retries` is
exhausted, wasting time and tokens.

**Current state**: `TieredExitConfig` (`tiered_exit.enabled = false`) implements exit when
`acceptance_score = 0.85` is met by a `quorum_fraction = 0.5` of proposals.

**What closes it**: Enable `tiered_exit.enabled = true` and measure the tradeoff between
early-exit task latency reduction and the score distribution of exited-early outputs
versus fully-retried outputs.

---

## Convergence Gate

**Problem**: When verified proposals converge on the same solution, additional retry waves
are wasteful.  The convergence gate (`convergence_gate.enabled = false`) can stop retrying
early but risks false convergence on locally optimal but globally suboptimal solutions.

**Current state**: Implemented.  `theta_converge = 0.87`, supermajority fractions for N=3
(0.67) and N≥4 (0.80), `score_floor = 0.80`.

**What closes it**: Enable the gate on a sample task set and verify that false-convergence
rate (tasks where the gate fired but a later retry would have produced a meaningfully higher
score) is acceptable.

---

## Per-Task Token Budget Enforcement

**Problem**: Tasks with large constraint corpora or deep retry loops can exhaust token
budgets silently.  The cost guard (`cost_guard.enabled = false`) enforces per-task budgets
but is disabled because the abort threshold needs deployment-specific tuning.

**Current state**: `CostGuardConfig` with `budget_tokens_per_task = 100 000`,
`budget_warning_fraction = 0.80`, `budget_abort_fraction = 1.00`.
`CostThresholdWarningEvent` and `BudgetExhaustedEvent` are wired into the engine.

**What closes it**: Enable in shadow mode (warning only, abort disabled) to measure actual
per-task token consumption, then set `budget_tokens_per_task` at a reasonable percentile
(e.g. P95) and enable abort.

---

## Persistent Reasoning Memory

**Problem**: The system discards all intermediate reasoning at task end.  Patterns that
appear across tasks — common failure modes, effective decompositions, constraint tension
archetypes — must be re-discovered on every run.

**Current state**: `ReasoningMemoryConfig` (`reasoning_memory.enabled = false`) implements
checkpoint writes (phases 1–4), induction cycles that distill cross-task patterns, and
retrieval boosts/penalties.  Storage uses the `H2AI_MEMORY_{tenant}` NATS KV bucket.
`InductionCycleCompletedEvent` is emitted after each batch.

**What closes it**: Enable on a long-running deployment with a stable task distribution.
Measure whether induction-derived archetype boosts improve first-wave verification pass
rates relative to the baseline without memory.

---

## Oracle Family Calibration Separation

**Problem**: Tasks from different oracle families (Syntactic for code, Semantic for factual
and reasoning, Human for human-evaluated tasks) likely have different calibration
characteristics.  A single shared calibration window mixes these distributions.

**Current state**: `OracleDomain::family()` maps domains to families.  The oracle gate
is opt-in (`oracle_gate.enabled = false`).  The calibration window is currently shared
per-tenant rather than per-family.

**What closes it**: Separate calibration windows per family (Syntactic/Semantic/Human),
track ECE per family, and verify that per-family p_mean patching improves overall ECE
compared to the pooled window.

---

## Constraint Ambiguity Detection

**Problem**: Ambiguous constraints (checks that different verifier personas interpret
differently) cause inconsistent verification scores across retries.  This is
indistinguishable from random noise without explicit detection.

**Current state**: `AmbiguityDetectionConfig` is wired into the engine and
`JudgePanelConfig.ambiguity_threshold = 2` specifies how many conflicting verdicts
constitute ambiguity.  Static scan seeding and weighted evidence accumulation are
implemented.  Repair routing additionally requires `gap_k1.enabled + gap_k1.auto_repair_enabled`.

**What closes it**: Measure the rate of `SpecAmbiguousSignal` fires in production, verify
that the triggered constraints are genuinely ambiguous (human review), and enable
`gap_k1.auto_repair_enabled` once the coherence probe has been validated.

---

## Plan-Awareness Probe

**Problem**: A generated plan may satisfy all constraint checks individually while
contradicting them in composite (e.g. a plan that correctly lists all required steps but
sequences them in a way that violates an implicit ordering constraint).

**Current state**: `AwarenessProbeConfig` (`awareness_probe.enabled = false`,
`mode = Shadow`) implements a batched judge call per wave that emits
`AwarenessProbeCompletedEvent`.  In Shadow mode it has no pipeline effect.

**What closes it**: Enable in Shadow mode and measure the rate of `CONTRADICTED` verdicts
on Hard constraints.  If the rate is significant and well-correlated with task failures,
switch to `mode = Active` which triggers one re-iteration per CONTRADICTED Hard constraint.

---

## Integration Wave for Constraint Oscillation

**Problem**: Sequential constraint repair can oscillate: fixing constraint A breaks
constraint B, and the next wave fixes B but breaks A.  This wastes all retry waves without
converging.

**Current state**: `IntegrationWaveConfig` (`enabled = true`, `plateau_threshold = 0.02`,
`min_retry_for_plateau = 2`) detects plateau (< 2% score improvement between waves) and
routes to a Branch-Solve-Merge integration wave.  The detection is enabled by default;
the integration wave itself requires additional wiring.

**What closes it**: Validate that the plateau detector fires specifically on oscillation
patterns (not on genuine hard tasks) by examining the `ConstraintFrontierEvent` and wave
scores for plateau-detected tasks.

---

## DPPM-MetaRefine Synthesis

**Problem**: Single-shot synthesis (`synthesis_enabled = true`) produces one candidate that
either passes or triggers a retry.  For tasks with multiple conflicting constraint clusters,
a parallel cluster-solving approach may converge more reliably.

**Current state**: `DPPMConfig` (`dppm.enabled = false`) implements parallel cluster solvers
(`max_parallel_solvers = 4`), merge-step retries (`merge_max_retries = 2`), and a
`MetaObserver` for oscillation detection (`meta_observer_enabled = true`).

**What closes it**: Enable on tasks that currently require 2+ MAPE-K retries and measure
whether DPPM reduces the `ZeroSurvivalEvent` rate compared to standard synthesis.

---

## Automatic Recalibration on Drift

**Problem**: When the BOCPD changepoint detector fires, the calibration coefficients are
stale but the system continues operating with them.  Manual recalibration requires an
operator to issue `POST /v1/calibrate`.

**Current state**: `auto_recalibrate_on_drift = false`.  The `CalibrationChangepoint`
NATS event is emitted, and `drift_staleness_ttl_secs = 3600` triggers a warning.

**What closes it**: Enable `auto_recalibrate_on_drift = true` on deployments where the
operator is confident that calibration is safe to run automatically (i.e. the adapter pool
is stable and the calibration cost is acceptable).  Monitor for spurious changepoints
triggering unnecessary recalibration.
