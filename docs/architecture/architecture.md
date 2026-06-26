# H2AI Architecture

H2AI Control Plane is a Rust runtime that coordinates pools of LLM adapters as an *adversarial committee*: independent generators, an independent verifier, and an independent auditor produce a resolved output that is more reliable than any single adapter. The runtime treats this committee as a physical system — an ensemble whose throughput, diversity, and quality are computable, calibrated, and bounded.

This document is the system-level map: phases, components, wire protocol, and enterprise deployment. The math is in [`math.md`](math.md). The HTTP/event/config surface is in [`reference.md`](reference.md). Operational details are in [`operations.md`](operations.md). Open questions are in [`research-state.md`](research-state.md).

---

## The Epistemological Architecture

H2AI is an **epistemic control plane**. Its job is not to run LLM inference — it is to coordinate the acquisition, validation, and grounding of knowledge about a problem. The output of a successful task is not a string; it is a belief that has survived four nested epistemological tests:

| Loop | Scope | Mechanism | Stops when |
|------|-------|-----------|------------|
| **TAO** (Thought–Action–Observation) | Within one agent | Iterative reasoning with `verify_pattern` check | Agent produces output matching the pattern, or `tao.max_turns` reached |
| **MAPE-K** (Monitor–Analyse–Plan–Execute) | Across the committee | Wave loop: generate → verify → audit → merge; ZeroSurvival → topology repair → retry | All proposals pass audit and merge; `decide()` returns `Return` |
| **Calibration** | Across tasks | USL fitting, CG measurement, confidence intervals | Meta-beliefs about agent quality are stable — `(α, β, CG)` CI widths drop below precision threshold |
| **Oracle / Grounding** | Across reality | HITL approval gate, automated oracle, GroundingChecker post-merge | Load-bearing claims verified against external truth or human approval |

**Why four loops?** Each operates at a different time-scale and tests a different epistemic property:

- The TAO loop tests **completeness** within a single reasoning chain.
- The MAPE-K loop tests **coherence** across the committee: do surviving beliefs form a consistent, constraint-compliant set?
- The calibration loop tests **meta-accuracy**: are the system's beliefs about its own agents correct? Wrong priors produce wrong `N_max`.
- The oracle/grounding loop tests **grounding**: does the coherent belief set correspond to something true in the world?

A system with only the first two loops is a sophisticated coherence engine — capable of producing internally consistent outputs that are confidently wrong. Calibration corrects the system's self-model over time; the oracle and grounding stages gate outputs on external truth verification.

---

## System Components

The control plane is organized into five primary crates:

| Crate | Role |
|-------|------|
| `h2ai-api` | REST + SSE gateway (axum). Submits tasks, serves event streams, hosts HITL approval endpoint |
| `h2ai-orchestrator` | `ExecutionEngine`: full MAPE-K wave loop, thinking loop, verification, merge, grounding checker, epistemic quality stage |
| `h2ai-autonomic` | Calibration harness, drift monitor (DDM + BOCPD), ceiling detector, epistemic diagnostics |
| `h2ai-config` | `H2AIConfig` structs (all thresholds/flags live here), prompt string constants in `prompts.rs` |
| `h2ai-types` | Shared event types: `H2AIEvent` enum, `MergeResolvedEvent`, `VerificationScoredEvent`, `TaskFailedEvent`, `ZeroSurvivalEvent`, etc. |

NATS JetStream is the durable event log, KV store, and (optionally) agent messaging bus. Every state transition is published to NATS and is replayable.

---

## Task Lifecycle

### Phase sequence overview

```
Pre-loop (once per task, in engine.rs):
  1. ORCA conformal margin adjustment
  2. Reasoning checkpoint bootstrap (if reasoning_memory.enabled)
  3. Signal subscription (if HITL or signal_wave_window_ms > 0)
  4. Conflict-beta accumulator load (if conflict_beta.enabled)
  5. Phase: bootstrap — assembles system context
  6. Phase: complexity — assess_task_complexity(); emits TaskComplexityAssessedEvent
  7. Phase: domain_coverage — pure check; may produce DiversityGuardDegradedEvent
  8. Complexity probe (if complexity_routing.enabled) — emits H2AIEvent::ComplexityProbe

Wave loop (0 to max_autonomic_retries):
  - Deadline check → DeadlineExceeded
  - TEE N-escalation
  - AgentDropout N-reduction (on retry ≥ 2 if last_wave_n_eff < dropout_threshold)
  - pipeline.run() — full multi-phase wave
  - controller.observe() — aggregates wave events
  - Violation frequency accumulation from BranchPrunedEvents
  - OOM guard (reads RSS every check_interval_waves; emits BudgetExhausted if exceeded)
  - Token budget charge; emits H2AIEvent::CostThresholdWarning if budget_warning_fraction crossed
  - Epistemic Leader Election (if leader_enabled)
  - Conflict-rate write — ConflictRateAccumulator to NATS
  - Reasoning checkpoint write — WaveCompleted(retry_count)
  - WaveContinue window (if signal_wave_window_ms > 0)
  - Gap I1 research — controller.run_gap_i1_research()
  - controller.decide() → Return / Retry / Fail / SpecAmbiguous / ComplexityOverflow
  - Pending ambiguity events published to NATS

On Return (success path):
  - Induction store record() (background)
  - record_success() on induction scheduler (background)
  - HITL gate (if enabled AND not oracle task AND require_approval OR q < threshold):
      Emits H2AIEvent::PendingApproval → waits for NATS signal or timeout
      Emits H2AIEvent::ApprovalResolved; on reject → HitlRejected error
  - Reasoning checkpoint written as Resolved
  - Signal consumer deleted
  - consensus_agreement_rate computed
  - Epistemic quality stage (if epistemic_quality.enabled):
      gap extraction, coherence check, MicroExplorerResolver recovery loop,
      ProvenanceMap build, optional re-verification, output rendering
  - Emits H2AIEvent::TieredExit if TEE gate fired
  - Emits H2AIEvent::BudgetExhausted if budget exhausted
  - Emits H2AIEvent::ConvergenceGate if convergence gate fired
  - Returns Ok(EngineOutput)
```

### Event publication order (post_run in task_pipeline.rs)

After the engine returns, `task_pipeline.rs::post_run()` publishes all wave events to NATS in this exact sequence via `publish_event_seq`:

| # | Event | Condition |
|---|-------|-----------|
| 1 | `TaskComplexityAssessed` | Always first |
| 2 | `ConstraintFrontier` | Only if `frontier_event.is_some()` |
| 3 | `VerificationScored` | One per `verification_events` entry |
| 4 | `ProposalFailed` | One per `failed_proposals` |
| 5 | `BranchPruned` | One per `pruned_events` |
| 6 | `SelectionResolved` | Single event |
| 7 | `TaskAttribution` | Always |
| 8 | `GenerationKnowledge` | Always |
| 9 | `VerifierComparison` | Only if `measure_verifier_ab` |
| 10 | `ShadowAudit` | Only if shadow context set |
| 11 | `CorrelatedEnsemble` | One per `correlated_warnings` |
| 12 | `ResearcherGrounding` | One per `researcher_grounding_events` |
| 13 | `DiversityGuardDegraded` | Only if `diversity_degraded_event.is_some()` |
| 14 | `CoherenceIncomplete` | Only if coherence state not closed |
| 15 | `LeaderElected` | One per `leader_elected_events` |
| 16 | `SocraticDiagnosis` | One per `socratic_diagnosis_events` |
| 17 | `ProvenanceRecorded` | Before `MergeResolved`; only if `provenance_map.is_some()` |
| 18 | `MergeResolved` | Terminal success event containing `resolved_output` |

After `MergeResolved` (background/async): OPRO trigger (if `j_eff.is_some()`), oracle dispatch (if `oracle_spec.is_some()`), `ctx.store.mark_resolved(task_id)`, violation feedback to knowledge provider, skill node persist to NATS KV, drift monitor feed (`consensus_agreement_rate`; may log `DriftEvent`), induction distillation trigger (every `induction_batch_size` resolved tasks), checkpoint GC (`delete_task_checkpoint`).

### Events published during engine execution (not in post_run)

These events are published to NATS as they occur, before `post_run()` runs:

- `H2AIEvent::CostThresholdWarning`
- `H2AIEvent::ComplexityProbe`
- `H2AIEvent::BudgetExhausted`
- `H2AIEvent::ConvergenceGate`
- `H2AIEvent::TieredExit`
- `H2AIEvent::ComplexityCeilingDetected`
- `H2AIEvent::PendingApproval`
- `H2AIEvent::ApprovalResolved`
- `H2AIEvent::ConstraintAmbiguityDetected`

---

## Thinking Loop (Phase −1)

When `thinking_loop.enabled = true` (default: `false`), a coverage-convergence brainstorm runs before the engine's pre-loop phases.

**Iteration structure.** The loop runs up to `cfg.max_iterations` iterations. On iteration 0, `max_archetypes` archetypes are instantiated. On subsequent iterations, the count scales by `1 - coverage_score` — proportional to how much coverage remains.

**Tension seeding.** On iteration 0 only, trigram Jaccard ≥ 0.05 comparisons select the top-3 matching tensions to inject into the next archetype selection prompt.

**Archetype selection.** `execute_chain()` at τ=0.2 selects archetypes; a markdown parser splits on `## Archetype` headers. The LLM self-reports which constraints its archetype targets via the `**Focus Constraints:**` field in the `THINKING_ARCHETYPE_MD_ITER1`/`THINKING_ARCHETYPE_MD_ITERN` prompt templates.

**Per-constraint coverage guarantee.** After `select_archetypes()` returns, `find_uncovered_constraints(archetypes, constraint_ids)` identifies constraint IDs with no dedicated archetype (empty `focus_constraints` covers nothing). For each uncovered constraint, `synthesize_coverage_archetype(cid)` constructs a specialist from the corpus description (falls back to generic text when absent). Every constraint gets at least one dedicated archetype before brainstorming begins.

**Prior boost/penalty.** Up to `max_archetype_boost = 0.15` when `net_confidence > 0.6` AND domain overlap; penalty up to `max_archetype_penalty = 0.20` when `avoid_for_tags` overlaps.

**Temperature scheduling.** Linear decay from `tau_max` to `tau_min` across iterations.

**Brainstorm and synthesis.** All archetypes run in parallel via `join_all`. Synthesis uses `tournament_merge()` at τ=0.3.

**Convergence.** Early stop when `coverage_score ≥ cfg.coverage_threshold` AND `prev_similarity ≥ cfg.convergence_threshold` AND LLM quality gate pass.

**Output.** Emits `ThinkingLoopCompletedEvent` after all iterations. Produces `ThinkingReport { shared_understanding, tensions, coverage_score, iteration, prev_similarity, retrieved_node_ids, skill_nodes_used, archetypes }`. The `shared_understanding` string is injected as `{thinking_context}` into downstream prompts. The final `coverage_score` is used as the `thinking_coverage_score` for the `j_eff_min` dynamic threshold.

---

## MAPE-K Control Loop

### MapeKController key fields

`conflict_graph`, `binary_checks`, `global_best_proposal: Option<(f64, String)>`, `global_best_constraint_reasons: HashMap<String, String>`, `compliance_score_history`, `violation_freq: HashMap<String, u32>`, `bypassed_verifier_constraints`, `last_wave_n_eff`, `tokens_used`, `tried_topologies`, `adapter_rotation_offset`, `mode_collapse_count`

### decide() logic

The `MapeKController::decide()` function maps each wave outcome to a decision via these branches (evaluated in order):

1. **SpecAmbiguous** (if `gap_k1.enabled`): Low Jaccard verifier reasons across last two waves + ambiguity scorecard → `MapeKDecision::SpecAmbiguous`

2. **Frozen verifier detection** (if `verifier_freeze.enabled` AND `retry_count >= min_waves_to_detect`): adds constraint IDs to `bypassed_verifier_constraints`

3. **`PipelineOutcome::Resolved`**:
   - Budget exhaustion check → `budget_exhausted` flag
   - Convergence gate check
   - TEE acceptance gate: if `k_accepted < k_required` AND not budget-exhausted AND more retries → `Retry`
   - Otherwise → `Return(finalize(merge_out))`

4. **`PipelineOutcome::Fatal`** → `Fail`

5. **`PipelineOutcome::EarlyExit`** → `handle_exit_reason()`:
   - Probe-based routing (if `complexity_routing.enabled`): `probe.complexity ≥ hitl_threshold` → `ComplexityOverflow { graft_first: false }`; `≥ decompose_threshold` AND `retry_count >= min_retries_before_graft` AND `corpus_viable` → `ComplexityOverflow { graft_first: true }`
   - `MultiplicationFailed` → `RetryPolicy::decide()` → `apply_retry_action()`
   - `DiversityFailed` (ModeCollapse) → same
   - `ZeroSurvival`: `FailureMode` mutations → integration wave check → intra-retry ceiling detector → retroactive induction → `RetryPolicy::decide()`
   - `HallucinationDetected` → sets `retry_context`, returns `Retry`
   - `OraclePostSelectionBlocked` → rotates adapter, returns `Retry`
   - `OracleBlocked` → `Fail`

### Ceiling detector (ceiling_detector.rs)

The intra-retry ceiling detector fires `ComplexityOverflow` when ≥ 2 of 3 signals cross threshold, gated by `complexity_routing.intra_retry.enabled` and `retry_count >= min_retry_count_for_detection`:

1. **Entropy signal**: Shannon entropy H of constraint-failure frequency distribution; normalized by `ln(n_unique_constraints)`; fires when `H < entropy_threshold` (peaked failure pattern)
2. **Retry slope**: `slope = (score[n-1] - score[n-2]) / score[n-2]`; fires when `slope < retry_slope_threshold` (stall)
3. **N_eff × CG product**: fires when `n_eff * cg_mean < n_eff_cg_product_threshold`

Config flag: `H2AIConfig.complexity_routing.intra_retry.enabled` (default: `false`).

### Retry state projected via PipelineParams

`MapeKController::params()` produces an immutable snapshot before each wave:

| Field | Purpose |
|-------|---------|
| `optimizer` | Agent count and merge thresholds |
| `force_topology` | Topology override from previous wave failure |
| `tau_reduction_factor` | Accumulated τ-reduction multiplier across retries |
| `tau_spread_factor` | τ-spread expansion from Talagrand feedback |
| `adapter_rotation_offset` | Round-robin offset for adapter assignment |
| `retry_context` | Constraint-feedback hint text from `RetryPolicy` |
| `tao_config` | Per-turn TAO configuration |
| `verification_config` | Verification gate thresholds |
| `pending_tombstone` | Constraint tombstone injected at topology phase on retry |
| `leader_context` | `Option<LeaderContextSnapshot>` — Krum-elected leader's prior proposal, Socratic question, per-slot constraint aspect assignment; `None` when `leader_enabled = false` or first wave |

---

## Grounding Chain (grounding_chain.rs)

The grounding chain is used in two contexts: C1 hallucination grounding (reactive during generation) and gap I1 research.

**Provider chain (in order):**

1. **SpecAnchorGrounder** (always, index 0) — pure fn, no I/O; extracts arch nouns from `task_description`, filters fabricated entities, returns spec-defined alternatives. Source: `SpecAnchor`.

2. **Tier provider** (index 1+):
   - **LlmResearcherGrounder** — LLM call with `GROUNDING_RESEARCHER` prompts at τ=0.3; returns `{alternatives, statement, implied}` as JSON. Source: `LlmResearcher`.
   - **WebSearchGrounder** — 2–3 DDG queries, concatenates non-empty results. Source: `WebSearch`.

3. **Distillation** (optional) — LLM call with `GROUNDING_DISTILL` prompts at τ=0.2 when `distill_enabled AND source == WebSearch AND len >= compress_threshold`.

---

## Verification / Judge Panel (verification.rs)

### Single-variant path: run()

- Per-constraint parallel evaluation. Eval cache: similarity ≥ 0.85 reuses score.
- Hard constraints: `consensus_passes` LLM passes (averaged).
- Binary-check constraints: τ=0.0 (greedy), appends `CHECK_EVIDENCE_FORMAT_INSTRUCTION`.
- `score_from_verdicts()`: binary check verdicts → `present_checks / n_checks`; else LLM float.
- Hard gate: all `r.hard_passes_scaled(ct_scale)` must pass; failure → overall = 0.0.
- Empty corpus: falls back to `COT_RUBRIC` holistic scoring.

### Multi-variant path: run_with_panel()

- Per proposal × per constraint: fires all panel variants in parallel.
- `aggregate_votes(votes_pass, votes_fail, diversity_kind, quorum_fraction)` → `Pass` / `Fail` / `Uncertain`:
  - Pass: `avg_score`
  - Fail (hard): `ht - 0.01`; sets `hard_fail = true`
  - Uncertain: `avg_score * uncertainty_weight`

**Panel diversity kinds:**
- `CrossFamily` (≥ 2 distinct adapter families): one variant per family, cap 3; supermajority vote (`quorum_fraction` default 0.67)
- `PersonaOnly` (single family): 3 variants (Literal, Contextual, Skeptical); unanimous agreement required — any dissent produces `Uncertain`

### VerificationScoredEvent fields

```
passed_checks: Option<u32>,         // #[serde(default)]
total_checks: Option<u32>,          // #[serde(default)]
score_lower: Option<f64>,           // Wilson score 95% CI lower bound
score_upper: Option<f64>,           // Wilson score 95% CI upper bound
per_check_verdicts: Vec<CheckVerdict>,  // #[serde(default)]
cache_hit: bool,                    // #[serde(default)]
```

---

## Merge Phase (merger.rs)

### MergeOutcome variants

- `Resolved { selection_resolved: SelectionResolvedEvent, resolved: Box<MergeResolvedEvent> }`
- `ZeroSurvival(ZeroSurvivalEvent)`

### MergeStrategy selection

| Strategy | Condition |
|----------|-----------|
| `ScoreOrdered` | `max_ci ≤ bft_threshold` — highest-scored surviving proposal |
| `ConsensusMedian` | `bft_threshold < max_ci ≤ krum_threshold` |
| `OutlierResistant { f: usize }` | `max_ci > krum_threshold` AND `krum_f > 0`; requires `n ≥ 2f+3`; Krum single-selection |
| `MultiOutlierResistant { f: usize, m: usize }` | Iterative Multi-Krum |

### OSP regime classification

When OSP is active, surviving proposals are classified before strategy dispatch:

- **`OSP-SingleSurvivor`**: `scores.len() ≤ 1`
- **`OSP-ClearLeader`**: `scores[0] - scores[1] ≥ 2·t_v`
- **`OSP-TightCluster`**: otherwise → semantic median

`SelectionResolvedEvent.merge_selection_mode: Option<String>` (`#[serde(default)]`) records which sub-path ran. Valid values: `"OSP-SingleSurvivor"`, `"OSP-ClearLeader"`, `"OSP-TightCluster"`, `"ScoreOrdered"`, `"ConsensusMedian"`, `"OutlierResistant-Krum"`, `"OutlierResistant-Weiszfeld"`, `"OutlierResistant-ConsensusMedian"`, `"MultiKrum"`.

### SelectionResolvedEvent fields

```
merge_selection_mode: Option<String>,   // #[serde(default)]
n_input_proposals: usize,              // #[serde(default)]
n_failed_proposals: usize,             // #[serde(default)]
```

### MergeResolvedEvent fields

```
j_eff: Option<f64>,                              // #[serde(default)]
oracle_gate_passed: Option<bool>,                // only on MergeResolved (post gate)
zone3_hints: Option<String>,                     // #[serde(default)]
contradiction_analysis: Option<ContradictionAnalysis>,  // #[serde(default)]
```

### ContradictionAnalysis

When `contradiction_explanation = true` (default: `false`), `merger.rs` populates `MergeResolvedEvent.contradiction_analysis`:

```
ContradictionAnalysis {
    n_valid: usize,
    n_total: usize,
    contradictions: Vec<ContradictionEntry>,
    rendered: String,
}
```

`resolved_output` is **never** modified — the analysis is a separate field only. `render_contradiction()` lives in `h2ai-autonomic` (not `h2ai-types` — circular dependency constraint).

---

## Post-Merge Stages

### Epistemic quality stage (epistemic_quality.enabled = true, default)

Runs inside `engine.rs::run_offline()` after the MAPE-K wave loop exits, before `EngineOutput` is returned to `task_pipeline.rs`. Stages:

1. Gap extraction and coherence check
2. `MicroExplorerResolver` recovery loop (concurrent, one resolver per gap)
3. `ProvenanceMap` build
4. Optional re-verification
5. Output rendering

`ProvenanceRecordedEvent` is published by `post_run()` at position 17 (before `MergeResolved`), only if `provenance_map.is_some()`.

### Grounding checker (GroundingChecker in h2ai-orchestrator)

The `GroundingChecker` implements `GapChecker` and wraps a composable `GroundingJudge` trait. It runs inside `run_epistemic_stage` in two positions:

1. **Pre-feedback-loop**: called on `out.resolved_output` (the merged output). Resulting `UngroundedContent` gaps become part of `static_gaps` passed to the feedback loop.
2. **Post-feedback-loop**: called on `final_output` only when `closed_ids` is non-empty — catches new ungrounded entities introduced by recovery patches.

**Gap production.** `GroundingChecker::check()` calls `judge.judge(output, spec)`, filters findings by `confidence ≥ min_confidence` (`grounding.min_confidence`, default 0.7), and emits one `Gap` per finding:

- `kind`: `GapKind::UngroundedContent`
- `source`: `GapSource::GroundingCheck`
- `id`: `grounding:{text_lowercased_underscored}`
- `description`: `[entity|claim] text: reason`
- `severity`: confidence ≥ 0.9 → `High`; ≥ 0.7 → `Medium`; else → `Low`

Config flag: `grounding.enabled` (default: `true`).

### Oracle dispatch (post-MergeResolved, background)

When `oracle_spec.is_some()`, oracle dispatch fires as a background task after `MergeResolved` is published. The `oracle_gate_passed: Option<bool>` field on `MergeResolvedEvent` carries the post-selection gate result.

---

## Calibration (calibration.rs)

The calibration harness measures three parameters from the adapter pool:

- **α** — serial bottleneck fraction
- **β₀** — pairwise reconciliation cost; resolved via three-tier cascade: epistemic formula (when embedding model configured + N_cal ≥ 3) → conflict-count override from rolling `ConflictRateAccumulator` → latency-based fallback
- **CG(i,j)** — Common Ground between every Explorer pair: mean pairwise Hamming distance on binary constraint-satisfaction fingerprints

These yield `N_max = sqrt((1−α) / β_eff)` where `β_eff = β₀ × (1 − CG_mean)`.

**USL fit phases:**
- Phase A: N=2
- Phase B: N=M ≥ 3; falls back to config defaults if M < 3

**CalibrationSource variants:** `Measured` (M ≥ 3 + ≥ 2 adapters), `SyntheticPriors`, `PartialFit`.

**Family constraint:**
- `require_diverse` (production) — aborts on single-family pool
- `single_family_ok` (development default) — allows single-family pool with warning

Fields present since 2026-05-16: `single_family_warning: bool`, `explorer_verification_family_match: bool`.

---

## Drift Monitoring (drift.rs)

The drift monitor receives `consensus_agreement_rate` as its input signal (one observation per resolved task).

**Two-layer detector:**

- **DDM (fast layer)**: O(1), window=20, k=2.5 sigma. Emits `CalibrationDriftWarning`.
- **BOCPD (slow layer)**: NIG conjugate prior (Adams & MacKay 2007); fires `CalibrationChangepoint` when P(r_t ≤ 4) > threshold.

**ORCA conformal margin adjustment.** An active changepoint causes `active_conformal_margin()` to return `drift_conformal_margin` (default 0.05). This margin is subtracted from the verification threshold at the start of each task (step 1 of the engine pre-loop). Margin TTL: `drift_staleness_ttl_secs` = 3600 s — drops to 0.0 after TTL.

`reset_after_recalibration()` resets both detectors.

---

## Key Configuration Flags

| Flag | Default | Effect |
|------|---------|--------|
| `thinking_loop.enabled` | `false` | Pre-dispatch expert analysis loop |
| `complexity_routing.enabled` | `false` | Complexity probe before dispatch |
| `complexity_routing.intra_retry.enabled` | `false` | Ceiling detector in ZeroSurvival path |
| `leader_enabled` | `false` | Epistemic leader election |
| `contradiction_explanation` | `false` | Populates `ContradictionAnalysis` on `MergeResolved` |
| `synthesis_wave_enabled` | `true` | Terminal synthesis wave on retry exhaustion |
| `gap_i1.enabled` | `false` | Web-grounded gap research during retries |
| `grounding.enabled` | `true` | Post-merge grounding checker |
| `epistemic_quality.enabled` | `true` | Post-merge provenance and gap annotation |

---

## Key Event Type Reference

### TaskFailedEvent

```
primary_cause: TerminalCause,                  // #[serde(default)]
contributing_causes: Vec<TerminalCause>,       // #[serde(default)]
top_violated_constraints: Vec<(String, u32)>,  // top-5 by frequency
last_selection_valid_count: Option<u32>,       // #[serde(default)]
```

`TerminalCause` variants: `LlmAdapterUnavailable`, `VerificationExhaustion`, `NoValidProposals`, `ComplexityOverflow`, `ContextExhaustion`, `OracleRejected`, `Timeout`, `Unknown`.

### ZeroSurvivalEvent

```
failure_mode: Option<FailureMode>,      // #[serde(default)]
n_eff_cosine_actual: Option<f64>,       // #[serde(default)]
```

`FailureMode` variants: `ConstrainedExploration`, `ModeCollapse`, `CorrelatedHallucination { cv, mean_jaccard_distance }`.

### BranchPrunedEvent

```
violated_constraints: Vec<ConstraintViolation>,
retry_count: u32,                // #[serde(default)]
bypass_reason: Option<String>,   // #[serde(default)]
```

### TopologyProvisionedEvent

```
constraint_tombstone: Option<String>,   // #[serde(default)] — only on ConstrainedExploration retries
```
