# Epistemic Committee Design

**Status:** Stages 1–5a implemented. Stage 5b (oracle) is future work.  
**Gaps addressed:** GAP-A5 (✅ closed), GAP-D4 (✅ closed), GAP-A4 (🟡 architectural fix done; loop gate and A/B measurement pending), GAP-C1 (🟡 adversarial probing in place; detector pending)  
**Companion documents:** [`gaps.md`](../architecture/gaps.md), [`research-state.md`](../architecture/research-state.md)

### Implementation Status

| Stage | Description | Status | Key deviation from design |
|---|---|---|---|
| 1 | `CalibrationSource` enum, `focus_mandate`/`rejection_criteria` on `ExplorerSlotConfig`, `CalibrationSource` on `CalibrationCompletedEvent` + `TaskAttributionEvent` | ✅ Done | `diverse_defaults()` removed entirely (became dead code once Stage 2 always runs) |
| 2 | LLM-derived decomposition (Path C), orthogonality pruner, Phase 0 in `tasks.rs`; `corpus_fallback` as utility | ✅ Done — with design correction | Path C **always runs** (no operator-bypass check); failure is hard `TaskFailed`, not a silent fallback to corpus. `corpus_fallback()` exists as a utility but is not wired as a production path. Path A (operator bypass) was removed. |
| 3 | `CoherenceState` + per-wave computation + `CoherenceIncomplete` event + `active_contradictions` | ✅ Done — exceeds spec | `active_contradictions: Vec<(ExplorerId, ExplorerId, String)>` added (not in original Stage 3 spec). `is_closed()` is computed and emitted as observability; **it is not yet a loop exit gate** (see §3.3 below). |
| 4 | `ADVERSARIAL_EVALUATOR_SYSTEM_PROMPT` + activation on `rejection_criteria` presence | ✅ Done — partial on A/B | Adversarial verifier defined in `h2ai-types/src/prompts.rs`, activated when any slot has non-empty `rejection_criteria` (auto-triggered by Path C). A/B score distribution comparison is **not done** (blocked by GAP-E1 oracle). |
| 5a | Prometheus label `h2ai_calibration_source`, startup warning on `SyntheticPriors`, `calibration_source` on `TaskAttributionEvent` | ✅ Done | All three surfaced. |
| 5b | `OracleExecutor` + `Phase6OracleResult` + calibration accumulation | ⏳ Future | Requires oracle harness (GAP-E1 open). |

---

## 1. Problem Statement

The H2AI execution engine currently treats proposals as **outputs to score**. A proposal is a text string; the verifier assigns a number; the highest number wins. This framing has three structural failures:

**Failure 1 — Committee is mathematically sized but semantically empty.**  
N comes from a USL formula or from the operator's manifest integer. Roles are either absent or filled with domain-agnostic CoT styles (`StepByStep`, `DevilsAdvocate`). Nobody is assigned responsibility for knowing anything specific. The question "why 3 agents, not 4?" has no answer except "calibration allows 3."

**Failure 2 — Verification confirms rubric compliance, not truth.**  
The constraint rubric is injected into the explorer's `system_context`. The LlmJudge verifier then scores proposals against that same rubric. For any task that is simple enough for the model to follow instructions, score = 1.0 on the first attempt, zero retries, zero epistemic work done. The MAPE-K retry loop adds value only when complexity causes the explorer to fail despite knowing the rubric — a condition that is empirically untested and rarely met in single-model deployments.

**Failure 3 — No grounding loop.**  
Proposals that survive coherence checks (verification + audit) are treated as knowledge. But the system cannot distinguish "the committee is confident and correct" from "the committee is confident and wrong." There is no loop that connects beliefs to external reality.

---

## 2. The Right Framing

Every non-trivial problem is a **team knowledge acquisition problem**. The team must:

1. Identify what it needs to know (epistemic division of labor)
2. Assign each knowledge domain to an agent with the right domain priors
3. Form beliefs through exploration
4. Test beliefs for internal coherence (constraints)
5. Resolve contradictions through adversarial inquiry
6. Ground load-bearing beliefs in external reality
7. Stop when the knowledge graph has reached principled closure — not when a retry budget runs out

The solution architecture is not a linear pipeline. It is a **graph of thinking, decisions, and executions** with loops at each level:

```
  ┌─────────────────────────────────────────────────────────┐
  │  TASK                                                   │
  │    description + constraint_corpus + pareto_weights     │
  └──────────────────────┬──────────────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────────────┐
  │  DECOMPOSITION  (new)                                   │
  │    What knowledge domains does this task require?       │
  │    → Vec<(domain, role_frame, cot_style, mandate)>      │
  │    → motivated N  (count of domains)                    │
  │    Pareto weights select the coverage/independence/cost │
  │    tradeoff point on the domain set                     │
  └──────────────────────┬──────────────────────────────────┘
                         │  N ≤ USL N_max  (budget ceiling)
                         ▼
  ┌─────────────────────────────────────────────────────────┐
  │  COMMITTEE EXECUTION  (parallel)                        │
  │    Each agent: role_frame + cot_style + focus_mandate   │
  │    Inner loop (TAO): explore hypothesis space           │
  │    Output: belief node  {claim, evidence, assumptions}  │
  └──────────────────────┬──────────────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────────────┐
  │  COHERENCE LOOP  (MAPE-K, extended)                     │
  │    Coherence test: do beliefs satisfy constraint axioms?│
  │    Contradiction detection: do beliefs conflict?        │
  │    Revision: targeted re-exploration of failed domains  │
  │    Stop: epistemic closure (no contradiction remains)   │
  └──────────────────────┬──────────────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────────────┐
  │  ADVERSARIAL REVIEW  (extended audit)                   │
  │    Verifier role: hostile reviewer — find the failure   │
  │    Not: "does this satisfy the rubric?"                 │
  │    But: "what is the most likely way this is wrong?"    │
  └──────────────────────┬──────────────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────────────┐
  │  SYNTHESIS  (belief integration)                        │
  │    Construct the most coherent view across all agents   │
  │    Justification: which agent contributed which claim   │
  │    Scope: where does the integrated belief hold?        │
  └──────────────────────┬──────────────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────────────┐
  │  GROUNDING LOOP  (oracle, when available)               │
  │    Connect load-bearing beliefs to external reality     │
  │    Code tasks: test suite execution                     │
  │    Structured output: schema/type validation            │
  │    Stop: all load-bearing beliefs grounded              │
  └──────────────────────┬──────────────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────────────┐
  │  OUTPUT                                                 │
  │    belief + justification chain + confidence + scope    │
  │    calibration_source label (measured vs. synthetic)    │
  └─────────────────────────────────────────────────────────┘
```

---

## 3. Design Components

### 3.1 Task Decomposition Layer

**Problem it solves:** GAP-A5 — committee composition is semantically unmotivated; N and roles must come from task decomposition.

**What it does:**  
Given `(task_description, constraint_corpus, pareto_weights)`, produce a `Vec<ExplorerSlotConfig>` where each slot has a motivated `role_frame`, `cot_style`, and `focus_mandate`. The count of slots IS the motivated N.

**The Pareto connection:**  
The decomposition problem is a multi-objective optimization over three axes:

| Pareto weight | Decomposition axis | Meaning |
|---|---|---|
| `containment` | Coverage completeness | Every active constraint domain has ≥ 1 specialist slot |
| `diversity` | Role orthogonality | Max pairwise embedding distance between `role_frame` vectors |
| `throughput` | Minimum N | Fewest slots that satisfy coverage and orthogonality requirements |

The Pareto-optimal decomposition maximises `containment × coverage + diversity × orthogonality` subject to `N ≤ N_max`. The weights already encode the operator's intent — a `containment: 0.7` task says "full domain coverage matters more than cost." The decomposition layer hears that signal first, before topology selection hears it.

**Implementation paths — as designed vs. as built:**

*Path A — Operator-specified bypass (removed):*  
The original design allowed operator-supplied `slot_configs` to bypass decomposition. This was removed. Path C always runs. Operator-supplied `slot_configs` (if any) are **appended** to the Path C result as additive operator context, then the combined set is re-pruned by orthogonality. They do not skip decomposition.

*Path B — Constraint-corpus-driven (`corpus_fallback`):*  
Implemented as a public utility (`crates/h2ai-orchestrator/src/decomposition.rs::corpus_fallback`): groups the active constraint corpus by `domains` field, one specialist slot per domain (security, performance, correctness, consistency, compliance). **Not wired as a production path.** Path C failure causes `TaskFailed`; there is no silent fallback to Path B. The function exists for testing and for future recovery scenarios.

*Path C — LLM-derived (always runs):*  
`run_decomposition_agent()` in `tasks.rs` Phase 0. A pre-dispatch call to the auditor adapter (most capable, τ=0.1): *"What are the N most cognitively distinct expert perspectives needed to solve this problem?"* Structured JSON response parsed into `Vec<ExplorerSlotConfig>` — each slot has motivated `role_frame`, `cot_style`, `focus_mandate`, and `rejection_criteria`. Returns `Result<Vec<ExplorerSlotConfig>, DecompositionError>`; failure propagates as `TaskFailed` with no retry. N = count of genuinely orthogonal roles, capped at N_max by orthogonality pruning.

**Design decision record:** Path A was removed because operator-specified slots were doing Path C's job manually — they required the operator to know the domain decomposition upfront, defeating the epistemic independence principle. Path B as a silent fallback was removed because if Path C fails, the task's goals are not achievable with the available decomposition — the fallback was hiding failures rather than surfacing them.

**USL N_max as budget guard:**  
The decomposition produces `desired_N`. If `desired_N > N_max`: prune by orthogonality — drop the slot whose `role_frame` embedding is closest to another retained slot (least independent perspective). Never pad to fill the budget. Never prune randomly.

---

### 3.2 Epistemic Role Assignment

**Problem it solves:** GAP-A5 (role content), GAP-A4 (verifier independence), GAP-A3 (ρ measurement via persona diversity).

**What changes in `ExplorerSlotConfig`:**

```rust
pub struct ExplorerSlotConfig {
    pub role_frame: String,      // exists, currently empty in diverse_defaults()
    pub cot_style: CotStyle,     // exists
    // NEW:
    pub focus_mandate: String,   // what this slot is responsible for covering
    pub rejection_criteria: String, // what this slot should specifically try to falsify
}
```

`focus_mandate` ensures coverage: slot 1 is responsible for CONSTRAINT-001 and CONSTRAINT-002; slot 2 owns CONSTRAINT-003. The verifier can then check "did this proposal address the security concerns the security-focused agent would raise?" — not just "does this text satisfy the rubric globally?"

`rejection_criteria` implements adversarial epistemology: each explorer is given a specific failure mode to look for, making them a partial adversary of their own proposal. The strongest proposal is one that survives its own agent's rejection criteria.

**Verifier role assignment:**  
The verifier (`EVALUATOR_SYSTEM_PROMPT`) should also carry a role. Not "check whether this satisfies the criteria" but "act as a hostile reviewer — identify the single most likely way this proposal fails silently." This restores epistemic independence: the verifier is not confirming rubric compliance, it is adversarially probing the proposal. This is the minimum viable fix for GAP-A4 without the architectural change of stripping the rubric from the explorer's context.

**`diverse_defaults()` fix:**  
The four default configs should gain non-empty `role_frame` strings:

```rust
ExplorerSlotConfig {
    role_frame: "You are a systems architect. Your first concern is what \
                 breaks under load and what is impossible to change later.".into(),
    cot_style: CotStyle::FirstPrinciples,
    ..
},
ExplorerSlotConfig {
    role_frame: "You are a security engineer. Before proposing anything, \
                 enumerate what an attacker can do with this interface.".into(),
    cot_style: CotStyle::DevilsAdvocate,
    ..
},
// etc.
```

---

### 3.3 Coherence-Based Stopping Criteria

**Problem it solves:** GAP-A4 (retry loop value), GAP-D1 (calibration measures latency not epistemic work).

**Correct framing:** Stopping when proposals reach acceptable quality IS the right criterion. The current MAPE-K loop is quality-gated in form — it stops when proposals pass the verification threshold or the budget exhausts. The problem is that "acceptable quality" is currently verifier-coherent (satisfies the rubric the explorer already saw), not oracle-coherent (correct against external reality). The budget `max_autonomic_retries` is the best available proxy until the oracle grounding loop (§3.4) closes. It becomes principled when calibrated to the empirical distribution of "iterations to coherent closure" — which requires oracle data to define what closure means against ground truth.

**What this stage adds:** An explicit coherence state that makes the quality gate legible and domain-aware, moving from "verifier threshold exceeded" (a scalar gate) to "no domain is uncovered" (a structural gate). This is a necessary intermediate step before the oracle loop can define calibrated quality thresholds per domain.

**Principled criterion:** The retry loop stops when the knowledge graph has reached **coherent closure** — no active constraint is violated by the best surviving proposal, and no proposal in the retained set contradicts another on its `focus_mandate` domain.

This is mostly already implemented: the MAPE-K retry fires when `RetryWithHints` is triggered by a failed verification pass. The gap is that the stopping condition is the budget, not the epistemic state. The fix:

1. Add a `CoherenceState` to the retry loop: `{uncovered_domains: Vec<ConstraintDomain>, active_contradictions: Vec<(ProposalId, ProposalId, Domain)>}`.
2. Compute `CoherenceState` after each verification round from the constraint violation map.
3. Stop when `CoherenceState::is_closed()` — empty uncovered domains AND empty active contradictions. This replaces the "all proposals pass verification" condition, which is currently equivalent for single-proposal outputs but diverges when N > 1 surviving proposals with different domain coverage.
4. If budget exhausts before closure: emit `CoherenceIncomplete {uncovered_domains}` alongside the best proposal, so callers know exactly what the output does not cover.

**Connection to oracle grounding (§3.4):** `CoherenceState::is_closed()` checks coherence against the constraint rubric — it remains verifier-coherent. The oracle grounding loop upgrades "closed" to mean "closed AND grounded" when an oracle is available. These are separate stopping conditions layered in sequence.

**Implementation note (2026-05-09).** `CoherenceState` is fully implemented with two fields:

```rust
pub struct CoherenceState {
    /// Constraint domains where any pruned proposal had violations. Sorted.
    pub uncovered_domains: Vec<String>,
    /// Pairs of surviving proposals that score on opposite sides of 0.5
    /// for any constraint in the same domain. Each entry: (a, b, domain).
    pub active_contradictions: Vec<(ExplorerId, ExplorerId, String)>,
}
```

`from_pruned(corpus, all_pruned)` computes uncovered domains from the cumulative pruned set. `with_contradictions(corpus, explorer_ids, satisfaction_matrix, constraint_ids)` adds contradiction pairs from Phase 4.5's static-constraint satisfaction matrix. `is_closed()` returns true only when both fields are empty.

`wave_coherence` is computed inside the MAPE-K retry loop after each wave (after `all_pruned.extend()` and after the Phase 4.5 frontier event) and is **reused at all exit paths** rather than being recomputed. It is surfaced via `tracing::trace!` with `uncovered_domains`, `active_contradictions.len()`, and `is_closed()`.

**Gap:** `is_closed()` is currently **observability only** — it is traced and emitted as `CoherenceIncomplete` at task completion when not closed, but the retry loop does **not** exit early when coherent. The loop still stops on budget exhaustion or "all proposals pass verification." Wiring `is_closed()` as an early exit condition is the next concrete step for GAP-D1 closure.

---

### 3.4 Oracle Grounding Loop

**Problem it solves:** GAP-E1 — no oracle integration; the system is epistemically ungrounded.

**Design:**  
The oracle loop runs after coherence closure, not instead of it. It is the last loop, not a replacement for the others. Its job: connect the surviving belief (best proposal) to external reality for the dimensions where a deterministic oracle exists.

**`OracleExecutor` as a `ToolExecutor` variant:**

```rust
pub enum OracleKind {
    TestSuite { command: String },           // cargo test, pytest, etc.
    SchemaValidation { schema_json: String },
    RegexMatch { pattern: String },
    // future: Symbolic { verifier: SymbolicKind }
}

pub struct OracleResult {
    pub passed: bool,
    pub oracle_kind: OracleKind,
    pub details: String,
    pub grounded_claims: Vec<String>,  // which belief claims were tested
}
```

**Integration point:**  
`OracleResult` events are emitted as `Phase6OracleResult` in the SSE stream. They are never blocking (the task output is already produced; oracle is async verification). They feed the calibration pipeline: accumulate `(q_confidence, oracle_passed)` pairs to build the calibration curve for conformal prediction (GAP-B3).

**Minimum viable oracle for code tasks:**  
`ShellExecutor` already exists. A `TestOracle` is `ShellExecutor` with a fixed allowlist command (`cargo nextest run`, `pytest`) and structured output parsing. This is a 1-week implementation.

---

### 3.5 Calibration Source Labeling

**Problem it solves:** GAP-D4 — synthetic calibration is indistinguishable from measured calibration.

**Design:**

```rust
pub enum CalibrationSource {
    Measured,        // USL fit from M ≥ 3 timing points, CG from ≥ 2 adapters
    PartialFit,      // USL fit valid; CG from fallback (< 2 adapters or empty corpus)
    SyntheticPriors, // Both USL and CG from config fallbacks
}
```

Added to `CalibrationCompletedEvent`. Set in `CalibrationHarness::run` based on which paths used fallbacks. Surfaced as:
- A Prometheus label: `h2ai_calibration_source{source="synthetic_priors"}`
- A startup warning when most recent stored calibration has `SyntheticPriors`
- A field on `TaskCompletedEvent` so outputs can be retrospectively filtered by calibration quality

This does not change routing behavior. It changes epistemic transparency: the system knows what it doesn't know about itself.

---

## 4. What Changes vs. What Stays

### Stays the same
- USL N_max as the hard budget ceiling on N
- MAPE-K as the implementation of the coherence loop
- CJT as the quality bound when beliefs are independent
- CRDT `ProposalSet` for idempotent merge
- Pareto weights as the tradeoff signal (now applied at decomposition AND topology)
- NATS/JetStream infrastructure
- `ExplorerSlotConfig` schema (extended, not replaced)

### Changes (as implemented)
- N is output of Phase 0 decomposition (Path C), not input from manifest integer
- `diverse_defaults()` **removed** — replaced by Path C always-on (was dead code once Phase 0 runs unconditionally)
- `ExplorerSlotConfig` gained `focus_mandate` and `rejection_criteria`; engine injects `[MANDATE]:` and `[FIND]:` preambles
- `ADVERSARIAL_EVALUATOR_SYSTEM_PROMPT` defined; activated automatically when any slot carries `rejection_criteria`
- `CoherenceState` computed per-wave with `uncovered_domains` + `active_contradictions`; `CoherenceIncomplete` event emitted at task close
- `CalibrationCompletedEvent` and `TaskAttributionEvent` both carry `calibration_source`; Prometheus gauge + startup warning wired
- `compiler::compile(manifest, corpus, include_rubric: bool)` — passing `false` withholds LlmJudge rubrics from explorer context

### New components
- **Task Decomposition Layer** — sits before `EngineInput` construction in `tasks.rs`
- **`CoherenceState`** — computed per retry round, drives stopping and `CoherenceIncomplete` event
- **`OracleExecutor`** — `ToolExecutor` variant, wired as Phase 6
- **Domain tags on `ConstraintDoc`** — enables constraint-corpus-driven decomposition

---

## 5. Implementation Sequence

The components have a natural dependency order that avoids regressions:

**Stage 1 — Epistemic transparency (no behavior change, 1 week)**
- `CalibrationSource` enum + `CalibrationCompletedEvent` field (GAP-D4)
- `focus_mandate` + `rejection_criteria` fields on `ExplorerSlotConfig` (schema only, unused)
- `domain` tag on `ConstraintDoc` YAML schema (schema only)
- Non-empty `role_frame` in `diverse_defaults()` (low risk, immediate diversity benefit)

**Stage 2 — Decomposition path B (constraint-corpus-driven, 1 week)**
- Domain tag library for constraint types
- `decompose_from_corpus(corpus, pareto_weights, n_max) -> Vec<ExplorerSlotConfig>`
- Wire into `tasks.rs` before `EngineInput` construction, activated by feature flag
- Update all ads-platform constraint YAMLs with `domain` tags

**Stage 3 — Coherence stopping criterion (1 week)**
- `CoherenceState` struct + `is_closed()` computation from constraint violation map
- Wire as secondary stopping condition in MAPE-K loop (alongside existing budget)
- `CoherenceIncomplete` event when budget exhausts before closure

**Stage 4 — Adversarial verifier role (3 days)**
- Verifier prompt variant with hostile-reviewer `role_frame`
- A/B test: measure score distribution shift (should see lower first-pass scores, higher oracle correlation)

**Stage 5 — Oracle grounding (1 week)**
- `OracleExecutor` for code tasks (wraps `ShellExecutor`)
- `Phase6OracleResult` event
- Wire oracle result accumulation into calibration pipeline

**Stage 6 — Decomposition path C (LLM-derived, 1 week)**
- Pre-dispatch LLM call producing `Vec<ExplorerSlotConfig>`
- Orthogonality pruning when `desired_N > N_max`
- Evaluation harness comparing path B vs. path C on oracle pass rate

---

## 6. Gaps Addressed

| Gap | Component | Status | How addressed |
|---|---|---|---|
| GAP-A5 | Task Decomposition Layer (§3.1) | ✅ Closed | Path C always runs; N and roles derived by LLM; failure is hard TaskFailed |
| GAP-A4 | Rubric separation (§3.3) + adversarial verifier (§3.2) + CoherenceState (§3.3) | 🟡 Architectural fix done; loop gate + A/B pending | `include_rubric=false` withholds rubric from explorer; adversarial verifier auto-activates; `CoherenceState` is per-wave observability — not yet a loop exit gate |
| GAP-A3 | Role-frame diversity in experiments (§3.2) | 🟡 Infrastructure ready | Path C produces non-empty `role_frame` and `rejection_criteria` per slot; ρ measurement experiment not yet run |
| GAP-C1 | `rejection_criteria` on slots (§3.2) | 🟡 Prompt-level done | Each explorer is given a specific failure mode to probe; adversarial verifier activates; correlated hallucination *detector* not implemented |
| GAP-D4 | Calibration Source Labeling (§3.5) | ✅ Closed | `CalibrationSource` on both `CalibrationCompletedEvent` and `TaskAttributionEvent`; Prometheus gauge + startup warning |
| GAP-D1 | Coherence stopping criterion (§3.3) | 🟡 Observability done; gate pending | `wave_coherence.is_closed()` traced per-wave; `CoherenceIncomplete` event emitted — loop exit on closure not yet wired |
| GAP-E1 | Oracle Grounding Loop (§3.4) | ❌ Open | Not implemented; `ShellExecutor` exists as building block |
| GAP-B3 | Oracle loop feeds conformal calibration (§3.4) | ❌ Open (blocked by GAP-E1) | Requires oracle data |

---

## 7. Open Questions Before Implementation

- **Decomposition path B vs. C ordering**: Should path B (constraint-corpus-driven) be the production default and path C (LLM-derived) be opt-in, or should they run in parallel with path C overriding path B when available?
- **`focus_mandate` ownership vs. constraint ownership**: Should each slot own a subset of constraints (slot 1 checks CONSTRAINT-001 and CONSTRAINT-002; slot 2 checks CONSTRAINT-003), or should all slots check all constraints from their domain perspective? Ownership enables coverage guarantees; full-coverage-per-slot enables cross-validation.
- **Adversarial verifier vs. rubric verifier**: Should the adversarial verifier *replace* the current rubric verifier, or run *alongside* it? Replacement is cleaner; parallel gives a score comparison that helps validate the approach.
- **`CoherenceIncomplete` output policy**: When the budget exhausts before coherence closure, should the system (a) emit the best proposal with a warning, (b) fail the task, or (c) let the Pareto weights decide (containment-heavy → fail; throughput-heavy → emit with warning)?
- **Oracle blocking vs. async**: The oracle loop is designed as async (non-blocking). Should it ever block task completion — e.g., for high-stakes tasks where the operator has set `require_oracle_grounding: true`?
