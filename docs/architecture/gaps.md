# H2AI Gaps — Research and Engineering Agenda

This document is the actionable companion to [`research-state.md`](research-state.md). Every gap
is a falsifiable question with a concrete research or engineering path.

---

## Navigation

| Section | What it covers |
|---|---|
| [Problem Space Map](#problem-space-map) | At-a-glance status and severity for all open/partial gaps |
| [Innovations Roadmap](#innovations-roadmap) | Cross-cutting solutions that close multiple gaps simultaneously |
| [Group A — Core Thesis](#brainstorm-group-a--core-thesis-validity) | Does the fundamental approach work and beat its competitors? |
| [Group B — Math Apparatus](#brainstorm-group-b--mathematical-formula-validity) | Are the formulas principled or arbitrary? |
| [Group D — Infrastructure](#brainstorm-group-d--infrastructure-and-operational-gaps) | Do the inputs to the math arrive correctly? |
| [Group F — Knowledge and Retrieval](#brainstorm-group-f--knowledge-and-retrieval) | Does knowledge injection improve outputs, and do constraint signals reshape routing? |
| [Group G — Reasoning Memory](#brainstorm-group-g--reasoning-memory) | Does the system learn across tasks? |
| [Group H — Skeptical Audit Resilience](#brainstorm-group-h--skeptical-audit-resilience) | One open production gap: small-N human rating calibration |
| [Group S — Signal Fidelity](#brainstorm-group-s--signal-fidelity) | Are internal signals accurate representations of reality? |
| [Gap Priority Matrix](#gap-priority-matrix) | Suggested implementation order |
| [Shared Infrastructure](#shared-infrastructure-required-for-group-a) | Pre-work that blocks Group A experiments |
| [Foundational Framing](#foundational-framing--every-problem-is-a-team-epistemology-problem) | Epistemic framing of the H2AI problem space |

---

## Problem Space Map

| Gap | Status | Severity | Innovation opportunity |
|---|---|---|---|
| **GAP-A1 Self-MoA vs. multi-family routing** | 🟡 PARTIAL | **Critical** | H2-P vs. B3 experiment runnable; TCC parameters unfit |
| **GAP-A2 USL N_max vs. quality curve** | 🟡 PARTIAL | **Critical** | N_IT primary sizer implemented; empirical validation open |
| **GAP-B1 β_eff functional form** | 🟡 PARTIAL | Medium | Epistemic β₀ wired; empirical validation open |
| GAP-B3 Attribution self-referential | 🟡 PARTIAL | Medium | Conformal prediction once oracle data exists |
| **GAP-B5 Proxy chain — rho_mean, p_mean, β_eff unvalidated** | 🟡 PARTIAL | **High** | Online ρ_EMA live after 30 obs; cold-start prior 0.45 unvalidated |
| GAP-D2 Compound task cost unconstrained | 🔴 OPEN | Low | Complexity bandit; HITL escalation on graft_first=false path open |
| **GAP-F4 Knowledge provider has no contrastive evaluation** | 🟡 PARTIAL | **High** | Phase 1b closed; Phases 2–3 open |
| **GAP-F5 Constraint violations don't reshape retrieval routing** | 🟡 PARTIAL | Medium | Steps 1–2 live; Step 3 unblocked; Step 4 deferred |
| **GAP-G1 Reasoning Memory Phases 2–4 unimplemented** | 🟡 PARTIAL | Medium | Phase 1 live; Layer 3 partial; Phases 2–4 designed, pending |
| **GAP-H4 Small-N Human Ratings — MoM ECE breaks below N=50** | 🔴 OPEN | Medium | Dirichlet-Categorical posterior + credible-interval circuit breaker |
| **GAP-S1 SRANI fires for technology-specific impl details** | 🟡 PARTIAL | Low | Implied-by suppression table or CFI-gated hint emission |

**Severity key** — Critical: threatens core thesis validity; High: corrupts math inputs or silently disables documented features; Medium: degrades confidence in results; Low: operational or presentation issue.

---

## Innovations Roadmap

### INNOVATION-5 — Structured Self-MoA Experiment Protocol

**Closes:** GAP-A1 (comparative signal).
**Status: COMPLETE (2026-06-18)** — H2-P achieved MergeResolved on Tier 1 (2 constraints, avg=0.833), Tier 2 (4 constraints, avg=0.667), and Tier 3 (6 constraints, avg=0.667; 1/3 proposals 1.0 on all 6 constraints) with the full DPPM+SRANI+manifest.context stack.
**Implementation:** `tests/e2e/scenarios/innovation-5/` — three e2e scenarios (Tier 1/2/3).

**Experiment arms:**
- **B3 — Self-MoA baseline:** `baseline.toml`, `max_autonomic_retries = 0`. Three explorers with τ-spread; verifier scores; synthesis selects winner. No MAPE-K enforcement loop.
- **H2-P — Full H2AI pipeline:** `h2ai.toml`, `max_autonomic_retries = 4`. Same three explorers plus MAPE-K retry waves pruning non-compliant proposals and regenerating with repair context.

**Primary metric:** Constraint compliance rate — fraction of `_expected.should_prune` patterns that H2-P rejects and B3 passes through. Secondary: j_eff, verification scores, token cost.

**Open:** A matched B3 run at current code level (post-DPPM+SRANI+manifest.context) has not been completed. The +41pp H₁ delta is against the original pre-stack B3 baseline.

---

## Foundational Framing — Every Problem Is a Team Epistemology Problem

Any non-trivial problem is a **team knowledge acquisition problem**: the team must discover what is
true about the problem domain, resolve contradictions between what different team members believe,
and produce a justified output that survives contact with reality. The solution is not a pipeline
— it is a **graph of thinking, decisions, and executions**, with loops wherever knowledge needs
to be refined.

### The knowledge graph

```
Nodes  : beliefs  — {claim, evidence, assumptions, scope, confidence}
Edges  : support  (B strengthens A)
         contradiction  (B and A cannot both be true)
         derivation     (B follows from A by rule R)
         grounding      (oracle/test connects A to external reality)

Loops  :
  coherence   → resolve contradictions between beliefs until none remain
  coverage    → ensure every required knowledge dimension has a justified belief
  grounding   → connect load-bearing beliefs to external reality
  revision    → update beliefs when new evidence contradicts old ones
```

### How H2AI phases map to epistemic operations

| Phase | Epistemic operation |
|---|---|
| Task decomposition (Phase 0) | Epistemic division of labor — assign knowledge responsibilities |
| TAO inner loop | Hypothesis exploration — form initial beliefs |
| Phase 2 topology | Determine how many independent knowledge contributors the budget allows |
| Phase 3.5 verification | Coherence test — do beliefs satisfy the constraint axioms? |
| Phase 4 audit | Final coherence gate before a belief is accepted as output |
| Phase 5a synthesis | Belief integration — construct the most coherent view across contributors |
| MAPE-K retry | Belief revision — update under new evidence from failed coherence tests |
| Phase 6 oracle | **Grounding** — connect beliefs to external reality |
| Calibration | Update meta-beliefs about team epistemic capabilities |

### The epistemological traditions each gap violates

| Gap | Epistemic violation |
|---|---|
| GAP-B3: attribution without oracle | Cannot distinguish confident-and-correct from confident-and-wrong |
| GAP-B5: proxy chain | Three proxies (rho_mean, p_mean, β_eff) all use CG_mean as input with no empirical validation; cold-start prior 0.45 unvalidated |

### Stopping criteria

| Loop | Current criterion | Principled criterion | Gap |
|---|---|---|---|
| TAO inner | `agent_max_tool_iterations` (budget) | No productive hypothesis extensions remain | Budget is proxy for epistemic exhaustion |
| MAPE-K retry | Proposals satisfy threshold OR retries exhausted; ZeroSurvival + is_closed() gate | Coherent closure: no active constraint violated, no domain uncovered | Quality threshold is rubric-coherent, not oracle-grounded |
| Calibration | Startup-automatic + POST /calibrate | Confidence intervals narrow enough for decision quality required | — |
| Oracle grounding | Phase 4.5 gate wired (NATS request/reply, `OracleGateConfig`); thinking loop Stage 2 inline oracle; `PendingClarificationEvent` suspension via `clarification_waiters`; `OracleClient` POSTs winning output to external `runner_uri`, receives `{ passed, score, details }` | All load-bearing beliefs grounded in at least one oracle test | — |

---

## Brainstorm Group A — Core Thesis Validity

---

### GAP-A1: Self-MoA vs. Multi-Family — Does Diversity Matter? 🟡 PARTIAL

**Status: PARTIAL** — TCC parameters are unfit priors; quality comparison not yet measured.

**Literature grounding.**
arXiv 2502.00674 (Li et al., 2025) — *"Rethinking Mixture-of-Agents: Is Mixing Different LLMs
Beneficial?"* — proposes Self-MoA: aggregating outputs from a single top-performing model with
temperature variation. Finds Self-MoA matches or outperforms cross-model MoA on most benchmarks.
The argument: mixing weaker models dilutes the strong model's signal.

arXiv 2512.17630 (Elgabry & Hamdi, 2025) — *"Confidence-Credibility Aware Weighted Ensembles"*
— explicitly invokes CJT and confirms: CJT advantage holds only when error diversity is actively
maintained. Minimising parameter convergence (distinct architectures) is essential.

arXiv 2411.01539 (Bradley, 2024) — *"LLMs and the Madness of Crowds"* — LLM errors are
systematically correlated across architecturally similar models. Naive majority-vote ensembles
reinforce errors when models share the same failure mode.

**Open.**
TCC parameters are unfit priors. The smoke test measured latency, not quality. `shadow_mode =
false` means unfitted priors are live in production. The 2×2 cross-family experiment requires a
second model family not currently available.

**Falsification condition.**
If H2-P ≤ B3 on Tier 3 tasks (6+ constraints) with oracle pass rate as the signal (not internal
verifier score), the Coverage routing adds cost without benefit and should be replaced by Precision
routing universally.

---

### GAP-A2: USL N_max vs. Actual Quality Curve 🟡 PARTIAL — **Critical**

**Status: PARTIAL** — N_IT promoted to primary sizer; empirical validation of the N_IT vs. quality curve still open.

`N_max = round(√((1−α)/β_eff))` is derived from USL's throughput model — not output quality. No
published paper applies USL to LLM multi-agent quality ceiling. The USL ceiling and Condorcet
n_optimal serve different purposes: USL caps cost; Condorcet maximises quality-per-agent. Using a
cost model as a quality predictor can cause over-sizing or under-sizing depending on ρ regime.

**Implemented.** `n_it_optimal` is primary N recommendation; `n_max_USL` is demoted to cost cap:

```rust
let n_target = calibration.n_it_optimal();            // info-theoretic target
let n_max    = calibration.n_max();                   // USL cost ceiling
let n_final  = n_target.min(n_max).min(cfg.calibration_max_ensemble_size);
```

USL is now explicitly documented as: "N_max is a cost heuristic drawn from throughput engineering.
It prevents runaway token cost but is not a quality predictor. The quality target is n_IT."

**Remaining open.** Empirical validation of N_IT vs. quality curve requires a labeled multi-N
benchmark set (see Shared Infrastructure). Monte Carlo simulation (Gaussian copula, `q_condorcet`
vs `monte_carlo_ensemble_quality`) can validate the formula before oracle data is available.

**Key literature.** arXiv 0808.1431 (Gunther, 2008) — foundational USL derivation, β is coherency
cost in compute platforms. arXiv 2006.04969 (Hamann & Reina, 2020) — USL describes swarm
throughput; quality metrics follow different scaling laws. arXiv 2509.19489 (Nowak, 2025) —
optimal compute allocation: m,n ∝ √B, not n→∞.

---

## Brainstorm Group B — Mathematical Formula Validity

---

### GAP-B1: β_eff Functional Form 🟡 PARTIAL — Medium

**Status: PARTIAL** — Epistemic β₀ wired. Empirical validation of the linear β_eff assumption remains open.

`β_eff = β₀ × (1 − CG_mean)` has a first-principles derivation under the assumption that conflict
resolution cost is linear in conflict count. The assumption is falsifiable: if conflict resolution
cost is super-linear (e.g. due to "Lost in the Middle" attention degradation in long synthesis
contexts), the formula needs a higher-order term.

A context-aware formula `β_eff(N) = β₀ × (1 - CG_mean) × (1 + γ × fill(N))` is computed in
`n_max_context_aware` but is not the default path. With `beta_quality` live
(`ConflictRateAccumulator`, `H2AI_CONFLICT_{tenant}` KV), empirical validation is possible:
regress `conflict_rate` against `(1 - CG_mean)` and `(1 - CG_mean) × fill(N)`. If bivariate
R² > univariate R², the attention term has explanatory power.

---

### GAP-B3: Attribution Formula Is Self-Referential 🟡 PARTIAL

Oracle-grounded calibration is available via `OracleAccumulator`. Remaining open: apply conformal
prediction:

```
conformal_margin(α) = quantile(|q_confidence - q_oracle|, 1-α)  over calibration split
output: q_confidence ± conformal_margin(0.10)   [90% coverage guarantee]
```

arXiv 2410.11594 (Wagner et al., 2024) — *"Black-box Uncertainty Quantification for
LLM-as-a-Judge"* — applicable to H2AI's Phase 3.5 verifier score intervals. Blocked on
sufficient oracle-grounded calibration data.

---

### GAP-B5: Proxy Chain — rho_mean, p_mean, β_eff All Unvalidated 🔴 OPEN — **High**

Three interconnected proxies form a chain of unvalidated assumptions. Each propagates error into
the Condorcet Q(N, p, ρ) model and the USL ceiling N_max. The chain:

1. `rho_mean = 1 − CG_mean` — correlation proxy
2. `p_mean = 0.5 + CG_mean / 2` — accuracy proxy (`sizing.rs:635`)
3. `β_eff = β₀ × (1 − CG_mean)` — conflict cost proxy

**rho_mean formula inversion problem.** `rho_mean = 1 − CG_mean` implies CG_mean = 0 (zero
constraint agreement) → ρ = 1.0 (fully correlated). This is backwards: low CG_mean means agents
disagree on constraints, which should indicate *less* error correlation. The formula
`ρ = CG_mean` (agreement → correlation) is derivable from Hamming geometry, but neither form is
derivable from first principles without additional assumptions about whether error correlation
tracks constraint specialization.

**Pragmatic resolution documented as operational convention:**
> "Operational convention: low CG (diverse constraint profiles) is assumed to indicate lower error
> correlation. This assumes error patterns track constraint specialisation. The assumption is
> unvalidated and replaced by empirical ρ_EMA once 30 task observations exist."

**Cold-start detail.** Online ρ_EMA (`rho_ema.rs`) returns a hard-coded prior of `0.45` before
30 pairwise observations accumulate. This prior enters Condorcet Q(N,p,ρ) directly — all ensemble
sizing decisions for the first ~30 tasks rest on this unvalidated assumption. External validation
against a held-out benchmark dataset is the correct fix.

**Sensitivity analysis:**

```python
import numpy as np
from scipy.stats import binom

def q_condorcet(n, p, rho):
    q_ind = sum(binom.pmf(k, n, p) for k in range(n//2 + 1, n+1))
    return p + (q_ind - p) * (1 - rho)

# Show sensitivity of Q to ρ=CG vs ρ=1-CG for CG_mean=0.7
cg = 0.70
p  = 0.5 + cg / 2   # = 0.85
print("Formula | rho | Q(N=5)")
for rho_formula, label in [(cg, "rho=CG"), (1-cg, "rho=1-CG")]:
    q = q_condorcet(5, p, rho_formula)
    print(f"{label:10s} | {rho_formula:.2f} | {q:.4f}")
# If the two give very different Q, the formula choice matters significantly
```

---

## Brainstorm Group D — Infrastructure and Operational Gaps

---

### GAP-D2: Compound Task Cost Is Unconstrained 🔴 OPEN — Low

A `CompoundTaskEngine` DAG fires a full wave for each subtask with no pre-execution cost estimate
or operator confirmation gate. Up to 75 LLM calls before synthesis for a 5-subtask compound.

`ComplexityOverflow { graft_first: true }` routes to DPPM-MetaRefine synthesis instead of silent
terminal failure. The `graft_first: false` path (HITL surface, `complexity >= hitl_threshold`)
still produces `TaskFailed` without active HITL escalation.

**Remaining open.** HITL escalation on the `graft_first: false` path; compound task cost
estimation before dispatch.

**Research approach.** Complexity bandit probe: call a lightweight adapter (smallest available)
to rate subtask complexity 1–5 before dispatching ensemble. Route 1–2 to single-adapter path;
3–5 to full ensemble. Thompson Sampling bandit improves probe accuracy over time.

---

## Brainstorm Group F — Knowledge and Retrieval

---

### GAP-F4: Knowledge Provider Has No Contrastive Evaluation 🟡 PARTIAL — **High**

**Status: PARTIAL** — Phase 1b (domain scoping) closed; Phases 2–3 open.

Phase 1: `GenerationKnowledgeEvent` emitted per task with `knowledge_injected: bool` and
`q_confidence`; published to NATS alongside `TaskAttributionEvent`. Offline query:
`mean(q_confidence | knowledge_injected=true) - mean(... | false)`. Phase 1b: `scope_by_domains`
in `skill_provider.rs`; `CompositeProvider.domain_scoping`; `knowledge_domain_scoping: bool` in
`H2AIConfig`. Domain scoping prevents auth and security constraint nodes from polluting billing
task retrieval context.

**Cross-reference.** GAP-F4 is the H2AI analogue of the Planner role in Solvita (arXiv 2605.15301):
structured retrieval routing over what context is assembled per role is more defensible than
prompt-level diversity (temperature spread). Full Solvita parity requires Phase 3 graph edge weight
updates via REINFORCE on calibrated oracle signal — depends on GAP-B3 closing first.

**Remaining open:**

**Phase 2 — Per-slot knowledge routing bandit.** Promote contrastive signal into Thompson Sampling
bandit. Maintain two arms per task domain: `(domain_tag, knowledge_on)` vs
`(domain_tag, knowledge_off)`. After sufficient observations, domains where knowledge injection
hurts are automatically routed to the passthrough path.

**Phase 3 — Graph edge weight updates (Solvita parity).** When `InductionStore` records a
high-hit-rate node on `MergeResolved`, record the verification score delta attributed to that node
retrieval. Update node's edge weight proportional to the delta. Requires closing GAP-B3 (calibrated
judge) first — REINFORCE gradients on biased soft rewards amplify judge bias into retrieval weights.
Cold-start note: Phase 3 may be net-negative for tenants with fewer than ~200 tasks.

---

### GAP-F5: Constraint Violations Don't Reshape Retrieval Routing 🟡 PARTIAL (Steps 1–2) — Medium

**Status: PARTIAL** — Steps 1–2 live in-memory; Step 3 unblocked; Step 4 deferred.

Steps 1–2 implemented: `CompositeProvider.violation_map: Arc<RwLock<HashMap<String, f32>>>`
accumulates violation penalties for non-Synthetic nodes co-occurring with topology retries
(delta=0.1, cap=0.9, applied before dedup/top_k in `query()`). Synthetic skill nodes permanently
exempt. Penalty map is in-memory only — resets on restart; NATS persistence deferred.

**Remaining open:**

**Step 3 — Retroactive induction trigger.** When `ZeroSurvival` fires on a domain with ≥10 prior
tasks, trigger an induction cycle immediately (don't wait for batch threshold). Unblocked:
`post_injection_pass_rates` field on `DomainSynthesis` (pipeline-resilience spec) provides the
per-injection quality signal Step 3's retroactive trigger needs to distinguish productive from
noise injections.

**Step 4 — Constraint difficulty map (NATS-persisted). ⚠️ DEFERRED.** Track empirical constraint
difficulty per `(constraint_id, model_lineage_key)` pair across all tasks. Deferred because: (a)
most historical failures are integration failures, not inherent constraint difficulty — the map
would mislabel constraints as "hard" when they are individually solvable; (b) difficulty map must
be stratified by `model_lineage_key` and use a decaying Beta posterior. Implement after
DPPM-MetaRefine stabilizes and difficulty signal is no longer polluted by MUS oscillation
artifacts.

---

## Brainstorm Group G — Reasoning Memory

---

### GAP-G1: Reasoning Memory Phases 2–4 Unimplemented 🟡 PARTIAL — Medium

**Status: PARTIAL** — Phase 1 live; Layer 3 partial path live; Phases 2–4 designed, pending implementation.

**Phase 1 (live).** `TaskReasoningCheckpoint` written at each engine phase gate; `TaskMetaState`
projected at resolution; per-tenant NATS KV buckets (`H2AI_CHECKPOINT_{tenant}` 7d TTL,
`H2AI_META_{tenant}` 90d TTL). `run_from_checkpoint` reads `CheckpointPhase` to skip completed
phases. Skill extraction provides a depth-stratified analogue: Topic nodes carry Socratic
diagnostic questions + resolution excerpts; Constraint-keyed and Reason-keyed Leaf nodes carry
per-constraint domain signals. `format_induction_priors` formats top-5 `KnowledgeNodePattern`
entries by `hit_rate` as prior context prepended to archetype selection `system_context` (Layer 3
partial path).

**Remaining open — Phases 2–4:**

**Phase 2 — Induction (Layer 2).** Two components with strict separation: `InductionWorker` trait
(pure computation — no I/O, testable with `MockInductionWorker`) and `InductionScheduler` (owns
JetStream subscription, NATS KV reads/writes, CAS swap).

`InductionScheduler` triggers when ≥ `induction_batch_size` (10) resolved tasks accumulate, or
`induction_max_interval_secs` (86400s) elapsed. Loads up to `induction_max_tasks_per_run` (50)
`TaskMetaState` records, calls `worker.distill()`, writes to staging key, CAS-swaps `latest` only
on full success — the previous snapshot is never touched on failure.

`AlgorithmicInductionWorker` — pure Rust, no LLM calls. Distillation steps:
1. **ArchetypePrior** — group `ArchetypeResult` entries by `archetype_name + domain_tags`;
   `net_confidence = weighted_mean(confidence, weight=2.0 if dominated_synthesis else 1.0)`;
   `avoid_for_tags` = tags where `net_confidence < 0.4` across ≥ 3 tasks.
2. **TensionPattern** — collect all tension strings; cluster by cosine similarity (threshold 0.85)
   if `EmbeddingModel` available, exact dedup otherwise; store `frequency` + `resolution_hint`
   from tasks that resolved the tension.
3. **RetryHintPattern** — group `(trigger_tags, exit_reason_kind, retry_context_that_resolved)`
   tuples; keep top hint per pair by `success_rate`.
4. **DecompositionTemplate** — group `shared_understanding` strings by
   `(quadrant, constraint_tags)`; select embedding centroid if model available, most recent
   otherwise.

`TenantMemoryStore` lives in `H2AI_MEMORY_{tenant_id}` KV bucket. Schema:
`{tenant_id, generated_at, task_count_seen, archetype_priors[], tension_patterns[],
retry_hint_patterns[], decomposition_templates[]}`. Published event:
`InductionCycleCompletedEvent` to `h2ai.telemetry.induction`.

New files: `crates/h2ai-orchestrator/src/induction/mod.rs` (trait + mock),
`induction/algorithmic.rs` (distillation), `induction/scheduler.rs` (I/O).
New types: `crates/h2ai-types/src/memory.rs`.

**Phase 3 — Thinking Loop Integration (Layer 3, full).** Before the thinking loop runs, load
`TenantMemoryStore` from `NatsClient::get_tenant_memory(&tenant_id)`. Thread as
`Option<TenantMemoryStore>` into `ThinkingLoopInput`.

- **Archetype priors** — `select_archetypes()` gives +0.15 weight boost to archetypes with
  `net_confidence > 0.6` + matching domain tags; -0.20 penalty to archetypes in `avoid_for_tags`
  matching current task.
- **Tension seeding** — Iteration 0 pre-loaded with top 3 `TensionPattern` entries matching
  constraint tags (Jaccard tag intersection). Injected as hypotheses: "previously observed
  tensions — validate, refute, or refine."
- **Retry hint priming** — `MapeKController::new()` receives `Vec<RetryHintPattern>` matching
  task tags as `primed_retry_hints`. When `ZeroSurvival` or `HallucinationDetected` fires, checks
  `primed_retry_hints` before computing retry context from scratch.

Full Layer 3 is blocked on Layer 2 (AlgorithmicInductionWorker) being live.

Config additions to `reference.toml`: `reasoning_memory_max_archetype_boost = 0.15`,
`reasoning_memory_max_archetype_penalty = 0.20`.

**Phase 4 — Hybrid Retrieval (Layer 4).** Tag-gate (Layer 3 baseline): Jaccard
`|tags_task ∩ tags_pattern| / |tags_task ∪ tags_pattern| ≥ 0.2`. O(1) per candidate — eliminates
irrelevant patterns before embedding work.

Embedding rerank (Layer 4 addition, only when tag-gate returns > 5 candidates): embed current
task description; compute cosine similarity against stored `TensionPattern.embedding`; final score
= `0.6 × jaccard + 0.4 × cosine`. Return top 3. Pattern embeddings precomputed during induction
— no embedding call at query time for stored patterns.

Config additions: `reasoning_memory_tag_gate_threshold = 0.2`,
`reasoning_memory_max_tension_candidates = 3`.

---

## Brainstorm Group H — Skeptical Audit Resilience

---

### GAP-H4: Small-N Human Ratings — MoM ECE Estimator Breaks Below N=50 🔴 OPEN — Medium

**Gap statement.**

The human oracle gateway (`OracleKind::HumanRating`) collects discrete ratings from human
evaluators and feeds them into `EnsembleCalibration` via ECE (Expected Calibration Error)
computation. The ECE estimator uses Method-of-Moments (MoM): it divides predictions into 10 bins
and computes mean confidence vs. mean accuracy per bin. This estimator has well-known breakdown at
small N:
- With N<50 ratings, each bin contains ≤5 samples — variance of the bin mean is O(1/√5) ≈ 45%
- Outlier ratings dominate; one unusual evaluator can flip a bin's calibration signal
- At N<10, MoM produces calibration estimates with confidence intervals wider than the [0,1] scale

For most tenants, human rating volumes will be N=3–30 per constraint domain. The calibration
output from human ratings is essentially noise at these volumes, yet it feeds directly into
`EnsembleCalibration` with the same weight as oracle-grounded accuracy estimates with N=1000+.

**Literature grounding.**

*Dirichlet-Categorical posterior* (Minka, 2000; Gelman et al., 2013) — the correct Bayesian model
for small-N count data on a discrete rating scale. The Dirichlet prior concentrates posterior mass
on the prior mean when N is small, and releases toward the empirical mean as N grows.

*Bayesian average for rating aggregation* (Laplace Smoothing generalization):
```
bayesian_mean = (sum_of_ratings + C × prior_mean) / (N + C)
```
where `C` is the effective prior count. Standard recommendation (MovieLens, Goodreads, IMDb):
`C = sqrt(mean_N)` where `mean_N` is the average rating count across all items.

*SSBC — Small-Sample Bootstrap Calibration* (Bröcker & Smith, 2007, Monthly Weather Review) —
conformal calibration valid down to N=47. The meteorological community standard for rank histogram
calibration with < 50 samples.

*Hybrid weight schedule* — practical recommendation from Bayesian A/B testing literature (Kohavi
et al., 2020): `weight = min(1.0, N / N_effective_min)` where N_effective_min ≈ 15 gives a 95%
credible interval of ±0.25 for a beta-binomial model on pass/fail ratings.

**N threshold tiers.**

| N range | Estimator | Action |
|---|---|---|
| N < 10 | Prior only | Use `human_rating_prior_mean` (configured per tenant); discard sample entirely for calibration update |
| 10 ≤ N < 30 | Bayesian average | `bayesian_mean = (sum + C × prior) / (N + C)`; weight in ECE update = `min(1.0, N / 15)` |
| 30 ≤ N < 50 | SSBC bootstrap | Bootstrap-corrected histogram calibration; credible interval width determines whether the calibration gate opens |
| N ≥ 50 | Standard MoM ECE | Full ECE computation; credible interval from SSBC used as confidence band |

**Credible-interval circuit breaker.**

Do not apply a human rating calibration update if the 95% credible interval of the updated ECE
exceeds `human_rating_max_credible_interval_width` (default: 0.30). This prevents noisy small-N
updates from overwriting calibration built from oracle-grounded data.

**Implementation plan.**

New function in `crates/h2ai-autonomic/src/calibration.rs`:

```rust
pub struct HumanRatingEstimate {
    pub bayesian_mean: f64,
    pub credible_interval_half_width: f64,  // 95% CI half-width
    pub effective_n: f64,                    // N adjusted for prior
    pub weight: f64,                         // weight for ECE update: min(1.0, N/N_min)
    pub estimator_used: HumanRatingEstimator,
}

pub enum HumanRatingEstimator {
    PriorOnly,
    BayesianAverage,
    BootstrapCorrected,
    StandardMoM,
}

pub fn estimate_human_rating(
    ratings: &[f64],          // raw ratings normalised to [0, 1]
    prior_mean: f64,
    prior_count: f64,         // C in Bayesian average formula
    n_min: f64,               // N_effective_min for weight schedule
) -> HumanRatingEstimate { ... }
```

The proposed integration point in `calibration.rs` would receive `HumanRatingEstimate`; apply the
`weight` to the ECE update and skip the update if `credible_interval_half_width >
max_credible_interval_width`. (`update_from_oracle_verdict` is not yet implemented in
`calibration.rs` — this integration is pending.)

**Config additions in `reference.toml`:**

```toml
[calibration]
human_rating_prior_mean = 0.5          # Dirichlet prior centre (0.5 = uninformative pass/fail)
human_rating_prior_count = 5.0         # effective prior observations C
human_rating_n_min = 15.0              # N_effective_min for hybrid weight schedule
human_rating_bootstrap_n_samples = 500 # SSBC bootstrap resamples for N in [30, 50)
human_rating_max_credible_interval_width = 0.30  # circuit breaker: skip update if CI too wide
```

**Test strategy.**
- Unit: `estimate_human_rating` returns `PriorOnly` for N<10, `BayesianAverage` for N=12,
  `BootstrapCorrected` for N=35, `StandardMoM` for N=60
- Unit: Bayesian average formula is mathematically correct against closed-form for 3 specific
  inputs
- Unit: circuit breaker fires when CI width > threshold; calibration update skipped
- Unit: weight = `min(1.0, N/15)` — verifies N=7.5 → 0.5, N=15 → 1.0, N=30 → 1.0
- Integration: simulate 8-rating stream; verify calibration does not update until N reaches 10

**Falsification condition.**
Inject synthetic human rating streams with N=5, N=15, N=40, N=100 and known ground-truth ECE.
If Bayesian estimator does not reduce RMSE vs. MoM for N=15 on 1000 bootstrap trials, the prior
is mis-specified and `human_rating_prior_count` needs tuning. Expected: RMSE(Bayesian) <
RMSE(MoM) for N ≤ 30 by at least 20%.

---

## Brainstorm Group S — Signal Fidelity

---

### GAP-S1: SRANI Fires for Technology-Specific Implementation Details 🟡 PARTIAL — Low

**Status: PARTIAL** — Core infrastructure false-positive (manifest.context exclusion) fixed.
Residual: SRANI still fires for technology-specific sub-terms.

`check_specification_grounding` receives an `effective_spec` built from
`manifest.description + manifest.context + constraint corpus text`. Core infrastructure terms
named in `manifest.context` (Redis, Kafka, ClickHouse, CockroachDB) are grounded; no harmful
"avoid Redis/Kafka" hints are injected.

**Observable trait.** SRANI still emits `shared_ungrounded` entries for technology-specific
sub-terms implied by but not explicitly named in spec/constraints/context. Examples: `"MergeTree"`
(ClickHouse table engine — strongly implied when ClickHouse is grounded in constraint binary
checks), `"BillingEvent"` (proposal-introduced named event type). ResearcherGrounding hint at
these sub-terms may redirect generation away from correct implementation choices: "use standard
idempotency patterns with TTL-based caches" when MergeTree is the correct ClickHouse choice.

**Cross-reference.** GAP-S1 residual is analogous to GAP-F5: constraint-mandated entities should
reinforce grounding signal, not suppress it. See `phases/srani.rs:extract_arch_nouns`.

**Candidate fixes:**
- **Implied-by suppression table:** Extend `extract_arch_nouns()` with a parent→sub-term map.
  When a grounded parent technology implies a sub-term, suppress from the ungrounded set. Example:
  `"ClickHouse"` grounded → suppress
  `{"MergeTree", "ReplacingMergeTree", "SummingMergeTree", "AggregatingMergeTree"}`;
  `"Redis"` grounded → suppress `{"EVAL", "Lua EVAL", "SETEX", "SETNX"}`. Requires maintaining
  the suppression table as the lexicon grows.
- **CFI-gated hint emission:** Suppress ResearcherGrounding hint when CFI < 0.4 (partial
  fabrication with no shared consensus). Simpler but loses information — all sub-term fabrication
  suppressed regardless of whether hints are harmful.

---

## Gap Priority Matrix

| Gap | Core thesis risk | Implementation cost | Data dependency | Suggested order |
|---|---|---|---|---|
| **GAP-F5 Step 3 retroactive induction trigger** | Medium | 3 days | GAP-I1 signal live | Week 2 (unblocked) |
| **GAP-G1 Phase 2 induction (Layer 2)** | Medium | 1 week | None | Week 2 |
| GAP-A1 TCC parameter fitting | Critical | 2 weeks | Oracle quality signal | Session 1 |
| GAP-A1 Full experiment (cross-family Coverage quadrant) | Critical | Timeline open | Second adapter family | Session 2+ |
| GAP-A2 USL quality curve empirical validation | High | 2 weeks | Labeled multi-N benchmark | Session 2 |
| **GAP-F4 Knowledge provider contrastive eval Phase 2** | High | 1 week | 50+ tasks per domain | Week 3 |
| GAP-D2 Compound task HITL escalation | Low | 1 week | None | Any |
| **GAP-H4 Dirichlet human rating posterior** | Medium | 1 week | Human rating data | Week 4 |

---

## Shared Infrastructure Required for Group A

Sessions 1 and 2 block on building a shared measurement harness:

1. **Labeled task set.** 100–200 tasks across code (test oracle), factual QA (reference answers),
   and constraint-heavy reasoning. Stratified by constraint count:
   - Tier 1: 1–2 constraints (simple) — validates B0 baseline
   - Tier 2: 3–5 constraints (moderate) — tests enforcement value
   - Tier 3: 6+ constraints (complex) — primary H2-P vs B3 battlefield

2. **JSON schema + cargo test oracle.** Minimum viable oracle for Session 1. Structured output
   tasks have zero-dependency validation. Code tasks need a ShellExecutor extension.

3. **Per-N quality measurement.** The benchmark harness extended to record oracle pass rate per
   adapter, per N value (2, 3, 5, 7, 9), and per task tier.

4. **Pairwise error correlation logging.** Per-adapter binary correct/incorrect logged per task.
   Stored to SQLite or Parquet for offline ρ analysis and ρ_EMA validation (online ρ EMA is now
   live in `rho_ema.rs`; offline logging enables retrospective validation of its convergence).

Building this harness is the pre-work for Session 1 and is the first concrete deliverable before
any gap-resolution experiments begin.
