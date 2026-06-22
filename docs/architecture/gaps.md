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
| **GAP-G1 Reasoning Memory Phase 4 unimplemented** | 🟡 PARTIAL | Medium | Phase 1 live; Phase 2 complete (2026-06-22); Phase 3 complete (2026-06-22): archetype prior boost/penalty wired in `select_archetypes()`, iteration-0 tension seeding via `select_tension_seeds`/`format_tension_seeds`, `run_distillation_cycle` triggered at `induction_batch_size` intervals in `post_run`, `InductionCycleCompletedEvent` publisher live, `load_semantic_memory` called before thinking loop; Phase 4 pending |
| **GAP-B6 Confidence-blind Byzantine merge kernel** | 🔴 OPEN | Medium | Confidence-weighted Krum/Weiszfeld once empirical score_lower distribution is characterized |
| **GAP-D6 Multi-wave latency and token characterization** | 🔴 OPEN | Low | Fast-path single-shot arm below decompose_threshold; cost-estimate gate before ensemble dispatch |
| **GAP-D7 Gap research confidence ceiling under local LLMs** | 🟡 PARTIAL | Medium | Mechanical-mechanism prompt refactor applied; two-stage extraction + check-text fallback open |
| **GAP-H4 Small-N Human Ratings — MoM ECE breaks below N=50** | 🔴 OPEN | Medium | Dirichlet-Categorical posterior + credible-interval circuit breaker |

**Severity key** — Critical: threatens core thesis validity; High: corrupts math inputs or silently disables documented features; Medium: degrades confidence in results; Low: operational or presentation issue.

---

## Innovations Roadmap

### INNOVATION-5 — Structured Self-MoA Experiment Protocol

**Closes:** GAP-A1 (comparative signal).
**Status: COMPLETE (2026-06-20)** — H2-P achieved MergeResolved on Tier 1 (2 constraints, j_eff=1.000, 2026-06-20), Tier 2 (4 constraints, avg_score=0.750, grounding events=0, 2026-06-20), and Tier 3 (6 constraints, j_eff=0.667 via one MAPE-K retry wave; 1/3 wave-1 proposals at score=1.00 on all 6 constraints, 2026-06-20).

**Reliability finding (e2e analysis, 2026-06-20).** Tier 1 (2 constraints) achieves j_eff=1.000 after framework improvements (corpus-seeded archetypes, ZeroSurvival induction trigger, LLM coverage phase). Tier 2 (4 constraints) reaches avg_score=0.750 with LLM-driven implied entity classification eliminating spurious technology hints. Tier 3 (6 constraints) exhibits a `ZeroSurvival` event in wave 0 (all 3 proposals pruned: 2 violating CONSTRAINT-TAU-2+CONSTRAINT-BFT-1, 1 violating all 6), followed by a MAPE-K retry wave producing 1/3 proposals at score=1.00 (j_eff=0.667). New failure patterns: repair oscillation (wave-1 fix for C-TAU-2/C-BFT-1 caused 2/3 proposals to regress on C-004/C-005/C-008) and no per-constraint archetype guarantee (coverage_score=0.98 but C-TAU-2 had no dedicated archetype in thinking loop iteration 0). Open work: cross-task ArchetypePrior/TensionPattern/DecompositionTemplate (reasoning memory Phase 2 pending).

**Implementation:** Scenario directories `tests/e2e/scenarios/innovation-5/` were archived during the 2026-06-23 scenario consolidation. Current canonical scenarios are under `tests/e2e/scenarios/`.

**Experiment arms:**
- **B3 — Self-MoA baseline:** `baseline.toml`, `max_autonomic_retries = 0`. Three explorers with τ-spread; verifier scores; synthesis selects winner. No MAPE-K enforcement loop.
- **H2-P — Full H2AI pipeline:** `h2ai.toml`, `max_autonomic_retries = 4`. Same three explorers plus MAPE-K retry waves pruning non-compliant proposals and regenerating with repair context.

**Primary metric:** Constraint compliance rate — fraction of `_expected.should_prune` patterns that H2-P rejects and B3 passes through. Secondary: j_eff, verification scores, token cost.

**Open:** A matched B3 run at current code level (post-DPPM+implied-entity-classification+manifest.context) has not been completed. The +41pp H₁ delta is against the original pre-stack B3 baseline.

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

### GAP-B6: Confidence-Blind Byzantine Merge Kernel 🔴 OPEN — Medium

The BFT merge kernel (`krum.rs`, `weiszfeld.rs`) computes proposal consensus using pairwise
semantic distance (embedding cosine similarity). It does not weight proposals by epistemic
confidence — a proposal with `score_lower = 0.12` (high uncertainty, 1/8 binary checks passing)
exerts the same gravitational pull on the Weiszfeld geometric median as a proposal with
`score_lower = 0.84` (well-evidenced, 9/10 checks passing).

**What is live.** `VerificationScoredEvent` carries `score_lower` and `score_upper` (Wilson score
95% credible intervals computed from `passed_checks / total_checks`). These are surfaced in the
event stream for full observability. The gap is between observability and algorithmic consumption:
the merge kernel has access to the confidence signal but does not use it.

**Why stopped here.** Wiring `score_lower` into the merge kernel requires a principled weighting
scheme. The two natural forms — confidence-gated Krum (exclude proposals whose `score_lower` is
below a threshold before ε-neighborhood selection) and confidence-weighted Weiszfeld (weight each
proposal's pull proportional to `score_lower`) — both require tuning a threshold derived from the
empirical distribution of `score_lower` values across the ensemble. Before sufficient data exists
to characterize that distribution, uncalibrated confidence weighting introduces a new bias rather
than removing one.

**Research path.** Collect `VerificationScoredEvent` across Tier-2 and Tier-3 tasks (target: 100+
merge events with `score_lower` populated). Regress oracle pass rate against
`max(score_lower) − min(score_lower)` in the ensemble. If the gap predicts oracle pass rate
(Spearman ρ > 0.3), the simplest principled intervention is a confidence gate before Krum: exclude
proposals with `score_lower < p10_threshold` (10th percentile of historical values) from
neighborhood computation, then apply unweighted Krum on survivors.

**Falsification condition.**
Run 50 Tier-3 tasks with full `score_lower` logging. If `min(score_lower_i)` in the ensemble does
not correlate with oracle pass rate of the Krum-selected proposal (p > 0.05), the confidence signal
adds no information beyond geometric distance and weighting is not justified.

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

### GAP-D6: Multi-Wave Orchestration — Latency and Token Budget 🔴 OPEN — Low

At N=3 explorers, 3 MAPE-K waves, and 6 constraints, the minimum inference call count is
approximately 30–50 LLM calls per task resolution (generation × waves + verification × proposals ×
constraints + synthesis + audit). On local hardware (llama.cpp, 20–40 tokens/s), end-to-end
wall-clock time for a Tier-3 6-constraint task is 15–40 minutes. On frontier API, the token cost
is $0.30–$3.00 depending on context size.

This is a deployment constraint, not a defect. H2AI is architected for high-complexity,
high-stakes tasks (enterprise compliance, distributed systems design, contract analysis) where the
overhead is warranted by the cost of undetected constraint violations. It is structurally
unsuitable for synchronous, latency-sensitive applications (user-facing APIs, real-time chat,
IDE autocomplete).

**Deployment constraints as specified:**

| Dimension | Local LLM (llama.cpp) | Frontier API |
|---|---|---|
| Wall-clock, Tier-3 (6 constraints, 3 waves) | 15–40 min | 2–8 min |
| Token budget, Tier-3 | 100k–200k tokens | 50k–150k tokens |
| Minimum viable (Tier-1, 2 constraints, 1 wave, no retry) | ~4 min | ~30 s |
| Synchronous use | Not supported | Not supported |

**Research path.** Fast-path routing: extend complexity routing with a `single_shot` arm below
`decompose_threshold`. Tasks classified as complexity ≤ 2 (single constraint, high p_mean prior)
bypass ensemble construction and go to a single-adapter direct call. Cost gate: before constructing
the explorer ensemble, emit a `TaskCostEstimateEvent` with projected call count and token budget;
surface to operator for confirmation when projected cost exceeds `cost_gate_threshold`.

---

### GAP-D7: Gap Research Confidence Ceiling Under Local LLMs 🟡 PARTIAL — Medium

**Status: PARTIAL** — Mechanical-mechanism prompt refactor applied; two-stage extraction and
check-text fallback remain open.

The gap research pipeline (`GapI1Config`, `run_gap_researcher()`) fires when a binary check has a
100% historical failure rate. The synthesis validator scores whether the synthesized belief
correction is mechanically actionable. In Tier-3 enterprise enforcement runs with
`synthesis_min_confidence = 0.1`, a local 122B model scored 9/9 synthesis calls below the
threshold. Zero adaptive context was injected into any MAPE-K wave; all waves ran on unmodified
repair context derived only from the current wave's failure signal.

**Root cause.** The synthesis validator asked: "does the correct belief make it structurally
impossible to repeat the wrong belief?" A local model reasoning about multi-constraint enterprise
compliance cannot answer this with >10% confidence — the question requires global counterfactual
reasoning across the full constraint space that is beyond local model capacity.

**What was changed.** `I1_GAP_EXTRACTOR_SYSTEM` now scopes extraction to "the single missing
mechanical mechanism" rather than "the belief gap." `I1_GAP_EXTRACTOR_TASK` requests "a specific
operation, command, or step" rather than "the incorrect concept the author held."
`I1_SYNTHESIS_VALIDATOR_TASK` now asks "does the correct belief name a specific concrete mechanism
that was absent from the failed proposals, and would its inclusion satisfy the constraint check?"
— answerable from local context rather than global counterfactual reasoning.

**Remaining open.** The prompt fix reduces the bar from global-counterfactual to local-mechanical,
which should raise scores above the 0.1 threshold. The structural ceiling is not eliminated — it
is lowered. If scores remain below 0.1, the model cannot even verify mechanical specificity.

**Research path.** Two-stage extraction: (1) extract the missing mechanism (simplified); (2)
validate that the mechanism text appears in the constraint check criteria (substring or semantic
overlap — near model-free). Mechanism extractable from the check text directly sidesteps the model
confidence floor entirely. Fallback: when `score < synthesis_min_confidence` after extraction,
inject the constraint check text verbatim as the repair slot rather than the synthesized belief
— coarser guidance but guaranteed non-empty and always relevant.

**Falsification condition.**
Re-run Tier-3 enterprise enforcement with updated prompts. If `gap_synthesis_confidence` moves
from 0.0–0.04 to ≥ 0.1 on ≥ 5 of 9 calls, the mechanical-mechanism framing resolves the
starvation. If scores remain below 0.1, model capacity is the ceiling and the two-stage validation
path is required.

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

## Brainstorm Group G — Reasoning Memory

---

### GAP-G1: Reasoning Memory Phase 4 Unimplemented 🟡 PARTIAL — Medium

**Status: PARTIAL** — Phase 1 live; Phase 2 complete (2026-06-22); Phase 3 complete (2026-06-22): full Layer 3 wiring live; Phase 4 pending.

**Phase 1 (live).** `TaskReasoningCheckpoint` written at each engine phase gate; `TaskMetaState`
projected at resolution; per-tenant NATS KV buckets (`H2AI_CHECKPOINT_{tenant}` 7d TTL,
`H2AI_META_{tenant}` no TTL). `run_from_checkpoint` reads `CheckpointPhase` to skip completed
phases. Skill extraction provides a depth-stratified analogue: Topic nodes carry Socratic
diagnostic questions + resolution excerpts; Constraint-keyed and Reason-keyed Leaf nodes carry
per-constraint domain signals. `format_induction_priors` formats top-5 `KnowledgeNodePattern`
entries by `hit_rate` as prior context prepended to archetype selection `system_context` (Layer 3
partial path).

**Remaining open — Phase 4:**

**Phase 2 — Induction (Layer 2). COMPLETE (2026-06-22).** Two components with strict separation:
`InductionScheduler` async trait (pure I/O interface, in `crates/h2ai-orchestrator/src/induction/mod.rs`)
and `AlgorithmicInductionWorker` (pure computation, no LLM calls, in `induction/algorithmic.rs`).
`NatsInductionScheduler` (`induction/nats_scheduler.rs`) owns NATS KV reads/writes with CAS-swap
(`kv.entry()` for revision, `kv.update()` for CAS); full-jitter backoff (base=5ms, cap=500ms, max 5
retries); `without_nats()` fallback for tests.

**What is live:** `RetryHintPattern` G-Counter (`trigger_tags`, `exit_reason_kind`, `hint_text`,
`success_count`/`attempt_count` u64, `success_rate()`, `merge_counts()`) in
`crates/h2ai-types/src/memory.rs`. `TenantMemoryStore` (tenant_id, generated_at, task_count_seen,
`retry_hint_patterns`, `archetype_priors`, `tension_patterns`, `decomposition_templates` — new fields
use `#[serde(default)]` for backward compat with stored JSON). `AlgorithmicInductionWorker` filters
stored patterns by tag overlap with `InductionContext.task_class_tags` (trigram-shingle Jaccard ≥ threshold),
sorts by `success_rate()` descending, returns top patterns as `InductionResult`. Trigram shingling pure
functions: `normalize_for_shingling`, `trigram_shingles`, `jaccard_shingles`, `cluster_by_similarity`.

**Semantic distillation data layer (complete 2026-06-22):** Three new types (`ArchetypePrior`, `TensionPattern`,
`DecompositionTemplate`) and `DistillationResult` (`empty()`, `is_empty()`). Pure distillation functions:
`distill_archetype_priors` (groups `ArchetypeResult` by name; `net_confidence` = unweighted mean;
`avoid_for_tags` populated when `sample_count ≥ MIN_SAMPLE_COUNT_FOR_AVOID=3 && net_confidence < 0.4`),
`distill_tension_patterns` (trigram Jaccard clustering at `TENSION_CLUSTER_THRESHOLD=0.6`; pre-computed
shingles), `distill_decomposition_templates` (groups by `(quadrant, sorted constraint_tags)`; picks
lowest-`retry_count` member for `shared_understanding`). `InductionScheduler` trait extended with
`run_distillation_cycle` and `load_semantic_memory` (default no-ops — all existing impls compile unchanged).
`AlgorithmicInductionWorker` implements both (in-memory from stored `TenantMemoryStore` fields).
`NatsInductionScheduler` implements both: `run_distillation_cycle` distills and persists `DistillationResult`
to `{tenant_id}.semantic` in `H2AI_MEMORY` KV; `load_semantic_memory` reads it back. `InductionCycleCompletedEvent`
struct added to `h2ai-types::events`.

**Pending (requires engine wiring in Phase 3):** `run_distillation_cycle` is not yet called by the engine
(`induction_batch_size` / `induction_max_interval_secs` trigger not wired). `InductionCycleCompletedEvent`
has no publisher. Archetype prior boost/penalty and tension seeding require `load_semantic_memory` integration
in `task_pipeline.rs` — these are Phase 3 wiring tasks, now unblocked.

**Phase 3 — Thinking Loop Integration (Layer 3, full). COMPLETE (2026-06-22).**
All three full Layer 3 paths are now live:

- **Archetype priors** — `select_archetypes()` in `thinking_loop.rs` applies `+max_archetype_boost`
  (default 0.15) to archetypes whose `ArchetypePrior.net_confidence > 0.6` with matching domain
  tags; `-max_archetype_penalty` (default 0.20) when `avoid_for_tags` overlaps current constraint
  tags. Adjustments clamp to [0.0, 1.0].
- **Tension seeding** — `select_tension_seeds` computes trigram Jaccard between joined
  `constraint_tags` and `TensionPattern.shingles`; top 3 matches (similarity ≥ 0.05) formatted
  via `format_tension_seeds` and appended to `research_context` at iteration 0. Archetypes and
  brainstorm prompts receive the "PREVIOUSLY OBSERVED TENSIONS — validate, refute, or refine" block.
- **Distillation trigger** — `post_run` increments `TenantState::resolved_task_count`
  (per-tenant `AtomicUsize`). When `count % induction_batch_size == 0`, a background tokio task
  calls `run_distillation_cycle` with up to `induction_max_tasks_per_run` recent `TaskMetaState`
  records and publishes `InductionCycleCompletedEvent`.
- **Semantic memory load** — `task_pipeline.rs` calls `load_semantic_memory` after building the
  induction scheduler (when `reasoning_memory.enabled`). The `DistillationResult` propagates
  through `ThinkingLoopArgs` → `ThinkingLoopInput` to `select_archetypes` and the tension-seed
  block builder.

**Phase 4 — Hybrid Retrieval (Layer 4).** Tag-gate (Layer 3 baseline): Jaccard
`|tags_task ∩ tags_pattern| / |tags_task ∪ tags_pattern| ≥ 0.2`. O(1) per candidate — eliminates
irrelevant patterns before embedding work.

Embedding rerank (Layer 4 addition, only when tag-gate returns > 5 candidates): embed current
task description; compute cosine similarity against stored `TensionPattern.embedding`; final score
= `0.6 × jaccard + 0.4 × cosine`. Return top 3. Pattern embeddings precomputed during induction
— no embedding call at query time for stored patterns.

Config additions: `reasoning_memory_tag_gate_threshold = 0.2`,
`reasoning_memory_max_tension_candidates = 3`.

**E2E run findings.** Across innovation-5 Tier-2 runs, j_eff is invariably 0.667 on every
successful task (exactly 1-of-3 explorers passes stochastically on wave 1). No run shows j_eff
improving across MAPE-K retry waves — when wave 1 fails entirely, subsequent waves fail at the
same rate and the task terminates via `TaskFailed`. This is consistent with the absence of Phase
2: without `AlgorithmicInductionWorker` distilling `RetryHintPattern` records from prior
`BranchPruned` history, `MapeKController` has no primed hints and constructs retry context from
scratch each wave using only the current wave's failure signal. Phase 2 is the mechanism that
turns MAPE-K from random restarts into directed repair: `RetryHintPattern` entries for
`(trigger_tags=["billing", "audit-log"], exit_reason_kind=ZeroSurvival)` would directly prime the
retry context for the CONSTRAINT-005 failure pattern present in every failed Tier-2 run.

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

## Gap Priority Matrix

### Dependency Graph

```
Shared Infrastructure (labeled task set + oracle + per-N harness + ρ logging)
├── unblocks → GAP-A1  (TCC fitting needs oracle quality signal)
├── unblocks → GAP-A2  (empirical N_IT vs. quality curve)
├── unblocks → GAP-B3  (conformal prediction needs oracle-grounded calibration data)
├── unblocks → GAP-B5  (cold-start prior 0.45 validation needs held-out benchmark)
└── unblocks → GAP-F4 Phase 3  (REINFORCE graph weights need calibrated oracle — via GAP-B3)

GAP-B3 (oracle data sufficient)
└── unblocks → GAP-F4 Phase 3

GAP-G1 Phase 2 COMPLETE (2026-06-22) — unblocks:
└── GAP-G1 Phase 3 full path (archetype prior boost/penalty + tension seeding engine wiring) — COMPLETE (2026-06-22)

Data accumulation — passive (tasks run, events emitted automatically):
  30+ task observations   → unblocks GAP-B5 ρ_EMA replaces cold-start prior 0.45
  30+ task observations   → unblocks GAP-B1 empirical regression of conflict_rate vs CG_mean
  50+ tasks per domain    → unblocks GAP-F4 Phase 2 (Thompson Sampling bandit)
  100+ merge events       → unblocks GAP-B6 (characterize score_lower distribution)

No upstream blockers (can start immediately):
  GAP-D7  falsification run (Tier-3 re-run with updated prompts)
  GAP-G1  Phase 4 (embedding rerank) — hybrid retrieval; deferred until Phase 3 shows signal
  GAP-H4  implementation (pure math + unit tests, no data needed)
  GAP-D2  HITL escalation + complexity bandit
  GAP-D6  fast-path routing + cost gate
  GAP-A2  Monte Carlo simulation path (Gaussian copula, no oracle needed)
  GAP-B5  sensitivity analysis script (computation only, rho=CG vs rho=1-CG)

Cannot start (hard external blockers):
  GAP-A1 full experiment   → second model family unavailable
```

### What To Start Next

| # | Gap | Action | Rationale | Unblocks |
|---|---|---|---|---|
| **1** | **GAP-D7** | Re-run Tier-3 enterprise enforcement scenario | Validates already-deployed prompt fix in ≤1 hour; determines whether two-stage extraction is needed or gap is resolved | Closes or scopes remaining D7 work |
| **2** | ~~**GAP-G1 Ph.3 (wiring)**~~ | ~~Wire `run_distillation_cycle` call into engine on `induction_batch_size` trigger; add archetype prior boost/penalty (+0.15/−0.20) to `select_archetypes()`; seed iteration 0 with top-3 matching `TensionPattern` entries~~ | **COMPLETE (2026-06-22)** | Turns cross-task ArchetypePrior/TensionPattern into live per-task MAPE-K guidance |
| **3** | **Shared Infra** | Build labeled task set + JSON/cargo oracle + per-N harness + ρ logging | Single deliverable that simultaneously unblocks 4 critical gaps (A1, A2, B3, B5); without it the core thesis cannot be validated | A1, A2, B3, B5, F4-Ph3 |
| **4** | **GAP-H4** | Implement `estimate_human_rating` (Bayesian avg + SSBC bootstrap + CI circuit breaker) in `calibration.rs` | Fully self-contained; complete spec with test strategy already written; no data needed to implement and unit-test | Closes H4 on first merge |

### Wait-For-Data Queue (implement when threshold reached)

| Gap | Trigger | What to build |
|---|---|---|
| GAP-B5 | 30 task observations (ρ_EMA live) | Validate ρ_EMA convergence; run sensitivity script to pick ρ=CG vs ρ=1-CG |
| GAP-B1 | 30 task observations | Regress `conflict_rate` vs `(1−CG_mean)` and `(1−CG_mean)×fill(N)`; promote context-aware β_eff if R²_bivariate > R²_univariate |
| GAP-F4 Ph.2 | 50 tasks per domain | Thompson Sampling bandit: `(domain_tag, knowledge_on)` vs `(domain_tag, knowledge_off)` arms |
| GAP-B6 | 100 merge events with `score_lower` | Spearman ρ of `max−min(score_lower)` vs oracle pass rate; if ρ > 0.3 implement confidence gate before Krum |
| GAP-B3 | Oracle data sufficient | Conformal prediction margin over calibration split |
| GAP-F4 Ph.3 | GAP-B3 closed | REINFORCE graph edge weight updates (requires calibrated oracle signal) |

### Deferred (external blocker)

| Gap | Blocker | Resume condition |
|---|---|---|
| GAP-A1 TCC fitting | Oracle quality signal (Shared Infra) | After Shared Infra + first labeled task set run |
| GAP-A1 full experiment | Second model family unavailable | When second adapter family is accessible |
| GAP-A2 empirical | Labeled multi-N benchmark (Shared Infra) | After Shared Infra; Monte Carlo simulation runnable now as approximation |
| GAP-D2 | None — low priority | Any sprint |
| GAP-D6 | None — low priority | Any sprint |

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
