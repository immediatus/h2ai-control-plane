# H2AI Gaps — Research and Engineering Agenda

This document is the actionable companion to [`research-state.md`](research-state.md). Every gap
is a falsifiable question with a concrete research or engineering path. Where previous editions
listed status only, this edition adds: literature grounding, innovative solution design,
mathematical improvement, and simulation protocol for every open gap.

---

## Navigation

| Section | What it covers |
|---|---|
| [Problem Space Map](#problem-space-map) | At-a-glance status and severity for all gaps |
| [Innovations Roadmap](#innovations-roadmap) | Cross-cutting solutions that close multiple gaps simultaneously |
| [Group A — Core Thesis](#brainstorm-group-a--core-thesis-validity) | Does the fundamental approach work and beat its competitors? |
| [Group B — Math Apparatus](#brainstorm-group-b--mathematical-formula-validity) | Are the formulas principled or arbitrary? |
| [Group D — Infrastructure](#brainstorm-group-d--infrastructure-and-operational-gaps) | Do the inputs to the math arrive correctly? |
| [Group E — Quality Measurement](#brainstorm-group-e--quality-measurement-infrastructure) | Can we measure what we claim to optimise? |
| [New Gaps](#new-gaps-from-2026-05-13-review) | Newly identified from critical review and arXiv research |
| [Shared Infrastructure](#shared-infrastructure-required-for-group-a) | Pre-work that blocks Group A experiments |

---

## Problem Space Map

| Gap | Status | Severity | Innovation opportunity |
|---|---|---|---|
| **GAP-A1 Self-MoA vs. multi-family routing** | 🟡 PARTIAL | **Critical** | H2-P vs. B3 experiment runnable today |
| **GAP-A2 USL N_max vs. quality curve** | 🔴 OPEN | **Critical** | Replace USL with N_IT as primary sizer; USL → cost cap only |
| **GAP-A6 Self-MoA as direct empirical competitor** | 🔴 OPEN | **Critical** | Structured experiment with constraint compliance as primary metric |
| **GAP-A7 Preference leakage in LlmJudge** | ✅ CLOSED 2026-05-16 | **High** | Cross-family panel + persona-diverse fallback; `ConstraintAmbiguityEvent` corpus quality signal |
| **GAP-B1 β_eff functional form** | 🔴 OPEN | Medium | First-principles derivation now available — unify with context-aware formula |
| GAP-B3 Attribution self-referential | 🟡 PARTIAL | Medium | Conformal prediction once oracle data exists |
| **GAP-B5 Proxy chain — rho_mean, p_mean, β_eff unvalidated** | 🔴 OPEN | **High** | Three interconnected heuristic proxies; online ρ_EMA mitigates rho after 30 obs |
| **GAP-B6 LLM-as-Judge validity — Krum on biased scores** | 🔴 OPEN | **High** | Pairwise ranking + adversarial critique; oracle calibration (blocked on GAP-A6) |
| **GAP-C5 Krum breakdown under majority correlated hallucination** *(new)* | 🔴 OPEN | **High** | Multi-family committee enforcement on ModeCollapse retry; structural pre-generation family diversity gate |
| **GAP-D4 Verification throughput** | ✅ CLOSED 2026-05-18 | — | Already parallel; rubric threshold bug fixed |
| **GAP-D5 Synthesis merge bottleneck — single sequential merge** | 🔴 OPEN | Medium | Hierarchical tournament merge; bounded context |
| **GAP-D6 State infrastructure complexity mismatch** | 🔴 OPEN | Low | BFT/CRDT overhead in single-region deployments |
| GAP-D2 Compound task cost unconstrained | 🔴 OPEN | Low | Complexity bandit probe |
| GAP-E1 Oracle integration | 🟢 WIRED | Blocking | Phase 4.5 gate wired; domain-specific automated oracles remaining |
| **GAP-E2 Talagrand feedback loop** | 🔴 OPEN | Medium | τ-spread KL update rule |
| **GAP-F1 Knowledge provider wired into generation pipeline** | ✅ CLOSED 2026-05-18 | — | Role-stratified retrieval via `KnowledgeProfile`/`profile_for_role` (Coordinator → CollapsedTree/top_k=3; Executor/Evaluator → TreeTraversal+PPR; Synthesizer → CollapsedTree+1 PPR hop); parallel per-slot queries in `generation.rs` Phase B1; `InductionStore` records high-hit-rate node IDs on `MergeResolved` and injects as `explicit_ids` on subsequent tasks; wired into `EngineInput.induction_store`; **partial**: `tasks.rs` and `recovery.rs` pass `induction_store: None` (follow-up: wire through AppState); graceful degradation when `[knowledge]` absent |
| **GAP-F2 ResumeSignal — JetStream HITL** | ✅ CLOSED 2026-05-19 | — | Replaces polling-based KV approval entirely. `H2AI_APPROVALS` KV + `approval_reaper.rs` deleted; `H2AI_SIGNALS` JetStream stream + durable per-task push consumers replace them. Engine parks at Merging phase with `tokio::select!` on signal or adaptive timeout; `POST /signal` (202 fire-and-forget) is the new operator endpoint; `POST /approve` → 301 redirect to `/signal`. Two signal types: `Approve` (HITL gate resolution) and `WaveContinue` (grounding injection at wave boundaries when `signal_wave_window_ms > 0`). Adaptive timeout decay: `effective_ms = timeout_ms × decay^hitl_timeouts_fired`, floored at `timeout_floor_ms`. |
| **GAP-F3 Wiki YAML generation tooling absent** | 🔴 OPEN | Low | `wiki/` subdirectory schema is defined and loaded by `YamlDirSource`; no CLI or LLM-assisted tooling exists to generate `wiki/<topic>.yaml` files from a constraint corpus |

**Severity key** — Critical: threatens core thesis validity; High: corrupts math inputs or silently disables documented features; Medium: degrades confidence in results; Low: operational or presentation issue.

**Infrastructure note (2026-05-14):** Delta checkpoint encoding (JSON Patch RFC 6902, `CheckpointKind::Base/Delta`, O(N) NATS KV storage) is now live in `h2ai-state`. Previously O(N²) checkpoint growth would have exhausted NATS KV space during the long multi-task experiment runs required by GAP-A6 and GAP-A1. This blocker is resolved; experiment runs are no longer storage-constrained.

**Infrastructure note (2026-05-15):** Persistent Reasoning Memory Phase 1 is live: `TaskReasoningCheckpoint` written at each engine phase gate, `TaskMetaState` projected at resolution, per-tenant NATS KV buckets (`H2AI_CHECKPOINT_{tenant}` 7d TTL, `H2AI_META_{tenant}` 90d TTL). Crash recovery: `run_from_checkpoint` reads `CheckpointPhase` to skip completed phases. Phases 2–4 are designed and pending implementation.

**Reasoning Memory — Pending Phases 2–4 Design**

| Phase | Layers | Value |
|-------|--------|-------|
| Phase 2 | Induction | First induction cycle; `TenantMemoryStore` populated from `TaskMetaState` history |
| Phase 3 | Thinking loop integration | Archetype priors + tension seeding; fewer iterations needed |
| Phase 4 | Hybrid retrieval | Embedding rerank on top of tag-gate; precision at scale |

**Layer 2 — Induction**

Two components with strict separation: `InductionWorker` trait (pure computation — no I/O, testable with `MockInductionWorker`) and `InductionScheduler` (owns JetStream subscription, NATS KV reads/writes, CAS swap).

`InductionScheduler` triggers a distillation cycle when ≥ `induction_batch_size` (10) resolved tasks accumulate, or `induction_max_interval_secs` (86400s) have elapsed. It loads up to `induction_max_tasks_per_run` (50) `TaskMetaState` records, calls `worker.distill()`, writes result to a staging key, then CAS-swaps `latest` only on full success — the previous snapshot is never touched on failure.

`AlgorithmicInductionWorker` — pure Rust, no LLM calls. Distillation steps:
1. **ArchetypePrior** — group `ArchetypeResult` entries by `archetype_name + domain_tags`; `net_confidence = weighted_mean(confidence, weight=2.0 if dominated_synthesis else 1.0)`; `avoid_for_tags` = tags where `net_confidence < 0.4` across ≥ 3 tasks.
2. **TensionPattern** — collect all tension strings; cluster by cosine similarity (threshold 0.85) if `EmbeddingModel` is available, exact dedup otherwise; store `frequency` + `resolution_hint` from tasks that resolved the tension.
3. **RetryHintPattern** — group `(trigger_tags, exit_reason_kind, retry_context_that_resolved)` tuples; keep top hint per pair by `success_rate`.
4. **DecompositionTemplate** — group `shared_understanding` strings by `(quadrant, constraint_tags)`; select embedding centroid if model available, most recent otherwise.

`TenantMemoryStore` lives in `H2AI_MEMORY_{tenant_id}` KV bucket. Schema: `{tenant_id, generated_at, task_count_seen, archetype_priors[], tension_patterns[], retry_hint_patterns[], decomposition_templates[]}`. Published event: `InductionCycleCompletedEvent` to `h2ai.telemetry.induction`.

New files: `crates/h2ai-orchestrator/src/induction/mod.rs` (trait + mock), `induction/algorithmic.rs` (distillation), `induction/scheduler.rs` (I/O). New types: `crates/h2ai-types/src/memory.rs`.

**Layer 3 — Thinking Loop Integration**

Before the thinking loop runs, load `TenantMemoryStore` from `NatsClient::get_tenant_memory(&tenant_id)`. Thread as `Option<TenantMemoryStore>` into `ThinkingLoopInput`.

- **Archetype priors** — `select_archetypes()` gives +0.15 weight boost to archetypes with `net_confidence > 0.6` + matching domain tags; -0.20 penalty to archetypes in `avoid_for_tags` matching current task.
- **Tension seeding** — Iteration 0 is pre-loaded with top 3 `TensionPattern` entries matching constraint tags (Jaccard tag intersection). Injected as hypotheses: "previously observed tensions — validate, refute, or refine."
- **Retry hint priming** — `MapeKController::new()` receives `Vec<RetryHintPattern>` matching task tags stored as `primed_retry_hints`. When `ZeroSurvival` or `HallucinationDetected` fires, checks `primed_retry_hints` before computing retry context from scratch.

**Layer 4 — Hybrid Retrieval**

Tag-gate (Layer 3 baseline): Jaccard `|tags_task ∩ tags_pattern| / |tags_task ∪ tags_pattern| ≥ 0.2`. O(1) per candidate — eliminates irrelevant patterns before any embedding work.

Embedding rerank (Layer 4 addition, only when tag-gate returns > 5 candidates): embed current task description; compute cosine similarity against stored `TensionPattern.embedding`; final score = `0.6 × jaccard + 0.4 × cosine`. Return top 3. Pattern embeddings are precomputed during induction — no embedding call at query time for stored patterns.

Config additions to `reference.toml`: `reasoning_memory_tag_gate_threshold = 0.2`, `reasoning_memory_max_tension_candidates = 3`, `reasoning_memory_max_archetype_boost = 0.15`, `reasoning_memory_max_archetype_penalty = 0.20`.

---

## Innovations Roadmap

Three cross-cutting innovations that each close multiple gaps without requiring new infrastructure.
Implement these before running any Group A experiments — the experiments will produce better-
grounded data if the math inputs are correct.

### INNOVATION-2 — Conflict-Count β₀ (replaces API-latency β)

**Closed:** GAP-D1 (2026-05-15). See `docs/architecture/reference.md → Conflict-Rate β (GAP-D1)`.  
**Cost:** 1 week to extend calibration harness.  
**Why now:** The constraint verifier already runs on every proposal in Phase B calibration. Counting
constraint violations per proposal pair is free — the data already flows through.

**Current state (wrong signal):**

```
β₀ = (z_M - z_2 × (M-1)) / ((M-1)(M-2))   where z = wall-clock time
```

Fast API (local LLM) → small z → small β₀ → large N_max.  
This is backwards: local single-model deployments produce the most correlated proposals and need
the *smallest* N_max, not the largest.

**Proposed state (principled signal):**

USL's β is defined as the cost of N(N-1) pairwise coherency checks. In H2AI, a coherency check
is a constraint conflict resolution. The coordination cost between agents i and j is proportional
to the number of constraints where they disagree.

```
conflict_rate(M) = (1/M(M-1)) × Σ_{i≠j} |constraints_violated_by_i XOR violated_by_j| / |corpus|

β_conflict = (conflict_rate(M) - conflict_rate(2)) / ((M-1)(M-2))
```

Two β values in parallel:
- `beta_latency` — existing timing β (useful for latency estimation only)
- `beta_quality` — new conflict-count β (drives N_max for quality bounding)

N_max uses `beta_quality`. Latency prediction uses `beta_latency`. These are separate concerns and
should never share the same estimate.

**First-principles derivation of β_eff:**

Given `beta_quality`, the β_eff formula now has a derivation:

```
β_eff = β_quality × (1 - CG_mean)
```

Interpretation: `β_quality` is the conflict cost per pair per unit CG distance.
`(1 - CG_mean)` is the expected CG distance between any two adapters in the pool.
Product: expected conflict cost per adapter pair in the actual pool.

This is not a heuristic — it follows from the definition if `conflict_rate ∝ (1 - CG)`. That
proportionality is an empirical claim to verify, but it is now falsifiable: measure
`conflict_rate` and `CG_mean` independently and fit the linear coefficient.

**Simulation:**

```python
# Validate: does constraint conflict rate scale linearly with (1 - CG_mean)?
# Use synthetic binary satisfaction vectors
import numpy as np

def conflict_rate(profiles):
    """profiles: (N, K) binary array where profiles[i,k] = agent i satisfies constraint k"""
    N, K = profiles.shape
    conflicts = 0
    for i in range(N):
        for j in range(i+1, N):
            conflicts += np.sum(profiles[i] != profiles[j]) / K
    return conflicts / (N * (N-1) / 2)

def hamming_cg(profiles):
    N, K = profiles.shape
    agree = 0
    for i in range(N):
        for j in range(i+1, N):
            agree += np.sum(profiles[i] == profiles[j]) / K
    return agree / (N * (N-1) / 2)

# Sweep CG_mean by varying cross-profile overlap probability p_agree
results = []
for p_agree in np.linspace(0.3, 0.95, 15):
    profiles = (np.random.rand(6, 100) < p_agree).astype(int)
    # Introduce controlled disagreement: flip with prob (1-p_agree)
    for i in range(1, 6):
        mask = np.random.rand(100) < (1 - p_agree)
        profiles[i] = np.where(mask, 1 - profiles[0], profiles[0])
    cg = hamming_cg(profiles)
    cr = conflict_rate(profiles)
    results.append((cg, cr))
    print(f"CG_mean={cg:.3f}, conflict_rate={cr:.3f}, 1-CG={1-cg:.3f}")

# Fit: conflict_rate ~ β₀ × (1 - CG_mean)
cgs, crs = zip(*results)
from numpy.polynomial import polynomial as P
coef = np.polyfit([1 - c for c in cgs], crs, 1)
print(f"Fit: conflict_rate = {coef[0]:.3f} × (1 - CG) + {coef[1]:.4f}")
print(f"If R² > 0.95, the linear form is validated.")
```

### INNOVATION-4 — N_IT as Primary Sizer; USL as Cost Cap

**Closes:** GAP-A2 (reframes rather than abandons USL).  
**Cost:** 1 week (routing logic change; no new math).  
**Why now:** `n_it_optimal` is already implemented in math.md §5.1 and matches `condorcet_n_optimal`
within ±1 for ρ ∈ [0.3, 0.95]. The information-theoretic framing is self-contained and valid
without the USL domain-transfer assumption. USL's role should be cost cap, not quality target.

**Current state:**

```
target N = n_optimal (Condorcet, maximises marginal Q gain)
ceiling N = N_max (USL, throughput model applied to quality)
```

Both are "quality" functions. USL's quality claim is unvalidated (no arXiv paper applies USL to
LLM agent ensembles — confirmed by our search). n_it_optimal is derived from independent
information theory and is correct on its own terms.

**Proposed state:**

```
target N = n_it_optimal = ceil(log(0.5) / log(1 - ρ))   [information-theoretic target]
ceiling N = N_max_USL                                     [cost ceiling, not quality target]
final N   = min(target_N, ceiling_N, calibration_max_ensemble_size)
```

The USL ceiling prevents runaway token cost when ρ is underestimated (which it currently is due
to the proxy chain). It is honest as a cost heuristic. It is not honest as a quality predictor.
Documents and code should describe N_max as a cost cap explicitly.

**Why information-theoretic N is sound:**

Marginal information gain of agent k in an N-agent ensemble with pairwise correlation ρ:

```
I_k = H(X) × (1 - ρ)^(k-1)     [geometric decay due to shared information]
```

Summing: total information = H(X) × (1 - (1-ρ)^N) / ρ

N_IT = min N: (1-ρ)^(N-1) < 0.5  →  N_IT = ceil(log(2) / log(1/(1-ρ)))

This is the point where the marginal information drops below half the per-agent entropy. Adding
more agents beyond this gives diminishing returns regardless of cost.

**Simulation — N_IT vs N_max across ρ values:**

```python
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

alpha = 0.12
beta0 = 0.039

def n_max_usl(cg_mean):
    beta_eff = beta0 * (1 - cg_mean)
    return max(1, round(np.sqrt((1 - alpha) / beta_eff)))

def n_it_optimal(rho):
    if rho <= 0: return 100
    return max(1, int(np.ceil(np.log(0.5) / np.log(max(1e-9, 1 - rho)))))

rho_values = np.linspace(0.05, 0.90, 50)
cg_values  = 1 - rho_values    # rho_proxy = 1 - CG_mean

n_usl = [n_max_usl(cg) for cg in cg_values]
n_it  = [n_it_optimal(rho) for rho in rho_values]

print("rho | N_IT | N_max_USL | gap")
for rho, n_i, n_u in zip(rho_values[::5], n_it[::5], n_usl[::5]):
    print(f"{rho:.2f} | {n_i:4d} | {n_u:9d} | {n_u - n_i:+4d}")
# Expected: N_max_USL >> N_IT for low ρ; USL is too permissive
```

---

### INNOVATION-5 — Structured Self-MoA Experiment Protocol

**Closes:** GAP-A6.  
**Cost:** 1 week (single-model, runnable today on devcontainer).  
**Why now:** The H2-P vs. B3 comparison does not require a second model family. It is the minimum
viable falsification of H2AI's core claim and can run immediately.

**Experiment design (phases):**

**Phase 0 — Baseline (B0):** Single shot, no retries.
**Phase B3 — Self-MoA:** Same model, τ ∈ {0.2, 0.7, 1.0}, 3 proposals, majority-vote on
  verification score, no constraint enforcement loop, no Phase 0 decomposition.
**Phase H2-P — H2AI Precision:** Same model, τ-spread, Phase 0 decomposition, MAPE-K constraint
  enforcement, phase 3.5 rubric scoring, full pipeline.

Primary metric: **constraint compliance rate** (fraction of constraint checks passed in oracle
evaluation — NOT just internal verifier score). Secondary: oracle pass rate on code tasks.

Key prediction: H2-P > B3 on 5+ constraint tasks even when the model is the same, because:
1. Phase 0 decomposition focuses each slot on a distinct constraint domain
2. MAPE-K enforcement iteratively raises constraint satisfaction
3. Self-MoA (B3) has no enforcement loop — it produces higher diversity but no constraint gate

If H2-P ≤ B3 on constraint-heavy tasks, the MAPE-K enforcement adds nothing above temperature
diversity and the core claim fails. Document that outcome and pivot to positioning H2AI as
"structured Self-MoA with better observability."

**Task selection:** S2 HumanEval set stratified by constraint count:
- Tier 1: 1–2 constraints (simple)
- Tier 2: 3–5 constraints (moderate)
- Tier 3: 6+ constraints (complex)

Hypothesis: H2-P advantage over B3 should grow with tier (more constraints = more MAPE-K value).
If advantage is flat across tiers, the hypothesis is wrong.

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
| Phase 6 oracle (GAP-E1) | **Grounding** — connect beliefs to external reality |
| Calibration | Update meta-beliefs about team epistemic capabilities |

### The epistemological traditions each gap violates

| Gap | Epistemic violation |
|---|---|
| GAP-A6: Self-MoA as competitor | Core diversity premise is empirically untested |
| GAP-E1: no oracle | Epistemic closure — internally coherent but ungrounded |
| GAP-B3: attribution without oracle | Cannot distinguish confident-and-correct from confident-and-wrong |
| GAP-B5: proxy chain | Three proxies (rho_mean, p_mean, β_eff) all use CG_mean as input with no empirical validation; cold-start prior 0.45 unvalidated |
| GAP-B6: judge validity | Krum and verifier consensus depend on LLM judge scores; judge bias (verbosity, self-preference) corrupts outlier rejection |

### Stopping criteria

| Loop | Current criterion | Principled criterion | Gap |
|---|---|---|---|
| TAO inner | `agent_max_tool_iterations` (budget) | No productive hypothesis extensions remain | Budget is proxy for epistemic exhaustion |
| MAPE-K retry | Proposals satisfy threshold OR retries exhausted; ZeroSurvival + is_closed() gate | Coherent closure: no active constraint violated, no domain uncovered | Quality threshold is rubric-coherent, not oracle-grounded |
| Calibration | Startup-automatic + POST /calibrate | Confidence intervals narrow enough for decision quality required | — |
| Oracle grounding | Phase 4.5 gate wired (NATS request/reply, `OracleGateConfig`); thinking loop Stage 2 inline oracle; `PendingClarificationEvent` suspension via `clarification_waiters` | All load-bearing beliefs grounded in at least one oracle test | GAP-E1: domain-specific automated test suites remaining |

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

**Innovative solution (available today).**
Run INNOVATION-5 (H2-P vs. B3 experiment protocol). H2-P is the minimum viable falsification:
same model, τ-spread, Phase 0 decomposition, full constraint enforcement loop. This can run on
the devcontainer today. The result either validates H2AI's Phase 0 + MAPE-K claim or reveals that
constraint enforcement adds nothing above temperature diversity.

**Falsification condition.**
If H2-P ≤ B3 on Tier 3 tasks (6+ constraints) with oracle pass rate as the signal (not internal
verifier score), the Coverage routing adds cost without benefit and should be replaced by Precision
routing universally.

---

### GAP-A2: USL N_max vs. Actual Quality Curve 🔴 OPEN — **Critical**

**Gap statement.**
`N_max = round(√((1−α)/β_eff))` is derived from USL's throughput model by setting `dX/dN = 0`.
USL models CPU/network throughput under parallelism — not output quality. No published arXiv paper
applies USL to LLM multi-agent quality ceiling. This is confirmed by our search: zero results for
`abs:"universal scalability law" AND abs:agent`. The domain transfer assumption is unvalidated.

Additionally, USL's ceiling and the Condorcet n_optimal serve different purposes: USL caps cost;
Condorcet maximises quality-per-agent. Using a cost model as a quality predictor can cause
over-sizing (high ρ → USL gives high N_max because β is suppressed, but Condorcet gives low N
because diversity is poor) or under-sizing (low ρ → large N_IT but USL may cap lower than optimal).

**Literature grounding.**
arXiv 0808.1431 (Gunther, 2008) — foundational USL derivation. β is defined as the "coherency
cost" — serialisation overhead per N(N-1) pairwise checks. Applicable to compute platforms;
domain transfer to quality requires explicit argument.

arXiv 2006.04969 (Hamann & Reina, 2020) — applies USL and Amdahl's law to robot swarms: the
closest bridge to autonomous agent systems. Finds USL describes swarm throughput; quality metrics
(task success rate) follow different scaling laws.

arXiv 2509.19489 (Nowak, 2025) — derives optimal compute allocation for self-consistency under
budget B = m×n: optimum is m,n ∝ √B, not n→∞. Directly constrains the ensemble budget
allocation, independent of USL.

**Innovative solution — INNOVATION-4: N_IT as primary sizer.**

Promote `n_it_optimal` to primary N recommendation. Demote N_max_USL to cost cap. The
information-theoretic formula has a self-contained derivation (see INNOVATION-4 above) and does
not require the USL domain-transfer assumption. In code: rename or document N_max as
`n_max_cost_ceiling` and adjust the planning logic:

```rust
let n_target = calibration.n_it_optimal(rho_mean);   // info-theoretic target
let n_max    = calibration.n_max();                   // USL cost ceiling
let n_final  = n_target.min(n_max).min(cfg.calibration_max_ensemble_size);
```

Document USL explicitly as: "N_max is a cost heuristic drawn from throughput engineering. It
prevents runaway token cost but is not a quality predictor. The quality target is n_IT."

**Simulation — validate N_IT vs empirical quality curve:**

```python
# Monte Carlo: quality as a function of N for given p and rho
import numpy as np
from scipy.stats import binom

def monte_carlo_ensemble_quality(n, p, rho, n_tasks=10000):
    """Simulate majority-vote ensemble quality with correlated errors."""
    # Generate correlated binary outcomes using Gaussian copula
    from scipy.stats import norm
    # Cholesky decomposition of correlation matrix
    corr_matrix = rho * np.ones((n, n)) + (1 - rho) * np.eye(n)
    L = np.linalg.cholesky(corr_matrix)
    u = norm.cdf(np.random.randn(n_tasks, n) @ L.T)
    correct = (u < p).astype(int)
    majority_correct = (correct.sum(axis=1) > n // 2).mean()
    return majority_correct

p, rho = 0.70, 0.30
print("N | Q_condorcet | Q_monte_carlo | N_IT | N_max_USL")
for n in [1, 2, 3, 5, 7, 9]:
    q_cjt = p + (sum(binom.pmf(k, n, p) for k in range(n//2+1, n+1)) - p) * (1 - rho)
    q_mc  = monte_carlo_ensemble_quality(n, p, rho)
    n_it  = max(1, int(np.ceil(np.log(0.5) / np.log(max(1e-9, 1-rho)))))
    n_usl = max(1, round(np.sqrt(0.88 / (0.039 * (1 - (1-rho))))))
    print(f"{n} | {q_cjt:.3f}       | {q_mc:.3f}         | {n_it}    | {n_usl}")
```

---

### GAP-A6: Self-MoA Is a Direct Empirical Competitor 🔴 OPEN — **Critical**

**Gap statement.**
arXiv 2502.00674 (Li et al., 2025) — Self-MoA beats multi-family MoA by 6.6% on AlpacaEval 2.0.
H2AI's full machinery must beat a single strong model at multiple temperatures to justify its
existence. The Coverage quadrant (cross-family committee) is the architectural bet against
Self-MoA; it has not been measured.

**Literature grounding.**
arXiv 2502.00674 — Self-MoA paper; primary threat.

arXiv 2505.24442 (Xie et al., 2025) — *"RMoA: Optimising MoA through Diversity Maximisation and
Residual Compensation"* — finds explicit diversity maximisation in agent selection is "the single
biggest lever for MoA quality." Validates H2AI's Phase 0 decomposition as the right intervention;
now needs empirical validation against B3 baseline.

arXiv 2601.16715 — *"Dynamic Expert-Guided Model Averaging"* — LLM-guided ensemble weighting
outperforms uniform averaging when the LLM's structural knowledge is partially correct. Validates
Phase 0 mandate-based slot selection.

arXiv 2503.03535 — *"Trade-offs in Ensembling, Merging, and Routing"* — systematic comparison:
ensembling dominates on distribution-shift tasks; routing wins in-distribution. Constraint-
compliance tasks under novel specifications are distribution-shift scenarios → ensemble approach
is architecturally correct for H2AI's target domain.

**H2AI's specific claim and where it can win.**
The Precision quadrant (τ-spread within one family) is architecturally aligned with Self-MoA.
The Coverage quadrant (cross-family) is the bet against it. H2AI's additional bet: the MAPE-K
enforcement loop adds constraint compliance above what any temperature-spread ensemble can achieve
without an enforcement mechanism.

**Innovative solution — INNOVATION-5: Structured experiment protocol.**

See INNOVATION-5 above. The key insight from the literature: H2AI should not try to beat Self-MoA
on general benchmarks. The correct metric is **constraint compliance rate on multi-constraint
tasks** — tasks where Self-MoA has no enforcement mechanism and MAPE-K has structural advantage.

**Concrete falsification table:**

| Condition | Task | Expected result if H2AI hypothesis holds |
|---|---|---|
| H2-P vs B3 | Tier 1 (1-2 constraints) | B3 ≈ H2-P (enforcement loop rarely fires) |
| H2-P vs B3 | Tier 3 (6+ constraints) | H2-P > B3 (enforcement loop provides value) |
| H2-C vs B3 | Any tier | Unmeasurable until second adapter family available |

If the Tier 3 advantage does not materialise, H2AI's value is observability and governance (audit
trail, constraint provenance) rather than quality improvement — which is still a valid product
position but requires rewritten documentation.

---

## New Gaps From 2026-05-13 Review

---

### GAP-A7: Preference Leakage in LlmJudge ✅ CLOSED 2026-05-16 — **High**

**Gap statement.**
When the constraint evaluation adapter (LlmJudge in Phase 3.5 and Phase 4) belongs to the same
LLM family as the explorer adapters, the judge systematically favours outputs that match its
family's stylistic patterns. This is not a neutral evaluation — it is an in-family affinity bias
that inflates verification scores for outputs that echo the judge's own training distribution.

**Literature grounding.**
arXiv 2502.01534 (Li et al., 2025) — *"Preference Leakage: A Contamination Problem in LLM-as-a-
Judge"* — demonstrates that when the same LLM family generates synthetic training data AND judges
it, the judge systematically favours the model's own output style. Feedback loop: the judge's
high scores reinforce generator outputs that look like the judge's own style, creating a
self-amplifying bias.

arXiv 2410.13341 (Dorner et al., 2024) — *"LLM as Judge Won't Beat Twice the Data"* — proves
that LLM judge self-preferencing bias is irreducible. Debiasing tools reduce but cannot eliminate
it. Pairwise evaluation (arXiv 2504.14716) is less biased than pointwise but introduces
Condorcet cycles at scale.

**Implementation (closed 2026-05-16).**

Phase 3.5 now uses `JudgePanel` (h2ai-orchestrator) built from the verification adapter plus cross-family explorer adapters:

- **`CrossFamily` panel** (≥2 distinct families): one variant per family, cap 3, `Literal` persona, `temperature_override: None`. Supermajority vote: `quorum = ceil(N × quorum_fraction)`. Pass / Fail / Uncertain.
- **`PersonaOnly` fallback** (single family): 3 variants — Literal (0.0), Contextual (0.2), Skeptical (0.4) — requiring unanimous agreement; any dissent → Uncertain.
- **Uncertain handling**: proposals with only uncertain constraint failures pass with `score × uncertainty_weight` (default 0.7). Hard non-uncertain failures still prune.
- **`ConstraintAmbiguityEvent`**: fire-and-forget tracing log when ≥`ambiguity_threshold` proposals in a wave produce uncertain votes on the same constraint — corpus quality signal.
- **Key insight from Prosa (2605.01630)**: binary rubric decomposition (each constraint evaluated independently) removes judge-model bias sensitivity more than cross-family diversity alone. The constraint corpus as an analytic rubric is the primary mitigation; panel diversity is second-order.

**Research backing:** PoLL (2404.18796) cross-family panel 7× cheaper than large single judge; CARE (2603.00039) same-family panels amplify correlated bias → unanimous rule for PersonaOnly; Prosa (2605.01630) binary rubric removes bias sensitivity; Logarithmic Scores (2604.00477) 3 judges captures ~90% quality gain; TCVA (2604.08595) generalized aggregation; Autorubric (2603.00077) few-shot calibration.

**Remaining open surface:** `uncertainty_weight = 0.7` is not calibrated per constraint severity. Future work: `constraint_severity: Strict | Standard` corpus metadata field mapping to per-constraint uncertainty weights. Empirical validation of cross-family vs. single-family panel Spearman correlation with oracle is pending.

---

## Brainstorm Group B — Mathematical Formula Validity

---

### GAP-B1: β_eff Functional Form Empirically Unvalidated 🔴 OPEN — Medium

**Gap statement.**
`β_eff = β₀ × (1 − CG_mean)` has a first-principles derivation under the assumption that conflict resolution cost is linear in conflict count (see math.md §2). However, the **linear-cost assumption is empirically unvalidated**. If conflict resolution is super-linear (e.g. due to "Lost in the Middle" attention degradation in long synthesis contexts), the formula needs a higher-order term.

**Literature grounding.**
arXiv 0808.1431 (Gunther, 2008) — USL β is defined as the per-pair coherency synchronisation
overhead. In H2AI terms, this is the per-adapter-pair constraint conflict resolution cost.

**Innovative solution — First-principles derivation (resolves the gap conceptually).**

In USL, β is the fraction of time spent on N(N-1) pairwise coherency checks. In H2AI's merge:

- Expected constraint conflicts between adapters i and j ∝ (1 - CG(i,j)) × |corpus|
- Mean expected conflicts ∝ (1 - CG_mean)
- Conflict resolution cost ∝ n_conflicts (approximately linear — one synthesis token per conflict)

Therefore: `β_eff ∝ β₀ × (1 - CG_mean)` is a first-principles consequence if conflict resolution
cost is linear in conflict count. The linear form is correct under this assumption.

The assumption is falsifiable: if conflict resolution cost is super-linear (e.g. due to attention
degradation in a long synthesis context), the formula needs a higher-order term.

**Improved formula unifying with context-aware term (from math.md §2.3):**

```
β_eff(N) = β₀ × (1 - CG_mean) × (1 + γ × fill(N))
fill(N)  = min(1, N × proposal_tokens / max_tokens)
```

This is already computed in `n_max_context_aware` but is not used in the primary `n_max()`
function. Unify them: make the attention term the standard formula, not a contextual variant.

**Empirical validation with INNOVATION-2 data:**
When the conflict-count β (INNOVATION-2) is available, regress `conflict_rate` against
`(1 - CG_mean)` and `(1 - CG_mean) × fill(N)`. If the bivariate R² > R² of the univariate fit,
the attention term has explanatory power.

---

### GAP-B3: Attribution Formula Is Self-Referential 🟡 PARTIAL

Oracle-grounded calibration requires GAP-E1 data. Once available, apply conformal prediction:

```
conformal_margin(α) = quantile(|q_confidence - q_oracle|, 1-α)  over calibration split
output: q_confidence ± conformal_margin(0.10)   [90% coverage guarantee]
```

arXiv 2410.11594 (Wagner et al., 2024) — *"Black-box Uncertainty Quantification for LLM-as-a-
Judge"* — sampling-based confidence intervals for LLM judge scores without white-box access.
Applicable to H2AI's Phase 3.5 verifier score intervals.

---

### GAP-B5: Proxy Chain — rho_mean, p_mean, β_eff All Unvalidated 🔴 OPEN — **High**

**Gap statement.**
Three interconnected proxies form a chain of unvalidated assumptions. Each propagates error into
the Condorcet Q(N, p, ρ) model and the USL ceiling N_max. The chain is:

1. `rho_mean = 1 − CG_mean` — correlation proxy
2. `p_mean = 0.5 + CG_mean / 2` — accuracy proxy (`sizing.rs:526`)
3. `β_eff = β₀ × (1 − CG_mean)` — conflict cost proxy

**p_mean proxy analysis (`sizing.rs:526`).**
`p_mean = 0.5 + CG_mean / 2` maps constraint agreement linearly to ground-truth accuracy.
This is only valid if the constraint corpus perfectly covers the failure modes of interest.
An ensemble that agrees on constraints but fails on unmodeled dimensions would have high CG_mean
but low true accuracy — the proxy would overestimate p_mean and recommend too few agents.
No empirical validation against held-out accuracy measurements exists.

**β_eff linear assumption.**
`β_eff = β₀ × (1 − CG_mean)` assumes conflict resolution cost scales linearly with conflict
count. This breaks if the LLM synthesiser exhibits super-linear attention degradation on long
contexts ("Lost in the Middle" effect). A context-aware `n_max_context_aware` correction exists
in `sizing.rs` but the linear base assumption is still the default path.

**rho_mean proxy analysis.**
`rho_mean = 1 − CG_mean` is used as the correlation correction in `Q(N, p, ρ)`. No derivation.
The formula implies CG_mean = 0 (zero constraint agreement) → ρ = 1.0 (fully correlated).
This is backwards: low CG_mean means agents disagree on constraints, which should indicate *less*
error correlation, not more.

**Analysis of the implicit logic.**

The intended interpretation: CG_mean measures *similarity* of behavioural profiles. High
similarity → high correlation. So `ρ = CG_mean` would be the natural proxy.

The actual formula `ρ = 1 - CG_mean` inverts this: high similarity → low ρ. The implicit
reasoning may be: "if agents agree (high CG), they're diverse in other ways, so ρ is low" — but
this is neither stated nor justified.

**Alternative derivation from Hamming geometry:**

If two agents have CG(i,j) = c (constraint agreement fraction), then:
- The fraction of constraints where they disagree = 1 - c
- Disagreement means independent error patterns on those constraints
- Agreement means correlated error patterns on those constraints
- If errors only occur on disagreed constraints: ρ ∝ c (agreement)

This gives `ρ_mean = CG_mean`, not `1 - CG_mean`.

However: agreement on constraint profiles doesn't mean agreement on *which wrong answer* to
give. Two agents can agree on satisfying constraint k but disagree on constraint l — yet both
hallucinate the same wrong entity. The SRANI CFI mechanism is precisely designed to detect this.

**Pragmatic conclusion:** Neither `ρ = CG_mean` nor `ρ = 1 - CG_mean` is derivable from first
principles without additional assumptions. Online ρ_EMA is now live (`rho_ema.rs`) and replaces
the proxy once 30 task observations accumulate. Until that threshold is reached, the system uses:

> "Operational convention: low CG (diverse constraint profiles) is assumed to indicate lower error
> correlation. This assumes error patterns track constraint specialisation. The assumption is
> unvalidated and replaced by empirical ρ_EMA once 30 task observations exist."

**Implementation cold-start detail (`rho_ema.rs`):**
The EMA returns a hard-coded prior of `0.45` before 30 pairwise observations accumulate. This
prior was chosen conservatively (mid-range correlation) but is unvalidated against any empirical
distribution of LLM ensemble error correlations. The prior directly enters the Condorcet Q(N,p,ρ)
model, meaning all ensemble sizing decisions for the first ~30 tasks of a tenant's lifetime rest
on this assumption. External validation against a held-out benchmark dataset is the correct fix.

**Simulation to reveal the assumption sensitivity:**

```python
import numpy as np

def q_condorcet(n, p, rho):
    from scipy.stats import binom
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

## Brainstorm Group C — Correlated Failure Modes

GAP-C1 through GAP-C4 are implemented (documented in research-state.md §3.3). GAP-C5 is new.

---

### GAP-C5: Krum Selection Fails Under Majority Correlated Hallucination 🔴 OPEN — **High**

**Gap statement.**

`OutlierResistant{f}` (Krum-style multi-Krum) selects the proposal with minimum sum of distances to its `N−f` nearest neighbours — the most central proposal in embedding space. This guarantees Byzantine fault tolerance when faults are *independent* outliers: Byzantine proposals cluster far from each other and from the correct answer, so the distance criterion excludes them.

The guarantee inverts when the majority of agents share a correlated hallucination. LLMs across different providers are pre-trained on the same internet corpus (Common Crawl, Wikipedia, a small set of code repositories). On tasks that activate a shared misconception — a plausible but false historical claim, a misremembered API signature, a canonical-but-wrong example — five adapters from five different providers can produce the same confident wrong answer. In this case:

- The hallucinated proposals are semantically clustered (low pairwise embedding distance).
- The correct proposal (if any agent produced it) is the outlier in distance space.
- Krum selects the cluster centroid — the hallucination — as the "safe" output.

This is not a corner case. Bradley (2024) — arXiv 2411.01539 — demonstrates that LLM errors are *systematically* correlated across architecturally similar models on shared-corpus content. Krum was designed for adversarial Byzantine nodes (federated learning, distributed state machines) — its Byzantine fraction proof assumes independent failures. That assumption does not hold for stochastically correlated LLM outputs.

**Current mitigations and their limits.**

| Mitigation | What it does | Why it is insufficient |
|---|---|---|
| GAP-C1: Token-Jaccard CV detector | Fires `CorrelatedEnsembleWarning` when `CV(distances) < 0.30` AND `mean_jaccard < 0.50`; triggers MAPE-K retry | Reactive — fires after generation, after Krum has already run. Retry reuses the same adapter pool; if the hallucination is cross-provider, grounded regeneration still produces correlated wrong answers. |
| GAP-C2: Shadow auditor | Concurrent auditor from a different family; promotes to AND-vote on high disagreement | Addresses audit bias, not generation diversity. Shadow auditor cannot vote a correlated hallucination into being correct. |
| GAP-A7: Cross-family judge rotation | Verifier adapter from non-majority family | Addresses verifier bias after Krum selection, not the Krum input distribution. |
| `family_constraint = "require_diverse"` | Warns / blocks monoculture explorer pools | Prevents single-provider monoculture. Does not prevent cross-provider shared training overlap on specific misconception domains. |

The gap: no mechanism prevents the correlated hallucination from *entering* the Krum input in the first place when the correlation source is shared training data rather than provider identity.

**Connection to other gaps.**

- **GAP-A6** (empirical ρ) — the only way to know whether a specific task domain activates majority correlated hallucination is to measure it with an oracle. Without GAP-A6 data, GAP-C5 cannot be quantified.
- **GAP-A7** (judge preference leakage) — the auditor phase may itself prefer the correlated output if judge and explorers share a family, compounding the error after Krum selection.

**Innovative solution — structural pre-generation family diversity + family rotation on ModeCollapse.**

Two levers, both incremental on existing infrastructure:

**1. Family-diversity gate at provisioning.** Require ≥ 2 distinct provider families in every explorer committee when `N ≥ 3`. The adapter factory already tracks family metadata; add a family-diversity check in `h2ai-provisioner` at slot assignment.

```toml
[provisioner]
min_explorer_families = 2   # proposed: enforce structural family diversity at provisioning
```

**2. Family rotation on ModeCollapse retry.** When the MAPE-K controller emits `ModeCollapse`, the current `adapter_rotation_offset` rotates within the configured pool. If the pool is all-same-family (or cross-provider same-corpus), rotation has no effect. Add: on `ModeCollapse`, rotate to the *least-used provider family* in the adapter factory pool.

```toml
[mape_k]
mode_collapse_family_rotation = true   # proposed: rotate provider family, not just offset
```

Both are wired to existing infrastructure: the adapter factory knows families, `MapeKController` already handles `ModeCollapse` as a named `ExitReason`.

**Falsification condition.**

Construct a task set activating a known common internet misconception (verifiable by external oracle). Run N=5 with (a) same-provider pool vs. (b) ≥3-family pool. Measure how often the Krum-selected output is oracle-correct in each condition. If family-diverse condition is significantly more accurate (p < 0.05), structural family diversity at provisioning is the mitigation. This can be run as an extension of the GAP-A6 benchmark — the task set already needs oracle-correct labels.

---

### GAP-B6: LLM-as-Judge Validity — Krum Operates on Potentially Biased Scores 🔴 OPEN — **High**

**Gap statement.**
The Krum-style epistemic leader election and verifier consensus phases both depend on
`VerificationScoredEvent` scores produced by an LLM judge. If the judge is biased (self-
preference, length bias, sycophancy), the Krum input distribution is corrupted and outlier
rejection selects the *judge-preferred* proposal, not the *correct* one.

**Why this matters for Byzantine robustness.**
The Krum algorithm was designed for distributed ML under Byzantine worker attacks. Its guarantees
hold when the scoring function is trustworthy. Substituting LLM-as-Judge introduces a new
failure mode: a flawed judge that consistently prefers verbose, confident-sounding proposals
will cause Krum to surface the most persuasive wrong answer, not the most accurate one.

**Known LLM judge failure modes (from literature):**
- **Self-preference / position bias**: judges rate outputs similar to their own training higher
- **Verbosity bias**: longer responses receive higher scores independent of accuracy
- **Sycophancy cascade**: if the judge has seen the proposal being judged (shared context), it
  tends to rate it higher

**Current mitigations.**
| Mitigation | Coverage |
|---|---|
| GAP-A7 cross-family judge rotation | Reduces family-level preference leakage; doesn't fix verbosity or position bias |
| Shadow auditor (concurrent independent verifier) | Second opinion but from same judge distribution |
| `verifier_consensus_passes = N` majority vote | Reduces single-call variance; amplifies systematic bias |

None of the mitigations address systematic bias in the judge itself.

**Path to resolution.**
1. **Calibrated judges**: measure judge accuracy on a held-out set with known-correct answers;
   use the measured judge accuracy as a discount factor on verification scores
2. **Comparative judging**: instead of absolute scores, ask judge to rank pairs of proposals;
   pairwise ranking is less susceptible to absolute-score bias
3. **Adversarial critique**: for each proposal, generate a dedicated critique (adversarial probe);
   score = f(judge_score, critique_score); this forces the judge to surface the failure mode
4. **Oracle bootstrap** (GAP-A6): only way to empirically measure judge calibration on the task
   distribution

**Effort estimate.** Pairwise ranking: 1 week. Adversarial critique integration: 2 weeks. Oracle calibration: blocked on GAP-A6.

---

## Brainstorm Group D — Infrastructure and Operational Gaps

---

### GAP-D4: Verification Throughput ✅ Already Parallel — **Closed**

**Status: closed on inspection (2026-05-18).**

The external feedback correctly identified Phase 3.5 verification as a potential bottleneck.
Code inspection (`verification.rs`) reveals both parallelism dimensions are already implemented:

- **Proposal-level**: `join_all(futures)` at `verification.rs:109` — all N proposal evaluations
  fire concurrently.
- **Constraint-level**: `join_all(futs)` at `verification.rs:537` (inside `eval_all`) — all M
  constraint evaluations fire concurrently per proposal.

The true bottleneck is the LLM adapter's HTTP concurrency capacity (one GPU inference queue for
local models), not missing parallelism in H2AI code. No action required on H2AI side.

**Actual bug found and fixed (2026-05-18):** The rubric fallback path (empty corpus) in
`eval_all` used a hardcoded `Hard { threshold: 0.45 }` severity, ignoring the caller's
`verify_threshold` config. This caused proposals to collapse to score 0.0 when the constraint
wiki was disabled — the hard gate fired at 0.45 regardless of `verify_threshold = 0.2`.
Fixed by threading `rubric_threshold` through `eval_all` and using the outer verification
threshold for the rubric constraint's hard gate.

---

### GAP-D5: Synthesis Merge Bottleneck — Single Sequential Synthesis Step 🔴 OPEN — Medium

**Gap statement.**
Phase 5a (synthesis/merge) is a single sequential LLM call that receives all N proposals
concatenated into one context. For N=5 proposals of 1000 tokens each with a 500-token system
prompt, the synthesis context is ~5500 tokens. This creates two problems:

1. **Length-sensitivity bias** — the synthesis LLM is exposed to "Lost in the Middle" attention
   degradation on long contexts. Proposals in the middle of the concatenated input receive less
   attention. Merger output quality degrades as N increases.

2. **No parallelism** — the synthesis call is inherently sequential (one call, one output). If the
   synthesis LLM fails or timeouts, there is no fallback within the phase.

**Literature grounding.**
arXiv 2307.03172 (Liu et al., 2023) — *"Lost in the Middle: How Language Models Use Long
Contexts"* — demonstrates that LLMs use context from the beginning and end most reliably; middle
items in long lists are systematically under-weighted. For 5+ proposals concatenated in a merge
context, the middle proposals are systematically disadvantaged.

**Innovative solution — hierarchical tournament merge.**

Instead of one N-way merge, use a bracket tournament:
- Round 1: pair proposals (N/2 independent merge calls in parallel)
- Round 2: merge the round-1 outputs (N/4 calls)
- Finals: single merge of the 2 finalists

Benefits:
- Each merge call sees at most 2 proposals — context length is bounded regardless of N
- Round 1 calls are parallel — same wall-clock time as 1-call merge
- Middle-position bias is eliminated (every proposal is in position 1 or 2 in its merge)

Trade-off: log₂(N) merge rounds vs. 1 round. For N=5: 3 rounds, each with bounded context.
The Krum/epistemic leader elected proposal should be seeded into round 1 position 1 (always "first"
position) to exploit the attention recency effect.

**Effort estimate.** 1 week for hierarchical merge. Requires new `MergeStrategy` variant in
`merger.rs` (`HierarchicalTournament` vs current `OneShot`). Config toggle to enable.

---

### GAP-D6: CRDT Semilattice on Binary Constraint Flags — Algorithmic Simplification Opportunity ✅ Not a deployment concern — Low

**Clarification on "complexity" framing.**
External reviewers may flag H2AI's state infrastructure (NATS JetStream + delta checkpoints +
BFT Krum + CRDT semilattice) as operationally complex. This conflates implementation complexity
with operational complexity. Deployed as a Docker container, all 16 crates compile to a single
binary; NATS runs embedded; no component is operator-facing. Rust's zero-cost abstractions and
lack of GC mean the full pipeline — including eigenvalue decomposition, event-sourced state,
and async Tokio orchestration — has a smaller runtime memory footprint than an equivalent Python
implementation of a simpler design. Operational surface area = `docker compose up`. This is
not a weakness.

**Actual narrow gap (algorithmic, not operational).**
The CRDT semilattice in the merger operates on binary constraint-satisfaction fingerprints:
for each constraint, a proposal either satisfies it (1) or doesn't (0). A semilattice join
over binary vectors is equivalent to bitwise OR (or AND depending on semantics). The CRDT
abstraction is correct and future-compatible with geo-distributed concurrent merges, but for
the current single-region single-JetStream deployment the CRDT machinery (lattice ordering,
join commutativity proofs) adds code complexity without producing a different result than
`bitwise_and(fingerprints)` on the surviving proposals.

**Recommendation.**
No action for correctness — the CRDT is not wrong and positions the system for multi-region
extension. As a code simplification opportunity: document within `merger.rs` that the semilattice
join on binary satisfaction vectors degenerates to bitwise AND/OR, making the algorithm
reviewable without knowledge of CRDT theory.

---

### GAP-D2: Compound Task Cost Is Unconstrained 🔴 OPEN — Low

A `CompoundTaskEngine` DAG fires a full wave for each subtask with no pre-execution cost estimate
or operator confirmation gate. Up to 75 LLM calls before synthesis for a 5-subtask compound.

**Research approach.** Complexity probe + bandit routing. Before dispatching ensemble, call a
lightweight adapter (smallest available) to rate subtask complexity 1–5. Route 1–2 to single-
adapter path; 3–5 to full ensemble. The probe cost is 1 small-model call vs. N full-model calls.
Use the existing Thompson Sampling bandit to improve probe accuracy over time.

**Effort estimate.** 1 week cost estimate + SSE event; 2 weeks complexity probe + bandit routing.

---

### GAP-D3: Calibration Bootstrapping — **Closed (2026-05-14)**

Closed by the Adaptive Prompt Harness (OPRO). `seed_all_bootstrap_priors` in `h2ai-api/src/bootstrap.rs` seeds Bayesian beta priors at startup from `AdapterProfile.tier` (Capable=0.78, Standard=0.62, Fast=0.45 j_eff median priors), providing principled calibration before any tasks have run. New deployments no longer return 503 on the first task for prompt-quality bootstrapping; ensemble physics calibration (α, β) still requires a `POST /calibrate` run.

---

## Brainstorm Group E — Quality Measurement Infrastructure

---

### GAP-E1: Oracle Integration 🟢 WIRED — Blocking

**Wired (2026-05-14).** Phase 4.5 oracle gate is live: NATS `request()` to `cfg.oracle_gate.subject` with configurable timeout before the Phase 5 merge step. The thinking loop Stage 2 (`brainstorm_one`) sends candidate solutions to the oracle inline; `synthesize` applies `oracle_confidence_bonus` when the oracle approved. On fail + low confidence, a matching `ClarificationTemplate` fires a `PendingClarificationEvent` that suspends the engine; `POST /{tenant_id}/tasks/{id}/clarify` resumes it with an operator answer. `oracle_gate_passed: Option<bool>` on `MergeResolvedEvent` tracks the gate outcome per task.

**Open.** Domain-specific automated test suites (code, factual QA, structured output) are the remaining work for automated oracle coverage.

**Innovative opportunity — FUSE-style zero-label verifier ensembling.**
arXiv 2604.18547 (Lee et al., 2026) — *"FUSE: Ensembling Verifiers with Zero Labeled Data"* —
proposes unsupervised ensembling of LLM verifiers using only agreement structure among verifiers.
Applicable to H2AI's cold-start problem: before domain-specific oracles exist, use FUSE-style
inter-verifier agreement to produce a calibrated ensemble score without ground truth labels.

**Oracle priority queue by implementation cost:**

| Oracle type | Tasks covered | Implementation cost |
|---|---|---|
| JSON schema validation | Structured output tasks | 1 day — already have typed deserialisation |
| Cargo test / pytest runner | Code tasks | 3 days — ShellExecutor extension |
| MMLU / TriviaQA lookup | Factual QA | 1 week — reference dataset integration |
| Symbolic verifier (Z3) | Formal reasoning | 2 weeks |
| Human rating | Open-ended writing | Open-ended |

Start with JSON schema validation (zero new dependencies, covers structured output tasks which
are H2AI's primary use case) and cargo test runner (enables the GAP-A6 experiment on HumanEval).

---

### GAP-E2: Talagrand Histogram Has No Feedback Loop 🔴 OPEN — Medium

**Gap statement.**
`TalagrandDiagnostic::from_verification_scores` computes the rank histogram. U-shape = over-
confident (increase τ spread); Λ-shape = under-dispersed (decrease τ spread). The feedback loop
is architecturally described but not implemented: `EnsembleCalibration::tau_spread_factor` is not
updated from Talagrand observations.

**Innovative solution — KL-divergence τ-spread update rule.**

arXiv 2605.07775 (Menet et al., 2026) — *"POETS: Uncertainty-Aware LLM Optimisation via Policy
Ensembles"* — uses Thompson Sampling over KL-regularised reward models for ensemble policy
updates. The KL framework directly applies to Talagrand histogram correction.

Define the Talagrand update rule:

```
KL_flat(H) = KL(H || Uniform)   [divergence from flat histogram]
U_score    = variance(H) / mean(H)   [U-shape detection: high variance relative to mean]
Λ_score    = max(H[middle]) / mean(H[edges])   [Λ-shape: centre mass exceeds edges]

Δτ = η × (U_score - Λ_score)
τ_new = clip(τ_current + Δτ, τ_min, τ_max)
```

Where η (learning rate) is fit from the time to cover one effective Talagrand window (default:
30 verification score observations).

Store `tau_spread_factor` history in `H2AI_CALIBRATION` KV alongside α and β₀. This persists τ
tuning across restarts — the system accumulates calibration from production traffic rather than
resetting on each deployment.

**arXiv precedent:**
Meteorological ensemble calibration (the origin of Talagrand histograms) uses exactly this
feedback: when the rank histogram is non-uniform, the spread parameter of the ensemble is adjusted.
The τ-spread in H2AI is the direct analogue of the ensemble spread parameter in NWP.

**Research question:** Is τ adjustment per-task-domain appropriate? A coding task may have
naturally high verification score variance (pass/fail) while a factual QA task may have low
variance. Per-domain Talagrand calibration requires enough observations per domain.

**Effort estimate.** 1 week for KL update rule implementation; 2 weeks for η tuning on a
representative task distribution.

---

## Gap Priority Matrix

| Gap | Core thesis risk | Implementation cost | Data dependency | Suggested order |
|---|---|---|---|---|
| **INNOVATION-2 Conflict-count β₀** | Critical | 1 week | Calibration runs | **Week 1** |
| **INNOVATION-5 H2-P vs B3 experiment** | Critical | 1 week | None (single-model) | **Week 1** |
| **INNOVATION-4 N_IT as primary sizer** | High | 1 week | None | **Week 2** |
| **GAP-A7 Preference leakage** | High | ~~1 week~~ | — | ✅ Closed 2026-05-16 |
| **GAP-B5 rho_mean documentation** | Medium | 2 days | None | **Week 2** |
| GAP-E1 Domain-specific oracles | Blocking for A/B | 1–3 weeks | Domain test suites | Session 1 |
| GAP-A1 TCC parameter fitting | Critical | 2 weeks | Oracle quality signal | Session 1 |
| GAP-A6 Full experiment (cross-family) | Critical | Timeline open | Second adapter family | Session 2+ |
| GAP-A2 USL quality curve experiment | High | 2 weeks | Shared task set | Session 2 |
| ~~GAP-D1 Calibration harness extension~~ | — | — | — | **Closed 2026-05-15** |
| GAP-E2 Talagrand feedback loop | Medium | 3 weeks | Task runs | Session 4 |
| GAP-B1 β_eff functional form fit | Medium | 2 weeks | Controlled calibration | Session 5 |
| ~~GAP-D3 Bootstrap calibration~~ | — | — | — | **Closed 2026-05-14** |
| ~~GAP-D4 Verification parallelism~~ | — | — | — | **Closed 2026-05-18** |
| GAP-D5 Hierarchical merge | Medium | 1 week | None | Week 3 |
| GAP-D2 Compound task cost | Low | 3 weeks | None | Any |

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
