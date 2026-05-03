# H2AI Control Plane — Architecture

H2AI Control Plane is a distributed multi-agent orchestration runtime that prevents LLM agent swarms from degrading under their own coordination cost. It measures the overhead of making N agents agree — using the Universal Scalability Law (USL) to model the serial planning bottleneck and the pairwise semantic reconciliation cost — and enforces typed constraints so agents share enough common ground to produce coherent results.

The system is best understood as an advanced distributed scheduler. Instead of scheduling static processes onto physical CPU cores, it schedules nondeterministic LLM inference tasks onto dynamically provisioned graph topologies, and bounds the coordination overhead at every step using measured parameters rather than configuration guesswork.

This document covers the system at the architectural level. For formula derivations, propositions, and limitations, see [`math.md`](math.md). For per-endpoint contracts and event payload schemas, see [`reference.md`](reference.md). For deployment topology, NATS cluster configuration, and observability, see [`operations.md`](operations.md).

---

## 1. System Overview

### The Coordination Cost Problem

Multi-agent systems exhibit retrograde behavior: beyond a critical N, more agents produce worse results than fewer. This has been confirmed empirically for LLM ensembles across multiple 2023–2025 studies. A framework that lets you add agents without bound is not neutral — it is actively harmful once the coordination ceiling is crossed.

Every existing multi-agent framework lets you set `max_agents` (or equivalent). The number comes from intuition, documentation examples, or trial and error. There is no principled basis for why three agents are better than five for a given task. H2AI's first responsibility is to derive that ceiling from measured parameters and enforce it.

### Positioning

The major multi-agent LLM frameworks as of 2026:

| Framework | Primary model | Language | State model | Scale target |
|---|---|---|---|---|
| **LangGraph** | Graph-based workflow DAG | Python | In-process / Redis | Single process, Python ecosystem |
| **AutoGen** | Conversational agent loop | Python | In-process thread storage | Research, prototyping |
| **CrewAI** | Role-based task delegation | Python | In-process | Small teams, quick setup |
| **Semantic Kernel** | Planner + plugin orchestration | C# / Python | In-process | Enterprise Microsoft stack |
| **MoA (Mixture-of-Agents)** | Layered ensembling | Research | In-process | Research benchmarks |
| **DSPy** | Programmatic prompt compilation | Python | In-process | Single program, optimization layer |
| **Ray** | Distributed compute fabric | Python | Object store | Generic distributed compute |
| **H2AI Control Plane** | Physics-bounded agent swarm | Rust | Event-sourced CRDT on NATS | Production, auditable, multi-node |

H2AI is positioned as a layer that solves a problem the others do not address: bounding ensemble size by measured coordination physics. It is complementary to DSPy (which optimizes individual prompts) and Ray (which provides a generic distributed compute fabric) — H2AI is the orchestration layer that sits above either, deciding *how many* agents to fan out, with what role assignment, and how to merge their outputs.

A more detailed feature comparison:

| Feature | LangChain / CrewAI | OpenAI Agents SDK | H2AI Control Plane |
|---------|-------------------|------------------|-------------------|
| Execution routing | Sequential / graph DSL | Handoffs, agents-as-tools | Deterministic DAGs |
| State management | LLM context window | Thread storage | Sovereign CRDTs (event-sourced) |
| Safety | Rules / guardrails | Guardrails | Typed `ConstraintDoc` predicates (Hard/Soft/Advisory) + ADR Auditor + Multiplication check |
| Scalability | Unbounded provisioning | Unbounded provisioning | MAPE-K autonomic shifting against N_max |
| Human integration | Chat / correction loop | Human-in-the-loop hooks | CRDT Merge Authority (O(1) resolution) |
| Agent identity | Named product variants | Function tools | `AgentDescriptor { model, tools }` — any LLM |
| Tool risk | Implicit, untracked | Function signatures | `AgentTool` flags → c_i, α, β₀, NKey scope |
| Iterative refinement | ReAct (manual prompt) | ReAct | TAO loop — harness-driven, pattern-verified |
| Quality filtering | None / rules | Guardrails | Phase 3.5 scored LLM-as-judge |
| Context management | Summarization | Compaction | Lost-in-Middle mitigation + keyword preservation |
| Harness attribution | None | None | **Unique: USL quality decomposition (baseline + topology + verification + TAO)** |
| Error taxonomy | Ad hoc | Ad hoc | 4-class typed (transient / recoverable / user-fixable / unexpected) |

### What H2AI Does Not Do Better

This is not a universal upgrade. CrewAI and LangGraph have a five-minute path to a working prototype; H2AI requires configuring NATS, a constraint corpus, and an adapter pool first. The primary runtime is Rust, so Python-native toolchains pay an integration cost. LangGraph Studio offers a visual workflow designer where H2AI selects topology automatically from calibration data — you trade manual graph control for physics-grounded automatic selection. AutoGen and CrewAI carry larger catalogs of pre-built agent roles and conversation patterns; H2AI focuses on three frontier topologies and an abstract role system, choosing depth (correctness, auditability) over breadth.

H2AI is the right choice when output correctness is auditable, when tool-using agents can write irreversible state, when the constraint space is explicit and should *block* non-compliant proposals, when the deployment must scale beyond a single process, or when ensemble size needs to be principled. It is not the right choice when you need a working prototype in thirty minutes, when all agents are pure-reasoning and audit is not required, when the team is fully Python-native, or when the task is simple enough that N=1 suffices.

---

## 2. Mathematical Foundation

The system is governed by USL extended with epistemic structure. Key quantities and how they shape architecture:

| Symbol | Formula | Architectural meaning |
|--------|---------|----------------------|
| α | measured | Serial contention fraction. Tool-using agents (Shell, FileSystem) raise α; pure-LLM agents have α ≈ 0. |
| β₀ | measured (EMA) | Baseline pairwise coherency cost per pair, computed from actual merge token cost: `β₀ = (merge_step_token_cost / N(N−1)/2) / mean_proposal_tokens`. |
| β_eff | β₀ × (1 − CG_mean) | Effective coherency cost; bounded at β₀ when CG_mean=0, near-zero when CG_mean=1. CG_mean < 0.10 collapses N_max to 1 via `cg_collapse_threshold`. |
| N_max | round(sqrt((1−α) / β_eff)) | Throughput ceiling — USL Proposition 1 (Gunther 1993). |
| N_optimal | argmax of marginal Condorcet gain per agent | CJT-derived quality target. |
| N_eff | EigenCalibration result | Effective independent ensemble size after correlation collapse, from spectral analysis of pairwise CG. |
| θ_coord | min(CG_mean − σ_CG, 0.3) | Minimum CG any Explorer pair must meet. |
| c_i | ∈ [0,1] | Role error cost. `max(c_i) > 0.85` switches `MergeStrategy` from `ScoreOrdered` to `ConsensusMedian`; `> 0.95` to `OutlierResistant` (BFT). |
| c_i_eff | c_i × 0.6^(t−1) | Effective error cost after t TAO iterations; the merge tier is re-evaluated post-loop. |
| filter_ratio | N_valid / N | Fraction of proposals passing Verification Phase. |
| Q_total | Q_baseline + G_topo + G_verify + G_tao | Harness attribution decomposition. |

The architectural significance is the *bridge* between mathematical structure and runtime decisions. USL was designed for shared-state distributed systems where α and β have physical meanings (lock queueing, cache-line coherency). For agent orchestration these parameters carry *reasoning-space* meanings: α is the serial fraction inherent to planning and synthesis, β is the pairwise cost of reconciling agents that have diverged. The mathematical structure is identical; the substrate is different.

USL alone gives a coordination ceiling but not a quality target. CJT (the Condorcet Jury Theorem) alone gives a quality gain but no cost constraint. H2AI uses both: the effective ensemble size is `min(N_optimal, N_max)`. When N_optimal exceeds N_max — when the quality benefit of a larger ensemble is real but coordination cost would degrade it — the system runs at N_max and compensates with topology. Hierarchical Tree reduces coordination cost from O(N²) to O(N) by grouping agents, raising the effective ceiling for sub-groups while keeping overall N bounded.

EigenCalibration (computing N_eff from the spectral structure of pairwise CG) detects correlated-hallucination collapse: an ensemble of N=6 agents whose pairwise agreement is too high is structurally an N_eff=2 ensemble, and the system reports this rather than over-counting independent signal.

See [`math.md`](math.md) for full formula derivations, propositions with proofs, calibration reference tables, simulation findings, and limitations.

---

## 3. Six Execution Phases

Execution proceeds through six phases driven by NATS JetStream events. All events are published to `h2ai.tasks.{task_id}` as immutable appends; if it happened, it is in the log. The full event vocabulary and structural guarantees are in [`reference.md`](reference.md); this section describes the phase-by-phase flow.

### Phase 0 — Calibration

Triggered by system startup or `POST /calibrate`, the calibration harness in `h2ai-autonomic` runs a small set of representative tasks (default 3) through the adapter pool. It measures α from wall-time fractions, β₀ from actual merge token cost (online EMA), and CG samples from Explorer pair agreement rates. From these it derives β_eff, N_max, and θ_coord, persisting `CoherencyCoefficients` to NATS KV via `CalibrationCompletedEvent`. No live task proceeds without valid calibration data — `POST /tasks` returns `503 CalibrationRequiredError` if the cache is empty.

### Phase 1 — Bootstrap

When a human POSTs a task manifest, `h2ai-context` reads the manifest and scans the local constraint corpus (ADRs and typed `ConstraintDoc` files via `h2ai-constraints::loader`). It compiles an immutable `system_context` string from the constraint corpus, the task description, and any explicit `context` field, then publishes `TaskBootstrappedEvent`. The Bootstrap phase compiles context unconditionally — the gate is not at the start of generation, it is the Auditor at Phase 4. The API responds `202 Accepted` with `task_id`; all further progress is observable on the SSE stream.

The key invariant: `system_context` is sealed at `TaskBootstrappedEvent`. No agent ever sees a different context than what the Auditor was briefed on. This creates a closed epistemic loop — Explorers and Auditor share exactly the constraint set the human provided.

### Phase 2 — Topology Provisioning

`h2ai-autonomic` reads the calibration coefficients and the task's `ParetoWeights` and `topology` field. It computes `β_eff` and `N_max` for the live task, then selects topology:

| Condition | Selected topology | Pareto profile (T/E/D) |
|---|---|---|
| Manifest provides `explorers.roles[]` | Team-Swarm Hybrid | 84% / 91% / 95% |
| Manifest sets `topology.kind: "hierarchical_tree"` | Hierarchical Tree | 96% / 96% / 60% |
| Manifest sets `topology.kind: "ensemble"` | Ensemble + CRDT | 84% / 84% / 90% |
| Auto: `N_requested ≤ N_max` AND `W_H` dominant | Ensemble + CRDT | 84% / 84% / 90% |
| Auto: `N_requested > N_max` OR `W_E` dominant | Hierarchical Tree | 96% / 96% / 60% |

Ensemble + CRDT connects all Explorers through NATS without a Coordinator — suitable for small, diverse swarms. Hierarchical Tree introduces one Coordinator and `k_opt = floor(N_max^flat)` sub-groups, reducing coordination edges from O(N²) to O(N). Team-Swarm Hybrid uses role-differentiated Explorers with review gates between specified pairs; its binding ceiling is `N_max^interface`, typically 3–5 concurrent sub-tasks, and `InterfaceSaturationWarningEvent` fires when active sub-tasks approach this ceiling.

The provisioner assigns τ values (from `AgentRole` canonical defaults when `roles[]` is provided, otherwise spread across `[τ_min, τ_max]` to enforce error decorrelation), assigns `RoleErrorCost` per node, computes `MergeStrategy` from `max(c_i)`, and publishes `TopologyProvisionedEvent`. The autonomic loop re-enters Phase 2 after `ZeroSurvivalEvent` or `MultiplicationConditionFailedEvent`, bounded by `max_retries`.

### Phase 2.5 — Multiplication Condition Gate

Before any inference token is generated, the orchestrator verifies all three conditions from Proposition 3 against the calibration data:

1. **Baseline competence** — each planned Explorer adapter has `p_correct > 0.5` on the calibration set. An Explorer below random chance degrades the collective.
2. **Error decorrelation** — pairwise agreement rate `ρ < 0.9` across all Explorer pairs. Two Explorers that make the same errors 90%+ of the time are structurally redundant. The fix is widening τ spread or routing to different model backends.
3. **Common Ground floor** — `CG_mean ≥ θ_coord` for all planned Explorer pairs.

If any condition fails, `MultiplicationConditionFailedEvent` names which condition failed and the measured values; Phase 2 is re-entered with adjusted parameters. The failure payload is included in `TaskFailedEvent` if retries exhaust, so the operator can diagnose which condition blocked execution.

### Phase 3 — Parallel Generation (TAO Loop)

The orchestrator fans out N Explorers into a `tokio::task::JoinSet` wrapped in `tokio::time::timeout`. Two execution paths exist depending on `EngineInput.nats_dispatch`. The direct path calls a locally-held `IComputeAdapter` in-process for local development and the Local plan. The NATS dispatch path uses `NatsDispatchAdapter`, which calls `AgentProvider::select_agent(&TaskRequirements)` to find a live edge agent by capability and cost tier, then publishes a `TaskPayload` to `h2ai.tasks.ephemeral.{task_id}` and awaits the `TaskResult` on the `H2AI_RESULTS` JetStream work-queue.

Each Explorer runs a TAO loop (`orchestrator::tao_loop::TaoLoop::run`) for up to `max_turns` (default 3). Turn 1 makes the initial `IComputeAdapter::execute()` call. The output is regex-checked against `TaoConfig.verify_pattern`, and optionally JSON-Schema-validated. On failure, an observation feedback string from `TaoConfig.retry_instruction` is appended and the loop retries — every prompt string the loop injects is config-driven. On `max_turns` reached, the last output is committed regardless. Repetition detection compares each failed turn's output to the previous via token-level Jaccard similarity; similarity ≥ `repetition_threshold` (default 0.92) emits `ProposalFailedEvent` immediately and converts a stuck loop into a MAPE-K retry.

Each successful turn emits `TaoIterationEvent`; each completed Explorer emits a `ProposalEvent` with `{explorer_id, tau, raw_output, token_cost, adapter_kind, tao_turns}`. Crashes, OOMs, and timeouts emit `ProposalFailedEvent`. When the JoinSet is fully drained, `GenerationPhaseCompletedEvent` is published — the stream is now closed.

The TAO physics from Definition 11: each iteration reduces effective role error cost as `c_i_eff = c_i × 0.6^(t−1)`. Shell agents (c_i=0.9) escape the BFT merge path after just t=2 turns (c_i_eff = 0.540). The merge strategy re-evaluates `max(c_i_eff)` from actual TAO turn counts before Phase 5.

The critical invariant: no Explorer reads another Explorer's output during Phase 3. Coordination cost α → 0 by graph construction.

### Phase 3.5 — Verification

When `GenerationPhaseCompletedEvent` fires, all `ProposalEvent` outputs are evaluated in parallel by an evaluator LLM using the system prompt, τ, and token budget from `VerificationConfig`. Each proposal receives a score `∈ [0, 1]` via JSON `{"score": float, "reason": string}`. Proposals at or above `threshold` (default 0.45) proceed to the Auditor; below-threshold proposals are soft-rejected via `BranchPrunedEvent` with `reason = "verification score X: <reason>"`. `VerificationScoredEvent` is published per proposal.

The fail-safe matters: parse failure or evaluator error defaults the score to 0.0. A hallucinating evaluator that returns unstructured output cannot silently pass proposals.

With P=N evaluators (Proposition 6), wall-clock cost is one T_eval regardless of ensemble size. For N≤6 this adds a constant 1–3 seconds. Simulation shows verification strictness (filter_ratio 1.0→0.0) delivers +21.9pp Q_total for Executor agents at an established ensemble (N=4); for Shell agents with 50% filter ratio the contribution is +45pp. Verification and TAO are the two highest-leverage tuning parameters once an ensemble is formed.

### Phase 3b — Review Gate (Team-Swarm Hybrid only)

When topology is `TeamSwarmHybrid` and a `ProposalEvent` from an Executor matches the `blocks` side of a declared `ReviewGate`, the orchestrator publishes `ReviewGateTriggeredEvent`. The Evaluator-role Explorer (τ ≈ 0.1, c_i ≈ 0.9) runs evaluation on only the blocked proposal and `system_context`. On approval, the proposal is forwarded to the Auditor unchanged; on rejection, `ReviewGateBlockedEvent` is published with the rejection reason and the proposal is tombstoned at the gate level — it never reaches the ADR Auditor.

The ADR Auditor only sees proposals that have passed all applicable review gates. Review gates are pre-Auditor; they do not replace it. If all Executor proposals are blocked, the count of gate-approved survivors is zero, triggering `ZeroSurvivalEvent` and autonomic retry.

### Phase 4 — Auditor Gate

The Auditor is a reactive stream processor, not a batch processor. It subscribes to `h2ai.tasks.{task_id}` as soon as `TopologyProvisionedEvent` fires, validating proposals as they arrive. For each proposal that has passed all review gates, it evaluates each `ConstraintDoc` in the corpus against the proposal text via `h2ai-constraints::eval_sync`. Predicates include `VocabularyPresence`, `NegativeKeyword`, `RegexMatch`, `NumericThreshold`, `LlmJudge`, and `Composite`.

Each constraint produces a score `∈ [0,1]`. `Hard` constraints gate the result — any Hard constraint below its threshold drives `compliance = 0.0`. `Soft` constraints contribute a weighted average. `Advisory` is informational. `constraint_error_cost = 1.0 − compliance` — derived from the compliance score, never hardcoded; this closes the loop between constraint evaluation and BFT merge strategy selection.

A pass produces `ValidationEvent`. A fail produces `BranchPrunedEvent` with the human-readable `reason`, the derived `constraint_error_cost`, and `violated_constraints: Vec<ConstraintViolation>` carrying `constraint_id`, `score`, `severity_label`, and optional `remediation_hint` per failed constraint. The branch is tombstoned — preserved for the Merge Authority UI but excluded from merge.

The Auditor reads `GenerationPhaseCompletedEvent` to know the stream is closed and counts surviving valid proposals. Survivors > 0 proceeds to Phase 5; survivors = 0 emits `ZeroSurvivalEvent`.

A structured response is required: `{"approved": bool, "reason": "..."}` JSON. A non-JSON response — regardless of content — rejects the proposal. This prevents a hallucinating auditor from passing constraints via free-text affirmations. The Auditor runs at τ = 0 with `RoleErrorCost c_i ≈ 0.9` — a false positive is near-catastrophic.

### Phase 4 → 2 — Autonomic Retry

`ZeroSurvivalEvent` triggers `RetryPolicy::decide` in `h2ai-autonomic`, which inspects the `BranchPrunedEvent` records and chooses one of three actions:

| Action | Trigger | Behavior |
|---|---|---|
| `RetryWithHints { topology, hints }` | Any pruned event has Hard `violated_constraints` with `remediation_hint.is_some()` | Collects unique remediation hints and passes them to the next Explorer generation as targeted repair guidance |
| `RetryWithTauReduction { topology, tau_factor: 0.7 }` | No structured hints AND majority of pruned reasons contain hallucination keywords | Reduces τ values by 0.7× — pushes Explorers toward more grounded outputs |
| `Retry(topology)` | Neither of the above | Plain topology escalation along the Pareto frontier (Ensemble → HierarchicalTree → TeamSwarmHybrid) |

Any variant may also escalate topology if the current one has been tried. A new `TopologyProvisionedEvent` re-enters Phase 2. Bounded by `max_retries` (default 3); on exhaustion, `TaskFailedEvent` carries the full diagnostic — every `BranchPrunedEvent`, every topology and τ set tried, and the multiplication-condition failure if Phase 2.5 was the blocker.

### Phase 5 — Merge and Human Resolution

`MergeStrategy` is selected by `MergeStrategy::from_role_costs()` at provisioning time from `max(c_i_eff)` recomputed post-TAO:

- **`ScoreOrdered`** (default, `max(c_i) ≤ 0.85`) — no coordination required. The semilattice compilation picks the highest verification-scored surviving proposal. O(1) reconciliation; epistemic diversity fully preserved.
- **`ConsensusMedian`** (`0.85 < max(c_i) ≤ 0.95`) — `ConsensusRequiredEvent` fires first. Selects the proposal with the highest mean pairwise semantic similarity (Condorcet voting). Useful for honest stochastic divergence; not Byzantine-fault-tolerant — vulnerable to coordinated identical proposals at f ≥ n/2.
- **`OutlierResistant { f }`** (`max(c_i) > 0.95` AND `krum_fault_tolerance > 0`) — `ConsensusRequiredEvent` fires first. Selects the proposal minimizing sum of semantic distances to its `n−f−2` nearest neighbours. Provably Byzantine-fault-tolerant for `n ≥ 2f+3` (Blanchard et al. 2017, Theorem 2). Engine returns `InsufficientQuorum` at provisioning if `n < 2f+3` — no Explorers dispatched. A cluster-coherence guard checks `cluster_coherent()` (mean pairwise semantic distance < 0.7) before running; an incoherent cluster falls back to `ConsensusMedian`.
- **`MultiOutlierResistant { f, m }`** — iteratively selects m survivors; the highest verification-scored survivor is the resolved output. Same coherence guard.

`SemilatticeCompiledEvent` fires with `{valid_proposals, pruned_proposals, merge_strategy}`. `h2ai-api` then renders the Merge Authority interface: a diff view grouped by target component, a Tombstone panel listing every `BranchPrunedEvent` with τ, attempted output, rejection reason, and per-constraint failure detail, an autonomic shift timeline showing every MAPE-K intervention, and a physics panel showing live `θ_coord`, `β_eff`, `N_max`, and current `MergeStrategy`.

The human makes one merge decision (select, synthesize, or reject). `MergeResolvedEvent` is published, the task is closed, and the SSE stream closes. The work is O(1) regardless of N — contradictions have already been resolved by the merge strategy.

### Compound Task Pipeline

An alternative entry point (`CompoundTaskEngine::run`) wraps a `TaskManifest` into an automatically planned and scheduled multi-step execution. `h2ai-planner::PlanningEngine::decompose` uses one LLM call to produce a `SubtaskPlan` (`PlanStatus::PendingReview`). `PlanReviewer::evaluate` runs structural checks first — empty plan and DFS White/Gray/Black cycle detection — then one LLM semantic review; rejection short-circuits without ever entering the scheduler. `SchedulingEngine::execute` partitions subtasks into waves via Kahn's topological sort, runs each wave concurrently via `join_all`, and injects completed dependency outputs into downstream subtask manifests. Each subtask dispatches through the `SubtaskExecutor` trait, decoupling scheduling from concrete execution.

---

## 4. Crate Structure

The workspace is fifteen crates with an acyclic dependency graph. Each crate has exactly one responsibility and one direction of dependency flow.

```
h2ai-config ──────────────────────────────────────────► (standalone)
h2ai-nats ────────────────────────────────────────────► h2ai-types
h2ai-constraints ────────────────────────────────────► h2ai-types

h2ai-agent (binary)
 ├── h2ai-types
 ├── h2ai-adapters → h2ai-types
 ├── h2ai-state → h2ai-nats · h2ai-types
 └── h2ai-nats → h2ai-types

h2ai-api
 └── h2ai-orchestrator → h2ai-nats
      ├── h2ai-autonomic → h2ai-config · h2ai-types
      ├── h2ai-state → h2ai-types
      ├── h2ai-context → h2ai-config · h2ai-constraints · h2ai-types
      ├── h2ai-constraints → h2ai-types
      ├── h2ai-adapters → h2ai-types
      ├── h2ai-provisioner → h2ai-nats · async-nats · h2ai-types
      ├── h2ai-memory → h2ai-types
      ├── h2ai-planner → h2ai-types
      └── h2ai-telemetry → h2ai-types
```

The rule in one sentence: every domain crate depends on `h2ai-types`. `h2ai-config` stands alone with no I/O dependencies. `h2ai-constraints` depends only on `h2ai-types` — predicate evaluation is pure computation. `h2ai-nats` owns NATS subject naming and NKey provisioning. Six crates may import `async-nats` directly — `h2ai-nats`, `h2ai-state`, `h2ai-memory`, `h2ai-telemetry`, `h2ai-provisioner`, and `h2ai-agent`. Only `h2ai-api` talks to HTTP. Nothing imports `h2ai-api`.

### Per-crate responsibilities

**`h2ai-types`** — the pure boundary. Every shared type used across crate boundaries lives here, with zero external I/O dependencies (only `serde`, `uuid`, `thiserror`, `async-trait`). It defines the UUID-backed ID newtypes, `CoherencyCoefficients` and `CoordinationThreshold`, `RoleErrorCost`, `MergeStrategy`, `SatisfactionFingerprint`, `MultiplicationCondition`, `ParetoWeights`, `AgentRole`, `TopologyKind`, `RoleSpec`, `ReviewGate`, `AdapterKind`, all configuration structs (`ExplorerConfig`, `AuditorConfig`, `TaoConfig`, `VerificationConfig`), `IComputeAdapter` and its request/response/error types, all 23 event structs and the `H2AIEvent` enum, the agent boundary types (`AgentTool`, `CostTier`, `AgentDescriptor`, `TaskRequirements`, `AgentState`, `TaskPayload`, `TaskResult`, `AgentHeartbeat`, `AgentTelemetryEvent`), and the compound-task types (`Subtask`, `SubtaskPlan`, `PlanStatus`, `SubtaskResult`). All boundary types carry `#[typeshare]` for Go/TypeScript binding generation.

**`h2ai-nats`** — owns the NATS subject namespace (`subjects` module) and the per-task ephemeral NKey lifecycle (`nkey` module). Every NATS subject string in the system originates here; no crate constructs subjects by hand.

**`h2ai-config`** — defines `H2AIConfig` carrying every physics threshold and role default: `bft_threshold`, `coordination_threshold_max`, `min_baseline_competence`, `max_error_correlation`, per-role `tau_*` and `cost_*`, token budgets, optimizer thresholds. No I/O.

**`h2ai-constraints`** — the typed `ConstraintDoc`/`ConstraintPredicate` system and synchronous predicate evaluator. The composable predicate types (`VocabularyPresence` with `AllOf`/`AnyOf`/`NoneOf` modes, `NegativeKeyword`, `RegexMatch`, `NumericThreshold`, `LlmJudge`, `Composite`), severity (`Hard`/`Soft`/`Advisory`), and the compliance formula. The loader scans a directory for `*.md` files using heading priority `## Hard Constraints` > `## Soft Constraints` > `## Advisory` > `## Constraints` (backward-compat).

**`h2ai-orchestrator`** — DAG builder, topology router, and harness engine. Builds the Explorer DAG from `TopologyProvisionedEvent`, runs context compaction before Phase 3, runs `TaoLoop::run` per Explorer (Phase 3) and `VerificationPhase::run` in parallel across proposals (Phase 3.5), classifies all errors into the four-class taxonomy, computes `HarnessAttribution`, and runs `SelfOptimizer::suggest` for the next task. Also hosts `NatsDispatchAdapter`, `CompoundTaskEngine`, and `SchedulingEngine`.

**`h2ai-autonomic`** — MAPE-K loop and calibration harness. Measures α and β₀, computes N_max and θ_coord, selects topology, assigns τ and `RoleErrorCost`, computes `MergeStrategy`, intercepts `ZeroSurvivalEvent`, and diagnoses retry actions via `RetryPolicy::decide`.

**`h2ai-state`** — the only crate that touches NATS for the task event log. Publishes events to `h2ai.tasks.{task_id}`, reads calibration data from NATS KV, replays event streams from offset 0 on crash recovery, compiles CRDT semilattice joins, and runs the BFT consensus path when required. Also implements `publish_task_payload` and `await_task_result_once` for the edge agent dispatch wire protocol — the `await` consumer must always be created *before* the publish to avoid a result-arrives-first race.

**`h2ai-context`** — Dark Knowledge Compiler. Compiles the immutable `system_context` from the constraint corpus and task manifest. Provides `bm25_search` (tantivy RAM index) and `rrf_fuse` for hybrid retrieval, and the `EmbeddingModel` trait with `semantic_jaccard` for BFT/Krum distance computations.

**`h2ai-adapters`** — implements `IComputeAdapter` for every compute backend (`AnthropicAdapter`, `OpenAIAdapter`, `OllamaAdapter`, `CloudGenericAdapter`, `MockAdapter`). `AdapterFactory::build` is the single place where `AdapterKind` resolves to a concrete adapter. Local llama.cpp inference (when wired) must use `tokio::task::spawn_blocking` — CPU-bound matrix operations cannot run on the async worker pool.

**`h2ai-tools`** — wires `AgentTool` capability flags to sandboxed executors via the `ToolExecutor` trait. `ShellExecutor` spawns `sh -c <command>` with `kill_on_drop(true)`, a 5-second timeout, and a 1MiB output cap. Tool execution is fully decoupled from the event log.

**`h2ai-provisioner`** — `AgentProvider` trait with `ensure_agent_capacity`, `terminate_agent`, and `select_agent`. `select_agent` applies a three-stage filter (capability, cost ceiling, then `SchedulingPolicy`). Two policies (`LeastLoadedPolicy`, `RoundRobinPolicy`) and three providers (`StaticProvider` for externally managed containers, `KubernetesProvider` for per-task Jobs, `NatsAgentProvider` for long-lived edge agents).

**`h2ai-memory`** — `MemoryProvider` trait makes edge agents stateless by managing conversation history and semantic context in the control plane. `InMemoryCache` for development; `NatsKvStore` persists across restarts.

**`h2ai-planner`** — `PlanningEngine::decompose` and `PlanReviewer::evaluate`. Pure LLM operations through `IComputeAdapter` only — no NATS, no state writes, no orchestration DAG. Decomposition logic can be tested with `MockAdapter` without any orchestrator setup.

**`h2ai-telemetry`** — `AuditProvider` trait with `DirectLogProvider` and `BrokerPublisherProvider`. `RedactionMiddleware` wraps any provider, scanning string fields for known API key patterns and replacing them with `[REDACTED]` before the event reaches the provider. Secret redaction is mandatory by construction, not by convention.

**`h2ai-api`** — the only crate that talks to HTTP. axum runs on the same Tokio runtime as the orchestrator and NATS consumer. Hosts the REST endpoints, the SSE/WebSocket event stream, the recovery endpoint, the merge endpoint, and the Prometheus metrics endpoint. Nothing imports `h2ai-api`.

**`h2ai-agent`** — the edge agent binary. Connects to NATS at startup, generates a stable `AgentId`, publishes `AgentHeartbeat` every 10 seconds with the live `active_tasks` count, subscribes to `h2ai.tasks.ephemeral.>` and processes only addressed messages, executes each `TaskPayload` through its local `IComputeAdapter`, publishes `TaskResult` to the `H2AI_RESULTS` JetStream stream, and listens on `h2ai.control.terminate.{agent_id}` for graceful shutdown.

### Enforcement

These rules are enforced by Cargo's dependency graph, not by convention. `h2ai-types`, `h2ai-config`, and `h2ai-constraints` will fail to compile if any I/O dependency is added. A crate cannot import `axum` unless it is `h2ai-api`. The boundary is the compiler.

### Thread Pool Isolation

Tokio runs two thread pools. The async worker pool handles NATS consumers, the MAPE-K loop, axum HTTP handlers, and cloud adapter HTTP calls — `num_cpus` threads by default. The blocking thread pool handles llama.cpp FFI inference via `spawn_blocking` — `max_blocking_threads` is set explicitly at runtime builder construction, calibrated to available RAM. Too many blocking threads spike OS scheduler contention (which raises α — the very coefficient the system is trying to minimize); too few drop inference throughput below N_max capacity.

---

## 5. Technology Stack

| Layer | Decision | Rationale |
|-------|----------|-----------|
| Language | Rust + Tokio | Compiler-verified CRDT state, zero-cost FFI to llama.cpp, no GC jitter contaminating β₀ measurements |
| Event log | NATS JetStream | Single static binary, Tokio-native `async-nats`, clusters natively for Cloud Plan |
| State model | Event-sourced CRDT | α → 0 during generation (no locks), full epistemic provenance chain, crash recovery = replay from offset 0 |
| Local compute | llama.cpp via Rust FFI | Zero-cost, 128GB RAM dedicated to model weights |
| HTTP layer | axum | Tokio-native, type-safe routing, same async runtime as orchestrator |
| Tracing | `tracing` + OpenTelemetry → Jaeger / Grafana Tempo | task_id as root span, DAG execution visible as a trace tree |
| Metrics | `metrics` + `metrics-exporter-prometheus` | USL physics gauges + hardware utilization |

NATS is chosen over Kafka because Kafka requires a JVM and a ZooKeeper/KRaft cluster consuming 1–4GB of RAM to operate. In Local Plan, a single workstation runs llama.cpp with 128GB RAM dedicated to model weights — RAM Kafka would consume should be serving inference. NATS runs as a single static binary in megabytes and provides both the event log (JetStream) and the calibration cache (KV) from one binary, with identical semantics across Local, Server, and Cloud deployments. The deeper point: NATS JetStream is not a message broker choice. It is the *shared immutable information ground* — the substrate that orchestrator, agents, audit system, and human UI all read from to reconstruct state at any point in time.

### Production Harness Components

A production agent harness requires twelve components (Anthropic/OpenAI/LangChain convergence, 2026):

| Component | H2AI Implementation | Crate |
|-----------|--------------------|----|
| Orchestration loop | TAO loop per Explorer — Thought-Action-Observation, max 3 turns | `orchestrator::tao_loop` |
| Tools | `ToolRegistry` with typed `AgentTool` key; `ShellExecutor` with `kill_on_drop`, 1MiB cap | `h2ai-tools` |
| Memory | `h2ai-memory` crate; `NatsKvStore` persists history; `SessionJournal` enables crash recovery | `h2ai-memory`, `orchestrator::session_journal` |
| Context management | Compaction with Lost-in-Middle mitigation; head+tail preservation; keyword injection | `context::compaction` |
| Prompt construction | `system_context` + `task` + TAO observation feedback; compacted before Phase 3 | `context` |
| Output parsing | `raw_output: String`; structured schema validation | `h2ai-types` |
| State management | `TaskStore` (write-through cache) + NATS JetStream (authoritative log); event-sourced CRDT | `state`, `h2ai-nats` |
| Error handling | 4-class error taxonomy: transient / recoverable / user-fixable / unexpected | `orchestrator::engine` |
| Guardrails / safety | Typed `ConstraintDoc` corpus (Hard/Soft/Advisory); ADR Auditor; Multiplication Condition; NKey scoping | `h2ai-constraints`, `adapters`, `context` |
| Verification loops | Phase 3.5 scored LLM-as-judge; `filter_ratio` feedback to attribution; graceful fallback | `orchestrator::verification` |
| Subagent orchestration | Topology provisioning (Ensemble, HierarchicalTree, TeamSwarmHybrid); MAPE-K retry | `autonomic`, `orchestrator` |
| Harness attribution | USL quality decomposition: baseline + topology_gain + verification_gain + tao_gain | `orchestrator::attribution` |

The unique differentiator is the last row: Harness Attribution does not exist in any competing framework. H2AI is the only system that can quantify — with USL math — how much each harness component contributed to output quality.

---

## 6. Deployment Plans

The system is C-first: Cloud Plan is the architectural foundation. Server Plan is the human interface layer on top. Local Plan is the full Cloud+Server stack on one machine. The CRDT state model, NATS topology, and event vocabulary are identical across all three.

| Plan | Hardware | Agent provider | Memory provider | Telemetry provider |
|------|----------|---------------|-----------------|-------------------|
| **Local** | Single workstation | StaticProvider | InMemoryCache | DirectLogProvider |
| **Server** | Dedicated server/VM | NatsAgentProvider (recommended) / StaticProvider | NatsKvStore | BrokerPublisherProvider + RedactionMiddleware |
| **Cloud** | Kubernetes cluster | KubernetesProvider | NatsKvStore | BrokerPublisherProvider + RedactionMiddleware |

### Edge Agent Model

Edge agents are ephemeral, stateless LLM-based containers described by `AgentDescriptor { model: String, tools: Vec<AgentTool> }`. The `model` selects the LLM backend (any string: `"llama3-70b"`, `"gpt-4o"`, `"claude-3-opus"`); `tools` is a set of capability flags (`Shell`, `WebSearch`, `CodeExecution`, `FileSystem`) granted at launch. The control plane dispatches a `TaskPayload` (carrying the full `AgentDescriptor`) over NATS, receives a `TaskResult`, and streams `AgentTelemetryEvent` entries. The agent has no persistent state — all context is assembled by `h2ai-memory` and injected via the payload. NATS credentials are scoped NKeys that expire when the task closes.

The `tools` field is not cosmetic. It is the primary input to three physics quantities:

- **c_i (role error cost):** determined by tool destructiveness — pure LLM c_i ≈ 0.1, Shell agent c_i ≈ 0.6–0.9. When `max(c_i) > 0.85`, `MergeStrategy` shifts toward `ConsensusMedian` and then `OutlierResistant`.
- **α (serial contention):** tool calls introduce serialization. A pool of Shell-capable agents has α ≈ 0.20–0.30, lowering N_max so the planner provisions fewer Explorers.
- **β₀ (pairwise coherency cost):** WebSearch and external API calls introduce retrieval nondeterminism, raising CG variance and therefore β_eff.

The full mapping of capability set to physics:

| Capability | α contribution | β₀ contribution | c_i (error cost) |
|---|---|---|---|
| Pure LLM | ~0 | ~0 | 0.1–0.3 |
| WebSearch | +0.01–0.02 | +0.005 | 0.2–0.4 |
| FileSystem | +0.02–0.05 | +0.010 | 0.4–0.6 |
| CodeExecution | +0.03–0.08 | +0.015 | 0.5–0.7 |
| Shell | +0.05–0.15 | +0.020 | 0.6–0.9 |

### Managed Agents Patterns

The Server and Cloud plans use three complementary patterns that together make the harness crash-safe and observable without an external database. **JetStream as authoritative session log**: every `H2AIEvent` is appended to a durable JetStream stream keyed by `task_id`; the stream is the single source of truth, and `TaskStore` is a write-through cache reconstructed by `SessionJournal::replay` on miss. **Harness recovery via wake pattern**: `GET /tasks/{task_id}/recover` replays the JetStream log, upserts the reconstructed `TaskState` into the live `TaskStore`, and returns current status — a server restart requires one HTTP call per in-flight task to restore the in-memory cache, not a database restore. **Live agent registry**: `NatsAgentProvider` maintains a live registry of managed agents that self-register via NATS heartbeats; the provisioner selects agents by `AgentDescriptor` match, dispatches `TaskPayload` over NATS, and issues soft-kills via `h2ai.agents.kill.{agent_id}`.

See [`operations.md`](operations.md) for per-plan startup sequences, NATS cluster configs, Kubernetes topology, observability metrics, and environment variables.

---

## 7. Design Rationale

### Why USL for Bounding Rather Than Empirical Tuning

Setting `max_agents = 3` because "3 seems reasonable" means it will be wrong for tasks with high agent divergence, wrong for tasks with low divergence, and wrong when the adapter pool changes. USL provides a closed-form expression for the throughput ceiling of a coordination-dependent system, empirically validated across CPU architectures, database connection pools, and (by structural analogy) human engineering teams. Its two-parameter form maps cleanly to the two types of coordination cost in agent orchestration.

The honest limitation: with M < 3 adapters, the USL fit degenerates and the system falls back to configured default parameters. Most small deployments run on conservative defaults (calibrated to produce N_max ≈ 4–6 for typical LLM ensembles), not empirically derived values. The "physics-derived N_max" is fully physics-derived only when M ≥ 3 adapters are available for calibration.

### Why NATS JetStream over Kafka or In-Process Queues

Multi-agent orchestration state must survive crashes. When five agents are mid-execution and the orchestrator restarts, the system must resume — not retry. Retrying costs inference tokens, time, and idempotency guarantees for tool-using agents that may have already written files or called external APIs. Kafka requires JVM RAM that competes with model weights in Local Plan. In-process queues (tokio channels, broadcast) do not survive restarts and prevent multi-node deployment. NATS JetStream provides immutable ordered append, KV semantics for the calibration cache, and identical behaviour across local, server, and cloud — from one binary. The deeper framing is that NATS is not chosen as a message broker; it is chosen as the shared immutable information ground.

### Why Event-Sourced CRDTs over a Database

CRDT semilattice (append-only LUB join) over NATS JetStream means no lock is required during agent generation — agents write to their own proposal subjects; the merge engine reads all of them after `GenerationPhaseCompletedEvent` closes the phase. CRDT join semantics are coordination-free by definition: any order of append operations produces the same merged state. Optimistic locking would require a sequence number handshake on every write, defeating the goal of zero-coordination during generation. PostgreSQL serializes writes behind a transaction lock — α=1 during generation. Redis adds a network round-trip per write. NATS JetStream is co-located with the orchestrator in Local Plan, provides equivalent event-sourcing semantics, and adds zero external process dependencies.

### Why Rust over Python or Go

The orchestrator must run inference-aware workloads (calibration timing, CRDT merge operations) on the same machine as llama.cpp model inference, without GC pauses or runtime overhead interfering with timing measurements. CRDT state is compiler-verified (the borrow checker ensures no concurrent mutation of semilattice state). Zero-cost FFI to llama.cpp shares one process without marshaling. No garbage collector means calibration timing is not contaminated by GC pauses that would corrupt α and β₀ estimates. Go's GC is low-latency but not zero-latency — under 128GB of model weights, GC pressure could produce timing spikes corrupting calibration. Python is appropriate for scripting calibration validation, not for the production runtime; the GIL prevents true parallel calibration phases and asyncio adds abstraction overhead obscuring timing.

### Why Scoped NKeys per Task, Not Long-Lived API Keys

Tool-using containers with Shell, CodeExecution, FileSystem access need NATS credentials. Long-lived credentials in containers create two risks: a compromised container can read other agents' payloads, and credentials survive task completion to be reused. Each task gets a fresh NATS NKey scoped to exactly the subjects that task's agents need to publish to: `h2ai.telemetry.{agent_id}`, `audit.events.{agent_id}`, `h2ai.results.{task_id}`. The NKey is provisioned at dispatch time and expires when the task closes. Scoping is enforced at the NATS server level — not by application code that could be bypassed. NKeys are Ed25519 key pairs with subject-level permission scoping built into the NATS auth model; the `allowed_publish` set is sized to match the agent's tool set.

### Why the Condorcet and USL Models Are Used Together

USL alone gives a coordination ceiling but not a quality target. CJT alone gives a quality gain but no cost constraint. Using both means the effective ensemble size is `min(N_optimal, N_max)` — large enough to gain from the Condorcet effect, small enough not to degrade from coordination overhead. When N_optimal exceeds N_max, the system runs at N_max and compensates with topology — Hierarchical Tree reduces coordination cost from O(N²) to O(N).

---

## 8. Differentiation

### Coordination Cost Is Measured, Not Guessed

Every other framework lets you set `max_agents`. H2AI derives N_max from measured parameters and re-runs calibration when the adapter pool changes, the task domain shifts, or models update. A framework that lets you add agents without bound is actively harmful once the coordination ceiling is crossed. H2AI tells you where that ceiling is.

### System State Is an Immutable Event Log

Every state transition — task bootstrapped, agent dispatched, proposal received, proposal rejected, merge decision made — is appended to NATS JetStream. Crash recovery is replay from offset 0. The same state model runs in development and production. In practice this means audit (every rejected proposal is a permanent record with the reason, the violated constraint, and the remediation hint — required for SOX, HIPAA, SOC2), reproducibility (the exact sequence of events that produced an output is replayable), and debugging (a JetStream query, not a log grep).

### Constraints Are Typed Predicates, Not Prompt Instructions

Every other framework encodes safety constraints in the system prompt as natural language ("do not use G1GC", "always use idempotency keys") — suggestions the LLM may or may not follow, with no verification mechanism. H2AI constraints are `ConstraintDoc` instances with typed predicate bodies (`VocabularyPresence`, `NegativeKeyword`, `RegexMatch`, `NumericThreshold`, `LlmJudge`, `Composite`), each with severity (`Hard`/`Soft`/`Advisory`). A proposal that violates ADR-004 ("use Redis Lua for atomic budget operations") is structurally impossible to reach the human — the Auditor rejects it before merge. This is defense-in-depth, not a single-layer probabilistic guardrail.

### Tool-Using Agent Risk Is Quantified

Every other framework treats a file-writing agent and a pure-reasoning agent identically at the orchestration layer. H2AI's `AgentTool` flags directly affect three measured quantities (α, β₀, c_i). When `max(c_i)` exceeds the BFT threshold, `MergeStrategy` switches automatically — Byzantine-resistant selection that minimizes damage from a single high-error-cost agent producing a wrong output with irreversible side effects. No configuration is required.

### Agent Credentials Are Task-Scoped

Long-lived credentials in containers let a compromised agent read other agents' payloads or impersonate future agents. H2AI scopes NATS NKeys to exactly the subjects that task's agents can publish to, expiring when the task closes. Scoping is enforced at the NATS server, not by application code. This is a security property no Python-based framework provides.

### Human Decision Is O(1), Not O(N)

The CRDT Merge Authority presents the merged diff grouped by target component (not raw agent outputs), the Tombstone panel with every rejected proposal and reason, the autonomic shift timeline, and the live physics panel. The human makes one decision. Contradictions have already been resolved by the merge strategy. The work is O(1) regardless of N.

### Layer Positioning vs DSPy and Ray

H2AI is not a competitor to DSPy or Ray. DSPy optimizes individual prompts (a sub-problem of "how to make one agent better"); Ray is a generic distributed compute fabric. H2AI is the orchestration layer above either: it decides *how many* agents to fan out, with what role assignment, and how to merge their outputs. A production deployment can plausibly compose DSPy-optimized prompt programs running as Explorers on a Ray cluster, with H2AI as the control plane bounding ensemble size, enforcing constraints, and computing harness attribution.

---

## 9. Theory to Implementation

The mathematical apparatus maps to concrete code locations as follows:

| Mathematical concept | H2AI implementation |
|---------------------|---------------------|
| Calibrate α, β₀, CG_mean | `POST /calibrate` → `CalibrationCompletedEvent`; `h2ai-autonomic` |
| Constraint fingerprint and diversity gate | Dark Knowledge Compiler in `crates/h2ai-context` compiles `system_context` unconditionally; diversity gate in Phase 3.5 checks mean pairwise Hamming distance on `SatisfactionFingerprint` across surviving proposals |
| N_max ceiling (Prop 1) | Explorer count capped at N_max during `TopologyProvisionedEvent`; `h2ai-types::CoherencyCoefficients::n_max()` |
| Multiplication Condition (Prop 3) | Phase 2.5 hard gate; `MultiplicationConditionFailedEvent` names which condition failed |
| θ_coord threshold (Prop 2) | Stored in calibration cache; enforced at topology construction in `h2ai-autonomic` |
| CRDT merge (Prop 4, 5) | `MergeStrategy::ScoreOrdered` / `MergeStrategy::ConsensusMedian` in `crates/h2ai-state` |
| BFT merge (Prop 5 safety constraint) | `MergeStrategy::OutlierResistant { f }` when `max(c_i) > 0.95`; `ConsensusRequiredEvent` signals this path; cluster-coherence guard in `crates/h2ai-state` |
| Auditor constraint checking (Def 10) | `BranchPrunedEvent` with constraint citation, emitted by Auditor adapter; predicate evaluation in `h2ai-constraints::eval_sync` |
| MAPE-K retry on zero survival | `ZeroSurvivalEvent` → `crates/h2ai-autonomic` `RetryPolicy::decide` → `RetryWithHints` / `RetryWithTauReduction` / `Retry` → new `TopologyProvisionedEvent` |
| Topology selection (three frontiers) | Phase 2: `roles[]` → `TeamSwarmHybrid`; explicit `kind` field; auto from `ParetoWeights` and N vs N_max |
| Abstract `AgentRole` enum | `h2ai-types::AgentRole` — Coordinator / Executor / Evaluator / Synthesizer / Custom |
| Review gate (intra-swarm Evaluator gate) | Phase 3b: `ReviewGateTriggeredEvent` → Evaluator runs → approve or `ReviewGateBlockedEvent` |
| N_max^interface (Team-Swarm binding ceiling) | `crates/h2ai-autonomic` computes from `CG(liaison, Coordinator)`; `InterfaceSaturationWarningEvent` + `h2ai_interface_n_max` metric |
| TAO error reduction (Def 11) | `TaoLoop::run` in `orchestrator::tao_loop`; `c_i_eff = c_i × 0.6^(t−1)` recomputed before Phase 5 |
| Verification filter gain (Prop 6) | `VerificationPhase::run` in `orchestrator::verification`; parallel evaluation; fail-safe 0.0 score on parse error |
| Harness attribution (Q_total decomposition) | `orchestrator::attribution::HarnessAttribution`; computed from actual TAO turns and `filter_ratio` |
| EigenCalibration (N_eff) | `h2ai-autonomic` calibration harness; spectral analysis of pairwise CG |
| Compound task decomposition | `h2ai-planner::PlanningEngine::decompose` + `PlanReviewer::evaluate`; `CompoundTaskEngine` and `SchedulingEngine` in `h2ai-orchestrator` |

The boundary between theory and code is intentional. Every mathematical quantity is observable as a metric or an event. Every event is replayable. Every configuration knob that affects physics has a default in `H2AIConfig` and a calibration value override path. The compiler enforces dependency direction; the runtime enforces phase order; the operator can always answer "why did the system make this decision" by reading the JetStream log.

See [`math.md`](math.md) for the formal math apparatus, [`reference.md`](reference.md) for full event and API schemas, and [`operations.md`](operations.md) for deployment specifics.
