# H2AI Control Plane

[![CI](https://github.com/immediatus/h2ai-control-plane/actions/workflows/ci.yml/badge.svg)](https://github.com/immediatus/h2ai-control-plane/actions/workflows/ci.yml)
[![Theoretical Foundation](https://img.shields.io/badge/Framework-The_Coordination_Constant-blue)](https://e-mindset.space/blog/coordination-constant-usl-human-ai-teams/)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-orange)](LICENSE)
[![Language](https://img.shields.io/badge/Language-Rust-orange)](https://www.rust-lang.org/)

> **A Rust LLM multi-agent orchestration runtime that replaces "spawn-N-agents-and-hope" frameworks with measured, math-grounded quality guarantees.**

Most multi-agent frameworks treat quality as an output property — something you observe after the fact, hope is good enough, and retry if it isn't. H2AI treats quality as a control variable: measured before generation starts, bounded by physics, and enforced by typed constraints at every stage.

The core claim: **adding more agents past a measurable ceiling actively degrades output quality**. The Universal Scalability Law gives you that ceiling exactly. H2AI computes it per deployment, enforces it per task, and closes the feedback loop via a MAPE-K autonomic engine that adjusts ensemble topology before quality falls off the cliff.

**H2AI Control Plane** is a distributed multi-agent orchestration runtime that prevents LLM agent swarms from degrading under their own coordination cost. It measures the overhead of making N agents agree, bounds ensemble size before that overhead exceeds the quality gain, and enforces typed constraints so agents share enough common ground to produce coherent results.

Reference implementation of the framework defined in **[One Equation Governs CPU Caches, Human Teams, and AI Agent Systems](https://e-mindset.space/blog/coordination-constant-usl-human-ai-teams/)**.

---

## The Coordination Cost Problem

When N agents work on a shared problem, two forces act in opposite directions.

**The quality force** pushes toward more agents: each independent perspective reduces the chance that a wrong answer survives to the output (Condorcet Jury Theorem). More agents → higher probability of majority-correct result.

**The coordination force** pushes back: to produce a *coherent* output, every pair of agents' conclusions must be checked for compatibility. That is O(N²) reconciliation operations. This is the split-brain problem applied to reasoning. Agents that started from the same context but reached divergent partial conclusions cannot simply be concatenated — their incompatibilities must be found and resolved. The cost of finding them grows quadratically.

These two forces have an intersection. The Universal Scalability Law describes it precisely:

```
X(N) = N / (1 + α(N−1) + β·N(N−1))

where:
  α = serial fraction — planning, context compilation, final synthesis;
      phases that cannot be parallelized regardless of N
  β = coherence-drag coefficient — the cost each new agent adds to producing a
      coherent final output. In LLM ensembles β has two physical components:
        (1) conflict reconciliation: at merge, every contradictory agent-pair must
            be detected and resolved — O(N²) constraint-fingerprint comparisons.
        (2) context-attention degradation: as N proposals fill the synthesis LLM's
            context, retrieval quality for proposals buried deep degrades ("Lost in
            the Middle", Liu et al. 2023) — super-linear in N.
```

The peak of X(N) is `N_max = √((1−α)/β_eff)`. Beyond N_max, adding agents actively degrades output quality because coherence-drag exceeds the Condorcet quality gain.

**This is not a new observation.** Brook's Law (1975) measured it in human engineering teams — communication channels grow as N(N−1)/2. CPU cache coherency protocols hit the same ceiling at a different scale. LLM agent swarms exhibit the same phenomenon for the same structural reason: pairwise synchronization overhead scales quadratically with group size when agents must reach mutual consistency.

The β coefficient is modulated by **Common Ground (CG)** — the agreement rate across the calibration adapter pool, measured as mean pairwise Hamming distance on constraint-satisfaction fingerprints. High CG means agents satisfied compatible constraints; low CG means they diverged and conflict reconciliation is costly. `β_eff = β₀ × (1 − CG_mean)`. At CG=1 (full overlap) β_eff ≈ 0; at CG=0 β_eff = β₀.

H2AI measures both forces, finds their intersection, and enforces a Common Ground floor (θ_coord) before allowing generation to start — preventing split brain before it begins rather than trying to repair it after.

---

## Why It Exists

| Problem | Standard Approach | H2AI Approach |
|---|---|---|
| Hallucination amplification | Hope the model self-corrects | Auditor node (τ→0) mathematically blocks propagation |
| Judge self-preference bias | Same model judges itself | `JudgePanel` — cross-family verification panel; supermajority aggregation (≥2 families) or unanimous persona-diversity rule (single family); `ConstraintAmbiguityEvent` fires when corpus rubric is ambiguous |
| State lives in the model | LLM context window (lossy) | Orchestrator owns state; models are stateless `f(ctx, τ) → text`. CRDTs track **constraint-satisfaction fingerprints** (metadata), never LLM text. Text is reconciled by the synthesis LLM. |
| Safety is probabilistic | "Don't do X" in the prompt | Topological interlocks — invalid output cannot reach the human by graph construction |
| More agents = worse results | Keep adding until it breaks | MAPE-K loop computes N_max, shifts topology before retrograde |
| Retry loop repeats same failed approach | Hope next attempt differs | Epistemic Leader — Krum-elected leader generates Socratic diagnostic question per failed wave; follower context forced to distinct constraint dimensions; credibility-weighted rotation on stagnation |
| Tacit knowledge is invisible | Agents guess team constraints | Dark Knowledge Compiler — typed `ConstraintDoc` predicates (Hard/Soft/Advisory) become hard Auditor gates; `constraint_error_cost = 1 − compliance` |
| Constraint corpus is static and fragile | Bulk file reload loses history | Constraint Wiki (`H2AI_CONSTRAINT_WIKI` KV) — hot-reload via NATS KV watch; `ConstraintSource` trait decouples corpus access from storage; `ConstraintSnapshot` in every checkpoint records which wiki revision was active |
| Flat constraint retrieval returns all-or-nothing | Inject entire corpus or keyword-match | Hierarchical Knowledge Provider — `Bm25WikiProvider` scores constraints via BM25+/PPR across a Global → Topic → Leaf tree; dual RAPTOR modes (TreeTraversal + CollapsedTree); PPR multi-hop expansion surfaces related constraints; `GlobalKnowledge` and `TopicKnowledge` context sections are injected at importance 1.0 and 0.8 respectively |
| Human babysits every step | Constant correction loop | Merge Authority — human resolves a structured CRDT diff once, at the end |
| Low-confidence outputs reach callers silently | Every output looks the same | HITL Approval Gate — `q_confidence < threshold` or `require_approval = true` parks the output in `H2AI_APPROVALS` KV; `PendingApprovalEvent` streams immediately; 30-minute reaper auto-rejects expired records |
| Prompt quality drifts without feedback | Manually tune prompts per deployment | Adaptive Prompt Harness — OPRO (arXiv 2309.03409) auto-improves prompts when j_eff EMA falls; Thompson bandit selects best variant |
| Edge agent secrets leak | Long-lived API keys in containers | Scoped NATS NKeys per task_id — token expires when task closes |
| Shell injection via LLM output | `sh -c <llm_string>` executes metacharacters | `ShellExecutor` uses JSON contract + `Command::new(cmd).args(args)` — no shell interpreter; PGID-scoped process group kill on timeout |
| Hardened wave still has full tooling | One tool policy for all waves | `WaveMode` (Normal/Hardened) on `TaskPayload`; `ToolRegistry::for_wave()` selects a reduced allowlist for `ConstrainedExploration` and `ModeCollapse` retries |
| Context lost between restarts | Agents rebuild context from scratch | MemoryProvider — control plane assembles and injects context before dispatch |
| Audit log is an afterthought | Logs scattered across containers | AuditProvider with redaction middleware — immutable telemetry on NATS JetStream |

---

## Quick Start

### Devcontainer (recommended)

Open in any devcontainer-compatible environment. NATS starts automatically as a sidecar.

```bash
git clone https://github.com/h2ai/control-plane.git
# Open in devcontainer — NATS and environment are pre-configured
```

### Local (Local Plan)

```bash
git clone https://github.com/h2ai/control-plane.git
cd h2ai-control-plane/deploy/local
docker compose up -d

# Seed your constraint corpus (your team's architectural decisions)
mkdir -p ../../constraints
cp -r ../../tests/e2e/constraints/* ../../constraints/

# Calibrate the adapter pool
curl -X POST http://localhost:8080/calibrate

# Task 1 — pure reasoning (no tools)
# 3 pure LLM explorers reason in parallel about the architecture decision.
# c_i ≈ 0.1 (text output, discard at zero cost) → CRDT merge.
# CONSTRAINT-004 and CONSTRAINT-005 are loaded from the corpus; others are skipped.
# tenant_id is a URL path segment — use "default" for single-tenant deployments.
curl -X POST http://localhost:8080/v1/default/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Design a budget enforcement mechanism that prevents double-billing during server restarts",
    "pareto_weights": {"diversity": 0.5, "containment": 0.4, "throughput": 0.1},
    "explorers": {"count": 3, "tau_min": 0.2, "tau_max": 0.85},
    "constraints": ["CONSTRAINT-004", "CONSTRAINT-005"]
  }'

# Task 2 — code generation (containment-weighted, tight τ band)
curl -X POST http://localhost:8080/v1/default/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Write and test a Redis Lua script for atomic budget check-and-decrement with 30s TTL idempotency",
    "pareto_weights": {"diversity": 0.3, "containment": 0.6, "throughput": 0.1},
    "explorers": {"count": 3, "tau_min": 0.2, "tau_max": 0.5},
    "constraints": ["CONSTRAINT-004"]
  }'

# Stream events in real time
curl -sN http://localhost:8080/v1/default/tasks/{task_id}/events

# If the task suspends with PendingClarificationEvent (oracle gate + low confidence),
# supply an operator answer to resume:
curl -X POST http://localhost:8080/v1/default/tasks/{task_id}/clarify \
  -H "Content-Type: application/json" \
  -d '{"answer": "your clarification here"}'

# Open the Merge Authority UI
open http://localhost:8080
```

### Enterprise (Kubernetes + Helm)

```bash
helm repo add h2ai https://h2ai.github.io/control-plane
helm repo update

kubectl create configmap constraint-corpus --from-file=./constraints/ -n h2ai

helm install h2ai h2ai/h2ai-control-plane \
  --namespace h2ai --create-namespace \
  --set ingress.enabled=true \
  --set ingress.hosts[0].host=h2ai.corp.example.com \
  --set serviceMonitor.enabled=true
```

---

## The Epistemological Architecture

Every problem submitted to H2AI is, at its core, a **team knowledge acquisition problem**. The system's job is not to produce text — it is to build a knowledge graph whose nodes are beliefs about the problem domain and whose edges are support, contradiction, derivation, and grounding relationships. When that graph reaches coherent closure, the system stops.

This framing maps directly to four nested control loops, each with an epistemic stopping criterion rather than a mechanical one:

| Loop | Scope | Stops when |
|------|-------|-----------|
| **TAO** (Thought–Action–Observation) | Within one agent | The agent has exhausted productive reasoning paths — no tool call changes the belief set |
| **MAPE-K** (Monitor–Analyse–Plan–Execute) | Across the committee | The knowledge graph has reached coherent closure — surviving proposals are consistent with the constraint corpus |
| **Calibration** | Across tasks | Calibration confidence intervals are tight enough — α, β, CG priors are stable across the adapter pool |
| **Oracle / Grounding** | Across reality | External truth has verified the claims that are load-bearing for the output — human or automated oracle confirms the result |

These loops are **not** a retry policy or a pipeline stage. They are four nested epistemological systems operating at different time-scales. The TAO loop refines a belief within a single agent-turn. The MAPE-K loop detects when the committee's collective knowledge has failed coherence and repairs it by restructuring the generation. The calibration loop updates the system's **meta-beliefs** — its beliefs about its own agent quality — so that future committees are better sized and composed. The grounding loop connects the entire system to external reality, preventing the ensemble from converging on a coherent but wrong answer.

The stopping criteria are what distinguish this from a retry loop: the system stops not when it has run a fixed number of iterations, but when it has **acquired enough knowledge**. Each loop has a precise definition of "enough":
- TAO: `tool_calls_made == 0` on the last iteration (no new evidence to gather)
- MAPE-K: `ZeroSurvival` not triggered — all proposals passed audit
- Calibration: confidence interval width on `(α, β, CG)` drops below the configured precision threshold
- Oracle: `q_confidence ≥ approval_threshold` or human operator provides explicit approval signal

This architecture is why H2AI is described as an **epistemic control plane** rather than an orchestration framework. Orchestration coordinates processes. H2AI coordinates the acquisition, validation, and grounding of knowledge.

---

## How It Works

### 1. Calibration — measure the physics, then spawn

Before any agent runs, the calibration harness measures three parameters from the adapter pool:

- **α** — serial bottleneck fraction: time in planning, context compilation, and synthesis — phases that serialize regardless of N. Measured as the fraction of total wall time that scales with N=1 behavior.
- **β₀** — pairwise reconciliation cost: how expensive it is to integrate each pair of agents' outputs into coherent agreement. Measured from merge-phase timing and output divergence. Scales N(N−1) in the USL model.
- **CG(i,j)** — Common Ground between every Explorer pair: mean pairwise Hamming distance on binary constraint-satisfaction fingerprints (1 bit per constraint: pass/fail). High CG = agents satisfied compatible constraints; low CG = divergence is costly to reconcile. This is a metadata measurement — the control plane never compares raw LLM text.

These yield `N_max = sqrt((1−α) / β_eff)` where `β_eff = β₀ × (1 − CG_mean)`. Past `N_max`, every additional agent makes results worse. No task starts without this data — calibration is not optional and not a one-time setup.

### 2. Bootstrap — dark knowledge compiled into explicit gates

The task manifest arrives. The **Dark Knowledge Compiler** assembles an immutable `system_context` from your constraint corpus and the manifest. Every `ConstraintDoc` becomes either a hard gate (fail = reject the proposal) or a weighted compliance score (`constraint_error_cost = 1 − compliance`). Every agent — Explorer and Auditor alike — receives exactly this context and nothing else. Tacit knowledge is now explicit and enforceable.

When `thinking_loop.enabled`, a structured multi-archetype brainstorm runs first — iterating until `coverage_score ≥ coverage_threshold`, with linear τ annealing and tension-targeted injection for unresolved gaps. The `ThinkingReport.shared_understanding` is injected into the decomposition prompt, improving decomposition quality on complex multi-domain tasks.

### 3. Provisioning — topology selected by physics, not gut feel

The MAPE-K controller reads `{α, β_eff, ParetoWeights}` and selects one of three topologies:

- **Ensemble + CRDT** — `N ≤ N_max`, diversity-weighted. All Explorers are peers. O(N²) edges — structurally fine for small N. Pareto: T=84%, E=84%, D=90%.
- **Hierarchical Tree** — `N > N_max` or containment-weighted. One Swarm Coordinator + k sub-groups; `k_opt = floor(N_max^flat)`. Coordination cost drops from O(N²) to O(N). Pareto: T=96%, E=96%, D=60%.
- **Team-Swarm Hybrid** — role-differentiated Explorers (Coordinator, Executor, Evaluator, Synthesizer) with declared review gates between specified pairs. The Evaluator forms a pre-Auditor gate that blocks Executor output before it can reach synthesis. Pareto: T=84%, E=91%, D=95%.

Before spawning a single inference token, the **Multiplication Condition Gate** enforces Proposition 3: competence > 0.5, error decorrelation ρ < 0.9, `CG_mean ≥ θ_coord`. Fail any one → re-enter provisioning with adjusted parameters.

### 4. Generation — parallel, isolated, edge agents with full TAO loops

N Explorers run in a `tokio::task::JoinSet`. No Explorer reads another's output — coordination cost during generation is structurally zero. Each Explorer is an **ephemeral, stateless edge agent** (`AgentDescriptor { model, tools }`) dispatched via NATS with a scoped NKey that expires when the task closes.

The `tools` field is not a feature flag — it directly affects USL quantities:

| Tool set | Effect on α | Effect on β₀ | Default c_i | Typical role |
|---|---|---|---|---|
| `[]` (pure LLM) | near 0 | 0 | 0.1–0.3 | Coordinator / Synthesizer |
| `[WebSearch]` | +0.01–0.02 | +0.005 | 0.2–0.4 | Evaluator |
| `[FileSystem]` | +0.02–0.05 | +0.01 | 0.4–0.6 | Executor |
| `[CodeExecution]` | +0.03–0.08 | +0.015 | 0.5–0.7 | Executor |
| `[Shell]` | +0.05–0.15 | +0.02 | 0.6–0.9 | Executor |

Inside each edge agent, `TaoAgent` runs a Thought→Action→Observation loop up to `agent_max_tool_iterations` turns. Each turn: the LLM is called with accumulated context and tool schemas in the system prompt; the response is parsed for a structured tool call `{"tool": "...", "input": {...}}` via a three-strategy pipeline (direct JSON parse → strip markdown fences → find first balanced `{...}` in text); the call is dispatched through `ToolRegistry::execute`; a `ToolCallRecord` is appended to the audit trail. Tool observations are UTF-8-safely truncated at `agent_max_observation_chars` (default 8192) before being injected back into context — preventing runaway context growth. The loop terminates when the LLM produces a final answer (no tool call) or the iteration budget is exhausted.

**Security invariant:** Each edge agent can only publish to `h2ai.telemetry.{agent_id}` and `h2ai.results.{task_id}`. It cannot read other agents' payloads, write to the orchestration bus, or retain credentials after the task closes. Enforced at the NATS server level — not by application code. Shell commands execute via `Command::new(cmd).args(args)` — no shell interpreter, no metacharacter injection possible.

### 5. MAPE-K gate — every wave reviewed, every retry typed

The Auditor spins up on `TopologyProvisionedEvent` — before generation starts. Proposals are validated as they arrive. Rejections become `BranchPrunedEvent` tombstones: permanently preserved with rejection reason and `constraint_error_cost`, visible in the Merge Authority UI. Failures are epistemic data, not noise.

Zero survivors → `ZeroSurvivalEvent`. The three-layer MAPE-K engine engages: `ExecutionPipeline` runs the 12 per-wave phase modules in order (topology → multiply → diversity → generation → hallucination → SRANI → verify → audit → frontier → oracle → synthesis → merge), `MapeKController` maps the wave's `ExitReason` to one of three `MapeKDecision` variants — `Return(output)`, `Retry`, or `Fail(error)` — and `engine.rs` orchestrates the loop. The retry adjusts `{N, τ, topology}` and reruns. Bounded by `max_retries`. Exhaustion → `TaskFailedEvent` with full diagnostic payload. The MAPE-K retry loop gains cross-wave guidance via an elected Epistemic Leader that diagnoses failures and formulates Socratic questions to prevent repeated mistakes. When `leader_enabled = true` (default: false), the highest-scoring Krum survivor is elected leader after each failed wave; it generates an EIG-ranked diagnostic question (deduplicated via belief buffer) that is injected as leader context in the next wave while followers receive the question paired with an assigned constraint aspect.

### 6. Merge — two layers, O(1) human decision

**Layer 1 — Metadata consensus (CRDT semilattice, deterministic):** The control plane aggregates binary constraint-satisfaction fingerprints from surviving proposals into a semilattice. The BFT threshold (e.g. 0.67) is applied to fingerprint agreement rate — not raw text similarity. This layer never touches LLM output text.

> The `bft_threshold` is a fractional agreement gate on constraint fingerprints — not PBFT. PBFT handles adversarial nodes with cryptographic guarantees at O(N²) message cost. Here the "Byzantine nodes" are stochastically diverging LLMs. A fractional threshold + Krum outlier rejection is the correct tool; full PBFT would be architectural overkill.

**Layer 2 — Semantic reconciliation (synthesis LLM):** The synthesis LLM receives only validated proposals. Two passes: (1) **critique** at low τ produces a structured gap analysis; (2) **synthesis** reads proposals + critique and produces the final coherent output. Before synthesis runs, a diversity gate rejects mono-culture ensembles — if mean pairwise Hamming distance across fingerprints falls below `diversity_threshold`, a MAPE-K retry fires.

The **Merge Authority UI** presents the valid proposals panel, tombstone panel (every rejection with reason and `c_i`), MAPE-K intervention timeline, and live physics panel (`θ_coord`, `β_eff`, `N_max`, `MergeStrategy`). One human decision. `MergeResolvedEvent` closes the task.

---

## The Scalability Ceiling

```
X(N) = N / (1 + α(N−1) + β_eff·N(N−1))

N_max = sqrt((1 − α) / β_eff)
```

The same law governs coordination-dependent systems at every scale. The parameters change; the structure does not.

| System | α (serial bottleneck) | β₀ (pairwise sync cost) | N_max | What α and β represent |
|---|---|---|---|---|
| CPU cache coherency | 0.02 | 0.0003 | ~57 | α = memory bus serialization; β = cache-line exchange protocol |
| Human engineering team | 0.10 | 0.0083 | ~10 | α = planning/review cycles; β = pairwise communication overhead (Brook's Law) |
| AI agents (same model) | 0.15 | 0.025 | ~4–5 | α = context compilation + synthesis; β = pairwise output reconciliation at low CG |
| AI agents (diverse backends) | 0.12 | 0.018 | ~6–7 | α = same; β lower because diverse models share less vocabulary, but diverge less on facts |

For AI agents, α captures the serial phases inherent to orchestration (you cannot parallelize task decomposition or final merge), and β captures how expensive it is to find and resolve contradictions between N agents' partial answers. Higher β = more divergence to reconcile = fewer agents before quality peaks.

Reference values: **α ≈ 0.10–0.15, β₀ ≈ 0.015–0.025, N_max ≈ 4–7** for typical LLM ensembles.
---

## The Event Vocabulary

All state is immutable event log entries on NATS JetStream. Crash recovery = replay from offset 0.

**Core orchestration events** (subject `h2ai.tasks.{task_id}`):
```
CalibrationCompletedEvent          → α, β₀, CG samples, θ_coord locked
TaskBootstrappedEvent              → system_context locked, constraint corpus compiled
TopologyProvisionedEvent           → DAG shape, τ values, RoleErrorCosts, MergeStrategy
MultiplicationConditionFailedEvent → which of 3 conditions failed, re-entering Phase 2
ProposalEvent                      → Explorer output appended, agent terminates
ProposalFailedEvent                → Explorer crashed/OOM/timeout, terminal state guaranteed
GenerationPhaseCompletedEvent      → JoinSet drained, stream closed
ReviewGateTriggeredEvent           → Evaluator gating an Executor proposal
ReviewGateBlockedEvent             → Evaluator rejected proposal (reason recorded)
ValidationEvent                    → Auditor: proposal passed
BranchPrunedEvent                  → Auditor: proposal rejected; violated_constraints[] per-constraint scores + remediation hints; constraint_error_cost = 1 − compliance
ZeroSurvivalEvent                  → all proposals pruned, autonomic retry fires
InterfaceSaturationWarningEvent    → active sub-tasks approaching N_max^interface
ConsensusRequiredEvent             → max(c_i) > 0.85; merge strategy escalates from ScoreOrdered to ConsensusMedian/OutlierResistant (fractional BFT threshold on fingerprints — not PBFT)
SemilatticeCompiledEvent           → merge ready, MergeStrategy recorded
MergeResolvedEvent                 → human O(1) decision, task closed; j_eff (Ensemble Efficiency Index) = Q_realized/Q_ceiling emitted; oracle_gate_passed: Option<bool>
OracleGateResultEvent              → Phase 4.5 oracle gate: gate_passed, confidence, checked/passed counts
PendingClarificationEvent          → engine suspended awaiting operator answer; POST /{tenant_id}/tasks/{id}/clarify to resume
OproTriggeredEvent                 → OPRO cycle started: j_eff_ema fell below threshold for adapter
PromptVariantPromotedEvent         → Thompson bandit promoted best variant as new prompt default
TaskFailedEvent                    → retries exhausted, full diagnostic payload
TaoIterationEvent                  → TAO loop turn result: tool_calls[] (tool, input_json, output, iteration) + total_token_cost
VerificationScoredEvent            → LLM-as-judge score per proposal (Phase 3.5); panel diversity_kind + quorum recorded
ConstraintAmbiguityEvent           → ≥ambiguity_threshold proposals produced uncertain votes on the same constraint; corpus quality signal
TaskAttributionEvent               → q_confidence decomposition: baseline × verification_filter × tao_uplift × topology_correction + synthesis_gain
PendingApprovalEvent               → HITL gate fired; proposed_output held for review; risk_level (Medium|High); timeout_at_ms
ApprovalResolvedEvent              → operator submitted POST /{tenant_id}/tasks/{id}/approve; approved bool + operator_id recorded
```

**Compound task events** (subject `h2ai.tasks.{task_id}`):
```
SubtaskPlanCreatedEvent            → CompoundTaskEngine decomposed task into SubtaskPlan (N subtasks)
SubtaskPlanReviewedEvent           → PlanReviewer approved or rejected SubtaskPlan (reason recorded)
SubtaskStartedEvent                → SchedulingEngine dispatched a subtask in topo-sort wave N
SubtaskCompletedEvent              → Subtask completed; output injected as context for dependents
```

**Edge agent telemetry events** (subject `h2ai.telemetry.*`):
```
AgentTelemetryEvent::LlmPromptSent        → tokens dispatched to edge agent LLM
AgentTelemetryEvent::LlmResponseReceived  → completion tokens received from edge agent
AgentTelemetryEvent::ShellCommandExecuted → command + args (structured, no raw shell string) + exit code
AgentTelemetryEvent::SystemError          → edge agent panic or unrecoverable error
```

---

## Repository Layout

| Crate | Responsibility |
|---|---|
| `h2ai-types` | Pure types — all events, manifests, USL physics structs, `IComputeAdapter` trait, `AdapterFamily`, `JudgePersona`, `PanelDiversityKind`; zero I/O deps |
| `h2ai-config` | `H2AIConfig` — physics thresholds, role defaults, all tunable parameters |
| `h2ai-constraints` | `ConstraintDoc` / `ConstraintPredicate` type system; Hard/Soft/Advisory severity; `compliance = hard_gate × soft_score` |
| `h2ai-knowledge` | Hierarchical BM25+/PPR knowledge provider — `KnowledgeSource` + `KnowledgeProvider` traits; `Bm25WikiProvider` (RAPTOR dual-mode: TreeTraversal + CollapsedTree); `ConstraintGraph` (Personalized PageRank multi-hop expansion); `YamlDirSource`; `PassthroughProvider` (delegates to `ConstraintResolver`); `ScoringConfig` (8 tunable parameters, all serde-defaulted); `KnowledgeProviderFactory` |
| `h2ai-nats` | NATS subject constants + scoped NKey provisioning per `task_id` |
| `h2ai-state` | CRDT semilattice, NATS JetStream I/O, checkpoint delta encoding (JSON Patch RFC 6902), per-tenant KV buckets |
| `h2ai-context` | Dark Knowledge Compiler — assembles immutable `system_context` from constraint corpus + task manifest |
| `h2ai-autonomic` | Calibration harness, USL solver (α/β/CG → N_max), eigenvalue computation (spawn_blocking) |
| `h2ai-orchestrator` | MAPE-K engine (`engine.rs` + `ExecutionPipeline` + `MapeKController`), all 16 phase modules, compound + scheduling engines, `JudgePanel` cross-family verification aggregation |
| `h2ai-provisioner` | `AgentProvider` — container pool management, capability filter, cost-tier ceiling, least-loaded scheduling |
| `h2ai-planner` | `PlanningEngine::decompose` + `PlanReviewer::evaluate` — LLM-driven task decomposition with structural cycle/empty checks |
| `h2ai-memory` | `MemoryProvider` — context history assembly; all context lives in the control plane, edge agents are stateless |
| `h2ai-telemetry` | `AuditProvider` — immutable audit log; secret redaction middleware; `ShellCommandExecuted.args` redacted per-element |
| `h2ai-adapters` | `IComputeAdapter` implementations: Anthropic, OpenAI, Ollama, CloudGeneric, Mock + `AdapterFactory` |
| `h2ai-tools` | `ToolExecutor` framework: `ShellExecutor` (JSON contract, no shell interpreter), `WebSearchExecutor`, `McpExecutor`, `WasmExecutor`; `ToolRegistry::for_wave(WaveMode)` |
| `h2ai-api` | axum REST gateway, SSE event stream, Merge Authority web UI, OPRO cycle, Thompson-sampling bandit |
| `h2ai-agent` | Edge agent binary — NATS task dispatch loop, `TaoAgent` TAO loop, `config_validation` fail-fast at startup |

**Dependency rule:** `h2ai-types`, `h2ai-config`, `h2ai-constraints`, `h2ai-knowledge` are pure — zero I/O (knowledge reads YAML at startup via std::fs, no async-nats dependency). Six crates use `async-nats` on dedicated subject namespaces: `h2ai-nats`, `h2ai-state`, `h2ai-memory`, `h2ai-telemetry`, `h2ai-provisioner`, `h2ai-agent`. `h2ai-api` is the only HTTP-facing crate; nothing imports it. Eigenvalue-intensive calibration paths use `tokio::task::spawn_blocking` so CPU-bound work never starves NATS consumers.

---

## Technology Stack

| Layer | Choice | Why |
|---|---|---|
| Language | Rust + Tokio | Compiler-verified CRDT state, zero-cost FFI to llama.cpp, no GC jitter in β₀ |
| Event log | NATS JetStream | Single static binary (MB of RAM), Tokio-native `async-nats`, clusters natively |
| State model | Event-sourced CRDT semilattice | α→0 during generation (no locks), crash recovery = replay. CRDTs operate on constraint-satisfaction fingerprints (metadata). LLM text is reconciled by the synthesis LLM — never by a CRDT. |
| Local compute | llama.cpp FFI | Zero-cost, 128GB RAM dedicated to weights |
| Edge agents | `AgentDescriptor { model, tools }` | Any LLM-based container described by model name + capability flags; stateless `f(ctx, τ) → result`, scoped NKeys per task |
| HTTP | axum | Tokio-native, same async runtime as orchestrator |
| Type bindings | `typeshare` + hand-authored | Rust types → TypeScript/Swift/Kotlin via typeshare CLI; Go structs hand-maintained (`typeshare` dropped Go in v1.13+) |
| Tracing | `tracing` + OpenTelemetry | task_id as root span, DAG execution visible in Jaeger/Grafana Tempo |
| Metrics | Prometheus `/metrics` | 5 series: n_eff_prior, n_eff_actual, epistemic_yield_ratio, mapek_interventions{mode_collapse}, mapek_interventions{constrained_exploration} |

---

## Deployment

The system is **C-first**: the distributed cluster is the architectural foundation, not a future upgrade. Local Plan is Cloud Plan running on one machine.

| Plan | Target | Stack |
|---|---|---|
| **Local — Local dev** | Single workstation (128GB RAM) | Static binary + nats-server, no container runtime required |
| **Server — Team node** | Dedicated server | 3-node NATS cluster + 2× h2ai + nginx + Prometheus + Grafana + Jaeger |
| **Cloud — Kubernetes** | Multi-region cluster | Helm chart, NATS StatefulSet, h2ai Deployment + HPA, ServiceMonitor |

---

## Documentation

| Document | Contents |
|---|---|
| [Architecture](docs/architecture/architecture.md) | System overview, positioning, tech stack, crate boundaries, deployment plans, event vocabulary |
| [Math](docs/architecture/math.md) | USL/CJT foundations, 10 definitions + 5 propositions, β_eff formula, calibration table |
| [Reference](docs/architecture/reference.md) | REST API, SSE event stream, configuration fields, Prometheus metrics, adapter guide, constraint corpus |
| [Operations](docs/architecture/operations.md) | Getting started, deployment plans, key metrics, alert rules, scaling, troubleshooting |
| [Research State](docs/architecture/research-state.md) | Project thesis, implemented gaps, open research questions, empirical benchmarking strategy |

---

## License

BSD 3-Clause License. See [LICENSE](LICENSE).

*In accordance with Clause 3, the name of the copyright holder may not be used to endorse or promote products derived from this software without specific prior written permission.*

---

## Citation

```bibtex
@article{polyulya2026coordination,
  title={The Coordination Constant — One Equation Governs CPU Caches, Human Teams, and AI Agent Systems},
  author={Polyulya, Yuriy},
  year={2026},
  url={https://e-mindset.space/blog/coordination-constant-usl-human-ai-teams/}
}
```
