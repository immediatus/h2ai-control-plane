# H2AI Control Plane

[![CI](https://github.com/immediatus/h2ai-control-plane/actions/workflows/ci.yml/badge.svg)](https://github.com/immediatus/h2ai-control-plane/actions/workflows/ci.yml)
[![Theoretical Foundation](https://img.shields.io/badge/Framework-The_Coordination_Constant-blue)](https://e-mindset.space/blog/coordination-constant-usl-human-ai-teams/)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-orange)](LICENSE)
[![Language](https://img.shields.io/badge/Language-Rust-orange)](https://www.rust-lang.org/)

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
| State lives in the model | LLM context window (lossy) | Orchestrator owns state; models are stateless `f(ctx, τ) → text`. CRDTs track **constraint-satisfaction fingerprints** (metadata), never LLM text. Text is reconciled by the synthesis LLM. |
| Safety is probabilistic | "Don't do X" in the prompt | Topological interlocks — invalid output cannot reach the human by graph construction |
| More agents = worse results | Keep adding until it breaks | MAPE-K loop computes N_max, shifts topology before retrograde |
| Tacit knowledge is invisible | Agents guess team constraints | Dark Knowledge Compiler — typed `ConstraintDoc` predicates (Hard/Soft/Advisory) become hard Auditor gates; `constraint_error_cost = 1 − compliance` |
| Human babysits every step | Constant correction loop | Merge Authority — human resolves a structured CRDT diff once, at the end |
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
cp -r ../../docs/examples/ads-platform/constraints/* ../../constraints/

# Calibrate the adapter pool
curl -X POST http://localhost:8080/calibrate

# Task 1 — pure reasoning (no tools)
# 3 pure LLM explorers reason in parallel about the architecture decision.
# c_i ≈ 0.1 (text output, discard at zero cost) → CRDT merge.
curl -X POST http://localhost:8080/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Design a budget enforcement mechanism that prevents double-billing during server restarts",
    "pareto_weights": {"diversity": 0.5, "containment": 0.4, "throughput": 0.1},
    "explorers": {"count": 3, "tau_min": 0.2, "tau_max": 0.85}
  }'

# Task 2 — code generation (containment-weighted, tight τ band)
curl -X POST http://localhost:8080/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Write and test a Redis Lua script for atomic budget check-and-decrement with 30s TTL idempotency",
    "pareto_weights": {"diversity": 0.3, "containment": 0.6, "throughput": 0.1},
    "explorers": {"count": 3, "tau_min": 0.2, "tau_max": 0.5}
  }'

# Stream events in real time
curl -sN http://localhost:8080/tasks/{task_id}/events

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

## How It Works

### 1. Calibration — measure the physics before spawning anything

The calibration harness runs representative tasks through the adapter pool and measures the two parameters that bound ensemble size:

- `α` — the **serial bottleneck fraction**: time spent in planning, context compilation, and final synthesis — phases that serialize regardless of how many agents run in parallel. Measured as the fraction of total wall time that scales with N=1 behavior.
- `β₀` — the **pairwise reconciliation cost**: how expensive it is to integrate each pair of agents' outputs into a coherent answer. Measured from merge phase timing and output divergence (CG_mean). Scales as N(N−1) in the USL model.
- `CG(i,j)` — **Common Ground** between every Explorer pair: mean pairwise Hamming distance on the binary constraint-satisfaction fingerprints each agent produces (1 bit per constraint: pass/fail). High CG means agents satisfied the same constraints; low CG means they diverged and conflict reconciliation is costly. This is a metadata measurement — the control plane never compares raw LLM text to compute CG.

From these it derives `N_max = sqrt((1−α) / β_eff)` — the agent count at which Condorcet quality gain and reconciliation cost intersect. Beyond `N_max`, every additional agent makes results worse. No task proceeds without this data.

### 2. Bootstrap — compile Dark Knowledge into explicit constraints

You submit a task manifest. The **Dark Knowledge Compiler** assembles an immutable `system_context` from your constraint corpus and the task manifest. Every agent — Explorer and Auditor alike — receives exactly this context and nothing else.

### 3. Provisioning — topology selected by physics, not guesswork

The autonomic loop reads `{α, β_eff, ParetoWeights}` and selects one of three topologies:
- **Ensemble + CRDT** — when `N ≤ N_max` and diversity weight dominates. No coordinator. All Explorers are peers. `O(N²)` edges, but structurally fine for small N. Pareto: T=84%, E=84%, D=90%.
- **Hierarchical Tree** — when `N > N_max` or containment weight dominates. One Swarm Coordinator + k sub-groups. Branching factor `k_opt = floor(N_max^flat)`. Coordination cost drops from `O(N²)` to `O(N)`. Pareto: T=96%, E=96%, D=60%.
- **Team-Swarm Hybrid** — when the manifest provides `explorers.roles[]`. Role-differentiated Explorers (Coordinator, Executor, Evaluator, Synthesizer) with declared review gates between specified pairs. The Evaluator forms a pre-Auditor gate that blocks Executor output. Pareto: T=84%, E=91%, D=95%.

Before spawning a single inference token, the **Multiplication Condition Gate** enforces all three conditions from Proposition 3: competence > 0.5, error decorrelation ρ < 0.9, Common Ground mean ≥ θ_coord. Fail any one → re-enter provisioning with adjusted parameters.

### 4. Generation — parallel, isolated, bounded

N Explorers run in a `tokio::task::JoinSet` wrapped in `tokio::time::timeout`. Each Explorer calls `IComputeAdapter::execute()` with its assigned `τ` value and terminates. No Explorer reads another Explorer's output. Coordination cost during generation is structurally zero. Every Explorer gets a guaranteed terminal state — `ProposalEvent` on success, `ProposalFailedEvent` on crash/OOM/timeout. The stream always closes with `GenerationPhaseCompletedEvent`.

For **Team-Swarm Hybrid** topologies, an additional Review Gate phase runs after generation: each Executor proposal is routed to its designated Evaluator (τ≈0.1, c_i≈0.9) before reaching the Auditor. Approved proposals proceed to Phase 5; blocked proposals are tombstoned at the gate with reason recorded (`ReviewGateBlockedEvent`) and visible in the Merge Authority UI.

### How Explorers Execute — The Edge Agent Dispatch Pipeline

The control plane never runs inference directly. Each Explorer is an **ephemeral, stateless edge agent** described by an `AgentDescriptor` and dispatched as a container over NATS. An agent is not a named product — it is any LLM-based container identified by the model it runs and the capabilities it has been granted:

```rust
pub struct AgentDescriptor {
    pub model: String,         // "llama3-70b", "gpt-4o", "claude-3-opus", ...
    pub tools: Vec<AgentTool>, // [] | [WebSearch] | [Shell, CodeExecution, FileSystem]
}

pub enum AgentTool {
    Shell,
    WebSearch,
    CodeExecution,
    FileSystem,
}
```

**`tools` are capability flags, not features.** They directly affect three USL quantities that the system measures and controls:

| Tool set | Effect on α | Effect on β₀ | Default c_i | Typical role |
|---|---|---|---|---|
| `[]` (pure LLM) | near 0 | 0 (text only) | 0.1–0.3 | Coordinator / Synthesizer |
| `[WebSearch]` | +0.01–0.02 | +0.005 (retrieval nondeterminism) | 0.2–0.4 | Evaluator |
| `[FileSystem]` | +0.02–0.05 | +0.01 (shared state writes) | 0.4–0.6 | Executor |
| `[CodeExecution]` | +0.03–0.08 | +0.015 (env side effects) | 0.5–0.7 | Executor |
| `[Shell]` | +0.05–0.15 | +0.02 (arbitrary side effects) | 0.6–0.9 | Executor |

A pure LLM agent is `f(context, τ) → text` — deterministic given its inputs, zero side effects, errors cost nothing to discard. A tool-using agent is `f(context, τ, external_state_t) → text + side_effects` — its output depends on the world at execution time, and wrong outputs may leave irreversible state.

This distinction flows directly into topology selection:
- High c_i from tool-using agents drives `max(c_i) > 0.85`, switching `MergeStrategy` from `ScoreOrdered` to `ConsensusMedian` (Condorcet voting) or `OutlierResistant` (for `max(c_i) > 0.95`).
- Tool-induced α increase lowers `N_max`, reducing the explorer count.
- WebSearch nondeterminism raises CG variance, lowering `CG_mean`, raising `β_eff`.
- High-c_i Executor agents trigger Review Gates in TeamSwarmHybrid topology — the wrong output cannot reach the Auditor by graph construction.

The full dispatch-and-await loop per Explorer:

1. **Context assembly** — `MemoryProvider::get_recent_history` retrieves prior session history; assembled into the `context` field of `TaskPayload`.
2. **Payload construction** — `TaskPayload { task_id, agent: AgentDescriptor, instructions, context, τ, max_tokens, wave_mode: WaveMode }` is built. The full descriptor and `wave_mode` travel with the payload so the edge agent's `ToolRegistry` is scoped to the correct allowlist for this wave.
3. **Capacity check** — `AgentProvider::ensure_agent_capacity(descriptor, task_load)` verifies the container pool. The provider selects the container image from `descriptor.model` and configures volume mounts and security contexts from `descriptor.tools` — no hardcoded image names in the orchestrator.
4. **Dispatch** — `TaskPayload` is published to `h2ai.tasks.ephemeral.{task_id}` on the durable JetStream `H2AI_TASKS` stream. A scoped **NKey** for this `task_id` is injected into the container — the edge agent has no NATS credentials before this moment and none after. The NKey's `allowed_publish` set is sized to match the tool set.
5. **TAO local tool loop** — inside the edge agent, `TaoAgent::run` iterates up to `agent_max_tool_iterations` turns. Each turn: call the LLM with accumulated context and all tool schemas injected into the system prompt; parse `{"tool": "...", "input": {...}}` from the response; dispatch to `ToolRegistry::execute(tool, input_json)`; append a `ToolCallRecord { tool, input_json, output, iteration }` to the audit trail. The loop terminates when the LLM produces a final answer (no tool field) or the iteration budget is exhausted.
6. **Live telemetry** — the control plane subscribes to `h2ai.telemetry.{task_id}`. Every `AgentTelemetryEvent` (LLM calls, shell commands, errors) is routed through `RedactionMiddleware` (secrets scrubbed before logging) into `AuditProvider::record_event` in real time.
7. **Result wait** — the control plane subscribes to `h2ai.results.{task_id}`. The edge agent publishes `TaskResult { answer, tool_calls: Vec<ToolCallRecord>, total_token_cost, .. }` when done and terminates. `tool_calls` carries the complete TAO iteration history for audit.
8. **Memory commit** — on `TaskResult`, `MemoryProvider::commit_new_memories` persists the output for future context assembly. `AuditProvider::flush` drains the audit buffer.
9. **NKey expiry** — the scoped NKey expires. The container loses all NATS permissions and is reaped.

**Security invariant:** An edge agent can only publish to `h2ai.telemetry.{its agent_id}`, `audit.events.{its agent_id}`, and `h2ai.results.{its task_id}`. It cannot read other agents' payloads, write to the orchestration event bus, or retain credentials after its task closes. This is enforced at the NATS server level — not by application code.

### 5. Auditor Gate — reactive, never idle

The Auditor spins up on `TopologyProvisionedEvent` — before generation starts. It validates proposals as they arrive against the compiled `system_context`. Every constraint in your corpus is a potential rejection. Rejected proposals become `BranchPrunedEvent` tombstones: permanently preserved in the log with rejection reason and constraint cost (`c_i`), visible in the Merge Authority UI.

If all proposals are pruned → `ZeroSurvivalEvent` → the MAPE-K loop adjusts `{N, τ}` and retries. Bounded by `max_retries`. Exhaustion → `TaskFailedEvent` with full diagnostic payload.

### 6. Merge — two layers, O(1) human decision

The merge phase has two deliberately separate layers:

**Layer 1 — Metadata consensus (CRDT semilattice, fast, deterministic):** The control plane aggregates the binary constraint-satisfaction fingerprints from all surviving proposals into a semilattice. This computes, in lock-free Rust, which proposals agree on which constraints. The BFT threshold (e.g. 0.67) is applied to this fingerprint agreement rate — not to raw text similarity. This layer never touches LLM output text. This is where `ConsensusRequiredEvent` fires and `MergeStrategy` is selected.

> **Note on "BFT" in this system:** The `bft_threshold` is a fractional agreement gate (e.g. 0.67) on constraint fingerprints — not PBFT (Practical Byzantine Fault Tolerance). PBFT is designed for adversarial nodes with cryptographic guarantees and costs O(N²) network rounds. Here the "Byzantine nodes" are hallucinating LLMs — stochastic divergence, not malicious actors. A fractional threshold + Krum outlier rejection is the correct tool for epistemic fault tolerance. Full PBFT would be architectural overkill.

**Layer 2 — Semantic reconciliation (synthesis LLM, slow, creative):** Once the metadata layer has identified which proposals cleared the BFT threshold, the synthesis LLM receives only those validated proposals. Synthesis runs in two passes: (1) a **critique pass** at low τ reads all verified proposals and produces a structured gap analysis; (2) a **synthesis pass** reads the proposals plus the critique and produces the final coherent output. The synthesis LLM does what CRDTs cannot: semantic reconciliation of natural-language proposals.

**Diversity gate:** Before synthesis runs, the system checks that the verified proposals are not collectively hallucinated — if mean pairwise Hamming distance across their fingerprints falls below `diversity_threshold`, all proposals are too similar and a MAPE-K retry fires. Synthesis on a mono-culture ensemble produces false confidence.

The **Merge Authority UI** presents:

- **Valid proposals panel** — diff view grouped by target component, τ and adapter shown per proposal
- **Tombstone panel** — every rejected proposal with Explorer ID, attempted output, rejection reason, and `c_i` weight of the violated constraint. Failures are epistemic data.
- **Autonomic shift timeline** — every MAPE-K intervention rendered as a timeline node
- **Physics panel** — live `θ_coord`, `β_eff`, `N_max`, current `MergeStrategy`

The human makes one decision. `MergeResolvedEvent` closes the task.

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
MergeResolvedEvent                 → human O(1) decision, task closed
TaskFailedEvent                    → retries exhausted, full diagnostic payload
TaoIterationEvent                  → TAO loop turn result: tool_calls[] (tool, input_json, output, iteration) + total_token_cost
VerificationScoredEvent            → LLM-as-judge score per proposal (Phase 3.5)
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

```
h2ai-control-plane/
├── Dockerfile                      # multi-stage: builder (rust+clang) → runtime (debian-slim)
├── typeshare.toml                  # typeshare CLI config (Rust→TypeScript/Swift/Kotlin; Go dropped in v1.13+)
├── bindings/
│   └── go/                         # hand-authored Go types (typeshare CLI dropped Go in v1.13+; maintained manually)
├── crates/
│   ├── h2ai-types/                 # Pure types boundary — zero I/O deps
│   │                               # All 23 core events + AgentTelemetryEvent, IComputeAdapter,
│   │                               # USL physics types, CoherencyCoefficients, MergeStrategy,
│   │                               # AgentState, TaskPayload, TaskResult (typeshare-annotated)
│   │                               # SubtaskId, SubtaskPlan, SubtaskResult, PlanStatus, Subtask
│   │                               # ConstraintViolation (per-constraint failure record in BranchPrunedEvent)
│   ├── h2ai-nats/                  # NATS subject constants + scoped NKey provisioning per task_id
│   ├── h2ai-config/                # H2AIConfig — physics thresholds and role defaults
│   ├── h2ai-constraints/           # ConstraintDoc/ConstraintPredicate type system + sync evaluator
│   │                               # VocabularyPresence/NegativeKeyword/RegexMatch/NumericThreshold/
│   │                               # LlmJudge/Composite predicates; Hard/Soft/Advisory severity;
│   │                               # compliance formula: hard_gate × soft_score
│   ├── h2ai-provisioner/           # AgentProvider + SchedulingPolicy + NatsAgentProvider
│   │                               # Capability filter + cost-tier ceiling + least-loaded scheduling
│   ├── h2ai-memory/                # MemoryProvider trait + InMemoryCache + NatsKvStore
│   │                               # Stateless edge agents: all context lives in the control plane
│   ├── h2ai-planner/               # PlanningEngine::decompose + PlanReviewer::evaluate
│   │                               # LLM-driven task decomposition into SubtaskPlan; structural
│   │                               # cycle/empty checks before semantic LLM review
│   ├── h2ai-telemetry/             # AuditProvider trait + DirectLogProvider + BrokerPublisherProvider
│   │                               # Immutable audit log with secret redaction middleware
│   │                               # ShellCommandExecuted.args redacted per-element
│   ├── h2ai-orchestrator/          # DAG builder + Pareto topology router + NatsDispatchAdapter
│   │                               # CompoundTaskEngine (decompose → review → schedule pipeline)
│   │                               # SchedulingEngine (Kahn topo-sort wave execution)
│   ├── h2ai-autonomic/             # MAPE-K loop + calibration harness + N_max calculator
│   ├── h2ai-state/                 # CRDT semilattice + NATS JetStream I/O + task dispatch wire protocol
│   ├── h2ai-context/               # Dark Knowledge Compiler + constraint corpus loader
│   │                               # corpus_keywords from ConstraintDoc::vocabulary()
│   ├── h2ai-adapters/              # IComputeAdapter: Anthropic, OpenAI, Ollama, CloudGeneric, Mock + AdapterFactory
│   ├── h2ai-tools/                 # Tool executor framework: ToolExecutor trait, ToolRegistry, ToolError
│   │                               # ShellExecutor (JSON contract, no shell interpreter, PGID kill on timeout)
│   │                               # WebSearchExecutor (GoogleSearchBackend, max 10 results)
│   │                               # McpExecutor (StdioMcpBackend, read_file/list_directory only)
│   │                               # WasmExecutor (RealWasmBackend via wasmtime, fuel-bounded JS sandbox)
│   │                               # ToolRegistry::for_wave(cfg, WaveMode) — live backends
│   │                               # ToolRegistry::for_wave_with_mocks(cfg, WaveMode) — test helper
│   ├── h2ai-api/                   # axum REST gateway + Merge Authority web UI
│   └── h2ai-agent/                 # Edge agent binary — heartbeat + NATS task dispatch loop
│                                   # TaoAgent: tool-call loop up to agent_max_tool_iterations turns
│                                   # config_validation::validate_tool_configs — fail-fast at startup
│                                   # builds ToolRegistry::for_wave(cfg, wave_mode) per task
├── nats/
│   ├── dev.conf                    # single-node JetStream config (Local Plan)
│   └── cluster.conf                # 3-node cluster config (Server/Cloud Plan)
├── deploy/
│   ├── local/docker-compose.yml          # h2ai + NATS, single workstation
│   ├── server/docker-compose.yml         # 3-node NATS + 2× h2ai + nginx + observability
│   ├── cloud/                            # raw Kubernetes manifests
│   └── helm/h2ai-control-plane/          # Helm chart for enterprise distribution
├── .devcontainer/                  # devcontainer: Rust toolchain + NATS sidecar
├── .github/workflows/
│   ├── ci.yml                      # fmt → clippy -D warnings → nextest → docker → helm lint
│   └── release.yml                 # image → ghcr.io, Helm chart → GitHub Pages, binary release
└── docs/
    ├── architecture/               # 5-file consolidated documentation
    │   ├── architecture.md         # System overview, crate boundaries, event vocabulary
    │   ├── math.md                 # USL/CJT foundations, formulas, calibration table
    │   ├── reference.md            # API, configuration, metrics, adapters, constraint corpus
    │   ├── operations.md           # Getting started, deployment, alerts, troubleshooting
    │   └── research-state.md       # Thesis, implemented gaps, open questions, benchmarks
    └── examples/
        └── ads-platform/           # Reference constraint corpus + integration test task manifests
            ├── constraints/        # 7 constraint docs derived from "Architecting Real-Time Ads Platform"
            └── tasks/              # 3 task manifests with expected Auditor outcomes
```

**Dependency rule:** `h2ai-types`, `h2ai-config`, and `h2ai-constraints` have zero external I/O dependencies — predicate evaluation is pure computation. `h2ai-planner` depends only on `h2ai-types` (LLM calls go through `IComputeAdapter` from `h2ai-types`; no NATS or HTTP deps). Six crates import `async-nats` directly, each on a dedicated subject namespace: `h2ai-nats` (subject constants + NKey provisioning), `h2ai-state` (task event log), `h2ai-memory` (context history), `h2ai-telemetry` (audit log), `h2ai-provisioner` (agent heartbeats + soft-kill), `h2ai-agent` (binary — heartbeat publisher + task subscriber). `h2ai-api` is the only crate that talks to HTTP. Nothing imports `h2ai-api`.

**Compute isolation:** Cloud HTTP adapters (Anthropic, OpenAI, Ollama) run async on the main worker pool. Future llama.cpp FFI inference uses `spawn_blocking` with `max_blocking_threads` explicitly set so CPU-bound inference never starves NATS consumers or HTTP handlers.

---

## Scripts

| Script | Purpose |
|---|---|
| `scripts/simulate.py` | Visualization: USL curves, β_eff vs CG coupling, N_max vs CG, CJT quality curves, N_eff eigenvalue vs scalar ρ, Talagrand rank histogram shapes |
| `scripts/baseline_eval.py` | **Production tool** — measures real p and ρ from live adapters against `eval_questions.jsonl`; output overrides `baseline_accuracy_proxy` in config. Run before high-stakes deployments. |
| `scripts/benchmark/` | Benchmark harness: GSM8K, HumanEval, TruthfulQA runners + B0/B1/B2/B3/H2AI baseline comparison. Runs not yet executed. |

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

## Constraint Corpus and Integration Examples

The **Dark Knowledge Compiler** reads your team's `ConstraintDoc` files. Each document becomes a typed `ConstraintDoc` with a `ConstraintPredicate` (vocabulary presence, negative keywords, regex, numeric threshold, LLM judge, or composites) and a severity (`Hard`, `Soft`, or `Advisory`). Hard constraints gate the merge; Soft constraints contribute a weighted compliance score; `constraint_error_cost = 1.0 − compliance` feeds back into the BFT merge strategy selector.

`docs/examples/ads-platform/` is a complete reference corpus derived from the blog series **[Architecting Real-Time Ads Platform](https://e-mindset.space/series/architecting-ads-platforms/)**:

| Constraint | Decision |
|---|---|
| CONSTRAINT-001 | Stateless request services — no per-user state across requests |
| CONSTRAINT-002 | gRPC internal / REST external — JSON overhead consumes 20% of latency budget at 1M QPS |
| CONSTRAINT-003 | Adaptive RTB timeouts via HdrHistogram — per-DSP P95, capped at 100ms global |
| CONSTRAINT-004 | Budget pacing with idempotency — Redis Lua atomic check-and-set, 30s TTL |
| CONSTRAINT-005 | Dual-ledger audit log — Kafka → ClickHouse append-only, 7-year retention, SOX |
| CONSTRAINT-006 | Java 21 + Generational ZGC — 32GB heap, <2ms P99.9 GC pauses |
| CONSTRAINT-007 | Tiered consistency — budget=strong, profiles=eventual, billing=linearizable |

The task manifests in `docs/examples/ads-platform/tasks/` are the input corpus for the integration test suite. They specify the expected Auditor outcomes — which proposals should be pruned, which constraint they violate — so that `cargo nextest run --test integration` can assert system behavior end-to-end.

---

## Documentation

All documentation is under `docs/architecture/`.

| Document | Contents |
|---|---|
| [Architecture](docs/architecture/architecture.md) | System overview, positioning, tech stack, crate boundaries, deployment plans, event vocabulary |
| [Math](docs/architecture/math.md) | USL/CJT foundations, 10 definitions + 5 propositions, β_eff formula, calibration table |
| [Reference](docs/architecture/reference.md) | REST API, SSE event stream, configuration fields, Prometheus metrics, adapter guide, constraint corpus |
| [Operations](docs/architecture/operations.md) | Getting started, deployment plans, key metrics, alert rules, scaling, troubleshooting |
| [Research State](docs/architecture/research-state.md) | Project thesis, implemented gaps, open research questions, empirical benchmarking strategy |

### Examples

| Document | Contents |
|---|---|
| [Ads Platform](docs/examples/ads-platform/) | 7 constraint docs + 3 integration test tasks derived from "Architecting Real-Time Ads Platform" |

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
