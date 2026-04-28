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
  β = pairwise reconciliation cost — integrating each new agent's output
      with every existing one; modulated by how much agents have diverged
```

The peak of X(N) is `N_max = √((1−α)/β_eff)`. Beyond N_max, adding agents actively degrades output quality because reconciliation cost exceeds the Condorcet quality gain.

**This is not a new observation.** Brook's Law (1975) measured it in human engineering teams — communication channels grow as N(N−1)/2. CPU cache coherency protocols hit the same ceiling at a different scale. LLM agent swarms exhibit the same phenomenon for the same structural reason: pairwise synchronization overhead scales quadratically with group size when agents must reach mutual consistency.

The β parameter is modulated by **Common Ground (CG)** — the semantic overlap between agents' outputs. When agents have high CG (compatible partial solutions, shared vocabulary), reconciliation is cheap. When they have split, each pair requires more work to reconcile. `β_eff = β₀ × (1 − CG_mean)` captures this: split agents cost more to coordinate than aligned ones.

H2AI measures both forces, finds their intersection, and enforces a Common Ground floor (θ_coord) before allowing generation to start — preventing split brain before it begins rather than trying to repair it after.

---

## Why It Exists

| Problem | Standard Approach | H2AI Approach |
|---|---|---|
| Hallucination amplification | Hope the model self-corrects | Auditor node (τ→0) mathematically blocks propagation |
| State lives in the model | LLM context window (lossy) | Sovereign CRDTs — orchestrator owns state, models are stateless `f(ctx, τ) → diff` |
| Safety is probabilistic | "Don't do X" in the prompt | Topological interlocks — invalid output cannot reach the human by graph construction |
| More agents = worse results | Keep adding until it breaks | MAPE-K loop computes N_max, shifts topology before retrograde |
| Tacit knowledge is invisible | Agents guess team constraints | Dark Knowledge Compiler — typed `ConstraintDoc` predicates (Hard/Soft/Advisory) become hard Auditor gates; `constraint_error_cost = 1 − compliance` |
| Human babysits every step | Constant correction loop | Merge Authority — human resolves a structured CRDT diff once, at the end |
| Edge agent secrets leak | Long-lived API keys in containers | Scoped NATS NKeys per task_id — token expires when task closes |
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
mkdir -p ../../adr
cp -r ../../docs/examples/ads-platform/adr/* ../../adr/

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

# Task 2 — code generation with tool-using executors
# Executors (CodeExecution + FileSystem) write and run code.
# c_i ≈ 0.7 → max(c_i) approaches BFT threshold.
# Evaluator (pure LLM, tau=0.1) forms a Review Gate before the Auditor.
curl -X POST http://localhost:8080/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Write and test a Redis Lua script for atomic budget check-and-decrement with 30s TTL idempotency",
    "pareto_weights": {"diversity": 0.3, "containment": 0.6, "throughput": 0.1},
    "explorers": {
      "roles": [
        {"agent_id": "executor_A", "role": "Executor", "tau": 0.4},
        {"agent_id": "executor_B", "role": "Executor", "tau": 0.5},
        {"agent_id": "evaluator",  "role": "Evaluator", "tau": 0.1}
      ],
      "review_gates": [
        {"reviewer": "evaluator", "blocks": "executor_A"},
        {"reviewer": "evaluator", "blocks": "executor_B"}
      ]
    },
    "constraints": ["ADR-004", "ADR-007"]
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

kubectl create configmap constraint-corpus --from-file=./adr/ -n h2ai

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
- `κ_base` — the **pairwise reconciliation cost**: how expensive it is to integrate each pair of agents' outputs into a coherent answer. Measured from merge phase timing and output divergence (CG_mean). Scales as N(N−1) in the USL model.
- `CG(i,j)` — **Common Ground** between every Explorer pair: vocabulary overlap of their outputs × temperature alignment. High CG means agents reached compatible conclusions; low CG means they have split and reconciliation is costly.

From these it derives `N_max = sqrt((1−α) / κ_eff)` — the agent count at which Condorcet quality gain and reconciliation cost intersect. Beyond `N_max`, every additional agent makes results worse. No task proceeds without this data.

### 2. Bootstrap — compile Dark Knowledge into explicit constraints

You submit a task manifest. The **Dark Knowledge Compiler** computes `J_eff = J(K_prompt, K_task_required)` — the Jaccard overlap between what you explicitly provided (manifest + constraint corpus) and what the task actually requires.

If `J_eff` is below threshold → synchronous `400 ContextUnderflowError`. The human must add constraints before proceeding. Nothing touches NATS.

If `J_eff` passes → an immutable `system_context` is compiled from your ADRs + manifest. Every agent — Explorer and Auditor alike — receives exactly this context and nothing else.

### 3. Provisioning — topology selected by physics, not guesswork

The autonomic loop reads `{α, κ_eff, ParetoWeights}` and selects one of three topologies:
- **Ensemble + CRDT** — when `N ≤ N_max` and diversity weight dominates. No coordinator. All Explorers are peers. `O(N²)` edges, but structurally fine for small N. Pareto: T=84%, E=84%, D=90%.
- **Hierarchical Tree** — when `N > N_max` or containment weight dominates. One Swarm Coordinator + k sub-groups. Branching factor `k_opt = floor(N_max^flat)`. Coordination cost drops from `O(N²)` to `O(N)`. Pareto: T=96%, E=96%, D=60%.
- **Team-Swarm Hybrid** — when the manifest provides `explorers.roles[]`. Role-differentiated Explorers (Coordinator, Executor, Evaluator, Synthesizer) with declared review gates between specified pairs. The Evaluator forms a pre-Auditor gate that blocks Executor output. Pareto: T=84%, E=91%, D=95%.

Before spawning a single inference token, the **Multiplication Condition Gate** enforces all three conditions from Proposition 3: competence > 0.5, error decorrelation ρ < 0.9, Common Ground mean ≥ θ_coord. Fail any one → re-enter provisioning with adjusted parameters.

### 4. Generation — parallel, isolated, bounded

N Explorers run in a `tokio::task::JoinSet` wrapped in `tokio::time::timeout`. Each Explorer calls `IComputeAdapter::execute()` with its assigned `τ` value and terminates. No Explorer reads another Explorer's output. Coordination cost during generation is structurally zero. Every Explorer gets a guaranteed terminal state — `ProposalEvent` on success, `ProposalFailedEvent` on crash/OOM/timeout. The stream always closes with `GenerationPhaseCompletedEvent`.

For **Team-Swarm Hybrid** topologies, an additional Review Gate phase runs after generation: each Executor proposal is routed to its designated Evaluator (τ≈0.1, c_i≈0.9) before reaching the ADR Auditor. Approved proposals proceed to Phase 5; blocked proposals are tombstoned at the gate with reason recorded (`ReviewGateBlockedEvent`) and visible in the Merge Authority UI.

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

| Tool set | Effect on α | Effect on κ_base | Default c_i | Typical role |
|---|---|---|---|---|
| `[]` (pure LLM) | near 0 | 0 (text only) | 0.1–0.3 | Coordinator / Synthesizer |
| `[WebSearch]` | +0.01–0.02 | +0.005 (retrieval nondeterminism) | 0.2–0.4 | Evaluator |
| `[FileSystem]` | +0.02–0.05 | +0.01 (shared state writes) | 0.4–0.6 | Executor |
| `[CodeExecution]` | +0.03–0.08 | +0.015 (env side effects) | 0.5–0.7 | Executor |
| `[Shell]` | +0.05–0.15 | +0.02 (arbitrary side effects) | 0.6–0.9 | Executor |

A pure LLM agent is `f(context, τ) → text` — deterministic given its inputs, zero side effects, errors cost nothing to discard. A tool-using agent is `f(context, τ, external_state_t) → text + side_effects` — its output depends on the world at execution time, and wrong outputs may leave irreversible state.

This distinction flows directly into topology selection:
- High c_i from tool-using agents drives `max(c_i) > 0.85`, switching `MergeStrategy` from `ScoreOrdered` to `ConsensusMedian` (Condorcet voting) or `Krum` (provably BFT, for `max(c_i) > 0.95` with `krum_fault_tolerance > 0`).
- Tool-induced α increase lowers `N_max`, reducing the explorer count.
- WebSearch nondeterminism raises CG variance, lowering `CG_mean`, raising `κ_eff`.
- High-c_i Executor agents trigger Review Gates in TeamSwarmHybrid topology — the wrong output cannot reach the Auditor by graph construction.

The full dispatch-and-await loop per Explorer:

1. **Context assembly** — `MemoryProvider::get_recent_history` retrieves prior session history; assembled into the `context` field of `TaskPayload`.
2. **Payload construction** — `TaskPayload { task_id, agent: AgentDescriptor, instructions, context, τ, max_tokens }` is built. The full descriptor travels with the payload so the edge agent knows what capabilities it has been granted.
3. **Capacity check** — `AgentProvider::ensure_agent_capacity(descriptor, task_load)` verifies the container pool. The provider selects the container image from `descriptor.model` and configures volume mounts and security contexts from `descriptor.tools` — no hardcoded image names in the orchestrator.
4. **Dispatch** — `TaskPayload` is published to `h2ai.tasks.ephemeral.{task_id}` on the durable JetStream `H2AI_TASKS` stream. A scoped **NKey** for this `task_id` is injected into the container — the edge agent has no NATS credentials before this moment and none after. The NKey's `allowed_publish` set is sized to match the tool set.
5. **Live telemetry** — the control plane subscribes to `h2ai.telemetry.{task_id}`. Every `AgentTelemetryEvent` (LLM calls, shell commands, errors) is routed through `RedactionMiddleware` (secrets scrubbed before logging) into `AuditProvider::record_event` in real time.
6. **Result wait** — the control plane subscribes to `h2ai.results.{task_id}`. The edge agent publishes `TaskResult` when done and terminates.
7. **Memory commit** — on `TaskResult`, `MemoryProvider::commit_new_memories` persists the output for future context assembly. `AuditProvider::flush` drains the audit buffer.
8. **NKey expiry** — the scoped NKey expires. The container loses all NATS permissions and is reaped.

**Security invariant:** An edge agent can only publish to `h2ai.telemetry.{its agent_id}`, `audit.events.{its agent_id}`, and `h2ai.results.{its task_id}`. It cannot read other agents' payloads, write to the orchestration event bus, or retain credentials after its task closes. This is enforced at the NATS server level — not by application code.

### 5. Auditor Gate — reactive, never idle

The Auditor spins up on `TopologyProvisionedEvent` — before generation starts. It validates proposals as they arrive against the compiled `system_context`. Every ADR constraint in your corpus is a potential rejection. Rejected proposals become `BranchPrunedEvent` tombstones: permanently preserved in the log with rejection reason and constraint cost (`c_i`), visible in the Merge Authority UI.

If all proposals are pruned → `ZeroSurvivalEvent` → the MAPE-K loop adjusts `{N, τ}` and retries. Bounded by `max_retries`. Exhaustion → `TaskFailedEvent` with full diagnostic payload.

### 6. Merge — O(1) human decision

Surviving proposals are merged using the provisioned strategy (`ScoreOrdered` / `ConsensusMedian` / `Krum`). The **Merge Authority UI** presents:

- **Valid proposals panel** — diff view grouped by target component, τ and adapter shown per proposal
- **Tombstone panel** — every rejected proposal with Explorer ID, attempted output, rejection reason, and `c_i` weight of the violated constraint. Failures are epistemic data.
- **Autonomic shift timeline** — every MAPE-K intervention rendered as a timeline node
- **Physics panel** — live `θ_coord`, `J_eff`, `κ_eff`, `N_max`, current `MergeStrategy`

The human makes one decision. `MergeResolvedEvent` closes the task.

---

## The Scalability Ceiling

```
X(N) = N / (1 + α(N−1) + κ_eff·N(N−1))

N_max = sqrt((1 − α) / κ_eff)
```

The same law governs coordination-dependent systems at every scale. The parameters change; the structure does not.

| System | α (serial bottleneck) | κ_base (pairwise sync cost) | N_max | What α and κ represent |
|---|---|---|---|---|
| CPU cache coherency | 0.02 | 0.0003 | ~57 | α = memory bus serialization; κ = cache-line exchange protocol |
| Human engineering team | 0.10 | 0.0083 | ~10 | α = planning/review cycles; κ = pairwise communication overhead (Brook's Law) |
| AI agents (same model) | 0.15 | 0.025 | ~4–5 | α = context compilation + synthesis; κ = pairwise output reconciliation at low CG |
| AI agents (diverse backends) | 0.12 | 0.018 | ~6–7 | α = same; κ lower because diverse models share less vocabulary, but diverge less on facts |

For AI agents, α captures the serial phases inherent to orchestration (you cannot parallelize task decomposition or final merge), and κ captures how expensive it is to find and resolve contradictions between N agents' partial answers. Higher κ = more divergence to reconcile = fewer agents before quality peaks.

Reference values: **α ≈ 0.10–0.15, κ_base ≈ 0.015–0.025, N_max ≈ 4–7** for typical LLM ensembles.

---

## The Event Vocabulary

All state is immutable event log entries on NATS JetStream. Crash recovery = replay from offset 0.

**Core orchestration events** (subject `h2ai.tasks.{task_id}`):
```
CalibrationCompletedEvent          → α, κ_base, CG samples, θ_coord locked
TaskBootstrappedEvent              → J_eff gate passed, system_context locked
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
ConsensusRequiredEvent             → max(c_i) > 0.85, switching CRDT → BFT
SemilatticeCompiledEvent           → merge ready, MergeStrategy recorded
MergeResolvedEvent                 → human O(1) decision, task closed
TaskFailedEvent                    → retries exhausted, full diagnostic payload
TaoIterationEvent                  → TAO loop turn result per Explorer per turn
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
AgentTelemetryEvent::ShellCommandExecuted → shell command run by edge agent (exit code recorded)
AgentTelemetryEvent::SystemError          → edge agent panic or unrecoverable error
```

---

## Repository Layout

```
h2ai-control-plane/
├── Dockerfile                      # multi-stage: builder (rust+clang) → runtime (debian-slim)
├── typeshare.toml                  # typeshare CLI config — Go bindings output to bindings/go/
├── bindings/
│   └── go/                         # generated Go types (from typeshare CLI, committed to repo)
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
│   │                               # compliance formula: hard_gate × soft_score; ADR loader (backward-compat)
│   ├── h2ai-provisioner/           # AgentProvider + SchedulingPolicy + NatsAgentProvider
│   │                               # Capability filter + cost-tier ceiling + least-loaded scheduling
│   ├── h2ai-memory/                # MemoryProvider trait + InMemoryCache + NatsKvStore
│   │                               # Stateless edge agents: all context lives in the control plane
│   ├── h2ai-planner/               # PlanningEngine::decompose + PlanReviewer::evaluate
│   │                               # LLM-driven task decomposition into SubtaskPlan; structural
│   │                               # cycle/empty checks before semantic LLM review
│   ├── h2ai-telemetry/             # AuditProvider trait + DirectLogProvider + BrokerPublisherProvider
│   │                               # Immutable audit log with secret redaction middleware
│   ├── h2ai-orchestrator/          # DAG builder + Pareto topology router + NatsDispatchAdapter
│   │                               # CompoundTaskEngine (decompose → review → schedule pipeline)
│   │                               # SchedulingEngine (Kahn topo-sort wave execution)
│   ├── h2ai-autonomic/             # MAPE-K loop + calibration harness + N_max calculator
│   ├── h2ai-state/                 # CRDT semilattice + NATS JetStream I/O + task dispatch wire protocol
│   ├── h2ai-context/               # Dark Knowledge Compiler + Jaccard + J_eff measurement
│   │                               # corpus_keywords from ConstraintDoc::vocabulary(); ADR shim
│   ├── h2ai-adapters/              # IComputeAdapter: Anthropic, OpenAI, Ollama, CloudGeneric, Mock + AdapterFactory
│   ├── h2ai-api/                   # axum REST gateway + Merge Authority web UI
│   └── h2ai-agent/                 # Edge agent binary — heartbeat publisher + NATS task dispatch loop
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
    ├── guides/                     # Getting started, constraint corpus, adapter development
    ├── reference/                  # API reference, configuration reference
    ├── operations/                 # Operations guide, troubleshooting
    ├── architecture/               # Design spec, math apparatus, runtime phases, crate boundaries
    └── examples/
        └── ads-platform/           # Reference constraint corpus + integration test task manifests
            ├── adr/                # 7 ADRs derived from "Architecting Real-Time Ads Platform"
            └── tasks/              # 3 task manifests with expected Auditor outcomes
```

**Dependency rule:** `h2ai-types`, `h2ai-config`, and `h2ai-constraints` have zero external I/O dependencies — predicate evaluation is pure computation. `h2ai-planner` depends only on `h2ai-types` (LLM calls go through `IComputeAdapter` from `h2ai-types`; no NATS or HTTP deps). Six crates import `async-nats` directly, each on a dedicated subject namespace: `h2ai-nats` (subject constants + NKey provisioning), `h2ai-state` (task event log), `h2ai-memory` (context history), `h2ai-telemetry` (audit log), `h2ai-provisioner` (agent heartbeats + soft-kill), `h2ai-agent` (binary — heartbeat publisher + task subscriber). `h2ai-api` is the only crate that talks to HTTP. Nothing imports `h2ai-api`.

**Compute isolation:** Cloud HTTP adapters (Anthropic, OpenAI, Ollama) run async on the main worker pool. Future llama.cpp FFI inference uses `spawn_blocking` with `max_blocking_threads` explicitly set so CPU-bound inference never starves NATS consumers or HTTP handlers.

---

## Research & Validation Scripts

`scripts/baseline_eval.py` is a **production tool**: measures real per-adapter accuracy (p)
and correlation (ρ) against `eval_questions.jsonl`. Run before high-stakes deployments to
override the CG_mean proxy with empirical values (`baseline_accuracy_proxy` config field).

The remaining scripts are **research/validation tools** — not required for deployment, but
run them to verify formula correctness after any change to calibration constants or physics
formulas. Each is the formal proof for a specific mathematical claim in
[`docs/architecture/research-state.md`](docs/architecture/research-state.md):

| Script | Validates |
|---|---|
| `validate_beta_coupling.py` | β_eff = β₀×(1−CG) is bounded; inverse form diverges at CG→0 |
| `validate_bft_methods.py` | Weiszfeld breakdown point 50%; Token Krum fails on LLM paraphrases |
| `validate_eigenvalue_calibration.py` | N_eff participation ratio detects hidden adapter redundancy |
| `validate_information_theory.py` | I_marginal decay, N_it_optimal, Slepian-Wolf efficiency |
| `validate_ensemble_theory.py` | CJT formula vs 100k-trial Monte Carlo; J_eff gate |
| `validate_conformal_vs_cjt.py` | CJT over-prediction at ρ≥0.6 vs conformal coverage guarantee |
| `validate_math.py` | Numerically validates all definitions and propositions in `docs/architecture/math-apparatus.md` (stdlib only, no dependencies) |
| `simulate_usl.py` | USL throughput curves, CG effect, Pareto matrix (generates PNGs to `scripts/output/`) |

---

## Technology Stack

| Layer | Choice | Why |
|---|---|---|
| Language | Rust + Tokio | Compiler-verified CRDT state, zero-cost FFI to llama.cpp, no GC jitter in κ_base |
| Event log | NATS JetStream | Single static binary (MB of RAM), Tokio-native `async-nats`, clusters natively |
| State model | Event-sourced CRDT | α→0 during generation (no locks), full provenance chain, crash recovery = replay |
| Local compute | llama.cpp FFI | Zero-cost, 128GB RAM dedicated to weights |
| Edge agents | `AgentDescriptor { model, tools }` | Any LLM-based container described by model name + capability flags; stateless `f(ctx, τ) → result`, scoped NKeys per task |
| HTTP | axum | Tokio-native, same async runtime as orchestrator |
| Type bindings | `typeshare` | Rust types → Go structs for edge agent contracts; no hand-maintained schemas |
| Tracing | `tracing` + OpenTelemetry | task_id as root span, DAG execution visible in Jaeger/Grafana Tempo |
| Metrics | Prometheus `/metrics` | 20 gauges: κ_eff, α, N_max, θ_coord, J_eff, VRAM, c_i per role, adapter latency |

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

The **Dark Knowledge Compiler** reads your team's constraint documents and ADRs. Each document becomes a typed `ConstraintDoc` with a `ConstraintPredicate` (vocabulary presence, negative keywords, regex, numeric threshold, LLM judge, or composites) and a severity (`Hard`, `Soft`, or `Advisory`). Hard constraints gate the merge; Soft constraints contribute a weighted compliance score; `constraint_error_cost = 1.0 − compliance` feeds back into the BFT merge strategy selector.

Every `## Constraints` bullet in a standard ADR is automatically parsed as a `Hard { threshold: 0.8 }` constraint — **existing ADR corpora require no changes**.

`docs/examples/ads-platform/` is a complete reference corpus derived from the blog series **[Architecting Real-Time Ads Platform](https://e-mindset.space/series/architecting-ads-platforms/)**:

| ADR | Decision |
|---|---|
| ADR-001 | Stateless request services — no per-user state across requests |
| ADR-002 | gRPC internal / REST external — JSON overhead consumes 20% of latency budget at 1M QPS |
| ADR-003 | Adaptive RTB timeouts via HdrHistogram — per-DSP P95, capped at 100ms global |
| ADR-004 | Budget pacing with idempotency — Redis Lua atomic check-and-set, 30s TTL |
| ADR-005 | Dual-ledger audit log — Kafka → ClickHouse append-only, 7-year retention, SOX |
| ADR-006 | Java 21 + Generational ZGC — 32GB heap, <2ms P99.9 GC pauses |
| ADR-007 | Tiered consistency — budget=strong, profiles=eventual, billing=linearizable |

The task manifests in `docs/examples/ads-platform/tasks/` are the input corpus for the integration test suite. They specify the expected Auditor outcomes — which proposals should be pruned, which ADR they violate — so that `cargo nextest run --test integration` can assert system behavior end-to-end.

---

## Documentation

### Guides

| Document | Contents |
|---|---|
| [Getting Started](docs/guides/getting-started.md) | First task end-to-end — Local Plan, Server Plan team node, Cloud Plan Kubernetes |
| [Agent Descriptor Guide](docs/guides/agent-descriptor.md) | Pure LLM vs. tool-using agents — how tools affect α, κ_base, c_i, topology selection, and NKey scoping; worked examples |
| [Constraint Corpus Guide](docs/guides/constraint-corpus.md) | ConstraintDoc/ConstraintPredicate type system, Hard/Soft/Advisory severity, compliance formula, ADR backward-compat, diagnosing low J_eff, remediation hints for MAPE-K retry |
| [Adapter Development](docs/guides/adapters.md) | Implementing `IComputeAdapter` for custom compute backends, testing, registration |
| [Theory to Implementation](docs/guides/theory-to-implementation.md) | Topology selection protocol, 7-topology catalog with Pareto scores, team-swarm configuration, worked example |

### Reference

| Document | Contents |
|---|---|
| [API Reference](docs/reference/api.md) | All REST endpoints, SSE event stream, complete JSON schemas for all 23 events, error codes |
| [Configuration Reference](docs/reference/configuration.md) | All environment variables, LLM adapter providers, 20 Prometheus metrics, Helm values |

### Architecture

| Document | Contents |
|---|---|
| [Design Specification](docs/architecture/design-specification.md) | System overview — positioning, tech stack, deployment plans, API contract, math summary |
| [Design Rationale](docs/architecture/design-rationale.md) | *Why* each major decision was made: NATS, USL, NKeys, Rust, event sourcing, CJT+USL together; honest gaps |
| [Differentiation](docs/architecture/differentiation.md) | How H2AI differs from LangGraph, AutoGen, CrewAI, Semantic Kernel — what it does better and worse |
| [Runtime Phases](docs/architecture/runtime-phases.md) | 6-phase execution flow + compound task pipeline, 23-event vocabulary, structural guarantees |
| [Crate Boundaries](docs/architecture/crate-boundaries.md) | Workspace layout, 15 crates, dependency rules, Tokio thread pool isolation |
| [Deployment](docs/architecture/deployment.md) | Three deployment plans, NATS clustering, Kubernetes topology, observability |
| [Math Apparatus](docs/architecture/math-apparatus.md) | USL/CJT theoretical foundations, 10 definitions + 5 propositions, calibration table, known limitations |
| [Research State](docs/architecture/research-state.md) | Project thesis, implemented math with validation evidence, gap analysis and fix status, innovation synthesis, script catalog |

### Operations

| Document | Contents |
|---|---|
| [Operations Guide](docs/operations/operations.md) | 6 key metrics, alert rules, scaling, rolling upgrade procedure, NATS backup |
| [Troubleshooting](docs/operations/troubleshooting.md) | ContextUnderflowError, zero-survival, Multiplication Condition failures, high α/κ |

### Examples

| Document | Contents |
|---|---|
| [Examples Overview](docs/examples/README.md) | What ADRs are, J_eff effect, how to run integration tests |
| [Ads Platform](docs/examples/ads-platform/README.md) | 7 ADRs + 3 integration test tasks derived from "Architecting Real-Time Ads Platform" |

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
