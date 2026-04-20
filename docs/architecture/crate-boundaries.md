# Crate Boundaries — Workspace Layout and Dependency Rules

The H2AI Control Plane is a Cargo workspace with fifteen crates. Each crate has exactly one responsibility and one direction of dependency flow. There are no circular dependencies. There are no exceptions.

---

## Workspace Layout

```
h2ai-control-plane/          (Cargo workspace root)
├── Cargo.toml               (workspace manifest, shared dependency versions)
├── typeshare.toml           (Go bindings config — output to bindings/go/)
└── crates/
    ├── h2ai-types/          # Pure types boundary — zero I/O dependencies
    ├── h2ai-nats/           # NATS subject constants + scoped NKey provisioning
    ├── h2ai-config/         # H2AIConfig — physics thresholds and role defaults
    ├── h2ai-constraints/    # ConstraintDoc type system — predicates, severity, compliance eval
    ├── h2ai-provisioner/    # AgentProvider + scheduling policies + NatsAgentProvider
    ├── h2ai-memory/         # MemoryProvider trait + InMemoryCache + NatsKvStore
    ├── h2ai-tools/          # ToolRegistry + ToolExecutor trait + ShellExecutor (sandboxed)
    ├── h2ai-planner/        # PlanningEngine::decompose + PlanReviewer::evaluate
    ├── h2ai-telemetry/      # AuditProvider trait + DirectLogProvider + BrokerPublisherProvider
    ├── h2ai-orchestrator/   # TAO loop + Verification + Attribution + NatsDispatchAdapter
    │                        # CompoundTaskEngine + SchedulingEngine
    ├── h2ai-autonomic/      # MAPE-K loop + calibration harness
    ├── h2ai-state/          # CRDT semilattice + NATS JetStream I/O + task dispatch wire protocol
    ├── h2ai-context/        # Dark Knowledge Compiler + J_eff measurement
    ├── h2ai-adapters/       # IComputeAdapter: llama.cpp FFI + cloud HTTP
    ├── h2ai-api/            # axum REST gateway + Merge Authority web UI
    └── h2ai-agent/          # Edge agent binary — heartbeat + task dispatch loop
```

---

## Dependency Graph

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
      ├── h2ai-autonomic → h2ai-config
      │    └── h2ai-types
      ├── h2ai-state
      │    └── h2ai-types
      ├── h2ai-context → h2ai-config · h2ai-constraints
      │    └── h2ai-types
      ├── h2ai-constraints
      │    └── h2ai-types
      ├── h2ai-adapters
      │    └── h2ai-types
      ├── h2ai-provisioner → h2ai-nats · async-nats
      │    └── h2ai-types
      ├── h2ai-memory
      │    └── h2ai-types
      ├── h2ai-planner
      │    └── h2ai-types
      └── h2ai-telemetry
           └── h2ai-types
```

**The rule in one sentence:** Every domain crate depends on `h2ai-types`. `h2ai-config` stands alone (no I/O deps). `h2ai-constraints` depends only on `h2ai-types` — predicate evaluation is pure computation, no I/O. `h2ai-nats` owns NATS subject naming and NKey provisioning. `h2ai-planner` depends only on `h2ai-types` — LLM calls go through the `IComputeAdapter` trait defined there; no NATS or HTTP deps. Six crates import `async-nats` directly — `h2ai-nats`, `h2ai-state`, `h2ai-memory`, `h2ai-telemetry`, `h2ai-provisioner`, and `h2ai-agent` — each on a separate subject namespace. Only `h2ai-api` talks to HTTP. Nothing imports `h2ai-api`.

---

## h2ai-types — The Pure Boundary

**Responsibility:** Define every shared type used across crate boundaries. Nothing else.

**Dependencies:** Zero external I/O dependencies. `serde`, `uuid`, `thiserror`, `async-trait` are permitted. `async-nats`, `axum`, `reqwest`, `tokio` runtime features are not.

**What lives here:**
- `TaskId`, `ExplorerId`, `SubtaskId` — UUID-backed newtypes with Display and serde
- `CoherencyCoefficients` — α, κ_base, CG samples; computes κ_eff() and N_max()
- `CoordinationThreshold` — θ_coord with from_calibration() constructor
- `RoleErrorCost` — c_i ∈ [0,1] validated weight
- `MergeStrategy` — CrdtSemilattice / BftConsensus with from_role_costs() selector
- `JeffectiveGap` — J_eff measurement with is_below_threshold()
- `MultiplicationCondition` — evaluate() returning which of 3 conditions failed
- `ParetoWeights` — diversity / containment / throughput weights (sum = 1.0)
- `AgentRole` — Coordinator / Executor / Evaluator / Synthesizer / Custom{name, tau, role_error_cost}; abstract topological roles, not domain-specific
- `TopologyKind` — Ensemble / HierarchicalTree{branching_factor} / TeamSwarmHybrid
- `RoleSpec` — {agent_id, role: AgentRole, tau: Option<f64>, role_error_cost: Option<f64>}; per-Explorer role assignment in the task manifest
- `ReviewGate` — {reviewer: String, blocks: String}; Evaluator-to-Executor dependency edge declared in the manifest
- `AdapterKind` — `LocalLlamaCpp { model_path: PathBuf, n_threads: usize }` / `CloudGeneric { endpoint: String, api_key_env: String }` / `OpenAI { api_key_env: String, model: String }` / `Anthropic { api_key_env: String, model: String }` / `Ollama { endpoint: String, model: String }`
- `ExplorerConfig`, `AuditorConfig` — τ values, role assignments; `AuditorConfig` also carries `max_tokens` and `prompt_template` (the `{constraints}` / `{proposal}` audit prompt — all LLM-facing strings live in config, never hardcoded)
- `TaoConfig` — `max_turns`, `verify_pattern`, and all four observation/retry strings (`observation_pass`, `observation_fail_pattern`, `observation_fail_schema`, `retry_instruction`); every string the TAO loop injects into a prompt comes from here
- `VerificationConfig` — `threshold`, `rubric`, `evaluator_system_prompt`, `evaluator_tau`, `evaluator_max_tokens`; every string and numeric constant sent to the evaluator LLM comes from here
- `IComputeAdapter` — the async trait all compute backends implement
- `ComputeRequest`, `ComputeResponse`, `AdapterError`
- All 23 event structs + `H2AIEvent` enum (internally tagged JSON)
- `AgentTool` — Shell / WebSearch / CodeExecution / FileSystem; capability flags carried as attributes of a descriptor
- `CostTier` — Low / Mid / High (ordered); agents declare their tier, tasks declare a maximum acceptable tier
- `AgentDescriptor` — {model: String, tools: Vec<AgentTool>, cost_tier: CostTier}; describes an LLM-based agent by model name, capability set, and cost category
- `TaskRequirements` — {max_cost_tier: CostTier, required_tools: Vec<AgentTool>}; passed to `AgentProvider::select_agent` to find a capable, cost-appropriate agent
- `AgentState` — Idle / Executing / AwaitingApproval / Failed(String); edge agent lifecycle
- `TaskPayload` — {task_id, agent_id, agent: AgentDescriptor, instructions, context, tau, max_tokens}; dispatched to edge agents over NATS core (ephemeral subject)
- `TaskResult` — {task_id, agent_id, output, token_cost, error}; published by edge agents to the `H2AI_RESULTS` JetStream work-queue stream
- `AgentHeartbeat` — {agent_id, descriptor, timestamp, active_tasks}; published by `h2ai-agent` every 10s to `h2ai.heartbeat.{agent_id}`
- `AgentTelemetryEvent` — LlmPromptSent / LlmResponseReceived / ShellCommandExecuted / SystemError; published to `h2ai.telemetry.*`
- `Subtask` — {id: SubtaskId, description, depends_on: Vec<SubtaskId>, role_hint: Option<AgentRole>}; one node in a subtask plan DAG
- `SubtaskPlan` — ordered Vec<Subtask> with `status: PlanStatus` and `parent_task_id`; adjacently-tagged serde (`{"status":"draft"}` for unit variants)
- `PlanStatus` — Draft / PendingReview / Approved / Rejected{reason} / Executing{completed,total} / Complete
- `SubtaskResult` — {subtask_id, output, token_cost, timestamp}; produced by each SchedulingEngine wave

All agent boundary types carry `#[typeshare]` — the CLI generates Go/TypeScript structs under `bindings/` for edge agent contract conformance.

**Why this boundary exists:** If `h2ai-adapters` imported `h2ai-state` to access event types, it would pull in the `async-nats` client as a transitive dependency. This means every compute adapter — including the llama.cpp FFI wrapper — would compile against the NATS client. The binary grows; the dependency graph becomes a web; local-only Local Plan must ship networking code it cannot use. `h2ai-types` breaks this: event types live in a crate with no I/O deps, so `h2ai-adapters` can import events without importing NATS.

---

## h2ai-nats — NATS Infrastructure

**Responsibility:** Own the NATS subject namespace and scoped NKey lifecycle. Every subject string in the system comes from here; no crate constructs NATS subjects by hand.

**Key behaviors:**
- `subjects` module — typed subject builders for `h2ai.tasks.{task_id}`, `h2ai.tasks.ephemeral.*`, `h2ai.telemetry.*`, `h2ai.memory.*`
- `nkey` module — generates per-task ephemeral NKeys scoped to a single `task_id`; the key expires when the task closes, so edge agents cannot retain NATS access

**Imports:** `h2ai-types` (for `TaskId`), `async-nats` (NKey generation), `nkeys`.

---

## h2ai-config — Runtime Configuration

**Responsibility:** Define `H2AIConfig` — the single struct that carries every physics threshold and role default used across crates. No I/O; callers load the file themselves.

**What lives here:**
- `j_eff_gate` (default 0.4) — ContextUnderflowError threshold
- `bft_threshold` (default 0.85) — switches CRDT → BFT when max(c_i) exceeds this
- `coordination_threshold_max` (default 0.3) — cap on θ_coord
- `min_baseline_competence` / `max_error_correlation` — Multiplication Condition constants
- `tau_*` — default τ per role (Coordinator 0.05, Executor 0.40, Evaluator 0.10, Synthesizer 0.80)
- `cost_*` — default role error cost per role (Coordinator 0.1, Executor 0.5, Evaluator 0.9, Synthesizer 0.1)
- `explorer_max_tokens` (default 1024) — token budget for Explorer generation calls
- `calibration_max_tokens` (default 256) — token budget for calibration probe calls
- `optimizer_threshold_step` (default 0.1) — step size for `SelfOptimizer` `verify_threshold` reduction
- `optimizer_threshold_floor` (default 0.3) — minimum `verify_threshold` the `SelfOptimizer` will suggest

All four new fields use `#[serde(default)]` so existing config files that predate them deserialize without error.

**Imports:** `serde`, `serde_json`, `thiserror` only. No `async-nats`, no `tokio`.

---

## h2ai-constraints — Constraint Type System and Evaluator

**Responsibility:** Define the composable `ConstraintDoc`/`ConstraintPredicate` type system and evaluate constraint compliance against proposal text. This crate replaces the keyword-bag ADR model with typed, machine-checkable predicates.

**Key types:**

```rust
pub enum ConstraintPredicate {
    VocabularyPresence { mode: VocabularyMode, terms: Vec<String> },
    NegativeKeyword { terms: Vec<String> },
    RegexMatch { pattern: String, must_match: bool },
    NumericThreshold { field_pattern: String, op: NumericOp, value: f64 },
    LlmJudge { rubric: String },
    Composite { op: CompositeOp, children: Vec<ConstraintPredicate> },
}

pub enum ConstraintSeverity {
    Hard { threshold: f64 },   // blocks merge if score < threshold; compliance → 0.0
    Soft { weight: f64 },       // weighted contribution to soft_score
    Advisory,                   // informational only; never blocks
}

pub struct ConstraintDoc {
    pub id: String,
    pub source_file: String,
    pub description: String,
    pub severity: ConstraintSeverity,
    pub predicate: ConstraintPredicate,
    pub remediation_hint: Option<String>,
}
```

**Compliance formula:**
```
hard_gate    = all Hard predicates produce score ≥ threshold
soft_score   = Σ(wᵢ × scoreᵢ) / Σwᵢ   (Soft constraints only)
compliance   = if hard_gate { soft_score } else { 0.0 }
constraint_error_cost = 1.0 − compliance   (fed into BranchPrunedEvent)
```

**`VocabularyMode` semantics:**
- `AllOf` — fractional score (`hits / total_terms`); all terms must appear for score = 1.0
- `AnyOf` — binary 1.0 if any term appears, 0.0 otherwise
- `NoneOf` — binary 1.0 if no term appears (negative keyword gate)

**`CompositeOp` identities:** `And` folds from 1.0 (vacuously true when empty); `Or` folds from 0.0 (vacuously false when empty). `Not` wraps a single child and inverts its score.

**`LlmJudge` path:** `eval_sync` returns 1.0 (pass-through); the async evaluation path in `h2ai-orchestrator::verification` calls the evaluator adapter.

**ADR backward-compatibility:** `crates/h2ai-context/src/adr.rs` re-exports:
```rust
pub type AdrConstraints = ConstraintDoc;
pub fn parse_adr(id: &str, content: &str) -> ConstraintDoc { parse_constraint_doc(id, content) }
pub fn load_corpus(dir) -> Result<Vec<ConstraintDoc>, io::Error> { load_constraint_corpus(dir) }
```
An ADR with `## Constraints` heading produces `Hard { threshold: 0.8 }` with a `VocabularyPresence { AllOf }` predicate. Existing ADR corpora require no changes.

**Loader (`loader.rs`):** Scans a directory recursively for `*.md` files. Heading priority: `## Hard Constraints` > `## Soft Constraints` > `## Advisory` > `## Constraints` (backward-compat). Each bullet becomes a term in the predicate's `terms` list.

**Imports:** `h2ai-types` (for `ConstraintViolation`, `ComplianceResult`), `regex`. No `async-nats`, no `tokio` runtime features, no HTTP.

---

## h2ai-orchestrator — DAG Builder, Topology Router, and Harness Engine

**Responsibility:** Fan out Explorers through the full production harness pipeline (TAO loop → Verification Phase → Auditor gate), enforce the Multiplication Condition gate, collect results, emit phase events, compute harness attribution.

**Key behaviors:**
- Builds the Explorer DAG from `TopologyProvisionedEvent`
- Runs context compaction (`context::compaction`) before Phase 3 to enforce token budget and Lost-in-Middle mitigation
- Wraps JoinSet in `tokio::time::timeout` — no Explorer hangs indefinitely
- Each Explorer runs `TaoLoop::run` (Phase 3): iterative TAO cycle up to `max_turns`, with regex pattern verification and optional JSON Schema validation per turn; emits `TaoIterationEvent` per turn
- Runs `VerificationPhase::run` in parallel across all proposals (Phase 3.5): scored LLM-as-judge, graceful 0.5 fallback on parse error; emits `VerificationScoredEvent` per proposal
- Emits `ProposalFailedEvent` on crash, OOM, or timeout
- Emits `GenerationPhaseCompletedEvent` when JoinSet is fully drained
- Verifies Multiplication Condition (Phase 2.5) before spawning any Explorer
- Executes review gate evaluation (Phase 3b) for `TeamSwarmHybrid`
- Classifies all errors into 4-class taxonomy (`error_class`): transient/recoverable/user-fixable/unexpected with typed retry policies
- Computes `HarnessAttribution` (Q_total decomposition) from actual TAO turns and filter ratio before merge
- Runs `SelfOptimizer::suggest(SuggestInput{...})` to tune N/max_turns/threshold for subsequent tasks; threshold bounds come from `H2AIConfig.optimizer_threshold_step/floor`

**`NatsDispatchAdapter`** implements `IComputeAdapter` and bridges the engine to remote edge agents:
1. Calls `AgentProvider::select_agent(&TaskRequirements)` to pick a target agent
2. Spawns an `await_task_result_once` consumer on the `H2AI_RESULTS` JetStream stream **before** publishing (ordering invariant)
3. Publishes a `TaskPayload` to core NATS `h2ai.tasks.ephemeral.{task_id}`
4. Awaits the `TaskResult` with a configurable timeout; maps it to `ComputeResponse`

`NatsDispatchConfig` carries `nats: Arc<NatsClient>`, `provider: Arc<dyn AgentProvider>`, `agent_descriptor: AgentDescriptor`, `task_requirements: TaskRequirements`, and `task_timeout: Duration`. `EngineInput.nats_dispatch: Option<NatsDispatchConfig>` — when `Some`, Phase 3 builds one `NatsDispatchAdapter` per explorer slot instead of drawing from `explorer_adapters`.

**`CompoundTaskEngine`** runs the three-step compound task pipeline:
1. `PlanningEngine::decompose` (in `h2ai-planner`) — one LLM call to decompose the task into a `SubtaskPlan`
2. `PlanReviewer::evaluate` (in `h2ai-planner`) — structural checks (empty plan, DFS cycle detection) then one LLM semantic review; returns `ReviewOutcome::Approved` or `Rejected{reason}`
3. `SchedulingEngine::execute` — Kahn's topo-sort splits subtasks into waves; each wave runs in parallel via `join_all`; completed subtask outputs are injected as `context` into dependents

**`SchedulingEngine`** in `scheduler.rs` uses the `SubtaskExecutor` trait to decouple scheduling from execution:
```rust
pub trait SubtaskExecutor: Send + Sync {
    async fn execute(&self, subtask_id: SubtaskId, manifest: TaskManifest) -> Result<SubtaskResult, SchedulerError>;
}
```

**Modules:** `engine`, `tao_loop`, `verification`, `attribution`, `error_class`, `self_optimizer`, `output_schema`, `nats_dispatch_adapter`, `compound`, `scheduler`

**Imports:** `h2ai-types`, `h2ai-state`, `h2ai-adapters`, `h2ai-context`, `h2ai-constraints`, `h2ai-autonomic`, `h2ai-provisioner`, `h2ai-planner`, `h2ai-tools` (optional tool execution in TAO loop).

---

## h2ai-autonomic — MAPE-K Loop and Calibration Harness

**Responsibility:** Measure α and κ_base, compute N_max and θ_coord, provision topology, and re-provision on ZeroSurvivalEvent or MultiplicationConditionFailedEvent.

**Key behaviors:**
- Runs calibration tasks against the adapter pool; stores CoherencyCoefficients in NATS KV
- Selects topology from manifest and calibration data: `TeamSwarmHybrid` if `roles[]` provided; explicit `ensemble`/`hierarchical_tree` if set; otherwise auto-selects from ParetoWeights and N vs N_max
- Assigns τ values across Explorers (spread to enforce error decorrelation)
- Assigns RoleErrorCost per node; computes MergeStrategy from max(c_i)
- Publishes TopologyProvisionedEvent
- Intercepts ZeroSurvivalEvent; diagnoses cause via `RetryPolicy::decide`:
  - Collects `remediation_hint` strings from `BranchPrunedEvent.violated_constraints` (Hard constraints with hints) → `RetryAction::RetryWithHints { topology, hints }` — targeted MAPE-K remediation
  - Falls back to keyword scan (hallucination signals in `BranchPrunedEvent.reason`) → `RetryAction::RetryWithTauReduction { topology, tau_factor: 0.7 }` when majority of pruned reasons indicate hallucination
  - Plain topology escalation → `RetryAction::Retry(topology)` otherwise
- Bounded by max_retries; publishes TaskFailedEvent with full diagnostic if retries exhausted

**Imports:** `h2ai-types`, `h2ai-state`, `h2ai-adapters`.

---

## h2ai-state — CRDT Semilattice and NATS JetStream I/O

**Responsibility:** Own all NATS I/O. Replay event logs. Compile CRDT semilattice joins. Run BFT consensus when MergeStrategy requires it.

**This is the only crate that touches NATS.** Every event append, every stream read, every KV store access goes through `state`. Other crates call `h2ai-state` functions; they never hold a NATS client directly.

**Key behaviors:**
- Publishes events to `h2ai.tasks.{task_id}` as immutable appends
- Reads calibration data from NATS KV store
- Replays event stream from offset 0 on crash recovery
- On Phase 5: replays validated events, constructs semilattice join of surviving proposals
- On BftConsensus path: runs BFT protocol over surviving proposals before emitting SemilatticeCompiledEvent
- `publish_task_payload(&TaskPayload)` — publishes a `TaskPayload` as JSON to core NATS subject `h2ai.tasks.ephemeral.{task_id}` for edge agent pickup
- `await_task_result_once(&TaskId, Duration)` — creates an ephemeral ordered JetStream consumer on the `H2AI_RESULTS` work-queue stream, filtered by task_id, and awaits the first matching `TaskResult` within a timeout

**NATS topology for task dispatch:** Core NATS (not JetStream) is used for `TaskPayload` dispatch — it is ephemeral broadcast. JetStream (`H2AI_RESULTS` with `WorkQueue` retention) is used for `TaskResult` collection — at-most-once delivery with ack. The `await_task_result_once` consumer must always be created **before** `publish_task_payload` is called to avoid a race where the result arrives before the consumer exists.

**Crash recovery invariant:** Full state is recoverable by replaying the NATS JetStream from offset 0. No external state store required.

---

## h2ai-context — Dark Knowledge Compiler

**Responsibility:** Measure J_eff, compile system_context, enforce the context sufficiency gate.

**Key behaviors:**
- Reads the submitted manifest and scans the local constraint corpus (via `h2ai-constraints::loader`)
- Computes J_eff = J(K_prompt, K_task_required) — Jaccard overlap; `corpus_keywords` is derived from `ConstraintDoc::vocabulary()` across all docs
- Returns ContextUnderflowError synchronously if J_eff < threshold (nothing written to NATS)
- Compiles immutable system_context string from constraint docs + manifest
- system_context is sealed at TaskBootstrappedEvent; no agent ever receives a different context
- `adr.rs` provides backward-compatible shims: `AdrConstraints = ConstraintDoc`, `parse_adr`, `load_corpus` — existing call sites require no changes

**Imports:** `h2ai-types`, `h2ai-constraints`, `h2ai-config`. No NATS or HTTP dependency.

---

## h2ai-adapters — Compute Backend Implementations

**Responsibility:** Implement `IComputeAdapter` for every compute backend. Isolate FFI and HTTP from the rest of the system. Provide `AdapterFactory` as the single place where `AdapterKind` → concrete adapter is resolved.

**Five concrete adapters:**

| Adapter | Kind | Auth | Notes |
|---|---|---|---|
| `AnthropicAdapter` | `Anthropic` | `x-api-key` header | POST `/v1/messages`; parses `content[].text` blocks |
| `OpenAIAdapter` | `OpenAI` | `Authorization: Bearer` | POST `/chat/completions`; sends `"model"` in body |
| `OllamaAdapter` | `Ollama` | none | POST `/api/chat`; temperature nested in `"options"`; `#[serde(default)]` on token counts |
| `CloudGenericAdapter` | `CloudGeneric` | `Authorization: Bearer` | Generic OpenAI-compatible endpoint without model field in kind |
| `MockAdapter` | `CloudGeneric` (sentinel) | — | Deterministic test double; returned for `provider=mock` |

**`AdapterFactory::build(kind: &AdapterKind) -> Result<Arc<dyn IComputeAdapter>, String>`** maps all five variants to their concrete adapter. `LocalLlamaCpp` returns `Err` (FFI not yet wired — use `Ollama` for local inference via Ollama server).

**Local inference path (llama.cpp FFI, future):**
- Must use `tokio::task::spawn_blocking` — CPU-bound matrix operations must not run on the async worker pool
- `max_blocking_threads` set explicitly in the runtime builder, calibrated to available RAM

**Why spawn_blocking for local inference:** llama.cpp matrix operations block for hundreds of milliseconds. If run on the async worker pool, every NATS consumer and HTTP handler waits. Tokio's blocking thread pool is separate, bounded, and does not affect async task scheduling.

**Imports:** `h2ai-types` only. No dependency on `h2ai-state` (no NATS client). No dependency on `h2ai-orchestrator`.

---

## h2ai-tools — Tool Execution Registry

**Responsibility:** Wire `AgentTool` capability flags to real, sandboxed executors. The TAO loop uses this crate to run rules-based verification steps (e.g. `cargo test`, `eslint`) as part of the iterative refinement cycle.

**The `ToolExecutor` trait:**
```rust
#[async_trait]
pub trait ToolExecutor: Send + Sync {
    async fn execute(&self, input: &str) -> Result<String, ToolError>;
}
```

**Key behaviors:**
- `ToolRegistry` maps `AgentTool` (typed enum key — not `Debug` string) to `Arc<dyn ToolExecutor>`
- `ShellExecutor`: spawns `sh -c <command>`, `kill_on_drop(true)` for process cleanup on cancellation, 5s timeout, 1MiB output cap
- `ToolError` covers: `NotRegistered`, `ShellFailed { exit_code, stderr }`, `Timeout`, `Io`

**Dependency rule:** `h2ai-tools` depends only on `h2ai-types` (for `AgentTool`) and `tokio`. It does **not** import `async-nats`, `state`, or `orchestrator`. Tool execution is fully decoupled from the event log.

**Imports:** `h2ai-types`, `tokio`, `thiserror`, `async-trait`.

---

## h2ai-provisioner — Agent Infrastructure Abstraction and Scheduling

**Responsibility:** Decouple task routing from container lifecycle management. Select the best available agent for a task using capability, cost, and load criteria.

**The `AgentProvider` trait:**
```rust
async fn ensure_agent_capacity(descriptor: &AgentDescriptor, task_load: usize) -> Result<(), ProvisionError>
async fn terminate_agent(agent_id: &AgentId) -> Result<(), ProvisionError>
async fn select_agent(requirements: &TaskRequirements) -> Result<AgentId, ProvisionError>
```

`select_agent` applies a three-stage filter before delegating to the active `SchedulingPolicy`:
1. **Capability filter** — exclude any agent missing a tool from `requirements.required_tools`
2. **Cost ceiling** — exclude any agent whose `cost_tier` exceeds `requirements.max_cost_tier`
3. **Policy selection** — among eligible agents, the `SchedulingPolicy` picks one

**`SchedulingPolicy` trait:**
```rust
fn select(&self, candidates: &[AgentCandidate]) -> Option<AgentId>
```

Two implementations:
- `LeastLoadedPolicy` (default) — sorts by `cost_tier` ascending, then `active_tasks` ascending, then `AgentId` as tiebreaker. A `Low`-tier agent with 99 active tasks always beats a `High`-tier agent with 0 tasks — cost tier takes absolute priority.
- `RoundRobinPolicy` — cycles through candidates sorted by `AgentId` using an `AtomicUsize` counter. Useful for load-testing or when all candidates are equivalent.

**Three `AgentProvider` implementations:**
- `StaticProvider` — assumes containers are externally managed (Podman/Docker local). Verifies availability via NATS heartbeats. Does not spawn processes. Sends soft-kill via NATS.
- `KubernetesProvider` — uses `kube-rs` to create a Kubernetes `Job` manifest per task. Selects image and mounts tool ConfigMaps based on `AgentDescriptor.tools`. Injects scoped NATS NKeys via env vars.
- `NatsAgentProvider` — for long-lived edge agents (the `h2ai-agent` binary). Maintains a live registry of `AgentRegistration` entries, keyed by `AgentId`, populated by subscribing to `h2ai.heartbeat.>`. Entries expire after a configurable TTL (default: 30s). `select_agent` filters the live registry by `TaskRequirements` and delegates to the active `SchedulingPolicy`. The `active_tasks` counter in each heartbeat is self-reported and used as a scheduling hint — it is stale by up to one heartbeat interval (10s) but sufficient for least-loaded approximation.

**NATS subjects:** `h2ai.heartbeat.{agent_id}` (heartbeat), `h2ai.control.terminate.{agent_id}` (soft-kill).

**Imports:** `h2ai-types` (for `AgentDescriptor`, `AgentTool`, `CostTier`, `TaskRequirements`, `AgentId`), `h2ai-nats` (subject constants), `async-nats` (heartbeat subscriptions and soft-kill dispatch). No dependency on `h2ai-state`, `h2ai-orchestrator`, or `h2ai-adapters`.

---

## h2ai-memory — Context and History Abstraction

**Responsibility:** Make edge agents stateless by managing conversation history and semantic context in the control plane.

**The `MemoryProvider` trait:**
```rust
async fn get_recent_history(session_id: &str, limit: usize) -> Result<Vec<EventRecord>, MemoryError>
async fn commit_new_memories(session_id: &str, memories: Vec<EventRecord>) -> Result<(), MemoryError>
async fn retrieve_relevant_context(session_id: &str, query: &str) -> Result<Vec<String>, MemoryError>
```

**Two implementations:**
- `InMemoryCache` — `dashmap::DashMap` for local development and testing. O(1) reads, no persistence.
- `NatsKvStore` — NATS Key-Value store backend. Persists history across control plane restarts. KV key pattern: `h2ai.memory.{session_id}`.

**Why this boundary exists:** Edge agents are ephemeral and stateless by design — they are described by `AgentDescriptor` but carry no history. Context assembly must not happen inside the container. `h2ai-memory` assembles the context before `TaskPayload` is constructed — the agent receives a complete, pre-assembled context and never needs access to history.

**Imports:** `h2ai-types` (for event types). `NatsKvStore` implementation imports `async-nats` directly.

---

## h2ai-planner — Task Decomposition and Plan Review

**Responsibility:** Convert a `TaskManifest` into a validated `SubtaskPlan` using LLM-driven decomposition and structural+semantic review.

**Key behaviors:**
- `PlanningEngine::decompose(manifest, adapter, tau) -> Result<SubtaskPlan, PlannerError>` — prompts the LLM to return a JSON array of subtasks with integer dependency indices; converts indices to `SubtaskId` references; returns `PlanStatus::PendingReview`
- `PlanReviewer::evaluate(plan, description, adapter, tau) -> Result<ReviewOutcome, PlannerError>` — runs structural checks (empty plan, DFS cycle detection with White/Gray/Black coloring) before any LLM call; on pass, one LLM semantic review returning `Approved` or `Rejected{reason}`
- Shared `parsing::extract_json` strips markdown fences and advances to `{` without truncating the JSON object at `rfind('}')` — lets serde_json handle object boundaries

**`ReviewOutcome` enum:**
```rust
pub enum ReviewOutcome {
    Approved,
    Rejected { reason: String },
}
```

**`PlannerError` variants:** `Adapter(String)` (LLM call failure), `ParseError(String)` (JSON parse failure)

**Why this crate is separate from `h2ai-orchestrator`:** Decomposition and review are pure LLM operations with no NATS, no state writes, and no orchestration DAG. Keeping them in `h2ai-planner` means the decomposition logic can be tested with `MockAdapter` without any orchestrator or NATS setup. `h2ai-orchestrator` imports `h2ai-planner` and wires it into `CompoundTaskEngine`.

**Imports:** `h2ai-types` only. No `async-nats`. No `tokio` runtime features. No `async-trait` (trait methods are free functions on unit structs).

---

## h2ai-telemetry — Audit Log and Secret Redaction

**Responsibility:** Route `AgentTelemetryEvent` entries to an immutable, redacted audit log. All edge agent activity is recorded before any other system sees it.

**The `AuditProvider` trait:**
```rust
async fn record_event(event: AgentTelemetryEvent) -> Result<(), AuditError>
async fn flush() -> Result<(), AuditError>
```

**Implementations:**
- `DirectLogProvider` — serializes events to JSON and writes to `stdout` via `tracing-appender`. Suitable for local development and log aggregation pipelines.
- `BrokerPublisherProvider` — publishes serialized events to NATS subject `h2ai.telemetry.*`. Enables stream replay and Grafana dashboard integration.
- `RedactionMiddleware` — wraps any `AuditProvider`. Scans `command`, `stdout`, and `stderr` string fields in `AgentTelemetryEvent` and replaces known API key patterns and secret regexes with `[REDACTED]` before the event reaches the provider.

**Why this boundary exists:** Secret redaction must be mandatory and cannot be bypassed. By placing `RedactionMiddleware` as the only public entry point for production providers, the architecture prevents unredacted telemetry from reaching the audit log by construction — not by convention.

**Imports:** `h2ai-types` (for `AgentTelemetryEvent`). `BrokerPublisherProvider` imports `async-nats` directly.

---

## h2ai-api — HTTP Gateway and Merge Authority UI

**Responsibility:** Accept task manifests, stream events to clients, render the Merge Authority interface.

**Key behaviors:**
- `POST /tasks` → validates manifest via `h2ai-context`, publishes TaskBootstrappedEvent via `h2ai-state`, uses `AppState.explorer_adapter` and `AppState.auditor_adapter`, returns 202 + task_id
- `POST /calibrate` → triggers calibration harness via `h2ai-autonomic` using `AppState.explorer_adapter`, returns 202
- `GET /tasks/{task_id}/events` → tails NATS JetStream subject, streams all 23 event types as SSE or WebSocket
- `GET /tasks/{task_id}/recover` → replays JetStream log via `SessionJournal`, upserts `TaskState` into `TaskStore`
- `POST /tasks/{task_id}/merge` → publishes `MergeResolvedEvent`, closes task
- `GET /metrics` → Prometheus endpoint: κ_eff, α, N_max, θ_coord, J_eff, VRAM, c_i per role

**`AppState`** holds `explorer_adapter: Arc<dyn IComputeAdapter>` and `auditor_adapter: Arc<dyn IComputeAdapter>` built from env vars via `AdapterFactory` at startup. Both default to `MockAdapter` when `H2AI_EXPLORER_PROVIDER`/`H2AI_AUDITOR_PROVIDER` are unset.

**This is the only crate that talks to HTTP.** axum runs on the same Tokio runtime as the orchestrator and NATS consumer. No second runtime.

**Stream lifecycle:** SSE/WebSocket stream opens on task creation and closes on MergeResolvedEvent (success) or TaskFailedEvent (failure with full diagnostic payload). The client sees the swarm working in real-time.

**Nothing imports `h2ai-api`.** It is the top of the dependency graph.

---

## Thread Pool Isolation

Tokio runs two thread pools in this system:

| Pool | Used for | Configuration |
|---|---|---|
| Async worker pool | NATS consumers, MAPE-K loop, axum HTTP handlers, cloud adapter HTTP calls | Default: `num_cpus` threads |
| Blocking thread pool | llama.cpp FFI inference (via `spawn_blocking`) | Explicit `max_blocking_threads(N)` |

`max_blocking_threads` is set at runtime builder construction, not left at Tokio's default (512). The correct value depends on available RAM and model size:
- Each blocking thread may hold a loaded model context
- Too many threads: OS scheduler contention spikes α — the contention coefficient the system is trying to minimize
- Too few threads: inference throughput drops below N_max Explorer capacity

The calibration harness measures the resulting α empirically. If α exceeds the reference range (0.10–0.15), reduce `max_blocking_threads`.

---

## h2ai-agent — Edge Agent Binary

**Responsibility:** Run a long-lived NATS subscriber that executes tasks on behalf of the control plane. This is a binary crate — it produces the `h2ai-agent` executable, not a library for other crates to import.

**What it does:**
- Connects to NATS at startup (`H2AI_NATS_URL` env var, default `nats://localhost:4222`)
- Generates a stable `AgentId` from `H2AI_AGENT_ID` env var or a fresh UUID on first start
- Publishes an `AgentHeartbeat` every 10 seconds to `h2ai.heartbeat.{agent_id}`, including the current `active_tasks` count — this is how `NatsAgentProvider` in the control plane discovers and tracks the agent
- Subscribes to `h2ai.tasks.ephemeral.>` and processes only messages where `payload.agent_id == self.agent_id` (other messages are silently skipped — the dispatch is addressed, not broadcast)
- Executes each received `TaskPayload` through its local `IComputeAdapter`, then publishes the `TaskResult` to the `H2AI_RESULTS` JetStream work-queue stream at `h2ai.results.{task_id}`
- Listens on `h2ai.control.terminate.{agent_id}` for graceful shutdown; also handles `SIGTERM`/`SIGINT`
- Increments/decrements a shared `Arc<AtomicU32>` counter around each task execution; the heartbeat reads this counter — the control plane sees load in near real-time

**Key types:**
- `HeartbeatTask` — owns the NATS client, `AgentId`, `AgentDescriptor`, and `Arc<AtomicU32>` counter. `.start()` returns a `JoinHandle` for the background publish loop.
- `DispatchLoop` — owns the NATS client, `AgentId`, `Arc<dyn IComputeAdapter>`, and `Arc<AtomicU32>` counter. `.run()` is the main select loop.

**Adapter selection:** The `MockAdapter` is used by default. Production deployments set `H2AI_EXPLORER_MODEL` and `H2AI_EXPLORER_PROVIDER` to configure a real adapter via `AdapterFactory`.

**Imports:** `h2ai-types`, `h2ai-adapters`, `h2ai-state` (to call `ensure_infrastructure` on startup), `h2ai-nats` (subject constants), `async-nats` (raw client for heartbeat and subscription). It is the only crate that simultaneously imports `h2ai-adapters` and `async-nats`.

---

## Enforcement

These rules are not conventions — they are enforced by Cargo's dependency graph:

- Six crates may import `async-nats` directly: `h2ai-nats` (subject constants + NKey provisioning), `h2ai-state` (task event log on `h2ai.tasks.*`), `h2ai-memory` (KV history store on `h2ai.memory.*`), `h2ai-telemetry` (audit log on `h2ai.telemetry.*`), `h2ai-provisioner` (agent heartbeats + soft-kill on `h2ai.agents.*`), and `h2ai-agent` (binary — heartbeat publisher + task subscriber). `h2ai-planner` and `h2ai-constraints` may NOT import `async-nats` — they interact with LLMs only through `IComputeAdapter`, and predicate evaluation is pure computation. No library crate beyond the six listed above may add `async-nats` to its `[dependencies]`.
- A crate cannot import `axum` unless it is `h2ai-api`. Nothing depends on `h2ai-api`.
- `h2ai-types`, `h2ai-config`, and `h2ai-constraints` will fail to compile if any I/O dependency is added (`async-nats`, `reqwest`, `axum`, `tokio` runtime features). This is caught at `cargo check`, not at runtime.

The boundary is the compiler. There is no runtime enforcement needed.
