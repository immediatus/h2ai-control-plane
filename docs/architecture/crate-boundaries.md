# Crate Boundaries — Workspace Layout and Dependency Rules

The H2AI Control Plane is a Cargo workspace with thirteen crates. Each crate has exactly one responsibility and one direction of dependency flow. There are no circular dependencies. There are no exceptions.

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
    ├── h2ai-provisioner/    # AgentProvider trait + StaticProvider + KubernetesProvider
    ├── h2ai-memory/         # MemoryProvider trait + InMemoryCache + NatsKvStore
    ├── h2ai-tools/          # ToolRegistry + ToolExecutor trait + ShellExecutor (sandboxed)
    ├── h2ai-telemetry/      # AuditProvider trait + DirectLogProvider + BrokerPublisherProvider
    ├── h2ai-orchestrator/   # TAO loop + Verification + Attribution + SelfOptimizer
    ├── h2ai-autonomic/      # MAPE-K loop + calibration harness
    ├── h2ai-state/          # CRDT semilattice + NATS JetStream I/O
    ├── h2ai-context/        # Dark Knowledge Compiler + J_eff measurement
    ├── h2ai-adapters/       # IComputeAdapter: llama.cpp FFI + cloud HTTP
    └── h2ai-api/            # axum REST gateway + Merge Authority web UI
```

---

## Dependency Graph

```
h2ai-config ──────────────────────────────────────────► (standalone)
h2ai-nats ────────────────────────────────────────────► h2ai-types

h2ai-api
 └── h2ai-orchestrator → h2ai-nats
      ├── h2ai-autonomic → h2ai-config
      │    └── h2ai-types
      ├── h2ai-state
      │    └── h2ai-types
      ├── h2ai-context → h2ai-config
      │    └── h2ai-types
      ├── h2ai-adapters
      │    └── h2ai-types
      ├── h2ai-provisioner
      │    └── h2ai-types
      ├── h2ai-memory
      │    └── h2ai-types
      └── h2ai-telemetry
           └── h2ai-types
```

**The rule in one sentence:** Every domain crate depends on `h2ai-types`. `h2ai-config` stands alone (no I/O deps). `h2ai-nats` owns NATS subject naming and NKey provisioning. Four crates import `async-nats` directly — `h2ai-nats`, `h2ai-state`, `h2ai-memory`, and `h2ai-telemetry` — each on a separate subject namespace. Only `h2ai-api` talks to HTTP. Nothing imports `h2ai-api`.

---

## h2ai-types — The Pure Boundary

**Responsibility:** Define every shared type used across crate boundaries. Nothing else.

**Dependencies:** Zero external I/O dependencies. `serde`, `uuid`, `thiserror`, `async-trait` are permitted. `async-nats`, `axum`, `reqwest`, `tokio` runtime features are not.

**What lives here:**
- `TaskId`, `ExplorerId` — UUID-backed newtypes with Display and serde
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
- `AdapterKind` — Local / Cloud
- `ExplorerConfig`, `AuditorConfig` — τ values, role assignments; `AuditorConfig` also carries `max_tokens` and `prompt_template` (the `{constraints}` / `{proposal}` audit prompt — all LLM-facing strings live in config, never hardcoded)
- `TaoConfig` — `max_turns`, `verify_pattern`, and all four observation/retry strings (`observation_pass`, `observation_fail_pattern`, `observation_fail_schema`, `retry_instruction`); every string the TAO loop injects into a prompt comes from here
- `VerificationConfig` — `threshold`, `rubric`, `evaluator_system_prompt`, `evaluator_tau`, `evaluator_max_tokens`; every string and numeric constant sent to the evaluator LLM comes from here
- `IComputeAdapter` — the async trait all compute backends implement
- `ComputeRequest`, `ComputeResponse`, `AdapterError`
- All 17 event structs + `H2AIEvent` enum (internally tagged JSON)
- `AgentDescriptor` — {model: String, tools: Vec<AgentTool>}; describes an LLM-based agent by model name and capability set
- `AgentTool` — Shell / WebSearch / CodeExecution / FileSystem; capability flags carried as attributes of a descriptor
- `AgentState` — Idle / Executing / AwaitingApproval / Failed(String); edge agent lifecycle
- `TaskPayload` — {task_id, agent: AgentDescriptor, instructions, context, tau, max_tokens}; dispatched to edge agents over NATS
- `TaskResult` — {task_id, agent_id, output, token_cost, error}; returned by edge agents
- `AgentTelemetryEvent` — LlmPromptSent / LlmResponseReceived / ShellCommandExecuted / SystemError; published to `h2ai.telemetry.*`

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

**Modules:** `engine`, `tao_loop`, `verification`, `attribution`, `error_class`, `self_optimizer`, `output_schema`

**Imports:** `h2ai-types`, `h2ai-state`, `h2ai-adapters`, `h2ai-context`, `h2ai-autonomic`, `h2ai-tools` (optional tool execution in TAO loop).

---

## h2ai-autonomic — MAPE-K Loop and Calibration Harness

**Responsibility:** Measure α and κ_base, compute N_max and θ_coord, provision topology, and re-provision on ZeroSurvivalEvent or MultiplicationConditionFailedEvent.

**Key behaviors:**
- Runs calibration tasks against the adapter pool; stores CoherencyCoefficients in NATS KV
- Selects topology from manifest and calibration data: `TeamSwarmHybrid` if `roles[]` provided; explicit `ensemble`/`hierarchical_tree` if set; otherwise auto-selects from ParetoWeights and N vs N_max
- Assigns τ values across Explorers (spread to enforce error decorrelation)
- Assigns RoleErrorCost per node; computes MergeStrategy from max(c_i)
- Publishes TopologyProvisionedEvent
- Intercepts ZeroSurvivalEvent; adjusts {N, τ}; republishes TopologyProvisionedEvent
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

**Crash recovery invariant:** Full state is recoverable by replaying the NATS JetStream from offset 0. No external state store required.

---

## h2ai-context — Dark Knowledge Compiler

**Responsibility:** Measure J_eff, compile system_context, enforce the context sufficiency gate.

**Key behaviors:**
- Reads the submitted manifest and scans the local ADR corpus
- Computes J_eff = J(K_prompt, K_task_required) — Jaccard overlap
- Returns ContextUnderflowError synchronously if J_eff < threshold (nothing written to NATS)
- Compiles immutable system_context string from ADRs + manifest
- system_context is sealed at TaskBootstrappedEvent; no agent ever receives a different context

**Imports:** `h2ai-types` only. Context compilation has no NATS or HTTP dependency.

---

## h2ai-adapters — Compute Backend Implementations

**Responsibility:** Implement IComputeAdapter for every compute backend. Isolate FFI and HTTP from the rest of the system.

**Two adapter kinds:**

**Local (llama.cpp FFI):**
- Called via `tokio::task::spawn_blocking` — runs on Tokio's bounded blocking thread pool
- Never runs on the async worker pool — inference work does not starve NATS consumers, MAPE-K loop, or HTTP handlers
- `max_blocking_threads` is set explicitly in the runtime builder, calibrated to available RAM and model weight size

**Cloud (HTTP):**
- Async HTTP client (`reqwest`) on the main async worker pool
- Standard timeout; failures surface as AdapterError

**Why spawn_blocking for local inference:** llama.cpp matrix operations are CPU-bound and block for hundreds of milliseconds. If run on the async worker pool, every NATS consumer and HTTP handler waits. Tokio's blocking thread pool is designed for exactly this: it is separate, bounded, and does not affect async task scheduling. The `max_blocking_threads` cap prevents OS scheduler thrash (too many threads competing for CPU time spikes α) while ensuring the pool is large enough to saturate available hardware.

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

## h2ai-provisioner — Agent Infrastructure Abstraction

**Responsibility:** Decouple task routing from container lifecycle management. The orchestrator asks "is there capacity?" — provisioner answers and acts.

**The `AgentProvider` trait:**
```rust
async fn ensure_agent_capacity(descriptor: &AgentDescriptor, task_load: usize) -> Result<(), ProvisionError>
async fn terminate_agent(agent_id: &str) -> Result<(), ProvisionError>
```

Agents are described by `AgentDescriptor { model: String, tools: Vec<AgentTool> }` — a model name and a set of capability flags (`Shell`, `WebSearch`, `CodeExecution`, `FileSystem`). Providers select the appropriate container image and tool mounts from this descriptor; there are no hard-coded agent type variants.

**Two implementations:**
- `StaticProvider` — assumes containers are externally managed (Podman/Docker local). Verifies availability via NATS heartbeats. Does not spawn processes. Sends soft-kill via NATS.
- `KubernetesProvider` — uses `kube-rs` to create a Kubernetes `Job` manifest per task. Selects image and mounts tool ConfigMaps based on `AgentDescriptor.tools`. Injects scoped NATS NKeys via env vars. Phase 2 target.

**Imports:** `h2ai-types` only (for `AgentDescriptor`, `AgentTool`, `TaskId`). No dependency on `h2ai-state`, `h2ai-orchestrator`, or `h2ai-adapters`.

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
- `POST /tasks` → validates manifest via `h2ai-context`, publishes TaskBootstrappedEvent via `h2ai-state`, returns 202 + task_id
- `POST /calibrate` → triggers calibration harness via `h2ai-autonomic`, returns 202
- `GET /tasks/{task_id}/events` → tails NATS JetStream subject, streams all 17 event types as SSE or WebSocket
- Merge Authority UI → renders valid proposals, tombstone panel, autonomic shift timeline, physics panel
- `GET /metrics` → Prometheus endpoint: κ_eff, α, N_max, θ_coord, J_eff, VRAM, c_i per role

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

## Enforcement

These rules are not conventions — they are enforced by Cargo's dependency graph:

- Exactly four crates may import `async-nats` directly: `h2ai-nats` (subject constants + NKey provisioning), `h2ai-state` (task event log on `h2ai.tasks.*`), `h2ai-memory` (KV history store on `h2ai.memory.*`), and `h2ai-telemetry` (audit log on `h2ai.telemetry.*`). No other crate may list `async-nats` in its `[dependencies]`.
- A crate cannot import `axum` unless it is `h2ai-api`. Nothing depends on `h2ai-api`.
- `h2ai-types` and `h2ai-config` will fail to compile if any I/O dependency is added (`async-nats`, `reqwest`, `axum`, `tokio` runtime features). This is caught at `cargo check`, not at runtime.

The boundary is the compiler. There is no runtime enforcement needed.
