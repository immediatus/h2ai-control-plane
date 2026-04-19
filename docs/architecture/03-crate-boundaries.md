# Crate Boundaries — Workspace Layout and Dependency Rules

The H2AI Control Plane is a Cargo workspace with seven crates. Each crate has exactly one responsibility and one direction of dependency flow. There are no circular dependencies. There are no exceptions.

---

## Workspace Layout

```
h2ai-control-plane/          (Cargo workspace root)
├── Cargo.toml               (workspace manifest, shared dependency versions)
└── crates/
    ├── h2ai-types/          # Pure types boundary — zero I/O dependencies
    ├── orchestrator/        # DAG builder + Pareto topology router
    ├── autonomic/           # MAPE-K loop + calibration harness
    ├── state/               # CRDT semilattice + NATS JetStream I/O
    ├── context/             # Dark Knowledge Compiler + J_eff measurement
    ├── adapters/            # IComputeAdapter: llama.cpp FFI + cloud HTTP
    └── api/                 # axum REST gateway + Merge Authority web UI
```

---

## Dependency Graph

```
api
 └── orchestrator
      ├── autonomic
      │    └── h2ai-types
      ├── state
      │    └── h2ai-types
      ├── context
      │    └── h2ai-types
      └── adapters
           └── h2ai-types
```

**The rule in one sentence:** Every crate depends on `h2ai-types`. Only `state` talks to NATS. Only `api` talks to HTTP. Nothing imports `api`.

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
- `TopologyKind` — FlatMesh / HierarchicalTree
- `AdapterKind` — Local / Cloud
- `ExplorerConfig`, `AuditorConfig` — τ values, role assignments
- `IComputeAdapter` — the async trait all compute backends implement
- `ComputeRequest`, `ComputeResponse`, `AdapterError`
- All 14 event structs + `H2AIEvent` enum (internally tagged JSON)

**Why this boundary exists:** If `adapters` imported `state` to access event types, it would pull in the `async-nats` client as a transitive dependency. This means every compute adapter — including the llama.cpp FFI wrapper — would compile against the NATS client. The binary grows; the dependency graph becomes a web; local-only profile A must ship networking code it cannot use. `h2ai-types` breaks this: event types live in a crate with no I/O deps, so `adapters` can import events without importing NATS.

---

## orchestrator — DAG Builder and Topology Router

**Responsibility:** Fan out Explorers into a `tokio::task::JoinSet`, enforce the Multiplication Condition gate, collect results, emit phase events.

**Key behaviors:**
- Builds the Explorer DAG from `TopologyProvisionedEvent`
- Wraps JoinSet in `tokio::time::timeout` — no Explorer hangs indefinitely
- Emits `ProposalFailedEvent` on crash, OOM, or timeout
- Emits `GenerationPhaseCompletedEvent` when JoinSet is fully drained
- Verifies Multiplication Condition (Phase 2.5) before spawning any Explorer

**Imports:** `h2ai-types`, `state` (to publish events), `adapters` (to call IComputeAdapter), `context` (to read compiled system_context), `autonomic` (MAPE-K feedback).

---

## autonomic — MAPE-K Loop and Calibration Harness

**Responsibility:** Measure α and κ_base, compute N_max and θ_coord, provision topology, and re-provision on ZeroSurvivalEvent or MultiplicationConditionFailedEvent.

**Key behaviors:**
- Runs calibration tasks against the adapter pool; stores CoherencyCoefficients in NATS KV
- Computes topology (FlatMesh vs HierarchicalTree) from ParetoWeights and calibration data
- Assigns τ values across Explorers (spread to enforce error decorrelation)
- Assigns RoleErrorCost per node; computes MergeStrategy from max(c_i)
- Publishes TopologyProvisionedEvent
- Intercepts ZeroSurvivalEvent; adjusts {N, τ}; republishes TopologyProvisionedEvent
- Bounded by max_retries; publishes TaskFailedEvent with full diagnostic if retries exhausted

**Imports:** `h2ai-types`, `state`, `adapters`.

---

## state — CRDT Semilattice and NATS JetStream I/O

**Responsibility:** Own all NATS I/O. Replay event logs. Compile CRDT semilattice joins. Run BFT consensus when MergeStrategy requires it.

**This is the only crate that touches NATS.** Every event append, every stream read, every KV store access goes through `state`. Other crates call `state` functions; they never hold a NATS client directly.

**Key behaviors:**
- Publishes events to `h2ai.tasks.{task_id}` as immutable appends
- Reads calibration data from NATS KV store
- Replays event stream from offset 0 on crash recovery
- On Phase 5: replays validated events, constructs semilattice join of surviving proposals
- On BftConsensus path: runs BFT protocol over surviving proposals before emitting SemilatticeCompiledEvent

**Crash recovery invariant:** Full state is recoverable by replaying the NATS JetStream from offset 0. No external state store required.

---

## context — Dark Knowledge Compiler

**Responsibility:** Measure J_eff, compile system_context, enforce the context sufficiency gate.

**Key behaviors:**
- Reads the submitted manifest and scans the local ADR corpus
- Computes J_eff = J(K_prompt, K_task_required) — Jaccard overlap
- Returns ContextUnderflowError synchronously if J_eff < threshold (nothing written to NATS)
- Compiles immutable system_context string from ADRs + manifest
- system_context is sealed at TaskBootstrappedEvent; no agent ever receives a different context

**Imports:** `h2ai-types` only. Context compilation has no NATS or HTTP dependency.

---

## adapters — Compute Backend Implementations

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

**Imports:** `h2ai-types` only. No dependency on `state` (no NATS client). No dependency on `orchestrator`.

---

## api — HTTP Gateway and Merge Authority UI

**Responsibility:** Accept task manifests, stream events to clients, render the Merge Authority interface.

**Key behaviors:**
- `POST /tasks` → validates manifest via `context`, publishes TaskBootstrappedEvent via `state`, returns 202 + task_id
- `POST /calibrate` → triggers calibration harness via `autonomic`, returns 202
- `GET /tasks/{task_id}/events` → tails NATS JetStream subject, streams all 14 event types as SSE or WebSocket
- Merge Authority UI → renders valid proposals, tombstone panel, autonomic shift timeline, physics panel
- `GET /metrics` → Prometheus endpoint: κ_eff, α, N_max, θ_coord, J_eff, VRAM, c_i per role

**This is the only crate that talks to HTTP.** axum runs on the same Tokio runtime as the orchestrator and NATS consumer. No second runtime.

**Stream lifecycle:** SSE/WebSocket stream opens on task creation and closes on MergeResolvedEvent (success) or TaskFailedEvent (failure with full diagnostic payload). The client sees the swarm working in real-time.

**Nothing imports api.** It is the top of the dependency graph.

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

- A crate cannot import `async-nats` unless it depends on `state`. `state` is the only crate that depends on `async-nats` directly.
- A crate cannot import `axum` unless it depends on `api`. Nothing depends on `api`.
- `h2ai-types` will fail to compile if any I/O dependency is added (`async-nats`, `reqwest`, `axum`, `tokio` runtime features). This is caught at `cargo check`, not at runtime.

The boundary is the compiler. There is no runtime enforcement needed.
