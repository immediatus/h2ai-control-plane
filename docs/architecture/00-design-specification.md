# H2AI Control Plane — System Design Specification

**Date:** 2026-04-19  
**Author:** Yuriy Polyulya  
**Status:** Approved (rev 2 — gap analysis fixes applied)

---

## Overview

H2AI Control Plane is a distributed multi-agent orchestration runtime that treats agent swarms as a control theory problem governed by the Universal Scalability Law (USL). It is the definitive, mathematically sound alternative to unbounded prompt-chaining frameworks (e.g., OpenClaw).

The system is best understood as an **advanced distributed scheduler**: instead of scheduling static processes onto physical CPU cores, it schedules nondeterministic LLM inference tasks onto dynamically provisioned graph topologies — and mathematically bounds the coordination overhead at every step.

### Core positioning vs. OpenClaw

| Feature | OpenClaw | H2AI Control Plane |
|---|---|---|
| Execution routing | Sequential prompt-chaining | Deterministic DAGs |
| State management | LLM context window | Sovereign CRDTs (event-sourced) |
| Safety | Prompt-based trust | Topological interlocks (Auditor gate) |
| Scalability | Unbounded provisioning | MAPE-K autonomic shifting against N_max |
| Human integration | Chat / correction loop | CRDT Merge Authority (O(1) resolution) |

---

## Technology Stack

| Layer | Decision | Rationale |
|---|---|---|
| Language | Rust + Tokio async runtime | Compiler-verified CRDT state, zero-cost FFI to llama.cpp, no GC jitter in κ_base |
| Event log | NATS JetStream | Single static binary (megabytes of RAM), Tokio-native `async-nats` crate, clusters natively for Profile C |
| State model | Event-sourced CRDT | α→0 during generation (no locks), full epistemic provenance chain, crash recovery = replay from offset 0 |
| Local compute | llama.cpp via Rust FFI | Zero-cost, 128GB RAM dedicated to model weights |
| HTTP layer | axum | Tokio-native, type-safe routing, same async runtime as orchestrator |
| Tracing | `tracing` + OpenTelemetry → Jaeger / Grafana Tempo | task_id as root span, DAG execution visible as trace tree |
| Metrics | `metrics` + `metrics-exporter-prometheus` | USL physics gauges + hardware utilization |

**Rust is the correct choice over Go** because: (1) CRDT semilattice correctness is enforced by the borrow checker and enum exhaustiveness, not discipline; (2) llama.cpp FFI is zero-cost with no CGO overhead; (3) no GC jitter in the MAPE-K background loop — κ_base is mathematically flat.

**NATS over Kafka/Redpanda** because: Kafka requires JVM overhead that competes with LLM weights for RAM in Profile A. NATS runs as a single static binary in megabytes. **NATS over custom WAL** because: implementing log compaction, consumer offsets, and replication is an engineering vanity project that stops you building a control plane and starts you building a database.

---

## Deployment Profiles

The system is designed **C-first**: Profile C (distributed cluster) is the architectural foundation. Profile B (team server interface) is the human layer on top. Profile A (local dev) is running the full C+B stack on a single machine.

### Profile A — Solo / Local Dev (bootstrap)
Single Fedora workstation (128GB RAM). Static Rust binary + nats-server running natively. Local llama.cpp for high-variance Explorer tasks, cloud APIs for Auditor gates. Used to prove MAPE-K physics before deploying to a team node.

### Profile B — Small Team / Enterprise Node (human interface layer)
Dedicated server or VM. Multiple humans submit async manifests via REST API. Web-based Merge Authority UI for senior engineers to resolve CRDT diffs. Dark Knowledge Compiler enforces team-wide ADR constraints. Dark Knowledge gap (J_eff) is highest here — the orchestrator bridges tacit team knowledge into explicit constraints.

### Profile C — Distributed Cluster (architectural foundation)
Kubernetes. Multiple orchestrator replicas share replicated CRDT state via NATS JetStream. Control Plane API is logically centralized (one CRDT lattice) but physically distributed. Compute adapters distributed across GPUs, machines, and cloud regions.

**Critical implication:** CRDT state is replicated via NATS JetStream from day one — never local-only in-memory. New orchestrator pods recover by replaying the event stream from offset 0.

---

## Section 1: Architecture & Component Boundaries

### Workspace layout

```
h2ai-control-plane/          (Cargo workspace)
├── crates/
│   ├── h2ai-types/          # Pure types boundary — zero external deps
│   ├── orchestrator/        # DAG builder + Pareto topology router
│   ├── autonomic/           # MAPE-K loop + calibration + N_max calculator
│   ├── state/               # CRDT semilattice logic + NATS JetStream I/O
│   ├── context/             # Dark Knowledge Compiler + Jaccard + J_eff measurement
│   ├── adapters/            # IComputeAdapter trait impl: llama.cpp FFI + cloud HTTP
│   └── api/                 # axum REST gateway + Merge Authority web UI
├── nats/                    # nats-server config (dev + cluster modes)
├── docs/
│   ├── architecture/
│   ├── examples/
│   └── superpowers/specs/
└── Cargo.toml               # workspace root
```

### Dependency rule

```
api → orchestrator → autonomic, state, context, adapters
                                    ↓         ↓       ↓        ↓
                               h2ai-types (all crates depend on this)
```

- `h2ai-types` owns: all event structs, `ComputeResponse`, `IComputeAdapter` trait, Pareto weight types, topology enums, calibration types, role error cost types, merge strategy enum. **Zero external dependencies.**
- `state` owns NATS JetStream I/O and CRDT semilattice logic. `adapters` never sees `state`. `state` never sees `adapters`.
- `api` depends only on `orchestrator`. Nothing imports `api`. The compute core is fully testable without HTTP.

### Key types in `h2ai-types`

Beyond the 14-event vocabulary (see Section 2), `h2ai-types` defines these load-bearing types:

**`CoherencyCoefficients`** — Measured calibration data. Contains `alpha` (serial contention), `kappa_base` (baseline pairwise coherency cost), and `cg_samples: Vec<f64>` (measured Common Ground values across Explorer pairs). Produced by the calibration harness before the first provisioning of any live task. Reference values for AI agents: α ≈ 0.10–0.15, κ_base ≈ 0.01, N_max ≈ 6–7.

**`RoleErrorCost`** — Per-role Byzantine error weight `c_i ∈ [0, 1]`. `c_i = 0` means the role's errors are costless (early Explorer drafts). `c_i = 1` means the role's errors propagate at full damage (Auditor falsely passing a hallucination to the human). Used by `autonomic` to adjust topology and by `state` to select merge strategy.

**`MergeStrategy`** — `CrdtSemilattice` (default) or `BftConsensus`. Selected by `autonomic` at provisioning time based on the maximum `c_i` in the active role set. CRDT is AP and preserves epistemic diversity. BFT is CP and provides safety guarantees at the cost of κ. Switch threshold: if `max(c_i) > 0.85`, switch to `BftConsensus`.

**`CoordinationThreshold`** — `θ_coord = min(CG_mean − σ_CG, 0.3)`. Computed during calibration. Every edge in the DAG must exceed θ_coord. If any Explorer pair falls below it, `autonomic` either adds a mediating Coordinator node or reduces N.

**`MultiplicationCondition`** — The three gating conditions from Proposition 3 that must all hold before Phase 3 can begin: (1) baseline competence > 0.5 on calibration set, (2) pairwise error correlation ρ < 0.9 across Explorer pairs, (3) `CG_mean ≥ θ_coord`. If any condition fails, `MultiplicationConditionFailedEvent` is published and the system re-enters Phase 2.

**`JeffectiveGap`** — `J_eff = J(K_prompt, K_task_required)`. The Jaccard overlap between what the human explicitly provided in the manifest and what the task actually requires. Low J_eff means high Dark Knowledge gap — the orchestrator throws `ContextUnderflowError` before any agent is spawned.

### Topology variants (both implemented)

**Flat Mesh** — Used when `N ≤ N_max` and `W_H` (diversity weight) is high. All Explorers communicate through the NATS event stream; no Coordinator. `O(N(N-1)/2)` coordination edges, but acceptable when N is small. No single point of failure. Auditor reads all proposals independently.

**Hierarchical Tree** — Used when `N > N_max` or `W_E` (containment weight) is dominant. One Swarm Coordinator + k sub-groups of Explorers. Coordination edges reduced to `O(N-1)`. Branching factor `k_opt ≈ floor(N_max^flat)` computed from the flat-mesh N_max as heuristic. Auditor gates each sub-group independently before results reach the Coordinator.

The topology selection is computable, not heuristic: it is determined entirely by `{α, κ_eff, W_X, W_E, W_H, N_requested}`.

### Tokio runtime — two thread pools, explicit bounds

**Async worker pool** (CPU core count threads): NATS consumer, MAPE-K background loop, axum HTTP, DAG orchestration, calibration harness. Never blocked by compute.

**Blocking pool** (explicitly bounded via `max_blocking_threads(N)` in runtime builder): llama.cpp FFI calls via `tokio::task::spawn_blocking`. N calibrated to machine RAM and model size. Unbounded blocking threads cause OS scheduler thrash that artificially spikes α — this bound is load-bearing for the framework's correctness.

---

## Section 2: Data Flow — Event Stream & Runtime Phases

### Event vocabulary (14 events, all published to `h2ai.tasks.{task_id}`)

| Event | Publisher | Meaning |
|---|---|---|
| `CalibrationCompletedEvent` | autonomic | α, κ_base, CG samples measured; θ_coord computed; CoherencyCoefficients locked |
| `TaskBootstrappedEvent` | context / api | J_eff computed, Dark Knowledge gate passed, system_context locked |
| `TopologyProvisionedEvent` | autonomic | DAG shape + Explorer count + τ values + RoleErrorCosts + MergeStrategy assigned |
| `MultiplicationConditionFailedEvent` | orchestrator | One of the 3 Proposition 3 conditions failed; re-entering Phase 2 |
| `ProposalEvent` | adapters (via orchestrator) | Explorer completed, output appended |
| `ProposalFailedEvent` | orchestrator | Explorer crashed, OOM, or timed out — terminal state guaranteed |
| `GenerationPhaseCompletedEvent` | orchestrator | JoinSet fully drained, stream closed |
| `ValidationEvent` | adapters (Auditor) | Proposal passed Auditor gate |
| `BranchPrunedEvent` | adapters (Auditor) | Proposal rejected, reason + c_i weight recorded, branch tombstoned |
| `ZeroSurvivalEvent` | orchestrator | All proposals rejected, triggering autonomic retry |
| `SemilatticeCompiledEvent` | state | CRDT join or BFT consensus complete, MergeStrategy recorded, ready for human resolution |
| `ConsensusRequiredEvent` | state | max(c_i) exceeded BFT threshold; merge switches from CRDT to BftConsensus |
| `MergeResolvedEvent` | api | Human performed O(1) merge decision, task closed |
| `TaskFailedEvent` | orchestrator | Autonomic retries exhausted, full diagnostic payload |

### Phase-by-phase flow

**Phase 0 — Calibration** *(new — runs once per operator environment, cached)*  
Before any live task, `autonomic` runs the calibration harness: a small set of representative tasks (configurable, default 3) through the adapter pool to measure empirical `α` (serial fraction) and `κ_base` (pairwise coherency cost). CG samples are computed across Explorer pairs using `CG(i,j) = J(K_i, K_j) × alignment(τ_i, τ_j)`. `θ_coord = min(CG_mean − σ_CG, 0.3)` is computed and stored in `CoherencyCoefficients`. `CalibrationCompletedEvent` is published. Calibration results are cached in the NATS KV store and reused until the adapter pool changes or the operator forces a recalibration. **No live task proceeds without valid calibration data.**

Reference coefficients (AI agent baseline, used as defaults before first calibration):
- α ≈ 0.10–0.15 (serial fraction from shared context window updates)
- κ_base ≈ 0.01 (pairwise token-level coherency cost)
- N_max ≈ 6–7 agents (typical AI swarm ceiling)

**Phase 1 — Bootstrap**  
Human POSTs manifest to `api`. `context` compiles ADRs, computes `J_eff = J(K_prompt, K_task_required)`. If `J_eff` below threshold → synchronous `ContextUnderflowError`, nothing written to NATS. If passed → `TaskBootstrappedEvent` published with locked `system_context`, Pareto weights, and `J_eff` value.

**Phase 2 — Topology Provisioning**  
`autonomic` consumes bootstrap event and `CoherencyCoefficients` from calibration cache. Computes `κ_eff = κ_base / mean(CG)`, `N_max = sqrt((1-α) / κ_eff)`. Selects topology:
- If `N_requested ≤ N_max` AND `W_H` dominant → **Flat Mesh**
- Otherwise → **Hierarchical Tree** with `k_opt = floor(N_max^flat)` branching factor

Assigns τ values per Explorer (spread across [τ_min, τ_max] to guarantee error decorrelation). Assigns `RoleErrorCost` per node. Computes `MergeStrategy` based on `max(c_i)`. Publishes `TopologyProvisionedEvent`. **MAPE-K retry re-enters here** after `ZeroSurvivalEvent` or `MultiplicationConditionFailedEvent`.

**Phase 2.5 — Multiplication Condition Gate** *(new — before any compute is spawned)*  
`orchestrator` verifies all three conditions from Proposition 3 before allowing Phase 3:
1. **Baseline competence:** Each proposed Explorer adapter must have p_correct > 0.5 on the calibration task set (measured in Phase 0).
2. **Error decorrelation:** Pairwise agreement rate ρ < 0.9 across Explorer pairs on the calibration set. If Explorers are too similar (same model, similar τ), they fail this gate. Fix: widen τ spread or route some Explorers to different model backends.
3. **Common Ground floor:** `CG_mean ≥ θ_coord` for all planned Explorer pairs.

If all three hold → proceed to Phase 3. If any fails → publish `MultiplicationConditionFailedEvent` (with which condition failed and why) → re-enter Phase 2 with adjusted parameters. Bounded by `max_retries`. Exhaustion → `TaskFailedEvent`.

**Phase 3 — Parallel Generation** *(structural guarantee: terminal state for every Explorer)*  
`orchestrator` fans out N Explorers into a `tokio::task::JoinSet` wrapped in `tokio::time::timeout`. Local Explorers → `spawn_blocking` → llama.cpp FFI. Cloud Explorers → async HTTP. On success: `ProposalEvent` published, Explorer terminates. On crash/OOM/timeout: `ProposalFailedEvent` published — every Explorer guaranteed a terminal state. JoinSet drained → `GenerationPhaseCompletedEvent` closes the stream. **No Explorer reads another Explorer's output.**

**Phase 4 — Auditor Gate** *(structural guarantee: reactive stream, never idle)*  
Auditor spins up immediately on `TopologyProvisionedEvent`. Validates each `ProposalEvent` as it arrives in real-time: publishes `ValidationEvent` (pass) or `BranchPrunedEvent` (fail + reason + `c_i` weight of the violated constraint). Reads `GenerationPhaseCompletedEvent` to know the stream is closed. If survivors > 0 → Phase 5. If survivors = 0 → `ZeroSurvivalEvent`.

**Phase 4→2 — Autonomic Retry**  
`autonomic` intercepts `ZeroSurvivalEvent`. Adjusts {N, τ, topology} and publishes a new `TopologyProvisionedEvent`. Bounded by `max_retries`. Exhaustion → `TaskFailedEvent` with full diagnostic.

**Phase 5 — Merge + Human Resolution**  
`state` checks `MergeStrategy` from the active `TopologyProvisionedEvent`:

- **CrdtSemilattice** (default): Replays validated events, constructs CRDT semilattice join of surviving proposals. `O(1)` reconciliation, preserves epistemic diversity.
- **BftConsensus** (when `max(c_i) > 0.85`): `ConsensusRequiredEvent` published first. State engine runs BFT consensus protocol over surviving proposals before presenting to human. Higher κ cost, but mathematically safe when role error cost is catastrophic.

Publishes `SemilatticeCompiledEvent` (with `merge_strategy` recorded). `api` renders Merge Authority UI. Human performs resolution → `MergeResolvedEvent` closes the task.

### Structural guarantees

1. **Every Explorer has a terminal state.** `ProposalEvent` OR `ProposalFailedEvent`. `GenerationPhaseCompletedEvent` closes the stream. Auditor never hangs.
2. **Auditor is never idle.** Reactive stream processor — validates as proposals arrive.
3. **Zero-survival is handled.** MAPE-K retries. Exhaustion → `TaskFailedEvent` with full diagnostic.
4. **Multiplication Condition is enforced.** All three Proposition 3 conditions checked before compute is spawned.
5. **Calibration precedes provisioning.** No live task proceeds without measured {α, κ_base, CG}.
6. **Merge strategy matches error stakes.** CRDT for normal operations; BFT when role error cost exceeds safety threshold.

---

## Section 3: Error Handling & Observability

Errors are state transitions, not bugs. Every failure is epistemic data.

### 3.1 API Contract — Async State Boundaries

**`POST /tasks`**
- Success → `202 Accepted` + `task_id`.
- Synchronous failure (J_eff gate miss) → `400 Bad Request` + `ContextUnderflowError` (includes measured J_eff and required threshold).
- Missing calibration → `503 Service Unavailable` + `CalibrationRequiredError`.

**`GET /tasks/{task_id}/events`**  
SSE or WebSocket. Tails NATS subject `h2ai.tasks.{task_id}`. Stream closes on `MergeResolvedEvent` or `TaskFailedEvent`. `TaskFailedEvent` payload: full array of `BranchPrunedEvent` reasons + c_i weights, all τ values tried, topologies tried, which Multiplication Condition failed (if applicable).

**`POST /calibrate`**  
Triggers calibration harness on the current adapter pool. Returns `202 Accepted` + `calibration_id`. Calibration result streamed on `h2ai.calibration.{calibration_id}`. Useful when adapters change or operator wants to force recalibration.

### 3.2 Merge Authority UI — Failures as Epistemic Data

The UI surfaces all failure states with full context.

**Pruned Proposals Panel:** `BranchPrunedEvent`s rendered alongside valid proposals, including the `c_i` weight of the violated constraint. Example: *"Explorer A (Local 70B, τ=0.8) proposed [Diff]. Auditor rejected: 'Violates ADR-004: Stateless Auth requirement' [c_i = 0.92 — high-stakes constraint]."*

**Autonomic Shift Timeline:** `ZeroSurvivalEvent` renders as timeline node: *"Zero valid proposals. Autonomic loop engaged. Topology shifted: N increased 3→5. τ range widened. Retrying."* `MultiplicationConditionFailedEvent` renders as: *"Pre-flight gate failed: error decorrelation ρ = 0.94 (threshold: 0.9). Widening τ spread and retrying."*

**Merge Strategy Indicator:** Prominently displayed in the UI header. Green = CrdtSemilattice (diversity preserved). Amber = BftConsensus (consensus required — high c_i detected). Includes explanation: *"BFT Consensus active: Auditor role error cost c_i = 0.91 exceeds safety threshold. Proposals must reach Byzantine agreement before human resolution."*

**Physics Panel:** Live θ_coord and J_eff displayed during task execution. If θ_coord is close to CG_mean, the operator sees they are near the coordination floor.

### 3.3 Telemetry — The Physics Dashboard

**Distributed Tracing** (`tracing` + OpenTelemetry → Jaeger / Grafana Tempo)  
- `task_id` as root span. Each Explorer spawn is a child span. Spans killed by timeout appear with timeout tag.
- Calibration harness runs appear as a dedicated `calibration` trace root.
- `MultiplicationConditionFailedEvent` appears as a span annotation with which condition failed.

**Prometheus `/metrics` — Full USL Physics Gauges**

| Metric | What it tracks |
|---|---|
| `h2ai_kappa_eff` | Effective Coherency. Rising → approaching N_max. |
| `h2ai_kappa_base` | Baseline pairwise coherency cost from calibration. |
| `h2ai_alpha` | Serial Contention. Spikes → shared resource bottleneck. |
| `h2ai_n_max` | Current scalability ceiling. |
| `h2ai_theta_coord` | Coordination threshold. If CG_mean approaches this, DAG is near the floor. |
| `h2ai_j_eff` | Dark Knowledge gap. Low J_eff → operator has not externalized enough tacit knowledge. |
| `h2ai_cg_mean` | Mean Common Ground across Explorer pairs in current task. |
| `h2ai_role_error_cost{role}` | Per-role c_i weight. Rising → merge strategy may switch to BFT. |
| `h2ai_adapter_vram_bytes{adapter}` | VRAM per llama.cpp adapter. Spikes → κ_base rises → N_max drops. |
| `h2ai_token_latency_seconds{adapter,quantile}` | p50/p99 per adapter. |
| `h2ai_autonomic_retry_total` | Rising → ADR constraints miscalibrated. |
| `h2ai_multiplication_condition_failures_total{condition}` | Counts per condition (competence / decorrelation / cg_floor). |
| `h2ai_merge_strategy{task_id}` | 0=CRDT, 1=BFT. Distribution shows how often high-stakes tasks appear. |
| `h2ai_calibration_age_seconds` | Time since last calibration. Alert if stale. |

---

## Appendix: Full Mathematical Foundation

### Universal Scalability Law (Extended)

$$X(N) = \frac{N}{1 + \alpha(N-1) + \kappa_{\text{eff}} N(N-1)}$$

- **α** — contention coefficient (serial bottleneck fraction). Measured empirically via calibration.
- **κ_eff** — effective coherency: `κ_base / mean(CG(i,j))`
- **N_max** — `sqrt((1-α) / κ_eff)` — throughput peaks here; adding agents beyond this degrades performance.

### Common Ground Coefficient

$$CG(i,j) = J(K_i, K_j) \times \text{alignment}(\tau_i, \tau_j)$$

Where `J(K_i, K_j)` is Jaccard overlap of knowledge sets and `alignment(τ_i, τ_j) = 1 - |τ_i - τ_j|`. High τ divergence increases alignment cost, reducing CG.

### Dark Knowledge Gap

$$J_{\text{eff}} = J(K_{\text{prompt}}, K_{\text{task\_required}})$$

If `J_eff` is near zero, the human has provided minimal explicit context relative to what the task requires. High Dark Knowledge gap means high κ — agents must guess tacit constraints, inflating coordination cost and hallucination risk. The Dark Knowledge Compiler throws `ContextUnderflowError` before spawning any agent.

### Coordination Threshold

$$\theta_{\text{coord}} = \min(CG_{\text{mean}} - \sigma_{CG},\ 0.3)$$

Every planned edge in the DAG must satisfy `CG(i,j) ≥ θ_coord`. Edges below the floor require either a mediating Coordinator node or a reduction in N.

### Multiplication Condition (Proposition 3)

For collective performance to exceed individual performance, **all three** must hold simultaneously:

1. **Baseline competence:** `p_correct > 0.5` for every Explorer on calibration tasks.
2. **Error decorrelation:** `ρ(err_i, err_j) < 0.9` for every Explorer pair (agreement rate < 90% on calibration set — structurally diverse enough to add information).
3. **Common Ground floor:** `CG_mean ≥ θ_coord` for all planned pairs.

Adding an agent that violates any condition makes the collective *worse*. This is not a gradient — it is a hard gate.

### Byzantine Expected Loss

$$L_i = c_i \times P(\text{hallucination}_i) \times \text{propagation}(\text{topology})$$

- In a Flat Mesh: `propagation = N - 1` (one hallucination contaminates all peers)
- In a Hierarchical Tree: `propagation ≤ k` (bounded by branching factor)
- Auditor gate reduces `P(hallucination)` reaching the Merge Authority to near zero for validated proposals

### Merge Strategy Selection (Proposition 5 Safety Boundary)

$$\text{MergeStrategy} = \begin{cases} \text{CrdtSemilattice} & \text{if } \max(c_i) \leq 0.85 \\ \text{BftConsensus} & \text{if } \max(c_i) > 0.85 \end{cases}$$

CRDT merge is AP — it preserves epistemic diversity but does not guarantee safety when error costs are catastrophic. BFT consensus is CP — it provides safety at the cost of higher κ (coordination overhead). The threshold 0.85 is configurable per deployment.

### Reference Calibration Values

| System type | α | κ_base | Typical N_max |
|---|---|---|---|
| CPU cache coherency | 0.02 | 0.0003 | ~57 cores |
| Human engineering team | 0.10 | 0.0083 | ~10 people |
| AI agent swarm (same model) | 0.15 | 0.025 | ~4–5 agents |
| AI agent swarm (diverse backends) | 0.12 | 0.018 | ~6–7 agents |
