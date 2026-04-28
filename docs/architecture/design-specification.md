# H2AI Control Plane — System Design Specification

**Date:** 2026-04-19 · **Author:** Yuriy Polyulya · **Status:** Approved (rev 3)

H2AI Control Plane is a distributed multi-agent orchestration runtime that prevents LLM agent swarms from degrading under their own coordination cost. It measures the overhead of making N agents agree — using the Universal Scalability Law to model the serial planning bottleneck (α) and the pairwise semantic reconciliation cost (β) — and enforces typed constraints so agents share enough common ground to produce coherent results.

The system is best understood as an **advanced distributed scheduler**: instead of scheduling static processes onto physical CPU cores, it schedules nondeterministic LLM inference tasks onto dynamically provisioned graph topologies — and bounds the coordination overhead at every step using measured parameters rather than configuration guesswork.

**For a full explanation of design decisions and tradeoffs:** see [Design Rationale](design-rationale.md).  
**For a comparison with LangGraph, AutoGen, CrewAI, and Semantic Kernel:** see [Differentiation](differentiation.md).

---

## Positioning

| Feature | LangChain / CrewAI | OpenAI Agents SDK | H2AI Control Plane |
|---------|-------------------|------------------|-------------------|
| Execution routing | Sequential / graph DSL | Handoffs, agents-as-tools | Deterministic DAGs |
| State management | LLM context window | Thread storage | Sovereign CRDTs (event-sourced) |
| Safety | Rules / guardrails | Guardrails | Typed `ConstraintDoc` predicates (Hard/Soft/Advisory) + ADR Auditor gate + J_eff gate + Multiplication check |
| Scalability | Unbounded provisioning | Unbounded provisioning | MAPE-K autonomic shifting against N_max |
| Human integration | Chat / correction loop | Human-in-the-loop hooks | CRDT Merge Authority (O(1) resolution) |
| Agent identity | Named product variants | Function tools | `AgentDescriptor { model, tools }` — any LLM |
| Tool risk | Implicit, untracked | Function signatures | `` `AgentTool` flags → c_i, α, β_base, NKey scope ``|
| Iterative refinement | ReAct (manual prompt) | ReAct | TAO loop — harness-driven, pattern-verified |
| Quality filtering | None / rules | Guardrails | Phase 3.5 scored LLM-as-judge (Verification Phase) |
| Context management | Summarization | Compaction | Lost-in-Middle mitigation + keyword preservation |
| Harness attribution | None | None | **Unique: USL quality decomposition (baseline + topology + verification + TAO)** |
| Error taxonomy | Ad hoc | Ad hoc | 4-class typed (transient / recoverable / user-fixable / unexpected) |

---

## Technology Stack

| Layer | Decision | Rationale |
|-------|----------|-----------|
| Language | Rust + Tokio | Compiler-verified CRDT state, zero-cost FFI to llama.cpp, no GC jitter in β_base |
| Event log | NATS JetStream | Single static binary, Tokio-native async-nats, clusters natively for Cloud Plan |
| State model | Event-sourced CRDT | α→0 during generation (no locks), full epistemic provenance chain, crash recovery = replay from offset 0 |
| Local compute | llama.cpp via Rust FFI | Zero-cost, 128GB RAM dedicated to model weights |
| HTTP layer | axum | Tokio-native, type-safe routing, same async runtime as orchestrator |
| Tracing | tracing + OpenTelemetry → Jaeger / Grafana Tempo | task_id as root span, DAG execution visible as trace tree |
| Metrics | metrics + metrics-exporter-prometheus | USL physics gauges + hardware utilization |

**NATS over Kafka/Redpanda:** Kafka requires JVM overhead that competes with LLM weights for RAM in Local Plan. NATS runs as a single static binary in megabytes, provides both the event log (JetStream) and the calibration cache (KV) from one binary, and file-backed persistence means crash recovery is replay from offset 0 — the same mechanism in development and production.

---

## Deployment Plans

The system is **C-first**: Cloud Plan is the architectural foundation. Server Plan is the human interface layer on top. Local Plan is the full Cloud+Server stack on one machine. The CRDT state model, NATS topology, and event vocabulary are identical across all three.

| Plan | Hardware | Agent provider | Memory provider | Telemetry provider |
|------|----------|---------------|-----------------|-------------------|
| **Local** | Single workstation | StaticProvider | InMemoryCache | DirectLogProvider |
| **Server** | Dedicated server/VM | NatsAgentProvider (recommended) / StaticProvider | NatsKvStore | BrokerPublisher + RedactionMiddleware |
| **Cloud** | Kubernetes cluster | KubernetesProvider | NatsKvStore | BrokerPublisher + RedactionMiddleware |

**Edge agent model (all plans):** Edge agents are ephemeral, stateless LLM-based containers described by `AgentDescriptor { model: String, tools: Vec<AgentTool> }`. The `model` selects the LLM backend (any string: `"llama3-70b"`, `"gpt-4o"`, `"claude-3-opus"`); `tools` is a set of capability flags (`Shell`, `WebSearch`, `CodeExecution`, `FileSystem`) that the container is granted at launch. The control plane dispatches a `TaskPayload` (which carries the full `AgentDescriptor`) over NATS, receives a `TaskResult`, and streams `AgentTelemetryEvent` entries. The agent has no persistent state — all context is assembled by `h2ai-memory` and injected via the payload. NATS credentials are scoped NKeys that expire when the task closes.

The `tools` field is not cosmetic. It is the primary input to three physics quantities:
- **c_i (role error cost):** determined by tool destructiveness — pure LLM c_i ≈ 0.1, Shell agent c_i ≈ 0.6–0.9. When `max(c_i) > 0.85`, `MergeStrategy` switches from CRDT to BFT.
- **α (serial contention):** tool calls introduce serialization. A pool of Shell-capable agents has α ≈ 0.20–0.30, lowering N_max and causing the planner to provision fewer explorers.
- **β_base (pairwise coherency cost):** WebSearch and external API calls introduce retrieval nondeterminism, raising CG variance and therefore β_eff.

See [`../guides/agent-descriptor.md`](../guides/agent-descriptor.md) for full parameter tables, topology examples, and FAQ.

See [`deployment.md`](deployment.md) for per-plan startup sequences, NATS cluster configs, Kubernetes topology, observability metrics, and environment variables.

### Managed Agents Patterns

The Server and Cloud plans use three complementary patterns that together make the harness crash-safe and observable without an external database:

**JetStream as authoritative session log (`NatsKvStore`).**
Every `H2AIEvent` is appended to a durable JetStream stream keyed by `task_id`. The stream is the single source of truth. `TaskStore` is a write-through cache — on cache miss, `SessionJournal::replay` reconstructs full `TaskState` by replaying from offset 0. No SQL, no Redis, no custom checkpoint format.

**Harness recovery via wake pattern (`SessionJournal`).**
`GET /tasks/{task_id}/recover` replays the JetStream log, upserts the reconstructed `TaskState` into the live `TaskStore`, and returns the current status JSON. A server restart therefore requires one HTTP call per in-flight task to restore the in-memory cache — not a database restore. `SessionJournal` is injected into `AppState` as `Arc<SessionJournal>` so handlers can replay on demand.

**Live agent registry (`NatsAgentProvider`).**
Instead of spinning up a Kubernetes Job per task (Cloud) or assuming containers are pre-started (Static), `NatsAgentProvider` maintains a live registry of managed agents that self-register via NATS heartbeats. The provisioner selects agents by `AgentDescriptor` match, dispatches `TaskPayload` over NATS, and issues soft-kills via `h2ai.agents.kill.{agent_id}`. This is the recommended provider for Server plan — it supports long-lived agents without container orchestration overhead.

---

## Architecture

The workspace contains 16 crates. The dependency flow is acyclic:

```
api
 └── orchestrator → autonomic · state · context · constraints · adapters · h2ai-tools
                    h2ai-provisioner · h2ai-memory · h2ai-telemetry · h2ai-nats
                    (all above depend on h2ai-types)
config ──────────────────────────────────────────────────────► (standalone)
constraints ─────────────────────────────────────────────────► h2ai-types (no I/O)
```

See [`crate-boundaries.md`](crate-boundaries.md) for workspace layout, each crate's responsibility, and enforcement rules.

---

## Runtime

Execution proceeds through six phases driven by NATS JetStream events:

| Phase | Name | Publisher | Gate |
|-------|------|-----------|------|
| 0 | Calibration | autonomic | `POST /calibrate`; result cached in KV |
| 1 | Bootstrap | context + api | J_eff gate; `ContextUnderflowError` if below 0.4 |
| 2 | Topology Provisioning | autonomic | Requires valid calibration data |
| 2.5 | Multiplication Condition | orchestrator | All 3 Proposition 3 conditions must hold |
| 3 | Parallel Generation (TAO) | orchestrator + adapters | Each Explorer runs TAO loop (max 3 turns); pattern-verified; every Explorer guaranteed terminal state |
| 3.5 | Verification Phase | orchestrator | Scored LLM-as-judge; proposals below threshold soft-rejected before Auditor |
| 3b | Review Gate (TeamSwarmHybrid only) | orchestrator | Evaluator approves or tombstones before Auditor |
| 4 | Auditor Gate | adapters + orchestrator | Reactive; ZeroSurvivalEvent triggers MAPE-K retry |
| 5 | Merge + Human Resolution | state + api | CRDT or BFT (recomputed from c_i_effective after TAO); human closes via POST /tasks/{id}/merge |

See [`runtime-phases.md`](runtime-phases.md) for full phase descriptions, the 23-event vocabulary, structural guarantees, and SSE stream lifecycle.

---

## Production Harness Components

A production agent harness requires 12 components (Anthropic/OpenAI/LangChain convergence, 2026). H2AI's implementation status:

| Component | H2AI Implementation | Crate |
|-----------|--------------------|----|
| **Orchestration loop** | TAO loop per Explorer — Thought-Action-Observation, max 3 turns, harness-driven | `orchestrator::tao_loop` |
| **Tools** | `ToolRegistry` with typed `AgentTool` key; `ShellExecutor` with `kill_on_drop`, 1MiB cap | `h2ai-tools` |
| **Memory** | `h2ai-memory` crate; `NatsKvStore` persists history across restarts; `SessionJournal` enables crash recovery via `GET /tasks/{id}/recover` | `h2ai-memory`, `orchestrator::session_journal` |
| **Context management** | Compaction with Lost-in-Middle mitigation; head+tail preservation; keyword injection | `context::compaction` |
| **Prompt construction** | `system_context` + `task` + TAO observation feedback; compacted before Phase 3 | `context` |
| **Output parsing** | `raw_output: String`; structured schema validation (planned) | `h2ai-types` |
| **State management** | `TaskStore` (write-through cache) + NATS JetStream (authoritative log); event-sourced CRDT; `SessionJournal` reconstructs state on replay | `state`, `h2ai-nats`, `orchestrator::session_journal` |
| **Error handling** | 4-class error taxonomy: transient / recoverable / user-fixable / unexpected | `orchestrator::engine` |
| **Guardrails / safety** | Typed `ConstraintDoc`/`ConstraintPredicate` corpus (Hard/Soft/Advisory); `constraint_error_cost = 1.0 − compliance`; ADR Auditor gate; J_eff gate; Multiplication Condition; NKey scoped NATS credentials | `h2ai-constraints`, `adapters`, `context` |
| **Verification loops** | Phase 3.5 scored LLM-as-judge; `filter_ratio` feedback to attribution; graceful fallback | `orchestrator::verification` |
| **Subagent orchestration** | Topology provisioning (Ensemble, HierarchicalTree, TeamSwarmHybrid); MAPE-K retry | `autonomic`, `orchestrator` |
| **Harness attribution** | USL quality decomposition: baseline + topology_gain + verification_gain + tao_gain | `orchestrator::attribution` |

**The unique differentiator:** Harness Attribution (row 12) does not exist in any competing framework. H2AI is the only system that can quantify — with USL math — how much each harness component contributed to output quality. See `math-apparatus.md §6` (Definitions 11–14, Propositions 6–8) for the formal treatment.

---

## API Contract — Error Model

**`POST /tasks`**
- `202 Accepted` + `task_id` — calibration valid, J_eff above threshold.
- `400 ContextUnderflowError` — J_eff below threshold; human must add explicit constraints.
- `400 InvalidParetoWeights` — `pareto_weights` do not sum to 1.0.
- `503 CalibrationRequiredError` — no calibration data cached; run `POST /calibrate` first.

**`GET /tasks/{task_id}/recover`**
- `200 OK` + status JSON — JetStream replay succeeded; `TaskState` upserted into live store.
- `404 TaskNotFound` — no events found for this `task_id` (task never started or ID is invalid).
- `400 InvalidRequest` — `task_id` is not a valid UUID.
- `500 InternalError` — NATS replay transport failure.

**`GET /tasks/{task_id}/events`** — SSE stream tailing `h2ai.tasks.{task_id}`. Closes on `MergeResolvedEvent` (success) or `TaskFailedEvent` (failure with full diagnostic: all pruned branches + c_i weights, topologies tried, τ ranges tried, which Multiplication Condition failed).

**`POST /tasks/{task_id}/merge`**
- `200 OK` — publishes `MergeResolvedEvent`, closes task.
- `404 TaskNotFound` — unknown task_id.
- `409 TaskAlreadyResolved` — task already closed.

**`POST /calibrate`** — `202 Accepted` + `calibration_id`. Progress streamed on `/calibrate/{id}/events`.

**`GET /ready`** — `503` if calibration data absent or NATS unreachable; Kubernetes removes pod from load balancer pool.

See [`../reference/api.md`](../reference/api.md) for full request/response schemas.

---

## Mathematical Foundation

The system is governed by the Universal Scalability Law extended with epistemic structure. Key quantities:

| Symbol | Formula | Meaning |
|--------|---------|---------|
| α | measured | Serial contention fraction |
| β_eff | β_base / CG_mean | Effective coherency cost per pair |
| N_max | round(sqrt((1−α) / β_eff)) | Throughput ceiling — USL Proposition 1 (Gunther 1993) |
| θ_coord | min(CG_mean − σ_CG, 0.3) | Minimum CG any Explorer pair must meet |
| J_eff | J(K_prompt, K_task_required) | Dark Knowledge Gap; gate at 0.4 |
| c_i | ∈ [0,1] | Role error cost; switches merge to BFT when max(c_i) > 0.85 |
| c_i_eff | c_i × 0.6^(t−1) | Effective error cost after t TAO iterations; BFT threshold re-evaluated post-loop |
| filter_ratio | N_valid / N | Fraction of proposals passing Verification Phase; lowers effective c_i |
| Q_total | Q_baseline + G_topo + G_verify + G_tao | Harness attribution decomposition (see math-apparatus.md §6) |

**How `AgentDescriptor.tools` changes the physics:**

A pure LLM agent (`tools: []`) contributes near-zero α during generation — it acquires no locks, produces no side effects, and terminates cleanly. Its c_i is low (0.1–0.3) because a wrong text output is free to discard.

A tool-using agent changes all three parameters. `Shell` commands serialize against shared state, directly raising α. `WebSearch` introduces retrieval nondeterminism — two agents querying the same source at different moments may diverge for environmental reasons, raising CG variance and therefore β_eff even if β_base is unchanged. High α and β_eff both lower N_max: the physics correctly responds by provisioning fewer explorers when the actual parallelism available is lower.

The c_i escalation is the most consequential effect: it can push `max(c_i) > 0.85` and force the BFT merge path, which is the only merge strategy that provides a correctness guarantee when conflicting outputs represent irreconcilable divergent states (e.g. two code generation agents that produced mutually incompatible file layouts).

**Simulation findings** (`python3 scripts/simulate_usl.py` → `scripts/output/`):

| Finding | Value | Plot |
|---------|-------|------|
| Shell agent (c_i=0.9) crosses BFT threshold | t=2, c_i_eff=0.540 | Plot 5 |
| Executor topology gain N=1→4 | +28pp | Plot 6 |
| Executor TAO gain (t=1→3, N=4) | +22pp | Plot 6 |
| Full harness lift: Executor | 50% → 100% (+50pp) | Plot 6 |
| Full harness lift: Shell agent | 10% → 100% (+90pp) | Plot 6 |
| Δ Q_total: TAO turns (1→4) | +21.9pp | Plot 7 |
| Δ Q_total: verify strictness (fr 1.0→0.0) | +21.9pp | Plot 7 |
| Δ Q_total: agents (1→N_max=5) | +29.0pp (front-loaded: N=1→2 gives +20pp) | Plot 7 |
| Marginal gain: N=4→5 vs TAO t=1→2 | +0.9pp vs +20pp (22× leverage) | Plot 7 |
| Proposition 8 monotonicity | ✓ verified all 3 parameters | Plot 7 |

**Operator implication:** Topology gain is front-loaded — the first additional agent gives +20pp, but N=4→5 gives only +0.9pp. Once an ensemble is formed (N≥3), the first additional TAO turn gives 22× more gain than adding another agent. The MAPE-K self-optimizer correctly prioritises TAO turns and verification strictness over N scaling within an established ensemble.

See [`math-apparatus.md`](math-apparatus.md) for formal definitions, propositions with proofs, calibration reference table, script cross-references, and citations — including §6 (Harness Physics Extensions: TAO error reduction, verification filter gain, harness attribution decomposition, 4-class error taxonomy) and §7 (Propositions 6–8: parallel verification speedup, TAO-USL convergence, attribution monotonicity). See [`../guides/agent-descriptor.md`](../guides/agent-descriptor.md) for tool-set-to-parameter tables and worked examples.
