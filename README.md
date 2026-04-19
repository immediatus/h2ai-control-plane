# H2AI Control Plane

[![CI](https://github.com/immediatus/h2ai-control-plane/actions/workflows/ci.yml/badge.svg)](https://github.com/immediatus/h2ai-control-plane/actions/workflows/ci.yml)
[![Theoretical Foundation](https://img.shields.io/badge/Framework-The_Coordination_Constant-blue)](https://e-mindset.space/blog/coordination-constant-usl-human-ai-teams/)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-orange)](LICENSE)
[![Language](https://img.shields.io/badge/Language-Rust-orange)](https://www.rust-lang.org/)

**H2AI Control Plane** is a distributed multi-agent orchestration runtime that treats LLM agent swarms as a control theory problem governed by the **Universal Scalability Law (USL)**.

Most agentic frameworks default to flat-mesh prompt-chaining: agents feed outputs to each other in a loop until consensus. This is mathematically wrong. When `N` agents must maintain mutual consistency, coordination cost grows as `κN(N-1)` — quadratic. At some `N`, that cost overtakes the value of adding agents. Throughput goes into retrograde. H2AI predicts the exact `N` at which this happens and shifts topology before it is crossed.

Reference implementation of the framework defined in **[One Equation Governs CPU Caches, Human Teams, and AI Agent Systems](https://e-mindset.space/blog/coordination-constant-usl-human-ai-teams/)**.

---

## Why It Exists

| Problem | Standard Approach | H2AI Approach |
|---|---|---|
| Hallucination amplification | Hope the model self-corrects | Auditor node (τ→0) mathematically blocks propagation |
| State lives in the model | LLM context window (lossy) | Sovereign CRDTs — orchestrator owns state, models are stateless `f(ctx, τ) → diff` |
| Safety is probabilistic | "Don't do X" in the prompt | Topological interlocks — invalid output cannot reach the human by graph construction |
| More agents = worse results | Keep adding until it breaks | MAPE-K loop computes N_max, shifts topology before retrograde |
| Tacit knowledge is invisible | Agents guess team constraints | Dark Knowledge Compiler — ADR corpus becomes hard Auditor gate |
| Human babysits every step | Constant correction loop | Merge Authority — human resolves a structured CRDT diff once, at the end |

---

## Quick Start

### Devcontainer (recommended)

Open in any devcontainer-compatible environment. NATS starts automatically as a sidecar.

```bash
git clone https://github.com/h2ai/control-plane.git
# Open in devcontainer — NATS and environment are pre-configured
```

### Local (Profile A)

```bash
git clone https://github.com/h2ai/control-plane.git
cd h2ai-control-plane/deploy/profile-a
docker compose up -d

# Seed your ADR corpus (your team's architectural decisions)
mkdir -p ../../adr
cp -r ../../docs/examples/ads-platform/adr/* ../../adr/

# Calibrate the adapter pool
curl -X POST http://localhost:8080/calibrate

# Submit your first task
curl -X POST http://localhost:8080/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Design a budget enforcement mechanism that prevents double-billing during server restarts",
    "pareto_weights": {"diversity": 0.5, "containment": 0.4, "throughput": 0.1},
    "explorers": {"count": 3, "tau_min": 0.2, "tau_max": 0.85}
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

kubectl create configmap adr-corpus --from-file=./adr/ -n h2ai

helm install h2ai h2ai/h2ai-control-plane \
  --namespace h2ai --create-namespace \
  --set ingress.enabled=true \
  --set ingress.hosts[0].host=h2ai.corp.example.com \
  --set serviceMonitor.enabled=true
```

---

## How It Works

### 1. Calibration — measure the physics before spawning anything

The calibration harness runs representative tasks through the adapter pool and measures empirically:

- `α` — the serial contention fraction (shared context lock time)
- `κ_base` — pairwise coherency cost (token exchange overhead between adapters)
- `CG(i,j)` — Common Ground between every Explorer pair

From these it derives `N_max = sqrt((1−α) / κ_eff)` — the exact agent count at which throughput peaks. Beyond `N_max`, every additional agent makes results worse. No task proceeds without this data.

### 2. Bootstrap — compile Dark Knowledge into explicit constraints

You submit a task manifest. The **Dark Knowledge Compiler** computes `J_eff = J(K_prompt, K_task_required)` — the Jaccard overlap between what you explicitly provided (manifest + ADR corpus) and what the task actually requires.

If `J_eff` is below threshold → synchronous `400 ContextUnderflowError`. The human must add constraints before proceeding. Nothing touches NATS.

If `J_eff` passes → an immutable `system_context` is compiled from your ADRs + manifest. Every agent — Explorer and Auditor alike — receives exactly this context and nothing else.

### 3. Provisioning — topology selected by physics, not guesswork

The autonomic loop reads `{α, κ_eff, ParetoWeights}` and selects:
- **Flat Mesh** — when `N ≤ N_max` and diversity weight dominates. No coordinator. `O(N²)` edges, but acceptable for small N.
- **Hierarchical Tree** — when `N > N_max` or containment weight dominates. One Swarm Coordinator + k sub-groups. Branching factor `k_opt = floor(N_max^flat)`. Coordination cost drops from `O(N²)` to `O(N)`.

Before spawning a single inference token, the **Multiplication Condition Gate** enforces all three conditions from Proposition 3: competence > 0.5, error decorrelation ρ < 0.9, Common Ground mean ≥ θ_coord. Fail any one → re-enter provisioning with adjusted parameters.

### 4. Generation — parallel, isolated, bounded

N Explorers run in a `tokio::task::JoinSet` wrapped in `tokio::time::timeout`. Each Explorer calls `IComputeAdapter::execute()` with its assigned `τ` value and terminates. No Explorer reads another Explorer's output. Coordination cost during generation is structurally zero. Every Explorer gets a guaranteed terminal state — `ProposalEvent` on success, `ProposalFailedEvent` on crash/OOM/timeout. The stream always closes with `GenerationPhaseCompletedEvent`.

### 5. Auditor Gate — reactive, never idle

The Auditor spins up on `TopologyProvisionedEvent` — before generation starts. It validates proposals as they arrive against the compiled `system_context`. Every ADR constraint in your corpus is a potential rejection. Rejected proposals become `BranchPrunedEvent` tombstones: permanently preserved in the log with rejection reason and constraint cost (`c_i`), visible in the Merge Authority UI.

If all proposals are pruned → `ZeroSurvivalEvent` → the MAPE-K loop adjusts `{N, τ}` and retries. Bounded by `max_retries`. Exhaustion → `TaskFailedEvent` with full diagnostic payload.

### 6. Merge — O(1) human decision

Surviving proposals are joined into a CRDT semilattice (or BFT consensus if `max(c_i) > 0.85`). The **Merge Authority UI** presents:

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

| System | α | κ_eff | N_max |
|---|---|---|---|
| CPU cache coherency | 0.02 | 0.0003 | ~57 |
| Human engineering team | 0.10 | 0.0083 | ~10 |
| AI agents (same model) | 0.15 | 0.025 | ~6 |
| AI agents (diverse backends) | 0.12 | 0.018 | ~7 |

Reference values for AI agents: **α ≈ 0.10–0.15, κ_base ≈ 0.01, κ_eff ≈ 0.018–0.025, N_max ≈ 6–7**.

---

## The 14-Event Vocabulary

All state is immutable event log entries on NATS JetStream. Crash recovery = replay from offset 0.

```
CalibrationCompletedEvent          → α, κ_base, CG samples, θ_coord locked
TaskBootstrappedEvent              → J_eff gate passed, system_context locked
TopologyProvisionedEvent           → DAG shape, τ values, RoleErrorCosts, MergeStrategy
MultiplicationConditionFailedEvent → which of 3 conditions failed, re-entering Phase 2
ProposalEvent                      → Explorer output appended, agent terminates
ProposalFailedEvent                → Explorer crashed/OOM/timeout, terminal state guaranteed
GenerationPhaseCompletedEvent      → JoinSet drained, stream closed
ValidationEvent                    → Auditor: proposal passed
BranchPrunedEvent                  → Auditor: proposal rejected (reason + c_i weight)
ZeroSurvivalEvent                  → all proposals pruned, autonomic retry fires
ConsensusRequiredEvent             → max(c_i) > 0.85, switching CRDT → BFT
SemilatticeCompiledEvent           → merge ready, MergeStrategy recorded
MergeResolvedEvent                 → human O(1) decision, task closed
TaskFailedEvent                    → retries exhausted, full diagnostic payload
```

---

## Repository Layout

```
h2ai-control-plane/
├── Dockerfile                      # multi-stage: builder (rust+clang) → runtime (debian-slim)
├── crates/
│   ├── h2ai-types/                 # Pure types boundary — zero I/O deps
│   │                               # All 14 event structs, IComputeAdapter trait,
│   │                               # USL physics types, CoherencyCoefficients, MergeStrategy
│   ├── orchestrator/               # DAG builder + Pareto topology router
│   ├── autonomic/                  # MAPE-K loop + calibration harness + N_max calculator
│   ├── state/                      # CRDT semilattice + NATS JetStream I/O
│   ├── context/                    # Dark Knowledge Compiler + Jaccard + J_eff measurement
│   ├── adapters/                   # IComputeAdapter: llama.cpp FFI + cloud HTTP
│   └── api/                        # axum REST gateway + Merge Authority web UI
├── nats/
│   ├── dev.conf                    # single-node JetStream config (Profile A)
│   └── cluster.conf                # 3-node cluster config (Profile B/C)
├── deploy/
│   ├── profile-a/docker-compose.yml      # h2ai + NATS, single workstation
│   ├── profile-b/docker-compose.yml      # 3-node NATS + 2× h2ai + nginx + observability
│   ├── profile-c/                         # raw Kubernetes manifests
│   └── helm/h2ai-control-plane/          # Helm chart for enterprise distribution
├── .devcontainer/                  # devcontainer: Rust toolchain + NATS sidecar
├── .github/workflows/
│   ├── ci.yml                      # fmt → clippy -D warnings → nextest → docker → helm lint
│   └── release.yml                 # image → ghcr.io, Helm chart → GitHub Pages, binary release
└── docs/
    ├── guides/                     # Getting started, ADR corpus, adapter development
    ├── reference/                  # API reference, configuration reference
    ├── operations/                 # Operations guide, troubleshooting
    ├── architecture/               # Design spec, USL physics, runtime phases, crate boundaries
    └── examples/
        └── ads-platform/           # Reference ADR corpus + integration test task manifests
            ├── adr/                # 7 ADRs derived from "Architecting Real-Time Ads Platform"
            └── tasks/              # 3 task manifests with expected Auditor outcomes
```

**Dependency rule:** `h2ai-types` has zero external I/O dependencies. Every other crate depends on it. `state` is the only crate that talks to NATS. `api` is the only crate that talks to HTTP. Nothing imports `api`.

**Compute isolation:** llama.cpp FFI calls run on Tokio's bounded blocking thread pool (`spawn_blocking`, `max_blocking_threads` explicitly set). The async worker pool — NATS consumer, MAPE-K loop, axum HTTP — is never starved by inference work.

---

## Technology Stack

| Layer | Choice | Why |
|---|---|---|
| Language | Rust + Tokio | Compiler-verified CRDT state, zero-cost FFI to llama.cpp, no GC jitter in κ_base |
| Event log | NATS JetStream | Single static binary (MB of RAM), Tokio-native `async-nats`, clusters natively |
| State model | Event-sourced CRDT | α→0 during generation (no locks), full provenance chain, crash recovery = replay |
| Local compute | llama.cpp FFI | Zero-cost, 128GB RAM dedicated to weights |
| HTTP | axum | Tokio-native, same async runtime as orchestrator |
| Tracing | `tracing` + OpenTelemetry | task_id as root span, DAG execution visible in Jaeger/Grafana Tempo |
| Metrics | Prometheus `/metrics` | 20 gauges: κ_eff, α, N_max, θ_coord, J_eff, VRAM, c_i per role, adapter latency |

---

## Deployment

The system is **C-first**: the distributed cluster is the architectural foundation, not a future upgrade. Profile A is Profile C running on one machine.

| Profile | Target | Stack |
|---|---|---|
| **A — Local dev** | Single workstation (128GB RAM) | Static binary + nats-server, no container runtime required |
| **B — Team node** | Dedicated server | 3-node NATS cluster + 2× h2ai + nginx + Prometheus + Grafana + Jaeger |
| **C — Kubernetes** | Multi-region cluster | Helm chart, NATS StatefulSet, h2ai Deployment + HPA, ServiceMonitor |

---

## ADR Corpus and Integration Examples

The **Dark Knowledge Compiler** reads your team's Architecture Decision Records and uses them as hard Auditor constraints. Every `## Constraints` bullet in an ADR becomes a condition the Auditor enforces — proposals that violate ADR constraints are tombstoned before they reach the human.

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
| [Getting Started](docs/guides/getting-started.md) | First task end-to-end — Profile A local, Profile B team node, Profile C Kubernetes |
| [ADR Corpus Guide](docs/guides/adr-corpus.md) | What ADRs are, how the compiler reads them, diagnosing low J_eff, minimum viable corpus |
| [Adapter Development](docs/guides/adapters.md) | Implementing `IComputeAdapter` for custom compute backends, testing, registration |
| [Theory to Implementation](docs/guides/theory-to-implementation.md) | Topology selection protocol, 7-topology catalog with Pareto scores, team-swarm configuration, worked example |

### Reference

| Document | Contents |
|---|---|
| [API Reference](docs/reference/api.md) | All REST endpoints, SSE event stream, complete JSON schemas for all 14 events, error codes |
| [Configuration Reference](docs/reference/configuration.md) | All environment variables, `adapters.toml` format, 20 Prometheus metrics, Helm values |

### Architecture

| Document | Contents |
|---|---|
| [Design Specification](docs/architecture/00-design-specification.md) | Full system design — all architectural decisions with rationale |
| [USL Physics](docs/architecture/01-usl-physics.md) | Mathematical foundation — USL, CG, κ_eff, N_max, Multiplication Condition, Dark Knowledge Gap |
| [Runtime Phases](docs/architecture/02-runtime-phases.md) | 6-phase execution flow, 14-event vocabulary, 10 structural guarantees |
| [Crate Boundaries](docs/architecture/03-crate-boundaries.md) | Workspace layout, dependency rules, Tokio thread pool isolation |
| [Deployment](docs/architecture/04-deployment.md) | Three profiles, NATS clustering, Kubernetes topology, observability |
| [Mathematics Apparatus](docs/architecture/05-math-apparatus.md) | All 10 definitions, 5 propositions with proofs, calibration reference table, safety constraints, event vocabulary |

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
