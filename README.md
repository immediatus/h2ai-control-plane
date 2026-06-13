# H2AI Control Plane

[![CI](https://github.com/immediatus/h2ai-control-plane/actions/workflows/ci.yml/badge.svg)](https://github.com/immediatus/h2ai-control-plane/actions/workflows/ci.yml)
[![Theoretical Foundation](https://img.shields.io/badge/Framework-The_Coordination_Constant-blue)](https://e-mindset.space/blog/coordination-constant-usl-human-ai-teams/)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-orange)](LICENSE)
[![Language](https://img.shields.io/badge/Language-Rust-orange)](https://www.rust-lang.org/)

> **Distributed systems consensus applied to stochastic text generators — a Rust multi-agent orchestration runtime that replaces "spawn-N-agents-and-hope" with measured, physics-grounded quality guarantees.**

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

Each row is a failure mode H2AI addresses structurally rather than probabilistically.

| Problem | H2AI Mechanism |
|---|---|
| **Hallucination amplification** | Auditor node (τ→0) blocks propagation before merge — mathematically, not by prompt |
| **More agents = worse results** | USL-bounded N_max: MAPE-K shifts topology before quality falls off the cliff |
| **Retry loop repeats same failure** | Progressive Verifier Feedback + Epistemic Leader: anchors on best prior proposal; Socratic diagnostics prevent repeated failure modes |
| **Constraint dispersion** (proposals satisfy different checks, none satisfies all) | Constraint-Informed Synthesis + Sequential Constraint Grafting: greedy set-cover picks orthogonal partials; iterative grafting loop monotonically improves score |
| **O(n³)+ tasks hallucinate regardless of retry depth** | Complexity Ceiling: `ComplexityProbe` routes pre-loop; AgentDropout reduces N on structurally-failing retries; BEYOND_BUDGET decomposition for sub-claims |
| **Ambiguous constraints produce random verifier votes** | Constraint Coherence: `CoherenceProbe` detects noise; `SpecRepairAdvisor` rewrites and hot-reloads the corpus without task restart |
| **Fabricated APIs pass constraint checks** | SRANI: Correlated Fabrication Index gates proposals; grounding escalation (spec anchor → researcher → web search) |
| **Retry loop fires even when first wave easily passes** | Tiered Early Exit: N escalates linearly wave-to-wave; exits immediately when quorum of proposals reach acceptance score — no full retry budget spent on simple tasks |
| **Generation budget depletes silently; converged proposals keep triggering new waves** | Cost Guard + Convergence Gate: token budget enforced per task with hint injection; semantic convergence (θ > 0.87 mean pairwise cosine) triggers early acceptance before budget exhaustion |
| **Calibration goes stale as model behavior changes** | Calibration Drift Detection: DDM fast-layer + BOCPD changepoint detectors emit `CalibrationDriftWarning`; adaptive recalibration gate prevents stale N_max from sizing the ensemble incorrectly |
| **Tacit knowledge is invisible** | Dark Knowledge Compiler: typed `ConstraintDoc` predicates become hard Auditor gates |

→ **Full problem-solution map** (24 mechanisms, each with implementation detail): [`docs/architecture/architecture.md § Problem-Solution Map`](docs/architecture/architecture.md#problem-solution-map)

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

# Check task status via SSE stream
curl -N http://localhost:8080/v1/default/tasks/{task_id}/events
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

## Safety Profiles

`[safety] profile` in your `h2ai.toml` selects a named tier. Four profiles are available:

| Profile | Use case | Shadow auditor | Strict | Krum f | Family diversity |
|---|---|---|---|---|---|
| `development` | Local dev, single adapter, devcontainer | off | false | 0 | `single_family_ok` |
| `production` | Staging and production | **on** | **true** | 1 | `require_diverse` |
| `strict` | Regulated / compliance environments | **on** | **true** | 2 | `require_diverse` |
| `custom` | E2E tests, research, partial overrides | manual | manual | manual | manual |

**Choose `development`** when you have one LLM endpoint or are iterating locally. All safety gates are relaxed — a single adapter family is fine, and the shadow auditor is off.

**Choose `production`** for any deployed environment with two or more adapter families. The shadow auditor runs in strict mode (audit failures abort the task, not just warn). Krum tolerates one Byzantine-equivalent outlier. The verifier and explorer pool must be from different model families — `VerifierExplorerFamilyConflict` fails the task immediately if they are not, before spending any tokens.

**Choose `strict`** for finance, healthcare, or any domain where a wrong but confident output has real-world consequences. Krum fault tolerance rises to `f = 2` (requires ≥7 explorer slots to be meaningful). Cross-family diversity is required.

**Choose `custom`** when you need a non-standard combination — for example, shadow auditor enabled but non-blocking (`strict = false`) in an E2E research scenario:

```toml
[safety]
profile = "custom"

[safety.shadow_auditor]
enabled = true
strict  = false
```

**Shadow auditor** runs a second independent audit pass on each wave result using a dedicated adapter (`adapter_key = "shadow_auditor"` in `[[adapter_profiles]]`). In strict mode it blocks the result; in non-strict mode it emits `ShadowAuditorResultEvent` for observability only.

Full field reference: `docs/architecture/reference.md § Safety Profiles`.

---

## The Epistemological Architecture

H2AI is an **epistemic control plane**. Its job is not to run LLM inference — it is to coordinate the acquisition, validation, and grounding of knowledge about a problem. The output of a successful task is not a string; it is a belief that has survived four nested epistemological tests: **TAO** (completeness within one agent), **MAPE-K** (coherence across the committee), **Calibration** (accuracy of the system's self-model), and **Oracle / Grounding** (correspondence to external reality).

The stopping criteria are epistemic, not mechanical. The system does not stop because it hit a retry limit — it stops because it has acquired enough knowledge. This is the architectural difference between H2AI and a pipeline with retries.

→ **Full epistemological architecture** (loop interactions, meta-accuracy, grounding model): [`docs/architecture/architecture.md § The Epistemological Architecture`](docs/architecture/architecture.md#the-epistemological-architecture)

---

## How It Works

Six phases run in sequence, all event-sourced to NATS JetStream:

1. **Calibration** — measures α (serial bottleneck), β₀ (pairwise reconciliation cost), and CG (Common Ground) from the adapter pool, yielding `N_max = √((1−α)/β_eff)`. No task starts without this.
2. **Bootstrap** — the Dark Knowledge Compiler assembles an immutable `system_context` from your constraint corpus. Every `ConstraintDoc` becomes a hard Auditor gate or a weighted compliance score.
3. **Provisioning** — the MAPE-K controller selects topology (Ensemble / Hierarchical Tree / Team-Swarm Hybrid) from physics, then enforces the Multiplication Condition Gate before spawning any inference token.
4. **Generation** — N Explorers run in parallel (`tokio::task::JoinSet`). No Explorer reads another's output. Each is a stateless edge agent with a scoped NKey that expires when the task closes. `TaoAgent` runs Thought→Action→Observation up to `agent_max_tool_iterations` turns inside each.
5. **MAPE-K gate** — the Auditor validates proposals as they arrive. Zero survivors triggers the three-layer retry engine: 12 per-wave phase modules, typed `MapeKDecision`, Epistemic Leader cross-wave diagnostics.
6. **Merge** — Layer 1: CRDT semilattice on constraint fingerprints (deterministic, never touches LLM text). Layer 2: OSP regime classification → two-pass critique+synthesis LLM. When the HITL gate fires, `POST /{tenant_id}/tasks/{id}/approve` parks the output for human review; one decision closes the task.

→ **Full How It Works** (topology trade-offs, TAO loop internals, security invariants, USL quantities by tool set): [`docs/architecture/architecture.md § How It Works`](docs/architecture/architecture.md#how-it-works)

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

All state is immutable event log entries on NATS JetStream. Crash recovery = replay from offset 0. Events cover core orchestration (`h2ai.tasks.{task_id}`), compound task scheduling, and edge agent telemetry (`h2ai.telemetry.*`).

→ **Full event vocabulary** (all 30+ events with fields and subjects): [`docs/architecture/reference.md § Event Vocabulary`](docs/architecture/reference.md#2-event-vocabulary)

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
| `h2ai-autonomic` | Calibration harness, USL solver (α/β/CG → N_max), eigenvalue computation (spawn_blocking); OSP merger (`AuditChannelBuilder`, `RetryAccumulator`, `OspRegime` dispatch) |
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
| [Research State](docs/architecture/research-state.md) | Project thesis, mathematical defensibility, open research questions, empirical validation status |

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
