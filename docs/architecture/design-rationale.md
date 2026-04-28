# H2AI Control Plane — Design Rationale

This document explains *why* each major design decision was made — the problem each choice solves, the alternative that was rejected, and the honest tradeoffs. It is a complement to the formal architecture docs, which describe *what* the system does. Read this when you want to understand the reasoning behind a design, not just the specification.

---

## Why NATS JetStream, Not Kafka or an In-Process Queue

**Problem:** Multi-agent orchestration state must survive crashes. When 5 agents are mid-execution and the orchestrator process restarts, the system must resume — not retry from the beginning. Retrying costs inference tokens, time, and idempotency guarantees for tool-using agents that may have already written files or called external APIs.

**Decision:** Use NATS JetStream as an immutable, ordered event log. Every state transition (task bootstrapped, agent dispatched, proposal received, branch pruned) is appended as an event. Crash recovery is replay from the last committed offset.

**Why not Kafka?** Kafka requires a JVM, a ZooKeeper/KRaft cluster, and 1–4GB of RAM to operate. In Local Plan (a single developer workstation running llama.cpp with 128GB RAM dedicated to model weights), Kafka would consume RAM that should be serving inference. NATS runs as a single static binary in megabytes. NATS JetStream provides both the event log and the KV store (for calibration cache) from one binary with identical semantics across local, server, and cloud deployments.

**Why not in-process queues (tokio channels, broadcast)?** In-process queues do not survive process restarts. They also prevent multi-node deployment — the event log is the only shared state between orchestrator instances in Cloud Plan.

**The deeper point:** NATS JetStream is not chosen as a message broker. It is chosen as the **shared immutable information ground** — the substrate that all participants (orchestrator, agents, audit system, human UI) read from to reconstruct system state at any point in time. This is not a messaging optimization; it is a state model decision.

---

## Why the Dark Knowledge Compiler Runs Before Generation

**Problem:** LLM generation is expensive and irreversible for tool-using agents. If the task manifest does not cover the constraint space — if the agent is about to propose something that violates an ADR it has never seen — every generated token is wasted, and tool-using agents may have already written files or called external services.

**Decision:** The `compile()` step measures `J_eff = j_positive × (1 − contamination)` before any inference token is generated. If J_eff falls below the configured threshold, the task is rejected with `ContextUnderflowError` synchronously. Nothing touches NATS. The human must provide more context before the system proceeds.

**What J_eff measures:** Semantic overlap between what the task manifest contains and what the constraint corpus requires. Low J_eff means agents would be making decisions in a domain they have not been grounded in — the semantic equivalent of split brain before generation even starts. High J_eff means agents share common ground with the constraint corpus and each other before they begin.

**The gate is not about safety in isolation.** It is about preventing the most expensive form of failure: agents that generate correct-looking but constraint-violating proposals that pass all internal checks and reach the human, who then discovers the ADR violation after reading the output.

**Honest limitation:** J_eff measured by token Jaccard (without an embedding model) catches vocabulary coverage, not semantic coverage. "Payment throttling" and "budget pacing" score near zero on token Jaccard despite being semantically equivalent. The `EmbeddingModel` path in `semantic_jaccard` closes this gap when an embedding model is available.

---

## Why Coordination Cost Uses USL, Not a Simple Max-N Parameter

**Problem:** Every multi-agent framework lets you set `max_agents`. The question is where that number comes from. Setting it to 3 because "3 seems reasonable" means it will be wrong for tasks with high agent divergence, wrong for tasks with low divergence, and wrong when the adapter pool changes.

**Decision:** Derive N_max from the measured coordination overhead using the Universal Scalability Law — `N_max = √((1−α)/β_eff)`. α captures serial bottleneck fraction; β_eff captures pairwise reconciliation cost as a function of agent divergence (CG_mean).

**Why USL, not a different model?** USL provides a closed-form expression for the throughput ceiling of a coordination-dependent system. It has been empirically validated across CPU architectures, database connection pools, and (by structural analogy) human engineering teams. Its two-parameter form (α, β) maps cleanly to the two types of coordination cost in agent orchestration: serial bottlenecks that exist regardless of N (α) and pairwise synchronization costs that grow with N (β).

**The key reframe:** USL was designed for shared-state distributed systems where α and β have physical meanings (queueing for a lock, cache-line coherency protocol). For agent orchestration, α and β have *reasoning-space* meanings: α is the serial fraction inherent to planning and synthesis, β is the pairwise cost of reconciling agents that have diverged. The mathematical structure is identical; the physical substrate is different.

**What the calibration actually measures:** The current two-phase timing harness measures wall-clock time for N concurrent API calls. This captures I/O scheduling overhead — a real but incomplete proxy for β. The ideal measurement is merge phase timing: how long does `MergeEngine` take as a function of N, divided by N(N−1)/2 pairs? The NATS event log records the timestamps to compute this; a future calibration harness will use them. The current measurement is conservative (network latency is more predictable than task-domain reconciliation cost) and corrects dynamically through the CG_mean coupling.

**Honest limitation:** With M < 3 adapters, the USL fit degenerates and the system falls back to configured default parameters. Most small deployments run on hardcoded conservative values, not empirically derived ones. This is safe (defaults are calibrated to produce N_max ≈ 4–6 for typical LLM ensembles) but should be understood: the "physics-derived N_max" is only fully physics-derived when M ≥ 3 adapters are available for calibration.

---

## Why Scoped NKeys Per Task, Not Long-Lived API Keys

**Problem:** Edge agents (tool-using containers with Shell, CodeExecution, FileSystem access) need NATS credentials to publish their results and telemetry. Long-lived credentials in containers create two risks: a compromised container can read other agents' payloads, and credentials survive task completion and can be reused.

**Decision:** Each task gets a fresh NATS NKey scoped to exactly the subjects that task's agents need to publish to: `h2ai.telemetry.{agent_id}`, `audit.events.{agent_id}`, `h2ai.results.{task_id}`. The NKey is provisioned at dispatch time and expires when the task closes. The scoping is enforced at the NATS server level — not by application code that could be bypassed.

**What this prevents:** A compromised executor container cannot read other agents' task payloads. It cannot write to the orchestration event bus. It cannot retain credentials to later impersonate another agent. The blast radius of any single agent compromise is bounded to the current task.

**Why NATS NKeys specifically?** NKeys are Ed25519 key pairs with subject-level permission scoping built into the NATS authentication model. No custom authorization middleware is needed. The `allowed_publish` set is sized to match the agent's tool set — a pure LLM agent gets fewer publish subjects than a Shell agent.

---

## Why Rust, Not Python or Go

**Problem:** The orchestrator must run inference-aware workloads (calibration timing, CRDT merge operations) on the same machine as llama.cpp model inference, without GC pauses or runtime overhead interfering with timing measurements.

**Decision:** Rust with Tokio. CRDT state is compiler-verified (the borrow checker ensures no concurrent mutation of semilattice state). Zero-cost FFI to llama.cpp means the orchestrator and the inference engine share the same process without marshaling overhead. No garbage collector means calibration timing measurements are not contaminated by GC pauses that would produce misleading α and β estimates.

**Why not Go?** Go's GC is low-latency but not zero-latency. In Local Plan with 128GB RAM dedicated to model weights, GC pressure from orchestrator allocations could produce timing spikes that corrupt calibration measurements. Go also lacks the FFI ergonomics for llama.cpp integration — the CGO boundary adds overhead that matters when timing microsecond-scale operations.

**Why not Python?** Python is appropriate for scripting the calibration validation (`validate_ensemble_theory.py`), not for the production runtime. The GIL prevents true parallel execution of calibration phases; asyncio adds abstraction overhead that obscures timing.

---

## Why Event-Sourced CRDTs, Not a Database

**Problem:** Multi-agent merge operations require an append-only record of every proposal, every rejection, and every merge decision. After the task closes, this record must be queryable for audit, reproducibility, and debugging. During execution, it must support concurrent writes from N agents without locking.

**Decision:** CRDT semilattice (append-only LUB join) over NATS JetStream. Each agent appends its proposal as an event; the merge engine reads all proposals and applies the semilattice join to produce the final state. No lock is required during agent generation — agents write to their own proposal subjects; the merge engine reads all of them after the `GenerationPhaseCompletedEvent` closes the phase.

**Why CRDT, not optimistic locking?** Optimistic locking requires a shared version counter that must be checked and incremented atomically. Under NATS JetStream, this would require a sequence number handshake for every write — defeating the goal of zero-coordination during generation. CRDT join semantics are coordination-free by definition: any order of append operations produces the same merged state.

**Why not PostgreSQL/Redis?** A relational database serializes writes behind a transaction lock — α=1 during generation. Redis requires a network round trip per write. NATS JetStream is co-located with the orchestrator in Local Plan, provides the same event-sourcing semantics as a dedicated event store, and adds zero external process dependencies.

---

## Why the Condorcet and USL Models Are Used Together

**Problem:** Neither model alone answers "what is the right ensemble size?"

- USL alone gives a coordination *ceiling* (N_max) but not a quality *target*. N=2 is always below N_max for reasonable parameters, but N=2 might not produce meaningfully better results than N=1.
- CJT alone gives a quality *gain* estimate but no cost constraint. CJT would suggest the largest N that fits in the token budget.

**Decision:** Use both. CJT gives `N_optimal = argmax marginal Condorcet gain per agent`. USL gives `N_max = coordination ceiling`. The effective ensemble size is `min(N_optimal, N_max)`. This ensures:
1. The ensemble is large enough to gain from the Condorcet effect (N ≥ N_optimal).
2. The ensemble is small enough not to degrade from coordination overhead (N ≤ N_max).

When N_optimal > N_max (coordination cost is high and quality benefit is also high), the system runs at N_max and compensates with topology: Hierarchical Tree reduces coordination cost from O(N²) to O(N) by grouping agents, effectively raising N_max for the sub-groups while keeping overall N bounded.

---

## What Is Not Yet Solved — Honest Gaps

**TalagrandDiagnostic is computed but not acted on.** The rank histogram diagnostic is implemented and returns calibration status (`WellCalibrated`, `OverConfident`, `UnderDispersed`, `UnderConfident`). No engine path currently adjusts τ spread or triggers re-calibration in response to these states. The diagnostic is informational until this feedback loop is wired.

**SelfOptimizer runs on every success but its suggestion is discarded on success.** The optimizer is applied only in the retry loop. On a successful first-pass run, its output is computed and dropped. The optimizer's value (suggesting τ/N adjustments to improve quality on next run) is not captured.

**CG_mean from token Jaccard underestimates semantic divergence in reasoning-heavy tasks.** Two adapters can produce different-vocabulary correct answers (low CG, both right) or similar-vocabulary wrong answers (high CG, both wrong). The `EmbeddingModel` path in `hybrid_search` closes this for retrieval; the same approach should be applied to CG measurement during calibration.

**β₀ calibration from merge phase timing is not yet implemented.** The current two-phase timing harness is a practical starting point. The correct measurement is merge phase cost as a function of N. Instrumentation exists (NATS event timestamps); the calibration harness needs to be updated to read merge spans rather than parallel execution spans.
