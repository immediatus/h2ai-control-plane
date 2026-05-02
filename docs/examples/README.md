# Examples

This directory contains reference projects that demonstrate H2AI Control Plane end-to-end. Each project serves two purposes simultaneously:

1. **Documentation** — concrete, realistic constraint documents and task manifests you can study and adapt for your own team.
2. **Integration test fixtures** — the task manifests are the input corpus for the system-level integration test suite. They verify that the J_eff gate, Auditor pruning, and MAPE-K retry all behave correctly on realistic inputs.

---

## Why constraints matter for H2AI

Every engineering team carries vast tacit knowledge: "we don't store sessions because of a compliance requirement," "payment retries happen async because synchronous calls caused duplicate charges," "service A must not query service B's database — that's what caused the Q3 incident." This knowledge lives in engineers' heads, in Slack threads, in post-mortems — not in any artifact an AI agent can read.

H2AI calls this the **Dark Knowledge Gap** (`J_eff`). When you submit a task, the Dark Knowledge Compiler measures the Jaccard overlap between what you explicitly provided and what the task actually requires. When the gap is large, agents hallucinate architectural decisions — and the Auditor rejects their proposals.

Constraint documents are the mechanism by which Dark Knowledge becomes explicit context. A well-written `## Constraints` section teaches the Auditor exactly what to reject.

### The J_eff effect in practice

**Without a constraint corpus:** A task about budget enforcement returns `ContextUnderflowError` — `J_eff = 0.12`, well below the 0.4 threshold. The system refuses to proceed because the constraint space is too underspecified.

**With a constraint corpus:** The same task returns `202 Accepted` — `J_eff = 0.71`. Three Explorers generate proposals. One proposes reading budget from a cache (faster, but stale). The Auditor catches it — "CONSTRAINT-004: budget checks must read from Redis atomic counters, never from cache" — and tombstones that branch. Two valid proposals reach the Merge Authority.

The constraint corpus is not overhead. It is the input that makes the Auditor work.

---

## Writing effective constraint documents

The compiler extracts three things from each constraint document:

1. **Prohibition statements** — phrases like "must not", "is forbidden", "is prohibited", "never"
2. **Requirement statements** — phrases like "must", "is required", "always", "shall"
3. **Scope identifiers** — service names, component names, compliance references, pattern names

A decision stated only as "we use JWT" gives the Auditor nothing to enforce. A decision stated as "Services must not store session tokens in any database, cache, or in-process store" gives the Auditor a specific, checkable constraint.

**The `## Constraints` section is the most important part of every document.** It should contain a bullet list of hard rules in imperative language. Every bullet becomes a potential Auditor rejection reason.

See the [Constraint Corpus Guide](../architecture/constraint-corpus.md) for the full typed format with explicit severity, predicate, and remediation sections.

---

## Reference projects

### ads-platform — Real-Time Ads Platform

Derived from the blog series *"Architecting Real-Time Ads Platform"* by Yuriy Polyulya. Seven constraint documents covering stateless services, gRPC/REST protocol split, adaptive RTB timeouts, budget pacing with idempotency, dual-ledger financial compliance, Java/ZGC runtime, and tiered data consistency.

**Why this project:** The ads platform decisions have sharp, verifiable constraint boundaries. An Explorer that proposes reading budget from cache, raising the global RTB timeout, or using G1GC instead of ZGC is unambiguously wrong given the constraints — the Auditor has high-confidence rejection criteria.

[View project →](ads-platform/README.md)

---

## Running as integration tests

```bash
# Start the stack
cd deploy/local && docker compose up -d

# Load the constraint corpus
export CORPUS_PATH=docs/examples/ads-platform/constraints
docker compose exec h2ai \
  sh -c "cp -r /workspace/$CORPUS_PATH/* /constraints/ && kill -HUP 1"

# Run calibration
curl -X POST http://localhost:8080/calibrate
# Wait for CalibrationCompletedEvent...

# Run the integration test suite (reads task manifests from docs/examples/)
cargo nextest run --test integration -- --test-threads=1
```

The integration test harness (`tests/integration/`) reads the `_expected` block from each task manifest JSON and asserts:
- `j_eff` is at or above `j_eff_min`
- Number of valid proposals is at or above `valid_proposals_min`
- Each entry in `should_prune` produces a `BranchPrunedEvent` citing the specified constraint
- The task reaches `SemilatticeCompiledEvent` (not `TaskFailedEvent`)

---

## Contributing a new example

A good example project has:
- At least 4 constraint documents with a strong `## Constraints` section
- At least 2 task manifests with distinct `should_prune` entries that cite different constraints
- At least 1 task manifest that exercises the MAPE-K retry path (`valid_proposals_min: 0` on first attempt, succeeds after retry)
- An `_expected` block in every task manifest so the integration test harness can assert outcomes

To add a project: create `docs/examples/{project-name}/` with the structure shown in `ads-platform/`, add it to this README, and add a test case in `tests/integration/`.
