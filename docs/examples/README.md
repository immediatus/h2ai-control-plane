# Examples

This directory contains reference projects that demonstrate H2AI Control Plane end-to-end. Each project serves two purposes simultaneously:

1. **Documentation** — concrete, realistic constraint documents and task manifests you can study and adapt for your own team.
2. **Integration test fixtures** — the task manifests are the input corpus for the system-level integration test suite. They verify that the J_eff gate, Auditor pruning, and MAPE-K retry all behave correctly on realistic inputs.

---

## Why constraints matter for H2AI

Every engineering team carries vast tacit knowledge: "we don't store sessions because of a compliance requirement," "payment retries happen async because synchronous calls caused duplicate charges," "service A must not query service B's database — that's what caused the Q3 incident." This knowledge lives in engineers' heads, in Slack threads, in post-mortems — not in any artifact an AI agent can read.

H2AI calls this the **Dark Knowledge Gap** (`J_eff`). When you submit a task, the Dark Knowledge Compiler measures the vocabulary overlap between what you explicitly provided and what the task actually requires. When the gap is large, agents hallucinate architectural decisions — and the Auditor rejects their proposals.

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

The full typed format uses explicit `severity`, `predicate`, and `remediation` fields — see the constraint documents in `ads-platform/constraints/` for worked examples.

---

## Reference projects

### ads-platform — Real-Time Ads Platform

Derived from the blog series *"Architecting Real-Time Ads Platform"* by Yuriy Polyulya. Seven constraint documents covering stateless services, gRPC/REST protocol split, adaptive RTB timeouts, budget pacing with idempotency, dual-ledger financial compliance, Java/ZGC runtime, and tiered data consistency.

**Why this project:** The ads platform decisions have sharp, verifiable constraint boundaries. An Explorer that proposes reading budget from cache, raising the global RTB timeout, or using G1GC instead of ZGC is unambiguously wrong given the constraints — the Auditor has high-confidence rejection criteria.

[View project →](ads-platform/README.md)

---

## Running the examples manually

```bash
# Start the stack (or use the devcontainer — NATS starts automatically)
cd deploy/local && docker compose up -d

# Copy the constraint corpus into the configured corpus_path
cp -r docs/examples/ads-platform/constraints/* /path/to/constraints/

# Run calibration (wait for CalibrationCompletedEvent in the SSE stream)
CAL=$(curl -s -X POST http://localhost:8080/calibrate | jq -r .calibration_id)
curl -sN "http://localhost:8080/calibrate/$CAL/events"

# Submit each task and observe the SSE stream
for task in docs/examples/ads-platform/tasks/*.json; do
  echo "=== Submitting $(basename $task) ==="
  RESP=$(curl -s -X POST http://localhost:8080/tasks \
    -H "Content-Type: application/json" -d @"$task")
  TASK_ID=$(echo "$RESP" | jq -r .task_id)
  echo "task_id: $TASK_ID"
  curl -sN "http://localhost:8080/tasks/$TASK_ID/events" | head -100
done
```

The `_expected` block in each manifest documents the observable system behaviour:
- `j_eff_min` — minimum Dark Knowledge coverage fraction expected
- `valid_proposals_min` — minimum proposals that should survive the Auditor gate
- `should_prune` — proposals the Auditor should reject, and which constraint they violate
- `should_pass` — a characterisation of a valid surviving proposal

These are human-readable contracts. A future integration test harness will assert them against live SSE streams.

---

## Contributing a new example

A good example project has:
- At least 4 constraint documents with a strong `## Constraints` section
- At least 2 task manifests with distinct `should_prune` entries that cite different constraints
- At least 1 task manifest that exercises the MAPE-K retry path (`valid_proposals_min: 0` on first attempt, succeeds after retry)
- An `_expected` block in every task manifest so the integration test harness can assert outcomes

To add a project: create `docs/examples/{project-name}/` with the structure shown in `ads-platform/`, add it to this README, and add a test case in `tests/integration/`.
