# Refactoring Plan — Codebase Alignment with 10-Prompt Architecture

> **For agentic workers:** This is a planning document, not an implementation plan. Read it in full before beginning any work. Each section is a self-contained refactoring concern.

**Goal:** Align the existing codebase with the 10-prompt ephemeral agent orchestration architecture before implementing new crates. These are issues identified during the agent-types extension that affect multiple crates and should be resolved first to avoid compounding debt.

---

## Why Refactor Before Building

The 10-prompt plan introduces three new crates (`h2ai-provisioner`, `h2ai-memory`, `h2ai-telemetry`) and modifies the orchestrator workflow. Several existing patterns are inconsistent or incomplete — resolving them now is cheaper than resolving them after new crates have been built on top of the broken foundations.

---

## R1: MergeEngine `retry_count` hardcoded to 0

**Location:** `crates/autonomic/src/merger.rs`

**Problem:** `ZeroSurvivalEvent` is constructed with `retry_count: 0` hardcoded. The MAPE-K loop needs the actual retry count to enforce `max_retries` correctly and populate `TaskFailedEvent` diagnostic payload.

**Fix:** Add `retry_count: u32` parameter to `MergeEngine::resolve`. Update the single call site in the orchestrator (or the autonomic loop) to pass the current retry count. Update the 5 existing merger tests to pass `0` explicitly — no behavior change, just makes the parameter visible.

**Files affected:**
- `crates/autonomic/src/merger.rs`
- `crates/autonomic/tests/merger_test.rs`

**Scope:** 1 function signature change, 5 test call sites, no behavioral change.

---

## R2: `tau: f64` is unvalidated across the codebase

**Location:** `crates/h2ai-types/src/adapter.rs` (`ComputeRequest`), `crates/h2ai-types/src/events.rs` (`ProposalEvent`), `crates/h2ai-types/src/agent.rs` (`TaskPayload`), `crates/h2ai-types/src/config.rs` (`ExplorerConfig`, `RoleSpec`)

**Problem:** `tau` appears 7+ times as a bare `f64` with domain `(0.0, 1.0]`. `RoleErrorCost` and `JeffectiveGap` already use validated newtypes for their `f64` fields. `tau` should be consistent.

**Fix:** Introduce `TauValue(f64)` newtype in `h2ai-types/src/physics.rs` with a `new(v: f64) -> Result<Self, TauError>` constructor that validates `v > 0.0 && v <= 1.0`. Replace all `tau: f64` fields with `tau: TauValue`. Update all constructors and tests that supply raw `f64` tau values.

**Files affected:** `physics.rs`, `adapter.rs`, `events.rs`, `agent.rs`, `config.rs`, all test files that construct these types.

**Scope:** Medium. Touches many files but each change is mechanical — add `.into()` or `TauValue::new(v).unwrap()` at test sites, update constructors.

**Priority:** Medium. Can be deferred to after Prompts 2–4 are implemented, but must be done before the orchestrator workflow (Prompt 9) is updated.

---

## R3: `agent_id: String` — no identity newtype

**Location:** `crates/h2ai-types/src/agent.rs` (`TaskResult`, `AgentTelemetryEvent` when implemented)

**Problem:** `TaskId` and `ExplorerId` are UUID-backed newtypes that prevent cross-assignment. `agent_id` is a bare `String`. Edge agents will be identified across `TaskResult`, `AgentTelemetryEvent`, provisioner logs, and memory provider keys — a newtype prevents accidental string mix-ups.

**Fix:** Add `AgentId(String)` newtype to `crates/h2ai-types/src/identity.rs`. Replace `agent_id: String` in `TaskResult` (and in `AgentTelemetryEvent` when Task 3 is complete). Implement `Display`, `From<&str>`, `From<String>`, `Serialize`, `Deserialize`.

**Files affected:** `identity.rs`, `agent.rs`, `agent_test.rs`.

**Scope:** Small. Best done before Task 3 (`AgentTelemetryEvent`) since that type also uses `agent_id`.

---

## R4: Complete the autonomic crate (deferred tasks)

**Status:** Tasks 5 and 6 of `docs/.plans/2026-04-19-autonomic.md` are pending.

**What remains:**
- Task 5: `RetryPolicy` — 5 tests
- Task 6: Wire `lib.rs`, final checks, 23 total tests green

**Dependency:** R1 (retry_count fix) must be done before Task 5 can be completed correctly.

**Priority:** High. The autonomic crate is used by the orchestrator (Prompt 9). It must be complete before the workflow update.

---

## R5: `h2ai-types` event count in lib.rs doc

**Location:** `crates/h2ai-types/src/lib.rs` line 16

**Problem:** Doc says "all 17 event structs" — this will be stale as `AgentTelemetryEvent` is added in Task 3.

**Fix:** After Task 3 is complete, update to "all 17 orchestration event structs + `AgentTelemetryEvent`".

**Scope:** One line. Do after Task 3.

---

## Recommended Execution Order

```
R3 (AgentId newtype)         ← before Task 3 (AgentTelemetryEvent)
Task 3 (AgentTelemetryEvent) ← complete current plan
Task 4 (final checks)        ← complete current plan
R1 (retry_count fix)         ← before autonomic Task 5
Autonomic Task 5 (RetryPolicy)
Autonomic Task 6 (wire lib.rs)
R2 (TauValue newtype)        ← before orchestrator workflow (Prompt 9)
R5 (lib.rs doc)              ← after Task 3
Prompts 2–10                 ← new crate implementations
```

---

## What This Plan Does Not Cover

- Architectural changes — all crate boundaries, dependency rules, and event schemas remain as documented in `docs/architecture/`
- Performance optimization — no benchmarks, no algorithmic changes
- UI / API layer — `crates/api` and `crates/orchestrator` are not yet implemented and are out of scope here
