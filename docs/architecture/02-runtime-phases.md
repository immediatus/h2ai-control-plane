# Runtime Phases — Execution Flow and Event Vocabulary

The H2AI Control Plane runtime is a deterministic state machine. Every state transition is an immutable event appended to a NATS JetStream log. There are no side-channel state mutations — if it happened, it is in the log.

This document describes the six runtime phases, the 14-event vocabulary, and the structural guarantees the system enforces.

---

## The 14-Event Vocabulary

All events are published to NATS subject `h2ai.tasks.{task_id}`. All are immutable, serialized with `serde` using internally-tagged JSON (`"event_type": "..."` + `"payload": {...}`).

| # | Event | Publisher | Phase |
|---|---|---|---|
| 1 | `CalibrationCompletedEvent` | autonomic | 0 |
| 2 | `TaskBootstrappedEvent` | context / api | 1 |
| 3 | `TopologyProvisionedEvent` | autonomic | 2 |
| 4 | `MultiplicationConditionFailedEvent` | orchestrator | 2.5 |
| 5 | `ProposalEvent` | adapters (via orchestrator) | 3 |
| 6 | `ProposalFailedEvent` | orchestrator | 3 |
| 7 | `GenerationPhaseCompletedEvent` | orchestrator | 3 |
| 8 | `ValidationEvent` | adapters (Auditor) | 4 |
| 9 | `BranchPrunedEvent` | adapters (Auditor) | 4 |
| 10 | `ZeroSurvivalEvent` | orchestrator | 4 |
| 11 | `ConsensusRequiredEvent` | state | 5 |
| 12 | `SemilatticeCompiledEvent` | state | 5 |
| 13 | `MergeResolvedEvent` | api | 5 |
| 14 | `TaskFailedEvent` | orchestrator | any |

---

## Phase 0 — Calibration

**Trigger:** System startup, or `POST /calibrate` from the operator.

**Publisher:** `crates/autonomic`

**What happens:**
The calibration harness runs a small set of representative tasks (default: 3) through the full adapter pool. It measures:
- `α` — the serial contention fraction, from the fraction of wall time spent in non-parallelizable work.
- `κ_base` — the baseline pairwise coherency cost, from token exchange overhead between adapter pairs.
- `CG(i,j)` — Common Ground samples across Explorer pairs, from agreement rates on calibration tasks.

From these, it computes:
- `κ_eff = κ_base / mean(CG)`
- `N_max = sqrt((1 − α) / κ_eff)`
- `θ_coord = min(CG_mean − σ_CG, 0.3)`

**Output event:** `CalibrationCompletedEvent` — carries `CoherencyCoefficients` and `CoordinationThreshold`. Cached in NATS KV store. Reused until the adapter pool changes or the operator forces recalibration.

**Gate:** No live task proceeds without valid calibration data. `POST /tasks` returns `503 CalibrationRequiredError` if calibration has not been completed.

---

## Phase 1 — Bootstrap

**Trigger:** Human POSTs a task manifest to `POST /tasks`.

**Publisher:** `crates/context` + `crates/api`

**What happens:**
1. `crates/context` reads the submitted manifest and scans the local ADR corpus.
2. Computes `J_eff = J(K_prompt, K_task_required)` — Jaccard overlap of the explicit context against task requirements.
3. If `J_eff < threshold` → synchronous `400 ContextUnderflowError` returned. Nothing written to NATS. The human must add more explicit constraints to the manifest.
4. If `J_eff ≥ threshold` → compiles an immutable `system_context` string from ADRs + manifest.
5. Publishes `TaskBootstrappedEvent` with `system_context`, `ParetoWeights` (from manifest), and `j_eff`.

**API response:** `202 Accepted` + `task_id`. The human disconnects. All further progress is available via `GET /tasks/{task_id}/events` (SSE or WebSocket).

**Key invariant:** `system_context` is immutable after this event. No agent ever sees a different context than what the Auditor was briefed on.

---

## Phase 2 — Topology Provisioning

**Trigger:** `CalibrationCompletedEvent` (cached) + `TaskBootstrappedEvent`.

**Publisher:** `crates/autonomic`

**What happens:**
1. Reads `CoherencyCoefficients` from calibration cache.
2. Reads `ParetoWeights` from the bootstrap event.
3. Computes `κ_eff`, `N_max`, selects topology:
   - **Flat Mesh** — if `N_requested ≤ N_max` AND diversity weight `W_H` is dominant. All Explorers connect through NATS; no Coordinator. Suitable for small, diverse swarms.
   - **Hierarchical Tree** — if `N_requested > N_max` OR containment weight `W_E` is dominant. One Swarm Coordinator + k sub-groups. Branching factor `k_opt = floor(N_max^flat)`. Coordination edges reduced from `O(N²)` to `O(N)`.
4. Assigns τ values per Explorer (spread across [τ_min, τ_max] to guarantee error decorrelation for Multiplication Condition 2).
5. Assigns `RoleErrorCost` (c_i) per node role.
6. Computes `MergeStrategy` from `max(c_i)`.
7. Publishes `TopologyProvisionedEvent`.

**Re-entry:** The autonomic loop re-enters Phase 2 after `ZeroSurvivalEvent` (adjusting {N, τ}) or `MultiplicationConditionFailedEvent` (adjusting parameters based on which condition failed). Bounded by `max_retries`.

---

## Phase 2.5 — Multiplication Condition Gate

**Trigger:** `TopologyProvisionedEvent`.

**Publisher:** `crates/orchestrator`

**What happens:**
Before any inference token is generated, the orchestrator verifies all three conditions from Proposition 3 against the calibration data:

**Condition 1 — Baseline competence:**
Each planned Explorer adapter must have `p_correct > 0.5` on the calibration task set (from Phase 0 measurements). An Explorer performing worse than random chance degrades the collective.

**Condition 2 — Error decorrelation:**
Pairwise agreement rate `ρ < 0.9` across all Explorer pairs on the calibration set. If two Explorers make the same errors 90%+ of the time, they are structurally redundant and add no information. Fix: widen τ spread or route to different model backends.

**Condition 3 — Common Ground floor:**
`CG_mean ≥ θ_coord` for all planned Explorer pairs.

**If all three hold:** Proceed to Phase 3.

**If any fails:** Publish `MultiplicationConditionFailedEvent` (naming which condition failed and the measured values). Re-enter Phase 2 with adjusted parameters. The failure payload is included in `TaskFailedEvent` if retries are exhausted, so the operator can diagnose which condition blocked execution.

---

## Phase 3 — Parallel Generation

**Trigger:** Multiplication Condition gate passes.

**Publisher:** `crates/orchestrator` (coordination) + `crates/adapters` (via orchestrator)

**What happens:**
1. The orchestrator fans out N Explorers into a `tokio::task::JoinSet`.
2. The entire JoinSet is wrapped in `tokio::time::timeout` — bounded wall time, no hanging.
3. Each Explorer calls `IComputeAdapter::execute()`:
   - **Local Explorers** → `tokio::task::spawn_blocking` → llama.cpp FFI (heavy CPU-bound work on the blocking thread pool, never starving the async pool)
   - **Cloud Explorers** → async HTTP on the main async pool
4. On success: `ProposalEvent` published with `{explorer_id, tau, raw_output, token_cost, adapter_kind}`. Explorer terminates.
5. On crash / OOM / timeout: `ProposalFailedEvent` published. Explorer gets a terminal state regardless.
6. JoinSet fully drained → `GenerationPhaseCompletedEvent` published. **The stream is now closed.**

**Critical invariant:** No Explorer reads another Explorer's output. During Phase 3, coordination cost `α → 0` by graph construction.

---

## Phase 4 — Auditor Gate

**Trigger:** `TopologyProvisionedEvent` (Auditor spins up immediately, does not wait for Phase 3).

**Publisher:** `crates/adapters` (Auditor) + `crates/orchestrator`

**What happens:**
The Auditor is a **reactive stream processor**, not a batch processor. It subscribes to `h2ai.tasks.{task_id}` as soon as the topology is provisioned and validates proposals as they arrive:

1. For each `ProposalEvent`: validates against compiled `system_context`. Checks that outputs do not hallucinate APIs, violate ADR constraints, or contradict explicit architectural decisions.
2. **Pass** → `ValidationEvent` published.
3. **Fail** → `BranchPrunedEvent` published with reason and `constraint_error_cost` (c_i of the violated constraint). Branch is tombstoned in the log — permanently preserved for the Merge Authority UI but excluded from the merge.
4. Reads `GenerationPhaseCompletedEvent` → knows the stream is closed → counts valid survivors.

**If survivors > 0:** Proceed to Phase 5.

**If survivors = 0:** Publish `ZeroSurvivalEvent` → MAPE-K autonomic retry.

**Auditor configuration:** τ = 0 (deterministic, no sampling variance). Routed to a large cloud reasoning model. The Auditor's adapter has `RoleErrorCost c_i ≈ 0.9` — a false positive (passing a hallucination) is near-catastrophic.

---

## Phase 4→2 — Autonomic Retry

**Trigger:** `ZeroSurvivalEvent`.

**Publisher:** `crates/autonomic`

**What happens:**
The MAPE-K loop intercepts the zero-survival state and diagnoses why:
- If constraints rejected everything: raise τ for some Explorers (more variance → broader search of the solution space).
- If Explorers were too similar: widen τ spread (enforce Condition 2).
- If N was too low to find a valid answer: increase N (if `N_new ≤ N_max` for the current topology).

Publishes a new `TopologyProvisionedEvent` with adjusted `{N, τ, topology}`. Re-enters Phase 2.

Bounded by `max_retries` (configurable, default: 3). If retries exhausted → `TaskFailedEvent` with full diagnostic: all `BranchPrunedEvent` reasons + c_i weights, all topologies tried, all τ sets tried, and `multiplication_condition_failure` if Phase 2.5 was the blocker.

---

## Phase 5 — Merge + Human Resolution

**Trigger:** Survivors > 0 after Phase 4.

**Publisher:** `crates/state` + `crates/api`

**What happens:**

**Step 5a — Merge strategy check:**
`crates/state` reads `MergeStrategy` from `TopologyProvisionedEvent`:
- `CrdtSemilattice` (default, `max(c_i) ≤ 0.85`): No coordination required. State engine replays validated events and constructs the CRDT semilattice join of surviving proposals. `O(1)` reconciliation. Epistemic diversity fully preserved.
- `BftConsensus` (`max(c_i) > 0.85`): `ConsensusRequiredEvent` published first. State engine runs BFT consensus protocol over surviving proposals. Higher κ cost. Provides mathematical safety when a single undetected Byzantine error would be catastrophic.

**Step 5b — Semilattice compiled:**
`SemilatticeCompiledEvent` published with `{valid_proposals, pruned_proposals, merge_strategy}`.

**Step 5c — Merge Authority UI:**
`crates/api` renders the Merge Authority interface:
- **Valid proposals panel:** Diff view grouped by target function/block. τ value, token cost, and adapter shown per proposal.
- **Pruned proposals (Tombstone) panel:** Every `BranchPrunedEvent` rendered with Explorer ID, τ, attempted output, rejection reason, and c_i weight of the violated constraint. Failures are epistemic data — the human sees what the swarm attempted.
- **Autonomic shift timeline:** Every `ZeroSurvivalEvent` and retry rendered as a timeline node. Human sees exactly when the MAPE-K loop intervened and what parameters it adjusted.
- **Physics panel:** Live `θ_coord`, `J_eff`, `κ_eff`, `N_max`, current `MergeStrategy`.

**Human resolution:** Human performs O(1) merge decision (select, synthesize, or reject). `MergeResolvedEvent` published. Task is closed.

---

## Structural Guarantees

| Guarantee | Mechanism |
|---|---|
| No agent spawned without measured context | J_eff gate in Phase 1; `ContextUnderflowError` if below threshold |
| No agent spawned without calibration data | Phase 0 gate; `CalibrationRequiredError` if uncalibrated |
| Every Explorer has a terminal state | `tokio::time::timeout` over JoinSet; `ProposalFailedEvent` on crash/OOM/timeout |
| Generation stream always closes | `GenerationPhaseCompletedEvent` after JoinSet drained |
| Auditor never hangs | Reactive stream; reads `GenerationPhaseCompletedEvent` as stream-closed signal |
| Auditor never idles | Spins up on `TopologyProvisionedEvent`, validates proposals as they arrive |
| Zero-survival is not terminal | MAPE-K retry loop with parameter adjustment; `TaskFailedEvent` only after `max_retries` |
| Multiplication Condition enforced | Phase 2.5 hard gate before Phase 3; compiler-exhaustive `H2AIEvent` enum |
| Merge strategy matches error stakes | `MergeStrategy` computed from `max(c_i)` at provisioning time |
| Full provenance preserved | Every state transition is an immutable log event; crash recovery = replay from offset 0 |

---

## SSE Event Stream

The API exposes `GET /tasks/{task_id}/events` as a Server-Sent Events or WebSocket stream that tails the NATS subject in real-time. The client receives all 14 event types as they occur. The stream closes on `MergeResolvedEvent` (success) or `TaskFailedEvent` (failure with full diagnostic).

This means the human liaison sees the swarm working in real-time: topology being provisioned, proposals arriving, the Auditor validating or pruning, the MAPE-K loop adjusting — all before the final merge is presented.
