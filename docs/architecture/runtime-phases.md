# Runtime Phases — Execution Flow and Event Vocabulary

The H2AI Control Plane runtime is a deterministic state machine. Every state transition is an immutable event appended to a NATS JetStream log. There are no side-channel state mutations — if it happened, it is in the log.

This document describes the nine runtime phases, the core event vocabulary, and the structural guarantees the system enforces.

---

## Core Orchestration Events

All events are published to NATS subject `h2ai.tasks.{task_id}`. All are immutable, serialized with `serde` using internally-tagged JSON (`"event_type": "..."` + `"payload": {...}`).

| # | Event | Publisher | Phase |
|---|---|---|---|
| 1 | `CalibrationCompletedEvent` | autonomic | 0 |
| 2 | `TaskBootstrappedEvent` | context / api | 1 |
| 3 | `TopologyProvisionedEvent` | autonomic | 2 |
| 4 | `MultiplicationConditionFailedEvent` | orchestrator | 2.5 |
| 5 | `ProposalEvent` | adapters (via orchestrator TAO loop) | 3 |
| 6 | `ProposalFailedEvent` | orchestrator | 3 |
| 7 | `TaoIterationEvent` | orchestrator | 3 (per turn) |
| 8 | `GenerationPhaseCompletedEvent` | orchestrator | 3 |
| 9 | `VerificationScoredEvent` | orchestrator | 3.5 (per proposal) |
| 10 | `ReviewGateTriggeredEvent` | orchestrator | 3b |
| 11 | `ReviewGateBlockedEvent` | orchestrator | 3b |
| 12 | `ValidationEvent` | adapters (Auditor) | 4 |
| 13 | `BranchPrunedEvent` | adapters (Auditor) | 4 |
| 14 | `ZeroSurvivalEvent` | orchestrator | 4 |
| 15 | `InterfaceSaturationWarningEvent` | autonomic | 2/3 |
| 16 | `ConsensusRequiredEvent` | state | 5 |
| 17 | `SemilatticeCompiledEvent` | state | 5 |
| 18 | `MergeResolvedEvent` | api | 5 |
| 19 | `TaskFailedEvent` | orchestrator | any |

---

## Edge Agent Telemetry Events

Published to NATS subject `h2ai.telemetry.{task_id}` by the `h2ai-telemetry` crate's `BrokerPublisherProvider`. Processed by `RedactionMiddleware` before persistence.

| Variant | Trigger |
|---|---|
| `AgentTelemetryEvent::LlmPromptSent` | Tokens dispatched to edge agent LLM (prompt_tokens recorded) |
| `AgentTelemetryEvent::LlmResponseReceived` | Completion tokens received (completion_tokens recorded) |
| `AgentTelemetryEvent::ShellCommandExecuted` | Shell command run by edge agent (command + exit_code recorded) |
| `AgentTelemetryEvent::SystemError` | Edge agent panic or unrecoverable error (message recorded) |

All four variants carry `task_id`, `agent_id`, and `timestamp`. `command` strings pass through `RedactionMiddleware` before logging.

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
2. Reads `ParetoWeights` and `topology` field from the bootstrap event.
3. Computes `κ_eff`, `N_max`, selects topology:

| Condition | Selected topology | Pareto profile |
|---|---|---|
| Manifest provides `explorers.roles[]` | **Team-Swarm Hybrid** | T=84%, E=91%, D=95% |
| Manifest sets `topology.kind: "hierarchical_tree"` | **Hierarchical Tree** | T=96%, E=96%, D=60% |
| Manifest sets `topology.kind: "ensemble"` | **Ensemble + CRDT** | T=84%, E=84%, D=90% |
| Auto: `N_requested ≤ N_max` AND `W_H` dominant | **Ensemble + CRDT** | T=84%, E=84%, D=90% |
| Auto: `N_requested > N_max` OR `W_E` dominant | **Hierarchical Tree** | T=96%, E=96%, D=60% |

   - **Ensemble + CRDT** (formerly "Flat Mesh") — all Explorers connect through NATS; no Coordinator. Suitable for small, diverse swarms.
   - **Hierarchical Tree** — one Coordinator + k sub-groups. Branching factor `k_opt = floor(N_max^flat)`. Coordination edges reduced from `O(N²)` to `O(N)`.
   - **Team-Swarm Hybrid** — role-differentiated Explorers (Coordinator, Executor, Evaluator, Synthesizer, Custom) with review gates between specified pairs. A Coordinator (τ≈0.05) routes sub-tasks; Evaluators form review gates that block Executor output before it reaches the ADR Auditor. The binding ceiling is `N_max^interface = sqrt((1−α_liaison)·CG(H_liaison, Coordinator)/κ_base)`, typically 3–5 concurrent sub-tasks. An `InterfaceSaturationWarningEvent` is emitted when active sub-tasks approach this ceiling.

4. Assigns τ values per Explorer: from role canonical defaults when `explorers.roles[]` is provided; otherwise spread across [τ_min, τ_max] to guarantee error decorrelation for Multiplication Condition 2.
5. Assigns `RoleErrorCost` (c_i) per node role.
6. Computes `MergeStrategy` from `max(c_i)`.
7. Publishes `TopologyProvisionedEvent` carrying `topology_kind`, resolved `RoleSpec[]`, and `ReviewGate[]`.

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

## Phase 3 — Parallel Generation (TAO Loop)

**Trigger:** Multiplication Condition gate passes.

**Publisher:** `crates/orchestrator` (coordination) + `crates/adapters` (via TAO loop)

**What happens:**
1. The orchestrator fans out N Explorers into a `tokio::task::JoinSet`.
2. The entire JoinSet is wrapped in `tokio::time::timeout` — bounded wall time, no hanging.
3. Each Explorer runs a **TAO loop** (`orchestrator::tao_loop::TaoLoop::run`) for up to `max_turns` (default 3):
   - **Turn 1:** Initial `IComputeAdapter::execute()` call with the task prompt.
   - **Pattern check:** If `TaoConfig.verify_pattern` set, output is regex-matched. Pass → commit immediately.
   - **Observation feedback:** On pattern or schema failure, the retry instruction from `TaoConfig.retry_instruction` (template: `[OBSERVATION turn {turn}]: output did not satisfy verification. Revise your response.`) is appended; loop retries. The exact text is config-driven — no prompt strings are hardcoded.
   - **Schema check (optional):** If `OutputSchemaConfig.schema_json` set, output is validated against JSON Schema. Failure injects `TaoConfig.observation_fail_schema` (template: `schema validation failed on turn {turn}: {error}; retrying`) for the next turn.
   - **Turn exhaustion:** On `max_turns` reached without match, last output is committed as the proposal.
   - **Local Explorers** → `tokio::task::spawn_blocking` → llama.cpp FFI
   - **Cloud Explorers** → async HTTP on the main async pool
4. On success: `TaoIterationEvent` per turn + `ProposalEvent` with `{explorer_id, tau, raw_output, token_cost, adapter_kind, tao_turns}`. Explorer terminates.
5. On crash / OOM / timeout: `ProposalFailedEvent` published. Explorer gets a terminal state regardless.
6. JoinSet fully drained → `GenerationPhaseCompletedEvent` published. **The stream is now closed.**

**TAO physics (Definition 11):** Each iteration reduces effective role error cost: `c_i_eff = c_i × 0.60^(t−1)`. Simulation shows Shell agents (c_i=0.9) escape the BFT merge path after just **t=2 turns** (c_i_eff=0.540). The merge strategy re-evaluates `max(c_i_eff)` using actual TAO turn counts before Phase 5.

**Critical invariant:** No Explorer reads another Explorer's output. During Phase 3, coordination cost `α → 0` by graph construction.

---

## Phase 3.5 — Verification Phase

**Trigger:** `GenerationPhaseCompletedEvent`.

**Publisher:** `crates/orchestrator`

**What happens:**
1. All `ProposalEvent` outputs are evaluated in **parallel** (`join_all`) by the evaluator LLM using the system prompt, τ, and token budget from `VerificationConfig` (defaults: `"You are a strict evaluator."`, τ=0.1, 128 tokens).
2. Each proposal receives a score `∈ [0, 1]` via JSON `{"score": float, "reason": string}` response.
3. Proposals with `score ≥ threshold` (default 0.45) proceed to the Auditor gate.
4. Proposals below threshold are **soft-rejected**: `BranchPrunedEvent` with `reason = "verification score X: <reason>"`. They are tombstoned before ADR evaluation.
5. `VerificationScoredEvent` published per proposal with `{explorer_id, score, reason, passed}`.
6. **Graceful degradation:** Parse failure → score defaults to 0.5 (neutral). The system degrades to unfiltered behavior rather than silently dropping all proposals.

**Parallelism (Proposition 6):** With P=N evaluators, wall-clock cost = one T_eval regardless of ensemble size. For N≤6 (AI layer N_max), this adds a constant ~1–3s to Phase 3.5 regardless of how many Explorers ran.

**Simulation finding:** Verification strictness (fr 1.0→0.0) delivers **+21.9pp Q_total** for Executor agents at an established ensemble (N=4). For Shell agents (c_i=0.9) with 50% filter ratio the contribution is **+45pp**. This equals the TAO gain range, making Verification and TAO the two highest-leverage tuning parameters once an ensemble is formed.

---

## Phase 3b — Review Gate Evaluation (Team-Swarm Hybrid only)

**Trigger:** `ProposalEvent` from an Executor-role Explorer, when `ReviewGate[]` declares that Executor's output requires Evaluator approval. Only active when topology is `TeamSwarmHybrid`.

**Publisher:** `crates/orchestrator`

**What happens:**
1. Orchestrator detects a `ProposalEvent` whose `explorer_id` matches the `blocks` side of a `ReviewGate`.
2. Publishes `ReviewGateTriggeredEvent` with `{gate_id, blocked_explorer_id, reviewer_explorer_id, proposal_ref}`.
3. The Evaluator-role Explorer (τ≈0.1, c_i≈0.9) runs its evaluation. It receives only the blocked proposal and `system_context` — it does not see other proposals.
4. **Evaluator approves** → proposal is forwarded to the ADR Auditor gate (Phase 4) unchanged.
5. **Evaluator rejects** → `ReviewGateBlockedEvent` published with `{gate_id, blocked_explorer_id, reviewer_explorer_id, rejection_reason}`. The proposal is tombstoned at the review gate level — it never reaches the ADR Auditor. The rejection is visible in the Merge Authority UI under the Tombstone panel, attributed to the gate rather than an ADR violation.

**Critical invariant:** The ADR Auditor (Phase 4) only sees proposals that have passed all applicable review gates. Review gates are pre-Auditor; they do not replace the Auditor.

**Re-entry:** If all Executor proposals are blocked by review gates and no proposals reach the Auditor, the count of gate-approved survivors after `GenerationPhaseCompletedEvent` is zero → `ZeroSurvivalEvent` → autonomic retry (Phase 4→2). The retry diagnostics distinguish gate blocks from ADR violations.

---

## Phase 4 — Auditor Gate

**Trigger:** `TopologyProvisionedEvent` (Auditor spins up immediately, does not wait for Phase 3).

**Publisher:** `crates/adapters` (Auditor) + `crates/orchestrator`

**What happens:**
The Auditor is a **reactive stream processor**, not a batch processor. It subscribes to `h2ai.tasks.{task_id}` as soon as the topology is provisioned and validates proposals as they arrive:

1. For each `ProposalEvent` that has passed all review gates (or for non-TeamSwarmHybrid topologies, all `ProposalEvent`s): validates against compiled `system_context`. Checks that outputs do not hallucinate APIs, violate ADR constraints, or contradict explicit architectural decisions.
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
| TAO loop always terminates | Hard `max_turns` cap (default 3); last output committed even if pattern never matched |
| TAO processes are cleaned up on timeout | `kill_on_drop(true)` on spawned shell processes; no process leaks on cancellation |
| Verification never silently drops proposals | Parse failure → neutral score 0.5; system degrades gracefully to unfiltered behavior |
| BFT merge re-evaluated after TAO | `max(c_i_eff)` recomputed from actual `tao_turns` counts before Phase 5; Shell agents may qualify for CRDT after t=2 |
| Generation stream always closes | `GenerationPhaseCompletedEvent` after JoinSet drained |
| Auditor never hangs | Reactive stream; reads `GenerationPhaseCompletedEvent` as stream-closed signal |
| Auditor never idles | Spins up on `TopologyProvisionedEvent`, validates proposals as they arrive |
| Auditor only sees verification-passed proposals | Phase 3.5 soft-rejects below-threshold proposals before Auditor evaluates |
| Zero-survival is not terminal | MAPE-K retry loop with parameter adjustment; `TaskFailedEvent` only after `max_retries` |
| Multiplication Condition enforced | Phase 2.5 hard gate before Phase 3; compiler-exhaustive `H2AIEvent` enum |
| Merge strategy matches error stakes | `MergeStrategy` computed from `max(c_i_eff)` (post-TAO) at merge time |
| Review gate always has a terminal state | Evaluator either approves (proposal forwarded) or blocks (`ReviewGateBlockedEvent`); no hanging evaluation |
| ADR Auditor only sees gate-approved proposals | TeamSwarmHybrid: Auditor subscribes only after review gate decisions are recorded; no double-validation |
| Topology selection is computable from manifest | Deterministic three-way rule in Phase 2; no topology ambiguity; operator can always predict which path |
| Full provenance preserved | Every state transition is an immutable log event; crash recovery = replay from offset 0 |
| Context fits token budget before generation | Compaction runs before Phase 3; `head+tail` strategy preserves ADR keywords at window ends |

---

## SSE Event Stream

The API exposes `GET /tasks/{task_id}/events` as a Server-Sent Events or WebSocket stream that tails the NATS subject in real-time. The client receives all 17 orchestration event types as they occur. The stream closes on `MergeResolvedEvent` (success) or `TaskFailedEvent` (failure with full diagnostic).

A separate `GET /tasks/{task_id}/telemetry` endpoint tails `h2ai.telemetry.{task_id}` and streams `AgentTelemetryEvent` entries in real-time. This gives the operator visibility into what edge agents (each described by `AgentDescriptor`) are executing — LLM calls, shell commands, errors — as they happen, with secrets already redacted by `RedactionMiddleware`.

This means the human liaison sees the swarm working in real-time: topology being provisioned, proposals arriving, the Auditor validating or pruning, the MAPE-K loop adjusting, and edge agent activity — all before the final merge is presented.
