# API Reference

All HTTP endpoints are served by the `crates/h2ai-api` axum gateway. The base URL is `http://<host>:8080` by default.

---

## Authentication

Authentication is not built into the control plane itself — it is expected to be handled by an ingress / API gateway layer (mTLS, JWT validation, OAuth2 proxy). All endpoints in this document assume the request has already been authenticated.

---

## Endpoints

### POST /tasks

Submit a task manifest. Returns immediately with a `task_id`. All further progress is available via the SSE event stream.

**Request body:**

```json
{
  "description": "string — natural language task description",
  "pareto_weights": {
    "diversity": 0.5,
    "containment": 0.3,
    "throughput": 0.2
  },
  "topology": {
    "kind": "auto",
    "branching_factor": null
  },
  "explorers": {
    "count": 4,
    "tau_min": 0.2,
    "tau_max": 0.9,
    "roles": [],
    "review_gates": []
  },
  "constraints": [
    "ADR-001",
    "ADR-007"
  ],
  "context": "optional — additional explicit constraints beyond the constraint corpus"
}
```

**Fields:**

| Field | Type | Required | Description |
|---|---|---|---|
| `description` | string | yes | Task description. Fed into the Dark Knowledge Compiler to measure `J_eff`. |
| `pareto_weights.diversity` | float | yes | Weight for epistemic diversity (`W_H`). Higher → prefers Ensemble with high τ spread. |
| `pareto_weights.containment` | float | yes | Weight for safety containment (`W_E`). Higher → prefers Hierarchical Tree with Coordinator. |
| `pareto_weights.throughput` | float | yes | Weight for raw throughput (`W_X`). Influences Explorer count selection. |
| `pareto_weights.*` | — | — | Must sum to 1.0. Returns 400 if violated. |
| `topology.kind` | enum | no | `"auto"` (default), `"ensemble"`, `"hierarchical_tree"`. When `explorers.roles[]` is non-empty the system always uses `TeamSwarmHybrid` regardless of this field. |
| `topology.branching_factor` | int | no | Override branching factor for `hierarchical_tree`. Default: `floor(N_max^flat)`. Ignored for other topology kinds. |
| `explorers.count` | int | yes | Requested Explorer count. System will reduce if above `N_max`. |
| `explorers.tau_min` | float | no | Minimum temperature. Default: 0.2. Ignored when `roles[]` is non-empty. |
| `explorers.tau_max` | float | no | Maximum temperature. Default: 0.9. Ignored when `roles[]` is non-empty. |
| `explorers.roles[]` | RoleSpec[] | no | Role-typed Explorer specs. When provided, triggers Team-Swarm Hybrid topology and overrides `tau_min`/`tau_max`. Each entry has `agent_id`, `role` (see AgentRole), and optional `tau` and `role_error_cost` overrides. |
| `explorers.review_gates[]` | ReviewGate[] | no | Dependency edges between Explorers. Each entry has `reviewer` (agent_id of the Evaluator-role Explorer) and `blocks` (agent_id of the Explorer whose output must be approved). Only valid with Team-Swarm Hybrid (i.e., when `roles[]` is non-empty). |
| `constraints` | string[] | no | ADR identifiers to explicitly include. The compiler always includes the full corpus; this field pins specific ADRs regardless of `J_eff`. |
| `context` | string | no | Additional explicit context not captured in ADRs. Raises `J_eff`. |

**AgentRole values:**

| Role | τ default | c_i default | Description |
|---|---|---|---|
| `"Coordinator"` | 0.05 | 0.1 | Low-entropy router; assigns sub-tasks to other Explorers. Internal node in Team-Swarm Hybrid. |
| `"Executor"` | 0.40 | 0.5 | Primary output producer. Subject to review gates when configured. |
| `"Evaluator"` | 0.10 | 0.9 | Review gate; evaluates and approves/blocks Executor output before it reaches the ADR Auditor. |
| `"Synthesizer"` | 0.80 | 0.1 | Combines or summarises other outputs. High τ, low c_i. |
| `"Custom"` | (required) | (required) | Arbitrary domain role. Must supply `tau` and `role_error_cost` explicitly. |

**Example manifests:**

*Ensemble + CRDT (default / explicit):*
```json
{
  "description": "Propose a token rotation strategy for the auth service",
  "pareto_weights": {"diversity": 0.5, "containment": 0.3, "throughput": 0.2},
  "topology": {"kind": "ensemble"},
  "explorers": {"count": 4, "tau_min": 0.2, "tau_max": 0.9},
  "constraints": ["ADR-001"]
}
```

*Hierarchical Tree (explicit, large swarm):*
```json
{
  "description": "Design the full data migration plan",
  "pareto_weights": {"diversity": 0.2, "containment": 0.6, "throughput": 0.2},
  "topology": {"kind": "hierarchical_tree", "branching_factor": 3},
  "explorers": {"count": 9, "tau_min": 0.1, "tau_max": 0.8},
  "constraints": ["ADR-005", "ADR-011"]
}
```

*Team-Swarm Hybrid (role-typed with review gate):*
```json
{
  "description": "Implement the payment webhook handler",
  "pareto_weights": {"diversity": 0.3, "containment": 0.4, "throughput": 0.3},
  "explorers": {
    "count": 4,
    "roles": [
      {"agent_id": "coord",  "role": "Coordinator"},
      {"agent_id": "impl_1", "role": "Executor"},
      {"agent_id": "impl_2", "role": "Executor"},
      {"agent_id": "review", "role": "Evaluator"}
    ],
    "review_gates": [
      {"reviewer": "review", "blocks": "impl_1"},
      {"reviewer": "review", "blocks": "impl_2"}
    ]
  },
  "constraints": ["ADR-003", "ADR-007"]
}
```

**Responses:**

`202 Accepted` — Task accepted, calibration data exists, `J_eff` above threshold.

```json
{
  "task_id": "task_01HXYZ...",
  "status": "accepted",
  "events_url": "/tasks/task_01HXYZ.../events",
  "j_eff": 0.67,
  "topology_kind": "Ensemble",
  "n_max": 6.3,
  "interface_n_max": null
}
```

`topology_kind` values: `"Ensemble"`, `"HierarchicalTree"`, `"TeamSwarmHybrid"`. `interface_n_max` is non-null only for `TeamSwarmHybrid` — it is the binding ceiling for concurrent sub-tasks between the human liaison and the Coordinator.

`400 Bad Request` — `J_eff` below threshold. The human must add more explicit context.

```json
{
  "error": "ContextUnderflowError",
  "j_eff": 0.18,
  "threshold": 0.4,
  "message": "Jaccard overlap between submitted context and task requirements is too low. Add explicit constraints, ADR references, or architectural decisions to the manifest.",
  "missing_coverage": [
    "authentication strategy",
    "database access policy",
    "service boundary constraints"
  ]
}
```

`400 Bad Request` — Pareto weights do not sum to 1.0.

```json
{
  "error": "InvalidParetoWeights",
  "message": "pareto_weights must sum to 1.0, got 0.85"
}
```

`503 Service Unavailable` — Calibration has not been run.

```json
{
  "error": "CalibrationRequiredError",
  "message": "No calibration data found. POST /calibrate before submitting tasks."
}
```

---

### GET /tasks/{task_id}/events

Server-Sent Events stream. Tails the NATS JetStream subject `h2ai.tasks.{task_id}` in real time. The client receives all 23 event types as they occur.

**Headers:**

```
Accept: text/event-stream
```

**Stream lifecycle:**
- Opens immediately on connection.
- Each event is a JSON object on a `data:` line followed by `\n\n`.
- Stream closes on `MergeResolvedEvent` (success) or `TaskFailedEvent` (failure).
- Reconnect by re-connecting to the same URL — the stream replays from the last seen event using the `Last-Event-ID` header.

**Reconnect:**

```
Last-Event-ID: 7
```

The server replays from offset 7 in the NATS stream.

**Event format:**

```
id: {sequence_number}
data: {"event_type": "...", "payload": {...}}

```

See [Event Vocabulary](#event-vocabulary) below for all 23 event schemas.

---

### GET /tasks/{task_id}

Returns the current task status without streaming.

**Response:**

```json
{
  "task_id": "task_01HXYZ...",
  "status": "generating",
  "phase": 3,
  "phase_name": "ParallelGeneration",
  "explorers_completed": 2,
  "explorers_total": 4,
  "proposals_valid": 2,
  "proposals_pruned": 0,
  "autonomic_retries": 0
}
```

**Status values:** `pending`, `provisioning`, `validating`, `generating`, `auditing`, `merging`, `resolved`, `failed`.

---

### POST /calibrate

Triggers the calibration harness on the current adapter pool. Calibration measures `α`, `κ_base`, and pairwise `CG` values across all configured Explorer adapters.

**Request body:** empty.

**Response:**

`202 Accepted`

```json
{
  "calibration_id": "cal_01HXYZ...",
  "status": "accepted",
  "events_url": "/calibrate/cal_01HXYZ.../events",
  "adapter_count": 5
}
```

Calibration replaces any previously cached coefficients. Running tasks are not affected — they use the coefficients from when they were provisioned.

---

### GET /calibrate/{calibration_id}/events

SSE stream for calibration progress. Closes on `CalibrationCompletedEvent`.

```
data: {"event_type":"CalibrationProgress","payload":{"task":1,"of":3,"adapter":"local-llama-8b"}}
data: {"event_type":"CalibrationProgress","payload":{"task":2,"of":3,"adapter":"cloud-gpt4o"}}
data: {"event_type":"CalibrationProgress","payload":{"task":3,"of":3,"adapter":"cloud-claude"}}
data: {"event_type":"CalibrationCompleted","payload":{"alpha":0.12,"kappa_base":0.021,"kappa_eff":0.019,"n_max":6.3,"theta_coord":0.28,"cg_mean":0.71,"cg_std_dev":0.09}}
```

---

### GET /calibrate/current

Returns the currently cached calibration coefficients.

```json
{
  "calibration_id": "cal_01HXYZ...",
  "calibrated_at": "2026-04-19T10:00:00Z",
  "alpha": 0.12,
  "kappa_base": 0.021,
  "kappa_eff": 0.019,
  "n_max": 6.3,
  "theta_coord": 0.28,
  "cg_mean": 0.71,
  "cg_std_dev": 0.09,
  "adapter_count": 5
}
```

`404` if no calibration has been run yet.

---

### POST /tasks/{task_id}/merge

Submit the human merge resolution. Closes the task.

**Request body:**

```json
{
  "resolution": "select",
  "selected_proposals": ["exp_A", "exp_B"],
  "synthesis_notes": "Combined exp_A's rotation strategy with exp_B's revocation approach.",
  "final_output": "string — the merged output text the human approved"
}
```

**Fields:**

| Field | Type | Description |
|---|---|---|
| `resolution` | enum | `select` (pick surviving proposals), `synthesize` (human-authored synthesis), `reject` (all pruned, task voided) |
| `selected_proposals` | string[] | Explorer IDs to include when `resolution=select` |
| `synthesis_notes` | string | Optional human annotation |
| `final_output` | string | Required when `resolution=synthesize` |

**Response:** `200 OK` — publishes `MergeResolvedEvent`, closes task.

---

### GET /health

Liveness probe. Returns `200` if the process is alive.

```json
{"status": "ok"}
```

---

### GET /ready

Readiness probe. Returns `200` only if calibration data exists and NATS is reachable.

```json
{"status": "ready", "calibration": "valid", "nats": "connected"}
```

Returns `503` if not ready — Kubernetes will remove the pod from the load balancer pool.

---

### GET /metrics

Prometheus metrics endpoint. See [Configuration Reference — Metrics](configuration.md#metrics) for the full gauge list.

---

## Event Vocabulary

All 17 events published to `h2ai.tasks.{task_id}`. Internally-tagged JSON: `"event_type"` + `"payload"`.

---

### CalibrationCompletedEvent

Published by `crates/h2ai-autonomic` at Phase 0 completion. Cached in NATS KV.

```json
{
  "event_type": "CalibrationCompleted",
  "payload": {
    "calibration_id": "cal_01HXYZ...",
    "alpha": 0.12,
    "kappa_base": 0.021,
    "kappa_eff": 0.019,
    "n_max": 6.3,
    "theta_coord": 0.28,
    "cg_mean": 0.71,
    "cg_std_dev": 0.09,
    "cg_samples": [0.68, 0.74, 0.71, 0.69, 0.73],
    "adapter_count": 5,
    "calibration_task_count": 3
  }
}
```

---

### TaskBootstrappedEvent

Published by `crates/h2ai-context` + `crates/h2ai-api` at Phase 1 completion.

```json
{
  "event_type": "TaskBootstrapped",
  "payload": {
    "task_id": "task_01HYYZ...",
    "j_eff": 0.67,
    "j_eff_threshold": 0.4,
    "system_context": "string — compiled immutable context from ADRs + manifest",
    "pareto_weights": {
      "diversity": 0.5,
      "containment": 0.3,
      "throughput": 0.2
    },
    "adr_count": 7,
    "bootstrapped_at": "2026-04-19T14:23:01Z"
  }
}
```

---

### TopologyProvisionedEvent

Published by `crates/h2ai-autonomic` at Phase 2 completion. Re-published on every MAPE-K retry.

```json
{
  "event_type": "TopologyProvisioned",
  "payload": {
    "task_id": "task_01HYYZ...",
    "topology_kind": "Ensemble",
    "n": 4,
    "n_max": 6.3,
    "interface_n_max": null,
    "kappa_eff": 0.019,
    "theta_coord": 0.28,
    "merge_strategy": "CrdtSemilattice",
    "retry_number": 0,
    "explorers": [
      {"explorer_id": "exp_A", "tau": 0.3, "adapter_kind": "Local",  "role": null,          "role_error_cost": 0.1},
      {"explorer_id": "exp_B", "tau": 0.5, "adapter_kind": "Cloud",  "role": null,          "role_error_cost": 0.1},
      {"explorer_id": "exp_C", "tau": 0.7, "adapter_kind": "Cloud",  "role": null,          "role_error_cost": 0.1},
      {"explorer_id": "exp_D", "tau": 0.9, "adapter_kind": "Local",  "role": null,          "role_error_cost": 0.1}
    ],
    "review_gates": [],
    "auditor": {
      "explorer_id": "auditor_0",
      "tau": 0.0,
      "adapter_kind": "Cloud",
      "role": "Evaluator",
      "role_error_cost": 0.9
    }
  }
}
```

`topology_kind` values: `"Ensemble"`, `"HierarchicalTree"`, `"TeamSwarmHybrid"`. The `role` field in each explorer entry is `null` for `Ensemble` and `HierarchicalTree`; populated for `TeamSwarmHybrid`. `interface_n_max` is non-null only for `TeamSwarmHybrid`.

---

### MultiplicationConditionFailedEvent

Published by `crates/h2ai-orchestrator` at Phase 2.5 when any of the 3 Proposition 3 conditions fails.

```json
{
  "event_type": "MultiplicationConditionFailed",
  "payload": {
    "task_id": "task_01HYYZ...",
    "failed_condition": "ErrorDecorrelation",
    "measured_value": 0.94,
    "threshold": 0.9,
    "description": "Pairwise agreement rate ρ=0.94 exceeds 0.9. Explorers exp_A and exp_B are structurally redundant. Widening τ spread.",
    "retry_number": 1
  }
}
```

`failed_condition` values: `BaselineCompetence`, `ErrorDecorrelation`, `CommonGroundFloor`.

---

### ProposalEvent

Published by `crates/h2ai-adapters` (via orchestrator) when an Explorer completes.

```json
{
  "event_type": "Proposal",
  "payload": {
    "task_id": "task_01HYYZ...",
    "explorer_id": "exp_A",
    "tau": 0.3,
    "adapter_kind": "Local",
    "raw_output": "string — Explorer's full output",
    "token_cost": 847,
    "latency_ms": 3241,
    "completed_at": "2026-04-19T14:23:09Z"
  }
}
```

---

### ProposalFailedEvent

Published by `crates/h2ai-orchestrator` when an Explorer crashes, runs out of memory, or times out.

```json
{
  "event_type": "ProposalFailed",
  "payload": {
    "task_id": "task_01HYYZ...",
    "explorer_id": "exp_D",
    "tau": 0.9,
    "adapter_kind": "Local",
    "failure_reason": "Timeout",
    "failed_at": "2026-04-19T14:23:30Z"
  }
}
```

`failure_reason` values: `Timeout`, `OOM`, `AdapterError`, `Crash`.

---

### GenerationPhaseCompletedEvent

Published by `crates/h2ai-orchestrator` after the JoinSet is fully drained. Signals the Auditor that the stream is closed.

```json
{
  "event_type": "GenerationPhaseCompleted",
  "payload": {
    "task_id": "task_01HYYZ...",
    "proposals_received": 3,
    "proposals_failed": 1,
    "completed_at": "2026-04-19T14:23:31Z"
  }
}
```

---

### ValidationEvent

Published by `crates/h2ai-adapters` (Auditor) when a proposal passes.

```json
{
  "event_type": "Validation",
  "payload": {
    "task_id": "task_01HYYZ...",
    "explorer_id": "exp_A",
    "validated_at": "2026-04-19T14:23:11Z"
  }
}
```

---

### BranchPrunedEvent

Published by `crates/h2ai-adapters` (Auditor) when a proposal fails validation. The branch is tombstoned — permanently preserved but excluded from merge.

```json
{
  "event_type": "BranchPruned",
  "payload": {
    "task_id": "task_01HYYZ...",
    "explorer_id": "exp_C",
    "tau": 0.7,
    "reason": "Proposes storing refresh tokens in Redis — violates ADR-001 stateless auth requirement",
    "violated_adr": "ADR-001",
    "constraint_error_cost": 0.72,
    "pruned_at": "2026-04-19T14:23:14Z"
  }
}
```

---

### ZeroSurvivalEvent

Published by `crates/h2ai-orchestrator` when all proposals are pruned. Triggers the MAPE-K retry loop.

```json
{
  "event_type": "ZeroSurvival",
  "payload": {
    "task_id": "task_01HYYZ...",
    "pruned_count": 4,
    "retry_number": 1,
    "autonomic_action": "WidenTauSpread",
    "previous_tau_range": [0.3, 0.8],
    "next_tau_range": [0.1, 0.95]
  }
}
```

`autonomic_action` values: `WidenTauSpread`, `IncreaseN`, `WidenTauAndIncreaseN`.

---

### ConsensusRequiredEvent

Published by `crates/h2ai-state` when `max(c_i) > 0.85` and BFT consensus is required before merge.

```json
{
  "event_type": "ConsensusRequired",
  "payload": {
    "task_id": "task_01HYYZ...",
    "max_role_error_cost": 0.9,
    "threshold": 0.85,
    "surviving_proposal_count": 3
  }
}
```

---

### SemilatticeCompiledEvent

Published by `crates/h2ai-state` when the CRDT semilattice join (or BFT consensus) is complete and the task is ready for human resolution.

```json
{
  "event_type": "SemilatticeCompiled",
  "payload": {
    "task_id": "task_01HYYZ...",
    "merge_strategy": "CrdtSemilattice",
    "valid_proposals": 2,
    "pruned_proposals": 1,
    "failed_proposals": 1,
    "compiled_at": "2026-04-19T14:23:16Z"
  }
}
```

---

### MergeResolvedEvent

Published by `crates/h2ai-api` when the human completes the Merge Authority resolution. Closes the task and the SSE stream.

```json
{
  "event_type": "MergeResolved",
  "payload": {
    "task_id": "task_01HYYZ...",
    "resolution": "select",
    "selected_proposals": ["exp_A", "exp_B"],
    "synthesis_notes": "Combined approach.",
    "resolved_by": "yuriy@example.com",
    "resolved_at": "2026-04-19T14:25:01Z",
    "total_duration_ms": 120000
  }
}
```

---

### TaskFailedEvent

Published by `crates/h2ai-orchestrator` when MAPE-K retries are exhausted. Closes the task and the SSE stream. Contains full diagnostic payload.

```json
{
  "event_type": "TaskFailed",
  "payload": {
    "task_id": "task_01HYYZ...",
    "failure_reason": "MaxRetriesExhausted",
    "retries_attempted": 3,
    "branch_pruned_events": [...],
    "topologies_tried": ["Ensemble", "Ensemble", "HierarchicalTree"],
    "tau_ranges_tried": [[0.3,0.8],[0.1,0.95],[0.1,0.95]],
    "multiplication_condition_failure": {
      "failed_condition": "BaselineCompetence",
      "measured_value": 0.48,
      "threshold": 0.5
    },
    "failed_at": "2026-04-19T14:28:00Z"
  }
}
```

`failure_reason` values: `MaxRetriesExhausted`, `MultiplicationConditionUnresolvable`, `CalibrationExpired`.

---

### ReviewGateTriggeredEvent

Published by `crates/h2ai-orchestrator` at Phase 3b when an Executor's proposal enters review gate evaluation. Only emitted for `TeamSwarmHybrid` topology.

```json
{
  "event_type": "ReviewGateTriggered",
  "payload": {
    "task_id": "task_01HYYZ...",
    "gate_id": "gate_review_impl_1",
    "blocked_explorer_id": "impl_1",
    "reviewer_explorer_id": "review",
    "proposal_ref": "exp_impl_1_proposal_01",
    "triggered_at": "2026-04-19T14:23:12Z"
  }
}
```

---

### ReviewGateBlockedEvent

Published by `crates/h2ai-orchestrator` at Phase 3b when an Evaluator rejects an Executor's proposal. The proposal is tombstoned at the gate level and never reaches the ADR Auditor.

```json
{
  "event_type": "ReviewGateBlocked",
  "payload": {
    "task_id": "task_01HYYZ...",
    "gate_id": "gate_review_impl_1",
    "blocked_explorer_id": "impl_1",
    "reviewer_explorer_id": "review",
    "rejection_reason": "Proposal uses synchronous HTTP calls inside async handler — violates ADR-007 non-blocking I/O requirement",
    "blocked_at": "2026-04-19T14:23:15Z"
  }
}
```

The `rejection_reason` is the Evaluator's natural language explanation. The blocked branch appears in the Merge Authority Tombstone panel attributed to `"ReviewGate"` rather than an ADR identifier.

---

### InterfaceSaturationWarningEvent

Published by `crates/h2ai-autonomic` when the number of concurrent active sub-tasks approaches `N_max^interface`. Only emitted for `TeamSwarmHybrid` topology. A warning — not a gate. The operator can use this to pace incoming work or scale the liaison team.

```json
{
  "event_type": "InterfaceSaturationWarning",
  "payload": {
    "task_id": "task_01HYYZ...",
    "active_subtasks": 4,
    "interface_n_max": 5,
    "saturation_ratio": 0.8,
    "warning_threshold": 0.75,
    "emitted_at": "2026-04-19T14:23:20Z"
  }
}
```

`saturation_ratio = active_subtasks / interface_n_max`. The warning fires when this ratio exceeds 0.75 (configurable). The `h2ai_interface_n_max` Prometheus gauge tracks this continuously.

---

## Error Codes Reference

| HTTP Status | Error Code | Meaning |
|---|---|---|
| 400 | `ContextUnderflowError` | `J_eff` below threshold |
| 400 | `InvalidParetoWeights` | Weights do not sum to 1.0 |
| 400 | `InvalidExplorerCount` | `count` < 1 or not an integer |
| 404 | `TaskNotFound` | Unknown `task_id` |
| 404 | `CalibrationNotFound` | Unknown `calibration_id` or no calibration run |
| 409 | `TaskAlreadyResolved` | `POST /tasks/{id}/merge` on a closed task |
| 503 | `CalibrationRequiredError` | No calibration data cached |
| 503 | `NatsUnavailable` | Cannot reach NATS JetStream |
