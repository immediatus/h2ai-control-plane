# H2AI Reference

Consolidated reference for the HTTP API, adapter development, agent descriptors, constraint corpus format, and configuration. All endpoints are served by `crates/h2ai-api` on `http://<host>:8080` by default.

---

## Table of Contents

1. [HTTP API](#http-api)
2. [Event Vocabulary](#event-vocabulary)
3. [Adapter Development](#adapter-development)
4. [Agent Descriptor](#agent-descriptor)
5. [Constraint Corpus](#constraint-corpus)
6. [Configuration](#configuration)

---

## HTTP API

### Authentication

Authentication is not built into the control plane — it is expected at the ingress layer (mTLS, JWT validation, OAuth2 proxy). All endpoints assume requests are pre-authenticated.

---

### POST /tasks

Submit a task manifest. Returns immediately with a `task_id`. All progress is available via the SSE event stream.

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
  "constraints": ["ADR-001", "ADR-007"],
  "context": "optional — additional explicit constraints beyond the constraint corpus"
}
```

**Fields:**

| Field | Type | Required | Description |
|---|---|---|---|
| `description` | string | yes | Task description. Compiled into the immutable `system_context` alongside the constraint corpus. |
| `pareto_weights.diversity` | float | yes | Weight for epistemic diversity (`W_H`). Higher → prefers Ensemble with high τ spread. |
| `pareto_weights.containment` | float | yes | Weight for safety containment (`W_E`). Higher → prefers Hierarchical Tree with Coordinator. |
| `pareto_weights.throughput` | float | yes | Weight for raw throughput (`W_X`). Influences Explorer count selection. |
| `pareto_weights.*` | — | — | Must sum to 1.0. Returns 400 if violated. |
| `topology.kind` | enum | no | `"auto"` (default), `"ensemble"`, `"hierarchical_tree"`. When `explorers.roles[]` is non-empty the system always uses `TeamSwarmHybrid`. |
| `topology.branching_factor` | int | no | Override branching factor for `hierarchical_tree`. Default: `floor(N_max^flat)`. Ignored for other topology kinds. |
| `explorers.count` | int | yes | Requested Explorer count. System will reduce if above `N_max`. |
| `explorers.tau_min` | float | no | Minimum temperature. Default: 0.2. Ignored when `roles[]` is non-empty. |
| `explorers.tau_max` | float | no | Maximum temperature. Default: 0.9. Ignored when `roles[]` is non-empty. |
| `explorers.roles[]` | RoleSpec[] | no | Role-typed Explorer specs. Triggers Team-Swarm Hybrid topology; overrides `tau_min`/`tau_max`. Each entry has `agent_id`, `role` (see AgentRole), optional `tau` and `role_error_cost`. |
| `explorers.review_gates[]` | ReviewGate[] | no | Dependency edges between Explorers. Each entry has `reviewer` (agent_id of Evaluator-role Explorer) and `blocks` (agent_id of blocked Explorer). Only valid with Team-Swarm Hybrid. |
| `constraints` | string[] | no | ADR identifiers to explicitly include. The compiler always includes the full corpus; this field pins specific ADRs into `system_context`. |
| `context` | string | no | Additional explicit context not captured in ADRs. Included verbatim in `system_context`. |

**AgentRole values:**

| Role | τ default | c_i default | Description |
|---|---|---|---|
| `"Coordinator"` | 0.05 | 0.1 | Low-entropy router; assigns sub-tasks to other Explorers. Internal node in Team-Swarm Hybrid. |
| `"Executor"` | 0.40 | 0.5 | Primary output producer. Subject to review gates when configured. |
| `"Evaluator"` | 0.10 | 0.9 | Review gate; evaluates and approves/blocks Executor output before it reaches the ADR Auditor. |
| `"Synthesizer"` | 0.80 | 0.1 | Combines or summarises other outputs. High τ, low c_i. |
| `"Custom"` | (required) | (required) | Arbitrary domain role. Must supply `tau` and `role_error_cost` explicitly. |

**Example — Ensemble + CRDT:**
```json
{
  "description": "Propose a token rotation strategy for the auth service",
  "pareto_weights": {"diversity": 0.5, "containment": 0.3, "throughput": 0.2},
  "topology": {"kind": "ensemble"},
  "explorers": {"count": 4, "tau_min": 0.2, "tau_max": 0.9},
  "constraints": ["ADR-001"]
}
```

**Example — Team-Swarm Hybrid with review gate:**
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

`202 Accepted`:
```json
{
  "task_id": "task_01HXYZ...",
  "status": "accepted",
  "events_url": "/tasks/task_01HXYZ.../events",
  "topology_kind": "Ensemble",
  "n_max": 6.3,
  "interface_n_max": null
}
```

`topology_kind` values: `"Ensemble"`, `"HierarchicalTree"`, `"TeamSwarmHybrid"`. `interface_n_max` is non-null only for `TeamSwarmHybrid` — it is the binding ceiling for concurrent sub-tasks between the human liaison and the Coordinator.

`400 Bad Request` — pareto weights do not sum to 1.0.

`503 Service Unavailable` — calibration has not been run.

---

### GET /tasks/{task_id}/events

Server-Sent Events stream. Tails the NATS JetStream subject `h2ai.tasks.{task_id}` in real time.

**Stream lifecycle:**
- Opens immediately on connection.
- Each event is a JSON object on a `data:` line followed by `\n\n`.
- Stream closes on `MergeResolvedEvent` (success) or `TaskFailedEvent` (failure).
- Reconnect by re-connecting to the same URL with `Last-Event-ID: {sequence_number}` header; the server replays from that offset.

---

### GET /tasks/{task_id}

Returns the current task status without streaming.

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

Triggers the calibration harness on the current adapter pool. Measures α, β₀, and pairwise CG across all configured Explorer adapters.

**Response `202 Accepted`:**
```json
{
  "calibration_id": "cal_01HXYZ...",
  "status": "accepted",
  "events_url": "/calibrate/cal_01HXYZ.../events",
  "adapter_count": 5
}
```

---

### GET /calibrate/{calibration_id}/events

SSE stream for calibration progress. Closes on `CalibrationCompletedEvent`.

```
data: {"event_type":"CalibrationProgress","payload":{"task":1,"of":3,"adapter":"local-llama-8b"}}
data: {"event_type":"CalibrationProgress","payload":{"task":2,"of":3,"adapter":"cloud-gpt4o"}}
data: {"event_type":"CalibrationProgress","payload":{"task":3,"of":3,"adapter":"cloud-claude"}}
data: {"event_type":"CalibrationCompleted","payload":{"alpha":0.12,"beta_base":0.021,"beta_eff":0.019,"n_max":6.3,"theta_coord":0.28,"cg_mean":0.71,"cg_std_dev":0.09}}
```

---

### GET /calibrate/current

Returns the currently cached calibration coefficients.

```json
{
  "calibration_id": "cal_01HXYZ...",
  "calibrated_at": "2026-04-19T10:00:00Z",
  "alpha": 0.12,
  "beta_base": 0.021,
  "beta_eff": 0.019,
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

```json
{
  "resolution": "select",
  "selected_proposals": ["exp_A", "exp_B"],
  "synthesis_notes": "Combined exp_A's rotation strategy with exp_B's revocation approach.",
  "final_output": "string — the merged output text the human approved"
}
```

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

### GET /ready

Readiness probe. Returns `200` only if calibration data exists and NATS is reachable. Returns `503` otherwise — Kubernetes will remove the pod from the load balancer pool.

### GET /metrics

Prometheus metrics endpoint.

---

### Error Codes

| HTTP Status | Error Code | Meaning |
|---|---|---|
| 400 | `InvalidParetoWeights` | Weights do not sum to 1.0 |
| 400 | `InvalidExplorerCount` | `count` < 1 or not an integer |
| 404 | `TaskNotFound` | Unknown `task_id` |
| 404 | `CalibrationNotFound` | Unknown `calibration_id` or no calibration run |
| 409 | `TaskAlreadyResolved` | `POST /tasks/{id}/merge` on a closed task |
| 503 | `CalibrationRequiredError` | No calibration data cached |
| 503 | `NatsUnavailable` | Cannot reach NATS JetStream |

---

## Event Vocabulary

All 24 events published to `h2ai.tasks.{task_id}`. Internally-tagged JSON: `"event_type"` + `"payload"`.

| Event | Phase | Description |
|---|---|---|
| `CalibrationCompleted` | 0 | α, β₀, CG samples locked |
| `TaskBootstrapped` | 1 | `system_context` compiled and locked |
| `TopologyProvisioned` | 2 | DAG shape, τ values, merge strategy |
| `MultiplicationConditionFailed` | 2.5 | Which of 3 conditions failed |
| `Proposal` | 3 | Explorer output appended |
| `ProposalFailed` | 3 | Explorer crashed/timed out |
| `GenerationPhaseCompleted` | 3 | JoinSet drained |
| `TaoIteration` | 3 | Per-explorer per-turn TAO loop result |
| `VerificationScored` | 3.5 | LLM-judge or oracle score per proposal |
| `ReviewGateTriggered` | 3b | Evaluator reviewing Executor proposal (TeamSwarmHybrid) |
| `ReviewGateBlocked` | 3b | Evaluator rejected proposal |
| `Validation` | 4 | Auditor: proposal passed |
| `BranchPruned` | 4 | Auditor: proposal rejected |
| `ZeroSurvival` | 4 | All proposals pruned — MAPE-K retry |
| `InterfaceSaturationWarning` | any | Active subtasks approaching N_max^interface |
| `ConsensusRequired` | 5 | max(c_i) > 0.85, switching to BFT |
| `SemilatticeCompiled` | 5 | Merge ready |
| `MergeResolved` | 5 | Human resolution, task closed |
| `TaskFailed` | any | Retries exhausted |
| `TaskAttribution` | 5 | Q_total decomposition with CI |
| `SubtaskPlanCreated` | compound | Task decomposed into SubtaskPlan |
| `SubtaskPlanReviewed` | compound | PlanReviewer approved/rejected SubtaskPlan |
| `SubtaskStarted` | compound | Subtask dispatched in topo-sort wave |
| `SubtaskCompleted` | compound | Subtask output ready for dependents |

---

### CalibrationCompletedEvent

```json
{
  "event_type": "CalibrationCompleted",
  "payload": {
    "calibration_id": "cal_01HXYZ...",
    "coefficients": {
      "alpha": 0.12,
      "beta_base": 0.021,
      "cg_samples": [0.68, 0.74, 0.71, 0.69, 0.73],
      "sample_timestamps": [1746000000, 1746000060, 1746000120, 1746000180, 1746000240]
    },
    "coordination_threshold": 0.28,
    "ensemble": {
      "n_optimal": 5,
      "p_mean": 0.855,
      "rho_mean": 0.145
    },
    "eigen": {
      "n_effective": 3.8,
      "n_pruned": 4,
      "lambda_values": [2.1, 1.4, 0.8, 0.3]
    },
    "pairwise_beta": 0.019,
    "cg_mode": "EmbeddingCosine",
    "timestamp": "2026-05-01T14:23:01Z"
  }
}
```

---

### TopologyProvisionedEvent

```json
{
  "event_type": "TopologyProvisioned",
  "payload": {
    "task_id": "task_01HYYZ...",
    "topology_kind": "Ensemble",
    "n": 4,
    "n_max": 6.3,
    "interface_n_max": null,
    "beta_eff": 0.019,
    "theta_coord": 0.28,
    "merge_strategy": "CrdtSemilattice",
    "retry_number": 0,
    "explorers": [
      {"explorer_id": "exp_A", "tau": 0.3, "adapter_kind": "Local",  "role": null, "role_error_cost": 0.1},
      {"explorer_id": "exp_B", "tau": 0.5, "adapter_kind": "Cloud",  "role": null, "role_error_cost": 0.1},
      {"explorer_id": "exp_C", "tau": 0.7, "adapter_kind": "Cloud",  "role": null, "role_error_cost": 0.1},
      {"explorer_id": "exp_D", "tau": 0.9, "adapter_kind": "Local",  "role": null, "role_error_cost": 0.1}
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

---

### ProposalEvent

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

### BranchPrunedEvent

```json
{
  "event_type": "BranchPruned",
  "payload": {
    "task_id": "task_01HYYZ...",
    "explorer_id": "exp_C",
    "reason": "Proposes storing refresh tokens in Redis — violates ADR-001 stateless auth requirement",
    "constraint_error_cost": 0.72,
    "violated_constraints": [
      {
        "constraint_id": "ADR-001",
        "score": 0.0,
        "severity_label": "Hard",
        "remediation_hint": "Use stateless JWT tokens; do not store session data in Redis"
      }
    ],
    "timestamp": "2026-05-01T14:23:14Z"
  }
}
```

---

### ZeroSurvivalEvent

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

### VerificationScoredEvent

```json
{
  "event_type": "VerificationScored",
  "payload": {
    "task_id": "task_01HYYZ...",
    "explorer_id": "exp_B",
    "score": 0.87,
    "reason": "Proposal correctly handles idempotency via the Lua atomic check-and-set pattern.",
    "passed": true,
    "timestamp": "2026-05-01T14:23:12Z"
  }
}
```

`score` is in [0, 1]. `passed` is `score ≥ verify_threshold` (default 0.3 after MAPE-K reduction).

---

### TaskAttributionEvent

```json
{
  "event_type": "TaskAttribution",
  "payload": {
    "task_id": "task_01HYYZ...",
    "q_predicted": 0.847,
    "q_measured": 1.0,
    "q_interval_lo": 0.791,
    "q_interval_hi": 0.903,
    "prediction_basis": "Empirical",
    "timestamp": "2026-05-01T14:23:17Z"
  }
}
```

`q_measured` is non-null only when a Tier 1 oracle ran. `q_interval_lo`/`q_interval_hi` are `null` when fewer than 2 CG calibration samples are available. `prediction_basis`: `"Heuristic"` (CG-proxy chain) or `"Empirical"` (measured from oracle tasks).

---

### ReviewGateBlockedEvent

```json
{
  "event_type": "ReviewGateBlocked",
  "payload": {
    "task_id": "task_01HYYZ...",
    "gate_id": "gate_review_impl_1",
    "blocked_explorer_id": "impl_1",
    "reviewer_explorer_id": "review",
    "rejection_reason": "Proposal uses synchronous HTTP calls inside async handler — violates ADR-007",
    "blocked_at": "2026-04-19T14:23:15Z"
  }
}
```

---

### TaskFailedEvent

```json
{
  "event_type": "TaskFailed",
  "payload": {
    "task_id": "task_01HYYZ...",
    "failure_reason": "MaxRetriesExhausted",
    "retries_attempted": 3,
    "topologies_tried": ["Ensemble", "Ensemble", "HierarchicalTree"],
    "tau_ranges_tried": [[0.3,0.8],[0.1,0.95],[0.1,0.95]],
    "failed_at": "2026-04-19T14:28:00Z"
  }
}
```

`failure_reason` values: `MaxRetriesExhausted`, `MultiplicationConditionUnresolvable`, `CalibrationExpired`.

---

## Adapter Development

An adapter is any compute backend that implements the `IComputeAdapter` trait. The orchestrator calls adapters through this interface without knowing whether the backend is a local llama.cpp model, a cloud API, or something else entirely.

### IComputeAdapter Trait

Defined in `crates/h2ai-types/src/adapter.rs`:

```rust
#[async_trait]
pub trait IComputeAdapter: Send + Sync + std::fmt::Debug {
    /// Execute one inference call. Must be cancel-safe — wrapped in tokio::time::timeout.
    async fn execute(&self, request: ComputeRequest) -> Result<ComputeResponse, AdapterError>;

    /// Identifies which backend produced a response (for telemetry and AuditorConfig).
    fn kind(&self) -> &AdapterKind;
}
```

### ComputeRequest and ComputeResponse

```rust
pub struct ComputeRequest {
    pub system_context: String,  // immutable context compiled at TaskBootstrappedEvent
    pub task: String,            // task description for this Explorer
    pub tau: TauValue,           // temperature assigned by autonomic; always 0.0 for Auditor
    pub max_tokens: u64,
}

pub struct ComputeResponse {
    pub output: String,
    pub token_cost: u64,
    pub adapter_kind: AdapterKind,
}

pub enum AdapterError {
    Timeout,
    OomPanic(String),
    NetworkError(String),
    FfiError(String),
}
```

### Built-in Adapters

`h2ai-adapters` ships five concrete adapters, all wired through `AdapterFactory::build`:

**AnthropicAdapter** — POST `/v1/messages`, `x-api-key` + `anthropic-version: 2023-06-01`. Token cost = `input_tokens + output_tokens`.

**OpenAIAdapter** — POST `/chat/completions`, `Authorization: Bearer`. Token cost = `usage.total_tokens`.

**OllamaAdapter** — POST `/api/chat`, no auth. Temperature nested as `"options": {"temperature": τ}`.

**CloudGenericAdapter** — OpenAI-compatible endpoint, configurable via `H2AI_EXPLORER_ENDPOINT`.

**MockAdapter** — Deterministic test double; returns a fixed string for every request.

```rust
use h2ai_adapters::factory::AdapterFactory;
use h2ai_types::config::AdapterKind;

let kind = AdapterKind::Anthropic {
    api_key_env: "ANTHROPIC_API_KEY".into(),
    model: "claude-3-5-haiku-20241022".into(),
};
let adapter: Arc<dyn IComputeAdapter> = AdapterFactory::build(&kind)?;
```

`LocalLlamaCpp` returns `Err` — use `Ollama` with a local Ollama server for local inference until the FFI is wired.

### Local llama.cpp Adapter

Local adapters **must** use `tokio::task::spawn_blocking` — CPU-bound inference must not block the async worker pool.

```rust
#[async_trait]
impl IComputeAdapter for LocalAdapter {
    async fn execute(&self, request: ComputeRequest) -> Result<ComputeResponse, AdapterError> {
        let model_path = self.model_path.clone();
        let kind = self.kind.clone();

        tokio::task::spawn_blocking(move || {
            let output = llama_cpp_ffi::generate(
                &model_path,
                &request.system_context,
                &request.task,
                request.tau.value() as f32,
                request.max_tokens,
            ).map_err(|e| AdapterError::FfiError(e.to_string()))?;

            Ok(ComputeResponse {
                output: output.text,
                token_cost: output.token_count,
                adapter_kind: kind,
            })
        })
        .await
        .map_err(|e| AdapterError::FfiError(format!("spawn_blocking join error: {e}")))?
    }
}
```

### Selecting Adapters at Runtime

Set `H2AI_EXPLORER_PROVIDER` and `H2AI_AUDITOR_PROVIDER` to one of `anthropic`, `openai`, `ollama`, `cloud`, or `mock`.

```bash
H2AI_EXPLORER_PROVIDER=anthropic
H2AI_EXPLORER_MODEL=claude-3-5-sonnet-20241022
H2AI_EXPLORER_API_KEY_ENV=ANTHROPIC_API_KEY

H2AI_AUDITOR_PROVIDER=anthropic
H2AI_AUDITOR_MODEL=claude-3-5-haiku-20241022
H2AI_AUDITOR_API_KEY_ENV=ANTHROPIC_API_KEY
```

To add a custom adapter, add a new `AdapterKind` variant to `h2ai-types/src/config.rs` and a matching arm in `h2ai-adapters/src/factory.rs`.

### Auditor Adapter Requirements

1. **τ is always 0.0** — deterministic; the adapter must use greedy decoding.
2. **`role_error_cost` must be ≥ 0.85** — this triggers `BftConsensus` when required.
3. **The Auditor's system prompt includes the constraint validation rubric** — compiled automatically; do not override `system_context`.
4. **Use the largest, most capable model available** — the Auditor is a gate, not a draft.

### Testing Adapters

Integration tests are `#[ignore]` by default:

```bash
ANTHROPIC_API_KEY=sk-ant-... \
cargo test -p h2ai-adapters --test integration_test -- --ignored --nocapture
```

Unit tests use `wiremock` for a local HTTP server — no live credentials needed.

---

## Agent Descriptor

An **edge agent** is any LLM-based stateless container that:
1. Receives a `TaskPayload` over NATS JetStream
2. Runs inference and produces a `TaskResult`
3. Publishes telemetry to `h2ai.telemetry.{task_id}` while running
4. Terminates — no persistent state

```rust
pub struct AgentDescriptor {
    pub model: String,         // "llama3-70b", "gpt-4o", "claude-3-opus", ...
    pub tools: Vec<AgentTool>,
}

pub enum AgentTool {
    Shell,
    WebSearch,
    CodeExecution,
    FileSystem,
}
```

### Pure LLM vs. Tool-Using Agent

**Pure LLM agent (`tools: []`):**
- α ≈ 0 during generation (no shared locks)
- Error independence holds (τ spread provides decorrelation)
- c_i is low (0.1–0.3) — wrong output is discarded at zero collateral cost
- CRDT semilattice merge applies

**Tool-using agent (`tools: [Shell, ...]`):**
- α increases: `Shell` serializes repository access; `FileSystem` serializes shared paths. Pools with tool-using agents have α in the 0.20–0.30 range, which lowers N_max.
- β_eff rises from retrieval nondeterminism: two agents calling `WebSearch` at different moments may receive different results, inflating CG variance and raising β_eff = β₀ × (1 − CG_mean), lowering N_max.
- Error independence weakens: correlated tool failures push ρ toward `max_error_correlation` (0.9), triggering `MultiplicationConditionFailedEvent`.
- c_i is high (0.6–0.9) — wrong tool-using outputs may have irreversible side effects; drives MergeStrategy to BFT.

**Tool set to physics parameter mapping:**

| Tool set | Effect on α | Effect on β₀ | Default c_i | Suggested role | τ range |
|---|---|---|---|---|---|
| `[]` | +0 | +0 | 0.1–0.3 | Coordinator, Synthesizer | 0.05–0.9 |
| `[WebSearch]` | +0.01–0.02 | +0.005 | 0.2–0.4 | Evaluator | 0.05–0.3 |
| `[FileSystem]` | +0.02–0.05 | +0.010 | 0.4–0.6 | Executor | 0.3–0.6 |
| `[CodeExecution]` | +0.03–0.08 | +0.015 | 0.5–0.7 | Executor | 0.3–0.6 |
| `[Shell]` | +0.05–0.15 | +0.020 | 0.6–0.9 | Executor | 0.3–0.5 |
| `[Shell, CodeExecution, FileSystem]` | +0.08–0.20 | +0.025 | 0.7–0.9 | Executor | 0.3–0.5 |

These are default ranges. The calibration harness measures actual α and CG for your specific adapter pool. The table is a prior — the harness is ground truth.

**BFT trigger:** If any agent has `c_i > 0.85`, `MergeStrategy` switches to BFT. Observe as `ConsensusRequiredEvent` in the task stream.

### How the Framework Uses the Descriptor

**`TopologyPlanner::provision`** — Reads `descriptor.tools` per role and applies default c_i if none is declared in the manifest. If `Shell` or `CodeExecution` is present in a TeamSwarmHybrid without a Review Gate, logs a warning.

**`KubernetesProvider::ensure_agent_capacity`** — Maps `descriptor.model` → container image; maps `descriptor.tools` → volume mounts and security contexts:
- `Shell` → writable workspace mount, `capabilities: {add: ["SYS_PTRACE"]}`
- `CodeExecution` → isolated sandbox volume, CPU/memory resource limits
- `FileSystem` → writable shared workspace mount
- `WebSearch` → egress NetworkPolicy allowing outbound HTTPS
- `[]` → no additional mounts, minimal attack surface

**`CalibrationHarness::run`** — Measures α and CG for the actual adapter pool. If tool-using adapters are present, measured α reflects their serialization cost. `N_max = sqrt((1−alpha) / beta_eff)` automatically accounts for tool overhead.

---

## Constraint Corpus

The Dark Knowledge Compiler turns your team's architectural decisions into machine-checkable rules. Constraints drive calibration (CG is Hamming distance on constraint-satisfaction fingerprints), Auditor gating (`BranchPrunedEvent`), and MAPE-K hints (`RetryWithHints`).

### Constraint Document Format

```markdown
# CONSTRAINT-{id}: {short title}

## Severity
Hard threshold=0.9

## Predicate
VocabularyPresence AllOf
- stateless
- jwt
- no session state

## Remediation
Ensure the proposal explicitly states that auth is stateless and does not store
session state.
```

**Severity options:**
- `Hard threshold=<float>` — blocks merge if score < threshold
- `Soft weight=<float>` — contributes to weighted soft score
- `Advisory` — informational only; never blocks

**Predicate options:**
- `VocabularyPresence AllOf|AnyOf|NoneOf` + bullet terms
- `NegativeKeyword` + bullet terms (fails if any term appears)
- `RegexMatch must_match=true|false` + a single regex pattern bullet
- `NumericThreshold field=<regex> op=lt|le|eq|ge|gt value=<float>`
- `LlmJudge` + rubric text (evaluated async via the auditor adapter)

Documents with only a `## Constraints` section are also valid — parsed as `VocabularyPresence { AllOf }` with `Hard { threshold: 0.8 }`.

### Compliance Formula

```
score_i        ∈ [0.0, 1.0]   (per predicate; VocabularyPresence AllOf is fractional)
hard_gate      = all Hard predicates have score_i ≥ threshold_i
soft_score     = Σ(w_i × score_i) / Σw_i   (Soft constraints only)
compliance     = if hard_gate { soft_score } else { 0.0 }
error_cost     = 1.0 − compliance            (recorded in BranchPrunedEvent)
```

`VocabularyMode` semantics:
- **`AllOf`** — fractional: `hits / total_terms`
- **`AnyOf`** — binary 1.0 if any term appears
- **`NoneOf`** — binary 1.0 if no term appears

### Writing Effective Constraints

Always prefer prohibitions over descriptions:

```markdown
## Predicate
VocabularyPresence AllOf
- jwt
- stateless
- no session state
- token expiry

## Remediation
The proposal must state auth is stateless and specify token expiry. No session
storage is permitted.
```

Always add a `## Remediation` hint to Hard constraints — without them the MAPE-K loop falls back to generic τ adjustment instead of targeted `RetryWithHints`.

### Corpus Organization

```
constraints/
├── architecture/
│   ├── CONSTRAINT-001-stateless-auth.md
│   └── CONSTRAINT-007-no-direct-db-access.md
├── security/
│   └── CONSTRAINT-010-gdpr-logging.md
└── deprecated/
    └── CONSTRAINT-005-redis-session-store.md  # Status: Deprecated
```

The compiler scans the entire directory recursively. Deprecated constraints are still indexed — they teach the Auditor about explicitly reversed decisions.

### Minimum Viable Corpus

Five constraints covering the highest ROI:
1. **Authentication and session management** — token lifecycle, revocation policy
2. **Database access policy** — which services can access which databases
3. **Service boundary rules** — synchronous vs. async communication
4. **Error handling and propagation** — retry policies, circuit breakers
5. **Sensitive data handling** — PII storage and logging requirements

---

## Configuration

H2AI Control Plane is configured via a three-layer stack: `reference.toml` (embedded defaults) → override TOML file (`H2AI_CONFIG` env var or `./h2ai.toml`) → `H2AI_<FIELD>` env vars (highest priority). See `crates/h2ai-config/reference.toml` for the canonical source of all defaults.

### Core Environment Variables

| Variable | Default | Description |
|---|---|---|
| `H2AI_PLAN` | `local` | Deployment plan hint (`local`, `server`, `cloud`). Affects default log verbosity and startup checks. |
| `H2AI_LISTEN_ADDR` | `0.0.0.0:8080` | HTTP bind address for the axum API gateway. |
| `H2AI_METRICS_ADDR` | `0.0.0.0:9090` | Prometheus `/metrics` bind address. Empty string to disable. |

### NATS

| Variable | Default | Description |
|---|---|---|
| `H2AI_NATS_URL` | `nats://localhost:4222` | NATS server URL. Comma-separate for clusters. |
| `H2AI_NATS_STREAM_NAME` | `H2AI_TASKS` | JetStream stream name for task events. |
| `H2AI_NATS_KV_BUCKET` | `H2AI_CALIBRATION` | KV bucket name for calibration cache. |
| `H2AI_NATS_STREAM_REPLICAS` | `1` | Stream replication factor. Set to `3` for Server/Cloud Plan clusters. |

### Runtime

| Variable | Default | Description |
|---|---|---|
| `H2AI_MAX_RETRIES` | `3` | Maximum MAPE-K retry cycles per task before `TaskFailedEvent`. |
| `H2AI_MAX_BLOCKING_THREADS` | `4` | Tokio blocking thread pool size for llama.cpp FFI. Calibrate to `floor(available_ram_gb / model_size_gb)`. |
| `H2AI_EXPLORER_TIMEOUT_SECS` | `120` | Wall time limit per Explorer call. Exceeded → `ProposalFailedEvent`. |
| `H2AI_BFT_THRESHOLD` | `0.85` | `max(c_i)` above which `MergeStrategy` switches to `BftConsensus`. |

### Physics Config (`H2AIConfig`)

#### USL + Multiplication Condition

| Field | Default | Description |
|---|---|---|
| `bft_threshold` | `0.85` | `max(c_i)` above which `MergeStrategy` switches to BFT consensus. |
| `krum_fault_tolerance` | `0` | Byzantine fault tolerance bound for Krum/Multi-Krum. `0` = Krum disabled. `n` = tolerate up to `n` Byzantine explorers; requires ≥ 2n+3 explorers. |
| `coordination_threshold_max` | `0.3` | Cap on computed θ_coord. |
| `min_baseline_competence` | `0.3` | Minimum competence threshold for Multiplication Condition (p > 0.5 proxy). |
| `max_error_correlation` | `0.9` | Maximum error correlation for Multiplication Condition. |
| `alpha_contention` | `0.12` | α contention constant: serial fraction that cannot be parallelized. |
| `beta_base_default` | `0.039` | β₀ base coherency cost per agent pair. Fallback when < 3 adapters available. At CG=0.4: β_eff = 0.039×(1−0.4) = 0.0234 → N_max ≈ 6. |
| `diversity_threshold` | `0.0` | Mean pairwise Hamming distance below which the swarm is flagged collectively hallucinated → `ZeroSurvivalEvent`. `0.0` disables. Recommended: `0.15`. |
| `context_pressure_gamma` | `0.5` | Sensitivity of β to context window fill. `0` disables. `0.5`: β doubles when context is 100% full. |

#### Agent Role Defaults

| Field | Default | Description |
|---|---|---|
| `tau_coordinator` | `0.05` | Default τ for Coordinator role. |
| `tau_executor` | `0.40` | Default τ for Executor role. |
| `tau_evaluator` | `0.10` | Default τ for Evaluator role. |
| `tau_synthesizer` | `0.80` | Default τ for Synthesizer role. |
| `cost_coordinator` | `0.1` | Role error cost c_i for Coordinator. |
| `cost_executor` | `0.5` | Role error cost c_i for Executor. |
| `cost_evaluator` | `0.9` | Role error cost c_i for Evaluator. |
| `cost_synthesizer` | `0.1` | Role error cost c_i for Synthesizer. |

#### Token Budgets

| Field | Default | Description |
|---|---|---|
| `explorer_max_tokens` | `1024` | Token budget per Explorer generation call. |
| `calibration_max_tokens` | `256` | Token budget per calibration probe call. |
| `max_concurrent_tasks` | `8` | Maximum concurrent tasks. Beyond this limit returns `503`. |
| `task_deadline_secs` | `null` | Hard deadline per task in seconds. `null` = no deadline. |

#### MAPE-K Autonomic Loop

| Field | Default | Description |
|---|---|---|
| `max_autonomic_retries` | `2` | Maximum MAPE-K retry cycles before `TaskFailedEvent`. |
| `optimizer_threshold_step` | `0.1` | How much `SelfOptimizer` lowers `verify_threshold` per MAPE-K step. |
| `optimizer_threshold_floor` | `0.3` | Minimum `verify_threshold` `SelfOptimizer` will suggest. |
| `tao_per_turn_factor` | `0.6` | Error decay factor per TAO turn: `c_i_eff = c_i × 0.6^(t−1)`. |
| `tau_spread_max_factor` | `2.0` | Maximum τ-spread expansion factor when Talagrand detects over-confidence (U-curve). |
| `optimizer_waste_threshold` | `0.5` | Fraction of proposals that must survive verification for the run to be considered efficient. |

#### CG Measurement and Embedding

| Field | Default | Description |
|---|---|---|
| `cg_collapse_threshold` | `0.10` | When `CG_embed` drops below this, forces `N_max=1`. Below 10% agreement, pairwise reconciliation is undefined. |
| `cg_agreement_threshold` | `0.85` | Cosine similarity threshold above which two adapter outputs count as "in agreement". |
| `calibration_cg_fallback` | `0.7` | `CG_mean` fallback when embedding model is not configured. |
| `embedding_model_name` | `AllMiniLmL6V2` | Embedding model for CG cosine agreement rate. Requires `fastembed-embed` Cargo feature. |
| `baseline_accuracy_proxy` | `0.0` | Override CG-derived accuracy proxy with a directly measured per-adapter baseline. Set by running `scripts/baseline_eval.py`. `0.0` = use `0.5 + CG_mean / 2` proxy. |

#### Thompson Sampling Bandit

| Field | Default | Description |
|---|---|---|
| `bandit_phase0_k` | `10` | Tasks before activating the bandit. During Phase 0, N = N_max_USL unconditionally. |
| `bandit_phase1_k` | `30` | Tasks before switching from ε-greedy (Phase 1) to pure Thompson Sampling (Phase 2). |
| `bandit_epsilon` | `0.3` | Phase 1 exploration probability. |
| `bandit_soft_reset_decay` | `0.3` | Decay toward prior when adapter version hash changes. |

#### Calibration Adapter Pool

| Field | Default | Description |
|---|---|---|
| `calibration_adapter_count` | `3` | Adapter instances to run during calibration. Must be ≥ 3 for USL two-point fit. |
| `calibration_tau` | `0.5` | τ value for calibration probe calls. |
| `calibration_tau_spread` | `[0.3, 0.7]` | Temperature range `[τ_min, τ_max]` for calibration adapter instances. |

#### Generative Synthesis

Controls the two-stage critique-then-write pipeline (MoA-style aggregation) that runs after verification. The engine calls the synthesis adapter twice — once to critique all verified proposals, once to write a unified answer — then re-verifies the result.

| Field | Default | Description |
|---|---|---|
| `synthesis_enabled` | `true` | Enable the synthesis phase. Set `false` to reproduce pre-synthesis (selection-only) behavior. |
| `synthesis_min_proposals` | `2` | Minimum verified proposals required to attempt synthesis. |
| `synthesis_tau` | `0.2` | τ for both critique and synthesis calls. |
| `synthesis_critique_max_tokens` | `1024` | Token budget for the critique call (Stage 1). |
| `synthesis_max_tokens` | `2048` | Token budget for the synthesis call (Stage 2). |

**Cost note:** The synthesis path adds 2 LLM calls (critique + synthesis) and 1 additional verification pass per task (~30–60% cost increase). `HarnessAttribution.synthesis_gain` reports `Q(synthesis) − max(Q(individual_proposals))`.

#### Event Snapshot Store

| Field | Default | Description |
|---|---|---|
| `snapshot_interval_events` | `50` | Events published per task before a state snapshot is written to NATS KV. `0` to disable. |

#### NATS Payload Offloading

| Field | Default | Description |
|---|---|---|
| `payload_offload_threshold_bytes` | `524288` | Byte length above which `system_context` is offloaded to content-addressed store. Default 512 KB — half of NATS JetStream 1 MB limit. |

#### Multi-Family Enforcement

| Field | Default | Description |
|---|---|---|
| `allow_single_family` | `false` | When `false`, calibration aborts if all non-Mock adapters belong to the same provider family. |

### Calibration Environment Variables

| Variable | Default | Description |
|---|---|---|
| `H2AI_CALIBRATION_TASKS` | `3` | Number of representative tasks the calibration harness runs. |
| `H2AI_CALIBRATION_MAX_AGE_SECS` | `86400` | Seconds before cached calibration is considered stale. `0` to disable expiry. |

### Constraint Corpus Environment Variables

| Variable | Default | Description |
|---|---|---|
| `H2AI_CONSTRAINT_CORPUS_PATH` | `./constraints` | Path to the directory containing `ConstraintDoc` Markdown files. Scanned recursively. Reloaded on `SIGHUP`. |
| `H2AI_ADR_RELOAD_INTERVAL_SECS` | `300` | Background corpus reload interval in seconds. `0` to disable background reload. |

### LLM Adapters

| Variable | Default | Description |
|---|---|---|
| `H2AI_EXPLORER_PROVIDER` | `mock` | Adapter type: `anthropic`, `openai`, `ollama`, `cloud`, `mock` |
| `H2AI_EXPLORER_MODEL` | `gpt-4o` | Model name sent to the provider |
| `H2AI_EXPLORER_API_KEY_ENV` | `OPENAI_API_KEY` | Name of the env var holding the API key |
| `H2AI_EXPLORER_ENDPOINT` | _(provider default)_ | Override endpoint URL |
| `H2AI_AUDITOR_PROVIDER` | `mock` | Adapter type for the Auditor |
| `H2AI_AUDITOR_MODEL` | `gpt-4o` | Auditor model name |
| `H2AI_AUDITOR_API_KEY_ENV` | `OPENAI_API_KEY` | Name of the env var holding the API key |
| `H2AI_AUDITOR_ENDPOINT` | _(provider default)_ | Override endpoint URL |
| `H2AI_EXPLORER2_PROVIDER` | `same` | Second explorer for USL timing Phase B. `same` = clone of explorer adapter. |

`role_error_cost` guidance:

| Role | Typical c_i | Rationale |
|---|---|---|
| Explorer (early draft) | 0.1 | Auditor will filter errors before human sees them |
| Swarm Coordinator | 0.5–0.7 | Error multiplied across sub-group |
| Auditor | 0.9 | False positive = hallucination reaches human unfiltered |

### Observability

| Variable | Default | Description |
|---|---|---|
| `H2AI_OTEL_ENDPOINT` | _(unset)_ | OpenTelemetry collector OTLP gRPC endpoint. Unset = tracing disabled. |
| `H2AI_OTEL_SERVICE_NAME` | `h2ai-control-plane` | Service name reported in traces. |
| `RUST_LOG` | `h2ai=info` | Log filter directive. |
| `RUST_BACKTRACE` | _(unset)_ | Set to `1` for backtraces on panic. |

### Metrics Reference

All metrics exposed at `GET /metrics` in Prometheus text format.

| Metric | Type | Labels | Description |
|---|---|---|---|
| `h2ai_alpha` | Gauge | — | Contention coefficient α from last calibration |
| `h2ai_beta_base` | Gauge | — | Baseline coherency coefficient β₀ |
| `h2ai_beta_eff` | Gauge | — | Effective coherency β_eff = β₀ × (1 − CG_mean) |
| `h2ai_n_max` | Gauge | — | Scalability ceiling N_max = sqrt((1−α) / β_eff) |
| `h2ai_theta_coord` | Gauge | — | Coordination threshold θ_coord |
| `h2ai_cg_mean` | Gauge | — | Mean Common Ground across Explorer pairs |
| `h2ai_role_error_cost` | Gauge | `role`, `adapter_id` | c_i per role per adapter |
| `h2ai_tasks_active` | Gauge | — | Tasks currently in flight |
| `h2ai_tasks_total` | Counter | `status` (resolved, failed) | Total tasks processed |
| `h2ai_proposals_total` | Counter | `outcome` (validated, pruned, failed) | Total proposals processed |
| `h2ai_zero_survival_total` | Counter | — | Total `ZeroSurvivalEvent` occurrences |
| `h2ai_autonomic_retries_total` | Counter | `action` | MAPE-K retry iterations |
| `h2ai_multiplication_condition_failures_total` | Counter | `condition` | Phase 2.5 gate failures per condition |
| `h2ai_merge_strategy_total` | Counter | `strategy` | How often each merge strategy is selected |
| `h2ai_calibration_age_seconds` | Gauge | — | Seconds since last successful calibration |
| `h2ai_adapter_latency_seconds` | Histogram | `adapter_id`, `adapter_kind` | Per-adapter inference latency |
| `h2ai_blocking_threads_active` | Gauge | — | Active Tokio blocking threads (llama.cpp FFI pool) |
| `h2ai_nats_publish_latency_seconds` | Histogram | — | NATS event publish latency |

### Helm Values Reference

Key overrides for enterprise deployments:

```yaml
replicaCount: 4

config:
  maxBlockingThreads: 16
  maxRetries: 5
  logLevel: "h2ai=warn"

nats:
  enabled: false
  natsUrl: nats://my-nats-cluster.internal:4222

serviceMonitor:
  enabled: true
  labels:
    release: kube-prometheus-stack

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: h2ai.corp.example.com
      paths:
        - path: /
          pathType: Prefix

resources:
  requests:
    cpu: 500m
    memory: 1Gi
  limits:
    cpu: "4"
    memory: 8Gi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 20
```
