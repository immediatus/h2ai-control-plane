# H2AI Reference

Operational surface of the control plane: HTTP API, event vocabulary, Prometheus metrics, configuration, adapters, agent descriptor, and constraint corpus. Authoritative source for every wire format. Defaults come from `crates/h2ai-config/reference.toml`; field semantics come from the Rust types in `crates/h2ai-types`.

---

## 1. HTTP API

The Axum router is wired in `crates/h2ai-api/src/routes/mod.rs`. Authentication is **not** built into the control plane; it is expected at the ingress layer (mTLS, JWT validation, OAuth2 proxy). All endpoints assume requests are pre-authenticated.

### Endpoint summary

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/tasks` | Submit a task manifest. |
| `GET` | `/tasks/:task_id/events` | SSE stream of `H2AIEvent` for a task. |
| `GET` | `/tasks/:task_id` | Current task status snapshot. |
| `POST` | `/tasks/:task_id/merge` | Resolve a Merge Authority decision. |
| `GET` | `/tasks/:task_id/recover` | Trigger snapshot+replay recovery for a task. |
| `POST` | `/calibrate` | Start a calibration run. |
| `GET` | `/calibrate/:cal_id/events` | SSE stream for an in-progress calibration. |
| `GET` | `/calibrate/current` | Last completed calibration. |
| `GET` | `/health` | Liveness probe. |
| `GET` | `/ready` | Readiness probe (depends on NATS connectivity and current calibration). |
| `GET` | `/metrics` | Prometheus exposition. |

### POST /tasks

Submits a task manifest. Returns immediately with `task_id`. Progress is observed via the SSE stream.

**Request body:**

```json
{
  "description": "string",
  "pareto_weights": {"diversity": 0.5, "containment": 0.3, "throughput": 0.2},
  "topology": {"kind": "auto"},
  "explorers": {
    "count": 4,
    "tau_min": 0.2,
    "tau_max": 0.9,
    "roles": [],
    "review_gates": []
  },
  "constraints": ["ADR-001", "ADR-007"],
  "context": "optional"
}
```

`pareto_weights.{diversity, containment, throughput}` must sum to 1.0. `topology.kind` is `"auto"`, `"ensemble"`, or `"hierarchical_tree"`. When `explorers.roles[]` is non-empty the system always selects `TeamSwarmHybrid`. `explorers.count` is requested — the system reduces to `N_max` if the request exceeds the calibrated ceiling.

**Response:** `202 Accepted` with `{"task_id": "...", "events_url": "/tasks/.../events"}`.

**Submission failures:**

- `400` — malformed manifest, weights not summing to 1.0, or invalid role spec.
- `503 CalibrationRequiredError` — no current calibration in `H2AI_CALIBRATION` KV.

### GET /tasks/:task_id/events

Server-Sent Events stream of `H2AIEvent` envelopes. Each event:

```
id: <NATS sequence number>
event: <event_type>
data: {"event_type": "...", "payload": {...}}

```

Reconnect with `Last-Event-ID: <sequence>` to resume from the last seen offset.

### POST /calibrate

```bash
curl -X POST http://localhost:8080/calibrate
```

Response: `{"calibration_id": "cal_...", "status": "accepted"}`. Calibration must finish before tasks can be submitted; in-flight task requests during a calibration return `503`. The harness writes the result to `H2AI_CALIBRATION` KV; subsequent `GET /calibrate/current` returns the most recent `CalibrationCompletedEvent` payload.

When `cfg.allow_single_family = false` (default), calibration aborts with `CalibrationFailed` if all non-Mock adapters share a provider family. Set `allow_single_family = true` to proceed with a warning.

---

## 2. Event Vocabulary

The discriminated union is `H2AIEvent` in `crates/h2ai-types/src/events.rs`. All variants are tagged-and-content serialised: `{"event_type": "...", "payload": {...}}`. Every payload field added since the initial release uses `#[serde(default)]`, so old serialised events remain readable.

### Variant index

| Event | Payload type | Phase |
|---|---|---|
| `CalibrationCompleted` | `CalibrationCompletedEvent` | calibration |
| `CalibrationFailed` | `{calibration_id, reason}` | calibration |
| `TaskBootstrapped` | `TaskBootstrappedEvent` | 1 |
| `TopologyProvisioned` | `TopologyProvisionedEvent` | 2 |
| `MultiplicationConditionFailed` | `MultiplicationConditionFailedEvent` | 2.5 / 2.6 |
| `Proposal` | `ProposalEvent` | 3 |
| `ProposalFailed` | `ProposalFailedEvent` | 3 |
| `GenerationPhaseCompleted` | `GenerationPhaseCompletedEvent` | 3 |
| `TaoIteration` | `TaoIterationEvent` | 3 |
| `Validation` | `ValidationEvent` | 3.5 |
| `VerificationScored` | `VerificationScoredEvent` | 3.5 |
| `BranchPruned` | `BranchPrunedEvent` | 3.5 / 4 |
| `ReviewGateTriggered` | `ReviewGateTriggeredEvent` | 3.5 |
| `ReviewGateBlocked` | `ReviewGateBlockedEvent` | 3.5 |
| `ConsensusRequired` | `ConsensusRequiredEvent` | 5 |
| `SemilatticeCompiled` | `SemilatticeCompiledEvent` | 5 |
| `MergeResolved` | `MergeResolvedEvent` | 5 |
| `ZeroSurvival` | `ZeroSurvivalEvent` | 5 → MAPE-K |
| `EpistemicYield` | `EpistemicYieldEvent` | post-merge (async) |
| `TaskAttribution` | `TaskAttributionEvent` | post-merge |
| `TaskFailed` | `TaskFailedEvent` | terminal |
| `InterfaceSaturationWarning` | `InterfaceSaturationWarningEvent` | any |
| `SubtaskPlanCreated` | `SubtaskPlanCreatedEvent` | planner |
| `SubtaskPlanReviewed` | `SubtaskPlanReviewedEvent` | planner |
| `SubtaskStarted` | `SubtaskStartedEvent` | subtask |
| `SubtaskCompleted` | `SubtaskCompletedEvent` | subtask |

### Key payloads

#### CalibrationCompletedEvent

```rust
struct CalibrationCompletedEvent {
    calibration_id: TaskId,
    coefficients: CoherencyCoefficients,            // alpha, beta_base, cg_samples, sample_timestamps
    coordination_threshold: CoordinationThreshold,
    ensemble: Option<EnsembleCalibration>,          // None when < 2 adapters
    eigen: Option<EigenCalibration>,                // None when < 2 adapters
    timestamp: DateTime<Utc>,
    pairwise_beta: Option<f64>,                     // β₀ from pairwise CG timing loop
    cg_mode: CgMode,                                // ConstraintProfile | EmbeddingCosine
    adapter_families: Vec<String>,
    explorer_verification_family_match: bool,       // judge bias warning
    single_family_warning: bool,                    // BFT diversity warning
    n_max_lo: f64,                                  // n_max(CG_mean − cg_std_dev)
    n_max_hi: f64,                                  // n_max(CG_mean + cg_std_dev)
    n_eff_cosine_prior: f64,                        // pool diversity prior
}
```

#### TopologyProvisionedEvent

```rust
struct TopologyProvisionedEvent {
    task_id: TaskId,
    topology_kind: TopologyKind,                    // Ensemble | HierarchicalTree | TeamSwarmHybrid
    explorer_configs: Vec<ExplorerConfig>,
    auditor_config: AuditorConfig,
    n_max: f64,
    interface_n_max: Option<f64>,
    beta_eff: f64,
    role_error_costs: Vec<RoleErrorCost>,
    merge_strategy: MergeStrategy,                  // ScoreOrdered | ConsensusMedian | OutlierResistant{f}
    coordination_threshold: CoordinationThreshold,
    review_gates: Vec<ReviewGate>,
    retry_count: u32,
    timestamp: DateTime<Utc>,
    constraint_tombstone: Option<String>,           // Some(_) only on ConstrainedExploration retries
}
```

#### ZeroSurvivalEvent

```rust
struct ZeroSurvivalEvent {
    task_id: TaskId,
    retry_count: u32,
    timestamp: DateTime<Utc>,
    n_eff_cosine_actual: Option<f64>,               // None when no EmbeddingModel configured
    failure_mode: Option<FailureMode>,              // ConstrainedExploration | ModeCollapse
}
```

#### TaoIterationEvent

Emitted once per TAO agent turn, capturing the tool call and its output for the audit trail.

```rust
struct TaoIterationEvent {
    task_id: TaskId,
    iteration: u8,
    tool_calls: Vec<ToolCallRecord>,
    total_token_cost: u32,
}

struct ToolCallRecord {
    tool: AgentTool,       // Shell | WebSearch | FileSystem | CodeExecution
    input_json: String,    // JSON payload sent to the executor
    output: String,        // executor result string
    iteration: u8,         // which TAO turn this call occurred in
}
```

`tool_calls` is `#[serde(default, skip_serializing_if = "Vec::is_empty")]` — turns with no tool calls serialise without the field, keeping old events readable.

#### EpistemicYieldEvent

```rust
struct EpistemicYieldEvent {
    task_id: TaskId,
    n_eff_cosine_actual: f64,
    n_eff_prior: f64,
    yield_ratio: f64,                               // n_eff_actual / N_requested
    adapters: Vec<String>,
}
```

Published asynchronously after `MergeResolved`. Never blocks task close.

#### MultiplicationConditionFailedEvent

`failure` is one of:

```rust
InsufficientCompetence { actual, required }
InsufficientDecorrelation { actual, threshold }
CommonGroundBelowFloor { cg_mean, theta }
InsufficientPoolDiversity { n_eff, threshold }    // Phase 2.6
```

#### TaskAttributionEvent

```rust
struct TaskAttributionEvent {
    task_id: TaskId,
    q_predicted: f64,
    q_measured: Option<f64>,                        // Tier 1 oracle pass-rate; None when no oracle
    q_interval_lo: Option<f64>,                     // 5th percentile (bootstrap or conformal)
    q_interval_hi: Option<f64>,                     // 95th percentile
    prediction_basis: PredictionBasis,              // Heuristic | Empirical
    waste_ratio: f64,                               // valid / total_evaluated
    applied_optimizations: Vec<AppliedOptimization>,
    timestamp: DateTime<Utc>,
}
```

---

## 3. Prometheus Metrics

The `/metrics` endpoint exposes exactly the counters and gauges defined in `crates/h2ai-api/src/metrics.rs`:

| Metric | Type | Meaning |
|---|---|---|
| `h2ai_n_eff_prior` | gauge | Effective independent adapters from the most recent calibration (cosine N_eff prior). |
| `h2ai_n_eff_actual` | gauge | Effective independent adapters from the most recent task's surviving wave (cosine N_eff actual). |
| `h2ai_epistemic_yield_ratio` | gauge | `n_eff_actual / N_requested` from the most recent `EpistemicYieldEvent`. |
| `h2ai_mapek_interventions_total{failure_mode="mode_collapse"}` | counter | Cumulative `ModeCollapse` MAPE-K interventions (adapter rotation). |
| `h2ai_mapek_interventions_total{failure_mode="constrained_exploration"}` | counter | Cumulative `ConstrainedExploration` interventions (tombstone injection). |

These five series cover the bivariate-CG control-loop signals. `h2ai_n_eff_prior` is updated on every `CalibrationCompletedEvent`. The other four are updated on every task that reaches a MAPE-K decision or successful merge.

---

## 4. Configuration

H2AI is configured by a layered stack:

1. `crates/h2ai-config/reference.toml` (embedded defaults, single source of truth).
2. An override TOML file (`H2AI_CONFIG=/path/to/h2ai.toml` or `./h2ai.toml`).
3. `H2AI_<FIELD_NAME>` environment variables (highest priority).

The TOML key is the lower-snake-case Rust field name. The env-var key is the upper-snake-case field name with an `H2AI_` prefix. Examples below quote the TOML form.

### Physics and gating

| Field | Default | Purpose |
|---|---|---|
| `alpha_contention` | `0.12` | USL serial fraction α. |
| `beta_base_default` | `0.039` | β₀ baseline (AI-agents tier). |
| `cg_collapse_threshold` | `0.10` | Forces `N_max = 1` when CG_embed falls below this. |
| `coordination_threshold_max` | `0.3` | Cap on derived θ_coord. |
| `min_baseline_competence` | `0.3` | Phase 2.5 competence floor. |
| `max_error_correlation` | `0.9` | Phase 2.5 ρ ceiling. |
| `diversity_threshold` | `0.0` | Phase 2.6 pool guard and MAPE-K boundary. `0.0` disables both. Recommended production value: `0.5`. |
| `bft_threshold` | `0.85` | `max(c_i)` above which `MergeStrategy` switches to `ConsensusMedian`. |
| `krum_fault_tolerance` | `0` | Byzantine bound `f` for Krum/Multi-Krum. `0` = disabled. |
| `krum_threshold` | `0.95` | `max(c_i)` above which Krum is preferred over `ConsensusMedian` (only when `krum_fault_tolerance > 0`). |
| `context_pressure_gamma` | `0.5` | Sensitivity of β to context-window fill. |
| `tao_per_turn_factor` | `0.6` | Quality factor per TAO turn (heuristic prior). |

### Calibration

| Field | Default | Purpose |
|---|---|---|
| `calibration_adapter_count` | `3` | Adapter instances in the harness. < 3 falls back to defaults. |
| `calibration_tau` | `0.5` | τ for calibration probes. |
| `calibration_tau_spread` | `[0.3, 0.7]` | τ range for cloned probes. |
| `calibration_max_tokens` | `256` | Per-probe token budget. |
| `calibration_cg_fallback` | `0.7` | CG_mean used when no corpus is provided. |
| `cg_agreement_threshold` | `0.85` | Cosine threshold for "in agreement" pairs. |
| `embedding_model_name` | `"AllMiniLmL6V2"` | Embedding model. Requires `fastembed-embed` feature. |
| `eigen_n_eff_delta` | `0.05` | Min N_eff increment to include the next adapter in `n_pruned`. |
| `baseline_accuracy_proxy` | `0.0` | When > 0, switches `EnsembleCalibration` to `Empirical` basis. |
| `auto_baseline_eval` | `false` | Auto-promote to `Empirical` after `auto_baseline_eval_min_tasks`. |
| `auto_baseline_eval_min_tasks` | `50` | Threshold for auto-promotion. |
| `allow_single_family` | `false` | Allow calibration to proceed with a monoculture pool. |

### MAPE-K and self-optimizer

| Field | Default | Purpose |
|---|---|---|
| `max_autonomic_retries` | `2` | MAPE-K retry budget per task. |
| `optimizer_threshold_step` | `0.1` | `verify_threshold` step on retries. |
| `optimizer_threshold_floor` | `0.3` | `verify_threshold` floor. |
| `optimizer_waste_threshold` | `0.5` | `valid / total_evaluated` below which a successful run is "wasteful" — triggers SelfOptimizer suggestions. |
| `tao_estimator_warmup` | `20` | Observations before `TaoMultiplierEstimator` is persisted. |
| `tao_estimator_ema_alpha` | `0.05` | EMA smoothing for tao multiplier drift. |
| `tau_spread_max_factor` | `2.0` | Max τ-spread expansion under Talagrand U-curve. |

### Bandit

| Field | Default | Purpose |
|---|---|---|
| `bandit_phase0_k` | `10` | Tasks before bandit activation. |
| `bandit_phase1_k` | `30` | Tasks before pure Thompson Sampling. |
| `bandit_epsilon` | `0.3` | Phase 1 ε-greedy exploration rate. |
| `bandit_soft_reset_decay` | `0.3` | Soft-reset toward prior on adapter version change. |
| `bandit_n_max_initial` | `4` | Warm-prior seed N_max at first startup. |

### Synthesis

| Field | Default | Purpose |
|---|---|---|
| `synthesis_enabled` | `true` | Enable Phase 5a critique→synthesis→re-verify. |
| `synthesis_min_proposals` | `2` | Minimum verified candidates before synthesis runs. |
| `synthesis_tau` | `0.2` | τ for critique and synthesis adapter calls. |
| `synthesis_critique_max_tokens` | `1024` | Critique stage budget. |
| `synthesis_max_tokens` | `2048` | Synthesis stage budget. |

### Shell Tool

| Field | Default | Purpose |
|---|---|---|
| `shell_allowlist` | `[]` | Commands permitted in Normal-mode waves. Empty = unrestricted. **Not safe for production** — populate with an explicit list before deployment. |
| `shell_hardened_allowlist` | `["ls","cat","git","find","echo","pwd"]` | Commands permitted in Hardened-mode waves (`ConstrainedExploration` / `ModeCollapse`). Must be a subset of `shell_allowlist` when `shell_allowlist` is non-empty. The system emits `tracing::warn!` at boot if any entry here is absent from `shell_allowlist`. Note: with `shell_allowlist = []` (unrestricted) both modes are unrestricted — this list takes effect only once `shell_allowlist` is populated. |
| `shell_timeout_secs` | `5` | Maximum seconds a shell tool invocation may run before the process group is killed (SIGKILL to the PGID). |

`ToolRegistry::for_wave(cfg, WaveMode::Normal)` selects `shell_allowlist`; `ToolRegistry::for_wave(cfg, WaveMode::Hardened)` selects `shell_hardened_allowlist`. The `wave_mode` field on `TaskPayload` carries the per-task mode from the NATS wire; the agent dispatch loop builds a fresh registry per task.

### Web Search Tool

| Field | Default | Purpose |
|---|---|---|
| `[web_search]` | absent | Section absent = WebSearch executor not registered. |
| `web_search.api_key_env` | — | Environment variable name holding the Google Custom Search API key. `validate_tool_configs` panics at startup if this var is missing or empty when the section is present. |
| `web_search.cx_env` | — | Environment variable name holding the Google Custom Search Engine (CX) ID. Panics at startup if missing or empty when the section is present. |
| `web_search.max_results` | — | Maximum results returned per query. Capped internally at 10 (Google API hard limit). |

The live backend sends HTTPS requests to `https://www.googleapis.com/customsearch/v1`. Only registered in `WaveMode::Normal`; absent in `WaveMode::Hardened`.

### Filesystem Tool (MCP)

| Field | Default | Purpose |
|---|---|---|
| `[mcp_filesystem]` | absent | Section absent = FileSystem executor not registered. |
| `mcp_filesystem.command` | — | Executable to launch as the MCP stdio server (e.g. `"npx"` or a compiled binary). |
| `mcp_filesystem.args` | `[]` | Arguments to pass to the MCP server command. |
| `mcp_filesystem.timeout_secs` | `5` | Maximum seconds to wait for an MCP response before sending SIGKILL to the server's process group. |

The `McpExecutor` enforces a read-only policy: only `read_file` and `list_directory` operations are permitted. Any other operation name returns `ToolError::NotPermitted`. This policy is enforced in the executor layer, not in the backend, so it holds regardless of which backend is wired. Only registered in `WaveMode::Normal`.

### WASM Code Execution Tool

| Field | Default | Purpose |
|---|---|---|
| `[wasm_executor]` | absent | Section absent = CodeExecution executor not registered. |
| `wasm_executor.interpreter_wasm_path` | — | Path to the WASM binary that acts as the JavaScript interpreter sandbox. `validate_tool_configs` panics at startup if this path does not exist when the section is present. |
| `wasm_executor.fuel_budget` | — | Wasmtime fuel units allocated per execution. Fuel exhaustion is a safe termination — the engine traps without crashing the host process. |

Requires the `wasm` cargo feature. Only `language = "javascript"` is accepted; other languages return `ToolError::NotPermitted`. No WASI imports are linked — the sandbox has no network or filesystem access by design. Registered in both `WaveMode::Normal` and `WaveMode::Hardened`.

### TAO Agent

| Field | Default | Purpose |
|---|---|---|
| `agent_max_tool_iterations` | `5` | Maximum tool-call turns the TAO agent may execute per wave. Prevents runaway tool loops. |

### Token budgets and concurrency

| Field | Default | Purpose |
|---|---|---|
| `explorer_max_tokens` | `1024` | Per-explorer generation budget. |
| `max_context_tokens` | unset | Cap kept after context compaction. |
| `max_concurrent_tasks` | `8` | 503 above this. |
| `task_deadline_secs` | unset | End-to-end task deadline. |

### NATS and storage

| Field | Default | Purpose |
|---|---|---|
| `nats_url` | `"nats://localhost:4222"` | NATS server URL. |
| `payload_offload_threshold_bytes` | `524_288` | `system_context` above this bytes is offloaded to a content-addressed blob and replaced with a hash reference (`ContextPayload::Ref`). Default is half of the JetStream 1 MB message ceiling. |
| `snapshot_interval_events` | `50` | Events between task snapshots. `0` disables. |

### Scheduler

| Field | Default | Purpose |
|---|---|---|
| `scheduler_policy` | `"CostAwareSpillover"` | `CostAwareSpillover` \| `LeastLoaded`. |
| `scheduler_spillover_threshold` | `10` | Per-tier queue depth before spillover. |

### Adapter profiles

`adapter_profiles` is an array of named adapter definitions used by `TaskProfile` routing:

```toml
[[adapter_profiles]]
name = "claude-sonnet"
[adapter_profiles.kind.Anthropic]
api_key_env = "ANTHROPIC_API_KEY"
model = "claude-sonnet-4-5"
```

### Role defaults

| Field | Default τ | Default c_i |
|---|---|---|
| Coordinator | `0.05` | `0.1` |
| Executor | `0.40` | `0.5` |
| Evaluator | `0.10` | `0.9` |
| Synthesizer | `0.80` | `0.1` |

Override via `tau_<role>` / `cost_<role>` keys.

---

## 5. Adapters

The `IComputeAdapter` trait lives in `crates/h2ai-types/src/adapter.rs`:

```rust
#[async_trait]
pub trait IComputeAdapter: Send + Sync + std::fmt::Debug {
    async fn execute(&self, request: ComputeRequest) -> Result<ComputeResponse, AdapterError>;
    fn kind(&self) -> &AdapterKind;
}
```

Built-in implementations in `crates/h2ai-adapters`: `Anthropic`, `OpenAI`, `Gemini`, `Ollama`, `LlamaCpp` (over HTTP — local server or `host.docker.internal:8000`), `CloudGeneric` (OpenAI-compatible), `Mock`.

`AdapterFactory::build(&AdapterKind)` returns `Arc<dyn IComputeAdapter>`. Local adapters that block must use `tokio::task::spawn_blocking` — CPU-bound inference must not block the async worker pool.

**Auditor adapter requirements.** τ is always 0.0 (deterministic). The role's default `c_i` is high (`cost_evaluator = 0.9`) to drive `MergeStrategy::ConsensusMedian` or `OutlierResistant` on disagreement. The auditor's output must be `{approved: bool, reason: string}` JSON; non-JSON output is treated as rejected (fail-safe).

---

## 6. Agent Descriptor

```rust
pub struct AgentDescriptor {
    pub model: String,
    pub tools: Vec<AgentTool>,
}

pub enum AgentTool {
    Shell,
    WebSearch,
    CodeExecution,
    FileSystem,
}

// Carried on every TaskPayload; selects the ToolRegistry allowlist for the task.
pub enum WaveMode {
    Normal,    // uses cfg.shell_allowlist; all configured executors registered
    Hardened,  // uses cfg.shell_hardened_allowlist; only Shell and CodeExecution registered
}
```

**Backend injection pattern.** Every tool executor wraps a `Box<dyn *Backend>` trait object. In production `ToolRegistry::for_wave(cfg, mode)` wires live backends (GoogleSearchBackend, StdioMcpBackend, RealWasmBackend). In tests `ToolRegistry::for_wave_with_mocks(cfg, mode)` injects mock backends (MockSearchBackend, MockMcpBackend, MockWasmBackend) without touching env vars, the filesystem, or spawning subprocesses. Both constructors apply identical WaveMode gating logic.

The TAO agent's local tool loop runs up to `agent_max_tool_iterations` turns. Each turn: call LLM with accumulated context, parse `{"tool": "...", "input": {...}}` from the response, dispatch to `ToolRegistry`, append a `ToolCallRecord` to the audit trail. The resulting `TaskResult.tool_calls` field carries the complete iteration history.

Tool presence shifts the calibrated physics: `Shell` and `FileSystem` raise α (serialised access to shared state); `WebSearch` raises β₀ (retrieval nondeterminism inflates CG variance); `CodeExecution` raises both. Default `c_i` ranges by tool set:

| Tool set | α impact | β₀ impact | Default c_i | Suggested role |
|---|---|---|---|---|
| `[]` | +0 | +0 | 0.1–0.3 | Coordinator, Synthesizer |
| `[WebSearch]` | +0.01–0.02 | +0.005 | 0.2–0.4 | Evaluator |
| `[FileSystem]` | +0.02–0.05 | +0.010 | 0.4–0.6 | Executor |
| `[CodeExecution]` | +0.03–0.08 | +0.015 | 0.5–0.7 | Executor |
| `[Shell]` | +0.05–0.15 | +0.020 | 0.6–0.9 | Executor |
| `[Shell, CodeExecution, FileSystem]` | +0.08–0.20 | +0.025 | 0.7–0.9 | Executor |

These are priors; the calibration harness measures actual α and β₀ for the deployed pool.

`KubernetesProvider::ensure_agent_capacity` maps `tools` to volume mounts and security contexts: `Shell` → writable workspace + `SYS_PTRACE`; `CodeExecution` → isolated sandbox volume with CPU/memory limits; `FileSystem` → shared writable mount; `WebSearch` → egress NetworkPolicy; `[]` → minimal attack surface.

---

## 7. Constraint Corpus

The corpus is a directory of markdown files. The Constraint Compiler reads them recursively and produces machine-checkable predicates.

### Format

```markdown
# CONSTRAINT-001: Stateless Authentication

## Severity
Hard threshold=0.9

## Predicate
VocabularyPresence AllOf
- jwt
- stateless
- no session state

## Remediation
The proposal must state authentication is JWT-based and stateless.
```

**Severity:** `Hard threshold=<float>` (blocks merge when `score < threshold`), `Soft weight=<float>` (contributes to weighted soft score), or `Advisory` (informational).

**Predicate kinds:**

- `VocabularyPresence AllOf|AnyOf|NoneOf` + bullet terms.
- `NegativeKeyword` + bullet terms (fails if any term appears).
- `RegexMatch must_match=true|false` + a single regex bullet.
- `NumericThreshold field=<regex> op=lt|le|eq|ge|gt value=<float>`.
- `LlmJudge` + rubric text (evaluated async via the auditor adapter).

A document with only a `## Constraints` section is parsed as `VocabularyPresence AllOf` with `Hard { threshold: 0.8 }`.

### Compliance

```
score_i     ∈ [0, 1]   (per-predicate; AllOf is fractional hits/total)
hard_gate   = all Hard predicates have score_i ≥ threshold_i
soft_score  = Σ(w_i × score_i) / Σ w_i      over Soft constraints
compliance  = soft_score if hard_gate else 0.0
error_cost  = 1 − compliance                  (recorded on BranchPrunedEvent)
```

### Operational guidance

- Always add `## Remediation` to Hard constraints — without it the MAPE-K loop cannot synthesise a targeted hint.
- Deprecated constraints should remain in the corpus under a `deprecated/` subdirectory; they teach the auditor about explicitly reversed decisions.
- A minimum viable corpus covers: authentication and session lifecycle, database access policy, service boundary rules (sync vs async), error handling and retries, sensitive-data handling.
