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
| `POST` | `/tasks/:task_id/approve` | Submit HITL approval decision (`approved`, `reviewer_note`, `operator_id`). |
| `POST` | `/tasks/:task_id/clarify` | Submit operator answer to a pending oracle clarification. |
| `GET` | `/tasks/:task_id/approval` | Current `ApprovalRecord` for tasks in `AwaitingApproval` phase; 404 otherwise. |
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
  "constraint_tags": ["eu_data", "financial_report"],
  "require_approval": false,
  "context": "optional",
  "tenant_id": "default"
}
```

`pareto_weights.{diversity, containment, throughput}` must sum to 1.0. `topology.kind` is `"auto"`, `"ensemble"`, or `"hierarchical_tree"`. When `explorers.roles[]` is non-empty the system always selects `TeamSwarmHybrid`. `explorers.count` is requested — the system reduces to `N_max` if the request exceeds the calibrated ceiling.

`constraint_tags` routes the task to a domain-specific subset of the constraint corpus via the wiki index (see §7). `constraints` provides explicit constraint IDs that are always included regardless of tags. `require_approval` forces a HITL review gate after merge even when `q_confidence` is high (see §9.5). `tenant_id` is optional; when omitted it defaults to `"default"`. It scopes all `TaskReasoningCheckpoint` and `TaskMetaState` writes to a per-tenant NATS KV bucket (see §10).

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

### POST /tasks/:task_id/clarify

Resumes an engine suspended by a `PendingClarificationEvent`. The engine waits up to `oracle_gate.timeout_secs` for an answer before timing out with `on_timeout` behavior.

**Request body:**
```json
{ "answer": "string" }
```

**Response:** `200 OK` when the answer was delivered; `404` when no pending clarification exists for this task; `503` when the task is not in a clarifiable state.

### POST /calibrate

```bash
curl -X POST http://localhost:8080/calibrate
```

Response: `{"calibration_id": "cal_...", "status": "accepted"}`. Calibration must finish before tasks can be submitted; in-flight task requests during a calibration return `503`. The harness writes the result to `H2AI_CALIBRATION` KV; subsequent `GET /calibrate/current` returns the most recent `CalibrationCompletedEvent` payload.

With `family_constraint = "require_diverse"` (the production/strict default), calibration aborts with `CalibrationFailed` if all non-Mock adapters share a provider family. The development default is `"single_family_ok"` — use that to proceed with a warning.

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
| `SelectionResolved` | `SelectionResolvedEvent` | 5 |
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
| `PendingApproval` | `PendingApprovalEvent` | post-merge (HITL gate) |
| `ApprovalResolved` | `ApprovalResolvedEvent` | post-approval |
| `CoherenceIncomplete` | `CoherenceIncompleteEvent` | post-merge (observability) |
| `CorrelatedEnsemble` | `CorrelatedEnsembleWarning` | 3.1 (C1 detection) |
| `CorrelatedFabrication` | `CorrelatedFabricationEvent` | 3.2 (SRANI) |
| `ResearcherGrounding` | `ResearcherGroundingEvent` | 3.1 / 3.2 / proactive |
| `DiversityGuardDegraded` | `DiversityGuardDegradedEvent` | 2.6 (domain coverage) |
| `OracleGateResult` | `OracleGateResultEvent` | 4.5 |
| `PendingClarification` | `PendingClarificationEvent` | 4.5 |
| `OproTriggered` | `OproTriggeredEvent` | post-merge (async) |
| `PromptVariantPromoted` | `PromptVariantPromotedEvent` | post-merge (async) |

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
InsufficientPoolDiversity { n_eff, threshold }          // Phase 2.6
VerifierExplorerFamilyConflict { explorer_family, verifier_family }   // hard gate, pre-MAPE-K
```

`VerifierExplorerFamilyConflict` is evaluated once, before the MAPE-K retry loop, in `h2ai-orchestrator/src/engine.rs`. It fires when `calibration.explorer_verification_family_match = true` and `cfg.safety.family_constraint = RequireDiverse` (production/strict default). Unlike the other variants — which may be resolved by MAPE-K retries — this variant marks the task permanently failed. No retry can resolve a deployment topology where the verification judge and the explorer pool share a provider family. The fix is a configuration change: route the verification adapter to a different model family and recalibrate.

#### TaskAttributionEvent

```rust
struct TaskAttributionEvent {
    task_id: TaskId,
    q_confidence: f64,          // heuristic confidence estimate (self-assessment, not oracle quality)
    q_measured: Option<f64>,                        // Tier 1 oracle pass-rate; None when no oracle
    q_interval_lo: Option<f64>,                     // 5th percentile (bootstrap or conformal)
    q_interval_hi: Option<f64>,                     // 95th percentile
    prediction_basis: PredictionBasis,              // Heuristic | Empirical
    waste_ratio: f64,                               // valid / total_evaluated
    applied_optimizations: Vec<AppliedOptimization>,
    approval_decision: Option<ApprovalDecision>,    // set when HITL gate was triggered
    timestamp: DateTime<Utc>,
}
```

#### PendingApprovalEvent

Emitted when the HITL gate fires after merge. Streams immediately to all connected SSE clients.

```rust
struct PendingApprovalEvent {
    task_id: TaskId,
    proposed_output: String,
    q_confidence: f64,
    risk_level: ApprovalRiskLevel,   // Medium | High (Low is never assigned)
    triggered_by: ApprovalTrigger,   // ManifestFlag | LowConfidence
    timeout_at_ms: u64,
    timestamp_ms: u64,
}
```

#### ApprovalResolvedEvent

Emitted by `POST /tasks/{id}/approve` after the CAS delete succeeds.

```rust
struct ApprovalResolvedEvent {
    task_id: TaskId,
    approved: bool,
    operator_id: String,
    reviewer_note: Option<String>,
    decided_at_ms: u64,
}
```

#### CoherenceIncompleteEvent

Emitted at task close when the surviving ensemble does not reach coherent closure over the constraint corpus. Non-blocking — the task still succeeds.

```rust
struct CoherenceIncompleteEvent {
    task_id: TaskId,
    uncovered_domains: Vec<String>,         // constraint domains where pruned proposals had violations
    active_contradictions: Vec<(String, String, String)>, // (explorer_a_id, explorer_b_id, domain)
    retries: u32,
    timestamp: DateTime<Utc>,
}
```

#### CorrelatedEnsembleWarning

Emitted (Phase 3.1) when the token-Jaccard pairwise CV check detects semantic clustering. Fires only when BOTH conditions hold: `cv < correlated_hallucination_cv_threshold` AND `mean_jaccard_distance < correlated_hallucination_min_jaccard_floor`.

```rust
struct CorrelatedEnsembleWarning {
    task_id: TaskId,
    cv: f64,                    // coefficient of variation of pairwise Jaccard distances
    mean_jaccard_distance: f64, // mean pairwise Jaccard distance across all proposal pairs
    retry_count: u32,
}
```

#### CorrelatedFabricationEvent

Emitted (Phase 3.2) when SRANI detects shared ungrounded architectural entities across proposals. CFI = max pairwise overlap of per-proposal ungrounded entity sets (absent from the task spec).

```rust
struct CorrelatedFabricationEvent {
    task_id: TaskId,
    cfi: f64,                               // Correlated Fabrication Index ∈ [0, 1]
    injection_pressure: f64,               // sigmoid((CFI − μ) / T); 0.0 when adaptive=false
    shared_ungrounded_entities: Vec<String>,
    proposal_count: usize,
    hint_injected: bool,
    timestamp: DateTime<Utc>,
}
```

#### ResearcherGroundingEvent

Emitted when external grounding is fetched — either reactively (C1/SRANI retry) or proactively (slot with `search_enabled: true`).

```rust
struct ResearcherGroundingEvent {
    task_id: TaskId,
    shared_assumption: String,    // assumption detected among correlated proposals (empty for proactive)
    literature_summary: String,
    slot: Option<String>,         // Some("slot_N") for proactive pre-steps; None for reactive
    source: GroundingSource,      // SpecAnchor | LlmResearcher (default) | WebSearch
}
```

#### DiversityGuardDegradedEvent

Emitted (Phase 2.6) when the union of slot `constraint_domains` covers less than `domain_coverage_threshold` of the corpus domain set.

```rust
struct DiversityGuardDegradedEvent {
    task_id: TaskId,
    reason: String,               // e.g. "coverage 0.25 < threshold 0.40"
    coverage_score: f64,
    slot_domains: Vec<String>,    // all domain tags assigned across all slots (flattened)
}
```

#### OracleGateResultEvent

Emitted (Phase 4.5) when the oracle gate responds. `oracle_gate_passed: Option<bool>` on `MergeResolvedEvent` reflects this result.

```rust
struct OracleGateResultEvent {
    task_id: TaskId,
    gate_passed: bool,
    confidence: f64,
    summary: String,
    checked_proposals: u32,
    passed_proposals: u32,
    timestamp: DateTime<Utc>,
}
```

#### PendingClarificationEvent

Emitted (Phase 4.5) when the oracle gate fails with low confidence and a matching `ClarificationTemplate` fires. The engine suspends via `clarification_waiters`; `POST /tasks/{id}/clarify` resumes it.

```rust
struct PendingClarificationEvent {
    task_id: TaskId,
    question: String,
    context: String,
    timeout_secs: u64,
    timestamp: DateTime<Utc>,
}
```

#### MergeResolvedEvent (updated field)

`MergeResolvedEvent` now carries `oracle_gate_passed: Option<bool>`:

- `Some(true)` — oracle gate ran and approved the surviving proposals.
- `Some(false)` — oracle gate ran and rejected; clarification was supplied or `on_timeout = "fail"` was overridden.
- `None` — oracle gate was disabled, timed out with `on_timeout = "skip"`, or was bypassed.

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
| `family_constraint` | `"single_family_ok"` | `"single_family_ok"` \| `"require_diverse"` \| `"disabled"`. Production profile sets `"require_diverse"`. |

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

### Correlated Hallucination and SRANI (C1 / GAP-C1)

| Field | Default | Purpose |
|---|---|---|
| `correlated_hallucination_cv_threshold` | `0.30` | CV of pairwise Jaccard distances below which C1 fires. Set to `0.0` to disable C1 entirely. |
| `correlated_hallucination_min_jaccard_floor` | `0.50` | Mean pairwise Jaccard distance must also be **below** this floor for C1 to fire. Joint AND condition prevents spurious retries on genuinely-diverse equidistant ensembles (CV=0 but all distances high). |
| `correlated_hallucination_min_proposals` | `2` | Minimum proposals required before C1 check runs. |
| `domain_coverage_threshold` | `0.40` | Minimum fraction of corpus domains that slot `constraint_domains` must cover. Below this, `DiversityGuardDegradedEvent` fires. |
| `require_bivariate_cg` | `false` | When `true`, tasks fail rather than warn when domain coverage is below threshold. |
| `[srani]` | — | SRANI correlated fabrication detection (entity-level cross-proposal overlap). |
| `srani.enabled` | `true` | Set to `false` to skip SRANI check entirely. |
| `srani.adaptive` | `true` | Use sigmoid gate with EMA-tracked midpoint (`true`) vs. static `warn_threshold`/`inject_threshold` pair (`false`). |
| `srani.ema_alpha` | `0.20` | EMA smoothing factor for adaptive midpoint. Lower = slower adaptation (longer memory horizon). 0.20 ≈ 5-task memory. |
| `srani.temperature` | `0.15` | Sigmoid temperature: controls gate sharpness. Lower = sharper cliff around midpoint. |
| `srani.gate_threshold` | `0.50` | Injection pressure above which grounding hint is injected. |
| `srani.warn_threshold` | `0.30` | CFI above which `CorrelatedFabricationEvent` is emitted (adaptive=false only). Also the cold-start midpoint lower bound. |
| `srani.inject_threshold` | `0.60` | CFI above which grounding hint is injected (adaptive=false only). Also the cold-start midpoint upper bound. |
| `srani.grounding_raw_max_chars` | `4000` | Maximum characters of raw web-search text fed into the distillation step. |
| `srani.grounding_hint_max_chars` | `1200` | Maximum characters of the grounding statement injected into the explorer hint block. |
| `srani.grounding_distill` | `true` | When `true` and a researcher adapter is available, distill raw web-search results with the LLM before injection. |

**NATS KV state:** SRANI EMA state (`srani_ema_cfi`, `srani_count`) is persisted at key `"srani_adaptive_state"` in `H2AI_ESTIMATOR`. Cold start: μ = 0.45 (midpoint of default thresholds) until count ≥ 5.

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
| `agent_max_observation_chars` | `8192` | Maximum UTF-8 byte length of a single tool observation appended to the agent context. Observations exceeding the limit are truncated with a diagnostic suffix `…[truncated N → max chars]`. Set to `0` to disable truncation entirely. Prevents `MaxTokensExceeded` on large shell or search outputs. |
| `tao.per_turn_timeout_secs` | `120` | Per-turn adapter call timeout in seconds. Increase for slow local models. Cloud models typically need 30s; 11B local models generating 1024-token responses need ≥120s. |
| `tao.repetition_threshold` | `0.92` | Token-overlap similarity threshold above which two consecutive responses are considered stuck (loop detected). Range [0.0, 1.0]. |

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

#### KV / Object Store buckets

| Bucket | Type | Key | TTL | Purpose |
|---|---|---|---|---|
| `H2AI_CALIBRATION` | KV | `current` | — | Latest `CalibrationCompletedEvent`. |
| `H2AI_SNAPSHOTS` | KV | `{task_id}` | — | Periodic `TaskState` snapshots for fast replay. |
| `H2AI_AGENT_MEMORY` | KV | `{session_id}` | — | Agent session memory (prior outputs). |
| `H2AI_TASK_CHECKPOINTS` | KV | `{task_id}` | 24 h | zstd-compressed `TaskCheckpoint` (phase outputs for crash recovery). |
| `H2AI_CHECKPOINT_PAYLOADS` | Object Store | SHA-256 hash | 24 h | Checkpoint payloads exceeding 800 KB (referenced by `TaskCheckpoint.object_store_ref`). |
| `H2AI_APPROVALS` | KV | `{task_id}` | 1 h | `ApprovalRecord` for tasks parked at the HITL gate. |
| `H2AI_CONSTRAINT_WIKI` | KV | `wiki_cache` | — | Serialised `WikiCache` (context_map + metas). Loaded at startup; `constraint_wiki.enabled = true` required. History=5. |
| `H2AI_CONSTRAINT_PAYLOADS` | Object Store | `{id}@{version}` | — | Full predicate payloads for non-Static constraints (LlmJudge, Oracle). Fetched lazily at Phase 4. |
| `H2AI_CHECKPOINT_{tenant_id}` | KV store | `task_id` string | 7 days | `TaskReasoningCheckpoint` (zstd-compressed). Per-tenant; bucket created on first task for each tenant. |
| `H2AI_META_{tenant_id}` | KV store | `task_id` string | no TTL | `TaskMetaState` projections (uncompressed JSON). Per-tenant; consumed by InductionScheduler (Phase 2). |

`TaskCheckpoint` schema (written after each phase boundary):

```rust
struct TaskCheckpoint {
    task_id: String,
    phase: String,               // "ParallelGeneration" | "AuditorGate" | "Merging"
    node_id: String,             // "hostname:PID" — owning node for split-brain detection
    lease_seq: u64,              // NATS KV revision at last write (used for CAS claims)
    proposals: Vec<String>,      // saved after ParallelGeneration
    auditor_survivors: Vec<usize>, // saved after AuditorGate
    resolved_output: Option<String>, // saved after Merging
    manifest_json: String,
    object_store_ref: Option<String>, // SHA-256 key in H2AI_CHECKPOINT_PAYLOADS
    created_at_ms: u64,
    updated_at_ms: u64,
    constraint_snapshot: Option<ConstraintSnapshot>, // None for pre-wiki tasks
}

struct ConstraintSnapshot {
    wiki_revision: u64,          // NATS KV revision at task creation — audit of which wiki version was active
    resolved_ids: Vec<String>,   // constraint IDs resolved by wiki lookup (tags + explicit IDs)
    evaluated_ids: Vec<String>,  // constraint IDs that were actually evaluated against proposals
    violation_ids: Vec<String>,  // constraint IDs that fired (failed Hard/Soft threshold); populated in Plan B
}
```

### State and Delta Checkpointing

```toml
[state]
# All NATS bucket and stream names — override to namespace multi-tenant deployments.
calibration_bucket = "H2AI_CALIBRATION"
snapshots_bucket = "H2AI_SNAPSHOTS"
agent_memory_bucket = "H2AI_AGENT_MEMORY"
estimator_bucket = "H2AI_ESTIMATOR"
shadow_auditor_bucket = "H2AI_SHADOW_AUDITOR"
approvals_bucket = "H2AI_APPROVALS"
prompt_variants_bucket = "H2AI_PROMPT_VARIANTS"
tasks_stream = "H2AI_TASKS"
telemetry_stream = "H2AI_TELEMETRY"
results_stream = "H2AI_RESULTS"

[state.delta]
enabled = true
base_interval = 10          # store a full base every N checkpoints
cache_ttl_secs = 60
cache_max_entries = 200
```

**Delta checkpoint NATS key scheme:**

| Key | Contents | Written when |
|---|---|---|
| `{task_id}/seq/{seq:08}` | `TaskCheckpointEntry` (Base or Delta) | Every `put_checkpoint_delta` call |
| `{task_id}/seq/latest` | Plain u32 seq number | CAS-updated after every write (3-retry loop) |

`seq=0` and every multiple of `base_interval` store a `CheckpointKind::Base(Box<TaskCheckpoint>)` (full snapshot). All other seqs store `CheckpointKind::Delta(Vec<PatchOperation>)` — the RFC 6902 diff against the nearest base. On read, `reconstruct_at_seq` fetches the base entry then applies the patch chain. Legacy flat-key checkpoints (pre-delta, no `/seq/` component) are detected by key format and read as a Base without migration.

### Reasoning Memory

```toml
[reasoning_memory]
enabled                     = false   # all checkpoint writes skipped when false
induction_batch_size        = 10
induction_max_interval_secs = 86400
induction_max_tasks_per_run = 50
tag_gate_threshold          = 0.2
max_archetype_boost         = 0.15
max_archetype_penalty       = 0.20
```

| Field | Default | Purpose |
|---|---|---|
| `enabled` | `false` | Master switch. When `false`, all `TaskReasoningCheckpoint` and `TaskMetaState` writes are skipped; no NATS buckets are created. |
| `induction_batch_size` | `10` | Number of `TaskMetaState` records consumed per `InductionScheduler` batch run (Phase 2). |
| `induction_max_interval_secs` | `86400` | Maximum interval between scheduled induction runs (Phase 2). |
| `induction_max_tasks_per_run` | `50` | Hard cap on tasks processed in a single induction run (Phase 2). |
| `tag_gate_threshold` | `0.2` | Minimum tag-overlap score for a `TaskMetaState` to be retrieved as a prior (Phase 2). |
| `max_archetype_boost` | `0.15` | Maximum score boost applied to an archetype with a strong positive prior (Phase 2). |
| `max_archetype_penalty` | `0.20` | Maximum score penalty applied to an archetype with a strong negative prior (Phase 2). |

Per-tenant NATS KV bucket name prefixes are configured under `[state]`:

| Field | Default | Purpose |
|---|---|---|
| `state.reasoning_checkpoint_bucket_prefix` | `"H2AI_CHECKPOINT"` | Prefix for per-tenant reasoning checkpoint buckets (`{prefix}_{tenant_id}`). |
| `state.task_meta_state_bucket_prefix` | `"H2AI_META"` | Prefix for per-tenant meta-state buckets (`{prefix}_{tenant_id}`). |

### Oracle Gate

```toml
[oracle_gate]
enabled = false
subject = "h2ai.oracle.gate"
timeout_secs = 30
on_timeout = "pass"         # pass | fail | skip
min_confidence = 0.7
# clarification_templates = [{ pattern = "...", question_template = "..." }]
```

| Field | Default | Purpose |
|---|---|---|
| `enabled` | `false` | Master switch; set `true` to wire Phase 4.5. |
| `subject` | `"h2ai.oracle.gate"` | NATS subject for `request()` calls. The oracle service subscribes here. |
| `timeout_secs` | `30` | How long to wait for an oracle reply before applying `on_timeout`. |
| `on_timeout` | `"pass"` | `pass` — treat timeout as approved; `fail` — treat as rejected; `skip` — proceed with no `oracle_gate_passed` field. |
| `min_confidence` | `0.7` | When oracle responds with `gate_passed=false` AND `confidence < min_confidence`, the engine attempts clarification via a matching `ClarificationTemplate`. |
| `clarification_templates` | `[]` | Array of `{ pattern: regex, question_template: string }`. Template placeholders: `{test_name}`, `{expected}`, `{actual}`, `{failure_delta}`. First matching pattern wins. If no template matches, the engine proceeds with `on_timeout` behaviour. |

### Adaptive Prompt Harness (OPRO)

```toml
[opro]
enabled = false
trigger_j_eff_threshold = 0.60
min_tasks_before_trigger = 10
suppress_n_tasks = 5        # suppress OPRO for N tasks after each trigger
graduation_tasks = 20       # total tasks before bandit graduation
promotion_margin = 0.05     # min improvement to promote a variant
ema_window = 10             # tasks in the j_eff EMA window

[calibration_bootstrap]
prior_weight = 5            # virtual task count for bootstrap Bayesian prior
```

| Field | Default | Purpose |
|---|---|---|
| `trigger_j_eff_threshold` | `0.60` | OPRO cycle fires when `j_eff_ema < threshold` and task count conditions are met. |
| `min_tasks_before_trigger` | `10` | Minimum observed tasks before any OPRO cycle is allowed. |
| `suppress_n_tasks` | `5` | Tasks to skip after each OPRO trigger (cooldown). |
| `graduation_tasks` | `20` | Total tasks observed before the Thompson bandit promotes the best variant. |
| `promotion_margin` | `0.05` | A challenger variant must beat the incumbent by this margin (in mean j_eff) to be promoted. |
| `ema_window` | `10` | EMA decay window (α = 2/(window+1)). Smaller = reacts faster to recent task quality. |
| `prior_weight` | `5` | Virtual observation count for bootstrap Beta priors. Tier medians: Capable=0.78, Standard=0.62, Fast=0.45. |

**Prompt variant NATS key scheme** (`H2AI_PROMPT_VARIANTS` bucket):

| Key | Contents |
|---|---|
| `{adapter_name}/{prompt_key}/{variant_id}` | `PromptVariant` JSON |
| `{adapter_name}/{prompt_key}/_active` | Active variant ID (plain string pointer) |
| `{adapter_name}/_opro_state` | `AdapterOproState` JSON (EMA, bandit arms, task counters) |

### HITL

| Field | Default | Purpose |
|---|---|---|
| `hitl.enabled` | `true` | Master switch; set `false` to bypass gate in dev/test. |
| `hitl.confidence_threshold` | `0.50` | `q_confidence` below this triggers human review. |
| `hitl.timeout_ms` | `1_800_000` | Review window length (30 minutes). After expiry the reaper auto-rejects. |

### Constraint Wiki

| Field | Default | Purpose |
|---|---|---|
| `constraint_wiki.enabled` | `false` | When `true`, corpus access routes through `NatsWikiConstraintSource` (NATS KV + Object Store). When `false`, falls back to `FsConstraintSource` (flat directory). |
| `constraint_wiki.corpus_path` | `"/constraints"` | Filesystem path for `FsConstraintSource` (used when `enabled = false`). Ignored when wiki is enabled. |
| `constraint_wiki.resolve_k` | `50` | Reserved: max constraints returned per `resolve_context` call (future Qdrant semantic search limit). |

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
tier = "Standard"           # Fast | Standard | Capable — seeds OPRO bootstrap prior
[adapter_profiles.kind.Anthropic]
api_key_env = "ANTHROPIC_API_KEY"
model = "claude-sonnet-4-5"
```

`tier` sets the Bayesian j_eff prior for the Adaptive Prompt Harness bootstrap: `Capable` → 0.78, `Standard` → 0.62, `Fast` → 0.45. Omitting the field defaults to `Standard`. The prior is seeded once at startup as `prior_weight` virtual observations and is superseded by real task data as tasks accumulate.

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

Built-in implementations in `crates/h2ai-adapters`: `Anthropic`, `OpenAI`, `Gemini`, `Ollama`, `LlamaCpp` (over HTTP — local server or `host.docker.internal:8000`), `CloudGeneric` (OpenAI-compatible), `A2a`, `Mock`.

**A2A Explorer Adapter** (`crates/h2ai-adapters/src/a2a.rs`): Client adapter for any [Agent2Agent (A2A)](https://a2aprotocol.ai) compatible remote agent. Adds a new diversity axis beyond model diversity: cross-framework N_eff gains. Delegates via JSON-RPC 2.0 polling; artifact text passes through a 4-stage extraction pipeline before entering the merge phase. Configured with `AdapterKind::A2a { ... }` — identical integration path as built-in adapters. Requires `endpoint`, `auth_scheme` (`"bearer"`, `"api_key"`, or `"none"`), `auth_token_env` (env var name), `timeout_minutes`, `poll_interval_ms`, `max_poll_interval_ms`, and `agent_card_cache_ttl_s`.

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

### Phase 4 Wiki Representation

The typed constraint wiki system introduces three types for structured constraint delivery:
- **`ConstraintMeta`**: Lightweight descriptor (~300 bytes) loaded at Phase 1 Bootstrap; includes id, summary, severity, predicate_kind, domains, and inline predicates for static evaluations.
- **`ConstraintPayload`**: Full descriptor fetched on-demand during Phase 4; carries the complete predicate for LlmJudge and Oracle evaluations.
- **`PredicateKind`**: Enum (Static | LlmJudge | Oracle) that gates lazy loading — Static predicates are inlined in ConstraintMeta; others are fetched from Predicate Store only when needed.

### ConstraintSource Abstraction

Corpus access is mediated by the `ConstraintSource` trait (`crates/h2ai-constraints/src/source.rs`); callers never invoke `load_corpus` directly.
- **`FsConstraintSource`**: wraps `load_corpus` for backward compatibility with flat-directory corpora; falls back to all docs when tags are supplied but no frontmatter domain metadata is present.
- **`NatsWikiConstraintSource`** (in `h2ai-api`): reads from NATS KV `H2AI_CONSTRAINT_WIKI` + Object Store; enables hot-reload without restart.
