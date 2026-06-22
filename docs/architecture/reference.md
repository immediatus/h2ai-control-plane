# H2AI Reference

Operational surface of the control plane: HTTP API, event vocabulary, Prometheus metrics, configuration, adapters, agent descriptor, and constraint corpus. Authoritative source for every wire format. Defaults come from `crates/h2ai-config/reference.toml`; field semantics come from the Rust types in `crates/h2ai-types`.

---

## 1. HTTP API

The Axum router is wired in `crates/h2ai-api/src/routes/mod.rs`. Authentication is **not** built into the control plane; it is expected at the ingress layer (mTLS, JWT validation, OAuth2 proxy). All endpoints assume requests are pre-authenticated.

### Endpoint summary

All task routes include `:tenant_id` as a URL path segment. The path value is authoritative — any `tenant_id` in the request body is overridden. Single-tenant deployments use `default` as the tenant ID (e.g. `/v1/default/tasks`).

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/:tenant_id/tasks` | Submit a task manifest. |
| `GET` | `/:tenant_id/tasks/:task_id/events` | SSE stream of `H2AIEvent` for a task. |
| `GET` | `/:tenant_id/tasks/:task_id` | Current task status snapshot. |
| `POST` | `/:tenant_id/tasks/:task_id/merge` | Resolve a Merge Authority decision. |
| `POST` | `/:tenant_id/tasks/:task_id/signal` | Inject a typed `SignalPayload` (`Approve` or `WaveContinue`) into a running task. Returns 202 immediately; engine consumes from JetStream. |
| `POST` | `/:tenant_id/tasks/:task_id/approve` | **Deprecated** — returns 301 redirect to `/signal`. Kept for one release. |
| `POST` | `/:tenant_id/tasks/:task_id/clarify` | Submit operator answer to a pending oracle clarification. |
| `GET` | `/:tenant_id/tasks/:task_id/approval` | **Deprecated** — returns 410 Gone; approval records removed. |
| `GET` | `/:tenant_id/tasks/:task_id/recover` | Trigger snapshot+replay recovery for a task. |
| `POST` | `/calibrate` | Start a calibration run. |
| `GET` | `/calibrate/:cal_id/events` | SSE stream for an in-progress calibration. |
| `GET` | `/calibrate/current` | Last completed calibration. |
| `GET` | `/health` | Liveness probe. |
| `GET` | `/ready` | Readiness probe (depends on NATS connectivity and current calibration). |
| `GET` | `/metrics` | Prometheus exposition. |

### POST /:tenant_id/tasks

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

`constraint_tags` routes the task to a domain-specific subset of the constraint corpus via the wiki index (see §7). `constraints` provides explicit constraint IDs that are always included regardless of tags. `require_approval` forces a HITL review gate after merge even when `q_confidence` is high (see §9.5). `tenant_id` is taken from the URL path and is always required. It scopes all estimators, HITL signals, and NATS KV keys to an isolated per-tenant namespace. The request body `tenant_id` field, if present, is overridden by the path value.

**Response:** `202 Accepted` with `{"task_id": "...", "events_url": "/{tenant_id}/tasks/.../events"}`.

**Submission failures:**

- `400` — malformed manifest, weights not summing to 1.0, or invalid role spec.
- `503 CalibrationRequiredError` — no current calibration in `H2AI_CALIBRATION` KV.

### GET /:tenant_id/tasks/:task_id/events

Server-Sent Events stream of `H2AIEvent` envelopes. Each event:

```
id: <NATS sequence number>
event: <event_type>
data: {"event_type": "...", "payload": {...}}

```

Reconnect with `Last-Event-ID: <sequence>` to resume from the last seen offset.

### POST /:tenant_id/tasks/:task_id/signal

Injects a typed `SignalPayload` into a running task. Returns `202 Accepted` immediately; the engine consumes the message from JetStream asynchronously.

**Request body:**

```json
{
  "payload": {
    "kind": "Approve",
    "data": {
      "approved": true,
      "operator_id": "alice@acme.com",
      "reviewer_note": "optional"
    }
  },
  "timeout_ms": 3600000
}
```

`timeout_ms` is optional. When supplied it overrides the engine's default review window for this task, clamped to `[signal_min_timeout_ms, signal_max_timeout_ms]`. Omit to use `hitl.timeout_ms`.

**Signal kinds:**

| `kind` | Purpose |
|---|---|
| `Approve` | Resolves the HITL gate. `approved: true` → task completes; `approved: false` → `TaskFailed`. |
| `WaveContinue` | Injects `grounding` text or `mandate_override` at a `WaveCompleted` pause. Only processed when `signal_wave_window_ms > 0`. |

**Response:** `202 Accepted`; `404` when task not found; `503` when task is not in a signalable state.

### POST /:tenant_id/tasks/:task_id/clarify

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
| `TaoIteration` | `TaoIterationEvent` | 3 — `tool_calls` omitted when empty |
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
| `ProvenanceRecorded` | `ProvenanceRecordedEvent` | post-merge (epistemic quality) |
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
| `ResearcherGrounding` | `ResearcherGroundingEvent` | 3.1 / 3.2 / proactive |
| `DiversityGuardDegraded` | `DiversityGuardDegradedEvent` | 2.6 (domain coverage) |
| `OracleGateResult` | `OracleGateResultEvent` | 4.5 |
| `PendingClarification` | `PendingClarificationEvent` | 4.5 |
| `OproTriggered` | `OproTriggeredEvent` | post-merge (async) |
| `PromptVariantPromoted` | `PromptVariantPromotedEvent` | post-merge (async) |
| `ConstraintAmbiguity` | `ConstraintAmbiguityEvent` | 3.5 (corpus quality signal) |
| `ComplexityProbe` | `ComplexityProbeEvent` | pre-loop |
| `ComplexityCeilingDetected` | `ComplexityCeilingDetectedEvent` | MAPE-K |
| `TaskComplexityAssessed` | `TaskComplexityAssessedEvent` | pre-loop (complexity module) |
| `ThinkingLoopCompleted` | `ThinkingLoopCompletedEvent` | Phase -1 |
| `AwarenessProbeCompleted` | `AwarenessProbeCompletedEvent` | Phase -1 (post thinking loop) |
| `LeaderElected` | `LeaderElectedEvent` | MAPE-K (between waves) |
| `SocraticDiagnosis` | `SocraticDiagnosisEvent` | MAPE-K (between waves) |
| `VerifierFrozen` | `VerifierFrozenEvent` | MAPE-K (per-constraint bypass) |
| `VerifierReasonContradiction` | `VerifierReasonContradictionEvent` | MAPE-K |
| `ConstraintFrontier` | `ConstraintFrontierEvent` | 4.5 (static constraint matrix) |
| `CostThresholdWarning` | `CostThresholdWarningEvent` | any (cost guard) |
| `BudgetExhausted` | `BudgetExhaustedEvent` | any (cost guard abort) |
| `ConvergenceGate` | `ConvergenceGateEvent` | MAPE-K (convergence gate) |
| `ConstraintCoherenceWarning` | `ConstraintCoherenceWarning` | corpus management |
| `ConstraintRepairAttempted` | `ConstraintRepairAttempted` | corpus management |
| `ConstraintVersionCreated` | `ConstraintVersionCreated` | corpus management |
| `ConstraintRepairFailed` | `ConstraintRepairFailed` | corpus management |
| `InductionCycleCompleted` | `InductionCycleCompletedEvent` | post-merge (cross-task memory) |
| `VerifierInstability` | `VerifierInstabilityEvent` | MAPE-K (verifier stability monitor) |
| `GenerationKnowledge` | `GenerationKnowledgeEvent` | Phase 3 (knowledge enrichment) |
| `ShadowAudit` | `ShadowAuditorResultEvent` | Phase 4 (shadow auditor) |
| `OracleCalibrationPatched` | `OracleCalibrationPatchedEvent` | post-merge (oracle empirical patch) |
| `AuditDomainPromoted` | `AuditDomainPromotedEvent` | Phase 4 (shadow auditor domain promotion) |
| `AuditDomainDemoted` | `AuditDomainDemotedEvent` | Phase 4 (shadow auditor domain demotion) |
| `VerifierComparison` | `VerifierComparisonEvent` | Phase 3.5 (cross-family verifier comparison) |

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
    adapter_families: Vec<String>,                   // populated from actual adapter registry
    explorer_verification_family_match: bool,        // true when cross-family panel is active
    single_family_warning: bool,                     // true when all calibration adapters share one family
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

Emitted by `POST /{tenant_id}/tasks/{id}/approve` after the CAS delete succeeds.

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

#### ResearcherGroundingEvent

Emitted when external grounding is fetched — either reactively (C1 hallucination retry via `GapResearchChain`) or proactively (slot with `search_enabled: true`).

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

Emitted (Phase 4.5) when the oracle gate fails with low confidence and a matching `ClarificationTemplate` fires. The engine suspends via `clarification_waiters`; `POST /{tenant_id}/tasks/{id}/clarify` resumes it.

```rust
struct PendingClarificationEvent {
    task_id: TaskId,
    question: String,
    context: String,
    timeout_secs: u64,
    timestamp: DateTime<Utc>,
}
```

#### ConstraintAmbiguityEvent

Emitted (fire-and-forget, `tracing::info!` target `"h2ai.engine"`) when ≥ `ambiguity_threshold` proposals in a wave return uncertain judge panel votes for the same constraint. Signals a constraint that is semantically underdetermined — the panel cannot reach a confident verdict because the constraint text is ambiguous. This is a corpus quality signal, not a task failure.

```rust
struct ConstraintAmbiguityEvent {
    task_id: TaskId,
    wave: u32,
    ambiguous_constraints: Vec<String>,          // constraint IDs above threshold
    uncertain_counts: HashMap<String, usize>,    // per-constraint uncertain vote counts
    timestamp: DateTime<Utc>,
}
```

**Operational use:** when `ambiguous_constraints` is non-empty across multiple tasks for the same constraint ID, the constraint definition should be reviewed and tightened. The `uncertain_counts` field shows how many proposals in that wave triggered disagreement.

---

#### ThinkingLoopCompletedEvent

Published after the final thinking loop iteration (Phase -1). When `thinking_loop.enabled = false`, all numeric fields are zero and `archetypes` is empty.

```rust
struct ThinkingLoopCompletedEvent {
    task_id: TaskId,
    enabled: bool,                       // false when thinking_loop.enabled = false
    iterations_run: u32,                 // how many iterations actually ran
    coverage_score: f64,                 // final coverage score from the last ThinkingReport
    shared_understanding_len: usize,     // char count (keeps event payload small)
    archetypes: Vec<String>,             // names of archetypes selected in the final iteration
    timestamp: DateTime<Utc>,
}
```

#### VerificationScoredEvent

```rust
struct VerificationScoredEvent {
    task_id: TaskId,
    explorer_id: ExplorerId,
    score: f64,
    reason: String,
    passed: bool,
    #[serde(default)]
    cache_hit: bool,                     // true when score reused from per-task eval cache
    #[serde(default)]
    passed_checks: Option<u32>,          // count of binary check_verdicts = true
    #[serde(default)]
    total_checks: Option<u32>,           // total binary check_verdicts (None when no binary checks)
    #[serde(default)]
    score_lower: Option<f64>,            // 95% Wilson CI lower bound on binary-check pass fraction
    #[serde(default)]
    score_upper: Option<f64>,            // 95% Wilson CI upper bound on binary-check pass fraction
    #[serde(default)]
    per_check_verdicts: Vec<CheckVerdict>, // per-check PRESENT/MISSING verdicts parsed from CoT
    timestamp: DateTime<Utc>,
}
```

`total_checks = 0` when the verifier response does not contain a visible `CHECK VERDICTS:` section. Thinking models (llama_cpp, Qwen3, R1) emit CHECK reasoning in hidden `<think>` tokens by default; the `CHECK_EVIDENCE_FORMAT_INSTRUCTION` prompt constant instructs the model to emit a `CHECK VERDICTS:` block in the visible final response to make the `has_check_markers` guard fire. Legacy events (before binary-check tracking) have all `Option` fields as `None`.

#### MergeResolvedEvent (updated fields)

`MergeResolvedEvent` carries two optional diagnostic fields:

**`oracle_gate_passed: Option<bool>`**
- `Some(true)` — oracle gate ran and approved the surviving proposals.
- `Some(false)` — oracle gate ran and rejected; clarification was supplied or `on_timeout = "fail"` was overridden.
- `None` — oracle gate was disabled, timed out with `on_timeout = "skip"`, or was bypassed.

**`zone3_hints: Option<String>`** *(added 2026-05-21)*
- `Some(text)` — OSP Zone 3 audit findings were generated by `AuditChannelBuilder`. The text contains only `constraint_id` and `remediation_hint` from the concordant `ConstraintViolation` records of failed proposals. It is injected into the next retry's synthesis prompt as positive guidance ("showed consistent difficulty in prior drafts").
- `None` — OSP is disabled (`[osp]` absent), no failed proposals this wave, `n_v > max_n_v_for_zone3`, or concordance threshold was not met.

**Operational use:** `zone3_hints` being populated consistently across retries for the same constraint ID indicates a systematically hard constraint. Review the constraint definition or increase `t_v` to relax the ClearLeader threshold for that domain.

---

#### ComplexityProbeEvent *(added 2026-05-29)*

Emitted once per task, before the first retry wave, when `complexity_routing.enabled = true`. Records the pre-dispatch probe verdict. Probe failure or timeout produces an event with `complexity = 2` (conservative default).

```rust
struct ComplexityProbeEvent {
    task_id: TaskId,
    complexity: u8,                  // 1–5 rating
    rationale: String,               // one-sentence explanation from the probe model
    decompose_recommended: bool,     // probe model's decomposition advisory
    probe_latency_ms: u64,
    timestamp: DateTime<Utc>,
}
```

**Operational use:** distribution of `complexity` per tenant indicates the workload mix. Consistent `complexity = 5` runs with `complexity_routing.hitl_threshold = 5` will all surface to HITL before any retry — operators should expect to see these in the HITL queue rather than the failure log.

---

#### ComplexityCeilingDetectedEvent *(added 2026-05-29)*

Emitted when the intra-retry ceiling detector fires (≥2 of 3 signals exceed thresholds) during a `ZeroSurvival` decision path. Indicates the task is hitting a structural ceiling rather than a quality ceiling.

```rust
struct ComplexityCeilingDetectedEvent {
    task_id: TaskId,
    retry_count: u32,
    entropy: f64,                    // failure_signature_entropy(last_wave_pruned)
    retry_slope: f64,                // (best_score_n - best_score_n-1) / best_score_n-1
    n_eff_cg_product: f64,           // N_eff × CG_mean from the last wave
    signals_fired: u8,               // 2 or 3
    timestamp: DateTime<Utc>,
}
```

**Operational use:** frequent firing for tasks the probe rated ≤ 3 means the probe is misclassifying. Either upgrade `complexity_probe_adapter` to a stronger model or lower `decompose_threshold`. The detector exists precisely as a safety net for probe miscalibration.

#### ProvenanceRecordedEvent

Emitted after the epistemic output quality pipeline completes for the resolved output. Requires `epistemic_quality.enabled = true`. Always fires after `MergeResolved` and before task close.

```rust
struct ProvenanceRecordedEvent {
    task_id: TaskId,
    document_confidence: String,  // "High" | "ReviewRecommended" | "RequiresReview" | "Unverified"
    provision_count: usize,       // total provisions tracked in ProvenanceMap
    open_gap_count: usize,        // gaps remaining after all MicroExplorerResolver passes
    timestamp: DateTime<Utc>,
}
```

`document_confidence` is the worst-wins aggregate over all provisions in the `ProvenanceMap`: if any provision is `Unverified`, the document is `Unverified`; `Verified` and `AutoCorrected` provisions both contribute to a `High` document-level rating, because `AutoCorrected` means the gap was patched and closed (no unresolved gap remains). `open_gap_count > 0` with `document_confidence = "High"` cannot occur — any unclosed gap leaves at least one provision below `AutoCorrected`.

**E2E assertions** in `replay.py`: `provenance_recorded` (event fired), `document_confidence_not_verified` (`document_confidence != "Unverified"`), `open_gap_count_min` (`open_gap_count >= N`).

#### VerifierReasonContradictionEvent.beyond_budget_count *(added 2026-05-29)*

`VerifierReasonContradictionEvent` gained a `beyond_budget_count: u32` field with `#[serde(default)]` (old events still deserialise). Non-zero values indicate the verifier reported sub-claims it could not evaluate within its own compute budget — distinguishes "rejected this" from "could not compute this". When `complexity_routing.verifier_decomposition_enabled = true` and the probe rates the task ≥ `decompose_threshold`, `BEYOND_BUDGET_VERIFIER_ADDENDUM` is appended to the verifier system prompt before the first wave, instructing sub-claim decomposition with VERIFIED / UNVERIFIED / BEYOND_BUDGET reporting.

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

### Safety Profiles

`[safety] profile` selects a named tier that overwrites all safety-relevant fields at startup. When `profile = "custom"`, no fields are overwritten and each field is set individually.

| Profile | When to use | shadow_auditor | strict | krum_f | diversity_threshold | family_constraint |
|---|---|---|---|---|---|---|
| `development` | Local dev, devcontainer, single-adapter setups | off | false | 0 | 0.0 | single_family_ok |
| `production` | Staging and production with ≥2 adapter families | **on** | **true** | 1 | 0.15 | require_diverse |
| `strict` | High-stakes deployments, regulated environments | **on** | **true** | 2 | 0.20 | require_diverse |
| `custom` | E2E tests, research scenarios, non-standard topologies | manual | manual | manual | manual | manual |

**`development`** — all safety gates relaxed. A single adapter family is allowed. Shadow auditor is disabled. Use this profile when you have one LLM endpoint or are iterating on constraint corpus design. `QuorumDegradedBelowMinimum` still fires in shadow mode.

**`production`** — shadow auditor enabled and running in strict mode (audit failures abort the task, not just log). Krum fault tolerance `f = 1` (tolerates one Byzantine-equivalent outlier). Family diversity is required — if all adapters share a provider, `VerifierExplorerFamilyConflict` marks the task failed before the first wave. Use this for all deployed environments.

**`strict`** — maximum safety tier. `f = 2` for Krum (requires ≥7 adapters for Krum to have a meaningful effect), bivariate CG required (`require_bivariate_cg = true` is set separately), diversity threshold 0.20. Intended for compliance-heavy domains (finance, healthcare, regulated DevOps).

**`custom`** — profile writes no fields. Combine with explicit `[safety.shadow_auditor]`, `[safety]` overrides, or env-vars. Used by E2E benchmark scenarios that need partial safety settings (e.g. shadow auditor on but strict=false).

**TOML examples:**

```toml
# Minimal devcontainer / local run — one LLM endpoint, no diversity requirement
[safety]
profile = "development"

# Production with two adapter families
[safety]
profile = "production"

# Research E2E: shadow auditor enabled but non-blocking (strict = false)
[safety]
profile = "custom"

[safety.shadow_auditor]
enabled = true
strict  = false

# Strict compliance deployment with additional per-field overrides
[safety]
profile = "strict"

[safety.shadow_auditor]
enabled     = true
strict      = true
adapter_key = "shadow_auditor"   # must be declared in [[adapter_profiles]]
```

**Shadow auditor sub-fields** (`[safety.shadow_auditor]`):

| Field | Default | Purpose |
|---|---|---|
| `enabled` | `false` (dev) / `true` (prod/strict) | Run a second independent audit pass on each wave result |
| `strict` | `false` (dev) / `true` (prod/strict) | When `true`, audit failures block the wave result; when `false`, failures are logged only |
| `adapter_key` | `"shadow_auditor"` | Named adapter from `[[adapter_profiles]]` used for the audit call |

**What each profile does to Krum:**

Krum requires ≥ `2f + 3` adapters to have any effect (`f = 1` → ≥5, `f = 2` → ≥7). With a single adapter family (`development`), Krum is disabled (`f = 0`). With `production`, a single outlier is tolerated. With `strict`, two independent outliers (or coordinated failures) are tolerated, but you must provision at least 7 explorer slots.

**Family constraint interaction:**

`require_diverse` checks at task submission time that the verification adapter and the explorer pool do not share a provider family. If they do, the task fails immediately with `VerifierExplorerFamilyConflict` — no LLM calls are made. This is not a runtime penalty; it is a hard configuration error. The fix is to route the verifier to a different model family and restart the server.

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
| `min_explorer_families` | `0` | BFT Lever 1 — minimum distinct model lineage families required in the explorer committee. `0` disables the check. When set to `2`, calibration emits `tracing::warn!` if fewer than 2 distinct `model_lineage_key()` values are present. `AdapterKind::model_lineage_key()` returns `cloud::{provider}::{endpoint}::{model}` for `CloudGeneric` adapters and `local::{model_path}` for `LocalLlamaCpp`. Enforcement is advisory-only (warn, never hard-fail) to allow monoculture dev environments. |

### Calibration probe (epistemic β₀)

```toml
[calibration_probe]
agents           = 3      # synthetic probe pool size for epistemic β₀
max_tokens       = 512    # token budget per probe call
neff_cg_exponent = 2.0    # k exponent in N_eff_adj = clamp(N_eff × CG_mean^k, 1, N_cal)
```

`[calibration_probe]` governs the epistemic β₀ path (active when `embedding_model` is configured and `calibration_adapter_count ≥ 3`). The three-tier β₀ resolution order is: (1) epistemic formula — `β₀ = max((1/N_eff_adj − 1/N_cal) / (N_cal − 1), 1e-6)` where `N_eff_adj = clamp(N_eff × CG_mean^k, 1, N_cal)`; (2) conflict-count online override from `ConflictRateAccumulator`; (3) latency-based fallback proxy. The epistemic path requires both an embedding model and ≥3 calibration adapters; without these, tier 2 or 3 is used.

| Field | Default | Purpose |
|---|---|---|
| `calibration_probe.agents` | `3` | Probe pool size. Must be ≥ 3 to activate the epistemic path. |
| `calibration_probe.max_tokens` | `512` | Token budget per probe adapter call. |
| `calibration_probe.neff_cg_exponent` | `2.0` | Exponent `k` in `N_eff_adj = clamp(N_eff × CG_mean^k, 1, N_cal)`. Higher k amplifies CG's dampening effect on N_eff. Mode collapse (N_eff≈1, CG≈0) → β₀≈0.333; ideal pool (N_eff=3, CG=0.9, k=2) → β₀≈0.039. |

### Calibration slow start (AIMD α adaptation)

```toml
[calibration_slow_start]
seed_alpha       = 0.15   # starting α for new adapter pools
decay_rate       = 0.95   # per-measurement multiplicative decay
reset_multiplier = 3.0    # α expansion factor on ZeroSurvival reset
reset_threshold  = 0.40   # α_measured below this triggers AIMD decay
```

AIMD slow-start governs how the calibration harness adapts the `alpha_contention` estimate over time. On each measurement: `α_new = max(α_cur × decay_rate, α_measured)` (steady decay toward the measured value). On a `ZeroSurvival` event: `α_new = min(α_cur × reset_multiplier, seed_alpha)` (multiplicative expansion, capped at seed). This prevents the system from over-committing to a low-α estimate after a failure event.

| Field | Default | Purpose |
|---|---|---|
| `calibration_slow_start.seed_alpha` | `0.15` | Initial α prior for new adapter pools. |
| `calibration_slow_start.decay_rate` | `0.95` | Per-measurement decay factor. Smaller = faster convergence to measured α. |
| `calibration_slow_start.reset_multiplier` | `3.0` | α expansion factor on ZeroSurvival. Caution: multiplier > seed_alpha/α_floor risks runaway expansion. |
| `calibration_slow_start.reset_threshold` | `0.40` | `α_measured` below this triggers AIMD decay; above this, α stays at measured value. |

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

### Complexity-Ceiling Routing

Pre-dispatch complexity probe + intra-retry ceiling detector. Routes structurally intractable tasks to /H1 synthesis-wave grafting or HITL before burning the full retry budget. Opt-in; default off in `reference.toml`, enabled in all six E2E scenarios (`tests/e2e/scenarios/*/`).

```toml
[complexity_routing]
enabled                          = false
complexity_probe_adapter         = "researcher"
complexity_probe_timeout_secs    = 30
decompose_threshold              = 4
hitl_threshold                   = 5
verifier_decomposition_enabled   = false

[complexity_routing.intra_retry]
enabled                          = false
entropy_threshold                = 0.6
retry_slope_threshold            = 0.05
n_eff_cg_product_threshold       = 0.3
min_retry_count_for_detection    = 2
```

| Field | Default | Purpose |
|---|---|---|
| `complexity_routing.enabled` | `false` | Master toggle. When `false`, no probe runs and no routing decisions are altered. |
| `complexity_routing.complexity_probe_adapter` | `"researcher"` | Adapter profile key for the probe call. Falls back to first explorer adapter when unavailable. |
| `complexity_routing.complexity_probe_timeout_secs` | `30` | Probe LLM call timeout. On timeout, defaults to `complexity = 2` (conservative). |
| `complexity_routing.decompose_threshold` | `4` | Probe score ≥ this routes to H1 grafting on first failure. |
| `complexity_routing.hitl_threshold` | `5` | Probe score ≥ this skips all retries and routes directly to HITL. |
| `complexity_routing.verifier_decomposition_enabled` | `false` | When `true` and probe score ≥ `decompose_threshold`, appends `BEYOND_BUDGET_VERIFIER_ADDENDUM` to the verifier system prompt before the first wave; instructs sub-claim decomposition with VERIFIED / UNVERIFIED / BEYOND_BUDGET reporting. |
| `complexity_routing.intra_retry.enabled` | `false` | Intra-retry ceiling detector (signals fire mid-loop on `ZeroSurvival`). |
| `complexity_routing.intra_retry.entropy_threshold` | `0.6` | Failure-signature Shannon entropy below this = concentrated failure mode = ceiling signal. |
| `complexity_routing.intra_retry.retry_slope_threshold` | `0.05` | Quality-score slope between consecutive waves below this = not converging. |
| `complexity_routing.intra_retry.n_eff_cg_product_threshold` | `0.3` | `N_eff × CG_mean` below this = correlated failure = ceiling signal. |
| `complexity_routing.intra_retry.min_retry_count_for_detection` | `2` | Minimum wave count before detector may fire; suppresses first-wave variance. |

### Tiered Early Exit

Fast-path acceptance gate that scales the number of explorers linearly across retry waves. On wave 0 the engine spawns `min_n` explorers; by wave `max_retries` it reaches `max_n`. A wave resolves immediately once enough proposals pass — avoiding full retry-budget consumption on straightforward tasks. Used by the `compliance-lite` benchmark scenario.

```toml
[tiered_exit]
enabled                   = false
min_n                     = 1
max_n                     = 5
quorum_fraction           = 0.5
acceptance_score          = 0.85
require_all_binary_checks = true
```

| Field | Default | Purpose |
|---|---|---|
| `tiered_exit.enabled` | `false` | Master toggle. When `false`, N is fixed at the calibrated value and acceptance is not checked mid-wave. |
| `tiered_exit.min_n` | `1` | Explorer count at wave 0. |
| `tiered_exit.max_n` | `5` | Explorer count at wave `max_retries`. Capped by `N_max` from calibration. |
| `tiered_exit.quorum_fraction` | `0.5` | Fraction of the wave's N that must pass to trigger early exit. k_for_wave = `max(1, ceil(n × quorum_fraction))`. |
| `tiered_exit.acceptance_score` | `0.85` | Minimum score a proposal must reach to count toward the quorum. |
| `tiered_exit.require_all_binary_checks` | `true` | When `true`, proposals that fail any binary check are excluded from the quorum count regardless of score. |

`TieredExitEvent` is published via `H2AIEvent::TieredExit` when the gate fires at resolution. The `n_for_wave` function interpolates linearly: `min_n + round((max_n − min_n) × wave / max_retries)`.

### Cost Guard

Per-task token-budget enforcement. Tracks cumulative generation tokens across all MAPE-K waves and blocks further retries once the budget is consumed. Provides a remaining-token hint injection to keep explorers aware of the shrinking window.

```toml
[cost_guard]
enabled                         = false
budget_tokens_per_task          = 100000
budget_warning_fraction         = 0.80
budget_abort_fraction           = 1.00
budget_prompt_injection_enabled = false
budget_injection_warn_fraction  = 0.50
budget_injection_max_complexity = 3
```

| Field | Default | Purpose |
|---|---|---|
| `cost_guard.enabled` | `false` | Master toggle. When `false`, no token tracking or injection occurs. |
| `cost_guard.budget_tokens_per_task` | `100000` | Total token budget for one task across all waves. |
| `cost_guard.budget_warning_fraction` | `0.80` | Fraction of budget at which `CostThresholdWarningEvent` is emitted. |
| `cost_guard.budget_abort_fraction` | `1.00` | Fraction of budget at which further retries are blocked. Setting below `1.0` creates a safety margin before hard exhaustion. |
| `cost_guard.budget_prompt_injection_enabled` | `false` | When `true`, a remaining-token hint is injected into `active_ctx` in the [50 %, 85 %) consumption window. |
| `cost_guard.budget_injection_warn_fraction` | `0.50` | Lower bound of the injection window (50 % budget consumed). |
| `cost_guard.budget_injection_max_complexity` | `3` | Hint injection is skipped for tasks with probe complexity above this threshold (high-complexity tasks generate long outputs regardless). |

Events: `CostThresholdWarningEvent` (at `budget_warning_fraction`), `BudgetExhaustedEvent` (at `budget_abort_fraction`, blocks retry). Both carry `tokens_used` and `budget_tokens`.

### Convergence Gate

Semantic early-exit gate. Fires acceptance when the surviving verified proposals from a wave are semantically close enough that another retry is unlikely to improve the result. Guarded by a `budget_floor_fraction` to prevent firing on mode-collapse (where proposals agree because they are all wrong).

```toml
[convergence_gate]
enabled                       = false
theta_converge                = 0.87
supermajority_fraction_n3     = 0.67
supermajority_fraction_n4plus = 0.80
score_floor                   = 0.80
min_wave                      = 1
budget_floor_fraction         = 0.30
```

| Field | Default | Purpose |
|---|---|---|
| `convergence_gate.enabled` | `false` | Master toggle. Requires an embedding model configured for `mean_pairwise_cosine()` to be meaningful. |
| `convergence_gate.theta_converge` | `0.87` | Mean pairwise cosine threshold above which surviving proposals are considered semantically converged. |
| `convergence_gate.supermajority_fraction_n3` | `0.67` | When N_surviving = 3, require this fraction (≥2 of 3) to exceed `score_floor`. |
| `convergence_gate.supermajority_fraction_n4plus` | `0.80` | When N_surviving ≥ 4, require this fraction to exceed `score_floor`. |
| `convergence_gate.score_floor` | `0.80` | Minimum score proposals must reach before their cosine distances contribute to the convergence check. |
| `convergence_gate.min_wave` | `1` | Do not fire on wave 0 regardless of cosine — allows at least one retry before early acceptance. |
| `convergence_gate.budget_floor_fraction` | `0.30` | Convergence gate is suppressed when cumulative budget consumed is below this fraction (prevents mode-collapse false positives early in the task when N_eff is low). |

`ConvergenceGateTriggeredEvent` is emitted when the gate fires, carrying `mean_cosine`, `n_surviving`, and `wave_index`. The `mean_pairwise_cosine()` function (`h2ai-autonomic::epistemic`) computes the upper-triangle mean cosine from surviving proposal embeddings.

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

### Correlated Hallucination and Universal Grounding

| Field | Default | Purpose |
|---|---|---|
| `correlated_hallucination_cv_threshold` | `0.30` | CV of pairwise Jaccard distances below which C1 fires. Set to `0.0` to disable C1 entirely. |
| `correlated_hallucination_min_jaccard_floor` | `0.50` | Mean pairwise Jaccard distance must also be **below** this floor for C1 to fire. Joint AND condition prevents spurious retries on genuinely-diverse equidistant ensembles (CV=0 but all distances high). |
| `domain_coverage_threshold` | `0.40` | Minimum fraction of corpus domains that slot `constraint_domains` must cover. Below this, `DiversityGuardDegradedEvent` fires. |
| `require_bivariate_cg` | `false` | When `true`, tasks fail rather than warn when domain coverage is below threshold. |
| `[grounding]` | — | Universal grounding checker (`GroundingChecker`) that verifies proposals/merged output against the spec boundary. |
| `grounding.enabled` | `true` | Set to `false` to skip the grounding check in development. |
| `grounding.max_tokens` | `8192` | Token budget for the `LlmGroundingJudge` researcher call. |
| `grounding.min_confidence` | `0.7` | Findings with `confidence < min_confidence` are discarded before gap production. |
| `grounding.tau` | `0.2` | Temperature for the `LlmGroundingJudge` call. Must be in [0, 1]; validated at config load. |
| `[gap_research]` | — | Settings for the `GapResearchChain` used by C1 reactive grounding and gap resolution. |
| `gap_research.grounding_distill` | `true` | When `true` and a researcher adapter is available, distill raw web-search results with the LLM before injection. |
| `gap_research.grounding_compress_threshold` | `800` | Maximum characters of a single grounding source's text before compression. Limits per-source token cost during distillation. |
| `gap_research.researcher_max_tokens` | `32768` | Max tokens for the `LlmResearcherGrounder` call. |
| `gap_research.distill_max_tokens` | `32768` | Max tokens for the web-search distillation LLM call. |
| `gap_research.gap_synthesis_max_tokens` | `32768` | Max tokens for GAP researcher synthesis calls. |

**Spec boundary construction:** The `effective_spec` passed to `GroundingChecker::new()` is built from three sources concatenated: `manifest.description`, `manifest.context` (the optional contextual background field — e.g. "We run Redis Cluster for caching"), and all constraint corpus text (`ConstraintDoc.description`, all entries of `ConstraintDoc.binary_checks`, `ConstraintDoc.pass_criteria` when present). Technologies named in any of these sources are treated as grounded and will not produce `UngroundedContent` gaps.

**Validation:** `H2AIConfig::validate()` is called by both `load_layered()` and `load_from_file()` and rejects configs where `grounding.tau ∉ [0, 1]` or `grounding.min_confidence ∉ [0, 1]` with a `ConfigLoadError::Validation` error before the config is returned to the caller.

### Pipeline Resilience

Four config structs (`crates/h2ai-config/src/lib.rs`) govern the pipeline resilience features. All must have two tests each: one via `H2AIConfig::default()` and one via `H2AIConfig::load_layered(None)` (verifying `reference.toml` is authoritative).

#### Verifier Freeze Detection

```toml
[verifier_freeze]
enabled                          = true
min_waves_to_detect              = 3
score_variance_threshold         = 0.05
reason_jaccard_threshold         = 0.7
reason_window_size               = 10
other_constraint_success_threshold = 0.5
bypass_hard_gate_on_freeze       = true
emit_event_only                  = false
```

| Field | Default | Purpose |
|---|---|---|
| `verifier_freeze.enabled` | `true` | Enable frozen verifier detection in `decide()`. |
| `verifier_freeze.min_waves_to_detect` | `3` | Minimum waves of history required before the signal can fire. |
| `verifier_freeze.score_variance_threshold` | `0.05` | Variance of per-wave scores below which "score not moving" holds. |
| `verifier_freeze.reason_jaccard_threshold` | `0.7` | Mean pairwise Jaccard of verifier reasons above which "reasons repeating" holds. |
| `verifier_freeze.reason_window_size` | `10` | Rolling window size for reason history per constraint; oldest entries evicted on overflow. |
| `verifier_freeze.other_constraint_success_threshold` | `0.5` | Minimum mean score required on at least one other constraint to confirm model capability (guards against model-ceiling false positives). |
| `verifier_freeze.bypass_hard_gate_on_freeze` | `true` | When `true`, frozen constraints enter `bypassed_verifier_constraints`; proposals failing only bypassed constraints pass pruning. |
| `verifier_freeze.emit_event_only` | `false` | When `true`, emit `VerifierFrozenEvent` but do not bypass. Useful for observability-only deployments. |

#### Generation Phase Timeout

```toml
[generation_phase]
timeout_secs = 300
```

| Field | Default | Purpose |
|---|---|---|
| `generation_phase.timeout_secs` | `300` | Wall-clock cap on the entire `join_all` across all explorers. On timeout: timed-out explorers emit `FailedReason::Timeout`; `generation_outcome` classifies the result; `AllTimedOut` routes to `ZeroSurvival`. |

#### OOM Circuit Breaker

```toml
[oom_guard]
enabled              = true
rss_abort_mb         = 4096
check_interval_waves = 1
```

| Field | Default | Purpose |
|---|---|---|
| `oom_guard.enabled` | `true` | Enable wave-boundary RSS polling. |
| `oom_guard.rss_abort_mb` | `4096` | RSS threshold in MB above which `oom_signal` fires → `BudgetExhausted` exit → clean checkpoint-and-exit. |
| `oom_guard.check_interval_waves` | `1` | Check RSS every N waves. |

#### Gap Feedback Quality

```toml
[gap_quality]
min_improvement_to_retain  = 0.1
min_post_injection_waves   = 2
```

| Field | Default | Purpose |
|---|---|---|
| `gap_quality.min_improvement_to_retain` | `0.1` | Minimum improvement in post-injection pass rate over `pre_injection_pass_rate` for a `DomainSynthesis` to be retained. Below this threshold after `min_post_injection_waves`, the entry is evicted as `Ineffective`. |
| `gap_quality.min_post_injection_waves` | `2` | Minimum post-injection observation waves before a verdict is issued (`Pending` until this many waves have elapsed). |

### Audit Gate

Controls Phase 4 auditor behaviour when the LLM returns a non-JSON response.

```toml
[audit_gate]
fail_open_on_parse_error = true   # true = pass through; false = reject (legacy behaviour)
```

| Field | Default | Purpose |
|---|---|---|
| `audit_gate.fail_open_on_parse_error` | `true` | When `true`, a non-JSON auditor response is treated as approved with an empty reason and a warning is logged (fail-open). When `false`, non-JSON is treated as rejection (legacy fail-safe). |

**Design invariant:** Proposals reaching Phase 4 have already passed the verifier gate. A non-JSON auditor response is an LLM formatting issue, not a constraint judgment. The default (`true`) prevents verified proposals from being silently dropped due to transient LLM response variance. Set to `false` only in environments where auditor reliability is independently verified and every non-JSON response should be treated as a safety rejection.

### Optimal Synthesis Policy (OSP)

OSP is disabled when the `[osp]` section is absent from `h2ai.toml`. When absent, `MergeEngine::resolve` uses legacy strategy dispatch unchanged.

```toml
[osp]
t_v                  = 0.125   # verifier noise temperature
concordance_alpha    = 0.1     # Hoeffding α for adaptive concordance threshold τ(N_f)
max_n_v_for_zone3    = 4       # Zone 3 suppressed when n_v (passing proposals) > this
accumulation_decay   = 0.7     # RetryAccumulator leaky λ (half-life ≈ 2 retries)
```

| Field | Default | Purpose |
|---|---|---|
| `osp.t_v` | `0.125` | Verifier noise temperature T_v. `ClearLeader` regime activates when Δ(top-2 scores) ≥ 2·T_v and P(correct) ≥ 0.92. Lower values make ClearLeader harder to reach (requires larger score gap). |
| `osp.concordance_alpha` | `0.1` | Hoeffding α used in adaptive concordance threshold τ(N_f) = clamp(0.5 + 0.5·√(−ln(α)/(2·N_f)), 0.5, 1.0). At α=0.1: τ(1)=1.0, τ(5)≈0.77, τ(10)≈0.66. Smaller α = stricter threshold = fewer Zone 3 injections. |
| `osp.max_n_v_for_zone3` | `4` | Zone 3 (AuditChannelBuilder) is suppressed when the number of passing proposals exceeds this value. Rationale: when many proposals pass, the merger has sufficient material without negative-signal guidance. |
| `osp.accumulation_decay` | `0.7` | RetryAccumulator decay λ. Per-retry update: `rate = λ·rate_old + (1−λ)·new_rate`. λ=0.7 gives half-life ≈ 2 retries. |

**OSP regime selection** (computed inside `MergeEngine::resolve`):

| Regime | Condition | Action |
|---|---|---|
| `ZeroSurvival` | N_v = 0 | WaveCollapse short-circuit (no LLM call) |
| `SingleSurvivor` | N_v = 1 | Return passing proposal directly |
| `ClearLeader` | Δ ≥ 2·T_v and P(correct) ≥ 0.92 | Select leader, skip ConsensusMedian |
| `TightCluster` | Δ < 2·T_v | ConsensusMedian over passing proposals only |

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

### Thinking Loop and Awareness Probe

`[thinking_loop]` is absent by default (opt-in). `thinking_loop.enabled = false` unless explicitly set.

| Field | Default | Purpose |
|---|---|---|
| `[thinking_loop]` | absent | Section absent = thinking loop disabled. |
| `thinking_loop.enabled` | `false` | Enable the Phase −1 multi-archetype brainstorm. |
| `thinking_loop.max_iterations` | `5` | Maximum brainstorm iterations before the loop terminates. |
| `thinking_loop.max_archetypes` | `4` | Maximum archetype count on iteration 0; contracts on subsequent iterations by coverage deficit. |
| `thinking_loop.coverage_threshold` | `0.75` | Loop terminates early when `coverage_score ≥ threshold`. |
| `thinking_loop.convergence_threshold` | `0.90` | Coverage score above which synthesis is considered converged (loop exits immediately). |
| `thinking_loop.tau_max` | `0.85` | Starting temperature — broad exploration on iteration 0. |
| `thinking_loop.tau_min` | `0.20` | Ending temperature — exploitation on the final iteration. |
| `thinking_loop.expansion_quality_floor` | `0.30` | Archetype count does not contract if fewer than this fraction pass the selection filter. |
| `thinking_loop.archetype_select_max_tokens` | `32768` | Token budget for each archetype-selection LLM call (ITER1 and ITERN). |
| `thinking_loop.brainstorm_max_tokens` | `32768` | Token budget per archetype brainstorm call. |
| `thinking_loop.quality_gate_max_tokens` | `64` | Token budget for the YES/NO quality gate. |
| `thinking_loop.synthesis_tournament_max_round_tokens` | `32768` | Token budget per pairwise tournament merge call. |
| `thinking_loop.oracle_timeout_secs` | `20` | Timeout for inline oracle check per archetype. |
| `thinking_loop.oracle_confidence_bonus` | `0.1` | `j_eff` boost applied when oracle passes. |

`[awareness_probe]` is absent by default. All fields require `enabled = true` to have any effect.

| Field | Default | Purpose |
|---|---|---|
| `[awareness_probe]` | absent | Section absent = probe disabled. |
| `awareness_probe.enabled` | `false` | Enable the Plan-Awareness Probe. Requires `thinking_loop.enabled = true`. |
| `awareness_probe.mode` | `"shadow"` | `"shadow"` — emit `AwarenessProbeCompletedEvent` only; `"active"` — re-iterate thinking loop on Hard non-gated `CONTRADICTED` verdicts. |
| `awareness_probe.judge_max_tokens` | `1024` | Token budget for the batched constraint judge (~100 tokens/constraint). |

`knowledge_domain_scoping = false` (top-level `H2AIConfig` field). When `true`, `CompositeProvider` pre-filters knowledge candidates to nodes whose `domains` intersect the task's domain tags before the BM25 query. Separate from the awareness probe.

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
| `H2AI_APPROVALS` | KV | `{task_id}` | 1 h | Legacy; still provisioned for backward compatibility but not actively written. Superseded by `H2AI_SIGNALS` JetStream stream for HITL signal delivery. |
| `H2AI_CONSTRAINT_WIKI` | KV | `wiki_cache` | — | Serialised `WikiCache` (context_map + metas). Loaded at startup; `constraint_wiki.enabled = true` required. History=5. |
| `H2AI_CONSTRAINT_PAYLOADS` | Object Store | `{id}@{version}` | — | Full predicate payloads for non-Static constraints (LlmJudge, Oracle). Fetched lazily at Phase 4. |
| `H2AI_CHECKPOINT_{tenant_id}` | KV store | `task_id` string | 7 days | `TaskReasoningCheckpoint` (zstd-compressed). Per-tenant; bucket created on first task for each tenant. |
| `H2AI_META_{tenant_id}` | KV store | `task_id` string | no TTL | `TaskMetaState` projections (uncompressed JSON). Per-tenant; consumed by InductionScheduler (Phase 2). |
| `H2AI_MEMORY` | KV store | `{tenant_id}.tag.{normalized_tag}` | no TTL | `TagPatternBucket` (vec of `RetryHintPattern`) for `NatsInductionScheduler` cross-task priming. Single shared bucket; tenant scoped via key prefix. A pattern with N trigger_tags appears in N tag buckets. |
| `H2AI_CONFLICT_{tenant_id}` | KV store | `"accumulator"` string | no TTL | `ConflictRateAccumulator` — rolling per-task conflict rates for `beta_quality` derivation. |
| `H2AI_CALIBRATION_RECORDS` | KV | `{adapter_profile}` | — | Global (not tenanted). `CalibrationRecord` per adapter profile — `n_useful_history` ring buffer `(N_useful: u8, N_max: u8, unix_minutes: u32)`. Written by the calibration harness; read by the epistemic β₀ `yield_from_history` path. Shared across all tenants because the adapter pool is shared infrastructure. |
| `H2AI_AUDITOR_HEALTH` | KV | `{adapter_profile}` | — | Global (not tenanted). `AuditorHealth` circuit-breaker state per adapter profile. Tracks `AuditorCircuitState` (Closed/Open/HalfOpen), consecutive failures, and `tripped_at` (unix millis). HalfOpen uses NATS KV `create` (CAS) as a probe lease serialiser. |
| `H2AI_PROBE_LEASE` | KV | `{adapter_profile}` | — | Global (not tenanted). Atomic probe-lease guards. `acquire_probe_lease` uses `kv.create()` (CAS, create-if-not-exists) with stale-lease eviction — only one caller wins per TTL window; `release_probe_lease` deletes the key. Serialises concurrent HalfOpen probe attempts across processes. |
| `H2AI_SKILLS` | KV | `{tenant_id_bucket_safe}/skills` | — | Per-tenant skill nodes extracted from resolved task traces (serialised JSON). Read and written by the `SkillStore` implementation in `nats.rs`. |

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
calibration_bucket             = "H2AI_CALIBRATION"
snapshots_bucket               = "H2AI_SNAPSHOTS"
task_checkpoints_bucket        = "H2AI_TASK_CHECKPOINTS"
checkpoint_payloads_bucket     = "H2AI_CHECKPOINT_PAYLOADS"
oracle_calibration_bucket      = "H2AI_ORACLE_CALIBRATION"
estimator_bucket               = "H2AI_ESTIMATOR"
sessions_bucket                = "H2AI_SESSIONS"
audit_shadow_bucket            = "H2AI_AUDIT_SHADOW"
approvals_bucket               = "H2AI_APPROVALS"
prompt_variants_bucket         = "H2AI_PROMPT_VARIANTS"
constraint_wiki_bucket         = "H2AI_CONSTRAINT_WIKI"
constraint_meta_bucket         = "H2AI_CONSTRAINT_META"
constraint_payloads_bucket     = "H2AI_CONSTRAINT_PAYLOADS"
calibration_records_bucket     = "H2AI_CALIBRATION_RECORDS"
auditor_health_bucket          = "H2AI_AUDITOR_HEALTH"
probe_lease_bucket             = "H2AI_PROBE_LEASE"
skills_bucket                  = "H2AI_SKILLS"

# Per-tenant bucket prefixes — actual bucket: {prefix}_{tenant_id}
reasoning_checkpoint_bucket_prefix = "H2AI_CHECKPOINT"  # TaskReasoningCheckpoint (7d TTL)
task_meta_state_bucket_prefix      = "H2AI_META"         # TaskMetaState projections
tenant_memory_bucket_prefix        = "H2AI_MEMORY"       # Defined but not consumed — NatsInductionScheduler uses hardcoded H2AI_MEMORY_BUCKET constant
conflict_beta_bucket_prefix        = "H2AI_CONFLICT"     # ConflictRateAccumulator

# JetStream stream names
tasks_stream             = "H2AI_TASKS"
telemetry_stream         = "H2AI_TELEMETRY"
results_stream           = "H2AI_RESULTS"
signals_stream           = "H2AI_SIGNALS"      # HITL signal delivery
signals_subject_prefix   = "h2ai.signals"      # subject prefix: {prefix}.{tenant_bucket_safe}.{task_id}

[state.delta]
enabled           = true
base_interval     = 10    # store a full base every N checkpoints
cache_ttl_secs    = 60
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
strict_audit_checkpoint     = false   # when true, a checkpoint write failure aborts the task
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
| `strict_audit_checkpoint` | `false` | When `true`, a `put_reasoning_checkpoint` failure propagates as `EngineError::CheckpointWriteFailed` and aborts the task. When `false` (default), write failures emit a warning and the task continues. Use `true` for regulated deployments that require a complete audit trail. |
| `induction_batch_size` | `10` | Number of `TaskMetaState` records consumed per `InductionScheduler` batch run (Phase 2). |
| `induction_max_interval_secs` | `86400` | Maximum interval between scheduled induction runs (Phase 2). |
| `induction_max_tasks_per_run` | `50` | Hard cap on tasks processed in a single induction run (Phase 2). |
| `tag_gate_threshold` | `0.2` | Minimum tag-overlap score for a `TaskMetaState` to be retrieved as a prior (Phase 2). |
| `max_archetype_boost` | `0.15` | Maximum score boost applied to an archetype with a strong positive prior (Phase 2). |
| `max_archetype_penalty` | `0.20` | Maximum score penalty applied to an archetype with a strong negative prior (Phase 2). |

Per-tenant NATS KV bucket name prefixes are configured under `[state]`:

| Field | Default | Purpose |
|---|---|---|
| `state.reasoning_checkpoint_bucket_prefix` | `"H2AI_CHECKPOINT"` | Prefix for per-tenant reasoning checkpoint buckets (`{prefix}_{tenant_id}`). 7-day TTL. |
| `state.task_meta_state_bucket_prefix` | `"H2AI_META"` | Prefix for per-tenant meta-state buckets (`{prefix}_{tenant_id}`). No TTL. |
| `state.tenant_memory_bucket_prefix` | `"H2AI_MEMORY"` | Defined in config but not consumed by production code. `NatsInductionScheduler` uses the hardcoded `H2AI_MEMORY_BUCKET` constant directly. |
| `state.conflict_beta_bucket_prefix` | `"H2AI_CONFLICT"` | Prefix for per-tenant conflict-rate accumulators (`{prefix}_{tenant_id}`). No TTL. |

**`TaskReasoningCheckpoint` schema and phase lifecycle:**

The engine writes progressive checkpoints at each gate. All writes are fire-and-forget (non-fatal) unless `strict_audit_checkpoint = true`.

```rust
struct TaskReasoningCheckpoint {
    task_id: TaskId,
    tenant_id: TenantId,
    created_at: u64,               // Unix seconds
    last_updated: u64,             // Unix seconds; updated on every write
    phase: ReasoningCheckpointPhase,

    // Set at task start (phase = Created)
    constraint_tags: Vec<String>,
    domain: Option<String>,
    task_quadrant: Option<TaskQuadrant>,
    system_context_with_rubric_hash: u64,
    constraint_corpus_fingerprint: u64,

    // Populated after thinking loop (phase >= ThinkingDone)
    shared_understanding: Option<String>,
    tensions: Option<Vec<String>>,
    archetype_selection: Option<Vec<ArchetypeSelection>>,  // name + confidence
    thinking_iterations: Option<u32>,

    // Appended after each adapter wave (phase = WaveCompleted(k), 0-based)
    completed_waves: Vec<CompletedWave>,   // wave_index + per-adapter output_hash + survived

    // Populated at resolution (phase = Resolved)
    retry_count: u32,
    retry_context_that_resolved: Option<String>,
    tried_topologies: Vec<TopologyKind>,
    tau_values_that_converged: Option<Vec<f64>>,
    resolved_attribution_json: Option<String>,  // HarnessAttribution serialized as JSON
    resolved_waste_ratio: Option<f64>,
}
```

Phase transition sequence:

| Phase | Written when | Key fields added |
|---|---|---|
| `Created` | Task accepted, before any LLM call | `constraint_tags`, `domain`, `task_quadrant`, hash fingerprints |
| `ThinkingDone` | Thinking loop completes | `shared_understanding`, `tensions`, `archetype_selection`, `thinking_iterations` |
| `WaveCompleted(k)` | Each adapter wave `k` finishes (0-based) | `completed_waves[k]` with per-adapter `output_hash` and `survived` flag |
| `MergeDone` | Merge phase output written | (no new fields; phase marker only) |
| `Resolved` | Task fully resolved, `TaskMetaState` projected | `retry_count`, `tried_topologies`, `resolved_attribution_json`, `resolved_waste_ratio` |

`AdapterWaveOutput` stores only a xxHash64 of the output text — full text is not stored to keep checkpoint size small. `resolved_attribution_json` and `resolved_waste_ratio` are written at resolution so that `run_from_checkpoint` can hydrate a complete `EngineOutput` without re-running inference, preventing zeroed attribution from corrupting downstream analytics.

**`TaskMetaState` projection:**

At resolution, `TaskReasoningCheckpoint::into_meta_state()` projects the checkpoint into an immutable `TaskMetaState` (drops wave-level detail; keeps reasoning artifacts and retrieval index). Stored in `H2AI_META_{tenant_id}` with no TTL. Read by `InductionScheduler` for distillation cycles (Phase 2).

**`run_from_checkpoint` recovery:**

When the engine is invoked with a task that already has a `TaskReasoningCheckpoint` in NATS, it loads the checkpoint and skips phases already completed:

- Phase ≥ `ThinkingDone` → skip thinking loop; restore `shared_understanding`, `tensions`, `archetype_selection`
- Phase ≥ `WaveCompleted(k)` → skip adapter waves 0..=k; resume from wave k+1
- Phase ≥ `MergeDone` → skip merge; go directly to resolution
- Phase = `Resolved` → deserialize `resolved_attribution_json` and restore `resolved_waste_ratio`; return the cached `EngineOutput` without any LLM calls

Phase ordering is enforced by `ReasoningCheckpointPhase::is_at_least()` using a total rank: `Created(0) < ThinkingDone(1) < WaveCompleted(k) = 2+k < MergeDone(MAX-1) < Resolved(MAX)`.

### Multi-Variant Judge Panel

Phase 3.5 verification now uses a `JudgePanel` instead of a single `LlmJudge` adapter.

**Panel construction** (in `phases/verify.rs`):
- If ≥2 distinct adapter families are present across `verification_adapter` + `explorer_adapters`: one cross-family variant per family, `PanelDiversityKind::CrossFamily`, cap 3 total.
- If 1 family: 3 persona variants (Literal, Contextual, Skeptical) at temperatures [0.0, 0.2, 0.4], `PanelDiversityKind::PersonaOnly`.

**Verdict aggregation** per constraint:
- `CrossFamily`: supermajority (configurable `quorum_fraction`, default 0.67) → Pass / Fail / Uncertain.
- `PersonaOnly`: unanimous agreement → Pass / Fail; any split → Uncertain.

**Uncertain proposals**: pass with `score × uncertainty_weight` (default 0.7) on uncertain constraints. Not pruned. Hard-fail gate skips uncertain constraints.

**`ConstraintAmbiguityEvent`**: emitted (fire-and-forget, tracing log) when ≥`ambiguity_threshold` (default 2) proposals in a wave show uncertain votes for the same constraint. Signals corpus quality issue.

**Config**: `[judge_panel]` section in `reference.toml`.

**Research**: PoLL (2404.18796), CARE (2603.00039), Prosa (2605.01630), Logarithmic Scores (2604.00477).

---

### Epistemic Output Quality

`[epistemic_quality]` is disabled by default. Enable to run the post-merge gap detection and provision annotation pipeline that emits `ProvenanceRecordedEvent`.

```toml
[epistemic_quality]
enabled                      = false
coherence_check_enabled      = true
coherence_min_severity       = "medium"
recovery_enabled             = true
recovery_max_passes          = 2
recovery_tau                 = 0.5
zero_valid_proposals_policy  = "fail"
output_mode                  = "clean"
```

| Field | Default | Purpose |
|---|---|---|
| `epistemic_quality.enabled` | `false` | Master switch. When `false`, the entire post-merge epistemic stage is skipped; `ProvenanceRecordedEvent` is never emitted. |
| `epistemic_quality.coherence_check_enabled` | `true` | Run `CoherenceChecker` after `SelectionPruningExtractor`. Adds one LLM call (τ=0.7, max_tokens=1024) per task when enabled. |
| `epistemic_quality.coherence_min_severity` | `"medium"` | Minimum `GapSeverity` (`"low"`, `"medium"`, `"high"`) for CoherenceChecker inter-provision conflicts to be included. |
| `epistemic_quality.recovery_enabled` | `true` | Run `MicroExplorerResolver` on each gap batch produced by `GapRegistry::dispatch_batches()`. When `false`, gaps are recorded but no recovery LLM calls are made. |
| `epistemic_quality.recovery_max_passes` | `2` | Maximum number of gap-resolution passes before the pipeline accepts the current `ProvenanceMap` state. |
| `epistemic_quality.recovery_tau` | `0.5` | Minimum score delta for a resolved patch to be accepted. Currently unused — the resolver uses binary acceptance (non-empty patch = score_delta=1.0). Reserved for a future continuous scoring mode. |
| `epistemic_quality.zero_valid_proposals_policy` | `"fail"` | Behaviour when `open_gap_count` equals `provision_count` (nothing resolved): `"fail"` → emit `TaskFailed(NoValidProposals)`; `"deliver_unverified"` → proceed and emit `ProvenanceRecordedEvent` with `document_confidence = "Unverified"`. Use `"deliver_unverified"` in audit pipelines that must surface output regardless of quality. |
| `epistemic_quality.output_mode` | `"clean"` | Output annotation mode. `"clean"` → prepend blockquote confidence header only. `"audit"` → header + per-provision annotations for every non-`Verified` provision (with `gap_ids`) + footer (`Document confidence: {label} | Provisions reviewed: {N}`). |

**Pipeline sequence** (when `enabled = true`): `SelectionPruningExtractor.extract_gaps_from_pruned()` → `CoherenceChecker.check()` (if `coherence_check_enabled`) → `GapRegistry::dispatch_batches()` (Kahn's topological sort) → per-batch `MicroExplorerResolver.resolve()` (if `recovery_enabled`) → `ProvenanceMap::document_confidence()` → `OutputRenderer::render_output()` → `ProvenanceRecordedEvent`.

**Gap ID scheme**: SelectionPruning gaps: `g{1-based-index}` (deterministic, deduplicated by description). CoherenceCheck conflicts: `coh-{1-based-index}`.

**`ProvisionConfidence` strict order** (worst-wins dominance): `Verified(0) < AutoCorrected(1) < ReviewRecommended(2) < RequiresReview(3) < Unverified(4)`. `derive(PartialOrd, Ord)` encodes the order. `AutoCorrected` collapses to `High` at document level — a patched provision has no unresolved gap.

**Implementation**: `crates/h2ai-orchestrator/src/gap_checkers/`, `gap_registry.rs`, `gap_resolvers/`, `provenance.rs`, `output_renderer.rs`.

---

### Conflict-Rate β

```toml
[conflict_beta]
enabled                  = true
max_samples              = 100
halflife_secs            = 604800   # 7 days — same halflife as CG samples
min_samples_for_override = 5        # production tasks needed before rolling overrides floor
```

| Field | Default | Purpose |
|---|---|---|
| `enabled` | `true` | Master switch. When `false`, no conflict rates are computed or stored; `beta_quality` remains `None` and `beta_eff` falls back to the latency-based proxy. |
| `max_samples` | `100` | Maximum number of per-task conflict rate samples retained in the rolling window. Oldest samples are evicted when cap is reached. |
| `halflife_secs` | `604800` | Exponential decay halflife (7 days). Samples older than 7 days contribute at 50% weight; 14-day-old samples at 25%. Matches `CG_HALFLIFE_SECS`. |
| `min_samples_for_override` | `5` | Rolling window must contain at least this many production task samples before it overrides the calibration floor. Prevents noisy early samples from corrupting N_max. |

Per-tenant accumulator bucket prefix is configured under `[state]`:

| Field | Default | Purpose |
|---|---|---|
| `state.conflict_beta_bucket_prefix` | `"H2AI_CONFLICT"` | Prefix for per-tenant conflict-rate accumulator buckets (`{prefix}_{tenant_id}`). No TTL — long-lived tenant record. |

**`ConflictRateAccumulator` schema:**

```rust
struct ConflictRateAccumulator {
    tenant_id: TenantId,
    calibration_floor: f64,           // from Phase B — never overwritten
    samples: Vec<ConflictRateSample>, // capped at max_samples
    beta_quality: f64,                // cached temporal-decay weighted mean
    total_tasks_seen: u64,
    last_updated: u64,                // Unix seconds
}

struct ConflictRateSample {
    conflict_rate: f64,   // mean pairwise Hamming(v_i, v_j)/K ∈ [1e-6, 1.0]
    n_adapters: u32,
    timestamp: u64,       // Unix seconds — required for halflife decay
}
```

**Data flow:**

1. Phase B calibration → `compute_conflict_rate(adapter_outputs, corpus)` → `CalibrationCompletedEvent.beta_quality` → written as `calibration_floor` to `H2AI_CONFLICT_{tenant}`
2. Production task verify phase → conflict rate from raw proposal texts → `WaveEvents.conflict_rate`
3. Engine after each wave → `ConflictRateAccumulator.push_sample()` → write to NATS (fire-and-forget)
4. Engine task start → load accumulator → if `total_tasks_seen >= min_samples_for_override`: override `complexity_out.n_max_ceiling` using `beta_quality`

**`beta_eff()` dispatch:**

```
if CoherencyCoefficients.beta_quality.is_some():
    beta_eff = beta_quality          # direct measurement — no CG adjustment
else:
    beta_eff = beta_base × (1−CG)   # latency-based proxy (legacy fallback)
```

### Empirical ρ EMA (`rho_ema`)

`RhoEmaState` accumulates pairwise error-correlation estimates from production task waves.
Once sufficient observations exist, it replaces the `rho_mean = 1 − CG_mean` proxy in
ensemble sizing and Q(N,p,ρ) Condorcet calculations.

**Parameters (hardcoded in `rho_ema.rs` — not yet in reference.toml):**

| Parameter | Value | Purpose |
|---|---|---|
| EMA alpha | `0.10` | Smoothing factor per update. Effective window ≈ 10 tasks. |
| Cold-start prior | `0.45` | Per-pair ρ prior before first observation (conservative mid-range). |
| Steady-state threshold | `30` observations | After 30 pairwise observations, `rho_mean()` is used in place of the CG proxy. |

**How it works:**
After each task wave, the engine computes pairwise centered score products for all
`(adapter_i, adapter_j)` pairs: `product = (score_i − p_mean) × (score_j − p_mean) / variance`,
clamped to `[−1, 1]`. Each pair's EMA is updated as:
`ema_new = (1 − α) × ema_old + α × product`.

`rho_mean()` returns the mean over all pair EMAs, clamped to `[0.0, 0.99]`.
Before any updates (empty map), returns `0.45`.

**State storage:** In-memory per `TenantState` in the API process (`Arc<RwLock<RhoEmaState>>`).
Not persisted to NATS KV — resets on server restart (steady-state reached within ~30 tasks
of production traffic).

**Transition to config:** When `rho_ema` steady state replaces the CG proxy, the transition
is logged via `n_observations` on `CalibrationCompletedEvent`. Future work: persist EMA
snapshot to `H2AI_CALIBRATION` KV alongside α and β so it survives restarts.

### Oracle Gate

```toml
[oracle_gate]
enabled = false
subject = "h2ai.oracle.gate"
timeout_secs = 30
on_timeout = "pass"         # pass | fail | skip
on_fail    = "evict"        # evict | pass | fail  (post-selection gate)
min_confidence = 0.7
# clarification_templates = [{ pattern = "...", question_template = "..." }]
```

| Field | Default | Purpose |
|---|---|---|
| `enabled` | `false` | Master switch; set `true` to wire Phase 4.5. |
| `subject` | `"h2ai.oracle.gate"` | NATS subject for `request()` calls. The oracle service subscribes here. |
| `timeout_secs` | `30` | How long to wait for an oracle reply before applying `on_timeout`. |
| `on_timeout` | `"pass"` | `pass` — treat timeout as approved; `fail` — treat as rejected; `skip` — proceed with no `oracle_gate_passed` field. |
| `on_fail` | `"evict"` | BFT Lever 2 — post-selection gate action when winner fails oracle check. `evict` — block winner, rotate adapter family, emit `CorrelatedEnsembleWarning`, retry; `pass` — ignore and proceed; `fail` — mark task failed. Implemented via `PostSelectionDecision` / `run_post_selection` in `phases/oracle.rs`. |
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

Signal delivery uses JetStream durable push consumers (live since 2026-05-19). The `H2AI_APPROVALS` KV bucket is still provisioned for backward compatibility but no longer written to; the `approval_reaper` background task has been removed.

| Field | Default | Purpose |
|---|---|---|
| `hitl.enabled` | `true` | Master switch; set `false` to bypass gate in dev/test. |
| `hitl.confidence_threshold` | `0.50` | `q_confidence` below this triggers human review. |
| `hitl.timeout_ms` | `14_400_000` | Base review window (4 hours). Decays exponentially on consecutive non-responses (see `timeout_decay`). |
| `hitl.timeout_decay` | `0.5` | Multiplier applied per consecutive timeout miss: `effective_ms = timeout_ms × decay^hitl_timeouts_fired`. Must be in (0.0, 1.0). |
| `hitl.timeout_floor_ms` | `300_000` | Minimum effective window regardless of decay (5 minutes). |
| `signal_wave_window_ms` | `0` | ms to pause at each `WaveCompleted` boundary waiting for a `WaveContinue` signal. `0` = disabled. |
| `signal_min_timeout_ms` | `60_000` | Lower bound for caller-supplied `timeout_ms` in `POST /signal` body; values below this are clamped up. |
| `signal_max_timeout_ms` | `86_400_000` | Upper bound for caller-supplied `timeout_ms` in `POST /signal` body; values above this are clamped down. |

**JetStream stream:** `H2AI_SIGNALS` (subject `h2ai.signals.>`). Per-task consumer: `SIGNAL-{task_id_no_dashes}` (durable push, created before the phase loop, deleted on task close).

**Signal subject:** `h2ai.signals.{tenant_bucket_safe}.{task_id}`.

**`hitl_timeouts_fired`** is stored in `TaskReasoningCheckpoint` with `#[serde(default)]` so old checkpoints deserialize cleanly. Resets to 0 on successful operator response.

### Constraint Wiki

| Field | Default | Purpose |
|---|---|---|
| `constraint_wiki.enabled` | `false` | When `true`, corpus access routes through `NatsWikiConstraintSource` (NATS KV + Object Store). When `false`, falls back to `YamlDirSource` (flat directory). |
| `constraint_wiki.corpus_path` | `"/constraints"` | Filesystem path for `YamlDirSource` (used when `enabled = false`). Ignored when wiki is enabled. |
| `constraint_wiki.resolve_k` | `50` | Reserved: max constraints returned per `resolve_context` call (future Qdrant semantic search limit). |

### Knowledge Provider

The `[knowledge]` section is optional. When absent, `PassthroughProvider` is used (delegates directly to `ConstraintResolver` — zero behaviour change from pre-knowledge operation). When present with `provider = "Bm25Wiki"`, `Bm25WikiProvider` is built at startup: BM25+ indexed over constraint leaf nodes and optional `wiki/` topic nodes, with Personalized PageRank for multi-hop expansion.

```toml
[knowledge]
provider = "Bm25Wiki"

[knowledge.source]
YamlDir = { path = "tests/e2e/constraints" }
```

The `[knowledge.source]` table is **externally-tagged**: the key (`YamlDir`) is the variant name and its value is the struct body. Currently only `YamlDir` is supported. The path is resolved relative to the process working directory at startup.

**Corpus layout under `YamlDir.path`:**
- `*.yaml` — constraint leaf files (standard `ConstraintDoc` schema, `id`, `domains`, `related`, etc.)
- `wiki/` — optional topic node files (see below); absence is graceful — a synthetic global node is built from constraint summaries
- `wiki/_overview.yaml` — optional global overview node (`id`, `depth: global`, `synthesis`, `domains`)
- `wiki/<topic>.yaml` — topic cluster node (`id`, `depth: topic`, `synthesis`, `domains`, `entry_points: [C-004, ...]`, `invariants`, `failure_modes`)

**ScoringConfig defaults** (all fields are optional in `[knowledge.scoring]`):

| Field | Default | Purpose |
|---|---|---|
| `knowledge.scoring.leaf_score_multiplier` | `0.7` | Multiplier on raw BM25+ score for direct leaf hits before boost application |
| `knowledge.scoring.id_in_query_boost` | `0.15` | Boost added when a constraint ID (e.g. `C-004`) appears literally in query text |
| `knowledge.scoring.entry_point_boost` | `0.10` | Boost for leaf nodes listed as `entry_points` in a matched topic cluster |
| `knowledge.scoring.ppr_score_multiplier` | `0.3` | Multiplier on raw PPR probability mass for PPR-expanded (multi-hop) nodes |
| `knowledge.scoring.ppr_alpha` | `0.15` | PPR teleportation probability (restart probability); standard value 0.15 |
| `knowledge.scoring.ppr_max_iter` | `20` | PPR power-iteration steps; 20 converges for graphs up to ~1k nodes |
| `knowledge.scoring.topic_cluster_top_k` | `3` | Max topic clusters matched per query in TreeTraversal mode |
| `knowledge.scoring.global_synthesis_max_chars` | `600` | Max characters retained in the synthesized global overview node |

**Relationship to `constraint_wiki`:** Both systems can be active simultaneously. `constraint_wiki` handles mandatory tag-based injection (always runs, enforces hard-gate predicates). `knowledge` handles semantic retrieval into `GlobalKnowledge`/`TopicKnowledge` context assembler sections (optional, runs when `[knowledge]` is configured). If a constraint ID appears in both paths, the context assembler deduplicates it with the higher-importance entry winning.

**Context assembler integration (+ live 2026-05-18):** The knowledge provider is invoked per explorer slot in `generation.rs` Phase B1. Each slot's `agent_role` maps via `profile_for_role()` to a `KnowledgeProfile` that selects RAPTOR mode (TreeTraversal / CollapsedTree), PPR `expand_hops` (0–2), `top_k`, and `domain_tag_boost`. Query results populate `ContextAssemblerInput`:
- `global_knowledge` → `SectionTag::GlobalKnowledge` (importance=1.0, preserve=true — survives all compression passes)
- `topic_knowledge` → `SectionTag::TopicKnowledge` (importance=0.8, preserve=false); only present when `domain_tag_boost=true` and matching domain nodes exist
- `constraint_tensions` → `SectionTag::ConstraintTension` (importance=0.85, preserve=false); only present for Synthesizer slots when `SurfacedTension` entries are non-empty

The `InductionStore` (key format `knowledge.{node_id}.{role}`) records `KnowledgeNodePattern{node_id, role, domain_tags, hit_rate}` after each accepted merge. On subsequent tasks with matching `domain_tags`, high-hit_rate `node_ids` are prepended as `explicit_ids` in the query — bypassing BM25+ scoring for known-good nodes. Phase 1 approximation: `record()` uses `domain_tags` as proxy node IDs until full node_id threading through `EngineOutput` is plumbed. Failures at any layer degrade silently to `None` — task execution never blocks on knowledge. The bucket name is passed as a constructor parameter to `InductionStore::create()`; no default bucket name constant is defined in the codebase.

**Partial wiring note (2026-05-18):** `InductionStore` field exists on `EngineInput` in `engine.rs`. However, `task_pipeline.rs` and `recovery.rs` currently pass `induction_store: None` when constructing `EngineInput` — induction boost is not active for any submitted tasks. Follow-up work: define a bucket name constant and wire `AppState.induction_store` into both handlers.

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

The corpus is a directory of YAML files (preferred) or legacy markdown files. The Constraint Compiler reads them and produces `ConstraintDoc` with a `Composite` predicate — all constraints share the same bytecode shape regardless of authoring format.

### Format

**Preferred: YAML with `semantic:` section** (`crates/h2ai-constraints/src/yaml.rs`)

```yaml
id: CONSTRAINT-001
title: Stateless Authentication
severity: Hard
threshold: 0.9
domains: [auth, session]
remediation_hint: "Authentication must be JWT-based with no server-side session state."
semantic:
  exclusions:
    - pattern: "server-side session or sticky session"
      passes: 3
  requirements:
    - concept: "JWT stateless authentication"
      passes: 3
quality:
  pass: "Proposal uses JWT with no session state."
  fail: "Proposal stores session state server-side."
```

`semantic.exclusions` produce `SemanticExclusion` children (structural anti-pattern gates). `semantic.requirements` produce `SemanticPresence` children (structural must-have gates). `semantic.orderings` produce `SemanticOrdering` children (operation-order gates). All are compiled into a `Composite(And([...gates..., LlmJudge]))` predicate.

**Legacy: markdown** (still loaded; logs deprecation warning when `predicates:` key present)

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

**Predicate kinds (legacy markdown):**

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
- **`YamlDirSource`** (`crates/h2ai-constraints/src/loader.rs`): implements `ConstraintSource` over a filesystem directory; loads YAML constraint files via `into_semantic_spec()`, deduplicates by ID, and logs a deprecation warning when the legacy `predicates:` key is present. Replaces the old `load_corpus()` function for production use.
- **`RuntimeConstraintStore`** (formerly `FsConstraintStore`, backward-compat alias preserved): builds an in-memory `ConstraintIndex` from any `ConstraintSource`; callers that held `FsConstraintStore` continue to compile without changes.
- **`NatsWikiConstraintSource`** (in `h2ai-api`): reads from NATS KV `H2AI_CONSTRAINT_WIKI` + Object Store; enables hot-reload without restart.
