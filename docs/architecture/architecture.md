# H2AI Architecture

H2AI Control Plane is a Rust runtime that coordinates pools of LLM adapters as an
adversarial committee: independent generators, an independent verifier, and an independent
auditor produce a resolved output that is more reliable than any single adapter.  The runtime
treats this committee as a physical system whose throughput, diversity, and quality are
computable, calibrated, and bounded by the Universal Scalability Law and the Condorcet Jury
Theorem.

Mathematical foundations are in [`math.md`](math.md).
Configuration reference is in [`reference.md`](reference.md).
Operations and deployment are in [`operations.md`](operations.md).
Research status and open questions are in [`research-state.md`](research-state.md).

---

## 1. System Overview

```
Client
  │  POST /v1/tasks
  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ h2ai-api  (Axum HTTP server, 0.0.0.0:8080)                         │
│  task_pipeline.rs — orchestrates the three-stage pipeline per task  │
│  recovery.rs      — resumes in-flight tasks on startup             │
│  oracle_worker.rs — consumes OracleResultEvent from NATS           │
│  metrics.rs       — Prometheus text exposition at /metrics          │
└─────────────────┬───────────────────────────────────────────────────┘
                  │ Arc<ExecutionEngine>
                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ h2ai-orchestrator                                                   │
│  engine.rs        — ExecutionEngine, MAPE-K wave driver            │
│  mape_k.rs        — MapeKController, PipelineParams, WaveEvents    │
│  decomposition.rs — Phase 0 constraint corpus + role decomposition │
│  phases/          — per-phase implementations (complexity, verify…) │
│  grounding_chain.rs — GapResearchChain (hallucination grounding)   │
│  verification.rs  — constraint evaluation, judge panel             │
│  oracle_gate.rs   — Phase 6 oracle dispatch                        │
└──────────┬──────────────────────────────────────────────────────────┘
           │ async_nats / Arc<dyn NatsBackend>
           ▼
      NATS JetStream                   LLM adapters (IComputeAdapter)
      (KV buckets + streams)           (Ollama, OpenAI, Anthropic, A2A…)
```

### Tenant isolation

Each tenant has its own `Arc<TenantState>` stored in a `DashMap<TenantId, Arc<TenantState>>`.
Per-tenant NATS KV bucket prefixes (`H2AI_CHECKPOINT_{tenant}`, `H2AI_META_{tenant}`,
`H2AI_MEMORY_{tenant}`, `H2AI_CONFLICT_{tenant}`) keep all persistent state isolated.

---

## 2. Task Pipeline

The `run_task_pipeline()` function in `crates/h2ai-api/src/task_pipeline.rs` orchestrates
every task through three sequential stages.  Dependencies are injected via
`Arc<dyn ThinkingLoopRunner>`, `Arc<dyn Decomposer>`, and `Arc<dyn EngineRunner>` traits
(mockable for testing).

### Stage 1 — Thinking loop (optional)

Disabled by default (`thinking_loop.enabled = false`).  When enabled, runs up to
`max_iterations = 5` rounds of archetype selection → parallel brainstorm → synthesis
tournament, producing `shared_understanding` text and `archetypes` (up to `max_archetypes = 4`).
Output is appended to the task context under `## Thinking Loop Analysis` before the
decomposition stage.

Key config (`ThinkingLoopConfig`): `coverage_threshold = 0.75`, `convergence_threshold = 0.90`,
`tau_max = 0.85`, `tau_min = 0.20`.

### Stage 2 — Decomposition (Phase 0)

Three-step LLM call sequence builds the constraint corpus and explorer role assignments:
1. Analysis — structural understanding of the task (`decomposition_step_max_tokens = 32768`).
2. Role design — assigns explorer personas and temperature spreads.
3. JSON formatting — serialises the decomposition into engine input (`decomposition_json_max_tokens = 32768`).

### Stage 3 — Engine (MAPE-K waves)

`ExecutionEngine::run_offline()` drives the MAPE-K autonomic control loop.
On success it returns `EngineOutput`; on failure it returns `EngineError` paired with
`EngineRunContext` containing partial events for skill extraction and HITL surfacing.

---

## 3. Execution Phases

Each MAPE-K wave runs the following phases in order.  The phase number prefixes come from
internal event naming and the order in which monitoring events are emitted.

| Phase | Name | Description |
|-------|------|-------------|
| 1.5 | Complexity assessment | `assess_task_complexity()` assigns a `TaskQuadrant` and TCC |
| 2 | Topology provisioning | Selects ensemble topology (Self-MoA, cross-family committee, CoT) based on quadrant |
| 2.6 | Diversity guard | `DomainCoverageEvent` emitted; fires `DiversityGuardDegradedEvent` when `domain_coverage_threshold = 0.40` is not met |
| 3 | TAO generation | Explorer agents generate candidate outputs (one per slot); each slot runs up to `agent_max_tool_iterations = 5` TAO loop turns |
| 3.5 | Verification | `verify_proposals()` evaluates each candidate against the constraint corpus; judge panel (`JudgePanelConfig`) runs multi-persona consensus |
| 4 | Audit | Shadow auditor compares primary verifier vs shadow verifier; promotes domains to AND-vote when disagreement rate exceeds threshold |
| 5a | Synthesis | Two-stage critique → write synthesis call when ≥ `synthesis_min_proposals = 2` verified proposals exist; sequential grafting available (disabled by default) |
| 6 | Oracle | `oracle_dispatch::fire()` publishes `OraclePendingEvent` to `h2ai.oracle.{tenant_id}.pending`; oracle worker consumes `OracleResultEvent` asynchronously |

### Phase 1.5 detail

`TaskComplexityAssessedEvent` fields reported per task:
- `tcc_structural`, `tcc_empirical`, `tcc_effective`: three TCC estimates
- `n_eff_pool`: pool-level N_eff from calibration
- `task_quadrant`: routing decision (`Precision / Coverage / Complex / Degenerate`)
- `probe_skipped`, `probe_skip_reason`: whether the LLM complexity probe was bypassed
- `heavy_fraction`, `tcc_mismatch`, `n_informative_static`

Degenerate guard (non-shadow mode): when `task_quadrant == Degenerate` the engine
immediately marks the task failed with `MultiplicationConditionFailure::InsufficientPoolDiversity`.
Quorum degradation guard: when `n_max_degraded()` is true the engine fails with
`MultiplicationConditionFailure::QuorumDegradedBelowMinimum`.
Bootstrap guard: when `calibration_quality == Bootstrap` (synthetic priors only, no real
adapter data), Phase 1.5 routes unconditionally to Coverage and skips the N-probe sampling.

### Phase 3.5 detail

`VerificationScoredEvent` fields per proposal:
- `score`: aggregate compliance score `[0, 1]`
- `score_lower`, `score_upper`: Wilson score confidence interval
- `passed`: whether score ≥ `verify_threshold` (default 0.45)
- `passed_checks`, `total_checks`: binary check counts
- `per_check_verdicts`: per-constraint verdict list (`CheckVerdict`)
- `cache_hit`: whether the evaluation was served from the per-task `EvalCache`

Cache hit threshold: `CACHE_SIMILARITY_THRESHOLD = 0.85` (Jaccard similarity).

---

## 4. MAPE-K Retry Loop

The `MapeKController` drives up to `max_autonomic_retries = 2` retry waves per task.
On each wave it:
1. Selects `PipelineParams` (topology, τ factors, adapter rotation, retry context, bypass hints).
2. Executes the pipeline, receiving `WaveEvents`.
3. Calls `observe()` on `WaveEvents` to update internal state.
4. Calls `RetryPolicy::decide()` to determine the next action.

### MAPE-K failure modes

`ZeroSurvivalEvent` is emitted when all proposals fail verification in a wave.
Its `failure_mode` field carries one of:

| Variant | Meaning |
|---------|---------|
| `ConstrainedExploration` | All slots hit hard constraint walls — search space is over-constrained |
| `ModeCollapse` | Proposals converged to a single mode (low Jaccard diversity) |
| `CorrelatedHallucination { cv, mean_jaccard_distance }` | Low coefficient of variation on pairwise distances AND mean distance below `correlated_hallucination_min_jaccard_floor = 0.50` |

Condition for `CorrelatedHallucination`: both `cv < correlated_hallucination_cv_threshold = 0.30`
AND `mean_jaccard_distance < 0.50` must hold simultaneously.

### MAPE-K WaveEvents

`WaveEvents` aggregates per-wave observables:
- `verification_events`: `Vec<VerificationScoredEvent>`
- `failed_proposals`: `Vec<ProposalFailedEvent>`
- `correlated_warnings`: `Vec<CorrelatedEnsembleWarning>`
- `filter_ratio`: `surviving / total_evaluated` (1.0 on early exit)
- `pruned_events`: `Vec<BranchPrunedEvent>` (accumulated so `RetryPolicy::decide` can extract `reason` strings for `RetryWithHints`)
- `conflict_rate`: mean pairwise constraint-conflict rate (`None` when < 2 proposals)
- `wave_token_cost`: sum of `ProposalEvent.token_cost` from this wave
- `best_passing_constraint_reasons`: per-constraint verifier reasons from best-passing proposal

---

## 5. Grounding Subsystems

Two independent grounding subsystems detect different classes of ungrounded content.

### GroundingChecker (gap_checkers/grounding.rs)

Runs after merge to check whether the merged output references entities not grounded in
the effective spec.  Seeds `UngroundedContent` gaps in the static gap list.

The inner judge is selected by `grounding.enabled`:
- `enabled = true` (default): `LlmGroundingJudge` — LLM call using `Reasoning` adapter profile,
  `max_tokens = 8192`, `tau = 0.2`.
- `enabled = false`: `HeuristicGroundingJudge` — rule-based fallback (no LLM call).

`GroundingChecker` wraps the active judge and implements the `GapChecker` trait.
`CompositeGroundingJudge` (combines an arbitrary list of `GroundingJudge` impls) is
available as a utility but not used in the current engine wiring.
Config: `GroundingConfig { enabled: true, max_tokens: 8192, min_confidence: 0.7, tau: 0.2 }`.

### GapResearchChain (grounding_chain.rs)

The optional entity-research pipeline.  When a fabricated entity is detected, it attempts
to find grounding statements from the spec or an LLM researcher.

Three `GroundingSource` tiers, in escalating cost order:

| Provider / Source | Strategy |
|-------------------|----------|
| `SpecAnchor` (`SpecAnchorGrounder`) | Extracts architectural nouns from the task spec; filters entities already grounded by spec text — no LLM call |
| `LlmResearcher` (`LlmResearcherGrounder`) | LLM call that classifies fabrications vs legitimate spec sub-components |
| `WebSearch` | External web search + distillation; requires `[web_search]` config and `gap_i1.enabled = true` |

Distillation and synthesis are controlled by `GapResearchConfig` (`gap_research` section):
`grounding_distill = true`, `grounding_compress_threshold = 800`,
`researcher_max_tokens = 32768`, `distill_max_tokens = 32768`, `gap_synthesis_max_tokens = 32768`.

The `GapResearchChain` is optional (`Option<Arc<GapResearchChain>>` in `TaskPipelineInput`);
it is constructed only when the config and available adapters support it.

---

## 6. Oracle Gate

`OracleGateConfig` controls post-merge validation before task delivery:

| Field | Default | Meaning |
|-------|---------|---------|
| `enabled` | false | Disabled by default; opt-in per scenario |
| `subject` | `h2ai.oracle.gate` | NATS subject for gate requests |
| `timeout_secs` | 30 | Seconds to wait for oracle response |
| `on_timeout` | `pass` | Action when timeout: pass or evict |
| `on_fail` | `evict` | Action when oracle verdict is fail |
| `min_confidence` | 0.7 | Minimum oracle confidence to accept pass verdict |

Oracle domain mapping (from `OracleDomain::family()`):
- `Code` → `OracleFamily::Syntactic`
- `Factual`, `Reasoning`, `Unknown` → `OracleFamily::Semantic`
- `Human` → `OracleFamily::Human`

---

## 7. Task Recovery

On startup, `recover_in_flight_tasks()` scans the `H2AI_TASK_CHECKPOINTS` KV bucket and
resumes all tasks that were in-flight when the node last restarted.

Strategy:
- **Own-node tasks** (`checkpoint.node_id == local_node_id()`): resume immediately.
- **Foreign-node tasks**: sleep random jitter `[0, 1500 ms]`, then attempt an optimistic
  compare-and-swap (CAS) claim.  If the CAS fails, another node won the race; skip silently.

All `N` foreign-node tasks are processed in parallel, so total wall time is bounded by
`max_jitter = 1500 ms` regardless of N.

`local_node_id()` returns `"hostname:PID"` to distinguish per-process instances.

---

## 8. NATS Event Stream

All per-task events are published as `H2AIEvent` enum variants serialised as
`{ "tag": "EventName", "content": <payload> }` JSON.

Default NATS subject: `h2ai.tasks.{task_id}`.
Special-cased subjects:
- `PendingApprovalEvent` → `h2ai.tasks.{task_id}.pending_approval`
- `ApprovalResolvedEvent` → `h2ai.tasks.{task_id}.approval_resolved`
- `OraclePendingEvent` → `h2ai.oracle.{tenant_id}.pending`

The `H2AIEvent` enum has 65+ variants covering every pipeline stage from
`TaskSubmittedEvent` through `TaskCompletedEvent` and `TaskFailedEvent`.

Key events:

| Event | When emitted |
|-------|-------------|
| `CalibrationCompletedEvent` | After POST /v1/calibrate completes |
| `TaskComplexityAssessedEvent` | Phase 1.5 — quadrant assigned |
| `ThinkingLoopCompletedEvent` | Stage 1 thinking loop (if enabled) |
| `TopologyProvisionedEvent` | Phase 2 — ensemble topology selected |
| `ProposalEvent` | Phase 3 — one per generated proposal |
| `VerificationScoredEvent` | Phase 3.5 — per-proposal verification result |
| `BranchPrunedEvent` | Phase 4 — pruned branch with `reason`, `violated_constraints`, `retry_count` |
| `ZeroSurvivalEvent` | All proposals pruned; carries `failure_mode` |
| `MergeResolvedEvent` | Phase 5a — winning output with `j_eff`, `oracle_gate_passed`, `zone3_hints` |
| `TaskFailedEvent` | Terminal failure; carries `primary_cause (TerminalCause)` and `contributing_causes` |
| `InductionCycleCompletedEvent` | Background induction run; `tasks_processed`, `archetype_priors_count` |

### TerminalCause values

`LlmAdapterUnavailable`, `VerificationExhaustion`, `NoValidProposals`,
`ComplexityOverflow`, `ContextExhaustion`, `OracleRejected`, `Timeout`, `Unknown`.

---

## 9. NATS Dispatch Mode

When `nats_dispatch_enabled = true`, explorer slots are dispatched to external `TaoAgent`
processes via NATS instead of calling in-process LLM adapters.

Key config:
- `nats_agent_ttl_secs = 30`: TTL for dispatched task slots
- `nats_agent_timeout_secs = 120`: single-agent task timeout
- `nats_agent_model = "local"`: model name in `AgentDescriptor`

---

## 10. HITL Approval Gate

`HitlConfig` controls when task outputs are held for human review before delivery.
`confidence_threshold`: `q_confidence` below this routes to HITL.
`timeout_ms`: max wait for approval; effective timeout decays per-miss by `timeout_decay = 0.5`,
floored at `timeout_floor_ms = 300_000 ms` (5 min).

When a human reviewer rejects via `POST /v1/approval`, the engine returns
`EngineError::HitlRejected { operator_id, reviewer_note }`.

---

## 11. NATS State Buckets

All persistent state uses NATS Key-Value.  Bucket names are configured in `StateConfig`.

### Global KV buckets (shared across tenants)

| Bucket | Purpose |
|--------|---------|
| `H2AI_SNAPSHOTS` | Per-task event snapshots for fast crash recovery |
| `H2AI_TASK_CHECKPOINTS` | In-flight task checkpoint records |
| `H2AI_CHECKPOINT_PAYLOADS` | Offloaded checkpoint payload blobs |
| `H2AI_ORACLE_CALIBRATION` | Rolling oracle observation window |
| `H2AI_ESTIMATOR` | `TaoMultiplierEstimator` state |
| `H2AI_SKILLS` | Skill (few-shot example) store |
| `H2AI_CALIBRATION` | Current calibration state per tenant |
| `H2AI_CALIBRATION_RECORDS` | Historical calibration records |
| `H2AI_AUDITOR_HEALTH` | Shadow auditor domain health state |
| `H2AI_PROBE_LEASE` | Distributed lease for calibration probe |
| `H2AI_SESSIONS` | Session journal state |
| `H2AI_AUDIT_SHADOW` | Shadow audit observation store |
| `H2AI_PROMPT_VARIANTS` | OPRO prompt variant store |
| `H2AI_APPROVALS` | HITL approval requests |
| `H2AI_ORACLE_HUMAN` | Human oracle pending queue |

### Per-tenant KV buckets (prefix `{name}_{tenant_bucket_safe}`)

| Prefix | Purpose |
|--------|---------|
| `H2AI_CHECKPOINT` | Reasoning checkpoints (phases 1–4) |
| `H2AI_META` | Per-tenant meta-state |
| `H2AI_MEMORY` | Distilled reasoning memory |
| `H2AI_CONFLICT` | Conflict-rate β accumulator |

### JetStream streams

| Stream | Purpose |
|--------|---------|
| `H2AI_TASKS` | Task submission and status events |
| `H2AI_TELEMETRY` | Pipeline telemetry events |
| `H2AI_RESULTS` | Completed task results |
| `H2AI_SIGNALS` | Wave-boundary control signals |

### Payload offloading

When `system_context` exceeds `payload_offload_threshold_bytes = 524 288` bytes (512 KB),
it is offloaded to `H2AI_CHECKPOINT_PAYLOADS` and the NATS message carries only a hash
reference (`ContextPayload::Ref`).

### Event snapshots

The engine writes a state snapshot to `H2AI_SNAPSHOTS` every `snapshot_interval_events = 50`
published events.  On crash recovery, the latest snapshot is loaded first and only events
after its sequence number are replayed.
