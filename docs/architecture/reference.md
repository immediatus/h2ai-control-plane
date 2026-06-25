# H2AI Reference

Authoritative surface for every configurable field, wire format, and extension point.
Defaults come from `crates/h2ai-config/reference.toml`; field semantics and types come
from `crates/h2ai-config/src/lib.rs`.  See [`architecture.md`](architecture.md) for the
execution model and [`operations.md`](operations.md) for deployment procedures.

---

## 1. Configuration Loading

Configuration is applied in three layers (last layer wins):
1. Compiled-in defaults (`reference.toml` baked into the binary).
2. Operator override file (`--config h2ai.toml`).
3. Environment variables: `H2AI_<FIELD_NAME>=value` (upper-case).

Create an override file containing only the fields you need to change.  Every field not
present in the override file uses the `reference.toml` default.

---

## 2. Core Physics Fields

| Field | Default | Range | Description |
|-------|---------|-------|-------------|
| `bft_threshold` | 0.85 | [0,1] | Byzantine fault tolerance threshold for BFT consensus |
| `coordination_threshold_max` | 0.3 | [0,1] | Maximum calibration-derived coordination threshold θ |
| `min_baseline_competence` | 0.3 | [0,1] | Minimum per-agent competence p₀ for multiplication condition |
| `max_error_correlation` | 0.9 | [0,1] | Maximum pairwise error correlation ρ |
| `alpha_contention` | 0.12 | [0,1] | USL contention coefficient α |
| `beta_base_default` | 0.039 | — | USL coherency cost β₀ (AI-agents tier) |
| `cg_collapse_threshold` | 0.10 | [0,1] | CG below this → force N_max = 1 |
| `cg_agreement_threshold` | 0.85 | [0,1] | Cosine similarity threshold for CG computation |
| `context_pressure_gamma` | 0.5 | [0,1] | Context fill sensitivity γ for context-aware N_max |

---

## 3. Agent Role Defaults

| Field | Default | Description |
|-------|---------|-------------|
| `tau_coordinator` | 0.05 | Temperature for coordinator role |
| `tau_executor` | 0.40 | Temperature for executor role |
| `tau_evaluator` | 0.10 | Temperature for evaluator role |
| `tau_synthesizer` | 0.80 | Temperature for synthesizer role |
| `cost_coordinator` | 0.1 | Error cost weight for coordinator |
| `cost_executor` | 0.5 | Error cost weight for executor |
| `cost_evaluator` | 0.9 | Error cost weight for evaluator |
| `cost_synthesizer` | 0.1 | Error cost weight for synthesizer |

---

## 4. Token Budgets

All full-generation LLM calls use `model_max_tokens = 32768`.
Intentional exceptions: `leader_diagnosis_max_tokens = 128` (one Socratic question),
`calibration_probe.max_tokens = 512` (short structure probe).

| Field | Default | Description |
|-------|---------|-------------|
| `model_max_tokens` | 32768 | Token budget for all full-generation calls |
| `explorer_max_tokens` | 32768 | Per-explorer generation call |
| `evaluator_max_tokens` | 32768 | Per-evaluator verification call |
| `calibration_max_tokens` | 32768 | Calibration adapter probe call |
| `decomposition_step_max_tokens` | 32768 | Steps 1 and 2 of decomposition |
| `decomposition_json_max_tokens` | 32768 | Step 3 (JSON formatting) of decomposition |
| `max_context_tokens` | 8192 | Maximum system context tokens after compaction |
| `synthesis_critique_max_tokens` | 32768 | Phase 5a critique call |
| `synthesis_max_tokens` | 32768 | Phase 5a synthesis call |
| `hallucination_check_max_tokens` | 32768 | Grounding researcher call |
| `generation_search_max_tokens` | 32768 | Per-slot generation grounding call |
| `leader_diagnosis_max_tokens` | 128 | Leader Socratic re-prompt |
| `tao_timeout_retry_max_tokens` | 4096 | Retry tokens after TAO-loop per-turn timeout |
| `evaluator_timeout_secs` | 600 | Per-evaluator call timeout |
| `tao_per_turn_timeout_secs` | 600 | TAO loop per-turn LLM call timeout |

---

## 5. Calibration

| Field | Default | Description |
|-------|---------|-------------|
| `calibration_tau` | 0.5 | Temperature for calibration probes |
| `calibration_adapter_count` | 3 | Adapters spawned per calibration run (min 3 for USL fit) |
| `calibration_tau_spread` | [0.3, 0.7] | Temperature range for calibration instances |
| `calibration_cg_fallback` | 0.7 | CG used when < 3 adapters ran calibration |
| `calibration_max_ensemble_size` | 9 | Maximum ensemble size for Condorcet search |
| `bandit_n_max_arms` | 6 | Maximum bandit arms (N values) |
| `bandit_prior_sigma` | 2.0 | Gaussian prior σ centred on N_max_USL |
| `bandit_prior_strength` | 5.0 | Pseudo-observation strength for warm prior |
| `bandit_soft_reset_decay` | 0.3 | Posterior blend fraction on adapter version change |
| `bandit_phase0_k` | 10 | Tasks before activating ε-greedy |
| `bandit_phase1_k` | 30 | Tasks before activating pure Thompson Sampling |
| `bandit_epsilon` | 0.3 | ε for Phase 1 ε-greedy |
| `bandit_n_max_initial` | 4 | Initial N_max seed for bandit warm prior |
| `eigen_n_eff_delta` | 0.05 | Minimum N_eff increment per adapter for pruning |
| `tao_estimator_warmup` | 20 | Minimum observations before TaoMultiplierEstimator state is persisted |
| `tao_per_turn_factor` | 0.6 | Quality factor gained per TAO loop turn (prior) |
| `tao_estimator_ema_alpha` | 0.05 | EMA smoothing for TaoMultiplierEstimator drift (~14-sample half-life) |
| `baseline_accuracy_proxy` | 0.0 | Directly measured p proxy (0.0 = use CG proxy) |
| `auto_baseline_eval` | false | Auto-switch to Empirical basis after sufficient oracle tasks |
| `auto_baseline_eval_min_tasks` | 50 | Minimum Tier 1 oracle tasks before auto baseline |

---

## 6. Verification and Self-Optimizer

| Field | Default | Description |
|-------|---------|-------------|
| `verify_threshold` | 0.45 | Initial verification pass threshold |
| `optimizer_threshold_step` | 0.1 | Step size for threshold reduction |
| `optimizer_threshold_floor` | 0.3 | Minimum verify_threshold |
| `optimizer_waste_threshold` | 0.5 | Fraction of proposals that must survive for run to be non-wasteful |
| `max_autonomic_retries` | 2 | MAPE-K maximum retry waves |
| `verifier_consensus_passes` | 1 | LLM judge passes per Hard constraint (≥2 reduces false-positives) |
| `correlated_hallucination_cv_threshold` | 0.30 | CV below which C1 detection may fire |
| `correlated_hallucination_min_jaccard_floor` | 0.50 | Minimum mean Jaccard distance for C1 to fire |
| `domain_coverage_threshold` | 0.40 | Minimum constraint domain coverage fraction |

---

## 7. Oracle and Drift

| Field | Default | Description |
|-------|---------|-------------|
| `oracle_window_size` | 200 | FIFO rolling window of oracle observations |
| `oracle_ece_alert_threshold` | 0.15 | ECE above this triggers alert |
| `oracle_pass_rate_floor` | 0.30 | Pass rate below this indicates systemic failure |
| `drift_ddm_window` | 20 | DDM sliding window size in tasks |
| `drift_ddm_k` | 2.5 | DDM detection threshold in standard deviations |
| `drift_bocpd_hazard_rate` | 0.01 | BOCPD per-observation changepoint probability |
| `drift_bocpd_changepoint_threshold` | 0.90 | Posterior mass threshold to fire `CalibrationChangepoint` |
| `auto_recalibrate_on_drift` | false | Trigger `POST /calibrate` automatically on changepoint |
| `drift_staleness_ttl_secs` | 3600 | Seconds before stale-calibration warning |
| `drift_conformal_margin` | 0.05 | ORCA conformal margin during active drift |

---

## 8. Synthesis

| Field | Default | Description |
|-------|---------|-------------|
| `synthesis_enabled` | true | Enable Phase 5a two-stage critique → write synthesis |
| `synthesis_wave_enabled` | true | Enable terminal synthesis wave after retries exhaust |
| `synthesis_min_proposals` | 2 | Minimum verified proposals before synthesis |
| `synthesis_tau` | 0.2 | Temperature for critique and synthesis calls |
| `synthesis_critique_max_tokens` | 32768 | Critique call token budget |
| `synthesis_max_tokens` | 32768 | Synthesis call token budget |
| `partial_pass_overhead_factor` | 5.0 | Overhead factor in partial-pass truncation budget formula |
| `sequential_grafting_enabled` | false | Enable sequential constraint grafting (opt-in) |
| `sequential_grafting_max_rounds` | 4 | Maximum graft rounds before stopping |

---

## 9. Diversity and Talagrand

| Field | Default | Description |
|-------|---------|-------------|
| `tau_spread_max_factor` | 2.0 | Maximum τ-spread expansion factor |
| `talagrand_eta` | 0.1 | Learning rate η for KL τ-spread update |
| `talagrand_tau_min` | 0.5 | Minimum τ-spread factor (contraction floor) |

---

## 10. API Server

| Field | Default | Description |
|-------|---------|-------------|
| `api_version` | `"v1"` | Current stable API version label |
| `listen_addr` | `"0.0.0.0:8080"` | HTTP server bind address |
| `nats_url` | `"nats://host.docker.internal:4222"` | NATS server URL |
| `max_concurrent_tasks` | 8 | Max in-flight tasks; 503 when exceeded |
| `nats_dispatch_enabled` | false | Explorer slots dispatched to TaoAgent via NATS |
| `nats_agent_ttl_secs` | 30 | TTL for NATS-dispatched agent task slots |
| `nats_agent_timeout_secs` | 120 | Timeout for single NATS-dispatched agent task |
| `nats_agent_model` | `"local"` | Model name in `AgentDescriptor` for NATS dispatch |
| `payload_offload_threshold_bytes` | 524 288 | Blob size above which `system_context` is offloaded |
| `snapshot_interval_events` | 50 | Events per task before a state snapshot is written |
| `signal_wave_window_ms` | 0 | ms to wait at each `WaveCompleted` for a `WaveContinue` signal (0 = disabled) |
| `signal_min_timeout_ms` | 60 000 | Minimum timeout for `POST /signal` requests |
| `signal_max_timeout_ms` | 86 400 000 | Maximum timeout for `POST /signal` requests |

---

## 11. Shell and TAO Loop

| Field | Default | Description |
|-------|---------|-------------|
| `shell_allowlist` | `[]` | Commands permitted in normal-mode waves (empty = unrestricted) |
| `shell_hardened_allowlist` | see below | Commands permitted in hardened-mode waves |
| `shell_timeout_secs` | 5 | Hard kill timeout per shell tool invocation |
| `agent_max_tool_iterations` | 5 | Maximum TAO loop tool-call iterations per task |
| `agent_max_observation_chars` | 8192 | Maximum bytes of a tool observation (0 = no limit) |

Default `shell_hardened_allowlist`: `["ls", "cat", "git", "find", "echo", "pwd"]`.

---

## 12. Feature-Flag Subsystems

These subsystems are all disabled by default and must be explicitly enabled in config.

### Thinking loop (`[thinking_loop]`)

Pre-execution archetype brainstorm and synthesis tournament.

| Field | Default | Description |
|-------|---------|-------------|
| `enabled` | false | Enable thinking loop |
| `max_iterations` | 5 | Maximum brainstorm rounds |
| `max_archetypes` | 4 | Maximum archetypes to explore |
| `coverage_threshold` | 0.75 | Minimum archetype coverage to stop early |
| `convergence_threshold` | 0.90 | Synthesis convergence threshold |
| `tau_max` | 0.85 | Maximum temperature during brainstorm |
| `tau_min` | 0.20 | Minimum temperature during brainstorm |
| `quality_gate_max_tokens` | 64 | Tokens for quality gate call |
| `archetype_select_max_tokens` | 32768 | Archetype selection call |
| `brainstorm_max_tokens` | 32768 | Per-archetype brainstorm call |
| `synthesis_tournament_max_round_tokens` | 32768 | Tournament synthesis call |

### Reasoning memory (`[reasoning_memory]`)

Persistent cross-task reasoning memory via induction cycles.

| Field | Default | Description |
|-------|---------|-------------|
| `enabled` | false | Enable reasoning memory |
| `induction_batch_size` | 10 | Tasks per induction batch |
| `induction_max_interval_secs` | 86400 | Maximum interval between induction cycles |
| `tag_gate_threshold` | 0.2 | Minimum tag match score for retrieval |
| `max_archetype_boost` | 0.15 | Maximum competence boost from archetype match |
| `max_archetype_penalty` | 0.20 | Maximum competence penalty from tension match |

### Grounding (`[grounding]`)

Hallucination detection and entity grounding chain.

| Field | Default | Description |
|-------|---------|-------------|
| `enabled` | true | Enable grounding chain (on by default) |
| `max_tokens` | 8192 | LLM researcher call token budget |
| `min_confidence` | 0.7 | Minimum confidence to accept a grounding result |
| `tau` | 0.2 | Temperature for LLM grounding calls |

### Gap research chain (`[gap_research]`)

Distillation and synthesis settings for the `GapResearchChain` — the optional pipeline
that researches ungrounded entities after `GroundingChecker` seeds `UngroundedContent`
gaps.  Active only when a `GapResearchChain` is wired into `TaskPipelineInput`.

| Field | Default | Description |
|-------|---------|-------------|
| `grounding_distill` | true | Distil raw researcher output before storing grounding statements |
| `grounding_compress_threshold` | 800 | Character threshold above which distillation fires |
| `researcher_max_tokens` | 32768 | Token budget for the `LlmResearcherGrounder` call |
| `distill_max_tokens` | 32768 | Token budget for the distillation LLM call |
| `gap_synthesis_max_tokens` | 32768 | Token budget for the gap synthesis LLM call |

### Complexity routing (`[complexity_routing]`)

Task complexity ceiling detection and HITL decomposition routing.

| Field | Default | Description |
|-------|---------|-------------|
| `enabled` | false | Enable complexity routing |
| `adapter` | `"researcher"` | Adapter profile for probe calls |
| `probe_timeout` | 30 s | Probe call timeout |
| `decompose_threshold` | 4 | TCC at or above this → suggest decomposition |
| `hitl_threshold` | 5 | TCC at or above this → route to HITL |
| `min_retries_before_graft` | 2 | Minimum retries before grafting is attempted |

### Tiered early exit (`[tiered_exit]`)

Exit the retry loop early when partial-pass proposals meet a quality bar.

| Field | Default | Description |
|-------|---------|-------------|
| `enabled` | false | Enable tiered early exit |
| `min_n` | 1 | Minimum N to consider early exit |
| `max_n` | 5 | Maximum N for tiered exit |
| `quorum_fraction` | 0.5 | Fraction of proposals that must pass |
| `acceptance_score` | 0.85 | Minimum score for acceptance |
| `require_all_binary_checks` | true | All binary checks must pass for acceptance |

### Convergence gate (`[convergence_gate]`)

Stop retrying when verified proposals converge.

| Field | Default | Description |
|-------|---------|-------------|
| `enabled` | false | Enable convergence gate |
| `theta_converge` | 0.87 | Convergence threshold |
| `supermajority_fraction_n3` | 0.67 | Supermajority fraction for N=3 |
| `supermajority_fraction_n4plus` | 0.80 | Supermajority fraction for N≥4 |
| `score_floor` | 0.80 | Minimum score for convergence |
| `min_wave` | 1 | Minimum wave number before gate may fire |

### Token budget enforcement (`[cost_guard]`)

Per-task token budget with warn and abort thresholds.

| Field | Default | Description |
|-------|---------|-------------|
| `enabled` | false | Enable cost guard |
| `budget_tokens_per_task` | 100 000 | Per-task token budget |
| `budget_warning_fraction` | 0.80 | Fraction at which `CostThresholdWarningEvent` fires |
| `budget_abort_fraction` | 1.00 | Fraction at which task is aborted with `BudgetExhaustedEvent` |

### Knowledge gap researcher (`[gap_i1]`)

LLM researcher loop for cold constraint checks with web search.

| Field | Default | Description |
|-------|---------|-------------|
| `enabled` | false | Enable researcher loop |
| `cold_check_threshold` | 0.0 | Only fire when constraint pass rate ≤ this |
| `synthesis_min_confidence` | 0.7 | Minimum confidence to accept `DomainSynthesis` |
| `max_gap_records_per_wave` | 3 | Maximum researcher calls per MAPE-K wave |
| `researcher_timeout_secs` | 90 | Budget per researcher call (web search + distillation) |

### Constraint coherence probe (`[gap_k1]`)

Pre-flight constraint coherence check and automated spec repair.

| Field | Default | Description |
|-------|---------|-------------|
| `enabled` | false | Enable coherence probe |
| `auto_repair_enabled` | false | Enable automated spec repair via `SpecRepairAdvisor` |
| `coherence_threshold` | 0.80 | Minimum LlmJudge pass rate to consider a check coherent |
| `instability_threshold` | 0.10 | Max Jaccard similarity between rejection reasons before instability fires |
| `repair_acceptance_threshold` | 0.90 | Self-consistency required to accept a rewrite |
| `probe_runs` | 5 | LlmJudge calls per check during probe |
| `repair_candidates` | 3 | Candidate rewrites per ambiguous check |
| `probe_cache_ttl_secs` | 86400 | TTL for coherence probe cache entries (24 h) |
| `repair_max_tokens` | 32768 | Token budget for spec repair generation |

### Judge panel (`[judge_panel]`)

Multi-persona Phase 3.5 verification panel.

| Field | Default | Description |
|-------|---------|-------------|
| `quorum_fraction` | 0.67 | Fraction of judge votes required for a verdict |
| `uncertainty_weight` | 0.7 | Weight applied to uncertain votes |
| `persona_temperatures` | [0.0, 0.2, 0.4] | Judge temperatures for multi-persona voting |
| `ambiguity_threshold` | 2 | Number of conflicting verdicts before ambiguity fires |

### Oracle gate (`[oracle_gate]`)

Post-merge oracle validation before delivery.

| Field | Default | Description |
|-------|---------|-------------|
| `enabled` | false | Enable oracle gate |
| `subject` | `h2ai.oracle.gate` | NATS subject for gate requests |
| `timeout_secs` | 30 | Seconds to wait for oracle verdict |
| `on_timeout` | `pass` | Action when oracle times out: `pass` or `evict` |
| `on_fail` | `evict` | Action when oracle rejects: `evict` |
| `min_confidence` | 0.7 | Minimum confidence to accept a pass verdict |

---

## 13. Adapter Profiles

Adapter profiles are named configurations in `adapter_profiles` (array).  Each entry has:

```toml
[[adapter_profiles]]
name = "claude-sonnet"
tier = "standard"
[adapter_profiles.kind.Anthropic]
api_key_env = "ANTHROPIC_API_KEY"
model = "claude-sonnet-4-5"
```

Supported `kind` variants: `Ollama`, `OpenAi`, `Anthropic`, `A2a`.

For A2A adapters:
```toml
[[adapter_profiles]]
name = "specialist-planner"
[adapter_profiles.kind.A2a]
endpoint            = "https://my-agent.example.com"
auth_scheme         = "bearer"       # "bearer", "api_key", or "none"
auth_token_env      = "A2A_TOKEN"
timeout_minutes     = 10
poll_interval_ms    = 2000
max_poll_interval_ms = 30000
agent_card_cache_ttl_s = 3600
```

---

## 14. Optional Executors

These executors are disabled when their config section is absent.

### Web search (`[web_search]`)

Google Custom Search integration for grounding researcher calls.

```toml
[web_search]
api_key_env = "GOOGLE_API_KEY"
cx_env      = "GOOGLE_CX"
max_results = 3
```

### MCP filesystem (`[mcp_filesystem]`)

Stdio subprocess MCP server for filesystem access.

```toml
[mcp_filesystem]
command     = "npx"
args        = ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"]
timeout_secs = 30
```

### WASM executor (`[wasm_executor]`)

QuickJS interpreter sandbox for safe script execution.

```toml
[wasm_executor]
interpreter_wasm_path = "assets/quickjs.wasm"
fuel_budget           = 1000000
```

---

## 15. H2AIEvent Wire Format

All events published to NATS are serialised as:

```json
{ "tag": "EventTypeName", "content": { ...fields... } }
```

Subject routing:
- Most events: `h2ai.tasks.{task_id}`
- `PendingApprovalEvent`: `h2ai.tasks.{task_id}.pending_approval`
- `ApprovalResolvedEvent`: `h2ai.tasks.{task_id}.approval_resolved`
- `OraclePendingEvent`: `h2ai.oracle.{tenant_id}.pending`
