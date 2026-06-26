# H2AI Operations

Deploying, calibrating, observing, and recovering an H2AI control plane.

---

## 1. Deployment

The runtime binary listens on `listen_addr` (default `0.0.0.0:8080`) and connects to NATS at `nats_url` (default `nats://host.docker.internal:4222`). Both values are set in `h2ai.toml`.

```bash
# Start NATS first, then the control plane
nats-server -c nats/dev.conf
h2ai-control-plane
# Override config: H2AI_CONFIG=/path/to/h2ai.toml h2ai-control-plane
```

Key runtime limits:

| Field | Default |
|---|---|
| `max_concurrent_tasks` | 8 |
| `snapshot_interval_events` | 50 |
| `payload_offload_threshold_bytes` | 524 288 |
| `tao_per_turn_timeout_secs` | 600 |
| `evaluator_timeout_secs` | 600 |
| `agent_max_tool_iterations` | 5 |
| `agent_max_observation_chars` | 8192 |

When `max_concurrent_tasks` is saturated, new task submissions return 503. There is no built-in queue; add one at the ingress if needed.

---

## 1b. Tool Executor Setup

Edge agents can be granted up to four tool executors, configured in `h2ai.toml`. Each section is optional — absent sections are silently skipped. At startup, `validate_tool_configs` performs fail-fast validation of any present section; missing env vars or missing WASM files cause an immediate panic rather than a runtime error.

### Shell Executor

Always registered. No additional configuration beyond the allowlist:

```toml
shell_allowlist         = []
shell_hardened_allowlist = ["ls", "cat", "git", "find", "echo", "pwd"]
shell_timeout_secs      = 5
```

`shell_allowlist = []` disables the allowlist (unrestricted). On timeout the agent uses PGID-scoped process group kill (SIGKILL) — no runaway child processes.

### Web Search Executor

Optional. Requires a Google Custom Search API key and CX ID.

```toml
[web_search]
api_key_env = "GOOGLE_CSE_API_KEY"
cx_env      = "GOOGLE_CSE_CX"
max_results = 3
```

`validate_tool_configs` panics at startup if either env var is missing or empty.

### Filesystem (MCP) Executor

Optional. Launches an MCP-compatible stdio server. The executor enforces a read-only policy: `read_file` and `list_directory` only. All write operations return an error at the executor layer.

```toml
[mcp_filesystem]
command      = "npx"
args         = ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"]
timeout_secs = 10
```

### WASM Code Execution Executor

Optional. Requires a compiled JavaScript interpreter WASM binary.

```toml
[wasm_executor]
interpreter_wasm_path = "/opt/h2ai/jsinterp.wasm"
fuel_budget           = 1_000_000
```

`validate_tool_configs` panics at startup if `interpreter_wasm_path` does not exist. Fuel exhaustion is a safe trap. The sandbox has no WASI imports: no network, no filesystem, no env vars. Only `language = "javascript"` is accepted.

---

## 1c. Multi-tenancy

Every task, estimator, and HITL signal is scoped to a tenant. Tenant identity is carried as a URL path segment — not a header or request body field. The `tenant_id` value in the request body is always overridden by the URL path value.

### HTTP routing

All task routes include `:tenant_id` as a path segment:

```
POST   /:tenant_id/tasks
GET    /:tenant_id/tasks/:task_id/events
GET    /:tenant_id/tasks/:task_id
POST   /:tenant_id/tasks/:task_id/merge
POST   /:tenant_id/tasks/:task_id/signal
POST   /:tenant_id/tasks/:task_id/approve   (308 redirect to /signal)
POST   /:tenant_id/tasks/:task_id/clarify
GET    /:tenant_id/tasks/:task_id/recover
```

Calibration and health endpoints (`/calibrate`, `/health`, `/ready`, `/metrics`) are global and not tenant-scoped. Single-tenant deployments use `default` as the tenant ID.

### Tenant isolation

| Layer | Mechanism |
|---|---|
| Estimators | `TenantRegistry` — `DashMap<TenantId, Arc<TenantState>>`, lazily created per tenant |
| NATS KV keys | Per-tenant prefix: `{tenant_id}/tao`, `{tenant_id}/bandit` |
| HITL signals | Subject-scoped: `h2ai.signals.{tenant_bucket_safe}.{task_id}` |
| Calibration | Shared (global) — new tenants inherit the default tenant calibration on first task |

No administrative step is required to add a tenant. A new tenant ID in the URL path creates isolated estimator state on first access.

---

## 2. NATS configuration

NATS is the authoritative event log and KV backing store. The control plane creates the following streams and KV buckets on first startup if absent:

| Name | Kind | Subject / Notes |
|---|---|---|
| `H2AI_TASKS` | Stream | `h2ai.tasks.>` — authoritative task event log |
| `H2AI_TELEMETRY` | Stream | `h2ai.telemetry.>` — telemetry and audit events |
| `H2AI_RESULTS` | Stream (WorkQueue) | `h2ai.results.>` — agent results, consumed exactly once |
| `H2AI_SIGNALS` | Stream | `h2ai.signals.>` — HITL signals, MaxAge 24h; subject per task: `h2ai.signals.{tenant_bucket_safe}.{task_id}` |
| `H2AI_CALIBRATION` | KV | Latest calibration result |
| `H2AI_SESSIONS` | KV (History 1) | Pipeline conversation history |
| `H2AI_SNAPSHOTS` | KV (History 1) | Per-task state snapshots — accelerates crash-recovery replay |
| `H2AI_ESTIMATOR` | KV | TAO and bandit state; keys: `{tenant_id}/tao`, `{tenant_id}/bandit` |
| `H2AI_SKILLS` | KV | Per-tenant skill nodes |
| `H2AI_ORACLE_CALIBRATION` | KV | Rolling oracle calibration window — max 200 entries |
| `H2AI_AUDIT_SHADOW` | KV | Shadow auditor promoted domains |
| `H2AI_TASK_CHECKPOINTS` | KV (History 1, TTL 24h) | Task phase checkpoints — used by `recover_in_flight_tasks` at startup |
| `H2AI_CHECKPOINT_PAYLOADS` | Object Store | Checkpoint payload overflow for entries > 800 KB |
| `H2AI_APPROVALS` | KV (History 1, TTL 1h) | HITL approval records |
| `H2AI_PROMPT_VARIANTS` | KV (History 5) | OPRO prompt variants |
| `H2AI_CALIBRATION_RECORDS` | KV | Per-adapter `CalibrationRecord` telemetry |
| `H2AI_AUDITOR_HEALTH` | KV | `AuditorHealth` circuit-breaker state |
| `H2AI_PROBE_LEASE` | KV (Memory storage) | Probe lease CAS tokens — ephemeral |
| `H2AI_CHECKPOINT_{tenant}` | KV (History 1, TTL 168h / 7d) | Per-task `TaskReasoningCheckpoint`, one bucket per tenant |
| `H2AI_META_{tenant}` | KV (History 1) | `TaskMetaState` projection, one bucket per tenant |
| `H2AI_MEMORY` | KV | `RetryHintPattern` cross-task priming; tenant scoping via key prefix |
| `H2AI_CONFLICT_{tenant}` | KV | `ConflictRateAccumulator` per tenant |

`payload_offload_threshold_bytes` (default 524 288) governs when `system_context` is written to a content-addressed blob so the NATS message stays under the JetStream message size limit. The Object Store (`H2AI_CHECKPOINT_PAYLOADS`) receives checkpoint entries exceeding 800 KB.

---

## 3. Calibration workflow

Calibration measures USL parameters for the configured adapter pool.

### Triggering calibration

```bash
# Trigger a fresh calibration
curl -X POST http://localhost:8080/calibrate
# {"calibration_id": "cal_...", "status": "accepted"}

# Stream calibration events
curl -sN http://localhost:8080/calibrate/cal_.../events

# Fetch the most recent completed calibration
curl http://localhost:8080/calibrate/current
```

### USL fit and calibration source

The harness runs two timing phases: N=2 adapters (Phase A) and N=M adapters (Phase B). When M ≥ 3, an analytical fit produces the USL coefficients. When M < 3, the config fallback values are used.

`CalibrationSource` is assigned as follows:

| Condition | Source |
|---|---|
| M ≥ 3 AND ≥ 2 adapters produced output | `Measured` |
| M < 3 AND < 2 adapters produced output | `SyntheticPriors` |
| Otherwise | `PartialFit` |

The `h2ai_calibration_basis` metric records the active `PredictionBasis` (0/1/2).

### Calibration safety gate

When `family_constraint = "require_diverse"` (production), a single-family adapter pool aborts calibration with `CalibrationFailed`. The development default is `family_constraint = "single_family_ok"`.

### Oracle-driven baseline update

After `n_observations ≥ 10` oracle observations, `patch_ensemble_p_from_oracle()` updates the calibration with an empirically derived baseline competence.

### When to recalibrate

| Trigger | Why |
|---|---|
| Adapter added or removed | USL coefficients are pool-specific |
| Adapter model version upgraded | `p_correct` and correlation shift |
| Sustained zero-survival rate | May indicate calibration drift |

---

## 3a. Calibration drift monitoring

`DriftMonitor` tracks `consensus_agreement_rate` — the fraction of verification events per task that passed — and detects when the LLM distribution has shifted without explicit recalibration.

### Algorithm parameters

- **DDM:** window = 20 tasks, k = 2.5 σ threshold
- **BOCPD:** hazard rate = 0.01, changepoint threshold = 0.90

A `CalibrationChangepoint` immediately subtracts `drift_conformal_margin` (0.05) from the verification threshold:

```
effective_threshold = base_threshold − active_conformal_margin()
```

The margin stays active for `drift_staleness_ttl_secs` (default 3600 s = 1 h). After TTL expiry the margin drops to 0.

### Responding to drift

**On drift warning:** monitor LLM provider status; no action required if it clears within hours.

**On changepoint:**
1. The ORCA conformal margin is already active — tasks continue with widened gates.
2. Trigger recalibration: `curl -X POST http://localhost:8080/calibrate`
3. After `CalibrationCompletedEvent`, `DriftMonitor` resets automatically.

### Automatic recalibration

`auto_recalibrate_on_drift = false` (default). Set to `true` to trigger `POST /calibrate` automatically on every changepoint. Only enable this if calibration is fast and LLM API costs are not a constraint.

---

## 4. HTTP API reference

### Task routes

| Method | Path | Description |
|---|---|---|
| POST | `/:tenant_id/tasks` | Submit task — returns 202 with `task_id` |
| GET | `/:tenant_id/tasks/:task_id/events` | SSE stream; closes on `MergeResolved` or `TaskFailed` |
| GET | `/:tenant_id/tasks/:task_id` | Task status (phase / status / proposals JSON) |
| POST | `/:tenant_id/tasks/:task_id/merge` | Force-resolve with provided output |
| GET | `/:tenant_id/tasks/:task_id/recover` | Replay NATS log, upsert TaskStore |
| POST | `/:tenant_id/tasks/:task_id/clarify` | Inject clarification answer into waiting task |
| POST | `/:tenant_id/tasks/:task_id/signal` | Inject `ResumeSignal` (Approve / WaveContinue) |
| POST | `/:tenant_id/tasks/:task_id/approve` | LEGACY — 308 redirect to `/signal` |
| GET | `/:tenant_id/tasks/:task_id/approval` | LEGACY — 410 Gone |

### Calibration routes

| Method | Path | Description |
|---|---|---|
| POST | `/calibrate` | Spawn calibration — returns 202 |
| GET | `/calibrate/:cal_id/events` | SSE stream for calibration progress |
| GET | `/calibrate/current` | Return current `CalibrationCompletedEvent` |

### Health and admin routes

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness — `{"status":"ok"}` |
| GET | `/ready` | Readiness — `"ready"` or `"missing"` (checks calibration) |
| GET | `/metrics` | Prometheus text format |
| POST | `/:tenant_id/admin/reset-experiment-state` | Resets `tau_spread`, bandit, `tao_multiplier`, `rho_ema` |

---

## 5. Observability

The `/metrics` endpoint exposes Prometheus series. Source of truth: `crates/h2ai-api/src/metrics.rs`.

| Metric | Type | When updated |
|---|---|---|
| `h2ai_n_eff_prior` | gauge | `CalibrationCompletedEvent` |
| `h2ai_n_eff_actual` | gauge | `EpistemicYieldEvent` |
| `h2ai_epistemic_yield_ratio` | gauge | `EpistemicYieldEvent` |
| `h2ai_mapek_interventions_total{failure_mode="mode_collapse"}` | counter | ModeCollapse retry |
| `h2ai_mapek_interventions_total{failure_mode="constrained_exploration"}` | counter | ConstrainedExploration retry |
| `h2ai_phase15_task_quadrant_total{quadrant}` | counter | Phase 1.5 complexity routing decision |
| `h2ai_oracle_ece_gauge` | gauge | Oracle calibration update |
| `h2ai_oracle_n_observations_total` | gauge | Rolling oracle observation count |
| `h2ai_oracle_coverage_rate` | gauge | Fraction of tasks with an `OracleSpec` |
| `h2ai_oracle_pass_rate` | gauge | Rolling oracle pass rate |
| `h2ai_oracle_residual_p90` | gauge | P90 of calibration residuals |
| `h2ai_calibration_basis` | gauge | `PredictionBasis` (0/1/2) |
| `h2ai_shadow_audit_total` | counter | Phase 4 shadow auditor observations |
| `h2ai_shadow_audit_disagreements_total` | counter | Primary/shadow disagreements |
| `h2ai_shadow_audit_promoted_domains` | gauge | Domains in two-auditor AND-vote mode |
| `h2ai_shadow_audit_disagreement_rate` | gauge | Rolling disagreement rate |
| `h2ai_safety_profile{profile}` | gauge | Active safety profile (1 = this profile) |
| `h2ai_safety_krum_fault_tolerance` | gauge | Krum fault tolerance setting |
| `h2ai_safety_diversity_threshold` | gauge | Diversity threshold setting |
| `h2ai_safety_shadow_auditor_enabled` | gauge | Shadow auditor enabled (1=yes, 0=no) |
| `h2ai_safety_require_bivariate_cg` | gauge | Bivariate CG required (1=yes, 0=no) |

### Reading the signals

- **Yield ratio < 0.5 sustained.** Pool is delivering fewer independent perspectives than calibrated. Investigate adapter family diversity.
- **`mode_collapse` rate climbing.** Pool is semantically near-degenerate — adapter rotation is being attempted but the pool is too correlated. Add a different model family.
- **`constrained_exploration` rate climbing.** Generation is diverse but the constraint corpus rejects everything. Check violated constraint patterns.
- **Both counters climbing together.** Systemic problem — recalibrate and audit family diversity simultaneously.

### Health probes

| Endpoint | Response |
|---|---|
| `GET /health` | `{"status":"ok"}` |
| `GET /ready` | `"ready"` when calibration is present; `"missing"` otherwise |

---

## 6. MAPE-K behaviour

The control loop runs after every `ZeroSurvival` event. Metrics `h2ai_mapek_interventions_total` track which branch fires.

- **`mode_collapse`** counter increments on each ModeCollapse retry.
- **`constrained_exploration`** counter increments on each ConstrainedExploration retry.

See the Prometheus metrics table (section 5) for the full metric definitions.

---

## 7. Backup and recovery

### What needs backing up

- **NATS JetStream file store** — the entire event log. This is the system's source of truth.
- **Calibration data** — stored in `H2AI_CALIBRATION` (included in JetStream backup). If lost, regenerate with `POST /calibrate`.
- **Constraint corpus** — lives in your VCS; not stored in the control plane.

### Crash recovery

In-flight tasks survive pod restarts via two complementary mechanisms:

1. **Checkpoint-based recovery** (`H2AI_TASK_CHECKPOINTS`): `recover_in_flight_tasks()` runs once at startup, before the HTTP listener binds. Own-node tasks resume immediately via `spawn_resume`. Foreign-node tasks apply random jitter (0–1500 ms) then race for ownership via optimistic CAS. The winner resumes; losers skip silently.

2. **Terminal guard**: `spawn_resume` scans the NATS event log; if `TaskFailed` or `MergeResolved` is found, the stale checkpoint is deleted and recovery returns without resuming.

`snapshot_interval_events` (default 50) governs how often a snapshot is written to `H2AI_SNAPSHOTS`. Setting it to 0 disables snapshotting — recovery then replays the entire event log from `H2AI_TASKS`.

Manual recovery for a specific task:

```bash
GET /:tenant_id/tasks/:task_id/recover
```

This replays the NATS event log and upserts the TaskStore for that task.

---

## 8. Common operational signals

| Symptom | Probable cause | First check |
|---|---|---|
| `503 CalibrationRequiredError` on every submit | No current calibration | `GET /calibrate/current` |
| `TaskFailed` with pool diversity error | Single-family pool with `family_constraint = "require_diverse"` | `CalibrationCompletedEvent` adapter family fields |
| `ZeroSurvival` with `failure_mode = ConstrainedExploration` | Constraint corpus too strict or task outside coverage | Violated constraint patterns |
| `ZeroSurvival` with `failure_mode = ModeCollapse` | Adapter rotation cannot find independent perspectives | Adapter family count and diversity |
| Yield ratio falling over time on identical workloads | Pool drift | Recalibrate; compare new calibration to historical |
| `h2ai_mapek_interventions_total{failure_mode="mode_collapse"}` rising | Pool monoculture | Deploy a second adapter family |
| `h2ai_mapek_interventions_total{failure_mode="constrained_exploration"}` rising | Corpus mismatch | Review corpus thresholds and task domain coverage |
| Agent panics at startup with env var error | `web_search.api_key_env` or `cx_env` not set | Export the env vars named in `[web_search]` before starting |
| Agent panics at startup with "does not exist" | `wasm_executor.interpreter_wasm_path` missing | Copy or build the WASM interpreter binary to the configured path |
| MCP tool returns write-not-allowed error | Agent requested a write op (`write_file` etc.) | MCP executor enforces read-only policy at the executor layer regardless of backend capability |
| WASM execution returns "fuel exhausted" | Script complexity exceeds `wasm_executor.fuel_budget` | Raise `fuel_budget`; simplify the script; check for infinite loops |
| TAO agent stops before task completion | `agent_max_tool_iterations` budget exhausted | Raise `agent_max_tool_iterations` in config |
| All proposals time out | `tao_per_turn_timeout_secs` too short for model response time | Raise `tao_per_turn_timeout_secs` (default 600 s) |
| Evaluator timeouts | `evaluator_timeout_secs` too short | Raise `evaluator_timeout_secs` (default 600 s) |
| `WaveContinue` signal has no effect | `signal_wave_window_ms = 0` (default, disabled) | Set `signal_wave_window_ms > 0` to enable wave-boundary injection |
| Observation text truncated in agent output | `agent_max_observation_chars` limit reached | Raise `agent_max_observation_chars` (default 8192) |

---

## 9. HITL signal operations

The HITL gate uses JetStream pull delivery. Each task creates a durable pull consumer on the `H2AI_SIGNALS` stream; the consumer is deleted when the task resolves.

### Sending an approval signal

```bash
curl -X POST http://localhost:8080/acme/tasks/{task_id}/signal \
  -H "Content-Type: application/json" \
  -d '{
    "payload": {
      "kind": "Approve",
      "data": {
        "approved": true,
        "reviewer_note": "LGTM"
      }
    }
  }'
```

`approved: false` rejects the task (`TaskFailed` is published). `reviewer_note` is optional.

### Sending a WaveContinue signal

`WaveContinue` is only processed when `signal_wave_window_ms > 0` (default 0, disabled):

```bash
curl -X POST http://localhost:8080/acme/tasks/{task_id}/signal \
  -H "Content-Type: application/json" \
  -d '{
    "payload": {
      "kind": "WaveContinue",
      "data": {
        "grounding": "Additional context: ...",
        "mandate_override": null
      }
    }
  }'
```

### Adaptive timeout decay

HITL default configuration:

| Field | Default |
|---|---|
| `hitl.enabled` | true |
| `hitl.confidence_threshold` | 0.50 |
| `hitl.timeout_ms` | 1 800 000 (30 min) |
| `hitl.timeout_decay` | 0.5 |
| `hitl.timeout_floor_ms` | 300 000 (5 min) |
| `signal_min_timeout_ms` | 60 000 |
| `signal_max_timeout_ms` | 86 400 000 |

Each HITL timeout increments `hitl_timeouts_fired` and reduces the effective window:

```
effective_ms = timeout_ms × timeout_decay ^ hitl_timeouts_fired
effective_ms = max(effective_ms, timeout_floor_ms)
```

Example with defaults:

| Consecutive timeouts | Effective window |
|---|---|
| 0 | 30 min |
| 1 | 15 min |
| 2 | 7.5 min |
| 3+ | 5 min (floor) |

`hitl_timeouts_fired` resets to 0 on the next successful operator response.

### Deprecated endpoints

`POST /:tenant_id/tasks/:task_id/approve` returns **308** to `/signal`. Clients should migrate.
`GET /:tenant_id/tasks/:task_id/approval` returns **410 Gone** — approval records no longer exist.

---

## 10. Epistemic output quality

The epistemic quality stage is controlled by the `[epistemic_quality]` config section.

### Defaults

| Field | Default |
|---|---|
| `enabled` | true |
| `coherence_check_enabled` | false |
| `coherence_min_severity` | `"medium"` |
| `recovery_enabled` | false |
| `recovery_max_passes` | 2 |
| `zero_valid_proposals_policy` | `"fail"` |
| `output_mode` | `"passthrough"` |

### `zero_valid_proposals_policy`

When `"fail"` (default), the task fails if no valid proposals remain after gap analysis. Set to `"deliver_unverified"` to accept annotated output despite unresolved gaps.

### `output_mode` effects

| `output_mode` | Behaviour |
|---|---|
| `"passthrough"` | Output text unchanged |
| `"clean"` | Prepends a confidence header |
| `"audit"` | Prepends confidence header, inlines per-provision annotations, appends footer |
