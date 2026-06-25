# H2AI Operations

Deploying, configuring, observing, and recovering an H2AI Control Plane instance.

---

## 1. Prerequisites

- **NATS Server** with JetStream enabled.  The default URL is `nats://host.docker.internal:4222`.
  In container environments NATS must be reachable from the container network.
- **LLM adapters** accessible at the configured endpoints (local Ollama, cloud API, or A2A agent URLs).
- **Rust toolchain** (stable) for building from source.

The devcontainer docker-compose configures NATS and Ollama as separate services that must
be started independently before launching the control plane.

---

## 2. Starting the Server

```bash
h2ai-control-plane --config h2ai.toml
```

The binary reads configuration in layers (last layer wins):
1. `crates/h2ai-config/reference.toml` — compiled-in defaults
2. `--config <path>` — operator override file
3. `H2AI_<FIELD>=value` environment variables (upper-case field name)

Example override: `H2AI_BFT_THRESHOLD=0.9`.

On startup the server:
1. Initialises NATS JetStream KV buckets and streams.
2. Runs `recover_in_flight_tasks()` — resumes all checkpointed in-flight tasks before
   accepting new requests.
3. Binds the Axum HTTP listener on `listen_addr = "0.0.0.0:8080"`.

---

## 3. HTTP API

All routes are served simultaneously.  The `api_version = "v1"` field in config
declares the current stable version; the actual route prefixes are compiled into the
binary and cannot be changed via config.

| Route | Method | Description |
|-------|--------|-------------|
| `/health` | GET | Liveness probe — returns 200 when server is up |
| `/ready` | GET | Readiness probe — returns 200 when NATS is connected |
| `/metrics` | GET | Prometheus text exposition |
| `/v1/tasks` | POST | Submit a new task |
| `/v1/tasks/{id}` | GET | Retrieve task state and events via SSE |
| `/v1/calibrate` | POST | Trigger a calibration run |
| `/v1/recovery` | POST | Manual task recovery trigger |
| `/v1/signal` | POST | Send a wave-boundary control signal |
| `/v1/approval` | POST/GET | HITL approval gate operations |
| `/v1/admin` | GET/POST | Administrative operations |

Maximum concurrent tasks: `max_concurrent_tasks = 8`.  Requests beyond this limit
receive HTTP 503.

---

## 4. Calibration

```bash
POST /v1/calibrate
```

Calibration runs `calibration_adapter_count = 3` adapter instances (minimum for USL
two-point fit) with temperatures spread across `calibration_tau_spread = [0.3, 0.7]`.
Fewer than 3 adapters fall back to `alpha_contention` + `beta_base_default` priors.

The result is a `CalibrationCompletedEvent` with:
- `coefficients`: `CoherencyCoefficients` (α, β_eff, CG_mean, N_max)
- `coordination_threshold`: derived θ (capped at `coordination_threshold_max = 0.3`)
- `ensemble`: `Option<EnsembleCalibration>` (p_mean, ρ_mean, n_optimal)
- `eigen`: `Option<EigenCalibration>` (N_eff, pruned adapters)
- `pairwise_beta`: `Option<f64>` — β₀ derived from the pairwise CG measurement loop; `None` when < 2 adapters ran
- `cg_mode`: `ConstraintProfile` | `EmbeddingCosine` — how CG was computed
- `adapter_families`: distinct non-Mock adapter families in the calibration pool (sorted)
- `explorer_verification_family_match`: true when explorer and verifier are the same non-Mock family (self-preference bias risk)
- `single_family_warning`: true when all non-Mock adapters belong to one family (BFT correlated hallucination risk)
- `n_max_lo`, `n_max_hi`: one-σ confidence interval on N_max
- `n_eff_cosine_prior`: pool-level semantic N_eff from cosine similarity matrix
- `calibration_quality`: `Domain` (empirically grounded) | `Bootstrap` (synthetic priors only)
- `calibration_source`: `Measured` (≥3 adapters USL + ≥2 CG) | `PartialFit` (one of the two) | `SyntheticPriors` (neither)
- `beta_quality`: `Option<f64>` — conflict-rate-based β from Phase B pairwise violation matrix

After calibration completes, `CalibrationDriftWarning` may be emitted if the
calibration is stale beyond `drift_staleness_ttl_secs = 3600` seconds.

---

## 5. NATS Infrastructure

### KV Buckets

All buckets are created automatically on first startup.  Bucket names are configurable
via the `[state]` section in `h2ai.toml` (defaults shown).

**Global buckets (shared across all tenants):**

| Bucket | Purpose |
|--------|---------|
| `H2AI_SNAPSHOTS` | Per-task event snapshots for crash recovery |
| `H2AI_TASK_CHECKPOINTS` | In-flight task checkpoint index |
| `H2AI_CHECKPOINT_PAYLOADS` | Offloaded checkpoint payload blobs (> 512 KB) |
| `H2AI_ORACLE_CALIBRATION` | Rolling oracle observation window |
| `H2AI_ESTIMATOR` | `TaoMultiplierEstimator` convergence state |
| `H2AI_SKILLS` | Extracted skill examples (few-shot memory) |
| `H2AI_CALIBRATION` | Live calibration state per tenant |
| `H2AI_CALIBRATION_RECORDS` | Historical calibration records |
| `H2AI_AUDITOR_HEALTH` | Shadow auditor domain health state |
| `H2AI_PROBE_LEASE` | Distributed probe lease (prevents duplicate calibrations) |
| `H2AI_SESSIONS` | Session journal state |
| `H2AI_AUDIT_SHADOW` | Shadow audit observation store |
| `H2AI_PROMPT_VARIANTS` | OPRO prompt variant candidates |
| `H2AI_APPROVALS` | HITL approval request queue |
| `H2AI_ORACLE_HUMAN` | Human oracle pending queue |

**Per-tenant KV buckets** (named `{prefix}_{tenant_bucket_safe}`):

| Prefix | Purpose |
|--------|---------|
| `H2AI_CHECKPOINT` | Reasoning checkpoint state (phases 1–4) |
| `H2AI_META` | Per-tenant meta-state (bandit posteriors, estimator) |
| `H2AI_MEMORY` | Distilled reasoning memory from induction |
| `H2AI_CONFLICT` | Conflict-rate β accumulator per constraint pair |

### JetStream Streams

| Stream | Purpose |
|--------|---------|
| `H2AI_TASKS` | Task submission and lifecycle events |
| `H2AI_TELEMETRY` | Pipeline telemetry events |
| `H2AI_RESULTS` | Completed task results |
| `H2AI_SIGNALS` | Wave-boundary control signals (`WaveContinue` etc.) |

### Payload offloading

`system_context` blobs exceeding `payload_offload_threshold_bytes = 524 288` bytes
(512 KB — half of NATS JetStream 1 MB default limit) are stored in
`H2AI_CHECKPOINT_PAYLOADS` and replaced with a hash reference (`ContextPayload::Ref`)
in the NATS message to prevent publish failures on large constraint corpora.

---

## 6. Safety Profiles

The `safety.profile` field selects a named profile that overwrites all safety-relevant
fields when the profile is not `custom`.

| Profile | Description |
|---------|-------------|
| `development` | Minimal restrictions; suitable for local testing |
| `production` | Standard guard rails; recommended for staging |
| `strict` | Maximum restrictions; highest false-positive rate |
| `custom` | All safety fields are set individually in config |

Active profile is reported at `/metrics` as `h2ai_safety_profile{profile="<name>"} 1`.

---

## 7. Agent Scheduler

The scheduler dispatches explorer slots to LLM adapters.

| Policy | Behaviour |
|--------|-----------|
| `CostAwareSpillover` (default) | Route to lowest cost tier with headroom below `scheduler_spillover_threshold = 10`. Spills to next tier when saturated. Falls back to least-loaded when all tiers saturated. |
| `LeastLoaded` | Always route to globally least-loaded adapter regardless of cost tier. |

---

## 8. Prometheus Metrics

All metrics are exposed in Prometheus text format at `GET /metrics`.

| Metric | Type | Description |
|--------|------|-------------|
| `h2ai_n_eff_prior` | gauge | Pool-level N_eff from last calibration |
| `h2ai_n_eff_actual` | gauge | Task-level N_eff from last `EpistemicYieldEvent` |
| `h2ai_epistemic_yield_ratio` | gauge | `n_eff_actual / N_requested` |
| `h2ai_mapek_interventions_total{failure_mode="mode_collapse"}` | counter | Mode-collapse MAPE-K interventions |
| `h2ai_mapek_interventions_total{failure_mode="constrained_exploration"}` | counter | Constrained-exploration interventions |
| `h2ai_phase15_task_quadrant_total{quadrant}` | counter | Phase 1.5 routing distribution |
| `h2ai_oracle_ece_gauge` | gauge | Current ECE (target < 0.05, alert > 0.15) |
| `h2ai_oracle_n_observations_total` | gauge | Rolling oracle observation count |
| `h2ai_oracle_coverage_rate` | gauge | Fraction of tasks with `OracleSpec` |
| `h2ai_oracle_pass_rate` | gauge | Rolling oracle pass rate (last 200 obs) |
| `h2ai_oracle_residual_p90` | gauge | P90 of calibration residuals |
| `h2ai_calibration_basis` | gauge | 0=Heuristic, 1=Bootstrap, 2=Conformal |
| `h2ai_oracle_tasks_total` | counter | Total resolved tasks |
| `h2ai_oracle_tasks_with_spec_total` | counter | Resolved tasks that carried `OracleSpec` |
| `h2ai_calibration_source{source}` | gauge | Active calibration source label (1 = active) |
| `h2ai_shadow_audit_total` | counter | Total Phase 4 shadow auditor observations |
| `h2ai_shadow_audit_disagreements_total` | counter | Primary–shadow disagreements |
| `h2ai_shadow_audit_promoted_domains` | gauge | Domains in two-auditor AND-vote mode |
| `h2ai_shadow_audit_disagreement_rate` | gauge | Rolling disagreement rate across all domains |
| `h2ai_safety_profile{profile}` | gauge | Active safety profile (1 = active) |
| `h2ai_safety_krum_fault_tolerance` | gauge | Krum fault tolerance setting |
| `h2ai_safety_diversity_threshold` | gauge | Diversity threshold setting |
| `h2ai_safety_shadow_auditor_enabled` | gauge | Shadow auditor enabled (1=yes, 0=no) |
| `h2ai_safety_require_bivariate_cg` | gauge | Bivariate CG check required |

`h2ai_phase15_task_quadrant_total` quadrant labels: `precision`, `coverage`, `complex`, `degenerate`.
`h2ai_calibration_source` labels: `measured`, `partial_fit`, `synthetic_priors`.

---

## 9. Crash Recovery

On any restart, `recover_in_flight_tasks()` runs before the HTTP listener binds.

- **Own-node tasks** (same `hostname:PID`): resumed immediately without racing.
- **Foreign-node tasks**: each applies a random jitter `[0, 1500 ms]` then attempts CAS
  ownership claim.  If the CAS fails, another node won the race and this node skips.

To minimise crash-recovery replay time, event snapshots are written to `H2AI_SNAPSHOTS`
every `snapshot_interval_events = 50` events.  Recovery loads the latest snapshot and
only replays events published after that snapshot's sequence number.

---

## 10. Debug Logging

Set `debug_log_path = "/tmp/h2ai-debug.ndjson"` in config to enable append-mode NDJSON
debug logging.  Each completed task appends one JSON line containing the full spec, all
proposals with scores, grounding events, and the merged output without truncation.
The directory must exist before the server starts.

---

## 11. Shell Tool Safety

When explorer agents use shell tool calls:
- `shell_allowlist`: commands permitted in normal-mode waves (empty = unrestricted; **not safe for production**).
- `shell_hardened_allowlist`: commands permitted during `ConstrainedExploration` and `ModeCollapse` retry waves.
  Default: `["ls", "cat", "git", "find", "echo", "pwd"]`.
- `shell_timeout_secs = 5`: hard kill timeout per shell invocation.

The server emits `tracing::warn!` at boot if any entry in `shell_hardened_allowlist` is
absent from `shell_allowlist` — that would grant more capability in the hardened state.
