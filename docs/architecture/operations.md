# H2AI Operations

Deploying, calibrating, observing, and recovering an H2AI control plane. Behavioural details for the bivariate-CG control loop and the infrastructure boundaries that limit it.

---

## 1. Deployment plans

Three deployment shapes, all using the same event model and CRDT state. The runtime binary is identical; the differences are NATS topology and agent provisioning.

### Local — single workstation

One `h2ai-control-plane` binary plus one `nats-server`. No container runtime required for the control plane itself. Suitable for development and small experiments.

```bash
nats-server -c nats/dev.conf
h2ai-control-plane --plan local --nats nats://localhost:4222
```

Agents run as pre-started Podman/Docker containers; the `StaticProvider` watches NATS heartbeats. Memory: `InMemoryCache`.

### Server — team node

NATS runs as a 3-node cluster (quorum fault tolerance). Multiple engineers submit manifests concurrently. The constraint corpus is a shared mount, reindexed on `SIGHUP`. Memory: `NatsKvStore` (persisted across restarts). Agent provider: `NatsAgentProvider` (live registry via NATS heartbeats) or `StaticProvider` with `docker compose`.

### Cloud — Kubernetes

```bash
kubectl apply -f deploy/cloud/namespace.yaml
kubectl create configmap constraint-corpus --from-file=./constraints/ -n h2ai
helm install h2ai h2ai/h2ai-control-plane \
  --namespace h2ai \
  --set ingress.enabled=true \
  --set serviceMonitor.enabled=true
```

Topology: stateless `Deployment/h2ai-control-plane`, `StatefulSet/nats` (3 nodes with file-store PVC), `ConfigMap/constraint-corpus`, `ServiceMonitor/h2ai` for Prometheus, ephemeral `Job/h2ai-agent-{task_id}` per task. Agent provider: `KubernetesProvider` — creates a Job with scoped NATS NKey credentials per task; the Job terminates when the task closes. Orchestrators are stateless — all task state lives in NATS JetStream — so horizontal scaling via `replicaCount` or HPA on `h2ai_tasks_active` is safe.

---

## 1b. Enterprise Tool Executor Setup

Edge agents can be granted up to four tool executors, configured in `h2ai.toml`. Each section is optional — absent sections are silently skipped. At startup, `h2ai_agent::config_validation::validate_tool_configs` performs fail-fast validation of any present section; missing env vars or missing WASM files cause an immediate panic rather than a confusing runtime error.

### Shell Executor

Always registered. No additional setup beyond the allowlist config:

```toml
shell_allowlist = ["ls", "cat", "git", "echo", "pwd", "find", "grep"]
shell_hardened_allowlist = ["ls", "cat", "git", "echo", "pwd"]
shell_timeout_secs = 5
```

`shell_allowlist = []` disables the allowlist (unrestricted). For production deployments, always populate it with an explicit list. The agent uses PGID-scoped process group kill (SIGKILL) on timeout — no runaway child processes.

### Web Search Executor

Requires a Google Custom Search API key and a Custom Search Engine (CX) ID. Registered only in `WaveMode::Normal`.

```toml
[web_search]
api_key_env = "GOOGLE_CSE_API_KEY"
cx_env      = "GOOGLE_CSE_CX"
max_results = 5
```

```bash
export GOOGLE_CSE_API_KEY="AIza..."
export GOOGLE_CSE_CX="017576662512468239146:omuauf_lfve"
```

`validate_tool_configs` panics at startup if either env var is missing or empty. `max_results` is capped at 10 by the Google API; requesting more silently returns 10.

### Filesystem (MCP) Executor

Launches an MCP-compatible stdio server. Registered only in `WaveMode::Normal`. The executor enforces a read-only policy: `read_file` and `list_directory` only. All write operations return an error at the executor layer regardless of what the backend supports.

```toml
[mcp_filesystem]
command      = "npx"
args         = ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"]
timeout_secs = 10
```

No startup validation beyond TOML parsing — the MCP process is only spawned at first use. If the command cannot be found, the executor returns `ToolError::InitializationFailed` on the first call.

### WASM Code Execution Executor

Requires the `wasm` cargo feature and a compiled JavaScript interpreter WASM binary. Registered in both `WaveMode::Normal` and `WaveMode::Hardened`.

```toml
[wasm_executor]
interpreter_wasm_path = "/opt/h2ai/jsinterp.wasm"
fuel_budget           = 1_000_000
```

`validate_tool_configs` panics at startup if `interpreter_wasm_path` does not exist. `fuel_budget` limits computation — fuel exhaustion is a safe trap, not a crash. The sandbox has no WASI imports: no network, no filesystem, no env vars. Only `language = "javascript"` is accepted.

---

## 1c. Multi-tenancy

H2AI supports multiple isolated tenants within a single control plane deployment. Every task, estimator, and approval record is scoped to a tenant. Tenant identity is carried as a URL path segment — not a header or query parameter.

### HTTP routing

All task routes include `:tenant_id` as a path segment:

```
POST   /:tenant_id/tasks
GET    /:tenant_id/tasks/:task_id/events
GET    /:tenant_id/tasks/:task_id
POST   /:tenant_id/tasks/:task_id/merge
POST   /:tenant_id/tasks/:task_id/approve
GET    /:tenant_id/tasks/:task_id/approval
POST   /:tenant_id/tasks/:task_id/clarify
GET    /:tenant_id/tasks/:task_id/recover
```

Calibration and health endpoints (`/calibrate`, `/health`, `/ready`, `/metrics`) are global — they are not tenant-scoped.

Single-tenant deployments use `default` as the tenant ID. The `tenant_id` field in the request body is always overridden by the URL path value — it is ignored if supplied.

### Tenant isolation guarantees

| Layer | Mechanism |
|---|---|
| Estimators | `TenantRegistry` — `DashMap<TenantId, Arc<TenantState>>`, lazily created per tenant |
| NATS KV keys | Per-tenant prefix: `{tenant_id}/tao`, `{tenant_id}/bandit`, `{tenant_id}/srani`, `{tenant_id}/{task_id}` (approvals) |
| Task ownership | `TaskStore::get_for_tenant()` returns `None` for cross-tenant access |
| Approval records | `ApprovalRecord.tenant_id` stored in the record; reaper uses the embedded tenant, never the URL |
| Calibration | Shared (global) — calibration runs measure the adapter pool, not tenant workloads. New tenants inherit the default tenant's calibration on first task submission |

### Adding a tenant

No administrative step required. A new tenant ID in the URL path is enough:

```bash
curl -X POST http://localhost:8080/v1/acme/tasks \
  -H "Content-Type: application/json" \
  -d '{"description": "...", "pareto_weights": {...}, "explorers": {...}}'
```

The `TenantRegistry` creates isolated estimator state on first access. Calibration is seeded from the default tenant automatically.

---

## 2. NATS configuration

NATS is the authoritative event log and the KV backing store. The runtime expects the following streams and KV buckets to exist (created by the control plane on first startup if absent):

| Subject / bucket | Storage | Retention | Replicas | Notes |
|---|---|---|---|---|
| `H2AI_TASKS` (`h2ai.tasks.>`) | File | WorkQueue, MaxAge 30d | 3 | Authoritative task event log. |
| `H2AI_TASKS_EPHEMERAL` (`h2ai.tasks.ephemeral.>`) | File | MaxAge 1d | 3 | Ephemeral diagnostics. |
| `H2AI_TELEMETRY` (`h2ai.telemetry.>`) | File | MaxAge 7d, MaxBytes 10 GB | 3 | Adapter telemetry. |
| `H2AI_CALIBRATION` KV | — | TTL none (invalidated by `POST /calibrate`) | 3 | Latest calibration. |
| `H2AI_AGENT_MEMORY` KV | — | per-session keys | 3 | Session memory. |
| `H2AI_ESTIMATOR` KV | — | — | 1 | TAO multiplier estimator + bandit state + SRANI adaptive EMA. Keys are prefixed by `{tenant_id}/` (e.g. `default/tao`, `acme/bandit`). |
| `H2AI_SNAPSHOTS` KV | — | History 1 | 1 | Per-task snapshots. |

JetStream message size limit defaults to 1 MB. `payload_offload_threshold_bytes` (default 524 288) governs when `system_context` is written to a content-addressed blob and replaced with a hash reference (`ContextPayload::Ref`) so the NATS message stays well under the limit.

3-node cluster config (illustrative):

```
port: 4222
cluster {
  name: h2ai-cluster
  listen: 0.0.0.0:6222
  routes: [
    nats-route://nats-0.nats.h2ai.svc:6222
    nats-route://nats-1.nats.h2ai.svc:6222
    nats-route://nats-2.nats.h2ai.svc:6222
  ]
}
jetstream { store_dir: "/data/jetstream"; max_memory_store: 8GB; max_file_store: 500GB }
```

---

## 3. Calibration workflow

Calibration measures α, β₀, CG, and the cosine N_eff prior across the configured adapter pool. It runs automatically at server startup and must be repeated whenever the pool changes.

### Startup behaviour

The server runs calibration synchronously before opening its HTTP listener. The startup log emits:

```
INFO: running startup calibration…
INFO: startup calibration complete — ready to accept tasks.
```

If the LLM is unreachable, a previously persisted calibration (loaded from NATS KV) is used as a fallback.

### Manual re-triggering

Use `POST /calibrate` to force a fresh calibration without restarting the server — for example after swapping an adapter model or adding capacity:

```bash
curl -X POST http://localhost:8080/calibrate
# {"calibration_id": "cal_...", "status": "accepted"}

curl -sN http://localhost:8080/calibrate/cal_.../events
# data: {"event_type":"CalibrationCompleted", "payload": {"coefficients":{"alpha":0.12,"beta_base":0.021,"cg_samples":[...]}, "n_max":6.3, "n_eff_cosine_prior": 2.7, ...}}
```

`GET /calibrate/current` returns the most recent `CalibrationCompletedEvent`. Tasks submitted while calibration is running receive `503 CalibrationRequiredError` — route traffic away from a recalibrating pod via labels to avoid downtime in Cloud Plan.

### What the harness measures

- **USL fit (Phase A and B).** Two-phase timing: 2 adapters (Phase A) and M = `calibration_adapter_count` adapters (Phase B). Analytical fit produces α and β₀ when M ≥ 3; falls back to `cfg.calibration_default_*` otherwise.
- **Hamming CG matrix.** Pairwise constraint-profile agreement on the configured corpus. Used to populate `cg_samples` and feed `EigenCalibration::from_cg_matrix`.
- **Cosine N_eff prior.** When an `EmbeddingModel` is configured, the harness embeds the calibration prompts, builds the cosine kernel, normalises K = C/N, and computes `n_eff_cosine_prior` via `EigenCalibration::from_cosine_matrix`. Without an embedding model it falls back to a closed-form estimate `1 + calibration_cg_fallback × (N − 1)`.
- **Family flags.** `single_family_warning` is set when all non-Mock adapters share a provider family. `explorer_verification_family_match` is set when the calibration adapter pool contains more than one distinct family — indicating that Phase 3.5 will use a `CrossFamily` judge panel (the stronger debiasing path). Both fields are now populated from the actual adapter registry (previously dead code, live since 2026-05-16).
- **Calibration safety gate.** When `family_constraint = "require_diverse"` (production/strict default), a single-family pool aborts calibration with `CalibrationFailed`. Set `family_constraint = "single_family_ok"` (development default) only with the documented warning understood.

### When to recalibrate

| Trigger | Why |
|---|---|
| New adapter added or removed | α/β₀/CG measurements are pool-specific. |
| Adapter model version upgraded | p_correct and ρ shift; `bandit_soft_reset_decay` blends old posterior toward the prior. |
| Sustained zero-survival rate | May indicate calibration drift. |
| Hardware change | Re-tune `H2AI_MAX_BLOCKING_THREADS` and recalibrate. |

`CG_HALFLIFE_SECS` (7 days, hard-coded) makes β_eff drift toward the conservative β₀ ceiling automatically when CG samples age out — but this is a safety net, not a substitute for fresh calibration.

---

## 4. Observability

The `/metrics` endpoint exposes exactly five Prometheus series — the bivariate-CG control-loop signals. See `crates/h2ai-api/src/metrics.rs` for the source of truth.

| Metric | Type | When updated |
|---|---|---|
| `h2ai_n_eff_prior` | gauge | On every `CalibrationCompletedEvent`. |
| `h2ai_n_eff_actual` | gauge | On every `EpistemicYieldEvent` (post-merge async). |
| `h2ai_epistemic_yield_ratio` | gauge | Same as above. `n_eff_actual / N_requested`. |
| `h2ai_mapek_interventions_total{failure_mode="mode_collapse"}` | counter | Each `ModeCollapse` retry. |
| `h2ai_mapek_interventions_total{failure_mode="constrained_exploration"}` | counter | Each `ConstrainedExploration` retry. |

### Reading the signals

- **Yield ratio < 0.5 sustained.** The pool is delivering fewer than half the independent perspectives the operator paid for. Investigate adapter family diversity and the cosine-N_eff prior.
- **`mode_collapse` rate climbing.** Pool is semantically near-degenerate — the runtime is rotating adapters but the pool is too small or too correlated for rotation to help. Add a different model family.
- **`constrained_exploration` rate climbing.** Generation is diverse, but the constraint corpus rejects everything. Either the corpus thresholds are too strict, or the task domain is outside the corpus's coverage. Check `BranchPruned.violated_constraints` for patterns.
- **`n_eff_prior` drops over successive calibrations.** Adapter pool is converging — add diversity before tasks start failing the Phase 2.6 guard.

The OpenTelemetry pipeline (`crates/h2ai-telemetry`) provides per-phase tracing spans for adapter latency, merge time, verification scoring, and synthesis. These are higher-cardinality and intended for distributed tracing rather than alerting.

### Health probes

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness — process is alive. |
| `GET /ready` | Readiness — NATS reachable AND `H2AI_CALIBRATION` KV has a current `CalibrationCompletedEvent`. |

---

## 5. MAPE-K behaviour

The control loop runs after every `ZeroSurvival` event. Operators do not configure it directly; they configure the boundary that decides which branch fires.

- **`diversity_threshold`** is the load-bearing knob. At `0.0` (the default), Phase 2.6 is disabled and the MAPE-K classifier always returns `ConstrainedExploration` for any wave with `n_eff > 0`. Production deployments should set it to `0.5`.
- **`max_autonomic_retries`** caps the loop at 2 retries per task by default. `TaskFailed` is emitted on exhaustion with a record of every topology and τ vector tried.
- **`adapter_rotation_offset`** is task-local. Two consecutive `ModeCollapse` retries advance the offset by 2; the next wave samples a rotated subset of the pool. The offset resets on task completion.
- **The Constraint Violation Tombstone** is written into `TopologyProvisionedEvent.constraint_tombstone` *only* on `ConstrainedExploration` retries. It contains constraint IDs, severity labels, and per-constraint scores — never raw proposal text. The orchestrator reads this back into the next wave's `system_context` so the explorers see what the previous wave failed.

### Interpreting the counters

The two `h2ai_mapek_interventions_total` series tell different stories:

- `mode_collapse` rising while `constrained_exploration` is flat → pool monoculture. Adapter rotation is being attempted but not helping.
- `constrained_exploration` rising while `mode_collapse` is flat → corpus mismatch. The committee is exploring; the constraints reject everything.
- Both rising in parallel → systemic problem. Calibrate, audit family diversity, and review the corpus thresholds simultaneously.

---

## 6. Backup and recovery

### What needs backing up

- **NATS JetStream file store** — the entire event log. This is the system's source of truth.
- **Constraint corpus** — lives in your VCS; not in the control plane.
- **Calibration data** — stored in `H2AI_CALIBRATION`, included in the JetStream backup. If lost, regenerate with `POST /calibrate`.

### Recovery from crash

In-flight tasks survive pod restarts because all state is in NATS JetStream. A new pod loads the latest snapshot from `H2AI_SNAPSHOTS` and replays only events with `sequence > last_sequence`. SSE clients reconnect via `Last-Event-ID`.

`snapshot_interval_events` (default 50) governs how often a snapshot is written. 0 disables snapshotting — recovery then replays the entire event log.

Manual recovery from a point in time:

```bash
nats stream backup H2AI_TASKS /backup/h2ai-tasks-$(date +%Y%m%d)/
# … restore later …
nats stream restore /backup/h2ai-tasks-20260101/
```

`GET /:tenant_id/tasks/:task_id/recover` triggers a manual snapshot+replay for a specific task — useful when investigating a stuck task.

---

## 7. Infrastructure boundaries

These are the system's hard limits. They are not bugs; they are physical or design constraints to be designed around.

- **NATS message size.** JetStream's default ceiling is 1 MB per message. `payload_offload_threshold_bytes` keeps event payloads well under this by hashing oversize `system_context` blobs. If you raise the JetStream limit, raise this in lockstep.
- **Single calibration in flight.** The harness runs one calibration at a time. Concurrent `POST /calibrate` requests during an in-flight calibration return 409. Cloud Plan deployments must route traffic away from a recalibrating pod.
- **Event-replay startup latency.** Without snapshots, recovery time is linear in the number of events for the task. Keep `snapshot_interval_events` at 50 unless you have a specific reason to raise it.
- **Starvation under sustained 503.** When `max_concurrent_tasks` is saturated, new submissions return 503. There is no built-in queue. If you need queueing, do it at the ingress.
- **Auditor as a single point of judgment.** Phase 4 is a single adapter call. If the auditor is biased, every task is biased. Mitigate by routing the auditor to a different model family from the explorers — `explorer_verification_family_match` flags this.
- **Judge panel configuration.** Phase 3.5 uses a multi-variant `JudgePanel`. Configure via `[judge_panel]` in `reference.toml` or override: `quorum_fraction` (CrossFamily supermajority, default 0.67), `uncertainty_weight` (score penalty for uncertain constraint verdicts, default 0.7 — consider 0.5 for hard safety constraints), `persona_temperatures` (PersonaOnly fallback temperatures, default [0.0, 0.2, 0.4]), `ambiguity_threshold` (uncertain-vote count before emitting a corpus quality warning, default 2). When `ConstraintAmbiguityEvent` appears repeatedly for the same constraint ID, the constraint definition likely needs tightening.
- **Cosine N_eff requires an embedding model.** When `cfg.embedding_model_name` is unset (and the `fastembed-embed` feature is off), the runtime falls back to `1 + calibration_cg_fallback × (N − 1)` for `n_eff_cosine_prior` and disables Phase 2.6 entirely. The system still runs, but the bivariate-CG safety net is downgraded to univariate Hamming.
- **Tokio blocking pool.** Local llama.cpp adapters use `spawn_blocking`. `H2AI_MAX_BLOCKING_THREADS` should be `floor(available_ram_gb / model_size_gb)`. Pool saturation manifests as Phase 3 timeouts; the calibration's α reflects this serialisation directly.

---

## 8. Common operational signals

| Symptom | Probable cause | First check |
|---|---|---|
| `503 CalibrationRequiredError` on every submit | No current calibration | `GET /calibrate/current` |
| `TaskFailed` with `MultiplicationConditionFailed { InsufficientPoolDiversity }` | Pool monoculture | `n_eff_cosine_prior` on last `CalibrationCompletedEvent` |
| `ZeroSurvival` on every wave with `failure_mode = ConstrainedExploration` | Corpus too strict, or task outside coverage | `BranchPruned.violated_constraints` patterns |
| `ZeroSurvival` on every wave with `failure_mode = ModeCollapse` | Adapter rotation cannot find independent perspectives | `single_family_warning`, `adapter_families` count |
| Yield ratio falling over time on identical workloads | Pool drift | Recalibrate; compare new `n_eff_cosine_prior` to historical |
| Auditor approving everything despite verifier rejections | Auditor too lax for the corpus | Move auditor to a stronger model family; check `explorer_verification_family_match` |
| `ConstraintAmbiguityEvent` logged repeatedly for the same constraint ID | Constraint text is semantically underdetermined | Review and tighten that constraint definition; consider splitting into two unambiguous constraints |
| High `ConstraintAmbiguityEvent` count with `PersonaOnly` panel (single family) | No cross-family adapters available — panel falls back to persona diversity | Deploy a second adapter family to activate `CrossFamily` panel and stronger debiasing |
| Slow Phase 3 with no events | Blocking pool saturated, or cloud rate-limited | `H2AI_MAX_BLOCKING_THREADS` vs. concurrent task count; adapter logs for 429s |
| Agent process panics at startup with "is missing or empty" | `web_search.api_key_env` or `cx_env` not set in environment | Export the env vars named in the TOML `[web_search]` section before starting the agent |
| Agent process panics at startup with "does not exist" | `wasm_executor.interpreter_wasm_path` points to a missing file | Copy or build the WASM interpreter binary to the configured path |
| `TaoIterationEvent.tool_calls` is empty despite tools being configured | WaveMode is Hardened but tool requested WebSearch or FileSystem | Only Shell and CodeExecution are available in Hardened mode; check `wave_mode` on `TaskPayload` |
| TAO agent stops before completing the task | `agent_max_tool_iterations` budget exhausted | Raise `agent_max_tool_iterations` in config; investigate whether the agent is looping on a tool error |
| MCP tool always returns `not allowed` or `permitted` error | Agent is requesting a write operation (not `read_file` / `list_directory`) | The MCP executor enforces read-only policy regardless of server capability; restrict tool use in the agent prompt |
| WASM execution returns "fuel exhausted" | Script complexity exceeds `wasm_executor.fuel_budget` | Raise `fuel_budget`; simplify the script; check for infinite loops |
| All proposals fail with `TAO timeout` | `tao.per_turn_timeout_secs` too short for model response time | Raise `per_turn_timeout_secs` in `[tao]` config; 11B local models generating 1024-token outputs typically need ≥120s |
| All proposals pruned with low vocabulary scores (~0.2–0.4) | `## constraints` threshold (0.20 default) may be too strict if corpus uses compound identifiers | Lower the threshold or add `## key terms` to constraint files; compound tokens like `idem:campaign_{id}` are split on delimiters before matching |
| Calibration fails with `env var LLAMACPP_API_KEY not set` | CloudGeneric adapter reads API key from env even for local servers | Set `LLAMACPP_API_KEY=local` (any non-empty value); the server ignores the key but the adapter client requires the env var to be present |

---

## 9. Knowledge Provider

The `[knowledge]` section is optional. When absent, `PassthroughProvider` wraps the existing `ConstraintResolver` — behaviour is identical to pre-knowledge operation. Add `[knowledge]` to opt into hierarchical BM25+/PPR retrieval via `Bm25WikiProvider`.

### Enabling the Bm25Wiki provider

```toml
[knowledge]
provider = "Bm25Wiki"

[knowledge.source]
YamlDir = { path = "/path/to/constraints" }
```

`YamlDir.path` is resolved relative to the process working directory. The provider is built synchronously at startup (no background reload). Startup time scales with corpus size — a 200-constraint corpus indexes in under 100ms.

### Corpus layout

```
constraints/
  CONSTRAINT-001.yaml        # standard constraint leaf files
  CONSTRAINT-002.yaml
  wiki/
    financial-systems.yaml   # topic node (depth: topic)
    _overview.yaml           # optional global overview (depth: global)
```

When `wiki/` is absent or empty, a synthetic global node is built from constraint summaries at startup. When `wiki/_overview.yaml` is absent, the synthetic global node has `NodeSource::Synthetic`; its `synthesis` field is the first 600 characters of each constraint's `description` joined by newlines (truncated to `global_synthesis_max_chars`).

**Topic YAML schema** (`wiki/<topic>.yaml`):
```yaml
id: financial-systems
depth: topic
synthesis: "Financial systems constraints cover atomicity, idempotency, and audit-log requirements..."
domains: [financial, payments]
entry_points: [CONSTRAINT-004, CONSTRAINT-005]
invariants: "All financial operations must be idempotent under retry."
failure_modes: "Non-idempotent debit on retry causes double-charge."
```

**Global YAML schema** (`wiki/_overview.yaml`):
```yaml
id: global-overview
depth: global
synthesis: "This constraint corpus covers financial systems, ML inference, and distributed systems..."
domains: [financial, ml, distributed]
```

### Tuning ScoringConfig

All fields are optional; omitting `[knowledge.scoring]` applies the defaults shown in `reference.md §4 Knowledge Provider`. The most impactful parameters:

- **`global_synthesis_max_chars`** — trim if the global node consumes too much context budget (default 600 chars)
- **`topic_cluster_top_k`** — raise to 5 on diverse corpora; lower to 1 on narrow single-domain corpora
- **`ppr_alpha`** — higher alpha (0.25+) reduces graph diffusion and keeps results closer to direct BM25 hits; lower alpha (0.10) allows more multi-hop expansion
- **`leaf_score_multiplier`** — raise toward 1.0 if you want direct BM25 hits to dominate PPR-expanded results

### Signals to watch

| Signal | Meaning |
|---|---|
| Server log `building knowledge provider (Bm25Wiki)` | Provider built successfully at startup |
| Server log `knowledge provider: passthrough` | No `[knowledge]` configured; using passthrough |
| `checks_present` below threshold in e2e results | Content checks may target knowledge that BM25 retrieval misses — try raising `topic_cluster_top_k` or adding `wiki/` topic nodes for the relevant domain |
