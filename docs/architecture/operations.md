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

## 2. NATS configuration

NATS is the authoritative event log and the KV backing store. The runtime expects the following streams and KV buckets to exist (created by the control plane on first startup if absent):

| Subject / bucket | Storage | Retention | Replicas | Notes |
|---|---|---|---|---|
| `H2AI_TASKS` (`h2ai.tasks.>`) | File | WorkQueue, MaxAge 30d | 3 | Authoritative task event log. |
| `H2AI_TASKS_EPHEMERAL` (`h2ai.tasks.ephemeral.>`) | File | MaxAge 1d | 3 | Ephemeral diagnostics. |
| `H2AI_TELEMETRY` (`h2ai.telemetry.>`) | File | MaxAge 7d, MaxBytes 10 GB | 3 | Adapter telemetry. |
| `H2AI_CALIBRATION` KV | — | TTL none (invalidated by `POST /calibrate`) | 3 | Latest calibration. |
| `H2AI_AGENT_MEMORY` KV | — | per-session keys | 3 | Session memory. |
| `H2AI_ESTIMATOR` KV | — | — | 1 | TaO estimator + bandit state. |
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

Calibration measures α, β₀, CG, and the cosine N_eff prior across the configured adapter pool. It must run before any task submission and must be repeated whenever the pool changes.

### Triggering

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
- **Cosine N_eff prior.** When an `EmbeddingModel` is configured, the harness embeds the calibration prompts, builds the cosine kernel, normalises K = C/N, and computes `n_eff_cosine_prior` via `EigenCalibration::from_cosine_matrix`. Without an embedding model it falls back to a closed-form estimate `1 + cg_fallback × (N − 1)`.
- **Family flags.** `single_family_warning` is set when all non-Mock adapters share a provider family. `explorer_verification_family_match` is set when the verifier and explorers come from the same non-Mock family — a flag for self-preference judge bias.
- **Calibration safety gate.** When `cfg.allow_single_family = false` (default), a single-family pool aborts calibration with `CalibrationFailed`. Override only with the documented warning understood.

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

`GET /tasks/:task_id/recover` triggers a manual snapshot+replay for a specific task — useful when investigating a stuck task.

---

## 7. Infrastructure boundaries

These are the system's hard limits. They are not bugs; they are physical or design constraints to be designed around.

- **NATS message size.** JetStream's default ceiling is 1 MB per message. `payload_offload_threshold_bytes` keeps event payloads well under this by hashing oversize `system_context` blobs. If you raise the JetStream limit, raise this in lockstep.
- **Single calibration in flight.** The harness runs one calibration at a time. Concurrent `POST /calibrate` requests during an in-flight calibration return 409. Cloud Plan deployments must route traffic away from a recalibrating pod.
- **Event-replay startup latency.** Without snapshots, recovery time is linear in the number of events for the task. Keep `snapshot_interval_events` at 50 unless you have a specific reason to raise it.
- **Starvation under sustained 503.** When `max_concurrent_tasks` is saturated, new submissions return 503. There is no built-in queue. If you need queueing, do it at the ingress.
- **Auditor as a single point of judgment.** Phase 4 is a single adapter call. If the auditor is biased, every task is biased. Mitigate by routing the auditor to a different model family from the explorers — `explorer_verification_family_match` flags this.
- **Cosine N_eff requires an embedding model.** When `cfg.embedding_model_name` is unset (and the `fastembed-embed` feature is off), the runtime falls back to `1 + cg_fallback × (N − 1)` for `n_eff_cosine_prior` and disables Phase 2.6 entirely. The system still runs, but the bivariate-CG safety net is downgraded to univariate Hamming.
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
| Slow Phase 3 with no events | Blocking pool saturated, or cloud rate-limited | `H2AI_MAX_BLOCKING_THREADS` vs. concurrent task count; adapter logs for 429s |
