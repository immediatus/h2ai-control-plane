# Configuration Reference

H2AI Control Plane is configured entirely via environment variables. All variables have safe defaults for Profile A. Profile B/C deployments should review every section.

---

## Core

| Variable | Default | Description |
|---|---|---|
| `H2AI_PROFILE` | `a` | Deployment profile hint (`a`, `b`, `c`). Affects default log verbosity and startup checks. |
| `H2AI_LISTEN_ADDR` | `0.0.0.0:8080` | HTTP bind address for the axum API gateway and Merge Authority UI. |
| `H2AI_METRICS_ADDR` | `0.0.0.0:9090` | Prometheus `/metrics` bind address. Set to empty string to disable. |

---

## NATS

| Variable | Default | Description |
|---|---|---|
| `H2AI_NATS_URL` | `nats://localhost:4222` | NATS server URL. For clusters, comma-separate multiple URLs: `nats://n1:4222,nats://n2:4222,nats://n3:4222`. The client round-robins and reconnects automatically. |
| `H2AI_NATS_STREAM_NAME` | `H2AI_TASKS` | JetStream stream name for task events. |
| `H2AI_NATS_KV_BUCKET` | `H2AI_CALIBRATION` | KV bucket name for calibration cache. |
| `H2AI_NATS_STREAM_REPLICAS` | `1` | Stream replication factor. Set to `3` for Profile B/C clusters. |
| `H2AI_NATS_MAX_FILE_STORE` | `100GB` | Maximum JetStream file store size. |

---

## Runtime Physics

| Variable | Default | Description |
|---|---|---|
| `H2AI_MAX_RETRIES` | `3` | Maximum MAPE-K retry cycles per task before `TaskFailedEvent`. |
| `H2AI_MAX_BLOCKING_THREADS` | `4` | Tokio blocking thread pool size for llama.cpp FFI. Calibrate to `floor(available_ram_gb / model_size_gb)`. Too high spikes α; too low caps throughput. |
| `H2AI_EXPLORER_TIMEOUT_SECS` | `120` | Wall time limit per Explorer call. Exceeded → `ProposalFailedEvent` with `failure_reason: Timeout`. |
| `H2AI_J_EFF_THRESHOLD` | `0.4` | Minimum Jaccard overlap for task acceptance. Below this → `ContextUnderflowError`. Lower values accept underspecified tasks; higher values enforce richer context. |
| `H2AI_BFT_THRESHOLD` | `0.85` | `max(c_i)` above which `MergeStrategy` switches from `CrdtSemilattice` to `BftConsensus`. |

---

## Calibration

| Variable | Default | Description |
|---|---|---|
| `H2AI_CALIBRATION_TASKS` | `3` | Number of representative tasks the calibration harness runs. More tasks = more accurate `α` and `κ_base` measurements, but longer calibration time. |
| `H2AI_CALIBRATION_MAX_AGE_SECS` | `86400` | Seconds before cached calibration is considered stale. Stale calibration triggers a `503 CalibrationRequiredError` on new task submissions. Set to `0` to disable expiry. |

---

## Dark Knowledge Compiler

| Variable | Default | Description |
|---|---|---|
| `H2AI_ADR_CORPUS_PATH` | `./adr` | Path to the directory containing ADR Markdown files. Scanned recursively for `*.md` files. Reloaded on `SIGHUP`. |
| `H2AI_ADR_RELOAD_INTERVAL_SECS` | `300` | Background corpus reload interval in seconds. Set to `0` to disable background reload (rely on `SIGHUP`). |

---

## Adapters

Adapter configuration controls which compute backends are available to the Explorer pool and the Auditor.

| Variable | Default | Description |
|---|---|---|
| `H2AI_ADAPTER_CONFIG_PATH` | `./adapters.toml` | Path to the adapter configuration file. See [Adapter Configuration](#adapter-configuration-file) below. |

### Adapter Configuration File

`adapters.toml` defines the adapter pool. The calibration harness measures all listed adapters.

```toml
# adapters.toml

[[explorer]]
id = "local-llama-8b"
kind = "local"
model_path = "/models/llama-3-8b-instruct.Q4_K_M.gguf"
context_size = 8192
gpu_layers = 0          # 0 = CPU only; set to -1 for full GPU offload
role_error_cost = 0.1

[[explorer]]
id = "cloud-gpt4o-mini"
kind = "cloud"
api_base = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"   # read from environment, never hardcoded
model = "gpt-4o-mini"
role_error_cost = 0.1

[[explorer]]
id = "cloud-claude-haiku"
kind = "cloud"
api_base = "https://api.anthropic.com"
api_key_env = "ANTHROPIC_API_KEY"
model = "claude-haiku-4-5-20251001"
role_error_cost = 0.1

[auditor]
id = "auditor-cloud"
kind = "cloud"
api_base = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"
model = "gpt-4o"
role_error_cost = 0.9   # Auditor: near-catastrophic false positive cost
# tau is always 0.0 for the Auditor — set automatically, not configurable
```

**`role_error_cost` guidance:**

| Role | Typical c_i | Rationale |
|---|---|---|
| Explorer (early draft) | 0.1 | Auditor will filter errors before human sees them |
| Swarm Coordinator | 0.5–0.7 | Error multiplied across sub-group |
| Auditor | 0.9 | False positive = hallucination reaches human unfiltered |

---

## Observability

| Variable | Default | Description |
|---|---|---|
| `H2AI_OTEL_ENDPOINT` | _(unset)_ | OpenTelemetry collector OTLP gRPC endpoint. If unset, tracing is disabled. Example: `http://jaeger:4317`. |
| `H2AI_OTEL_SERVICE_NAME` | `h2ai-control-plane` | Service name reported in traces. |
| `RUST_LOG` | `h2ai=info` | Log filter directive. `h2ai=debug` for verbose output. Supports per-module overrides: `h2ai_autonomic=debug,h2ai_adapters=warn`. |
| `RUST_BACKTRACE` | _(unset)_ | Set to `1` for backtraces on panic. `full` for full backtraces. |

---

## Metrics Reference

All metrics are exposed at `GET /metrics` in Prometheus text format.

| Metric | Type | Labels | Description |
|---|---|---|---|
| `h2ai_alpha` | Gauge | — | Contention coefficient α from last calibration |
| `h2ai_kappa_base` | Gauge | — | Baseline coherency coefficient κ_base |
| `h2ai_kappa_eff` | Gauge | — | Effective coherency κ_eff = κ_base / mean(CG) |
| `h2ai_n_max` | Gauge | — | Scalability ceiling N_max = sqrt((1−α) / κ_eff) |
| `h2ai_theta_coord` | Gauge | — | Coordination threshold θ_coord |
| `h2ai_cg_mean` | Gauge | — | Mean Common Ground across Explorer pairs |
| `h2ai_j_eff` | Gauge | `task_id` | Dark Knowledge Gap J_eff for the last completed task |
| `h2ai_role_error_cost` | Gauge | `role`, `adapter_id` | c_i per role per adapter |
| `h2ai_tasks_active` | Gauge | — | Tasks currently in flight |
| `h2ai_tasks_total` | Counter | `status` (resolved, failed) | Total tasks processed |
| `h2ai_proposals_total` | Counter | `outcome` (validated, pruned, failed) | Total proposals processed |
| `h2ai_zero_survival_total` | Counter | — | Total `ZeroSurvivalEvent` occurrences |
| `h2ai_autonomic_retries_total` | Counter | `action` (WidenTauSpread, IncreaseN, ...) | MAPE-K retry iterations |
| `h2ai_multiplication_condition_failures_total` | Counter | `condition` (BaselineCompetence, ErrorDecorrelation, CommonGroundFloor) | Phase 2.5 gate failures per condition |
| `h2ai_merge_strategy_total` | Counter | `strategy` (CrdtSemilattice, BftConsensus) | How often each merge strategy is selected |
| `h2ai_calibration_age_seconds` | Gauge | — | Seconds since last successful calibration |
| `h2ai_adapter_vram_bytes` | Gauge | `adapter_id` | VRAM in use per local llama.cpp adapter |
| `h2ai_adapter_latency_seconds` | Histogram | `adapter_id`, `adapter_kind` | Per-adapter inference latency |
| `h2ai_blocking_threads_active` | Gauge | — | Active Tokio blocking threads (llama.cpp FFI pool) |
| `h2ai_nats_publish_latency_seconds` | Histogram | — | NATS event publish latency |

---

## Helm Values Reference

When deploying with Helm, all environment variables map to `values.yaml` fields. The full values file with documentation is at `deploy/helm/h2ai-control-plane/values.yaml`.

Key overrides for enterprise deployments:

```yaml
# Number of control plane replicas
replicaCount: 4

config:
  maxBlockingThreads: 16     # tune to your GPU node memory
  maxRetries: 5              # more patience for high-stakes tasks
  jEffThreshold: "0.5"       # stricter context requirement
  logLevel: "h2ai=warn"      # quieter in production

# Bring your own NATS cluster (e.g., NATS Managed)
nats:
  enabled: false
  natsUrl: nats://my-nats-cluster.internal:4222

# Enable Prometheus Operator scraping
serviceMonitor:
  enabled: true
  labels:
    release: kube-prometheus-stack

# Ingress with TLS
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
  tls:
    - secretName: h2ai-tls
      hosts:
        - h2ai.corp.example.com

# Resource tuning for production
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
