# Configuration Reference

H2AI Control Plane is configured entirely via environment variables. All variables have safe defaults for Local Plan. Server/Cloud Plan deployments should review every section.

---

## Core

| Variable | Default | Description |
|---|---|---|
| `H2AI_PLAN` | `local` | Deployment plan hint (`local`, `server`, `cloud`). Affects default log verbosity and startup checks. |
| `H2AI_LISTEN_ADDR` | `0.0.0.0:8080` | HTTP bind address for the axum API gateway and Merge Authority UI. |
| `H2AI_METRICS_ADDR` | `0.0.0.0:9090` | Prometheus `/metrics` bind address. Set to empty string to disable. |

---

## NATS

| Variable | Default | Description |
|---|---|---|
| `H2AI_NATS_URL` | `nats://localhost:4222` | NATS server URL. For clusters, comma-separate multiple URLs: `nats://n1:4222,nats://n2:4222,nats://n3:4222`. The client round-robins and reconnects automatically. |
| `H2AI_NATS_STREAM_NAME` | `H2AI_TASKS` | JetStream stream name for task events. |
| `H2AI_NATS_KV_BUCKET` | `H2AI_CALIBRATION` | KV bucket name for calibration cache. |
| `H2AI_NATS_STREAM_REPLICAS` | `1` | Stream replication factor. Set to `3` for Server/Cloud Plan clusters. |
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

## Physics Config File (`H2AIConfig`)

Runtime physics parameters may also be loaded from a JSON file via `H2AIConfig::load_from_file`. All fields are optional in the file — missing fields fall back to the defaults shown below.

| Field | Default | Description |
|---|---|---|
| `j_eff_gate` | `0.4` | Context sufficiency gate. Identical to `H2AI_J_EFF_THRESHOLD`. |
| `bft_threshold` | `0.85` | BFT merge strategy switch point. |
| `coordination_threshold_max` | `0.3` | Cap on computed θ_coord. |
| `min_baseline_competence` | `0.5` | Minimum c_i floor for Multiplication Condition. |
| `max_error_correlation` | `0.9` | Maximum error correlation for Multiplication Condition. |
| `tau_coordinator` | `0.05` | Default τ for Coordinator role. |
| `tau_executor` | `0.40` | Default τ for Executor role. |
| `tau_evaluator` | `0.10` | Default τ for Evaluator role. |
| `tau_synthesizer` | `0.80` | Default τ for Synthesizer role. |
| `cost_coordinator` | `0.1` | Default role error cost c_i for Coordinator. |
| `cost_executor` | `0.5` | Default role error cost c_i for Executor. |
| `cost_evaluator` | `0.9` | Default role error cost c_i for Evaluator. |
| `cost_synthesizer` | `0.1` | Default role error cost c_i for Synthesizer. |
| `max_context_tokens` | `null` | Token budget for context compaction. `null` = no limit. |
| `explorer_max_tokens` | `1024` | Token budget per Explorer generation call. |
| `calibration_max_tokens` | `256` | Token budget per calibration probe call. |
| `optimizer_threshold_step` | `0.1` | How much `SelfOptimizer` lowers `verify_threshold` per MAPE-K step. |
| `optimizer_threshold_floor` | `0.3` | Minimum `verify_threshold` the `SelfOptimizer` will suggest. |

---

## Calibration

| Variable | Default | Description |
|---|---|---|
| `H2AI_CALIBRATION_TASKS` | `3` | Number of representative tasks the calibration harness runs. More tasks = more accurate `α` and `κ_base` measurements, but longer calibration time. |
| `H2AI_CALIBRATION_MAX_AGE_SECS` | `86400` | Seconds before cached calibration is considered stale. Stale calibration triggers a `503 CalibrationRequiredError` on new task submissions. Set to `0` to disable expiry. |

### Planned Calibration Config Fields (added in Task 1)

The following fields are planned and will be added as part of the multi-adapter calibration
fix (Gap P5). They are not yet active; the system currently falls back silently to config
defaults when fewer than 3 adapters are present.

| Field (`H2AIConfig`) | Type | Default | Description |
|---|---|---|---|
| `calibration_adapter_count` | `usize` | `3` | Number of adapter instances to run during calibration. Must be ≥ 3 for the two-phase USL fit to produce a valid β₀. When < 3, calibration will return a hard error rather than silently using defaults. |
| `calibration_tau_spread` | `[f64; 2]` | `[0.3, 0.7]` | Temperature range `[τ_min, τ_max]` for calibration adapters, linearly spaced across all M instances. Wider spread produces more diverse CG samples for calibration; narrower spread reduces adapter output variance but may underestimate CG_mean. |

---

## Dark Knowledge Compiler

| Variable | Default | Description |
|---|---|---|
| `H2AI_CONSTRAINT_CORPUS_PATH` | `./adr` | Path to the directory containing constraint documents (ADRs and typed `ConstraintDoc` Markdown files). Scanned recursively for `*.md` files. Reloaded on `SIGHUP`. Accepts `H2AI_ADR_CORPUS_PATH` as a deprecated fallback for backward compatibility. |
| `H2AI_ADR_RELOAD_INTERVAL_SECS` | `300` | Background corpus reload interval in seconds. Set to `0` to disable background reload (rely on `SIGHUP`). |

---

## Adapters

The explorer and auditor adapters are selected by environment variable at startup. The adapter factory resolves the provider name to a concrete `IComputeAdapter` implementation. Defaults to `mock` (deterministic test double).

See the [LLM Adapters](#llm-adapters) section below for the full variable reference.

**`role_error_cost` guidance** (set in `AuditorConfig` / `ExplorerConfig`):

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
| `h2ai_kappa_eff` | Gauge | — | Effective coherency κ_eff = κ_base × (1 − CG_mean) |
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

---

## LLM Adapters

The explorer and auditor adapters are configured independently. Both default to `mock` (deterministic test double). Set provider and model at startup to use real LLMs.

### Explorer (used for proposal generation and calibration)

| Variable | Default | Description |
|---|---|---|
| `H2AI_EXPLORER_PROVIDER` | `mock` | Adapter type: `anthropic`, `openai`, `ollama`, `cloud`, `mock` |
| `H2AI_EXPLORER_MODEL` | `gpt-4o` | Model name sent to the provider |
| `H2AI_EXPLORER_API_KEY_ENV` | `OPENAI_API_KEY` | Name of the env var holding the API key |
| `H2AI_EXPLORER_ENDPOINT` | _(provider default)_ | Override endpoint URL. Required for `ollama` and `cloud`. |

### Auditor (used for ADR constraint gate)

| Variable | Default | Description |
|---|---|---|
| `H2AI_AUDITOR_PROVIDER` | `mock` | Adapter type: same values as explorer |
| `H2AI_AUDITOR_MODEL` | `gpt-4o` | Auditor model name |
| `H2AI_AUDITOR_API_KEY_ENV` | `OPENAI_API_KEY` | Name of the env var holding the API key |
| `H2AI_AUDITOR_ENDPOINT` | _(provider default)_ | Override endpoint URL |

### Provider defaults

| Provider | Default endpoint | Auth header |
|---|---|---|
| `anthropic` | `https://api.anthropic.com` | `x-api-key: <value of H2AI_EXPLORER_API_KEY_ENV>` |
| `openai` | `https://api.openai.com/v1` | `Authorization: Bearer <value of H2AI_EXPLORER_API_KEY_ENV>` |
| `ollama` | `http://localhost:11434` | none |
| `cloud` | _(must set H2AI_EXPLORER_ENDPOINT)_ | `Authorization: Bearer <value of H2AI_EXPLORER_API_KEY_ENV>` |
| `mock` | n/a | n/a |

### Example: Anthropic explorer + Claude Haiku auditor

```bash
H2AI_EXPLORER_PROVIDER=anthropic
H2AI_EXPLORER_MODEL=claude-3-5-sonnet-20241022
H2AI_EXPLORER_API_KEY_ENV=ANTHROPIC_API_KEY

H2AI_AUDITOR_PROVIDER=anthropic
H2AI_AUDITOR_MODEL=claude-3-5-haiku-20241022
H2AI_AUDITOR_API_KEY_ENV=ANTHROPIC_API_KEY

ANTHROPIC_API_KEY=sk-ant-...
```

### Example: local Ollama

```bash
H2AI_EXPLORER_PROVIDER=ollama
H2AI_EXPLORER_MODEL=llama3.2
H2AI_EXPLORER_ENDPOINT=http://localhost:11434

H2AI_AUDITOR_PROVIDER=ollama
H2AI_AUDITOR_MODEL=llama3.2
H2AI_AUDITOR_ENDPOINT=http://localhost:11434
```
