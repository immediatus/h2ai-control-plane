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

## Physics Configuration (`H2AIConfig`)

Runtime physics parameters are loaded via a three-layer stack (later layers win):

1. **`reference.toml`** — embedded in the binary; provides all defaults with inline comments.
   This is the canonical source of truth. See `crates/h2ai-config/reference.toml`.
2. **Override TOML file** — operator-supplied file containing only the fields to change.
   Specify a path in `H2AI_CONFIG`, or drop a `h2ai.toml` in the working directory.
   Unspecified fields fall through to `reference.toml`.
3. **`H2AI__<FIELD>` env vars** — highest priority; override any field from either file.
   Double-underscore separator, upper-case field name. Example: `H2AI__J_EFF_GATE=0.5`.

Config source discovery at startup (first match wins):

| Priority | Condition | Action |
|---|---|---|
| 1 | `H2AI_CONFIG` env var is set | Load that file as the override TOML (error if missing) |
| 2 | `./h2ai.toml` exists | Load it as the override TOML |
| 3 | Neither | Use `reference.toml` defaults only |

Startup logs which source is active.

A complete JSON file may also be loaded with `H2AIConfig::load_from_file` (used in tests and tooling), but unlike the TOML loader it requires **all** fields to be present — there is no fallback to `reference.toml` in that path.

### USL + Multiplication Condition

| Field | Default | Description |
|---|---|---|
| `j_eff_gate` | `0.4` | Context sufficiency gate. Below this → `ContextUnderflowError`. |
| `bft_threshold` | `0.85` | `max(c_i)` above which `MergeStrategy` switches to BFT consensus. |
| `krum_fault_tolerance` | `0` | Byzantine fault tolerance bound for Krum/Multi-Krum. `0` = Krum disabled (ConsensusMedian used). `n` = tolerate up to `n` Byzantine explorers; requires ≥ 2n+3 explorers. |
| `krum_threshold` | `0.95` | `max(c_i)` above which Krum is preferred over ConsensusMedian. Only active when `krum_fault_tolerance > 0`. |
| `coordination_threshold_max` | `0.3` | Cap on computed θ_coord. |
| `min_baseline_competence` | `0.3` | Minimum competence threshold for Multiplication Condition (p > 0.5 proxy). |
| `max_error_correlation` | `0.9` | Maximum error correlation for Multiplication Condition. |
| `alpha_contention` | `0.12` | α contention constant: serial fraction that cannot be parallelized. Typical LLM ensemble value. Also accepts alias `calibration_alpha_single_adapter`. |
| `beta_base_default` | `0.039` | β₀ base coherency cost per agent pair. Fallback when < 3 adapters available. Also accepts alias `kappa_eff_factor`. At CG=0.4: β_eff = 0.039/0.4 ≈ 0.097 → N_max ≈ 3. |
| `diversity_threshold` | `0.85` | Jaccard similarity above which all proposals are considered uniformly hallucinated → `ZeroSurvivalEvent`. |
| `context_pressure_gamma` | `0.5` | Sensitivity of β to context window fill. `0` disables. `0.5` (default): β doubles when context is 100% full. Range [0,1]. |

### Agent Role Defaults

| Field | Default | Description |
|---|---|---|
| `tau_coordinator` | `0.05` | Default τ for Coordinator role. |
| `tau_executor` | `0.40` | Default τ for Executor role. |
| `tau_evaluator` | `0.10` | Default τ for Evaluator role. |
| `tau_synthesizer` | `0.80` | Default τ for Synthesizer role. |
| `cost_coordinator` | `0.1` | Role error cost c_i for Coordinator. |
| `cost_executor` | `0.5` | Role error cost c_i for Executor. |
| `cost_evaluator` | `0.9` | Role error cost c_i for Evaluator. |
| `cost_synthesizer` | `0.1` | Role error cost c_i for Synthesizer. |

### Token Budgets

| Field | Default | Description |
|---|---|---|
| `explorer_max_tokens` | `1024` | Token budget per Explorer generation call. |
| `calibration_max_tokens` | `256` | Token budget per calibration probe call. |
| `max_context_tokens` | `null` | Context compaction token budget. `null` = no limit. |
| `max_concurrent_tasks` | `8` | Maximum concurrent tasks. Requests beyond this limit receive `503 Service Unavailable`. |
| `task_deadline_secs` | `null` | Hard deadline per task in seconds. `null` = no deadline. |

### MAPE-K Autonomic Loop

| Field | Default | Description |
|---|---|---|
| `max_autonomic_retries` | `2` | Maximum MAPE-K retry cycles before `TaskFailedEvent`. |
| `optimizer_threshold_step` | `0.1` | How much `SelfOptimizer` lowers `verify_threshold` per MAPE-K step. |
| `optimizer_threshold_floor` | `0.3` | Minimum `verify_threshold` `SelfOptimizer` will suggest. |
| `tao_per_turn_factor` | `0.6` | Error decay factor per TAO turn: `c_i_eff = c_i × 0.6^(t−1)`. |
| `tau_spread_max_factor` | `2.0` | Maximum τ-spread expansion factor when Talagrand detects over-confidence (U-curve). `2.0` means τ spread can at most double. |

### CG Measurement and Embedding

| Field | Default | Description |
|---|---|---|
| `cg_collapse_threshold` | `0.10` | When `CG_embed` drops below this, forces `N_max=1`. Below 10% agreement, pairwise reconciliation is undefined. |
| `cg_agreement_threshold` | `0.85` | Cosine similarity threshold above which two adapter outputs count as "in agreement". Used to compute `CG_embed = fraction(cosine(embed_i, embed_j) > θ)`. |
| `calibration_cg_fallback` | `0.7` | `CG_mean` fallback when embedding model is not configured. Conservative; overestimates coordination quality. |
| `embedding_model_name` | `AllMiniLmL6V2` | Embedding model for CG cosine agreement rate. Requires `fastembed-embed` Cargo feature. Options: `AllMiniLmL6V2` (22 MB, ~8 ms/sentence), `BgeSmallEnV1_5` (109 MB, ~5 ms, better MTEB STS). Models cached to `~/.cache/fastembed/`. |
| `baseline_accuracy_proxy` | `0.0` | Override CG-derived accuracy proxy with a directly measured per-adapter baseline accuracy. Set by running `scripts/baseline_eval.py`. `0.0` = use `0.5 + CG_mean / 2` proxy. |

### Thompson Sampling Bandit (Adaptive N Selection)

| Field | Default | Description |
|---|---|---|
| `bandit_phase0_k` | `10` | Tasks before activating the bandit. During Phase 0, N = N_max_USL unconditionally. |
| `bandit_phase1_k` | `30` | Tasks before switching from ε-greedy (Phase 1) to pure Thompson Sampling (Phase 2). |
| `bandit_epsilon` | `0.3` | Phase 1 exploration probability: probability of picking a random N instead of the TS arm. |
| `bandit_soft_reset_decay` | `0.3` | Decay toward prior when adapter version hash changes. `0.3` = 30% pull toward prior, 70% of learned posterior preserved. |

### Calibration Adapter Pool

| Field | Default | Description |
|---|---|---|
| `calibration_adapter_count` | `3` | Adapter instances to run during calibration. Must be ≥ 3 for USL two-point fit to produce real measurements. < 3 falls back to `alpha_contention` and `beta_base_default`. |
| `calibration_tau` | `0.5` | τ value for calibration probe calls. |
| `calibration_tau_spread` | `[0.3, 0.7]` | Temperature range `[τ_min, τ_max]` for calibration adapter instances. Instances linearly spaced: `τ_i = τ_min + (τ_max − τ_min) × i/(M−1)`. |

### Empirical Baseline Automation

| Field | Default | Description |
|---|---|---|
| `auto_baseline_eval` | `false` | When true, switches to `Empirical` prediction basis automatically after enough Tier 1 oracle tasks complete. |
| `auto_baseline_eval_min_tasks` | `50` | Minimum Tier 1 oracle task count before auto baseline evaluation triggers. |

### Adapter Profiles

| Field | Default | Description |
|---|---|---|
| `adapter_profiles` | `[]` | Named adapter configurations. Reference by name via `AdapterFactory::build_from_profiles`. Names must be unique; first match wins. |

### Self-Optimizer

| Field | Default | Description |
|---|---|---|
| `optimizer_threshold_step` | `0.1` | How much `SelfOptimizer` lowers `verify_threshold` per MAPE-K step. |
| `optimizer_threshold_floor` | `0.3` | Minimum `verify_threshold` `SelfOptimizer` will suggest. |
| `optimizer_waste_threshold` | `0.5` | Fraction of proposals that must survive verification for the run to be considered efficient. Below this, `SelfOptimizer` suggestions are applied on the success path to reduce future waste. |

### Scheduler

| Field | Default | Description |
|---|---|---|
| `scheduler_policy` | `CostAwareSpillover` | Dispatch policy. `CostAwareSpillover`: route to lowest cost tier with headroom, spill to next tier when saturated. `LeastLoaded`: always route to globally least-loaded agent. |
| `scheduler_spillover_threshold` | `10` | Queue depth per cost tier at which `CostAwareSpillover` routes to the next tier. Ignored by `LeastLoaded`. |

### Multi-Family Enforcement

| Field | Default | Description |
|---|---|---|
| `allow_single_family` | `false` | When `false` (default), calibration aborts if all non-Mock adapters belong to the same provider family. Weiszfeld BFT correlated-hallucination protection requires diverse families. Set `true` to allow single-family pools with a warning. |

### NATS Payload Offloading

| Field | Default | Description |
|---|---|---|
| `payload_offload_threshold_bytes` | `524288` | Byte length above which `system_context` is offloaded to a content-addressed store and replaced with a `ContextPayload::Ref { hash, byte_len }` in the NATS message. Default 512 KB — half of the NATS JetStream 1 MB default message-size limit. Prevents publish failures on large constraint corpora. Set higher to keep more contexts inline; set lower to be more conservative with NATS headroom. |

### Event Snapshot Store

`SessionJournal` writes a periodic snapshot of each task's in-memory state to the `H2AI_SNAPSHOTS` NATS KV bucket. On crash-recovery, `replay` loads the latest snapshot and only replays events with a JetStream sequence number greater than `last_sequence`, reducing CPU cost from O(all events) to O(recent events).

| Field | Default | Description |
|---|---|---|
| `snapshot_interval_events` | `50` | Events published per task before a state snapshot is written to NATS KV. Set to `0` to disable snapshotting entirely. Lower values protect against longer replays but write more frequently to KV. |

Snapshots are stored at key `snapshots/{task_id}/latest` in `H2AI_SNAPSHOTS` (history=1, latest-only). The write is fire-and-forget via `tokio::spawn`; a failed write logs a warning but never blocks the task hot-path. If the snapshot fails or is absent, `replay` falls back to full replay from sequence 0.

---

## Calibration

| Variable | Default | Description |
|---|---|---|
| `H2AI_CALIBRATION_TASKS` | `3` | Number of representative tasks the calibration harness runs. More tasks = more accurate `α` and `κ_base` measurements, but longer calibration time. |
| `H2AI_CALIBRATION_MAX_AGE_SECS` | `86400` | Seconds before cached calibration is considered stale. Stale calibration triggers a `503 CalibrationRequiredError` on new task submissions. Set to `0` to disable expiry. |

See the **Calibration Adapter Pool** section in the Physics Config File table above for all calibration-related `H2AIConfig` fields.

---

## Dark Knowledge Compiler

| Variable | Default | Description |
|---|---|---|
| `H2AI_CONSTRAINT_CORPUS_PATH` | `./constraints` | Path to the directory containing `ConstraintDoc` Markdown files. Scanned recursively for `*.md` files. Reloaded on `SIGHUP`. |
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
| `h2ai_kappa_eff` | Gauge | — | Effective coherency β_eff = β₀ × (1 − CG_mean) |
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
