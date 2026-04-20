# Deployment — Three Plans

The H2AI Control Plane is **C-first**: the distributed cluster is the architectural foundation, not a future upgrade. Local Plan is Cloud Plan running on one machine. Server Plan is Cloud Plan with a team-facing web UI layered on top. The CRDT state model, event-sourced log, and NATS JetStream topology are identical across all three.

**Edge agent model (all plans):** Edge agents are ephemeral, stateless LLM-based containers described by an `AgentDescriptor` (model name + tool capabilities). The control plane dispatches a `TaskPayload` to each agent over NATS, receives a `TaskResult`, and streams `AgentTelemetryEvent` entries in real-time. The agent has no persistent state — all context is assembled by `h2ai-memory` and injected via the payload. The agent's NATS credentials are scoped NKeys that expire when the task closes.

---

## Local Plan — Local Development

**Hardware:** Single workstation. Reference: Fedora Linux, 128 GB RAM, dedicated to model weights.

**What runs:**
- One static Rust binary (`h2ai-control-plane`) — all crates compiled into a single executable
- One `nats-server` binary — no Docker, no container runtime required for the control plane itself
- Local llama.cpp Explorers (weight files on local NVMe or tmpfs)
- Cloud Auditor (one large reasoning model via HTTP — avoids loading a second heavy model locally)
- LLM-based edge agent containers via Podman or Docker (managed externally; `StaticProvider` verifies availability via NATS heartbeats, does not spawn them)

**Startup sequence:**
```bash
# Start NATS
nats-server -c nats/dev.conf

# Start edge agent containers (externally managed in Local Plan)
podman run -d --name openclaw-1 \
  -e NATS_URL=nats://host.containers.internal:4222 \
  ghcr.io/h2ai/openclaw:latest

# Start control plane
h2ai-control-plane --plan local --nats nats://localhost:4222
```

**NATS config (`nats/dev.conf`):**
```
port: 4222
jetstream {
  store_dir: "/var/lib/nats/jetstream"
  max_memory_store: 4GB
  max_file_store: 100GB
}
```

**Agent provider:** `StaticProvider` — containers are pre-started and their NATS heartbeat subjects (`h2ai.agents.heartbeat.{agent_id}`) are monitored. `ensure_agent_capacity` checks heartbeat liveness; if no live agent is found, it returns `ProvisionError::NoCapacity`. No containers are spawned by the control plane in Local Plan.

**Memory provider:** `InMemoryCache` — session history lives in process. Fast, zero deps. Suitable for development where restarts are acceptable.

**Telemetry provider:** `DirectLogProvider` — `AgentTelemetryEvent` entries are serialized to JSON and written to stdout via `tracing-appender`. No NATS publishing in Local Plan.

**Use:** Proves the physics before deploying to a team node. Developer submits manifests via `curl` or the local Merge Authority UI at `http://localhost:8080`. Full SSE event stream and telemetry stream available immediately.

**Local Plan = Cloud Plan on one machine.** The same binary, the same event log, the same CRDT state model. Scaling to Cloud Plan requires no architectural change — only adding NATS cluster nodes, additional binary instances, and switching providers (`NatsKvStore`, `BrokerPublisherProvider`, `KubernetesProvider`).

---

## Server Plan — Team Node

**Hardware:** Dedicated server. Reference: 32–128 GB RAM, multi-core, network-accessible.

**What runs:**
- One or more `h2ai-control-plane` binary instances (orchestrator replicas)
- NATS JetStream cluster (3-node minimum for quorum)
- Web-based Merge Authority UI accessible to the engineering team
- Mix of local model Explorers and cloud Explorers depending on available GPU

**What changes from Local Plan:**
- NATS runs as a cluster, not a single server — see NATS Clustering below
- Multiple engineers submit async manifests concurrently via REST
- Merge Authority UI is the primary interface (browser, not curl)
- Dark Knowledge Compiler enforces team-wide ADR constraints — the ADR corpus is mounted as a shared volume, not a local directory
- Prometheus metrics scraped by a team-facing Grafana instance
- **Agent provider:** `StaticProvider` — edge agent containers run on the team server, managed via `docker compose` or Podman Quadlets. Heartbeats monitored over NATS cluster.
- **Memory provider:** `NatsKvStore` — session history persisted across control plane restarts in NATS KV. KV bucket: `H2AI_AGENT_MEMORY`.
- **Telemetry provider:** `BrokerPublisherProvider` wrapped in `RedactionMiddleware` — `AgentTelemetryEvent` entries published to `h2ai.telemetry.*` and visible in Grafana dashboard.

**What does not change:**
- The event model — NATS JetStream subjects are identical
- The CRDT state — semilattice joins work the same way across replicas
- The 17-event vocabulary — no schema changes across plans
- The dependency rules — state is the only NATS-touching crate for core events
- The `TaskPayload` / `TaskResult` wire format — same JSON schema (typeshare-generated Go types used by all edge agents)

**Orchestrator replicas:** Multiple `h2ai-control-plane` instances share state through NATS JetStream. Each instance consumes from the same subject namespace. CRDT semantics mean concurrent appends from different instances produce the correct merged state on replay — no locks, no leader election for the generation phase.

**Team ADR corpus:**
```
/shared/adr/
  ├── ADR-001-stateless-auth.md
  ├── ADR-007-no-direct-db-access.md
  └── ...
```

Mounted read-only at startup. The Dark Knowledge Compiler indexes this corpus at boot and reindexes on SIGHUP.

---

## Cloud Plan — Distributed Cluster

**Hardware:** Kubernetes cluster. Reference: multiple nodes across availability zones, GPU nodes for local inference, cloud provider API for cloud Explorers.

**What runs:**
- `h2ai-control-plane` Deployment (N replicas, horizontally scalable)
- NATS JetStream StatefulSet (3 or 5 nodes, spread across AZs)
- Compute adapters across GPUs, machines, and cloud regions
- Prometheus + Grafana for USL physics metrics
- Jaeger or Grafana Tempo for distributed tracing (task_id as root span)

**Kubernetes topology:**
```
Namespace: h2ai
├── Deployment/h2ai-control-plane    # orchestrator replicas
├── StatefulSet/nats                 # 3-node JetStream cluster
├── Service/nats                     # ClusterIP, port 4222
├── Service/h2ai-api                 # LoadBalancer, port 443
├── ConfigMap/nats-config            # cluster NATS config
├── ConfigMap/adr-corpus             # team ADR documents
├── ConfigMap/agent-tools            # tool binaries mounted into edge agent pods
├── PersistentVolumeClaim/nats-data  # JetStream file store
├── ServiceMonitor/h2ai              # Prometheus scrape config
└── Job/h2ai-agent-{task_id}         # ephemeral edge agent pods (KubernetesProvider)
```

**Agent provider:** `KubernetesProvider` — on `ensure_agent_capacity`, creates a Kubernetes `Job` manifest for an isolated edge agent container. The image and tool ConfigMaps are selected from the `AgentDescriptor`; scoped NATS NKey injected via `NATS_CREDS` env var. Job terminates when the task closes. No long-running agent pods.

**Memory provider:** `NatsKvStore` — same as Server Plan. KV bucket replicas match NATS StatefulSet replica count.

**Telemetry provider:** `BrokerPublisherProvider` + `RedactionMiddleware` — `AgentTelemetryEvent` on `h2ai.telemetry.*`. Scraped by Grafana dashboard. Retained in JetStream stream `H2AI_TELEMETRY` for 7 days.

**Horizontal scaling:** Orchestrator replicas are stateless with respect to task execution — all task state lives in NATS JetStream. Adding replicas increases the number of concurrent tasks the system can run. Each replica independently consumes from the NATS subject for its assigned tasks.

**GPU node affinity:** Local llama.cpp Explorers run as separate pods with GPU node affinity. The `adapters` crate routes `AdapterKind::Local` calls to these pods via internal service addresses.

**Cross-region Explorers:** Cloud adapter HTTP calls route to different LLM providers per Explorer, enforcing τ spread across model backends as well as temperature spread. This satisfies Multiplication Condition 2 (error decorrelation ρ < 0.9) even when all Explorers use cloud endpoints.

**NKey scoping (all plans):** Each task_id receives a dedicated NATS NKey JWT scoped to publish on `h2ai.tasks.ephemeral.{task_id}.*` and subscribe on `h2ai.telemetry.{task_id}`. The key is injected into the edge agent container at dispatch time and revoked when `TaskResult` is received or `TaskFailedEvent` fires.

---

## NATS Clustering

All three plans use the same NATS JetStream configuration. Local Plan runs a single server. Server Plan and Cloud Plan run a cluster.

**3-node cluster config (`nats/cluster.conf`):**
```
port: 4222
server_name: $SERVER_NAME

jetstream {
  store_dir: "/data/jetstream"
  max_memory_store: 8GB
  max_file_store: 500GB
}

cluster {
  name: h2ai-cluster
  listen: 0.0.0.0:6222
  routes: [
    nats-route://nats-0.nats.h2ai.svc:6222
    nats-route://nats-1.nats.h2ai.svc:6222
    nats-route://nats-2.nats.h2ai.svc:6222
  ]
}
```

**Stream configurations:**
```
Stream: H2AI_TASKS
  Subjects:  h2ai.tasks.>
  Storage:   File
  Replicas:  3 (Server/Cloud Plan) / 1 (Local Plan)
  Retention: WorkQueue
  MaxAge:    30d

Stream: H2AI_TASKS_EPHEMERAL
  Subjects:  h2ai.tasks.ephemeral.>
  Storage:   File
  Replicas:  3 (Server/Cloud Plan) / 1 (Local Plan)
  Retention: WorkQueue
  MaxAge:    1d   ← ephemeral agent tasks expire faster

Stream: H2AI_TELEMETRY
  Subjects:  h2ai.telemetry.>
  Storage:   File
  Replicas:  3 (Server/Cloud Plan) / 1 (Local Plan)
  Retention: Limits
  MaxAge:    7d
  MaxBytes:  10GB
```

**KV stores:**
```
Bucket: H2AI_CALIBRATION
  Replicas: 3 (Server/Cloud Plan) / 1 (Local Plan)
  TTL:      none (invalidated on adapter pool change or POST /calibrate)

Bucket: H2AI_AGENT_MEMORY
  Replicas: 3 (Server/Cloud Plan) / 1 (Local Plan)
  TTL:      none per key (entries managed by h2ai-memory NatsKvStore)
  Key pattern: h2ai.memory.{session_id}
```

**Why NATS, not Kafka or Redis:**
- Single static binary, megabytes of RAM overhead — Local Plan runs it on a workstation alongside llama.cpp without a container runtime
- Tokio-native `async-nats` client — same async runtime as the orchestrator, no blocking bridge
- JetStream provides both the event log (stream) and the calibration cache (KV) from one binary
- File-backed persistence means crash recovery = replay from offset 0, which is the same mechanism used in development and production

---

## Observability

All three plans expose the same observability surface. The difference is where it is scraped.

**Prometheus metrics (`GET /metrics`):**

| Metric | Type | Description |
|---|---|---|
| `h2ai_alpha` | Gauge | Contention coefficient α, current calibration |
| `h2ai_kappa_base` | Gauge | Baseline coherency coefficient κ_base |
| `h2ai_kappa_eff` | Gauge | Effective coherency κ_eff = κ_base / mean(CG) |
| `h2ai_n_max` | Gauge | Scalability ceiling N_max |
| `h2ai_theta_coord` | Gauge | Coordination threshold θ_coord |
| `h2ai_j_eff` | Gauge | Dark Knowledge Gap J_eff, last task |
| `h2ai_role_error_cost` | Gauge (labeled) | c_i per role (explorer, coordinator, auditor) |
| `h2ai_tasks_active` | Gauge | Tasks currently in flight |
| `h2ai_tasks_total` | Counter | Total tasks processed |
| `h2ai_proposals_pruned_total` | Counter | BranchPrunedEvents total |
| `h2ai_zero_survival_total` | Counter | ZeroSurvivalEvents total |
| `h2ai_autonomic_retries_total` | Counter | MAPE-K retry iterations total |
| `h2ai_vram_bytes` | Gauge | VRAM in use by local inference |
| `h2ai_blocking_threads_active` | Gauge | Active blocking threads (llama.cpp FFI pool) |
| `h2ai_edge_agents_active` | Gauge | Edge agent containers currently executing tasks |
| `h2ai_telemetry_events_total` | Counter | AgentTelemetryEvent entries recorded total |
| `h2ai_telemetry_redactions_total` | Counter | Secret redaction hits by RedactionMiddleware |
| `h2ai_agent_token_cost_total` | Counter | Total tokens consumed by edge agents |
| `h2ai_agent_errors_total` | Counter | AgentTelemetryEvent::SystemError occurrences total |

**Distributed tracing:**
- Every task creates a root span keyed on `task_id`
- Each phase is a child span: calibration, bootstrap, provisioning, generation, audit, merge
- Each Explorer call is a child span of generation: adapter_kind, tau, token_cost
- Exported via OpenTelemetry to Jaeger (Local Plan) or Grafana Tempo (Server/Cloud Plan)

**Alert thresholds (reference):**
- `h2ai_kappa_eff` approaching `(1 − h2ai_alpha) / h2ai_n_max²` — topology approaching retrograde
- `h2ai_blocking_threads_active` near `max_blocking_threads` — inference pool saturated, α will spike
- `h2ai_zero_survival_total` rate increasing — Auditor rejection rate high, check ADR corpus coverage

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `H2AI_NATS_URL` | `nats://localhost:4222` | NATS server or cluster URL |
| `H2AI_ADR_CORPUS_PATH` | `./adr` | Path to ADR corpus directory |
| `H2AI_MAX_BLOCKING_THREADS` | `4` | Tokio blocking pool size for llama.cpp FFI |
| `H2AI_MAX_RETRIES` | `3` | MAPE-K retry limit before TaskFailedEvent |
| `H2AI_CALIBRATION_TASKS` | `3` | Number of representative tasks for calibration harness |
| `H2AI_LISTEN_ADDR` | `0.0.0.0:8080` | HTTP bind address for axum gateway |
| `H2AI_METRICS_ADDR` | `0.0.0.0:9090` | Prometheus metrics bind address |
| `H2AI_OTEL_ENDPOINT` | _(unset)_ | OpenTelemetry collector endpoint (disables tracing if unset) |
| `H2AI_J_EFF_THRESHOLD` | `0.4` | Minimum J_eff before ContextUnderflowError |
| `H2AI_PLAN` | `local` | Deployment plan hint (`local`, `server`, `cloud`) — affects default logging verbosity |
| `H2AI_AGENT_PROVIDER` | `static` | Agent provider: `static` (pre-started containers) or `kubernetes` |
| `H2AI_MEMORY_PROVIDER` | `inmemory` | Memory provider: `inmemory` (dev) or `nats_kv` (production) |
| `H2AI_TELEMETRY_PROVIDER` | `direct_log` | Telemetry provider: `direct_log` (stdout) or `broker` (NATS publish) |
| `H2AI_NKEY_SEED` | _(unset)_ | Operator NKey seed for signing scoped agent JWTs (required for edge agents) |
| `H2AI_AGENT_IMAGE` | _(unset)_ | Default container image for ephemeral agents (used by KubernetesProvider; overridable per-descriptor via Helm values) |
| `H2AI_REDACTION_PATTERNS` | _(built-in)_ | Comma-separated additional regex patterns for RedactionMiddleware |
