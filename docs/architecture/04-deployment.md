# Deployment — Three Profiles

The H2AI Control Plane is **C-first**: the distributed cluster is the architectural foundation, not a future upgrade. Profile A is Profile C running on one machine. Profile B is Profile C with a team-facing web UI layered on top. The CRDT state model, event-sourced log, and NATS JetStream topology are identical across all three.

---

## Profile A — Local Development

**Hardware:** Single workstation. Reference: Fedora Linux, 128 GB RAM, dedicated to model weights.

**What runs:**
- One static Rust binary (`h2ai-control-plane`) — all seven crates compiled into a single executable
- One `nats-server` binary — no Docker, no container runtime required
- Local llama.cpp Explorers (weight files on local NVMe or tmpfs)
- Cloud Auditor (one large reasoning model via HTTP — avoids loading a second heavy model locally)

**Startup sequence:**
```bash
# Start NATS
nats-server -c nats/dev.conf

# Start control plane
h2ai-control-plane --profile a --nats nats://localhost:4222
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

**Use:** Proves the physics before deploying to a team node. Developer submits manifests via `curl` or the local Merge Authority UI at `http://localhost:8080`. Full SSE event stream available immediately.

**Profile A = Profile C on one machine.** The same binary, the same event log, the same CRDT state model. Scaling to Profile C requires no architectural change — only adding NATS cluster nodes and additional binary instances.

---

## Profile B — Team Node

**Hardware:** Dedicated server. Reference: 32–128 GB RAM, multi-core, network-accessible.

**What runs:**
- One or more `h2ai-control-plane` binary instances (orchestrator replicas)
- NATS JetStream cluster (3-node minimum for quorum)
- Web-based Merge Authority UI accessible to the engineering team
- Mix of local model Explorers and cloud Explorers depending on available GPU

**What changes from Profile A:**
- NATS runs as a cluster, not a single server — see NATS Clustering below
- Multiple engineers submit async manifests concurrently via REST
- Merge Authority UI is the primary interface (browser, not curl)
- Dark Knowledge Compiler enforces team-wide ADR constraints — the ADR corpus is mounted as a shared volume, not a local directory
- Prometheus metrics scraped by a team-facing Grafana instance

**What does not change:**
- The event model — NATS JetStream subjects are identical
- The CRDT state — semilattice joins work the same way across replicas
- The 14-event vocabulary — no new events, no schema changes
- The dependency rules — state is the only NATS-touching crate

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

## Profile C — Distributed Cluster

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
├── PersistentVolumeClaim/nats-data  # JetStream file store
└── ServiceMonitor/h2ai              # Prometheus scrape config
```

**Horizontal scaling:** Orchestrator replicas are stateless with respect to task execution — all task state lives in NATS JetStream. Adding replicas increases the number of concurrent tasks the system can run. Each replica independently consumes from the NATS subject for its assigned tasks.

**GPU node affinity:** Local llama.cpp Explorers run as separate pods with GPU node affinity. The `adapters` crate routes `AdapterKind::Local` calls to these pods via internal service addresses.

**Cross-region Explorers:** Cloud adapter HTTP calls route to different LLM providers per Explorer, enforcing τ spread across model backends as well as temperature spread. This satisfies Multiplication Condition 2 (error decorrelation ρ < 0.9) even when all Explorers use cloud endpoints.

---

## NATS Clustering

All three profiles use the same NATS JetStream configuration. Profile A runs a single server. Profiles B and C run a cluster.

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

**Stream configuration:**
```
Stream name: H2AI_TASKS
Subjects: h2ai.tasks.>
Storage: File
Replicas: 3 (Profiles B/C) / 1 (Profile A)
Retention: WorkQueue (tasks are finite; old streams can be purged after MergeResolvedEvent)
MaxAge: 30d
```

**KV store (calibration cache):**
```
Bucket: H2AI_CALIBRATION
Replicas: 3 (Profiles B/C) / 1 (Profile A)
TTL: none (invalidated on adapter pool change or operator POST /calibrate)
```

**Why NATS, not Kafka or Redis:**
- Single static binary, megabytes of RAM overhead — Profile A runs it on a workstation alongside llama.cpp without a container runtime
- Tokio-native `async-nats` client — same async runtime as the orchestrator, no blocking bridge
- JetStream provides both the event log (stream) and the calibration cache (KV) from one binary
- File-backed persistence means crash recovery = replay from offset 0, which is the same mechanism used in development and production

---

## Observability

All three profiles expose the same observability surface. The difference is where it is scraped.

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

**Distributed tracing:**
- Every task creates a root span keyed on `task_id`
- Each phase is a child span: calibration, bootstrap, provisioning, generation, audit, merge
- Each Explorer call is a child span of generation: adapter_kind, tau, token_cost
- Exported via OpenTelemetry to Jaeger (Profile A local) or Grafana Tempo (Profiles B/C)

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
| `H2AI_PROFILE` | `a` | Deployment profile hint (a/b/c) — affects default logging verbosity |
