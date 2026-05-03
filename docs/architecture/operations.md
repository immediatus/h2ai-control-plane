# H2AI Operations Guide

Getting started, deployment reference, monitoring, and troubleshooting for H2AI Control Plane.

---

## Quick Start

### Prerequisites

| Requirement | Local Plan | Server Plan | Cloud Plan |
|---|---|---|---|
| Docker + Compose | required | required | — |
| Kubernetes 1.28+ | — | — | required |
| Helm 3.x | — | — | required |
| Cloud LLM API key | optional | required | required |

Both explorer and auditor default to `mock`. To get real LLM output you need at least one cloud API key (OpenAI, Anthropic, or an Ollama endpoint). Use a capable reasoning model for the Auditor role.

### 1. Clone and configure

```bash
git clone https://github.com/h2ai/control-plane.git
cd h2ai-control-plane
cp .env.example .env
```

Edit `.env`:

```bash
# Explorer adapter (proposal generation + calibration)
H2AI_EXPLORER_PROVIDER=anthropic
H2AI_EXPLORER_MODEL=claude-3-5-sonnet-20241022
H2AI_EXPLORER_API_KEY_ENV=ANTHROPIC_API_KEY

# Auditor adapter (constraint gate — use a capable reasoning model)
H2AI_AUDITOR_PROVIDER=anthropic
H2AI_AUDITOR_MODEL=claude-3-5-haiku-20241022
H2AI_AUDITOR_API_KEY_ENV=ANTHROPIC_API_KEY

# The actual key
ANTHROPIC_API_KEY=sk-ant-...
```

### 2. Start the stack

```bash
cd deploy/local
docker compose up -d
docker compose ps   # both h2ai and nats should show "running"
```

NATS monitoring: `http://localhost:8222`. API: `http://localhost:8080`.

### 3. Seed your constraint corpus

```bash
mkdir -p constraints
cat > constraints/CONSTRAINT-001-stateless-auth.md << 'EOF'
# CONSTRAINT-001: Stateless Authentication

## Severity
Hard threshold=0.8

## Predicate
VocabularyPresence AllOf
- jwt
- stateless
- no session state
- token expiry

## Remediation
The proposal must state that authentication is JWT-based and stateless. No session
tokens may be written to any storage medium. Token expiry must be specified.
EOF

docker compose restart h2ai   # reload corpus
```

See [Constraint Corpus](reference.md#constraint-corpus) for the full format reference.

### 4. Run calibration

Before submitting tasks, the system must measure α, β₀, and CG across the adapter pool:

```bash
curl -s -X POST http://localhost:8080/calibrate | jq .
```

```json
{"calibration_id": "cal_01HXYZ...", "status": "accepted"}
```

Stream calibration progress:

```bash
curl -sN http://localhost:8080/calibrate/cal_01HXYZ.../events
```

```
data: {"event_type":"CalibrationCompleted","payload":{"coefficients":{"alpha":0.12,"beta_base":0.021,...},"n_max":6.3}}
```

Calibration is cached in NATS KV. Repeat only when the adapter pool changes.

### 5. Submit a task

```bash
curl -s -X POST http://localhost:8080/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Design a stateless JWT refresh token rotation strategy for our API gateway",
    "pareto_weights": {"diversity": 0.5, "containment": 0.3, "throughput": 0.2},
    "explorers": {"count": 3, "tau_min": 0.3, "tau_max": 0.8}
  }' | jq .
```

```json
{"task_id": "task_01HYYZ...", "status": "accepted", "events_url": "/tasks/task_01HYYZ.../events"}
```

### 6. Watch the swarm

```bash
curl -sN http://localhost:8080/tasks/task_01HYYZ.../events
```

You will see `TopologyProvisioned` → `Proposal` (per explorer) → `GenerationPhaseCompleted` → `Validation`/`BranchPruned` → `SemilatticeCompiled`.

### 7. Resolve in the Merge Authority

Open `http://localhost:8080`. The Merge Authority UI shows valid proposals (with diffs), tombstoned proposals (with constraint violations), and the live physics panel (β_eff, N_max, θ_coord). Select, synthesize, or reject proposals and click **Resolve**.

---

## Deployment Plans

H2AI is **C-first**: the distributed cluster is the architectural foundation. Local Plan is Cloud Plan on one machine. The CRDT state model, event-sourced log, and NATS JetStream topology are identical across all three plans.

### Local Plan — Single Workstation

**What runs:** One `h2ai-control-plane` binary + one `nats-server`. No container runtime required for the control plane itself. Edge agent containers managed externally via Podman or Docker.

**Startup:**
```bash
nats-server -c nats/dev.conf
podman run -d --name agent-1 -e NATS_URL=nats://host.containers.internal:4222 ghcr.io/h2ai/openclaw:latest
h2ai-control-plane --plan local --nats nats://localhost:4222
```

**Agent provider:** `StaticProvider` — containers are pre-started; NATS heartbeats monitored. Control plane does not spawn containers.

**Memory provider:** `InMemoryCache` — fast, zero deps, suitable for development.

### Server Plan — Team Node

**What changes from Local Plan:**
- NATS runs as a 3-node cluster (quorum fault tolerance)
- Multiple engineers submit manifests concurrently
- Constraint corpus mounted as a shared volume
- Prometheus + Grafana for team-facing metrics
- **Agent provider:** `NatsAgentProvider` — live registry via NATS heartbeats; or `StaticProvider` with `docker compose`
- **Memory provider:** `NatsKvStore` — persisted across control plane restarts

**What does not change:** Event model, CRDT state, 24-event vocabulary, `TaskPayload`/`TaskResult` wire format.

**Team constraint corpus:**
```bash
/shared/constraints/
  ├── CONSTRAINT-001-stateless-auth.md
  ├── CONSTRAINT-007-no-direct-db-access.md
  └── ...
```

Mounted read-only at startup; reindexed on SIGHUP.

### Cloud Plan — Kubernetes

```bash
kubectl apply -f deploy/cloud/namespace.yaml
kubectl create configmap constraint-corpus --from-file=./constraints/ -n h2ai

helm repo add h2ai https://h2ai.github.io/control-plane
helm install h2ai h2ai/h2ai-control-plane \
  --namespace h2ai \
  --set ingress.enabled=true \
  --set ingress.hosts[0].host=h2ai.corp.example.com \
  --set serviceMonitor.enabled=true
```

**Kubernetes topology:**
```
Namespace: h2ai
├── Deployment/h2ai-control-plane    # orchestrator replicas (stateless)
├── StatefulSet/nats                 # 3-node JetStream cluster
├── Service/nats                     # ClusterIP, port 4222
├── Service/h2ai-api                 # LoadBalancer, port 443
├── ConfigMap/constraint-corpus      # team constraint documents
├── PersistentVolumeClaim/nats-data  # JetStream file store
├── ServiceMonitor/h2ai              # Prometheus scrape config
└── Job/h2ai-agent-{task_id}         # ephemeral edge agent pods
```

**Agent provider:** `KubernetesProvider` — creates a Kubernetes `Job` per task with scoped NATS NKey credentials. Jobs terminate when the task closes.

**Horizontal scaling:** Orchestrators are stateless — all task state lives in NATS JetStream. Scale by increasing `replicaCount` or via HPA on `h2ai_tasks_active`.

---

## NATS Configuration

**3-node cluster config:**
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

**Streams:**
```
H2AI_TASKS:          h2ai.tasks.>,   File, Replicas=3, Retention=WorkQueue, MaxAge=30d
H2AI_TASKS_EPHEMERAL: h2ai.tasks.ephemeral.>, File, Replicas=3, MaxAge=1d
H2AI_TELEMETRY:      h2ai.telemetry.>, File, Replicas=3, Retention=Limits, MaxAge=7d, MaxBytes=10GB
```

**KV stores:**
```
H2AI_CALIBRATION:   Replicas=3, TTL=none (invalidated by POST /calibrate)
H2AI_AGENT_MEMORY:  Replicas=3, Key=h2ai.memory.{session_id}
H2AI_ESTIMATOR:     Replicas=1, stores tao_estimator and bandit_state
H2AI_SNAPSHOTS:     Replicas=1, Key=snapshots/{task_id}/latest, History=1
```

**Why NATS:** single static binary, megabytes overhead, Tokio-native `async-nats` client, JetStream provides both event log and KV from one binary. Crash recovery = replay from offset (same in dev and prod).

---

## Monitoring

### Key metrics

**`h2ai_beta_eff`** — most important gauge. When it rises, the system approaches the scalability ceiling. If it reaches `(1 − α) / N_max²`, adding Explorers hurts, not helps.

Alert: `h2ai_beta_eff > 0.035`

**`h2ai_alpha`** — contention coefficient. Sustained α > 0.20 means serial fraction is too high — investigate NATS publish latency and blocking thread pool saturation.

Alert: `h2ai_alpha > 0.20`

**`h2ai_zero_survival_total` (rate)** — a non-zero rate means the Auditor is rejecting everything. Check `h2ai_autonomic_retries_total` alongside.

Alert: `rate(h2ai_zero_survival_total[10m]) > 0.1`

**`h2ai_calibration_age_seconds`** — stale calibration means N_max is wrong.

Alert: `h2ai_calibration_age_seconds > 86400`

**`h2ai_blocking_threads_active`** — near-saturation → Explorer timeouts increase.

Alert: `h2ai_blocking_threads_active / H2AI_MAX_BLOCKING_THREADS > 0.9`

**`h2ai_cg_mean`** — trend awareness. Sustained CG < 0.4 means β_eff is rising and N_max is falling. Recalibrate after significant corpus or adapter changes.

### Alert rules

```yaml
groups:
  - name: h2ai
    rules:
      - alert: H2AIBetaEffHigh
        expr: h2ai_beta_eff > 0.035
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "β_eff high — approaching scalability ceiling"
          description: "β_eff={{ $value }}. Consider reducing Explorer count or widening model diversity."

      - alert: H2AIAlphaHigh
        expr: h2ai_alpha > 0.20
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "α high — serial contention detected"
          description: "Check NATS publish latency and blocking thread pool saturation."

      - alert: H2AIZeroSurvivalRateHigh
        expr: rate(h2ai_zero_survival_total[10m]) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High zero-survival rate — Auditor rejecting everything"

      - alert: H2AICalibrationStale
        expr: h2ai_calibration_age_seconds > 86400
        labels:
          severity: warning
        annotations:
          summary: "Calibration data is stale (>24h)"

      - alert: H2AIBlockingPoolSaturated
        expr: h2ai_blocking_threads_active / on() h2ai_blocking_threads_max > 0.9
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Blocking thread pool near saturation"

      - alert: H2AINATSPublishSlow
        expr: histogram_quantile(0.99, rate(h2ai_nats_publish_latency_seconds_bucket[5m])) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "NATS publish p99 latency > 100ms"
```

### Grafana dashboard

Import `deploy/server/grafana/h2ai-dashboard.json`. Includes USL physics panel (β_eff, α, N_max, θ_coord), task throughput, proposal outcomes, MAPE-K activity, adapter latency histograms, blocking thread pool utilization.

---

## Scaling

### Scaling the control plane

```bash
kubectl scale deployment h2ai-control-plane --replicas=6 -n h2ai
# or via Helm
helm upgrade h2ai h2ai/h2ai-control-plane --set autoscaling.maxReplicas=20 --reuse-values
```

### Scaling NATS (maintenance window required)

```bash
kubectl scale statefulset nats --replicas=5 -n h2ai
```

Do not scale NATS during active task execution — quorum reconfiguration is a maintenance operation. 3 nodes = tolerates 1 failure; 5 nodes = tolerates 2 failures.

### High-load tuning

```bash
kubectl set env deployment/h2ai-control-plane H2AI_MAX_BLOCKING_THREADS=16 -n h2ai
kubectl set env deployment/h2ai-control-plane H2AI_CALIBRATION_TASKS=1 -n h2ai
kubectl set env deployment/h2ai-control-plane H2AI_EXPLORER_TIMEOUT_SECS=180 -n h2ai
```

---

## Upgrading

### Rolling upgrade (Cloud Plan)

```bash
helm upgrade h2ai h2ai/h2ai-control-plane --set image.tag=0.2.0 --reuse-values
kubectl rollout status deployment/h2ai-control-plane -n h2ai
```

In-flight tasks survive pod restarts — all state is in NATS JetStream. A new pod replays the event log from the last snapshot and resumes from the current phase. SSE clients reconnect via `Last-Event-ID`.

### Post-upgrade calibration

After any upgrade that changes adapter versions or model weights:

```bash
curl -X POST https://h2ai.corp.example.com/calibrate
```

Stale calibration produces inaccurate N_max — either underprovisioning or throughput retrograde.

---

## Backup and Recovery

### What needs backup

- **NATS JetStream file store** — the entire event log for all tasks
- **Constraint corpus** — lives in your git repository (not in the control plane)
- **Calibration data** — stored in NATS KV; backed up with the file store; regenerate with `POST /calibrate` if lost

### NATS backup

```bash
# Local Plan
tar -czf nats-backup-$(date +%Y%m%d).tar.gz /var/lib/nats/jetstream/

# Cloud Plan
nats stream backup H2AI_TASKS /backup/h2ai-tasks-$(date +%Y%m%d)/
```

### Recovery

```bash
# Restore (Local Plan)
tar -xzf nats-backup-20260419.tar.gz -C /var/lib/nats/
docker compose restart nats
docker compose restart h2ai
```

In-flight tasks resume from their last committed event. Proposals that were generating at crash time will be retried by the MAPE-K loop.

### Recalibration triggers

| Event | Why |
|---|---|
| New adapter added to pool | New α/β₀ measurements needed for full pool |
| Adapter model version upgraded | p_correct and ρ may have changed |
| Sustained `h2ai_zero_survival_total` rate increase | May indicate calibration drift |
| `h2ai_calibration_age_seconds` alert fires | Data is stale by policy |
| Hardware change (RAM, GPU added) | Re-tune `H2AI_MAX_BLOCKING_THREADS` and recalibrate |

Recalibration takes as long as `H2AI_CALIBRATION_TASKS` full inference cycles. Tasks submitted during calibration receive `503 CalibrationRequiredError`. To avoid downtime on Cloud Plan, route traffic away from the recalibrating instance using pod labels.

---

## Troubleshooting

### Tasks fail with `ZeroSurvivalEvent` on every attempt

**Diagnosis:**
```bash
curl -sN http://localhost:8080/tasks/{task_id}/events | grep BranchPruned
```

| Cause | Fix |
|---|---|
| Constraint corpus empty or wrong path | Check `H2AI_CONSTRAINT_CORPUS_PATH`; run `ls $H2AI_CONSTRAINT_CORPUS_PATH/*.md` |
| Hard constraints unsatisfiable for task domain | Review `BranchPrunedEvent.violated_constraints`; relax thresholds or add remediation hints |
| Task description uses domain language not in corpus | Add relevant constraint doc, or add explicit `context` to task manifest |
| Corpus not reloaded after changes | Send `SIGHUP` or restart the container |

### `CalibrationRequiredError` on task submission

```bash
curl -X POST http://localhost:8080/calibrate
curl -sN http://localhost:8080/calibrate/{calibration_id}/events  # watch progress
```

### Task stuck — no events after `TopologyProvisionedEvent`

**Causes:**
- **Blocking pool saturated** — `h2ai_blocking_threads_active` equals `H2AI_MAX_BLOCKING_THREADS`. Reduce concurrent tasks or increase `H2AI_MAX_BLOCKING_THREADS`.
- **Cloud API rate limit** — check adapter logs for 429 responses.
- **Explorer timeout too short** — increase `H2AI_EXPLORER_TIMEOUT_SECS`.

```bash
curl http://localhost:9090/metrics | grep h2ai_blocking_threads
nats stream view H2AI_TASKS --subject "h2ai.tasks.{task_id}"
```

### `MultiplicationConditionFailedEvent` — ErrorDecorrelation

**Meaning:** Two or more Explorers make the same errors (ρ too high). Structurally redundant.

**Fix:**
1. Widen τ spread: `{"explorers": {"tau_min": 0.1, "tau_max": 0.95}}`
2. Add a second model backend to the adapter pool
3. Reduce Explorer count if only one model is available

### `MultiplicationConditionFailedEvent` — BaselineCompetence

**Meaning:** An adapter performs below chance on calibration. Adding it makes the collective worse.

**Fix:** Remove the underperforming adapter. Check model loading, context size, and prompt format. Re-run calibration.

### `MultiplicationConditionFailedEvent` — CommonGroundFloor

**Meaning:** CG_mean below θ_coord — adapters too epistemically distant; coordination cost exceeds diversity benefit.

**Fix:** Reduce τ spread, use more similar model families, or reduce N.

### High α (contention)

| Symptom | Likely cause |
|---|---|
| α spikes correlate with NATS publish latency | NATS disk I/O bottleneck — check `nats-data` volume throughput |
| α spikes correlate with blocking thread saturation | Blocking pool too large for available CPU — reduce `H2AI_MAX_BLOCKING_THREADS` |
| α spikes during merge resolution | Structural — merge step is inherently serial |

### High β_eff (coordination cost)

| Symptom | Likely cause |
|---|---|
| `cg_mean` dropped | Adapter pool changed — recalibrate |
| `beta_base` rose | Token exchange overhead increased — check network latency between Explorer pods |
| N_max dropped | Combined effect of rising α and β — system will select smaller swarms automatically |

### Diagnosing a `TaskFailedEvent`

```bash
nats stream view H2AI_TASKS --subject "h2ai.tasks.{task_id}" | \
  python3 -c "
import sys,json
events = [json.loads(l) for l in sys.stdin if l.startswith('{')]
failed = [e for e in events if e.get('event_type') == 'TaskFailed']
print(json.dumps(failed[0], indent=2))
"
```

| Field | What to check |
|---|---|
| `multiplication_condition_failure` | Which of the 3 conditions blocked (see above) |
| `branch_pruned_events[*].reason` | Patterns in Auditor rejections → ADR coverage gap |
| `branch_pruned_events[*].constraint_error_cost` | High c_i → safety-critical constraint violated consistently |
| `tau_values_tried` | If already spanning [0.0, 1.0], τ widening cannot help — root cause elsewhere |
| `topologies_tried` | If HierarchicalTree was tried, N_max was hit — adapter pool may need recalibration |

### NATS cluster not forming (Server Plan)

Each NATS node must reach the others on port 6222:

```bash
docker compose exec nats-0 wget -q -O- http://nats-1:8222/routez
```

If `"routes": []`, verify container names match the routes in `cluster.conf` and port 6222 is not firewalled.

### Slow task completion

Profile with adapter latency histogram:

```bash
curl -s http://localhost:9090/metrics | grep h2ai_adapter_latency
```

- Local adapters slow → verify `spawn_blocking` is used for all FFI; check `H2AI_MAX_BLOCKING_THREADS` saturation
- Cloud adapters slow → check provider latency; consider switching to a faster endpoint or model
