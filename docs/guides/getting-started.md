# Getting Started

This guide walks through running H2AI Control Plane for the first time: from zero to your first task resolved by the Merge Authority.

---

## Prerequisites

| Requirement | Profile A | Profile B | Profile C |
|---|---|---|---|
| Docker + Compose | required | required | — |
| Kubernetes 1.28+ | — | — | required |
| Helm 3.x | — | — | required |
| Cloud LLM API key | recommended (Auditor) | required | required |

The Auditor is always routed to a cloud reasoning model. You need at minimum one cloud API key — OpenAI, Anthropic, or a compatible provider.

---

## Profile A — Local Dev (fastest path)

### 1. Clone and configure

```bash
git clone https://github.com/h2ai/control-plane.git
cd h2ai-control-plane
cp .env.example .env
```

Edit `.env`:

```bash
# Cloud API key for the Auditor
AUDITOR_API_KEY=sk-...
AUDITOR_API_BASE=https://api.openai.com/v1
AUDITOR_MODEL=gpt-4o

# Optional: local model for Explorers
# Leave empty to use cloud for all adapters
LOCAL_MODEL_PATH=/models/llama-3-8b-instruct.Q4_K_M.gguf
```

### 2. Start the stack

```bash
cd deploy/profile-a
docker compose up -d
```

Check that both containers are healthy:

```bash
docker compose ps
# NAME       STATUS
# h2ai       running
# nats       running (healthy)
```

NATS monitoring is available at `http://localhost:8222`. The H2AI API is at `http://localhost:8080`.

### 3. Seed your ADR corpus

The Dark Knowledge Compiler reads ADRs from the `./adr/` directory. Without ADRs, `J_eff` will be low and the system will reject tasks with `ContextUnderflowError`.

Create at minimum one ADR:

```bash
mkdir -p adr
cat > adr/ADR-001-stateless-auth.md << 'EOF'
# ADR-001: Stateless Authentication

## Status
Accepted

## Context
The API layer must not store session tokens. Compliance requirement CR-2024-07.

## Decision
All authentication is JWT-based. No session store. Tokens are validated on every
request via the shared signing key. Services must not write auth state to any
database.

## Consequences
- Services are horizontally scalable without sticky sessions.
- Token revocation requires short expiry + refresh token rotation.
EOF
```

Restart h2ai to pick up the new corpus:

```bash
docker compose restart h2ai
```

### 4. Run calibration

Before submitting tasks, the system must measure `α`, `κ_base`, and `CG` across the adapter pool:

```bash
curl -s -X POST http://localhost:8080/calibrate | jq .
```

```json
{
  "calibration_id": "cal_01HXYZ...",
  "status": "accepted"
}
```

Stream the calibration progress:

```bash
curl -sN http://localhost:8080/calibrate/cal_01HXYZ.../events
```

```
data: {"event_type":"CalibrationCompleted","payload":{"alpha":0.12,"kappa_base":0.021,"n_max":6.3,"theta_coord":0.28}}
```

Calibration is now cached. You will not need to repeat this unless your adapter pool changes.

### 5. Submit your first task

```bash
curl -s -X POST http://localhost:8080/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Design a stateless JWT refresh token rotation strategy for our API gateway",
    "pareto_weights": {
      "diversity": 0.5,
      "containment": 0.3,
      "throughput": 0.2
    },
    "explorers": {
      "count": 3,
      "tau_min": 0.3,
      "tau_max": 0.8
    }
  }' | jq .
```

```json
{
  "task_id": "task_01HYYZ...",
  "status": "accepted",
  "events_url": "/tasks/task_01HYYZ.../events"
}
```

### 6. Watch the swarm work

```bash
curl -sN http://localhost:8080/tasks/task_01HYYZ.../events
```

You will see events arrive in real time:

```
data: {"event_type":"TopologyProvisioned","payload":{"topology":"FlatMesh","n":3,"tau_values":[0.3,0.55,0.8],"merge_strategy":"CrdtSemilattice"}}

data: {"event_type":"Proposal","payload":{"explorer_id":"exp_A","tau":0.3,"token_cost":847,...}}

data: {"event_type":"Proposal","payload":{"explorer_id":"exp_B","tau":0.55,"token_cost":921,...}}

data: {"event_type":"Proposal","payload":{"explorer_id":"exp_C","tau":0.8,"token_cost":1103,...}}

data: {"event_type":"GenerationPhaseCompleted","payload":{"proposal_count":3}}

data: {"event_type":"Validation","payload":{"explorer_id":"exp_A"}}
data: {"event_type":"Validation","payload":{"explorer_id":"exp_B"}}
data: {"event_type":"BranchPruned","payload":{"explorer_id":"exp_C","reason":"Proposes storing refresh tokens in Redis — violates ADR-001 stateless auth requirement","constraint_error_cost":0.72}}

data: {"event_type":"SemilatticeCompiled","payload":{"valid_proposals":2,"pruned_proposals":1,"merge_strategy":"CrdtSemilattice"}}
```

### 7. Resolve in the Merge Authority

Open `http://localhost:8080` in a browser. The Merge Authority UI shows:

- **Valid proposals panel** — diffs from exp_A and exp_B, grouped by affected component
- **Tombstone panel** — exp_C's proposal with the rejection reason and constraint cost
- **Physics panel** — live `θ_coord`, `κ_eff`, `N_max`, `J_eff`

Select, synthesize, or reject proposals. Click **Resolve**. The task closes with `MergeResolvedEvent`.

---

## Profile B — Team Node

```bash
cd deploy/profile-b
docker compose up -d
```

The Merge Authority UI is available at `http://<server-ip>`. Multiple team members can submit manifests concurrently. All task state is replicated across the 3-node NATS cluster — any node failure is tolerated without data loss.

See [Deployment — Profile B](../architecture/04-deployment.md) for team ADR corpus setup.

---

## Profile C — Kubernetes

```bash
# Create namespace
kubectl apply -f deploy/profile-c/namespace.yaml

# Upload your ADR corpus
kubectl create configmap adr-corpus --from-file=./adr/ -n h2ai

# Install via Helm
helm repo add h2ai https://h2ai.github.io/control-plane
helm install h2ai h2ai/h2ai-control-plane \
  --namespace h2ai \
  --set ingress.enabled=true \
  --set ingress.hosts[0].host=h2ai.corp.example.com \
  --set serviceMonitor.enabled=true
```

---

## What to read next

| Topic | Document |
|---|---|
| All REST endpoints and event schemas | [API Reference](../reference/api.md) |
| All environment variables and Helm values | [Configuration Reference](../reference/configuration.md) |
| Writing ADRs that the compiler understands | [ADR Corpus Guide](adr-corpus.md) |
| Implementing a custom compute adapter | [Adapter Development](adapters.md) |
| Monitoring, alerting, upgrading | [Operations Guide](../operations/operations.md) |
| Diagnosing common problems | [Troubleshooting](../operations/troubleshooting.md) |
