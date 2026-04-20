# Getting Started

This guide walks through running H2AI Control Plane for the first time: from zero to your first task resolved by the Merge Authority.

---

## Prerequisites

| Requirement | Local Plan | Server Plan | Cloud Plan |
|---|---|---|---|
| Docker + Compose | required | required | — |
| Kubernetes 1.28+ | — | — | required |
| Helm 3.x | — | — | required |
| Cloud LLM API key | optional (real results) | required | required |

Both explorer and auditor default to `mock` (deterministic test double). To get real LLM output you need at least one cloud API key — OpenAI, Anthropic, or an Ollama endpoint. The Auditor should be routed to a capable reasoning model for reliable ADR constraint enforcement.

---

## Local Plan — Local Dev (fastest path)

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

# Auditor adapter (ADR constraint gate — use a capable reasoning model)
H2AI_AUDITOR_PROVIDER=anthropic
H2AI_AUDITOR_MODEL=claude-3-5-haiku-20241022
H2AI_AUDITOR_API_KEY_ENV=ANTHROPIC_API_KEY

# The actual key value (not the var name)
ANTHROPIC_API_KEY=sk-ant-...

# Alternative: OpenAI
# H2AI_EXPLORER_PROVIDER=openai
# H2AI_EXPLORER_MODEL=gpt-4o-mini
# H2AI_EXPLORER_API_KEY_ENV=OPENAI_API_KEY
# OPENAI_API_KEY=sk-...

# Alternative: local Ollama (no API key needed)
# H2AI_EXPLORER_PROVIDER=ollama
# H2AI_EXPLORER_MODEL=llama3.2
# H2AI_EXPLORER_ENDPOINT=http://localhost:11434
```

### 2. Start the stack

```bash
cd deploy/local
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

### 3. Seed your constraint corpus

The Dark Knowledge Compiler reads constraint documents from the `./adr/` directory. Without constraints, `J_eff` will be low and the system will reject tasks with `ContextUnderflowError`.

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
data: {"event_type":"CalibrationCompleted","payload":{"calibration_id":"cal_01HXYZ...","coefficients":{"alpha":0.12,"kappa_base":0.021,"cg_samples":[]},"coordination_threshold":{"value":0.28},"timestamp":"2026-04-19T10:00:00Z"}}
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
data: {"event_type":"TopologyProvisioned","payload":{"task_id":"task_01HYYZ...","topology_kind":"Ensemble","explorer_configs":[{"tau":0.3},{"tau":0.55},{"tau":0.8}],"merge_strategy":"CrdtSemilattice","n_max":6.3,"kappa_eff":0.019,"retry_count":0,...}}

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

## Server Plan — Team Node

```bash
cd deploy/server
docker compose up -d
```

The Merge Authority UI is available at `http://<server-ip>`. Multiple team members can submit manifests concurrently. All task state is replicated across the 3-node NATS cluster — any node failure is tolerated without data loss.

See [Deployment — Server Plan](../architecture/deployment.md) for team constraint corpus setup.

---

## Cloud Plan — Kubernetes

```bash
# Create namespace
kubectl apply -f deploy/cloud/namespace.yaml

# Upload your constraint corpus
kubectl create configmap constraint-corpus --from-file=./adr/ -n h2ai

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
| Writing constraints that the compiler understands | [Constraint Corpus Guide](constraint-corpus.md) |
| Implementing a custom compute adapter | [Adapter Development](adapters.md) |
| Monitoring, alerting, upgrading | [Operations Guide](../operations/operations.md) |
| Diagnosing common problems | [Troubleshooting](../operations/troubleshooting.md) |
