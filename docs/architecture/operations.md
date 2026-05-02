# Operations Guide

This guide covers day-to-day operation of H2AI Control Plane in production: monitoring, alerting, scaling, upgrading, and backup.

---

## Monitoring

### Key metrics to watch

The system exposes 20+ Prometheus metrics. In practice, six metrics drive operational decisions:

**`h2ai_kappa_eff`** — The most important single gauge.  
When this rises, the system is approaching the scalability ceiling. If it reaches `(1 − α) / N_max²`, adding Explorers will make results worse, not better.

Alert: `h2ai_kappa_eff > 0.035` (approaching retrograde for typical AI agent α)

**`h2ai_alpha`** — Contention coefficient.  
Spikes indicate a shared resource bottleneck: context lock contention, NATS publish queue building up, or the Merge Authority resolution step becoming a bottleneck. Sustained α > 0.20 means the serial fraction is too high — investigate NATS publish latency and blocking thread pool saturation.

Alert: `h2ai_alpha > 0.20`

**`h2ai_zero_survival_total` (rate)** — Rate of zero-survival events.  
A non-zero rate means the Auditor is rejecting everything repeatedly. This indicates ADR constraints are either too strict for the current task domain or Explorers are systematically missing coverage. Check `h2ai_autonomic_retries_total` alongside this.

Alert: `rate(h2ai_zero_survival_total[10m]) > 0.1`

**`h2ai_calibration_age_seconds`** — Staleness of calibration data.  
If the adapter pool has changed since calibration (new models added, API providers changed), calibration data is stale. Stale calibration means N_max is wrong — the system may be provisioning too many or too few Explorers.

Alert: `h2ai_calibration_age_seconds > 86400`

**`h2ai_blocking_threads_active`** — llama.cpp FFI pool utilization.  
If this approaches `H2AI_MAX_BLOCKING_THREADS`, the blocking pool is saturated. New Explorer calls queue behind existing ones — wall time rises, timeouts increase, `ProposalFailedEvent` count climbs.

Alert: `h2ai_blocking_threads_active / H2AI_MAX_BLOCKING_THREADS > 0.9`

**`h2ai_j_eff`** — Dark Knowledge gap per task.  
Sustained low J_eff (even for tasks that pass the threshold) means the constraint corpus is thin relative to what tasks require. Invest in constraint authorship before adding more Explorers.

No alert — trend awareness. Watch for steady decline over weeks.

---

### Recommended alert rules

```yaml
# prometheus/alerts.yml

groups:
  - name: h2ai
    rules:
      - alert: H2AIKappaEffHigh
        expr: h2ai_kappa_eff > 0.035
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "κ_eff is high — approaching scalability ceiling"
          description: "κ_eff={{ $value }}. N_max may be lower than expected. Consider reducing Explorer count or widening model diversity."

      - alert: H2AIAlphaHigh
        expr: h2ai_alpha > 0.20
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "α is high — serial contention detected"
          description: "Check NATS publish latency and blocking thread pool saturation."

      - alert: H2AIZeroSurvivalRateHigh
        expr: rate(h2ai_zero_survival_total[10m]) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High zero-survival rate — Auditor rejecting everything"
          description: "ADR constraints may be too strict or Explorers are missing domain coverage."

      - alert: H2AICalibrationStale
        expr: h2ai_calibration_age_seconds > 86400
        labels:
          severity: warning
        annotations:
          summary: "Calibration data is stale (>24h)"
          description: "Run POST /calibrate if the adapter pool has changed."

      - alert: H2AIBlockingPoolSaturated
        expr: h2ai_blocking_threads_active / on() h2ai_blocking_threads_max > 0.9
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Blocking thread pool near saturation"
          description: "llama.cpp FFI pool is {{ $value | humanizePercentage }} utilized. Explorer timeouts likely. Consider reducing local adapter count or increasing H2AI_MAX_BLOCKING_THREADS."

      - alert: H2AINATSPublishSlow
        expr: histogram_quantile(0.99, rate(h2ai_nats_publish_latency_seconds_bucket[5m])) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "NATS publish p99 latency > 100ms"
          description: "Event log writes are slow. Check NATS cluster health and disk I/O."
```

---

### Grafana dashboard

A reference Grafana dashboard JSON is provided at `deploy/server/grafana/h2ai-dashboard.json`. Import it via **Dashboards → Import → Upload JSON**.

The dashboard includes:
- USL physics panel (`κ_eff`, `α`, `N_max`, `θ_coord` time series)
- Task throughput and phase duration heatmaps
- Proposal outcome breakdown (validated / pruned / failed)
- MAPE-K activity (retry rate, autonomic action breakdown)
- Adapter performance (latency histograms per adapter)
- Blocking thread pool utilization

---

## Scaling

### Scaling the control plane (Cloud Plan)

The h2ai orchestrator is stateless with respect to task execution — all state is in NATS JetStream. Scale horizontally by increasing `replicaCount` or letting the HPA handle it:

```bash
# Manual scale
kubectl scale deployment h2ai-control-plane --replicas=6 -n h2ai

# Or update Helm values
helm upgrade h2ai h2ai/h2ai-control-plane \
  --set autoscaling.maxReplicas=20 \
  --reuse-values
```

The HPA scales on CPU and memory. For workloads dominated by cloud adapter calls (async, low CPU), consider scaling on custom metrics from the `h2ai_tasks_active` gauge instead.

### Scaling NATS

NATS JetStream is a StatefulSet. It does not scale horizontally during operation without a plan:

- **3 nodes:** minimum for quorum (tolerates 1 failure)
- **5 nodes:** tolerates 2 failures — use for high-availability Cloud Plan
- Do not scale NATS during active task execution. Quorum reconfiguration is a maintenance operation.

```bash
# Scale NATS (maintenance window required)
kubectl scale statefulset nats --replicas=5 -n h2ai
```

### Tuning for high-load plans

When tasks per minute is high, tune these values before scaling replicas:

```bash
# Increase blocking pool for more concurrent local inference
kubectl set env deployment/h2ai-control-plane H2AI_MAX_BLOCKING_THREADS=16 -n h2ai

# Reduce calibration task count for faster startup (less accurate α/κ)
kubectl set env deployment/h2ai-control-plane H2AI_CALIBRATION_TASKS=1 -n h2ai

# Increase Explorer timeout for slow cloud providers
kubectl set env deployment/h2ai-control-plane H2AI_EXPLORER_TIMEOUT_SECS=180 -n h2ai
```

---

## Upgrading

### Rolling upgrade (Cloud Plan)

Kubernetes handles rolling updates automatically. The deployment uses `RollingUpdate` strategy with `maxUnavailable: 0` by default.

```bash
# Upgrade to a new image tag
helm upgrade h2ai h2ai/h2ai-control-plane \
  --set image.tag=0.2.0 \
  --reuse-values

# Monitor the rollout
kubectl rollout status deployment/h2ai-control-plane -n h2ai
```

In-flight tasks at upgrade time: tasks survive pod restarts because all state is in NATS JetStream. A new pod that picks up an in-flight task replays the event log from offset 0 and resumes from the current phase. The client's SSE stream reconnects automatically using `Last-Event-ID`.

### Upgrading NATS

NATS upgrades require a rolling restart of the StatefulSet. Update the image tag in the NATS Helm dependency and perform a rolling restart:

```bash
helm upgrade h2ai h2ai/h2ai-control-plane \
  --set nats.image.tag=2.11 \
  --reuse-values

kubectl rollout restart statefulset/nats -n h2ai
kubectl rollout status statefulset/nats -n h2ai
```

### Post-upgrade calibration

After any upgrade that changes adapter versions or model weights, re-run calibration:

```bash
curl -X POST https://h2ai.corp.example.com/calibrate
```

Calibration data from a previous binary version may not reflect changed adapter behavior. Stale calibration produces inaccurate `N_max`, leading to either underprovisioning (wasted diversity) or overprovisioning (throughput retrograde).

---

## Backup and recovery

### What needs backup

**NATS JetStream file store** — contains the entire event log for all tasks. This is the only persistent state. Back it up.

**Constraint corpus** — your team's architectural knowledge (ADRs and typed ConstraintDocs). This lives in your git repository, not in the control plane. Ensure the constraint repository is backed up through normal git processes.

**Adapter configuration** (`adapters.toml`) — also in git. Not runtime state.

**Calibration data** — stored in the NATS KV store (backed by the file store). Backed up with the file store. If lost, run `POST /calibrate` to regenerate.

### NATS backup

```bash
# Local Plan — back up the JetStream data directory
tar -czf nats-backup-$(date +%Y%m%d).tar.gz /var/lib/nats/jetstream/

# Cloud Plan — use the nats CLI tool
nats stream backup H2AI_TASKS /backup/h2ai-tasks-$(date +%Y%m%d)/
```

For Cloud Plan, consider a CronJob that runs nightly:

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: nats-backup
  namespace: h2ai
spec:
  schedule: "0 2 * * *"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
            - name: backup
              image: natsio/nats-box:latest
              command:
                - sh
                - -c
                - |
                  nats stream backup H2AI_TASKS /backup/h2ai-$(date +%Y%m%d)/ \
                    --server nats://nats.h2ai.svc.cluster.local:4222
              volumeMounts:
                - name: backup-storage
                  mountPath: /backup
          restartPolicy: OnFailure
          volumes:
            - name: backup-storage
              persistentVolumeClaim:
                claimName: nats-backup
```

### Recovery

Full recovery from the NATS file store is a replay operation:

```bash
# Restore the file store (Local Plan)
tar -xzf nats-backup-20260419.tar.gz -C /var/lib/nats/

# Restart NATS — it will replay from the restored file store
docker compose restart nats

# Restart h2ai — it reconnects to NATS and resumes in-flight tasks
docker compose restart h2ai
```

In-flight tasks at backup time will resume from their last committed event. Proposals that were being generated at crash time will be retried by the MAPE-K loop (if `ProposalFailedEvent` was not yet committed, the timeout will fire and publish it on resume).

---

## Recalibration triggers

Recalibrate after any of these events:

| Event | Why |
|---|---|
| New adapter added to pool | New α/κ measurements needed for the full pool |
| Adapter model version upgraded | p_correct and ρ may have changed |
| Sustained `h2ai_zero_survival_total` rate increase | May indicate calibration drift |
| `h2ai_calibration_age_seconds` alert fires | Data is stale by policy |
| Hardware change (RAM, GPU added) | Blocking thread pool capacity changed; re-tune `H2AI_MAX_BLOCKING_THREADS` and recalibrate |
| α spike that does not resolve | Serial fraction changed — investigate and recalibrate |

Recalibration takes as long as `H2AI_CALIBRATION_TASKS` full inference cycles. With 3 calibration tasks and 5 adapters, expect 3–10 minutes depending on adapter latency. Tasks submitted during calibration receive `503 CalibrationRequiredError` until the new data is committed.

To avoid downtime during recalibration on Cloud Plan, route traffic away from the recalibrating instance using pod labels before starting calibration.
