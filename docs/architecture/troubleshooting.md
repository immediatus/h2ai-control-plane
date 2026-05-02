# Troubleshooting

---

## Task submission errors

### `ContextUnderflowError` on every task

**Symptom:** `POST /tasks` always returns 400 with `j_eff` below `threshold`.

**Diagnosis:**

```bash
# Check the current J_eff threshold
curl http://localhost:8080/calibrate/current | jq .

# Check what the response says is missing
curl -X POST http://localhost:8080/tasks \
  -H "Content-Type: application/json" \
  -d '{"description":"...","pareto_weights":{"diversity":0.5,"containment":0.3,"throughput":0.2},"explorers":{"count":3}}' \
  | jq .missing_coverage
```

**Causes and fixes:**

| Cause | Fix |
|---|---|
| Constraint corpus directory is empty or wrong path | Check `H2AI_CONSTRAINT_CORPUS_PATH`. Run `ls $H2AI_CONSTRAINT_CORPUS_PATH/*.md`. |
| ADRs exist but constraints section is thin | Add explicit `## Constraints` bullets or use typed `ConstraintDoc` format. See [Constraint Corpus Guide](../guides/constraint-corpus.md). |
| Task description uses domain language not in ADRs | Add the relevant ADR, or add explicit `context` to the task manifest. |
| `H2AI_J_EFF_THRESHOLD` is set too high | Lower the threshold if your corpus is intentionally minimal: `--set config.jEffThreshold="0.25"`. |
| h2ai did not reload the corpus after ADR changes | Send `SIGHUP` to the process or restart the container. |

---

### `CalibrationRequiredError` on task submission

**Symptom:** `POST /tasks` returns 503.

**Fix:**

```bash
curl -X POST http://localhost:8080/calibrate
# wait for CalibrationCompletedEvent on the events stream, then retry
```

If calibration itself fails, check the calibration events stream:

```bash
curl -sN http://localhost:8080/calibrate/{calibration_id}/events
```

---

## Task execution problems

### Task gets stuck — no events after `TopologyProvisionedEvent`

**Symptom:** SSE stream stops after topology provisioned. No `ProposalEvent` or `MultiplicationConditionFailedEvent` arrives.

**Diagnosis:**

```bash
# Check if Explorers are running
docker logs h2ai | grep explorer

# Check NATS for events on the task subject
nats stream view H2AI_TASKS --subject "h2ai.tasks.{task_id}"

# Check blocking thread pool
curl http://localhost:9090/metrics | grep h2ai_blocking_threads
```

**Causes:**

- **Blocking pool saturated** — `h2ai_blocking_threads_active` equals `H2AI_MAX_BLOCKING_THREADS`. Local Explorers are queued. Either reduce concurrent tasks, increase `H2AI_MAX_BLOCKING_THREADS`, or reduce local adapter count.
- **Cloud API rate limit** — Cloud Explorers are waiting on 429 responses. Check adapter logs. The adapter will retry after the `retry-after` header value, but if all cloud slots are rate-limited, the task will time out.
- **Explorer timeout too short** — If a model is slow, the `H2AI_EXPLORER_TIMEOUT_SECS` fires before the Explorer completes. Increase the timeout.

---

### High `ZeroSurvivalEvent` rate

**Symptom:** Tasks repeatedly trigger the MAPE-K retry loop. `h2ai_zero_survival_total` is climbing.

**Diagnosis — check the `BranchPrunedEvent` reasons:**

```bash
# Pull pruned branch reasons from the event log
nats stream view H2AI_TASKS --subject "h2ai.tasks.{task_id}" | grep BranchPruned
```

**Scenario A — All proposals rejected for the same ADR violation:**  
The Auditor is consistently catching a specific constraint. The Explorers are not aware of or not respecting it.

Fix: The constraint is valid but not prominent enough. Strengthen the ADR wording. Move the constraint to the explicit `constraints` field in the task manifest for immediate effect.

**Scenario B — All proposals rejected for different reasons:**  
The task is genuinely hard — the solution space is narrow and Explorers are exploring the wrong regions. The MAPE-K loop is widening τ but not finding valid proposals.

Fix: Reduce `J_eff_threshold` temporarily, add more explicit `context` to the manifest, or reduce the number of constraints being enforced for this task class.

**Scenario C — Proposals rejected with vague reasons:**  
The Auditor is hallucinating rejections. This happens when the Auditor's `system_context` is ambiguous.

Fix: Review the compiled `system_context` in the `TaskBootstrappedEvent` payload. Rewrite ambiguous ADR sections.

**Scenario D — All proposals fail before reaching the Auditor (`ProposalFailedEvent`):**  
This is not a zero-survival scenario — it is an Explorer failure scenario. The MAPE-K loop treats it the same way but the root cause is different (adapter errors, not Auditor rejections). Check adapter logs for OOM, timeout, or API errors.

---

### `MultiplicationConditionFailedEvent` — `ErrorDecorrelation`

**Symptom:** Phase 2.5 gate fails with `ρ = 0.94, threshold = 0.9`. Task retries Phase 2 repeatedly.

**Meaning:** Two or more Explorers make the same errors on the calibration set. They are structurally redundant — adding both gives no diversity benefit.

**Fix options:**

1. **Widen τ spread** — Increase the gap between `tau_min` and `tau_max` in the task manifest:
   ```json
   {"explorers": {"tau_min": 0.1, "tau_max": 0.95}}
   ```

2. **Route to different model backends** — If all Explorers use the same model, add a second model to the adapter pool. Same model, same weights → high ρ by construction.

3. **Reduce Explorer count** — If you only have one model and cannot widen τ enough, reduce `count`. Two Explorers with different τ may pass where four fail.

---

### `MultiplicationConditionFailedEvent` — `BaselineCompetence`

**Symptom:** Phase 2.5 gate fails with `p_correct = 0.44, threshold = 0.5`.

**Meaning:** One of the adapters performs worse than random chance on the calibration task set. Adding it to the swarm makes the collective worse.

**Fix:**
- Remove the underperforming adapter from the pool.
- If the adapter is expected to perform better, check: is the model loaded correctly? Is the context size sufficient? Is the prompt format correct for this model family?
- Re-run calibration after fixing: `POST /calibrate`.

---

### `MultiplicationConditionFailedEvent` — `CommonGroundFloor`

**Symptom:** Phase 2.5 gate fails with `cg_mean = 0.21, theta_coord = 0.28`.

**Meaning:** Explorer pairs are too epistemically distant. Their Common Ground is below the coordination floor — they would need so many tokens to verify consistency that coordination cost exceeds diversity benefit.

**Fix:**
- Reduce τ spread (bring τ values closer together).
- Use adapters from more similar model families.
- Or accept the constraint and reduce N — fewer, more similar Explorers with CG above θ_coord.

---

## NATS problems

### NATS healthcheck failing in docker compose

```bash
docker compose logs nats
```

Common cause: `nats.conf` has incorrect `store_dir` path, or the volume mount is missing write permissions.

```bash
# Check if the conf file is mounted correctly
docker compose exec nats cat /etc/nats/nats.conf

# Check JetStream store directory
docker compose exec nats ls -la /data/jetstream
```

### NATS cluster not forming (Server Plan)

Each NATS node must be able to reach the others on port 6222. In docker compose, container names are resolvable as hostnames within the network.

```bash
# Test connectivity between nodes
docker compose exec nats-0 wget -q -O- http://nats-1:8222/routez
```

If routes are empty (`"routes": []`), the cluster is not connected. Verify:
- Container names match the routes in `cluster.conf` (`nats-0`, `nats-1`, `nats-2`)
- Port 6222 is not blocked by a firewall rule
- All three containers are running before the first task is submitted

### NATS KV calibration cache missing after restart

The KV bucket is stored in the JetStream file store. If the volume was not persisted, the bucket is lost.

Fix: ensure the `nats-data` volume is a named volume (not anonymous) in docker compose, and that the PVC is bound in Kubernetes.

Workaround: re-run calibration after restart: `POST /calibrate`.

---

## Performance problems

### High α (contention)

`h2ai_alpha` rising above 0.20 indicates a serial bottleneck. Causes:

| Symptom | Likely cause |
|---|---|
| α spikes correlate with NATS publish latency | NATS disk I/O bottleneck. Check `nats-data` volume throughput. |
| α spikes correlate with blocking thread saturation | Blocking pool too large for available CPU — OS scheduler thrash. Reduce `H2AI_MAX_BLOCKING_THREADS`. |
| α spikes during Merge Authority resolution | Many concurrent tasks waiting for human resolution. This is structural — the merge step is inherently serial. |

### High κ_eff (coordination cost)

`h2ai_kappa_eff` rising means coordination cost is increasing. Causes:

| Symptom | Likely cause |
|---|---|
| `cg_mean` dropped | Adapter pool changed — new diverse models added. Recalibrate. |
| `kappa_base` rose | Token exchange overhead increased. Check network latency between Explorer pods. |
| N_max dropped | Combined effect of rising α and κ. System will select smaller swarms automatically. |

### Slow task completion

End-to-end task latency = calibration (one-time) + provisioning + generation + auditing + merge.

The dominant cost is usually generation. Profile by checking the `h2ai_adapter_latency_seconds` histogram:

```bash
curl -s http://localhost:9090/metrics | grep h2ai_adapter_latency
```

If local adapters dominate: check that `spawn_blocking` is being used correctly (no FFI on the async pool), and that `H2AI_MAX_BLOCKING_THREADS` is not saturated.

If cloud adapters dominate: check API provider latency. Consider switching the slow adapter to a faster endpoint or model.

---

## Diagnosing a `TaskFailedEvent`

When a task fails after exhausted retries, the `TaskFailedEvent` payload contains the full diagnostic. Parse it:

```bash
# Get the TaskFailedEvent from the event log
nats stream view H2AI_TASKS --subject "h2ai.tasks.{task_id}" | \
  python3 -c "import sys,json; events = [json.loads(l) for l in sys.stdin if l.startswith('{')]; failed = [e for e in events if e.get('event_type') == 'TaskFailed']; print(json.dumps(failed[0], indent=2))"
```

Read the payload fields:

| Field | What to check |
|---|---|
| `multiplication_condition_failure` | Which of the 3 conditions blocked. See condition-specific sections above. |
| `branch_pruned_events[*].reason` | What the Auditor rejected and why. Pattern in rejections → ADR coverage gap. |
| `branch_pruned_events[*].constraint_error_cost` | High c_i rejections → safety-critical constraint being violated consistently. |
| `tau_values_tried` | If all τ sets already span `[0.0, 1.0]`, τ widening cannot help. Root cause is elsewhere. |
| `topologies_tried` | If HierarchicalTree was tried, N_max was hit. Adapter pool may need recalibration. |
