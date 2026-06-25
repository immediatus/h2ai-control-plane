# Oracle System

The oracle is the ground-truth feedback signal that tells the calibration subsystem whether
a winning output was actually correct.  It operates entirely asynchronously: the engine
delivers its output to the client and simultaneously fires an `OraclePendingEvent`; the
oracle worker later receives the verdict and updates the rolling calibration window.

All values in this document are sourced from `crates/h2ai-api/src/oracle/mod.rs`,
`crates/h2ai-orchestrator/src/oracle.rs`, `crates/h2ai-config/src/lib.rs`, and
`crates/h2ai-config/reference.toml`.

---

## 1. Data Flow

```
ExecutionEngine (Phase 6)
  │  oracle_dispatch::fire()
  │  publishes OraclePendingEvent
  ▼
NATS subject: h2ai.oracle.{tenant_id}.pending
  │
  ▼
OracleWorker (oracle_worker.rs)
  │  calls OracleClient.evaluate (external HTTP oracle service)
  │  receives OracleResultEvent
  ▼
NATS: H2AI_ORACLE_CALIBRATION bucket updated
  │  patch_ensemble_p_from_oracle() called when n_obs ≥ 10
  ▼
MetricsState updated (h2ai_oracle_* gauges)
```

### OraclePendingEvent fields

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | `TaskId` | Unique task identifier |
| `tenant_id` | `TenantId` | Tenant scope |
| `winning_output` | `String` | Output selected by merge engine |
| `q_confidence` | `f64` | Engine's predicted correctness probability |
| `n_used` | `u32` | Actual ensemble size used for this task |
| `oracle_spec` | `OracleSpec` | Evaluation specification including domain |
| `domain` | `OracleDomain` | Domain tag for family routing |

---

## 2. Oracle Domain and Family

`OracleDomain::family()` maps task domain to oracle evaluation family:

| Domain | Family |
|--------|--------|
| `Code` | `Syntactic` |
| `Factual` | `Semantic` |
| `Reasoning` | `Semantic` |
| `Unknown` | `Semantic` |
| `Human` | `Human` |

The family determines which external oracle service is invoked.

---

## 3. Calibration Window

The oracle worker maintains a rolling FIFO observation window per tenant.

### Window management

Window size cap: `oracle_window_size = 200` observations.
When the window exceeds the cap, the oldest observations are drained first
(`enforce_fifo_cap()` in `oracle/mod.rs`).

### OracleObservation fields

Each completed oracle evaluation appends one `OracleObservation`:
- `q_confidence`: engine's predicted probability for this task
- `y_oracle`: binary oracle verdict (true = passed)
- `residual`: `|q_confidence − y_oracle|`

---

## 4. Calibration Statistics

All statistics are recomputed on every new observation.

### Expected Calibration Error (ECE)

```
ECE = (1/n) × Σᵢ |q_confidence_i − y_oracle_i|
```

A perfectly calibrated system has ECE = 0; a system that always predicts 1.0 but passes
only half the time has ECE = 0.5.

Target: ECE < 0.05.
Alert threshold: `oracle_ece_alert_threshold = 0.15`.

### Pass rate

```
pass_rate = count(y_oracle_i == true) / n
```

Floor: `oracle_pass_rate_floor = 0.30`.  Dropping below this value indicates the oracle
is consistently rejecting engine outputs and should trigger manual investigation.

### Residual P90

The 90th percentile of residuals, using the Angelopoulos–Bates finite-sample index:

```
idx = ceil((n + 1) × 0.9) − 1    (clamped to [0, n−1])
```

Tracks the width of the upper tail of calibration errors; used as a proxy for conformal
interval width in the `/metrics` endpoint.

### Calibration basis

The basis determines how much to trust the calibration statistics:

| Window size | ECE | Basis | Meaning |
|-------------|-----|-------|---------|
| n < 10 | any | Heuristic (0) | Insufficient data; use prior |
| 10 ≤ n < 30 | any | Bootstrap (1) | CLT not yet satisfied |
| n ≥ 30 | ECE < 0.15 | Conformal (2) | Statistically grounded |
| n ≥ 30 | ECE ≥ 0.15 | Heuristic (0) | Quality regression; revert to prior |

---

## 5. Ensemble p_mean Patching

Once `n_observations ≥ 10`, `patch_ensemble_p_from_oracle()` updates the in-memory
calibration state with an empirically derived baseline competence:

```rust
EnsembleCalibration::from_measured_p(pass_rate.clamp(0.5, 1.0), cg_mean, max_ensemble_size)
```

This replaces the CG-proxy formula `p_mean = 0.5 + CG/2` with directly observed accuracy.
The update is applied to the tenant's live calibration in-memory and does not require a
full `POST /v1/calibrate` recalibration cycle.

---

## 6. Oracle Gate

The oracle gate (`oracle_gate.rs`) is an optional pre-delivery validation step.
Default: **disabled** (`oracle_gate.enabled = false`).

| Config field | Default | Meaning |
|-------------|---------|---------|
| `enabled` | false | Disabled; opt-in per scenario |
| `subject` | `h2ai.oracle.gate` | NATS subject for gate requests |
| `timeout_secs` | 30 | Seconds to wait for oracle verdict |
| `on_timeout` | `pass` | Deliver output when oracle times out |
| `on_fail` | `evict` | Evict proposal when oracle rejects |
| `min_confidence` | 0.7 | Minimum confidence to accept a pass verdict |

When enabled, the oracle gate runs between Phase 5a (synthesis) and final delivery.
A verdict of fail with `on_fail = evict` causes the MAPE-K loop to retry the task.

---

## 7. Drift Detection Interaction

The DDM (Drift Detection Method) and BOCPD (Bayesian Online Changepoint Detection)
monitors track calibration drift across the rolling oracle window:

- `drift_ddm_window = 20`: sliding window size for DDM.
- `drift_ddm_k = 2.5`: detection threshold in standard deviations.
- `drift_bocpd_hazard_rate = 0.01`: per-observation changepoint probability.
- `drift_bocpd_changepoint_threshold = 0.90`: posterior mass threshold to fire `CalibrationChangepoint`.

When drift is detected, `drift_conformal_margin = 0.05` is subtracted from the
verification threshold as a conservative coverage guarantee (ORCA conformal margin).

Automatic recalibration on drift: `auto_recalibrate_on_drift = false` (operator opt-in).
Stale-calibration warning after `drift_staleness_ttl_secs = 3600` seconds.

---

## 8. Prometheus Metrics

The following oracle metrics are exposed at `/metrics`:

| Metric | Type | Description |
|--------|------|-------------|
| `h2ai_oracle_ece_gauge` | gauge | Current ECE (target < 0.05, alert > 0.15) |
| `h2ai_oracle_n_observations_total` | gauge | Rolling observation count |
| `h2ai_oracle_coverage_rate` | gauge | Fraction of tasks that carried an `OracleSpec` |
| `h2ai_oracle_pass_rate` | gauge | Rolling oracle pass rate (last 200 observations) |
| `h2ai_oracle_residual_p90` | gauge | P90 of calibration residuals |
| `h2ai_calibration_basis` | gauge | Basis: 0=Heuristic, 1=Bootstrap, 2=Conformal |
| `h2ai_oracle_tasks_total` | counter | Total successfully resolved tasks |
| `h2ai_oracle_tasks_with_spec_total` | counter | Tasks that carried an `OracleSpec` |
| `h2ai_calibration_source{source}` | gauge | Active calibration source (1 = active) |

`h2ai_calibration_source` labels: `measured`, `partial_fit`, `synthetic_priors`.
