# Oracle System

The oracle is the ground-truth feedback signal that tells the calibration subsystem whether
a winning output was actually correct. It operates entirely asynchronously: the engine
delivers its output to the client and simultaneously fires an `OraclePendingEvent`; the
oracle worker later receives the verdict and updates the rolling calibration window.

All values in this document are sourced from `crates/h2ai-api/src/oracle/mod.rs`,
`crates/h2ai-orchestrator/src/oracle_gate.rs`, `crates/h2ai-types/src/sizing.rs`,
`crates/h2ai-config/src/lib.rs`, and `crates/h2ai-config/reference.toml`.

---

## Overview

```
ExecutionEngine (Phase 6)
  │  publishes OraclePendingEvent
  ▼
NATS subject: h2ai.oracle.*.pending
  │
  ▼
OracleWorker (oracle_worker.rs)
  │  evaluates verdict (y_oracle), computes residual
  │  appends OracleObservation to FIFO window
  │  recomputes ECE, pass_rate, residual P90, calibration basis
  ▼
patch_ensemble_p_from_oracle()   (when n_observations ≥ 10)
  │  publishes OracleCalibrationPatchedEvent
  ▼
MetricsState updated (h2ai_oracle_* gauges)
```

---

## 1. Data Flow

OracleWorker subscribes to `h2ai.oracle.*.pending`. On receipt of an `OraclePendingEvent`
it routes the request by domain family (see §2), evaluates the verdict, records an
`OracleObservation`, and publishes an `OracleResultEvent` to `h2ai.oracle.results`.

### OraclePendingEvent fields

| Field | Type | Notes |
|-------|------|-------|
| `task_id` | `TaskId` | Unique task identifier |
| `winning_output` | `String` | Output selected by merge engine |
| `q_confidence` | `f64` | Engine's predicted correctness probability |
| `n_used` | `u32` | Actual ensemble size used for this task |
| `oracle_spec` | `OracleSpec` | Evaluation specification |
| `domain` | `OracleDomain` | Domain tag for family routing |
| `oracle_specs` | `Vec<OracleSpec>` | Additional specs (`#[serde(default)]`) |
| `tenant_id` | `TenantId` | Tenant scope (`#[serde(default)]`) |

---

## 2. Oracle Domain and Family

`OracleDomain::family()` determines which evaluation family is invoked:

| Domain | Family |
|--------|--------|
| `Code` | `Syntactic` |
| `Factual` | `Semantic` |
| `Reasoning` | `Semantic` |
| `Unknown` | `Semantic` |
| `Human` | `Human` |

### FUSE multi-oracle reduction

When multiple oracle specs are present, results are combined via FUSE:

1. Group verdict scores by `OracleFamily` (`Syntactic=0`, `Semantic=1`, `Human=2`).
2. Take `min(score)` within each family.
3. Average the per-family minima across all represented families.
4. The task passes if `final_score ≥ 0.5`.

---

## 3. Calibration Window

The oracle worker maintains a rolling FIFO observation window per tenant.

### OracleObservation fields

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | `String` | Task this observation belongs to |
| `q_confidence` | `f64` | Ensemble confidence score |
| `y_oracle` | `bool` | Binary oracle verdict |
| `residual` | `f64` | `\|q_confidence − y_oracle as f64\|` — nonconformity score |
| `domain` | `OracleDomain` | Domain of the evaluated task |
| `timestamp_ms` | `u64` | Wall-clock time of observation |

### Window management

- Default cap: `oracle_window_size = 200` observations.
- `enforce_fifo_cap(observations, cap)` drains the oldest entries when `len > cap`.

---

## 4. Calibration Statistics

All statistics are recomputed from the full window on every new observation.

### Expected Calibration Error (ECE)

ECE is the mean of the pre-computed per-observation residuals:

```
ECE = (1/n) × Σᵢ residual_i
```

where `residual_i = |q_confidence_i − y_oracle_i as f64|`.

Alert threshold: `oracle_ece_alert_threshold = 0.15`.

### Pass rate

```
pass_rate = count(y_oracle == true) / n
```

Floor: `oracle_pass_rate_floor = 0.30`.

### Residual P90 (Angelopoulos–Bates Theorem 1)

```
idx = ⌈(n + 1) × 0.9⌉ − 1    (clamped to [0, n−1])
p90 = sorted_residuals[idx]
```

---

## 5. Calibration Basis

The basis (encoded as an integer) determines how much statistical trust to place in the
current window:

| Condition | Basis | Code |
|-----------|-------|------|
| n < 10 | Heuristic | 0 |
| 10 ≤ n < 30 | Bootstrap | 1 |
| n ≥ 30 AND ECE < 0.15 | Conformal | 2 |
| n ≥ 30 AND ECE ≥ 0.15 | Heuristic (quality regression) | 0 |

---

## 6. Ensemble p_mean Patching

`patch_ensemble_p_from_oracle()` (oracle/mod.rs lines 102–126) updates the in-memory
calibration state with an empirically derived baseline competence.

**Activation:** only when `n_observations ≥ 10`.

**Steps:**

1. Clamp `pass_rate` to `[0.5, 1.0]`.
2. Derive `cg_mean` from the existing calibration: `cg_mean = (1.0 - rho_mean).clamp(EPSILON, 1.0)`. Fallback value: `0.3`.
3. Call `EnsembleCalibration::from_measured_p(clamped_pass_rate, cg_mean, max_ensemble_size)`.
4. Publish `OracleCalibrationPatchedEvent` on success.

---

## 7. Oracle Gate

The oracle gate (`oracle_gate.rs`) is an optional pre-delivery validation step that runs
a synchronous NATS request/reply exchange before the output is returned to the client.

Default: **disabled** (`oracle_gate.enabled = false`).

### OracleGateConfig fields

| Field | Default | Description |
|-------|---------|-------------|
| `enabled` | `false` | Opt-in; disabled by default |
| `subject` | `"h2ai.oracle.gate"` | NATS request/reply subject |
| `timeout_secs` | `30` | Seconds to wait for a verdict |
| `on_timeout` | `"pass"` | Action when oracle does not respond in time |
| `on_fail` | `"evict"` | Action when oracle rejects the output |
| `min_confidence` | `0.7` | Minimum confidence to accept a pass verdict |
| `clarification_templates` | `[]` | Templates for clarification requests |

### OracleGateResultEvent fields

| Field | Type |
|-------|------|
| `task_id` | `String` |
| `gate_passed` | `bool` |
| `confidence` | `f64` |
| `summary` | `String` |
| `checked_proposals` | `u32` |
| `passed_proposals` | `u32` |
| `timestamp` | `DateTime<Utc>` |

---

## 8. Events

### OracleResultEvent

| Field | Type | Notes |
|-------|------|-------|
| `task_id` | `TaskId` | |
| `q_confidence` | `f64` | |
| `n_used` | `u32` | |
| `passed` | `bool` | |
| `score` | `f64` | |
| `residual` | `f64` | |
| `domain` | `OracleDomain` | |
| `duration_ms` | `u64` | |
| `timestamp_ms` | `u64` | |
| `verdict` | `Option<OracleVerdict>` | `#[serde(default, skip_serializing_if = ...)]` |
| `tenant_id` | `TenantId` | `#[serde(default)]` |

### OracleCalibrationPatchedEvent

Published to confirm a successful `patch_ensemble_p_from_oracle()` call.

| Field | Type |
|-------|------|
| `task_id` | `TaskId` |
| `oracle_pass_rate` | `f64` |
| `n_observations` | `usize` |
| `p_mean_before` | `f64` |
| `p_mean_after` | `f64` |
| `rho_mean` | `f64` |
| `timestamp` | `DateTime<Utc>` |

### CalibrationDriftWarning

| Field | Type |
|-------|------|
| `n_observations` | `usize` |
| `ece` | `f64` |
| `timestamp_ms` | `u64` |

### OracleSuspectEvent

| Field | Type |
|-------|------|
| `pass_rate` | `f64` |
| `n_observations` | `usize` |
| `reason` | `String` |
| `timestamp_ms` | `u64` |

---

## 9. Health Alerts

| Alert | Condition | NATS subject |
|-------|-----------|--------------|
| ECE drift | n ≥ 30 AND `ece > oracle_ece_alert_threshold` (0.15) | `h2ai.oracle.calibration_drift` |
| Low pass rate | n ≥ 30 AND `pass_rate < oracle_pass_rate_floor` (0.30) | `h2ai.oracle.suspect` |

Both alerts publish their respective event structs (see §8) to the listed subjects.

---

## 10. Prometheus Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `h2ai_oracle_ece_gauge` | gauge | Current ECE |
| `h2ai_oracle_n_observations_total` | gauge | Rolling observation count |
| `h2ai_oracle_coverage_rate` | gauge | Fraction of tasks carrying an `OracleSpec` |
| `h2ai_oracle_pass_rate` | gauge | Rolling oracle pass rate |
| `h2ai_oracle_residual_p90` | gauge | P90 of calibration residuals |
| `h2ai_calibration_basis` | gauge | 0=Heuristic, 1=Bootstrap, 2=Conformal |
| `h2ai_oracle_tasks_total` | counter | Total successfully resolved tasks |
| `h2ai_oracle_tasks_with_spec_total` | counter | Tasks that carried an `OracleSpec` |
| `h2ai_calibration_source{source}` | gauge | Active calibration source (1 = active) |

`h2ai_calibration_source` labels: `measured`, `partial_fit`, `synthetic_priors`.
