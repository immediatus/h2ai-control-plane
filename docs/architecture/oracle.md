# Oracle System

The oracle is the ground-truth signal that tells the engine whether a winning output is actually correct.
After `MergeEngine` selects a winner it emits an `OraclePendingEvent`; `OracleWorker` calls `OracleClient.evaluate`,
which POSTs the output to an external HTTP oracle service, then publishes an `OracleResultEvent`.
In the background, `OracleAccumulator` consumes that stream and continuously re-calibrates the ensemble.

The control plane is oracle-agnostic: it has no knowledge of schema validation, Z3, pytest, human ratings,
or any other evaluation strategy. All of that lives in the external oracle service.

---

## OracleSpec

Per-task oracle config in `task.json`:

```json
"oracle_spec": {
  "runner_uri": "http://oracle-service:9090/evaluate",
  "timeout_ms": 5000,
  "domain": "rtb"
}
```

- `runner_uri` — HTTP URL of the external oracle service
- `timeout_ms` — milliseconds before the HTTP call is abandoned (failure → `passed=false, score=0.0`)
- `domain` — domain tag forwarded to the oracle service and stored in calibration observations

---

## HTTP Protocol Contract

**Request (POST `<runner_uri>`):**

```json
{
  "task_id": "t-abc123",
  "output": "<winning output text>",
  "domain": "rtb"
}
```

**Response (2xx):**

```json
{
  "passed": true,
  "score": 0.95,
  "details": { "...any JSON..." }
}
```

- `passed` — bool, required
- `score` — f64 in [0.0, 1.0], required
- `details` — any JSON object, stored as-is, not interpreted by the control plane

On any error (timeout, network failure, non-2xx, bad JSON): `passed=false, score=0.0, details={"error":"<reason>"}`.

The oracle service is fully responsible for evaluation strategy, internal config, and response shape of `details`.

---

## Multi-Oracle FUSE

A single task can be evaluated by multiple oracle services simultaneously. When `oracle_specs: Vec<OracleSpec>` is non-empty on `OraclePendingEvent`, `OracleWorker` runs the primary `oracle_spec` and all additional `oracle_specs` in sequence, then aggregates via **worst-of-family reduction** (`fuse_reduce_by_family` in `oracle_worker.rs`).

### Oracle families

`OracleFamily` partitions oracle types by their correlated failure mode:

| Family | OracleDomain | Typical examples |
|---|---|---|
| `Syntactic` | `Code` | JSON Schema validator, Z3 symbolic, pytest runner |
| `Semantic` | `Factual`, `Reasoning`, `Unknown` | LLM judge, factual QA verifier, reference-answer matcher |
| `Human` | `Human` | Human rating gateway |

Oracles in the same family share failure modes — a malformed JSON output will simultaneously fail a JSON Schema validator *and* a Z3 constraint that expects a valid document. Counting both failures as independent FUSE votes would double-penalise a single root cause.

### FUSE reduction algorithm

```
fuse_reduce_by_family(verdicts: [(OracleFamily, bool, f64)]) → (bool, f64)

1. Group verdicts by family.
2. Within each family: take min(score)  — worst-case within a correlated group.
3. Across families: take mean(family_min) — independent signals aggregate equally.
4. pass ← final_score ≥ 0.5
```

An empty input returns `(false, 0.0)`. A single oracle follows the single-oracle path (backward-compatible — no FUSE overhead).

### Configuration

```json
"oracle_spec": {
  "runner_uri": "http://schema-oracle:9090/evaluate",
  "timeout_ms": 5000,
  "domain": "rtb"
},
"oracle_specs": [
  {
    "runner_uri": "http://semantic-oracle:9091/evaluate",
    "timeout_ms": 3000,
    "domain": "reasoning"
  }
]
```

`oracle_specs` defaults to `[]` (single-oracle path) — all existing deployments without this field are unaffected.

---

## OracleWorker

Thin NATS→HTTP bridge:

1. Subscribe to `h2ai.oracle.*.pending`
2. Deserialize `OraclePendingEvent`
3. Call `OracleClient.evaluate(spec, task_id, output)` — primary oracle
4. If `oracle_specs` non-empty: call each additional spec; reduce all verdicts via `fuse_reduce_by_family`
5. Build `OracleResultEvent` from aggregated `(passed, score)`
6. Publish to `h2ai.oracle.results` and reply subject

---

## Calibration Loop

After every oracle result, `OracleAccumulator.handle_result` runs the full calibration pipeline:

### 1. Observation Window

Observations are stored in NATS KV. New results are appended and the window is capped
at `oracle_window_size` (FIFO, oldest dropped first).

### 2. Calibration Basis

Determined by `determine_calibration_basis` using CLT-grounded thresholds:

| n observations | ECE | Basis |
|---|---|---|
| n < 10 | any | 0 — Heuristic (insufficient data) |
| 10 ≤ n < 30 | any | 1 — Bootstrap (coarse intervals) |
| n ≥ 30 | ECE < 0.15 | 2 — Conformal (valid coverage) |
| n ≥ 30 | ECE ≥ 0.15 | 0 — Heuristic (quality regression) |

These thresholds are **mathematical constants, not config** (CLT: Lehmann & Romano 2005;
bootstrap minimum: DiCiccio & Efron 1996).

### 3. ECE Metric

```
ECE = (1/n) × Σ |q_confidence_i − y_oracle_i|
```

`q_confidence` is the ensemble's predicted pass probability; `y_oracle` is the binary oracle outcome.
Lower ECE = better calibration.

### 4. Ensemble Patch (n ≥ 10)

When n ≥ 10, `patch_ensemble_p_from_oracle` updates `EnsembleCalibration.p_mean` to the
measured oracle pass rate (clamped to `[0.5, 1.0]`). This is innovation INNOVATION-1 (GAP-B2):
oracle ground truth feeds back into ensemble sizing.

Emits `OracleCalibrationPatchedEvent` on the NATS task stream.

### 5. Health Alerts (n ≥ 30)

| Condition | NATS subject | Log level |
|---|---|---|
| ECE > `oracle_ece_alert_threshold` | `h2ai.oracle.calibration_drift` | WARN |
| pass_rate < `oracle_pass_rate_floor` | `h2ai.oracle.suspect` | WARN |

---

## Prometheus Metrics

`OracleAccumulator` updates these gauges after every result:

| Metric | Description |
|---|---|
| `oracle_ece` | Current ECE over the window |
| `oracle_n_observations` | Window size |
| `oracle_pass_rate` | Fraction of passing observations |
| `oracle_residual_p90` | P90 of residuals (Angelopoulos-Bates Theorem 1) |
| `oracle_calibration_basis` | 0=Heuristic, 1=Bootstrap, 2=Conformal |

---

## Configuration

All oracle config lives in `h2ai.toml` under `[oracle]` (or in `reference.toml`):

```toml
[oracle]
oracle_window_size = 100           # FIFO cap on observation window
oracle_ece_alert_threshold = 0.20  # ECE WARN threshold (n≥30)
oracle_pass_rate_floor = 0.40      # pass_rate WARN threshold (n≥30)
```

---

## Testing

### Unit Tests

```bash
cargo test --package h2ai-api -- oracle
```

### E2E Tests

The `dsp-onboarding` scenario exercises the oracle path end-to-end.
Its `task.json` specifies `runner_uri` pointing to an oracle sidecar service.

```bash
python3 tests/e2e/replay.py dsp-onboarding
```

### Debug Logging

```bash
RUST_LOG=debug python3 tests/e2e/replay.py dsp-onboarding
```

Key log patterns:

| Pattern | Meaning |
|---|---|
| `oracle result processed` | OracleAccumulator processed a result; shows ece, basis, n_obs |
| `oracle: patching ensemble p_mean` | Calibration feedback applied (n≥10) |
| `CalibrationDrift: ECE exceeded` | ECE alert fired (n≥30) |
| `OracleSuspect: pass rate below floor` | Pass rate alert fired (n≥30) |
