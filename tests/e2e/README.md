# E2E Tests

End-to-end tests that submit scenarios to a running H2AI server, capture the full SSE event stream, check assertions, and save artifacts for regression comparison.

## Structure

```
tests/e2e/
  scenarios/
    <name>/
      task.json    — scenario definition, constraints, expected assertions
      h2ai.toml    — server config for this scenario (gitignored)
  constraints/     — constraint corpus YAML files
  replay.py        — test runner
  client.py        — HTTP client (localhost:8080)
  results/         — run artifacts (gitignored)
```

## Prerequisites

Build the binary and ensure NATS is running (NATS is already up inside the devcontainer):

```bash
cargo build --release
```

`replay.py` starts and stops the server automatically for each scenario with the correct config.

## Run

```bash
# All scenarios
python3 tests/e2e/replay.py

# Single scenario
python3 tests/e2e/replay.py dsp-onboarding

# Multiple scenarios
python3 tests/e2e/replay.py dsp-onboarding ml-feature-latency

# List available scenarios
python3 tests/e2e/replay.py --list

# Baseline: direct LLM without H2AI pipeline
python3 tests/e2e/replay.py --baseline dsp-onboarding
```

Multi-family pruning assertions (requires auditor adapter configured):
```bash
H2AI_E2E_MULTIFAMILY=1 python3 tests/e2e/replay.py
```

## Scenarios

| Scenario | Constraints |
|---|---|
| `budget-enforcement-crash-recovery` | CONSTRAINT-004, CONSTRAINT-005 |
| `dsp-onboarding` | CONSTRAINT-003 |
| `ml-feature-latency` | CONSTRAINT-001, CONSTRAINT-006, CONSTRAINT-007 |

## Adding a Scenario

1. Create `tests/e2e/scenarios/<name>/`:
   - `task.json` — `description`, `config`, `pareto_weights`, `explorers`, `constraints`, `context`, `_timeout_s`, `_expected`
   - `h2ai.toml` — server config (copy from an existing scenario, adjust as needed)

2. Run `python3 tests/e2e/replay.py <name>` to verify.

## Output

```
tests/e2e/results/<scenario>/<timestamp>/
  events.jsonl          — raw SSE event stream
  output.txt            — merged output from MergeResolved event
  summary.json          — j_eff, verification scores, SRANI signals, assertion results
  _server_logs/h2ai.log — server logs (always present; INFO level by default)
```

Compare `summary.json` across runs to track regressions or improvements.

## Debug Logging

Server logs are written at INFO level by default. To capture full debug traces
(oracle calibration, bandit updates, SRANI signals):

```bash
RUST_LOG=debug python3 tests/e2e/replay.py dsp-onboarding
```

Logs are saved to `tests/e2e/results/<scenario>/<timestamp>/_server_logs/h2ai.log`
regardless of `RUST_LOG`.

## Oracle

The oracle evaluates whether the winning output is correct. It is configured per task
in `oracle_spec.test_suite`. Six oracle types are supported:

| Prefix | What it checks |
|---|---|
| `schema:` | JSON Schema validation |
| `multiple-choice` / `free-form` | LlmJudge semantic match against a reference answer |
| `z3:` | Z3 SMT symbolic verification |
| `human:<channel>` | Human rater via NATS gateway |
| `staged-code:` | Staged code test harness |
| `test-suite:` or bare name | Forks a local binary; passes winning output as a temp file |

See `docs/architecture/oracle.md` for the full reference including calibration loop,
FUSE semantic dominance, and Prometheus metrics.
