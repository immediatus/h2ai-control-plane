# E2E Tests

End-to-end tests that submit scenarios to a running H2AI server, capture the full SSE event stream, check assertions, and save artifacts for regression comparison.

## Structure

```
tests/e2e/
  scenarios/
    benchmark/<name>/
      task.json    — scenario definition, constraints, expected assertions
      h2ai.toml    — server config for this scenario
  constraints/     — constraint corpus YAML files (loaded by h2ai.toml corpus_path)
  replay.py        — test runner
  client.py        — HTTP client (localhost:8080)
  scripts/
    analyze_failed_run.py  — post-run failure analysis helper
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
# All benchmark scenarios
python3 tests/e2e/replay.py \
  benchmark/terminal-bft-adversarial \
  benchmark/osworld-long-horizon \
  benchmark/tau-bench-constraint-drift \
  benchmark/hle-expert-consensus

# Single scenario
python3 tests/e2e/replay.py benchmark/terminal-bft-adversarial

# List available scenarios
python3 tests/e2e/replay.py --list

# Baseline: direct LLM without H2AI pipeline
python3 tests/e2e/replay.py --baseline benchmark/terminal-bft-adversarial

# Compare H2AI vs baseline delta
python3 tests/e2e/replay.py --compare benchmark/hle-expert-consensus

# Run k times to get pass^k reliability estimate
python3 tests/e2e/replay.py --trials 3 benchmark/terminal-bft-adversarial
```

## Benchmark Scenarios

| Scenario | Benchmark | Constraints | Domain |
|---|---|---|---|
| `benchmark/terminal-bft-adversarial` | Terminal-Bench | CONSTRAINT-BFT-1, BFT-2 | Production safety, expand-contract migrations |
| `benchmark/osworld-long-horizon` | OSWorld-Verified | CONSTRAINT-OSW-1, 008, 004 | Pipeline backpressure, lock-free aggregation |
| `benchmark/tau-bench-constraint-drift` | τ²-bench | CONSTRAINT-TAU-1, TAU-2, 008 | Multi-tenant isolation, atomic quota changes |
| `benchmark/hle-expert-consensus` | HLE | CONSTRAINT-HLE-1, HLE-2 | BFT consensus safety and liveness proofs |

## Adding a Benchmark Scenario

1. Create `tests/e2e/scenarios/benchmark/<name>/`:
   - `task.json` — must include `_benchmark` (benchmark name string) plus: `description`, `pareto_weights`, `explorers`, `constraints`, `context`, `_timeout_s`, `_expected`
   - `h2ai.toml` — server config (copy from an existing benchmark scenario)

2. Add constraint YAML files to `tests/e2e/constraints/` and update the wiki under `constraints/wiki/`.

3. Run `python3 tests/e2e/replay.py benchmark/<name>` to verify.

## Output

```
tests/e2e/results/<scenario>/<timestamp>/
  events.jsonl          — raw SSE event stream
  output.txt            — merged output from MergeResolved event
  summary.json          — benchmark, git_sha, j_eff, verification scores, SRANI signals, assertion results
  _server_logs/h2ai.log — server logs (always present; INFO level by default)
```

`summary.json` includes `benchmark` (which benchmark this scenario maps to) and `git_sha` (short commit SHA) for tracking improvements across commits.

## Debug Logging

Server logs are written at INFO level by default. To capture full debug traces
(oracle calibration, bandit updates, SRANI signals):

```bash
RUST_LOG=debug python3 tests/e2e/replay.py benchmark/terminal-bft-adversarial
```

## Failure Analysis

```bash
python3 tests/e2e/scripts/analyze_failed_run.py tests/e2e/results/benchmark/terminal-bft-adversarial/<timestamp>
```

Prints: assertion failures, suspicious numeric values in output, constraint binary checks, suggestions.

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

See `docs/architecture/oracle.md` for the full reference.
