# E2E Scenario Tests

End-to-end tests that submit a task to a running H2AI server, capture the full SSE event stream, and check scored assertions. Each scenario exercises a distinct combination of pipeline features.

## Structure

```
tests/e2e/
  scenarios/
    <name>/
      task.json    — task description, constraints, oracle assertions
      h2ai.toml    — server config: exactly which features are ON and why
  constraints/     — constraint corpus YAML files referenced by corpus_path
  replay.py        — run a scenario end-to-end
  analyze_events.py — summarise results from events.jsonl
  results/         — run artifacts (gitignored)
```

## Prerequisites

```bash
cargo build --release    # build the server binary
# Ensure Ollama is running on port 8080 and NATS on port 4222:
#   ollama serve                        (or your preferred LLM server)
#   docker run -p 4222:4222 nats:latest
```

`replay.py` starts and stops the server automatically per scenario.

## Run

```bash
# Single scenario
python3 tests/e2e/replay.py compliance-lite

# Sequential suite (all scenarios)
python3 tests/e2e/replay.py \
  compliance-lite \
  complexity-routing \
  terminal-bft-adversarial \
  tau-bench-constraint-drift \
  hle-expert-consensus \
  multi-constraint-billing

# Summarise latest results for all scenarios
python3 tests/e2e/analyze_events.py --all

# Summarise a specific run
python3 tests/e2e/analyze_events.py complexity-routing --run latest

# View constraint binary check verdicts
python3 tests/e2e/analyze_events.py multi-constraint-billing --checks

# Debug trace (full oracle + grounding signals)
RUST_LOG=debug python3 tests/e2e/replay.py terminal-bft-adversarial
```

---

## Scenarios

Six scenarios cover independent axes of the H2AI pipeline. Each `h2ai.toml` is self-documenting — read the header block to see what is ON, what is OFF, and why.

### 1. `compliance-lite` — Baseline fast path

**Task:** Draft a SaaS contract addendum satisfying one legal constraint (liability cap, term-limited license, GDPR processor terms).

**Expected result:** PASS

**Key features:**
- `tiered_exit = true` — exits after the first explorer whose score meets the bar (N=1–3). Validates the fast-path USL ensemble sizing. Expect fewer than 5 adapters to run.
- `thinking_loop = false`, `adapter_enable_thinking = false` — no chain-of-thought; baseline comparison for latency vs quality tradeoff.
- `gap_i1 = false` — single well-specified constraint; no grounding research needed.
- `safety = "development"` — relaxed safety gates.

**Why it matters:** Ensures the simplest path through the pipeline works correctly and quickly. A regression here means the core orchestration loop is broken.

---

### 2. `complexity-routing` — Thinking loop + complexity ceiling detection

**Task:** Prove MVC is NP-hard via a polynomial-time 3-SAT→MVC reduction; prove the 2-approximation guarantee via LP relaxation.

**Expected result:** PASS on capable models (Qwen3/DeepSeek-R1); VerificationExhaustion on weaker models.

**Key features:**
- `thinking_loop (2 iters)` — iterative refinement: sketch → formal proof. Expect ThinkingIterationStarted/Completed events.
- `complexity_routing.verifier_decomposition_enabled = true` — **unique to this scenario**: tasks scoring ≥3 get a BEYOND_BUDGET check injected into the verifier suite. Validates NP-hardness classification.
- `complexity_routing.agent_dropout` — reduces ensemble size when all explorers converge on the same wrong answer (N_eff collapses).
- `safety = "development"`, `shadow_auditor strict=false` — advisory auditing, non-blocking.
- `gap_i1 = true` — grounding validates cited complexity-theory facts.

**Why it matters:** Tests the full complexity-routing stack (ComplexityProbe → BEYOND_BUDGET injection → ceiling detection → grafting). The only scenario that uses `verifier_decomposition_enabled`.

---

### 3. `terminal-bft-adversarial` — Production safety + BFT pruning

**Task:** Design a zero-downtime PostgreSQL column rename on a 12M-row table under continuous write load. Some proposals suggest unsafe DDL (direct ALTER — acquires full table lock, causes downtime).

**Expected result:** PASS — unsafe proposals are Krum-pruned or shadow_auditor-vetoed before reaching the verifier.

**Key features:**
- `safety = "production"` — Krum f=1 aggregation (discards the 1 most-divergent proposal). Requires ≥5 adapters. Validates BFT-robust aggregation.
- `shadow_auditor strict=true` — AND-gate veto: an auditor veto blocks the proposal regardless of verifier score. Expect AuditorVetoEvent for unsafe DDL proposals.
- `thinking_loop (3 iters)` — migration plan → rollback procedure → concurrency safety analysis.
- `gap_i1 = true` — validates PostgreSQL DDL lock semantics.

**Why it matters:** Only scenario that tests `strict=true` shadow auditor veto (AND-gate). The production safety machinery (Krum + binding auditor) is exercised here.

---

### 4. `tau-bench-constraint-drift` — Multi-constraint under cache drift

**Task:** Immediately reduce tenant 'acme-corp' rate limit from 10,000→2,000 req/min. Redis cache TTL=60s causes drift — cached values lag behind the database for up to 60 seconds. Three constraints must hold simultaneously: TAU-1 (data isolation), TAU-2 (atomic transitions), CONSTRAINT-008 (no distributed locks).

**Expected result:** PASS — a cache-invalidation + pub-sub approach satisfies all three. Simpler approaches that propose Redis SETNX locks fail CONSTRAINT-008.

**Key features:**
- `safety = "production"` — Krum pruning. Validates that lock-based proposals are pruned before scoring.
- `thinking_loop (3 iters)` — drift window analysis → invalidation strategy → atomicity proof.
- 3 simultaneous constraints: unlike single-constraint scenarios, the verifier runs all three binary checks.
- `gap_i1 = true` — validates Redis TTL semantics and pub-sub delivery guarantees.

**Why it matters:** Tests multi-constraint enforcement under real-world drift conditions. Shows how Krum handles proposals that satisfy 2/3 constraints but violate CONSTRAINT-008.

---

### 5. `hle-expert-consensus` — Capability ceiling (expected failure)

**Task:** Design a BFT consensus protocol for n nodes with f Byzantine (equivocating) nodes. Prove safety under equivocation, liveness bound, and adversarial correctness — a graduate-level multi-sub-claim formal proof.

**Expected result:** VerificationExhaustion — current local models cannot produce a complete BFT correctness proof. This scenario validates graceful failure: the system should emit `TaskFailed(VerificationExhaustion)`, not hallucinate a passing answer.

**Key features:**
- `thinking_loop (5 iters, highest in suite)` — maximum attempts before exhaustion. Validates the convergence detector: all 5 iterations fire, coverage plateaus at ~0.40–0.60, then NoConvergence is emitted.
- `complexity_routing.intra_retry min_retry_count_for_detection=1` — detect ceiling on the FIRST retry (faster fail-fast vs. the default 2 retries).
- `safety = "production"` — Krum. On hard BFT proofs, proposals maximally diverge — Krum's effect is most visible here.

**Why it matters:** Validates the failure path. Without this scenario, a regression that makes all tasks "pass" (by lowering the verifier bar) would go undetected. Correct behavior: fail gracefully, not silently pass.

---

### 6. `multi-constraint-billing` — Multi-constraint enforcement + CSPR

**Task:** Design the quota change mechanism for a multi-tenant SaaS billing service satisfying three constraints simultaneously: CONSTRAINT-004 (idempotency — retried requests within 30s must not double-apply), CONSTRAINT-005 (immutable audit log), CONSTRAINT-008 (no distributed locks).

**Expected result:** PASS

**Key features:**
- `cspr = true` — Constraint-Sensitive Proposal Ranking: proposals are re-ranked by constraint coverage before aggregation. Proposals that use locks (violating CONSTRAINT-008) rank lower even if their overall quality score is high. Expect ConstraintRankingAppliedEvent.
- `ambiguity_detection = true` — detects semantic conflicts between constraints before generation (e.g. "MUST record every retry" vs "MUST NOT double-apply"). If the score exceeds 0.6, AmbiguityFlaggedEvent fires and a disambiguation prompt is injected.
- `knowledge_domain_scoping = true` — restricts model reasoning to the billing domain; suppresses irrelevant suggestions.
- `safety = "development"` — no BFT pruning; isolates constraint-enforcement logic from production safety machinery.
- `max_autonomic_retries = 6` — extra retries because satisfying 3 constraints simultaneously requires more iterations than single-constraint tasks.

**Why it matters:** Only scenario using `cspr` and `ambiguity_detection`. Tests the constraint-interaction pipeline that the production safety scenarios (terminal-bft, tau-bench) don't exercise.

---

## Feature Matrix

| Feature | compliance-lite | complexity-routing | terminal-bft | tau-bench | hle-consensus | multi-constraint-billing |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `safety` profile | develop | develop† | **production** | **production** | **production** | develop |
| Krum BFT pruning | — | — | **✓ f=1** | **✓ f=1** | **✓ f=1** | — |
| shadow_auditor strict | — | advisory | **✓ AND-gate** | — | — | — |
| `thinking_loop` | **OFF** | 2 iter | 3 iter | 3 iter | **5 iter** | 3 iter |
| `tiered_exit` | **✓ N≤3** | — | — | — | — | — |
| `gap_i1` grounding | — | ✓ | ✓ | ✓ | ✓ | ✓ |
| `complexity_routing` | — | **✓ BEYOND** | ✓ | ✓ | ✓ | ✓ |
| `cspr` ranking | — | — | — | — | — | **✓** |
| `ambiguity_detection` | — | — | — | — | — | **✓** |
| `knowledge_domain_scoping` | — | — | — | — | — | **✓** |
| N adapters | 5 | 5 | **7** | **7** | **7** | 5 |
| Constraints | 1 | 2 | 2 | 3 | 2 | 3 |
| Expected outcome | PASS | PASS† | PASS | PASS | **FAIL††** | PASS |

† complexity-routing `develop` profile has shadow_auditor advisory (strict=false) — non-blocking audit.
† complexity-routing: PASS on Qwen3/R1, may VerificationExhaust on weaker models.
†† hle-consensus: expected VerificationExhaustion on all current local models.

---

## Output Artifacts

```
tests/e2e/results/<scenario>/<timestamp>/
  events.jsonl     — raw SSE event stream (all pipeline events)
  output.txt       — merged text from MergeResolvedEvent
  summary.json     — j_eff, verification scores, constraint checks, assertion results
  _server_logs/    — h2ai-control-plane server logs (INFO by default)
```

## Oracle Types

The oracle in `task.json` evaluates whether the winning output is correct:

| Prefix | Check |
|---|---|
| `schema:` | JSON Schema validation |
| `multiple-choice` / `free-form` | LLM semantic judge against a reference answer |
| `z3:` | Z3 SMT symbolic verification |
| `test-suite:` | Forks a local binary, passes winning output as temp file |
| `human:<channel>` | Human rater via NATS gateway |

See `docs/architecture/oracle.md` for details.

## Adding a Scenario

1. Create `tests/e2e/scenarios/<name>/`:
   - `task.json` — task description, constraints, oracle assertions
   - `h2ai.toml` — server config (copy a scenario with similar features; document what is ON and why)
2. Add any new constraint YAML files to `tests/e2e/constraints/`.
3. Add the scenario path to the config tests in `crates/h2ai-config/tests/config_test.rs`.
4. Run `python3 tests/e2e/replay.py <name>` to verify.
