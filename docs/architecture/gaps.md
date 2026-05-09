# H2AI Gaps — Research and Engineering Agenda

This document is the actionable companion to [`research-state.md`](research-state.md). Where `research-state.md` names limitations as architectural properties, this document converts each gap into a falsifiable question with a concrete research or engineering path. Each gap is a candidate session topic; groups are arranged so that related gaps can share a single instrumentation harness or task set.

Gaps are not listed by severity alone — they are listed by what they can falsify. The ones that can kill the core thesis are first.

---

## Foundational Framing — Every Problem Is a Team Epistemology Problem

Before the individual gaps, the meta-framing that connects them.

Any non-trivial problem is a **team knowledge acquisition problem**: the team must discover what is true about the problem domain, resolve contradictions between what different team members believe, and produce a justified output that survives contact with reality. The solution is not a pipeline — it is a **graph of thinking, decisions, and executions**, with loops wherever knowledge needs to be refined.

### The knowledge graph

```
Nodes  : beliefs  — {claim, evidence, assumptions, scope, confidence}
Edges  : support  (B strengthens A)
         contradiction  (B and A cannot both be true)
         derivation     (B follows from A by rule R)
         grounding      (oracle/test connects A to external reality)

Loops  :
  coherence   → resolve contradictions between beliefs until none remain
  coverage    → ensure every required knowledge dimension has a justified belief
  grounding   → connect load-bearing beliefs to external reality
  revision    → update beliefs when new evidence contradicts old ones
```

The loops do not have fixed counts. Each runs until its epistemic stopping criterion is met — not until a retry budget is exhausted.

### How H2AI phases map to epistemic operations

| Phase | Epistemic operation |
|---|---|
| Task decomposition (GAP-A5) | Epistemic division of labor — assign knowledge responsibilities |
| TAO inner loop | Hypothesis exploration — form initial beliefs |
| Phase 2 topology | Determine how many independent knowledge contributors the budget allows |
| Phase 3.5 verification | Coherence test — do beliefs satisfy the constraint axioms? |
| Phase 4 audit | Final coherence gate before a belief is accepted as output |
| Phase 5a synthesis | Belief integration — construct the most coherent view across contributors |
| MAPE-K retry | Belief revision — update under new evidence from failed coherence tests |
| Phase 6 oracle (GAP-E1) | **Grounding** — connect beliefs to external reality |
| Calibration | Update meta-beliefs about team epistemic capabilities |

### The epistemological traditions each gap violates

| Gap | Epistemic violation |
|---|---|
| GAP-A5: unmotivated committee composition | No epistemic division of labor — roles do not map to knowledge domains |
| GAP-A4: verification circularity | Epistemic independence broken — verifier uses same evidence as explorer |
| GAP-C1: correlated hallucination | Epistemic echo chamber — all agents share the same false prior |
| GAP-C2: single auditor | Single point of epistemic judgment — no adversarial check on the judge |
| GAP-A3: ρ proxy chain | Measuring belief correlation by proxy, not directly |
| GAP-E1: no oracle | Epistemic closure — belief system is internally coherent but ungrounded |
| GAP-D4: synthetic calibration unlabeled | Epistemic meta-uncertainty — the system does not know what it doesn't know about itself |
| GAP-B3: attribution without oracle | Confidence without ground truth — cannot distinguish "confident and correct" from "confident and wrong" |

### The Pareto point as epistemic optimum

The committee's Pareto weights `(containment, diversity, throughput)` are the task-specific encoding of the epistemological tradeoff:

- **Containment** → coverage completeness: does every knowledge domain required by the task have at least one justified belief?
- **Diversity** → epistemic independence: are the contributors' belief-formation processes sufficiently decorrelated that their errors are independent?
- **Throughput** → epistemic efficiency: minimum knowledge contributors sufficient to achieve coverage and independence

The Pareto-optimal committee is the one where you cannot increase coverage or independence without exceeding the USL budget (N_max). This is the same math the planner already applies to topology selection — it should be applied first, upstream, to committee composition.

### Stopping criteria for the loops

| Loop | Current criterion | Epistemically principled criterion | Gap |
|---|---|---|---|
| TAO inner | `agent_max_tool_iterations` (budget) | No productive hypothesis extensions remain | Budget is proxy for epistemic exhaustion |
| MAPE-K retry | Proposals satisfy verifier threshold OR `max_autonomic_retries` exhausted | Coherent closure: no active constraint violated, no domain uncovered | Quality threshold is rubric-coherent, not oracle-grounded |
| Calibration | Manual re-run | Confidence intervals narrow enough for decision quality required | GAP-D1: calibration measures latency, not epistemic work |
| Oracle grounding | Not wired | All load-bearing beliefs grounded in at least one oracle test | GAP-E1: no oracle integration |

**The stopping criteria framing:** Stopping when "proposals achieve acceptable quality" is the correct epistemic criterion. The current MAPE-K loop IS quality-gated in form — it stops when proposals pass the verification threshold. The problem is one level deeper: "acceptable quality" is currently defined as "verifier-coherent" (satisfies the rubric the explorer already saw), not "oracle-coherent" (correct in external reality). The budget `max_autonomic_retries` is the best available proxy for quality until the oracle grounding loop (GAP-E1) closes. It becomes principled when calibrated to the empirical distribution of "iterations to coherent closure" — which requires the oracle to define what closure means against ground truth.

The budget-based stopping criteria are not mechanical — they are quality proxies operating against an ungrounded quality signal. They become fully principled when: (a) the coherence criterion is added (GAP-A4 Stage 3), and (b) the oracle loop calibrates "acceptable" against external reality (GAP-E1).

---

## Brainstorm Group A — Core Thesis Validity

---

### ~~GAP-A0: Verifier Precision — Local LLM Gives Holistic 1.0 Scores~~ ✅ CLOSED (code)

**Status: CLOSED (code complete) — evaluator model upgrade pending.**

The local 11B model returned `score=1.0, reason=""` for every proposal regardless of content,
defeating the entire MAPE-K retry loop. Two complementary fixes were implemented:

**1. Criteria decomposition (Anchored CoT scoring).**
`criteria.checks: Vec<String>` added to the YAML constraint schema. When checks are present,
the rubric builder appends a numbered binary checklist and the instruction:
`Score = count(PRESENT) / N. Ignore the Pass/Partial/Fail guide above — compute arithmetically.`
All 7 ads-platform constraints were updated with 4 binary yes/no checks each.
The evaluator system prompt (`EVALUATOR_SYSTEM_PROMPT` in `h2ai-types`) was extended with an
Anchored CoT scoring path triggered when "Binary compliance checks" appears in the rubric.

The math: with binary checks the score becomes a proper fraction (e.g., 3/4 = 0.75), crossing
the `hard` severity threshold of 0.8. A contraction ratio of 1 − (recall × fix_rate) ≈ 0.325
gives gap_k = 0.30 × 0.325^k → 9.9/10 after 3 retries, if the evaluator follows the format.

**2. Hint injection (Self-Refine / Reflexion).**
`retry_context: Option<String>` added to the MAPE-K engine state. When `RetryWithHints` fires,
`remediation_hints` from violated constraints are formatted into a constraint feedback block and
injected into the explorer's `system_context` for the next generation round. Anti-accumulation:
the hint block is always built from the immutable `system_context` base, not from the previous
`active_ctx`, so hints do not stack across retries.

**Residual gap.**
The local 11B model (4-bit quantized, llama.cpp) does not follow structured CoT output formats.
The verifier returns `{"score": 1.0, "reason": ""}` without performing the binary check steps,
so `RetryWithHints` never fires. The code path is correct and tested; it requires an evaluator
model with stronger instruction-following (≥13B, or a fine-tuned 7B). See
`results/evaluation-2026-05-08-v4.md` for the baseline vs. framework comparison that confirmed
this gap.

**Files changed.**
`crates/h2ai-constraints/src/yaml.rs`, `crates/h2ai-types/src/prompts.rs`,
`crates/h2ai-orchestrator/src/engine.rs`,
`docs/examples/ads-platform/constraints/*.yaml` (all 7),
tests in `crates/h2ai-constraints/tests/source_test.rs` and
`crates/h2ai-orchestrator/tests/engine_test.rs`.

---

These three gaps share a single instrumented task set. They answer the question: *does the fundamental approach work at all?* Run them together.

---

### ~~GAP-A1: Self-MoA vs. Multi-Family — Does Diversity Matter?~~ ✅ CLOSED

**Status: CLOSED** — Phase 1.5 routing implemented and armed (`shadow_mode = false`).
TCC parameters fitted from S2 dataset: `k_soft=1.0`, `k_type=1.0`, `k_cross=2.5`, `θ_tcc=2.0` (precision threshold), `θ_cov=2.5` (coverage threshold). Updated in `crates/h2ai-config/reference.toml`.
Multi-family adapter selection (highest-p_mean family) deferred to when a multi-family calibration pool is available; documented as `TODO(gap-a1-multi-family)` in `engine.rs`.

**Resolution summary.**
Task-geometry-adaptive routing (Phase 1.5) is live. Each task is routed to one of four quadrants based on TCC (task coverage complexity) and N_eff (pool diversity):

- **Precision tasks** (TCC < 2.0, pool diverse): within-family τ-spread, 2–3 slots
- **Coverage tasks** (TCC ≥ 2.5, pool diverse): cross-family committee, CJT logic
- **Complex tasks** (TCC ≥ 2.5, poor pool diversity): max ensemble, forced synthesis
- **Degenerate tasks** (both low): fail fast, no wasted compute

The S2 dataset pipeline (`scripts/s2_dataset/`) was built and run: 40 seeds selected from HumanEval, 30 constraint corpora generated via local LLM, 20 validated (structural TCC=3.75 for all). The B0 vs. B3 smoke experiment validated the routing infrastructure (B3 latency 4.5× B0, confirming τ-spread applies 3 calls per task).

**Deferred (not blocking).**
Multi-family highest-p_mean adapter selection and the full 2×2 cross-family vs. single-family experiment require a multi-model calibration pool not present in the current single-LLM deployment. The routing infrastructure is in place; this comparison is future work when additional adapter families are available.

---

### ~~GAP-A5: Committee Composition Is Semantically Unmotivated — N and Roles Must Come From Task Decomposition~~ ✅ CLOSED

**Status: CLOSED** — Epistemic decomposition implemented as Phase 0 (2026-05-09). See [design spec](../superpowers/specs/2026-05-09-epistemic-committee-design.md).

**Resolution summary.**
The committee was semantically empty: `diverse_defaults()` returned four slots with `role_frame = ""` and generic CoT styles. N came from USL or the operator manifest integer, not the problem.

**What changed:**

- `ExplorerSlotConfig` gained `focus_mandate: String` and `rejection_criteria: String` (both `#[serde(default)]`, backwards-compatible).
- `diverse_defaults()` **removed**. It became dead code once Phase 0 always runs.
- Engine context builder wires `[MANDATE]: {focus_mandate}` and `[FIND]: {rejection_criteria}` as preamble before each agent's system context when non-empty.
- `run_decomposition_agent()` in `crates/h2ai-orchestrator/src/decomposition.rs` — Path C: pre-dispatch LLM call to the auditor adapter (most capable, τ=0.1) producing `Vec<ExplorerSlotConfig>` with motivated `role_frame`, `cot_style`, `focus_mandate`, and `rejection_criteria`. Returns `Result<Vec<ExplorerSlotConfig>, DecompositionError>` — failure propagates as `TaskFailed`, no silent fallback.
- `corpus_fallback()` — exists as a public utility (groups `ConstraintDoc` by `domains`, one slot per domain). **Not wired as a production path.** Available for future recovery scenarios and testing.
- `prune_by_orthogonality()` — drops the least independent slot (highest mean cosine similarity to retained peers) when `len > N_max`. No padding.
- `tasks.rs` Phase 0: Path C **always runs** before `EngineInput` construction. Operator-supplied `slot_configs` (if any) are **appended** to the Path C result as additive context, then the combined set is re-pruned. They do not bypass decomposition.

**The flow after implementation:**

```
Phase 0 — always runs:
  run_decomposition_agent()  →  Vec<ExplorerSlotConfig> (Path C, τ=0.1)
  Err(DecompositionError)    →  TaskFailed event, task aborted (no fallback)
  append manifest.explorers.slot_configs (additive operator context)
  prune_by_orthogonality() if len > N_max
  set manifest.explorers.slot_configs for EngineInput
```

**Why Path A (operator bypass) was removed:** Operator-specified slots required the operator to pre-know the domain decomposition — doing Path C's job manually without the LLM's domain knowledge. The bypass violated the epistemic independence principle: if the operator specifies "security engineer, performance engineer" they have already made the decomposition decision, but without the task description's constraint context.

**Why Path B (corpus_fallback) is not the fallback:** If Path C fails (LLM adapter error, parse error), the task cannot produce a principled committee. Falling back to corpus-driven slots would silently continue with a lower-quality decomposition, masking the failure. The correct response is `TaskFailed` so the operator knows decomposition was not achievable.

**Remaining research questions (falsification conditions open):**
- Does Path C decomposition outperform fixed `diverse_defaults` on oracle pass rate with equal or fewer agent calls? Requires oracle harness (GAP-E1).
- Does the decomposition need to be visible to the verifier? A verifier that knows "slot 2 was the security-focused agent" could ask domain-specific questions.
- Should the decomposition update across retries — if security concerns were missed, add a security specialist in retry 2?

**Files changed.**
`crates/h2ai-types/src/manifest.rs` (`ExplorerSlotConfig` extended, `diverse_defaults()` removed),
`crates/h2ai-orchestrator/src/engine.rs` (context builder preamble injection, `diverse_defaults()` fallback removed),
`crates/h2ai-orchestrator/src/decomposition.rs` (new module: prompt, parser, pruner, `corpus_fallback`, `run_decomposition_agent`),
`crates/h2ai-orchestrator/src/lib.rs` (module registration),
`crates/h2ai-api/src/routes/tasks.rs` (Phase 0: always-on decomposition, operator slots appended post-Path-C).

---

### GAP-A4: Verification Circularity — Rubric-in-Context Creates Self-Reference

**Partial progress (2026-05-09).** Three of four planned fixes are implemented. One is observability-only (not yet a gate). A/B measurement is blocked by GAP-E1.

**What is now implemented:**

1. **Rubric separation (`include_rubric=false`).** `compiler::compile(manifest, corpus, include_rubric: bool)` — the production call in `engine.rs` passes `false`, which withholds all `LlmJudge` rubric text (and constraint IDs) from the explorer's `system_context`. The verifier retains the rubric via `ConstraintPredicate::LlmJudge`. The explorer must reason from domain expertise and the task description alone, not from rubric scaffolding. Vocabulary-presence constraints (term lists) are always included regardless of this flag. This directly addresses failure mode 1 (self-confirmation): the explorer can no longer score 1.0 by instruction-following alone.

2. **Adversarial verifier.** `ADVERSARIAL_EVALUATOR_SYSTEM_PROMPT` defined in `h2ai-types/src/prompts.rs` with hostile-reviewer framing: *"Your goal is NOT to check whether this proposal follows the rubric. Find the single most likely way this proposal fails silently."* Activated automatically in `tasks.rs` when any slot config carries non-empty `rejection_criteria` — which is always true when Path C decomposition runs.

3. **`CoherenceState` with two closure dimensions.** Computed per-wave inside the MAPE-K retry loop (after `all_pruned.extend()` and Phase 4.5 frontier event):
   - `uncovered_domains`: constraint domains where any pruned proposal had violations.
   - `active_contradictions: Vec<(ExplorerId, ExplorerId, String)>`: pairs of surviving proposals scoring on opposite sides of 0.5 on any constraint in the same domain (from Phase 4.5 satisfaction matrix).
   - `is_closed()` requires both fields to be empty.
   - `CoherenceIncomplete` event emitted from `tasks.rs` at task close when `!is_closed()`.

**What is not yet done:**

4. **`is_closed()` as a loop gate.** `wave_coherence.is_closed()` is computed and traced per-wave (`h2ai.coherence` trace target) but the retry loop does **not** exit early when coherent. The loop still stops on budget exhaustion or "all proposals pass verification." Wiring early exit is the remaining concrete step.

5. **A/B score distribution comparison.** The adversarial verifier activates automatically (no flag to run standard and adversarial side-by-side). Measuring score distribution shift — "adversarial verifier produces lower first-pass scores and better oracle correlation" — requires either dual-run capability or oracle ground truth (GAP-E1).

**Architectural accuracy note.** The rubric-separation fix (`include_rubric=false`) addresses the *structural* circularity: the explorer no longer receives the rubric it will be scored against. Whether this produces measurably different behavior (lower first-pass scores, more genuine exploration) is empirically unvalidated. The fix is architecturally necessary but its effect is conditional on the explorer's reasoning capability — a model that ignores missing rubric cues and reasons from task description anyway would show no difference.

**Gap statement.**
The constraint rubric — including all binary pass/fail checks and remediation hints — is injected into the explorer's `system_context` by `compiler.rs` before generation begins. The LlmJudge verifier (Phase 3.5) then scores each proposal against that same rubric. This means the verification step is not an independent judgment; it is a check of whether the explorer followed instructions that were already provided to it.

Two compounding problems:

1. **Self-confirmation, not multi-agent reasoning.** For tasks where the constraint is clearly stated and the model is capable, score=1.0 is achieved on the first attempt with zero retries. The MAPE-K retry loop fires only when task complexity causes the explorer to fail *despite knowing the rubric*. In single-model devcontainer deployments, this condition is rarely met. The entire retry machinery is effectively a no-op for well-specified, low-complexity tasks.

2. **The retry loop appends what was already present.** When `RetryWithHints` fires, it injects `remediation_hints` from violated constraints into `retry_context`. These hints were already derivable from the rubric in `system_context`. The only genuinely new information the retry loop adds is: "you failed this check specifically." Whether that delta drives improvement or just reinforces an already-known instruction is empirically untested.

The deeper problem: the "independent verification" claim requires that the verifier's judgment be formed from knowledge independent of the explorer's prompt. This is architecturally impossible when both receive the same rubric. A verifier that also has the rubric in context will correctly score high-compliance proposals highly *regardless of whether the proposals are actually correct* — because the verifier is checking rubric adherence, not ground truth.

**Research approach.**

1. **Rubric-blind verifier experiment.** Run the same task set twice: once with the standard rubric-in-context setup, once with a verifier that receives *only* the raw proposal and the task description (no rubric). Compare: (a) correlation between rubric-blind score and test-oracle pass rate; (b) correlation between rubric-aware score and test-oracle pass rate. If rubric-blind scores correlate *better* with oracle outcomes, the rubric injection is contaminating the verification signal.

2. **Retry loop value measurement.** On a stratified task set (simple/medium/complex by constraint count and description length), measure: retry firing rate, score improvement per retry, and final oracle pass rate. If retry improvement is near-zero on simple tasks and positive only on complex tasks, the retry loop's value is domain-gated and the system should be documented accordingly.

3. **Separate rubric from context.** Architectural option: strip the rubric from `system_context` for the explorer and inject it only into the verifier prompt. This makes the verifier an independent judge while the explorer is a genuine problem-solver without scaffolding. The tradeoff: lower first-pass scores (explorers generate more freely), but verifier independence is restored.

**Falsification condition.**
If rubric-blind verifier scores correlate more strongly with oracle outcomes than rubric-aware scores (Spearman rank correlation), the current rubric injection is actively degrading verification quality by replacing independent judgment with rubric compliance checking.

**Interaction with GAP-C2 (single auditor bias).** A rubric-aware auditor has the same self-reference problem. If the auditor is scoring "did this follow the rubric" rather than "is this correct," auditor diversity (GAP-C2) does not help: multiple rubric-aware auditors will agree on rubric compliance and disagree only on stylistic preferences.

**Open questions for brainstorm.**
- Should the rubric remain in explorer context as guidance but be *withheld* from the verifier to create genuine judge independence?
- Can the verifier be given the rubric *after* forming a preliminary judgment (two-pass scoring) to separate independent judgment from rubric-check confirmation?
- Is there a prompt structure that makes the verifier's task genuinely harder than "check each criterion" — e.g., adversarial probing ("what is the weakest part of this proposal?")?
- For the devcontainer (single model family), what is the minimum task complexity at which the retry loop first produces a measurable quality improvement?

**Effort estimate.** 1 week for rubric-blind experiment harness, 1 week runs and correlation analysis. Architectural option (rubric separation) is a 2-day code change; measuring its effect requires the same harness.

**Critical missing piece: `role_frame` is empty in `diverse_defaults()`.**

`ExplorerSlotConfig` has two fields: `cot_style` (reasoning format instruction) and `role_frame` (personality/identity injection prepended to system context). `ExplorerSlotConfig::diverse_defaults()` populates all four `cot_style` variants but leaves **`role_frame: String::new()`** on every slot. This is the distinction that matters:

- `cot_style` says "think this way" — it changes output structure and sequencing but does not change what the model considers salient or which parts of its training activate.
- `role_frame` says "you are this kind of reasoner" — a model told `"You are a skeptical senior engineer whose first instinct is to find the production failure mode"` will attend to different features of the problem than the same model with no framing, even at the same temperature. This is cognitive context, not sampling variance.

The implication for single-model diversity: if `role_frame` produces measurably lower error correlation (ρ) than temperature variation alone, then the single-model deployment is not fundamentally limited — it is *under-configured*. The same model running `[Implementer, SecurityAuditor, PerformanceEngineer, DevilsAdvocate]` personas may produce genuinely decorrelated proposals on tasks where those roles have different priors.

The literature signal is mixed: arXiv 2506.07962 finds error correlation driven by training data and architecture, not prompting. But this likely tests temperature + minor system prompt variation — not full persona framing that activates different professional reasoning modes. arXiv 2508.09654 finds temperature insufficient; it does not test persona diversity. The persona effect on ρ is an open empirical question for H2AI's specific task domains.

**Verification persona as independent judge.** If the verifier receives `role_frame: "You are a hostile reviewer. Your goal is to find the single most likely way this proposal violates a constraint or fails silently."`, it is no longer checking rubric adherence — it is actively attacking the proposal. This partially restores the independence that rubric-in-context removes, without requiring a separate model family.

---

### GAP-A2: USL N_max vs. Actual Quality Curve

**Gap statement.**
`N_max = round(√((1−α)/β_eff))` is derived from USL's throughput model by setting `dX/dN = 0`. USL models throughput under parallelism. H2AI uses it to bound quality under ensemble size. The domain transfer is unvalidated: there is no published result showing that quality in an LLM ensemble peaks at the USL throughput optimum.

The formula is used as both a ceiling (stop adding agents when merge cost exceeds diversity benefit) and an implicit quality predictor. These are different claims. The ceiling use may be valid as a cost heuristic even if the quality predictor is wrong.

**Research approach.**
Using the same task set from GAP-A1, vary N systematically: 2, 3, 5, 7, 9. For each N:
1. Record oracle pass rate (quality signal)
2. Record merge-phase wall-clock time (cost signal)
3. Record N_max as predicted by calibration at that run

Fit the empirical quality-vs-N curve and find its actual peak. Measure how far the actual peak is from the predicted N_max.

**Possible outcomes and responses:**
- Quality peaks within ±1 of N_max → USL framing is a valid proxy; keep the formula, add an empirical confidence note.
- Quality peaks earlier than N_max → N_max is a loose ceiling; switch to `EnsembleCalibration::n_optimal` (Condorcet-based) as the primary target and demote N_max to a hard cost cap.
- Quality peaks later than N_max → N_max is too conservative; recalibrate β₀ measurement or add a γ parameter that stretches the ceiling for high-diversity pools.
- No clear quality peak (quality is flat or monotone) → ensemble size is not a meaningful variable for this task class; simplify.

**Falsification condition.**
If the empirical quality-vs-N curve shows no relationship to the calibrated N_max across task types, retire the USL formula as a quality predictor and keep it only as a cost ceiling, clearly labelled.

**Open questions for brainstorm.**
- How do we control for token cost when comparing N=2 vs. N=9 (N=9 is inherently more expensive, so raw pass rate is not a fair comparison — quality/cost must be the metric)?
- Should N_max be compared against mean quality peak or median? What about the tail (P90 task difficulty)?
- Does β₀ need to be re-fit for each task domain, or is a single calibration portable across domains?
- Is the iterative `n_max_context_aware` solver converging to meaningful values in practice, or is the γ parameter always near-zero?

**Effort estimate.** 2 weeks: 1 week harness instrumentation, 1 week runs and curve fitting.

---

### GAP-A3: ρ Is a Proxy — Measure Error Correlation Directly

**Gap statement.**
`rho_mean = 1 − CG_mean` is an untested heuristic. The CJT quality bound `Q(N, p, ρ) = p + (Q_ind − p)(1 − ρ)` is only meaningful when ρ reflects actual error correlation between adapters. CG_mean measures constraint-profile behavioural agreement; it does not measure whether two adapters make the same mistake on novel tasks.

The proxy can be wrong in both directions:
- Two adapters can agree on constraints (high CG_mean → low ρ_proxy → CJT predicts large benefit) while sharing a hallucination vector from common training data (high actual ρ → CJT benefit collapses).
- Two adapters can disagree on constraint satisfaction (low CG_mean → high ρ_proxy → CJT predicts little benefit) while having genuinely independent error patterns (low actual ρ → more benefit than predicted).

**Research approach.**
1. Run `scripts/baseline_eval.py` on a labeled task set to get binary correct/incorrect per adapter per task.
2. Compute measured pairwise error correlation: `ρ_measured(i,j) = Pearson corr(errors_i, errors_j)` over the task set.
3. Compare `ρ_measured` against `1 − CG_mean` and `1 − N_eff/N`.
4. Fit the best linear predictor of `ρ_measured` from the two proxy signals.

**Expected finding based on existing literature.**
arXiv 2506.07962 finds error correlation is driven by training data and architecture, not prompting. If that holds: ρ measured on the calibration corpus will be a stable predictor of ρ on held-out tasks within the same domain. If it does not hold: ρ is task-dependent and requires online per-task estimation.

**Implementation path when measurement is available.**
Add `EnsembleCalibration::from_measured_rho` alongside the existing `from_measured_p`. This breaks the last proxy link in the CJT chain. The `prediction_basis` field already distinguishes `Heuristic` from `Empirical` for p — extend the same flag to ρ. Until measured ρ exists, the system continues with the proxy and the `Heuristic` label.

**Falsification condition.**
If `ρ_measured` and `1 − CG_mean` are uncorrelated (|r| < 0.3 across a representative adapter pool), the CJT chain is not producing meaningful quality bounds. Either measure ρ online (expensive) or replace Q with a cruder but honest signal (e.g., survival rate from audit alone).

**Open questions for brainstorm.**
- How large does the labeled task set need to be to get a stable ρ estimate per adapter pair? (Statistical power question: ρ estimates from <50 tasks are noisy.)
- Should ρ be task-domain-specific? An adapter pair might have correlated errors on code tasks but independent errors on factual QA.
- Is the embedding-cosine N_eff a better predictor of ρ_measured than Hamming CG? If yes, the bivariate CG claim has direct empirical support.
- Online ρ estimation per task: is it feasible to update ρ_mean from each task's verification scores using an EMA?
- **Does persona/role-frame diversity reduce ρ measurably compared to temperature-only variation?** The ρ measurement experiment should include three adapter conditions: (a) same model, temperature variation only; (b) same model, `cot_style` variation only; (c) same model, `role_frame` + `cot_style` variation. If condition (c) produces significantly lower ρ than (a) or (b), persona framing is a viable single-model diversity mechanism and the single-family limitation is a configuration gap, not a fundamental architectural one.

**Effort estimate.** 3 weeks: 1 week for oracle labeling of task set, 1 week for correlation analysis (including persona conditions), 1 week for `from_measured_rho` implementation.

---

## Brainstorm Group B — Mathematical Formula Validity

These gaps target specific formulas in the math apparatus. They do not threaten the core thesis but they do determine whether the formulas are principled or arbitrary.

---

### GAP-B1: β_eff Formula Has No Derivation

**Gap statement.**
`β_eff = β₀ × (1 − CG_mean)` is stated as a design choice. The formula is monotone decreasing in CG_mean and bounded — both good properties — but the specific linear form `(1 − CG_mean)` has no derivation from USL theory or from empirical fitting. It could equally be `(1 − CG_mean)²`, `exp(−λ × CG_mean)`, or `1/(1 + k × CG_mean)`. The choice determines how fast N_max grows as the pool's constraint agreement increases.

**Research approach.**
Run calibrations at controlled CG_mean levels (achieved by varying the constraint corpus density and the adapter pool composition across homogeneous → heterogeneous) and measure actual merge-phase latency at each N. Fit candidate functional forms:

```
β_measured(CG_mean) vs β₀ × f(CG_mean)
```

for f = {linear, quadratic, exponential, hyperbolic}. Select by AIC/BIC.

If the linear form wins, add a note citing the empirical fit. If a non-linear form wins, update the formula and the `n_max()` derivation accordingly. If no parametric form fits well (R² < 0.7), replace with a monotone spline interpolation over measured (CG_mean, β_eff) pairs.

**Falsification condition.**
If merge-phase latency shows no relationship to CG_mean (β_measured is flat across the CG_mean range), then CG_mean does not predict merge cost and should be decoupled from β_eff. In that case, β_eff reverts to a fixed β₀ and CG_mean's role is reduced to the CJT ρ proxy only.

**Open questions for brainstorm.**
- Can we construct adapter pools with controlled CG_mean values? (Requires generating constraint corpora with known overlap properties.)
- Is merge-phase latency the right proxy for β? Or is the number of constraint violations resolved during synthesis a better signal?
- Should γ (attention-sensitivity coefficient in `n_max_context_aware`) be fit simultaneously with the β_eff functional form, or independently?

**Effort estimate.** 2 weeks: 1 week controlled calibration runs, 1 week curve fitting.

---

### GAP-B2: Hamming CG Is a Neologism

**Gap statement.**
"Hamming Common Ground" does not appear in the published literature. It is H2AI-specific terminology. Readers searching for the foundational reference will not find one. The underlying concept — pairwise Hamming similarity on binary constraint-satisfaction fingerprints — is sound, but the naming implies a published grounding that does not exist.

This is not a correctness gap but a credibility gap. Presenting a project-specific heuristic as if it has an external citation undermines trust when the citation cannot be found.

**Research approach.**
Two options:
1. Rename to "constraint-profile agreement" or "constraint-satisfaction similarity" throughout the codebase and docs, with a note that it is an H2AI-specific operationalisation of Hamming distance applied to binary satisfaction vectors.
2. Formalise and publish: write a short technical note or arXiv preprint defining Hamming CG as a diversity measure for LLM ensemble constraint compliance, survey related work (Hamming distance in CSP, ensemble diversity measures in ML), and cite it internally. This turns the gap into a contribution.

Option 2 is the higher-value path if the term is going to appear in external documentation or papers.

**Open questions for brainstorm.**
- Is "Common Ground" the right intuition to preserve? In communication theory, common ground means shared background knowledge — is that what CG_mean measures for LLM adapters?
- Are there existing diversity measures in the ensemble learning literature (Q-statistic, κ-statistic, disagreement measure) that Hamming CG reduces to in specific cases? If so, cite those instead of coining a new term.
- How should the docs distinguish H2AI-specific definitions from standard references?

**Effort estimate.** 1 week for terminology audit and rename, or 4–6 weeks for the arXiv note path.

---

### GAP-B3: Attribution Formula Is Self-Referential

**Gap statement.**
`Q_total = base_quality × verification_filter_ratio × tao_uplift_factor × topology_correction + synthesis_gain` uses only internal signals. Every factor is derived from the same LLM verifier used throughout the pipeline. A verifier blind spot inflates individual scores and synthesis scores equally — the decomposition is internally consistent but cannot measure correctness against the outside world.

The formula is currently presented as a quality decomposition. It is actually a confidence decomposition: it measures how confident the system is in its own output, not whether the output is correct. These are correlated but not equal.

**Research approach.**
Ground `q_confidence` against an external oracle on a labeled task set. For each task, record `q_confidence_predicted` and `q_oracle_measured` (test pass rate, fact-check result, domain expert rating). Compute the calibration curve: does `q_confidence = 0.9` correspond to 90% oracle accuracy?

Use conformal prediction (arXiv 2406.09714) to construct calibrated coverage intervals:
1. Fit the conformal predictor on a calibration split: compute residuals `|q_confidence − q_oracle|`.
2. At inference time, output `q_confidence ± conformal_margin(α)` where the margin guarantees coverage with probability `1 − α`.

This upgrades `q_confidence` from a point estimate with bootstrap CG-variance intervals to a statistically calibrated quality interval grounded in oracle data.

**Short-term palliative — ✅ Done.**
`Q_total` renamed to `q_confidence` in `HarnessAttribution` and `TaskAttributionEvent`. `prediction_basis` field was already present. The API surface now clearly signals `Heuristic` vs `Empirical`.

**Falsification condition.**
If the calibration curve shows poor correlation between `q_confidence` and oracle accuracy (Pearson r < 0.5), the attribution formula is producing noise as quality signals. Simplify to: report the raw audit survival rate and synthesis gain delta, remove the multiplicative decomposition.

**Open questions for brainstorm.**
- What oracle is available for H2AI's intended task domains? Code: test suites. Factual QA: reference datasets. Structured reasoning: formal verifiers. Open-ended writing: none without human rating.
- Is the conformal prediction path feasible without a fixed labeled dataset? (Exchangeability assumption in conformal prediction requires i.i.d. tasks — does H2AI's workload satisfy this?)
- Should `synthesis_gain` be removed from the formula until an oracle can distinguish verifier-preference gain from actual quality gain?
- What UI change communicates the distinction between confidence and quality to an operator who does not read the docs?

**Effort estimate.** Oracle data collection is domain-specific and open-ended. Conformal predictor code: 3 days once oracle data exists. Rename/palliative: 1 day.

---

## Brainstorm Group C — Adversarial Committee Integrity

These gaps concern the system's ability to detect and route around correlated failure. They are orthogonal to the core thesis but determine whether the committee produces reliable outputs when under pressure.

---

### GAP-C1: Krum Breaks Under Correlated Hallucination

**Gap statement.**
The `OutlierResistant{f}` merge strategy is derived from Blanchard et al. (2017) Krum, whose breakdown-point proof assumes Byzantine faults are independent outliers in distance space. LLM correlated hallucinations are the opposite: when multiple adapters from families sharing Common Crawl training data encounter the same false premise, their outputs cluster tightly in Jaccard-distance space. Krum selects the geometric median — the hallucinated answer — as its output with high confidence.

Existing mitigations (family conflict gate, cosine N_eff guard, slot-config CoT diversity) reduce the probability but do not eliminate the failure mode. When they fail, the system provides no signal that it has confidently selected a wrong answer.

**Research approach.**
Design a correlated hallucination detector that fires before `MergeEngine::resolve` outputs the Krum result:

1. Compute all pairwise Jaccard distances among surviving proposals.
2. Compute the coefficient of variation (CV) of the distance matrix: low CV = homogeneous cluster (correlated regime), high CV = spread cloud (independent regime).
3. If `CV(distances) < correlated_hallucination_threshold` → emit `CorrelatedEnsembleWarning` event before emitting `MergeResolved`.
4. Route the warning to the operator via SSE and to the MAPE-K loop as a new intervention option: force Phase 5a synthesis with an explicit cross-proposal contradiction-seeking prompt.

The threshold requires empirical calibration. Instrument a controlled experiment: inject a known false premise into the system context shared by all adapters, record the CV of the distance matrix at the merge step when the hallucination fires. The CV at which the hallucination becomes dominant is the threshold prior.

**Research question for the brainstorm.**
Does the synthesis phase (Phase 5a) actually recover from correlated hallucination? The synthesis LLM receives all correlated proposals and is asked to reconcile them. If all N proposals assert the same wrong fact, synthesis may simply amplify the false consensus. A dedicated contradiction-seeking prompt (explicitly asking: "what factual claims in these proposals cannot all be true simultaneously?") is needed. Whether this works depends on the synthesis model's reasoning capability — an empirical question.

**Falsification condition.**
If the correlated hallucination detector fires frequently in production but Phase 5a synthesis does not improve oracle accuracy in those cases, the system needs a different intervention: escalate to human review, reject the task with a `CorrelatedEnsembleDetected` failure, or require a verified external oracle call before proceeding.

**Open questions for brainstorm.**
- Is Jaccard distance the right metric for detecting correlated LLM outputs, or would cosine distance on embeddings be more sensitive?
- How do we distinguish "legitimate consensus" (all adapters are correct and agree) from "correlated hallucination" (all adapters are wrong and agree)? Without an oracle, the distance matrix looks the same.
- Should `CorrelatedEnsembleWarning` be an operator alert or an automatic MAPE-K intervention?
- What is the interaction between this detector and the cosine N_eff guard at Phase 2.6? Can Phase 2.6 be strengthened to prevent the scenario rather than detecting it at merge time?

**Effort estimate.** 3 weeks: 1 week detector implementation, 2 weeks threshold calibration with injected hallucinations.

---

### GAP-C2: Single Auditor Is a Systematic Bias Point

**Gap statement.**
Phase 4 is a single LLM adapter call. The auditor is the final non-negotiable gate — all surviving proposals pass through it. If the auditor has a bias on a task domain (positional bias, length bias, self-preference for its own model family's style — Zheng et al. 2023, arXiv 2410.02736), every task in that domain is systematically biased.

The family conflict gate (`VerifierExplorerFamilyConflict`) addresses the most obvious monoculture case. It does not address: subtle cross-family stylistic preferences, training-data biases toward certain answer formats, or capability gaps in specific reasoning domains.

A multi-auditor committee is architecturally consistent with the adversarial-committee thesis but is not currently implemented.

**Research approach.**
Before building multi-auditor, measure the need. Instrument a shadow second auditor on a random 10–20% of tasks (different family from the primary auditor, enforced by the existing family conflict gate logic). Log every case where the two auditors disagree. Track:
- Disagreement rate overall and by task type
- On tasks with oracle answers: which auditor was right more often?
- Do disagreements cluster by task domain, proposal length, or model family?

This data answers: is single-auditor bias a real problem at the current task distribution, or a theoretical concern that rarely fires in practice?

**Implementation path when data supports it.**
Two-auditor majority vote is the minimal implementation: both auditors must approve, or the proposal is pruned. Three-auditor Krum-style (geometric median of approval confidences) is the full solution. The Phase 4 code path already loops over surviving proposals; extending to multiple auditors is a loop over auditor adapters per proposal.

**Falsification condition.**
If shadow auditor data shows disagreement rate < 5% and the primary auditor is right on 90%+ of disagreements (by oracle), the SPOF risk is low and multi-auditor overhead is not justified. Document this empirically and close the gap as low-priority.

**Open questions for brainstorm.**
- Should auditors be allowed to provide a confidence score alongside `{approved, reason}`? A low-confidence approval could trigger a second auditor without running a full shadow auditor on every task.
- Is the family conflict gate sufficient to ensure meaningful auditor diversity, or do we need a more nuanced diversity requirement (e.g., at least one auditor trained primarily on code, at least one on reasoning)?
- If two auditors disagree, should the system escalate to a third auditor (the "rubber meets the road" case) or fall back to human review?
- What is the latency cost of a second auditor? Phase 4 is already sequential; adding a second auditor doubles Phase 4 latency.

**Effort estimate.** 4 weeks for shadow auditor data collection, 1 week for analysis, 2 weeks for two-auditor implementation when justified.

---

### GAP-C3: Cosine N_eff Guard Is Optional — Bivariate CG Can Be Silently Downgraded

**Gap statement.**
Phase 2.6 (the cosine N_eff pool diversity guard) requires `fastembed` and a configured embedding model. Without them, the system falls back to "closed-form n_eff_cosine_prior" and disables Phase 2.6. The system continues to run but the bivariate CG safety net — the most novel piece of the framework and the primary guard against correlated hallucination at pool level — is silently absent.

An operator who has not read `research-state.md §6` will run the system believing bivariate CG is active when it is not. The degradation is not surfaced in logs, metrics, or the SSE event stream.

**Research approach.**
This is primarily an engineering gap with a research question attached: when bivariate CG is downgraded to univariate, what is the measured increase in correlated hallucination rate on the same task set?

Run the GAP-C1 hallucination injection experiment twice: once with fastembed active (Phase 2.6 enabled), once without (Phase 2.6 disabled). Measure how often the correlated hallucination detector (from GAP-C1) fires in each condition. If the rate is materially higher without Phase 2.6 (>2× increase), the silent degradation is a safety issue.

**Engineering fix regardless of research outcome.**
Change the degradation from silent to loud:
1. When `embedding_model` is `None` and `diversity_threshold > 0`: emit `DiversityGuardDegradedEvent` in the SSE stream and increment a Prometheus counter `h2ai_phase26_disabled_total`.
2. Add a configuration flag `require_bivariate_cg: bool` (default `false`). When `true`, fail the task at Phase 2.6 with `InsufficientPoolDiversity` rather than silently proceeding.
3. Add a startup warning (not panic) when `diversity_threshold > 0` and no embedding model is configured.

**Open questions for brainstorm.**
- Should `require_bivariate_cg` default to `true` in production configurations and `false` only in dev/test profiles?
- Is there a lightweight fallback for embedding computation that avoids the full fastembed dependency — e.g., TF-IDF cosine similarity as a proxy? Would this be good enough to preserve Phase 2.6 semantics without the full embedding model?
- What Prometheus alert rule should operators set on `h2ai_phase26_disabled_total`?

**Effort estimate.** 2 days engineering fix, 2 weeks for the hallucination rate comparison experiment.

---

## Brainstorm Group D — Infrastructure and Operational Gaps

These gaps are engineering problems that interact with the math. They do not falsify the thesis but they corrupt the math's inputs if left unaddressed.

---

### GAP-D1: Calibration Measures Latency, Not Quality Cost

**Gap statement.**
Phase A and Phase B of the calibration harness measure wall-clock time to fit α and β₀. USL's β is a coherency cost — in H2AI's framing, the cost of reconciling proposals that disagree on constraints. Wall-clock merge latency is a proxy for this cost, but it conflates:
- The synthesis LLM's computation time (hardware-dependent, not task-dependent)
- The number of constraint conflicts requiring resolution (task-dependent, the signal we want)
- Network latency to the LLM API (infrastructure noise)

A fast merge that resolves zero constraint conflicts (because proposals are correlated and trivially agree) produces a low measured β₀, which raises N_max — precisely the opposite of what should happen when correlated proposals are detected.

**Research approach.**
Extend the calibration harness to record a second β signal: **constraint conflict count per merge**. During Phase B, after all adapters respond, run the constraint verifier on each proposal and record:
- Number of constraint violations per proposal
- Pairwise constraint disagreement rate (proposal i violates constraint k that proposal j satisfies)
- Total number of constraint conflicts resolved by the synthesis pass

Fit β₀ from both the timing signal and the conflict-count signal. Compare the resulting N_max values. If they diverge significantly (>2 agents difference), the conflict-count β₀ is the more principled value for quality bounding; the timing β₀ remains useful for latency estimation.

**Open questions for brainstorm.**
- Should the runtime maintain two β₀ values: `beta_quality` (from conflict counts) and `beta_latency` (from timing)? Which one drives N_max in the planner?
- How do we weight the conflict-count β₀ against the timing β₀ when they disagree? Is a combined cost function sensible?
- Does the constraint corpus size affect the conflict-count β₀ in ways that make it incomparable across deployments with different corpus densities?
- Is online β₀ tracking (the existing `beta_from_token_spans` EMA) measuring latency or quality cost? If latency, it needs a parallel EMA on conflict counts.

**Effort estimate.** 1 week to extend the calibration harness, 1 week to compare β₀ signals on real calibration runs.

---

### GAP-D2: Compound Task Cost Is Unconstrained

**Gap statement.**
A `CompoundTaskEngine` DAG fires a full 6-phase H2AI wave (N adapters × verification + audit) for each subtask. The cost is: `N_subtasks × N_adapters × (generation + verification + audit) token cost`. A compound task with 5 subtasks, 5 adapters per wave, and 3 verification passes per proposal costs approximately 75 LLM calls before synthesis. This is not estimated or communicated to the operator before execution begins.

There is no pre-execution cost estimate, no cost cap, and no mechanism for the operator to inspect the planned DAG and abort before the first adapter call fires.

**Research approach.**
Two complementary solutions:

**Lightweight complexity probe (pre-execution).**
Before dispatching any subtask ensemble, call a single light adapter (smallest available model) with a modified prompt: "Given this subtask description, estimate on a 1–5 scale how many independent reasoning steps or domain-knowledge lookups are required." Route subtasks rated 1–2 to a single-adapter execution path; route subtasks rated 3–5 to the full ensemble. The complexity probe costs one small-model call per subtask vs. N full-model calls.

This is a bandit problem: the probe's prediction is uncertain and should be updated from outcomes. The existing `h2ai-orchestrator/src/bandit.rs` Thompson Sampling implementation is the right machinery.

**Pre-execution cost estimate and operator confirmation.**
Before executing the DAG, emit `CompoundTaskCostEstimate {subtask_count, estimated_adapter_calls, estimated_token_budget}` as an SSE event. Add a configuration flag `require_compound_task_approval: bool`. When enabled, the engine waits for an explicit `POST /tasks/{id}/approve` before proceeding.

**Open questions for brainstorm.**
- What complexity proxy correlates best with actual ensemble benefit? Subtask description length? Number of constraints that apply? Required tool diversity?
- Should the bandit over complexity-probe accuracy be shared across tasks (improving with each compound task run) or reset per task?
- Is `require_compound_task_approval` the right UX, or should the system use cost caps (`max_adapter_calls_per_task`) and self-manage?
- How does the SchedulingEngine currently handle a subtask that is substantially more expensive than its siblings? Is there any fairness or budget mechanism today?

**Effort estimate.** 1 week for cost estimation and SSE event, 2 weeks for complexity probe + bandit routing.

---

### GAP-D3: Calibration Bootstrapping Has No Defined Path

**Gap statement.**
Every task execution requires calibration data in `H2AI_CALIBRATION` KV. A new deployment with an empty KV store will return `503` on every task request until calibration has been run. The calibration harness exists but there is no documented, automated, or integrated path from "blank Kubernetes namespace" to "calibrated and ready to serve tasks."

An operator following the deployment docs (Kubernetes manifests, Helm chart) will encounter a silent `503` with no guidance unless they have read the calibration section of `operations.md` carefully.

**Research approach.**
This is an engineering gap, not a research gap, but it has a research angle: what is the minimum viable calibration that a new deployment can run immediately without a domain-specific constraint corpus or task prompts?

Design a **bootstrap calibration mode** that:
1. Uses a built-in set of synthetic calibration prompts covering basic capability areas (code, factual, reasoning)
2. Runs with the configured adapter pool against these prompts
3. Produces a conservative `CalibrationCompletedEvent` (wide confidence intervals, large `n_max_lo`/`n_max_hi` spread) that unblocks task execution
4. Marks the result as `calibration_quality: Bootstrap` (vs. `Domain`) so operators know they are running on synthetic priors

Domain-specific calibration then overrides the bootstrap result when the operator runs it with production task prompts.

**Open questions for brainstorm.**
- What makes a good synthetic calibration prompt set? Should it be task-domain-agnostic (basic capability probes) or should it ask the operator to specify a domain at deployment time?
- Should the Kubernetes Helm chart include a `calibration` Job that runs automatically on first install?
- How often should production deployments re-run calibration? The temporal decay (GAP-B1 area: 7-day CG halflife) already creates pressure — but there is no automated recalibration trigger when the halflife-decayed β_eff becomes too conservative.
- What is the minimum adapter pool size for a meaningful calibration? The two-phase fit (Phase A with 2 adapters, Phase B with M ≥ 3) implies a minimum of 3 adapters. Can the bootstrap calibration work with 2?

**Effort estimate.** 1 week for bootstrap calibration mode, 1 week for Helm chart Job integration.

---

### ~~GAP-D4: Synthetic Calibration Is Indistinguishable From Measured Calibration~~ ✅ CLOSED

**Status: CLOSED (2026-05-09).** All surfacing points implemented.

- `CalibrationSource` enum (`Measured`/`PartialFit`/`SyntheticPriors`) in `h2ai-types/src/events.rs`.
- `CalibrationHarness::run` sets `calibration_source` from `m < 3` (USL fallback) and `adapter_outputs.len() < 2` (CG fallback): both fallback → `SyntheticPriors`; neither → `Measured`; one → `PartialFit`.
- `CalibrationCompletedEvent` carries `calibration_source` (`#[serde(default = Measured)]` for backwards-compatible deserialisation).
- `TaskAttributionEvent` carries `calibration_source` — outputs can be retrospectively filtered by calibration quality.
- Prometheus gauge `h2ai_calibration_source` with three states (`measured`, `partial_fit`, `synthetic_priors`) in `h2ai-api/src/metrics.rs`; updated on each `CalibrationCompletedEvent`.
- Startup warning (`tracing::warn!`) in `h2ai-api/src/main.rs` when the persisted calibration has `SyntheticPriors`.

**Remaining open questions (not blocking).**
- Is `calibration_cg_fallback = 0.70` a defensible prior for single-model deployments? At CG=0.70, `β_eff = 0.039 × 0.30 = 0.0117`, giving `N_max ≈ 9`. For a single-model pool the true CG may be much higher, making this prior optimistic.
- Should `PartialFit` also trigger a startup warning, or only `SyntheticPriors`?

**Gap statement.**
When `CalibrationHarness::run` is called with a single adapter (< 2) or fewer than 3 timing points (M < 3), it silently falls back to hardcoded config values: `α = cfg.alpha_contention` (0.12), `β₀ = cfg.beta_base_default` (0.039), and `CG_mean = cfg.calibration_cg_fallback` (0.70). The resulting `CalibrationCompletedEvent` carries `calibration_quality: Default::default()` — there is no enum variant that flags the event as "synthetic priors only" vs. "fitted from measurement."

Downstream consequences:
- `N_max = round(√((1 − 0.12) / (0.039 × (1 − 0.70)))) = round(√(0.88 / 0.0117)) ≈ round(8.67) = 9` — a specific-looking number computed entirely from priors, indistinguishable from a USL fit.
- Every routing decision in single-adapter deployments (all devcontainer runs, any single-model CI environment) executes this formula. The routing looks scientifically grounded; it is running on hand-tuned constants.
- Operators reading `CalibrationCompletedEvent` in the SSE stream or NATS KV have no signal that the calibration is synthetic. Metrics, logs, and the event schema are identical for both paths.

**Relationship to GAP-D3.** GAP-D3 is about the missing automation path from blank deployment to calibrated state. GAP-D4 is the complementary problem: even when the system does run calibration, it proceeds silently on priors without any indication that the result is not based on measurement. GAP-D3 fixes the operational workflow; GAP-D4 fixes the epistemic transparency.

**Engineering fix (no research required).**

1. Extend `CalibrationQuality` enum (if one exists) or `CalibrationCompletedEvent` with a `calibration_source` field:
   ```rust
   pub enum CalibrationSource {
       Measured,        // USL fit from M ≥ 3 timing points, CG from ≥ 2 adapters
       PartialFit,      // USL fit valid; CG from fallback (< 2 adapters or empty corpus)
       SyntheticPriors, // Both USL and CG from config fallbacks
   }
   ```
2. Set `calibration_source` in `CalibrationHarness::run` based on which paths took the fallback.
3. Emit `calibration_source` in the SSE stream and surface it as a Prometheus label: `h2ai_calibration_source{source="synthetic_priors"}`.
4. Add a startup warning (not panic) when the most recent stored calibration has `calibration_source = SyntheticPriors`.

**Interaction with GAP-A2 and GAP-A3.** Both gaps require valid calibration as their starting condition. Any experiment run against a deployment using `SyntheticPriors` calibration produces meaningless data: the N_max being "validated" was never derived from measurement. The `calibration_source` label is a prerequisite for interpreting any experiment output.

**Open questions for brainstorm.**
- Should `PartialFit` calibration (valid USL, synthetic CG) be treated as acceptable for production routing, or should it also trigger a warning?
- Is `calibration_cg_fallback = 0.70` a conservative or optimistic prior? At CG=0.70, `β_eff = 0.039 × 0.30 = 0.0117`, giving `N_max ≈ 9`. If the actual CG of a single-model pool at different temperatures is much higher (e.g., 0.95), the prior is dangerously optimistic: the true `β_eff ≈ 0.039 × 0.05 = 0.002` gives `N_max ≈ 21`, but the fallback suppresses it to 9. The prior is not calibrated to the single-model case it most commonly covers.
- Could the system measure a rough CG even in single-adapter mode by comparing outputs at different τ values on the same prompt? This would at least produce a CG proxy without requiring a second adapter family.
- Should tasks submitted to a `SyntheticPriors`-calibrated instance be auto-tagged with `calibration_quality: synthetic` in their `TaskCompletedEvent` to allow retrospective filtering?

**Effort estimate.** 2 days for the enum extension and labeling, 1 day for Prometheus metric and SSE surfacing, 1 week to measure the CG of single-model τ-spread pools and determine whether 0.70 is a defensible prior.

---

## Brainstorm Group E — Quality Measurement Infrastructure

These gaps concern the measurement instruments themselves. Without them, all of Group A is blocked.

---

### GAP-E1: No Oracle Integration

**Gap statement.**
The attribution formula, the CJT quality bound, the conformal prediction path (GAP-B3), and the ρ measurement (GAP-A3) all require an external oracle: a ground-truth signal that is independent of the system's own verifier. Currently no oracle is wired into the production system. `scripts/baseline_eval.py` exists for offline measurement but there is no online oracle signal.

Without an oracle, `q_confidence` is the only quality signal. The system cannot distinguish between "my verifier is confident and correct" and "my verifier is confident and wrong." This is the foundational gap underlying all quality claims.

**Research approach.**
Oracle selection by task domain:
- **Code tasks:** Test suite execution. ShellExecutor is already in the toolchain; a `TestOracle` executor that runs `pytest`/`cargo test` and returns pass/fail is a direct extension of the existing executor pattern.
- **Math/formal reasoning:** Symbolic verifier (Z3, Lean) for formally specified tasks.
- **Factual QA:** Reference dataset lookup (TriviaQA, MMLU, HumanEval for code) for benchmark tasks.
- **Structured output tasks:** JSON Schema validation, regex match, or typed deserialisation as a lightweight oracle.
- **Open-ended writing:** No automated oracle — human rating required, or use a separate independent strong model family explicitly not in the adapter pool.

**Minimum viable oracle.**
Implement `OracleExecutor` as a `ToolExecutor` variant that runs a test command and returns `{passed: bool, details: String}`. Wire it as Phase 6 (post-merge oracle check): after `MergeResolved`, run the oracle on the winning proposal and emit `OracleResult {passed, score, oracle_type}`. This event never blocks task close; it is asynchronous. Accumulate `OracleResult` events to build the calibration dataset for conformal prediction.

**Open questions for brainstorm.**
- What fraction of H2AI's intended workload admits a deterministic oracle? If >50% are open-ended, the oracle path is only partially useful.
- Should oracle results be used to update calibration online (upgrading `prediction_basis` from `Heuristic` to `Empirical` as data accumulates)?
- How do we handle oracle unavailability for a task at runtime? Should tasks be tagged with `oracle_type: None | TestSuite | Symbolic | Reference` at submission time?
- Privacy: oracle test suites may contain proprietary logic. Can the oracle run inside the agent sandbox (WasmExecutor) to avoid exfiltration?

**Effort estimate.** 1 week for `TestOracle` executor on code tasks, 2 weeks for online calibration update pipeline. Non-code oracles are domain-specific and open-ended.

---

### GAP-E2: Talagrand Histogram Has No Feedback Loop

**Gap statement.**
`TalagrandDiagnostic::from_verification_scores` computes the rank histogram of verification scores across the ensemble. A flat histogram indicates a calibrated ensemble (uniform spread of quality); a U-shape indicates over-confidence (adapters cluster at high and low scores); a Λ-shape indicates under-dispersion (adapters cluster near the mean). The diagnostic is computed and emitted as `DiversityWarningEvent` and used as a soft ρ correction in attribution.

However, the Talagrand histogram in meteorological ensemble forecasting is used to drive **τ-spread adjustment**: when the histogram is U-shaped (over-confident), increase τ spread to encourage more diversity; when Λ-shaped (under-dispersed), decrease τ spread. In H2AI, the τ-spread adjustment is mentioned in `research-state.md §3.1` (Talagrand wired to τ-spread adjustment and `DiversityWarningEvent`) but the actual adjustment loop is not described in the architecture docs.

**Research approach.**
Clarify and close the feedback loop:
1. Define the τ-spread update rule explicitly: `Δτ_spread = η × (histogram_U_score − histogram_Λ_score)` where the scores are derived from the histogram's deviation from uniformity (e.g., KL divergence from uniform).
2. Apply the update to `EnsembleCalibration::tau_spread_factor` at the `SelfOptimizer` post-merge step.
3. Track the τ-spread history in the `H2AI_CALIBRATION` KV alongside α and β₀.

The research question: how quickly does τ-spread need to adjust? A fast adjustment (high η) may overshoot; a slow adjustment (low η) may lag task-distribution shifts. The right learning rate is an empirical question.

**Open questions for brainstorm.**
- Is τ-spread the right knob? Would adjusting the adapter pool composition (removing under-dispersed adapters) be more effective?
- Should the Talagrand adjustment be per-task-domain rather than global? A task domain with naturally high output variance may need a different τ spread than a low-variance domain.
- What is the interaction between τ-spread adjustment and the Condorcet n_optimal calculation? Increasing τ spread increases diversity (lowers ρ) which raises the Condorcet optimum — does the system re-run Phase 2 topology selection after a τ-spread update?
- Is `DiversityWarningEvent` surfaced to the operator in a way that prompts action? Currently it is an SSE event but there is no prescribed operator response.

**Effort estimate.** 1 week to define and implement the τ-spread update rule, 2 weeks to tune η on a representative task set.

---

## Brainstorm Group F — Nomenclature and Presentation

These gaps do not affect correctness but affect credibility and operator understanding. Address these before any external documentation or publication.

---

### GAP-F1: Q_total Presented as Quality, Not Confidence ✅ Done

`total_quality` renamed to `q_confidence` in `HarnessAttribution` and `AttributionInterval`; `q_predicted` renamed to `q_confidence` in `TaskAttributionEvent`. `prediction_basis` field was already present. Architecture and attribution docstrings updated to distinguish confidence from oracle-grounded quality.

---

### GAP-F2: β_eff Formula Presented as Derived, Not Fitted ✅ Done

Added `Note (GAP-B1)` callout in `math.md §2` explicitly stating `β_eff = β₀ × (1 − CG_mean)` is an empirical heuristic, not derived from USL theory.

---

### GAP-F3: CRDT Terminology Implies Synthesis ✅ Done

Renamed `SemilatticeCompiledEvent` → `SelectionResolvedEvent` and `H2AIEvent::SemilatticeCompiled` → `H2AIEvent::SelectionResolved` across all 9 affected files. Added the clarifying note to `architecture.md §Phase 5`. The `SelectionResolved` event is now also published to NATS (previously it was built but never emitted).

---

## Gap Priority Matrix

| Gap | Core thesis risk | Implementation cost | Data dependency | Suggested session order |
|---|---|---|---|---|
| ~~GAP-A0 Verifier precision / holistic bias~~ | Critical | ✅ Code done | Evaluator model upgrade | — |
| ~~GAP-A1 Self-MoA vs. multi-family~~ | Critical | ✅ Done | — | — |
| ~~GAP-A5 Committee composition semantically unmotivated~~ | Critical | ✅ Done — Path C always-on, diverse_defaults() removed | Oracle comparison validates | — |
| GAP-A4 Verification circularity | High | 🟡 Architectural fix done (rubric separation + adversarial verifier + CoherenceState). Loop gate + A/B pending | Rubric-blind experiment; loop gate is 1-day code change | Session 1 |
| GAP-A2 USL N_max vs. quality curve | High | Low | Task set (shared with A1) | Session 1 |
| GAP-A3 ρ proxy chain | High | Low | Labeled task set | Session 2 |
| GAP-C1 Krum correlated hallucination | High | Medium | Injection experiment | Session 3 |
| GAP-C2 Single auditor bias | Medium | Medium | Shadow auditor data | Session 4 |
| GAP-B3 Attribution without oracle | Medium | Low (code), High (oracle) | Oracle data | Session 2 |
| ~~GAP-D4 Synthetic calibration not labeled~~ | Medium | ✅ Done — all surfacing points live | None | — |
| GAP-E1 Oracle integration | Blocking (for A2, A3, B3) | Medium | Domain-specific | Session 2 |
| GAP-D1 Calibration measures latency | Medium | Low — CoherenceState observability done; loop gate is the remaining step | Calibration runs | Session 5 |
| GAP-B1 β_eff functional form | Low | Medium | Controlled calibration runs | Session 5 |
| GAP-C3 Phase 2.6 silent downgrade | Low | Low (2 days) | None | Any |
| GAP-D3 Bootstrap calibration | Low | Low | None | Any |
| GAP-D2 Compound task cost | Low | Low | None | Any |
| GAP-E2 Talagrand feedback loop | Low | Low | Task runs | Session 6 |
| GAP-B2 CG neologism | None | Low | None | Any |
| ~~GAP-F1 q_confidence rename~~ | None | ✅ Done | — | — |
| ~~GAP-F2 β_eff heuristic label~~ | None | ✅ Done | — | — |
| ~~GAP-F3 CRDT terminology~~ | None | ✅ Done | — | — |

---

## Shared Infrastructure Required for Group A

Sessions 1 and 2 block on building a shared measurement harness:

1. **Labeled task set** — 100–200 tasks across code (test oracle), factual QA (reference answers), and constraint-heavy reasoning. Stratified by "requires knowledge diversity" vs. "requires reasoning diversity" per the GAP-A1 taxonomy hypothesis.
2. **Oracle runner** — see GAP-E1. Test-suite oracle for code tasks is the minimum viable oracle for Sessions 1 and 2.
3. **Per-N quality measurement** — the `scripts/benchmark/` harness extended to record oracle pass rate per adapter, per N value, and per cell in the 2×2 matrix from GAP-A1.
4. **Pairwise error correlation logging** — per-adapter binary correct/incorrect logged per task, stored to a local SQLite or parquet file for offline ρ analysis.

Building this shared infrastructure is the pre-work for Session 1 and should be the first concrete deliverable before any gap-resolution runs begin.
