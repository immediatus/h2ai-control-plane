# H2AI Research State

This document is the project's critical self-assessment. It states what the system claims, where the math is genuinely defensible, where it is honest engineering heuristic, where empirical validation is missing, and how H2AI sits in the external landscape. It is not a roadmap and it does not track historical work.

For implementation details: [`architecture.md`](architecture.md). For formulas: [`math.md`](math.md). For the operational surface: [`reference.md`](reference.md), [`operations.md`](operations.md).

---

## 1. The epistemic contamination problem

The Condorcet Jury Theorem and the USL coherency model both require one thing that is easy to state and hard to enforce: **agent independence**. An agent that has been contaminated by another agent's side effects is no longer an independent voter; its output is downstream of the contaminating agent's choices. When independence is broken, the CJT quality gain collapses toward zero and the USL β measurement becomes a noisy average over correlated states.

Three distinct threats break independence in practice:

### 1.1 Shared state mutation

When agents execute shell commands or write files to a shared workspace, their observations become coupled. Agent A's `git checkout`, file write, or database mutation changes what Agent B reads on the next iteration. The agents are no longer exploring the problem space independently — they are contaminating each other's state space. CJT requires that each voter's decision be statistically independent; shared mutable state makes that impossible.

**Architectural response:** `WasmExecutor` evaluates scripts inside a `wasmtime` sandbox with no WASI host imports — zero filesystem, network, or OS access. No agent can mutate shared state via code execution. `McpExecutor` enforces a read-only boundary (`read_file`, `list_directory` only) at the executor layer, regardless of what the MCP server supports. Agents that read the same file get the same content and diverge only in their reasoning — the intended source of diversity. `ShellExecutor` allowlists restrict commands to read-only operations in production configurations; the allowlist is the operator's explicit commitment about which commands leave no side effects.

### 1.2 Affinity bias (verifier/explorer monoculture)

If the agent generating a proposal and the agent verifying that proposal belong to the same model family, the verifier will systematically overlook the explorer's blind spots — because they share the same pre-training biases, the same hallucination vectors, and the same confidence patterns. A GPT-4 verifier evaluating a GPT-4 proposal measures agreement, not correctness. Self-preference bias (Zheng et al. 2023; arXiv 2410.02736) is well-documented; same-family bias is a structural amplification of it.

**Architectural response:** Two complementary mechanisms address affinity bias:

1. **`VerifierExplorerFamilyConflict` gate** (`engine.rs`): when `calibration.explorer_verification_family_match = true` and `family_constraint = RequireDiverse`, the task fails immediately before the MAPE-K loop. No retry can resolve a topology where judge and explorers share a family.

2. **Multi-variant judge panel (implemented 2026-05-16):** Phase 3.5 now builds a `JudgePanel` from the verification adapter plus cross-family explorer adapters. When ≥2 families are present, a `CrossFamily` supermajority panel fires all variants in parallel per constraint. When only one family is available, a `PersonaOnly` panel (Literal/Contextual/Skeptical) requires unanimous agreement. Uncertain verdicts pass with a configurable score penalty rather than pruning — the constraint corpus is the primary bias mitigation (binary rubric decomposition removes judge-model sensitivity per Prosa 2605.01630); panel diversity is a second-order guard. `explorer_verification_family_match` and `adapter_families` in `CalibrationCompletedEvent` are now populated from the actual adapter registry.

### 1.3 Execution latency (α spike)

If every tool call during the TAO loop requires a NATS round-trip between the edge agent and the central orchestrator, the Amdahl serial fraction α spikes. The NATS latency — even at sub-millisecond levels — accumulates across `agent_max_tool_iterations` turns for each of N parallel agents. At the USL ceiling, α is the dominant term: `N_max = √((1 − α) / β_eff)`. A high-α deployment drives N_max toward 1, making the committee economically and computationally unviable.

**Architectural response:** The TaoAgent loop runs entirely inside the edge agent binary. No tool call crosses the NATS boundary. The orchestrator receives one `TaskResult` message at the end — the complete answer plus the audit trail of `ToolCallRecord` entries. During the TAO loop, the orchestrator's event bus is silent for that agent. α captures only the genuinely serial phases: task bootstrap (constraint compilation), topology provisioning, and merge. The TAO loop itself is fully parallel and contributes zero to α.

---

## 2. The system thesis

H2AI Control Plane orchestrates pools of LLM adapters as an *adversarial committee*. The runtime claims:

- **Committee composition is epistemically motivated.** Phase 0 (Path C) derives `Vec<ExplorerSlotConfig>` from the task description via LLM decomposition. Each slot carries a `role_frame`, `cot_style`, `focus_mandate` (constraint domains owned), and `rejection_criteria` (specific failure mode to probe). N = count of genuinely orthogonal roles, capped by USL N_max.
- The reliability of an N-adapter committee is bounded by **two diversity signals** that must both be measured: Hamming Common Ground on constraint behaviour, and cosine N_eff on semantic embedding.
- The throughput-vs-coordination trade-off is captured by USL (Gunther 1993) with a CG-coupled coherency cost: `β_eff = β₀ × (1 − CG_mean)`, `N_max = round(√((1 − α) / β_eff))`.
- The quality of the committee is bounded by a correlation-corrected Condorcet Jury Theorem (Nitzan & Paroush 1982; Ladha 1992): `Q(N, p, ρ) = p + (Q_ind(N, p) − p)(1 − ρ)`.
- **Verification is structurally independent.** Explorer context is compiled with `include_rubric=false` — LlmJudge rubrics are withheld from the explorer and retained only by the verifier. The verifier uses `ADVERSARIAL_EVALUATOR_SYSTEM_PROMPT` (hostile-reviewer framing) by default.
- Failures are classifiable: post-zero-survival, the cosine N_eff distinguishes `ConstrainedExploration` (diverse generation rejected by constraints) from `ModeCollapse` (correlated hallucination). MAPE-K routes each to a different intervention (tombstone vs adapter rotation).
- **Coherence state is tracked per-wave.** `CoherenceState { uncovered_domains, active_contradictions }` is computed after each verification round. `CoherenceIncomplete` is emitted when the task closes without epistemic closure. The loop exit gate on `is_closed()` is pending.
- **Thinking loop (optional pre-execution coverage-convergence).** Gated by `thinking_loop.enabled` (default `false`). When enabled, a structured multi-archetype brainstorm runs before decomposition and produces a `ThinkingReport` whose `shared_understanding` field is injected as `{thinking_context}` into the Phase 0 decomposition prompt. The loop iterates until `coverage_score ≥ coverage_threshold` or `max_iterations` is reached. Adaptive archetype count contracts by coverage deficit each iteration; `scheduled_tau` linearly decays temperature from `tau_max` to `tau_min` across iterations; `tension injection` directs new archetypes at explicitly named open tensions from the previous report. A quality floor gate prevents contraction when the archetype pass-rate falls below `expansion_quality_floor`, guarding against MoA degradation.
- **Ensemble Efficiency Index (j_eff).** `j_eff = Q_realized / Q_ceiling` is emitted in every `MergeResolvedEvent`. `Q_realized = condorcet_quality(n_valid, filter_ratio, rho_mean)` measures the quality achieved by the proposals that survived verification. `Q_ceiling = condorcet_quality(n_agents, p_mean, 0.0)` measures the theoretical ceiling under zero correlation. The ratio tells operators what fraction of the ensemble's quality potential was realised. Dynamic threshold: `j_eff_min = pareto_weights.diversity × thinking_coverage_score` — collapses to zero when no thinking context was generated and tightens as coverage improves.
- **Shadow auditor with adaptive AND-vote promotion.** A second adapter from a different family can run concurrently with the primary auditor via `tokio::join!`. `ShadowAuditorAccumulator` tracks per-domain disagreement rates over a sliding window and auto-promotes domains to AND-vote mode when the rate exceeds the configured threshold. Promotions persist to NATS KV. Default: `enabled=false` (observe mode off).
- **Grounded oracle loop (wired, phase 4.5, 2026-05-14).** Phase 4.5 oracle gate (NATS request/reply, `OracleGateConfig`) queries an external oracle before the merge step. On fail + low confidence, a `PendingClarificationEvent` suspends the engine; `POST /tasks/{id}/clarify` resumes it with an operator answer. The thinking loop's Stage 2 (`brainstorm_one`) sends candidate solutions inline to the oracle; `synthesize` applies `oracle_confidence_bonus` when the oracle approved. `oracle_gate_passed: Option<bool>` on `MergeResolvedEvent` tracks the gate outcome per task.
- **Adaptive prompt harness via OPRO (2026-05-14).** Per-adapter j_eff EMA drives OPRO cycles (arXiv 2309.03409) when output quality falls below `trigger_j_eff_threshold`. Thompson-sampling bandit selects among stored `PromptVariant` entries; best variant is promoted after `graduation_tasks`. Bootstrap priors seed Bayesian beta distributions from `AdapterProfile.tier` (Capable=0.78, Standard=0.62, Fast=0.45 j_eff medians) — providing principled Bayesian priors before any tasks have run, without requiring empirical data. `OproTriggeredEvent` and `PromptVariantPromotedEvent` make the optimisation auditable.
- **Delta checkpoint encoding (storage bottleneck resolved, 2026-05-14).** `CheckpointKind::Delta` stores JSON Patch (RFC 6902) diffs between checkpoints; `CheckpointKind::Base` snapshots every `base_interval`. Storage is O(N) rather than O(N²), unblocking long multi-task experiment runs that would otherwise exhaust NATS KV space. LRU cache + CAS pointer keep reconstruction latency sub-millisecond.
- **Correlated hallucination detected (2026-05-11; joint-threshold fix 2026-05-12).** Token-Jaccard pairwise CV detector (`compute_cv` in `correlated_hallucination.rs`) fires `CorrelatedEnsembleWarning` when **both** `CV(distances) < correlated_hallucination_cv_threshold` (default 0.30) **and** `mean_jaccard_distance < correlated_hallucination_min_jaccard_floor` (default 0.50) over ≥ `correlated_hallucination_min_proposals` surviving proposals. The joint gate prevents false positives on uniform-but-diverse ensembles (high mean Jaccard, zero CV). Reactive researcher path via `SraniGroundingChain`: `SpecAnchorGrounder` (spec entities, always) + `LlmResearcherGrounder` (tier 0) + `WebSearchGrounder` (tier 1, escalated on persistent CFI). Proactive researcher path: slots with `search_enabled: true` receive a researcher pre-pass before Phase 3 generation. N = 2 edge case: `compute_cv` returns `None` for diverse N = 2 (statistically uninterpretable single-point distribution) and `Some(cv = 0.0)` only when both proposals are identical.
- **SRANI correlated fabrication detection (2026-05-12).** Specification-Relative Architectural Noun Intersection detects shared ungrounded architectural entities across proposals. CFI = max pairwise overlap of per-proposal entity sets absent from the task specification. Adaptive sigmoid gate: `injection_pressure = σ((CFI − μ) / T)` where μ is an EMA of observed CFI values (α=0.20, ≈5-task memory horizon), T=0.15 (sigmoid temperature, fixed). `injection_pressure ≥ 0.20` emits `CorrelatedFabricationEvent`; `≥ gate_threshold` (default 0.50) also injects a grounding hint into `retry_context`. EMA midpoint μ cold-starts at `(warn_threshold + inject_threshold) / 2 = 0.45` until count ≥ 5, then tracks the deployment's observed CFI distribution. State persists to NATS KV (`srani_adaptive_state`) and survives restarts. `adaptive=false` falls back to static `warn_threshold` / `inject_threshold` behavior. Web-search grounding results are distilled with the researcher LLM before injection (`grounding_distill=true`, max 1200 chars hint). See `crates/h2ai-orchestrator/src/srani_gate.rs`, `srani_grounding.rs`.
- **Domain coverage guard (2026-05-11).** Before the MAPE-K wave loop, the engine checks whether the union of `constraint_domains` across all `ExplorerSlotConfig`s covers the corpus domain tag set. Below `domain_coverage_threshold`, `DiversityGuardDegradedEvent` is emitted. With `require_bivariate_cg = true`, coverage failure halts the task with `InsufficientPoolDiversity`. Phase 2.6 degradation (no embedding model) is now loud: `DiversityGuardDegradedEvent` fires and a startup warning is emitted rather than silently falling back to the closed-form prior.
- **Calibration is labelled.** `CalibrationSource::Measured / PartialFit / SyntheticPriors` on every `CalibrationCompletedEvent` and `TaskAttributionEvent`. Prometheus gauge and startup warning surface the label.
- Every successful task produces a confidence decomposition `q_confidence ≈ baseline × verification_filter × tao_uplift × topology_correction + synthesis_gain` (`crates/h2ai-orchestrator/src/attribution.rs`). This is a self-confidence score, not oracle-grounded quality.

The differentiating claim is not the math itself — most components have decades of literature behind them. It is that all four signals (USL, CJT, eigenvalue diversity, Talagrand calibration) are tracked together, calibrated together, and used together to bound a single committee-execution loop.

---

## 3. What is genuinely defensible

In decreasing order of mathematical rigor and confidence in domain transfer:

**Eigenvalue calibration via participation ratio.** `EigenCalibration::from_cosine_matrix` applies `(Σ λ)² / Σ λ²` (Choueifaty & Coignard 2008) to the embedding cosine kernel. This is a direct measurement: at full independence it returns N; at full correlation it returns 1. There is no contested domain transfer — the participation ratio is the right tool to answer "how many independent perspectives are in this pool?"

**Generation-first ProposalSet LUB.** The CRDT semilattice over `(generation, score)` provides crash-safe idempotency and TAO-monotonic ordering. All CRDT axioms hold. This is selection, not content synthesis — see §3.

**Three-tier merge dispatch.** `ScoreOrdered → ConsensusMedian → OutlierResistant{f}` correctly escalates with the maximum role error cost. The Krum-style `OutlierResistant` selection has a proven breakdown point (Blanchard et al. 2017) under independent faults. The honest limitation is correlated hallucination — see §3.

**USL N_max as a calibrated upper bound.** Even framed as a phenomenological heuristic, calibrating N from measured `(α, β_eff)` is strictly better than "spawn until expensive." The N(N−1) merge complexity is real; the `β₀ × (1 − CG_mean)` coupling correctly captures that divergent agents cost more to reconcile.

**Talagrand rank histogram.** Borrowed from ensemble weather forecasting (Leutbecher & Palmer 2008). Flat = calibrated, U-shape = over-confident, Λ-shape = under-dispersed. The diagnostic is wired to τ-spread adjustment and `DiversityWarningEvent`. No domain-transfer issue — calibration histograms are domain-agnostic.

**Bivariate CG.** Phase 2.6 and the MAPE-K `FailureMode` classifier exist precisely because Hamming CG and cosine N_eff measure different things and must be tracked together. Two adapters can share constraint profiles by accident (high Hamming CG) while being semantically near-identical (low cosine N_eff). The runtime detects and routes around this.

**Temporal CG decay.** Ebbinghaus 7-day half-life on CG samples creates automatic recalibration pressure with a conservative failure mode (β_eff drifts toward β₀, lowering N_max). This is operationally sound — staleness errs on the side of caution.

---

## 4. Mathematical weaknesses (current architectural properties)

These are real limitations of the current system. They are not bugs to fix on a deadline; they are structural properties of the math being applied.

### 4.1 The independence-assumption chain

USL, CJT, Krum, and CRDT semantics all assume failure independence in different ways. The literature has individually found each assumption violated for LLM ensembles:

- **CJT independence.** Lefort et al. (arXiv 2409.00094): "CJT predicted accuracy gains do not materialise for LLM ensembles due to significant overlap in decision-making processes." The root mechanism is shared training data: virtually every commercially available LLM was pre-trained on a corpus dominated by Common Crawl, Wikipedia, and a small number of code repositories. When a task activates a specific hallucination vector present in that shared corpus — a plausible but false historical claim, a misremembered API signature, a wrong formula — five adapters from five different providers may produce the same confident wrong answer. The `(1 − ρ)` correction in CJT and the bivariate-CG Phase 2.6 guard reduce but do not eliminate this risk. The definitive mitigation is an empirical oracle (test execution, fact-check API) — without one, the system cannot distinguish correlated hallucination from genuine consensus. `family_constraint = "require_diverse"` (production/strict default) and `single_family_warning` address the most obvious monoculture case; they do not address cross-provider shared training overlap.
- **Krum Byzantine assumption.** arXiv 2512.20184: traditional consensus is designed for deterministic state machines and is incompatible with stochastic multi-agent reasoning. See §3.3 for detail.
- **USL coherency assumption.** arXiv 2602.03794: homogeneous agents saturate fast due to correlated outputs; USL's single-N parameter cannot distinguish homogeneous from heterogeneous pools. The bivariate-CG extension (Hamming + cosine N_eff) is the direct response to this: cosine N_eff distinguishes semantic homogeneity that Hamming CG misses.

The runtime corrects with `(1 − ρ)` in CJT and the bivariate-CG check at Phase 2.6. The corrections do not eliminate the underlying assumption; they bound its damage. When the constraint corpus is sparse, Hamming CG is near-zero regardless of true pool diversity, and compounding four formulas over a noisy base does not increase precision.

### 4.2 The O(N²) synthesis cost is not bypassed by DAG topology

A natural objection: "the system uses a DAG with Kahn's topological sort for orchestration, which is O(N). The `HierarchicalTree` topology further reduces explorer coordination to O(N). So where does USL's O(N²) β term come from? Isn't it a mathematical artefact of a topology the system doesn't actually use?"

The objection conflates two separate cost layers:

- **Orchestration coordination cost** — routing proposals through the DAG, dispatching subtasks, managing the JoinSet. This IS O(N) for HierarchicalTree and is precisely why that topology is chosen when N > N_max. The α coefficient models this serial-fraction cost.
- **Synthesis reconciliation cost** — after proposals arrive, the system must reconcile them. Two O(N²) costs persist regardless of topology:
  1. `CG_mean` is the mean over all `N×(N−1)/2` pairwise Hamming comparisons across surviving proposals. This is a measurement, not an approximation.
  2. The synthesis LLM receives all N surviving proposals concatenated. Identifying cross-proposal constraint conflicts is a pairwise comparison problem. "Lost in the Middle" attention degradation (Liu et al. 2023) is measured as super-linear in N for retrieval of proposals buried deep in context — β_ctx captures this via the context-fill fraction term.

The HierarchicalTree topology reduces α (orchestration). It does not reduce β (synthesis reconciliation). β is fitted from merge-phase timing, which captures the synthesis cost directly. The DAG does not make β irrelevant; it makes α smaller, which shifts the N_max ceiling — which is the intended effect.

### 4.3 The β_eff double-duty problem

`CG_mean` modulates two quantities pointing the same direction:

```
β_eff    = β₀ × (1 − CG_mean)        high CG → low β_eff → N_max grows
rho_mean = 1 − CG_mean               high CG → low ρ → CJT predicts more benefit
```

Both effects say "high agreement → use more agents." But high agreement can mean *good consensus* (truly high p) or *correlated hallucination* (high ρ). One scalar cannot distinguish the two. The runtime mitigates this by tracking cosine N_eff independently and by Talagrand calibration, but the underlying coupling remains.

### 4.4 Correlated hallucination under outlier-resistant merge

The `OutlierResistant{f}` (Krum) breakdown-point proof assumes Byzantine faults are *independent* outliers in distance space. LLMs from the same family produce semantically identical wrong answers whose embeddings cluster tightly. If ≥50% of agents share a correlated hallucination, a Byzantine-resistant selector confidently selects that hallucination as the geometric median — this is the Krum inversion problem: the Byzantine guarantee requires that the corrupted minority is the outlier, but shared training data makes the corrupted majority the centroid. The structural fix (family-diversity gate at provisioning + family rotation on ModeCollapse retry) is not yet implemented. Mitigations currently deployed:

- `single_family_warning` on `CalibrationCompletedEvent` flags monoculture pools.
- `family_constraint = "require_diverse"` (production/strict default) refuses to calibrate a monoculture pool.
- Phase 2.6 cosine-N_eff guard rejects pools with `n_eff_cosine_prior < 1.0 + diversity_threshold`.
- Phase 0 decomposition (Path C, always-on) populates `rejection_criteria` and `focus_mandate` on every slot — each explorer is given a domain-specific failure mode to probe, reducing the probability that all agents share the same false confidence. The adversarial verifier then applies hostile-reviewer scoring.
- `ExplorerSlotConfig` carries distinct `role_frame` strings (via Path C) and distinct `cot_style` values per slot, activating different domain reasoning priors.
- **Token-Jaccard CV detection with joint threshold (2026-05-11/12).** `compute_cv` (`correlated_hallucination.rs`) fires `CorrelatedEnsembleWarning` when **both** `CV(distances) < correlated_hallucination_cv_threshold` (default 0.30) **and** `mean_jaccard_distance < correlated_hallucination_min_jaccard_floor` (default 0.50). The joint gate prevents false positives on uniform-but-diverse ensembles: low CV alone is ambiguous (zero-distance clusters and equidistant-diverse ensembles both produce CV=0); the mean_jaccard floor distinguishes tight clusters (small mean) from uniform spread (large mean). This is a detection-and-retry mechanism, not a prevention: it fires after Phase 3 generation and reroutes through `SraniGroundingChain`-enriched regeneration.
- **SRANI entity-level correlated fabrication detection (2026-05-12).** Above the Jaccard cluster signal, SRANI tracks shared architectural entities absent from the task specification. CFI (Correlated Fabrication Index) = max pairwise overlap of per-proposal ungrounded entity sets. The adaptive sigmoid gate (`srani_gate.rs`) replaces the former static threshold pair: `injection_pressure = σ((CFI − μ) / T)` provides a continuous signal with a self-calibrating midpoint. Grounding escalates through `SraniGroundingChain`: spec-anchor (always) → LLM researcher (tier 0) → web search + LLM distillation (tier 1). The EMA midpoint is deployment-specific and persists across restarts, making the detector self-adapting to the task distribution without requiring oracle data.

These are mitigations, not solutions. The Byzantine guarantee remains conditioned on independent faults. The structural fix is not yet implemented.

### 4.5 LUB is selection, not synthesis

The `ProposalSet` CRDT merge picks a winning proposal and discards the rest. It does not synthesise content. The optional Phase 5a synthesis pipeline (critique → synthesis → re-verify) provides MoA-style generative aggregation when ≥ `synthesis_min_proposals` candidates survive audit, and `HarnessAttribution.synthesis_gain` records the delta. When synthesis does not run (insufficient candidates, disabled, or re-verification regression), the system falls back to LUB selection.

### 4.6 Verification circularity

Phase 3.5 uses a verification adapter (LLM-as-Judge) and Phase 4 uses an auditor adapter. Both are LLMs. Their biases (self-preference, length bias, position bias — Zheng et al. 2023; arXiv 2410.02736; arXiv 2410.21819) propagate through the entire decomposition. `synthesis_gain` is measured by the same verifier, so a verifier blind spot inflates the individual scores and the synthesis score equally.

**Architectural mitigations now in place (2026-05-09):**

- **Rubric separation.** `compiler::compile(manifest, corpus, include_rubric=false)` is the production call. `LlmJudge` rubric text is withheld from the explorer's `system_context`; the verifier retains it via `ConstraintPredicate::LlmJudge`. The explorer must produce proposals from task description and domain expertise, not from rubric scaffolding. This restores structural verifier independence: the verifier is judging proposals against criteria the explorer did not receive.

- **Adversarial verifier framing.** `ADVERSARIAL_EVALUATOR_SYSTEM_PROMPT` replaces the standard rubric-compliance prompt when any explorer slot carries `rejection_criteria` (the default for Path C decomposition). The verifier is instructed to find the single most likely silent failure mode rather than checking criterion adherence.

- **`CoherenceState` per-wave.** `uncovered_domains` and `active_contradictions` are computed and traced after each verification round. `CoherenceIncomplete` is emitted at task close when `!is_closed()`.

- **Multi-variant judge panel (implemented 2026-05-16).** `VerificationPhase::run_with_panel()` fires all panel variants in parallel per constraint and aggregates votes via `aggregate_votes()`. `CrossFamily` panels use supermajority (`quorum_fraction` 0.67); `PersonaOnly` panels require unanimity. Uncertain constraints pass with `score × uncertainty_weight` (0.7). `ConstraintAmbiguityEvent` (fire-and-forget tracing log) signals corpus constraints that consistently produce panel disagreement — the primary corpus quality diagnostic.

**Remaining limitation.** The rubric-separation fix is structurally sound but empirically unvalidated — whether explorers produce genuinely different output when rubric scaffolding is absent depends on model capability. The adversarial verifier framing does not eliminate judge bias; it changes its direction (from "is this compliant?" to "where does this fail?"). Without a Tier 1 oracle, `synthesis_gain` still conflates verifier-preference gain with actual quality improvement. The multi-variant panel reduces same-family self-preference bias but cannot eliminate correlated errors within a `PersonaOnly` panel — the `CrossFamily` path is required for genuine decorrelation. `uncertainty_weight = 0.7` is not calibrated per constraint severity; hard safety constraints where uncertain compliance is near-equivalent to non-compliance may require `uncertainty_weight ≈ 0.5`.

### 4.7 The proxy chain

The flow is:

```
CG_Hamming → β_eff → N_max          (USL chain)
CG_Hamming → (p_proxy, ρ_proxy) → Q (CJT chain, with prediction_basis = Heuristic)
embedding cosine kernel → N_eff      (eigen chain, direct measurement)
```

`compare.py` (benchmark tool — see `docs/architecture/reference.md`) measures `p` directly and switches `prediction_basis` to `Empirical`, breaking part of the proxy chain. The ρ estimate remains a proxy unless an external oracle is configured.

### 4.8 Language choice: Rust over Python

H2AI is written in Rust. The most common objection is that Python enables faster iteration for AI research, and that the Rust borrow checker imposes cognitive overhead on prompt engineering and API exploration.

This is the correct tradeoff for a *different problem*. H2AI is not a research scratchpad or a single-shot LLM call orchestrator. It is a production runtime where:

- **Async correctness is load-bearing** — NATS JetStream consumers, Tokio JoinSets across N parallel agent tasks, snapshot/replay machinery, and SSE streaming all run concurrently. Data races in Python async code are silent; in Rust, the compiler rejects them before the binary is produced.
- **Memory safety at the FFI boundary** — `RealWasmBackend` links `wasmtime` via FFI, and the `llama.cpp` adapter uses raw C bindings. Python FFI errors (use-after-free, null pointer) produce SIGSEGV at 2 AM in production; Rust makes them compile-time errors.
- **PGID process-group kill and signal handling** — the `ShellExecutor` sends `SIGKILL` to a process group via `libc::kill`. Doing this correctly in an async Tokio context requires precise lifetime management that Python cannot express statically.
- **CRDT and event-sourcing correctness** — `ProposalSet` LUB semantics, snapshot/replay ordering, and NATS sequence tracking are invariants that must hold across restarts. Rust's type system encodes them; Python tests approximate them.

The maintenance burden argument is real: contributors unfamiliar with Rust will find the crate graph harder to navigate than a Python monorepo. The counter-claim is that the production bug surface is correspondingly smaller. For deployments where correctness failures mean financial loss (FinTech) or security incidents (DevOps remediation), the cognitive overhead is the right trade.

---

## 5. Empirical gaps

### 5.1 No benchmark numbers in this repository

A benchmark harness exists under `the benchmark harness` with five baselines (`B0` single-shot, `B1` majority vote, `B2` MoA, `B3` Self-MoA, `H2` H2AI). The runs themselves are not in the repository. Until they are executed, the system's quality claim is unfalsified.

The MoA paper (Wang et al. 2024, arXiv 2406.04692) achieves 65.1% on AlpacaEval 2.0 vs GPT-4o's 57.5%. "More Agents Is All You Need" (arXiv 2402.05120) shows majority voting scales log-linearly with N. Self-MoA (Li et al. 2025, arXiv 2502.00674) reports N samples from one strong model beats N diverse models by 6.6%. H2AI must be measured against all three on a representative task suite.

The strongest empirical competitor is MoA. Whether USL-bounded N + bivariate CG + adversarial committee outperforms simpler generative aggregation is the load-bearing empirical question.

### 5.2 Attribution intervals depend on calibration variance

`HarnessAttribution::compute` produces a point estimate. `bootstrap_interval` (1000 resamples over CG samples, when `cg_samples.len() >= 2`) supplies `q_interval_lo`/`q_interval_hi`. Conformal intervals require an oracle signal that is not yet wired into production deployments. Until then, the interval reflects CG-sample variance, not ground-truth uncertainty.

### 5.3 Open empirical questions

- **Does role diversity reduce error correlation?** arXiv 2506.07962 finds error correlation is driven by training data and architecture, not prompting; arXiv 2508.09654 finds temperature alone is insufficient. Whether `ExplorerSlotConfig` system-prompt diversity measurably reduces ρ on a verifiable task set is unknown.
- **Self-MoA vs. multi-adapter.** If Li et al. (2025) holds for H2AI's target task domain, inter-adapter CG optimisation is optimising the wrong variable. Empirical test: same-model × temperature vs multi-family × role on the same task suite.
- **Does USL `N_max` produce better quality/cost than naive scaling?** The core thesis. Without benchmark data, it is plausible but unproven.
- **What is the right `cg_collapse_threshold`?** The default `0.10` is analytical. Empirical data on the CG_embed value at which TAO first-pass rate drops sharply would replace the prior.

---

## 6. Infrastructure boundaries that limit the math

The math is calibrated on the assumption that infrastructure does not silently distort the signal. Several infrastructure choices interact with the math in ways operators must understand.

- **NATS message size.** JetStream's 1 MB default message ceiling is well below modern LLM context budgets (1M-token contexts ≈ 4–8 MB). `payload_offload_threshold_bytes` keeps `system_context` bytes well under 1 MB by replacing them with hash references. When the JetStream ceiling is raised, this knob must be raised in lockstep — otherwise large constraint corpora silently truncate.
- **Event replay.** Recovery time without snapshots is linear in the task's event count. `snapshot_interval_events = 50` keeps replay bounded; raising it linearly increases recovery latency and does not improve correctness.
- **Scheduler starvation.** `CostAwareSpillover` (the default) routes to the next cost tier when a tier's queue exceeds `scheduler_spillover_threshold = 10`. Without spillover, low-tier agents form deep queues while high-tier agents idle. The math is unaffected, but Phase 3 timeouts become a calibration-drift signal that does not actually reflect the pool.
- **Tool-using agent file-system races.** Multiple shell agents share a workspace volume. CRDT event-log coordination does not mediate file-system concurrency. The calibrated α reflects measured serialisation cost, but uncoordinated writes still produce non-deterministic outputs that show up as inflated ρ, not as α. Per-task volume mounts or ephemeral containers are required for correctness.
- **Auditor bias mitigation via shadow auditor (2026-05-11).** Phase 4 now supports a concurrent shadow adapter (`H2AI_SHADOW_AUDITOR_PROVIDER`) from a different family. In observe mode (default), shadow results are recorded as `ShadowAuditorResultEvent` without affecting task outcomes. When per-domain disagreement rate exceeds 5% over 30 observations, the `ShadowAuditorAccumulator` auto-promotes that domain to AND-vote mode — both auditors must approve or the proposal is pruned. Promotions are persisted to NATS KV and survive restarts. `explorer_verification_family_match` still flags the monoculture case; the shadow auditor addresses cross-family bias that the family gate cannot see.
- **Multi-variant judge panel (implemented 2026-05-16).** Phase 3.5 now uses `JudgePanel` with parallel variant dispatch. The `[judge_panel]` config section controls `quorum_fraction`, `uncertainty_weight`, `persona_temperatures`, and `ambiguity_threshold`. The `PersonaOnly` fallback applies when only one adapter family is deployed; this is a weaker signal than `CrossFamily` because same-family persona variants share training correlations. The `ambiguity_threshold` counter is a function of how many proposals see uncertain verdicts per wave — in low-traffic deployments, this counter accumulates slowly and may underreport corpus quality issues.
- **Embedding model is required for the bivariate-CG safety net.** Without `fastembed-embed` and a configured model, the runtime falls back to a closed-form `n_eff_cosine_prior` and disables Phase 2.6. The system still runs, but the bivariate-CG guarantees are downgraded to univariate Hamming. **This degradation is now loud (2026-05-11):** `DiversityGuardDegradedEvent` is emitted to NATS, a startup warning fires, and `require_bivariate_cg = true` will fail the task rather than silently proceeding.

---

## 7. External landscape

The combination of *USL-bounded N* + *bivariate CG* + *MAPE-K failure-mode routing* + *CRDT-convergent merge with optional generative synthesis* + *Harness Attribution* has no direct analogue in published frameworks.

Layer positioning:

- **Inference layer** (vLLM, TGI, llama.cpp, Ollama) — H2AI delegates here via adapters.
- **Adapter-internal optimisation** (DSPy) — DSPy compiles prompts and few-shot weights inside one adapter. It is complementary to H2AI: a DSPy-optimised adapter can sit inside `IComputeAdapter` while H2AI orchestrates the swarm.
- **Distributed compute fabric** (Ray, Kubernetes) — Ray and K8s map agents to hardware. H2AI decides *how many* agents and *what roles* before delegating.
- **Topology and coordination layer (H2AI's home)** — bounding N, calibrating ρ, routing failure modes, attributing quality.
- **Agentic frameworks** (LangChain/LangGraph, AutoGen, CrewAI, OpenAI Swarm) — these compose tools and memory. They do not bound N from measurement, do not classify failure modes, and do not produce a quality decomposition.
- **Empirical aggregators** (MoA, Self-MoA) — the strongest empirical competitors. MoA wins on simplicity; H2AI's claim is that calibrated bounding outperforms unbounded aggregation on quality/cost. The claim is empirically unverified at the time of writing.

### Key papers to cite and differentiate against

Papers are grouped by the architectural concern they address. Each entry states what H2AI specifically uses the paper for and whether the claim is validated, contested, or pending empirical testing.

#### Ensemble quality theory (CJT foundation)

| Paper | H2AI use | Status |
|---|---|---|
| Condorcet (1785) | The `condorcet_quality(N, p, ρ)` formula is the primary quality bound. Every ensemble sizing decision flows from this theorem. | Mathematically proven. Domain transfer to LLMs is the open question. |
| Nitzan & Paroush (1982) | Extends CJT to non-uniform competence; justifies role-differentiated explorer slots with distinct `focus_mandate`. | Proven. |
| Ladha (1992) | Adds the `(1 − ρ)` correlation correction to `Q_ind`. Source of the `ρ` term in `condorcet_quality`. Without this correction, CJT over-predicts quality when agents share training data. | Proven. Applied in `h2ai-types/src/sizing.rs`. |
| Lefort et al. (2024) — arXiv 2409.00094 | **Adversarial result.** "CJT predicted accuracy gains do not materialise for LLM ensembles due to significant overlap in decision-making processes." Motivates the `(1−ρ)` correction and the bivariate-CG guard. Shows plain majority-vote ensembles underperform. | Published result. H2AI's `ρ` correction directly addresses this. Empirical validation pending. |
| Bradley (2024) — arXiv 2411.01539 | *"LLMs and the Madness of Crowds."* LLM errors are **systematically correlated** across architecturally similar models, especially on shared-corpus hallucinations. Justifies the online ρ EMA (`rho_ema.rs`) replacing the CG proxy. | Published result. Motivates empirical ρ measurement. |
| Elgabry & Hamdi (2025) — arXiv 2512.17630 | *"Confidence-Credibility Aware Weighted Ensembles."* Explicitly invokes CJT and confirms the diversity maintenance requirement — CJT advantage collapses without active error decorrelation. Validates the Phase 0 role decomposition as the structural diversity mechanism. | Published result. Supports Phase 0 decomposition design. |

#### Scalability and coordination cost (USL foundation)

| Paper | H2AI use | Status |
|---|---|---|
| Gunther (1993) | USL `X(N) = N / (1 + α(N−1) + β·N(N−1))` is the coordination cost model. `N_max = √((1−α)/β_eff)` is derived by setting `dX/dN = 0`. Applied in `h2ai-autonomic/src/calibration.rs`. | Proven for throughput systems. Domain transfer to LLM quality is contested — USL was derived for CPU/network throughput; no published work validates N_max as a quality ceiling for LLM ensembles. |
| Gunther (2008) — arXiv 0808.1431 | Foundational derivation of β as the **coherency synchronisation cost fraction per adapter pair**. Used to justify `β_eff = β₀ × (1 − CG_mean)`: CG_mean is the fraction of constraints where adapters agree, so `(1 − CG_mean)` is the expected conflict fraction, which is proportional to β. | Foundational. The linear proportionality assumption is an open empirical question. |
| Hamann & Reina (2020) — arXiv 2006.04969 | Applies USL and Amdahl's Law to robot swarms — the closest published bridge to autonomous agent systems. **Key finding:** USL describes swarm *throughput*; quality metrics follow different scaling laws. This is the primary justification for treating N_max as a cost ceiling, not a quality predictor; `n_it_optimal` (`N_IT = ceil(log(0.5)/log(1−ρ))`) is the information-theoretic quality target, and planning logic uses `min(N_IT, N_max)`. | Published result. Directly motivates the N_IT / N_max split in `n_it_optimal`. |
| Nowak (2025) — arXiv 2509.19489 | Derives the optimal compute allocation for self-consistency under budget B = m×n: optimum is `m, n ∝ √B`. Independently confirms that ensemble sizing has a sweet spot and unbounded scaling is sub-optimal. Complements the USL ceiling. | Published result. Consistent with N_max design. |
| arXiv 2512.08296 | *"Towards a Science of Scaling Agent Systems."* Coordination overhead model consistent with USL framing; provides a secondary validation that agent coordination cost is a real, measurable phenomenon. | Published result. Contextual support. |

#### Ensemble diversity and composition

| Paper | H2AI use | Status |
|---|---|---|
| Li et al. (2025) — arXiv 2502.00674 (Self-MoA) | *"Rethinking Mixture-of-Agents: Is Mixing Different LLMs Beneficial?"* Self-MoA (single model, temperature variation) beats cross-model MoA by 6.6% on AlpacaEval 2.0. This is **H2AI's primary empirical threat**: if temperature diversity alone is sufficient, the cross-family coordination cost is pure overhead. H2AI's counter-claim is that MAPE-K constraint enforcement adds value that temperature spread alone cannot provide on constraint-heavy tasks. | Published result. Empirical comparison pending — structured H2-P vs. B3 experiment protocol not yet run. |
| Xie et al. (2025) — arXiv 2505.24442 (RMoA) | *"Optimising MoA through Diversity Maximisation and Residual Compensation."* Identifies explicit diversity maximisation in agent selection as the single biggest quality lever. Validates Phase 0 decomposition as the correct structural intervention. The residual compensation mechanism is related to the critique-synthesis pass in Phase 5a. | Published result. Directly supports Phase 0 design. |
| arXiv 2601.16715 | *"Dynamic Expert-Guided Model Averaging."* LLM-guided ensemble weighting outperforms uniform averaging when the LLM has partial structural knowledge. Validates Phase 0 mandate-based slot selection over random role assignment. | Published result. Supports role-frame design. |
| arXiv 2503.03535 | *"Trade-offs in Ensembling, Merging, and Routing."* Systematic comparison: ensembling dominates on distribution-shift tasks; routing wins in-distribution. Constraint-compliance tasks under novel specifications are distribution-shift scenarios — validates the ensemble approach for H2AI's primary use case. | Published result. Supports Coverage quadrant design. |
| arXiv 2402.05120 | *"More Agents Is All You Need."* Shows majority voting scales log-linearly with N. Establishes the naive scaling baseline H2AI must beat. | Published result. Baseline for empirical comparison. |
| arXiv 2602.03794 | *"Understanding Agent Scaling via Diversity."* Homogeneous agents saturate quickly due to correlated outputs; USL's single-N parameter cannot distinguish homogeneous from heterogeneous pools. Direct motivation for the bivariate-CG extension: cosine N_eff distinguishes semantic homogeneity that Hamming CG misses. | Published result. Core motivation for Phase 2.6 cosine guard. |
| arXiv 2506.07962 | Error correlation is driven by training data and architecture, not by prompting strategy. Challenges the assumption that role-frame diversity in prompts meaningfully reduces ρ. If this holds for H2AI's task domain, `ExplorerSlotConfig` diversity provides behavioural diversity without reducing error correlation — MAPE-K enforcement, not diversity, would be the value driver. | Published result. **Open empirical question.** |
| arXiv 2508.09654 | Temperature alone is insufficient for diversity; training loss distribution governs. Supports the claim that multi-family (Coverage quadrant) is necessary for genuine error decorrelation, not just temperature-spread within one model. | Published result. Supports Coverage quadrant design. |
| JMLR 2023 (unified diversity decomposition) | Unified bias-variance-diversity decomposition: `ensemble error = bias² + variance/N + covariance×(N−1)/N`. Failed proposals increase the covariance term disproportionately. This is the theoretical basis for j_eff's "double penalty": failures reduce both `n_valid` (fewer Condorcet voters) and `filter_ratio` (lower per-voter competence), penalising covariance twice — once through count, once through individual quality. | Proven (regression). Domain transfer to LLM quality scores is the open question. Applied in `compute_j_eff` design rationale. |
| Wang et al. (2024) — arXiv 2406.04692 (MoA) | *"Mixture-of-Agents enhances large language model capabilities."* Achieves 65.1% on AlpacaEval 2.0 vs GPT-4o's 57.5% with generative aggregation. Primary baseline: H2AI's claim is that USL-bounded + bivariate-CG + MAPE-K outperforms unbounded MoA on quality/cost. | Published result. Empirical comparison pending. |

#### Merge strategy and Byzantine fault tolerance

| Paper | H2AI use | Status |
|---|---|---|
| Blanchard et al. (2017) — NeurIPS | *"Machine learning with adversaries: Byzantine tolerant gradient descent (Krum)."* Theorem 2 gives the Krum breakdown-point proof: with f Byzantine agents and N−2f independent non-Byzantine agents, Krum selects a non-Byzantine candidate. H2AI's `OutlierResistant{f}` merge strategy is Krum-style selection applied to proposal embeddings. The `krum_f` parameter is the breakdown-point budget. | Proven. Applied in `h2ai-orchestrator/src/engine.rs`. The proof assumes independent faults — correlated hallucination violates this. |
| arXiv 2512.20184 | Traditional consensus protocols (PBFT) are designed for deterministic state machines and are incompatible with stochastic multi-agent LLM reasoning. Justifies H2AI's use of fractional fingerprint agreement (`bft_threshold`) rather than full PBFT. Documents the impossibility result for deterministic consensus in stochastic agent systems. | Published result. Motivates the fingerprint-agreement design. |
| arXiv 2511.10400 | *"Rethinking the Reliability of Multi-Agent Systems via BFT."* Frames multi-agent reliability in terms of Byzantine fault tolerance. Contextual grounding for the merge tier's three-strategy escalation. | Published result. Contextual. |
| arXiv 2507.14928 | *"Byzantine-Robust Decentralised LLM Coordination."* Closest architectural prior art to H2AI. Explores Byzantine-robust consensus for LLM agent networks. H2AI's differentiation: USL-bounded N + bivariate CG + typed constraint gates before merge. | Published result. Architectural comparison point. |
| Pillutla et al. (2019) — arXiv 1912.13445 | *"Robust aggregation for federated learning."* Weiszfeld's iterative re-weighted least-squares algorithm for geometric median computation. H2AI's `OutlierResistant` merge uses the Krum variant rather than geometric median, but this paper provides the theoretical foundation for `O(1/t)` convergence per iteration. | Published result. Theoretical foundation for robust merge. |
| Vardi & Zhang (2000) — PNAS | *"The multivariate L1-median and associated data depth."* Geometric median theory underlying the `OutlierResistant` merge strategy's distance-space selection. | Proven. Mathematical foundation. |
| arXiv 2510.18893 (CodeCRDT) | Independently confirms that CRDT semantics apply to LLM-agent output ordering. Validates the `ProposalSet` LUB design: CRDT merge on `(generation, score)` pairs provides crash-safe idempotency. | Published result. Independent validation of CRDT design. |

#### Verification and evaluation biases

| Paper | H2AI use | Status |
|---|---|---|
| Zheng et al. (2023) — arXiv 2306.05685 | *"Judging LLM-as-a-Judge with MT-Bench."* Foundational LLM-as-Judge bias paper. Documents self-preference bias, length bias, and position bias in LLM evaluators. Motivates `VerifierExplorerFamilyConflict` gate and rubric-separation (`include_rubric=false`). | Published result. Applied in engine family-gate design. |
| arXiv 2410.02736 | Self-preference bias in LLM evaluation: models rate their own outputs higher. Motivates the `family_constraint = "require_diverse"` production default — the verifier and explorer must come from different families. | Published result. Applied in family-gate design. |
| arXiv 2410.21819 | Position bias and length bias in LLM-as-Judge evaluations. Documents that judges favour longer responses and proposals appearing earlier in context. Motivates the adversarial framing (`ADVERSARIAL_EVALUATOR_SYSTEM_PROMPT`) as a counter-bias measure. | Published result. Applied in verifier prompt design. |
| Li et al. (2025) — arXiv 2502.01534 | *"Preference Leakage: A Contamination Problem in LLM-as-a-Judge."* When the same LLM family generates training data AND judges it, the judge systematically favours the family's own output style. Motivates cross-family judge panel. | Published result. Applied — implemented 2026-05-16 via `JudgePanel` cross-family panel + `PersonaOnly` fallback. |
| Dorner et al. (2024) — arXiv 2410.13341 | *"LLM as Judge Won't Beat Twice the Data."* Proves LLM judge self-preferencing bias is irreducible by prompt engineering alone — only architectural separation addresses it. Validates the `VerifierExplorerFamilyConflict` hard gate as the correct structural fix rather than a prompt-level workaround. | Published result. Validates family-gate design. |
| arXiv 2504.14716 | Pairwise evaluation is less biased than pointwise scoring but introduces Condorcet cycles at scale. H2AI uses pointwise constraint compliance scores (not pairwise ranking), which avoids Condorcet cycles but inherits pointwise biases. Contextual awareness for the verification design. | Published result. Contextual awareness. |
| Wagner et al. (2024) — arXiv 2410.11594 | *"Black-box Uncertainty Quantification for LLM-as-a-Judge."* Sampling-based confidence intervals for LLM judge scores without white-box model access. The path to calibrated uncertainty intervals on `VerificationScoredEvent`. Currently unimplemented — `verification_score` is a point estimate. | Published result. Implementation path for verification confidence intervals. |

#### Diversity measurement and calibration

| Paper | H2AI use | Status |
|---|---|---|
| Choueifaty & Coignard (2008) — *Journal of Portfolio Management* 35(1) | Participation ratio `(Σλ)² / Σλ²` from portfolio diversification theory. H2AI applies this to the eigendecomposition of the embedding cosine kernel: `N_eff = (Σλ)² / Σλ²` is the number of independent directions in the pool's semantic embedding space. At full independence `N_eff = N`; at full correlation `N_eff = 1`. This is a direct measurement, not an approximation. | Proven in portfolio context. Domain transfer to embedding kernels is direct — same mathematics. |
| Leutbecher & Palmer (2008) — *Journal of Computational Physics* | *"Ensemble forecasting."* Source of the Talagrand rank histogram diagnostic for ensemble calibration. Flat histogram = calibrated spread, U-shape = over-confident (increase τ), Λ-shape = under-dispersed (decrease τ). Borrowed from numerical weather prediction where the technique is a standard calibration tool. Applied in `h2ai-autonomic/src/calibration.rs`. | Proven in meteorological context. Domain transfer is direct — calibration histograms are domain-agnostic. |

#### Context window and attention limitations

| Paper | H2AI use | Status |
|---|---|---|
| Liu et al. (2023) | *"Lost in the Middle: How Language Models Use Long Contexts."* Retrieval quality for proposals buried in a long context degrades super-linearly. H2AI models this via `β_ctx(N) = β_eff × (1 + γ × fill(N))` — the context-fill fraction amplifies the coordination cost as N grows. This is the physical motivation for the O(N²) synthesis cost despite O(N) orchestration topology. | Published result. Applied in `β_ctx` formula in `math.md` §2.3. |

#### Conformal prediction and uncertainty quantification

| Paper | H2AI use | Status |
|---|---|---|
| arXiv 2406.09714 | *"LLM Validity via Conformal Prediction."* Path to oracle-grounded confidence intervals with coverage guarantees. H2AI's `q_confidence` is currently a self-assessed decomposition score, not a conformal interval. This paper provides the methodology to upgrade it once an oracle accumulator has enough data. | Published result. Future implementation path — not yet implemented. |

#### Thinking loop and multi-round reasoning

| Paper | H2AI use | Status |
|---|---|---|
| Yao et al. (2023) — arXiv 2309.13007 (ReConcile) | *"ReConcile: Round-Table Conference Improves Reasoning via Consensus among Diverse LLMs."* ACL 2024. Multi-model round-table where agents discuss, attempt to persuade, and converge via confidence-weighted voting. Key finding: model diversity is the primary quality driver — a collection of mediocre reasoners interacting outperforms a single strong reasoner (+11.4% over prior multi-agent baselines). Applied in the thinking loop synthesis stage: archetype outputs are weighted by self-reported confidence; unresolved "tensions" (the "unresolved debate") drive the next iteration's archetype selection. | Published result. Applied in `thinking_loop.rs` synthesis. |
| Chen et al. (2025) — arXiv 2503.19855 (Think Twice) | *"Think Twice: Multi-Round Test-Time Thinking."* Iterative refinement where each round receives only a distilled summary of prior reasoning, not intermediate steps, forcing independent re-evaluation. Consistent gains on DeepSeek-R1 and QwQ-32B. Improvements plateau after ~4 rounds (diminishing returns). Applied: each thinking loop iteration receives only `ThinkingReport.shared_understanding` + `tensions`, not raw archetype outputs, matching the "discard intermediates" principle. `max_thinking_iterations = 5` gives one buffer iteration beyond the empirical plateau. | Published result. Applied in convergence design of `thinking_loop.rs`. |
| Chen et al. (2024) — arXiv 2402.10178 (TDAG) | *"TDAG: A Multi-Agent Framework based on Dynamic Task Decomposition and Agent Generation."* Neural Networks 2025. Dynamically generates a tailored sub-agent per subtask, adjusting subsequent agents based on what earlier agents resolved — agent generation in response to the problem, not pre-configured globally. Applied: archetype selection is the "agent generation" step; on iteration N>1, selection adapts to unresolved tensions from the synthesis report, directly mirroring TDAG's adaptive subtask adjustment. | Published result. Applied in archetype selection design. |
| Tseng et al. (2025) — arXiv 2506.06254 (PersonaAgent) | *"PersonaAgent: Test-Time Persona Alignment for LLM Agents."* NeurIPS 2025. Agent iteratively rewrites its persona prompt to minimise a "textual loss" between simulated and ground-truth responses — rapid convergence with no fine-tuning. Shows persona quality is a continuous variable optimisable at inference time. Applied: archetypes whose perspectives did not contribute to resolving tensions in the prior iteration are replaced or sharpened in the next iteration's selection prompt, implementing inference-time persona refinement. | Published result. Applied in iteration N>1 archetype refinement. |
| A-HMAD (2025) | *Adaptive Heterogeneous Multi-Agent Debate.* Operationalises Minsky's Society of Mind as diverse LLM agents with distinct latent-space reasoning trajectories. Introduces dynamic debate routing: specific tensions are directed to specific agent pairs rather than all-to-all discussion, reducing cross-talk and improving coverage orthogonality. Applied: each archetype's `scope` field implements this routing — the archetype reasons only within its assigned problem slice; on subsequent iterations, tensions are injected to direct new archetypes at precisely the gaps the prior synthesis identified. | 2025 preprint. Applied in archetype `scope` design and tension injection. |

#### Judge panel and bias mitigation

| Paper | H2AI use | Status |
|---|---|---|
| Es et al. (2024) — arXiv 2404.18796 (PoLL) | *"Replacing Judges with Juries."* Cross-family panel of diverse models outperforms single large judge at 1/7th cost. Basis for `JudgePanel` `CrossFamily` construction rule: one variant per family, cap 3. | Published result. Applied in `judge_panel.rs`. |
| Zhao et al. (2026) — arXiv 2603.00039 (CARE) | *"Confounder-Aware Aggregation."* Majority vote amplifies bias when judges share latent confounders. Motivates the split aggregation rule: `CrossFamily` → supermajority; `PersonaOnly` → unanimity. | Published result. Applied in `aggregate_votes()`. |
| Pezeshkpour et al. (2026) — arXiv 2605.01630 (Prosa) | *"Rubric-Based Evaluation."* Binary rubric decomposition removes judge-model bias sensitivity more than cross-family diversity. The constraint corpus as analytic rubric is the primary mitigation. | Published result. Applied — constraint corpus structure is the primary bias guard. |
| arXiv 2604.00477 (Logarithmic Scores) | Quality improves logarithmically with panel size; 3 judges captures ~90% gain. Justifies panel cap of 3. | Published result. Applied in panel cap design. |
| arXiv 2603.00077 (Autorubric) | Binary criteria + ensemble evaluation + few-shot calibration as the correct combination for rubric-based ensemble judging. | Published result. Applied in persona design rationale. |
| arXiv 2412.12509 | High temperature increases variance without improving accuracy for evaluation tasks — interpretive diversity (persona) preferable to stochastic diversity (temperature). Motivates `persona_temperatures` capped at 0.4. | Published result. Applied in `persona_temperatures` default. |
| Mirzadeh et al. (2025) — arXiv 2512.05379 | *"Mitigating Self-Preference by Authorship Obfuscation."* Stylistic obfuscation only partially helps; family boundary is the effective mitigation. | Published result. Validates `CrossFamily` over persona-only as primary debiasing path. |
| arXiv 2404.13076 | LLM evaluators recognise and favour their own generations even without explicit authorship cues — same-family judges inflate scores. | Published result. Motivates `CrossFamily` panel over single-family variants. |

#### Verifier quality and ensemble verifiers

| Paper | H2AI use | Status |
|---|---|---|
| Lee et al. (2026) — arXiv 2604.18547 (FUSE) | *"Ensembling Verifiers with Zero Labeled Data."* Unsupervised ensembling of LLM verifiers using only inter-verifier agreement structure. Directly applicable to H2AI's cold-start problem: before domain-specific oracles exist, FUSE-style inter-verifier agreement can produce a calibrated ensemble verification score without ground truth labels. | Published result. Future implementation path — not yet implemented. |
| Menet et al. (2026) — arXiv 2605.07775 (POETS) | *"Uncertainty-Aware LLM Optimisation via Policy Ensembles."* Uses Thompson Sampling over KL-regularised reward models for ensemble policy updates. The KL framework directly applies to the Talagrand histogram correction for τ-spread adjustment. Motivates the `Δτ = η × (U_score − Λ_score)` update rule for automated τ-spread calibration — not yet implemented. | Published result. Future implementation path — not yet implemented. |
| arXiv 2303.16634 (G-Eval) | *"G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment."* G-Eval-style chain-of-thought rubric is the structural model for `LlmJudge` constraint predicates. Each `ConstraintDoc.predicate = LlmJudge { rubric }` follows the G-Eval pattern: rubric defines the scoring chain-of-thought, the verifier LLM executes it and returns a score. | Published result. Applied in constraint predicate design. |

---

## 8. References

### Foundational mathematics

- **Condorcet, M. J. A. N. (1785).** *Essai sur l'application de l'analyse à la probabilité des décisions rendues à la pluralité des voix.* The original Condorcet Jury Theorem: a committee of independent competent voters reaches the correct majority decision with probability approaching 1 as N → ∞.
- **Nitzan, S. & Paroush, J. (1982).** *Optimal decision rules in uncertain dichotomous choice situations.* International Economic Review, 23(2). Extends CJT to non-uniform voter competence; justifies differentiated role assignment.
- **Ladha, K. K. (1992).** *The Condorcet jury theorem, free speech, and correlated votes.* American Journal of Political Science, 36(3). Adds the `(1 − ρ)` correlation correction. Source of the `Q(N, p, ρ) = p + (Q_ind(N,p) − p)(1−ρ)` formula implemented in `h2ai-types/src/sizing.rs::condorcet_quality`.
- **Choueifaty, Y. & Coignard, Y. (2008).** *Towards maximum diversification.* Journal of Portfolio Management, 35(1). Participation ratio `N_eff = (Σλ)² / Σλ²` from portfolio theory. H2AI applies this to LLM embedding cosine kernels to measure the number of semantically independent perspectives in an adapter pool.
- **Gunther, N. J. (1993).** *A simple capacity model of massively parallel transaction systems (The Universal Scalability Law).* CMG. The throughput model `X(N) = N / (1 + α(N−1) + βN(N−1))` whose peak gives the ensemble ceiling `N_max = √((1−α)/β_eff)`.
- **Gunther, N. J. (2008).** *Guerrilla capacity planning.* arXiv 0808.1431. Derives β as the coherency synchronisation cost fraction per agent pair. H2AI maps β to constraint conflict resolution cost via `β_eff = β₀ × (1 − CG_mean)`.
- **Weiszfeld, E. (1937).** *Sur le point pour lequel la somme des distances de n points donnés est minimum.* Tôhoku Mathematical Journal. Foundational algorithm for geometric median computation (`OutlierResistant` merge uses Krum as a proxy; Weiszfeld provides the theoretical L1-median basis).
- **Vardi, Y. & Zhang, C.-H. (2000).** *The multivariate L1-median and associated data depth.* Proceedings of the National Academy of Sciences. Geometric median theory underpinning the `OutlierResistant{f}` merge strategy's distance-space proposal selection.
- **Levandowsky, M. & Winter, D. (1971).** *Distance between sets.* Nature, 234(5323). Jaccard distance satisfies the triangle inequality and is a valid metric. Referenced in correlated hallucination detection (`Token-Jaccard CV` in `correlated_hallucination.rs`).

### Scalability and coordination overhead

- **Amdahl, G. M. (1967).** *Validity of the single processor approach to achieving large scale computing capabilities.* AFIPS. Serial fraction model underlying the α term in USL. The α in H2AI captures constraint compilation, topology provisioning, and merge — genuinely serial phases that do not parallelize.
- **Brooks, F. P. (1975).** *The Mythical Man-Month.* Addison-Wesley. Brook's Law: communication channels grow as N(N−1)/2 in human teams. H2AI draws the structural analogy to LLM agent pairwise reconciliation cost.
- **Hamann, H. & Reina, A. (2020).** *Swarm robotics: A review from the swarm engineering perspective.* arXiv 2006.04969. Applies USL and Amdahl's Law to autonomous swarm agents. Key finding: USL models throughput; quality (task success rate) follows different scaling laws. This is the primary justification for treating `N_max` as a cost ceiling and `N_IT` as the quality target.
- **Nowak, S. (2025).** *Optimal compute allocation for self-consistency.* arXiv 2509.19489. Derives `m, n ∝ √B` as the optimal budget split between proposal count and verification passes. Consistent with H2AI's `N_max` design and independently motivates the existence of an ensemble size sweet spot.

### LLM ensembles and multi-agent systems

- **Wang, J. et al. (2024).** *Mixture-of-Agents enhances large language model capabilities.* arXiv 2406.04692. Generative aggregation baseline. Achieves 65.1% on AlpacaEval 2.0 vs GPT-4o's 57.5%. The MoA architecture that H2AI's Phase 5a synthesis pass is directly compared to.
- **Li, J. et al. (2025).** *Rethinking Mixture-of-Agents: Is Mixing Different LLMs Beneficial? (Self-MoA).* arXiv 2502.00674. Self-MoA with a single strong model beats cross-family MoA by 6.6% on AlpacaEval. The primary empirical threat to H2AI's diversity claim. H2AI's counter-claim: MAPE-K constraint enforcement provides value above temperature spread on constraint-heavy tasks — a direct empirical test is needed.
- **Xie, et al. (2025).** *RMoA: Optimising MoA through Diversity Maximisation and Residual Compensation.* arXiv 2505.24442. Explicit diversity maximisation in agent selection is the single biggest quality lever. Validates H2AI's Phase 0 role decomposition. The "residual compensation" mechanism is the MoA analogue of H2AI's Phase 5a synthesis critique pass.
- **arXiv 2601.16715.** *Dynamic Expert-Guided Model Averaging.* LLM-guided ensemble weighting outperforms uniform averaging when the LLM's structural knowledge about the problem domain is even partially correct. Validates Phase 0's `focus_mandate` assignment as the structural basis for role-differentiated weighting.
- **arXiv 2503.03535.** *Trade-offs in Ensembling, Merging, and Routing.* Systematic comparison: ensembling dominates on distribution-shift tasks; routing wins in-distribution. Constraint-compliance tasks under novel architectural specifications are precisely distribution-shift scenarios, making H2AI's ensemble approach architecturally correct for its target domain.
- **arXiv 2402.05120.** *More Agents Is All You Need.* Majority voting scales log-linearly with N. Establishes the naive scaling baseline H2AI must outperform on quality/cost ratio.
- **arXiv 2512.08296.** *Towards a Science of Scaling Agent Systems.* Coordination overhead in multi-agent systems is a real, measurable phenomenon consistent with USL framing. Provides a secondary theoretical anchor for the N_max ceiling.
- **Lefort, P. et al. (2024).** *Empirical limits of the Condorcet Jury Theorem in LLM ensembles.* arXiv 2409.00094. CJT predicted accuracy gains do not materialise in naive LLM majority-vote ensembles due to shared training-data overlap. The `(1 − ρ)` correction in `condorcet_quality` and the bivariate-CG Phase 2.6 guard directly address this finding.
- **Bradley, N. (2024).** *LLMs and the Madness of Crowds.* arXiv 2411.01539. LLM errors are systematically correlated across architecturally similar models — the same hallucination vectors activate across provider families when the shared pre-training corpus contains the same plausible false claims. Motivates the online ρ EMA (`rho_ema.rs`) as the only empirically grounded ρ estimate.
- **Elgabry, O. & Hamdi, A. (2025).** *Confidence-Credibility Aware Weighted Ensembles.* arXiv 2512.17630. Explicitly applies CJT to LLM ensembles and confirms that CJT advantage holds only when error diversity is actively maintained — minimising parameter convergence (distinct architectures) is essential. Validates H2AI's cross-family requirement.
- **arXiv 2602.03794.** *Understanding Agent Scaling via Diversity.* Homogeneous agents saturate quickly because correlated outputs provide diminishing marginal information. USL's single-N parameter cannot distinguish homogeneous from heterogeneous pools. Direct motivation for the bivariate-CG extension: cosine N_eff distinguishes semantic homogeneity that Hamming CG misses.
- **arXiv 2506.07962.** Error correlation in LLM ensembles is driven by training data distribution and model architecture, not by prompting strategy. This challenges the assumption that `role_frame` diversity in prompts measurably reduces ρ. If confirmed for H2AI's task domain, the value driver is MAPE-K enforcement, not prompt diversity — a direct falsification target for experiments.
- **arXiv 2508.09654.** Temperature variation is insufficient to produce independent errors; training loss distribution governs the correlation structure. Supports the Coverage quadrant (cross-family adapters) as the structural mechanism for genuine error decorrelation, not the Precision quadrant (τ-spread within one family).

### Byzantine fault tolerance and CRDT merge

- **Blanchard, P., El Mhamdi, E. M., Guerraoui, R., & Stainer, J. (2017).** *Machine learning with adversaries: Byzantine tolerant gradient descent.* NeurIPS. Theorem 2 proves Krum's breakdown-point: with f Byzantine agents and N−2f independent non-Byzantine agents, Krum selects a non-Byzantine candidate. H2AI's `OutlierResistant{f}` merge strategy applies this to proposal embeddings via `krum_f`. The proof assumes fault independence — correlated hallucination violates this assumption.
- **Pillutla, K., Kakade, S. M., & Harchaoui, Z. (2019).** *Robust aggregation for federated learning.* arXiv 1912.13445. Weiszfeld's iterative algorithm for computing the geometric median with `O(1/t)` convergence per iteration. H2AI uses the Krum variant rather than full geometric median, but this paper provides the convergence theory for robust aggregation under Byzantine agents.
- **arXiv 2512.20184.** Traditional consensus protocols (PBFT) are incompatible with stochastic multi-agent LLM reasoning — they are designed for deterministic state machines where equivocation is adversarial, not stochastic. Justifies H2AI's use of fractional fingerprint agreement (`bft_threshold` on constraint satisfaction vectors) rather than full PBFT, and motivates the `bft_threshold` framing note in architecture documentation.
- **arXiv 2511.10400.** *Rethinking the Reliability of Multi-Agent Systems via Byzantine Fault Tolerance.* Frames multi-agent reliability in BFT terms. Contextual grounding for the three-tier merge escalation: `ScoreOrdered → ConsensusMedian → OutlierResistant`.
- **arXiv 2507.14928.** *Byzantine-Robust Decentralised LLM Coordination.* Closest architectural prior art to H2AI. H2AI's differentiators: USL-bounded N before generation starts, bivariate CG typed constraint gates at Phase 2.5, and MAPE-K routing of failure modes rather than post-hoc BFT repair.
- **arXiv 2510.18893 (CodeCRDT).** Independently confirms that CRDT semantics apply to LLM-agent output ordering. Validates the `ProposalSet` LUB design: CRDT merge on `(generation, score)` pairs provides crash-safe idempotency and TAO-monotonic ordering without centralised coordination.

### Verification and evaluation biases

- **Zheng, L. et al. (2023).** *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* arXiv 2306.05685. Foundational LLM-as-Judge evaluation. Documents self-preference bias (models rate own-family outputs higher), length bias (longer responses score better independent of quality), and position bias (earlier proposals score higher). Motivates `VerifierExplorerFamilyConflict` gate and `include_rubric=false` rubric separation.
- **arXiv 2410.02736.** Self-preference bias in LLM evaluation — models systematically rate outputs from their own family higher, even when they cannot identify the source. Motivates the `family_constraint = "require_diverse"` production default.
- **arXiv 2410.21819.** Position and length biases in LLM-as-Judge. The adversarial verifier framing (`ADVERSARIAL_EVALUATOR_SYSTEM_PROMPT`) partially counteracts these biases by instructing the verifier to look for failures rather than confirm compliance.
- **Li, J. et al. (2025).** *Preference Leakage: A Contamination Problem in LLM-as-a-Judge.* arXiv 2502.01534. When the same LLM family generates training data AND judges it, a self-amplifying preference loop forms — the judge reinforces outputs that match the family's own style. Motivates cross-family judge rotation as the structural fix; implemented via `JudgePanel` with `CrossFamily` supermajority and `PersonaOnly` unanimous fallback.
- **Dorner, F. et al. (2024).** *LLM as Judge Won't Beat Twice the Data.* arXiv 2410.13341. LLM judge self-preferencing bias is irreducible through prompt engineering alone. Only architectural separation (different family for judge and defendant) provides a structural fix. Validates the `VerifierExplorerFamilyConflict` hard gate as the correct — and only adequate — response.
- **arXiv 2504.14716.** Pairwise evaluation is less biased than pointwise scoring but introduces Condorcet cycles at scale. H2AI uses pointwise constraint compliance scores; this paper provides the theoretical reason pairwise ranking was not chosen.
- **Wagner, M. et al. (2024).** *Black-box Uncertainty Quantification for LLM-as-a-Judge.* arXiv 2410.11594. Sampling-based confidence intervals for LLM judge scores without white-box access. Provides the methodology for upgrading `verification_score` from a point estimate to a calibrated interval with coverage guarantees. Currently unimplemented.
- **arXiv 2303.16634 (G-Eval).** *G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment.* G-Eval-style chain-of-thought rubrics for evaluation. The `ConstraintDoc.predicate = LlmJudge { rubric }` structure follows this pattern: each constraint defines a scoring rubric that the verifier executes as a structured chain-of-thought.

### Conformal prediction and uncertainty quantification

- **arXiv 2406.09714.** *LLM Validity via Conformal Prediction.* Conformal prediction with LLM-generated calibration sets. Provides the methodology for oracle-grounded confidence intervals with finite-sample coverage guarantees on `q_confidence`. The path to replacing the current heuristic attribution decomposition with statistically rigorous intervals — not yet implemented.

### Ensemble verifiers and oracle bootstrapping

- **Lee, J. et al. (2026).** *FUSE: Ensembling Verifiers with Zero Labeled Data.* arXiv 2604.18547. Unsupervised ensembling of LLM verifiers using only inter-verifier agreement structure — no ground truth labels required. Directly applicable to H2AI's cold-start problem: before domain-specific oracles exist, FUSE-style agreement can produce a calibrated ensemble verification score. Future implementation path for automated oracle integration — not yet implemented.
- **Menet, A. et al. (2026).** *POETS: Uncertainty-Aware LLM Optimisation via Policy Ensembles.* arXiv 2605.07775. Thompson Sampling over KL-regularised reward models for ensemble policy updates. The KL-divergence framework applies to Talagrand histogram correction: `Δτ = η × (U_score − Λ_score)` is the H2AI adaptation for τ-spread auto-calibration — not yet implemented.

### Calibration and ensemble spread

- **Leutbecher, M. & Palmer, T. N. (2008).** *Ensemble forecasting.* Journal of Computational Physics, 227(7). Talagrand rank histogram diagnostic borrowed from numerical weather prediction: flat = calibrated ensemble spread, U-shape = over-confident (increase τ to widen spread), Λ-shape = under-dispersed (decrease τ). Applied in `h2ai-autonomic/src/calibration.rs::TalagrandDiagnostic`. Domain transfer is direct — rank histograms are domain-agnostic calibration tools.
- **Ebbinghaus, H. (1885).** *Über das Gedächtnis.* Memory decay with exponential half-life. H2AI applies this to CG sample weighting: `exp(−(now − t) / CG_HALFLIFE_SECS)` with `CG_HALFLIFE_SECS = 604_800` (7 days). As calibration samples age, β_eff drifts toward the conservative prior β₀, automatically recalibrating toward caution without operator intervention.

### Context window and attention

- **Liu, N. F. et al. (2023).** *Lost in the Middle: How Language Models Use Long Contexts.* arXiv 2307.03172. Retrieval quality for content in the middle of long contexts degrades super-linearly. H2AI models this via `β_ctx(N) = β_eff × (1 + γ × fill(N))` where `fill(N) = min(1, N × proposal_tokens / max_tokens)`. This makes the synthesis coordination cost grow faster than linearly with N, providing a second incentive (beyond USL) to keep N below `N_max`.
