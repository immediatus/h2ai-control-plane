# H2AI Research State

This document is the project's critical self-assessment. It states what the system claims, where the math is genuinely defensible, where it is honest engineering heuristic, where empirical validation is missing, and how H2AI sits in the external landscape. It is not a roadmap and it does not track historical work.

For implementation details: [`architecture.md`](architecture.md). For formulas: [`math.md`](math.md). For the operational surface: [`reference.md`](reference.md), [`operations.md`](operations.md).

---

## 1. The system thesis

H2AI Control Plane orchestrates pools of LLM adapters as an *adversarial committee*. The runtime claims:

- The reliability of an N-adapter committee is bounded by **two diversity signals** that must both be measured: Hamming Common Ground on constraint behaviour, and cosine N_eff on semantic embedding.
- The throughput-vs-coordination trade-off is captured by USL (Gunther 1993) with a CG-coupled coherency cost: `β_eff = β₀ × (1 − CG_mean)`, `N_max = round(√((1 − α) / β_eff))`.
- The quality of the committee is bounded by a correlation-corrected Condorcet Jury Theorem (Nitzan & Paroush 1982; Ladha 1992): `Q(N, p, ρ) = p + (Q_ind(N, p) − p)(1 − ρ)`.
- Failures are classifiable: post-zero-survival, the cosine N_eff distinguishes `ConstrainedExploration` (diverse generation rejected by constraints) from `ModeCollapse` (correlated hallucination). MAPE-K routes each to a different intervention (tombstone vs adapter rotation).
- Every successful task produces an attribution decomposition `Q_total ≈ baseline × verification_filter × tao_uplift × topology_correction + synthesis_gain` (`crates/h2ai-orchestrator/src/attribution.rs`).

The differentiating claim is not the math itself — most components have decades of literature behind them. It is that all four signals (USL, CJT, eigenvalue diversity, Talagrand calibration) are tracked together, calibrated together, and used together to bound a single committee-execution loop.

---

## 2. What is genuinely defensible

In decreasing order of mathematical rigor and confidence in domain transfer:

**Eigenvalue calibration via participation ratio.** `EigenCalibration::from_cosine_matrix` applies `(Σ λ)² / Σ λ²` (Choueifaty & Coignard 2008) to the embedding cosine kernel. This is a direct measurement: at full independence it returns N; at full correlation it returns 1. There is no contested domain transfer — the participation ratio is the right tool to answer "how many independent perspectives are in this pool?"

**Generation-first ProposalSet LUB.** The CRDT semilattice over `(generation, score)` provides crash-safe idempotency and TAO-monotonic ordering. All CRDT axioms hold. This is selection, not content synthesis — see §3.

**Three-tier merge dispatch.** `ScoreOrdered → ConsensusMedian → OutlierResistant{f}` correctly escalates with the maximum role error cost. The Krum-style `OutlierResistant` selection has a proven breakdown point (Blanchard et al. 2017) under independent faults. The honest limitation is correlated hallucination — see §3.

**USL N_max as a calibrated upper bound.** Even framed as a phenomenological heuristic, calibrating N from measured `(α, β_eff)` is strictly better than "spawn until expensive." The N(N−1) merge complexity is real; the `β₀ × (1 − CG_mean)` coupling correctly captures that divergent agents cost more to reconcile.

**Talagrand rank histogram.** Borrowed from ensemble weather forecasting (Leutbecher & Palmer 2008). Flat = calibrated, U-shape = over-confident, Λ-shape = under-dispersed. The diagnostic is wired to τ-spread adjustment and `DiversityWarningEvent`. No domain-transfer issue — calibration histograms are domain-agnostic.

**Bivariate CG.** Phase 2.6 and the MAPE-K `FailureMode` classifier exist precisely because Hamming CG and cosine N_eff measure different things and must be tracked together. Two adapters can share constraint profiles by accident (high Hamming CG) while being semantically near-identical (low cosine N_eff). The runtime detects and routes around this.

**Temporal CG decay.** Ebbinghaus 7-day half-life on CG samples creates automatic recalibration pressure with a conservative failure mode (β_eff drifts toward β₀, lowering N_max). This is operationally sound — staleness errs on the side of caution.

---

## 3. Mathematical weaknesses (current architectural properties)

These are real limitations of the current system. They are not bugs to fix on a deadline; they are structural properties of the math being applied.

### 3.1 The independence-assumption chain

USL, CJT, Krum, and CRDT semantics all assume failure independence in different ways. The literature has individually found each assumption violated for LLM ensembles:

- **CJT independence.** Lefort et al. (arXiv 2409.00094): "CJT predicted accuracy gains do not materialise for LLM ensembles due to significant overlap in decision-making processes."
- **Krum Byzantine assumption.** arXiv 2512.20184: traditional consensus is designed for deterministic state machines and is incompatible with stochastic multi-agent reasoning.
- **USL coherency assumption.** arXiv 2602.03794: homogeneous agents saturate fast due to correlated outputs; USL's single-N parameter cannot distinguish homogeneous from heterogeneous pools.

The runtime corrects with `(1 − ρ)` in CJT and the bivariate-CG check at Phase 2.6. The corrections do not eliminate the underlying assumption; they bound its damage. When the constraint corpus is sparse, Hamming CG is near-zero regardless of true pool diversity, and compounding four formulas over a noisy base does not increase precision.

### 3.2 The β_eff double-duty problem

`CG_mean` modulates two quantities pointing the same direction:

```
β_eff    = β₀ × (1 − CG_mean)        high CG → low β_eff → N_max grows
rho_mean = 1 − CG_mean               high CG → low ρ → CJT predicts more benefit
```

Both effects say "high agreement → use more agents." But high agreement can mean *good consensus* (truly high p) or *correlated hallucination* (high ρ). One scalar cannot distinguish the two. The runtime mitigates this by tracking cosine N_eff independently and by Talagrand calibration, but the underlying coupling remains.

### 3.3 Correlated hallucination under outlier-resistant merge

The `OutlierResistant{f}` (Krum) breakdown-point proof assumes Byzantine faults are *independent* outliers in distance space. LLMs from the same family produce semantically identical wrong answers whose embeddings cluster tightly. If ≥50% of agents share a correlated hallucination, a Byzantine-resistant selector confidently selects that hallucination as the geometric median. Mitigations:

- `single_family_warning` on `CalibrationCompletedEvent` flags monoculture pools.
- `allow_single_family = false` (default) refuses to calibrate a monoculture pool.
- Phase 2.6 cosine-N_eff guard rejects pools with `n_eff_cosine_prior < 1.0 + diversity_threshold`.
- `ExplorerSlotConfig` (when populated) forces distinct CoT strategies (`StepByStep`, `DevilsAdvocate`, `FirstPrinciples`, `BackwardChaining`) per slot, reducing simultaneous-failure probability for same-family pools.

These are mitigations, not solutions. The Byzantine guarantee remains conditioned on independent faults.

### 3.4 LUB is selection, not synthesis

The `ProposalSet` CRDT merge picks a winning proposal and discards the rest. It does not synthesise content. The optional Phase 5a synthesis pipeline (critique → synthesis → re-verify) provides MoA-style generative aggregation when ≥ `synthesis_min_proposals` candidates survive audit, and `HarnessAttribution.synthesis_gain` records the delta. When synthesis does not run (insufficient candidates, disabled, or re-verification regression), the system falls back to LUB selection.

### 3.5 Verification circularity

Phase 3.5 uses a verification adapter (LLM-as-Judge) and Phase 4 uses an auditor adapter. Both are LLMs. Their biases (self-preference, length bias, position bias — Zheng et al. 2023; arXiv 2410.02736; arXiv 2410.21819) propagate through the entire decomposition. `synthesis_gain` is measured by the same verifier, so a verifier blind spot inflates the individual scores and the synthesis score equally. The system flags `explorer_verification_family_match` to surface judge-bias risk; it does not eliminate it. Without a Tier 1 oracle (test execution, fact-check API, domain-specific verifier), the "provable quality" framing is aspirational.

### 3.6 The proxy chain

The flow is:

```
CG_Hamming → β_eff → N_max          (USL chain)
CG_Hamming → (p_proxy, ρ_proxy) → Q (CJT chain, with prediction_basis = Heuristic)
embedding cosine kernel → N_eff      (eigen chain, direct measurement)
```

`scripts/baseline_eval.py` measures `p` directly and switches `prediction_basis` to `Empirical`, breaking part of the proxy chain. The ρ estimate remains a proxy unless an external oracle is configured.

---

## 4. Empirical gaps

### 4.1 No benchmark numbers in this repository

A benchmark harness exists under `scripts/benchmark/` with five baselines (`B0` single-shot, `B1` majority vote, `B2` MoA, `B3` Self-MoA, `H2` H2AI). The runs themselves are not in the repository. Until they are executed, the system's quality claim is unfalsified.

The MoA paper (Wang et al. 2024, arXiv 2406.04692) achieves 65.1% on AlpacaEval 2.0 vs GPT-4o's 57.5%. "More Agents Is All You Need" (arXiv 2402.05120) shows majority voting scales log-linearly with N. Self-MoA (Li et al. 2025, arXiv 2502.00674) reports N samples from one strong model beats N diverse models by 6.6%. H2AI must be measured against all three on a representative task suite.

The strongest empirical competitor is MoA. Whether USL-bounded N + bivariate CG + adversarial committee outperforms simpler generative aggregation is the load-bearing empirical question.

### 4.2 Attribution intervals depend on calibration variance

`HarnessAttribution::compute` produces a point estimate. `bootstrap_interval` (1000 resamples over CG samples, when `cg_samples.len() >= 2`) supplies `q_interval_lo`/`q_interval_hi`. Conformal intervals require an oracle signal that is not yet wired into production deployments. Until then, the interval reflects CG-sample variance, not ground-truth uncertainty.

### 4.3 Open empirical questions

- **Does role diversity reduce error correlation?** arXiv 2506.07962 finds error correlation is driven by training data and architecture, not prompting; arXiv 2508.09654 finds temperature alone is insufficient. Whether `ExplorerSlotConfig` system-prompt diversity measurably reduces ρ on a verifiable task set is unknown.
- **Self-MoA vs. multi-adapter.** If Li et al. (2025) holds for H2AI's target task domain, inter-adapter CG optimisation is optimising the wrong variable. Empirical test: same-model × temperature vs multi-family × role on the same task suite.
- **Does USL `N_max` produce better quality/cost than naive scaling?** The core thesis. Without benchmark data, it is plausible but unproven.
- **What is the right `cg_collapse_threshold`?** The default `0.10` is analytical. Empirical data on the CG_embed value at which TAO first-pass rate drops sharply would replace the prior.

---

## 5. Infrastructure boundaries that limit the math

The math is calibrated on the assumption that infrastructure does not silently distort the signal. Several infrastructure choices interact with the math in ways operators must understand.

- **NATS message size.** JetStream's 1 MB default message ceiling is well below modern LLM context budgets (1M-token contexts ≈ 4–8 MB). `payload_offload_threshold_bytes` keeps `system_context` bytes well under 1 MB by replacing them with hash references. When the JetStream ceiling is raised, this knob must be raised in lockstep — otherwise large constraint corpora silently truncate.
- **Event replay.** Recovery time without snapshots is linear in the task's event count. `snapshot_interval_events = 50` keeps replay bounded; raising it linearly increases recovery latency and does not improve correctness.
- **Scheduler starvation.** `CostAwareSpillover` (the default) routes to the next cost tier when a tier's queue exceeds `scheduler_spillover_threshold = 10`. Without spillover, low-tier agents form deep queues while high-tier agents idle. The math is unaffected, but Phase 3 timeouts become a calibration-drift signal that does not actually reflect the pool.
- **Tool-using agent file-system races.** Multiple shell agents share a workspace volume. CRDT event-log coordination does not mediate file-system concurrency. The calibrated α reflects measured serialisation cost, but uncoordinated writes still produce non-deterministic outputs that show up as inflated ρ, not as α. Per-task volume mounts or ephemeral containers are required for correctness.
- **Auditor as a single point of judgment.** Phase 4 is one adapter call. If the auditor is biased on the corpus, every task is biased. `explorer_verification_family_match` flags the most common failure mode (judge-from-the-same-family). Multi-auditor consensus is not currently part of the design.
- **Embedding model is required for the bivariate-CG safety net.** Without `fastembed-embed` and a configured model, the runtime falls back to a closed-form `n_eff_cosine_prior` and disables Phase 2.6. The system still runs, but the bivariate-CG guarantees are downgraded to univariate Hamming.

---

## 6. External landscape

The combination of *USL-bounded N* + *bivariate CG* + *MAPE-K failure-mode routing* + *CRDT-convergent merge with optional generative synthesis* + *Harness Attribution* has no direct analogue in published frameworks.

Layer positioning:

- **Inference layer** (vLLM, TGI, llama.cpp, Ollama) — H2AI delegates here via adapters.
- **Adapter-internal optimisation** (DSPy) — DSPy compiles prompts and few-shot weights inside one adapter. It is complementary to H2AI: a DSPy-optimised adapter can sit inside `IComputeAdapter` while H2AI orchestrates the swarm.
- **Distributed compute fabric** (Ray, Kubernetes) — Ray and K8s map agents to hardware. H2AI decides *how many* agents and *what roles* before delegating.
- **Topology and coordination layer (H2AI's home)** — bounding N, calibrating ρ, routing failure modes, attributing quality.
- **Agentic frameworks** (LangChain/LangGraph, AutoGen, CrewAI, OpenAI Swarm) — these compose tools and memory. They do not bound N from measurement, do not classify failure modes, and do not produce a quality decomposition.
- **Empirical aggregators** (MoA, Self-MoA) — the strongest empirical competitors. MoA wins on simplicity; H2AI's claim is that calibrated bounding outperforms unbounded aggregation on quality/cost. The claim is empirically unverified at the time of writing.

### Key papers to cite and differentiate against

| Paper | Relevance |
|---|---|
| Wang et al. (2024) — arXiv 2406.04692 (MoA) | Generative aggregation baseline. |
| Li et al. (2025) — arXiv 2502.00674 (Self-MoA) | Single-model aggregation may beat multi-family; load-bearing for H2AI's diversity claim. |
| arXiv 2402.05120 — More Agents Is All You Need | Naive scaling baseline. |
| arXiv 2512.08296 — Towards a Science of Scaling Agent Systems | Coordination overhead model; aligned with USL framing. |
| Lefort et al. (2024) — arXiv 2409.00094 | CJT failure for LLM ensembles. |
| arXiv 2602.03794 — Understanding Agent Scaling via Diversity | Supports cosine-N_eff approach. |
| arXiv 2507.14928 — Byzantine-Robust Decentralised LLM Coordination | Closest architectural prior art. |
| arXiv 2511.10400 — Rethinking the Reliability of MAS via BFT | Framing context for the merge tier. |
| arXiv 2510.18893 — CodeCRDT | Independent confirmation that CRDT applies to LLM-agent ordering. |
| arXiv 2406.09714 — LLM Validity via Conformal Prediction | Path to oracle-grounded intervals. |
| Zheng et al. (2023) — arXiv 2306.05685 | LLM-as-Judge biases. |
| arXiv 2506.07962 | Error correlation driven by training data, not prompting. |
| arXiv 2508.09654 | Temperature is insufficient for diversity; training loss governs. |

---

## 7. References

- Gunther, N. J. (1993). *Universal Scalability Law*. CMG.
- Condorcet, M. J. A. N. (1785). *Essai sur l'application de l'analyse à la probabilité des décisions rendues à la pluralité des voix*.
- Nitzan, S. & Paroush, J. (1982). *International Economic Review* 23(2).
- Ladha, K. K. (1992). *American Journal of Political Science* 36(3).
- Choueifaty, Y. & Coignard, Y. (2008). *Journal of Portfolio Management* 35(1).
- Blanchard, P., El Mhamdi, E. M., Guerraoui, R., Stainer, J. (2017). *Machine learning with adversaries: Byzantine tolerant gradient descent (Krum)*. NeurIPS.
- Pillutla, K., Kakade, S. M., Harchaoui, Z. (2019). *Robust aggregation for federated learning*. arXiv 1912.13445.
- Vardi, Y. & Zhang, C.-H. (2000). *The multivariate L1-median and associated data depth*. PNAS.
- Leutbecher, M. & Palmer, T. N. (2008). *Ensemble forecasting*. *Journal of Computational Physics*.
- Liu, N. F. et al. (2023). *Lost in the Middle: How Language Models Use Long Contexts*.
- Wang, J. et al. (2024). *Mixture-of-Agents enhances large language model capabilities*. arXiv 2406.04692.
- Li, J. et al. (2025). *Self-MoA*. arXiv 2502.00674.
- Lefort, P. et al. (2024). *Empirical CJT failure on LLM ensembles*. arXiv 2409.00094.
- Zheng, L. et al. (2023). *Judging LLM-as-a-Judge with MT-Bench*. arXiv 2306.05685.
