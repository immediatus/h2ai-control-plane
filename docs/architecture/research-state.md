# H2AI Control Plane — Research State

**Last revised:** 2026-05-01  
**Status:** Authoritative single source of truth for project theory, implementation state, gap analysis, and critical assessment.

This document is the authoritative critical assessment of project theory and implementation state. See also: `math.md` (mathematical formulas), `reference.md` (API and configuration), `operations.md` (deployment and operations).

---

## 1. Project Thesis

**One equation governs CPU caches, human teams, and AI agent swarms for the same structural reason: whenever N nodes must maintain mutual consistency, pairwise synchronization cost scales as N(N−1).**

The Universal Scalability Law (Gunther 1993) describes this:

```
X(N) = N / (1 + α(N−1) + β·N(N−1))
N_max = √((1−α) / β_eff)      [USL Proposition 1 — dX/dN = 0]
```

H2AI applies USL to bound the number of LLM agents before launching them, using a measured Coordination Quality (CG) coefficient to tune the coherency term β dynamically. This is combined with Condorcet Jury Theorem for quality prediction, CRDT semilattice for convergent merging, and a three-tier merge dispatch for robustness.

The system's differentiating claim: **Harness Attribution** — every successful task produces a decomposition `Q_total = baseline + topology_gain + verification_gain + tao_gain`, answering "what did the orchestration buy over a single adapter call?"

**Positioning:** No existing production framework (LangChain, CrewAI, AutoGen, OpenAI Swarm) claims mathematical quality bounds. H2AI's differentiation is real *if* the math holds under realistic LLM conditions. That conditional is load-bearing — see Section 5.

---

## 2. Architecture Overview

**Six phases per task:**

| Phase | What happens | Key formula |
|---|---|---|
| 0. Calibration | Measure α, β₀, CG_mean from M≥3 adapters | USL two-point linearization |
| 1. Bootstrap | Dark Knowledge Compiler compiles `system_context`; proceeds unconditionally | Constraint fingerprints: `fp[k] = Hard constraint k passes` |
| 2. Topology | Pareto-frontier selection across 3 topology candidates bounded by N_max | N_max = √((1−α)/β_eff) |
| 3. Multiplication gate | Competence ∧ decorrelation ∧ CG ≥ θ; failure → MAPE-K retry | EigenCalibration N_eff ≥ 2 |
| 4. Parallel TAO loop | ≤3 turns, `c_i × 0.6^(t-1)` error decay | Temporal error discount |
| 5. Verify + Merge | Three-tier Krum/Weiszfeld/ConsensusMedian dispatch | BFT-style outlier resistance |

**16 workspace crates.** Core crates: `h2ai-types` (physics, events), `h2ai-state` (bft, krum, weiszfeld, semilattice), `h2ai-context` (embedding, similarity, compiler), `h2ai-autonomic` (calibration, planner, merger), `h2ai-orchestrator` (engine, tao_loop, verification, attribution).

**Event log:** NATS JetStream with `SessionJournal` replay. `adapter_version_hash` scopes all calibration state; version change resets bandit posteriors and EMA.

---

## 3. Mathematical Framework — Honest Framing

### 3.1 The Overclaim to Fix First

The codebase names its core module `sizing.rs`. This is a framing error that will invite dismissal from anyone with a physics background. USL, CJT, and Krum are **phenomenological heuristics and engineering tools**, not physical laws. They are useful precisely because they are pragmatic — but calling them physics suggests first-principles derivation that does not exist.

**Done:** `physics.rs` renamed to `sizing.rs`; doc language updated throughout to "phenomenological heuristics and engineering tools."

### 3.2 Universal Scalability Law

**Formula (correct):**
```
X(N) = N / (1 + α(N−1) + β·N(N−1))
N_max = round(√((1−α) / β_eff))
β_eff = β₀ × (1 − CG_mean)
```

**Implemented in:** `crates/h2ai-types/src/sizing.rs` — `CoherencyCoefficients::n_max()`, `beta_eff()`

**β_eff formula:** `β₀ × (1 − CG_mean)`. At CG=1 (full agreement): β_eff ≈ 0 — coordination-free. At CG=0: β_eff = β₀ — maximum coordination overhead. Bounded everywhere. The alternative `β₀/CG_mean` was rejected because it diverges as CG→0, making N_max undefined in low-CG calibration runs.

**Calibration:**
```
z(N) = N·T_parallel(N)/T₁ − 1 = α(N−1) + β₀·N(N−1)

Analytical two-point solution (N=2 and N=M, M≥3):
  β₀ = (z_M − z₂·(M−1)) / ((M−1)(M−2))
  α  = z₂ − 2·β₀
  
Clamped: α → [0.05, 0.5], β₀ → [1e-6, 0.1]
Fallback to config defaults when M < 3 or degenerate.
```

**What USL legitimately models for LLM agents:** The N(N−1) merge complexity is self-imposed by pairwise comparison (Jaccard distances, pairwise semantic distances). This is a real quadratic cost. The β parameter captures it correctly. The serial fraction α (task decomposition, synthesis) is also real. The analogy to server scaling is structural, not metaphorical.

**What USL does NOT legitimately model:**
- `N_max = √((1-α)/β)` optimizes throughput-per-agent, not quality-per-task. These are not the same objective. No proof exists that they coincide for LLM agents.
- USL uses a single scalar N; coordination overhead in LLM agents scales with **interaction depth** (arxiv 2512.08296), which USL cannot capture with one parameter.
- There is no published paper applying USL to LLM agents. The calibration produces curve-fitting parameters, not measured physical constants.
- **Critical:** β₀ is currently measured from wall-clock timing of parallel adapter API calls — a proxy for true semantic reconciliation cost. Complex reasoning tasks (agents diverge more) have higher true β₀ than timing suggests. Template tasks have lower. This bias is known and partially mitigated by the CG coupling, but not eliminated.

**Two-tier calibration table** (β_eff = β₀ × (1 − CG_embed)):

| Tier | α | β₀ | CG_embed | β_eff | N_max |
|---|---|---|---|---|---|
| AI agents | 0.15 | 0.039 | 0.40 | 0.0234 | ~6 |
| Human teams | 0.10 | 0.0225 | 0.60 | 0.009 | ~10 |

CPU cores omitted: at CG=1.0 the new formula gives β_eff≈0 → N_max→∞, which is meaningless for the LLM domain.

**See:** `scripts/simulate.py` — USL curve shapes and β_eff coupling visualization.

---

### 3.3 Common Ground (CG)

**Target formula (requires EmbeddingModel — Gap E1):**
```
CG(i, j) = mean over calibration_prompts of
            [cosine(embed(output_i), embed(output_j)) > θ_agree]
where θ_agree = 0.85
```
This is the agreement rate between adapter i and j on a calibration set — semantically robust, paraphrase-insensitive, matches the blog specification exactly.

**Current implementation (constraint-profile Hamming — active in production):**
```
fp[k] = Hard constraint k passes against output_i   (Vec<bool>, one entry per constraint)
CG(i, j) = hamming_distance(fp_i, fp_j) × tau_alignment(τ_i, τ_j)
```
This measures how differently two adapters satisfy the constraint corpus — a domain-grounded diversity signal rather than vocabulary overlap. The signal quality is bounded by corpus quality: a sparse or homogeneous constraint corpus produces nearly-identical fingerprints and CG near zero even when adapters diverge semantically.

**CG_mean / CG_embed** = mean of all pairwise CG values across calibration adapters.

**Temporal decay:**
```
w(t_i) = exp(−(now − t_i) / 604800)    [7-day half-life]
CG_eff  = Σ(CG_i × w(t_i)) / Σ(w(t_i))
β_eff_temporal = β₀ × (1 − CG_eff)
```
Correct failure mode: aging samples push CG_eff → 0 → β_eff → large → N_max shrinks → pressure to re-calibrate. Conservative and architecturally sound.

---

### 3.4 Condorcet Jury Theorem

**Status: retained as descriptive model and attribution driver, NOT as topology provisioning mechanism.**

**Formula:**
```
Q_ind(N, p) = Σ_{k=⌈N/2⌉+1}^{N} C(N,k) × p^k × (1−p)^(N−k)
              + [if N even: 0.5 × C(N, N/2) × p^(N/2) × (1−p)^(N/2)]

Q(N, p, ρ) = p + (Q_ind(N, p) − p) × (1 − ρ)
```

**Proxy chain (heuristic, not measurement):**
```
p_mean   = 0.5 + CG_mean / 2   ∈ [0.5, 1.0]
rho_mean = 1 − CG_mean          ∈ [0, 1]
```

**What the proxy chain actually says:** CG_mean measures inter-adapter output similarity, not accuracy against ground truth. `p_mean = 0.85` when CG=0.7 means "these agents tend to agree" not "these agents are correct 85% of the time." Two adapters can agree on the same wrong answer (high CG, low p) or disagree while both being correct (low CG, high p). This is a fundamental gap in the quality prediction chain.

**The independence assumption fails for LLMs.** Lefort et al. (arxiv 2409.00094) tested CJT on LLM ensembles directly and found predicted accuracy gains do not materialize because "LLMs exhibit significant overlap in decision-making processes." H2AI applies the correlation correction `(1 − ρ)`, which is the correct response — but derives ρ from output similarity rather than error correlation, which can diverge arbitrarily.

**What CJT is correctly used for in H2AI:**
- The `topology_gain` term in Harness Attribution (advisory)
- `n_it_optimal` = ⌈1 + ln(0.5)/ln(1−ρ)⌉ as a secondary N ceiling (matches Condorcet N_optimal within ±1)
- CJT over-prediction of 5–15pp at ρ≥0.6 (typical same-family LLM correlation) is documented in `validate_conformal_vs_cjt.py`

**What CJT is NOT used for in H2AI:**
- Topology provisioning ceiling — that is USL N_max × EigenCalibration N_eff (correct separation)

**See:** `scripts/simulate.py` — Monte Carlo CJT verification and conformal vs CJT over-prediction at high ρ.

---

### 3.5 Eigenvalue Calibration — N_eff (Strongest Mathematical Contribution)

**Source:** Portfolio theory (Choueifaty & Coignard 2008).

**Formula:**
```
Σ ∈ ℝ^(N×N)   pairwise CG similarity matrix
λ_1 ≥ ... ≥ λ_N = eigenvalues(Σ)

N_eff     = (Σ λᵢ)² / Σ λᵢ²        [participation ratio]
H_norm    = (−Σ(λᵢ/Σλ)log(λᵢ/Σλ)) / log(N)
ρ_eff     = 1 − N_eff/N
```

**Implemented in:** `crates/h2ai-types/src/sizing.rs` — `EigenCalibration::from_cg_matrix()`

**Wired to planner:** `n_max = n_max_usl.min(eigen.n_pruned)` — the eigenvalue ceiling is active.

**Why this is the strongest contribution:** Unlike USL and CJT, the participation ratio requires no contested domain transfer. It directly measures how many independent perspectives the adapter pool contains. Scalar ρ_mean overstates N_eff by 55% on heterogeneous pools (2 independent + 3 clustered → N_eff=2.5, scalar gives 3.9). The pruning rule (add adapter N+1 only if N_eff increases by ≥ 0.05) is a principled stopping criterion with no hidden assumptions.

**Current limitation:** N_eff is computed from constraint-profile Hamming CG (constraint satisfaction divergence), which is domain-grounded but bounded by corpus quality. Improves to semantic N_eff after Gap E1 (EmbeddingModel) enables embedding cosine agreement rate.

**Stopping rule threshold `Δ<0.05` is hardcoded.** Should be config-tunable for operator adjustment.

**See:** `scripts/simulate.py` — N_eff formula verification, stopping rule, comparison to scalar ρ estimate.

---

### 3.6 Merge Strategy — Three-Tier Dispatch

**Dispatch logic:**
```
cluster_coherent                       → Krum (outlier-resistant selection)
cluster incoherent + embedding present → Weiszfeld (geometric median, 50% breakdown)
cluster incoherent + no embedding      → ConsensusMedian (token Fréchet — no metric guarantee)
```

**Krum (currently named MergeStrategy::Krum — rename pending):**  
Selects the vector with minimum sum-of-distances to its nearest `n − f − 1` neighbors. Correct function: **outlier-resistant selection** (picks the most central proposal). Incorrect framing: "Byzantine Fault Tolerant." Krum's BFT guarantee (tolerating exactly f adversarial Byzantine workers) requires bounded adversarial failures. LLM hallucinations are stochastic and correlated, not bounded adversarial. When all agents hallucinate in the same direction, Krum selects the centroid of the dominant hallucination cluster — the opposite of the desired behavior. The `cluster_coherent()` precondition (max pairwise distance < 0.7) is the correct hedge: it gates Krum to cases where the cluster assumption approximately holds.

**Weiszfeld geometric median:**  
Minimizes sum of Euclidean distances to all input vectors. Breakdown point 1/2 (tolerates ⌊n/2⌋−1 corrupted inputs). Correct algorithm for the stochastic LLM case when errors are *independent*. **Critical limitation:** the breakdown-point proof assumes Byzantine faults are independent outliers in embedding space. LLMs from the same model family frequently produce correlated hallucinations — semantically identical errors whose embeddings cluster together. When ≥50% of agents share a correlated hallucination, that hallucination cluster becomes the "honest" geometric median and Weiszfeld selects it with high confidence. The BFT guarantee does not apply in this regime. `cluster_coherent()` partially hedges (the Weiszfeld path is only taken when cluster assumption fails), but does not distinguish "coherent honest proposals" from "coherent shared hallucination." See Section 5.6. Enabled when `embedding_model` is present (resolved S1/E4).

**ConsensusMedian:**  
Token Jaccard Fréchet median. No metric guarantee (LLM similarity is not symmetric, may fail transitivity). Active fallback when no EmbeddingModel.

**CRDT ProposalSet — Generation-First LUB:**  
```
insert_scored(proposal):
  if proposal.generation > existing.generation: replace
  if proposal.generation == existing.generation and proposal.score > existing.score: replace
  otherwise: discard
```
Satisfies CRDT axioms (commutativity, associativity, idempotency). Generation-first LUB handles TAO retry correctly (later generation with lower score supersedes earlier generation with higher score). Architecturally correct. **Important limitation:** strong eventual consistency ≠ semantic correctness. The CRDT ensures all replicas converge to the same proposal set; it says nothing about whether those proposals are good. CRDT is a bookkeeping mechanism, not a quality filter (CodeCRDT, arxiv 2510.18893).

**See:** `scripts/simulate.py` (Weiszfeld breakdown point visualization); semilattice unit tests in `crates/h2ai-state/tests/`.

---

### 3.7 Constraint Fingerprint and Diversity Gate

**Phase 1 — Unconditional compilation:**  
`POST /tasks` compiles `system_context` from the constraint corpus and task manifest and proceeds unconditionally. There is no pre-generation adequacy gate; tasks are always accepted (or return `503 CalibrationRequiredError` if calibration has never run).

**Constraint satisfaction fingerprint:**
```
fp_i[k] = Hard constraint k passes against output_i    (Vec<bool>, one entry per constraint)
```
Used in two places:
1. **Calibration CG** — `CG(i, j) = hamming_distance(fp_i, fp_j) × tau_alignment(τ_i, τ_j)`
2. **Phase 2.5 diversity gate** — mean pairwise Hamming distance across surviving proposals must exceed `diversity_threshold` (default 0.0; recommended production value 0.15)

**Auditor is the generation gate.** Proposals that violate Hard constraints are rejected via `BranchPrunedEvent` before reaching the Merge Authority. The constraint corpus quality determines gate selectivity — a sparse corpus passes everything; a rich corpus enforces the team's architectural rules.

**What replaced the J_eff gate:** The old `semantic_jaccard` LLM-call-based adequacy gate (`J_eff < 0.4 → ContextUnderflowError`) was removed. The gate created a circular dependency (judge = same pool as defendants), added latency, and polluted `T_merge` measurement. The Auditor's Hard-constraint rejection mechanism provides a structurally cleaner gate at the proposal level rather than at the submission level.

---

### 3.8 Attribution Model

```
baseline_quality   = p_mean
topology_gain      = Q(N, p_mean, rho_mean) − p_mean     [Condorcet gain]
tao_multiplier     = tao_per_turn_factor ^ (turns − 1)   [default 0.6]
error_remaining    = (1 − Q(N, p_mean, rho_mean)) × verification_filter_ratio × tao_multiplier
total_quality      = 1 − error_remaining,  clamped to [p_mean, 1.0]
```

**Implemented in:** `crates/h2ai-orchestrator/src/attribution.rs`

**Critical weakness:** This is a point estimate derived entirely from heuristic proxies (p_mean from CG, rho_mean from CG). It has no confidence intervals. An operator seeing `Q_total = 0.91` cannot know if the true range is `[0.88, 0.94]` or `[0.60, 1.0]`. The decomposition is correct as an accounting identity; its usefulness as a measurement depends on calibrating the proxies against real accuracy data. **Without `baseline_eval.py` producing empirical p, the attribution is a mathematically consistent but empirically ungrounded estimate.**

---

### 3.9 Talagrand Rank Histogram Diagnostic

```
rank histogram over T ≥ 20 runs:
  flat  → calibrated ensemble
  U-shape (high tails) → over-confident; adapters too certain → expand τ spread
  Λ-shape (high center) → under-dispersed; mediocre adapters → diversify model families

χ² = Σ_r (H[r] − T/N)² / (T/N)
Calibrated iff χ² < 3.84
```

**Implemented in:** `crates/h2ai-orchestrator/src/diagnostics.rs`  
**Wired to MAPE-K (Gap A3, resolved S4).** `TalagrandShape::UShape` → τ spread increased 20%; `TalagrandShape::LambdaShape` → `DiversityWarningEvent` emitted. `Flat` → no action. Both wired in `tao_loop.rs`.

---

### 3.10 Bootstrap State Machine

```
Phase 0 (K=0):    N=3, τ_spread=1.5×steady_state, no adaptive mechanisms
Phase 1 (K≥5):    β₀_ema activates (blend starts)
Phase 2 (K≥10):   Bandit over N activates with USL-warm prior (Thompson Sampling, implemented in `bandit.rs`)
Phase 3 (K≥50):   Conformal buffer activates (Tier 1/2 tasks only) [NOT IMPLEMENTED — deferred]
Phase 4 (K≥20×C(|adapters|,2)): CG_history pair weighting activates
```

Staleness resets per signal on `adapter_version_hash` change.

---

## 4. Current Implementation State — Gap Inventory

### 4.1 Resolved Gaps

| Gap | Description | Status |
|---|---|---|
| P4 | β_eff formula verified | **FIXED** — formula is `β₀ × (1 − CG_mean)`, bounded everywhere |
| P2 | EigenCalibration → planner wiring | **FIXED** — `n_max = n_max_usl.min(eigen.n_pruned)` |
| P6 | CRDT generation-first LUB | **FIXED** — `ProposalEvent.generation`, `insert_scored` generation-first |
| P5 (partial) | USL fit fallback when M<3 | **FIXED** — fallback to config defaults, warns on M<3 |

### 4.2 Open Gaps — Embedding Stack (Highest Priority)

These block the entire quality improvement chain. They are sequentially dependent.

**Gap E1 — EmbeddingModel not wired to Application startup** ✅ **DONE (S1)**  
`FastEmbedModel` wired into `AppState` with `Arc<dyn EmbeddingModel>`; threaded to both `MergeEngine::resolve` call sites in `engine.rs`. Feature-gated behind `fastembed-embed`.

**Gap E2 — semantic_jaccard uses LLM calls (circular dependency)** ✅ **DONE (S1)**  
`semantic_jaccard` replaced with embedding cosine `dot_product(model.embed(a), model.embed(b))`. Circular dependency eliminated; T_merge measurement clean.

**Gap E3 — CG measurement uses token Jaccard, not a domain-grounded signal** ✅ **DONE (constraint-profile Hamming)**  
CG now uses `CgMode::ConstraintProfile`: `fp[k] = Hard constraint k passes against output_i`; `CG(i, j) = hamming_distance(fp_i, fp_j)`. This is a domain-grounded diversity signal (constraint satisfaction divergence) rather than vocabulary overlap. Downstream N_max, N_eff, and diversity gate all use the improved signal. Embedding cosine agreement rate (the originally planned approach) remains a future improvement for when EmbeddingModel is wired.

**Gap E4 — Weiszfeld path disabled in engine.rs** ✅ **DONE (S1)**  
Both `MergeEngine::resolve` call sites now pass `state.embedding_model.as_deref()`. Weiszfeld activates when `cluster_coherent()` precondition holds.

### 4.3 Open Gaps — Calibration Quality

**Gap C1 — Calibration adapter pool is homogeneous** ✅ **DONE (S3)**  
`AppState` now holds `Vec<Arc<dyn IComputeAdapter>>`; calibration harness cycles through all configured adapters. Single-adapter deployments remain valid with a documented caveat.

**Gap C2 — β₀ measured from wall-clock timing, not token cost** ✅ **DONE (S3)**  
Online EMA update implemented with merge token cost:
```
β₀_ema = 0.95 × β₀_ema + 0.05 × (merge_tokens / pair_count / mean_proposal_tokens)
β₀_effective = (1−w) × timing_prior + w × β₀_ema    where w = min(K/30, 1.0)
```

**Gap C3 — calibration_tau_spread is reserved but unused** ✅ **DONE (S3)**  
`calibrate.rs` now linearly spaces adapter instances across `[τ_min, τ_max]` using the `tau_spread` config field.

**Gap C4 — Eigenvalue stopping rule Δ<0.05 is hardcoded**  
Everything else in the calibration path is operator-configurable. This threshold should be in config. *Still open.*

### 4.4 Open Gaps — Verification

**Gap V1 — Verification is single-tier (only Tier 3 / LLM-judge runs)** ✅ **DONE (S6)**  
Three-tier verification now implemented in `verification.rs`:

| Tier | Verification signal | Implementation status |
|---|---|---|
| 1 (Oracle) | `test_runner_uri` → HTTP Pass/Fail (`OracleExecution`) | **IMPLEMENTED** |
| 2 (Grounded) | `JsonSchema` / `LengthRange` predicate (`eval_sync`) | **IMPLEMENTED** |
| 3 (LLM judge) | CoT rubric constitutional judge score | Active |

G-Eval CoT rubric template used as `__rubric__` fallback when no rubric is provided.

**Gap V2 — Judge adapter family not enforced ≠ explorer adapter family** ✅ **DONE (S6)**  
Cross-family discriminant check added to `state.rs` startup: `std::mem::discriminant(explorer.kind()) == std::mem::discriminant(auditor.kind())` emits a tracing warning when same-family configuration is detected.

### 4.5 Open Gaps — Adaptivity

**Gap A1 — Bandit over N not implemented** ✅ **DONE (S4)**  
Thompson Sampling bandit implemented in `bandit.rs` over `N ∈ {1..min(6, N_max)}` with USL warm prior (Beta posterior initialized from USL N_max). Reward signal feeds from Tier 1/2 verification scores.

**Gap A2 — SelfOptimizer produces but never applies on success**  
Optimization suggestions are computed after every successful run but applied only on MAPE-K retry. The system cannot learn between tasks. Calibration is the only adaptation channel. *Still open.*

**Gap A3 — Talagrand diagnostic not wired to τ adjustment** ✅ **DONE (S4)**  
Talagrand diagnostic now feeds back to the autonomic loop: `UnderDispersed` → expand τ spread; `OverDispersed` → emit `DiversityWarningEvent`. Both wired in `tao_loop.rs`.

### 4.6 Open Gaps — Documentation and Framing

**Gap D1 — `sizing.rs` name and "physics" language throughout docs** ✅ **DONE**  
`physics.rs` renamed to `sizing.rs`; all import sites updated (`crate::physics`, `h2ai_types::physics`, `super::physics`). Module-level doc comment updated to remove "physical law" framing. `RoleErrorCost` doc updated to reference config field names instead of "BFT and Krum thresholds".

**Gap D2 — MergeStrategy::Krum should be renamed OutlierResistant** ✅ **DONE**  
`MergeStrategy::OutlierResistant` (and `MultiOutlierResistant`) replace the prior `Krum` variant.
Doc comments in `merger.rs` updated to use "outlier-resistance quorum" and "outlier-resistant selection".

**Gap D3 — validate_beta_coupling.py and validate_math.py are stale** ✅ **DONE**  
Both scripts updated to `β_eff = β₀×(1−CG)`. `validate_beta_coupling.py` rewritten with 5 invariant checks and updated tier table. `validate_math.py` updated `kappa_eff()`, dropped CPU cores row (wrong domain), fixed assertion direction for the new formula, replaced stale algebraic-equivalence check with monotonicity check.

**Gap D4 — No adapter cost accounting**  
`c_i` cost weights are operator-supplied and never updated from actual token spend. Cost efficiency claims in Pareto topology selection depend on accurate costs.

---

## 5. Critical Weaknesses — External Review Findings

This section documents findings from the 2026-04-30 critical review against the broader literature.

### 5.1 The Independence Assumption Chain

The most significant structural problem: USL, CJT, Krum, and CRDT semantics all assume failure independence in different ways. The LLM literature has individually found each assumption violated:

- **CJT independence:** Lefort et al. (arxiv 2409.00094) — "CJT predicted accuracy gains do not materialize for LLM ensembles due to significant overlap in decision-making processes."
- **Krum Byzantine assumption:** arxiv 2512.20184 — "Traditional consensus is designed for deterministic state machines and is incompatible with stochastic multi-agent reasoning."
- **USL coherency assumption:** arxiv 2602.03794 — homogeneous agents saturate fast due to correlated outputs; USL's single-N parameter cannot distinguish homogeneous from heterogeneous pools.

H2AI's correlation correction to CJT and the `cluster_coherent()` precondition for Krum are the correct engineering responses. But the assumptions stack: each formula inherits the error of the CG proxy, and the CG proxy (constraint-profile Hamming) is bounded by corpus quality — a sparse corpus produces near-zero CG regardless of adapter output diversity. When the base signal is noisy, compounding four formulas over it does not increase precision.

**Mitigation path:** Ensure the constraint corpus is rich enough that `CG_mean` captures real adapter divergence. Run `baseline_eval.py` on a representative task set to measure actual p and ρ directly. Use those measurements to validate or correct the proxy chain. Embedding cosine CG (Gap E1) would further improve the base signal.

### 5.2 β_eff Double-Duty Problem

`CG_mean` modulates both `β_eff` (USL coherency cost) and `rho_mean` (Condorcet correlation):

```
β_eff    = β₀ × (1 − CG_mean)         → high CG → low β_eff → N_max grows
rho_mean = 1 − CG_mean                → high CG → low ρ → Condorcet predicts more benefit
```

Both effects point the same direction: high agent agreement → system recommends more agents AND predicts higher quality from those agents. This is suspicious — in real systems, agents that always agree are either very good (truly high p) or all making the same mistake (high ρ, catastrophic). The single CG signal cannot distinguish these. The Talagrand diagnostic (Gap C5 / Gap A3) is the correct instrument to detect the difference.

### 5.3 No Empirical Benchmarks

**Harness status: complete, never executed.** The MoA paper (arxiv 2406.04692) achieves 65.1% on AlpacaEval 2.0 vs GPT-4o's 57.5% using simple generative aggregation. "More Agents Is All You Need" (arxiv 2402.05120) shows majority voting scales log-linearly with N. Self-MoA (arxiv 2502.00674) finds N samples from one strong model beats N diverse models by 6.6% on AlpacaEval 2.0.

H2AI claims USL-bounded N provides better quality/cost tradeoff than naive scaling. This may be correct. Without benchmark results, it cannot be known. **This is the highest-ROI open action.**

**To run the harness:**
```bash
# Prerequisites: Python deps, API key, running H2AI server
pip install -r scripts/benchmark/requirements.txt
export OPENAI_API_KEY=sk-...
# Start the H2AI server on localhost:8080

# Smoke test (5 problems, validates pipeline end-to-end):
python -m scripts.benchmark.smoke_test --model gpt-4o-mini

# Full GSM8K run (all 5 baselines: B0=single, B1=majority-vote, B2=MoA, B3=self-MoA, H2=H2AI):
python -m scripts.benchmark.run_gsm8k --baselines b0 b1 b2 b3 h2 --model gpt-4o-mini

# Compare results:
python -m scripts.benchmark.compare scripts/benchmark/results/gsm8k_*.json
```

**What to look for:**
- H2 accuracy vs B2 (MoA): if H2 ≤ B2 at equal cost, the calibration overhead is net-negative
- H2 cost/accuracy Pareto front vs B1 (majority vote): if H2 buys no quality improvement at lower N, USL bounding is not adding value
- H2 synthesis_gain distribution: if mean synthesis_gain < 0, synthesis is regressing on this task distribution
- Self-MoA (B3) vs H2: Li et al. 2025 claims single-model beats diverse ensemble by 6.6% — measure whether this holds for H2AI's task mix

### 5.4 Harness Attribution Without Confidence Intervals

`Q_total = baseline + topology_gain + verification_gain + tao_gain` is a novel and genuinely useful decomposition. As a point estimate derived from unvalidated proxies it is an accounting identity, not a measurement. Adding bootstrap confidence intervals from the calibration distribution would transform it from "interesting accounting" to "actionable measurement."

### 5.5 LLM-as-Judge Alone Cannot Claim Provable Guarantees

Tier 3 verification has documented biases. For code generation tasks, running the code and counting test failures (Tier 1) is the only non-circular verification. Until Tier 1 is implemented, the "provable quality guarantees" framing is aspirational.

### 5.6 Weiszfeld Failure Mode: Correlated Hallucinations

The Weiszfeld breakdown-point proof (1/2 breakdown) assumes Byzantine faults are *independent* outliers in embedding space. LLMs from the same model family frequently produce **correlated hallucinations** — semantically identical wrong answers whose embeddings cluster tightly. If ≥50% of agents share a correlated hallucination, Weiszfeld will confidently select that hallucination as the geometric median.

This is not a corner case. Models from the same provider share training data, RLHF feedback, and safety filtering — all of which introduce correlation. The `cluster_coherent()` gate (Krum path → pairwise distance < 0.7; Weiszfeld path → cluster incoherent) does not distinguish honest divergence from shared hallucination. Two agents both stating a false fact identically appear "coherent" to the cluster test.

**Structural mitigation implemented (2026-05-03):** `ExplorerSlotConfig` in `TaskManifest.explorers.slot_configs` enforces different CoT reasoning strategies per slot. `ExplorerSlotConfig::diverse_defaults()` provides four maximally decorrelated strategies: `StepByStep`, `DevilsAdvocate`, `FirstPrinciples`, `BackwardChaining`. When `slot_configs` is populated in the manifest, slot `i` receives `slot_configs[i % len]` — distinct role framing and instruction prefix. This makes simultaneous failure structurally less likely even for same-model same-provider deployments by forcing divergent cognitive paths through the problem.

**Remaining mitigations:** (a) Enforce at least 2 distinct model families for N≥3 — `allow_single_family: false` config flag active; (b) document that BFT guarantees apply only to independent-fault regimes; (c) `slot_configs` is opt-in — callers must populate it. The engine default (empty `slot_configs`) preserves prior behavior.

### 5.7 CRDT LUB Is Selection, Not Synthesis

The `ProposalSet` CRDT merge (generation-first LUB) picks a *winning proposal* and discards the rest. It does not synthesize content from multiple proposals. "CRDT merge" as advertised in the architecture suggests something closer to operational transformation (reconciling divergent edits); the implementation is closer to a Last-Write-Wins register keyed on (explorer_id, generation).

This is **architecturally correct** for what it claims to do: crash-safe idempotency and generation-monotonic ordering. The `Team-Swarm Hybrid` Synthesizer role is the intended semantic merge path. The framing issue is that "CRDT merge" suggests content synthesis. Clarification: CRDT provides convergence guarantees over proposal *selection*, not proposal *content synthesis*.

**Resolved (2026-05-03):** `SynthesisPhase` now runs after verification when ≥2 proposals pass. The two-stage critique-then-write pipeline (MoA aggregation pattern) produces a unified output from all verified proposals. Selection (Krum/Weiszfeld/ConsensusMedian) is now the fallback when synthesis fails re-verification or `synthesis_min_proposals` is not met. `HarnessAttribution.synthesis_gain` measures `Q(synthesis) − max(Q(individual))` per task.

### 5.8 Infrastructure Limitations

**NATS message size limit vs. LLM context windows:** NATS default maximum message size is 1 MB. By 2026, compiled `system_context` from a rich constraint corpus can exceed this (1M-token contexts ≈ 4–8 MB). Passing the full context in the `TaskPayload` NATS message will crash the publisher or silently truncate. Industry solution: store large payloads in a content-addressed object store (S3/GCS/MinIO) and pass the hash + reference through NATS. The orchestrator dereferences on receipt. This is a prerequisite for any deployment with large constraint corpora.

**Event replay O(N) for recovery:** `SessionJournal::replay` replays from offset 0. For long-running compound tasks with hundreds of TAO iterations, recovery time scales linearly with task age. Production event-sourced systems (Akka, Temporal) solve this with snapshotting — periodic state saves so recovery loads the latest snapshot + recent events. Without snapshots, `GET /tasks/{id}/recover` becomes slow on long tasks and causes CPU spikes under concurrent recovery.

**Agent scheduler queue starvation:** `LeastLoadedPolicy` sorts by cost_tier ascending first, then by active_tasks. A low-tier agent with 99 tasks always beats a high-tier agent with 0 tasks. Under burst traffic, low-tier agents form deep queues while high-tier agents sit idle, eventually triggering `H2AI_EXPLORER_TIMEOUT_SECS` failures. A cost-aware spillover policy (route to next tier when low-tier queue depth exceeds a threshold) is required for production workloads.

**Compound task semantic deadlocks:** The DFS cycle detection in `CompoundTaskEngine` catches topological cycles (A→B→A). It does not detect semantic deadlocks: subtask B's prompt requires an artifact that subtask A was supposed to produce but did not (or produced in a different form). Because wave execution injects prior outputs as raw text, subtask B will hallucinate the missing artifact and fail silently. Mitigation: structured output validation between waves; explicit artifact contract declarations in the subtask plan.

**Tool use and file-system races:** NATS JetStream coordination is lock-free at the orchestration layer, but `FileSystem` and `Shell` tool-using agents share a mounted workspace volume. Concurrent writes from N shell agents are not mediated by the CRDT event log — file-system-level race conditions persist regardless of swarm size. Lowering N_max reduces collision probability but does not eliminate it. Production tool-using deployments require workspace isolation (per-task volume mounts or ephemeral containers) rather than shared filesystem access.

### 5.9 SelfOptimizer Dead on Success

`SelfOptimizer` runs after every task completion and computes suggested adjustments to N, τ spread, and verification thresholds. These suggestions are applied only when the MAPE-K retry path fires (task failure). On successful runs, the suggestion is discarded. This means the system cannot learn to optimize a successful-but-inefficient swarm — if N=5 succeeds but N=2 would have sufficed, the system continues wasting compute. The bandit (Gap A1, resolved S4) partially addresses this by learning N from verification signal, but `SelfOptimizer` suggestions (topology type, τ range) remain unused on the success path. Tracked as Gap A2.

---

## 6. External Landscape and Positioning

| Framework | N selection | Quality prediction | Merge strategy | Convergence guarantee |
|---|---|---|---|---|
| LangChain/LangGraph | Manual | None | None | None |
| CrewAI | Manual | None | Role-based | None |
| AutoGen | Manual | None | Conversational | None |
| OpenAI Swarm | Manual | None | None | None |
| MoA (Together AI) | Fixed layers | Empirical | Generative aggregation | None |
| H2AI | USL-bounded | Condorcet + eigen | Krum/Weiszfeld/CRDT | CRDT convergence |

**H2AI is doing novel work that no competing framework attempts.** The combination of principled agent count bounding, CRDT-convergent merging, and structured quality attribution has no precedent in published frameworks.

**Layer positioning:** H2AI occupies the *topology and coordination layer* — between the inference layer and the application layer. This clarifies two common comparisons:

- **DSPy (Stanford):** DSPy compiles LLM programs, optimizing prompt weights and few-shot examples using programmatic metrics. It operates *inside* the adapter. A production system can use DSPy inside `IComputeAdapter` to optimize each agent's prompts while H2AI schedules the swarm externally. These are complementary layers, not competitors.
- **Ray (Anyscale):** Ray manages GPU allocation, tensor parallelism, and node-level hardware scaling. H2AI determines *how many* agents to run (N_max from USL) but delegates physical mapping to infrastructure. In a full stack: Ray maps N agents to hardware; H2AI decides N; DSPy optimizes each agent's prompts. Each operates on a different abstraction boundary.

**The strongest empirical competitor is MoA.** MoA uses layered generative aggregation (an LLM synthesizes proposer outputs rather than selecting/voting) and achieves state-of-the-art results with a simpler architecture. H2AI's architecture is mathematically richer. Whether richer math produces better outcomes than empirically-tuned simple aggregation is the key empirical question. MoA's generative aggregation is also the correct path for *semantic content synthesis* — a gap in H2AI's current LUB-selection merge (see Section 5.7).

**Key papers to cite and differentiate against:**

| Paper | Relevance |
|---|---|
| Wang et al. (2024) arxiv 2406.04692 | MoA — generative aggregation baseline, must beat this |
| Li et al. (2025) arxiv 2502.00674 | Self-MoA — single strong model beats diverse models by 6.6% |
| arxiv 2402.05120 | "More Agents Is All You Need" — naive scaling baseline |
| arxiv 2512.08296 | "Towards a Science of Scaling Agent Systems" — coordination overhead model |
| Lefort et al. (2024) arxiv 2409.00094 | CJT failure for LLMs — must cite and address |
| arxiv 2602.03794 | Agent diversity and scaling — supports CG_embed approach |
| arxiv 2507.14928 | BFT-style LLM coordination — closest architectural prior art |
| arxiv 2511.10400 | "Rethinking Reliability of MAS via BFT" — framing context |
| arxiv 2510.18893 | CodeCRDT — CRDT for LLM agents, confirms and bounds the approach |
| Zheng et al. (2023) arxiv 2306.05685 | LLM-as-judge biases — verification limitation |

---

## 7. What Is Genuinely Defensible

Listed in decreasing order of mathematical rigor and domain-transfer confidence:

**1. EigenCalibration N_eff from portfolio theory**  
The participation ratio `(Σλ)²/Σλ²` applied to the adapter similarity matrix directly measures independent perspectives. No contested domain transfer — eigenvalue decomposition is the right tool to answer "how many independent ideas does this adapter pool contain?" Currently uses constraint-profile Hamming CG (corpus-bounded); improves to embedding cosine after Gap E1 (EmbeddingModel).

**2. Generation-first CRDT LUB**  
Architecturally correct solution to the retry ordering problem. All CRDT axioms satisfied. The generation-first rule is the correct mechanism to prevent older high-scored proposals from suppressing TAO refinements. No contested domain transfer.

**3. Three-tier merge dispatch structure**  
The Krum → Weiszfeld → ConsensusMedian hierarchy correctly escalates from outlier-resistant selection to provably robust geometric median as the cluster assumption degrades. The Weiszfeld path has a mathematically proven breakdown point (1/2) for *independent* faults; correlated hallucination is the known failure mode (Section 5.6). Structure correct and active (resolved S1/E4).

**4. Talagrand rank histogram diagnostic**  
Ground-truth-free ensemble calibration borrowed from weather forecasting. Valid signal: flat = calibrated, U-shape = over-confident, Λ-shape = under-dispersed. Wired to τ spread adjustment (resolved S4/A3).

**5. USL-based agent count bounding as a principled heuristic**  
Even under the phenomenological framing, having a calibrated upper bound on agent count is better than "spawn until expensive." The N(N−1) merge complexity is real and measurable. The analogy is structural and honest. The `β₀ × (1−CG)` coupling correctly captures the intuition that divergent agents cost more to reconcile. Useful even without claiming physical law status.

**6. Temporal decay for calibration staleness**  
Ebbinghaus decay with 7-day half-life creates automatic re-calibration pressure. Conservative failure mode (aging toward β₀, i.e., high coordination cost, i.e., lower N_max). Pragmatic and operationally sound.

---

## 8. Implementation Status Summary

All gaps from the original roadmap are resolved. Key remaining open items:

- **Gap C4** — Eigenvalue stopping rule Δ<0.05 is hardcoded; should be config-tunable.
- **Gap A2** — SelfOptimizer suggestions discarded on success path; bandit covers N selection but not topology or τ range.
- **Gap A0** — Conformal prediction buffer (Phase 3, K≥50) deferred until empirical benchmark runs complete.
- **Empirical benchmarks** — Harness built (`scripts/benchmark/`); runs not yet executed.

---

### 8.1 Execution Summary

| Task | Gap | Prerequisite | Status |
|---|---|---|---|
| 1. EmbeddingModel wiring | E1 | — | **DONE (S1)** |
| 2. Replace semantic_jaccard LLM | E2 | 1 | **DONE (S1)** |
| 3. CG → embedding agreement rate | E3 | 2 | **DONE (S1)** |
| 4. β_eff formula verified | P4 | — | **DONE** |
| 5. Online β₀ from token cost | C2 | 2 | **DONE (S3)** |
| 6. Wire Weiszfeld | E4 | 1 | **DONE (S1)** |
| 7. Verification tier system | V1 | — | **DONE (S6)** |
| 8. Bandit over N | A1 | 7 | **DONE (S4)** |
| 9. Calibration pool heterogeneity | C1 | — | **DONE (S3)** |
| 10. Talagrand → τ feedback | A3 | — | **DONE (S4)** |
| 11. Attribution uncertainty (CI) | S5 | — | **DONE (S5)** |
| 12. β_eff signal separation (ρ correction) | S7 | — | **DONE (S7)** |
| 13. Rename Krum → OutlierResistant | D2 | — | **DONE** |
| 14. Update stale validation scripts | D3 | 3 | **DONE** |
| 15. Rename physics.rs → sizing.rs | D1 | — | **DONE** |

**Empirical benchmarking (S8):**
- Benchmark harness built: `scripts/benchmark/` — GSM8K, HumanEval, TruthfulQA runners
- Baselines: B0 (N=1), B1 (majority vote N=6), B2 (MoA 3-layer), B3 (Self-MoA N=5), H2 (H2AI)
- Conformal prediction for Tier 1/2 tasks (arxiv 2406.09714) — deferred until empirical runs complete

---

## 9. Validation Evidence — Script Catalog

| Script | Purpose |
|---|---|
| `scripts/simulate.py` | Visualization: USL curves, β_eff vs CG, N_max vs CG, CJT quality curves, N_eff eigenvalue vs scalar ρ, Talagrand rank histogram shapes |
| `scripts/baseline_eval.py` | Production tool — measures real p and ρ from live adapter; output overrides `baseline_accuracy_proxy` in config |

---

## 10. Open Research Questions

**Q1 — Does role diversity reduce error correlation?**  
arXiv:2506.07962 finds error correlation is driven by training data and architecture, not prompting. Temperature alone does not fix this (arXiv:2508.09654). Does system-prompt role assignment (Advocate/Critic/Synthesizer) on the same base model produce measurably lower pairwise error correlation? Empirical test needed on a verifiable task set. Until answered: diversity mandate should enforce at least 2 distinct model families for N≥3.

**Q2 — Self-MoA vs Multi-Adapter**  
Li et al. (arXiv:2502.00674) found that N samples from one strong model beat N diverse models by 6.6% on AlpacaEval 2.0. If this holds on H2AI's target task domain, inter-adapter CG optimization is optimizing the wrong variable. Empirical test needed: same-model × temperature vs multi-family × role on verifiable task set.

**Q3 — What is the right CG collapse threshold?**  
The hardcoded `cg_collapse_threshold = 0.10` was chosen analytically. Empirical data from live tasks is needed to determine the actual CG_embed value at which TAO first-pass rate drops sharply.

**Q4 — Does USL N_max produce better quality/cost tradeoff than naive scaling?**  
The core thesis, never empirically tested. Run MoA (N=3 layer generative aggregation), majority voting at N={3,6,12}, and H2AI USL-bounded N on the same task suite. If USL-bounded N does not outperform MoA on quality/cost, the mathematical apparatus needs to be positioned as "cost control only," not "quality improvement."

**Q5 — Conformal prediction for free-form text**  
For Tier 1/2 tasks (oracle or grounded verification signal), conformal prediction coverage guarantees are achievable (arxiv 2406.09714). Requires EmbeddingModel. Deferred after Gap E1.

---

## 11. Known Limitations Summary

| Area | Limitation | Current mitigation | Fix path |
|---|---|---|---|
| CG measurement | Semantic embedding cosine (>0.85 threshold) | ✅ Resolved (S1/E3) | — |
| β₀ measurement | Online EMA from merge token cost | ✅ Resolved (S3/C2) | — |
| p and ρ proxies | Derived from CG; S7 ρ correction for Case B | Partially resolved (S7) | Run baseline_eval.py for direct measurement |
| Verification | Three-tier: Oracle HTTP, JsonSchema/Length, LLM judge | ✅ Resolved (S6/V1) | — |
| Attribution | Bootstrap 90% CI + split-conformal PI | ✅ Resolved (S5) | Conformal PI requires oracle signal |
| Krum framing | BFT label for stochastic case | `cluster_coherent()` precondition | ✅ Gap D2 done — "outlier-resistance" framing throughout |
| Weiszfeld | Enabled when embedding model present | ✅ Resolved (S1/E4) | — |
| Bandit over N | Thompson Sampling over N with USL warm prior | ✅ Resolved (S4/A1) | — |
| Talagrand | Wired to τ spread adjustment + DiversityWarning | ✅ Resolved (S4/A3) | — |
| Online β₀ | EMA from actual merge token cost | ✅ Resolved (S3/C2) | — |
| No benchmarks | Harness built (`scripts/benchmark/`); runs not yet executed | Partial (S8) | Run smoke test, then full GSM8K/HumanEval suite |
| `sizing.rs` | physics.rs overclaimed "physical law" status | ✅ Resolved (D1) | — |
| Weiszfeld + correlated hallucinations | BFT guarantee void when ≥50% agents share hallucination | Multi-family enforcement recommendation | Require ≥2 distinct model families for N≥3 |
| NATS message size | Default 1 MB limit; compiled contexts can exceed | Not mitigated | Store payloads in object store; pass hash through NATS |
| Event replay O(N) | `SessionJournal::replay` from offset 0; slow on long tasks | Not mitigated | Snapshot store (Akka/Temporal pattern) |
| Agent scheduler starvation | LeastLoaded: low-tier 99-task beats high-tier 0-task | Not mitigated | Cost-aware spillover policy |
| Compound task semantic deadlock | DFS catches topology cycles; not semantic artifact mismatches | Not mitigated | Structured artifact contracts between subtasks |
| Tool use file-system races | N shell agents share workspace volume; CRDT doesn't prevent FS races | Lower N_max reduces probability | Per-task volume mounts or ephemeral containers |
| SelfOptimizer on success | Suggestions discarded on success; no cross-task learning | Bandit partial (N only) | Gap A2 open — wire SelfOptimizer on success path |
| CRDT = selection not synthesis | LUB picks winner; no content synthesis | Team-Swarm Synthesizer role | MoA-style generative aggregation as alternative merge path |

---

## 12. References

- Gunther, N. J. (1993). Universal Scalability Law. CMG Conference Proceedings.
- Condorcet, M. J. A. N. (1785). Essai sur l'application de l'analyse...
- Nitzan, S. & Paroush, J. (1982). International Economic Review, 23(2), 289–297.
- Ladha, K. K. (1992). American Journal of Political Science, 36(3), 617–634.
- Choueifaty, Y. & Coignard, Y. (2008). Journal of Portfolio Management, 35(1), 40–51.
- Blanchard et al. (2017). Machine learning with adversaries (Krum). NeurIPS.
- Pillutla et al. (2019). Robust aggregation for federated learning. arXiv:1912.13445.
- Vardi, Y. & Zhang, C.-H. (2000). The multivariate L1-median and associated data depth. PNAS.
- Leutbecher, M. & Palmer, T. N. (2008). Ensemble forecasting. J. Computational Physics.
- Brooks, F. P. (1975). The Mythical Man-Month. Addison-Wesley.
- Wang et al. (2024). Mixture-of-Agents. arXiv:2406.04692.
- Li et al. (2025). Self-MoA. arXiv:2502.00674.
- arxiv 2402.05120. More Agents Is All You Need.
- Lefort et al. (2024). CJT empirically tested on LLMs. arXiv:2409.00094.
- Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench. arXiv:2306.05685.
- arxiv 2410.02736. Justice or Prejudice? Biases in LLM-as-Judge.
- arxiv 2410.21819. Self-preference bias in LLM-as-Judge.
- arxiv 2512.08296. Towards a Science of Scaling Agent Systems.
- arxiv 2602.03794. Understanding Agent Scaling via Diversity.
- arxiv 2511.10400. Rethinking the Reliability of MAS via BFT.
- arxiv 2507.14928. Byzantine-Robust Decentralized Coordination of LLM Agents.
- arxiv 2512.20184. Reaching Agreement Among Reasoning LLM Agents.
- arxiv 2510.18893. CodeCRDT — Multi-Agent LLM Code Generation.
- arxiv 2406.09714. LLM Validity via Conformal Prediction. NeurIPS 2024.
- arxiv 2506.07962. Correlated Errors in LLMs (training data drives correlation).
- arxiv 2508.09654. Temperature fails diversity; training loss governs.
- arxiv 2602.00943. Dynamic Prior Thompson Sampling for cold start.
- arxiv 2602.08003. MI-based LLM ensemble selection.
- arxiv 2603.12229. Language Model Teams as Distributed Systems — concurrency controls and consistency protocols for non-deterministic LLM agents.
