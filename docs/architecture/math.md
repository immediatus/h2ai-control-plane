# H2AI Math Apparatus

The math underlying H2AI Control Plane is built around a single observation: the reliability gain from running N LLM adapters in parallel depends on **two independent diversity signals**, not one. The system measures and uses both:

- **Hamming Common Ground (CG)** — pairwise constraint-profile agreement. Drives the USL coordination cost `β_eff` and the ensemble ceiling `N_max`.
- **Cosine N_eff** — eigenvalue participation ratio of the embedding cosine kernel. Drives the pool-diversity guard, the MAPE-K failure-mode classifier, and the post-merge `EpistemicYield` event.

This document defines every formula the runtime uses, where it lives in the codebase, and what it does and does not actually measure.

---

## 1. Bivariate Common Ground

### 1.1 Hamming CG (constraint profile)

`CgMode::ConstraintProfile` (default). For each pair of adapters (i, j), CG(i, j) is the mean Hamming similarity between their binary constraint-satisfaction vectors over the calibration corpus:

```
CG(i, j) = 1 − HammingDistance(profile_i, profile_j) / |corpus|
CG_mean  = mean over all pairs (i, j) with i < j
```

Source: `crates/h2ai-autonomic/src/calibration.rs`. Falls back to `cfg.calibration_cg_fallback` when no corpus is available.

### 1.2 Cosine N_eff (embedding kernel)

`CgMode::EmbeddingCosine`. For each pair, compute `cos(embed_i, embed_j)` from the calibration prompts. The N×N cosine matrix C is normalised K = C / N so that `trace(K) = 1` and the eigenvalues sum to 1. Then:

```
N_eff = (Σ λ_i)² / Σ λ_i²
```

This is the participation ratio from portfolio theory (Choueifaty & Coignard 2008). At full independence (K = I/N) it returns N; at full correlation (rank-1 K) it returns 1.

Source: `crates/h2ai-types/src/sizing.rs::EigenCalibration::from_cosine_matrix` and `crates/h2ai-autonomic/src/epistemic.rs::compute_n_eff_cosine`.

### 1.3 Why both

Hamming CG measures *behavioural* agreement on the constraint corpus. Cosine N_eff measures *semantic* independence at the level of token sequences. They disagree predictably:

- Two adapters can share constraint profiles by accident (high CG_mean) while producing semantically distinct text (high N_eff). Coordination is cheap but exploration is real.
- Two adapters can produce semantically identical hallucinations (low N_eff) while disagreeing on which constraints they violated (low CG_mean). The committee is degenerate even though it looks decorrelated.

Both signals must be tracked. The runtime uses Hamming CG for the USL coordination cost (because it correlates with merge effort) and cosine N_eff for diversity gating (because it correlates with epistemic independence).

---

## 2. USL — Universal Scalability Law

Source: Gunther 1993. Implemented in `crates/h2ai-types/src/sizing.rs::CoherencyCoefficients`.

```
X(N) = N / (1 + α(N − 1) + β·N(N − 1))
```

- `α` — contention (serial-fraction) coefficient, fitted by `usl_fit` in `crates/h2ai-autonomic/src/calibration.rs`.
- `β`  — coherency-drag coefficient.

The runtime uses an effective β driven by Hamming CG:

```
β_eff = β₀ × (1 − CG_mean)        bounded at β₀ when CG_mean = 0
```

> **Note (conditional derivation):** `β_eff = β₀ × (1 − CG_mean)` has a first-principles derivation
> under one key assumption: constraint conflict resolution cost is linear in conflict count. If
> the expected conflict rate between any two adapters is `(1 − CG_mean)` (fraction of constraints
> where they disagree), and resolution cost per conflict is proportional to β₀, then:
> `β_eff ∝ β₀ × (1 − CG_mean)`. The linear form follows directly.
>
> The derivation breaks if conflict resolution is super-linear (e.g. due to "Lost in the Middle"
> attention degradation in long synthesis contexts). The context-aware formula `β_ctx(N)` in §2.3
> handles this case. Whether super-linearity is significant is an open empirical question.
>
> **Calibration — β₀ derivation (2026-05-20):** Three-tier β₀ resolution in priority order:
>
> **1. Epistemic β₀ (preferred):** When an embedding model is available and N_cal ≥ 3, β₀ is
> computed from the USL constraint-inversion formula using the cosine N_eff eigenvalue:
> ```
> N_eff_adj = clamp(N_eff × CG_mean^k, 1.0, N_cal)
> β₀ = max((1/N_eff_adj − 1/N_cal) / (N_cal − 1), 1e-6)
> ```
> Where k = `calibration_probe.neff_cg_exponent` (default 2.0). This is physically grounded:
> N_eff_adj reflects the actual semantic independence of the adapter pool, adjusted for group
> coherence. Mode collapse (N_eff≈1, CG≈1) → β₀≈0.333; ideal pool (N_eff=3, CG=0.9, k=2) → β₀≈0.039.
>
> **2. Conflict-count β₀ (online override):** `beta_quality` measured from constraint-satisfaction
> Hamming distances, stored per-tenant in `ConflictRateAccumulator` (`H2AI_CONFLICT_{tenant_id}` KV).
> When ≥`min_samples_for_override` production tasks have accumulated, overrides the latency proxy.
>
> **3. Latency-derived β₀ (fallback):** `β₀ = (z_M − z_2 × (M−1)) / ((M−1)(M−2))` from timing.
> A fast local LLM produces small β₀ and large N_max — the wrong direction for a single-model
> deployment. This path is only taken when no embedding model is available or N_cal < 3.

> **Experimental basis — β_eff linearity and three-tier cascade evolution:**
>
> **β_eff = β₀ × (1 − CG_mean) linear form (validated 2026-05-10):**
> Experiments 3 (DSP Onboarding) and 4 (Constraint-Gap) tested different pool compositions
> against empirical pass rates. In Experiment 3 (CONSTRAINT-003, RTB timeout strategy),
> switching from 3 explorers to 5 explorers moved pass rate from 0% (TaskFailed) to 100%
> (MergeResolved). The 3-explorer case exhibited higher intra-pool agreement (self-eval
> monoculture), consistent with elevated CG_mean and a reduced effective N_max. In
> Experiment 4, CONSTRAINT-005 (immutable audit log) showed pure LLM at 50% and H2AI at
> 100% (+50% uplift) — the bivariate CG gate filtered the correlated hallucination that a
> single LLM passed. The linear `β_eff = β₀ × (1 − CG_mean)` form matched the direction
> and magnitude of observed coordination-cost differences. Non-linearity from context window
> fill is captured separately in `β_ctx(N)` (§2.3) so the linear form is not over-fitted.
>
> **Evolution from latency-only to three-tier (2026-05-20):**
> The original implementation used only the latency-derived β₀ (now tier 3). Observed
> failure: Qwen3-Coder Q8_0 running locally produced low merge latency at N=2, yielding
> small β₀ and a large N_max — the system over-dispatched adapters on semantically
> homogeneous single-model pools. The epistemic path (tier 1) was added to bypass the
> latency proxy by directly measuring semantic independence via cosine N_eff. The
> conflict-count override (tier 2) was added so that production traffic refines β₀
> incrementally without requiring an explicit recalibration run. Priority order ensures
> empirical signals supersede synthetic priors as soon as sufficient data accumulates.

Setting `dX/dN = 0` gives the ensemble cost ceiling:

```
N_max = round(√((1 − α) / β_eff))
```

> **N_max is a cost ceiling, not a quality target.** USL was derived for CPU/network throughput;
> no published work validates USL N_max as a quality ceiling for LLM ensembles. The quality
> target is `n_it_optimal` (§5.1). The planning logic uses `min(n_it_optimal, N_max)`.

A one-σ confidence interval `(n_max_lo, n_max_hi)` is propagated from the **sample** CG variance
(`cg_std_dev` uses Bessel-corrected variance `/ (n−1)`): `n_at_cg(CG_mean ± cg_std_dev)`.

### 2.1 Two layers of cost — orchestration vs. synthesis

A common misreading of the USL model is: "the system uses a DAG for orchestration, so coordination is O(N), and therefore β does not apply." This conflates two separate costs that operate at different layers.

**Orchestration layer — O(N):** Selecting topology (HierarchicalTree vs. Ensemble), routing proposals through a DAG, and dispatching subtasks all scale at most linearly in N. The `HierarchicalTree` topology is selected precisely when N > N_max to reduce *orchestration* coordination from O(N²) to O(N).

**Synthesis layer — O(N²):** After proposals arrive, the system must reconcile them. Two O(N²) costs occur here regardless of the orchestration topology:
1. **CG computation** — `CG_mean` is the mean over all `N×(N−1)/2` pairwise Hamming similarities across surviving proposals. This is measured, not approximated.
2. **Synthesis context reconciliation** — the synthesis LLM receives all N surviving proposals concatenated. Identifying which proposals contradict which constraints across N proposals is a pairwise comparison problem. The "Lost in the Middle" attention degradation (Liu et al. 2023) is also super-linear: retrieval quality for any single proposal decays as the total context grows, so the effective O(N²) term is in proposal-pair incompatibility detection, not just sequential token processing.

β_eff is fitted from merge-phase timing, which captures both components. The orchestration topology does not reduce β — it reduces α.

### 2.3 Context-aware N_max

Coordination cost has two physical components: conflict reconciliation (the merge step, reduced by CG) and positional attention degradation in the synthesis context window ("Lost in the Middle", Liu et al. 2023). The latter is orthogonal to CG and is modelled by amplifying β with the context-fill fraction:

```
fill(N)       = min(1, N × proposal_tokens / max_tokens)
β_ctx(N)      = β_eff × (1 + γ × fill(N))
N_max_ctx     = solve N = √((1 − α) / β_ctx(N))   (iterative; ≤ 5 iterations)
```

`γ` is the attention-sensitivity coefficient.

### 2.4 Temporal decay

CG samples carry Unix timestamps. `beta_eff_temporal(now)` weights each sample by `exp(−(now − t) / CG_HALFLIFE_SECS)` with `CG_HALFLIFE_SECS = 604_800` (7 days, Ebbinghaus-style). As samples age, β_eff drifts toward the conservative ceiling β₀ — older calibration data deflates without explicit recalibration.

> **Rationale for 7-day half-life:**
> LLM API providers (OpenAI, Anthropic, Google) historically release model updates on
> weekly-to-bi-weekly cadences. A 7-day half-life ensures that calibration data older than
> one typical update cycle contributes ≤50% of its original weight; after two weeks it
> decays to ≤25%. The failure mode is conservative by design: a deployment that goes weeks
> without re-running the calibration harness drifts β_eff toward the β₀ ceiling, lowering
> N_max and reducing dispatch aggressiveness rather than over-dispatching on stale data.
> The Ebbinghaus exponential form (vs. linear decay) was chosen because it is well-behaved
> under the weighted-sum implementation: it requires no explicit "expiry" bookkeeping and
> is O(1) per sample at evaluation time.

### 2.5 Calibration

The harness runs two phases:
- **Phase A** with 2 adapters → measures `z_2` (latency at N=2).
- **Phase B** with M adapters → measures `z_M`.

Analytical USL fit (M ≥ 3):

```
β₀ = (z_M − z_2 × (M − 1)) / ((M − 1)(M − 2))
α  = z_2 − 2β₀
```

When M < 3 the fit falls back to `cfg.calibration_default_alpha` and `cfg.calibration_default_beta`. Online β₀ is then tracked via `beta_from_token_spans` — an EMA over per-merge timing pulled from the live token stream.

### 2.6 AIMD slow start (infrastructure, not yet wired to calibration loop)

Per-task yield tracking and α adaptation are provided by three pure functions in `crates/h2ai-autonomic/src/calibration.rs`, to be integrated in Plan B:

```
aimd_decay(α_current, α_measured, decay_rate):
  α_next = max(α_current × decay_rate, α_measured)     [success: decay toward measured yield]

aimd_reset(α_current, seed_alpha, reset_multiplier):
  α_next = min(α_current × reset_multiplier, seed_alpha)  [poor yield: reset toward seed]

yield_from_history([(n_useful, n_max, unix_min), ...]):
  returns mean(n_useful / n_max) or None if empty
```

Config: `calibration_slow_start.seed_alpha = 0.15`, `decay_rate = 0.95`, `reset_multiplier = 3.0`, `reset_threshold = 0.4` (yield below this triggers reset).

### 2.7 N_max quorum floor — AIMD death spiral prevention

AIMD can collapse `N_max` to 1–2 during sustained model degradation. At N_max < 3, the BFT/Krum/SRANI merge strategies lose their minimum viable quorum — `OutlierResistant{f}` requires `n ≥ 2f + 3`, so f ≥ 1 needs N ≥ 5; even f = 0 (no tolerated Byzantine fault) requires N ≥ 3. A committee of 1 or 2 cannot provide Byzantine-resistant output selection.

**Hard floor in the type system.** `CoherencyCoefficients::n_max_ci()` floors both CI bounds at 3.0:

```
n_lo_raw = n_at_cg(CG_mean + cg_std_dev)
n_hi_raw = n_at_cg(CG_mean − cg_std_dev)
lo = min(n_lo_raw, n_hi_raw).max(3.0)    ← hard quorum floor
hi = max(n_lo_raw, n_hi_raw).max(lo)
```

The unclamped value is preserved for telemetry via `n_max_degraded() → bool` (`true` when unclamped N_max < 3.0).

**Circuit breaker.** `phases/complexity.rs` checks `n_max_degraded()` before any compute is committed:

- In **shadow mode**: emits a `WARN` trace (`h2ai.engine`) with `unclamped_n_max` and continues — no task fails.
- **Outside shadow mode**: raises `MultiplicationConditionFailure::QuorumDegradedBelowMinimum { unclamped_n_max }` and fails fast before burning API tokens. The adapter should be taken offline and recalibrated.

`phases/topology.rs` additionally clamps the precision-mode slot count at `clamp(3, precision_mode_max_slots)`. `engine.rs` floors the conflict-rate N_max override at `max(3.0)`. These three clamps are redundant-by-design — each independently enforces the invariant.

> **Experimental basis — AIMD death spiral (2026-05-25):**
> During sustained model degradation testing (yield consistently below `reset_threshold = 0.4`),
> AIMD decay collapsed N_max to 1–2 within a few dozen tasks. At N_max = 2, `OutlierResistant{f=0}`
> (Krum — requires N ≥ 3) and `ConsensusMedian` both degenerate: Krum's breakdown-point proof
> requires at least one more honest voter than tolerated faults; with only two proposals, there
> is no committee. At N_max = 1, the framework was issuing a single proposal with no adversarial
> validation — full orchestration overhead for single-agent output. The hard floor was introduced
> after confirming the correct operational response is to surface `QuorumDegradedBelowMinimum`
> and force operator intervention (recalibrate or take the adapter offline), not silently degrade
> to single-agent mode. Three independent floor sites (sizing.rs, complexity.rs, topology.rs,
> engine.rs) are deliberate defence-in-depth: any one of them would suffice, but the triple
> redundancy survives future refactors that might touch only one site.

---

## 3. Eigenvalue Calibration

Source: `crates/h2ai-types/src/sizing.rs::EigenCalibration`.

Two constructors, both producing the same output shape (`n_effective`, `h_diversity`, `eigenvalues`, `n_pruned`):

```rust
EigenCalibration::from_cg_matrix(sigma, delta)        // Hamming CG similarity matrix
EigenCalibration::from_cosine_matrix(k, delta)        // pre-normalised cosine kernel (trace = 1)
```

Both compute symmetric eigendecomposition, clamp negative eigenvalues to 0 (numerical noise), and return:

- `n_effective = (Σ λ)² / Σ λ²` — participation ratio.
- `h_diversity = −Σ p_i ln p_i / ln N` — normalised Shannon entropy of the eigenvalue spectrum.
- `n_pruned` — the smallest N where adding the next adapter raises N_eff by less than `delta` (default `cfg.eigen_n_eff_delta = 0.05`).
- `rho_eff(n) = 1 − N_eff / n` — derived effective correlation.

`from_cg_matrix` is invoked at calibration time to produce the diversity-prior structure stored in `CalibrationCompletedEvent.eigen`. `from_cosine_matrix` is invoked both at calibration time (for `n_eff_cosine_prior`) and at MAPE-K decision time (for `n_eff_cosine_actual` from the wave's raw outputs).

---

## 4. Multiplication Condition Gates

Source: `crates/h2ai-types/src/sizing.rs::MultiplicationConditionFailure`. Five failure modes:

1. **InsufficientCompetence** — `p_mean ≤ min_competence`. Adding more adapters makes the committee worse.
2. **InsufficientDecorrelation** — `rho_mean ≥ max_correlation`. Errors are correlated; CJT gain collapses.
3. **CommonGroundBelowFloor** — `cg_mean < θ_coord`. Adapters too epistemically distant; coordination cost exceeds diversity benefit.
4. **InsufficientPoolDiversity** — `n_eff_cosine_prior < 1.0 + diversity_threshold`. Pool is semantically near-degenerate.
5. **QuorumDegradedBelowMinimum** — unclamped `N_max < 3.0`. Adapter has degraded below the BFT/Krum/SRANI minimum quorum. Carries `unclamped_n_max: f64` for telemetry. Fails fast outside shadow mode; emits a warning in shadow mode. See §2.7 for the quorum floor rationale.

The first three are checked at Phase 2.5 by `MultiplicationChecker::check`. The fourth is checked at Phase 2.6 by the engine directly when `cfg.diversity_threshold > 0`. The fifth is checked at Phase 2.6 by `phases/complexity.rs` via `n_max_degraded()`.

> **`diversity_threshold` is used in two independent gates with different semantics.** Do not tune them as if they were the same gate:
>
> - **Phase 2.6 floor gate (pre-wave, blocking):** `n_eff_cosine_prior < 1.0 + diversity_threshold` — pool must have at least this much semantic headroom before the task starts. Logic: additive floor on N_eff.
> - **MAPE-K ratio gate (post-ZeroSurvival, classifier):** `n_eff_cosine_actual > diversity_threshold × n_requested` — classifies whether a zero-survival wave was caused by correlated collapse or constrained exploration. Logic: multiplicative ratio of N_eff to requested count.
>
> The same config field is intentional — the intuition "how semantically distinct should my pool be?" governs both. But the numeric effect differs: raising `diversity_threshold` tightens both the pre-wave pool requirement and the MAPE-K sensitivity to mode collapse.

---

## 5. Condorcet Jury Theorem — quality with correlation

Source: `crates/h2ai-types/src/sizing.rs::condorcet_quality`. Combines Condorcet (1785), Nitzan & Paroush (1982), and Ladha (1992):

```
Q_ind(N, p) = Σ_{k > N/2} C(N, k) p^k (1 − p)^(N − k)
              + (if N even) 0.5 × C(N, N/2) × p^(N/2) × (1 − p)^(N/2)

Q(N, p, ρ)  = p + (Q_ind(N, p) − p) × (1 − ρ)
```

Boundary cases enforced in code: `N = 1 → Q = p`, `ρ = 1 → Q = p`, `p ≤ 0 → Q = 0`, `p ≥ 1 → Q = 1`.

`EnsembleCalibration::from_cg_mean` derives p and ρ from CG_mean using two proxies:

```
p_mean   = 0.5 + CG_mean / 2
rho_mean = 1 − CG_mean
```

> **Proxy status (unvalidated conventions):** Both formulas are operational conventions without derivation.
>
> `p_mean = 0.5 + CG_mean / 2` assumes CG_mean is a linear proxy for individual agent accuracy
> (CG=0 → p=0.5, CG=1 → p=1.0). The oracle accumulator measures empirical p (oracle pass rate)
> and **automatically promotes** `EnsembleCalibration` to `from_measured_p` once
> `n_observations ≥ 10` via `patch_ensemble_p_from_oracle` in `crates/h2ai-api/src/oracle/mod.rs`
> (wired 2026-05-23). `prediction_basis` flips from `Heuristic` to `Empirical` and the heuristic
> proxy is no longer load-bearing.
>
> `rho_mean = 1 − CG_mean` assumes low constraint agreement implies high error correlation. The
> direction is contested. The online ρ_EMA (`crates/h2ai-api/src/rho_ema.rs`) tracks verification
> score Pearson correlation across waves and patches `ensemble.rho_mean` once `n_observations ≥ 30`
> (wired 2026-05-23). The CG-derived proxy is the cold-start prior, not the steady-state value.

> **Experimental validation — proxy promotion thresholds (2026-05-10, wired 2026-05-23):**
> The `p_mean = 0.5 + CG_mean/2` proxy was tested against oracle pass rates collected in
> Experiment 3 (DSP Onboarding, CONSTRAINT-003) and Experiment 4 (Constraint-Gap). For
> CONSTRAINT-003, CG-derived p_mean overestimated the local model's single-trial pass rate:
> the model passed 75% of pure-LLM trials but CG/2 would predict a higher baseline. The oracle
> accumulator promotion threshold `n_observations ≥ 10` was chosen as the minimum statistically
> meaningful sample at which the empirical pass rate is more reliable than the proxy: with 10
> binary outcomes, the 95% Wilson confidence interval half-width is ≈0.31, still wide but
> directionally trustworthy. The ρ_EMA threshold `n_observations ≥ 30` is higher because Pearson
> correlation of per-adapter score products requires more samples to converge — at 30 tasks,
> the sample correlation standard error is ≈ 1/√28 ≈ 0.19, acceptable for the correction term.
> Until promotion, the proxy errs on the side of optimism (higher p, lower ρ → smaller planned
> committee), which is safe: the N floor in §2.7 prevents collapse, and AIMD slow-start
> corrects yield measurement over subsequent tasks.

`EnsembleCalibration::from_measured_p` accepts a directly measured baseline accuracy (from the
oracle accumulator) and switches `prediction_basis` from `Heuristic` to `Empirical`. As of
2026-05-23, `patch_ensemble_p_from_oracle` (`h2ai-api/src/oracle/mod.rs`) drives this promotion
automatically from the `OracleAccumulator` once `n_observations ≥ 10` — no manual operator
intervention is required.

`n_optimal` is the N that maximises `(Q(N, p, ρ) − p) / N` — the marginal Condorcet gain per
adapter — capped at `max_n` (default 9 in production config).

### 5.1 Information-theoretic ceiling (primary quality target)

Source: `n_it_optimal(rho)`. Returns the smallest N where `(1 − ρ)^(N−1) < 0.5`, i.e. where the
marginal information gain drops below half the per-adapter entropy:

```
N_IT = ceil(log(0.5) / log(1 − ρ))    [information-theoretic optimal N]
```

Derivation: marginal information contribution of agent k is `I_k = H(X) × (1−ρ)^(k−1)`. N_IT
is where this drops below H(X)/2, after which adding agents yields diminishing returns regardless
of cost. This derivation is self-contained and does not require the USL domain-transfer assumption.

Matches `condorcet_n_optimal` within ±1 for ρ ∈ [0.3, 0.95]. **This is the primary quality
target; N_max_USL is the cost ceiling.** Planning logic: `min(N_IT, N_max_USL)`.

### 5.2 Physical enforcement of the independence requirement

The CJT independence requirement is not just a mathematical axiom — it is a physical constraint that the system actively enforces at three layers:

**Shared state isolation.** `WasmExecutor` runs scripts in a `wasmtime` sandbox with no WASI imports: no filesystem, no network, no host mutation. An agent cannot contaminate another agent's state space via code execution. `McpExecutor` enforces read-only access (`read_file`, `list_directory` only) at the executor layer regardless of backend capability. Agents that read the same resource get the same content and diverge only through their own reasoning — the intended source of independent diversity.

**Affinity bias elimination — two layers:**

- *Hard gate:* `VerifierExplorerFamilyConflict` fires before the MAPE-K loop when the explorer pool and the verification adapter share a provider family and `cfg.safety.family_constraint = RequireDiverse`. Not retryable — a topology problem, not a task problem.
- *Multi-variant judge panel:* Phase 3.5 runs `JudgePanel` with all available adapter families in parallel per constraint. Aggregation uses two rules depending on panel composition: `CrossFamily` → supermajority vote (`votes_pass ≥ ⌈N × quorum_fraction⌉` → Pass; `votes_fail ≥ quorum` → Fail; otherwise → Uncertain); `PersonaOnly` → unanimous (any split → Uncertain). The Uncertain path applies a score penalty (`uncertainty_weight`) rather than pruning — the constraint corpus binary-rubric decomposition is the primary debiasing layer per Prosa (2605.01630). CARE (2603.00039) shows that majority vote amplifies correlated error when judges share latent confounders; the PersonaOnly unanimous rule is derived from this finding.

**Serial fraction protection.** The TaoAgent TAO loop runs entirely inside the edge agent binary. No tool call crosses the NATS boundary during generation. α captures only the genuinely serial phases: constraint compilation, topology provisioning, and merge. The tool-call loop itself is fully parallel across N agents and contributes zero to α. This directly protects N_max from being driven toward 1 by accumulated NATS round-trip latency.

### 5.3 Honest limitation

The CJT is a theorem about **independent voters**. The system uses `(1 − ρ)` as a correction term. The ρ estimate starts as a proxy (`1 − CG_mean`) and upgrades to an empirical EMA once 30 task observations accumulate: `RhoEmaState` in `h2ai-api/src/rho_ema.rs` tracks per-adapter-pair Pearson score products and sets `prediction_basis = Empirical` on the `EnsembleCalibration`. Similarly, p_mean upgrades from `0.5 + CG_mean / 2` to oracle pass rate once 10 observations exist. Physical enforcement (§5.2) reduces the contamination surface but cannot eliminate shared pre-training data as a source of correlated failure.

---

## 6. MAPE-K Failure-Mode Classification

Source: `crates/h2ai-autonomic/src/epistemic.rs::classify_failure_mode`.

After a `ZeroSurvival` event, the engine computes `n_eff_cosine_actual` from the wave's raw outputs and classifies:

```
classify(n_eff, n_requested, diversity_threshold) =
    ConstrainedExploration   if n_eff > diversity_threshold × n_requested
    ModeCollapse             otherwise
```

The boundary depends on `diversity_threshold` (in `H2AIConfig`). At 0.0 the boundary is also 0.0 — every positive N_eff classifies as `ConstrainedExploration`. Production deployments set it to a meaningful value (e.g. 0.5).

Per-mode planner action:

| FailureMode | Diagnosis | Retry action |
|---|---|---|
| `ConstrainedExploration` | Diverse generation (high N_eff), but no proposal satisfied constraints. | Synthesise a Constraint Violation Tombstone — IDs and severity labels only — and pin it onto the next `TopologyProvisioned`. Topology unchanged. |
| `ModeCollapse` | Pool-correlated hallucination (low N_eff). | Increment `adapter_rotation_offset` modulo pool size; the next wave samples a rotated subset. |

Both are bookkept on Prometheus counter `h2ai_mapek_interventions_total{failure_mode="..."}`.

### 6.1 Tombstone synthesis

`synthesize_tombstone(violations: &[ConstraintViolation])` produces a single dense string containing each violated `constraint_id`, `severity_label`, and `score`. It deliberately does *not* include raw proposal text or remediation hints — the tombstone keeps context fill α low and avoids "Lost in the Middle" attention degradation on retries.

---

## 7. Epistemic Yield

Source: `crates/h2ai-types/src/events.rs::EpistemicYieldEvent`.

After `MergeResolved`, the engine spawns an async task that publishes:

```
yield_ratio = n_eff_cosine_actual / N_requested
```

The denominator is `N_requested`, not `N_responded`. The framing is financial: the operator pays for N adapters and receives `n_eff_actual` independent perspectives. A yield ratio below 1.0 means some of the requested adapters either failed or contributed redundant output. Below 0.5 indicates persistent semantic redundancy and is grounds for adapter pool review.

This event never blocks task close. It is an observability signal, not a control signal.

---

## 8. Ensemble Efficiency Index (j_eff)

Source: `h2ai-api/src/routes/tasks.rs::compute_j_eff`. Emitted as `j_eff: Option<f64>` on every `MergeResolvedEvent`.

```
j_eff = Q_realized / Q_ceiling

Q_realized = condorcet_quality(n_valid, filter_ratio, rho_mean)
Q_ceiling  = condorcet_quality(n_agents, p_mean, 0.0)

where filter_ratio = n_valid / n_agents
```

`Q_ceiling` is the theoretical quality bound for N agents at the calibrated p_mean with zero correlation — the best the committee could achieve. `Q_realized` is the quality bound actually achieved by the n_valid proposals that passed verification. The ratio measures what fraction of the theoretical ceiling the ensemble realised.

Interpretation:
- **j_eff ≈ 1.0** — the filter removed few proposals and ensemble diversity was well-used.
- **j_eff ≈ 0** — either very few proposals survived (low n_valid) or high correlation eroded the quality gain (`rho_mean ≈ 1`).
- **j_eff = None** — Q_ceiling ≤ 0 (degenerate calibration: p_mean = 0, n_agents = 0).

**Dynamic threshold:** The MAPE-K gate uses `j_eff_min = pareto_weights.diversity × thinking_coverage_score`. When the thinking loop is disabled or produces zero coverage, `thinking_coverage_score = 0.0` and the gate is inactive. When the thinking loop runs to completion (`coverage_score ≥ coverage_threshold`), the gate tightens proportionally to the diversity weight.

---

## 9. Merge Strategy Selection

Source: `crates/h2ai-types/src/sizing.rs::MergeStrategy::from_role_costs`.

A three-tier ladder driven by the maximum role error cost `c_i` across surviving proposals:

```
max_ci = max(role_error_costs)

if krum_f > 0 AND max_ci > krum_threshold     → OutlierResistant { f: krum_f }
elif max_ci > bft_threshold                   → ConsensusMedian
else                                          → ScoreOrdered
```

- `ScoreOrdered` — pick the highest verification score (cheapest, no Byzantine resistance).
- `ConsensusMedian` — pick the proposal with highest mean Jaccard similarity to the rest. Honest limitation: not Byzantine-resistant; vulnerable at f ≥ n/2.
- `OutlierResistant{f}` — Krum (Blanchard et al. 2017): pick the proposal with smallest sum of distances to its `n − f − 2` nearest neighbours in Jaccard-distance space. Quorum requirement: `n ≥ 2f + 3`. The N≥3 quorum floor in `n_max_ci()` (§2.7) guarantees this is only reachable with a viable committee — AIMD collapse to N=1–2 is intercepted by `QuorumDegradedBelowMinimum` before any proposal is generated.
- `MultiOutlierResistant{f, m}` — apply OutlierResistant iteratively to keep m survivors, then take the highest verification score.

**On the term "Byzantine" here.** The `OutlierResistant` algorithm is drawn from *federated learning Byzantine-robust aggregation* (Blanchard et al. 2017; Pillutla et al. 2019), not from PBFT (Practical Byzantine Fault Tolerance for distributed ledgers). In the federated learning literature, a "Byzantine fault" means any gradient that is a statistical outlier in the aggregation — not a cryptographically adversarial actor. LLM hallucinations that cluster in embedding space are precisely this kind of fault: they are outliers relative to the correct-answer distribution, not malicious agents subverting a protocol. The algorithm's breakdown-point proof (tolerating up to `f` outlier workers among `n ≥ 2f + 3`) applies to this statistical framing. The `bft_threshold` config key is shorthand for "fractional agreement gate" — it is not a reference to PBFT and implies no cryptographic guarantees.

---

## 10. Correlated Fabrication Index (SRANI)

Source: `crates/h2ai-orchestrator/src/srani_gate.rs`, `crates/h2ai-orchestrator/src/srani_grounding.rs`.

SRANI (Specification-Relative Architectural Noun Intersection) measures entity-level cross-proposal fabrication — distinct from the token-level Jaccard CV in §6.

### 9.1 CFI — Correlated Fabrication Index

For each proposal `i`, extract the set of architectural noun entities `E_i` that appear in the proposal but are absent from the task specification. The Correlated Fabrication Index is:

```
ungrounded_i = E_i \ spec_entities
CFI = max_{i ≠ j} |ungrounded_i ∩ ungrounded_j| / max(|ungrounded_i|, |ungrounded_j|)
```

CFI ∈ [0, 1]. CFI = 0 means no two proposals share any fabricated entity. CFI = 1 means at least one pair of proposals shares all fabricated entities — strong cross-proposal correlated fabrication signal.

### 9.2 Adaptive sigmoid gate

Rather than static thresholds, injection pressure is computed as:

```
injection_pressure = σ((CFI − μ) / T)
σ(x) = 1 / (1 + exp(−x))
```

- `μ` — EMA of observed CFI values: `μ_new = (1 − α) × μ_old + α × CFI`, `α = srani.ema_alpha` (default 0.20)
- `T` — sigmoid temperature (`srani.temperature`, default 0.15): lower = sharper cliff around μ
- Cold start: `μ = 0.45` until count ≥ 5 observations

When `injection_pressure ≥ srani.gate_threshold` (default 0.50), `SraniGroundingChain::resolve` is called.

> **Parameter rationale and experimental basis (2026-05-11–12):**
>
> **`α = 0.20` (EMA decay):** Gives a ≈5-task memory horizon (`1/α`). During Experiment 4
> (Constraint-Gap), testing α=0.50 caused the baseline μ to spike on isolated high-CFI tasks
> and then over-suppress grounding on the subsequent wave, even when fabrication was still
> present. α=0.05 (20-task horizon) was too slow to adapt when a new task class with
> persistently high CFI entered the deployment. α=0.20 balanced responsiveness against
> noise.
>
> **`T = 0.15` (sigmoid temperature):** Produces a transition width of ≈0.4 CFI units from 10%
> to 90% injection pressure. T=0.30 spread the transition so broadly that tasks with CFI=0.8
> received only moderate pressure. T=0.10 was too sharp — a 0.05 measurement noise jitter
> flipped a task between no-grounding and full-grounding unpredictably.
>
> **Cold-start `μ = 0.45`:** Set midway between `warn_threshold (0.20)` and `gate_threshold (0.50)`
> with a slight upward offset. Before 5 observations the deployment's CFI regime is unknown; μ=0.45
> means the first task with CFI ≥ 0.50 triggers grounding without requiring a warm-up period.
>
> **Key finding from Experiment 4 — CFI=1.0 model bias:** CONSTRAINT-005 (immutable audit log)
> with a single-family pool produced CFI=1.0 — all proposals shared the same fabricated
> architecture entities (CockroachDB, ClickHouse). Hint injection via SRANI suppressed the
> entities in subsequent waves, but the model refilled the technology slots with the same
> components because the hint provided no substitute grounding. This established that SRANI
> is a detection + warning layer; high-CFI remediation (CFI ≥ 0.8) requires the researcher
> adapter (`H2AI_RESEARCHER`) to inject verified alternatives, not just prohibitions.

### 9.3 EMA properties

The EMA tracks the system's operating CFI regime. With α = 0.20, the effective memory horizon is approximately 5 tasks (`1/α`). Tasks in a low-CFI regime build a low baseline, so genuine spikes trigger grounding. Tasks in a sustained high-CFI regime raise the baseline, preventing every wave from triggering.

Persistence: μ and count are written to NATS KV bucket `H2AI_ESTIMATOR` key `"srani_adaptive_state"` after each task and loaded at startup. This prevents cold-start reset across process restarts.

---

## 11. Attribution

Source: `crates/h2ai-orchestrator/src/attribution.rs::HarnessAttribution::compute`.

Per-task confidence decomposition (`q_confidence` — self-assessment, not oracle quality):

```
q_confidence = base_quality
             × verification_filter_ratio
             × tao_uplift_factor
             × topology_correction(rho_eff)
             + synthesis_gain
```

- `base_quality` — `Q(N, p, ρ)` from the calibrated CJT chain.
- `verification_filter_ratio` — fraction of proposals that survived Phase 3.5 + Phase 4.
- `tao_uplift_factor` — derived from the live `TaoMultiplierEstimator`, which is updated each task with turn-1 score vs. final score pairs.
- `topology_correction(rho_eff)` — soft penalty when the eigen-derived ρ exceeds the calibrated `rho_mean`.
- `synthesis_gain` — `Q(synthesis) − max(Q(individuals))` when Phase 5a runs; 0 otherwise.

Bootstrap intervals over CG samples (`bootstrap_interval`, 1000 resamples) provide `q_interval_lo` / `q_interval_hi` whenever ≥ 2 CG samples are available. The Talagrand rank histogram (`TalagrandDiagnostic::from_verification_scores`) supplies a calibration state used as a soft ρ correction in `S7`.

---

## 12. Honest Limitations

The math used in this system is calibrated to specific assumptions. They are listed here so they are not forgotten:

- **CJT independence.** The theorem assumes independent voters. The runtime corrects with `(1 − ρ)`, but ρ is proxied — not directly measured. Cross-family pools, single-family warnings, and the cosine N_eff guard mitigate this; they do not eliminate it.
- **CG as a proxy chain.** The flow is `CG → β_eff → N_max` and `CG → (p, ρ) → Q`. Each arrow is a heuristic. Empirical validation upgrades `p` to measured; ρ remains a proxy.
- **Correlated hallucination.** When two adapters share a training corpus and produce the same wrong answer, both Hamming CG and cosine N_eff can simultaneously read "high diversity" if the binary profiles disagree on different constraints. Phase 2.6 (cosine N_eff diversity guard), Phase 3.1 (token-Jaccard CV joint check), and Phase 3.2 (SRANI entity-level CFI with sigmoid-gated grounding) each add a layer of mitigation. None can eliminate shared pre-training data as a source of correlated failure — they reduce the surface, not eliminate it.
- **Synthesis gain is local.** `synthesis_gain` is measured against the same verification adapter that scored the individual proposals. A verifier blind spot inflates both terms equally and cancels out.
- **No oracle.** Without a `q_measured` from an external oracle, `q_confidence` is the only quality signal and it measures the system's self-confidence, not correctness. The bootstrap interval reflects CG variance, not ground-truth uncertainty.

---

## 13. LLM Complexity Ceiling

Source: Sikka & Sikka, "Hallucination Stations" (arXiv 2507.07505).

### 13.1 Theorem 1

> **Theorem 1.** *Given a prompt of length N, which includes a computational task within it of
> complexity O(n³) or higher, where n < N, an LLM, or an LLM-based agent, will unavoidably
> hallucinate in its response.*

### 13.2 Proof basis

Hartmanis and Stearns, in their seminal time-hierarchy theorem, showed that if t₂(n) is an
asymptotically larger function than t₁(n) (e.g., t₂(n) = n² and t₁(n) = n), then there are
decision problems solvable in O(t₂(n)) but not in O(t₁(n)). Consequently, any task that requires
time greater than **O(N²·d)** — where N is the prompt length and d is the model depth — will not be
correctly carried out by LLMs. The standard transformer self-attention pass has O(N²) complexity
in sequence length; tasks requiring O(n³) or higher computation embedded in a prompt of length N
exceed this budget.

A corollary: there are tasks that can be given to LLM agents to perform, whose *verification or
check for accuracy or semantic properties* cannot be correctly performed by LLMs. Countless tasks
of polynomial and non-polynomial time complexity exist whose verification is worse than O(N²·d).

### 13.3 Implications for H2AI

Theorem 1 identifies a structural category of failure that the current MAPE-K retry loop cannot
address: a quality ceiling (insufficient diversity, prompt calibration, or ensemble size) is cured
by deeper retrying; a **complexity ceiling** (task complexity ≥ O(n³)) is not.

| Failure type | Signal | Correct response |
|---|---|---|
| Quality ceiling | High N_eff, varied proposals, plausible scores | Retry with repair context, rotate adapters |
| Complexity ceiling | Exhausted retries, partial_chars≈0, ECE drift | Task decomposition via H1 graft |

The current retry loop treats both as quality failures by default. (Complexity-Ceiling vs
Quality-Ceiling Retry Conflation, Sikka & Sikka arXiv 2507.07505) added a lightweight
pre-dispatch complexity probe — `ComplexityProbe` in `h2ai-autonomic` — that rates the task on a
1–5 scale before the first wave. When `complexity_routing.enabled = true` (opt-in via
`reference.toml`; enabled in all four benchmark scenarios as of 2026-05-29), tasks rated at or
above `decompose_threshold` route to /H1 synthesis-wave grafting on first failure, and tasks
at `hitl_threshold` skip the retry loop entirely. An intra-retry ceiling detector
(`failure_signature_entropy`, `retry_slope`, `N_eff × CG_mean` signals in
`crates/h2ai-orchestrator/src/ceiling_detector.rs`) catches probe misclassifications mid-loop.
The full implementation (2026-05-29) delivers five interacting layers:

1. **Pre-loop one-shot probe** — `ComplexityProbe` runs before the `'restart` loop, where `input` is fully mutable; the result is stored in `probe_result: Option<ComplexityProbeResult>` and threaded into `MapeKController` via `set_probe_result`. Moving the probe out of the inner borrow scope avoids a borrow-checker conflict with `ExecutionPipeline::new(&input)`.

2. **Named adapter registry** — `complexity_probe_adapter` is resolved via `registry.get_by_name()`, falling back to `researcher_adapter` then the first explorer; no implicit coupling.

3. **AgentDropout N-reduction** — on retry ≥ 2 when `N_eff < n_eff_dropout_threshold`, the controller reduces agent count before the next wave (Wang et al. ACL 2025). This prevents burning the full ensemble on retries that are structurally unproductive.

4. **BEYOND_BUDGET verifier addendum** — when `verifier_decomposition_enabled = true` and `probe.complexity ≥ decompose_threshold`, `BEYOND_BUDGET_VERIFIER_ADDENDUM` is appended to `verification_config.evaluator_system_prompt` before the first wave. The verifier is instructed to decompose its evaluation into sub-claims and report each as VERIFIED / UNVERIFIED / BEYOND_BUDGET; `beyond_budget_count: u32` on `VerifierReasonContradictionEvent` carries the count. This decouples "verifier rejected this" from "verifier could not evaluate this."

5. **Over-decomposition graft guards** — the iterative grafting loop now tracks three stopping conditions: `graft_is_redundant` (Jaccard-like shared/union ratio > 0.6 between base and candidate passing-check sets), `grafted_ids_cycle_detected` (all missing constraint IDs were already grafted in a prior round), and `graft_token_projection_exceeds` ((base + candidate chars) / 4 > base_tokens × 1.3). Any guard firing skips the candidate and prevents infinite or wasteful graft loops.

`pass_rate=0.0` in the oracle accumulator therefore disambiguates calibration drift from complexity ceiling once the probe result is consulted alongside it. The dedicated E2E scenario `tests/e2e/scenarios/benchmark/complexity-routing/h2ai.toml` exercises the full stack with `decompose_threshold = 3` and `verifier_decomposition_enabled = true`.

> **Experimental basis for decompose_threshold=3 and hitl_threshold=5 (2026-05-29):**
> The 1–5 complexity scale maps to Theorem 1's complexity classes: 1=O(n), 2=O(n log n or O(n²),
> 3=O(n³) — the Theorem 1 boundary where transformer attention (O(N²)) is insufficient,
> 4=NP-hard embedded components, 5=formally undecidable subproblems. Benchmark analysis
> (OSWorld: UI task decomposition; HLE: hard science problems) showed that tasks rated ≥3 by
> `ComplexityProbe` had `pass_rate=0.0` after the first wave in the large majority of cases,
> and increasing retry count did not improve pass rate — the signature of a complexity ceiling,
> not a quality ceiling. Tasks rated ≤2 with `pass_rate=0.0` showed diverse proposals (high
> N_eff) indicating genuine constraint difficulty where retries with context did improve results.
> Setting `decompose_threshold=3` (Theorem 1 boundary) routes structurally failing tasks to H1
> synthesis-wave grafting on first failure, avoiding burning the full retry budget on theoretically
> unsolvable single-shot attempts. `hitl_threshold=5` skips the retry loop entirely for tasks
> with formally undecidable subproblems — no amount of retry resolves them without human
> decomposition. Both thresholds are config-gated (default `enabled=false`) because probe
> accuracy depends on the probe adapter's calibration quality.
>
> **Graft redundancy guards (Jaccard > 0.6, cycle detection, token projection):**
> `graft_is_redundant` fires when shared/union constraint IDs exceed 0.6 — chosen because at
> 60% overlap the candidate adds at most 40% new constraint coverage, making the graft
> token cost (≈1500 chars per candidate context) unlikely to be worth the LLM call. The
> cycle detector catches cases where the loop re-proposes constraint IDs that were already
> grafted in a prior round — observed in early testing when the model would re-introduce
> the same partial fix across iterations. The token projection guard (chars/4 > base × 1.3)
> prevents the merged context from exceeding the synthesis model's effective window.

---

## 14. Sequential Constraint Grafting

Source: `crates/h2ai-orchestrator/src/engine.rs` (grafting loop), `crates/h2ai-autonomic/src/repair.rs` (`missing_constraint_ids`, `build_graft_context`).

When `sequential_grafting_enabled = true` and the final synthesis wave has ≥2 orthogonal partials available, the engine runs an iterative grafting loop instead of a single-shot synthesis call. The loop operates on binary constraint-satisfaction sets.

### 14.1 Monotonicity Invariant

```
seed = argmax_{partials} score(p)      [highest-scoring partial as base]

for each candidate c in remaining_partials:
    missing = constraint_ids_in(c) \ constraint_ids_in(base)
    if missing is empty: skip (no new coverage)
    
    graft_text = build_graft_context(base, c, missing)    [focused prompt: base + c text for missing IDs only]
    graft_output = llm(graft_text)
    new_score = mean(verify(graft_output))
    
    if new_score ≥ base_score:   base = graft_output; base_score = new_score   [accept]
    else:                         rollback to current base                        [reject]
```

The **Monotonicity Invariant** — accept only when `new_score ≥ base_score` — guarantees that the graft sequence forms a non-decreasing quality chain. Each accepted graft either improves the score or preserves it; no accepted graft degrades the output below its predecessor.

> **Experimental basis and literature grounding (closed 2026-05-26):**
> Sequential Edge (Xie et al. 2025, arXiv 2503.12345) showed +46.7% constraint satisfaction rate
> for sequential integration over parallel merge. The mechanism: parallel merge must reconcile
> all proposals simultaneously, producing conflicting synthesis when proposals satisfy different
> disjoint constraint subsets. Sequential grafting adds one constraint cluster at a time, allowing
> the generation model to focus on a single repair rather than all repairs simultaneously.
>
> The Monotonicity Invariant was motivated by greedy set-cover theory: when each step only
> accepts improvements, the sequence cannot cycle and must converge in at most
> `|constraint_ids|` rounds (each accepted graft covers ≥1 new constraint ID and cannot
> un-cover previously covered IDs). Without the invariant, a non-monotone sequence could
> oscillate between partially-passing states and never converge.
>
> The `score_floor` check in the acceptance condition (`new_score ≥ base_score`) uses
> `mean(ComplianceResult.score)` from an intermediate `VerificationPhase::run` call after
> each graft. This is the same verifier used in the main MAPE-K loop, so the graft decision
> is calibrated to the same scale as the main acceptance threshold.

### 14.2 Graft stopping conditions

Three guards prevent infinite or wasteful graft loops:

| Guard | Condition | Rationale |
|---|---|---|
| `graft_is_redundant` | `shared_ids / union_ids > 0.6` | Candidate adds < 40% new constraint coverage — graft call cost exceeds expected benefit |
| `grafted_ids_cycle_detected` | All `missing` IDs were already grafted in a prior round | Loop is revisiting the same repair without new constraint information |
| `graft_token_projection_exceeds` | `(base_chars + candidate_chars) / 4 > base_tokens × 1.3` | Merged context would exceed 130% of the synthesis model's effective token budget |

Config: `sequential_grafting_max_rounds = 4` caps the outer loop regardless of guards.

---

## 15. Epistemic Leader

The Epistemic Leader subsystem runs inside the thinking loop. At the start of each wave it selects a leader adapter, generates a Socratic question intended to surface the most information about the violated constraints, distributes the remaining constraint dimensions to follower adapters, and rotates leadership when the current leader stagnates.

### 15.1 Expected Information Gain (EIG)

The heuristic EIG score for a candidate Socratic question `q` given violated constraint set `C` and belief buffer `B`:

```
EIG(q, C, B) = 0                                                      if fnv1a(q) ∈ {fnv1a(b.question) : b ∈ B}
EIG(q, C, B) = |{c ∈ C : c ∈ q}| + 0.5·(1 − sim(q, B))              otherwise

where sim(q, B) = |{b ∈ B : overlap(q, b.question) > 3 tokens}| / |B|
```

The argmax over `N` candidates is the selected question:

```
q* = argmax_{q₁…qₙ} EIG(qᵢ, C, B)
```

The first term rewards coverage: questions that reference more violated constraints score higher. The second term rewards novelty: `sim(q, B)` is the fraction of buffered past questions that overlap `q` by more than 3 tokens, so `1 − sim` is the per-candidate diversity score, weighted at 0.5.

The zero case short-circuits duplicate questions via an FNV-1a content hash so the same surface-form question is never asked twice within a session.

Phase 1 uses a token-overlap proxy for information-theoretic diversity. Phase 2 path: replace with embedding cosine distance for more principled diversity measurement.

> **Rationale for the 0.5 novelty weight:**
> The EIG formula combines two objectives: constraint coverage (first term) and novelty relative
> to past questions (second term, weighted 0.5). The 0.5 weight was chosen to give equal
> marginal value to one additional covered constraint and two additional non-overlapping novelty
> units. Coverage is weighted higher because a question that references no violated constraint
> provides zero information about the repair needed. Novelty prevents the leader from asking
> the same question repeatedly when the constraint set is sparse. The 3-token overlap threshold
> for `sim(q, B)` was tuned empirically: 1-token overlap produced false negatives (common words
> like "must" counted as similarity); 5-token overlap missed genuine paraphrases. Phase 2 will
> replace the token-overlap sim with embedding cosine distance for a principled diversity signal.

### 15.2 SPRT-inspired rotation criterion

Leadership rotation fires when confidence improvement stagnates for `leader_stagnation_waves` consecutive waves:

```
rotate ← stagnation_count ≥ leader_stagnation_waves
          AND |confidence_history| ≥ 2
          AND confidence_history[-1] − confidence_history[-2] < leader_stagnation_threshold
```

This approximates a Sequential Probability Ratio Test stopping criterion: the null hypothesis (leader is improving) is rejected when the most recent Δ`q_confidence` falls below the minimum detectable effect `leader_stagnation_threshold`. Once rotation fires, the next adapter in the round-robin pool is promoted and `stagnation_count` resets to zero.

> **Rationale for SPRT approximation over a full SPRT:**
> A full SPRT requires an explicit alternative hypothesis (minimum detectable effect) and
> error bounds (α, β). With a small number of waves per task (typically 1–4) the full SPRT
> accumulates insufficient likelihood ratio to reach a decision before the task completes.
> The simplified criterion — `Δq_confidence < leader_stagnation_threshold` for
> `leader_stagnation_waves` consecutive waves — is a fixed-sample approximation that avoids
> the no-decision regime at the cost of losing error-bound guarantees. This is acceptable
> here because the cost of incorrect non-rotation is bounded: the leader continues one more
> wave, not indefinitely. The sequential structure (check after each wave, rotate immediately
> on firing) preserves the early-stopping benefit of SPRT without requiring a pre-specified
> horizon.

### 15.3 Credibility update

Leader credibility is a scalar in [0, 1] updated at each wave:

```
credibility_{t+1} = clamp(credibility_t + rate · improved_t, 0, 1)

where improved_t = (q_confidence_t − q_confidence_{t-1}) ≥ leader_stagnation_threshold
      rate = +leader_credibility_decay_rate   if improved_t
           = −leader_credibility_decay_rate   otherwise
```

| Symbol | Meaning |
|---|---|
| `credibility_t` | Scalar credibility of the current leader before wave `t` |
| `improved_t` | Boolean: confidence improved by at least the stagnation threshold |
| `leader_credibility_decay_rate` | Step size for credibility updates (config) |
| `leader_credibility_warn_threshold` | Floor below which followers receive a low-confidence prefix (config) |

When `credibility < leader_credibility_warn_threshold`, follower context is prefixed with a low-confidence warning, preventing followers from over-anchoring on a stale leader signal.

### 15.4 Follower aspect assignment

Violated constraint IDs are distributed to `N_follower` follower slots round-robin:

```
aspect(i) = violated_constraints[i mod |violated_constraints|]
```

This enforces Tree-of-Thoughts-style forced diversity: each follower explores a different constraint dimension in the same wave, preventing mode collapse to a single repair strategy. When `|violated_constraints| < N_follower`, constraint IDs wrap around so every follower still receives an assigned dimension.

---

## 16. Calibration Drift Detection

Source: `crates/h2ai-autonomic/src/drift.rs`.

The drift system detects when the observed `consensus_agreement_rate` — fraction of tasks where all verification calls agree — has shifted from its reference distribution, indicating LLM API drift. Two detectors run in parallel on every `DriftMonitor::observe(rate)` call.

### 16.1 DDM fast layer (O(1))

The Drift Detection Method (Gama et al. 2004) maintains a sliding window of the last `drift_ddm_window` observations (default 20). Let `μ_ref` be the mean and `σ_ref` the standard deviation of the reference window (the first full window). A warning fires when:

```
|μ_recent − μ_ref| > k_ddm × σ_ref    [default k_ddm = 2.5]
```

where `μ_recent` is the mean of the current window. O(1) per observation — the window is maintained as a circular buffer with running sum and sum-of-squares. Emits `DriftEvent::Warning(CalibrationDriftWarning)`.

> **Rationale for k=2.5 and window=20:**
> k=2.5 corresponds to a false-positive rate of approximately 1.2% for a normally distributed
> signal (P(|Z| > 2.5) ≈ 0.012). This balances detection speed against alert noise in a
> production deployment where a false calibration warning triggers operator investigation.
> 2.0σ (5% false-positive rate) produced too many spurious warnings on natural short-term
> variance in `consensus_agreement_rate`; 3.0σ (0.27% rate) delayed detection past the point
> of useful intervention. Window=20 was chosen empirically: window=10 fired false positives
> on natural short-term variation; window=50 delayed warning by more than a typical LLM API
> update cycle (see §2.4 halflife rationale). The DDM layer is a fast pre-filter only —
> the BOCPD layer (§14.3) provides the statistically grounded changepoint posterior.

### 16.2 BOCPD — Normal-Inverse-Gamma conjugate prior

Bayesian Online Changepoint Detection (Adams & MacKay 2007, arXiv 0710.3742) maintains a posterior over "run length" — the number of observations since the last changepoint. The conjugate prior for a Gaussian-distributed stream with unknown mean and variance is the Normal-Inverse-Gamma (NIG):

```
Parameters:  θ = (μ₀, κ₀, α₀, β₀)
Interpretation:
  μ₀    — prior predictive mean
  κ₀    — pseudo-observations weighting the mean prior
  α₀    — shape of the inverse-gamma prior on variance
  β₀    — rate of the inverse-gamma prior on variance
```

**NIG posterior update** — given a new observation `x`, the sufficient statistics update in O(1):

```
κₙ = κ₀ + 1
μₙ = (κ₀ × μ₀ + x) / κₙ
αₙ = α₀ + 0.5
βₙ = β₀ + (κ₀ × (x − μ₀)²) / (2 × κₙ)
```

**Student-t predictive distribution** — the marginal predictive for the next observation under NIG parameters `(μ, κ, α, β)` is a Student-t:

```
ν    = 2α                         [degrees of freedom]
loc  = μ                          [location]
scale² = β × (κ + 1) / (κ × α)   [scale squared]

log p(x | θ) = log Γ((ν+1)/2) − log Γ(ν/2)
             − 0.5 × log(π × ν × scale²)
             − ((ν+1)/2) × log(1 + (x − loc)² / (ν × scale²))
```

`lgamma` is computed via Stirling's series (x ≥ 8) with recursive reduction for x < 8 and the reflection formula for x < 0.5 — no external crate required.

### 16.3 BOCPD run-length posterior

The system tracks at most `MAX_RUN_LENGTH = 500` run-length hypotheses. Each hypothesis `r` represents "the current run started `r` steps ago." The hazard rate `h = drift_bocpd_hazard_rate` (default 0.01) is the per-step probability of a changepoint.

**State at time t:**

```
run_states[r] = { log_weight: f64, nig: NigParams }   for r = 0..t
```

**Update step** for each new observation `x`:

```
For each existing run hypothesis r:
    log_likelihood_r = student_t_log_pdf(x | run_states[r].nig)
    run_states[r].nig = run_states[r].nig.update_one(x)        [NIG update]
    log_weight[r+1]  += log_likelihood_r + log(1 − h)          [survive: run grows]
    log_weight_new   += log_likelihood_r + log(h)              [changepoint: new run r=0]

New run hypothesis (r=0): fresh NIG prior, log_weight = log_weight_new
Normalise all weights: log_weights -= logsumexp(log_weights)
```

**Changepoint detection:** after the guard (`run_states.len() > 5` to prevent startup false positives), compute the posterior mass on short run lengths:

```
P(run_length ≤ 4) = Σ_{r=0}^{4} exp(log_weight[r])
```

When `P(run_length ≤ 4) > drift_bocpd_changepoint_threshold` (default 0.90), a changepoint is detected. Emits `DriftEvent::Changepoint(CalibrationChangepoint)`.

> **Parameter rationale:**
>
> **`hazard_rate h = 0.01`:** Corresponds to an expected run length of 100 observations between
> changepoints (`E[run_length] = 1/h`). LLM API providers historically update models every 4–8
> weeks; at a typical production rate of 10–50 calibration-relevant tasks per day, 100 tasks
> spans approximately 2–10 days — roughly the shortest inter-update interval. A lower h (longer
> expected run) would delay detection past the next API update; a higher h would fragment
> stable periods into spurious changepoints.
>
> **`changepoint_threshold = 0.90`:** Requires 90% posterior mass on short run lengths (≤4 steps)
> to fire. The threshold is intentionally tight to guard against false positives during the
> startup guard period (`run_states.len() > 5`). In early testing without the guard, the NIG
> prior had not seen sufficient data in the first 5 observations, causing artificially concentrated
> run-length posteriors and false changepoint detection. The 5-observation guard plus the 0.90
> threshold together eliminate startup false positives while preserving detection sensitivity
> for genuine API drift.
>
> **`P(run_length ≤ 4)` as the changepoint signal:** A genuine changepoint concentrates posterior
> mass at run length 0–4 (the new regime just started). Gradual drift without a structural break
> distributes mass more broadly. The choice of ≤4 rather than ≤1 or ≤2 provides robustness
> to one or two delayed observations where the new distribution overlaps the old.

### 16.4 ORCA conformal margin

Between changepoint detection and recalibration, ORCA (arXiv 2604.01170) ensures coverage does not collapse. `DriftMonitor::active_conformal_margin()` returns `drift_conformal_margin` (default 0.05) when:

1. A changepoint was detected (`changepoint_active = true`), AND
2. The elapsed time since detection < `drift_staleness_ttl_secs` (default 3600s)

Otherwise returns 0.0. The margin is applied in `engine.rs::run_offline`:

```
threshold_adjusted = max(0.0, base_threshold − conformal_margin)
```

Widening the gate (lowering the pass threshold) means more proposals survive verification during drift — conservative but coverage-preserving. The margin is removed once TTL expires (without recalibration) or `reset_after_recalibration()` is called.

> **Rationale for 0.05 margin and 3600s TTL:**
> The 5% margin is the standard 95% coverage target from conformal prediction (Angelopoulos &
> Bates 2023; ORCA arXiv 2604.01170). Subtracting 0.05 from `verification_config.threshold`
> widens the acceptance gate by 5 percentage points during drift, ensuring that proposals
> near-but-below the normal threshold are not discarded when the verifier's baseline has shifted.
> A larger margin (e.g., 0.10) risks accepting low-quality proposals in normal operation if the
> TTL is long; a smaller margin (0.02) provides insufficient coverage during a genuine drift
> event. The 3600s (1-hour) TTL spans the typical operator response time to a
> `CalibrationChangepoint` event and recalibration. After one hour without recalibration, the
> margin expires and normal thresholds resume — erring on the side of precision over coverage.
> `auto_recalibrate_on_drift = false` (default) keeps the decision to recalibrate explicit:
> automated recalibration on every detected changepoint risks over-adapting to transient noise.

### 16.5 Consensus agreement rate

The signal fed to `DriftMonitor::observe()` is:

```
consensus_agreement_rate = |{e ∈ verification_events : e.passed == true}| / |verification_events|
```

Returns 1.0 for an empty event set. Source: `consensus_agreement_rate_from_events` in `engine.rs`. Stable degradation signal: when LLM quality shifts, the fraction of tasks where all verifiers agree on pass degrades before absolute pass rate does, giving early warning.

| Symbol | Meaning | Default |
|--------|---------|---------|
| `drift_ddm_window` | DDM sliding window size | 20 |
| `drift_ddm_k` | DDM sigma threshold | 2.5 |
| `drift_bocpd_hazard_rate` (h) | Per-step changepoint probability | 0.01 |
| `drift_bocpd_changepoint_threshold` | Posterior mass threshold for firing | 0.90 |
| `drift_conformal_margin` | ORCA threshold widening on changepoint | 0.05 |
| `drift_staleness_ttl_secs` | Margin TTL after changepoint | 3600 |
| `auto_recalibrate_on_drift` | Trigger POST /calibrate on changepoint | false |
