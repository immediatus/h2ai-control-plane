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

Setting `dX/dN = 0` gives the ensemble ceiling:

```
N_max = round(√((1 − α) / β_eff))
```

A one-σ confidence interval `(n_max_lo, n_max_hi)` is propagated from the empirical CG variance: `n_at_cg(CG_mean ± cg_std_dev)`.

### 2.1 Context-aware N_max

Coordination cost has two physical components: conflict reconciliation (the merge step, reduced by CG) and positional attention degradation in the synthesis context window ("Lost in the Middle", Liu et al. 2023). The latter is orthogonal to CG and is modelled by amplifying β with the context-fill fraction:

```
fill(N)       = min(1, N × proposal_tokens / max_tokens)
β_ctx(N)      = β_eff × (1 + γ × fill(N))
N_max_ctx     = solve N = √((1 − α) / β_ctx(N))   (iterative; ≤ 5 iterations)
```

`γ` is the attention-sensitivity coefficient.

### 2.2 Temporal decay

CG samples carry Unix timestamps. `beta_eff_temporal(now)` weights each sample by `exp(−(now − t) / CG_HALFLIFE_SECS)` with `CG_HALFLIFE_SECS = 604_800` (7 days, Ebbinghaus-style). As samples age, β_eff drifts toward the conservative ceiling β₀ — older calibration data deflates without explicit recalibration.

### 2.3 Calibration

The harness runs two phases:
- **Phase A** with 2 adapters → measures `z_2` (latency at N=2).
- **Phase B** with M adapters → measures `z_M`.

Analytical USL fit (M ≥ 3):

```
β₀ = (z_M − z_2 × (M − 1)) / ((M − 1)(M − 2))
α  = z_2 − 2β₀
```

When M < 3 the fit falls back to `cfg.calibration_default_alpha` and `cfg.calibration_default_beta`. Online β₀ is then tracked via `beta_from_token_spans` — an EMA over per-merge timing pulled from the live token stream.

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

Source: `crates/h2ai-types/src/sizing.rs::MultiplicationConditionFailure`. Four failure modes:

1. **InsufficientCompetence** — `p_mean ≤ min_competence`. Adding more adapters makes the committee worse.
2. **InsufficientDecorrelation** — `rho_mean ≥ max_correlation`. Errors are correlated; CJT gain collapses.
3. **CommonGroundBelowFloor** — `cg_mean < θ_coord`. Adapters too epistemically distant; coordination cost exceeds diversity benefit.
4. **InsufficientPoolDiversity** — `n_eff_cosine_prior < 1.0 + diversity_threshold`. Pool is semantically near-degenerate.

The first three are checked at Phase 2.5 by `MultiplicationChecker::check`. The fourth is checked at Phase 2.6 by the engine directly when `cfg.diversity_threshold > 0`.

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

`EnsembleCalibration::from_measured_p` accepts a directly measured baseline accuracy from `scripts/baseline_eval.py` and switches `prediction_basis` from `Heuristic` to `Empirical`.

`n_optimal` is the N that maximises `(Q(N, p, ρ) − p) / N` — the marginal Condorcet gain per adapter — capped at `max_n` (default 9 in production config).

### 5.1 Information-theoretic ceiling

Source: `n_it_optimal(rho)`. Returns the smallest N where `(1 − ρ)^(N−1) < 0.5`, i.e. where the marginal information gain drops below half the per-adapter entropy. Matches the Condorcet `n_optimal` within ±1 for ρ ∈ [0.3, 0.95].

### 5.2 Honest limitation

The CJT is a theorem about **independent voters**. The system uses `(1 − ρ)` as a correction term, but it does *not* directly measure ρ — it proxies it from `1 − CG_mean` (Hamming) or `1 − N_eff / N` (cosine). When two adapters from the same family hallucinate the same answer, both proxies underestimate the true correlation. This is mitigated by Phase 2.6 (cosine N_eff guard), `single_family_warning` on `CalibrationCompletedEvent`, and `explorer_verification_family_match` flagging — but not eliminated. Empirical baseline accuracy from `scripts/baseline_eval.py` upgrades `PredictionBasis::Heuristic → Empirical` for the p estimate; the ρ estimate remains a proxy.

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

## 8. Merge Strategy Selection

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
- `OutlierResistant{f}` — Krum (Blanchard et al. 2017): pick the proposal with smallest sum of distances to its `n − f − 2` nearest neighbours in Jaccard-distance space. Quorum requirement: `n ≥ 2f + 3`.
- `MultiOutlierResistant{f, m}` — apply OutlierResistant iteratively to keep m survivors, then take the highest verification score.

---

## 9. Attribution

Source: `crates/h2ai-orchestrator/src/attribution.rs::HarnessAttribution::compute`.

Per-task quality decomposition:

```
Q_total = base_quality
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

## 10. Honest Limitations

The math used in this system is calibrated to specific assumptions. They are listed here so they are not forgotten:

- **CJT independence.** The theorem assumes independent voters. The runtime corrects with `(1 − ρ)`, but ρ is proxied — not directly measured. Cross-family pools, single-family warnings, and the cosine N_eff guard mitigate this; they do not eliminate it.
- **CG as a proxy chain.** The flow is `CG → β_eff → N_max` and `CG → (p, ρ) → Q`. Each arrow is a heuristic. Empirical validation upgrades `p` to measured; ρ remains a proxy.
- **Correlated hallucination.** When two adapters share a training corpus and produce the same wrong answer, both Hamming CG and cosine N_eff can simultaneously read "high diversity" if the binary profiles disagree on different constraints. Phase 2.6 reduces but does not solve this.
- **Synthesis gain is local.** `synthesis_gain` is measured against the same verification adapter that scored the individual proposals. A verifier blind spot inflates both terms equally and cancels out.
- **No oracle.** Without a `q_measured` from an external oracle, `q_predicted` is the only quality signal. The bootstrap interval reflects CG variance, not ground-truth uncertainty.
