# Mathematical Apparatus

Authoritative reference for every formula, constant, and statistical model used by the
H2AI Control Plane.  All values are sourced directly from `crates/h2ai-types/src/sizing.rs`,
`crates/h2ai-config/reference.toml`, and `crates/h2ai-api/src/oracle/mod.rs`.

---

## 1. Universal Scalability Law (USL)

The USL models throughput of a parallel system as a function of concurrency N:

```
X(N) = N / (1 + α(N−1) + β·N(N−1))
```

- **α** (contention): fraction of work that must serialise regardless of parallelism.
  Default `alpha_contention = 0.12` (AI-agent workloads).
- **β** (coherency cost): per-agent-pair coordination overhead.
  Default `beta_base_default = 0.039` (AI-agents tier; human teams ≈ 0.0225; CPU cores ≈ 0.0003).

### Maximum ensemble size

The USL-derived ceiling on useful ensemble size:

```
N_max = round(√((1 − α) / β_eff))
```

Implemented in `CoherencyCoefficients::n_max()`.
The result is floored at 3 (BFT quorum minimum) by the engine before use.

### Effective coherency cost (β_eff)

```
β_eff = max(β₀ × (1 − CG_mean), 1e−6)
```

If a directly measured `beta_quality` value is present in `CoherencyCoefficients`, it
takes precedence over the proxy formula.  Implemented in `CoherencyCoefficients::beta_eff()`.

The bounded form `β₀ × (1 − CG)` is deliberate: unlike `β₀ / CG` it never diverges as
CG approaches zero.

### Temporal decay of β_eff

Calibration coefficients age via exponential decay:

```
β_eff(t) = β_eff(0) × 2^(−Δt / CG_HALFLIFE_SECS)    CG_HALFLIFE_SECS = 604 800 (7 days)
```

Implemented in `CoherencyCoefficients::beta_eff_temporal()`.

### Context-aware N_max

When per-slot context fill is known, β is inflated to model context-pressure slowdown:

```
β_ctx(N) = β_eff × (1 + γ × fill(N))
```

The equation `N_max = round(√((1 − α) / β_ctx(N)))` is solved iteratively (≤ 5 iterations,
convergence tolerance 0.5).  `γ` is `context_pressure_gamma` (default 0.5).
Implemented in `CoherencyCoefficients::n_max_context_aware()`.

Constants: `USL_SOLVER_MAX_ITERS = 5`, `USL_SOLVER_CONVERGENCE_TOL = 0.5`.

### Quorum degradation

`CoherencyCoefficients::n_max_degraded()` returns `true` when the unclamped N_max is
below 3.0.  In non-shadow mode the engine fails with
`MultiplicationConditionFailure::QuorumDegradedBelowMinimum` rather than running a
sub-quorum pool that would disable BFT safety mechanisms.

---

## 2. Multiplication Condition

Before accepting a task, the engine checks five preconditions in `MultiplicationCondition::check()`.
Failure of any one aborts immediately:

| Variant | Trigger |
|---------|---------|
| `InsufficientCompetence` | `p_mean < min_baseline_competence` (default 0.30) |
| `InsufficientDecorrelation` | `ρ_mean > max_error_correlation` (default 0.90) |
| `CommonGroundBelowFloor` | `CG_mean < cg_collapse_threshold` (default 0.10) |
| `InsufficientPoolDiversity` | task-level N_eff below `n_eff_complex_threshold` |
| `QuorumDegradedBelowMinimum` | USL-derived N_max < 3 |

---

## 3. Common Ground (CG)

Common Ground is the mean pairwise agreement across explorer outputs, measured as 1 minus the
normalised Hamming distance between per-constraint satisfaction profiles.
`CG ∈ [0, 1]`; 1 = perfect agreement, 0 = total disagreement.

### Proxy calibration formulas

When only CG is available, `EnsembleCalibration::from_cg_mean()` derives:

```
p_mean = 0.5 + CG / 2
ρ_mean = 1 − CG
```

### Empirical path

`EnsembleCalibration::from_empirical(p_mean, rho_empirical, max_n)` accepts directly
measured competence and correlation values, bypassing the proxy formulas entirely.

---

## 4. Condorcet Jury Theorem (CJT)

For an ensemble of N agents with individual competence p and pairwise error correlation ρ,
the ensemble majority-vote accuracy is:

```
Q(N, p, ρ) = p + (Q_ind(N, p) − p) × (1 − ρ)
```

where `Q_ind(N, p)` is the independent-agent majority vote probability.
Even-N ensembles subtract 0.5 times the tie probability from `Q_ind`.
Implemented in `condorcet_quality()`.

### Optimal iteration count

The minimum number of independent generation passes to reduce residual uncertainty below 50%:

```
n_it_optimal(ρ) = ceil(1 + log(0.5) / log(1 − ρ))
```

Clamped to `[1, 9]`.  Implemented in `n_it_optimal()`.
`RHO_UPPER_CLAMP = 0.99` prevents division by zero.

---

## 5. EigenCalibration

Eigendecomposition of the pairwise agreement matrix yields the effective number of
independent information sources in the pool (participation ratio):

```
N_eff = (Σλᵢ)² / Σλᵢ²
```

Two construction paths:
- `EigenCalibration::from_cg_matrix`: pairwise Hamming-based agreement matrix.
- `EigenCalibration::from_cosine_matrix`: embedding cosine similarity matrix.

`eigen_n_eff_delta = 0.05` is the minimum N_eff increment for an adapter to be included
in the pruned set.

---

## 6. Task Complexity and Quadrant Routing

Phase 1.5 assigns each task to a routing quadrant based on the Task Complexity Coefficient
(TCC) and pool N_eff:

| Quadrant | Description | Topology route |
|----------|-------------|----------------|
| `Precision` | Low TCC, high N_eff | Self-MoA |
| `Coverage` | High TCC, high N_eff | Cross-family committee |
| `Complex` | High TCC, high N_eff (heavy) | Forced CoT + synthesis |
| `Degenerate` | Low N_eff (both below threshold) | `MultiplicationConditionFailed` (non-shadow) |

`UnambiguousPrecision` and `UnambiguousCoverage` are internal sub-variants for
high-confidence routing decisions.

---

## 7. Thompson Sampling Bandit

The bandit selects ensemble size N across three learning phases keyed on total task count k:

| Phase | k range | Strategy |
|-------|---------|----------|
| 0 | k < 10 | Pure exploration: N = N_max_USL |
| 1 | 10 ≤ k < 30 | ε-greedy, `bandit_epsilon = 0.3` |
| 2 | k ≥ 30 | Pure Thompson Sampling |

Arms span `N ∈ [1, bandit_n_max_arms]` (default 6).
Prior: Gaussian centred on N_max_USL, `σ = bandit_prior_sigma = 2.0`,
strength `bandit_prior_strength = 5.0` pseudo-observations.
On adapter version-hash change, the posterior is soft-reset:
`posterior = (1 − 0.3) × existing + 0.3 × prior` (`bandit_soft_reset_decay = 0.3`).

---

## 8. τ-Alignment

Explorer temperature diversity is quantified with an exponential alignment kernel:

```
τ_alignment(a, b) = exp(TAU_ALIGNMENT_DECAY_COEFF × |a − b|) = exp(−3 × |a − b|)
```

`TAU_ALIGNMENT_DECAY_COEFF = −3.0`.  Used during topology provisioning to weight
temperature spread within an ensemble.

### Talagrand KL τ-spread adaptation

The Talagrand diagnostic adapts τ-spread per wave to prevent over-confidence:

```
τ_spread_next = τ_spread × (1 + η × (U_score − Λ_score))
```

`talagrand_eta = 0.1` (learning rate η), `talagrand_tau_min = 0.5` (contraction floor),
`tau_spread_max_factor = 2.0` (expansion ceiling).

---

## 9. Empirical ρ Estimation (RhoEmaState)

After each task wave, pairwise Pearson score products are tracked per adapter pair with
an exponential moving average:

```
EMA_t = (1 − α) × EMA_{t−1} + α × score_product_{a,b}
```

`α = 0.10`, effective window ≈ 10 tasks.  Conservative initial prior: `0.45`.
Once `n_observations ≥ 30` (CLT threshold), `rho_mean()` replaces the `1 − CG_mean`
proxy.  Output clamped to `[0, 0.99]`.

---

## 10. Optimal Synthesis Policy (OSP)

`OspConfig` governs zone-based synthesis routing (`crates/h2ai-types/src/sizing.rs`):

| Field | Default | Meaning |
|-------|---------|---------|
| `t_v` | 0.125 | Verification threshold for Zone-1 pass-through |
| `concordance_alpha` | 0.1 | Kendall-τ confidence level for score ordering |
| `max_n_v_for_zone3` | 4 | Max verified proposals before Zone-3 synthesis is triggered |
| `accumulation_decay` | 0.7 | Score decay for round-2 candidate accumulation |

When `osp` is absent from config, the merger falls back to legacy strategy dispatch.
Available `MergeStrategy` variants: `ScoreOrdered`, `ConsensusMedian`,
`OutlierResistant{f}`, `MultiOutlierResistant{f, m}`.

---

## 11. Oracle Statistics

The oracle calibration service maintains a rolling FIFO window capped at
`oracle_window_size = 200` observations.

### Expected Calibration Error (ECE)

```
ECE = (1/n) × Σᵢ |q_confidence_i − y_oracle_i|
```

`q_confidence_i` is the engine's predicted success probability for task i;
`y_oracle_i ∈ {0, 1}` is the binary oracle verdict.

### Calibration basis thresholds

| Window size n | ECE | Basis |
|---|-----|-------|
| n < 10 | any | Heuristic (0) |
| 10 ≤ n < 30 | any | Bootstrap (1) |
| n ≥ 30 | ECE < 0.15 | Conformal (2) |
| n ≥ 30 | ECE ≥ 0.15 | Heuristic (0) — quality regression |

Target ECE < 0.05.  Alert threshold: `oracle_ece_alert_threshold = 0.15`.
Pass-rate floor: `oracle_pass_rate_floor = 0.30`.

### Residual P90

Using the Angelopoulos–Bates Theorem 1 finite-sample quantile index:

```
idx = ceil((n + 1) × 0.9) − 1
```

### Ensemble p_mean patching

Once `n_observations ≥ 10`, `EnsembleCalibration::from_measured_p()` updates the
ensemble baseline competence from the empirical oracle pass rate.

---

## 12. Wilson Score Confidence Intervals

`VerificationScoredEvent.score_lower` and `.score_upper` are Wilson score confidence
intervals on the binary constraint pass rate observed across all checks in a verification
pass.

---

## 13. Constants Summary

| Constant | Value | Source |
|----------|-------|--------|
| `N_MAX_ENSEMBLE_CAP` | 9 | `sizing.rs` |
| `RHO_UPPER_CLAMP` | 0.99 | `sizing.rs` |
| `CG_HALFLIFE_SECS` | 604 800 | `sizing.rs` |
| `TAU_ALIGNMENT_DECAY_COEFF` | −3.0 | `sizing.rs` |
| `USL_SOLVER_MAX_ITERS` | 5 | `sizing.rs` |
| `USL_SOLVER_CONVERGENCE_TOL` | 0.5 | `sizing.rs` |
| `QUORUM_FLOOR` | 3 | `phases/complexity.rs` |
| `CACHE_SIMILARITY_THRESHOLD` | 0.85 | `verification.rs` |
| `alpha_contention` (default) | 0.12 | `reference.toml` |
| `beta_base_default` (default) | 0.039 | `reference.toml` |
| `context_pressure_gamma` (default) | 0.5 | `reference.toml` |
| `CG_HALFLIFE_SECS` (days) | 7 | `sizing.rs` |
| `bandit_phase0_k` | 10 | `reference.toml` |
| `bandit_phase1_k` | 30 | `reference.toml` |
| `oracle_window_size` | 200 | `reference.toml` |
| `oracle_ece_alert_threshold` | 0.15 | `reference.toml` |
| `oracle_pass_rate_floor` | 0.30 | `reference.toml` |
