# H2AI Math Reference

This document defines every formula and constant the H2AI Control Plane runtime uses to size ensembles, measure constraint agreement, fit calibration parameters, select output, and calibrate oracle confidence. All values are sourced from `crates/h2ai-types/src/sizing.rs`, `crates/h2ai-config/reference.toml`, and supporting modules noted per section. No formula appears here that is not directly implemented in source.

---

## 1. USL Scalability Model

Source: `crates/h2ai-types/src/sizing.rs` — `CoherencyCoefficients`.

The Universal Scalability Law (Gunther 1993) models throughput as a function of ensemble size N:

```
X(N) = N / (1 + α(N − 1) + β · N(N − 1))
```

- `α` — contention (serial-fraction) coefficient.
- `β` — coherency-drag coefficient.

### 1.1 N_max — ensemble cost ceiling

Setting `dX/dN = 0` and solving gives the maximum throughput point:

```
N_max = round( √( (1 − α) / β_eff ) )
```

Rust implementation:

```rust
pub fn n_max(&self) -> f64 {
    let beta_eff = self.beta_eff().max(f64::EPSILON);
    ((1.0 - self.alpha).max(0.0) / beta_eff).sqrt().round()
}
```
<!-- sizing.rs -->

### 1.2 β_eff — effective coherency coefficient

When `beta_quality` is explicitly set it overrides the proxy. Otherwise the proxy is derived from `CG_mean`:

```
β_eff = β₀ × (1 − CG_mean)    [proxy; floor 1e-6]
```

Rust implementation:

```rust
pub fn beta_eff(&self) -> f64 {
    self.beta_quality.map_or_else(
        || {
            let cg = self.cg_mean().clamp(0.0, 1.0);
            (self.beta_base * (1.0 - cg)).max(1e-6)
        },
        |bq| bq.max(1e-6),
    )
}
```
<!-- sizing.rs -->

### 1.3 β_eff with Ebbinghaus temporal decay

CG samples carry Unix timestamps. Each sample is weighted by an exponential decay before computing `CG_mean`:

```
weight(t) = exp( −(now − t) / CG_HALFLIFE_SECS )
```

`CG_HALFLIFE_SECS = 604_800` (7 days). When timestamps are missing the method falls back to `beta_eff()`.
<!-- sizing.rs -->

### 1.4 Context-aware N_max

Positional attention degradation in long synthesis contexts is modelled by amplifying `β_eff` with the context-fill fraction `f(N)`:

```
f(N)     = min(1, N × proposal_tokens / max_tokens)
β_ctx(N) = β_eff × (1 + γ × f(N))
N_max_ctx = round( √( (1 − α) / β_ctx(N) ) )   [iterate until convergence]
```

Rust:

```rust
pub fn n_max_context_aware(&self, proposal_tokens: f64, max_tokens: f64, gamma: f64) -> f64 {
    let fill = (n * proposal_tokens / max_tokens).min(1.0);
    let beta_ctx = beta_eff * gamma.mul_add(fill, 1.0);
    let n_new = ((1.0 - alpha).max(0.0) / beta_ctx.max(f64::EPSILON)).sqrt().round();
    // converges in ≤ USL_SOLVER_MAX_ITERS = 5 steps, tol = USL_SOLVER_CONVERGENCE_TOL = 0.5
}
```
<!-- sizing.rs -->

Convergence constants: `USL_SOLVER_MAX_ITERS = 5`, `USL_SOLVER_CONVERGENCE_TOL = 0.5`.

---

## 2. CG — Constraint-Profile Agreement

### 2.1 Hamming path (default)

`CgMode::ConstraintProfile`. For each pair of adapters (i, j), CG(i, j) is derived from the Hamming distance between their binary constraint-satisfaction vectors over the calibration corpus. `CG_mean` is the mean over all pairs.

Falls back to `cfg.calibration_cg_fallback` (default 0.7) when no corpus is available.

### 2.2 Embedding cosine path

`CgMode::EmbeddingCosine`. For each pair (i, j):

```
CG(i, j) = 1.0  if semantic_jaccard(o_i, o_j) ≥ cg_agreement_threshold
            0.0  otherwise
```

Then `CG_mean × tau_alignment(tau_i, tau_j)`.

`cg_agreement_threshold` default: 0.85.
<!-- sizing.rs -->

### 2.3 τ-alignment

Each CG observation is attenuated by the alignment between the τ values of the two adapters:

```
tau_alignment(a, b) = exp( TAU_ALIGNMENT_DECAY_COEFF × |a − b| )
```

`TAU_ALIGNMENT_DECAY_COEFF = −3.0`. Same τ → 1.0; |diff| = 1.0 → exp(−3) ≈ 0.05.
<!-- sizing.rs -->

---

## 3. Ensemble Sizing

### 3.1 N_max confidence interval

`n_max_ci()` propagates a ±1σ band from Bessel-corrected `cg_std_dev` through the N_max formula:

```
lo = min( n_at_cg(CG_mean + σ), n_at_cg(CG_mean − σ) ).max(3.0)
hi = max( n_at_cg(CG_mean + σ), n_at_cg(CG_mean − σ) ).max(lo)
```

Hard BFT/Krum floor on both bounds: **3.0**. The unclamped value is preserved for telemetry via `n_max_degraded()`.
<!-- sizing.rs -->

### 3.2 Condorcet quality with error correlation

```
Q_ind(N, p) = Σ_{k > N/2} C(N,k) · p^k · (1−p)^(N−k)
              + (N even) 0.5 × C(N, N/2) × p^(N/2) × (1−p)^(N/2)

Q(N, p, ρ) = p + (Q_ind(N, p) − p) × (1 − ρ)
```

Clamped to [0, 1]. Boundary cases: `N = 1 → Q = p`; `ρ = 1 → Q = p`.

Rust:

```rust
pub fn condorcet_quality(n_agents: usize, p: f64, rho: f64) -> f64 {
    // Q(N,p,ρ) = p + (Q_independent(N,p) − p) × (1 − ρ)
    (q_ind - p).mul_add(1.0 - rho, p).clamp(0.0, 1.0)
}
```
<!-- sizing.rs -->

### 3.3 N_optimal (Condorcet, marginal gain)

`n_optimal` is the N that maximises the marginal Condorcet gain per adapter:

```
N_optimal = argmax_{N=1..max_n}  (Q(N, p, ρ) − p) / N
```

N = 1 scores 0. When ρ = 1 all N score 0 and N = 1 is returned. `max_n` is bounded by `calibration_max_ensemble_size` (default 9; `N_MAX_ENSEMBLE_CAP = 9`).
<!-- sizing.rs -->

### 3.4 N_it — information-theoretic optimal ensemble size

```
N_it = ⌈ 1 + log(0.5) / log(1 − ρ) ⌉
```

Rust:

```rust
pub fn n_it_optimal(rho: f64) -> usize {
    let n = 1.0 + 0.5_f64.log(1.0 - rho);
    (n.ceil() as usize).clamp(1, N_MAX_ENSEMBLE_CAP)  // N_MAX_ENSEMBLE_CAP = 9
}
```
<!-- sizing.rs -->

---

## 4. Proxy Chain

### 4.1 CG_mean → p_mean and ρ_mean (cold-start priors)

`EnsembleCalibration::from_cg_mean()` derives the Condorcet inputs from `CG_mean`:

```rust
let rho_mean = (1.0 - cg).clamp(0.0, 1.0);
let p_mean   = (0.5 + cg / 2.0).clamp(0.5, 1.0);
```
<!-- sizing.rs -->

These are cold-start priors. `p_mean` is promoted to oracle pass rate once `n_observations ≥ 10`; `rho_mean` is replaced by `RhoEMA` once `n_observations ≥ 30`.

### 4.2 n_eff_cosine_prior fallback (no embedding model)

When no embedding model is configured, `n_eff_cosine_prior` is estimated as:

```
n_eff_cosine_prior = 1 + (N − 1) × calibration_cg_fallback
```

Capped at N. Default `calibration_cg_fallback = 0.7`.
<!-- sizing.rs -->

### 4.3 RhoEMA — online ρ replacement

Source: `crates/h2ai-api/src/rho_ema.rs`.

**Cold-start prior:** `0.45`, inserted on first pair observation via `or_insert(0.45)`.

**Update rule:**

```
score_product = (score_a − p_mean) × (score_b − p_mean) / variance    [clamped to [−1, 1]]
ema_next      = (1 − α) × ema_prev + α × score_product
```

**Steady-state threshold:** 30 observations (CLT threshold).

**Upper clamp:** `RHO_UPPER_CLAMP = 0.99`.
<!-- rho_ema.rs -->

### 4.4 CoordinationThreshold

Source: `crates/h2ai-types/src/sizing.rs`.

```rust
pub fn from_calibration(cc: &CoherencyCoefficients, max: f64) -> Self {
    let spread = cc.cg_mean() - cc.cg_std_dev();
    Self(spread.clamp(0.0, max))
}
```
<!-- sizing.rs -->

`coordination_threshold_max` default: 0.30.

---

## 5. Calibration Fitting

### 5.1 USL two-point fit (α and β₀ from timing)

Phase A runs with 2 adapters; Phase B with M adapters. Normalized latency ratios:

```
z_2 = 2 × t2_parallel / t1 − 1
z_M = M × t_M_parallel / t1 − 1
```

Fit (requires M ≥ 3):

```
β₀ = (z_M − z_2 × (M − 1)) / ((M − 1)(M − 2))
α  = z_2 − 2β₀
```

Rust:

```rust
pub fn usl_fit(t1, t2_parallel, m, t_m_parallel, ...) -> (f64, f64) {
    let z2    = 2.0 * t2_parallel / t1 - 1.0;
    let z_m   = m_f * t_m_parallel / t1 - 1.0;
    let beta_denom = (m_f - 1.0) * (m_f - 2.0);
    let beta0 = (z_m - z2 * (m_f - 1.0)) / beta_denom;
    let alpha = 2.0f64.mul_add(-beta0, z2);  // α = z₂ − 2β₀
    // Clamps: α ∈ [0.05, 0.5], β₀ ∈ [1e-6, 0.1]
}
```
<!-- sizing.rs -->

Falls back to `cfg.alpha_contention` and `cfg.beta_base_default` when M < 3, degenerate timings, or negative fitted parameters.

### 5.2 Epistemic β₀ override (embedding model + M ≥ 3)

When an embedding model is available and `N_cal ≥ 3`, β₀ is computed from the cosine N_eff eigenvalue:

```
N_eff_adj = clamp( N_eff × CG_mean^k, 1.0, N_cal )
β₀        = max( (1/N_eff_adj − 1/N_cal) / (N_cal − 1), 1e-6 )
```

`k = calibration_probe.neff_cg_exponent` (default 2.0).
<!-- sizing.rs -->

### 5.3 β₀ from merge spans (pairwise_beta)

```
pairs_i = max(1, n_i × (n_i − 1) / 2)
β₀      = mean( elapsed_i / pairs_i ) / T1_secs
```

Clamped to [1e-9, 0.1].
<!-- sizing.rs -->

### 5.4 Calibration basis selection

Source: `crates/h2ai-types/src/sizing.rs` and oracle accumulator.

| Condition | Basis |
|---|---|
| n < 10 | Heuristic (0) |
| 10 ≤ n < 30 | Bootstrap (1) |
| n ≥ 30 AND ECE < 0.15 | Conformal (2) |
| n ≥ 30 AND ECE ≥ 0.15 | Heuristic — quality regression (0) |

### 5.5 AIMD slow start

Per-task α adaptation:

```
[success]  α_next = max( α_current × decay_rate, alpha_measured )
[failure, yield < reset_threshold]  α_next = min( α_current × reset_multiplier, seed_alpha )
```

Defaults: `seed_alpha = 0.15`, `decay_rate = 0.95`, `reset_multiplier = 3.0`, `reset_threshold = 0.4`.
<!-- sizing.rs / calibration.rs -->

---

## 6. EigenCalibration

Source: `crates/h2ai-types/src/sizing.rs::EigenCalibration`.

Given the eigenvalue spectrum `{λ_i}` of the CG or cosine kernel matrix (negative eigenvalues clamped to 0):

```
N_eff       = ( Σ λ_i )² / Σ λ_i²                         [participation ratio]
h_diversity = −Σ (p_i × ln p_i) / ln(N)     where p_i = λ_i / Σ λ  [normalised Shannon entropy]
ρ_eff(n)    = (1 − N_eff / n).clamp(0.0, 1.0)              [derived effective correlation]
```
<!-- sizing.rs -->

---

## 7. Task Complexity Coefficient (TCC)

Source: `crates/h2ai-types/src/sizing.rs`.

**Structural TCC:**

```
TCC_structural = 1
               + k_soft  × soft_fraction
               + k_type  × type_diversity
               + k_cross × soft_fraction × type_diversity
```

**Effective TCC (heavy-dominant context):**

```
TCC_effective = TCC_structural × (1 + k_heavy × heavy_fraction)
```

Applied only when the heavy constraint fraction exceeds the threshold that makes this path dominant.
<!-- sizing.rs -->

---

## 8. OSP Selection Math

Source: `crates/h2ai-types/src/sizing.rs`.

### 8.1 Concordance threshold τ(N_f)

Given `N_f` finalised proposals and significance level `α = concordance_alpha`:

```
τ(N_f) = clamp( 0.5 + √( −ln(α) / (2 · N_f) ), 0.5, 1.0 )
```
<!-- sizing.rs -->

### 8.2 ClearLeader condition

```
Δ = scores[0] − scores[1] ≥ 2 · t_v
```

Default `t_v = 0.125`.
<!-- sizing.rs -->

---

## 9. Oracle Calibration Math

Source: oracle accumulator and `crates/h2ai-types/src/sizing.rs`.

### 9.1 ECE (Expected Calibration Error)

```
ECE = (1/n) × Σ |q_confidence_i − y_oracle_i|
```

Mean of pre-computed residuals. Used to select the calibration basis (§5.4).

### 9.2 Oracle P90 (Angelopoulos–Bates)

```
idx = ⌈(n + 1) × 0.9⌉ − 1    [clamped to valid range]
P90 = sorted_residuals[idx]
```
<!-- sizing.rs -->

### 9.3 p_mean promotion

Once `n_observations ≥ 10`, `patch_ensemble_p_from_oracle` in `crates/h2ai-api/src/oracle/mod.rs` replaces the heuristic `p_mean = 0.5 + CG_mean / 2` with the empirical oracle pass rate. `prediction_basis` flips from `Heuristic` to `Empirical`.

---

## 10. CoordinationThreshold

Source: `crates/h2ai-types/src/sizing.rs`.

```rust
pub fn from_calibration(cc: &CoherencyCoefficients, max: f64) -> Self {
    let spread = cc.cg_mean() - cc.cg_std_dev();
    Self(spread.clamp(0.0, max))
}
```
<!-- sizing.rs -->

The derived floor is bounded above by `coordination_threshold_max` (default 0.30).

---

## 11. Key Constants

### 11.1 Hard-coded constants

| Constant | Value | Source |
|---|---|---|
| `CG_HALFLIFE_SECS` | 604 800 (7 days) | sizing.rs |
| `USL_SOLVER_MAX_ITERS` | 5 | sizing.rs |
| `USL_SOLVER_CONVERGENCE_TOL` | 0.5 | sizing.rs |
| `N_MAX_ENSEMBLE_CAP` | 9 | sizing.rs |
| `RHO_UPPER_CLAMP` | 0.99 | rho_ema.rs |
| `TAU_ALIGNMENT_DECAY_COEFF` | −3.0 | sizing.rs |
| β_eff floor | 1e-6 | sizing.rs |
| BFT/Krum floor (`n_max_ci`) | 3.0 | sizing.rs |
| n_max_degraded threshold | 3.0 | sizing.rs |
| α fit clamp | [0.05, 0.5] | sizing.rs |
| β₀ fit clamp | [1e-6, 0.1] | sizing.rs |
| RhoEMA cold-start prior | 0.45 | rho_ema.rs |
| RhoEMA steady-state threshold | 30 observations | rho_ema.rs |
| CalibrationBootstrap `prior_weight` | 5 | config |
| `min_krum_quorum` | 2f + 3 | Blanchard 2017 Theorem 2 |

### 11.2 Default parameter values (reference.toml)

| Field | Default |
|---|---|
| `alpha_contention` | 0.12 |
| `beta_base_default` | 0.039 |
| `context_pressure_gamma` | 0.5 |
| `calibration_cg_fallback` | 0.7 |
| `cg_agreement_threshold` | 0.85 |
| `cg_collapse_threshold` | 0.10 |
| `bft_threshold` | 0.85 |
| `coordination_threshold_max` | 0.30 |
| `krum_threshold` (SafetyConfig) | 0.30 |
| `krum_fault_tolerance` (SafetyConfig) | 0 (development) |
| `diversity_threshold` | 0.0 (development) |
| `tau_coordinator` | 0.05 |
| `tau_executor` | 0.40 |
| `tau_evaluator` | 0.10 |
| `tau_synthesizer` | 0.80 |
| `calibration_max_ensemble_size` | 9 |
| `verifier_consensus_passes` | 1 |
| `correlated_hallucination_cv_threshold` | 0.30 |
| `correlated_hallucination_min_jaccard_floor` | 0.50 |
| `calibration_probe.neff_cg_exponent` | 2.0 |
