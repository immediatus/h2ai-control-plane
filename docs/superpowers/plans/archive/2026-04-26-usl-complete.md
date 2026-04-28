# USL-Complete Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade H2AI Control Plane so every physics computation is governed by the Universal Scalability Law (USL) instead of the simpler Amdahl's law, with mathematically provable semantics throughout the stack.

**Architecture:** Two-phase USL calibration derives α and β₀ from timing measurements; `n_max()` uses the USL Proposition 1 formula `round(√((1−α)/β_eff))`; ConsensusMedian becomes a true semantic Fréchet median; ProposalSet implements a CRDT powerset join-semilattice with LUB semantics; Pareto topology selection is a formal weighted scalarization.

**Tech Stack:** Rust (tokio, serde, futures), Python (numpy, matplotlib), NATS JetStream

---

## Context and Root Problems

The project claims "governed by Universal Scalability Law" but the implementation uses Amdahl's law (`N_max = floor(1/α)`), dropping the β coherency term entirely. The `kappa_base` field exists in `CoherencyCoefficients` but is marked "informational telemetry only" and not used. This creates five concrete gaps:

| # | Problem | File | Impact |
|---|---------|------|--------|
| P1 | `n_max()` uses Amdahl, not USL | `physics.rs:78` | N_max overestimates ceiling; ignores coherency cost |
| P2 | β₀ derived from circular formula | `calibration.rs:129` | `kappa_base = kappa_eff_factor × (2 − CG_mean)` — uses CG_mean to derive a β that gets divided by CG_mean |
| P3 | Calibration is single-phase | `calibration.rs:33` | Only measures α; β₀ not calibrated at all |
| P4 | `select_topology` is ad-hoc if/else | `planner.rs:91` | Pareto weights barely used; no mathematical backing |
| P5 | ConsensusMedian is sync, token-only | `bft.rs:18` | Fréchet median property not documented; synonyms score as outliers |

Secondary issues:
- `ProposalSet::insert_scored` uses `or_insert` (first-wins), not LUB semantics for the powerset lattice
- `TopologyProvisionedEvent` (events.rs:40) has `kappa_eff: f64` field — must be renamed `beta_eff` with serde alias
- `planner.rs:31` calls `input.cc.kappa_eff()` — must change to `beta_eff()` after Task 1
- `planner_test.rs:61` asserts `Ensemble` for balanced weights — but weighted Pareto gives TeamSwarmHybrid (highest mean score 0.900 vs 0.860 vs 0.841); test must be updated
- `simulate_usl.py` already implements the correct math but uses variable names (`kappa`) that contradict the blog/docs nomenclature (`beta`)
- `math-apparatus.md` Section 3 documents Amdahl formula instead of USL Proposition 1 derivation

**Pareto score verification (weights T=0.34, E=0.33, D=0.33):**
- HierarchicalTree: 0.34×0.96 + 0.33×0.96 + 0.33×0.60 = **0.841**
- TeamSwarmHybrid: 0.34×0.84 + 0.33×0.91 + 0.33×0.95 = **0.900** ← wins with equal weights
- Ensemble:        0.34×0.84 + 0.33×0.84 + 0.33×0.90 = **0.860**

---

## Mathematical Reference (all tasks derive from this)

**USL (Gunther 1993):**
```
X(N) = N / (1 + α(N−1) + β·N(N−1))
```
- α ∈ [0,0.5]: serial/contention fraction (Amdahl term)
- β ≥ 0: coherency/synchronization cost per pair

**β_eff = β₀ / κ̄** where κ̄ = CG_mean (Definition 6, blog):
- High CG_mean → lower β_eff → higher N_max (better constraint corpus unlocks more agents)
- Low CG_mean → higher β_eff → lower N_max (divergent outputs require more coordination)

**Proposition 1 (USL N_max derivation):**
```
dX/dN = 0  →  (1 + α(N−1) + β·N(N−1)) − N(α + β(2N−1)) = 0
           →  1 − α − β·N² = 0  [first-order approximation for large N]
           →  N_max = √((1−α) / β_eff)
```
Discrete form: `round(√((1−α) / β_eff))`

**Verification:** AI agent tier (α=0.15, β₀=0.01, CG_mean=0.4):
- β_eff = 0.01/0.4 = 0.025
- N_max = √(0.85/0.025) = √34 = 5.83 → round → 6
- X(5) = 5/(1+0.15·4+0.025·20) = 5/1.1 = 2.381
- X(6) = 6/(1+0.15·5+0.025·30) = 6/1.5 = 2.400  ← discrete peak ✓
- X(7) = 7/(1+0.15·6+0.025·42) = 7/1.94 = 2.373  ← falls

**USL linearization for β₀ calibration:**
```
z(N) = N·T_parallel(N)/T₁ − 1 = α(N−1) + β₀·N(N−1)

Two measurements: N=2 (phase A) and N=M (phase B):
  z₂ = α + 2β₀        [N=2: α(2−1) + β₀·2(2−1)]
  z_M = α(M−1) + β₀·M(M−1)

Analytical solution:
  β₀ = (z_M − z₂·(M−1)) / ((M−1)(M−2))   [only valid when M≥3]
  α  = z₂ − 2β₀
```
Fallback when M<3: use `alpha_contention` and `beta_base_default`.

**Fréchet median (Fréchet 1948, Vardi-Zhang 2000):**
```
m* = argmin_{x ∈ S} Σᵢ d(x, sᵢ)
```
In metric space (P(Tokens), d_J) where d_J = 1 − J(A,B), `ConsensusMedian` IS the Fréchet median. Properties:
- Breakdown point 1/2 (tolerates up to n/2 − 1 outliers)
- Strictly stronger than Krum's ⌊(n−3)/4⌋/n breakdown point
- When semantic_jaccard replaces token jaccard, paraphrases of the same answer cluster correctly

**CRDT powerset lattice (Shapiro et al. 2011):**
- Partial order: S₁ ≤ S₂ iff S₁ ⊆ S₂ (containment)  
- Join (LUB): S₁ ⊔ S₂ = S₁ ∪ S₂ with tie-breaking by max score
- Three semilattice axioms: commutativity, associativity, idempotency → all hold for set union

**Pareto scalarization (multi-objective optimization):**
```
topology* = argmax_i (w_T · T_i + w_E · E_i + w_D · D_i)
```
where T=throughput, E=containment, D=diversity, w = pareto_weights from task manifest.

---

## File Map

| Task | Files Modified |
|------|---------------|
| 1 | `crates/h2ai-types/src/physics.rs`, `crates/h2ai-types/src/events.rs`, `crates/h2ai-types/tests/physics_test.rs` |
| 2 | `crates/h2ai-config/src/lib.rs` |
| 3 | `crates/h2ai-autonomic/src/calibration.rs` |
| 4 | `crates/h2ai-autonomic/src/planner.rs`, `crates/h2ai-autonomic/tests/planner_test.rs` |
| 5 | `crates/h2ai-state/src/bft.rs`, `crates/h2ai-autonomic/src/merger.rs` |
| 6 | `crates/h2ai-state/src/semilattice.rs` |
| 7 | `crates/h2ai-state/tests/nats_test.rs` |
| 8 | `scripts/simulate_usl.py`, `scripts/validate_ensemble_theory.py` |
| 9 | `docs/architecture/math-apparatus.md`, `docs/architecture/design-specification.md`, `docs/guides/theory-to-implementation.md` |
| 10 | Workspace verification |

---

## Task 1: USL Physics — `CoherencyCoefficients` and `n_max()`

**Files:**
- Modify: `crates/h2ai-types/src/physics.rs`
- Modify: `crates/h2ai-types/tests/physics_test.rs`

**What changes:**
1. Rename field `kappa_base` → `beta_base` with `#[serde(alias = "kappa_base")]` for backward compat with existing serialized events
2. Rename method `kappa_eff()` → `beta_eff()` implementing `beta_base / cg_mean`
3. Fix `n_max()` to use USL Proposition 1 formula
4. Add `PhysicsError::InvalidBetaBase` variant
5. Update `new()` to validate `beta_base ≥ 0`
6. In `crates/h2ai-types/src/events.rs` line 40: rename `kappa_eff: f64` → `beta_eff: f64` in `TopologyProvisionedEvent` with `#[serde(alias = "kappa_eff")]` for backward compat
7. Update tests

- [ ] **Step 1: Write the failing test for USL n_max**

In `crates/h2ai-types/tests/physics_test.rs`, add:

```rust
#[test]
fn coherency_coefficients_usl_n_max_ai_agents() {
    // AI-agent tier from blog: α=0.15, β₀=0.01, CG_mean=0.4
    // β_eff = 0.01/0.4 = 0.025
    // N_max = round(√(0.85/0.025)) = round(√34) = round(5.831) = 6
    let cc = CoherencyCoefficients::new(0.15, 0.01, vec![0.4]).unwrap();
    let n_max = cc.n_max();
    assert!(
        (n_max - 6.0).abs() < 1.0,
        "AI-agent tier N_max must be ≈6, got {n_max}"
    );
}

#[test]
fn coherency_coefficients_beta_eff_divides_by_cg_mean() {
    // β_eff = β₀ / CG_mean = 0.01 / 0.4 = 0.025
    let cc = CoherencyCoefficients::new(0.15, 0.01, vec![0.4]).unwrap();
    let beta_eff = cc.beta_eff();
    assert!(
        (beta_eff - 0.025).abs() < 1e-10,
        "β_eff = β₀/CG_mean = 0.025, got {beta_eff}"
    );
}

#[test]
fn coherency_coefficients_human_team_tier() {
    // Human team tier: α=0.10, β₀=0.005, CG_mean=0.6
    // β_eff = 0.005/0.6 ≈ 0.00833
    // N_max = round(√(0.90/0.00833)) = round(√108) = round(10.39) = 10
    let cc = CoherencyCoefficients::new(0.10, 0.005, vec![0.6]).unwrap();
    let n_max = cc.n_max();
    assert!(
        (n_max - 10.0).abs() < 1.5,
        "Human team N_max must be ≈10, got {n_max}"
    );
}

#[test]
fn coherency_coefficients_cpu_core_tier() {
    // CPU tier: α=0.02, β₀=0.0003, CG_mean=1.0
    // β_eff = 0.0003/1.0 = 0.0003
    // N_max = round(√(0.98/0.0003)) = round(√3267) = round(57.16) = 57
    let cc = CoherencyCoefficients::new(0.02, 0.0003, vec![1.0]).unwrap();
    let n_max = cc.n_max();
    assert!(
        (n_max - 57.0).abs() < 2.0,
        "CPU core tier N_max must be ≈57, got {n_max}"
    );
}
```

- [ ] **Step 2: Run the tests to confirm they fail**

```bash
cd /workspaces/h2ai-control-plane
cargo test --package h2ai-types usl_n_max 2>&1 | head -30
```
Expected: FAIL — `beta_eff` method not found, `n_max` gives wrong value.

- [ ] **Step 3: Update `CoherencyCoefficients` in `physics.rs`**

Replace lines 36–96 of `crates/h2ai-types/src/physics.rs`:

```rust
/// Calibrated coherency parameters for a set of compute adapters.
///
/// `alpha` is the serial contention fraction from USL calibration.
/// `beta_base` (β₀) is the base coherency cost per pair, measured from two-phase
/// calibration timing via USL linearization. Divided by CG_mean to give β_eff.
/// `n_max()` computes the USL-optimal ceiling: `round(√((1−α) / β_eff))`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherencyCoefficients {
    pub alpha: f64,
    /// β₀ — base coherency cost per agent pair, measured from calibration timing.
    /// Divided by κ̄ = CG_mean to yield β_eff = β₀ / κ̄.
    #[serde(alias = "kappa_base")]
    pub beta_base: f64,
    pub cg_samples: Vec<f64>,
}

impl CoherencyCoefficients {
    pub fn new(alpha: f64, beta_base: f64, cg_samples: Vec<f64>) -> Result<Self, PhysicsError> {
        if !(0.0..1.0).contains(&alpha) {
            return Err(PhysicsError::InvalidAlpha(alpha));
        }
        if cg_samples.is_empty() {
            return Err(PhysicsError::EmptyCgSamples);
        }
        Ok(Self { alpha, beta_base, cg_samples })
    }

    /// β_eff = β₀ / κ̄ (Definition 6, USL+CG coupling).
    ///
    /// Higher CG_mean → lower β_eff → higher N_max.
    /// Lower CG_mean → higher β_eff → lower N_max (divergent outputs cost more to merge).
    pub fn beta_eff(&self) -> f64 {
        self.beta_base / self.cg_mean().max(f64::EPSILON)
    }

    /// Maximum useful ensemble size from USL Proposition 1.
    ///
    /// N_max = round(√((1 − α) / β_eff)).
    /// Derived by setting dX/dN = 0 in X(N) = N/(1 + α(N−1) + β·N(N−1)).
    /// Beyond N_max the USL throughput curve enters retrograde (X decreasing).
    pub fn n_max(&self) -> f64 {
        let beta_eff = self.beta_eff().max(f64::EPSILON);
        ((1.0 - self.alpha).max(0.0) / beta_eff).sqrt().round()
    }

    pub fn cg_mean(&self) -> f64 {
        self.cg_samples.iter().sum::<f64>() / self.cg_samples.len() as f64
    }

    pub fn cg_std_dev(&self) -> f64 {
        let mean = self.cg_mean();
        let variance = self.cg_samples.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / self.cg_samples.len() as f64;
        variance.sqrt()
    }
}
```

Also add `InvalidBetaBase` to `PhysicsError`:
```rust
#[error("beta_base must be ≥ 0, got {0}")]
InvalidBetaBase(f64),
```

- [ ] **Step 4: Update existing tests in `physics_test.rs` to use new field/method names**

Replace the `coherency_coefficients_computes_kappa_eff` test:
```rust
#[test]
fn coherency_coefficients_beta_eff_computation() {
    let cc = CoherencyCoefficients {
        alpha: 0.12,
        beta_base: 0.020,
        cg_samples: vec![0.6, 0.7, 0.65],
    };
    // CG_mean = (0.6+0.7+0.65)/3 = 0.65
    // β_eff = 0.020 / 0.65 ≈ 0.03077
    let beta_eff = cc.beta_eff();
    let expected = 0.020 / 0.65;
    assert!(
        (beta_eff - expected).abs() < 1e-10,
        "β_eff = β₀/CG_mean = {expected:.6}, got {beta_eff:.6}"
    );
}
```

Replace the `coherency_coefficients_computes_n_max` test:
```rust
#[test]
fn coherency_coefficients_computes_n_max_usl() {
    let cc = CoherencyCoefficients {
        alpha: 0.12,
        beta_base: 0.020,
        cg_samples: vec![0.65],
    };
    // β_eff = 0.020/0.65 ≈ 0.03077
    // N_max = round(√(0.88/0.03077)) = round(√28.6) = round(5.35) = 5
    let n_max = cc.n_max();
    let expected = ((1.0_f64 - 0.12) / (0.020 / 0.65)).sqrt().round();
    assert!(
        (n_max - expected).abs() < 1e-9,
        "N_max = round(√((1−α)/β_eff)) = {expected:.3}, got {n_max}"
    );
}
```

Update `coherency_coefficients_serde_round_trip`:
```rust
#[test]
fn coherency_coefficients_serde_round_trip() {
    let cc = CoherencyCoefficients {
        alpha: 0.10,
        beta_base: 0.015,
        cg_samples: vec![0.55, 0.70, 0.62],
    };
    let json = serde_json::to_string(&cc).unwrap();
    let back: CoherencyCoefficients = serde_json::from_str(&json).unwrap();
    assert_eq!(cc.alpha, back.alpha);
    assert_eq!(cc.beta_base, back.beta_base);
    assert_eq!(cc.cg_samples.len(), back.cg_samples.len());
}
```

Add backward-compat serde test:
```rust
#[test]
fn coherency_coefficients_kappa_base_alias_loads_as_beta_base() {
    // Existing serialized events use "kappa_base" — must still deserialize correctly.
    let json = r#"{"alpha":0.12,"kappa_base":0.021,"cg_samples":[0.68,0.74,0.71]}"#;
    let cc: CoherencyCoefficients = serde_json::from_str(json).unwrap();
    assert!((cc.alpha - 0.12).abs() < 1e-10);
    assert!((cc.beta_base - 0.021).abs() < 1e-10);
}
```

All other tests using `kappa_base` struct literal must change to `beta_base`.

- [ ] **Step 5: Run tests to confirm they pass**

```bash
cargo test --package h2ai-types 2>&1 | tail -20
```
Expected: all tests pass, no compilation errors.

- [ ] **Step 6: Commit**

```bash
git add crates/h2ai-types/src/physics.rs crates/h2ai-types/tests/physics_test.rs
git commit -m "feat(physics): USL n_max, beta_eff — replace Amdahl with Proposition 1"
```

---

## Task 2: Config — Replace `kappa_eff_factor` with `beta_base_default`

**Files:**
- Modify: `crates/h2ai-config/src/lib.rs`

**What changes:** Remove `kappa_eff_factor` (a magic constant with no physical meaning) and replace with `beta_base_default` (the β₀ for the deployment tier when M<3 adapters run). Default 0.01 matches the AI-agents tier from the blog calibration table.

- [ ] **Step 1: Grep all usages of `kappa_eff_factor`**

```bash
grep -rn "kappa_eff_factor" /workspaces/h2ai-control-plane/
```
Expected usages: `h2ai-config/src/lib.rs` (field + default fn), `h2ai-autonomic/src/calibration.rs` (consumer). Those are the only two files to update.

- [ ] **Step 2: Update `H2AIConfig`**

In `crates/h2ai-config/src/lib.rs`, make these changes:

Replace:
```rust
    /// κ multiplier for coordination cost: κ_eff = kappa_eff_factor × (2 − CG_mean).
    #[serde(default = "default_kappa_eff_factor")]
    pub kappa_eff_factor: f64,
```
With:
```rust
    /// β₀ (beta_base_default) — base coherency cost per agent pair for this deployment tier.
    /// Used as calibration fallback when fewer than 3 adapters are available.
    /// Default 0.01 = AI-agents tier (blog calibration table).
    /// Use 0.005 for human-team tier, 0.0003 for CPU-core tier.
    #[serde(default = "default_beta_base", alias = "kappa_eff_factor")]
    pub beta_base_default: f64,
```

Replace the default function:
```rust
fn default_beta_base() -> f64 {
    0.01
}
```

Remove `fn default_kappa_eff_factor`.

Update `Default` impl, replacing `kappa_eff_factor: 0.019` with `beta_base_default: 0.01`.

- [ ] **Step 3: Run compilation check**

```bash
cargo check --package h2ai-config 2>&1
```
Expected: one compile error — `calibration.rs` still references `kappa_eff_factor`. Fix in Task 3.

- [ ] **Step 4: Commit after Task 3 is done (these two are coupled)**

---

## Task 3: Two-Phase USL Calibration

**Files:**
- Modify: `crates/h2ai-autonomic/src/calibration.rs`

**What changes:** Add Phase A (run 2 adapters) and Phase B (run all M adapters) to measure both T₂ and T_M. Derive β₀ analytically from USL linearization. When M<3, fall back to `beta_base_default`.

The current code already measures `t_parallel` (total wall-clock) and per-adapter times. We extend this with a two-phase approach.

- [ ] **Step 1: Write a unit test for `usl_fit`**

At the bottom of `crates/h2ai-autonomic/src/calibration.rs`, add:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usl_fit_recovers_known_params() {
        // Ground truth: α=0.15, β₀=0.01
        // T₁ = 1.0 (single-adapter baseline)
        // At N=2: X(2) = 2/(1+0.15+0.02) = 2/1.17 ≈ 1.709
        //   T_parallel(2) = T₁/X(2) = 1.0/1.709 = 0.585
        //   z₂ = 2*T_parallel(2)/T₁ − 1 = 2*0.585 − 1 = 0.170
        // At N=4: X(4) = 4/(1+0.45+0.12) = 4/1.57 ≈ 2.548
        //   T_parallel(4) = 1.0/2.548 = 0.393
        //   z_4 = 4*0.393 − 1 = 0.571
        let t1 = 1.0_f64;
        let t2_parallel = t1 / (2.0 / (1.0 + 0.15 * 1.0 + 0.01 * 2.0 * 1.0));
        let t4_parallel = t1 / (4.0 / (1.0 + 0.15 * 3.0 + 0.01 * 4.0 * 3.0));
        let (alpha, beta0) = CalibrationHarness::usl_fit(t1, t2_parallel, 4, t4_parallel, 0.12, 0.01);
        assert!(
            (alpha - 0.15).abs() < 0.01,
            "USL fit: α should be ≈0.15, got {alpha:.4}"
        );
        assert!(
            (beta0 - 0.01).abs() < 0.002,
            "USL fit: β₀ should be ≈0.01, got {beta0:.6}"
        );
    }

    #[test]
    fn usl_fit_fallback_when_m_less_than_3() {
        let (alpha, beta0) = CalibrationHarness::usl_fit(1.0, 0.8, 2, 0.8, 0.12, 0.01);
        assert_eq!(alpha, 0.12, "fallback α when M<3");
        assert_eq!(beta0, 0.01, "fallback β₀ when M<3");
    }
}
```

- [ ] **Step 2: Run to confirm fail**

```bash
cargo test --package h2ai-autonomic usl_fit 2>&1 | head -20
```
Expected: FAIL — `usl_fit` not found.

- [ ] **Step 3: Add `usl_fit` method and restructure `CalibrationHarness`**

Replace the entire `crates/h2ai-autonomic/src/calibration.rs` with the new two-phase implementation:

```rust
use chrono::Utc;
use futures::future::join_all;
use h2ai_config::H2AIConfig;
use h2ai_context::jaccard::{jaccard, tokenize};
use h2ai_types::adapter::{ComputeRequest, IComputeAdapter};
use h2ai_types::events::CalibrationCompletedEvent;
use h2ai_types::identity::TaskId;
use h2ai_types::physics::{
    CoherencyCoefficients, CoordinationThreshold, EnsembleCalibration, PhysicsError, TauValue,
    tau_alignment,
};
use thiserror::Error;
use tokio::time::Instant;

#[derive(Debug, Error)]
pub enum CalibrationError {
    #[error("adapter error: {0}")]
    Adapter(String),
    #[error("physics error: {0}")]
    Physics(#[from] PhysicsError),
    #[error("need at least 1 adapter to calibrate")]
    NoAdapters,
}

pub struct CalibrationInput<'a> {
    pub calibration_id: TaskId,
    pub task_prompts: Vec<String>,
    pub adapters: Vec<&'a dyn IComputeAdapter>,
    pub cfg: &'a H2AIConfig,
}

pub struct CalibrationHarness;

impl CalibrationHarness {
    pub async fn run(
        input: CalibrationInput<'_>,
    ) -> Result<CalibrationCompletedEvent, CalibrationError> {
        if input.adapters.is_empty() {
            return Err(CalibrationError::NoAdapters);
        }
        let tau = TauValue::new(input.cfg.calibration_tau)
            .expect("calibration_tau must be in [0,1]");
        let m = input.adapters.len();

        // ── Phase A: run the first 2 adapters in parallel to measure T₂ ──────
        // If M < 2, skip and rely on fallback.
        let (t1, t2_parallel, t_m_parallel, adapter_outputs) = if m >= 2 {
            let phase_a_adapters = &input.adapters[..2];
            let (phase_a_outputs, t2_wall) =
                Self::run_adapters_parallel(phase_a_adapters, &input.task_prompts, tau, input.cfg)
                    .await?;

            // Per-adapter mean time as T₁ proxy
            let t1_proxy = phase_a_outputs.iter().map(|(_, t)| t).sum::<f64>() / 2.0;

            // ── Phase B: run all M adapters in parallel to measure T_M ─────
            let (all_outputs, t_m_wall) =
                Self::run_adapters_parallel(&input.adapters, &input.task_prompts, tau, input.cfg)
                    .await?;

            let outputs: Vec<Vec<String>> = all_outputs.into_iter().map(|(o, _)| o).collect();
            (t1_proxy, t2_wall, t_m_wall, outputs)
        } else {
            // M == 1: single adapter, no parallelism measurement
            let (single_out, t_single) =
                Self::run_adapters_parallel(&input.adapters, &input.task_prompts, tau, input.cfg)
                    .await?;
            let outputs: Vec<Vec<String>> = single_out.into_iter().map(|(o, _)| o).collect();
            (t_single, t_single, t_single, outputs)
        };

        // ── Derive α and β₀ via USL linearization ──────────────────────────
        let (alpha, beta_base) = Self::usl_fit(
            t1,
            t2_parallel,
            m,
            t_m_parallel,
            input.cfg.alpha_contention,
            input.cfg.beta_base_default,
        );

        // ── CG_mean from pairwise Jaccard across all adapter outputs ────────
        let (cg_samples, ensemble) = if adapter_outputs.len() < 2 {
            (vec![input.cfg.calibration_cg_fallback], None)
        } else {
            let cal_tau = TauValue::new(input.cfg.calibration_tau)
                .expect("calibration_tau must be in [0,1]");
            let align = tau_alignment(cal_tau, cal_tau); // = 1.0 when taus equal

            let mut pairs = Vec::new();
            for i in 0..adapter_outputs.len() {
                for j in (i + 1)..adapter_outputs.len() {
                    let oi = adapter_outputs[i].join(" ");
                    let oj = adapter_outputs[j].join(" ");
                    let ki = tokenize(&oi);
                    let kj = tokenize(&oj);
                    pairs.push(jaccard(&ki, &kj) * align);
                }
            }
            let cg_mean_val: f64 = pairs.iter().sum::<f64>() / pairs.len() as f64;
            let ec = if input.cfg.baseline_accuracy_proxy > 0.0 {
                EnsembleCalibration::from_measured_p(
                    input.cfg.baseline_accuracy_proxy,
                    cg_mean_val,
                    9,
                )
            } else {
                EnsembleCalibration::from_cg_mean(cg_mean_val, 9)
            };
            (pairs, Some(ec))
        };

        let cc = CoherencyCoefficients::new(alpha, beta_base, cg_samples)?;
        let coordination_threshold =
            CoordinationThreshold::from_calibration(&cc, input.cfg.coordination_threshold_max);

        Ok(CalibrationCompletedEvent {
            calibration_id: input.calibration_id,
            coefficients: cc,
            coordination_threshold,
            ensemble,
            timestamp: Utc::now(),
        })
    }

    /// Fit USL parameters α and β₀ from two parallel timing measurements.
    ///
    /// Uses the linearisation z(N) = N·T_parallel(N)/T₁ − 1 = α(N−1) + β₀·N(N−1).
    /// With two data points at N=2 and N=M:
    ///   β₀ = (z_M − z₂·(M−1)) / ((M−1)(M−2))
    ///   α  = z₂ − 2·β₀
    ///
    /// Falls back to (alpha_fallback, beta_fallback) when:
    /// - M < 3 (denominator is zero)
    /// - derived α or β₀ are degenerate (negative or > bounds)
    pub fn usl_fit(
        t1: f64,
        t2_parallel: f64,
        m: usize,
        t_m_parallel: f64,
        alpha_fallback: f64,
        beta_fallback: f64,
    ) -> (f64, f64) {
        if m < 3 || t1 < 1e-9 || t2_parallel < 1e-9 || t_m_parallel < 1e-9 {
            return (alpha_fallback, beta_fallback);
        }
        let m_f = m as f64;
        let z2 = 2.0 * t2_parallel / t1 - 1.0;
        let z_m = m_f * t_m_parallel / t1 - 1.0;

        let beta_denom = (m_f - 1.0) * (m_f - 2.0);
        if beta_denom.abs() < 1e-9 {
            return (alpha_fallback, beta_fallback);
        }
        let beta0 = (z_m - z2 * (m_f - 1.0)) / beta_denom;
        let alpha = z2 - 2.0 * beta0;

        // Degenerate measurement (e.g. super-linear speedup, negative params) — use fallback.
        // Must check BEFORE clamping; clamping would mask the degenerate case.
        if beta0 < 0.0 || alpha < 0.0 {
            return (alpha_fallback, beta_fallback);
        }

        let alpha_clamped = alpha.clamp(0.05, 0.5);
        let beta_clamped = beta0.clamp(1e-6, 0.1);
        (alpha_clamped, beta_clamped)
    }

    /// Run a slice of adapters concurrently on all prompts.
    /// Returns (outputs_per_adapter, wall_clock_seconds).
    async fn run_adapters_parallel(
        adapters: &[&dyn IComputeAdapter],
        prompts: &[String],
        tau: TauValue,
        cfg: &H2AIConfig,
    ) -> Result<(Vec<(Vec<String>, f64)>, f64), CalibrationError> {
        let t_wall_start = Instant::now();
        let futures: Vec<_> = adapters
            .iter()
            .map(|adapter| {
                let prompts_ref = prompts;
                async move {
                    let t0 = Instant::now();
                    let mut outputs = Vec::new();
                    for prompt in prompts_ref.iter() {
                        let req = ComputeRequest {
                            system_context: String::new(),
                            task: prompt.clone(),
                            tau,
                            max_tokens: cfg.calibration_max_tokens,
                        };
                        let resp = adapter
                            .execute(req)
                            .await
                            .map_err(|e| CalibrationError::Adapter(e.to_string()))?;
                        outputs.push(resp.output);
                    }
                    Ok::<_, CalibrationError>((outputs, t0.elapsed().as_secs_f64()))
                }
            })
            .collect();

        let results: Vec<Result<(Vec<String>, f64), CalibrationError>> = join_all(futures).await;
        let t_wall = t_wall_start.elapsed().as_secs_f64();

        let mut per_adapter = Vec::with_capacity(results.len());
        for r in results {
            per_adapter.push(r?);
        }
        Ok((per_adapter, t_wall))
    }
}
```

- [ ] **Step 4: Run tests**

```bash
cargo test --package h2ai-autonomic 2>&1 | tail -20
```
Expected: all pass (including usl_fit tests).

- [ ] **Step 5: Commit tasks 2+3 together**

```bash
git add crates/h2ai-config/src/lib.rs crates/h2ai-autonomic/src/calibration.rs
git commit -m "feat(calibration): two-phase USL fitting for α and β₀, replace kappa_eff_factor with beta_base_default"
```

---

## Task 4: Pareto Topology Selection — Weighted Scalarization

**Files:**
- Modify: `crates/h2ai-autonomic/src/planner.rs`

**What changes:** Replace the ad-hoc if/else `select_topology` with a formal weighted dot product over the three Pareto-frontier topologies. The frontier topologies are:

| Topology | T (throughput) | E (containment) | D (diversity) |
|----------|---------------|-----------------|---------------|
| HierarchicalTree | 0.96 | 0.96 | 0.60 |
| TeamSwarmHybrid | 0.84 | 0.91 | 0.95 |
| Ensemble | 0.84 | 0.84 | 0.90 |

Selection: `argmax_i(w_T · T_i + w_E · E_i + w_D · D_i)`.

`TeamSwarmHybrid` is forced when review gates exist (gates imply oversight structure).

- [ ] **Step 1: Write a failing test**

Add to `crates/h2ai-autonomic/src/planner.rs` (or a new test file):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use h2ai_types::config::ParetoWeights;

    #[test]
    fn select_topology_containment_heavy_gives_hierarchical() {
        // w_E dominates → HierarchicalTree (highest E score 0.96)
        let weights = ParetoWeights { throughput: 0.1, containment: 0.8, diversity: 0.1 };
        let result = TopologyPlanner::select_topology(&weights, &[], 9.0);
        assert!(
            matches!(result, TopologyKind::HierarchicalTree { .. }),
            "containment-heavy weights → HierarchicalTree, got {:?}", result
        );
    }

    #[test]
    fn select_topology_diversity_heavy_gives_team_swarm() {
        // w_D dominates → TeamSwarmHybrid (highest D score 0.95)
        let weights = ParetoWeights { throughput: 0.1, containment: 0.1, diversity: 0.8 };
        let result = TopologyPlanner::select_topology(&weights, &[], 9.0);
        assert!(
            matches!(result, TopologyKind::TeamSwarmHybrid),
            "diversity-heavy weights → TeamSwarmHybrid, got {:?}", result
        );
    }

    #[test]
    fn select_topology_review_gates_override_weights() {
        // Even with throughput-heavy weights, review gates force TeamSwarmHybrid
        let weights = ParetoWeights { throughput: 0.9, containment: 0.05, diversity: 0.05 };
        let gate = h2ai_types::config::ReviewGate::default();
        let result = TopologyPlanner::select_topology(&weights, &[gate], 9.0);
        assert!(
            matches!(result, TopologyKind::TeamSwarmHybrid),
            "review gates must force TeamSwarmHybrid"
        );
    }

    #[test]
    fn select_topology_equal_weights_selects_hierarchical() {
        // Equal weights → all three topologies score (0.96+0.96+0.60)/3 vs (0.84+0.91+0.95)/3
        // HierarchicalTree: (0.96+0.96+0.60)/3 = 0.840
        // TeamSwarmHybrid: (0.84+0.91+0.95)/3 = 0.900  ← wins
        // Ensemble: (0.84+0.84+0.90)/3 = 0.860
        // Wait — TeamSwarmHybrid should win here.
        let weights = ParetoWeights { throughput: 0.333, containment: 0.333, diversity: 0.334 };
        let result = TopologyPlanner::select_topology(&weights, &[], 9.0);
        assert!(
            matches!(result, TopologyKind::TeamSwarmHybrid),
            "equal weights → TeamSwarmHybrid (highest mean score), got {:?}", result
        );
    }
}
```

- [ ] **Step 2: Run to confirm fail**

```bash
cargo test --package h2ai-autonomic select_topology 2>&1 | head -20
```

- [ ] **Step 3: Rewrite `select_topology` in `planner.rs`**

Replace lines 91–108:

```rust
    fn select_topology(
        pareto_weights: &ParetoWeights,
        review_gates: &[ReviewGate],
        _n_max: f64,
    ) -> TopologyKind {
        if !review_gates.is_empty() {
            return TopologyKind::TeamSwarmHybrid;
        }

        // Pareto-frontier topologies with (T, E, D) scores.
        // These scores match the Pareto matrix in docs/guides/theory-to-implementation.md.
        struct Candidate {
            score_t: f64,
            score_e: f64,
            score_d: f64,
            make: fn(f64) -> TopologyKind,
        }
        let candidates: [Candidate; 3] = [
            Candidate { score_t: 0.96, score_e: 0.96, score_d: 0.60,
                make: |n| TopologyKind::HierarchicalTree { branching_factor: Some((n.floor() as u8).max(2)) } },
            Candidate { score_t: 0.84, score_e: 0.91, score_d: 0.95,
                make: |_| TopologyKind::TeamSwarmHybrid },
            Candidate { score_t: 0.84, score_e: 0.84, score_d: 0.90,
                make: |_| TopologyKind::Ensemble },
        ];

        let wt = pareto_weights.throughput as f64;
        let we = pareto_weights.containment as f64;
        let wd = pareto_weights.diversity as f64;

        let (best_idx, _) = candidates
            .iter()
            .enumerate()
            .map(|(i, c)| (i, wt * c.score_t + we * c.score_e + wd * c.score_d))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((2, 0.0)); // default: Ensemble

        (candidates[best_idx].make)(_n_max)
    }
```

Also remove the `kappa_eff` variable from `provision()` since it's no longer directly used in topology selection:
```rust
// In provision(), remove:
let kappa_eff = input.cc.kappa_eff();
// Replace in the returned event with:
let beta_eff = input.cc.beta_eff();
// Update TopologyProvisionedEvent field if it uses kappa_eff
```

Check the `TopologyProvisionedEvent` struct — if it has a `kappa_eff` field, update it to `beta_eff` with serde alias.

- [ ] **Step 4: Update `planner.rs` to call `beta_eff()` and use `beta_eff` field**

In `crates/h2ai-autonomic/src/planner.rs` `provision()`, replace:
```rust
let kappa_eff = input.cc.kappa_eff();
```
with:
```rust
let beta_eff = input.cc.beta_eff();
```

And replace `kappa_eff,` in the event constructor with `beta_eff,`.

- [ ] **Step 5: Update `planner_test.rs` — fix field name and topology assertion**

In `crates/h2ai-autonomic/tests/planner_test.rs`:

a) Line 129: `assert!(event.kappa_eff > 0.0)` → `assert!(event.beta_eff > 0.0)`.

b) Lines 44–62: `planner_selects_ensemble_when_weights_balanced` asserts `Ensemble` but balanced weights (0.34, 0.33, 0.33) give TeamSwarmHybrid under Pareto scalarization (score 0.900 vs Ensemble 0.860). Rename and fix:

```rust
#[test]
fn planner_selects_team_swarm_when_weights_balanced() {
    let cc = cc();
    let weights = ParetoWeights::new(0.34, 0.33, 0.33).unwrap();
    let cfg = H2AIConfig::default();
    let event = TopologyPlanner::provision(ProvisionInput {
        task_id: TaskId::new(),
        cc: &cc,
        pareto_weights: &weights,
        role_specs: &two_roles(),
        review_gates: vec![],
        auditor_config: auditor(),
        explorer_adapter: adapter(),
        force_topology: None,
        retry_count: 0,
        cfg: &cfg,
    });
    // TeamSwarmHybrid has highest mean Pareto score (0.900) with equal weights.
    // Scores: HierarchicalTree=0.841, TeamSwarmHybrid=0.900, Ensemble=0.860
    assert_eq!(event.topology_kind, TopologyKind::TeamSwarmHybrid);
}
```

- [ ] **Step 6: Run tests**

```bash
cargo test --package h2ai-autonomic 2>&1 | tail -20
```

- [ ] **Step 7: Commit**

```bash
git add crates/h2ai-autonomic/src/planner.rs crates/h2ai-autonomic/tests/planner_test.rs
git commit -m "feat(planner): Pareto topology selection via weighted scalarization, rename kappa_eff→beta_eff"
```

---

## Task 5: ConsensusMedian as Async Semantic Fréchet Median

**Files:**
- Modify: `crates/h2ai-state/src/bft.rs`
- Modify: `crates/h2ai-autonomic/src/merger.rs`

**What changes:** Make `ConsensusMedian::resolve` async and accept `Option<&dyn IComputeAdapter>`. Use `semantic_jaccard` for pairwise similarity so that paraphrases of the same answer cluster correctly. Document the Fréchet median property and breakdown point.

The mathematical claim: `ConsensusMedian` selects the Fréchet median `m* = argmin_{x∈S} Σᵢ d(x, sᵢ)` in metric space `(P(Tokens), d_J)`. With `n` proposals, it tolerates `⌊n/2⌋ − 1` outliers (breakdown point 1/2), vs Krum's `⌊(n−3)/4⌋/n`.

- [ ] **Step 1: Write failing async test**

```rust
#[cfg(test)]
mod tests {
    // ... existing tests ...

    #[tokio::test]
    async fn frechet_median_selects_semantically_central_proposal() {
        // Two semantically close proposals + one outlier
        // Without an adapter (token Jaccard mode), confirm basic behavior
        let p1 = prop("stateless JWT authentication token rotation ADR-001 compliant");
        let p2 = prop("JWT auth token stateless rotation ADR-001 implementation");
        let outlier = prop("Redis session store sliding window expiry database cache");
        let proposals = vec![p1.clone(), p2.clone(), outlier];
        let selected = ConsensusMedian::resolve(&proposals, None).await.unwrap();
        assert!(
            selected.raw_output == p1.raw_output || selected.raw_output == p2.raw_output,
            "Fréchet median must select from the close pair, got: {}",
            selected.raw_output
        );
    }
}
```

- [ ] **Step 2: Run to confirm fail**

```bash
cargo test --package h2ai-state frechet_median 2>&1 | head -20
```

- [ ] **Step 3: Update `bft.rs`**

Replace the entire `crates/h2ai-state/src/bft.rs`:

```rust
//! Fréchet Median proposal selection (ConsensusMedian).
//!
//! ## Mathematical foundation
//!
//! In metric space (𝒫(Tokens), d_J) where d_J(A,B) = 1 − J(A,B), the **Fréchet median**
//! (Fréchet 1948) is:
//!
//!   m* = argmin_{x ∈ S} Σᵢ d(x, sᵢ)
//!
//! Minimising the sum of distances is equivalent to maximising the sum of similarities:
//!
//!   m* = argmax_{x ∈ S} Σᵢ semantic_jaccard(x, sᵢ)
//!
//! **Breakdown point:** The Fréchet median is resistant to `⌊n/2⌋ − 1` outliers
//! (breakdown point 1/2, Vardi & Zhang 2000). This is strictly stronger than Krum's
//! breakdown point of `⌊(n−3)/4⌋/n` and does not require the cluster assumption.
//!
//! **When to use:**
//! - Honest stochastic diversity (LLMs producing different but correct outputs)
//! - Medium error costs (above BFT threshold, below Krum threshold)
//! - Any case where Krum's cluster assumption is violated
//!
//! When `adapter` is `Some(...)`, uses semantic Jaccard (synonyms score as close).
//! When `adapter` is `None`, falls back to token Jaccard (deterministic, no I/O).

use futures::future::join_all;
use h2ai_context::jaccard::{jaccard, tokenize};
use h2ai_context::similarity::semantic_jaccard;
use h2ai_types::adapter::IComputeAdapter;
use h2ai_types::events::ProposalEvent;
use std::cmp::Ordering;
use std::collections::HashSet;

pub struct ConsensusMedian;

impl ConsensusMedian {
    /// Fréchet median: returns the proposal with minimum sum of distances to all others.
    ///
    /// Equivalently, maximises mean pairwise semantic similarity.
    /// Ties broken by position (earlier index wins) for determinism.
    ///
    /// When `adapter` is `Some`, uses `semantic_jaccard` (paraphrase-aware).
    /// When `adapter` is `None`, uses token Jaccard (offline/test mode).
    pub async fn resolve(
        proposals: &[ProposalEvent],
        adapter: Option<&dyn IComputeAdapter>,
    ) -> Option<&ProposalEvent> {
        if proposals.is_empty() {
            return None;
        }
        if proposals.len() == 1 {
            return proposals.first();
        }

        let n = proposals.len();
        let outputs: Vec<&str> = proposals.iter().map(|p| p.raw_output.as_str()).collect();

        // Compute all pairwise similarities concurrently.
        // For n proposals there are n*(n-1)/2 pairs; at n≤9 this is at most 36 calls.
        let mut sims = vec![vec![1.0f64; n]; n]; // diagonal = 1.0 (self-similarity)

        let pairs: Vec<(usize, usize)> = (0..n)
            .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
            .collect();

        let pair_sims = join_all(
            pairs.iter().map(|&(i, j)| semantic_jaccard(outputs[i], outputs[j], adapter)),
        )
        .await;

        for (k, &(i, j)) in pairs.iter().enumerate() {
            sims[i][j] = pair_sims[k];
            sims[j][i] = pair_sims[k];
        }

        // Fréchet median: argmax of sum of similarities to all others
        proposals
            .iter()
            .enumerate()
            .max_by(|(i, _), (j, _)| {
                let si: f64 = sims[*i].iter().sum::<f64>() - 1.0; // subtract self-sim
                let sj: f64 = sims[*j].iter().sum::<f64>() - 1.0;
                si.partial_cmp(&sj).unwrap_or(Ordering::Equal)
            })
            .map(|(_, p)| p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use h2ai_types::config::AdapterKind;
    use h2ai_types::identity::{ExplorerId, TaskId};
    use h2ai_types::physics::TauValue;

    fn prop(text: &str) -> ProposalEvent {
        ProposalEvent {
            task_id: TaskId::new(),
            explorer_id: ExplorerId::new(),
            tau: TauValue::new(0.5).unwrap(),
            raw_output: text.into(),
            token_cost: text.len() as u64,
            adapter_kind: AdapterKind::CloudGeneric {
                endpoint: "mock".into(),
                api_key_env: "NONE".into(),
            },
            timestamp: Utc::now(),
        }
    }

    #[tokio::test]
    async fn empty_proposals_returns_none() {
        assert!(ConsensusMedian::resolve(&[], None).await.is_none());
    }

    #[tokio::test]
    async fn single_proposal_returns_itself() {
        let p = prop("only proposal");
        let result = ConsensusMedian::resolve(&[p.clone()], None).await.unwrap();
        assert_eq!(result.raw_output, p.raw_output);
    }

    #[tokio::test]
    async fn selects_consensus_not_outlier() {
        let ca = prop("JWT stateless auth ADR-001 compliant token rotation");
        let cb = prop("JWT stateless authentication compliant ADR-001 rotation");
        let outlier = prop("Redis session store sliding window expiry completely different");
        let proposals = vec![ca.clone(), cb.clone(), outlier];
        let selected = ConsensusMedian::resolve(&proposals, None).await.unwrap();
        assert!(
            selected.raw_output == ca.raw_output || selected.raw_output == cb.raw_output,
            "expected consensus proposal, got: {}", selected.raw_output
        );
    }

    #[tokio::test]
    async fn two_identical_proposals_returns_first_by_stability() {
        let p1 = prop("identical stateless JWT auth ADR-001");
        let p2 = prop("identical stateless JWT auth ADR-001");
        let proposals = vec![p1.clone(), p2];
        let result = ConsensusMedian::resolve(&proposals, None).await.unwrap();
        assert_eq!(result.raw_output, p1.raw_output);
    }

    #[tokio::test]
    async fn frechet_median_selects_semantically_central_proposal() {
        let p1 = prop("stateless JWT authentication token rotation ADR-001 compliant");
        let p2 = prop("JWT auth token stateless rotation ADR-001 implementation");
        let outlier = prop("Redis session store sliding window expiry database cache");
        let proposals = vec![p1.clone(), p2.clone(), outlier];
        let selected = ConsensusMedian::resolve(&proposals, None).await.unwrap();
        assert!(
            selected.raw_output == p1.raw_output || selected.raw_output == p2.raw_output,
            "Fréchet median must select from the close pair, got: {}",
            selected.raw_output
        );
    }
}
```

- [ ] **Step 4: Update `merger.rs` to pass adapter to ConsensusMedian**

In `crates/h2ai-autonomic/src/merger.rs`, `ConsensusMedian` arms need `.await` and the adapter:

```rust
MergeStrategy::ConsensusMedian => ConsensusMedian::resolve(&result.valid_proposals, adapter)
    .await
    .map(|p| p.raw_output.clone())
    .unwrap_or_default(),
```

And in the Krum/MultiKrum fallback arms:
```rust
} else {
    ConsensusMedian::resolve(proposals, adapter)
        .await
        .map(|p| p.raw_output.clone())
        .unwrap_or_default()
}
```

- [ ] **Step 5: Run all state and autonomic tests**

```bash
cargo test --package h2ai-state --package h2ai-autonomic 2>&1 | tail -30
```

- [ ] **Step 6: Commit**

```bash
git add crates/h2ai-state/src/bft.rs crates/h2ai-autonomic/src/merger.rs
git commit -m "feat(bft): ConsensusMedian as async semantic Fréchet median with breakdown point 1/2"
```

---

## Task 6: ProposalSet — LUB Semantics for CRDT Powerset Lattice

**Files:**
- Modify: `crates/h2ai-state/src/semilattice.rs`

**What changes:** Replace `or_insert` (first-wins, arbitrary) with `and_modify` taking max-score (LUB). Add a `join()` method documenting the CRDT semantics. The powerset join-semilattice `(2^E, ⊆, ∪)` keyed by ExplorerId has join `S₁ ⊔ S₂ = S₁ ∪ S₂` where conflicts resolve by max score.

**Note:** In normal operation each explorer produces exactly one proposal, so `or_insert` and max-score LUB are operationally equivalent. The LUB change is mathematically necessary for the CRDT proof (idempotency requires `join(x, x) = x` which holds for max-score but only accidentally for first-wins).

- [ ] **Step 1: Write failing tests (add after existing tests in `semilattice.rs`)**

Add a `prop_with_id` helper and two new tests. The existing `prop()` helper generates a random `ExplorerId` each call; `prop_with_id` pins it:

```rust
fn prop_with_id(text: &str, id: ExplorerId) -> ProposalEvent {
    ProposalEvent {
        task_id: TaskId::new(),
        explorer_id: id,
        tau: TauValue::new(0.5).unwrap(),
        raw_output: text.into(),
        token_cost: 1,
        adapter_kind: AdapterKind::CloudGeneric {
            endpoint: "mock".into(),
            api_key_env: "NONE".into(),
        },
        timestamp: Utc::now(),
    }
}

#[test]
fn insert_scored_keeps_higher_score_for_same_explorer() {
    let task_id = TaskId::new();
    let explorer_id = ExplorerId::new();
    let low  = prop_with_id("low score output",  explorer_id.clone());
    let high = prop_with_id("high score output", explorer_id.clone());

    let mut set = ProposalSet::new();
    set.insert_scored(low,  0.3);
    set.insert_scored(high, 0.9); // same explorer, higher score — LUB must win

    let result = SemilatticeResult::compile(task_id, set, vec![]);
    assert_eq!(result.valid_proposals.len(), 1, "one explorer → one slot");
    assert_eq!(
        result.valid_proposals[0].raw_output, "high score output",
        "LUB must keep higher-scored proposal"
    );
}

#[test]
fn join_is_idempotent() {
    let task_id = TaskId::new();
    let explorer_id = ExplorerId::new();
    let p = prop_with_id("proposal text", explorer_id.clone());

    let mut s1 = ProposalSet::new();
    s1.insert_scored(p.clone(), 0.7);
    let mut s2 = ProposalSet::new();
    s2.insert_scored(p, 0.7);

    let joined = ProposalSet::join(s1, s2);
    let result = SemilatticeResult::compile(task_id, joined, vec![]);
    assert_eq!(result.valid_proposals.len(), 1, "join(S, S) = S (idempotent)");
}
```

- [ ] **Step 2: Run to confirm fail**

```bash
cargo test --package h2ai-state insert_scored_updates 2>&1 | head -20
```

- [ ] **Step 3: Update `ProposalSet` in `semilattice.rs`**

Replace `insert_scored` and add `join`:

```rust
    /// Insert or update using max-score LUB semantics.
    ///
    /// If the explorer already has a proposal, the higher-scored one is kept.
    /// This implements the join-semilattice LUB for the powerset lattice keyed by ExplorerId:
    ///   S₁ ⊔ S₂ = S₁ ∪ S₂  with  conflict resolution by max(score₁, score₂).
    ///
    /// The three CRDT axioms (Shapiro et al. 2011) are satisfied:
    /// - Commutativity: join(S₁, S₂) = join(S₂, S₁)  [max is commutative]
    /// - Associativity: join(join(S₁,S₂),S₃) = join(S₁,join(S₂,S₃))  [set union is associative]
    /// - Idempotency:   join(S, S) = S  [max(x,x)=x for all x]
    pub fn insert_scored(&mut self, proposal: ProposalEvent, score: f64) {
        self.0
            .entry(proposal.explorer_id.clone())
            .and_modify(|(existing_proposal, existing_score)| {
                if score > *existing_score {
                    *existing_proposal = proposal.clone();
                    *existing_score = score;
                }
            })
            .or_insert((proposal, score));
    }

    /// Join two proposal sets (CRDT merge).
    ///
    /// join(S₁, S₂) = S₁ ∪ S₂ with max-score conflict resolution per explorer.
    /// This is the standard CRDT merge for a grow-only set extended with scores.
    pub fn join(mut lhs: Self, rhs: Self) -> Self {
        for (explorer_id, (proposal, score)) in rhs.0 {
            lhs.0
                .entry(explorer_id)
                .and_modify(|(ep, es)| {
                    if score > *es {
                        *ep = proposal.clone();
                        *es = score;
                    }
                })
                .or_insert((proposal, score));
        }
        lhs
    }
```

- [ ] **Step 4: Run tests**

```bash
cargo test --package h2ai-state 2>&1 | tail -20
```

- [ ] **Step 5: Commit**

```bash
git add crates/h2ai-state/src/semilattice.rs
git commit -m "feat(semilattice): CRDT max-score LUB for ProposalSet join-semilattice"
```

---

## Task 7: Update Tests That Use Old Field Names

**Files:**
- Modify: `crates/h2ai-state/tests/nats_test.rs`
- Check: `crates/h2ai-state/tests/nats_infra_test.rs`

All tests constructing `CoherencyCoefficients` with `kappa_base` must change to `beta_base`.

- [ ] **Step 1: Find all usages**

```bash
grep -rn "kappa_base" /workspaces/h2ai-control-plane/ --include="*.rs"
```

- [ ] **Step 2: Update `nats_test.rs`**

Replace all occurrences of:
```rust
CoherencyCoefficients::new(0.12, 0.021, vec![0.68, 0.74, 0.71])
```
with:
```rust
CoherencyCoefficients::new(0.12, 0.010, vec![0.68, 0.74, 0.71])
```
(Second arg is now `beta_base`, not `kappa_base`. Use 0.010 as the AI-tier default β₀.)

- [ ] **Step 3: Run all tests**

```bash
cargo test --workspace 2>&1 | grep -E "FAILED|error" | head -30
```

- [ ] **Step 4: Fix any remaining compilation errors from the field rename**

Check `TopologyProvisionedEvent` in `h2ai-types/src/events.rs`:

```bash
grep -rn "kappa_eff" /workspaces/h2ai-control-plane/crates/ --include="*.rs"
```

If `TopologyProvisionedEvent` has `kappa_eff: f64`, add `#[serde(alias = "kappa_eff")]` to it and update construction sites.

- [ ] **Step 5: Commit**

```bash
git add crates/h2ai-state/tests/nats_test.rs
git commit -m "fix(tests): update kappa_base → beta_base across test suite"
```

---

## Task 8: Update Simulation Scripts

**Files:**
- Modify: `scripts/simulate_usl.py`
- Modify: `scripts/validate_ensemble_theory.py`

**Situation:** `simulate_usl.py` already implements the **correct** USL math (`n_max`, `kappa_eff`, `usl`). The problem is naming inconsistency — the blog and docs now use `beta` not `kappa`. Update variable names and add USL curve-fit validation to `validate_ensemble_theory.py`.

- [ ] **Step 1: Update variable names in `simulate_usl.py`**

In `simulate_usl.py`:
- Rename function `kappa_eff(kappa_base, cg)` → `beta_eff(beta_base, cg)`
- Update all call sites: `kappa_eff(kb, cg)` → `beta_eff(kb, cg)`
- Rename parameter `kappa` → `beta` in `usl(N, alpha, kappa)` → `usl(N, alpha, beta)`
- Update all call sites
- Rename `kappa_e` → `beta_e` in `harness_attribution` signature and internal usage
- Update `LAYERS` to use `beta_base` naming in labels and variables
- Update the CROSS-REFERENCE MAP comments (§ Definition 4 now refers to β_eff, § Definition 5 to Extended USL with β)
- Change `KAPPA_AI = 0.025` → `BETA_AI = 0.025`

Specific changes in LAYERS constant:
```python
LAYERS = [
    # (label, alpha, beta_base, cg_mean, color)
    ("CPU cores  (α=0.02, β_eff=0.0003, N_max≈57)",   0.02, 0.0003, 1.0, "#2563eb"),
    ("Human teams (α=0.10, β_eff=0.0083, N_max≈10)",  0.10, 0.005,  0.6, "#16a34a"),
    ("AI agents  (α=0.15, β_eff=0.025,  N_max≈6)",    0.15, 0.01,   0.4, "#dc2626"),
]
```

In the plot loop:
```python
for label, alpha, bb, cg, color in LAYERS:
    be = beta_eff(bb, cg)
    X = usl(N, alpha, be)
    nm = n_max(alpha, be)
```

- [ ] **Step 2: Add USL curve-fit validation to `validate_ensemble_theory.py`**

At the end of `validate_ensemble_theory.py`, add a new validation section:

```python
# ── USL Curve-Fit Validation ──────────────────────────────────────────────────
print("\n[5] USL two-phase calibration recovery ...")

def usl_fit(t1, t2_parallel, m, tm_parallel):
    """Mirror of CalibrationHarness::usl_fit in Rust."""
    if m < 3 or t1 < 1e-9 or t2_parallel < 1e-9 or tm_parallel < 1e-9:
        return None, None
    m_f = float(m)
    z2 = 2.0 * t2_parallel / t1 - 1.0
    z_m = m_f * tm_parallel / t1 - 1.0
    beta_denom = (m_f - 1.0) * (m_f - 2.0)
    if abs(beta_denom) < 1e-9:
        return None, None
    beta0 = (z_m - z2 * (m_f - 1.0)) / beta_denom
    alpha = z2 - 2.0 * beta0
    return max(0.05, min(0.5, alpha)), max(1e-6, min(0.1, beta0))

def usl_throughput(N, alpha, beta):
    return N / (1.0 + alpha * (N - 1) + beta * N * (N - 1))

# Test recovery for all three calibration tiers
TIERS = [
    ("AI agents",   0.15, 0.01, 0.4),   # α, β₀, CG_mean
    ("Human teams", 0.10, 0.005, 0.6),
    ("CPU cores",   0.02, 0.0003, 1.0),
]

for label, true_alpha, true_beta0, cg_mean in TIERS:
    true_beta_eff = true_beta0 / cg_mean
    t1 = 1.0  # normalize
    t2_sim = t1 / usl_throughput(2, true_alpha, true_beta_eff)
    t4_sim = t1 / usl_throughput(4, true_alpha, true_beta_eff)

    # Test usl_fit with ground truth inputs
    recovered_alpha, recovered_beta0 = usl_fit(t1, t2_sim, 4, t4_sim)
    assert recovered_alpha is not None, f"usl_fit returned None for {label}"
    # β₀ recovery: fit recovers β_eff; we get β₀ back by multiplying by CG_mean
    recovered_beta_eff = recovered_alpha  # placeholder — fit recovers effective params
    # What we actually recover is α and β_eff directly from timing
    alpha_err = abs(recovered_alpha - true_alpha)
    # Note: the fit recovers β₀ directly, not β_eff
    beta_err = abs(recovered_beta0 - true_beta0)
    assert alpha_err < 0.01, f"{label}: α recovery error {alpha_err:.4f} > 0.01"
    assert beta_err < 0.002, f"{label}: β₀ recovery error {beta_err:.6f} > 0.002"
    print(f"  ✓ {label}: α={recovered_alpha:.4f} (Δ={alpha_err:.4f}), β₀={recovered_beta0:.6f} (Δ={beta_err:.6f})")

# Test fallback when M < 3
alpha_fb, beta_fb = usl_fit(1.0, 0.8, 2, 0.8)
assert alpha_fb is None, "M=2 must return None (fallback case)"
print("  ✓ M<3 fallback correctly returns None")

# Verify N_max formula for all tiers
def n_max(alpha, beta):
    return round(math.sqrt((1 - alpha) / beta))

expected_n_max = [6, 10, 57]  # AI, Human, CPU
for (label, true_alpha, true_beta0, cg_mean), expected in zip(TIERS, expected_n_max):
    nm = n_max(true_alpha, true_beta0 / cg_mean)
    assert abs(nm - expected) <= 1, f"{label}: N_max={nm}, expected≈{expected}"
    print(f"  ✓ {label}: N_max={nm} (expected {expected})")

print("[5] USL calibration recovery PASSED")
```

- [ ] **Step 3: Run the scripts to verify**

```bash
cd /workspaces/h2ai-control-plane
python3 scripts/validate_ensemble_theory.py 2>&1 | tail -20
python3 scripts/simulate_usl.py 2>&1 | tail -10
```
Expected: all assertions pass, PNGs saved to `scripts/output/`.

- [ ] **Step 4: Commit**

```bash
git add scripts/simulate_usl.py scripts/validate_ensemble_theory.py
git commit -m "feat(scripts): rename kappa→beta in simulate_usl.py, add USL calibration recovery validation"
```

---

## Task 9: Update Documentation

**Files:**
- Modify: `docs/architecture/math-apparatus.md`
- Modify: `docs/architecture/design-specification.md`
- Modify: `docs/guides/theory-to-implementation.md`

### 9.1 `math-apparatus.md`

**Section 3 — rewrite completely:**

Replace the existing Section 3 (Contention and Coordination) with:

```markdown
## 3. Universal Scalability Law and Coherency Coefficients

**Source:** Gunther (1993). Extended with CG coupling: blog.e-mindset.space/coordination-constant-usl-human-ai-teams/
**Implemented in:** `crates/h2ai-types/src/physics.rs`, `crates/h2ai-autonomic/src/calibration.rs`

### 3.1 USL Formula

```
X(N) = N / (1 + α(N−1) + β_eff·N(N−1))
```

Parameters:
- `α ∈ [0.05, 0.5]`: serial contention fraction — measured from two-phase calibration
- `β_eff = β₀ / κ̄`: effective coherency cost per agent pair
  - `β₀`: base coherency cost, measured from USL linearization
  - `κ̄` = CG_mean: mean Common Ground across adapter pairs

### 3.2 β_eff and CG Coupling (Definition 6)

```
β_eff = β₀ / κ̄  where  κ̄ = CG_mean
```

**Why this coupling exists:** When agents produce similar outputs (high CG_mean), they have established common ground. Per-pair synchronization cost is lower because agents agree on vocabulary and framing — they spend less effort reconciling divergent outputs. Formally, β₀ is the infrastructure overhead; dividing by κ̄ discounts it by the degree of established common ground.

**Consequence:** Higher CG_mean → lower β_eff → higher N_max. A richer constraint corpus that improves common ground directly raises the scalability ceiling.

### 3.3 N_max Derivation — Proposition 1

Setting dX/dN = 0 in X(N):

```
dX/dN = [(1 + α(N−1) + β·N(N−1)) − N(α + β(2N−1))] / (denominator)² = 0

Numerator = 0:
  1 + αN − α + βN² − βN − αN − 2βN² + βN = 0
  1 − α − βN² = 0
  N_max = √((1−α) / β_eff)
```

Discrete form (Rust): `round(√((1−α) / β_eff))`

**Calibrated ceilings for each tier:**

| Tier | α | β₀ | CG_mean | β_eff | N_max |
|------|---|-----|---------|-------|-------|
| CPU cores | 0.02 | 0.0003 | 1.0 | 0.0003 | 57 |
| Human teams | 0.10 | 0.005 | 0.6 | 0.0083 | 10 |
| AI agents | 0.15 | 0.01 | 0.4 | 0.025 | 6 |

Verification (AI tier): X(5)=2.381, X(6)=2.400 (peak), X(7)=2.373 → discrete N_max=6 ✓

### 3.4 Two-Phase Calibration Protocol

**USL linearization:** z(N) = N·T_parallel(N)/T₁ − 1 = α(N−1) + β₀·N(N−1)

**Phase A:** Run 2 adapters in parallel → measure T₂. Compute T₁ = mean per-adapter time.

**Phase B:** Run all M adapters in parallel → measure T_M.

**Analytical solution (M ≥ 3):**
```
z₂  = 2·T₂/T₁ − 1   [N=2 data point]
z_M = M·T_M/T₁ − 1  [N=M data point]

β₀ = (z_M − z₂·(M−1)) / ((M−1)(M−2))
α  = z₂ − 2·β₀
```

**Fallback:** When M < 3, use `alpha_contention` and `beta_base_default` from config.

**What this measures:** Actual I/O and scheduling serialization under the calibration workload. β₀ is the per-pair synchronization overhead before CG discount.
```

**Section 5 — add Fréchet Median subsection:**

After the existing cluster coherence content, add:

```markdown
### 6.3 Fréchet Median and ConsensusMedian

`ConsensusMedian` implements the **Fréchet median** (Fréchet 1948) in metric space (𝒫(Tokens), d_J):

```
m* = argmin_{x ∈ S} Σᵢ d_J(x, sᵢ)
   = argmax_{x ∈ S} Σᵢ semantic_jaccard(x, sᵢ)
```

**Breakdown point:** 1/2 (Vardi & Zhang 2000). Tolerates up to ⌊n/2⌋ − 1 outliers.
Compare Krum: breakdown point ⌊(n−3)/4⌋/n — weaker guarantee, requires cluster assumption.

**When to use:**
- Honest stochastic diversity (different phrasings of correct answer)
- Medium error costs (above BFT threshold)
- When cluster assumption for Krum is violated (diverse honest outputs)
```

**Section 7 — add CRDT subsection:**

```markdown
### 7. CRDT Powerset Lattice — ProposalSet

**Implemented in:** `crates/h2ai-state/src/semilattice.rs`
**Reference:** Shapiro et al. (2011) — Conflict-Free Replicated Data Types

`ProposalSet` is a **join-semilattice** on the powerset of proposals, keyed by `ExplorerId`:
- Partial order: S₁ ≤ S₂ iff S₁ ⊆ S₂ (containment)
- Join (LUB): S₁ ⊔ S₂ = S₁ ∪ S₂ with max-score conflict resolution per explorer

The three CRDT axioms hold for this `join` operation:
- **Commutativity:** max(s₁, s₂) = max(s₂, s₁) for any scores s₁, s₂
- **Associativity:** set union is associative
- **Idempotency:** join(S, S) = S because max(s, s) = s for any s

This means `ProposalSet::join` is a valid CRDT merge — proposal sets from concurrent writers can be merged in any order and the result is deterministic.
```

### 9.2 `design-specification.md`

In the math table, update:
- `κ_eff` references → `β_eff`
- `N_max = sqrt((1−α) / κ_eff)` → `N_max = round(√((1−α) / β_eff))`
- Add row: `β_eff = β₀ / κ̄` where κ̄ = CG_mean

### 9.3 `theory-to-implementation.md`

Update the Pareto scores table to note it corresponds to the weighted scalarization formula:
```
topology* = argmax_i(w_T·T_i + w_E·E_i + w_D·D_i)
```

- [ ] **Step 1: Apply math-apparatus.md changes**

Make the section 3 rewrite, section 6 Fréchet median addition, and section 7 CRDT addition described above.

- [ ] **Step 2: Update design-specification.md**

```bash
grep -n "kappa\|κ_eff\|N_max" /workspaces/h2ai-control-plane/docs/architecture/design-specification.md | head -20
```
Then apply targeted edits.

- [ ] **Step 3: Update theory-to-implementation.md**

```bash
grep -n "Pareto\|select_topology\|kappa" /workspaces/h2ai-control-plane/docs/guides/theory-to-implementation.md | head -20
```
Then add the scalarization formula above the topology table.

- [ ] **Step 4: Commit**

```bash
git add docs/
git commit -m "docs: USL Proposition 1, β_eff, two-phase calibration, Fréchet median, CRDT proof"
```

---

## Task 10: Full Workspace Verification

**Files:** None modified — verification only.

- [ ] **Step 1: Run full workspace test suite**

```bash
cd /workspaces/h2ai-control-plane
cargo test --workspace 2>&1 | tee /tmp/test_results.txt
tail -30 /tmp/test_results.txt
```
Expected: zero failures, zero compilation errors.

- [ ] **Step 2: Confirm no `kappa_base` or `kappa_eff_factor` structural usages remain**

```bash
grep -rn "kappa_base\|kappa_eff_factor" /workspaces/h2ai-control-plane/crates/ --include="*.rs" \
  | grep -v "alias\|test\|//\|doc"
```
Expected: zero hits (only serde aliases and comments remain).

- [ ] **Step 3: Run simulation scripts**

```bash
python3 scripts/validate_ensemble_theory.py 2>&1 | tail -15
python3 scripts/simulate_usl.py 2>&1 | tail -5
```
Expected: all assertions pass, 7 PNGs saved to `scripts/output/`.

- [ ] **Step 4: Verify three-tier calibration table numerically**

```bash
python3 -c "
import math
tiers = [
    ('CPU cores',   0.02, 0.0003, 1.0),
    ('Human teams', 0.10, 0.005,  0.6),
    ('AI agents',   0.15, 0.01,   0.4),
]
for label, alpha, beta0, cg in tiers:
    beta_eff = beta0 / cg
    n_max = round(math.sqrt((1 - alpha) / beta_eff))
    # Verify at discrete peak
    def x(N): return N / (1 + alpha*(N-1) + beta_eff*N*(N-1))
    print(f'{label}: N_max={n_max}, X(N_max-1)={x(n_max-1):.4f}, X(N_max)={x(n_max):.4f}, X(N_max+1)={x(n_max+1):.4f}')
    assert x(n_max) >= x(n_max-1) and x(n_max) >= x(n_max+1), f'N_max={n_max} is not the peak for {label}'
    print(f'  ✓ Peak verified')
"
```

- [ ] **Step 5: Final commit tag**

```bash
git tag -a v0.2.0-usl -m "USL-complete: β_eff coupling, two-phase calibration, Fréchet median, CRDT LUB, Pareto scalarization"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] P1 (Amdahl n_max) → Task 1 fixes `n_max()` to USL Proposition 1
- [x] P2 (circular β₀ formula) → Task 3 replaces with USL linearization from timing
- [x] P3 (single-phase calibration) → Task 3 adds two-phase protocol
- [x] P4 (ad-hoc topology) → Task 4 replaces with weighted Pareto scalarization
- [x] P5 (sync token-only ConsensusMedian) → Task 5 makes async semantic Fréchet median
- [x] ProposalSet LUB → Task 6
- [x] Test suite updates → Task 7
- [x] Script updates → Task 8
- [x] Documentation → Task 9
- [x] Workspace verification → Task 10

**Backward compatibility:**
- [x] `kappa_base` serde alias on `CoherencyCoefficients.beta_base` — existing events deserialize
- [x] `kappa_eff_factor` serde alias on `H2AIConfig.beta_base_default` — existing configs load
- [x] `TopologyProvisionedEvent.kappa_eff` confirmed at events.rs:40 — `#[serde(alias = "kappa_eff")]` added and renamed to `beta_eff` in Task 1

**Verified test breakages (pre-corrected in plan):**
- [x] `planner_test.rs:61` asserts `Ensemble` with balanced weights — WRONG under Pareto scalarization (TeamSwarmHybrid wins 0.900). Updated in Task 4 Step 5.
- [x] `planner_test.rs:129` uses `event.kappa_eff` — updated to `event.beta_eff` in Task 4 Step 5.
- [x] `semilattice.rs` LUB test needs `prop_with_id` helper — added in Task 6 Step 1.
- [x] `usl_fit` checked negatives after clamping (bug) — fixed to check BEFORE clamping in Task 3.

**Pareto score table verified numerically:**
- HierarchicalTree (T=0.96, E=0.96, D=0.60): wins when w_T or w_E dominant
- TeamSwarmHybrid (T=0.84, E=0.91, D=0.95): wins when w_D dominant OR equal weights
- Ensemble (T=0.84, E=0.84, D=0.90): never wins (dominated by TeamSwarmHybrid on all axes)
- Note: Ensemble is on the Pareto frontier by the strict definition (no single topology strictly dominates it) but is dominated in practice by TeamSwarmHybrid+Ensemble combination

**Mathematical proofs included:**
- [x] USL Proposition 1 derivation (dX/dN=0 → N_max = √((1−α)/β_eff))
- [x] β_eff coupling rationale (CG_mean as discount on synchronization cost)
- [x] Fréchet median breakdown point 1/2 (Vardi-Zhang 2000)
- [x] CRDT semilattice axioms (Shapiro et al. 2011)
- [x] Pareto scalarization optimality (standard multi-objective)

**Type consistency:**
- Task 1 introduces `beta_base` and `beta_eff()` 
- Task 2 introduces `beta_base_default` in config
- Task 3 reads `cfg.beta_base_default` (not `kappa_eff_factor`)
- Task 4 calls `cc.beta_eff()` (not `cc.kappa_eff()`)
- Task 5 signature `resolve(proposals, adapter)` used consistently in merger
- All consistent ✓
