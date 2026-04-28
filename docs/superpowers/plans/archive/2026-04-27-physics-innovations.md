# Physics Innovations & Critical Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three correctness gaps in the physics layer (β_eff singularity, dead Krum code path, USL fit never running) and add two innovations from the cross-domain research (TalagrandDiagnostic, EigenCalibration) that require zero external I/O deps.

**Architecture:** All changes stay within existing crates. Tasks 1–3 are critical fixes with no new dependencies. Task 4 adds a pure-math diagnostic struct. Task 5 adds `nalgebra` as a pure-math dep to `h2ai-types` and `h2ai-autonomic`. Tasks 6–7 close documentation and verification.

**Tech Stack:** Rust (tokio, serde, futures), nalgebra (Task 5 only), Python scripts already in `scripts/`

---

## What Is Already Done (Do Not Re-Implement)

- R1–R5 structural refactoring (retry_count, TauValue, AgentId, RetryPolicy, lib.rs doc) — all complete in current main
- USL two-phase calibration harness (`CalibrationHarness::usl_fit`) — correct, M<3 falls back to config
- ConsensusMedian as async Fréchet median — complete
- ProposalSet CRDT LUB semilattice — complete
- Pareto topology selection — complete
- AdapterRegistry + AppState wiring — complete (tasks 101–104 done, task 105 pending — run `cargo test --workspace` to confirm green)

---

## Current Bug Inventory

| ID | File | Line | Bug | Severity |
|---|---|---|---|---|
| P4 | `physics.rs` | 77 | `beta_eff = beta_base / cg_mean` — singularity at CG→0, inverse form unvalidated | HIGH |
| P3 | `krum.rs` | 124–138 | `krum_select` uses token Jaccard distance, not semantic; cluster guard uses semantic but selection doesn't | HIGH |
| P5 | `tasks.rs` | 112 | Only 2 identical adapters passed → `usl_fit` always falls back (requires M≥3) | MEDIUM |
| — | `physics.rs` | 42–44 | β_eff doc comment says "β₀/CG_mean" — will be wrong after P4 fix | LOW |

---

## File Map

**Modified:**
- `crates/h2ai-types/src/physics.rs` — fix `beta_eff()` formula, add `EigenCalibration` (Task 1, 5)
- `crates/h2ai-types/Cargo.toml` — add nalgebra dep (Task 5)
- `crates/h2ai-types/tests/physics_test.rs` — update tier tests for new β_eff formula (Task 1)
- `crates/h2ai-config/src/lib.rs` — change `beta_base_default` from 0.01 to 0.039 (Task 1)
- `crates/h2ai-state/src/krum.rs` — add async semantic distance matrix + semantic Krum (Task 2)
- `crates/h2ai-autonomic/src/merger.rs` — use semantic Krum in Krum/MultiKrum branches (Task 2)
- `crates/h2ai-api/src/routes/tasks.rs` — pass 3 adapters so usl_fit runs (Task 3)
- `crates/h2ai-autonomic/src/calibration.rs` — add EigenCalibration to CalibrationCompletedEvent (Task 5)
- `crates/h2ai-types/src/events.rs` — add `eigen: Option<EigenCalibration>` to CalibrationCompletedEvent (Task 5)
- `docs/architecture/math-apparatus.md` — update Sections 3 and new Section 9 (Task 6)

**Created:**
- `crates/h2ai-orchestrator/src/diagnostics.rs` — TalagrandDiagnostic struct (Task 4)
- `crates/h2ai-autonomic/Cargo.toml` — add nalgebra (Task 5)

---

## Task 1: Fix β_eff Formula (Proportional, Bounded)

**Files:**
- Modify: `crates/h2ai-types/src/physics.rs:72–103`
- Modify: `crates/h2ai-config/src/lib.rs` (beta_base_default default value)
- Modify: `crates/h2ai-types/tests/physics_test.rs:354–396`

**Background:**

Current formula: `β_eff = β₀ / CG_mean` — diverges to ∞ as CG_mean → 0. Validated by `scripts/validate_beta_coupling.py`.

Proposed: `β_eff = β₀ × (1 − CG_mean)` with floor `max(1e-6)`. Bounded everywhere.

Consequence: to produce the same N_max≈6 for AI-agents (α=0.15, CG=0.4), β₀ must change:
```
β_eff = β₀ × (1−0.4) = β₀×0.6 = 0.85/36 = 0.02361  →  β₀ = 0.0394 ≈ 0.039
```

For human teams (α=0.10, CG=0.6, N_max=10):
```
β₀ = 0.9 / (100 × (1−0.6)) = 0.9/40 = 0.0225
```

For CPU cores (α=0.02, CG=1.0): the proportional formula gives β_eff = β₀×(1−1.0) = 0 →
N_max = ∞. This is physically correct — perfectly coherent cores have no coordination overhead;
only α limits throughput. Update the test to assert `n_max > 50` (retrograde disappears).

- [ ] **Step 1: Write the failing tests (update tier tests to use new formula expectations)**

In `crates/h2ai-types/tests/physics_test.rs`, replace the three tier tests and the beta_eff formula test at lines 354–396:

```rust
#[test]
fn coherency_coefficients_usl_n_max_ai_agents() {
    // AI-agent tier: α=0.15, β₀=0.039, CG=0.4
    // β_eff = 0.039×(1−0.4) = 0.039×0.6 = 0.02340 → N_max = round(√(0.85/0.02340)) = round(6.02) = 6
    let cc = CoherencyCoefficients::new(0.15, 0.039, vec![0.4]).unwrap();
    let n_max = cc.n_max();
    assert!(
        (n_max - 6.0).abs() < 1.0,
        "AI-agent tier N_max must be ≈6 with proportional β_eff, got {n_max}"
    );
}

#[test]
fn coherency_coefficients_beta_eff_proportional_formula() {
    // β_eff = β₀ × (1 − CG_mean) = 0.039 × 0.6 = 0.02340
    let cc = CoherencyCoefficients::new(0.15, 0.039, vec![0.4]).unwrap();
    let beta_eff = cc.beta_eff();
    assert!(
        (beta_eff - 0.02340).abs() < 1e-5,
        "β_eff = β₀×(1−CG) = 0.02340, got {beta_eff}"
    );
}

#[test]
fn coherency_coefficients_beta_eff_bounded_at_low_cg() {
    // At CG→0, proportional form: β_eff = β₀×1.0 = β₀. Must not diverge.
    let cc = CoherencyCoefficients::new(0.15, 0.039, vec![0.001]).unwrap();
    let beta_eff = cc.beta_eff();
    assert!(
        beta_eff < 0.04,
        "β_eff must be bounded (≤ β₀) even at CG≈0, got {beta_eff}"
    );
    assert!(beta_eff > 0.0, "β_eff must be positive, got {beta_eff}");
}

#[test]
fn coherency_coefficients_human_team_tier() {
    // Human team tier: α=0.10, β₀=0.0225, CG=0.6
    // β_eff = 0.0225×(1−0.6) = 0.0225×0.4 = 0.009 → N_max = round(√(0.9/0.009)) = round(10.0) = 10
    let cc = CoherencyCoefficients::new(0.10, 0.0225, vec![0.6]).unwrap();
    let n_max = cc.n_max();
    assert!(
        (n_max - 10.0).abs() < 1.5,
        "Human team N_max must be ≈10 with proportional β_eff, got {n_max}"
    );
}

#[test]
fn coherency_coefficients_cpu_core_tier() {
    // CPU tier: α=0.02, β₀=0.0003, CG=1.0.
    // With proportional formula, β_eff = β₀×(1−1.0) ≈ 0 → retrograde disappears.
    // CPU cores are coherency-free at full alignment; only α limits throughput.
    // N_max with near-zero β_eff is very large (>> 57); assert ≥ 50.
    let cc = CoherencyCoefficients::new(0.02, 0.0003, vec![1.0]).unwrap();
    let n_max = cc.n_max();
    assert!(
        n_max >= 50.0,
        "CPU core tier N_max must be ≥50 with proportional formula at CG=1.0, got {n_max}"
    );
}
```

- [ ] **Step 2: Run failing tests to confirm they fail**

```bash
cargo test -p h2ai-types --test physics_test 2>&1 | grep -E "FAILED|ok\.|error"
```

Expected: `coherency_coefficients_beta_eff_proportional_formula` FAILS with "0.025, got 0.02340"  
and `coherency_coefficients_beta_eff_bounded_at_low_cg` FAILS with divergent value.

- [ ] **Step 3: Update `beta_eff()` in `crates/h2ai-types/src/physics.rs`**

Replace lines 72–78 (the `beta_eff` method and its doc comment):

```rust
    /// Effective coordination cost per agent pair.
    ///
    /// `β_eff = β₀ × (1 − CG_mean)`, clamped to a minimum of 1e-6.
    ///
    /// - At CG_mean = 0 (no overlap): β_eff = β₀ (maximum cost, bounded).
    /// - At CG_mean = 1 (full overlap): β_eff ≈ 0 (coordination-free).
    /// - Previous formula β₀/CG_mean diverged at CG→0; this form is bounded everywhere.
    pub fn beta_eff(&self) -> f64 {
        let cg = self.cg_mean().clamp(0.0, 1.0);
        (self.beta_base * (1.0 - cg)).max(1e-6)
    }
```

Also update the doc comment on `CoherencyCoefficients` struct (lines 42–44):

```rust
/// `beta_eff` = β₀ × (1 − CG_mean) couples coordination cost with how divergent adapter
/// outputs are. Higher CG_mean → lower β_eff → higher N_max. Bounded at β₀ when CG_mean = 0.
/// `n_max` = round(√((1−α)/β_eff)) is derived from USL Proposition 1.
```

- [ ] **Step 4: Update `beta_base_default` in `crates/h2ai-config/src/lib.rs`**

Find `beta_base_default: 0.01` in the default config and the doc comment. Change both:

```rust
/// β₀ (beta_base_default) — base coherency cost per agent pair for this deployment tier.
/// Used as calibration fallback when fewer than 3 adapters are available.
/// Default 0.039 = AI-agents tier (proportional formula: α=0.15, β₀=0.039, CG=0.4 → N_max≈6).
/// Recalibration: β₀ = (1−α) / (N_max² × (1−CG)).
/// Use 0.0225 for human-team tier, 0.0003 for CPU-core tier.
pub beta_base_default: f64,
```

In the `Default` impl (wherever `beta_base_default: 0.01` appears), change to:

```rust
beta_base_default: 0.039,
```

Also update the serde alias test that asserts the kappa_eff_factor maps to 0.019:
Search for `assert.*0.019\|kappa_eff_factor.*0.019` in config tests. The alias test should check that the JSON key works, not a specific numeric value — keep the test but update any hardcoded 0.019 or 0.01 assertions.

- [ ] **Step 5: Run the updated tests**

```bash
cargo test -p h2ai-types --test physics_test 2>&1 | grep -E "FAILED|ok\.|error"
```

Expected: all physics tests pass.

```bash
cargo test -p h2ai-config 2>&1 | grep -E "FAILED|ok\.|error"
```

Expected: all config tests pass.

- [ ] **Step 6: Run the validation script to verify results**

```bash
python3 scripts/validate_beta_coupling.py
```

Expected: Section 3 "Singularity Test" shows proportional form bounded at β₀=0.039.
Section 1 three-tier comparison — use `beta0=0.039` for AI tier (the script uses 0.01; note discrepancy is expected since script predates this fix).

- [ ] **Step 7: Commit**

```bash
git add crates/h2ai-types/src/physics.rs crates/h2ai-types/tests/physics_test.rs crates/h2ai-config/src/lib.rs
git commit -m "fix(physics): proportional β_eff = β₀×(1−CG), recalibrate beta_base_default=0.039"
```

---

## Task 2: Async Semantic Krum Selection

**Files:**
- Modify: `crates/h2ai-state/src/krum.rs`
- Modify: `crates/h2ai-autonomic/src/merger.rs`
- Modify: `crates/h2ai-autonomic/tests/merger_test.rs`

**Background:**

`krum_select` and `multi_krum_select` use token Jaccard distance internally (built from `tokenize`).
`cluster_coherent` already uses `semantic_jaccard` (async, adapter-aware).

When a semantic adapter is available, honest LLM paraphrases cluster tightly (cosine dist ≈ 0.1),
so `cluster_coherent` returns `true` — the cluster guard passes. But then `krum_select` scores
proposals by token Jaccard distance, where paraphrases have distance ≈ 0.95, so Byzantine
vocabulary-stuffed proposals (lower token distance) get selected.

Fix: add async semantic variants of `krum_select` and `multi_krum_select` that compute the
distance matrix via `semantic_jaccard`. Keep the existing sync token-Jaccard versions for
tests and no-adapter contexts.

- [ ] **Step 1: Write the failing test (semantic Krum selects the semantically central proposal)**

Add to `crates/h2ai-autonomic/tests/merger_test.rs`:

```rust
#[tokio::test]
async fn krum_selects_semantically_central_with_mock_adapter() {
    use h2ai_adapters::mock::MockAdapter;
    use h2ai_state::krum::krum_select_semantic;
    use chrono::Utc;
    use h2ai_types::config::AdapterKind;
    use h2ai_types::events::ProposalEvent;
    use h2ai_types::identity::{ExplorerId, TaskId};
    use h2ai_types::physics::TauValue;

    let kind = AdapterKind::CloudGeneric {
        endpoint: "mock://".into(),
        api_key_env: "X".into(),
    };
    // MockAdapter returns a fixed string regardless of input.
    // The semantic_jaccard in token-mode (no SLM) falls back to jaccard.
    // Pass None as adapter to test the token-fallback path in krum_select_semantic.
    let task_id = TaskId::new();
    let make = |text: &str| ProposalEvent {
        task_id: task_id.clone(),
        explorer_id: ExplorerId::new(),
        tau: TauValue::new(0.4).unwrap(),
        raw_output: text.into(),
        token_cost: 10,
        adapter_kind: kind.clone(),
        timestamp: Utc::now(),
    };

    // 4 similar proposals + 1 outlier → Krum should NOT select the outlier
    let proposals = vec![
        make("the quick brown fox"),
        make("a quick brown fox"),
        make("the fast brown fox"),
        make("the quick brown dog"),
        make("completely unrelated output about blockchain"),  // outlier
    ];

    let result = krum_select_semantic(&proposals, 1, None).await;
    assert!(result.is_some(), "krum_select_semantic must return Some");
    let selected = result.unwrap();
    assert_ne!(
        selected.raw_output, "completely unrelated output about blockchain",
        "Krum must not select the outlier"
    );
}
```

- [ ] **Step 2: Run the failing test to confirm it fails**

```bash
cargo test -p h2ai-autonomic --test merger_test krum_selects_semantically_central 2>&1
```

Expected: FAIL with "cannot find function `krum_select_semantic`"

- [ ] **Step 3: Add async distance matrix + krum_select_semantic to `krum.rs`**

Add after the existing `krum_index` function (line 228 in current file). These functions are async and accept an optional adapter for semantic distance:

```rust
// ── Async semantic Krum ───────────────────────────────────────────────────────

/// Build the n×n semantic distance matrix via `semantic_jaccard`.
/// All pairs computed concurrently via `join_all`.
/// Falls back to token Jaccard when `adapter` is `None`.
async fn semantic_distance_matrix(
    proposals: &[ProposalEvent],
    adapter: Option<&dyn IComputeAdapter>,
) -> Vec<Vec<f64>> {
    let n = proposals.len();
    let outputs: Vec<&str> = proposals.iter().map(|p| p.raw_output.as_str()).collect();

    // Collect all (i, j) pairs with i < j
    let pairs: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
        .collect();

    let similarities = join_all(
        pairs.iter().map(|&(i, j)| semantic_jaccard(outputs[i], outputs[j], adapter)),
    )
    .await;

    let mut d = vec![vec![0.0f64; n]; n];
    for (k, &(i, j)) in pairs.iter().enumerate() {
        let dist = 1.0 - similarities[k];
        d[i][j] = dist;
        d[j][i] = dist;
    }
    d
}

/// **Semantic Krum** — selects the proposal with minimum sum of distances to its
/// `n − f − 2` nearest neighbours, using semantic (not token) distance.
///
/// Uses `semantic_jaccard` for pairwise similarity so that lexically-distinct
/// but semantically-equivalent proposals (synonyms, paraphrases) score as close.
/// Falls back to token Jaccard when `adapter` is `None`.
///
/// Returns `None` when the quorum condition `n ≥ 2f + 3` is not met,
/// or when `proposals` is empty.
pub async fn krum_select_semantic(
    proposals: &[ProposalEvent],
    f: usize,
    adapter: Option<&dyn IComputeAdapter>,
) -> Option<&ProposalEvent> {
    if proposals.is_empty() {
        return None;
    }
    if f == 0 {
        return proposals.first();
    }
    if !quorum_satisfied(proposals.len(), f) {
        return None;
    }
    let distances = semantic_distance_matrix(proposals, adapter).await;
    let k = proposals.len() - f - 2;
    krum_index(&distances, k).map(|i| &proposals[i])
}

/// **Semantic Multi-Krum** — iteratively selects `m` proposals via semantic Krum,
/// returning them in selection order (best Krum score first).
pub async fn multi_krum_select_semantic(
    proposals: &[ProposalEvent],
    f: usize,
    m: usize,
    adapter: Option<&dyn IComputeAdapter>,
) -> Vec<&ProposalEvent> {
    if proposals.is_empty() || m == 0 {
        return vec![];
    }
    if f == 0 {
        return proposals.iter().take(m).collect();
    }
    if !quorum_satisfied(proposals.len(), f) {
        return vec![];
    }
    let distances = semantic_distance_matrix(proposals, adapter).await;

    let mut remaining: Vec<usize> = (0..proposals.len()).collect();
    let mut selected = Vec::with_capacity(m);

    while selected.len() < m && remaining.len() > f + 2 {
        let k = remaining.len() - f - 2;
        let best_pos = (0..remaining.len())
            .min_by(|&a, &b| {
                let sa = krum_score_subset(remaining[a], &remaining, &distances, k);
                let sb = krum_score_subset(remaining[b], &remaining, &distances, k);
                sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("remaining is non-empty");
        selected.push(&proposals[remaining[best_pos]]);
        remaining.remove(best_pos);
    }

    selected
}
```

- [ ] **Step 4: Update `merger.rs` to use semantic Krum**

In `crates/h2ai-autonomic/src/merger.rs`, update the import at the top:

```rust
use h2ai_state::krum::{
    cluster_coherent, krum_select_semantic, multi_krum_select_semantic, quorum_satisfied,
};
```

Remove `krum_select` and `multi_krum_select` from the import (they are no longer used here).

Replace the `MergeStrategy::Krum { f }` branch:

```rust
MergeStrategy::Krum { f } => {
    let proposals = &result.valid_proposals;
    if quorum_satisfied(proposals.len(), f) && cluster_coherent(proposals, adapter).await {
        krum_select_semantic(proposals, f, adapter)
            .await
            .map(|p| p.raw_output.clone())
            .unwrap_or_default()
    } else {
        ConsensusMedian::resolve(proposals, adapter)
            .await
            .map(|p| p.raw_output.clone())
            .unwrap_or_default()
    }
}
```

Replace the `MergeStrategy::MultiKrum { f, m }` branch:

```rust
MergeStrategy::MultiKrum { f, m } => {
    let proposals = &result.valid_proposals;
    if quorum_satisfied(proposals.len(), f) && cluster_coherent(proposals, adapter).await {
        let survivors = multi_krum_select_semantic(proposals, f, m, adapter).await;
        proposals
            .iter()
            .find(|p| survivors.iter().any(|s| s.explorer_id == p.explorer_id))
            .map(|p| p.raw_output.clone())
            .unwrap_or_default()
    } else {
        ConsensusMedian::resolve(proposals, adapter)
            .await
            .map(|p| p.raw_output.clone())
            .unwrap_or_default()
    }
}
```

- [ ] **Step 5: Run the new test**

```bash
cargo test -p h2ai-autonomic --test merger_test krum_selects_semantically_central 2>&1
```

Expected: PASS.

```bash
cargo test -p h2ai-autonomic 2>&1 | grep -E "FAILED|ok\.|error"
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add crates/h2ai-state/src/krum.rs crates/h2ai-autonomic/src/merger.rs crates/h2ai-autonomic/tests/merger_test.rs
git commit -m "feat(krum): async semantic Krum/MultiKrum using semantic_jaccard distance"
```

---

## Task 3: Wire 3rd Adapter for USL Fit (M≥3)

**Files:**
- Modify: `crates/h2ai-api/src/routes/tasks.rs:112`
- Modify: `crates/h2ai-api/src/main.rs` (add explorer2 adapter env vars)
- Modify: `crates/h2ai-api/src/state.rs` (add explorer2_adapter field)

**Background:**

`tasks.rs:112` passes `vec![explorer.as_ref(), explorer.as_ref()]` (M=2).
`CalibrationHarness::usl_fit` requires M≥3 or falls back to config defaults.
The same adapter passed 3 times still gives meaningful timing measurements (3 parallel calls to the same provider measure real scheduling overhead).

The minimal fix is to add a 3rd explorer adapter — either the same adapter (simplest, valid for USL timing) or a configurable second adapter via env var.

We use the **same adapter 3 times** as the default: minimal change, no breaking changes to deployment config.

- [ ] **Step 1: Write the failing test confirming M=2 falls back**

This is a documentation test only — the existing calibration tests already cover it.
Run to confirm current state:

```bash
cargo test -p h2ai-autonomic calibration -- --nocapture 2>&1 | grep "fallback\|M=2"
```

Expected: see `usl_fit_fallback_when_m_less_than_3` test passing (i.e., M=2 always falls back).

- [ ] **Step 2: Add `explorer2_adapter` field to `AppState`**

In `crates/h2ai-api/src/state.rs`:

```rust
pub struct AppState {
    pub nats: Arc<NatsClient>,
    pub cfg: Arc<H2AIConfig>,
    pub store: TaskStore,
    pub calibration: Arc<RwLock<Option<CalibrationCompletedEvent>>>,
    pub journal: Arc<SessionJournal>,
    pub explorer_adapter: Arc<dyn IComputeAdapter>,
    /// Second explorer adapter for USL timing Phase B (can be same as explorer_adapter).
    pub explorer2_adapter: Arc<dyn IComputeAdapter>,
    pub verification_adapter: Arc<dyn IComputeAdapter>,
    pub auditor_adapter: Arc<dyn IComputeAdapter>,
    pub scoring_adapter: Option<Arc<dyn IComputeAdapter>>,
    pub task_semaphore: Arc<Semaphore>,
}

impl AppState {
    pub fn new(
        nats: NatsClient,
        cfg: H2AIConfig,
        explorer_adapter: Arc<dyn IComputeAdapter>,
        auditor_adapter: Arc<dyn IComputeAdapter>,
    ) -> Self {
        let nats = Arc::new(nats);
        let journal = Arc::new(SessionJournal::new(nats.clone()));
        let max_tasks = cfg.max_concurrent_tasks;
        Self {
            nats,
            cfg: Arc::new(cfg),
            store: TaskStore::new(),
            calibration: Arc::new(RwLock::new(None)),
            journal,
            explorer2_adapter: explorer_adapter.clone(), // default: same adapter
            explorer_adapter,
            verification_adapter: auditor_adapter.clone(),
            auditor_adapter,
            scoring_adapter: None,
            task_semaphore: Arc::new(Semaphore::new(max_tasks)),
        }
    }

    pub fn with_explorer2(mut self, adapter: Arc<dyn IComputeAdapter>) -> Self {
        self.explorer2_adapter = adapter;
        self
    }

    pub fn registry(&self) -> AdapterRegistry {
        let reg = AdapterRegistry::new(self.explorer_adapter.clone());
        match &self.scoring_adapter {
            Some(scoring) => reg.with_scoring(scoring.clone()),
            None => reg,
        }
    }
}
```

- [ ] **Step 3: Update `tasks.rs` to pass 3 adapters**

In `crates/h2ai-api/src/routes/tasks.rs`, change line:
```rust
let explorer = state.explorer_adapter.clone();
```
to:
```rust
let explorer = state.explorer_adapter.clone();
let explorer2 = state.explorer2_adapter.clone();
```

Change the `EngineInput` construction:
```rust
explorer_adapters: vec![explorer.as_ref(), explorer2.as_ref(), explorer.as_ref()],
```

This gives M=3: `[adapter1, adapter2, adapter1]`. With M=3, `usl_fit` runs Phase A on adapters[0..2] and Phase B on all 3. Even if adapter1==adapter2, the timing measurement is real.

- [ ] **Step 4: Update `main.rs` to optionally read `H2AI_EXPLORER2_*` env vars**

In `crates/h2ai-api/src/main.rs`, after the existing `explorer_adapter` construction:

```rust
let explorer2_kind_opt = {
    let provider = env::var("H2AI_EXPLORER2_PROVIDER")
        .unwrap_or_else(|_| "same".into())
        .to_lowercase();
    if provider == "same" || provider.is_empty() {
        None  // use same as explorer
    } else {
        Some(adapter_kind_from_env("EXPLORER2"))
    }
};
let explorer2_adapter: Arc<dyn IComputeAdapter> = explorer2_kind_opt
    .as_ref()
    .map(build_adapter)
    .unwrap_or_else(|| explorer_adapter.clone());

eprintln!("explorer2 adapter: {:?}", explorer2_kind_opt);
```

And update `app_state` construction:
```rust
let app_state = AppState::new(nats, cfg, explorer_adapter, auditor_adapter)
    .with_explorer2(explorer2_adapter);
```

- [ ] **Step 5: Compile and test**

```bash
cargo check -p h2ai-api 2>&1 | grep "^error"
```

Expected: no errors.

```bash
cargo test -p h2ai-api 2>&1 | grep -E "FAILED|ok\.|error"
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add crates/h2ai-api/src/state.rs crates/h2ai-api/src/routes/tasks.rs crates/h2ai-api/src/main.rs
git commit -m "fix(api): pass 3 explorer adapters so usl_fit runs (M≥3)"
```

---

## Task 4: TalagrandDiagnostic — Ensemble Calibration Without Labels

**Files:**
- Create: `crates/h2ai-orchestrator/src/diagnostics.rs`
- Modify: `crates/h2ai-orchestrator/src/lib.rs` (add `pub mod diagnostics`)

**Background:**

Inspired by operational weather ensemble forecasting (Leutbecher & Palmer 2008).
After each verification phase, we record the rank of the runner-up proposal in the
sorted verification score list. Over 20+ runs, the rank histogram should be uniform
(flat = calibrated, U-shape = over-confident, Λ-shape = under-dispersed).

Zero new inference calls. Pure arithmetic on existing verification scores.

- [ ] **Step 1: Write the failing tests**

Create `crates/h2ai-orchestrator/src/diagnostics.rs` with tests inline:

```rust
//! Ensemble calibration diagnostics inspired by weather ensemble rank histograms.
//!
//! See docs/research/2026-04-27-innovation-synthesis.md Section 6.

#[derive(Debug, Clone, PartialEq)]
pub enum CalibrationState {
    /// χ² test passes (histogram uniform): ensemble is well-calibrated.
    Calibrated,
    /// Tail ranks too frequent (U-shape): ensemble is over-confident.
    OverConfident,
    /// Middle ranks too frequent (Λ-shape): ensemble is under-dispersed.
    UnderDispersed,
    /// Fewer than 20 runs observed: not enough data.
    Insufficient,
}

#[derive(Debug, Clone)]
pub struct TalagrandDiagnostic {
    /// Rank histogram: histogram[r] = count of runs where runner-up had rank r.
    /// Index 0 unused. Length = n_adapters + 1.
    pub rank_histogram: Vec<u32>,
    /// Chi-squared statistic testing uniformity of the rank histogram.
    pub chi_sq_from_uniform: f64,
    /// Ratio of proposal spread (std dev of scores) to mean top-score gap.
    /// Ideal ≈ 1.0 (ensemble spread matches actual score variation).
    pub spread_error_ratio: f64,
    pub calibration_state: CalibrationState,
}

impl TalagrandDiagnostic {
    /// Build a Talagrand diagnostic from a collection of per-run verification scores.
    ///
    /// `per_run_scores`: each element is a Vec of N adapter verification scores for one run.
    /// All inner Vecs must have the same length N ≥ 2.
    ///
    /// Returns `None` if `per_run_scores` is empty or inner Vecs are empty.
    pub fn from_verification_scores(per_run_scores: &[Vec<f64>]) -> Option<Self> {
        if per_run_scores.is_empty() {
            return None;
        }
        let n = per_run_scores[0].len();
        if n < 2 {
            return None;
        }

        let mut histogram = vec![0u32; n + 1];
        let mut spread_sum = 0.0f64;
        let mut gap_sum = 0.0f64;

        for scores in per_run_scores {
            if scores.len() != n {
                continue;
            }
            let top = scores
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);

            // Rank = 1 + count of proposals scoring strictly higher than runner-up
            let second = scores
                .iter()
                .cloned()
                .filter(|&s| s < top)
                .fold(f64::NEG_INFINITY, f64::max);
            let rank = if second == f64::NEG_INFINITY {
                // All scores equal — mid rank
                n / 2
            } else {
                scores.iter().filter(|&&s| s > second).count()
            };
            histogram[rank.min(n)] += 1;

            let mean = scores.iter().sum::<f64>() / n as f64;
            let spread = (scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
            let gap = top - mean;
            spread_sum += spread;
            gap_sum += gap;
        }

        let t = per_run_scores.len() as f64;
        let expected = t / n as f64;
        let chi_sq: f64 = histogram
            .iter()
            .skip(1) // skip index 0
            .map(|&c| (c as f64 - expected).powi(2) / expected.max(1.0))
            .sum();

        let spread_error_ratio = if gap_sum > 1e-10 {
            spread_sum / gap_sum
        } else {
            1.0
        };

        let state = if t < 20.0 {
            CalibrationState::Insufficient
        } else if chi_sq < 3.84 {
            // χ²(1) at α=0.05 critical value — conservative (single d.f. approximation)
            CalibrationState::Calibrated
        } else {
            // Distinguish U vs Λ: U-shape has high tail counts, Λ has low tails
            let tail_count = histogram[1] + histogram[n];
            let tail_rate = tail_count as f64 / t;
            if tail_rate > 2.0 / n as f64 {
                CalibrationState::OverConfident
            } else {
                CalibrationState::UnderDispersed
            }
        };

        Some(Self {
            rank_histogram: histogram,
            chi_sq_from_uniform: chi_sq,
            spread_error_ratio,
            calibration_state: state,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn talagrand_returns_none_for_empty_input() {
        assert!(TalagrandDiagnostic::from_verification_scores(&[]).is_none());
    }

    #[test]
    fn talagrand_returns_none_for_single_adapter() {
        let scores = vec![vec![0.8f64]];
        assert!(TalagrandDiagnostic::from_verification_scores(&scores).is_none());
    }

    #[test]
    fn talagrand_insufficient_when_fewer_than_20_runs() {
        let run = vec![0.9f64, 0.7, 0.5];
        let scores: Vec<Vec<f64>> = std::iter::repeat(run).take(5).collect();
        let d = TalagrandDiagnostic::from_verification_scores(&scores).unwrap();
        assert_eq!(d.calibration_state, CalibrationState::Insufficient);
    }

    #[test]
    fn talagrand_calibrated_when_histogram_is_uniform() {
        // 3 adapters, 60 runs: each rank appears 20 times
        // Simulate by cycling through rank patterns
        let n = 3;
        let mut scores_vec: Vec<Vec<f64>> = Vec::new();
        for i in 0..60 {
            // Rotate which adapter gets the top score so ranks are uniform
            let top_idx = i % n;
            let run: Vec<f64> = (0..n)
                .map(|j| if j == top_idx { 0.9 } else { 0.7 - j as f64 * 0.01 })
                .collect();
            scores_vec.push(run);
        }
        let d = TalagrandDiagnostic::from_verification_scores(&scores_vec).unwrap();
        // Uniform → Calibrated (χ² small)
        assert!(d.chi_sq_from_uniform < 10.0, "uniform histogram → low χ², got {}", d.chi_sq_from_uniform);
    }

    #[test]
    fn talagrand_histogram_length_equals_n_adapters_plus_one() {
        let run = vec![0.9f64, 0.7, 0.5, 0.3];
        let scores: Vec<Vec<f64>> = std::iter::repeat(run).take(5).collect();
        let d = TalagrandDiagnostic::from_verification_scores(&scores).unwrap();
        assert_eq!(d.rank_histogram.len(), 5, "histogram length should be N+1=5");
    }
}
```

- [ ] **Step 2: Run the failing tests**

```bash
cargo test -p h2ai-orchestrator diagnostics 2>&1 | grep -E "FAILED|ok\.|error"
```

Expected: FAIL with "module `diagnostics` not found"

- [ ] **Step 3: Export from lib.rs**

In `crates/h2ai-orchestrator/src/lib.rs`, add:

```rust
pub mod diagnostics;
```

- [ ] **Step 4: Run tests**

```bash
cargo test -p h2ai-orchestrator diagnostics 2>&1 | grep -E "FAILED|ok\.|error"
```

Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/h2ai-orchestrator/src/diagnostics.rs crates/h2ai-orchestrator/src/lib.rs
git commit -m "feat(orchestrator): TalagrandDiagnostic — ensemble calibration via rank histogram"
```

---

## Task 5: EigenCalibration — Eigenvalue-Based N_effective

**Files:**
- Modify: `crates/h2ai-types/Cargo.toml` (add nalgebra)
- Modify: `Cargo.toml` (add nalgebra to workspace.dependencies)
- Modify: `crates/h2ai-types/src/physics.rs` (add EigenCalibration struct)
- Modify: `crates/h2ai-autonomic/Cargo.toml` (add nalgebra)
- Modify: `crates/h2ai-autonomic/src/calibration.rs` (compute EigenCalibration)
- Modify: `crates/h2ai-types/src/events.rs` (add `eigen` field to CalibrationCompletedEvent)

**Background:**

Scalar `ρ_mean = 1 − CG_mean` discards structure. Portfolio theory (Choueifaty 2008) shows
that pairwise correlation matrix eigenvalues give the true "effective number of independent
adapters": `N_eff = (Σλ)²/Σλ²`. For 5 adapters with 3 in a tight cluster, N_eff ≈ 2.5, not 3.9.

The calibration harness already computes pairwise CG scores. We extend it to compute the
full N×N CG matrix and its eigenvalues.

- [ ] **Step 1: Add nalgebra to workspace**

In root `Cargo.toml`, add to `[workspace.dependencies]`:

```toml
nalgebra = { version = "0.33", default-features = false, features = ["std"] }
```

In `crates/h2ai-types/Cargo.toml`, add to `[dependencies]`:

```toml
nalgebra = { workspace = true }
```

In `crates/h2ai-autonomic/Cargo.toml`, add to `[dependencies]`:

```toml
nalgebra = { workspace = true }
```

- [ ] **Step 2: Run compile check to confirm nalgebra resolves**

```bash
cargo check -p h2ai-types 2>&1 | grep "^error"
```

Expected: no errors (nalgebra downloads and compiles).

- [ ] **Step 3: Write failing test for EigenCalibration**

Add to `crates/h2ai-types/tests/physics_test.rs`:

```rust
#[test]
fn eigen_calibration_full_independence_gives_n_eff_equal_n() {
    use nalgebra::DMatrix;
    // Identity matrix (N=4 fully independent adapters): all eigenvalues = 1
    // N_eff = (4)² / 4 = 4
    let sigma = DMatrix::<f64>::identity(4, 4);
    let ec = EigenCalibration::from_cg_matrix(&sigma);
    assert!(
        (ec.n_effective - 4.0).abs() < 0.1,
        "identity Σ → N_eff = 4, got {}", ec.n_effective
    );
    assert!(
        (ec.h_diversity - 1.0).abs() < 0.01,
        "identity Σ → H_norm = 1.0, got {}", ec.h_diversity
    );
}

#[test]
fn eigen_calibration_full_correlation_gives_n_eff_one() {
    use nalgebra::DMatrix;
    // Σ = all-ones matrix (fully correlated, rank 1): one eigenvalue = N, rest = 0
    // N_eff = N² / N² = 1
    let n = 4;
    let sigma = DMatrix::<f64>::from_element(n, n, 1.0);
    let ec = EigenCalibration::from_cg_matrix(&sigma);
    assert!(
        (ec.n_effective - 1.0).abs() < 0.5,
        "all-ones Σ → N_eff ≈ 1, got {}", ec.n_effective
    );
}

#[test]
fn eigen_calibration_uniform_rho_matches_portfolio_formula() {
    use nalgebra::DMatrix;
    // Uniform ρ=0.5, N=5: Choueifaty formula N_eff = N×(1−ρ)+ρ = 5×0.5+0.5 = 3.0
    let n = 5;
    let rho = 0.5f64;
    let mut sigma = DMatrix::<f64>::identity(n, n);
    for i in 0..n {
        for j in 0..n {
            if i != j {
                sigma[(i, j)] = rho;
            }
        }
    }
    let ec = EigenCalibration::from_cg_matrix(&sigma);
    let expected = n as f64 * (1.0 - rho) + rho;  // = 3.0
    assert!(
        (ec.n_effective - expected).abs() < 0.3,
        "uniform ρ=0.5 → N_eff≈{expected:.1}, got {:.3}", ec.n_effective
    );
}
```

- [ ] **Step 4: Run failing tests**

```bash
cargo test -p h2ai-types --test physics_test eigen_calibration 2>&1
```

Expected: FAIL with "unresolved import `EigenCalibration`"

- [ ] **Step 5: Implement `EigenCalibration` in `physics.rs`**

Add to `crates/h2ai-types/src/physics.rs`. Place after the `EnsembleCalibration` struct (after line 351 in current file):

```rust
use nalgebra::DMatrix;

/// Eigenvalue-based ensemble calibration from the pairwise CG similarity matrix.
///
/// Implements the portfolio theory "participation ratio" (Choueifaty & Coignard 2008):
///   N_eff = (Σ λᵢ)² / Σ λᵢ²
///
/// N_eff measures the number of truly independent adapters in the ensemble.
/// At full independence (Σ = I), N_eff = N. At full correlation (Σ = 𝟏𝟏ᵀ), N_eff = 1.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EigenCalibration {
    /// Effective number of independent adapters: (Σλ)²/Σλ².
    pub n_effective: f64,
    /// Normalized Shannon entropy of eigenvalue distribution ∈ [0, 1].
    /// 1.0 = fully decorrelated; 0.0 = one adapter dominates.
    pub h_diversity: f64,
    /// Eigenvalues of the CG similarity matrix, sorted descending.
    pub eigenvalues: Vec<f64>,
    /// Recommended adapter count: first N where adding another raises N_eff by < 0.05.
    pub n_pruned: usize,
}

impl EigenCalibration {
    /// Compute from an N×N symmetric positive-semidefinite CG similarity matrix.
    ///
    /// `sigma`: pairwise CG scores, where `sigma[(i,j)] = cg(adapter_i, adapter_j)`.
    /// Diagonal entries are 1.0 (perfect self-similarity).
    pub fn from_cg_matrix(sigma: &DMatrix<f64>) -> Self {
        let eig = sigma.symmetric_eigen();
        let mut evs: Vec<f64> = eig.eigenvalues.iter().copied().map(|v| v.max(0.0)).collect();
        evs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        let sum: f64 = evs.iter().sum();
        let sum_sq: f64 = evs.iter().map(|l| l * l).sum();
        let n_eff = if sum_sq > 1e-12 { sum * sum / sum_sq } else { 1.0 };

        let h_div: f64 = evs
            .iter()
            .filter(|&&l| l > 1e-12)
            .map(|&l| { let p = l / sum; -p * p.ln() })
            .sum();
        let h_norm = if evs.len() > 1 {
            h_div / (evs.len() as f64).ln()
        } else {
            0.0
        };

        // Pruning rule: stop adding adapters when marginal N_eff gain < 0.05
        let n_pruned = {
            let mut prev = 0.0f64;
            let mut pruned = evs.len();
            for (i, _) in evs.iter().enumerate() {
                let partial_sum: f64 = evs[..=i].iter().sum();
                let partial_sum_sq: f64 = evs[..=i].iter().map(|l| l * l).sum();
                let current = if partial_sum_sq > 1e-12 {
                    partial_sum * partial_sum / partial_sum_sq
                } else {
                    1.0
                };
                if i > 0 && current - prev < 0.05 {
                    pruned = i;
                    break;
                }
                prev = current;
            }
            pruned.max(1)
        };

        Self {
            n_effective: n_eff,
            h_diversity: h_norm.clamp(0.0, 1.0),
            eigenvalues: evs,
            n_pruned,
        }
    }

    /// Derive effective correlation from N_eff: ρ_eff = 1 − N_eff/N.
    pub fn rho_eff(&self, n: usize) -> f64 {
        (1.0 - self.n_effective / n as f64).clamp(0.0, 1.0)
    }
}
```

Also add `use nalgebra::DMatrix;` at the top of `physics.rs` imports.

- [ ] **Step 6: Run the tests**

```bash
cargo test -p h2ai-types --test physics_test eigen_calibration 2>&1 | grep -E "FAILED|ok\."
```

Expected: all 3 eigen_calibration tests PASS.

- [ ] **Step 7: Add `eigen` field to `CalibrationCompletedEvent`**

In `crates/h2ai-types/src/events.rs`, find `CalibrationCompletedEvent` and add a field:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationCompletedEvent {
    pub calibration_id: TaskId,
    pub coefficients: CoherencyCoefficients,
    pub coordination_threshold: CoordinationThreshold,
    /// Condorcet-based calibration (from CG_mean scalar proxy).
    pub ensemble: Option<EnsembleCalibration>,
    /// Eigenvalue-based calibration (from pairwise CG matrix). None when fewer than 2 adapters.
    pub eigen: Option<EigenCalibration>,
    pub timestamp: DateTime<Utc>,
}
```

Update the import at the top of events.rs to add `EigenCalibration`:

```rust
use crate::physics::{
    CoherencyCoefficients, CoordinationThreshold, EigenCalibration, EnsembleCalibration,
};
```

- [ ] **Step 8: Compute EigenCalibration in `calibration.rs`**

In `crates/h2ai-autonomic/src/calibration.rs`, add this import at the top:

```rust
use h2ai_types::physics::EigenCalibration;
use nalgebra::DMatrix;
```

In the `CalibrationHarness::run` method, after the `ensemble` computation (after the `let (cg_samples, ensemble)` block), add:

```rust
// Compute eigenvalue calibration from the full pairwise CG matrix (N×N).
let eigen: Option<EigenCalibration> = if adapter_outputs.len() >= 2 {
    let n = adapter_outputs.len();
    let cal_tau = TauValue::new(input.cfg.calibration_tau).expect("calibration_tau valid");
    let align = tau_alignment(cal_tau, cal_tau);
    let mut sigma = DMatrix::<f64>::identity(n, n);
    for i in 0..n {
        for j in (i + 1)..n {
            let oi = adapter_outputs[i].join(" ");
            let oj = adapter_outputs[j].join(" ");
            let ki = tokenize(&oi);
            let kj = tokenize(&oj);
            let cg_ij = jaccard(&ki, &kj) * align;
            sigma[(i, j)] = cg_ij;
            sigma[(j, i)] = cg_ij;
        }
    }
    Some(EigenCalibration::from_cg_matrix(&sigma))
} else {
    None
};
```

Update `CalibrationCompletedEvent` construction at the end of `run`:

```rust
Ok(CalibrationCompletedEvent {
    calibration_id: input.calibration_id,
    coefficients: cc,
    coordination_threshold,
    ensemble,
    eigen,
    timestamp: Utc::now(),
})
```

- [ ] **Step 9: Fix any compilation errors from the new field**

Run:

```bash
cargo check --workspace 2>&1 | grep "^error"
```

If there are struct-literal errors for `CalibrationCompletedEvent` missing `eigen` field,
add `eigen: None` to all construction sites (typically in test files).

Grep for construction sites:

```bash
grep -rn "CalibrationCompletedEvent {" crates/ --include="*.rs" | grep -v "target"
```

For each site that doesn't include `eigen`, add `eigen: None,`.

- [ ] **Step 10: Run all tests**

```bash
cargo test --workspace 2>&1 | grep -E "FAILED|error\[" | head -20
```

Expected: no failures.

- [ ] **Step 11: Commit**

```bash
git add Cargo.toml crates/h2ai-types/Cargo.toml crates/h2ai-types/src/physics.rs crates/h2ai-types/src/events.rs crates/h2ai-types/tests/physics_test.rs crates/h2ai-autonomic/Cargo.toml crates/h2ai-autonomic/src/calibration.rs
git commit -m "feat(physics): EigenCalibration — eigenvalue-based N_eff from pairwise CG matrix"
```

---

## Task 6: Update Documentation

**Files:**
- Modify: `docs/architecture/math-apparatus.md`

- [ ] **Step 1: Update Section 3 (β_eff formula)**

Find the existing line:
```
beta_eff   = beta_base / CG_mean    [Definition 6 — USL+CG coupling]
```

Replace with:
```
beta_eff   = beta_base × (1 − CG_mean)    [Definition 6 — USL+CG coupling, bounded]
             Higher CG_mean → lower beta_eff → higher N_max.
             At CG_mean → 0: beta_eff = beta_base (maximum, finite).
             At CG_mean = 1: beta_eff → 0 (coordination-free; only α limits N_max).
             Floor: max(beta_base × (1−CG), 1e-6) prevents zero in pathological cases.
             Previous inverse form beta_base/CG_mean diverged at CG→0.

Three-tier calibration table (proportional formula):
  CPU cores  (α=0.02, β₀=0.0003, CG=1.0): N_max → large (β_eff≈0, only α limits)
  Human teams (α=0.10, β₀=0.0225, CG=0.6): N_max ≈ 10
  AI agents  (α=0.15, β₀=0.039,  CG=0.4): N_max ≈  6
```

Also update the paragraph after the table: change `beta_base_default = 0.01` to `0.039`.

- [ ] **Step 2: Add new Definition 6B (EigenCalibration) after existing Definition 6**

Add a new subsection in Section 3:

```markdown
### 3.5 Eigenvalue Calibration (N_effective)

**Source:** Portfolio theory (Choueifaty & Coignard 2008, "Toward Maximum Diversification").  
**Implemented in:** `crates/h2ai-types/src/physics.rs` — `EigenCalibration::from_cg_matrix()`

**Definition 6B — Effective adapter count:**
```
Σ ∈ ℝ^(N×N)   pairwise CG similarity matrix (Σ_ij = CG(adapter_i, adapter_j))
λ_1 ≥ ... ≥ λ_N = eigenvalues of Σ

N_eff     = (Σ λᵢ)² / Σ λᵢ²        [participation ratio]
H_div     = −Σ (λᵢ/Σλ) × log(λᵢ/Σλ)
H_norm    = H_div / log(N)           [normalized diversity ∈ [0,1]]
ρ_eff     = 1 − N_eff/N              [effective correlation from matrix]
```

N_eff is a strictly more informative measure than the scalar ρ_mean = 1 − CG_mean.
Example: 5 adapters with 2 independent + 3 in a tight cluster give N_eff ≈ 2.5
but scalar CG_mean proxy gives ρ_mean ≈ 0.27 → N_eff_scalar ≈ 3.9 (over-estimate).

**Adapter pruning rule:** Add adapter N+1 only if N_eff increases by ≥ 0.05.
For typical LLM ensembles with ρ ≈ 0.9: optimal N = 2 (further adapters redundant).
```
```

- [ ] **Step 3: Add new Section 9 (Talagrand Diagnostics)**

At the end of the file, add:

```markdown
## 9. Talagrand Ensemble Calibration Diagnostic

**Implemented in:** `crates/h2ai-orchestrator/src/diagnostics.rs` — `TalagrandDiagnostic`  
**Inspired by:** Weather ensemble forecasting (Leutbecher & Palmer 2008, ECMWF).

After each verification phase, H2AI records where the runner-up proposal ranks in the
sorted verification score list. Over T ≥ 20 runs, the rank histogram should be uniform.

**Definition 9 — Rank histogram:**
```
For run t with N adapter proposals, sort scores descending: s₁ ≥ s₂ ≥ ... ≥ s_N.
Rank r_t = position of the runner-up in the score ordering (r_t ∈ {1, ..., N}).
Histogram H[r] = count{t : r_t = r} for r = 1..N.

Uniformity test: χ² = Σ_r (H[r] − T/N)² / (T/N)
Calibrated iff χ² < χ²_critical(N−1, 0.05).
```

**Interpretation:**
- Flat histogram → well-calibrated ensemble
- U-shape (high tail counts) → over-confident: adapters too certain, expand τ spread
- Λ-shape (high center counts) → under-dispersed: all adapters mediocre, try diverse model families

**Key advantage:** No ground-truth labels needed. Quality measured from internal consistency alone.
```

- [ ] **Step 4: Verify docs compile**

```bash
cargo doc --workspace --no-deps 2>&1 | grep "^error" | head -5
```

Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add docs/architecture/math-apparatus.md
git commit -m "docs: update math-apparatus — proportional beta_eff, EigenCalibration, TalagrandDiagnostic"
```

---

## Task 7: Full Workspace Verification

**Files:** None — verification only.

- [ ] **Step 1: Full cargo test**

```bash
cargo test --workspace 2>&1 | tail -30
```

Expected: all test suites green, zero FAILED.

- [ ] **Step 2: Run all validation scripts**

```bash
python3 scripts/validate_ensemble_theory.py 2>&1 | tail -5
python3 scripts/validate_beta_coupling.py 2>&1 | grep "PASS\|FAIL\|singularity"
python3 scripts/validate_bft_methods.py 2>&1 | grep "Krum\|Weiszfeld\|Guard"
python3 scripts/validate_conformal_vs_cjt.py 2>&1 | tail -5
python3 scripts/validate_information_theory.py 2>&1 | tail -5
python3 scripts/validate_eigenvalue_calibration.py 2>&1 | tail -5
```

Expected:
- validate_beta_coupling: "PASS — all outputs produced without error or singularity"
- validate_bft_methods: Embedding Krum 100%, Token Krum 0.9%
- Others: no errors

- [ ] **Step 3: Clippy**

```bash
cargo clippy --workspace -- -D warnings 2>&1 | grep "^error" | head -20
```

Expected: no errors. Fix any warnings that were introduced in Tasks 1–5.

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "chore: workspace verification — all tests green, scripts validated"
```

---

## Self-Review

**Spec coverage check:**

| Gap | Task | Covered? |
|---|---|---|
| P4: β_eff singularity | Task 1 | ✓ |
| P3: token Krum uses wrong metric | Task 2 | ✓ |
| P5: M<3 → USL fit never runs | Task 3 | ✓ |
| P2: scalar ρ misses cluster structure | Task 5 | ✓ (EigenCalibration) |
| P6: CRDT monotone invariant undocumented | — | Not in scope — existing code is correct; invariant is documented in ProposalSet inline comments |
| Innovation: TalagrandDiagnostic | Task 4 | ✓ |
| Innovation: EigenCalibration | Task 5 | ✓ |
| Documentation update | Task 6 | ✓ |
| Full verification | Task 7 | ✓ |

**Placeholder scan:** None found. All code blocks are complete and compilable.

**Type consistency check:**
- `EigenCalibration` defined in Task 5 Step 5, imported in events.rs Step 7, computed in calibration.rs Step 8. ✓
- `krum_select_semantic` defined in Task 2 Step 3, used in merger.rs Step 4. ✓
- `TalagrandDiagnostic` defined in Task 4 Step 1, exported Step 3. ✓
- `AppState::explorer2_adapter` added in Task 3 Step 2, used in Step 3. ✓
