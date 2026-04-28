# Knowledge Retrieval Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add RRF-based hybrid search to `h2ai-context` and Ebbinghaus temporal decay to `CoherencyCoefficients`, making the retrieval and calibration layers aware of recency.

**Architecture:** Two independent additions. (1) `h2ai-context/src/fusion.rs` — a new module with `rrf_fuse` and `hybrid_search` that fuses token-Jaccard and embedding-cosine rankings via Reciprocal Rank Fusion. (2) `h2ai-types/src/physics.rs` — a new `beta_eff_temporal` method and `CG_HALFLIFE_SECS` constant on `CoherencyCoefficients` that weights CG samples by exponential age decay.

**Tech Stack:** Rust, `h2ai-context` crate, `h2ai-types` crate, no new dependencies.

---

## File Map

| File | Change |
|---|---|
| `crates/h2ai-context/src/fusion.rs` | **Create** — `rrf_fuse`, `rank_by_jaccard`, `rank_by_embedding`, `hybrid_search`, inline tests |
| `crates/h2ai-context/src/lib.rs` | **Modify** — add `pub mod fusion;` |
| `crates/h2ai-types/src/physics.rs` | **Modify** — add `CG_HALFLIFE_SECS` const and `beta_eff_temporal` method to `CoherencyCoefficients`, add inline tests |

---

### Task 1: RRF fusion + hybrid search (`h2ai-context/src/fusion.rs`)

**Files:**
- Create: `crates/h2ai-context/src/fusion.rs`
- Modify: `crates/h2ai-context/src/lib.rs`

- [ ] **Step 1: Write the failing tests**

Add a new file `crates/h2ai-context/src/fusion.rs` with just the test module (no implementation yet):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rrf_fuse_single_list_preserves_order() {
        let list = vec![(0usize, 0.9), (1, 0.7), (2, 0.3)];
        let fused = rrf_fuse(&[list], RRF_K);
        assert_eq!(fused[0].0, 0);
        assert_eq!(fused[1].0, 1);
        assert_eq!(fused[2].0, 2);
    }

    #[test]
    fn rrf_fuse_two_agreeing_lists_amplifies_top_doc() {
        let list_a = vec![(0usize, 0.9), (1, 0.5)];
        let list_b = vec![(0usize, 0.8), (1, 0.4)];
        let fused = rrf_fuse(&[list_a, list_b], RRF_K);
        assert_eq!(fused[0].0, 0, "doc ranked 1st in both lists must win");
        assert!(fused[0].1 > fused[1].1);
    }

    #[test]
    fn rrf_fuse_disagreeing_lists_give_equal_scores() {
        // list_a: [0, 1]  list_b: [1, 0] → each doc appears once at rank 1 and once at rank 2
        let list_a = vec![(0usize, 0.9), (1, 0.5)];
        let list_b = vec![(1usize, 0.9), (0, 0.5)];
        let fused = rrf_fuse(&[list_a, list_b], RRF_K);
        let score_0 = fused.iter().find(|(i, _)| *i == 0).unwrap().1;
        let score_1 = fused.iter().find(|(i, _)| *i == 1).unwrap().1;
        assert!((score_0 - score_1).abs() < 1e-9, "mirrored ranks → equal RRF score");
    }

    #[test]
    fn rrf_fuse_empty_input_returns_empty() {
        let fused: Vec<(usize, f64)> = rrf_fuse(&[], RRF_K);
        assert!(fused.is_empty());
    }

    #[test]
    fn hybrid_search_empty_docs_returns_empty() {
        let result = hybrid_search("query", &[], None, RRF_K);
        assert!(result.is_empty());
    }

    #[test]
    fn hybrid_search_returns_all_docs() {
        let docs = ["jwt auth token", "redis cache store", "stateless session"];
        let result = hybrid_search("jwt authentication", &docs, None, RRF_K);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn hybrid_search_relevant_doc_ranks_first_without_model() {
        let docs = ["jwt auth token stateless", "redis cache store", "tcp socket"];
        let result = hybrid_search("jwt authentication", &docs, None, RRF_K);
        assert_eq!(result[0].0, 0, "jwt doc must rank first for jwt query");
    }

    #[test]
    fn hybrid_search_with_model_semantic_doc_ranks_higher() {
        use crate::embedding::EmbeddingModel;
        struct AuthModel;
        impl EmbeddingModel for AuthModel {
            fn embed(&self, text: &str) -> Vec<f32> {
                if text.contains("auth") || text.contains("jwt") || text.contains("bearer") {
                    vec![1.0, 0.0]
                } else {
                    vec![0.0, 1.0]
                }
            }
        }
        let docs = ["bearer token mechanism", "redis cache store"];
        let result = hybrid_search("jwt authentication", &docs, Some(&AuthModel), RRF_K);
        assert_eq!(result[0].0, 0, "semantic auth match must rank above unrelated doc");
    }
}
```

- [ ] **Step 2: Verify tests fail**

Run: `cargo test -p h2ai-context fusion 2>&1`
Expected: `error[E0425]: cannot find function rrf_fuse` or similar — confirms stubs are needed.

- [ ] **Step 3: Write the implementation**

Replace the full contents of `crates/h2ai-context/src/fusion.rs` with:

```rust
use crate::embedding::{semantic_jaccard, EmbeddingModel};
use crate::jaccard::{jaccard, tokenize};
use std::collections::HashMap;

/// RRF constant from Cormack et al. 2009. Lower k → top ranks matter more.
pub const RRF_K: f64 = 60.0;

/// Fuse multiple ranked lists using Reciprocal Rank Fusion.
///
/// Each input list is `Vec<(doc_index, score)>` sorted descending by score.
/// Returns a fused list sorted descending by RRF score.
///
/// `rrf_score(d) = Σ 1.0 / (k + rank_i(d))` where `rank_i(d)` is the 1-based
/// position of document `d` in list `i`. Documents absent from a list contribute
/// nothing from that list.
pub fn rrf_fuse(ranked_lists: &[Vec<(usize, f64)>], k: f64) -> Vec<(usize, f64)> {
    let mut scores: HashMap<usize, f64> = HashMap::new();
    for list in ranked_lists {
        for (rank, &(doc_idx, _)) in list.iter().enumerate() {
            *scores.entry(doc_idx).or_insert(0.0) += 1.0 / (k + (rank + 1) as f64);
        }
    }
    let mut fused: Vec<(usize, f64)> = scores.into_iter().collect();
    fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    fused
}

fn rank_by_jaccard(query: &str, docs: &[&str]) -> Vec<(usize, f64)> {
    let q_tokens = tokenize(query);
    let mut ranked: Vec<(usize, f64)> = docs
        .iter()
        .enumerate()
        .map(|(i, doc)| (i, jaccard(&q_tokens, &tokenize(doc))))
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked
}

fn rank_by_embedding(query: &str, docs: &[&str], model: Option<&dyn EmbeddingModel>) -> Vec<(usize, f64)> {
    let mut ranked: Vec<(usize, f64)> = docs
        .iter()
        .enumerate()
        .map(|(i, doc)| (i, semantic_jaccard(query, doc, model)))
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked
}

/// Hybrid search combining token Jaccard and embedding cosine similarity via RRF.
///
/// When `model` is `None` both streams use token Jaccard — no penalty for
/// deployments without an embedding model.
///
/// Returns `(doc_index, rrf_score)` sorted descending. All input documents are
/// represented in the output even if their score is zero.
pub fn hybrid_search(
    query: &str,
    docs: &[&str],
    model: Option<&dyn EmbeddingModel>,
    k: f64,
) -> Vec<(usize, f64)> {
    if docs.is_empty() {
        return vec![];
    }
    let jaccard_ranks = rank_by_jaccard(query, docs);
    let embedding_ranks = rank_by_embedding(query, docs, model);
    rrf_fuse(&[jaccard_ranks, embedding_ranks], k)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rrf_fuse_single_list_preserves_order() {
        let list = vec![(0usize, 0.9), (1, 0.7), (2, 0.3)];
        let fused = rrf_fuse(&[list], RRF_K);
        assert_eq!(fused[0].0, 0);
        assert_eq!(fused[1].0, 1);
        assert_eq!(fused[2].0, 2);
    }

    #[test]
    fn rrf_fuse_two_agreeing_lists_amplifies_top_doc() {
        let list_a = vec![(0usize, 0.9), (1, 0.5)];
        let list_b = vec![(0usize, 0.8), (1, 0.4)];
        let fused = rrf_fuse(&[list_a, list_b], RRF_K);
        assert_eq!(fused[0].0, 0, "doc ranked 1st in both lists must win");
        assert!(fused[0].1 > fused[1].1);
    }

    #[test]
    fn rrf_fuse_disagreeing_lists_give_equal_scores() {
        let list_a = vec![(0usize, 0.9), (1, 0.5)];
        let list_b = vec![(1usize, 0.9), (0, 0.5)];
        let fused = rrf_fuse(&[list_a, list_b], RRF_K);
        let score_0 = fused.iter().find(|(i, _)| *i == 0).unwrap().1;
        let score_1 = fused.iter().find(|(i, _)| *i == 1).unwrap().1;
        assert!((score_0 - score_1).abs() < 1e-9, "mirrored ranks → equal RRF score");
    }

    #[test]
    fn rrf_fuse_empty_input_returns_empty() {
        let fused: Vec<(usize, f64)> = rrf_fuse(&[], RRF_K);
        assert!(fused.is_empty());
    }

    #[test]
    fn hybrid_search_empty_docs_returns_empty() {
        let result = hybrid_search("query", &[], None, RRF_K);
        assert!(result.is_empty());
    }

    #[test]
    fn hybrid_search_returns_all_docs() {
        let docs = ["jwt auth token", "redis cache store", "stateless session"];
        let result = hybrid_search("jwt authentication", &docs, None, RRF_K);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn hybrid_search_relevant_doc_ranks_first_without_model() {
        let docs = ["jwt auth token stateless", "redis cache store", "tcp socket"];
        let result = hybrid_search("jwt authentication", &docs, None, RRF_K);
        assert_eq!(result[0].0, 0, "jwt doc must rank first for jwt query");
    }

    #[test]
    fn hybrid_search_with_model_semantic_doc_ranks_higher() {
        use crate::embedding::EmbeddingModel;
        struct AuthModel;
        impl EmbeddingModel for AuthModel {
            fn embed(&self, text: &str) -> Vec<f32> {
                if text.contains("auth") || text.contains("jwt") || text.contains("bearer") {
                    vec![1.0, 0.0]
                } else {
                    vec![0.0, 1.0]
                }
            }
        }
        let docs = ["bearer token mechanism", "redis cache store"];
        let result = hybrid_search("jwt authentication", &docs, Some(&AuthModel), RRF_K);
        assert_eq!(result[0].0, 0, "semantic auth match must rank above unrelated doc");
    }
}
```

- [ ] **Step 4: Export `fusion` from `lib.rs`**

In `crates/h2ai-context/src/lib.rs`, add `pub mod fusion;` after the existing module declarations:

```rust
pub mod adr;
pub mod compaction;
pub mod compiler;
pub mod embedding;
pub mod fusion;
pub mod jaccard;
pub mod similarity;
```

- [ ] **Step 5: Run tests and verify they pass**

Run: `cargo test -p h2ai-context fusion 2>&1`

Expected output:
```
running 8 tests
test fusion::tests::hybrid_search_empty_docs_returns_empty ... ok
test fusion::tests::hybrid_search_relevant_doc_ranks_first_without_model ... ok
test fusion::tests::hybrid_search_returns_all_docs ... ok
test fusion::tests::hybrid_search_with_model_semantic_doc_ranks_higher ... ok
test fusion::tests::rrf_fuse_disagreeing_lists_give_equal_scores ... ok
test fusion::tests::rrf_fuse_empty_input_returns_empty ... ok
test fusion::tests::rrf_fuse_single_list_preserves_order ... ok
test fusion::tests::rrf_fuse_two_agreeing_lists_amplifies_top_doc ... ok
test result: ok. 8 passed; 0 failed
```

- [ ] **Step 6: Run full crate tests to confirm no regressions**

Run: `cargo test -p h2ai-context 2>&1 | grep -E "^test result|FAILED"`

Expected: all `test result: ok`, no `FAILED`.

---

### Task 2: Temporal decay in `CoherencyCoefficients`

**Files:**
- Modify: `crates/h2ai-types/src/physics.rs`

- [ ] **Step 1: Write the failing tests**

In `crates/h2ai-types/src/physics.rs`, locate the `#[cfg(test)]` block at the bottom of the file and append these tests:

```rust
    #[test]
    fn beta_eff_temporal_fresh_sample_equals_beta_eff() {
        let cc = CoherencyCoefficients { alpha: 0.1, beta_base: 0.02, cg_samples: vec![0.6] };
        let now = 1_000_000u64;
        // Sample timestamped at now → weight = e^0 = 1.0 → identical to beta_eff()
        let result = cc.beta_eff_temporal(now, &[now]);
        let expected = cc.beta_eff();
        assert!((result - expected).abs() < 1e-9, "fresh sample: {result} vs {expected}");
    }

    #[test]
    fn beta_eff_temporal_stale_sample_approaches_beta_base() {
        let cc = CoherencyCoefficients { alpha: 0.1, beta_base: 0.05, cg_samples: vec![0.8] };
        // 100 halflives later → weight ≈ 0 → cg_eff ≈ 0 → beta_eff ≈ beta_base
        let now = CG_HALFLIFE_SECS * 100;
        let result = cc.beta_eff_temporal(now, &[0u64]);
        assert!(
            (result - cc.beta_base).abs() < 0.001,
            "stale sample must approach beta_base={}, got {result}", cc.beta_base
        );
    }

    #[test]
    fn beta_eff_temporal_mismatched_timestamps_falls_back() {
        let cc = CoherencyCoefficients { alpha: 0.1, beta_base: 0.02, cg_samples: vec![0.6, 0.7] };
        // 1 timestamp for 2 samples → mismatch → fallback to beta_eff()
        let result = cc.beta_eff_temporal(1_000_000, &[1_000_000u64]);
        assert!((result - cc.beta_eff()).abs() < 1e-9);
    }

    #[test]
    fn beta_eff_temporal_empty_timestamps_falls_back() {
        let cc = CoherencyCoefficients { alpha: 0.1, beta_base: 0.02, cg_samples: vec![0.6] };
        let result = cc.beta_eff_temporal(1_000_000, &[]);
        assert!((result - cc.beta_eff()).abs() < 1e-9);
    }

    #[test]
    fn beta_eff_temporal_recent_low_cg_dominates_old_high_cg() {
        // old sample: high CG=0.9 → low beta (helpful); recent sample: low CG=0.2 → high beta
        // After aging, old sample's weight ≈ 0, so result is dominated by recent low-CG sample
        let cc = CoherencyCoefficients { alpha: 0.1, beta_base: 0.05, cg_samples: vec![0.9, 0.2] };
        let now = CG_HALFLIFE_SECS * 10;
        let timestamps = [0u64, now]; // first sample is 10 halflives old, second is fresh
        let result = cc.beta_eff_temporal(now, &timestamps);
        // cg_eff ≈ 0.2 (fresh dominates) → beta = 0.05 * (1 - 0.2) = 0.04
        let fresh_only_beta = cc.beta_base * (1.0 - 0.2);
        assert!(
            (result - fresh_only_beta).abs() < 0.005,
            "recent low-CG sample must dominate: expected ≈{fresh_only_beta:.4}, got {result:.4}"
        );
    }
```

- [ ] **Step 2: Verify tests fail**

Run: `cargo test -p h2ai-types beta_eff_temporal 2>&1`

Expected: `error[E0599]: no method named beta_eff_temporal found` — confirms method is missing.

- [ ] **Step 3: Add `CG_HALFLIFE_SECS` constant and `beta_eff_temporal` method**

In `crates/h2ai-types/src/physics.rs`, locate the line that begins `/// Effective coordination cost` (the `beta_eff` doc comment, around line 74). Add the constant just before the `impl CoherencyCoefficients` block:

```rust
/// Halflife for CG sample decay under Ebbinghaus temporal weighting.
/// 7 days: a sample one week old contributes at 50% weight.
pub const CG_HALFLIFE_SECS: u64 = 604_800;
```

Then inside `impl CoherencyCoefficients`, after the `cg_std_dev` method (around line 108), add:

```rust
    /// Effective coordination cost with Ebbinghaus temporal decay.
    ///
    /// CG samples are weighted by `e^(-(now_secs − t) / CG_HALFLIFE_SECS)`.
    /// Older samples fade toward zero weight; as all samples age, `cg_eff → 0`
    /// and `beta_eff_temporal → beta_base` (conservative, no CG discount).
    ///
    /// Falls back to `beta_eff()` when `sample_timestamps` is empty or its
    /// length does not match `cg_samples`.
    pub fn beta_eff_temporal(&self, now_secs: u64, sample_timestamps: &[u64]) -> f64 {
        if sample_timestamps.len() != self.cg_samples.len() || sample_timestamps.is_empty() {
            return self.beta_eff();
        }
        let halflife = CG_HALFLIFE_SECS as f64;
        let weights: Vec<f64> = sample_timestamps
            .iter()
            .map(|&t| (-(now_secs.saturating_sub(t) as f64) / halflife).exp())
            .collect();
        let total_weight: f64 = weights.iter().sum();
        if total_weight < 1e-15 {
            return self.beta_eff();
        }
        let cg_eff: f64 = self.cg_samples
            .iter()
            .zip(&weights)
            .map(|(cg, w)| cg * w)
            .sum::<f64>()
            / total_weight;
        (self.beta_base * (1.0 - cg_eff.clamp(0.0, 1.0))).max(1e-6)
    }
```

Also add `CG_HALFLIFE_SECS` to the `use h2ai_types::physics::{...}` import line in `crates/h2ai-types/tests/physics_test.rs` so the test can reference it:

```rust
use h2ai_types::physics::{
    CoherencyCoefficients, CoordinationThreshold, EigenCalibration, EnsembleCalibration,
    JeffectiveGap, MergeStrategy, MultiplicationCondition, MultiplicationConditionFailure,
    RoleErrorCost, TauValue, n_it_optimal, CG_HALFLIFE_SECS,
};
```

- [ ] **Step 4: Run tests and verify they pass**

Run: `cargo test -p h2ai-types beta_eff_temporal 2>&1`

Expected output:
```
running 5 tests
test beta_eff_temporal_empty_timestamps_falls_back ... ok
test beta_eff_temporal_fresh_sample_equals_beta_eff ... ok
test beta_eff_temporal_mismatched_timestamps_falls_back ... ok
test beta_eff_temporal_recent_low_cg_dominates_old_high_cg ... ok
test beta_eff_temporal_stale_sample_approaches_beta_base ... ok
test result: ok. 5 passed; 0 failed
```

- [ ] **Step 5: Run full crate tests to confirm no regressions**

Run: `cargo test -p h2ai-types 2>&1 | grep -E "^test result|FAILED"`

Expected: all `test result: ok`, no `FAILED`.

---

### Task 3: Full workspace verification

**Files:** None — verification only.

- [ ] **Step 1: Run workspace tests**

Run: `cargo test --workspace 2>&1 | grep -E "^test result|FAILED|^error"`

Expected: every line reads `test result: ok`, no `FAILED`, no `error`.

- [ ] **Step 2: Run clippy**

Run: `cargo clippy --workspace -- -D warnings 2>&1 | grep -E "^error"`

Expected: no output (zero errors).

- [ ] **Step 3: Confirm new public API is accessible**

Run: `cargo doc -p h2ai-context -p h2ai-types --no-deps 2>&1 | grep -E "^error"`

Expected: no output (docs compile cleanly).
