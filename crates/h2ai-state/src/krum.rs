//! Krum and Multi-Krum outlier-resistant proposal selection.
//!
//! ## Algorithm
//!
//! Both algorithms operate in the metric space `(𝒫(Tokens), d_J)` where
//! `d_J(A, B) = 1 − J(A, B)` is the Jaccard distance. Jaccard distance is a
//! valid metric (satisfies triangle inequality — Levandowsky & Winter 1971).
//!
//! Krum selects the proposal with the minimum sum of distances to its
//! `n − f − 2` nearest neighbours — the proposal most deeply embedded in
//! the densest cluster.
//!
//! ## Theoretical precondition (Blanchard et al. 2017, Theorem 2)
//!
//! The proof that Krum selects a non-Byzantine proposal rests on a **cluster
//! assumption**: honest proposals must form a cluster of diameter Δ that is
//! small relative to the distance from honest proposals to Byzantine ones.
//!
//! **This assumption does not hold unconditionally for LLM outputs.** LLMs
//! hallucinate stochastically, not adversarially. Diverse high-quality outputs
//! may be lexically distant (different phrasings, synonyms, sentence structure)
//! and thus violate the cluster assumption. When honest proposals are spread
//! across Jaccard space, Krum picks the most "average" output — not necessarily
//! the best one — and its BFT safety guarantee no longer applies.
//!
//! **Before applying Krum, call [`cluster_coherent`] to verify the precondition.**
//! If cluster coherence fails, fall back to `ConsensusMedian`, which handles
//! honest stochastic divergence without requiring a cluster assumption.
//!
//! ## Quorum requirement
//!
//! | f (outliers) | n_min |
//! |--------------|-------|
//! | 1            | 5     |
//! | 2            | 7     |
//! | 3            | 9     |
//! | f            | 2f+3  |
//!
//! Call [`quorum_satisfied`] before invoking [`krum_select`] or
//! [`multi_krum_select`]. The engine enforces this via `EngineError::InsufficientQuorum`.
//!
//! ## When to use Krum vs ConsensusMedian
//!
//! - **Krum**: high `role_error_cost` (adversarial/unreliable adapters), embedding
//!   model present, quorum satisfied, AND `cluster_coherent` passes.
//! - **ConsensusMedian**: stochastic LLM diversity where outputs legitimately differ,
//!   OR no embedding model is available. No cluster assumption required.
//!
//! **Krum always requires a semantic embedding model.** Without one, `cluster_coherent`,
//! `krum_select_semantic`, and `multi_krum_select_semantic` all return failure values
//! (`false` / `None` / `[]`) so the merger falls back to `ConsensusMedian` automatically.
//! Token Jaccard alone cannot rescue the cluster assumption for long LLM outputs: two
//! adapters writing the same code with different variable names appear as outliers to each
//! other, allowing a degenerate stopword-heavy output to win the Krum selection.

/// Maximum mean pairwise Jaccard distance below which the Krum cluster assumption
/// is considered approximately satisfied. Above this threshold, honest proposals
/// are too lexically diverse for Krum's BFT proof to apply.
pub const MAX_CLUSTER_DIAMETER: f64 = 0.7;

use h2ai_context::embedding::semantic_jaccard;
use h2ai_context::embedding::EmbeddingModel;
use h2ai_types::events::ProposalEvent;
use std::collections::HashSet;

// Token Jaccard is a valid metric (triangle inequality holds), but it breaks down
// on long LLM outputs where lexically-distinct paraphrases of the same correct answer
// are scored as far apart as hallucinations — destroying Krum's cluster assumption.
// Used only by `mean_pairwise_distance` as a distance utility; Krum and cluster_coherent
// refuse to run without an embedding model (see guards in those functions).
fn tokenize(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_lowercase())
        .filter(|t| t.len() > 1)
        .collect()
}

fn token_jaccard(a: &str, b: &str) -> f64 {
    let ta = tokenize(a);
    let tb = tokenize(b);
    if ta.is_empty() && tb.is_empty() {
        return 0.0;
    }
    let intersection = ta.intersection(&tb).count() as f64;
    let union = ta.union(&tb).count() as f64;
    intersection / union
}

// ── Public helpers ───────────────────────────────────────────────────────────

/// Minimum proposal count to tolerate `f` Byzantine faults via Krum.
/// Derived from n ≥ 2f + 3 (Blanchard et al. 2017).
pub const fn min_quorum(f: usize) -> usize {
    2 * f + 3
}

/// Return `true` when `n` proposals satisfy the Krum quorum for `f` Byzantine faults.
///
/// Requires `n ≥ 2f + 3` (equivalently `n ≥ min_quorum(f)`), which is the minimum
/// needed for Krum's score function to have enough neighbours to distinguish an
/// honest cluster from up to `f` outliers (Blanchard et al. 2017, Theorem 2).
pub fn quorum_satisfied(n: usize, f: usize) -> bool {
    n >= min_quorum(f)
}

/// Mean pairwise semantic distance across all proposal pairs.
///
/// Uses `semantic_jaccard` for pairwise similarity so that lexically-distinct
/// but semantically-equivalent proposals (synonyms, paraphrases) are recognised
/// as close. All pairs are scored in parallel via `join_all`.
///
/// Returns 0.0 for fewer than 2 proposals (trivially coherent).
pub async fn mean_pairwise_distance(
    proposals: &[ProposalEvent],
    embedding_model: Option<&dyn EmbeddingModel>,
) -> f64 {
    let n = proposals.len();
    if n < 2 {
        return 0.0;
    }
    let outputs: Vec<&str> = proposals.iter().map(|p| p.raw_output.as_str()).collect();
    let pairs: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
        .collect();
    let similarities: Vec<f64> = pairs
        .iter()
        .map(|&(i, j)| match embedding_model {
            Some(m) => semantic_jaccard(outputs[i], outputs[j], Some(m)),
            None => token_jaccard(outputs[i], outputs[j]),
        })
        .collect();
    let total: f64 = similarities.iter().map(|s| 1.0 - s).sum();
    total / similarities.len() as f64
}

/// Returns `true` if proposals form a sufficiently tight cluster for Krum's BFT
/// proof (Blanchard et al. 2017, Theorem 2) to apply.
///
/// **Requires a semantic embedding model.** Returns `false` immediately when
/// `embedding_model` is `None` — token Jaccard is unreliable for long LLM outputs
/// (paraphrased honest proposals score as distant as hallucinations), which would
/// silently break the BFT cluster assumption. Without embeddings the caller must
/// fall back to `ConsensusMedian`.
///
/// When `false`, callers should fall back to `ConsensusMedian`.
pub async fn cluster_coherent(
    proposals: &[ProposalEvent],
    embedding_model: Option<&dyn EmbeddingModel>,
) -> bool {
    if embedding_model.is_none() {
        return false;
    }
    mean_pairwise_distance(proposals, embedding_model).await < MAX_CLUSTER_DIAMETER
}

// ── Private helpers (shared by sync and async Krum) ─────────────────────────

/// Sum of Jaccard distances from proposal `idx` to its `k` nearest neighbours in `subset`.
///
/// `k` is typically `n − f − 2`: each proposal is scored against enough neighbours
/// to expose an outlier while excluding the `f` worst-case Byzantine peers.
/// A lower score means the proposal is more deeply embedded in the densest cluster.
pub fn krum_score_subset(idx: usize, subset: &[usize], distances: &[Vec<f64>], k: usize) -> f64 {
    let mut dists: Vec<f64> = subset
        .iter()
        .filter(|&&j| j != idx)
        .map(|&j| distances[idx][j])
        .collect();
    dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    dists.iter().take(k).sum()
}

/// Return the global index of the proposal with the minimum Krum score.
///
/// Scans all `n` proposals in `distances`, scoring each with `krum_score_subset`
/// over the full set.  Returns `None` only when `distances` is empty.
pub fn krum_index(distances: &[Vec<f64>], k: usize) -> Option<usize> {
    let n = distances.len();
    let all: Vec<usize> = (0..n).collect();
    let scores: Vec<f64> = (0..n)
        .map(|i| krum_score_subset(i, &all, distances, k))
        .collect();
    scores
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
}

// ── Async semantic Krum ───────────────────────────────────────────────────────

/// Build the n×n pairwise distance matrix from cosine-based semantic similarity.
///
/// Each entry `d[i][j] = 1 − semantic_jaccard(i, j)` converts a similarity in [0, 1]
/// to a distance in [0, 1] suitable for the Jaccard-metric Krum proof.
/// Falls back to token Jaccard when `embedding_model` is `None`.
async fn semantic_distance_matrix(
    proposals: &[ProposalEvent],
    embedding_model: Option<&dyn EmbeddingModel>,
) -> Vec<Vec<f64>> {
    let n = proposals.len();
    let outputs: Vec<&str> = proposals.iter().map(|p| p.raw_output.as_str()).collect();
    let pairs: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
        .collect();
    let similarities: Vec<f64> = pairs
        .iter()
        .map(|&(i, j)| match embedding_model {
            Some(m) => semantic_jaccard(outputs[i], outputs[j], Some(m)),
            None => token_jaccard(outputs[i], outputs[j]),
        })
        .collect();
    let mut d = vec![vec![0.0f64; n]; n];
    for (k, &(i, j)) in pairs.iter().enumerate() {
        let dist = 1.0 - similarities[k];
        d[i][j] = dist;
        d[j][i] = dist;
    }
    d
}

/// Select the single most-central proposal using semantic Krum.
///
/// Uses `semantic_distance_matrix` (cosine-to-distance conversion) rather than
/// raw token Jaccard, so that paraphrased but semantically-equivalent proposals
/// are treated as close neighbours — the same BFT guarantee as the text path but
/// robust to lexical variation.
///
/// Returns `None` when `embedding_model` is `None` (token Jaccard silently breaks
/// the BFT cluster assumption on long outputs — see module doc), when quorum is
/// unsatisfied, or when `proposals` is empty.
pub async fn krum_select_semantic<'a>(
    proposals: &'a [ProposalEvent],
    f: usize,
    embedding_model: Option<&'a dyn EmbeddingModel>,
) -> Option<&'a ProposalEvent> {
    embedding_model?;
    if proposals.is_empty() {
        return None;
    }
    if f == 0 {
        return proposals.first();
    }
    if !quorum_satisfied(proposals.len(), f) {
        return None;
    }
    let distances = semantic_distance_matrix(proposals, embedding_model).await;
    let k = proposals.len() - f - 2;
    krum_index(&distances, k).map(|i| &proposals[i])
}

/// Select up to `m` proposals by iteratively applying semantic Krum.
///
/// Each iteration picks the remaining proposal with the minimum Krum score and
/// removes it from the candidate pool, repeating until `m` proposals are collected
/// or fewer than `f + 3` candidates remain.  Uses the same semantic distance matrix
/// as `krum_select_semantic`, giving consistent BFT behaviour across both functions.
///
/// Returns an empty vec when `embedding_model` is `None` — same rationale as
/// `krum_select_semantic`: token Jaccard silently breaks the BFT guarantee on real
/// LLM outputs.
pub async fn multi_krum_select_semantic<'a>(
    proposals: &'a [ProposalEvent],
    f: usize,
    m: usize,
    embedding_model: Option<&'a dyn EmbeddingModel>,
) -> Vec<&'a ProposalEvent> {
    if embedding_model.is_none() {
        return vec![];
    }
    if proposals.is_empty() || m == 0 {
        return vec![];
    }
    if f == 0 {
        return proposals.iter().take(m).collect();
    }
    if !quorum_satisfied(proposals.len(), f) {
        return vec![];
    }
    let distances = semantic_distance_matrix(proposals, embedding_model).await;
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
