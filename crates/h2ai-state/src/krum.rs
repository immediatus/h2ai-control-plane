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
//! - **Krum**: high `role_error_cost` (adversarial/unreliable adapters), quorum
//!   satisfied, AND `cluster_coherent` passes. Provides outlier rejection.
//! - **ConsensusMedian**: stochastic LLM diversity where outputs legitimately
//!   differ. No cluster assumption required. The correct default for honest divergence.

/// Maximum mean pairwise Jaccard distance below which the Krum cluster assumption
/// is considered approximately satisfied. Above this threshold, honest proposals
/// are too lexically diverse for Krum's BFT proof to apply.
pub const MAX_CLUSTER_DIAMETER: f64 = 0.7;

use h2ai_context::embedding::EmbeddingModel;
use h2ai_context::similarity::semantic_jaccard;
use h2ai_types::events::ProposalEvent;

// ── Public helpers ───────────────────────────────────────────────────────────

/// Minimum proposal count to tolerate `f` Byzantine faults via Krum.
/// Derived from n ≥ 2f + 3 (Blanchard et al. 2017).
pub const fn min_quorum(f: usize) -> usize {
    2 * f + 3
}

/// Returns `true` if `n` proposals satisfy the Krum quorum for fault bound `f`.
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
        .map(|&(i, j)| semantic_jaccard(outputs[i], outputs[j], embedding_model))
        .collect();
    let total: f64 = similarities.iter().map(|s| 1.0 - s).sum();
    total / similarities.len() as f64
}

/// Returns `true` if proposals form a sufficiently tight cluster for Krum's BFT
/// proof (Blanchard et al. 2017, Theorem 2) to apply.
///
/// Uses semantic distance so that paraphrased but semantically-identical proposals
/// are correctly identified as coherent even when lexically distant.
/// When `false`, the cluster assumption is violated and callers should fall back
/// to `ConsensusMedian` which handles honest stochastic divergence.
pub async fn cluster_coherent(
    proposals: &[ProposalEvent],
    embedding_model: Option<&dyn EmbeddingModel>,
) -> bool {
    mean_pairwise_distance(proposals, embedding_model).await < MAX_CLUSTER_DIAMETER
}

// ── Private helpers (shared by sync and async Krum) ─────────────────────────

/// Krum score for proposal at global index `idx`, considering only proposals
/// whose global indices are in `subset`. Uses the `k` nearest neighbours.
pub fn krum_score_subset(idx: usize, subset: &[usize], distances: &[Vec<f64>], k: usize) -> f64 {
    let mut dists: Vec<f64> = subset
        .iter()
        .filter(|&&j| j != idx)
        .map(|&j| distances[idx][j])
        .collect();
    dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    dists.iter().take(k).sum()
}

/// Find the index (in `0..n`) with minimum Krum score using `k` neighbours.
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

/// Build the n×n semantic distance matrix via `semantic_jaccard`.
/// All pairs computed concurrently via `join_all`.
/// Falls back to token Jaccard when `adapter` is `None`.
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
        .map(|&(i, j)| semantic_jaccard(outputs[i], outputs[j], embedding_model))
        .collect();
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
pub async fn krum_select_semantic<'a>(
    proposals: &'a [ProposalEvent],
    f: usize,
    embedding_model: Option<&'a dyn EmbeddingModel>,
) -> Option<&'a ProposalEvent> {
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

/// **Semantic Multi-Krum** — iteratively selects `m` proposals via semantic Krum.
pub async fn multi_krum_select_semantic<'a>(
    proposals: &'a [ProposalEvent],
    f: usize,
    m: usize,
    embedding_model: Option<&'a dyn EmbeddingModel>,
) -> Vec<&'a ProposalEvent> {
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
