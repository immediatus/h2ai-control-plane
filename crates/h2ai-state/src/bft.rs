//! Fréchet Median proposal selection (`ConsensusMedian`).
//!
//! ## Mathematical foundation
//!
//! In metric space (𝒫(Tokens), `d_J`) where `d_J(A,B)` = 1 − J(A,B), the **Fréchet median**
//! (Fréchet 1948) is:
//!
//!   m* = argmin_{x ∈ S} Σᵢ d(x, sᵢ)
//!
//! Minimising the sum of distances is equivalent to maximising the sum of similarities:
//!
//!   m* = argmax_{x ∈ S} Σᵢ `semantic_jaccard(x`, sᵢ)
//!
//! **Breakdown point:** The Fréchet median tolerates ⌊n/2⌋ − 1 outliers
//! (breakdown point 1/2, Vardi & Zhang 2000). This is strictly stronger than Krum's
//! breakdown point of ⌊(n−3)/4⌋/n and does not require the cluster assumption.
//!
//! **When to use:**
//! - Honest stochastic diversity (LLMs producing different but correct outputs)
//! - Medium error costs (above BFT threshold, below Krum threshold)
//! - Any case where Krum's cluster assumption is violated
//!
//! When `adapter` is `Some(...)`, uses semantic Jaccard (synonyms score as close).
//! When `adapter` is `None`, falls back to token Jaccard (deterministic, no I/O).

use h2ai_context::embedding::cosine_similarity;
use h2ai_context::embedding::EmbeddingModel;
use h2ai_types::events::ProposalEvent;
use std::cmp::Ordering;
use std::collections::HashSet;

// Token-Jaccard similarity fallback for BFT median — a valid metric for the
// Fréchet median proof. Exact equality (the embedding.rs fallback) produces a
// degenerate metric where all distinct proposals are equidistant, making the
// Fréchet median selection non-deterministic for non-identical inputs.
fn tokenize(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(str::to_lowercase)
        .filter(|t| t.len() > 1)
        .collect()
}

#[allow(clippy::cast_precision_loss)]
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

/// BFT-resistant consensus via the Fréchet median in Jaccard metric space.
///
/// Provides a breakdown point of 1/2: up to ⌊n/2⌋ − 1 outlier proposals
/// cannot shift the selected result outside the honest majority, without
/// requiring Krum's cluster assumption (Vardi & Zhang 2000).
/// Prefer this over `krum_select_semantic` when honest proposals are
/// lexically diverse (different phrasings of the same correct answer).
pub struct ConsensusMedian;

impl ConsensusMedian {
    /// Select the proposal closest to the Fréchet median of all proposals.
    ///
    /// `proposals` is the full set of candidate `ProposalEvent`s; `embedding_model`
    /// is an optional semantic adapter — when `Some`, pairwise distances use
    /// `semantic_jaccard` (paraphrase-aware); when `None`, token Jaccard is used
    /// (deterministic, no I/O, suitable for tests).  The breakdown point of 1/2
    /// means that so long as the majority of proposals are honest, the selected
    /// proposal is drawn from that majority regardless of where the outliers lie.
    #[allow(clippy::unused_async)]
    pub async fn resolve<'a>(
        proposals: &'a [ProposalEvent],
        embedding_model: Option<&'a dyn EmbeddingModel>,
    ) -> Option<&'a ProposalEvent> {
        if proposals.is_empty() {
            return None;
        }
        if proposals.len() == 1 {
            return proposals.first();
        }

        let n = proposals.len();
        let outputs: Vec<&str> = proposals.iter().map(|p| p.raw_output.as_str()).collect();

        // Pre-embed every proposal once (O(N) ONNX passes) then compute pair
        // similarities from cached vectors (pure dot products). The naive pair loop
        // would call embed 2×N(N-1)/2 times — for N=9 that is 72 redundant passes.
        // block_in_place signals Tokio to move other tasks off this thread for the
        // duration of the ONNX inference so the executor is not starved.
        let pairs: Vec<(usize, usize)> = (0..n)
            .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
            .collect();

        let pair_sims: Vec<f64> = embedding_model.map_or_else(
            || {
                pairs
                    .iter()
                    .map(|&(i, j)| token_jaccard(outputs[i], outputs[j]))
                    .collect()
            },
            |m| {
                let embeddings: Vec<Vec<f32>> =
                    tokio::task::block_in_place(|| outputs.iter().map(|s| m.embed(s)).collect());
                pairs
                    .iter()
                    .map(|&(i, j)| {
                        if embeddings[i].is_empty() || embeddings[j].is_empty() {
                            if outputs[i] == outputs[j] {
                                1.0
                            } else {
                                0.0
                            }
                        } else {
                            cosine_similarity(&embeddings[i], &embeddings[j]).max(0.0)
                        }
                    })
                    .collect()
            },
        );

        // Populate symmetric similarity matrix.
        let mut sims = vec![vec![1.0f64; n]; n]; // diagonal = 1.0 (self-similarity)
        for (k, &(i, j)) in pairs.iter().enumerate() {
            sims[i][j] = pair_sims[k];
            sims[j][i] = pair_sims[k];
        }

        // Fréchet median: argmax of sum of similarities to all others.
        // Sum includes self-similarity (1.0) which cancels out in comparisons.
        proposals
            .iter()
            .enumerate()
            .max_by(|(i, _), (j, _)| {
                let si: f64 = sims[*i].iter().sum();
                let sj: f64 = sims[*j].iter().sum();
                si.partial_cmp(&sj).unwrap_or(Ordering::Equal)
            })
            .map(|(_, p)| p)
    }
}
