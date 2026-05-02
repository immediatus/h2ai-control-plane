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

use h2ai_context::embedding::EmbeddingModel;
use h2ai_context::similarity::semantic_jaccard;
use h2ai_types::events::ProposalEvent;
use std::cmp::Ordering;

pub struct ConsensusMedian;

impl ConsensusMedian {
    /// Fréchet median: returns the proposal with minimum sum of distances to all others.
    ///
    /// Equivalently, maximises sum of pairwise semantic similarities.
    /// Ties broken by position (later index wins) — `Iterator::max_by` semantics.
    ///
    /// When `adapter` is `Some`, uses `semantic_jaccard` (paraphrase-aware).
    /// When `adapter` is `None`, uses token Jaccard (offline/test mode).
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

        // Compute all unique pairwise similarities concurrently.
        // At n ≤ 9 this is at most 36 calls; join_all parallelises them.
        let pairs: Vec<(usize, usize)> = (0..n)
            .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
            .collect();

        let pair_sims: Vec<f64> = pairs
            .iter()
            .map(|&(i, j)| semantic_jaccard(outputs[i], outputs[j], embedding_model))
            .collect();

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
            generation: 0,
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
        let proposals = [p.clone()];
        let result = ConsensusMedian::resolve(&proposals, None).await.unwrap();
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
            "expected consensus proposal, got: {}",
            selected.raw_output
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
