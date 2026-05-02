use crate::embedding::{semantic_jaccard, EmbeddingModel};
use crate::jaccard::{jaccard, tokenize};
use std::collections::HashMap;

/// RRF constant from Cormack et al. 2009. Lower k → top ranks matter more.
pub const RRF_K: f64 = 60.0;

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

fn rank_by_embedding(
    query: &str,
    docs: &[&str],
    model: Option<&dyn EmbeddingModel>,
) -> Vec<(usize, f64)> {
    let mut ranked: Vec<(usize, f64)> = docs
        .iter()
        .enumerate()
        .map(|(i, doc)| (i, semantic_jaccard(query, doc, model)))
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked
}

/// When `model` is `None` both streams use token Jaccard — no penalty for
/// deployments without an embedding model.
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
        assert!(
            (score_0 - score_1).abs() < 1e-9,
            "mirrored ranks → equal RRF score"
        );
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
        let docs = [
            "jwt auth token stateless",
            "redis cache store",
            "tcp socket",
        ];
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
        assert_eq!(
            result[0].0, 0,
            "semantic auth match must rank above unrelated doc"
        );
    }
}
