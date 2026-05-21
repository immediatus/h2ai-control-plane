#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::significant_drop_tightening,
    clippy::significant_drop_in_scrutinee,
    clippy::unused_async,
    clippy::default_trait_access,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::cast_possible_wrap,
    clippy::doc_markdown,
    clippy::manual_let_else,
    clippy::match_wildcard_for_single_variants,
    clippy::similar_names,
    clippy::match_same_arms,
    clippy::literal_string_with_formatting_args,
    clippy::redundant_clone,
    clippy::redundant_closure_for_method_calls,
    clippy::useless_format,
    clippy::option_if_let_else,
    clippy::map_unwrap_or,
    clippy::cloned_instead_of_copied,
    clippy::trivially_copy_pass_by_ref,
    clippy::cast_lossless,
    clippy::uninlined_format_args,
    clippy::needless_pass_by_value,
    clippy::explicit_iter_loop,
    clippy::needless_borrow,
    clippy::large_futures,
    clippy::manual_string_new,
    clippy::needless_lifetimes,
    clippy::elidable_lifetime_names,
    clippy::redundant_else,
    clippy::stable_sort_primitive,
    clippy::type_complexity,
    clippy::wildcard_imports,
    clippy::single_match_else,
    clippy::missing_fields_in_debug,
    clippy::doc_link_with_quotes,
    clippy::implicit_hasher,
    clippy::needless_collect,
    clippy::suboptimal_flops,
    clippy::missing_const_for_fn,
    clippy::needless_type_cast,
    clippy::unreadable_literal,
    clippy::no_effect_underscore_binding
)]
use chrono::Utc;
use h2ai_context::embedding::EmbeddingModel;
use h2ai_state::bft::ConsensusMedian;
use h2ai_types::config::AdapterKind;
use h2ai_types::events::ProposalEvent;
use h2ai_types::identity::{ExplorerId, TaskId};
use h2ai_types::sizing::TauValue;

fn cloud() -> AdapterKind {
    AdapterKind::CloudGeneric {
        endpoint: "https://x.com".into(),
        api_key_env: "K".into(),
        model: None,
    }
}

fn proposal(task_id: TaskId, output: &str, token_cost: u64) -> ProposalEvent {
    ProposalEvent {
        task_id,
        explorer_id: ExplorerId::new(),
        tau: TauValue::new(0.1).unwrap(),
        generation: 0,
        raw_output: output.into(),
        token_cost,
        adapter_kind: cloud(),
        timestamp: Utc::now(),
    }
}

#[tokio::test]
async fn condorcet_selects_proposal_most_similar_to_group() {
    let tid = TaskId::new();
    // Two proposals share vocabulary; one outlier does not.
    // Condorcet should pick one of the consensus pair, not the outlier.
    let a = proposal(
        tid.clone(),
        "stateless JWT auth token ADR-001 compliant",
        50,
    );
    let b = proposal(
        tid.clone(),
        "stateless authentication JWT ADR-001 token rotation",
        60,
    );
    let outlier = proposal(
        tid.clone(),
        "blockchain proof-of-work hash rainbow table",
        10,
    );
    let proposals = vec![a.clone(), b.clone(), outlier.clone()];
    let result = ConsensusMedian::resolve(&proposals, None).await.unwrap();
    assert!(
        result.raw_output == a.raw_output || result.raw_output == b.raw_output,
        "Condorcet should pick consensus proposal, not outlier; got: {}",
        result.raw_output
    );
}

#[tokio::test]
async fn condorcet_returns_none_when_no_proposals() {
    let result = ConsensusMedian::resolve(&[], None).await;
    assert!(result.is_none());
}

#[tokio::test]
async fn condorcet_single_proposal_returns_it() {
    let tid = TaskId::new();
    let p = proposal(tid, "only", 200);
    let proposals = [p.clone()];
    let result = ConsensusMedian::resolve(&proposals, None).await.unwrap();
    assert_eq!(result.raw_output, "only");
}

#[tokio::test]
async fn two_identical_proposals_returns_first_by_stability() {
    let tid = TaskId::new();
    let p1 = proposal(tid.clone(), "identical stateless JWT auth ADR-001", 10);
    let p2 = proposal(tid.clone(), "identical stateless JWT auth ADR-001", 10);
    let proposals = vec![p1.clone(), p2];
    let result = ConsensusMedian::resolve(&proposals, None).await.unwrap();
    assert_eq!(result.raw_output, p1.raw_output);
}

#[tokio::test]
async fn frechet_median_selects_semantically_central_proposal() {
    let tid = TaskId::new();
    let p1 = proposal(
        tid.clone(),
        "stateless JWT authentication token rotation ADR-001 compliant",
        50,
    );
    let p2 = proposal(
        tid.clone(),
        "JWT auth token stateless rotation ADR-001 implementation",
        50,
    );
    let outlier = proposal(
        tid.clone(),
        "Redis session store sliding window expiry database cache",
        10,
    );
    let proposals = vec![p1.clone(), p2.clone(), outlier];
    let selected = ConsensusMedian::resolve(&proposals, None).await.unwrap();
    assert!(
        selected.raw_output == p1.raw_output || selected.raw_output == p2.raw_output,
        "Fréchet median must select from the close pair, got: {}",
        selected.raw_output
    );
}

// ── Mock EmbeddingModel ───────────────────────────────────────────────────────

/// Fixed-vector mock: all texts map to the same vector (L2-normalised).
struct FixedEmbedding(Vec<f32>);
impl EmbeddingModel for FixedEmbedding {
    fn embed(&self, _text: &str) -> Vec<f32> {
        self.0.clone()
    }
}

/// Empty-embedding mock: always returns an empty vec, triggering the
/// exact-equality fallback inside `ConsensusMedian`.
struct EmptyEmbedding;
impl EmbeddingModel for EmptyEmbedding {
    fn embed(&self, _text: &str) -> Vec<f32> {
        vec![]
    }
}

/// Axis-aligned mock: assigns orthogonal unit vectors based on the first char.
struct AxisEmbedding;
impl EmbeddingModel for AxisEmbedding {
    fn embed(&self, text: &str) -> Vec<f32> {
        match text.chars().next() {
            Some('a') => vec![1.0, 0.0, 0.0],
            Some('b') => vec![0.9, 0.436, 0.0], // close to 'a' cluster
            Some('c') => vec![0.9, 0.436, 0.0],
            _ => vec![0.0, 0.0, 1.0], // outlier direction
        }
    }
}

// ── ConsensusMedian with embedding model ─────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn consensus_median_with_fixed_embedding_returns_some() {
    // All embeddings identical → all similarities 1.0 → any proposal equally valid.
    let model = FixedEmbedding(vec![1.0, 0.0, 0.0]);
    let tid = TaskId::new();
    let proposals = vec![
        proposal(tid.clone(), "alpha jwt stateless", 10),
        proposal(tid.clone(), "beta redis session", 10),
        proposal(tid.clone(), "gamma blockchain hash", 10),
    ];
    let result = ConsensusMedian::resolve(&proposals, Some(&model)).await;
    assert!(
        result.is_some(),
        "should return Some with a fixed embedding"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn consensus_median_with_embedding_rejects_outlier() {
    // Use axis-aligned embeddings: 'a'/'b'/'c' cluster together, '_' is outlier.
    let model = AxisEmbedding;
    let tid = TaskId::new();
    let a = proposal(tid.clone(), "alpha jwt stateless auth token", 10);
    let b = proposal(tid.clone(), "beta stateless auth token rotation", 10);
    let c = proposal(tid.clone(), "charlie jwt bearer token compliant", 10);
    let outlier = proposal(tid.clone(), "zzz blockchain proof-of-work hash", 10);
    let proposals = vec![a.clone(), b.clone(), c.clone(), outlier.clone()];
    let result = ConsensusMedian::resolve(&proposals, Some(&model))
        .await
        .unwrap();
    assert_ne!(
        result.raw_output, outlier.raw_output,
        "Fréchet median with embedding must not select the outlier; got: {}",
        result.raw_output
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn consensus_median_empty_embedding_identical_strings_returns_one() {
    // EmptyEmbedding + identical text → sim=1.0 → any result valid
    let model = EmptyEmbedding;
    let tid = TaskId::new();
    let proposals = vec![
        proposal(tid.clone(), "identical stateless JWT", 10),
        proposal(tid.clone(), "identical stateless JWT", 10),
    ];
    let result = ConsensusMedian::resolve(&proposals, Some(&model)).await;
    assert!(result.is_some());
    assert_eq!(result.unwrap().raw_output, "identical stateless JWT");
}

#[tokio::test(flavor = "multi_thread")]
async fn consensus_median_empty_embedding_different_strings_returns_one() {
    // EmptyEmbedding + different strings → sim=0.0 between all pairs.
    // With n=3 and all cross-similarities = 0, the total sim for each is just
    // 1.0 (self) + 0.0 + 0.0 = 1.0; max_by picks the first (stable).
    let model = EmptyEmbedding;
    let tid = TaskId::new();
    let proposals = vec![
        proposal(tid.clone(), "aaa bbb ccc", 10),
        proposal(tid.clone(), "ddd eee fff", 10),
        proposal(tid.clone(), "ggg hhh iii", 10),
    ];
    let result = ConsensusMedian::resolve(&proposals, Some(&model)).await;
    assert!(
        result.is_some(),
        "must return Some even with fully orthogonal empty embeddings"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn consensus_median_with_embedding_single_proposal_returns_it() {
    let model = FixedEmbedding(vec![1.0, 0.0]);
    let tid = TaskId::new();
    let p = proposal(tid, "sole proposal", 42);
    let proposals = vec![p.clone()];
    let result = ConsensusMedian::resolve(&proposals, Some(&model))
        .await
        .unwrap();
    assert_eq!(result.raw_output, "sole proposal");
}

#[tokio::test(flavor = "multi_thread")]
async fn consensus_median_with_embedding_empty_proposals_returns_none() {
    let model = FixedEmbedding(vec![1.0, 0.0]);
    let result = ConsensusMedian::resolve(&[], Some(&model)).await;
    assert!(result.is_none());
}
