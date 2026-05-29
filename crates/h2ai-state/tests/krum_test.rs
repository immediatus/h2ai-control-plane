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
use h2ai_state::krum::{
    cluster_coherent, krum_index, krum_score_subset, krum_select_semantic, mean_pairwise_distance,
    min_quorum, multi_krum_select_semantic, quorum_satisfied,
};
use h2ai_types::config::AdapterKind;
use h2ai_types::events::ProposalEvent;
use h2ai_types::identity::{ExplorerId, TaskId};
use h2ai_types::sizing::TauValue;
use std::collections::HashSet;

fn tokenize(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(str::to_lowercase)
        .filter(|t| t.len() > 1)
        .collect()
}

fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let intersection = a.intersection(b).count() as f64;
    let union = a.union(b).count() as f64;
    intersection / union
}

fn build_token_sets(proposals: &[ProposalEvent]) -> Vec<HashSet<String>> {
    proposals.iter().map(|p| tokenize(&p.raw_output)).collect()
}

fn jaccard_distance_matrix(token_sets: &[HashSet<String>]) -> Vec<Vec<f64>> {
    let n = token_sets.len();
    let mut d = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let dist = 1.0 - jaccard(&token_sets[i], &token_sets[j]);
            d[i][j] = dist;
            d[j][i] = dist;
        }
    }
    d
}

fn krum_select(proposals: &[ProposalEvent], f: usize) -> Option<&ProposalEvent> {
    if proposals.is_empty() {
        return None;
    }
    if f == 0 {
        return proposals.first();
    }
    if !quorum_satisfied(proposals.len(), f) {
        return None;
    }
    let distances = jaccard_distance_matrix(&build_token_sets(proposals));
    let k = proposals.len() - f - 2;
    krum_index(&distances, k).map(|i| &proposals[i])
}

fn multi_krum_select(proposals: &[ProposalEvent], f: usize, m: usize) -> Vec<&ProposalEvent> {
    if proposals.is_empty() || m == 0 {
        return vec![];
    }
    if f == 0 {
        return proposals.iter().take(m).collect();
    }
    if !quorum_satisfied(proposals.len(), f) {
        return vec![];
    }
    let distances = jaccard_distance_matrix(&build_token_sets(proposals));
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
            model: None,
            provider: Default::default(),
        },
        timestamp: Utc::now(),
    }
}

// ── quorum helpers ────────────────────────────────────────────────────────────

#[test]
fn min_quorum_formula() {
    assert_eq!(min_quorum(0), 3);
    assert_eq!(min_quorum(1), 5);
    assert_eq!(min_quorum(2), 7);
    assert_eq!(min_quorum(3), 9);
}

#[test]
fn quorum_satisfied_boundary() {
    assert!(!quorum_satisfied(4, 1), "n=4 < 5 should fail for f=1");
    assert!(quorum_satisfied(5, 1), "n=5 = 2*1+3 should pass for f=1");
    assert!(!quorum_satisfied(6, 2), "n=6 < 7 should fail for f=2");
    assert!(quorum_satisfied(7, 2), "n=7 = 2*2+3 should pass for f=2");
}

// ── krum_select ───────────────────────────────────────────────────────────────

#[test]
fn krum_empty_returns_none() {
    assert!(krum_select(&[], 1).is_none());
}

#[test]
fn krum_f_zero_returns_first() {
    let proposals = vec![prop("alpha stateless jwt"), prop("beta redis session")];
    let result = krum_select(&proposals, 0).unwrap();
    assert_eq!(result.raw_output, "alpha stateless jwt");
}

#[test]
fn krum_quorum_violated_returns_none() {
    // n=4, f=1: need n ≥ 5, so this must return None
    let proposals: Vec<_> = (0..4).map(|_| prop("stateless jwt auth")).collect();
    assert!(
        krum_select(&proposals, 1).is_none(),
        "n=4 < 5 must return None for f=1"
    );
}

#[test]
fn krum_selects_honest_against_single_outlier() {
    // n=5, f=1: quorum satisfied.
    // 4 honest proposals clustered around "stateless jwt auth"; 1 Byzantine outlier.
    let honest_texts = [
        "stateless jwt auth token validation ADR-001",
        "stateless jwt authentication token refresh ADR-001",
        "jwt stateless token auth mechanism ADR-001",
        "stateless authentication jwt bearer token ADR-001",
    ];
    let byzantine = "completely wrong redis session store rainbow table";
    let mut proposals: Vec<ProposalEvent> = honest_texts.iter().map(|t| prop(t)).collect();
    proposals.push(prop(byzantine));

    let selected = krum_select(&proposals, 1).unwrap();
    assert_ne!(
        selected.raw_output, byzantine,
        "Krum must not select Byzantine outlier; selected: {}",
        selected.raw_output
    );
}

#[test]
fn krum_selects_honest_against_two_byzantine_outliers() {
    // n=7, f=2: quorum satisfied.
    // 5 honest proposals + 2 Byzantine outliers.
    let honest_texts = [
        "stateless jwt auth token ADR-001",
        "stateless jwt authentication mechanism ADR-001",
        "jwt stateless auth rotation ADR-001",
        "stateless token authentication jwt ADR-001",
        "jwt bearer stateless authentication ADR-001",
    ];
    let byz1 = "blockchain proof-of-work cryptocurrency mining hash";
    let byz2 = "redis session store sliding window expiry cookie";
    let mut proposals: Vec<ProposalEvent> = honest_texts.iter().map(|t| prop(t)).collect();
    proposals.push(prop(byz1));
    proposals.push(prop(byz2));

    let selected = krum_select(&proposals, 2).unwrap();
    assert!(
        selected.raw_output != byz1 && selected.raw_output != byz2,
        "Krum must not select a Byzantine outlier; selected: {}",
        selected.raw_output
    );
}

// ── multi_krum_select ─────────────────────────────────────────────────────────

#[test]
fn multi_krum_empty_returns_empty() {
    assert!(multi_krum_select(&[], 1, 3).is_empty());
}

#[test]
fn multi_krum_m_zero_returns_empty() {
    let p = vec![prop("stateless jwt")];
    assert!(multi_krum_select(&p, 0, 0).is_empty());
}

#[test]
fn multi_krum_quorum_violated_returns_empty() {
    let proposals: Vec<_> = (0..4).map(|_| prop("stateless jwt auth")).collect();
    assert!(multi_krum_select(&proposals, 1, 2).is_empty());
}

#[test]
fn multi_krum_returns_m_survivors_all_honest() {
    // n=7, f=2, m=3: must return 3 honest proposals, none Byzantine.
    let honest_texts = [
        "stateless jwt auth token ADR-001",
        "stateless jwt authentication ADR-001",
        "jwt stateless token rotation ADR-001",
        "stateless auth mechanism jwt ADR-001",
        "jwt bearer stateless auth ADR-001",
    ];
    let byz1 = "blockchain hash completely wrong";
    let byz2 = "redis session expiry cookie wrong";
    let mut proposals: Vec<ProposalEvent> = honest_texts.iter().map(|t| prop(t)).collect();
    proposals.push(prop(byz1));
    proposals.push(prop(byz2));

    let survivors = multi_krum_select(&proposals, 2, 3);
    assert_eq!(survivors.len(), 3, "must return exactly m=3 survivors");
    for s in &survivors {
        assert!(
            s.raw_output != byz1 && s.raw_output != byz2,
            "Multi-Krum survivor must not be Byzantine; got: {}",
            s.raw_output
        );
    }
}

#[test]
fn multi_krum_f_zero_returns_first_m() {
    let proposals: Vec<_> = ["alpha jwt", "beta auth", "gamma stateless"]
        .iter()
        .map(|t| prop(t))
        .collect();
    let survivors = multi_krum_select(&proposals, 0, 2);
    assert_eq!(survivors.len(), 2);
    assert_eq!(survivors[0].raw_output, "alpha jwt");
    assert_eq!(survivors[1].raw_output, "beta auth");
}

#[test]
fn krum_all_identical_proposals_returns_some() {
    // When all proposals have identical content, Krum should still return one.
    // f=0 path → first element; this also confirms no panic on zero-distance inputs.
    let proposals: Vec<_> = (0..5).map(|_| prop("stateless jwt auth")).collect();
    assert!(
        krum_select(&proposals, 0).is_some(),
        "krum with f=0 must return first proposal even when all identical"
    );
    // With f=1 and n=5 (quorum satisfied), all proposals are at distance 0 from each other.
    // The algorithm must still return Some (any proposal is correct — all are honest).
    assert!(
        krum_select(&proposals, 1).is_some(),
        "krum must return Some when quorum satisfied and all proposals identical"
    );
}

#[test]
fn multi_krum_m_larger_than_selectable_returns_partial() {
    // n=5, f=1, m=10: the inner loop exits when remaining.len() <= f+2 = 3,
    // so we can select at most n - (f+2) = 5 - 3 = 2 proposals.
    let honest_texts = [
        "stateless jwt auth token ADR-001",
        "stateless jwt authentication ADR-001",
        "jwt stateless token rotation ADR-001",
        "stateless auth mechanism jwt ADR-001",
        "jwt bearer stateless auth ADR-001",
    ];
    let proposals: Vec<_> = honest_texts.iter().map(|t| prop(t)).collect();
    let survivors = multi_krum_select(&proposals, 1, 10);
    // Should return fewer than m (capped by selectable count), but never panic.
    assert!(
        survivors.len() < 10,
        "multi_krum with m > selectable should return fewer than m"
    );
    assert!(!survivors.is_empty(), "should return at least one proposal");
}

// ── cluster_coherent / mean_pairwise_distance ─────────────────────────────────

#[tokio::test]
async fn mean_pairwise_distance_zero_for_single_proposal() {
    assert_eq!(
        mean_pairwise_distance(&[prop("stateless jwt")], None).await,
        0.0
    );
}

#[tokio::test]
async fn mean_pairwise_distance_zero_for_empty() {
    assert_eq!(mean_pairwise_distance(&[], None).await, 0.0);
}

#[tokio::test]
async fn mean_pairwise_distance_one_for_totally_disjoint_pair() {
    // No shared tokens → Jaccard = 0 → distance = 1.0
    let proposals = vec![prop("alpha bravo charlie"), prop("delta echo foxtrot")];
    let d = mean_pairwise_distance(&proposals, None).await;
    assert!(
        (d - 1.0).abs() < 1e-9,
        "totally disjoint proposals must have distance 1.0, got {d}"
    );
}

#[tokio::test]
async fn mean_pairwise_distance_zero_for_identical_pair() {
    let text = "stateless jwt auth token";
    let proposals = vec![prop(text), prop(text)];
    let d = mean_pairwise_distance(&proposals, None).await;
    assert!(
        d.abs() < 1e-9,
        "identical proposals must have distance 0.0, got {d}"
    );
}

#[tokio::test]
async fn cluster_coherent_no_embedding_always_returns_false() {
    // Without an embedding model, cluster_coherent returns false regardless of content.
    // Token Jaccard cannot reliably satisfy the BFT cluster assumption for LLM outputs;
    // the merger must fall back to ConsensusMedian in this configuration.
    let proposals = vec![
        prop("stateless jwt auth token ADR-001"),
        prop("stateless jwt authentication token ADR-001"),
        prop("jwt stateless auth rotation ADR-001"),
        prop("stateless bearer jwt auth ADR-001"),
        prop("jwt bearer stateless authentication ADR-001"),
    ];
    assert!(
        !cluster_coherent(&proposals, None).await,
        "cluster_coherent must return false when no embedding model is provided"
    );
}

#[tokio::test]
async fn cluster_coherent_diverse_proposals_returns_false() {
    // Each proposal shares no tokens with any other → mean distance = 1.0 > MAX_CLUSTER_DIAMETER
    let proposals = vec![
        prop("alpha bravo charlie delta"),
        prop("echo foxtrot golf hotel"),
        prop("india juliet kilo lima"),
        prop("mike november oscar papa"),
        prop("quebec romeo sierra tango"),
    ];
    assert!(
        !cluster_coherent(&proposals, None).await,
        "maximally diverse proposals must NOT be cluster_coherent"
    );
}

#[tokio::test]
async fn cluster_coherent_single_proposal_no_embedding_returns_false() {
    // No embedding model → false even for a single proposal.
    // The merger falls back to ConsensusMedian (which also picks the one proposal).
    assert!(!cluster_coherent(&[prop("anything here")], None).await);
}

#[test]
fn multi_krum_m_equals_one_with_f_nonzero() {
    // n=5, f=1, m=1: should return exactly 1 best proposal.
    let honest_texts = [
        "stateless jwt auth token ADR-001",
        "stateless jwt authentication ADR-001",
        "jwt stateless token rotation ADR-001",
        "stateless auth mechanism ADR-001",
    ];
    let byzantine = "blockchain hash completely wrong different";
    let mut proposals: Vec<_> = honest_texts.iter().map(|t| prop(t)).collect();
    proposals.push(prop(byzantine));
    let survivors = multi_krum_select(&proposals, 1, 1);
    assert_eq!(survivors.len(), 1);
    assert_ne!(
        survivors[0].raw_output, byzantine,
        "multi_krum m=1 must not select Byzantine proposal"
    );
}

// ── Mock EmbeddingModel ───────────────────────────────────────────────────────

/// Fixed-vector mock: every text maps to the same pre-set vector.
struct FixedEmbedding(Vec<f32>);
impl EmbeddingModel for FixedEmbedding {
    fn embed(&self, _text: &str) -> Vec<f32> {
        self.0.clone()
    }
}

/// Variable-vector mock: selects a vector by hashing the first char of the text.
/// Texts starting with the same character get the same embedding.
struct FirstCharEmbedding;
impl EmbeddingModel for FirstCharEmbedding {
    fn embed(&self, text: &str) -> Vec<f32> {
        match text.chars().next() {
            Some('a') => vec![1.0, 0.0, 0.0],
            Some('b') => vec![0.9, 0.436, 0.0], // ~cos(~25°) close to 'a'
            Some('c') => vec![0.9, 0.436, 0.0],
            Some('d') => vec![0.9, 0.436, 0.0],
            Some('e') => vec![0.9, 0.436, 0.0],
            // outlier: orthogonal to everything above
            _ => vec![0.0, 0.0, 1.0],
        }
    }
}

/// Empty-embedding mock: always returns an empty vec.
struct EmptyEmbedding;
impl EmbeddingModel for EmptyEmbedding {
    fn embed(&self, _text: &str) -> Vec<f32> {
        vec![]
    }
}

// ── krum_select_semantic ──────────────────────────────────────────────────────

#[tokio::test]
async fn krum_select_semantic_no_embedding_returns_none() {
    let proposals = vec![prop("stateless jwt auth")];
    let result = krum_select_semantic(&proposals, 1, None).await;
    assert!(result.is_none(), "None embedding model must return None");
}

#[tokio::test]
async fn krum_select_semantic_empty_proposals_returns_none() {
    let model = FixedEmbedding(vec![1.0, 0.0]);
    let result = krum_select_semantic(&[], 1, Some(&model)).await;
    assert!(result.is_none(), "empty proposals must return None");
}

#[tokio::test]
async fn krum_select_semantic_f_zero_returns_first() {
    let model = FixedEmbedding(vec![1.0, 0.0]);
    let proposals = vec![prop("first proposal"), prop("second proposal")];
    let result = krum_select_semantic(&proposals, 0, Some(&model)).await;
    assert!(result.is_some());
    assert_eq!(result.unwrap().raw_output, "first proposal");
}

#[tokio::test]
async fn krum_select_semantic_quorum_fail_returns_none() {
    let model = FixedEmbedding(vec![1.0, 0.0]);
    // n=4, f=1: need n >= 5
    let proposals: Vec<_> = (0..4).map(|i| prop(&format!("proposal {i}"))).collect();
    let result = krum_select_semantic(&proposals, 1, Some(&model)).await;
    assert!(result.is_none(), "quorum not satisfied must return None");
}

#[tokio::test(flavor = "multi_thread")]
async fn krum_select_semantic_real_selection_rejects_outlier() {
    // n=5, f=1: quorum satisfied.
    // 4 proposals with 'a'-prefix embeddings cluster together; 1 'z'-prefix is outlier.
    let model = FirstCharEmbedding;
    let proposals = vec![
        prop("alpha cluster jwt"),
        prop("alpha stateless auth"),
        prop("alpha bearer token"),
        prop("alpha rotation adr"),
        prop("zzz outlier blockchain hash"),
    ];
    let result = krum_select_semantic(&proposals, 1, Some(&model)).await;
    assert!(result.is_some());
    assert_ne!(
        result.unwrap().raw_output,
        "zzz outlier blockchain hash",
        "semantic krum must reject the outlier"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn krum_select_semantic_empty_embedding_fallback() {
    // With an empty-embedding model, identical texts → sim=1.0; different → 0.0
    let model = EmptyEmbedding;
    // n=5, f=1 — quorum satisfied; all identical → all at distance 0 from each other
    let proposals: Vec<_> = (0..5).map(|_| prop("identical text")).collect();
    let result = krum_select_semantic(&proposals, 1, Some(&model)).await;
    assert!(
        result.is_some(),
        "should return Some for identical proposals"
    );
}

// ── multi_krum_select_semantic ────────────────────────────────────────────────

#[tokio::test]
async fn multi_krum_select_semantic_no_embedding_returns_empty() {
    let proposals = vec![prop("stateless jwt")];
    let result = multi_krum_select_semantic(&proposals, 1, 1, None).await;
    assert!(result.is_empty(), "None embedding must return empty vec");
}

#[tokio::test]
async fn multi_krum_select_semantic_empty_proposals_returns_empty() {
    let model = FixedEmbedding(vec![1.0, 0.0]);
    let result = multi_krum_select_semantic(&[], 1, 2, Some(&model)).await;
    assert!(result.is_empty());
}

#[tokio::test]
async fn multi_krum_select_semantic_m_zero_returns_empty() {
    let model = FixedEmbedding(vec![1.0, 0.0]);
    let proposals = vec![prop("stateless jwt")];
    let result = multi_krum_select_semantic(&proposals, 1, 0, Some(&model)).await;
    assert!(result.is_empty());
}

#[tokio::test]
async fn multi_krum_select_semantic_f_zero_returns_first_m() {
    let model = FixedEmbedding(vec![1.0, 0.0]);
    let proposals: Vec<_> = ["alpha", "beta", "gamma"].iter().map(|t| prop(t)).collect();
    let result = multi_krum_select_semantic(&proposals, 0, 2, Some(&model)).await;
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].raw_output, "alpha");
    assert_eq!(result[1].raw_output, "beta");
}

#[tokio::test]
async fn multi_krum_select_semantic_quorum_fail_returns_empty() {
    let model = FixedEmbedding(vec![1.0, 0.0]);
    // n=4, f=1: need n >= 5
    let proposals: Vec<_> = (0..4).map(|i| prop(&format!("p{i}"))).collect();
    let result = multi_krum_select_semantic(&proposals, 1, 2, Some(&model)).await;
    assert!(result.is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn multi_krum_select_semantic_real_selection_rejects_outliers() {
    // n=7, f=2, m=3: should select 3 honest proposals.
    let model = FirstCharEmbedding;
    let proposals = vec![
        prop("alpha cluster jwt auth"),
        prop("alpha stateless bearer"),
        prop("alpha rotation adr compliant"),
        prop("alpha jwt token refresh"),
        prop("alpha bearer stateless auth"),
        prop("zzz outlier blockchain hash one"),
        prop("zzz outlier redis session two"),
    ];
    let result = multi_krum_select_semantic(&proposals, 2, 3, Some(&model)).await;
    assert_eq!(result.len(), 3);
    for s in &result {
        assert!(
            !s.raw_output.starts_with("zzz"),
            "must not select outlier; got: {}",
            s.raw_output
        );
    }
}

// ── cluster_coherent with embedding model ─────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn cluster_coherent_with_tight_embedding_returns_true() {
    // All proposals embed to the same vector → distance 0 → coherent
    let model = FixedEmbedding(vec![1.0, 0.0, 0.0]);
    let proposals: Vec<_> = (0..5).map(|_| prop("stateless jwt auth")).collect();
    let result = cluster_coherent(&proposals, Some(&model)).await;
    assert!(result, "identical embeddings must be cluster_coherent");
}

#[tokio::test(flavor = "multi_thread")]
async fn cluster_coherent_with_dispersed_embedding_returns_false() {
    // Orthogonal embeddings → cosine similarity 0 → distance 1.0 > MAX_CLUSTER_DIAMETER
    struct AxisEmbedding;
    impl EmbeddingModel for AxisEmbedding {
        fn embed(&self, text: &str) -> Vec<f32> {
            match text.chars().next() {
                Some('a') => vec![1.0, 0.0, 0.0],
                Some('b') => vec![0.0, 1.0, 0.0],
                _ => vec![0.0, 0.0, 1.0],
            }
        }
    }
    let model = AxisEmbedding;
    let proposals = vec![prop("alpha one"), prop("beta two"), prop("charlie three")];
    let result = cluster_coherent(&proposals, Some(&model)).await;
    assert!(
        !result,
        "orthogonal embeddings must NOT be cluster_coherent"
    );
}

// ── mean_pairwise_distance with embedding model ───────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn mean_pairwise_distance_with_identical_embeddings_is_zero() {
    let model = FixedEmbedding(vec![1.0, 0.0]);
    let proposals: Vec<_> = (0..3).map(|_| prop("jwt stateless")).collect();
    let d = mean_pairwise_distance(&proposals, Some(&model)).await;
    assert!(d.abs() < 1e-9, "identical embeddings → distance 0, got {d}");
}

#[tokio::test(flavor = "multi_thread")]
async fn mean_pairwise_distance_with_empty_embeddings_uses_equality_fallback() {
    // EmptyEmbedding → falls back to exact string comparison.
    // All identical → sim=1.0, distance=0.0
    let model = EmptyEmbedding;
    let proposals = vec![prop("same text"), prop("same text"), prop("same text")];
    let d = mean_pairwise_distance(&proposals, Some(&model)).await;
    assert!(
        d.abs() < 1e-9,
        "identical strings with empty embedding → 0.0, got {d}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn mean_pairwise_distance_with_empty_embedding_different_strings() {
    // EmptyEmbedding + different strings → sim=0.0, distance=1.0
    let model = EmptyEmbedding;
    let proposals = vec![prop("aaa bbb"), prop("ccc ddd")];
    let d = mean_pairwise_distance(&proposals, Some(&model)).await;
    assert!(
        (d - 1.0).abs() < 1e-9,
        "different strings with empty embedding → 1.0, got {d}"
    );
}
