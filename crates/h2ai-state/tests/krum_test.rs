use chrono::Utc;
use h2ai_state::krum::{
    cluster_coherent, krum_select, mean_pairwise_distance, min_quorum, multi_krum_select,
    quorum_satisfied,
};
use h2ai_types::config::AdapterKind;
use h2ai_types::events::ProposalEvent;
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
    assert_eq!(mean_pairwise_distance(&[prop("stateless jwt")], None).await, 0.0);
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
    assert!((d - 1.0).abs() < 1e-9, "totally disjoint proposals must have distance 1.0, got {d}");
}

#[tokio::test]
async fn mean_pairwise_distance_zero_for_identical_pair() {
    let text = "stateless jwt auth token";
    let proposals = vec![prop(text), prop(text)];
    let d = mean_pairwise_distance(&proposals, None).await;
    assert!(d.abs() < 1e-9, "identical proposals must have distance 0.0, got {d}");
}

#[tokio::test]
async fn cluster_coherent_tight_cluster_returns_true() {
    // Proposals with substantial token overlap → mean distance well below 0.7
    let proposals = vec![
        prop("stateless jwt auth token ADR-001"),
        prop("stateless jwt authentication token ADR-001"),
        prop("jwt stateless auth rotation ADR-001"),
        prop("stateless bearer jwt auth ADR-001"),
        prop("jwt bearer stateless authentication ADR-001"),
    ];
    assert!(
        cluster_coherent(&proposals, None).await,
        "tight jwt cluster must be cluster_coherent"
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
async fn cluster_coherent_single_proposal_returns_true() {
    // Zero distance for 1 proposal → trivially coherent
    assert!(cluster_coherent(&[prop("anything here")], None).await);
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
