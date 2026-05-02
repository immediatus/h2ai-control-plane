use chrono::Utc;
use h2ai_autonomic::merger::{MergeEngine, MergeOutcome};
use h2ai_state::semilattice::ProposalSet;
use h2ai_types::config::AdapterKind;
use h2ai_types::events::{BranchPrunedEvent, ProposalEvent};
use h2ai_types::identity::{ExplorerId, TaskId};
use h2ai_types::physics::{MergeStrategy, RoleErrorCost, TauValue};

fn adapter() -> AdapterKind {
    AdapterKind::CloudGeneric {
        endpoint: "https://api.test".into(),
        api_key_env: "K".into(),
    }
}

fn proposal(task_id: &TaskId, explorer_id: ExplorerId, output: &str, cost: u64) -> ProposalEvent {
    ProposalEvent {
        task_id: task_id.clone(),
        explorer_id,
        tau: TauValue::new(0.4).unwrap(),
        generation: 0,
        raw_output: output.into(),
        token_cost: cost,
        adapter_kind: adapter(),
        timestamp: Utc::now(),
    }
}

fn pruned(task_id: &TaskId, explorer_id: &ExplorerId) -> BranchPrunedEvent {
    BranchPrunedEvent {
        task_id: task_id.clone(),
        explorer_id: explorer_id.clone(),
        reason: "ADR violation".into(),
        constraint_error_cost: RoleErrorCost::new(0.9).unwrap(),
        violated_constraints: vec![],
        timestamp: Utc::now(),
    }
}

#[tokio::test]
async fn merge_engine_resolves_crdt_when_valid_proposals_exist() {
    let task_id = TaskId::new();
    let mut set = ProposalSet::new();
    set.insert(proposal(&task_id, ExplorerId::new(), "answer A", 10));
    set.insert(proposal(&task_id, ExplorerId::new(), "answer B", 20));

    let outcome =
        MergeEngine::resolve(task_id, set, vec![], MergeStrategy::ScoreOrdered, 0, None).await;
    assert!(matches!(outcome, MergeOutcome::Resolved { .. }));
}

#[tokio::test]
async fn merge_engine_emits_zero_survival_when_all_pruned() {
    let task_id = TaskId::new();
    let explorer_id = ExplorerId::new();
    let mut set = ProposalSet::new();
    set.insert(proposal(&task_id, explorer_id.clone(), "output", 5));
    let pruned_events = vec![pruned(&task_id, &explorer_id)];

    let outcome = MergeEngine::resolve(
        task_id,
        set,
        pruned_events,
        MergeStrategy::ScoreOrdered,
        0,
        None,
    )
    .await;
    assert!(matches!(outcome, MergeOutcome::ZeroSurvival(_)));
}

#[tokio::test]
async fn merge_engine_zero_survival_when_proposal_set_empty() {
    let task_id = TaskId::new();
    let outcome = MergeEngine::resolve(
        task_id,
        ProposalSet::new(),
        vec![],
        MergeStrategy::ScoreOrdered,
        0,
        None,
    )
    .await;
    assert!(matches!(outcome, MergeOutcome::ZeroSurvival(_)));
}

#[tokio::test]
async fn merge_engine_consensus_median_selects_a_proposal() {
    let task_id = TaskId::new();
    let mut set = ProposalSet::new();
    let a = "stateless JWT auth ADR-001 compliant token rotation";
    let b = "blockchain hash table proof-of-work completely different";
    set.insert(proposal(&task_id, ExplorerId::new(), a, 100));
    set.insert(proposal(&task_id, ExplorerId::new(), b, 10));

    let outcome = MergeEngine::resolve(
        task_id,
        set,
        vec![],
        MergeStrategy::ConsensusMedian,
        0,
        None,
    )
    .await;
    if let MergeOutcome::Resolved { resolved, .. } = outcome {
        assert!(
            !resolved.resolved_output.is_empty(),
            "ConsensusMedian must select a non-empty output"
        );
    } else {
        panic!("expected Resolved");
    }
}

#[tokio::test]
async fn merge_engine_resolved_outcome_carries_semilattice_compiled_event() {
    let task_id = TaskId::new();
    let mut set = ProposalSet::new();
    set.insert(proposal(&task_id, ExplorerId::new(), "output", 5));

    let outcome =
        MergeEngine::resolve(task_id, set, vec![], MergeStrategy::ScoreOrdered, 0, None).await;
    if let MergeOutcome::Resolved { compiled, .. } = outcome {
        assert!(!compiled.valid_proposals.is_empty());
    } else {
        panic!("expected Resolved");
    }
}

#[tokio::test]
async fn merge_engine_krum_selects_honest_proposal() {
    // n=5, f=1: quorum satisfied. Krum must not pick the outlier.
    let task_id = TaskId::new();
    let mut set = ProposalSet::new();
    set.insert(proposal(
        &task_id,
        ExplorerId::new(),
        "stateless jwt auth token ADR-001",
        10,
    ));
    set.insert(proposal(
        &task_id,
        ExplorerId::new(),
        "stateless jwt authentication ADR-001",
        10,
    ));
    set.insert(proposal(
        &task_id,
        ExplorerId::new(),
        "jwt stateless auth rotation ADR-001",
        10,
    ));
    set.insert(proposal(
        &task_id,
        ExplorerId::new(),
        "stateless bearer token jwt ADR-001",
        10,
    ));
    set.insert(proposal(
        &task_id,
        ExplorerId::new(),
        "blockchain hash rainbow table wrong",
        10,
    ));

    let outcome = MergeEngine::resolve(
        task_id,
        set,
        vec![],
        MergeStrategy::OutlierResistant { f: 1 },
        0,
        None,
    )
    .await;
    if let MergeOutcome::Resolved { resolved, .. } = outcome {
        assert_ne!(
            resolved.resolved_output, "blockchain hash rainbow table wrong",
            "Krum must not select Byzantine outlier"
        );
    } else {
        panic!("expected Resolved");
    }
}

#[tokio::test]
async fn merge_engine_multi_krum_returns_honest_output() {
    // n=7, f=2, m=3: Multi-Krum selects 3 survivors; merger picks highest-scored.
    let task_id = TaskId::new();
    let mut set = ProposalSet::new();
    let honest = [
        ("stateless jwt auth token ADR-001", 9),
        ("stateless jwt authentication ADR-001", 8),
        ("jwt stateless auth rotation ADR-001", 7),
        ("stateless bearer jwt token ADR-001", 6),
        ("jwt bearer stateless ADR-001", 5),
    ];
    for (text, score) in honest {
        set.insert_scored(
            proposal(&task_id, ExplorerId::new(), text, 10),
            score as f64,
        );
    }
    set.insert(proposal(
        &task_id,
        ExplorerId::new(),
        "blockchain hash wrong",
        10,
    ));
    set.insert(proposal(
        &task_id,
        ExplorerId::new(),
        "redis session wrong expiry",
        10,
    ));

    let outcome = MergeEngine::resolve(
        task_id,
        set,
        vec![],
        MergeStrategy::MultiOutlierResistant { f: 2, m: 3 },
        0,
        None,
    )
    .await;
    if let MergeOutcome::Resolved { resolved, .. } = outcome {
        assert!(
            resolved.resolved_output.contains("jwt")
                || resolved.resolved_output.contains("stateless"),
            "Multi-Krum output should be from honest cluster; got: {}",
            resolved.resolved_output
        );
        assert!(
            !resolved.resolved_output.contains("blockchain")
                && !resolved.resolved_output.contains("redis"),
            "Multi-Krum must not select Byzantine output"
        );
    } else {
        panic!("expected Resolved");
    }
}

#[tokio::test]
async fn merge_engine_krum_quorum_violated_falls_back_to_consensus_median() {
    // n=4, f=1: quorum not satisfied (need 5). Falls back to ConsensusMedian.
    let task_id = TaskId::new();
    let mut set = ProposalSet::new();
    set.insert(proposal(
        &task_id,
        ExplorerId::new(),
        "jwt auth stateless token",
        10,
    ));
    set.insert(proposal(
        &task_id,
        ExplorerId::new(),
        "jwt stateless bearer auth",
        10,
    ));
    set.insert(proposal(
        &task_id,
        ExplorerId::new(),
        "stateless jwt token auth",
        10,
    ));
    set.insert(proposal(
        &task_id,
        ExplorerId::new(),
        "blockchain wrong hash",
        10,
    ));

    let outcome = MergeEngine::resolve(
        task_id,
        set,
        vec![],
        MergeStrategy::OutlierResistant { f: 1 },
        0,
        None,
    )
    .await;
    if let MergeOutcome::Resolved { resolved, .. } = outcome {
        assert!(
            !resolved.resolved_output.is_empty(),
            "quorum-violated Krum must fall back to ConsensusMedian and return non-empty output"
        );
    } else {
        panic!("expected Resolved");
    }
}

#[tokio::test]
async fn merge_engine_krum_incoherent_cluster_falls_back_to_consensus_median() {
    // n=5, f=1: quorum satisfied, but all proposals are maximally diverse (incoherent cluster).
    // cluster_coherent() must return false, triggering ConsensusMedian fallback.
    let task_id = TaskId::new();
    let mut set = ProposalSet::new();
    set.insert(proposal(
        &task_id,
        ExplorerId::new(),
        "alpha bravo charlie delta",
        10,
    ));
    set.insert(proposal(
        &task_id,
        ExplorerId::new(),
        "echo foxtrot golf hotel",
        10,
    ));
    set.insert(proposal(
        &task_id,
        ExplorerId::new(),
        "india juliet kilo lima",
        10,
    ));
    set.insert(proposal(
        &task_id,
        ExplorerId::new(),
        "mike november oscar papa",
        10,
    ));
    set.insert(proposal(
        &task_id,
        ExplorerId::new(),
        "quebec romeo sierra tango",
        10,
    ));

    let outcome = MergeEngine::resolve(
        task_id,
        set,
        vec![],
        MergeStrategy::OutlierResistant { f: 1 },
        0,
        None,
    )
    .await;
    if let MergeOutcome::Resolved { resolved, .. } = outcome {
        assert!(
            !resolved.resolved_output.is_empty(),
            "incoherent-cluster Krum must fall back to ConsensusMedian and return non-empty output"
        );
    } else {
        panic!("expected Resolved");
    }
}

#[tokio::test]
async fn merge_engine_multi_krum_incoherent_cluster_falls_back_to_consensus_median() {
    // n=7, f=2: quorum satisfied, but all proposals are maximally diverse.
    // cluster_coherent() must return false, triggering ConsensusMedian fallback.
    let task_id = TaskId::new();
    let mut set = ProposalSet::new();
    let diverse = [
        "alpha bravo charlie delta echo",
        "foxtrot golf hotel india juliet",
        "kilo lima mike november oscar",
        "papa quebec romeo sierra tango",
        "uniform victor whiskey xray yankee",
        "zulu apple banana cherry date",
        "elderberry fig grape honeydew kiwi",
    ];
    for text in diverse {
        set.insert(proposal(&task_id, ExplorerId::new(), text, 10));
    }

    let outcome = MergeEngine::resolve(
        task_id,
        set,
        vec![],
        MergeStrategy::MultiOutlierResistant { f: 2, m: 3 },
        0,
        None,
    )
    .await;
    if let MergeOutcome::Resolved { resolved, .. } = outcome {
        assert!(
            !resolved.resolved_output.is_empty(),
            "incoherent-cluster MultiKrum must fall back to ConsensusMedian"
        );
    } else {
        panic!("expected Resolved");
    }
}

#[tokio::test]
async fn merge_engine_zero_survival_carries_retry_count() {
    let task_id = TaskId::new();
    let outcome = MergeEngine::resolve(
        task_id,
        ProposalSet::new(),
        vec![],
        MergeStrategy::ScoreOrdered,
        7,
        None,
    )
    .await;
    if let MergeOutcome::ZeroSurvival(event) = outcome {
        assert_eq!(event.retry_count, 7);
    } else {
        panic!("expected ZeroSurvival");
    }
}

#[tokio::test]
async fn merge_engine_multi_krum_quorum_violated_falls_back_to_consensus_median() {
    // n=6, f=2: quorum not satisfied (need 7). Falls back to ConsensusMedian.
    let task_id = TaskId::new();
    let mut set = ProposalSet::new();
    set.insert(proposal(
        &task_id,
        ExplorerId::new(),
        "jwt auth stateless token",
        10,
    ));
    set.insert(proposal(
        &task_id,
        ExplorerId::new(),
        "jwt stateless bearer auth",
        10,
    ));
    set.insert(proposal(
        &task_id,
        ExplorerId::new(),
        "stateless jwt token auth",
        10,
    ));
    set.insert(proposal(
        &task_id,
        ExplorerId::new(),
        "auth jwt token stateless bearer",
        10,
    ));
    set.insert(proposal(
        &task_id,
        ExplorerId::new(),
        "token stateless jwt authentication",
        10,
    ));
    set.insert(proposal(
        &task_id,
        ExplorerId::new(),
        "blockchain wrong hash table",
        10,
    ));

    let outcome = MergeEngine::resolve(
        task_id,
        set,
        vec![],
        MergeStrategy::MultiOutlierResistant { f: 2, m: 3 },
        0,
        None,
    )
    .await;
    if let MergeOutcome::Resolved { resolved, .. } = outcome {
        assert!(
            !resolved.resolved_output.is_empty(),
            "quorum-violated MultiKrum must fall back to ConsensusMedian and return non-empty output"
        );
    } else {
        panic!("expected Resolved");
    }
}

#[tokio::test]
async fn merge_resolved_event_contains_timing_fields() {
    let task_id = TaskId::new();
    let mut set = ProposalSet::new();
    set.insert(proposal(&task_id, ExplorerId::new(), "answer A", 10));
    set.insert(proposal(&task_id, ExplorerId::new(), "answer B", 20));
    set.insert(proposal(&task_id, ExplorerId::new(), "answer C", 15));

    let outcome =
        MergeEngine::resolve(task_id, set, vec![], MergeStrategy::ScoreOrdered, 0, None).await;
    if let MergeOutcome::Resolved { compiled, .. } = outcome {
        assert!(
            compiled.merge_elapsed_secs.is_some(),
            "merge_elapsed_secs must be populated"
        );
        assert_eq!(compiled.n_input_proposals, 3);
    } else {
        panic!("expected Resolved");
    }
}

#[tokio::test]
async fn merge_n_input_proposals_includes_pruned_count() {
    let task_id = TaskId::new();
    let explorer_id = ExplorerId::new();
    let mut set = ProposalSet::new();
    set.insert(proposal(
        &task_id,
        ExplorerId::new(),
        "surviving answer",
        10,
    ));
    set.insert(proposal(&task_id, explorer_id.clone(), "pruned answer", 10));
    let pruned_events = vec![pruned(&task_id, &explorer_id)];

    // 2 proposals in set + 1 pruned event → n_input_proposals = proposals.len() + pruned.len() = 3
    let outcome = MergeEngine::resolve(
        task_id,
        set,
        pruned_events,
        MergeStrategy::ScoreOrdered,
        0,
        None,
    )
    .await;
    if let MergeOutcome::Resolved { compiled, .. } = outcome {
        assert_eq!(
            compiled.n_input_proposals, 3,
            "must count proposals in set plus pruned events"
        );
    } else {
        panic!("expected Resolved");
    }
}

#[tokio::test]
async fn krum_selects_semantically_central_with_semantic_variant() {
    use h2ai_state::krum::krum_select_semantic;

    let task_id = TaskId::new();
    let kind = AdapterKind::CloudGeneric {
        endpoint: "https://api.test".into(),
        api_key_env: "K".into(),
    };
    let make = |text: &str| ProposalEvent {
        task_id: task_id.clone(),
        explorer_id: ExplorerId::new(),
        tau: TauValue::new(0.4).unwrap(),
        generation: 0,
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
        make("completely unrelated output about blockchain cryptocurrency"),
    ];

    let result = krum_select_semantic(&proposals, 1, None).await;
    assert!(result.is_some(), "krum_select_semantic must return Some");
    let selected = result.unwrap();
    assert_ne!(
        selected.raw_output, "completely unrelated output about blockchain cryptocurrency",
        "Krum must not select the outlier"
    );
}
