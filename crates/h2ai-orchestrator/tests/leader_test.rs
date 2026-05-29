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
use h2ai_orchestrator::leader::{
    assign_follower_aspects, eig_score, fnv1a, select_best_and_runner_up, should_rotate,
    update_credibility, BeliefRecord, LeaderState,
};
use h2ai_types::identity::ExplorerId;

// ── should_rotate ──────────────────────────────────────────────────────────────

#[test]
fn should_rotate_fires_at_threshold() {
    let state = LeaderState {
        term: 2,
        leader_explorer_id: ExplorerId::new(),
        prior_proposal: "proposal".into(),
        socratic_question: "why?".into(),
        confidence_history: vec![0.5, 0.5],
        stagnation_count: 1,
        belief_buffer: vec![],
        credibility_score: 1.0,
        follower_aspects: vec![],
    };
    assert!(should_rotate(&state, 0.02, 1));
    assert!(!should_rotate(&state, 0.02, 2));
}

// ── eig_score / fnv1a ──────────────────────────────────────────────────────────

#[test]
fn belief_buffer_dedup_skips_hash_collision() {
    let record = BeliefRecord {
        question_hash: fnv1a("What if we simplify the design?"),
        question_text: "What if we simplify the design?".into(),
        outcome_scores: vec![0.4],
    };
    let buffer = vec![record];
    let score = eig_score(
        "What if we simplify the design?",
        &["C1".to_string(), "C2".to_string()],
        &buffer,
    );
    assert_eq!(score, 0.0);
}

#[test]
fn eig_score_ranks_diverse_question_higher() {
    let buffer: Vec<BeliefRecord> = vec![];
    let constraints = vec!["C1".to_string(), "C2".to_string(), "C3".to_string()];
    let score_diverse = eig_score("What about C1 and C3 interaction?", &constraints, &buffer);
    let score_narrow = eig_score("What about C1?", &constraints, &buffer);
    assert!(score_diverse > score_narrow);
}

// ── update_credibility ─────────────────────────────────────────────────────────

#[test]
fn credibility_clamps_at_bounds() {
    let score = update_credibility(0.0, false, 0.2);
    assert_eq!(score, 0.0);
    let score = update_credibility(1.0, true, 0.2);
    assert_eq!(score, 1.0);
}

// ── assign_follower_aspects ────────────────────────────────────────────────────

#[test]
fn follower_aspects_round_robin_over_clusters() {
    let aspects = assign_follower_aspects(&["C1".to_string(), "C2".to_string()], 4);
    assert_eq!(aspects.len(), 4);
    assert_eq!(aspects[0], aspects[2]);
    assert_eq!(aspects[1], aspects[3]);
}

// ── select_best_and_runner_up ─────────────────────────────────────────────────

#[test]
fn select_best_and_runner_up_returns_ordered_pair() {
    let scores = vec![
        (ExplorerId::new(), 0.6f64),
        (ExplorerId::new(), 0.8f64),
        (ExplorerId::new(), 0.5f64),
    ];
    let (winner, runner_up) = select_best_and_runner_up(&scores).unwrap();
    assert_eq!(winner, scores[1].0);
    assert_eq!(runner_up, Some(scores[0].0.clone()));
}

// ── LeaderState::to_snapshot ──────────────────────────────────────────────────

#[test]
fn to_snapshot_copies_all_fields() {
    use h2ai_orchestrator::leader::LeaderState;
    let leader_id = ExplorerId::new();
    let state = LeaderState {
        term: 3,
        leader_explorer_id: leader_id.clone(),
        prior_proposal: "my proposal".into(),
        socratic_question: "What if we retry?".into(),
        confidence_history: vec![0.5, 0.7],
        stagnation_count: 2,
        belief_buffer: vec![BeliefRecord {
            question_hash: fnv1a("old question"),
            question_text: "old question".into(),
            outcome_scores: vec![0.6],
        }],
        credibility_score: 0.75,
        follower_aspects: vec!["throughput".into()],
    };
    let violated = vec!["C1".to_string(), "C2".to_string()];
    let snap = state.to_snapshot(violated.clone());

    assert_eq!(snap.term, 3);
    assert_eq!(snap.leader_explorer_id, leader_id);
    assert_eq!(snap.socratic_question, "What if we retry?");
    assert_eq!(snap.prior_proposal, "my proposal");
    assert!((snap.credibility_score - 0.75).abs() < 1e-9);
    assert_eq!(snap.follower_aspects, vec!["throughput"]);
    assert_eq!(snap.violated_constraints, violated);
    assert_eq!(snap.belief_buffer_questions, vec!["old question"]);
}

// ── build_leader_prefix ───────────────────────────────────────────────────────

#[test]
fn build_leader_prefix_for_leader_slot_contains_question() {
    use h2ai_orchestrator::leader::{build_leader_prefix, LeaderContextSnapshot};
    let leader_id = ExplorerId::new();
    let snap = LeaderContextSnapshot {
        term: 1,
        leader_explorer_id: leader_id.clone(),
        socratic_question: "Why does the system fail?".into(),
        prior_proposal: "Use caching.".into(),
        credibility_score: 0.9,
        follower_aspects: vec!["latency".into()],
        violated_constraints: vec!["C1".into()],
        belief_buffer_questions: vec![],
    };
    let prefix = build_leader_prefix(&snap, &leader_id);
    assert!(
        prefix.contains("LEADER CONTEXT"),
        "must contain LEADER CONTEXT"
    );
    assert!(prefix.contains("Why does the system fail?"));
    assert!(prefix.contains("Use caching."));
    assert!(prefix.contains("C1"));
}

#[test]
fn build_leader_prefix_for_follower_slot_contains_follower_context() {
    use h2ai_orchestrator::leader::{build_leader_prefix, LeaderContextSnapshot};
    let leader_id = ExplorerId::new();
    let follower_id = ExplorerId::new();
    let snap = LeaderContextSnapshot {
        term: 2,
        leader_explorer_id: leader_id.clone(),
        socratic_question: "Is the algorithm correct?".into(),
        prior_proposal: "Leader's answer".into(),
        credibility_score: 0.6,
        follower_aspects: vec![],
        violated_constraints: vec![],
        belief_buffer_questions: vec![],
    };
    let prefix = build_leader_prefix(&snap, &follower_id);
    assert!(
        prefix.contains("FOLLOWER CONTEXT"),
        "must be follower context"
    );
    assert!(prefix.contains("Is the algorithm correct?"));
    assert!(
        !prefix.contains("Leader's answer"),
        "follower must not see prior proposal"
    );
}

#[test]
fn build_leader_prefix_with_belief_buffer_questions_includes_history() {
    use h2ai_orchestrator::leader::{build_leader_prefix, LeaderContextSnapshot};
    let leader_id = ExplorerId::new();
    let snap = LeaderContextSnapshot {
        term: 1,
        leader_explorer_id: leader_id.clone(),
        socratic_question: "New question".into(),
        prior_proposal: "prior".into(),
        credibility_score: 0.8,
        follower_aspects: vec![],
        violated_constraints: vec![],
        belief_buffer_questions: vec!["Question from wave 1".into()],
    };
    let prefix = build_leader_prefix(&snap, &leader_id);
    assert!(prefix.contains("Question from wave 1"));
}

// ── build_follower_prefix_with_aspect ─────────────────────────────────────────

#[test]
fn build_follower_prefix_with_high_credibility_no_warning() {
    use h2ai_orchestrator::leader::{build_follower_prefix_with_aspect, LeaderContextSnapshot};
    let snap = LeaderContextSnapshot {
        term: 1,
        leader_explorer_id: ExplorerId::new(),
        socratic_question: "Is caching the right call?".into(),
        prior_proposal: String::new(),
        credibility_score: 0.85,
        follower_aspects: vec!["throughput".into()],
        violated_constraints: vec![],
        belief_buffer_questions: vec![],
    };
    let prefix = build_follower_prefix_with_aspect(&snap, 0, 0.5);
    assert!(prefix.contains("FOLLOWER CONTEXT"));
    assert!(prefix.contains("throughput"), "should have assigned aspect");
    assert!(
        !prefix.contains("low-confidence"),
        "high credibility → no warning"
    );
}

#[test]
fn build_follower_prefix_with_low_credibility_shows_warning() {
    use h2ai_orchestrator::leader::{build_follower_prefix_with_aspect, LeaderContextSnapshot};
    let snap = LeaderContextSnapshot {
        term: 1,
        leader_explorer_id: ExplorerId::new(),
        socratic_question: "What if we're wrong?".into(),
        prior_proposal: String::new(),
        credibility_score: 0.2,
        follower_aspects: vec!["consistency".into()],
        violated_constraints: vec![],
        belief_buffer_questions: vec![],
    };
    let prefix = build_follower_prefix_with_aspect(&snap, 0, 0.5);
    assert!(
        prefix.contains("low-confidence"),
        "low credibility → warning shown"
    );
}

#[test]
fn build_follower_prefix_out_of_range_slot_uses_default_aspect() {
    use h2ai_orchestrator::leader::{build_follower_prefix_with_aspect, LeaderContextSnapshot};
    let snap = LeaderContextSnapshot {
        term: 1,
        leader_explorer_id: ExplorerId::new(),
        socratic_question: "How?".into(),
        prior_proposal: String::new(),
        credibility_score: 0.8,
        follower_aspects: vec!["latency".into()], // only index 0 exists
        violated_constraints: vec![],
        belief_buffer_questions: vec![],
    };
    // slot_index=5 is out of bounds → fallback to "constraint resolution"
    let prefix = build_follower_prefix_with_aspect(&snap, 5, 0.5);
    assert!(prefix.contains("constraint resolution"));
}

// ── generate_socratic_question ────────────────────────────────────────────────

#[tokio::test]
async fn generate_socratic_question_with_mock_adapter_returns_non_empty() {
    use h2ai_config::H2AIConfig;
    use h2ai_orchestrator::leader::generate_socratic_question;
    use h2ai_test_utils::mock_adapter;

    let adapter = mock_adapter("What if the retry logic is fundamentally broken?");
    let cfg = H2AIConfig::default();
    let (question, rank, _dedup) = generate_socratic_question(
        &adapter,
        "prior proposal text",
        &["C1".to_string()],
        &[],
        &cfg,
    )
    .await;

    assert!(!question.is_empty());
    assert_eq!(rank, 1);
}

#[tokio::test]
async fn generate_socratic_question_with_dup_in_buffer_uses_fallback() {
    use h2ai_config::H2AIConfig;
    use h2ai_orchestrator::leader::{generate_socratic_question, BeliefRecord};
    use h2ai_test_utils::mock_adapter;

    let question_text = "What if the retry logic is fundamentally broken?";
    let adapter = mock_adapter(question_text);
    let cfg = H2AIConfig {
        leader_eig_candidates: 1,
        ..H2AIConfig::default()
    };
    // Put the same question in the belief buffer → it will be detected as a dup
    let buffer = vec![BeliefRecord {
        question_hash: fnv1a(question_text),
        question_text: question_text.into(),
        outcome_scores: vec![0.5],
    }];
    let (question, _rank, dedup) =
        generate_socratic_question(&adapter, "prior", &["C1".to_string()], &buffer, &cfg).await;

    // All candidates were dups → fell back to constraint-based fallback question
    assert!(!question.is_empty());
    assert_eq!(dedup, 1, "one dup was tried");
    assert!(question.contains("C1") || question.contains("assumption"));
}

// ── eig_score with non-empty buffer (diversity bonus) ─────────────────────────

#[test]
fn eig_score_reduces_when_question_overlaps_with_buffer() {
    // A question with many shared words with buffer entry should score lower (similarity penalty).
    let long_question =
        "What if the system retries indefinitely when the timeout expires during heavy load?";
    let buffer_similar = vec![BeliefRecord {
        question_hash: fnv1a("different hash"),
        question_text: "retries indefinitely when the timeout expires during heavy load".into(),
        outcome_scores: vec![0.5],
    }];
    let buffer_empty: Vec<BeliefRecord> = vec![];
    let constraints = vec!["C1".to_string()];

    let score_empty = eig_score(long_question, &constraints, &buffer_empty);
    let score_similar = eig_score(long_question, &constraints, &buffer_similar);
    // With a highly similar buffer entry, diversity_bonus is reduced → lower score
    assert!(
        score_similar <= score_empty,
        "similar buffer should reduce score: empty={score_empty}, similar={score_similar}"
    );
}

// ── select_best_and_runner_up edge cases ──────────────────────────────────────

#[test]
fn select_best_and_runner_up_with_single_entry_has_no_runner_up() {
    let id = ExplorerId::new();
    let scores = vec![(id.clone(), 0.9f64)];
    let (winner, runner_up) = select_best_and_runner_up(&scores).unwrap();
    assert_eq!(winner, id);
    assert!(runner_up.is_none());
}

#[test]
fn select_best_and_runner_up_empty_returns_none() {
    assert!(select_best_and_runner_up(&[]).is_none());
}

// ── assign_follower_aspects edge cases ────────────────────────────────────────

#[test]
fn assign_follower_aspects_empty_constraints_uses_default() {
    let aspects = assign_follower_aspects(&[], 3);
    assert_eq!(aspects.len(), 3);
    assert!(aspects.iter().all(|a| a == "constraint resolution"));
}

#[test]
fn assign_follower_aspects_zero_followers_returns_empty() {
    let aspects = assign_follower_aspects(&["C1".to_string()], 0);
    assert!(aspects.is_empty());
}
