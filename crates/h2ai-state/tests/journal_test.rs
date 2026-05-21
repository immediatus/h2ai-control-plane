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
use h2ai_state::journal::{EventJournal, InMemoryBackend};
use h2ai_types::events::{H2AIEvent, MergeResolvedEvent, ZeroSurvivalEvent};
use h2ai_types::identity::TaskId;

#[tokio::test]
async fn append_and_replay_preserves_order() {
    let journal = EventJournal::new(InMemoryBackend::new());
    let tid = TaskId::new();

    journal
        .append(H2AIEvent::ZeroSurvival(ZeroSurvivalEvent {
            task_id: tid.clone(),
            retry_count: 0,
            timestamp: Utc::now(),
            n_eff_cosine_actual: None,
            failure_mode: None,
        }))
        .await
        .unwrap();

    journal
        .append(H2AIEvent::MergeResolved(MergeResolvedEvent {
            task_id: tid.clone(),
            resolved_output: "done".into(),
            j_eff: None,
            oracle_gate_passed: None,
            timestamp: Utc::now(),
            zone3_hints: None,
        }))
        .await
        .unwrap();

    let events = journal.replay(0).await.unwrap();
    assert_eq!(events.len(), 2);
    assert!(matches!(events[0], H2AIEvent::ZeroSurvival(_)));
    assert!(matches!(events[1], H2AIEvent::MergeResolved(_)));
}

#[tokio::test]
async fn replay_from_offset_returns_tail() {
    let journal = EventJournal::new(InMemoryBackend::new());
    let tid = TaskId::new();

    for i in 0..5u32 {
        journal
            .append(H2AIEvent::ZeroSurvival(ZeroSurvivalEvent {
                task_id: tid.clone(),
                retry_count: i,
                timestamp: Utc::now(),
                n_eff_cosine_actual: None,
                failure_mode: None,
            }))
            .await
            .unwrap();
    }

    let tail = journal.replay(3).await.unwrap();
    assert_eq!(tail.len(), 2);
}

#[tokio::test]
async fn replay_empty_journal_returns_empty_vec() {
    let journal = EventJournal::new(InMemoryBackend::new());
    let events = journal.replay(0).await.unwrap();
    assert!(events.is_empty());
}
