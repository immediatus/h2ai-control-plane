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
use h2ai_config::IntraRetryDetectorConfig;
use h2ai_orchestrator::ceiling_detector::{
    count_ceiling_signals, failure_signature_entropy, retry_slope,
};
use h2ai_types::events::{BranchPrunedEvent, ConstraintViolation};
use h2ai_types::identity::{ExplorerId, TaskId};
use h2ai_types::sizing::RoleErrorCost;

fn make_pruned(constraint_ids: &[&str]) -> BranchPrunedEvent {
    BranchPrunedEvent {
        task_id: TaskId::new(),
        explorer_id: ExplorerId::new(),
        reason: "test".into(),
        raw_output: String::new(),
        constraint_error_cost: RoleErrorCost::new(0.5).unwrap(),
        violated_constraints: constraint_ids
            .iter()
            .map(|id| ConstraintViolation {
                constraint_id: id.to_string(),
                score: 0.0,
                severity_label: "Hard".into(),
                remediation_hint: None,
                constraint_description: String::new(),
                verifier_reason: None,
                check_verdicts: vec![],
                criteria_pass: None,
            })
            .collect(),
        timestamp: chrono::Utc::now(),
    }
}

// ── failure_signature_entropy ──────────────────────────────────────────────

#[test]
fn entropy_high_when_failures_spread_across_many_constraints() {
    let events = vec![
        make_pruned(&["c1"]),
        make_pruned(&["c2"]),
        make_pruned(&["c3"]),
        make_pruned(&["c4"]),
    ];
    let h = failure_signature_entropy(&events);
    assert!(h > 0.6, "expected entropy > 0.6, got {h}");
}

#[test]
fn entropy_low_when_all_proposals_fail_same_constraint() {
    let events = vec![
        make_pruned(&["c1"]),
        make_pruned(&["c1"]),
        make_pruned(&["c1"]),
        make_pruned(&["c1"]),
    ];
    let h = failure_signature_entropy(&events);
    assert!(h < 0.6, "expected entropy < 0.6, got {h}");
}

#[test]
fn entropy_returns_1_when_no_pruned_events() {
    let h = failure_signature_entropy(&[]);
    assert!((h - 1.0).abs() < 1e-9, "expected 1.0, got {h}");
}

// ── retry_slope ────────────────────────────────────────────────────────────

#[test]
fn retry_slope_positive_when_score_improves() {
    let slope = retry_slope(&[0.2, 0.4]);
    // (0.4 - 0.2) / 0.2 = 1.0
    assert!(
        (slope - 1.0).abs() < 1e-6,
        "expected slope ≈ 1.0, got {slope}"
    );
}

#[test]
fn retry_slope_near_zero_when_score_stagnates() {
    let slope = retry_slope(&[0.3, 0.31]);
    assert!(slope < 0.05, "expected slope < 0.05, got {slope}");
}

#[test]
fn retry_slope_infinity_when_fewer_than_2_scores() {
    // Insufficient history must not be misread as a stall (INFINITY > any threshold).
    assert_eq!(retry_slope(&[]), f64::INFINITY);
    assert_eq!(retry_slope(&[0.5]), f64::INFINITY);
}

#[test]
fn count_signals_slope_does_not_fire_on_empty_history() {
    // All-ZeroSurvival waves leave quality_history empty.  Signal 2 must stay silent
    // so the intra-retry ceiling does not prematurely abort the retry loop.
    let events = vec![
        make_pruned(&["c1"]),
        make_pruned(&["c1"]),
        make_pruned(&["c1"]),
    ];
    let cfg = IntraRetryDetectorConfig {
        enabled: true,
        entropy_threshold: 0.6,
        retry_slope_threshold: 0.05,
        n_eff_cg_product_threshold: 0.3,
        min_retry_count_for_detection: 2,
    };
    // Empty score_history — slope signal must be silent.
    let count = count_ceiling_signals(&events, &[], 1.0, 0.7, &cfg);
    // Signal 1 fires (entropy=0 from single constraint) but signal 2 must NOT.
    // n_eff×cg_mean = 0.7 > 0.3 → signal 3 silent. Total ≤ 1.
    assert!(count < 2, "ceiling must not fire on empty quality history; got {count} signals");
}

// ── count_ceiling_signals ─────────────────────────────────────────────────

#[test]
fn count_signals_zero_when_all_clear() {
    // Diverse failures → high entropy (no signal 1)
    let events = vec![
        make_pruned(&["c1"]),
        make_pruned(&["c2"]),
        make_pruned(&["c3"]),
        make_pruned(&["c4"]),
    ];
    // Good slope → no signal 2
    let scores = vec![0.2, 0.4];
    // High n_eff × cg_mean → no signal 3
    let cfg = IntraRetryDetectorConfig {
        enabled: true,
        entropy_threshold: 0.6,
        retry_slope_threshold: 0.05,
        n_eff_cg_product_threshold: 0.3,
        min_retry_count_for_detection: 2,
    };
    let count = count_ceiling_signals(&events, &scores, 1.0, 0.8, &cfg);
    assert_eq!(count, 0, "expected 0 signals, got {count}");
}

#[test]
fn count_signals_2_when_entropy_and_product_fire() {
    // Peaked failures → low entropy (signal 1)
    let events = vec![
        make_pruned(&["c1"]),
        make_pruned(&["c1"]),
        make_pruned(&["c1"]),
        make_pruned(&["c1"]),
    ];
    // Good slope → no signal 2
    let scores = vec![0.2, 0.4];
    // Low n_eff × cg_mean → signal 3
    let cfg = IntraRetryDetectorConfig {
        enabled: true,
        entropy_threshold: 0.6,
        retry_slope_threshold: 0.05,
        n_eff_cg_product_threshold: 0.3,
        min_retry_count_for_detection: 2,
    };
    let count = count_ceiling_signals(&events, &scores, 0.2, 0.5, &cfg);
    // entropy signal + product signal = 2
    assert!(count >= 2, "expected ≥2 signals, got {count}");
}
