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
/// Integration test: CSPR-v2 repair context is correctly built when two
/// constraints with opposing SemanticOrdering predicates are both violated.
///
/// Exercises the full chain:
///   ConstraintConflictGraph::build → build_repair_context → output format
/// without requiring a live adapter or NATS connection.
use h2ai_autonomic::repair::{build_repair_context, RepairInput, RepairTarget};
use h2ai_constraints::conflict::ConstraintConflictGraph;
use h2ai_constraints::types::{ConstraintDoc, ConstraintPredicate, ConstraintSeverity};

fn make_doc_with_hint(id: &str, first: &str, then: &str, hint: &str) -> ConstraintDoc {
    ConstraintDoc {
        id: id.to_owned(),
        source_file: format!("{id}.yaml"),
        description: String::new(),
        severity: ConstraintSeverity::Hard { threshold: 0.5 },
        predicate: ConstraintPredicate::SemanticOrdering {
            first: first.to_owned(),
            then: then.to_owned(),
            passes: 3,
        },
        remediation_hint: Some(hint.to_owned()),
        domains: vec![],
        mandatory_for_tags: vec![],
        related_to: vec![],
        binary_checks: vec![],
        version: 1,
        repair_provenance: None,
        pass_criteria: None,
    }
}

#[test]
fn cspr_repair_context_for_competing_constraints() {
    let docs = vec![
        make_doc_with_hint(
            "C-005",
            "account debit",
            "Kafka publish",
            "Debit the budget atomically first, then publish the billing event.",
        ),
        make_doc_with_hint(
            "C-999",
            "Kafka publish",
            "account debit",
            "Publish to Kafka before debiting to ensure the audit trail precedes state change.",
        ),
    ];
    let graph = ConstraintConflictGraph::build(&docs);

    assert!(
        graph.are_conflicting("C-005", "C-999"),
        "opposing SemanticOrdering pair must be detected as conflicting"
    );

    let prior_text = "We publish the billing event to Kafka first, then debit the Redis budget.";
    let system_ctx = "You are an expert system designer. Evaluate the task below.";

    let targets = vec![
        RepairTarget {
            constraint_id: "C-005".to_owned(),
            constraint_description: "account debit must occur before Kafka publish".to_owned(),
            remediation_hint: Some(
                "Debit the budget atomically first, then publish the billing event.".to_owned(),
            ),
            criteria_pass: None,
            verifier_reasons: vec![],
        },
        RepairTarget {
            constraint_id: "C-999".to_owned(),
            constraint_description: "Kafka publish must occur before account debit".to_owned(),
            remediation_hint: Some("Publish to Kafka before debiting.".to_owned()),
            criteria_pass: None,
            verifier_reasons: vec![],
        },
    ];
    let ctx = build_repair_context(RepairInput {
        prior_proposal_text: prior_text,
        targets: &targets,
        zone3_hints: None,
        conflict_graph: &graph,
        retry_count: 1,
        attempts_remaining: 2,
        system_context_with_rubric: system_ctx,
        checks: &[],
        partial_passes: &[],
        prior_best_score: None,
        domain_syntheses: &[],
        coupled_constraint_hints: &[],
        passing_constraint_pins: &[],
    });

    assert!(
        ctx.starts_with(system_ctx),
        "system context must be the first thing in the repair context"
    );
    assert!(
        ctx.contains(prior_text),
        "prior proposal text must appear verbatim in repair context"
    );
    assert!(
        ctx.contains("COMPETING CONSTRAINTS DETECTED"),
        "competing constraint block must be emitted"
    );
    assert!(
        ctx.contains("Fix C-005 first"),
        "C-005 must be identified as the first repair priority"
    );
    assert!(ctx.contains("C-005"), "repair target C-005 must appear");
    assert!(ctx.contains("C-999"), "repair target C-999 must appear");
    assert!(
        ctx.contains("Debit the budget atomically first"),
        "C-005 remediation hint must be included"
    );
}

#[test]
fn cspr_repair_context_non_conflicting_constraints_no_meta_repair() {
    let docs = vec![
        make_doc_with_hint("A", "x", "y", "Do X before Y"),
        make_doc_with_hint("B", "a", "b", "Do A before B"),
    ];
    let graph = ConstraintConflictGraph::build(&docs);
    assert!(!graph.are_conflicting("A", "B"));

    let targets = vec![
        RepairTarget {
            constraint_id: "A".to_owned(),
            constraint_description: "x must occur before y".to_owned(),
            remediation_hint: Some("Do X before Y".to_owned()),
            criteria_pass: None,
            verifier_reasons: vec![],
        },
        RepairTarget {
            constraint_id: "B".to_owned(),
            constraint_description: "a must occur before b".to_owned(),
            remediation_hint: Some("Do A before B".to_owned()),
            criteria_pass: None,
            verifier_reasons: vec![],
        },
    ];
    let ctx = build_repair_context(RepairInput {
        prior_proposal_text: "prior",
        targets: &targets,
        zone3_hints: None,
        conflict_graph: &graph,
        retry_count: 0,
        attempts_remaining: 3,
        system_context_with_rubric: "CTX",
        checks: &[],
        partial_passes: &[],
        prior_best_score: None,
        domain_syntheses: &[],
        coupled_constraint_hints: &[],
        passing_constraint_pins: &[],
    });

    assert!(
        !ctx.contains("COMPETING CONSTRAINTS DETECTED"),
        "non-conflicting pairs must NOT emit MetaRepair block"
    );
    assert!(ctx.contains("REPAIR TARGET 1"));
    assert!(ctx.contains("REPAIR TARGET 2"));
}
