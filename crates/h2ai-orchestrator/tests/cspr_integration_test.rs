/// Integration test: CSPR-v2 repair context is correctly built when two
/// constraints with opposing SemanticOrdering predicates are both violated.
///
/// Exercises the full chain:
///   ConstraintConflictGraph::build → build_repair_context → output format
/// without requiring a live adapter or NATS connection.
use h2ai_autonomic::repair::{build_repair_context, RepairInput};
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

    let ctx = build_repair_context(RepairInput {
        prior_proposal_text: prior_text,
        violated_ids: &["C-005".to_owned(), "C-999".to_owned()],
        violated_hints: &[
            Some("Debit the budget atomically first, then publish the billing event.".to_owned()),
            Some("Publish to Kafka before debiting.".to_owned()),
        ],
        conflict_graph: &graph,
        retry_count: 1,
        attempts_remaining: 2,
        system_context_with_rubric: system_ctx,
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

    let ctx = build_repair_context(RepairInput {
        prior_proposal_text: "prior",
        violated_ids: &["A".to_owned(), "B".to_owned()],
        violated_hints: &[
            Some("Do X before Y".to_owned()),
            Some("Do A before B".to_owned()),
        ],
        conflict_graph: &graph,
        retry_count: 0,
        attempts_remaining: 3,
        system_context_with_rubric: "CTX",
    });

    assert!(
        !ctx.contains("COMPETING CONSTRAINTS DETECTED"),
        "non-conflicting pairs must NOT emit MetaRepair block"
    );
    assert!(ctx.contains("REPAIR TARGET 1"));
    assert!(ctx.contains("REPAIR TARGET 2"));
}
