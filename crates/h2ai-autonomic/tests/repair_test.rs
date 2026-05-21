use h2ai_autonomic::repair::{build_repair_context, RepairInput};
use h2ai_constraints::conflict::ConstraintConflictGraph;
use h2ai_constraints::types::{ConstraintDoc, ConstraintPredicate, ConstraintSeverity};

fn make_ordering_doc(id: &str, first: &str, then: &str) -> ConstraintDoc {
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
        remediation_hint: Some(format!("Ensure {first} before {then}")),
        domains: vec![],
        mandatory_for_tags: vec![],
        related_to: vec![],
    }
}

#[test]
fn repair_context_contains_prior_proposal_section() {
    let docs = vec![make_ordering_doc("A", "debit", "publish")];
    let graph = ConstraintConflictGraph::build(&docs);
    let ctx = build_repair_context(RepairInput {
        prior_proposal_text: "My prior proposal text here.",
        violated_ids: &["A".to_owned()],
        violated_hints: &[Some("Ensure debit before publish".to_owned())],
        conflict_graph: &graph,
        retry_count: 1,
        attempts_remaining: 2,
        system_context_with_rubric: "SYSTEM CTX",
    });
    assert!(
        ctx.contains("SYSTEM CTX"),
        "must preserve system context prefix"
    );
    assert!(
        ctx.contains("My prior proposal text here."),
        "must embed prior proposal"
    );
    assert!(
        ctx.contains("PRIOR PROPOSAL"),
        "must have prior proposal section header"
    );
    assert!(ctx.contains("REPAIR TARGET"), "must have repair target");
    assert!(
        ctx.contains("Ensure debit before publish"),
        "must include remediation hint"
    );
}

#[test]
fn competing_constraints_produce_meta_repair_block() {
    let docs = vec![
        make_ordering_doc("A", "debit", "publish"),
        make_ordering_doc("B", "publish", "debit"),
    ];
    let graph = ConstraintConflictGraph::build(&docs);
    let ctx = build_repair_context(RepairInput {
        prior_proposal_text: "Prior text.",
        violated_ids: &["A".to_owned(), "B".to_owned()],
        violated_hints: &[
            Some("Ensure debit before publish".to_owned()),
            Some("Ensure publish before debit".to_owned()),
        ],
        conflict_graph: &graph,
        retry_count: 2,
        attempts_remaining: 1,
        system_context_with_rubric: "CTX",
    });
    assert!(
        ctx.contains("COMPETING CONSTRAINTS DETECTED"),
        "must flag conflict"
    );
    assert!(
        ctx.contains("Fix A first"),
        "must identify hard-gate priority"
    );
}

#[test]
fn no_prior_text_falls_back_gracefully() {
    let graph = ConstraintConflictGraph::build(&[]);
    let ctx = build_repair_context(RepairInput {
        prior_proposal_text: "",
        violated_ids: &["X".to_owned()],
        violated_hints: &[None],
        conflict_graph: &graph,
        retry_count: 0,
        attempts_remaining: 3,
        system_context_with_rubric: "CTX",
    });
    assert!(ctx.contains("CTX"));
    assert!(ctx.contains("REPAIR TARGET") || ctx.contains("CONSTRAINT FEEDBACK"));
}

#[test]
fn repair_context_includes_attempt_count() {
    let graph = ConstraintConflictGraph::build(&[]);
    let ctx = build_repair_context(RepairInput {
        prior_proposal_text: "prior",
        violated_ids: &["Y".to_owned()],
        violated_hints: &[Some("fix Y".to_owned())],
        conflict_graph: &graph,
        retry_count: 1,
        attempts_remaining: 2,
        system_context_with_rubric: "CTX",
    });
    assert!(ctx.contains('2'), "must mention attempts remaining");
}

#[test]
fn repair_context_emits_conflict_block_and_breaks_loop() {
    // Line 73 (break 'outer) AND line 74 (closing brace of non-conflicting pair):
    // Use 3 violated IDs — first pair (A,B) does NOT conflict, second pair (A,C) DOES.
    // A(first=debit, then=publish) conflicts with C(first=publish, then=debit).
    // B(first=credit, then=withdraw) does not conflict with A or C.
    let doc_a = make_ordering_doc("A", "debit", "publish");
    let doc_b = make_ordering_doc("B", "credit", "withdraw");
    let doc_c = make_ordering_doc("C", "publish", "debit");
    let graph = ConstraintConflictGraph::build(&[doc_a, doc_b, doc_c]);
    let ctx = build_repair_context(RepairInput {
        prior_proposal_text: "my proposal",
        violated_ids: &["A".to_owned(), "B".to_owned(), "C".to_owned()],
        violated_hints: &[
            Some("fix A".to_owned()),
            Some("fix B".to_owned()),
            Some("fix C".to_owned()),
        ],
        conflict_graph: &graph,
        retry_count: 0,
        attempts_remaining: 3,
        system_context_with_rubric: "CTX",
    });
    assert!(
        ctx.contains("COMPETING CONSTRAINTS"),
        "conflict block must appear"
    );
    assert!(
        ctx.contains("REPAIR TARGET"),
        "repair targets must still be emitted"
    );
}
