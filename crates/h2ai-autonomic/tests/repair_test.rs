use h2ai_autonomic::repair::{build_repair_context, PartialPass, RepairInput, RepairTarget};
use h2ai_constraints::conflict::ConstraintConflictGraph;
use h2ai_constraints::types::{ConstraintDoc, ConstraintPredicate, ConstraintSeverity};
use h2ai_types::gap_i1::DomainSynthesis;

// ── Shared builders ────────────────────────────────────────────────────────────

fn ordering_doc(id: &str, first: &str, then: &str) -> ConstraintDoc {
    ConstraintDoc {
        id: id.to_owned(),
        source_file: format!("{id}.yaml"),
        description: format!("{first} must occur before {then}"),
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
        binary_checks: vec![],
        version: 1,
        repair_provenance: None,
        pass_criteria: None,
    }
}

fn target(id: &str, hint: Option<&str>, reasons: &[(f64, &str)]) -> RepairTarget {
    RepairTarget {
        constraint_id: id.to_owned(),
        constraint_description: format!("{id} requirement description"),
        remediation_hint: hint.map(str::to_owned),
        criteria_pass: None,
        verifier_reasons: reasons.iter().map(|(s, r)| (*s, r.to_string())).collect(),
    }
}

fn target_with_pass(id: &str, pass: &str, reasons: &[(f64, &str)]) -> RepairTarget {
    RepairTarget {
        constraint_id: id.to_owned(),
        constraint_description: format!("{id} requirement description"),
        remediation_hint: None,
        criteria_pass: if pass.is_empty() {
            None
        } else {
            Some(pass.to_owned())
        },
        verifier_reasons: reasons.iter().map(|(s, r)| (*s, r.to_string())).collect(),
    }
}

fn empty_graph() -> ConstraintConflictGraph {
    ConstraintConflictGraph::build(&[])
}

fn repair(
    prior: &str,
    targets: &[RepairTarget],
    zone3: Option<&str>,
    graph: &ConstraintConflictGraph,
) -> String {
    build_repair_context(RepairInput {
        prior_proposal_text: prior,
        targets,
        zone3_hints: zone3,
        conflict_graph: graph,
        retry_count: 1,
        attempts_remaining: 2,
        system_context_with_rubric: "SYSTEM_CTX",
        checks: &[],
        partial_passes: &[],
        prior_best_score: None,
        domain_syntheses: &[],
        coupled_constraint_hints: &[],
        passing_constraint_pins: &[],
    })
}

// ── build_repair_context ───────────────────────────────────────────────────────

/// Tests for system context prefix and prior-proposal embedding.
mod system_context_and_prior_proposal {
    use super::*;

    #[test]
    fn output_starts_with_system_context_rubric() {
        let ctx = repair(
            "prior text",
            &[target("A", Some("fix A"), &[])],
            None,
            &empty_graph(),
        );
        assert!(ctx.starts_with("SYSTEM_CTX"));
    }

    #[test]
    fn non_empty_prior_embeds_proposal_with_section_header() {
        let ctx = repair(
            "My prior proposal.",
            &[target("A", Some("fix A"), &[])],
            None,
            &empty_graph(),
        );
        assert!(ctx.contains("PRIOR PROPOSAL"));
        assert!(ctx.contains("My prior proposal."));
    }

    #[test]
    fn empty_prior_emits_constraint_feedback_header_instead() {
        let ctx = repair("", &[target("A", None, &[])], None, &empty_graph());
        assert!(ctx.contains("CONSTRAINT FEEDBACK"));
        assert!(!ctx.contains("PRIOR PROPOSAL"));
    }

    #[test]
    fn attempt_count_is_embedded_in_repair_instructions() {
        let ctx = repair(
            "prior",
            &[target("Y", Some("fix Y"), &[])],
            None,
            &empty_graph(),
        );
        assert!(ctx.contains('2'), "attempts_remaining=2 must appear");
    }

    #[test]
    fn output_ends_with_repair_instructions_sentinel() {
        let ctx = repair("prior", &[target("A", None, &[])], None, &empty_graph());
        assert!(ctx.ends_with("--- END REPAIR INSTRUCTIONS ---"));
    }
}

/// Tests for the three-slot sandwich template per repair target.
mod repair_target_slot_selection {
    use super::*;

    #[test]
    fn slot_a_emitted_when_verifier_reason_present() {
        let targets = vec![target("Z", None, &[(0.7, "missing lock acquisition")])];
        let ctx = repair("prior", &targets, None, &empty_graph());
        assert!(ctx.contains("VERIFIER INTERPRETATION"));
        assert!(ctx.contains("missing lock acquisition"));
        assert!(ctx.contains("70%"), "score must appear in slot A");
        assert!(!ctx.contains("GUIDANCE"));
    }

    #[test]
    fn slot_b_emitted_when_hint_present_and_reason_absent() {
        let targets = vec![target("Z", Some("use a mutex"), &[])];
        let ctx = repair("prior", &targets, None, &empty_graph());
        assert!(ctx.contains("GUIDANCE"));
        assert!(ctx.contains("use a mutex"));
        assert!(!ctx.contains("VERIFIER INTERPRETATION"));
    }

    #[test]
    fn slot_c_emitted_when_only_description_available() {
        let targets = vec![target("Z", None, &[])];
        let ctx = repair("prior", &targets, None, &empty_graph());
        assert!(ctx.contains("REPAIR TARGET"));
        assert!(ctx.contains("Z requirement description"));
        assert!(!ctx.contains("GUIDANCE"));
        assert!(!ctx.contains("VERIFIER INTERPRETATION"));
    }

    #[test]
    fn repair_target_header_includes_constraint_id() {
        let targets = vec![target("GDPR-001", Some("hint"), &[])];
        let ctx = repair("prior", &targets, None, &empty_graph());
        assert!(ctx.contains("REPAIR TARGET 1 — GDPR-001"));
    }
}

/// Tests for conflicting-constraint detection and meta-repair block.
mod conflict_detection {
    use super::*;

    #[test]
    fn competing_constraints_block_emitted_when_pair_conflicts() {
        let docs = vec![
            ordering_doc("A", "debit", "publish"),
            ordering_doc("B", "publish", "debit"),
        ];
        let graph = ConstraintConflictGraph::build(&docs);
        let targets = vec![
            target("A", Some("fix A"), &[]),
            target("B", Some("fix B"), &[]),
        ];
        let ctx = build_repair_context(RepairInput {
            prior_proposal_text: "Prior text.",
            targets: &targets,
            zone3_hints: None,
            conflict_graph: &graph,
            retry_count: 2,
            attempts_remaining: 1,
            system_context_with_rubric: "CTX",
            checks: &[],
            partial_passes: &[],
            prior_best_score: None,
            domain_syntheses: &[],
            coupled_constraint_hints: &[],
            passing_constraint_pins: &[],
        });
        assert!(ctx.contains("COMPETING CONSTRAINTS DETECTED"));
        assert!(ctx.contains("Fix A first"));
    }

    #[test]
    fn no_competing_constraints_block_when_no_conflict() {
        let docs = vec![ordering_doc("A", "debit", "publish")];
        let graph = ConstraintConflictGraph::build(&docs);
        let targets = vec![target("A", Some("fix A"), &[])];
        let ctx = repair("prior", &targets, None, &graph);
        assert!(!ctx.contains("COMPETING CONSTRAINTS DETECTED"));
    }

    #[test]
    fn conflict_block_breaks_after_first_conflicting_pair() {
        let docs = vec![
            ordering_doc("A", "debit", "publish"),
            ordering_doc("B", "credit", "withdraw"),
            ordering_doc("C", "publish", "debit"),
        ];
        let graph = ConstraintConflictGraph::build(&[
            ordering_doc("A", "debit", "publish"),
            ordering_doc("B", "credit", "withdraw"),
            ordering_doc("C", "publish", "debit"),
        ]);
        let targets = vec![
            target("A", Some("fix A"), &[]),
            target("B", Some("fix B"), &[]),
            target("C", Some("fix C"), &[]),
        ];
        let ctx = build_repair_context(RepairInput {
            prior_proposal_text: "my proposal",
            targets: &targets,
            zone3_hints: None,
            conflict_graph: &graph,
            retry_count: 0,
            attempts_remaining: 3,
            system_context_with_rubric: "CTX",
            checks: &[],
            domain_syntheses: &[],
            partial_passes: &[],
            prior_best_score: None,
            coupled_constraint_hints: &[],
            passing_constraint_pins: &[],
        });
        assert!(ctx.contains("COMPETING CONSTRAINTS"));
        assert!(
            ctx.contains("REPAIR TARGET"),
            "repair targets must follow conflict block"
        );
        let _ = docs; // suppress unused warning
    }
}

/// Tests for Zone 3 OSP audit text appended after repair targets.
mod zone3_hints {
    use super::*;

    #[test]
    fn zone3_section_appended_after_all_repair_targets() {
        let targets = vec![target("A", Some("fix A"), &[])];
        let ctx = repair(
            "prior",
            &targets,
            Some("OSP audit: constraint C-003 borderline"),
            &empty_graph(),
        );
        assert!(ctx.contains("OSP AUDIT CONTEXT"));
        assert!(ctx.contains("OSP audit: constraint C-003 borderline"));
        let repair_pos = ctx.find("REPAIR TARGET").unwrap();
        let osp_pos = ctx.find("OSP AUDIT CONTEXT").unwrap();
        assert!(
            osp_pos > repair_pos,
            "zone3 must appear after repair targets"
        );
    }

    #[test]
    fn empty_zone3_string_produces_no_audit_section() {
        let targets = vec![target("A", Some("fix A"), &[])];
        let ctx = repair("prior", &targets, Some(""), &empty_graph());
        assert!(!ctx.contains("OSP AUDIT CONTEXT"));
    }

    #[test]
    fn none_zone3_produces_no_audit_section() {
        let targets = vec![target("A", Some("fix A"), &[])];
        let ctx = repair("prior", &targets, None, &empty_graph());
        assert!(!ctx.contains("OSP AUDIT CONTEXT"));
    }
}

/// Tests for positive assertion framing.
mod positive_assertion_framing {
    use super::*;

    #[test]
    fn slot_a_with_criteria_pass_emits_target_behavior_block() {
        let targets = vec![target_with_pass(
            "C-008",
            "Uses lock-free Redis Lua EVAL on all quota state mutations",
            &[(0.3, "uses Redlock on charge path")],
        )];
        let ctx = repair("prior", &targets, None, &empty_graph());
        assert!(
            ctx.contains("TARGET BEHAVIOR"),
            "must emit TARGET BEHAVIOR header"
        );
        assert!(
            ctx.contains("Uses lock-free Redis Lua EVAL"),
            "must include pass criterion text"
        );
    }

    #[test]
    fn slot_a_without_criteria_pass_emits_positive_fallback_not_target_behavior() {
        let targets = vec![target_with_pass("C-008", "", &[(0.3, "uses Redlock")])];
        let ctx = repair("prior", &targets, None, &empty_graph());
        assert!(
            !ctx.contains("TARGET BEHAVIOR"),
            "must NOT emit TARGET BEHAVIOR when criteria_pass is absent"
        );
        assert!(ctx.contains("satisfies the constraint requirement"));
    }

    #[test]
    fn slot_a_does_not_contain_old_prohibition_text() {
        let targets = vec![target_with_pass("C-008", "", &[(0.3, "reason")])];
        let ctx = repair("prior", &targets, None, &empty_graph());
        assert!(
            !ctx.contains("avoids the above failure mode"),
            "old prohibition text must be gone"
        );
        assert!(
            !ctx.contains("Do not reuse patterns"),
            "old negation directive must be gone"
        );
    }
}

// ── coupled_constraint_hints (moved from repair.rs) ───────────────────────────

#[test]
fn repair_context_includes_coupled_constraint_hints() {
    let graph = ConstraintConflictGraph::build(&[]);
    let targets: Vec<RepairTarget> = vec![];
    let domain_syntheses: Vec<DomainSynthesis> = vec![];
    let partial_passes: Vec<PartialPass> = vec![];

    let input = RepairInput {
        prior_proposal_text: "",
        targets: &targets,
        zone3_hints: None,
        conflict_graph: &graph,
        retry_count: 1,
        attempts_remaining: 2,
        system_context_with_rubric: "",
        checks: &[],
        partial_passes: &partial_passes,
        prior_best_score: None,
        domain_syntheses: &domain_syntheses,
        coupled_constraint_hints: &[(
            "CONSTRAINT-TAU-2".to_string(),
            Some("quota audit must use PostgreSQL INSERT-only".to_string()),
        )],
        passing_constraint_pins: &[],
    };
    let ctx = build_repair_context(input);
    assert!(
        ctx.contains("CONSTRAINT-TAU-2"),
        "repair context must include coupled constraint id"
    );
    assert!(
        ctx.contains("quota audit must use PostgreSQL INSERT-only"),
        "repair context must include coupled constraint hint text"
    );
    assert!(
        ctx.contains("MUST NOT BE BROKEN"),
        "repair context must frame coupled hints as a non-break constraint"
    );
}
