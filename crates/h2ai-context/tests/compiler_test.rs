use h2ai_constraints::types::{
    CompositeOp, ConstraintDoc, ConstraintPredicate, ConstraintSeverity, VocabularyMode,
};
use h2ai_context::compiler::compile;

// ── SemanticPresence child in Composite ──────────────────────────────────────

#[test]
fn semantic_presence_child_injects_concept_into_context() {
    let doc = ConstraintDoc {
        id: "C-PRES".to_string(),
        source_file: "test.yaml".into(),
        description: String::new(),
        severity: ConstraintSeverity::Hard { threshold: 0.9 },
        predicate: ConstraintPredicate::Composite {
            op: CompositeOp::And,
            children: vec![
                ConstraintPredicate::SemanticPresence {
                    concept: "idempotency_key".to_string(),
                    passes: 1,
                },
                ConstraintPredicate::LlmJudge {
                    rubric: "response must mention idempotency_key".to_string(),
                },
            ],
        },
        remediation_hint: None,
        domains: vec![],
        mandatory_for_tags: vec![],
        related_to: vec![],
    };
    let result = compile("task with presence constraint", &[doc], false);
    assert!(
        result.system_context.contains("idempotency_key"),
        "SemanticPresence concept must appear in compiled context"
    );
}

// ── SemanticExclusion child in Composite ─────────────────────────────────────

#[test]
fn semantic_exclusion_child_injects_pattern_into_context() {
    let doc = ConstraintDoc {
        id: "C-EXCL".to_string(),
        source_file: "test.yaml".into(),
        description: String::new(),
        severity: ConstraintSeverity::Hard { threshold: 0.9 },
        predicate: ConstraintPredicate::Composite {
            op: CompositeOp::And,
            children: vec![
                ConstraintPredicate::SemanticExclusion {
                    pattern: "server-side session".to_string(),
                    passes: 1,
                },
                ConstraintPredicate::LlmJudge {
                    rubric: "must not use server-side session".to_string(),
                },
            ],
        },
        remediation_hint: None,
        domains: vec![],
        mandatory_for_tags: vec![],
        related_to: vec![],
    };
    let result = compile("stateless jwt design", &[doc], false);
    assert!(
        result.system_context.contains("server-side session"),
        "SemanticExclusion pattern must appear in compiled context"
    );
}

// ── Composite with vocabulary child ──────────────────────────────────────────

#[test]
fn composite_with_vocabulary_presence_child_appends_terms_block() {
    let doc = ConstraintDoc {
        id: "C-VOC".to_string(),
        source_file: "test.yaml".into(),
        description: String::new(),
        severity: ConstraintSeverity::Soft { weight: 1.0 },
        predicate: ConstraintPredicate::Composite {
            op: CompositeOp::And,
            children: vec![
                ConstraintPredicate::VocabularyPresence {
                    mode: VocabularyMode::AnyOf,
                    terms: vec!["redis".to_string(), "idempotency".to_string()],
                },
                ConstraintPredicate::LlmJudge {
                    rubric: "use redis idempotency".to_string(),
                },
            ],
        },
        remediation_hint: None,
        domains: vec![],
        mandatory_for_tags: vec![],
        related_to: vec![],
    };
    let result = compile("budget mutation design", &[doc], true);
    assert!(
        result.system_context.contains("redis"),
        "vocabulary term 'redis' must appear in compiled context"
    );
    assert!(
        result.system_context.contains("idempotency"),
        "vocabulary term 'idempotency' must appear in compiled context"
    );
    assert!(
        result.system_context.contains("C-VOC"),
        "constraint id must appear"
    );
}

// ── Non-Composite predicate with vocabulary (NegativeKeyword) ────────────────

#[test]
fn negative_keyword_predicate_appends_terms_block() {
    let doc = ConstraintDoc {
        id: "C-NEG".to_string(),
        source_file: "test.yaml".into(),
        description: String::new(),
        severity: ConstraintSeverity::Advisory,
        predicate: ConstraintPredicate::NegativeKeyword {
            terms: vec!["session_state".to_string(), "sticky_session".to_string()],
        },
        remediation_hint: None,
        domains: vec![],
        mandatory_for_tags: vec![],
        related_to: vec![],
    };
    let result = compile("stateless design", &[doc], false);
    assert!(
        result.system_context.contains("session_state"),
        "NegativeKeyword term must appear in compiled context"
    );
    assert!(
        result.system_context.contains("sticky_session"),
        "NegativeKeyword term must appear in compiled context"
    );
}

// ── Non-Composite predicate with empty vocabulary ────────────────────────────

#[test]
fn non_composite_predicate_with_no_vocab_emits_no_extra_block() {
    let doc = ConstraintDoc {
        id: "C-NOVEC".to_string(),
        source_file: "test.yaml".into(),
        description: String::new(),
        severity: ConstraintSeverity::Advisory,
        // LlmJudge alone (non-Composite) has no vocabulary
        predicate: ConstraintPredicate::LlmJudge {
            rubric: "some rubric".to_string(),
        },
        remediation_hint: None,
        domains: vec![],
        mandatory_for_tags: vec![],
        related_to: vec![],
    };
    let result = compile("task", &[doc], true);
    // The constraint id should not appear since it's a bare LlmJudge (not wrapped in Composite)
    // but no vocabulary block should be emitted either — just the task manifest
    assert!(
        result.system_context.contains("task"),
        "manifest must always be present"
    );
}

// ── Composite without LlmJudge child (no rubric found) ───────────────────────

#[test]
fn composite_without_llm_judge_child_uses_active_id_entry() {
    let doc = ConstraintDoc {
        id: "C-NOJUDGE".to_string(),
        source_file: "test.yaml".into(),
        description: String::new(),
        severity: ConstraintSeverity::Soft { weight: 1.0 },
        predicate: ConstraintPredicate::Composite {
            op: CompositeOp::And,
            children: vec![ConstraintPredicate::SemanticPresence {
                concept: "circuit_breaker".to_string(),
                passes: 1,
            }],
        },
        remediation_hint: None,
        domains: vec![],
        mandatory_for_tags: vec![],
        related_to: vec![],
    };
    // include_rubric=true but no LlmJudge child — falls back to ACTIVE_ID entry
    let result = compile("resilience design", &[doc], true);
    assert!(
        result.system_context.contains("C-NOJUDGE"),
        "constraint id must appear even without LlmJudge child"
    );
    assert!(
        result.system_context.contains("circuit_breaker"),
        "SemanticPresence concept must still be injected"
    );
}

#[test]
fn compiled_system_context_contains_adr_source_name() {
    let doc = ConstraintDoc::new_llm_judge(
        "ADR-004",
        "All budget mutations MUST use a Redis Lua idempotency key. No per-request state may be stored in service memory.",
    );
    let result = compile(
        "prevent double-billing on restart using redis idempotency budget mutations memory",
        &[doc],
        true,
    );
    assert!(result.system_context.contains("ADR-004"));
}

#[test]
fn compiled_system_context_contains_manifest() {
    let manifest =
        "prevent double-billing on restart using redis idempotency budget mutations memory";
    let doc = ConstraintDoc::new_llm_judge(
        "ADR-004",
        "All budget mutations MUST use a Redis Lua idempotency key.",
    );
    let result = compile(manifest, &[doc], true);
    assert!(result.system_context.contains(manifest));
}

#[test]
fn compile_with_empty_corpus_uses_manifest_only() {
    let manifest = "redis idempotency budget mutations memory";
    let result = compile(manifest, &[], false);
    assert!(result.system_context.contains(manifest));
}

#[test]
fn compile_multiple_constraints_includes_all_ids() {
    let doc_a = ConstraintDoc::new_llm_judge(
        "ADR-001",
        "Use stateless JWT tokens for authentication. No server-side session state.",
    );
    let doc_b = ConstraintDoc::new_llm_judge(
        "ADR-002",
        "Internal services MUST use gRPC for inter-service communication. REST is not permitted internally.",
    );
    let result = compile("implement stateless jwt grpc auth", &[doc_a, doc_b], true);
    assert!(result.system_context.contains("ADR-001"));
    assert!(result.system_context.contains("ADR-002"));
}

#[test]
fn include_rubric_true_injects_llm_judge_rubric_and_id() {
    let doc = ConstraintDoc::new_llm_judge("C-001", "The proposal must be stateless.");
    let result = compile("task description", &[doc], true);
    assert!(
        result
            .system_context
            .contains("The proposal must be stateless."),
        "rubric text must appear when include_rubric=true"
    );
    assert!(
        result.system_context.contains("C-001"),
        "constraint ID must appear when include_rubric=true"
    );
}

#[test]
fn include_rubric_false_withholds_rubric_but_keeps_id_and_hint() {
    // Use Soft constraint to test withholding rubric behavior
    // (Hard constraints always get rubric regardless of include_rubric flag)
    let mut doc = ConstraintDoc::new_soft_llm_judge("C-001", "The proposal must be stateless.");
    doc.remediation_hint = Some("Avoid sticky sessions.".into());
    let result = compile("task description", &[doc], false);
    assert!(
        !result
            .system_context
            .contains("The proposal must be stateless."),
        "rubric scoring text must NOT appear when include_rubric=false for Soft constraints"
    );
    assert!(
        result.system_context.contains("C-001"),
        "constraint ID must still appear so explorer knows what to satisfy"
    );
    assert!(
        result.system_context.contains("Avoid sticky sessions."),
        "remediation hint must still appear to guide the explorer"
    );
    assert!(
        result.system_context.contains("task description"),
        "task manifest must always be present"
    );
}

#[test]
fn include_rubric_false_still_includes_vocabulary_constraints() {
    let vocab_doc = ConstraintDoc {
        id: "C-002".to_string(),
        source_file: "test.md".into(),
        description: String::new(),
        severity: ConstraintSeverity::Advisory,
        predicate: ConstraintPredicate::VocabularyPresence {
            mode: VocabularyMode::AnyOf,
            terms: vec!["idempotency".to_string()],
        },
        remediation_hint: None,
        domains: vec![],
        mandatory_for_tags: vec![],
        related_to: vec![],
    };
    let result = compile("task description", &[vocab_doc], false);
    assert!(
        result.system_context.contains("idempotency"),
        "vocabulary terms must still appear when include_rubric=false"
    );
}

#[test]
fn hard_constraint_always_gets_rubric_regardless_of_include_rubric_flag() {
    let doc = ConstraintDoc::new_llm_judge("C-HARD", "The proposal must keep T_global ≤ 100ms.");
    // new_llm_judge uses Hard severity with threshold 0.8
    let result = compile("task description", &[doc], false);
    assert!(
        result
            .system_context
            .contains("The proposal must keep T_global ≤ 100ms."),
        "Hard constraint rubric must appear even when include_rubric=false"
    );
}

#[test]
fn semantic_ordering_constraint_injects_ordering_requirement_and_hint() {
    let doc = ConstraintDoc {
        id: "C-005".to_string(),
        source_file: "test.yaml".into(),
        description: String::new(),
        severity: ConstraintSeverity::Hard { threshold: 0.9 },
        predicate: ConstraintPredicate::Composite {
            op: CompositeOp::And,
            children: vec![
                ConstraintPredicate::SemanticOrdering {
                    first: "account debit".to_string(),
                    then: "Kafka publish".to_string(),
                    passes: 1,
                },
                ConstraintPredicate::LlmJudge {
                    rubric: "account debit must precede Kafka publish".to_string(),
                },
            ],
        },
        remediation_hint: Some("Debit Redis first, then publish to Kafka.".into()),
        domains: vec![],
        mandatory_for_tags: vec![],
        related_to: vec![],
    };
    let result = compile("task description", &[doc], false);
    assert!(
        result.system_context.contains("account debit"),
        "first ordering term must appear in context"
    );
    assert!(
        result.system_context.contains("Kafka publish"),
        "then ordering term must appear in context"
    );
    assert!(
        result.system_context.contains("Debit Redis first"),
        "remediation hint must appear for SemanticOrdering constraints"
    );
    assert!(
        result.system_context.contains("C-005"),
        "constraint ID must appear"
    );
}
