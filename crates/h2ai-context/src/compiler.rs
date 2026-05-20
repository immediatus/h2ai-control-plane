use h2ai_config::prompts::{
    COMPILER_COMPOSITE_EXCLUSION_DETAIL, COMPILER_COMPOSITE_ORDERING_DETAIL,
    COMPILER_COMPOSITE_PRESENCE_DETAIL, COMPILER_CONSTRAINT_ACTIVE_ID,
    COMPILER_CONSTRAINT_GUIDANCE_SUFFIX, COMPILER_CONSTRAINT_HARD_RUBRIC,
    COMPILER_CONSTRAINT_VOCABULARY, COMPILER_TASK_MANIFEST,
};
use h2ai_constraints::types::{ConstraintDoc, ConstraintPredicate, ConstraintSeverity};

/// Output produced by a successful [`compile`] call.
pub struct CompilerResult {
    /// Assembled system-context string injected into LLM prompts.
    pub system_context: String,
}

/// Compile a task manifest and constraint corpus into a system-context string.
///
/// All constraints are compiled to `Composite` bytecode; this function renders each
/// `Composite` constraint as one coherent block: structural gates (exclusion/presence/ordering)
/// followed by the rubric (for Hard or include_rubric=true) or just the constraint ID.
/// Vocabulary terms are always appended regardless of severity.
pub fn compile(manifest: &str, corpus: &[ConstraintDoc], include_rubric: bool) -> CompilerResult {
    CompilerResult {
        system_context: build_system_context(manifest, corpus, include_rubric),
    }
}

fn build_system_context(manifest: &str, corpus: &[ConstraintDoc], include_rubric: bool) -> String {
    let mut parts = vec![COMPILER_TASK_MANIFEST.render(&[("manifest", manifest)])];
    for doc in corpus {
        match &doc.predicate {
            ConstraintPredicate::Composite { children, .. } => {
                let is_hard = matches!(doc.severity, ConstraintSeverity::Hard { .. });
                // LlmJudge rubric is always the last child (guaranteed by into_constraint_doc()).
                let rubric = children.iter().rev().find_map(|c| {
                    if let ConstraintPredicate::LlmJudge { rubric } = c {
                        Some(rubric.as_str())
                    } else {
                        None
                    }
                });
                let mut entry = match rubric {
                    Some(r) if include_rubric || is_hard => COMPILER_CONSTRAINT_HARD_RUBRIC
                        .render(&[("id", &doc.id), ("rubric", r.trim())]),
                    _ => COMPILER_CONSTRAINT_ACTIVE_ID.render(&[("id", &doc.id)]),
                };
                // Inject structural child details.
                for child in children {
                    match child {
                        ConstraintPredicate::SemanticOrdering { first, then, .. } => {
                            entry.push_str(
                                &COMPILER_COMPOSITE_ORDERING_DETAIL
                                    .render(&[("first", first), ("then", then)]),
                            );
                        }
                        ConstraintPredicate::SemanticPresence { concept, .. } => {
                            entry.push_str(
                                &COMPILER_COMPOSITE_PRESENCE_DETAIL.render(&[("concept", concept)]),
                            );
                        }
                        ConstraintPredicate::SemanticExclusion { pattern, .. } => {
                            entry.push_str(
                                &COMPILER_COMPOSITE_EXCLUSION_DETAIL
                                    .render(&[("pattern", pattern)]),
                            );
                        }
                        _ => {}
                    }
                }
                if let Some(h) = &doc.remediation_hint {
                    entry.push_str(COMPILER_CONSTRAINT_GUIDANCE_SUFFIX);
                    entry.push_str(h);
                }
                parts.push(entry);
                // Vocabulary-based children still get their term lists injected.
                let vocab = doc.vocabulary();
                if !vocab.is_empty() {
                    let mut terms: Vec<&str> = vocab.iter().map(|s| s.as_str()).collect();
                    terms.sort();
                    let terms_block = terms
                        .iter()
                        .map(|k| format!("- {k}"))
                        .collect::<Vec<_>>()
                        .join("\n");
                    parts.push(
                        COMPILER_CONSTRAINT_VOCABULARY
                            .render(&[("id", &doc.id), ("terms", &terms_block)]),
                    );
                }
            }
            _ => {
                // Vocabulary fallback: VocabularyPresence, NegativeKeyword.
                // Used only by programmatically constructed constraints not from SemanticSpec.
                let vocab = doc.vocabulary();
                if !vocab.is_empty() {
                    let mut terms: Vec<&str> = vocab.iter().map(|s| s.as_str()).collect();
                    terms.sort();
                    let terms_block = terms
                        .iter()
                        .map(|k| format!("- {k}"))
                        .collect::<Vec<_>>()
                        .join("\n");
                    parts.push(
                        COMPILER_CONSTRAINT_VOCABULARY
                            .render(&[("id", &doc.id), ("terms", &terms_block)]),
                    );
                }
            }
        }
    }
    parts.join("\n\n")
}

#[cfg(test)]
mod compiler_tests {
    use super::*;
    use h2ai_constraints::types::{
        CompositeOp, ConstraintDoc, ConstraintPredicate, ConstraintSeverity,
    };

    fn llm_judge_doc(id: &str, rubric: &str) -> ConstraintDoc {
        ConstraintDoc::new_llm_judge(id, rubric)
    }

    #[test]
    fn include_rubric_true_injects_llm_judge_rubric_and_id() {
        let doc = llm_judge_doc("C-001", "The proposal must be stateless.");
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
        use h2ai_constraints::types::{ConstraintPredicate, VocabularyMode};
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
        let doc = llm_judge_doc("C-HARD", "The proposal must keep T_global ≤ 100ms.");
        // llm_judge_doc uses Hard severity with threshold 0.8
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
}
