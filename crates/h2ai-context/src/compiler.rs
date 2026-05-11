use h2ai_constraints::types::{ConstraintDoc, ConstraintPredicate, ConstraintSeverity};

/// Output produced by a successful [`compile`] call.
pub struct CompilerResult {
    /// Assembled system-context string injected into LLM prompts.
    pub system_context: String,
}

/// Compile a task manifest and constraint corpus into a system-context string.
///
/// `include_rubric` controls whether `LlmJudge` constraint rubrics are injected into
/// the explorer's context. Pass `false` (the production default) to withhold rubrics
/// from soft/advisory `LlmJudge` constraints — the verifier retains them via
/// `ConstraintPredicate::LlmJudge`. Hard-severity `LlmJudge` constraints always inject
/// their full rubric regardless of this flag: they represent non-negotiable requirements
/// the explorer must know upfront.
///
/// Vocabulary-based constraints (term lists) are always included regardless of this flag.
pub fn compile(manifest: &str, corpus: &[ConstraintDoc], include_rubric: bool) -> CompilerResult {
    CompilerResult {
        system_context: build_system_context(manifest, corpus, include_rubric),
    }
}

fn build_system_context(manifest: &str, corpus: &[ConstraintDoc], include_rubric: bool) -> String {
    let mut parts = vec![format!("## Task Manifest\n{manifest}")];
    for doc in corpus {
        match &doc.predicate {
            ConstraintPredicate::LlmJudge { rubric } => {
                let is_hard = matches!(doc.severity, ConstraintSeverity::Hard { .. });
                if include_rubric || is_hard {
                    let mut entry = format!("## {} Constraint\n{}", doc.id, rubric.trim());
                    if let Some(h) = &doc.remediation_hint {
                        entry.push_str(&format!("\nGuidance: {h}"));
                    }
                    parts.push(entry);
                } else {
                    // Withhold the scoring rubric but always inject the constraint ID and
                    // remediation guidance so explorers know what behavioral requirements
                    // they must satisfy. Rubric scoring details stay with the verifier.
                    let mut hint = format!("## Active Constraint: {}", doc.id);
                    if let Some(h) = &doc.remediation_hint {
                        hint.push_str(&format!("\nRequirement: {h}"));
                    }
                    parts.push(hint);
                }
            }
            _ => {
                let vocab = doc.vocabulary();
                if !vocab.is_empty() {
                    let mut terms: Vec<&str> = vocab.iter().map(|s| s.as_str()).collect();
                    terms.sort();
                    parts.push(format!(
                        "## {} Constraints\n{}",
                        doc.id,
                        terms
                            .iter()
                            .map(|k| format!("- {k}"))
                            .collect::<Vec<_>>()
                            .join("\n")
                    ));
                }
            }
        }
    }
    parts.join("\n\n")
}

#[cfg(test)]
mod compiler_tests {
    use super::*;
    use h2ai_constraints::types::{ConstraintDoc, ConstraintPredicate, ConstraintSeverity};

    fn llm_judge_doc(id: &str, rubric: &str) -> ConstraintDoc {
        ConstraintDoc {
            id: id.to_string(),
            source_file: "test.md".into(),
            description: String::new(),
            severity: ConstraintSeverity::Hard { threshold: 0.8 },
            predicate: ConstraintPredicate::LlmJudge {
                rubric: rubric.to_string(),
            },
            remediation_hint: None,
            domains: vec![],
            mandatory_for_tags: vec![],
            related_to: vec![],
        }
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
        let doc = ConstraintDoc {
            id: "C-001".to_string(),
            source_file: "test.md".into(),
            description: String::new(),
            severity: ConstraintSeverity::Soft { weight: 1.0 },
            predicate: ConstraintPredicate::LlmJudge {
                rubric: "The proposal must be stateless.".to_string(),
            },
            remediation_hint: Some("Avoid sticky sessions.".into()),
            domains: vec![],
            mandatory_for_tags: vec![],
            related_to: vec![],
        };
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
}
