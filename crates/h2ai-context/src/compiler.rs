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
/// followed by the rubric (for Hard or `include_rubric=true`) or just the constraint ID.
/// Vocabulary terms are always appended regardless of severity.
#[must_use]
pub fn compile(manifest: &str, corpus: &[ConstraintDoc], include_rubric: bool) -> CompilerResult {
    CompilerResult {
        system_context: build_system_context(manifest, corpus, include_rubric),
    }
}

fn build_system_context(manifest: &str, corpus: &[ConstraintDoc], include_rubric: bool) -> String {
    let mut parts = vec![COMPILER_TASK_MANIFEST.render(&[("manifest", manifest)])];
    for doc in corpus {
        if let ConstraintPredicate::Composite { children, .. } = &doc.predicate {
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
                Some(r) if include_rubric || is_hard => {
                    COMPILER_CONSTRAINT_HARD_RUBRIC.render(&[("id", &doc.id), ("rubric", r.trim())])
                }
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
                            &COMPILER_COMPOSITE_EXCLUSION_DETAIL.render(&[("pattern", pattern)]),
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
                let mut terms: Vec<&str> = vocab.iter().map(std::string::String::as_str).collect();
                terms.sort_unstable();
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
        } else {
            // Vocabulary fallback: VocabularyPresence, NegativeKeyword.
            // Used only by programmatically constructed constraints not from SemanticSpec.
            let vocab = doc.vocabulary();
            if !vocab.is_empty() {
                let mut terms: Vec<&str> = vocab.iter().map(std::string::String::as_str).collect();
                terms.sort_unstable();
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
    parts.join("\n\n")
}
