use h2ai_constraints::types::{ConstraintDoc, ConstraintPredicate};

/// Output produced by a successful [`compile`] call.
pub struct CompilerResult {
    /// Assembled system-context string injected into LLM prompts.
    pub system_context: String,
}

/// Compile a task manifest and constraint corpus into a system-context string.
///
/// For LlmJudge constraints the rubric is embedded directly.
/// For vocabulary-based constraints the terms are listed as keywords.
pub fn compile(manifest: &str, corpus: &[ConstraintDoc]) -> CompilerResult {
    CompilerResult {
        system_context: build_system_context(manifest, corpus),
    }
}

fn build_system_context(manifest: &str, corpus: &[ConstraintDoc]) -> String {
    let mut parts = vec![format!("## Task Manifest\n{manifest}")];
    for doc in corpus {
        match &doc.predicate {
            ConstraintPredicate::LlmJudge { rubric } => {
                parts.push(format!("## {} Constraint\n{}", doc.id, rubric.trim()));
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
