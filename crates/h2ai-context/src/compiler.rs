use h2ai_constraints::types::ConstraintDoc;

/// Output produced by a successful [`compile`] call.
pub struct CompilerResult {
    /// Assembled system-context string injected into LLM prompts.
    pub system_context: String,
}

/// Compile a task manifest and constraint corpus into a system-context string.
///
/// Concatenates the manifest and all constraint vocabularies under Markdown headings.
/// Tasks are accepted unconditionally — irrelevant tasks surface as verification
/// failures downstream. The j_eff gate has been removed.
pub fn compile(manifest: &str, corpus: &[ConstraintDoc]) -> CompilerResult {
    CompilerResult {
        system_context: build_system_context(manifest, corpus),
    }
}

fn build_system_context(manifest: &str, corpus: &[ConstraintDoc]) -> String {
    let mut parts = vec![format!("## Task Manifest\n{manifest}")];
    for doc in corpus {
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
    parts.join("\n\n")
}
