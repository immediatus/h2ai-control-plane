use crate::adr::AdrConstraints;
use crate::jaccard::{jaccard, tokenize};
use h2ai_config::H2AIConfig;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ContextError {
    #[error("J_eff={j_eff:.3} < {threshold:.1} — context underflow; add more ADR constraints")]
    ContextUnderflow { j_eff: f64, threshold: f64 },
}

pub struct CompilerResult {
    pub system_context: String,
    pub j_eff: f64,
}

pub fn compile(
    manifest: &str,
    corpus: &[AdrConstraints],
    task_required_keywords: &str,
    cfg: &H2AIConfig,
) -> Result<CompilerResult, ContextError> {
    let k_prompt = tokenize(manifest);
    let k_required = tokenize(task_required_keywords);
    let j_eff = jaccard(&k_prompt, &k_required);

    if j_eff < cfg.j_eff_gate {
        return Err(ContextError::ContextUnderflow {
            j_eff,
            threshold: cfg.j_eff_gate,
        });
    }

    Ok(CompilerResult {
        system_context: build_system_context(manifest, corpus),
        j_eff,
    })
}

fn build_system_context(manifest: &str, corpus: &[AdrConstraints]) -> String {
    let mut parts = vec![format!("## Task Manifest\n{manifest}")];
    for adr in corpus {
        if !adr.keywords.is_empty() {
            parts.push(format!(
                "## {} Constraints\n{}",
                adr.source,
                adr.keywords
                    .iter()
                    .map(|k| format!("- {k}"))
                    .collect::<Vec<_>>()
                    .join("\n")
            ));
        }
    }
    parts.join("\n\n")
}
