use crate::embedding::EmbeddingModel;
use crate::jaccard::tokenize;
use crate::similarity::semantic_jaccard;
use h2ai_config::H2AIConfig;
use h2ai_constraints::types::ConstraintDoc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ContextError {
    #[error(
        "J_eff={j_eff:.3} < {threshold:.1} — context underflow; add more constraint documents"
    )]
    ContextUnderflow { j_eff: f64, threshold: f64 },
}

pub struct CompilerResult {
    pub system_context: String,
    /// Signed J_eff = jaccard(manifest, positive_vocab) × (1 − contamination).
    ///
    /// `contamination` measures the fraction of manifest tokens that belong to the
    /// corpus's negative vocabulary (NoneOf / NegativeKeyword constraints). When the
    /// task manifest explicitly uses prohibited terms (e.g. "use G1GC" against an
    /// ADR that bans G1GC), J_eff is penalised — the Auditor needs more scrutiny, not less.
    ///
    /// Limitation: "avoid G1GC" also contains "g1gc", so manifests that name a prohibited
    /// term in order to reject it receive a small false-positive penalty. The Auditor's
    /// full predicate evaluation is the authoritative compliance check.
    pub j_eff: f64,
    /// Fraction of manifest tokens that matched corpus negative vocabulary (0.0 = none).
    pub contamination: f64,
}

pub async fn compile(
    manifest: &str,
    corpus: &[ConstraintDoc],
    task_required_keywords: &str,
    cfg: &H2AIConfig,
    embedding_model: Option<&dyn EmbeddingModel>,
) -> Result<CompilerResult, ContextError> {
    let manifest_tokens = tokenize(manifest);

    // Collect negative vocabulary from the corpus — terms a compliant proposal must NOT contain.
    // NoneOf + NegativeKeyword predicates, tokenised to individual words.
    let negative_vocab: std::collections::HashSet<String> = corpus
        .iter()
        .flat_map(|doc| doc.negative_vocabulary())
        .flat_map(|term| tokenize(&term))
        .collect();

    // Contamination: fraction of manifest tokens that are prohibited constraint vocabulary.
    // A manifest that names prohibited technology (e.g. "use G1GC heap") scores high here,
    // causing J_eff to be penalised — signalling likely constraint violation to the gate.
    let contamination = if manifest_tokens.is_empty() || negative_vocab.is_empty() {
        0.0
    } else {
        let negative_hits = manifest_tokens.intersection(&negative_vocab).count();
        negative_hits as f64 / manifest_tokens.len() as f64
    };

    // Positive coverage: semantic similarity between manifest and the required-keyword set.
    // Uses SLM adapter when available for semantic understanding (synonyms, paraphrases).
    // Falls back to token-level Jaccard when adapter is None — zero extra cost.
    let j_positive = semantic_jaccard(manifest, task_required_keywords, embedding_model);

    // Signed J_eff: reward domain vocabulary coverage, penalise negative-term contamination.
    let j_eff = j_positive * (1.0 - contamination);

    if j_eff < cfg.j_eff_gate {
        return Err(ContextError::ContextUnderflow {
            j_eff,
            threshold: cfg.j_eff_gate,
        });
    }

    Ok(CompilerResult {
        system_context: build_system_context(manifest, corpus),
        j_eff,
        contamination,
    })
}

fn build_system_context(manifest: &str, corpus: &[ConstraintDoc]) -> String {
    let mut parts = vec![format!("## Task Manifest\n{manifest}")];
    for doc in corpus {
        let vocab = doc.vocabulary();
        if !vocab.is_empty() {
            parts.push(format!(
                "## {} Constraints\n{}",
                doc.id,
                vocab
                    .iter()
                    .map(|k| format!("- {k}"))
                    .collect::<Vec<_>>()
                    .join("\n")
            ));
        }
    }
    parts.join("\n\n")
}
