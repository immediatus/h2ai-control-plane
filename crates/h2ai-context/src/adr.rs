//! Backward-compatibility shim: ADR loading now delegates to h2ai-constraints.
use h2ai_constraints::loader::{load_corpus as load_constraint_corpus, parse_constraint_doc};
use h2ai_constraints::types::ConstraintDoc;
use std::path::Path;

/// Kept for external callers that still reference AdrConstraints.
pub type AdrConstraints = ConstraintDoc;

pub fn load_corpus(dir: impl AsRef<Path>) -> Result<Vec<ConstraintDoc>, std::io::Error> {
    load_constraint_corpus(dir)
}

pub fn parse_adr(source: &str, content: &str) -> ConstraintDoc {
    parse_constraint_doc(source, content)
}
