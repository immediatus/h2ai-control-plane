use std::collections::HashSet;

use crate::types::{ConstraintDoc, ConstraintSeverity, ConstraintTier};

/// Zero-cost corpus-level complexity metadata.
///
/// Derived from constraint metadata alone — no I/O, no LLM calls.
/// Used by Phase 1.5 for structural TCC computation and probe-skip decisions.
#[derive(Debug, Clone)]
pub struct CorpusComplexityMetadata {
    /// FNV-1a hash of sorted constraint IDs — stable corpus fingerprint for caching.
    pub corpus_hash: u64,
    /// Total constraint count (all tiers).
    pub n_constraints: usize,
    /// Fraction of constraints with Soft severity.
    pub soft_fraction: f64,
    /// Type diversity: fraction of distinct predicate variant names present.
    pub type_diversity: f64,
    /// Fraction of constraints that are Static tier.
    pub static_coverage: f64,
    /// Fraction of constraints that are Heavy tier (`OracleExecution`).
    pub heavy_fraction: f64,
    /// `TCC_structural`: formula-based complexity prior.
    /// = 1.0 + `k_soft` × `soft_fraction` + `k_type` × `type_diversity`
    ///       + `k_cross` × `soft_fraction` × `type_diversity`
    pub tcc_structural: f64,
}

/// Number of distinct `ConstraintPredicate` variants tracked for type diversity.
/// Must stay in sync with `predicate_variant_name`'s match arms.
const N_PREDICATE_VARIANTS: usize = 12;

/// Default `TCC_structural` coefficients — theoretical initial priors from the GAP-A1
/// solution spec (§2.3). Fitted values will replace these after the GAP-A1 experiment.
/// Override via `[task_complexity]` config section in reference.toml.
const K_SOFT: f64 = 2.0;
const K_TYPE: f64 = 1.0;
const K_CROSS: f64 = 1.5;

/// Compute zero-cost corpus-level complexity metadata.
///
/// Pure function — safe to call on the hot path without any I/O.
#[must_use]
pub fn compute_corpus_complexity(corpus: &[ConstraintDoc]) -> CorpusComplexityMetadata {
    compute_corpus_complexity_with_coefficients(corpus, K_SOFT, K_TYPE, K_CROSS)
}

/// Same as `compute_corpus_complexity` but with configurable coefficients.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn compute_corpus_complexity_with_coefficients(
    corpus: &[ConstraintDoc],
    k_soft: f64,
    k_type: f64,
    k_cross: f64,
) -> CorpusComplexityMetadata {
    if corpus.is_empty() {
        return CorpusComplexityMetadata {
            corpus_hash: 0,
            n_constraints: 0,
            soft_fraction: 0.0,
            type_diversity: 0.0,
            static_coverage: 1.0,
            heavy_fraction: 0.0,
            tcc_structural: 1.0,
        };
    }

    let n = corpus.len();

    let mut ids: Vec<&str> = corpus.iter().map(|d| d.id.as_str()).collect();
    ids.sort_unstable();
    let corpus_hash = fnv1a(&ids);

    let soft_count = corpus
        .iter()
        .filter(|d| matches!(d.severity, ConstraintSeverity::Soft { .. }))
        .count();
    let soft_fraction = soft_count as f64 / n as f64;

    let predicate_types: HashSet<&'static str> = corpus
        .iter()
        .map(|d| predicate_variant_name(&d.predicate))
        .collect();
    let type_diversity = (predicate_types.len() as f64 / N_PREDICATE_VARIANTS as f64).min(1.0);

    let tier_counts = corpus
        .iter()
        .fold((0usize, 0usize, 0usize), |(s, l, h), d| match d.tier() {
            ConstraintTier::Static => (s + 1, l, h),
            ConstraintTier::Light => (s, l + 1, h),
            ConstraintTier::Heavy => (s, l, h + 1),
        });
    let static_coverage = tier_counts.0 as f64 / n as f64;
    let heavy_fraction = tier_counts.2 as f64 / n as f64;

    let tcc_structural = k_cross.mul_add(
        soft_fraction * type_diversity,
        k_type.mul_add(type_diversity, k_soft.mul_add(soft_fraction, 1.0)),
    );

    CorpusComplexityMetadata {
        corpus_hash,
        n_constraints: n,
        soft_fraction,
        type_diversity,
        static_coverage,
        heavy_fraction,
        tcc_structural,
    }
}

const fn predicate_variant_name(pred: &crate::types::ConstraintPredicate) -> &'static str {
    use crate::types::ConstraintPredicate::{
        Composite, JsonSchema, LengthRange, LlmJudge, NegativeKeyword, NumericThreshold,
        OracleExecution, RegexMatch, SemanticExclusion, SemanticOrdering, SemanticPresence,
        VocabularyPresence,
    };
    match pred {
        VocabularyPresence { .. } => "VocabularyPresence",
        NegativeKeyword { .. } => "NegativeKeyword",
        RegexMatch { .. } => "RegexMatch",
        NumericThreshold { .. } => "NumericThreshold",
        LlmJudge { .. } => "LlmJudge",
        Composite { .. } => "Composite",
        OracleExecution { .. } => "OracleExecution",
        JsonSchema { .. } => "JsonSchema",
        LengthRange { .. } => "LengthRange",
        SemanticPresence { .. } => "SemanticPresence",
        SemanticOrdering { .. } => "SemanticOrdering",
        SemanticExclusion { .. } => "SemanticExclusion",
    }
}

fn fnv1a(ids: &[&str]) -> u64 {
    const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
    const FNV_PRIME: u64 = 1_099_511_628_211;
    let mut hash = FNV_OFFSET;
    for id in ids {
        for byte in id.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash ^= u64::from(b',');
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}
