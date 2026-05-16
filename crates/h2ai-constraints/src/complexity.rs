use crate::types::{ConstraintDoc, ConstraintSeverity, ConstraintTier};
use std::collections::HashSet;

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
    /// Fraction of constraints that are Heavy tier (OracleExecution).
    pub heavy_fraction: f64,
    /// TCC_structural: formula-based complexity prior.
    /// = 1.0 + k_soft × soft_fraction + k_type × type_diversity
    ///       + k_cross × soft_fraction × type_diversity
    pub tcc_structural: f64,
}

/// Number of distinct ConstraintPredicate variants tracked for type diversity.
/// Must stay in sync with `predicate_variant_name`'s match arms.
const N_PREDICATE_VARIANTS: usize = 12;

/// Default TCC_structural coefficients — theoretical initial priors from the GAP-A1
/// solution spec (§2.3). Fitted values will replace these after the GAP-A1 experiment.
/// Override via `[task_complexity]` config section in reference.toml.
const K_SOFT: f64 = 2.0;
const K_TYPE: f64 = 1.0;
const K_CROSS: f64 = 1.5;

/// Compute zero-cost corpus-level complexity metadata.
///
/// Pure function — safe to call on the hot path without any I/O.
pub fn compute_corpus_complexity(corpus: &[ConstraintDoc]) -> CorpusComplexityMetadata {
    compute_corpus_complexity_with_coefficients(corpus, K_SOFT, K_TYPE, K_CROSS)
}

/// Same as `compute_corpus_complexity` but with configurable coefficients.
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

    // FNV-1a corpus fingerprint over sorted IDs
    let mut ids: Vec<&str> = corpus.iter().map(|d| d.id.as_str()).collect();
    ids.sort_unstable();
    let corpus_hash = fnv1a(&ids);

    let soft_count = corpus
        .iter()
        .filter(|d| matches!(d.severity, ConstraintSeverity::Soft { .. }))
        .count();
    let soft_fraction = soft_count as f64 / n as f64;

    // Type diversity: fraction of the N_PREDICATE_VARIANTS predicate kinds present.
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

    let tcc_structural = 1.0
        + k_soft * soft_fraction
        + k_type * type_diversity
        + k_cross * soft_fraction * type_diversity;

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

fn predicate_variant_name(pred: &crate::types::ConstraintPredicate) -> &'static str {
    use crate::types::ConstraintPredicate::*;
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
            hash ^= *byte as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash ^= b',' as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ConstraintDoc, ConstraintPredicate, ConstraintSeverity, VocabularyMode};

    fn make_doc(
        id: &str,
        severity: ConstraintSeverity,
        pred: ConstraintPredicate,
    ) -> ConstraintDoc {
        ConstraintDoc {
            id: id.into(),
            source_file: "test".into(),
            description: "test constraint".into(),
            severity,
            predicate: pred,
            remediation_hint: None,
            domains: vec![],
            mandatory_for_tags: vec![],
            related_to: vec![],
        }
    }

    #[test]
    fn empty_corpus_returns_unit_tcc() {
        let meta = compute_corpus_complexity(&[]);
        assert_eq!(meta.n_constraints, 0);
        assert!((meta.tcc_structural - 1.0).abs() < 1e-9);
        assert!((meta.static_coverage - 1.0).abs() < 1e-9);
    }

    #[test]
    fn all_hard_static_corpus_has_low_tcc() {
        let corpus = vec![
            make_doc(
                "c1",
                ConstraintSeverity::Hard { threshold: 0.9 },
                ConstraintPredicate::VocabularyPresence {
                    mode: VocabularyMode::AllOf,
                    terms: vec!["auth".into()],
                },
            ),
            make_doc(
                "c2",
                ConstraintSeverity::Hard { threshold: 0.9 },
                ConstraintPredicate::RegexMatch {
                    pattern: ".*".into(),
                    must_match: true,
                },
            ),
        ];
        let meta = compute_corpus_complexity(&corpus);
        assert_eq!(meta.n_constraints, 2);
        assert!((meta.soft_fraction).abs() < 1e-9);
        assert!((meta.heavy_fraction).abs() < 1e-9);
        // 2 distinct types (VocabularyPresence, RegexMatch) out of N_PREDICATE_VARIANTS=12
        // tcc = 1.0 + 0 + K_TYPE * (2/12) + 0 = 1.0 + 1.0 * 0.1667 ≈ 1.167
        let expected = 1.0 + K_TYPE * (2.0 / N_PREDICATE_VARIANTS as f64);
        assert!(
            (meta.tcc_structural - expected).abs() < 1e-6,
            "tcc={} expected={}",
            meta.tcc_structural,
            expected
        );
    }

    #[test]
    fn heavy_constraint_increases_heavy_fraction() {
        let corpus = vec![
            make_doc(
                "c1",
                ConstraintSeverity::Hard { threshold: 0.9 },
                ConstraintPredicate::OracleExecution {
                    test_runner_uri: "http://localhost".into(),
                    test_suite: "suite".into(),
                    timeout_secs: 30,
                },
            ),
            make_doc(
                "c2",
                ConstraintSeverity::Soft { weight: 0.5 },
                ConstraintPredicate::VocabularyPresence {
                    mode: VocabularyMode::AnyOf,
                    terms: vec!["x".into()],
                },
            ),
        ];
        let meta = compute_corpus_complexity(&corpus);
        assert!((meta.heavy_fraction - 0.5).abs() < 1e-9);
        assert!((meta.soft_fraction - 0.5).abs() < 1e-9);
        assert!((meta.static_coverage - 0.5).abs() < 1e-9);
    }

    #[test]
    fn corpus_hash_is_stable_across_insertion_order() {
        let a = make_doc(
            "a",
            ConstraintSeverity::Advisory,
            ConstraintPredicate::LengthRange {
                min_chars: None,
                max_chars: Some(100),
            },
        );
        let b = make_doc(
            "b",
            ConstraintSeverity::Advisory,
            ConstraintPredicate::LengthRange {
                min_chars: None,
                max_chars: Some(200),
            },
        );
        let h1 = compute_corpus_complexity(&[a.clone(), b.clone()]).corpus_hash;
        let h2 = compute_corpus_complexity(&[b, a]).corpus_hash;
        assert_eq!(h1, h2, "hash must be order-independent");
    }
}
