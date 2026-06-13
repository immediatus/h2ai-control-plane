//! GAP-F8/F9: constraint ambiguity detection — evidence model, static scanner,
//! and pure scoring helpers.
//!
//! Zero LLM calls and zero I/O in this module. The dynamic accumulation path
//! lives in the MAPE-K controller (h2ai-orchestrator); rewrite synthesis is
//! delegated to the existing GAP-K1 `SpecRepairAdvisor` (h2ai-autonomic).

use std::collections::HashMap;
use std::collections::HashSet;

use crate::types::{ConstraintDoc, ConstraintPredicate};

/// Sentinel check index for scorecards backed only by dynamic (runtime) evidence,
/// where the ambiguous check could not be pinpointed by the static scanner.
pub const DYNAMIC_ONLY_CHECK_IDX: usize = usize::MAX;

/// Weights + threshold for ambiguity scoring. Embedded in `H2AIConfig` as the
/// `[ambiguity_detection]` block. Lives in this crate so the pure scoring
/// functions carry no h2ai-config dependency.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AmbiguityDetectionConfig {
    /// Enable static scan seeding and evidence accumulation. Default: false.
    pub enabled: bool,
    /// Accumulated score at/above which the repair path fires. Default: 0.6.
    pub score_threshold: f32,
    pub weight_multi_storage: f32,
    pub weight_fm_negation: f32,
    pub weight_remediation_conflict: f32,
    pub weight_cross_check_negation: f32,
    /// Reserved for a future load-time LLM meta-validator (not wired in v1).
    pub weight_llm_confirmed: f32,
    pub weight_jaccard_freeze_wave: f32,
}

impl Default for AmbiguityDetectionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            score_threshold: 0.6,
            weight_multi_storage: 0.20,
            weight_fm_negation: 0.30,
            weight_remediation_conflict: 0.15,
            weight_cross_check_negation: 0.20,
            weight_llm_confirmed: 0.25,
            weight_jaccard_freeze_wave: 0.15,
        }
    }
}

/// One piece of evidence that a constraint check is ambiguous.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum AmbiguityEvidence {
    /// Check text names ≥2 storage systems without an explicit OR/either construction.
    MultiStorageConflict { systems: Vec<String> },
    /// A storage term required by the check appears negated in the rubric's
    /// guidance/failure-mode text.
    FmTermNegation { term: String, negated_in: String },
    /// First storage system in the check contradicts the first in the remediation hint.
    RemediationContradiction {
        check_system: String,
        hint_system: String,
    },
    /// Another check in the same constraint negates a term required by this check.
    CrossCheckNegation {
        this_term: String,
        negating_check_idx: usize,
    },
    /// Reserved for a future load-time LLM meta-validator (not wired in v1).
    LlmMetaValidated { reason: String },
    /// Cross-wave verifier-reason Jaccard fell below `gap_k1.instability_threshold`.
    JaccardFreezeWave { wave: u32, cross_wave_jaccard: f32 },
}

impl AmbiguityEvidence {
    #[must_use]
    pub fn weight(&self, cfg: &AmbiguityDetectionConfig) -> f32 {
        match self {
            Self::MultiStorageConflict { .. } => cfg.weight_multi_storage,
            Self::FmTermNegation { .. } => cfg.weight_fm_negation,
            Self::RemediationContradiction { .. } => cfg.weight_remediation_conflict,
            Self::CrossCheckNegation { .. } => cfg.weight_cross_check_negation,
            Self::LlmMetaValidated { .. } => cfg.weight_llm_confirmed,
            Self::JaccardFreezeWave { .. } => cfg.weight_jaccard_freeze_wave,
        }
    }
}

impl std::fmt::Display for AmbiguityEvidence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MultiStorageConflict { systems } => write!(
                f,
                "multi-storage conflict: check names [{}] without OR/either",
                systems.join(", ")
            ),
            Self::FmTermNegation { term, negated_in } => {
                let one_line = negated_in.replace('\n', " ");
                write!(f, "term '{term}' negated in rubric guidance: {one_line}")
            }
            Self::RemediationContradiction {
                check_system,
                hint_system,
            } => write!(
                f,
                "check requires '{check_system}' but remediation hint uses '{hint_system}'"
            ),
            Self::CrossCheckNegation {
                this_term,
                negating_check_idx,
            } => write!(
                f,
                "term '{this_term}' negated by sibling check {negating_check_idx}"
            ),
            Self::LlmMetaValidated { reason } => {
                let one_line = reason.replace('\n', " ");
                write!(f, "LLM meta-validation: {one_line}")
            }
            Self::JaccardFreezeWave {
                wave,
                cross_wave_jaccard,
            } => write!(
                f,
                "cross-wave verifier divergence at wave {wave}: jaccard={cross_wave_jaccard:.3}"
            ),
        }
    }
}

/// Whether a threshold-crossing scorecard can be routed to spec repair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PatchMode {
    /// Static scan pinpointed the ambiguous check; safe to repair at this index.
    Precise { check_idx: usize },
    /// Only dynamic evidence exists; check index unknown. Event only, no repair.
    DiagnosticOnly,
}

/// Per-(constraint, check) ambiguity evidence accumulator.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AmbiguityScorecard {
    pub constraint_id: String,
    /// Real check index, or `DYNAMIC_ONLY_CHECK_IDX` when unknown.
    pub check_idx: usize,
    /// Accumulated weighted score, capped at 1.0.
    pub score: f32,
    pub evidence: Vec<AmbiguityEvidence>,
    /// Set when the threshold has fired once — prevents repeat triggers in one run.
    pub rewrite_applied: bool,
}

impl AmbiguityScorecard {
    #[must_use]
    pub fn new(constraint_id: String, check_idx: usize) -> Self {
        Self {
            constraint_id,
            check_idx,
            score: 0.0,
            evidence: Vec::new(),
            rewrite_applied: false,
        }
    }

    /// `Precise` when any static (non-JaccardFreezeWave) evidence pinpointed the
    /// check; `DiagnosticOnly` otherwise. The corpus is never repaired at an index
    /// the static scanner did not confirm.
    #[must_use]
    pub fn patch_mode(&self) -> PatchMode {
        let has_static = self
            .evidence
            .iter()
            .any(|e| !matches!(e, AmbiguityEvidence::JaccardFreezeWave { .. }));
        if has_static && self.check_idx != DYNAMIC_ONLY_CHECK_IDX {
            PatchMode::Precise {
                check_idx: self.check_idx,
            }
        } else {
            PatchMode::DiagnosticOnly
        }
    }
}

/// Pure: returns a new scorecard with `ev` appended and the score updated.
/// Never mutates the input.
#[must_use]
pub fn score_evidence(
    scorecard: &AmbiguityScorecard,
    ev: AmbiguityEvidence,
    cfg: &AmbiguityDetectionConfig,
) -> AmbiguityScorecard {
    let mut updated = scorecard.clone();
    updated.score = (updated.score + ev.weight(cfg)).min(1.0);
    updated.evidence.push(ev);
    updated
}

/// Word-bag Jaccard similarity: |A ∩ B| / |A ∪ B|. Returns 1.0 when both bags
/// are empty. **Case-sensitive** — normalize both inputs to lowercase before
/// calling when comparing natural-language strings that may differ only in case.
/// Single shared implementation — `retry.rs` and `spec_repair.rs`
/// (h2ai-autonomic) import this instead of carrying private copies.
#[must_use]
pub fn jaccard(a: &str, b: &str) -> f64 {
    let bag_a: HashSet<&str> = a.split_whitespace().collect();
    let bag_b: HashSet<&str> = b.split_whitespace().collect();
    let union = bag_a.union(&bag_b).count();
    if union == 0 {
        return 1.0;
    }
    bag_a.intersection(&bag_b).count() as f64 / union as f64
}

/// Pure: from a list of verifier reasons, returns the pair with minimum Jaccard
/// similarity — the two furthest-apart interpretations. These are the correct
/// lead inputs for the repair prompt (not the two most common reasons, which may
/// be paraphrases of the same camp). `None` when fewer than 2 reasons.
#[must_use]
pub fn most_divergent_pair(reasons: &[String]) -> Option<(&str, &str)> {
    if reasons.len() < 2 {
        return None;
    }
    let mut min_j = f64::MAX;
    let mut best = (0usize, 1usize);
    for i in 0..reasons.len() {
        for j in (i + 1)..reasons.len() {
            let sim = jaccard(&reasons[i], &reasons[j]);
            if sim < min_j {
                min_j = sim;
                best = (i, j);
            }
        }
    }
    Some((&reasons[best.0], &reasons[best.1]))
}

/// Storage system vocabulary for the static heuristics. Longer names first so
/// "postgresql" wins over its "postgres" prefix at the same position.
const STORAGE_SYSTEMS: &[&str] = &[
    "cockroachdb",
    "postgresql",
    "clickhouse",
    "cassandra",
    "dynamodb",
    "mongodb",
    "rocksdb",
    "leveldb",
    "sqlite",
    "postgres",
    "redis",
    "kafka",
];

const NEGATION_WORDS: &[&str] = &[
    "avoid",
    "not",
    "never",
    "prohibit",
    "prohibits",
    "prohibited",
    "instead",
    "don't",
];

/// Storage systems found in `text`, as (byte_position, name), ordered by first
/// appearance. Prefix overlaps (postgresql/postgres) resolve to the longer name.
fn storage_systems_in(text: &str) -> Vec<(usize, String)> {
    let lower = text.to_lowercase();
    let mut found: Vec<(usize, String)> = Vec::new();
    for sys in STORAGE_SYSTEMS {
        if let Some(pos) = lower.find(sys) {
            found.push((pos, (*sys).to_string()));
        }
    }
    // Sort by position (asc), then by name length (desc) so "postgresql" comes
    // before "postgres" when both match at the same byte offset.
    // dedup_by only removes consecutive duplicates — correctness depends on the
    // list being sorted by position first.
    found.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| b.1.len().cmp(&a.1.len())));
    found.dedup_by(|b, a| b.0 == a.0);
    found
}

/// True when `term` appears within `window` tokens of a negation word in `text`.
fn term_negated_in(text: &str, term: &str, window: usize) -> bool {
    let tokens: Vec<String> = text
        .to_lowercase()
        .split_whitespace()
        .map(|t| {
            t.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'')
                .to_string()
        })
        .collect();
    let term_positions: Vec<usize> = tokens
        .iter()
        .enumerate()
        .filter(|(_, t)| t.contains(term))
        .map(|(i, _)| i)
        .collect();
    let neg_positions: Vec<usize> = tokens
        .iter()
        .enumerate()
        .filter(|(_, t)| NEGATION_WORDS.contains(&t.as_str()))
        .map(|(i, _)| i)
        .collect();
    term_positions
        .iter()
        .any(|tp| neg_positions.iter().any(|np| tp.abs_diff(*np) <= window))
}

/// All LlmJudge rubric text in the predicate tree, joined with newlines.
fn rubric_text(predicate: &ConstraintPredicate) -> String {
    match predicate {
        ConstraintPredicate::LlmJudge { rubric } => rubric.clone(),
        ConstraintPredicate::Composite { children, .. } => children
            .iter()
            .map(rubric_text)
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

/// Pure: scans one `ConstraintDoc` for static ambiguity evidence. Returns
/// `(check_idx, evidence)` pairs over `doc.binary_checks`. Deterministic,
/// zero LLM calls, zero I/O.
#[must_use]
pub fn scan_constraint(doc: &ConstraintDoc) -> Vec<(usize, AmbiguityEvidence)> {
    let mut out = Vec::new();
    let rubric = rubric_text(&doc.predicate);
    // Guidance lines: rubric text excluding the binary checks themselves —
    // failure-mode and pass/fail prose where negations indicate contradiction.
    let guidance_lines: Vec<&str> = rubric
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter(|l| !doc.binary_checks.iter().any(|c| l.contains(c.as_str())))
        .collect();

    for (idx, check) in doc.binary_checks.iter().enumerate() {
        let systems = storage_systems_in(check);
        let lower = check.to_lowercase();

        // 1. Multi-storage conflict: ≥2 systems, no OR/either escape hatch.
        if systems.len() >= 2 && !lower.contains(" or ") && !lower.contains("either") {
            out.push((
                idx,
                AmbiguityEvidence::MultiStorageConflict {
                    systems: systems.iter().map(|(_, s)| s.clone()).collect(),
                },
            ));
        }

        // 2. FM term negation: a required storage term is negated in guidance text.
        for (_, term) in &systems {
            if let Some(line) = guidance_lines.iter().find(|l| term_negated_in(l, term, 5)) {
                out.push((
                    idx,
                    AmbiguityEvidence::FmTermNegation {
                        term: term.clone(),
                        negated_in: (*line).trim().to_string(),
                    },
                ));
                break; // one FmTermNegation per check is enough signal
            }
        }

        // 3. Remediation contradiction: first system in check vs first in hint.
        if let Some(hint) = &doc.remediation_hint {
            let hint_systems = storage_systems_in(hint);
            if let (Some((_, check_first)), Some((_, hint_first))) =
                (systems.first(), hint_systems.first())
            {
                if check_first != hint_first {
                    out.push((
                        idx,
                        AmbiguityEvidence::RemediationContradiction {
                            check_system: check_first.clone(),
                            hint_system: hint_first.clone(),
                        },
                    ));
                }
            }
        }

        // 4. Cross-check negation: a sibling check negates a term this check requires.
        for (j, other) in doc.binary_checks.iter().enumerate() {
            if j == idx {
                continue;
            }
            for (_, term) in &systems {
                if term_negated_in(other, term, 5) {
                    out.push((
                        idx,
                        AmbiguityEvidence::CrossCheckNegation {
                            this_term: term.clone(),
                            negating_check_idx: j,
                        },
                    ));
                }
            }
        }
    }
    out
}

/// Pure: builds the initial scorecard map for a corpus from the static scan.
/// Empty when `cfg.enabled = false` (zero cost on the disabled path).
#[must_use]
pub fn seed_scorecards(
    corpus: &[ConstraintDoc],
    cfg: &AmbiguityDetectionConfig,
) -> HashMap<(String, usize), AmbiguityScorecard> {
    let mut map: HashMap<(String, usize), AmbiguityScorecard> = HashMap::new();
    if !cfg.enabled {
        return map;
    }
    for doc in corpus {
        for (idx, ev) in scan_constraint(doc) {
            let key = (doc.id.clone(), idx);
            let card = map
                .remove(&key)
                .unwrap_or_else(|| AmbiguityScorecard::new(doc.id.clone(), idx));
            map.insert(key, score_evidence(&card, ev, cfg));
        }
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> AmbiguityDetectionConfig {
        AmbiguityDetectionConfig {
            enabled: true,
            ..AmbiguityDetectionConfig::default()
        }
    }

    #[test]
    fn config_defaults() {
        let c = AmbiguityDetectionConfig::default();
        assert!(!c.enabled);
        assert!((c.score_threshold - 0.6).abs() < f32::EPSILON);
        assert!((c.weight_multi_storage - 0.20).abs() < f32::EPSILON);
        assert!((c.weight_fm_negation - 0.30).abs() < f32::EPSILON);
        assert!((c.weight_remediation_conflict - 0.15).abs() < f32::EPSILON);
        assert!((c.weight_cross_check_negation - 0.20).abs() < f32::EPSILON);
        assert!((c.weight_llm_confirmed - 0.25).abs() < f32::EPSILON);
        assert!((c.weight_jaccard_freeze_wave - 0.15).abs() < f32::EPSILON);
    }

    #[test]
    fn jaccard_identical_is_1() {
        assert!((jaccard("redis lua eval", "redis lua eval") - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn jaccard_disjoint_is_0() {
        assert!(jaccard("redis lua", "kafka stream").abs() < f64::EPSILON);
    }

    #[test]
    fn jaccard_empty_is_1() {
        assert!((jaccard("", "") - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn score_evidence_is_pure_and_does_not_mutate_original() {
        let card = AmbiguityScorecard::new("C-1".into(), 0);
        let ev = AmbiguityEvidence::FmTermNegation {
            term: "cockroachdb".into(),
            negated_in: "Avoid CockroachDB on the charge path".into(),
        };
        let a = score_evidence(&card, ev.clone(), &cfg());
        let b = score_evidence(&card, ev, &cfg());
        assert!((a.score - b.score).abs() < f32::EPSILON);
        assert_eq!(a.evidence, b.evidence);
        assert!(card.evidence.is_empty(), "original must not be mutated");
        assert!((card.score).abs() < f32::EPSILON);
    }

    #[test]
    fn score_evidence_caps_at_1() {
        let mut card = AmbiguityScorecard::new("C-1".into(), 0);
        for _ in 0..10 {
            card = score_evidence(
                &card,
                AmbiguityEvidence::FmTermNegation {
                    term: "t".into(),
                    negated_in: "n".into(),
                },
                &cfg(),
            );
        }
        assert!((card.score - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn patch_mode_precise_with_static_evidence() {
        let card = score_evidence(
            &AmbiguityScorecard::new("C-1".into(), 4),
            AmbiguityEvidence::FmTermNegation {
                term: "cockroachdb".into(),
                negated_in: "avoid".into(),
            },
            &cfg(),
        );
        assert_eq!(card.patch_mode(), PatchMode::Precise { check_idx: 4 });
    }

    #[test]
    fn patch_mode_diagnostic_only_with_jaccard_only() {
        let card = score_evidence(
            &AmbiguityScorecard::new("C-1".into(), DYNAMIC_ONLY_CHECK_IDX),
            AmbiguityEvidence::JaccardFreezeWave {
                wave: 3,
                cross_wave_jaccard: 0.034,
            },
            &cfg(),
        );
        assert_eq!(card.patch_mode(), PatchMode::DiagnosticOnly);
    }

    #[test]
    fn most_divergent_pair_returns_minimum_jaccard_pair() {
        let reasons = vec![
            "redis is the right ledger choice here".to_string(),
            "redis is the correct ledger choice here".to_string(),
            "cockroachdb on charge path violates failure mode".to_string(),
        ];
        let (a, b) = most_divergent_pair(&reasons).expect("pair");
        let pair = [a, b];
        assert!(
            pair.iter().any(|s| s.contains("cockroachdb")),
            "must include the divergent interpretation, got {pair:?}"
        );
    }

    #[test]
    fn most_divergent_pair_none_for_single() {
        assert!(most_divergent_pair(&["only one".to_string()]).is_none());
        assert!(most_divergent_pair(&[]).is_none());
    }

    #[test]
    fn evidence_display_renders_one_line_for_all_variants() {
        let cases = vec![
            AmbiguityEvidence::MultiStorageConflict {
                systems: vec!["cockroachdb".into(), "clickhouse".into()],
            },
            AmbiguityEvidence::FmTermNegation {
                term: "cockroachdb".into(),
                negated_in: "Avoid CockroachDB on the charge path\nsecond line".into(),
            },
            AmbiguityEvidence::RemediationContradiction {
                check_system: "cockroachdb".into(),
                hint_system: "redis".into(),
            },
            AmbiguityEvidence::CrossCheckNegation {
                this_term: "cockroachdb".into(),
                negating_check_idx: 2,
            },
            AmbiguityEvidence::LlmMetaValidated {
                reason: "multi-line\nreason".into(),
            },
            AmbiguityEvidence::JaccardFreezeWave {
                wave: 3,
                cross_wave_jaccard: 0.034,
            },
        ];
        for ev in &cases {
            let s = ev.to_string();
            assert!(
                !s.contains('\n'),
                "Display for {ev:?} must not contain newlines, got: {s:?}"
            );
        }
    }

    #[test]
    fn patch_mode_diagnostic_only_static_evidence_on_sentinel_index() {
        // Static evidence on DYNAMIC_ONLY_CHECK_IDX → DiagnosticOnly
        // (check index was never pinpointed despite having static evidence)
        let card = score_evidence(
            &AmbiguityScorecard::new("C-1".into(), DYNAMIC_ONLY_CHECK_IDX),
            AmbiguityEvidence::FmTermNegation {
                term: "cockroachdb".into(),
                negated_in: "avoid cockroachdb".into(),
            },
            &cfg(),
        );
        assert_eq!(card.patch_mode(), PatchMode::DiagnosticOnly);
    }

    use crate::types::{ConstraintPredicate, ConstraintSeverity};

    fn doc(
        checks: Vec<&str>,
        rubric_extra: &str,
        hint: Option<&str>,
    ) -> crate::types::ConstraintDoc {
        let rubric = format!("{}\n{}", checks.join("\n"), rubric_extra);
        crate::types::ConstraintDoc {
            id: "C-TEST".into(),
            source_file: "test.md".into(),
            description: "test constraint".into(),
            severity: ConstraintSeverity::Hard { threshold: 0.5 },
            predicate: ConstraintPredicate::LlmJudge { rubric },
            remediation_hint: hint.map(String::from),
            domains: vec![],
            mandatory_for_tags: vec![],
            related_to: vec![],
            binary_checks: checks.into_iter().map(String::from).collect(),
            version: 1,
            repair_provenance: None,
            pass_criteria: None,
        }
    }

    #[test]
    fn scan_detects_multi_storage_conflict() {
        let d = doc(
            vec!["Does the proposal use CockroachDB for state and ClickHouse for audit?"],
            "",
            None,
        );
        let evidence = scan_constraint(&d);
        assert!(evidence.iter().any(|(idx, e)| *idx == 0
            && matches!(e, AmbiguityEvidence::MultiStorageConflict { systems }
                if systems.contains(&"cockroachdb".to_string())
                && systems.contains(&"clickhouse".to_string()))));
    }

    #[test]
    fn scan_no_false_positive_on_or_construction() {
        let d = doc(
            vec!["Does the proposal use Redis or CockroachDB for the ledger?"],
            "",
            None,
        );
        let evidence = scan_constraint(&d);
        assert!(!evidence
            .iter()
            .any(|(_, e)| matches!(e, AmbiguityEvidence::MultiStorageConflict { .. })));
    }

    #[test]
    fn scan_detects_fm_negation_in_rubric_guidance() {
        let d = doc(
            vec!["Does the proposal use CockroachDB for operational state?"],
            "FM-2: Avoid CockroachDB on the synchronous charge path.",
            None,
        );
        let evidence = scan_constraint(&d);
        assert!(evidence.iter().any(|(idx, e)| *idx == 0
            && matches!(e, AmbiguityEvidence::FmTermNegation { term, .. }
                if term == "cockroachdb")));
    }

    #[test]
    fn scan_detects_remediation_contradiction() {
        let d = doc(
            vec!["Does the proposal use CockroachDB for operational state?"],
            "",
            Some("Use Redis Lua EVAL for atomic debits."),
        );
        let evidence = scan_constraint(&d);
        assert!(evidence.iter().any(|(idx, e)| *idx == 0
            && matches!(e, AmbiguityEvidence::RemediationContradiction { check_system, hint_system }
                if check_system == "cockroachdb" && hint_system == "redis")));
    }

    #[test]
    fn scan_detects_cross_check_negation() {
        let d = doc(
            vec![
                "Does the proposal use CockroachDB for operational state?",
                "Does the proposal never place CockroachDB on the charge path?",
            ],
            "",
            None,
        );
        let evidence = scan_constraint(&d);
        assert!(evidence.iter().any(|(idx, e)| *idx == 0
            && matches!(e, AmbiguityEvidence::CrossCheckNegation { this_term, negating_check_idx }
                if this_term == "cockroachdb" && *negating_check_idx == 1)));
    }

    #[test]
    fn scan_fm_negation_does_not_fire_on_check_line_itself() {
        // The check contains "never" + a storage system, but the check line is also
        // part of the rubric. The guidance-line exclusion filter must suppress
        // FmTermNegation — a check must not self-negate.
        let d = doc(
            vec!["Does the proposal never use CockroachDB for the charge path?"],
            "",
            None,
        );
        assert!(!scan_constraint(&d)
            .iter()
            .any(|(_, e)| matches!(e, AmbiguityEvidence::FmTermNegation { .. })));
    }

    #[test]
    fn scan_clean_check_produces_no_evidence() {
        let d = doc(
            vec!["Does the proposal include an idempotency key for every charge request?"],
            "FM-1: Duplicate charges must be rejected.",
            Some("Generate a UUID v4 idempotency key per request."),
        );
        assert!(scan_constraint(&d).is_empty());
    }

    #[test]
    fn scan_constraint_005_shape_detects_check_4() {
        // Mirrors CONSTRAINT-005 from the INNOVATION-5 Tier 2 run: check 4 requires
        // CockroachDB while the rubric's FM-005-2 warns against it on the charge path
        // and the hint leads with Redis.
        let d = doc(
            vec![
                "Does the proposal persist every charge attempt?",
                "Does the proposal make audit records immutable?",
                "Does the proposal separate operational state from audit state?",
                "Does the proposal retain audit records for 7 years?",
                "Does the proposal use a dual-ledger model: CockroachDB for operational state, ClickHouse for immutable audit?",
            ],
            "FM-005-2: Avoid CockroachDB on the synchronous charge path — latency budget.",
            Some("Use Redis for the hot ledger and append-only ClickHouse for audit."),
        );
        let evidence = scan_constraint(&d);
        let on_check_4: Vec<_> = evidence.iter().filter(|(idx, _)| *idx == 4).collect();
        assert!(
            on_check_4.len() >= 2,
            "expected ≥2 evidence items on check 4, got {evidence:?}"
        );
    }

    #[test]
    fn seed_scorecards_empty_when_disabled() {
        let d = doc(
            vec!["Does the proposal use CockroachDB and ClickHouse together?"],
            "",
            None,
        );
        let cfg_off = AmbiguityDetectionConfig::default(); // enabled = false
        assert!(seed_scorecards(&[d], &cfg_off).is_empty());
    }

    #[test]
    fn seed_scorecards_accumulates_per_check() {
        let d = doc(
            vec!["Does the proposal use CockroachDB for state and ClickHouse for audit?"],
            "FM-1: Avoid CockroachDB on the charge path.",
            None,
        );
        let cards = seed_scorecards(&[d], &cfg());
        let card = cards
            .get(&("C-TEST".to_string(), 0))
            .expect("card for check 0");
        assert!(card.evidence.len() >= 2);
        assert!(card.score > 0.0);
        assert_eq!(card.patch_mode(), PatchMode::Precise { check_idx: 0 });
    }
}
