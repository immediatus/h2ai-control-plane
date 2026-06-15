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
    /// A check implies strict/universal use of a system but positive_examples show it
    /// inside try/except — the check is over-constrained relative to the examples.
    pub weight_positive_example_conflict: f32,
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
            weight_positive_example_conflict: 0.35,
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
    /// The check implies strict/universal use of a system ("every", "before", etc.)
    /// but a positive example in the rubric shows that system inside a try/except block,
    /// meaning the system is used conditionally. The check is over-constrained relative
    /// to the rubric's own authoritative positive examples.
    PositiveExampleConflict {
        term: String,
        example_snippet: String,
    },
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
            Self::PositiveExampleConflict { .. } => cfg.weight_positive_example_conflict,
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
            Self::PositiveExampleConflict {
                term,
                example_snippet,
            } => write!(
                f,
                "check demands strict '{term}' but positive example shows try/except: {example_snippet}"
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

/// Extracts code-block contents from the "--- Positive Examples ---" section of a
/// compiled `LlmJudge` rubric. The rubric is produced by `ConstraintYaml::build_rubric`
/// and embeds examples as fenced ` ``` ` blocks. Returns only the text *inside* each
/// fence, not the fence markers themselves.
fn extract_positive_code_blocks(rubric: &str) -> Vec<String> {
    const POS_MARKER: &str = "--- Positive Examples";
    let Some(pos_start) = rubric.find(POS_MARKER) else {
        return vec![];
    };
    let section = &rubric[pos_start..];
    let mut blocks = Vec::new();
    let mut rest = section;
    while let Some(open) = rest.find("```") {
        rest = &rest[open + 3..];
        // skip optional language tag on the opening line
        if let Some(newline) = rest.find('\n') {
            rest = &rest[newline + 1..];
        }
        if let Some(close) = rest.find("```") {
            blocks.push(rest[..close].to_string());
            rest = &rest[close + 3..];
        } else {
            break;
        }
    }
    blocks
}

/// Keywords that indicate a check makes a universal or strict-ordering claim —
/// "every", "before", "must", "always", "all".
const STRICT_CLAIM_WORDS: &[&str] = &["every", "before", "must", "always", "all"];

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

        // 5. Positive-example conflict: the check implies a strict/universal requirement
        //    for a storage system ("every", "before", "must", "always", "all") but a
        //    positive example in the rubric shows that system inside a try/except block —
        //    meaning the system is used conditionally, not as a hard prerequisite.
        //    Catches the pattern: check says "published to X before ACK" but
        //    positive_example has `try: X / except: local_fallback`.
        let implies_strict = lower
            .split_whitespace()
            .any(|w| STRICT_CLAIM_WORDS.contains(&w.trim_matches(|c: char| !c.is_alphabetic())));
        if implies_strict {
            let pos_blocks = extract_positive_code_blocks(&rubric);
            'outer: for (_, term) in &systems {
                for block in &pos_blocks {
                    let block_lower = block.to_lowercase();
                    let has_term = block_lower.contains(term.as_str());
                    let has_fallback =
                        block_lower.contains("except") || block_lower.contains("catch");
                    if has_term && has_fallback {
                        let snippet: String = block
                            .lines()
                            .find(|l| l.to_lowercase().contains(term.as_str()))
                            .unwrap_or_default()
                            .trim()
                            .chars()
                            .take(80)
                            .collect();
                        out.push((
                            idx,
                            AmbiguityEvidence::PositiveExampleConflict {
                                term: term.clone(),
                                example_snippet: snippet,
                            },
                        ));
                        break 'outer;
                    }
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
